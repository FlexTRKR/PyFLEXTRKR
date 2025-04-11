"""
Calculate monthly total, MCS precipitation amount and frequency within a period, save output to a netCDF file.
"""
import numpy as np
import sys, os
import xarray as xr
import pandas as pd
import time
from pyflextrkr.ft_utilities import load_config
from dask.distributed import Client, LocalCluster, progress

def process_month_chunked(month_ds, chunk_days=5):
    """Process one month of data in time chunks to reduce memory pressure"""
    # Current month's time value for output
    out_time = month_ds.time[0]
    out_time_str = out_time.dt.strftime('%Y-%m').item()
    
    # Get total times in month
    ntimes = len(month_ds.time)
    
    # Initialize accumulators for summations
    totprecip_sum = None
    mcsprecip_sum = None
    mcscloudct_sum = None
    mcspcpct_sum = None
    
    # Process in chunks of days to limit memory usage
    step = chunk_days * 24  # hours
    for start_idx in range(0, ntimes, step):
        end_idx = min(start_idx + step, ntimes)
        print(f"  Processing days {start_idx//24 + 1}-{end_idx//24} of month {out_time_str}")
        
        # Extract chunk of data and compute immediately to free memory
        chunk_ds = month_ds.isel(time=slice(start_idx, end_idx)).compute()
        
        # Extract variables
        mcs_mask = chunk_ds.mcs_mask
        precipitation = chunk_ds.precipitation
        
        # Compute statistics for this chunk
        chunk_mcsprecip = precipitation.where(mcs_mask > 0).sum(dim='time')
        chunk_totprecip = precipitation.sum(dim='time')
        chunk_mcscloudct = (mcs_mask > 0).sum(dim='time')
        chunk_mcspcpct = (precipitation.where(mcs_mask > 0) > pcp_thresh).sum(dim='time')
        
        # Accumulate results
        if totprecip_sum is None:
            totprecip_sum = chunk_totprecip
            mcsprecip_sum = chunk_mcsprecip
            mcscloudct_sum = chunk_mcscloudct
            mcspcpct_sum = chunk_mcspcpct
        else:
            totprecip_sum += chunk_totprecip
            mcsprecip_sum += chunk_mcsprecip
            mcscloudct_sum += chunk_mcscloudct
            mcspcpct_sum += chunk_mcspcpct
        
        # Explicitly delete chunk data to free memory
        del chunk_ds, mcs_mask, precipitation
        del chunk_mcsprecip, chunk_totprecip, chunk_mcscloudct, chunk_mcspcpct
    
    return {
        'time': out_time,
        'ntimes': ntimes,
        'totprecip': totprecip_sum,
        'mcsprecip': mcsprecip_sum,
        'mcscloudct': mcscloudct_sum,
        'mcspcpct': mcspcpct_sum
    }

if __name__ == "__main__":

    # Get inputs from command line
    config_file = sys.argv[1]
    start_datetime = sys.argv[2]
    end_datetime = sys.argv[3]

    # Set up a local Dask cluster optimized for large memory node
    # For HPC with 128 CPUs and 512GB memory
    n_workers = 12  # Use fewer workers with more memory each
    threads_per_worker = 10  # Use more threads per worker (12×10=120 cores total)
    memory_limit = '40GB'  # Each worker gets 40GB (12×40GB = 480GB total)
    chunk_days = 5  # Number of days to process in each chunk

    cluster = LocalCluster(
        n_workers=n_workers, 
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        local_directory='/tmp',
    )
    client = Client(cluster)
    print(client)

    # Rain rate threshold to compute MCS precipitation frequency
    pcp_thresh = 2.0  # [mm/h]

    # Load configuration file
    config = load_config(config_file)
    # Get config parameters
    pixeltracking_outpath = config.get("pixeltracking_outpath")
    stats_outpath = config.get("stats_outpath")
    startdate = config.get("startdate")
    enddate = config.get("enddate")
    in_dir = os.path.dirname(os.path.normpath(pixeltracking_outpath)) + "/"
    # Get preset-specific configuration
    presets = config.get("zarr_output_presets", {})
    mask_filebase = presets.get("mask", {}).get("out_filebase", "mcs_mask_latlon_")
    tbpr_filebase = presets.get("tbpr", {}).get("out_filebase", "tb_pr_latlon_")
    # Input mask Zarr store
    mask_file = f"{in_dir}{mask_filebase}{startdate}_{enddate}.zarr"
    tbpr_file = f"{in_dir}{tbpr_filebase}{startdate}_{enddate}.zarr"

    # Convert input datetimes to output datetime strings
    sdatetime_str = pd.Timestamp(start_datetime).strftime('%Y%m%d')
    edatetime_str = pd.Timestamp(end_datetime).strftime('%Y%m%d')

    # Output directory
    out_dir = f'{stats_outpath}monthly/'
    os.makedirs(out_dir, exist_ok=True)
    # output_filename = f'{out_dir}monthly_mcs_rainmap.nc'
    output_filename = f'{out_dir}mcs_monthly_rainmap_{sdatetime_str}_{edatetime_str}.nc'

    # Check if the input file is a directory (Zarr store)
    if not os.path.isdir(tbpr_file):
        print(f'ERROR: Zarr store does not exist or is not a directory: {tbpr_file}')
        sys.exit('Code will exit now.')
    if not os.path.isdir(mask_file):
        print(f'ERROR: Zarr store does not exist or is not a directory: {mask_file}')
        sys.exit('Code will exit now.')

    # Read mask Zarr store, subset times
    ds_m = xr.open_zarr(mask_file).sel(time=slice(start_datetime, end_datetime))
    # Read Tb/precipitation Zarr store, subset times
    ds_p = xr.open_zarr(tbpr_file).sel(time=slice(start_datetime, end_datetime))
    # Merge datasets
    ds = xr.merge([ds_p, ds_m], combine_attrs='override')
    # import pdb; pdb.set_trace()

    # Group by month and apply the processing function
    monthly_results = []
    monthly_groups = ds.resample(time='1MS')

    # Use parallel processing
    delayed_results = []
    for month_start, month_ds in monthly_groups:
        print(f"Processing month: {pd.Timestamp(month_start).strftime('%Y-%m')}")
        # Submit the processing job to the dask cluster
        delayed_result = client.submit(process_month_chunked, month_ds, chunk_days=chunk_days)
        delayed_results.append(delayed_result)
    
    # Clear line and show progress tracking
    print("\nTracking progress of all months processing in parallel:")
    progress(delayed_results)
    
    # Gather results (will wait for completion)
    results = client.gather(delayed_results)

    # Prepare data for output dataset
    times = [r['time'] for r in results]
    ntimes_values = [r['ntimes'] for r in results]
    totprecip_values = [r['totprecip'] for r in results]
    mcsprecip_values = [r['mcsprecip'] for r in results]
    mcscloudct_values = [r['mcscloudct'] for r in results]
    mcspcpct_values = [r['mcspcpct'] for r in results]

    # Create output dataset
    var_dict = {
        'precipitation': (['time', 'lat', 'lon'], np.stack([r.values for r in totprecip_values])),
        'mcs_precipitation': (['time', 'lat', 'lon'], np.stack([r.values for r in mcsprecip_values])),
        'mcs_precipitation_count': (['time', 'lat', 'lon'], np.stack([r.values for r in mcspcpct_values])),
        'mcs_cloud_count': (['time', 'lat', 'lon'], np.stack([r.values for r in mcscloudct_values])),
        'ntimes': (['time'], np.array(ntimes_values)),
    }

    coord_dict = {
        'time': (['time'], [t.values for t in times]),
        'lat': (['lat'], ds['lat'].values),
        'lon': (['lon'], ds['lon'].values),
    }

    gattr_dict = {
        'Title': 'Monthly MCS precipitation statistics',
        'contact':'Zhe Feng, zhe.feng@pnnl.gov',
        'start_date': start_datetime,
        'end_date': end_datetime,
        'created_on': time.ctime(time.time()),
    }

    # Create output dataset
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

    # Add variable attributes
    dsout['lon'].attrs['long_name'] = 'Longitude'
    dsout['lon'].attrs['units'] = 'degree'
    dsout['lat'].attrs['long_name'] = 'Latitude'
    dsout['lat'].attrs['units'] = 'degree'
    dsout['ntimes'].attrs['long_name'] = 'Number of hours during the month'
    dsout['ntimes'].attrs['units'] = 'count'
    dsout['precipitation'].attrs['long_name'] = 'Total precipitation'
    dsout['precipitation'].attrs['units'] = 'mm'
    dsout['mcs_precipitation'].attrs['long_name'] = 'MCS precipitation'
    dsout['mcs_precipitation'].attrs['units'] = 'mm'
    dsout['mcs_precipitation_count'].attrs['long_name'] = 'Number of hours MCS precipitation is recorded'
    dsout['mcs_precipitation_count'].attrs['units'] = 'hour'
    dsout['mcs_cloud_count'].attrs['long_name'] = 'Number of hours MCS cloud is recorded'
    dsout['mcs_cloud_count'].attrs['units'] = 'hour'

    # Save the output file
    fillvalue = np.nan
    comp = dict(zlib=True, _FillValue=fillvalue, dtype='float32')
    encoding = {var: comp for var in dsout.data_vars}

    print(f'Writing output file ...')
    dsout.to_netcdf(path=output_filename, mode='w', format='NETCDF4', 
                    unlimited_dims='time', encoding=encoding)
    print(f'Done: {output_filename}')

    # Close the dask client
    client.close()