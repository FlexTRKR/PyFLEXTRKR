"""
Calculate monthly total, MCS precipitation amount and frequency within a period, save output to a netCDF file.

Usage: python calc_tbpf_mcs_monthly_rainmap_zarr.py -c CONFIG.yml -s STARTDATE -e ENDDATE
Optional arguments:
--zoom Z (HEALPix zoom level)
--parallel (use parallel processing)
--nworkers N (number of Dask workers)
--threads T (threads per worker)
--memory M (memory limit per worker, e.g. '40GB')
--chunk_days N (days per processing chunk)
--pcp_thresh P (precipitation threshold in mm/h)
"""
import numpy as np
import sys, os
import xarray as xr
import pandas as pd
import time
import psutil
import argparse
import cftime
import intake
import requests
import easygems.healpix as egh
from pyflextrkr.ft_utilities import load_config, convert_cftime_to_standard
from pyflextrkr.zarr_tools import setup_dask_client
import logging

def setup_logging():
    """
    Set up the logging configuration
    """
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

def parse_cmd_args():
    # Define and retrieve the command-line arguments...
    parser = argparse.ArgumentParser(
        description="Calculate monthly MCS precipitation statistics."
    )
    parser.add_argument("-c", "--config", help="yaml config file for tracking", required=True)
    parser.add_argument("-s", "--start", help="first time to process, format=YYYY-mm-ddTHH", required=True)
    parser.add_argument("-e", "--end", help="last time to process, format=YYYY-mm-ddTHH", required=True)
    parser.add_argument("--zoom", help="HEALPix zoom level", type=int, default=None)
    parser.add_argument("--parallel", help="use parallel processing", action="store_true")
    parser.add_argument("--nworkers", help="number of Dask workers", type=int, default=12)
    parser.add_argument("--threads", help="threads per worker", type=int, default=1)
    parser.add_argument("--memory", help="memory limit per worker, e.g. '40GB' (default: auto)", default=None)
    parser.add_argument("--chunk_days", help="number of days to process in each chunk", type=int, default=5)
    parser.add_argument("--pcp_thresh", help="precipitation threshold in mm/h", type=float, default=2.0)
    args = parser.parse_args()

    # Put arguments in a dictionary
    args_dict = {
        'config_file': args.config,
        'start_datetime': args.start,
        'end_datetime': args.end,
        'zoom': args.zoom,
        'parallel': args.parallel,
        'n_workers': args.nworkers,
        'threads_per_worker': args.threads,
        'memory_limit': args.memory,
        'chunk_days': args.chunk_days,
        'pcp_thresh': args.pcp_thresh,
    }

    return args_dict

def process_month_chunked(month_ds, chunk_days=5, pcp_thresh=2.0):
    """Process one month of data in time chunks to reduce memory pressure"""
    # Current month's time value for output
    out_time = month_ds.time[0]
    out_time_str = out_time.dt.strftime('%Y-%m').item()
    _year = out_time.dt.year.item()
    _month = out_time.dt.month.item()
    _day = out_time.dt.day.item()
    # Create standard datetime using pandas
    std_time = pd.Timestamp(_year, _month, _day, 0, 0, 0)
    
    # Get total times in month
    ntimes = len(month_ds.time)
    
    # Detect time interval in hours
    if ntimes > 1:
        # Get the first two timestamps
        time_values = month_ds.time.values
        
        # Calculate time interval in hours based on type
        if isinstance(time_values[0], (cftime._cftime.DatetimeNoLeap, cftime._cftime.Datetime360Day)):
            # For cftime objects
            time_interval = (time_values[1] - time_values[0]).total_seconds() / 3600.0
        else:
            # For numpy datetime64 objects
            time_interval = (pd.Timestamp(time_values[1]) - pd.Timestamp(time_values[0])).total_seconds() / 3600.0
            
        print(f"  Detected time interval: {time_interval:.1f} hours")
    else:
        # Default to 1 hour if only one timestamp
        time_interval = 1.0
        print(f"  Using default time interval: {time_interval:.1f} hours")
    
    # Calculate steps per day based on time interval
    steps_per_day = 24.0 / time_interval
    
    # Initialize accumulators for summations
    totprecip_sum = None
    mcsprecip_sum = None
    mcscloudct_sum = None
    mcspcpct_sum = None
    
    # Process in chunks of days to limit memory usage
    step = int(chunk_days * steps_per_day)  # Convert chunk_days to timesteps
    for start_idx in range(0, ntimes, step):
        end_idx = min(start_idx + step, ntimes)
        start_day = int(start_idx / steps_per_day) + 1
        end_day = int(end_idx / steps_per_day)
        print(f"  Processing days {start_day}-{end_day} of month {out_time_str}")
        
        # Extract chunk of data and compute immediately to free memory
        chunk_ds = month_ds.isel(time=slice(start_idx, end_idx)).compute()
        
        # Extract variables
        mcs_mask = chunk_ds.mcs_mask
        precipitation = chunk_ds.precipitation
        
        # Compute statistics for this chunk - multiply by time_interval to get mm
        chunk_mcsprecip = (precipitation.where(mcs_mask > 0) * time_interval).sum(dim='time')
        chunk_totprecip = (precipitation * time_interval).sum(dim='time')
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
        'time': std_time,
        'ntimes': ntimes,
        'time_interval': time_interval,
        'totprecip': totprecip_sum,
        'mcsprecip': mcsprecip_sum,
        'mcscloudct': mcscloudct_sum,
        'mcspcpct': mcspcpct_sum
    }

def get_memory_usage():
    """Get current memory usage in a human-readable format"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    # Convert to GB
    memory_gb = memory_info.rss / (1024 ** 3)
    return memory_gb

def format_time(seconds):
    """Format time in seconds to hours, minutes, seconds"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# def setup_dask_client(parallel, n_workers, threads_per_worker, memory_limit=None, logger=None):
#     """
#     Set up a Dask client for parallel processing
    
#     Args:
#         parallel: bool
#             Whether to use parallel processing
#         n_workers: int
#             Number of workers for the Dask cluster
#         threads_per_worker: int
#             Number of threads per worker
#         memory_limit: str, optional
#             Memory limit for each worker (e.g. '40GB'). Use None for auto detection.
#         logger: logging.Logger, optional
#             Logger for status messages
            
#     Returns:
#         dask.distributed.Client or None: Dask client if parallel is True, None otherwise
#     """
#     if logger is None:
#         logger = logging.getLogger(__name__)
        
#     if not parallel:
#         logger.info("Running in sequential mode (parallel=False)")
#         return None
    
#     memory_str = "auto" if memory_limit is None else memory_limit
#     logger.info(f"Setting up Dask cluster with {n_workers} workers, {threads_per_worker} threads per worker, {memory_str} memory")
    
#     cluster = LocalCluster(
#         n_workers=n_workers,
#         threads_per_worker=threads_per_worker,
#         memory_limit=memory_limit,
#     )
#     client = Client(cluster)
#     logger.info(f"Dask dashboard: {client.dashboard_link}")
    
#     return client

def write_netcdf(results, ds, output_filename, zoom, pcp_thresh, logger=None):
    """
    Write monthly precipitation statistics to a NetCDF file.
    
    Args:
        results: list
            Results from process_month_chunked function
        ds: xarray.Dataset
            Original dataset with coordinates
        output_filename: str
            Output NetCDF file path
        zoom: int
            HEALPix zoom level
        pcp_thresh: float
            Precipitation threshold in mm/h
        logger: logging.Logger, optional
            Logger for status messages
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f'Preparing data for output file: {output_filename}')
    
    # Prepare data for output dataset
    times = [r['time'] for r in results]
    ntimes_values = [r['ntimes'] for r in results]
    time_interval = results[0]['time_interval']
    
    totprecip_values = [r['totprecip'] for r in results]
    mcsprecip_values = [r['mcsprecip'] for r in results]
    mcscloudct_values = [r['mcscloudct'] for r in results]
    mcspcpct_values = [r['mcspcpct'] for r in results]

    # Create output variables
    var_dict = {
        'precipitation': (['time', 'cell'], np.stack([r.values for r in totprecip_values])),
        'mcs_precipitation': (['time', 'cell'], np.stack([r.values for r in mcsprecip_values])),
        'mcs_precipitation_count': (['time', 'cell'], np.stack([r.values for r in mcspcpct_values])),
        'mcs_cloud_count': (['time', 'cell'], np.stack([r.values for r in mcscloudct_values])),
        'ntimes': (['time'], np.array(ntimes_values)),
    }
    # Create coordinates
    coord_dict = {
        'time': times,
        'cell': (['cell'], ds['cell'].values),
        'lat': (['cell'], ds['lat'].values),
        'lon': (['cell'], ds['lon'].values),
    }
    
    # Add crs if available
    if 'crs' in ds:
        coord_dict['crs'] = ds['crs'].values
    
    # Create global attributes
    gattr_dict = {
        'Title': 'Monthly MCS precipitation statistics',
        'contact': 'Zhe Feng, zhe.feng@pnnl.gov',
        'created_on': time.ctime(time.time()),
        'grid_type': 'HEALPix',
        'zoom_level': zoom,
        'time_interval': time_interval,
        'precipitation_threshold': pcp_thresh,
    }

    # Create output dataset
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

    # Add variable attributes
    dsout['cell'].attrs['long_name'] = 'HEALPix cell index'
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
    dsout['mcs_precipitation_count'].attrs['long_name'] = 'Number of hours MCS precipitation exceeds threshold'
    dsout['mcs_precipitation_count'].attrs['units'] = 'hour'
    dsout['mcs_cloud_count'].attrs['long_name'] = 'Number of hours MCS cloud is recorded'
    dsout['mcs_cloud_count'].attrs['units'] = 'hour'

    # Save the output file
    fillvalue = np.nan
    comp = dict(zlib=True, _FillValue=fillvalue, dtype='float32')
    encoding = {var: comp for var in dsout.data_vars}

    logger.info(f'Writing output file: {output_filename}')
    dsout.to_netcdf(path=output_filename, mode='w', format='NETCDF4', 
                    unlimited_dims='time', encoding=encoding)
    logger.info(f'Successfully wrote: {output_filename}')
    
    return dsout


def main():
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Start timing
    start_time = time.time()
    initial_memory = get_memory_usage()
    logger.info(f"Initial memory usage: {initial_memory:.2f} GB")

    # Get the command-line arguments
    args_dict = parse_cmd_args()
    config_file = args_dict.get('config_file')
    start_datetime = args_dict.get('start_datetime')
    end_datetime = args_dict.get('end_datetime')
    zoom = args_dict.get('zoom')
    parallel = args_dict.get('parallel')
    n_workers = args_dict.get('n_workers')
    threads_per_worker = args_dict.get('threads_per_worker')
    memory_limit = args_dict.get('memory_limit')
    chunk_days = args_dict.get('chunk_days')
    pcp_thresh = args_dict.get('pcp_thresh')

    # Set up a local Dask cluster for parallel processing
    # client = setup_dask_client(parallel, n_workers, threads_per_worker, memory_limit, logger)
    client = setup_dask_client(parallel=parallel, n_workers=n_workers, threads_per_worker=threads_per_worker, logger=logger)
    
    # Load configuration file
    config = load_config(config_file)

    # ---------- INPUT CONFIGURATION ----------
    root_path = config.get("root_path")
    pixel_path_name = config.get("pixel_path_name")
    # pixeltracking_outpath = config.get("pixeltracking_outpath")
    stats_outpath = config.get("stats_outpath")
    startdate = config.get("startdate")
    enddate = config.get("enddate")
    pcp_varname = config.get('pcp_varname')
    pcp_convert_factor = config.get('pcp_convert_factor', 1)

    # Get preset-specific configuration
    presets = config.get("zarr_output_presets", {})
    preset_mask = presets.get("mask", {})
    preset_healpix = presets.get("healpix", {})
    
    # ---------- HEALPIX CONFIGURATION ----------
    catalog_file = config.get("catalog_file")
    catalog_location = config.get("catalog_location", "")
    catalog_source = config.get("catalog_source", "")
    catalog_params = config.get("catalog_params", {})
    catalog_zoom = catalog_params.get("zoom")
    hp_zoom = preset_healpix.get("zoom", catalog_zoom)
    hp_version = preset_healpix.get("version", "v1")
    # Update zoom level if provided in command line
    if zoom is not None:
        hp_zoom = zoom
        catalog_params["zoom"] = hp_zoom

    # MCS mask HEALPix Zarr file
    hp_filebase = preset_healpix.get("out_filebase", "mcs_mask_")
    mask_file = f"{root_path}{pixel_path_name}/{hp_filebase}hp{hp_zoom}_{hp_version}.zarr"

    # Convert input datetimes to output datetime strings
    sdatetime_str = pd.Timestamp(start_datetime).strftime('%Y%m%d')
    edatetime_str = pd.Timestamp(end_datetime).strftime('%Y%m%d')

    # Output directory
    out_dir = f'{stats_outpath}monthly/'
    os.makedirs(out_dir, exist_ok=True)
    output_filename = f'{out_dir}mcs_monthly_rainmap_hp{hp_zoom}_{sdatetime_str}_{edatetime_str}.nc'

    # Check catalog file availability
    if catalog_file:
        if catalog_file.startswith(('http://', 'https://')):
            # Handle URL case
            try:
                response = requests.head(catalog_file, timeout=10)
                if response.status_code >= 400:
                    print(f"Catalog URL {catalog_file} returned status code {response.status_code}. Skipping remap.")
                    sys.exit('Code will exit now.')
            except requests.exceptions.RequestException as e:
                print(f"Error accessing catalog URL {catalog_file}: {str(e)}. Skipping remap.")
                sys.exit('Code will exit now.')
        elif os.path.isfile(catalog_file) is False:
            # Handle local file case
            print(f"Catalog file {catalog_file} does not exist. Skipping remap.")
            sys.exit('Code will exit now.')
    else:
        print("Catalog file not specified in config. HEALPix remapping requires a catalog.")
        sys.exit('Code will exit now.')
    
    # Load the HEALPix catalog
    print(f"Loading HEALPix catalog: {catalog_file}")
    in_catalog = intake.open_catalog(catalog_file)
    if catalog_location:
        in_catalog = in_catalog[catalog_location]

    # Get the DataSet from the catalog
    ds_p = in_catalog[catalog_source](**catalog_params).to_dask()
    # Add lat/lon coordinates to the HEALPix DataSet
    ds_p = ds_p.pipe(egh.attach_coords)
    # Check the calendar type of the time coordinate
    calendar = ds_p['time'].dt.calendar
   
    # Convert ds_p time coordinate to standard calendar
    if calendar not in ['proleptic_gregorian', 'gregorian', 'standard']:
        print(f"Converting ds_p's {calendar} calendar to proleptic_gregorian calendar")
        # Create new times with proleptic_gregorian calendar
        standard_times = convert_cftime_to_standard(ds_p['time'].values)
        
        # Create a new dataset with standard calendar
        # ds_p_converted = ds_p.copy()
        ds_p['time'] = standard_times
    else:
        print(f"Dataset already uses standard calendar: {calendar}")

    # Rename the variable and apply conversion factor
    ds_p = ds_p.rename({pcp_varname: 'precipitation'})
    if pcp_convert_factor != 1:
        ds_p['precipitation'] = ds_p['precipitation'] * pcp_convert_factor

    # Check mask directory (Zarr store)
    if not os.path.isdir(mask_file):
        print(f'ERROR: Zarr store does not exist or is not a directory: {mask_file}')
        sys.exit('Code will exit now.')

    # Read mask Zarr store, subset times
    ds_m = xr.open_zarr(mask_file).sel(time=slice(start_datetime, end_datetime))

    # Find common time range
    common_times = sorted(set(ds_p['time'].values).intersection(set(ds_m['time'].values)))
    if not common_times:
        logger.warning("No common time values between datasets!")
        return None
    else:
        # Select only the common times in both datasets
        ds_p_subset = ds_p.sel(time=common_times)
        ds_m_subset = ds_m.sel(time=common_times)

    # Merge the datasets
    ds = xr.merge([ds_p_subset, ds_m_subset], combine_attrs='override')
    print(f"Successfully merged datasets with {len(common_times)} common time points")
    
    # Group by month and apply the processing function
    # monthly_results = []
    monthly_groups = ds.resample(time='1MS')

    # Check if client exists for parallel processing
    if client is not None:
        # Use parallel processing
        delayed_results = []
        for month_start, month_ds in monthly_groups:
            logger.info(f"Processing month: {pd.Timestamp(month_start).strftime('%Y-%m')}")
            # Submit the processing job to the dask cluster
            delayed_result = client.submit(process_month_chunked, month_ds, chunk_days=chunk_days, pcp_thresh=pcp_thresh)
            delayed_results.append(delayed_result)
        
        # Clear line and show progress tracking
        logger.info("Tracking progress of all months processing in parallel:")
        from dask.distributed import progress
        progress(delayed_results)
        
        # Gather results (will wait for completion)
        results = client.gather(delayed_results)
    else:
        # Serial processing
        logger.info("Running in serial mode")
        results = []
        for month_start, month_ds in monthly_groups:
            logger.info(f"Processing month: {pd.Timestamp(month_start).strftime('%Y-%m')}")
            result = process_month_chunked(month_ds, chunk_days=chunk_days, pcp_thresh=pcp_thresh)
            results.append(result)

    # Write output to NetCDF file
    write_netcdf(results, ds, output_filename, hp_zoom, pcp_thresh, logger)

    # Calculate and print timing and memory usage
    end_time = time.time()
    final_memory = get_memory_usage()
    elapsed_time = end_time - start_time
    memory_change = final_memory - initial_memory
    
    logger.info("\n" + "="*50)
    logger.info(f"Performance Summary:")
    logger.info(f"  Total execution time: {format_time(elapsed_time)} (HH:MM:SS)")
    logger.info(f"  Initial memory usage: {initial_memory:.2f} GB")
    logger.info(f"  Final memory usage:   {final_memory:.2f} GB")
    logger.info(f"  Memory change:        {memory_change:.2f} GB")
    logger.info("="*50)

    # Cleanup client
    if client and parallel:
        logger.info("Shutting down Dask client")
        client.close()


if __name__ == "__main__":
    main()