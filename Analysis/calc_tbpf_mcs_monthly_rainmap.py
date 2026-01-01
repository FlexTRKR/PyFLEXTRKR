"""
Calculate monthly total, MCS precipitation amount and frequency, save output to a netCDF file.

Uses xarray's lazy evaluation approach - builds computation graph, then computes all at once.
"""
import numpy as np
import glob, sys, os
import xarray as xr
import time, datetime, calendar, pytz
import logging
import psutil
from pyflextrkr.ft_utilities import load_config

def setup_logging():
    """
    Set up the logging configuration
    """
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
        level=logging.INFO
    )

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / (1024 ** 3)
    return memory_gb

def format_time(seconds):
    """Format time in seconds to HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def main():
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Start timing
    script_start_time = time.time()
    initial_memory = get_memory_usage()
    logger.info(f"Script started - Initial memory usage: {initial_memory:.2f} GB")
    logger.info("Method: Lazy evaluation (build graph, compute once)")

    # Get inputs from command line
    config_file = sys.argv[1]
    year = (sys.argv[2])
    month = (sys.argv[3]).zfill(2)
    
    logger.info(f"Processing: Year={year}, Month={month}")

    # Get inputs from configuration file
    config_start = time.time()
    config = load_config(config_file)
    pixel_dir = config['pixeltracking_outpath']
    output_monthly_dir = config['stats_outpath'] + 'monthly/'
    pcpvarname = 'precipitation'
    logger.info(f"Config loaded in {time.time() - config_start:.2f} seconds")

    # Output file name
    output_filename = f'{output_monthly_dir}mcs_rainmap_{year}{month}.nc'

    # Find all pixel files in a month
    file_search_start = time.time()
    mcsfiles = sorted(glob.glob(f'{pixel_dir}/mcstrack_{year}{month}*_*.nc'))
    nfiles = len(mcsfiles)
    logger.info(f"Found {nfiles} files in {time.time() - file_search_start:.2f} seconds")
    logger.info(f"Pixel directory: {pixel_dir}")
    os.makedirs(output_monthly_dir, exist_ok=True)

    if nfiles > 0:

        # Read and concatenate data
        logger.info("Opening files with xarray (lazy loading)...")
        open_start = time.time()
        ds = xr.open_mfdataset(mcsfiles, concat_dim='time', combine='nested')
        open_time = time.time() - open_start
        logger.info(f"Opened {nfiles} files in {open_time:.2f} seconds")
        
        ntimes = ds.sizes['time']
        logger.info(f"Total timesteps: {ntimes}")
        logger.info(f"Dataset shape: {dict(ds.sizes)}")

        # Get coordinates
        coord_start = time.time()
        longitude = ds['longitude'].isel(time=0).compute()
        latitude = ds['latitude'].isel(time=0).compute()
        coord_time = time.time() - coord_start
        logger.info(f"Loaded coordinates in {coord_time:.2f} seconds")
        
        current_memory = get_memory_usage()
        logger.info(f"Current memory usage: {current_memory:.2f} GB (change: +{current_memory - initial_memory:.2f} GB)")

        # Build computation graph (lazy - no computation yet)
        logger.info("Building computation graph (lazy evaluation)...")
        graph_start = time.time()
        
        # Sum MCS counts over time to get number of hours
        mcscloudct = (ds['cloudtracknumber'] > 0).sum(dim='time')

        # Sum MCS precipitation over time, use cloudtracknumber > 0 as mask
        mcsprecip = ds[pcpvarname].where(ds['cloudtracknumber'] > 0).sum(dim='time')

        # Sum total precipitation over time
        totprecip = ds[pcpvarname].sum(dim='time')

        # Convert all MCS track numbers to 1 for summation purpose
        mcspcpmask = xr.where(ds['pcptracknumber'] > 0, 1, 0)
        # Sum MCS PF counts overtime to get the number of hours
        mcspcpct = mcspcpmask.sum(dim='time')
        
        graph_time = time.time() - graph_start
        logger.info(f"Computation graph built in {graph_time:.2f} seconds")
        logger.info("Graph contains lazy operations - no actual data loaded yet")
        
        # Compute Epoch Time for the month
        months = np.zeros(1, dtype=int)
        months[0] = calendar.timegm(datetime.datetime(int(year), int(month), 1, 0, 0, 0, tzinfo=pytz.UTC).timetuple())

        ############################################################################
        # Prepare output dataset
        # NOTE: .data on lazy arrays returns dask arrays (still lazy!)
        # Actual computation happens during to_netcdf() call
        logger.info("="*60)
        logger.info(f"Preparing output dataset: {output_filename}")
        logger.info("Creating variable dictionary with lazy arrays...")
        
        var_dict = {
            'longitude': (['lat', 'lon'], longitude.data, longitude.attrs),
            'latitude': (['lat', 'lon'], latitude.data, latitude.attrs),
            'precipitation': (['time', 'lat', 'lon'], totprecip.expand_dims('time', axis=0).data),
            'mcs_precipitation': (['time', 'lat', 'lon'], mcsprecip.expand_dims('time', axis=0).data),
            'mcs_precipitation_count': (['time', 'lat', 'lon'], mcspcpct.expand_dims('time', axis=0).data),
            'mcs_cloud_count': (['time', 'lat', 'lon'], mcscloudct.expand_dims('time', axis=0).data),
            'ntimes': (['time'], xr.DataArray(ntimes).expand_dims('time', axis=0).data),
        }
        
        logger.info("Variable dictionary created (arrays still lazy)")
        logger.info("="*60)
        
        # Create output dataset
        logger.info("Creating output dataset...")
        coord_dict = {
            'time': (['time'], months),
            'lat': (['lat'], ds['lat'].data),
            'lon': (['lon'], ds['lon'].data),
        }
        gattr_dict = {
            'title': 'MCS precipitation accumulation',
            'contact':'Zhe Feng, zhe.feng@pnnl.gov',
            'created_on':time.ctime(time.time()),
        }
        dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)
        dsout.time.attrs['long_name'] = 'Epoch Time (since 1970-01-01T00:00:00)'
        dsout.time.attrs['units'] = 'Seconds since 1970-1-1 0:00:00 0:00'
        dsout.lon.attrs['long_name'] = 'Longitude'
        dsout.lon.attrs['units'] = 'degree'
        dsout.lat.attrs['long_name'] = 'Latitude'
        dsout.lat.attrs['units'] = 'degree'
        dsout.ntimes.attrs['long_name'] = 'Number of times in the month'
        dsout.ntimes.attrs['units'] = 'count'
        dsout.precipitation.attrs['long_name'] = 'Total precipitation'
        dsout.precipitation.attrs['units'] = 'mm'
        dsout.mcs_precipitation.attrs['long_name'] = 'MCS precipitation'
        dsout.mcs_precipitation.attrs['units'] = 'mm'
        dsout.mcs_precipitation_count.attrs['long_name'] = 'Number of hours MCS precipitation is recorded'
        dsout.mcs_precipitation_count.attrs['units'] = 'hour'
        dsout.mcs_cloud_count.attrs['long_name'] = 'Number of hours MCS cloud is recorded'
        dsout.mcs_cloud_count.attrs['units'] = 'hour'

        fillvalue = np.nan
        # Set encoding/compression for all variables
        comp = dict(zlib=True, _FillValue=fillvalue, dtype='float32')
        encoding = {var: comp for var in dsout.data_vars}

        # Write to NetCDF - THIS IS WHERE COMPUTATION ACTUALLY HAPPENS!
        logger.info("Writing to NetCDF...")
        logger.info("  (Lazy computation will be triggered during write)")
        netcdf_write_start = time.time()
        dsout.to_netcdf(path=output_filename, mode='w', format='NETCDF4', unlimited_dims='time', encoding=encoding)
        netcdf_write_time = time.time() - netcdf_write_start
        
        logger.info(f"NetCDF write completed in {netcdf_write_time:.2f} seconds")
        logger.info(f"  (This includes loading {nfiles} files + computing statistics + writing)")
        
        write_memory = get_memory_usage()
        logger.info(f"Memory after write: {write_memory:.2f} GB (change: +{write_memory - initial_memory:.2f} GB)")
        logger.info(f"Output saved: {output_filename}")

    else:
        logger.warning(f"No files found. Code exits.")
    
    # Final summary
    script_time = time.time() - script_start_time
    final_memory = get_memory_usage()
    memory_change = final_memory - initial_memory
    
    logger.info("\n" + "="*60)
    logger.info("Performance Summary:")
    logger.info(f"  Method: Lazy evaluation (computation during write)")
    logger.info(f"  Total execution time: {format_time(script_time)} (HH:MM:SS)")
    if nfiles > 0:
        logger.info(f"  File open time:       {open_time:.2f} seconds ({open_time/script_time*100:.1f}%)")
        logger.info(f"  Graph building:       {graph_time:.2f} seconds ({graph_time/script_time*100:.1f}%)")
        logger.info(f"  NetCDF write time:    {netcdf_write_time:.2f} seconds ({netcdf_write_time/script_time*100:.1f}%)")
        logger.info(f"    (includes I/O + computation + compression)")
    logger.info(f"  Initial memory:       {initial_memory:.2f} GB")
    logger.info(f"  Final memory:         {final_memory:.2f} GB")
    logger.info(f"  Memory change:        {memory_change:.2f} GB")
    logger.info("="*60)


if __name__ == "__main__":
    main()