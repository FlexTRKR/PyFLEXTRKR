import numpy as np
import time
import os, sys, glob
import logging
import xarray as xr
import pandas as pd
from netCDF4 import Dataset
from wrf import getvar, ALL_TIMES
from itertools import repeat
from multiprocessing import Pool
from pyflextrkr.ftfunctions import olr_to_tb

def preprocess_wrf_tb_rainrate(config):
    """
    Preprocess WRF output file to get Tb and rain rate.

    Args:
        config: dictionary
            Dictionary containing config parameters
    
    Returns:
        None.
    """

    # Set the logging message level
    logger = logging.getLogger(__name__)
    
    # Get inputs from config
    run_parallel = config['run_parallel']
    n_workers = config['nprocesses']
    indir = config['wrfout_path']
    outdir = config['clouddata_path']
    inbasename = config['wrfout_basename']
    outbasename = config['databasename']

    # Create output directory
    os.makedirs(outdir, exist_ok=True)

    # Find all WRF files
    filelist = sorted(glob.glob(f'{indir}/{inbasename}*'))
    nfiles = len(filelist)
    logger.info(f'Number of WRF files: {nfiles}')

    # Create a list with a pair of WRF filenames that are adjacent in time
    filepairlist = []
    for ii in range(0, nfiles-1):
        ipair = [filelist[ii], filelist[ii+1]]
        filepairlist.append(ipair)
    nfilepairs = len(filepairlist)
    
    if run_parallel == 0:
        # Serial version
        for ifile in range(0, nfilepairs):
            status = calc_rainrate_tb(filepairlist[ifile], outdir, inbasename, outbasename)
    elif run_parallel == 1:
        # Parallel version
        # Use starmap to unpack the iterables as arguments
        # For arguments that are the same for each iterable, use itertools "repeat" to duplicate those
        # Example follows this stackoverflow post
        # https://stackoverflow.com/questions/5442910/python-multiprocessing-pool-map-for-multiple-arguments
        # Refer to Pool.starmap:
        # https://docs.python.org/dev/library/multiprocessing.html#multiprocessing.pool.Pool.starmap
        pool = Pool(n_workers)
        pool.starmap(calc_rainrate_tb, zip(filepairlist, repeat(outdir), repeat(inbasename), repeat(outbasename)) )
        pool.close()
    else:
        sys.exit('Valid parallelization flag not set.')
        
    return

    
def calc_rainrate_tb(filepairnames, outdir, inbasename, outbasename):
    """
    Calculates rain rates from a pair of WRF output files and write to netCDF
    ----------
    filepairnames: list
        A list of filenames in pair
    outdir: string
        Output file directory.
    inbasename: string
        Input file basename.
    outbasename: string
        Output file basename.

    Returns
    ----------
    status: 0 or 1
        Returns status = 1 if success.
    """

    logger = logging.getLogger(__name__)
    status = 0
    
    # Filenames with full path
    filein_t1 = filepairnames[0]
    logger.debug(f'Reading input f1: {filein_t1}')

    filein_t2 = filepairnames[1]
    logger.debug(f'Reading input f2: {filein_t2}')

    # Get filename
    fname_t1 = os.path.basename(filein_t1)
    fname_t2 = os.path.basename(filein_t2)

    # Get basename string position
    idx0 = fname_t1.find(inbasename)
    ftime_t2 = fname_t2[idx0+len(inbasename):]

    fileout = f'{outdir}/{outbasename}{ftime_t2}.nc'

    # Read WRF files
    ncfile_t1 = Dataset(filein_t1)
    ncfile_t2 = Dataset(filein_t2)
    # Read time as characters
    times_char_t1 = ncfile_t1.variables['Times'][:]
    times_char_t2 = ncfile_t2.variables['Times'][:]

    # Read WRF lat/lon
    DX = getattr(ncfile_t1, 'DX')
    DY = getattr(ncfile_t1, 'DY')
    XLONG = getvar(ncfile_t1, 'XLONG')
    XLAT = getvar(ncfile_t1, 'XLAT')
    ny, nx = np.shape(XLAT)

    # Create a list with the file pairs
    wrflist = [Dataset(filein_t1), Dataset(filein_t2)]

    # Extract the 'Times' variable
    wrftimes = getvar(wrflist, 'Times', timeidx=ALL_TIMES, method='cat')

    # Convert datetime object to WRF string format
    Times_str_t1 = pd.to_datetime(wrftimes[0].data).strftime('%Y-%m-%d_%H:%M:%S')
    logger.info(Times_str_t1)
    # Get the length of the string
    strlen_t1 = len(Times_str_t1)

    # Convert np.datetime64 to Epoch time in seconds since 1970-01-01T00:00:00 and put into a numpy array
    ntimes = len(wrftimes)
    basetimes = np.full(ntimes, np.NAN, dtype=float)
    # Loop over each time
    for tt in range(0, ntimes):
        basetimes[tt] = wrftimes[tt].values.tolist()/1e9

    # Calculate basetime difference in [seconds]
    delta_times = np.diff(basetimes)

    # Read accumulated precipitation and OLR
    RAINNC = getvar(wrflist, 'RAINNC', timeidx=ALL_TIMES, method='cat')
    RAINC = getvar(wrflist, 'RAINC', timeidx=ALL_TIMES, method='cat')
    OLR_orig = getvar(wrflist, 'OLR', timeidx=ALL_TIMES, method='cat')
    # Add grid-scale and convective precipitation
    RAINALL = RAINNC + RAINC

    # Create an array to store rainrates and brightness temperature
    rainrate = np.zeros((ntimes-1,ny,nx), dtype=float)
    OLR = np.zeros((ntimes-1,ny,nx), dtype = float)

    # Loop over all times-1
    for tt in range(0, ntimes-1):
        # Calculate rainrate, convert unit to [mm/h]
        rainrate[tt,:,:] = 3600. * (RAINALL.isel(Time=tt+1).data - RAINALL.isel(Time=tt).data) / delta_times[tt]
        OLR[tt,:,:] = OLR_orig.isel(Time=tt+1).data

    ncfile_t1.close()
    ncfile_t2.close()
    
    # Convert OLR to IR brightness temperature
    tb = olr_to_tb(OLR)

    # Write single time frame to netCDF output
    for tt in range(0, ntimes-1):

        # Define xarray dataset
        var_dict = {
            'Times': (['time','char'], times_char_t1),
            'lon2d': (['lat','lon'], XLONG.data),
            'lat2d': (['lat','lon'], XLAT.data),
            'tb': (['time','lat','lon'], np.expand_dims(tb[tt,:,:], axis=0)),
            'rainrate': (['time','lat','lon'], np.expand_dims(rainrate[tt,:,:], axis=0)),
        }
        coord_dict = {
            'time': (['time'], np.expand_dims(basetimes[tt], axis=0)),
            'char': (['char'], np.arange(0, strlen_t1)),
        }
        gattr_dict = {
            'Title': 'WRF calculated rainrate and brightness temperature',
            'Contact': 'Zhe Feng: zhe.feng@pnnl.gov',
            'Institution': 'Pacific Northwest National Laboratory',
            'created on': time.ctime(time.time()),
            'Original_File1': filein_t1,
            'Original_File2': filein_t2,
            'DX': DX,
            'DY': DY,
        }
        dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

        # Specify attributes
        dsout['time'].attrs['long_name'] = 'Epoch time (seconds since 1970-01-01 00:00:00)'
        dsout['time'].attrs['units'] = 'seconds since 1970-01-01 00:00:00'
        dsout['time'].attrs['_FillValue'] = np.NaN
        dsout['Times'].attrs['long_name'] = 'WRF-based time'
        dsout['lon2d'].attrs['long_name'] = 'Longitude'
        dsout['lon2d'].attrs['units'] = 'degrees_east'
        dsout['lat2d'].attrs['long_name'] = 'Latitude'
        dsout['lat2d'].attrs['units'] = 'degrees_north'
        dsout['tb'].attrs['long_name'] = 'Brightness temperature'
        dsout['tb'].attrs['units'] = 'K'
        dsout['rainrate'].attrs['long_name'] = 'rainrate'
        dsout['rainrate'].attrs['units'] = 'mm hr-1'

        # Write to netcdf file
        encoding_dict = {
            # 'base_time': {'zlib':True, 'dtype':'int64'},
            'time':{'zlib':True, 'dtype':'float'},
            'Times':{'zlib':True},
            'lon2d':{'zlib':True, 'dtype':'float32'},
            'lat2d':{'zlib':True, 'dtype':'float32'},
            'tb':{'zlib':True, 'dtype':'float32'},
            'rainrate': {'zlib':True, 'dtype':'float32'},
        }
        dsout.to_netcdf(path=fileout, mode='w', format='NETCDF4', unlimited_dims='time', encoding=encoding_dict)
        logger.info(f'{fileout}')
        status = 1
        return (status)




