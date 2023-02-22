import numpy as np
import time
import os, sys, glob
import logging
from netCDF4 import Dataset
import xarray as xr
import pandas as pd
from wrf import (getvar, vinterp, ALL_TIMES)
from itertools import repeat
from multiprocessing import Pool
from pyflextrkr.ftfunctions import olr_to_tb

def preprocess_wrf(config):
    """
    Preprocess WRF output file to get Tb, rain rate, 
    3D radar reflectivity regridded to constant altitude, melting level height.

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
    outbasename = config['regrid_basename']

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
            status = calc_rainrate_tb_ze(filepairlist[ifile], outdir, inbasename, outbasename, config)
    elif run_parallel == 1:
        # Parallel version
        # Use starmap to unpack the iterables as arguments
        # For arguments that are the same for each iterable, use itertools "repeat" to duplicate those
        # Example follows this stackoverflow post
        # https://stackoverflow.com/questions/5442910/python-multiprocessing-pool-map-for-multiple-arguments
        # Refer to Pool.starmap:
        # https://docs.python.org/dev/library/multiprocessing.html#multiprocessing.pool.Pool.starmap
        pool = Pool(n_workers)
        pool.starmap(calc_rainrate_tb_ze, 
            zip(filepairlist, repeat(outdir), repeat(inbasename), repeat(outbasename), repeat(config)))
        pool.close()
    else:
        sys.exit('Valid parallelization flag not set.')

    return


#---------------------------------------------------------------------------------
def calc_rainrate_tb_ze(filepairnames, outdir, inbasename, outbasename, config):
    """
    Process a pair of WRF output files and write to netCDF

    Args:
        filepairnames: list
            A list of filenames in pair
        outdir: string
            Output file directory
        outbasename: string
            Output file basename
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        fileout: string
            Output filename
    """
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    # Get vertical interpolation levels
    interp_levels = np.asarray(config['interp_levels'], dtype=float)
    
    # Filenames with full path
    filein_t1 = filepairnames[0]
    logger.debug(f'Reading input f1: {filein_t1}')
    filein_t2 = filepairnames[1]
    logger.debug(f'Reading input f2: {filein_t2}')

    # Get the time and data string from filename (wrfout_d0x_yyyy-mm-dd_hh:mm:ss)
    fname_t1 = os.path.basename(filein_t1)
    fname_t2 = os.path.basename(filein_t2)

    # Get basename string position
    idx0 = fname_t1.find(inbasename)
    ftime_t2 = fname_t2[idx0 + len(inbasename):]

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
    nx = ncfile_t1.dimensions['west_east'].size
    ny = ncfile_t1.dimensions['south_north'].size
    nz = ncfile_t1.dimensions['bottom_top'].size

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
    basetimes = np.full(ntimes, np.NAN, dtype=np.float64)
    # Loop over each time
    for tt in range(0, ntimes):
        basetimes[tt] = wrftimes[tt].values.tolist()/1.e9

    # Calculate basetime difference in [seconds]
    delta_times = np.diff(basetimes)

    # Read accumulated grid scale precipitation and OLR
    RAINNC = getvar(wrflist, 'RAINNC', timeidx=ALL_TIMES, method='cat')
    RAINC = getvar(wrflist, 'RAINC', timeidx=ALL_TIMES, method='cat')
    OLR_orig = getvar(wrflist,'OLR', timeidx=ALL_TIMES, method='cat')
    REFL_10CM = getvar(wrflist, 'REFL_10CM', timeidx=ALL_TIMES, method='cat')
    # Get temperature in celsius, model height ASL for mass grid
    TC = getvar(wrflist, 'tc', timeidx=ALL_TIMES, method='cat')
    HASL = getvar(wrflist, 'height', timeidx=ALL_TIMES, method='cat', units="km")
    # Add grid-scale and convective precipitation
    RAINALL = RAINNC + RAINC

    # Create an array to store variables
    rainrate = np.zeros((ntimes-1,ny,nx), dtype=float)
    OLR = np.zeros((ntimes-1,ny,nx), dtype=float)
    nz_regrid = len(interp_levels)
    dbz_regrid = np.zeros((ntimes-1,nz_regrid,ny,nx), dtype=float)
    temperature_c = np.zeros((ntimes-1,nz,ny,nx), dtype=float)
    height_asl = np.zeros((ntimes-1,nz,ny,nx), dtype=float)

    # Loop over all times-1
    for tt in range(0, ntimes-1):
        # Calculate rainrate, convert unit to [mm/h]
        rainrate[tt,:,:] = 3600. * (RAINALL.isel(Time=tt+1).data - RAINALL.isel(Time=tt).data) / delta_times[tt]
        OLR[tt,:,:] = OLR_orig.isel(Time=tt+1).data

        # reflectivity interpolation
        dbz = np.zeros((nz,ny,nx), dtype = float)
        dbz = REFL_10CM.isel(Time=tt+1).data
        dbz_linear = 10.0**(dbz/10.0)
        refl_reg = vinterp(ncfile_t2, field=dbz_linear, vert_coord='ght_msl', interp_levels=interp_levels, extrapolate=False)
        dbz_regrid[tt,:,:,:] = 10.0 * np.log10(refl_reg)
        del dbz, dbz_linear, refl_reg

        temperature_c[tt,:,:,:] = TC.isel(Time=tt+1).data
        height_asl[tt,:,:,:] = HASL.isel(Time=tt+1).data

    # Set nan to -35, which is the default value of WRF
    dbz_regrid[np.isnan(dbz_regrid)] = -35.

    del TC, HASL, REFL_10CM, RAINALL, OLR_orig

    # find melting level heights
    melting_height = get_melting_height(height_asl, temperature_c, ntimes, nx, ny, nz)
    
    # Convert OLR to Tb
    tb = olr_to_tb(OLR)

    # Close WRF files
    ncfile_t1.close()
    ncfile_t2.close()

    # Make a map of good data (model data is all good so set to 1)
    # mask = np.ones((ny,nx))

    # Write single time frame to netCDF output
    for tt in range(0, ntimes-1):

        # Define xarray dataset
        var_dict = {
            'base_time': (['time'], np.expand_dims(basetimes[tt+1], axis=0)),
            # 'Times': (['time','char'], times_char_t1),
            'lon2d': (['lat','lon'], XLONG.data),
            'lat2d': (['lat','lon'], XLAT.data),
            # 'mask': (['lat','lon'], mask),
            'tb': (['time','lat','lon'], np.expand_dims(tb[tt,:,:], axis=0)),
            'reflectivity': (['time',"level",'lat','lon'], np.expand_dims(dbz_regrid[tt,:,:,:], axis=0)),
            'rainrate': (['time','lat','lon'], np.expand_dims(rainrate[tt,:,:], axis=0)),
            'meltinglevelheight': (['time','lat','lon'], np.expand_dims(melting_height[tt,:,:], axis=0)),
        }
        coord_dict = {
            'time': (['time'], np.expand_dims(basetimes[tt+1], axis=0)),
            'level': (['level'], interp_levels),
            # 'char': (['char'], np.arange(0, strlen_t1)),
        }
        gattr_dict = {
            'Title': 'WRF rainrate, brightness temperature, reflectivity and melting level height',
            'contact': 'Zhe Feng: zhe.feng@pnnl.gov',
            'Institution': 'Pacific Northwest National Laboratory',
            'created on': time.ctime(time.time()),
            'Original_File1': filein_t1,
            'Original_File2': filein_t2,
            'DX': DX,
            'DY': DY,
        }
        dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

        # Specify attributes
        dsout['base_time'].attrs['long_name'] = 'Epoch time (seconds since 1970-01-01 00:00:00)'
        dsout['base_time'].attrs['units'] = 'seconds since 1970-01-01 00:00:00'
        dsout['time'].attrs['long_name'] = 'Epoch time (seconds since 1970-01-01 00:00:00)'
        dsout['time'].attrs['units'] = 'seconds since 1970-01-01 00:00:00'
        # dsout['Times'].attrs['long_name'] = 'WRF-based time'
        dsout['level'].attrs['long_name'] = 'reflectivity height'
        dsout['level'].attrs['units'] = 'km'
        dsout['lon2d'].attrs['long_name'] = 'Longitude'
        dsout['lon2d'].attrs['units'] = 'degrees_east'
        dsout['lat2d'].attrs['long_name'] = 'Latitude'
        dsout['lat2d'].attrs['units'] = 'degrees_north'
        # dsout.mask.attrs['long_name'] = 'mask'
        # dsout.mask.attrs['units'] = '1 = good data, 0 = bad'
        dsout['tb'].attrs['long_name'] = 'Brightness temperature'
        dsout['tb'].attrs['units'] = 'Kelvin'
        dsout['reflectivity'].attrs['long_name'] = 'Radar reflectivity (lamda = 10 cm)'
        dsout['reflectivity'].attrs['units'] = 'dBZ'
        dsout['meltinglevelheight'].attrs['long_name'] = 'melting level height ASL'
        dsout['meltinglevelheight'].attrs['units'] = 'km'
        dsout['rainrate'].attrs['long_name'] = 'rainrate'
        dsout['rainrate'].attrs['units'] = 'mm hr-1'

        # Write to netcdf file
        fillvalue = np.nan
        # Set encoding/compression for all variables
        comp = dict(zlib=True, _FillValue=fillvalue, dtype='float32')
        encoding = {var: comp for var in dsout.data_vars}
        # Update base_time variable dtype as 'double' for better precision
        bt_dict = {
            'base_time': {'zlib':True, 'dtype':'float64'},
            'time': {'zlib':True, 'dtype':'float64'},
        }
        encoding.update(bt_dict)
        dsout.to_netcdf(path=fileout, mode='w', format='NETCDF4', unlimited_dims='time', encoding=encoding)

        logger.info(f'Output: {fileout}')
        logger.debug(f"{(time.time() - start_time)} seconds")
        return fileout


#---------------------------------------------------------------------------------
def get_melting_height(height_asl, temperature_c, ntimes, nx, ny, nz):
    """
    Calculate melting level height (written by Jianfeng Li)

    Args:
        height_asl: np.array
            Height above sea level
        temperature_c: np.array
            Temperature in Celcius
        ntimes: int
            Number of times
        nx: int
            Number of points in x-direction
        ny: int
            Number of points in y-direction
        nz: int
            Number of points in z-direction

    Returns:
        melting_height: np.array
            Melting-level height.
    """
    logger = logging.getLogger(__name__)

    # Multiply temperatures between adjacent vertical levels
    temperature_c_adjacent = np.zeros((ntimes - 1, nz - 1, ny, nx), dtype=float)
    temperature_c_adjacent[:, :, :, :] = temperature_c[:, 0:nz - 1, :, :] * temperature_c[:, 1:nz, :, :]
    # Get temperature and height for adjacent levels
    temperature_c_low = np.full((ntimes - 1, nz - 1, ny, nx), np.nan, dtype=float)
    temperature_c_up = np.full((ntimes - 1, nz - 1, ny, nx), np.nan, dtype=float)
    height_asl_low = np.full((ntimes - 1, nz - 1, ny, nx), np.nan, dtype=float)
    height_asl_up = np.full((ntimes - 1, nz - 1, ny, nx), np.nan, dtype=float)
    # Locate points where temperature changes to negative
    # Adjacent layers where temperature are both positive or negative are excluded
    loc = np.where(temperature_c_adjacent <= 0.)
    # Save temperature and height above/below these locations
    temperature_c_low[loc] = (temperature_c[:, 0:nz - 1, :, :])[loc]
    temperature_c_up[loc] = (temperature_c[:, 1:nz, :, :])[loc]
    height_asl_low[loc] = (height_asl[:, 0:nz - 1, :, :])[loc]
    height_asl_up[loc] = (height_asl[:, 1:nz, :, :])[loc]
    # del loc
    # del temperature_c, height_asl
    melting_height = np.full((ntimes - 1, ny, nx), np.nan, dtype=float)

    # Loop over each vertical level
    for zz in range(0, nz - 1):
        if (np.any(~np.isnan(temperature_c_low[:, zz, :, :]))):
            # Get the height and temperature at this vertical level
            height_asl_up_tmp = height_asl_up[:, zz, :, :]
            height_asl_low_tmp = height_asl_low[:, zz, :, :]
            temperature_c_up_tmp = temperature_c_up[:, zz, :, :]
            temperature_c_low_tmp = temperature_c_low[:, zz, :, :]
            # Calculate the difference between levels (up - low)
            tmph = height_asl_up_tmp - height_asl_low_tmp
            tmpc = temperature_c_up_tmp - temperature_c_low_tmp
            # Location where dT = 0
            locconstant = np.where((~np.isnan(tmpc)) & (tmpc == 0.))
            # Location where dT != 0
            locvalid = np.where((~np.isnan(tmpc)) & (tmpc != 0.))
            if (len(locconstant[0]) > 0):
                # Use the height level above
                melting_height[locconstant] = height_asl_up_tmp[locconstant]
            if (len(locvalid[0]) > 0):
                # Interpolate height using dT/dz
                melting_height[locvalid] = height_asl_low_tmp[locvalid] + tmph[locvalid] / tmpc[locvalid] * (
                            0. - temperature_c_low_tmp[locvalid])
            del tmph, tmpc, locconstant, locvalid, height_asl_up_tmp, height_asl_low_tmp, \
                temperature_c_up_tmp, temperature_c_low_tmp
    del temperature_c_adjacent, height_asl_low, height_asl_up, temperature_c_low, temperature_c_up

    #filter out those melting level too high and unrealistic, just in case melting level is in the stratosphere
    #melting_height[np.where((~np.isnan(melting_height)) & (melting_height > 12.))]=np.nan
    return melting_height