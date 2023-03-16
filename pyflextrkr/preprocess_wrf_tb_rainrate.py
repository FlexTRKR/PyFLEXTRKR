import numpy as np
import time
import os, sys, glob
import logging
import xarray as xr
import pandas as pd
import xesmf as xe
# import dask
# from dask.distributed import wait
from itertools import repeat
from multiprocessing import Pool
from pyflextrkr.ft_regrid_func import make_weight_file, make_grid4regridder
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
    regrid_input = config.get('regrid_input', False)

    # Create output directory
    os.makedirs(outdir, exist_ok=True)

    # Find all WRF files
    filelist = sorted(glob.glob(f'{indir}/{inbasename}*'))
    nfiles = len(filelist)
    logger.info(f'Number of WRF files: {nfiles}')

    # Check regridding option
    if regrid_input:
        # logger.info('Regridding input is requested.')
        weight_filename = make_weight_file(filelist[0], config)

    # Create a list with a pair of WRF filenames that are adjacent in time
    filepairlist = []
    for ii in range(0, nfiles-1):
        ipair = [filelist[ii], filelist[ii+1]]
        filepairlist.append(ipair)
    nfilepairs = len(filepairlist)
    
    if run_parallel == 0:
        # Serial version
        for ifile in range(0, nfilepairs):
            status = calc_rainrate_tb(filepairlist[ifile], outdir, inbasename, outbasename, config)
    elif run_parallel == 1:
        # Parallel version
        # Dask
        # results = []
        # for ifile in range(0, nfilepairs):
        #     result = dask.delayed(calc_rainrate_tb)(filepairlist[ifile], outdir, inbasename, outbasename, config)
        #     results.append(result)
        # final_result = dask.compute(*results)
        # wait(final_result)

        # Use starmap to unpack the iterables as arguments
        # For arguments that are the same for each iterable, use itertools "repeat" to duplicate those
        # Example follows this stackoverflow post
        # https://stackoverflow.com/questions/5442910/python-multiprocessing-pool-map-for-multiple-arguments
        # Refer to Pool.starmap:
        # https://docs.python.org/dev/library/multiprocessing.html#multiprocessing.pool.Pool.starmap
        pool = Pool(n_workers)
        pool.starmap(
            calc_rainrate_tb, 
            zip(filepairlist, repeat(outdir), repeat(inbasename), repeat(outbasename), repeat(config)),
        )
        pool.close()
    else:
        sys.exit('Valid parallelization flag not set.')
        
    return

    
def calc_rainrate_tb(filepairnames, outdir, inbasename, outbasename, config):
    """
    Calculates rain rates from a pair of WRF output files and write to netCDF

    Args:
        filepairnames: list
            A list of filenames in pair
        outdir: string
            Output file directory.
        inbasename: string
            Input file basename.
        outbasename: string
            Output file basename.
        config: dictionary
            Dictionary containing config parameters

    Returns:
        status: 0 or 1
            Returns status = 1 if success.
    """

    logger = logging.getLogger(__name__)
    status = 0

    regrid_input = config.get('regrid_input', False)
    time_dimname = config.get('time_dimname', 'time')
    x_dimname = config.get('x_dimname', 'lon')
    y_dimname = config.get('y_dimname', 'lat')
    x_coordname = config.get('x_coordname', 'longitude')
    y_coordname = config.get('y_coordname', 'latitude')
    tb_varname = config.get('tb_varname', 'tb')
    pcp_varname = config.get('pcp_varname', 'rainrate')
    
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
    # Output filename
    fileout = f'{outdir}/{outbasename}{ftime_t2}.nc'

    # Read in WRF data files
    ds_in = xr.open_mfdataset([filein_t1, filein_t2], concat_dim='Time', combine='nested')
    Times = ds_in['Times'].load()
    XLONG = ds_in['XLONG'][0,:,:].squeeze()
    XLAT = ds_in['XLAT'][0,:,:].squeeze()
    ny, nx = np.shape(XLAT)
    DX = ds_in.attrs['DX']
    DY = ds_in.attrs['DY']
    
    ntimes = len(Times)
    Times_str = []
    basetimes = np.full(ntimes, np.NAN, dtype=float)
    for tt in range(0, ntimes):
        # Decode bytes to string with UTF-8 encoding, then replace "_" with "T"
        # to make time string: YYYY-MO-DDTHH:MM:SS
        tstring = Times[tt].item().decode("utf-8").replace("_", "T")
        Times_str.append(tstring)
        # Convert time string to Epoch time
        basetimes[tt] = pd.to_datetime(tstring).timestamp()
    # wrfdatetimes = pd.to_datetime(Times_str)

    # Calculate basetime difference in [seconds]
    delta_times = np.diff(basetimes)

    # Read accumulated precipitation and OLR
    RAINNC = ds_in['RAINNC'].load()
    RAINC = ds_in['RAINC'].load()
    OLR_orig = ds_in['OLR'].load()
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

    # Check regridding option
    if regrid_input:
        logger.info('Regridding input is requested.')

        weight_filename = config.get('weight_filename')
        regrid_method = config.get('regrid_method', 'conservative')

        # Make source & destination grid data for regridder
        grid_src, grid_dst = make_grid4regridder(filein_t1, config)
        # Retrieve Regridder
        regridder = xe.Regridder(grid_src, grid_dst, method=regrid_method, weights=weight_filename)

        # Make output coordinates
        x_coord_dst = grid_dst['lon']
        y_coord_dst = grid_dst['lat']
        if (x_coord_dst.ndim == 1) & (y_coord_dst.ndim == 1):
            x_coord_out, y_coord_out = np.meshgrid(x_coord_dst, y_coord_dst)
        else:
            x_coord_out = x_coord_dst
            y_coord_out = y_coord_dst
        # Regrid variables
        OLR_out = regridder(OLR)
        rainrate_out = regridder(rainrate)
    else:
        x_coord_out = XLONG.data
        y_coord_out = XLAT.data
        OLR_out = OLR
        rainrate_out = rainrate

    ds_in.close()
    
    # Convert OLR to IR brightness temperature
    tb_out = olr_to_tb(OLR_out)

    # Write single time frame to netCDF output
    for tt in range(0, ntimes-1):

        # Use the next time to be consitent with output filename
        _bt = basetimes[tt+1]
        _TimeStr = Times_str[tt+1]

        # Define xarray dataset
        var_dict = {
            # 'Times': ([time_dimname,'char'], times_char_t1),
            x_coordname: ([y_dimname,x_dimname], x_coord_out),
            y_coordname: ([y_dimname,x_dimname], y_coord_out),
            tb_varname: ([time_dimname,y_dimname,x_dimname], np.expand_dims(tb_out[tt,:,:], axis=0)),
            pcp_varname: ([time_dimname,y_dimname,x_dimname], np.expand_dims(rainrate_out[tt,:,:], axis=0)),
        }
        coord_dict = {
            time_dimname: ([time_dimname], np.expand_dims(_bt, axis=0)),
            # 'char': (['char'], np.arange(0, strlen_t1)),
        }
        gattr_dict = {
            'Title': 'WRF calculated rainrate and brightness temperature',
            'Contact': 'Zhe Feng: zhe.feng@pnnl.gov',
            'Institution': 'Pacific Northwest National Laboratory',
            'created on': time.ctime(time.time()),
            'Original_File1': filein_t1,
            'Original_File2': filein_t2,
            # 'DX': DX,
            # 'DY': DY,
        }
        dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

        # Specify attributes
        dsout[time_dimname].attrs['long_name'] = 'Epoch time (seconds since 1970-01-01 00:00:00)'
        dsout[time_dimname].attrs['units'] = 'seconds since 1970-01-01 00:00:00'
        # dsout[time_dimname].attrs['_FillValue'] = np.NaN
        dsout[time_dimname].attrs['tims_string'] = _TimeStr
        # dsout['Times'].attrs['long_name'] = 'WRF-based time'
        dsout[x_coordname].attrs['long_name'] = 'Longitude'
        dsout[x_coordname].attrs['units'] = 'degrees_east'
        dsout[y_coordname].attrs['long_name'] = 'Latitude'
        dsout[y_coordname].attrs['units'] = 'degrees_north'
        dsout[tb_varname].attrs['long_name'] = 'Brightness temperature'
        dsout[tb_varname].attrs['units'] = 'K'
        dsout[pcp_varname].attrs['long_name'] = 'Precipitation rate'
        dsout[pcp_varname].attrs['units'] = 'mm hr-1'

        # Write to netcdf file
        encoding_dict = {
            time_dimname:{'zlib':True, 'dtype':'float'},
            # 'Times':{'zlib':True},
            x_coordname:{'zlib':True, 'dtype':'float32'},
            y_coordname:{'zlib':True, 'dtype':'float32'},
            tb_varname:{'zlib':True, 'dtype':'float32'},
            pcp_varname: {'zlib':True, 'dtype':'float32'},
        }
        dsout.to_netcdf(path=fileout, mode='w', format='NETCDF4', unlimited_dims=time_dimname, encoding=encoding_dict)
        logger.info(f'{fileout}')
        status = 1
        return (status)




