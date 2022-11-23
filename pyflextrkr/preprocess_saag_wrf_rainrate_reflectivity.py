import os
import sys
import glob
import logging
import numpy as np
import xarray as xr
import pandas as pd
import dask
# from dask.distributed import Client, LocalCluster
from pyflextrkr.ft_utilities import load_config, setup_logging

if __name__ == "__main__":

    # Set the logging message level
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load configuration file
    config_file = sys.argv[1]
    config = load_config(config_file)
    # Get inputs from config
    run_parallel = config['run_parallel']
    n_workers = config['nprocesses']
    indir = config['wrfout_path']
    outdir = config['clouddata_path']
    inbasename = config['wrfout_basename']
    outbasename = config['databasename']
    time_format = config['time_format']
    x_dimname = config['x_dimname']
    y_dimname = config['y_dimname']
    t_dimname = config['time_dimname']
    x_varname = config['x_varname']
    y_varname = config['y_varname']
    terrain_file = config['terrain_file']
    outfile_freq = '15min'
    # outfile_freq = '1H'

    # Create output directory
    os.makedirs(outdir, exist_ok=True)

    # Find all WRF files
    filelist = sorted(glob.glob(f'{indir}/{inbasename}*'))
    nfiles = len(filelist)

    # Test only a few files
    filelist = filelist[60:62]
    nfiles = len(filelist)
    logger.info(f'Number of WRF files: {nfiles}')

    # # Create a list with a pair of WRF filenames that are adjacent in time
    # filepairlist = []
    # for ii in range(0, nfiles-1):
    #     ipair = [filelist[ii], filelist[ii+1]]
    #     filepairlist.append(ipair)
    # nfilepairs = len(filepairlist)

    # Read grid data
    dsg = xr.open_dataset(terrain_file)
    lon = dsg[x_varname].squeeze()
    lat = dsg[y_varname].squeeze()

    # Read input data
    drop_vars = ['Q2', 'T2', 'U10', 'V10', 'OLR', 'AFWA_CAPE_MU', 'AFWA_CIN_MU', 'C1H', 'C2H', 'C3H', 'C4H', 'C1F', 'C2F', 'C3F', 'C4F']
    ds = xr.open_mfdataset(filelist, concat_dim='Time', combine='nested', drop_variables=drop_vars)
    # Rename time dimension name
    ds = ds.rename({'Time':t_dimname})
    ntime = ds.sizes[t_dimname]
    Times = ds['Times'].load()

    # Calculate rain amount in [mm], then divide by timedelta to get [mm/h]
    total_rain = (ds['RAINNC'] + ds['I_RAINNC'] * 100).diff(t_dimname)
    # Drop variables
    ds = ds.drop_vars(['RAINNC', 'I_RAINNC'])

    # Convert time string to DatetimeIndex
    ftimes = []
    for ii in range(0, ntime):
        # Decode bytes to string
        time_str = Times[ii].item().decode('utf-8')
        ftimes.append(pd.to_datetime(time_str, format='%Y-%m-%d_%H:%M:%S'))
    # Convert list to Pandas DatetimeIndex
    ftimes = pd.to_datetime(ftimes)
    # Get time difference in hours
    timediff = (ftimes.to_series().diff()[1:] / pd.Timedelta(hours=1)).values
    # Convert to DataArray
    ftimes = xr.DataArray(ftimes, coords={t_dimname:ds[t_dimname]}, dims=(t_dimname))

    # Calculate mean/median time difference (they should be the same)
    timediff_avg = np.nanmean(timediff)
    timediff_med = np.nanmedian(timediff)
    if timediff_avg != timediff_med:
        logger.warning(f'Time difference may not be the same for all times: mean ({timediff_avg}), median ({timediff_med})!')

    # Drop variable 'Times'
    ds = ds.drop_vars('Times')
    # Replace time coordinate in the dataset
    ds[t_dimname] = ftimes

    # Remove the first time
    ds = ds.isel(time=slice(1, ntime))
    t_coord = ds[t_dimname]

    # Calculate rain rate in [mm/h]
    rainrate = total_rain / timediff_med
    # Convert rain rate to DataArray by assigning coordinates & dimensions
    coords = {t_dimname:t_coord, y_varname:lat, x_varname:lon}
    rain_rate = xr.DataArray(rainrate, coords=coords, dims=(t_dimname, y_dimname, x_dimname))
    rr_attrs = {
        'description': 'Rain rate',
        'units': 'mm/h',
    }
    # Add rainrate to dataset
    ds['rainrate'] = rain_rate
    ds['rainrate'].attrs = rr_attrs

    # Resample dataset for output
    dates, datasets = zip(*ds.resample(time=outfile_freq))
    # Get number of days
    ndates = len(datasets)

    # Create a list of output filenames
    # Output file datetime format: yyyy-mm-dd-hh_mm_ss
    filenames_out = []
    for idate in range(0, ndates):
        filenames_out.append(outdir + outbasename + datasets[idate].indexes[t_dimname][0].strftime('%Y-%m-%d_%H_%M_%S') + '.nc')

    # Set encoding/compression for all variables & coordinates
    comp = dict(zlib=True, dtype='float32')
    encoding = {var: comp for var in datasets[0].data_vars}
    encoding.update({var: comp for var in datasets[0].coords})
    kwargs = {'encoding':encoding, 'format':'NETCDF4', 'unlimited_dims':t_dimname}

    # Write to netCDF
    dsout = xr.save_mfdataset(datasets, filenames_out, compute=False, **kwargs)

    # Execute compute
    logger.info("Writing out files ... ")
    dsout.compute()

    # import pdb; pdb.set_trace()