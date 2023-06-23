"""
Calculate grid-level rain rate PDF by different types of convection for land & ocean and save output to a netCDF file.
Precipitation types: all, MCS, non-MCS deep convection, congestus

>python calc_mcs_rainrate_hist_byregion.py -c config.yml -l landfrac_range -o oceanfrac_range
Optional arguments:
-s start_datetime yyyy-mm-ddThh:mm:ss
-e end_datetime yyyy-mm-ddThh:mm:ss
--extent domain extent lonmin lonmax latmin latmax

Zhe Feng, PNNL
contact: Zhe.Feng@pnnl.gov
"""
import numpy as np
import glob
import xarray as xr
import pandas as pd
import time
import argparse
from pyflextrkr.ft_utilities import load_config

#-----------------------------------------------------------------------
def parse_cmd_args():
    # Define and retrieve the command-line arguments...
    parser = argparse.ArgumentParser(
        description="Calculate rain rate PDF by different types of convection."
    )
    parser.add_argument("-c", "--config", help="yaml config file for tracking", required=True)
    parser.add_argument("-l", "--land", nargs='+', help="land fraction range (min max)", type=float, required=True)
    parser.add_argument("-o", "--ocean", nargs='+', help="ocean fraction range (min max)", type=float, required=True)
    parser.add_argument("-s", "--start", help="first time in time series to plot, format=YYYY-mm-ddTHH:MM:SS", default=None)
    parser.add_argument("-e", "--end", help="last time in time series to plot, format=YYYY-mm-ddTHH:MM:SS", default=None)
    parser.add_argument("--extent", nargs='+', help="domain extent (lonmin lonmax latmin latmax)", type=float, default=None)
    parser.add_argument("--region", help="region name", default="fulldomain")
    args = parser.parse_args()

    # Put arguments in a dictionary
    args_dict = {
        'config_file': args.config,
        'start_datetime': args.start,
        'end_datetime': args.end,
        'extent': args.extent,
        'region': args.region,
        'land': args.land,
        'ocean': args.ocean,
    }

    return args_dict

#-----------------------------------------------------------------------
def calc_pdf(datafiles, outfile, lon_bounds, lat_bounds, rrbins, config):
    status = 0

    # Congestus Tb and rainrate thresholds
    tb_thresh_congestus = 310.0     # K
    rr_thresh_congestus = 0.5       # mm/h

    landmask_file = config['landmask_filename']
    landmask_varname = config['landmask_varname']
    land_range = config['land_range']
    ocean_range = config['ocean_range']

    # Read data
    print(f'Reading input data ...')
    dropvars_list = ['numclouds','pcptracknumber']
    ds = xr.open_mfdataset(datafiles, concat_dim='time', combine='nested', drop_variables=dropvars_list)
    print(f'Finish reading data.')
    # Get input data latitude/longitude arrays
    longitude = ds['longitude'].isel(time=0).load()
    latitude = ds['latitude'].isel(time=0).load()

    # Read landmask
    dslm = xr.open_dataset(landmask_file)
    landmask = dslm[landmask_varname].squeeze().load()

    # Create a mask for the region
    region_mask = (longitude >= min(lon_bounds)) & (longitude <= max(lon_bounds)) & (latitude >= min(lat_bounds)) & (latitude <= max(lat_bounds))
    # Create mask for land & ocean
    l_mask = (landmask >= min(land_range)) & (landmask <= max(land_range))
    o_mask = (landmask >= min(ocean_range)) & (landmask <= max(ocean_range))
    # Combine region and land/ocean masks
    land_mask = (region_mask.data == True) & (l_mask.data == True)
    ocean_mask = (region_mask.data == True) & (o_mask.data == True)
    # Convert to DataArray
    land_mask = xr.DataArray(land_mask, coords={'lat':ds['lat'], 'lon':ds['lon'], }, dims=('lat','lon'))
    ocean_mask = xr.DataArray(ocean_mask, coords={'lat':ds['lat'], 'lon':ds['lon']}, dims=('lat','lon'))


    # Range of rain rate
    rr_range = (np.min(rrbins), np.max(rrbins))

    # Land all precipitation
    totpcp_land = ds['precipitation'].where(land_mask == True, drop=True)
    totpcp_land_pdf, bins = np.histogram(totpcp_land, bins=rrbins, range=rr_range, density=False)
    del totpcp_land

    # Land MCS precipitation
    mcspcp_land = ds['precipitation'].where(ds['cloudtracknumber'] > 0).where(land_mask == True, drop=True)
    mcspcp_land_pdf, bins = np.histogram(mcspcp_land, bins=rrbins, range=rr_range, density=False)
    del mcspcp_land

    # Land Non-MCS deep convection (cloudnumber > 0: CCS & cloudtracknumber == NaN: non-MCS)
    idcpcp_land = ds['precipitation'].where((ds['cloudnumber'] > 0) & (np.isnan(ds['cloudtracknumber']))).where(land_mask == True, drop=True)
    idcpcp_land_pdf, bins = np.histogram(idcpcp_land, bins=rrbins, range=rr_range, density=False)
    del idcpcp_land

    # Land congestus (Tb between CCS and tb_thresh_congestus, rain rate > rr_thresh_congestus)
    congpcp_land = ds['precipitation'].where(np.isnan(ds['cloudnumber']) & (ds['tb'] < tb_thresh_congestus) & (ds['precipitation'] > rr_thresh_congestus)).where(land_mask == True, drop=True)
    congpcp_land_pdf, bins = np.histogram(congpcp_land, bins=rrbins, range=rr_range, density=False)
    del congpcp_land

    # Ocean all precipitation
    totpcp_ocean = ds['precipitation'].where(ocean_mask == True, drop=True)
    totpcp_ocean_pdf, bins = np.histogram(totpcp_ocean, bins=rrbins, range=rr_range, density=False)
    del totpcp_ocean

    # Ocean MCS precipitation
    mcspcp_ocean = ds['precipitation'].where(ds['cloudtracknumber'] > 0).where(ocean_mask == True, drop=True)
    mcspcp_ocean_pdf, bins = np.histogram(mcspcp_ocean, bins=rrbins, range=rr_range, density=False)
    del mcspcp_ocean

    # Ocean Non-MCS deep convection (cloudnumber > 0: CCS & cloudtracknumber == NaN: non-MCS)
    idcpcp_ocean = ds['precipitation'].where((ds['cloudnumber'] > 0) & (np.isnan(ds['cloudtracknumber']))).where(ocean_mask == True, drop=True)
    idcpcp_ocean_pdf, bins = np.histogram(idcpcp_ocean, bins=rrbins, range=rr_range, density=False)
    del idcpcp_ocean

    # Ocean congestus (Tb between CCS and tb_thresh_congestus, rain rate > rr_thresh_congestus)
    congpcp_ocean = ds['precipitation'].where(np.isnan(ds['cloudnumber']) & (ds['tb'] < tb_thresh_congestus) & (ds['precipitation'] > rr_thresh_congestus)).where(ocean_mask == True, drop=True)
    congpcp_ocean_pdf, bins = np.histogram(congpcp_ocean, bins=rrbins, range=rr_range, density=False)
    del congpcp_ocean


    # Define xarray output dataset
    print('Writing output to netCDF file ...')
    var_dict = {
        'total_land': (['bins'], totpcp_land_pdf),
        'mcs_land': (['bins'], mcspcp_land_pdf),
        'idc_land': (['bins'], idcpcp_land_pdf),
        'congestus_land': (['bins'], congpcp_land_pdf),
        'total_ocean': (['bins'], totpcp_ocean_pdf),
        'mcs_ocean': (['bins'], mcspcp_ocean_pdf),
        'idc_ocean': (['bins'], idcpcp_ocean_pdf),
        'congestus_ocean': (['bins'], congpcp_ocean_pdf),
    }
    coord_dict = {'bins': (['bins'], rrbins[:-1])}
    gattr_dict = {
        'title': 'Precipitation PDF by types',
        'lon_bounds': lon_bounds,
        'lat_bounds': lat_bounds,
        'landfrac_range': land_range,
        'oceanfrac_range': ocean_range,
        'contact': 'Zhe Feng, zhe.feng@pnnl.gov',
        'created_on': time.ctime(time.time()),
    }
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

    dsout.bins.attrs['long_name'] = 'Rain rate bins'
    dsout.bins.attrs['units'] = 'mm/h'
    dsout.total_land.attrs['long_name'] = 'Land total precipitation'
    dsout.total_land.attrs['units'] = 'count'
    dsout.mcs_land.attrs['long_name'] = 'Land MCS precipitation'
    dsout.mcs_land.attrs['units'] = 'count'
    dsout.idc_land.attrs['long_name'] = 'Land isolated deep convection precipitation'
    dsout.idc_land.attrs['units'] = 'count'
    dsout.congestus_land.attrs['long_name'] = 'Land congestus precipitation'
    dsout.congestus_land.attrs['units'] = 'count'
    dsout.total_ocean.attrs['long_name'] = 'Ocean total precipitation'
    dsout.total_ocean.attrs['units'] = 'count'
    dsout.mcs_ocean.attrs['long_name'] = 'Ocean MCS precipitation'
    dsout.mcs_ocean.attrs['units'] = 'count'
    dsout.idc_ocean.attrs['long_name'] = 'Ocean isolated deep convection precipitation'
    dsout.idc_ocean.attrs['units'] = 'count'
    dsout.congestus_ocean.attrs['long_name'] = 'Ocean congestus precipitation'
    dsout.congestus_ocean.attrs['units'] = 'count'

    fillvalue = np.nan
    # Set encoding/compression for all variables
    comp = dict(zlib=True, dtype='float')
    encoding = {var: comp for var in dsout.data_vars}
    # Write to file
    dsout.to_netcdf(path=outfile, mode='w', format='NETCDF4', encoding=encoding)
    print('Output saved as: ', outfile)

    status = 1
    return status



if __name__ == "__main__":
    
    # Get the command-line arguments...
    args_dict = parse_cmd_args()
    config_file = args_dict.get('config_file')
    start_datetime = args_dict.get('start_datetime')
    end_datetime = args_dict.get('end_datetime')
    extent = args_dict.get('extent')
    region = args_dict.get('region')
    land_range = args_dict.get('land')
    ocean_range = args_dict.get('ocean')

    # Set up the rain rate bins (linear)
    rrbins = np.arange(1, 201, 1)
    # Set up the rain rate bins (log)
    # rrbins = np.logspace(np.log10(0.01), np.log10(100.0), 100)

    # Get inputs from configuration file
    config = load_config(config_file)

    # If start_datetime or end_datetime is not specified, get it from config
    # That means the entire tracking period is used for the calculation
    if start_datetime is None:
        # Convert Epoch time to Timestamp, then to string ('%Y-%m-%dT%H:%M:%S')
        start_datetime = pd.to_datetime(config['start_basetime'], unit='s').strftime('%Y-%m-%dT%H:%M:%S')
    if end_datetime is None:
        # Convert Epoch time to Timestamp, then to string ('%Y-%m-%dT%H:%M:%S')
        end_datetime = pd.to_datetime(config['end_basetime'], unit='s').strftime('%Y-%m-%dT%H:%M:%S')
    # If extent is not specified, use geolimit from config
    if extent is None:
        geolimits = config['geolimits']
        # geolimits: [lat_min, lon_min, lat_max, lon_max]
        lat_bounds = [geolimits[0], geolimits[2]]
        lon_bounds = [geolimits[1], geolimits[3]]
    else:
        # extent: [lonmin, lonmax, latmin, latmax]
        lon_bounds = [extent[0], extent[1]]
        lat_bounds = [extent[2], extent[3]]
    
    # Get pixel directory, output directory
    pixel_dir = config['pixeltracking_outpath']
    outdir = config['stats_outpath']
    # Add land_range, ocean_range to config
    config['land_range'] = land_range
    config['ocean_range'] = ocean_range

    # Generate time marks within the start/end datetime
    file_datetimes = pd.date_range(start=start_datetime, end=end_datetime, freq='1D').strftime('%Y%m%d')
    # Find all files from these dates
    datafiles = []
    for tt in range(0, len(file_datetimes)):
        datafiles.extend(sorted(glob.glob(f'{pixel_dir}mcstrack_{file_datetimes[tt]}*.nc')))
    print(f'Number of files: {len(datafiles)}')

    # Output filename
    outfile = f'{outdir}mcs_rainrate_hist_{file_datetimes[0]}_{file_datetimes[-1]}_{region}.nc'

    # Call function
    status = calc_pdf(datafiles, outfile, lon_bounds, lat_bounds, rrbins, config)
