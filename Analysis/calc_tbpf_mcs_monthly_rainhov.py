"""
Calculate monthly total and MCS precipitation Hovmoller diagram and save output to a netCDF file.
"""
import numpy as np
import glob, sys, os
import xarray as xr
import time
from pyflextrkr.ft_utilities import load_config

if __name__ == "__main__":

    # Get inputs from command line
    config_file = sys.argv[1]
    year = (sys.argv[2])
    month = (sys.argv[3]).zfill(2)
    startlat = float(sys.argv[4])
    endlat = float(sys.argv[5])
    startlon = float(sys.argv[6])
    endlon = float(sys.argv[7])

    # Get inputs from configuration file
    config = load_config(config_file)
    pixel_dir = config['pixeltracking_outpath']
    output_monthly_dir = config['stats_outpath'] + 'monthly/'
    pcpvarname = config['track_field_for_speed']

    # Output file name
    output_filename = f'{output_monthly_dir}mcs_rainhov_{year}{month}.nc'

    # Find all pixel files in a month
    mcsfiles = sorted(glob.glob(f'{pixel_dir}/mcstrack_{year}{month}??_????.nc'))
    print(pixel_dir)
    print(year, month)
    print('Number of files: ', len(mcsfiles))
    os.makedirs(output_monthly_dir, exist_ok=True)

    # Read data
    ds = xr.open_mfdataset(mcsfiles, concat_dim='time', combine='nested')
    print('Finish reading input files.')

    # Mask out non-MCS precipitation as 0 for averaging Hovmoller purpose
    mcspcp = ds[pcpvarname].where(ds['pcptracknumber'] > 0, other=0)

    # Select a latitude band and time period
    mcspreciphov = mcspcp.sel(lat=slice(startlat, endlat), lon=slice(startlon, endlon)).mean(dim='lat')
    totpreciphov = ds[pcpvarname].sel(lat=slice(startlat, endlat), lon=slice(startlon, endlon)).mean(dim='lat')

    # Select time slice matching the chunk
    timehov = ds['time']
    lonhov = ds['lon'].sel(lon=slice(startlon, endlon))

    # Convert xarray decoded time back to Epoch Time in seconds
    basetime = np.array([tt.tolist()/1e9 for tt in ds.time.values])

    ############################################################################
    # Write output file
    print('Writing Hovmoller to netCDF file ...')
    var_dict = {
        'precipitation': (['time', 'lon'], totpreciphov.data),
        'mcs_precipitation': (['time', 'lon'], mcspreciphov.data),
    }
    coord_dict = {
        'lon': (['lon'], lonhov.data),
        'time': (['time'], basetime),
    }
    gattr_dict = {
        'title': 'MCS precipitation Hovmoller',
        'startlat': startlat,
        'endlat': endlat,
        'startlon': startlon,
        'endlon': endlon,
        'contact': 'Zhe Feng, zhe.feng@pnnl.gov',
        'created_on': time.ctime(time.time()),
    }
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)
    dsout.lon.attrs['long_name'] = 'Longitude'
    dsout.lon.attrs['units'] = 'degree'
    dsout.time.attrs['long_name'] = 'Epoch Time (since 1970-01-01T00:00:00)' 
    dsout.time.attrs['units'] = 'seconds since 1970-01-01T00:00:00'
    dsout.precipitation.attrs['long_name'] = 'Total precipitation'
    dsout.precipitation.attrs['units'] = 'mm/h'
    dsout.mcs_precipitation.attrs['long_name'] = 'MCS precipitation'
    dsout.mcs_precipitation.attrs['units'] = 'mm/h'

    fillvalue = np.nan
    # Set encoding/compression for all variables
    comp = dict(zlib=True, _FillValue=fillvalue, dtype='float32')
    encoding = {var: comp for var in dsout.data_vars}

    dsout.to_netcdf(path=output_filename, mode='w', format='NETCDF4', unlimited_dims='time', encoding=encoding)
    print(f'Output saved: {output_filename}')