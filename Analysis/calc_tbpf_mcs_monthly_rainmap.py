"""
Calculate monthly total, MCS precipitation amount and frequency, save output to a netCDF file.
"""
import numpy as np
import glob, sys, os
import xarray as xr
import time, datetime, calendar, pytz
from pyflextrkr.ft_utilities import load_config

if __name__ == "__main__":

    # Get inputs from command line
    config_file = sys.argv[1]
    year = (sys.argv[2])
    month = (sys.argv[3]).zfill(2)

    # Get inputs from configuration file
    config = load_config(config_file)
    pixel_dir = config['pixeltracking_outpath']
    output_monthly_dir = config['stats_outpath'] + 'monthly/'
    pcpvarname = 'precipitation'

    # Output file name
    output_filename = f'{output_monthly_dir}mcs_rainmap_{year}{month}.nc'

    # Find all pixel files in a month
    mcsfiles = sorted(glob.glob(f'{pixel_dir}/mcstrack_{year}{month}??_????.nc'))
    nfiles = len(mcsfiles)
    print(pixel_dir)
    print(year, month)
    print('Number of files: ', nfiles)
    os.makedirs(output_monthly_dir, exist_ok=True)

    if nfiles > 0:

        # Read and concatinate data
        ds = xr.open_mfdataset(mcsfiles, concat_dim='time', combine='nested')
        print('Finish reading input files.')
        ntimes = ds.dims['time']
        longitude = ds['longitude'].isel(time=0)
        latitude = ds['latitude'].isel(time=0)

        # Sum MCS precipitation over time, use cloudtracknumber > 0 as mask
        mcsprecip = ds[pcpvarname].where(ds['cloudtracknumber'] > 0).sum(dim='time')

        # Sum total precipitation over time
        totprecip = ds[pcpvarname].sum(dim='time')

        # Convert all MCS track number to 1 for summation purpose
        pcpnumber = ds['pcptracknumber'].values
        pcpnumber[pcpnumber > 0] = 1

        # Convert numpy array to DataArray
        mcspcpmask = xr.DataArray(pcpnumber, coords={'time':ds.time, 'lat':ds.lat, 'lon':ds.lon}, dims=['time','lat','lon'])

        # Sum MCS PF counts overtime to get number of hours
        mcspcpct = mcspcpmask.sum(dim='time')

        # Compute Epoch Time for the month
        months = np.zeros(1, dtype=int)
        months[0] = calendar.timegm(datetime.datetime(int(year), int(month), 1, 0, 0, 0, tzinfo=pytz.UTC).timetuple())

        ############################################################################
        # Write output file
        var_dict = {
            'longitude': (['lat', 'lon'], longitude.data, longitude.attrs),
            'latitude': (['lat', 'lon'], latitude.data, latitude.attrs),
            'precipitation': (['time', 'lat', 'lon'], totprecip.expand_dims('time', axis=0).data),
            'mcs_precipitation': (['time', 'lat', 'lon'], mcsprecip.expand_dims('time', axis=0).data),
            'mcs_precipitation_count': (['time', 'lat', 'lon'], mcspcpct.expand_dims('time', axis=0).data),
            'ntimes': (['time'], xr.DataArray(ntimes).expand_dims('time', axis=0).data),
        }
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

        fillvalue = np.nan
        # Set encoding/compression for all variables
        comp = dict(zlib=True, _FillValue=fillvalue, dtype='float32')
        encoding = {var: comp for var in dsout.data_vars}

        dsout.to_netcdf(path=output_filename, mode='w', format='NETCDF4', unlimited_dims='time', encoding=encoding)
        print(f'Output saved: {output_filename}')

    else:
        print(f'No files found. Code exits.')