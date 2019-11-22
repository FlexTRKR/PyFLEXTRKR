import time
import numpy as np
import xarray as xr
from netCDF4 import Dataset, stringtochar, num2date

def write_cloudid_wrf(cloudid_outfile, file_basetime, file_datestring, file_timestring, \
                        out_lat, out_lon, out_ir, \
                        cloudtype, convcold_cloudnumber, cloudnumber, \
                        nclouds, ncorepix, ncoldpix, ncorecoldpix, nwarmpix, \
                        cloudid_version, cloudtb_threshs, geolimits, mintb_thresh, maxtb_thresh, area_thresh, **kwargs):
        """
        Writes cloudid variables to netCDF file.

        **kwargs: 
        Expects these optional arguments:
                precipitation: np.ndarray(float)
                reflectivity: np.ndarray(float) 
                pf_number: np.ndarray(int)
                convcold_cloudnumber_orig: np.ndarray(int)
                cloudnumber_orig: np.ndarray(int)
                linkpf: int
                pf_smooth_window: int
                pf_dbz_thresh: float
                pf_link_area_thresh: float

        """
        #         missing_value_int = -9999

        # Define variable list
        varlist = {'basetime': (['time'], file_basetime), \
                        'filedate': (['time', 'ndatechar'], np.array([stringtochar(np.array(file_datestring))])), \
                        'filetime': (['time', 'ntimechar'], np.array([stringtochar(np.array(file_timestring))])), \
                        'latitude': (['lat', 'lon'], out_lat), \
                        'longitude': (['lat', 'lon'], out_lon), \
                        'tb': (['time', 'lat', 'lon'], np.expand_dims(out_ir, axis=0)), \
                        'cloudtype': (['time', 'lat', 'lon'], cloudtype), \
                        'convcold_cloudnumber': (['time', 'lat', 'lon'], convcold_cloudnumber), \
                        'cloudnumber': (['time', 'lat', 'lon'], cloudnumber), \
                        'nclouds': (['time'], nclouds), \
                        # 'ncorepix': (['time', 'clouds'], ncorepix), \
                        # 'ncoldpix': (['time', 'clouds'], ncoldpix), \
                        'ncorecoldpix': (['time', 'clouds'], ncorecoldpix), \
                        # 'nwarmpix': (['time', 'clouds'], nwarmpix), \
                        }
        # Now check for optional arguments, add them to varlist if provided
        if 'precipitation' in kwargs:
                varlist['precipitation'] = (['time', 'lat', 'lon'], kwargs['precipitation'])
        if 'reflectivity' in kwargs:
                varlist['reflectivity'] = (['time', 'lat', 'lon'], kwargs['reflectivity'])
        if 'pf_number' in kwargs:
                varlist['pf_number'] = (['time', 'lat', 'lon'], kwargs['pf_number'])
        if 'convcold_cloudnumber_orig' in kwargs:
                varlist['convcold_cloudnumber_orig'] = (['time', 'lat', 'lon'], kwargs['convcold_cloudnumber_orig'])
        if 'cloudnumber_orig' in kwargs:
                varlist['cloudnumber_orig'] = (['time', 'lat', 'lon'], kwargs['cloudnumber_orig'])

        # Define coordinate list
        coordlist = {'time': (['time'], file_basetime), \
                        'lat': (['lat'], np.squeeze(out_lat[:, 0])), \
                        'lon': (['lon'], np.squeeze(out_lon[0, :])), \
                        'clouds': (['clouds'],  np.arange(1, nclouds+1)), \
                        'ndatechar': (['ndatechar'], np.arange(0, 32)), \
                        'ntimechar': (['ntimechar'], np.arange(0, 16))}

        # Define global attributes
        gattrlist = {'title': 'Statistics about convective features identified in the data from ' + \
                        file_datestring[0:4] + '/' + file_datestring[4:6] + '/' + file_datestring[6:8] + ' ' + \
                        file_timestring[0:2] + ':' + file_timestring[2:4] + ' UTC', \
                        'institution': 'Pacific Northwest National Laboratory', \
                        'convections': 'CF-1.6', \
                        'contact': 'Katelyn Barber: katelyn.barber@pnnl.gov', \
                        'created_on': time.ctime(time.time()), \
                        'cloudid_cloud_version': cloudid_version, \
                        'tb_threshold_core':  cloudtb_threshs[0], \
                        'tb_threshold_coldanvil': cloudtb_threshs[1], \
                        'tb_threshold_warmanvil': cloudtb_threshs[2], \
                        'tb_threshold_environment': cloudtb_threshs[3], \
                        'minimum_cloud_area': area_thresh}
        # Now check for optional arguments, add them to gattrlist if provided
        if 'linkpf' in kwargs:
                gattrlist['linkpf'] = kwargs['linkpf']
        if 'pf_smooth_window' in kwargs:
                gattrlist['pf_smooth_window'] = kwargs['pf_smooth_window']
        if 'pf_dbz_thresh' in kwargs:
                gattrlist['pf_dbz_thresh'] = kwargs['pf_dbz_thresh']
        if 'pf_link_area_thresh' in kwargs:
                gattrlist['pf_link_area_thresh'] = kwargs['pf_link_area_thresh']

        # Define xarray dataset
        ds_out = xr.Dataset(varlist, coords=coordlist, attrs=gattrlist)

        # Specify variable attributes
        ds_out.time.attrs['long_name'] = 'epoch time (seconds since 01/01/1970 00:00) in epoch of file'

        ds_out.basetime.attrs['long_name'] = 'epoch time (seconds since 01/01/1970 00:00) in epoch of file'

        ds_out.lat.attrs['long_name'] = 'Vector of latitudes, y-coordinate in Cartesian system'
        ds_out.lat.attrs['standard_name'] = 'latitude'
        ds_out.lat.attrs['units'] = 'degrees_north'
        ds_out.lat.attrs['valid_min'] = geolimits[0]
        ds_out.lat.attrs['valid_max'] = geolimits[2]

        ds_out.lon.attrs['long_name'] = 'Vector of longitudes, x-coordinate in Cartesian system'
        ds_out.lon.attrs['standard_name'] = 'longitude'
        ds_out.lon.attrs['units'] = 'degrees_east'
        ds_out.lon.attrs['valid_min'] = geolimits[1]
        ds_out.lon.attrs['valid_max'] = geolimits[2]

        ds_out.clouds.attrs['long_name'] = 'number of distict convective cores identified'
        ds_out.clouds.attrs['units'] = 'unitless'

        ds_out.ndatechar.attrs['long_name'] = 'number of characters in date string'
        ds_out.ndatechar.attrs['units'] = 'unitless'

        ds_out.ntimechar.attrs['long_name'] = 'number of characters in time string'
        ds_out.ntimechar.attrs['units'] = 'unitless'

        ds_out.basetime.attrs['long_name'] = 'epoch time (seconds since 01/01/1970 00:00) of file'
        ds_out.basetime.attrs['standard_name'] = 'time'

        ds_out.filedate.attrs['long_name'] = 'date string of file (yyyymmdd)'
        ds_out.filedate.attrs['units'] = 'unitless'

        ds_out.filetime.attrs['long_name'] = 'time string of file (hhmm)'
        ds_out.filetime.attrs['units'] = 'unitless'

        ds_out.latitude.attrs['long_name'] = 'cartesian grid of latitude'
        ds_out.latitude.attrs['units'] = 'degrees_north'
        ds_out.latitude.attrs['valid_min'] = geolimits[0]
        ds_out.latitude.attrs['valid_max'] = geolimits[2]

        ds_out.longitude.attrs['long_name'] = 'cartesian grid of longitude'
        ds_out.longitude.attrs['units'] = 'degrees_east'
        ds_out.longitude.attrs['valid_min'] = geolimits[1]
        ds_out.longitude.attrs['valid_max'] = geolimits[3]

        ds_out.tb.attrs['long_name'] = 'brightness temperature'
        ds_out.tb.attrs['units'] = 'K'
        ds_out.tb.attrs['valid_min'] = mintb_thresh
        ds_out.tb.attrs['valid_max'] = maxtb_thresh                     

        ds_out.cloudtype.attrs['long_name'] = 'grid of cloud classifications'
        ds_out.cloudtype.attrs['values'] = '1 = core, 2 = cold anvil, 3 = warm anvil, 4 = other'
        ds_out.cloudtype.attrs['units'] = 'unitless'
        ds_out.cloudtype.attrs['valid_min'] = 1
        ds_out.cloudtype.attrs['valid_max'] = 5
        ds_out.cloudtype.attrs['_FillValue'] = 5

        ds_out.convcold_cloudnumber.attrs['long_name'] = 'grid with each classified cloud given a number'
        ds_out.convcold_cloudnumber.attrs['units'] = 'unitless'
        ds_out.convcold_cloudnumber.attrs['valid_min'] = 0
        ds_out.convcold_cloudnumber.attrs['valid_max'] = nclouds+1
        ds_out.convcold_cloudnumber.attrs['comment'] = 'extend of each cloud defined using cold anvil threshold'
        ds_out.convcold_cloudnumber.attrs['_FillValue'] = 0

        ds_out.cloudnumber.attrs['long_name'] = 'grid with each classified cloud given a number'
        ds_out.cloudnumber.attrs['units'] = 'unitless'
        ds_out.cloudnumber.attrs['valid_min'] = 0
        ds_out.cloudnumber.attrs['valid_max'] = nclouds+1
        ds_out.cloudnumber.attrs['comment'] = 'extend of each cloud defined using warm anvil threshold'
        ds_out.cloudnumber.attrs['_FillValue'] = 0

        ds_out.nclouds.attrs['long_name'] = 'number of distict convective cores identified in file'
        ds_out.nclouds.attrs['units'] = 'unitless'

        #     ds_out.ncorepix.attrs['long_name'] = 'number of convective core pixels in each cloud feature'
        #     ds_out.ncorepix.attrs['units'] = 'unitless'

        #     ds_out.ncoldpix.attrs['long_name'] = 'number of cold anvil pixels in each cloud feature'
        #     ds_out.ncoldpix.attrs['units'] = 'unitless'

        ds_out.ncorecoldpix.attrs['long_name'] = 'number of convective core and cold anvil pixels in each cloud feature'
        ds_out.ncorecoldpix.attrs['units'] = 'unitless'

        #     ds_out.nwarmpix.attrs['long_name'] = 'number of warm anvil pixels in each cloud feature'
        #     ds_out.nwarmpix.attrs['units'] = 'unitless'

        # Now check for optional arguments, define attributes if provided
        if 'precipitation' in kwargs:
                ds_out.precipitation.attrs['long_name'] = 'Precipitation'
                ds_out.precipitation.attrs['units'] = 'mm/h'
                ds_out.precipitation.attrs['_FillValue'] = np.nan
        if 'reflectivity' in kwargs:
                ds_out.reflectivity.attrs['long_name'] = 'Radar reflectivity'
                ds_out.reflectivity.attrs['units'] = 'dBZ'
                ds_out.reflectivity.attrs['_FillValue'] = np.nan
        if 'pf_number' in kwargs:
                ds_out.pf_number.attrs['long_name'] = 'Precipitation Feature number'
                ds_out.pf_number.attrs['units'] = 'unitless'
                ds_out.pf_number.attrs['_FillValue'] = 0
        if 'convcold_cloudnumber_orig' in kwargs:
                ds_out.convcold_cloudnumber_orig.attrs['long_name'] = 'Number of cloud system in this file that given pixel belongs to (before linked by pf_number)'
                ds_out.convcold_cloudnumber_orig.attrs['units'] = 'unitless'
                ds_out.convcold_cloudnumber_orig.attrs['_FillValue'] = 0
        if 'cloudnumber_orig' in kwargs:
                ds_out.cloudnumber_orig.attrs['long_name'] = 'Number of cloud system in this file that given pixel belongs to (before linked by pf_number)'
                ds_out.cloudnumber_orig.attrs['units'] = 'unitless'
                ds_out.cloudnumber_orig.attrs['_FillValue'] = 0

        # Specify encoding list
        encodelist = {'time': {'zlib':True, 'units': 'seconds since 1970-01-01'}, \
                        'basetime': {'dtype':'int64', 'zlib':True, 'units': 'seconds since 1970-01-01'}, \
                        'lon': {'zlib':True}, \
                        'lon': {'zlib':True}, \
                        'clouds': {'zlib':True}, \
                        'filedate': {'dtype':'str', 'zlib':True}, \
                        'filetime': {'dtype':'str', 'zlib':True}, \
                        'longitude': {'zlib':True, '_FillValue':np.nan}, \
                        'latitude': {'zlib':True, '_FillValue':np.nan}, \
                        'tb': {'zlib':True, '_FillValue':np.nan}, \
                        'cloudtype': {'zlib':True}, \
                        'convcold_cloudnumber': {'dtype':'int', 'zlib':True}, \
                        'cloudnumber': {'dtype':'int', 'zlib':True}, \
                        'nclouds': {'dtype':'int', 'zlib':True},  \
                        #                         'ncorepix': {'dtype': 'int', 'zlib':True, '_FillValue': -9999},  \
                        #                         'ncoldpix': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                        'ncorecoldpix': {'dtype':'int', 'zlib':True}, \
                        #                         'nwarmpix': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                        }
        # Now check for optional arguments, add them to encodelist if provided
        if 'precipitation' in kwargs:
                encodelist['precipitation'] = {'zlib':True}
        if 'reflectivity' in kwargs:
                encodelist['reflectivity'] = {'zlib':True}
        if 'pf_number' in kwargs:
                encodelist['pf_number'] = {'zlib':True}
        if 'convcold_cloudnumber_orig' in kwargs:
                encodelist['convcold_cloudnumber_orig'] = {'zlib':True}
        if 'cloudnumber_orig' in kwargs:
                encodelist['cloudnumber_orig'] = {'zlib':True}

        # Write netCDF file
        ds_out.to_netcdf(path=cloudid_outfile, mode='w', format='NETCDF4_CLASSIC', unlimited_dims='time', encoding=encodelist)
        # print('Output cloudid file: ' + cloudid_outfile)