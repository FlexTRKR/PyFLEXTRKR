def idcell_csapr(input_filename, file_datestring, file_timestring, file_basetime, \
                datasource, datadescription, cloudid_version, \
                dataoutpath, startdate, enddate, \
                pixel_radius, area_thresh, miss_thresh, mincellpix, \
                **kwargs):
    """
    Identifies convective cells from CSAPR data.

    Arguments:
    input_filename - path to raw data directory
    datafiledatestring - string with year, month, and day of data
    datafiletimestring - string with the hour and minute of thedata
    datafilebase - header for the raw data file
    datasource - source of the raw data
    datadescription - description of data source, included in all output file names
    variablename - name of tb data in raw data file
    cloudid_version - version of cloud identification being run, set at the start of the beginning of run_test.py
    dataoutpath - path to destination of the output
    latname - name of latitude variable in raw data file
    longname - name of longitude variable in raw data file
    geolimits - 4-element array with plotting boundaries [lat_min, lon_min, lat_max, lon_max]
    startdate - data to start processing in yyyymmdd format
    enddate - data to stop processing in yyyymmdd format
    pixel_radius - radius of pixels in km
    area_thresh - minimum area thershold to define a feature in km^2
    miss_thresh - minimum amount of data required in order for the file to not to be considered corrupt. 
    mincellpix - minimum size threshold for a cell
    """
    ##########################################################
    # Load modules

    from netCDF4 import Dataset, stringtochar, num2date
    import os
    import numpy as np
    import sys
    import datetime
    import calendar
    import time
    import xarray as xr
    import datetime
    import pandas as pd
    from scipy.signal import medfilt2d
    from scipy.ndimage import label, filters
    from ftfunctions import sort_renumber, sort_renumber2vars
    import netcdf_io as net
    np.set_printoptions(threshold=np.inf)

    ##########################################################
 
    # Read input data
    rawdata = Dataset(input_filename, 'r')
    out_lat = rawdata['latitude'][:]
    out_lon = rawdata['longitude'][:]
    # original_time = rawdata['time'][:]
    # basetime_units = rawdata['time'].units
    comp_ref = rawdata['comp_ref'][:]
    conv_mask_inflated = rawdata['conv_mask_inflated'][:]
    conv_mask1 = rawdata['conv_mask1'][:]
    conv_mask2 = rawdata['conv_mask2'][:]
    rawdata.close()

    # Replace very small reflectivity values with nan
    comp_ref[np.where(comp_ref < -30)] = np.nan

    # Multiply the inflated cell number with conv_mask2 to get the actual cell size without inflation
    conv_mask_noinflate = (conv_mask_inflated * conv_mask2).astype(int)
    # Sort and renumber the cells
    # The number of pixels for each cell is calculated from the cellmask without inflation (conv_mask_sorted_noinflate)
    # Therefore it is the actual size of the cells, but will be different from the inflated mask that is used for tracking
    conv_mask_sorted_noinflate, conv_mask_sorted, conv_npix = sort_renumber2vars(conv_mask_noinflate, conv_mask_inflated, 1)

    # Get number of cells
    nclouds = np.nanmax(conv_mask_sorted)


    #######################################################
    # output data to netcdf file, only if clouds present
    # if nclouds > 0:
    cloudid_outfile = dataoutpath + datasource + '_' + datadescription + '_cloudid' + cloudid_version + '_' + file_datestring + '_' + file_timestring + '.nc'
    print('outcloudidfile: ', cloudid_outfile)
    
    # Check if file exists, if it does delete it
    if os.path.isfile(cloudid_outfile):
        os.remove(cloudid_outfile)

    # Write output to netCDF file
    # net.write_cellid_radar()

    # Put time and nclouds in a numpy array so that they can be set with a time dimension
    out_basetime = np.zeros(1, dtype=float)
    out_basetime[0] = file_basetime
    
    out_nclouds = np.zeros(1, dtype=int)
    out_nclouds[0] = nclouds
    
    # Define variable list
    varlist = {'basetime': (['time'], out_basetime), \
                    # 'filedate': (['time', 'ndatechar'], np.array([stringtochar(np.array(file_datestring))])), \
                    # 'filetime': (['time', 'ntimechar'], np.array([stringtochar(np.array(file_timestring))])), \
                    'latitude': (['lat', 'lon'], out_lat), \
                    'longitude': (['lat', 'lon'], out_lon), \
                    'comp_ref': (['time', 'lat', 'lon'], np.expand_dims(comp_ref, axis=0)), \
                    'conv_mask1': (['time', 'lat', 'lon'], np.expand_dims(conv_mask1, axis=0)), \
                    'conv_mask2': (['time', 'lat', 'lon'], np.expand_dims(conv_mask2, axis=0)), \
                    'convcold_cloudnumber': (['time', 'lat', 'lon'], np.expand_dims(conv_mask_sorted, axis=0)), \
                    'cloudnumber': (['time', 'lat', 'lon'], np.expand_dims(conv_mask_sorted, axis=0)), \
                    'cloudnumber_noinflate': (['time', 'lat', 'lon'], np.expand_dims(conv_mask_sorted_noinflate, axis=0)), \
                    'nclouds': (['time'], out_nclouds), \
                    # 'nclouds': (out_nclouds), \
                    # 'ncorecoldpix': (['time', 'nclouds'], np.expand_dims(conv_npix, axis=0)), \
                    # 'nclouds': (['time'], out_nclouds), \
                    'ncorecoldpix': (['clouds'], conv_npix), \
                    }
    # Define coordinate list
    coordlist = {'time': (['time'], out_basetime), \
                    'lat': (['lat'], np.squeeze(out_lat[:, 0])), \
                    'lon': (['lon'], np.squeeze(out_lon[0, :])), \
                    'clouds': (['clouds'],  np.arange(1, nclouds+1)), \
                    # 'ndatechar': (['ndatechar'], np.arange(0, 32)), \
                    # 'ntimechar': (['ntimechar'], np.arange(0, 16)), \
                }

    # Define global attributes
    gattrlist = {'title': 'Convective cells identified in the data from ' + \
                    file_datestring[0:4] + '/' + file_datestring[4:6] + '/' + file_datestring[6:8] + ' ' + \
                    file_timestring[0:2] + ':' + file_timestring[2:4] + ' UTC', \
                    'institution': 'Pacific Northwest National Laboratory', \
                    'convections': 'CF-1.6', \
                    'contact': 'Zhe Feng, zhe.feng@pnnl.gov', \
                    'created_on': time.ctime(time.time()), \
                    'cloudid_cloud_version': cloudid_version, \
                    'minimum_cloud_area': area_thresh}
    
    # Define xarray dataset
    ds_out = xr.Dataset(varlist, coords=coordlist, attrs=gattrlist)
    # import pdb; pdb.set_trace()
    # Specify variable attributes
    ds_out.time.attrs['long_name'] = 'Base time in Epoch'
    ds_out.time.attrs['units'] = 'Seconds since 1970-1-1 0:00:00 0:00'

    ds_out.basetime.attrs['long_name'] = 'Base time in Epoch'
    ds_out.basetime.attrs['units'] = 'Seconds since 1970-1-1 0:00:00 0:00'

    ds_out.lat.attrs['long_name'] = 'Vector of latitudes, y-coordinate in Cartesian system'
    ds_out.lat.attrs['standard_name'] = 'latitude'
    ds_out.lat.attrs['units'] = 'degrees_north'

    ds_out.lon.attrs['long_name'] = 'Vector of longitudes, x-coordinate in Cartesian system'
    ds_out.lon.attrs['standard_name'] = 'longitude'
    ds_out.lon.attrs['units'] = 'degrees_east'

    ds_out.nclouds.attrs['long_name'] = 'Number of convective cells identified'
    ds_out.nclouds.attrs['units'] = 'unitless'
    
    ds_out.ncorecoldpix.attrs['long_name'] = 'Number of pixels in each convective cells'
    ds_out.ncorecoldpix.attrs['units'] = 'unitless'

    ds_out.latitude.attrs['long_name'] = 'Cartesian grid of latitude'
    ds_out.latitude.attrs['units'] = 'degrees_north'

    ds_out.longitude.attrs['long_name'] = 'Cartesian grid of longitude'
    ds_out.longitude.attrs['units'] = 'degrees_east'

    ds_out.comp_ref.attrs['long_name'] = 'Composite Reflectivity'
    ds_out.comp_ref.attrs['units'] = 'dBZ'
    ds_out.comp_ref.attrs['_FillValue'] = np.nan

    ds_out.conv_mask1.attrs['long_name'] = 'Convective Region Mask After Reflectivity Threshold and Peakedness Steps'
    ds_out.conv_mask1.attrs['units'] = 'unitless'
    ds_out.conv_mask1.attrs['_FillValue'] = 0

    ds_out.conv_mask2.attrs['long_name'] = 'Convective Region Mask After Reflectivity Threshold, Peakedness, and Expansion Steps'
    ds_out.conv_mask2.attrs['units'] = 'unitless'
    ds_out.conv_mask2.attrs['_FillValue'] = 0

    ds_out.convcold_cloudnumber.attrs['long_name'] = 'Grid with each classified cell given a number'
    ds_out.convcold_cloudnumber.attrs['units'] = 'unitless'
    ds_out.convcold_cloudnumber.attrs['_FillValue'] = 0

    ds_out.cloudnumber.attrs['long_name'] = 'Grid with each classified cell given a number'
    ds_out.cloudnumber.attrs['units'] = 'unitless'
    ds_out.cloudnumber.attrs['_FillValue'] = 0
    
    ds_out.cloudnumber_noinflate.attrs['long_name'] = 'Grid with each classified cell given a number without inflation'
    ds_out.cloudnumber_noinflate.attrs['units'] = 'unitless'
    ds_out.cloudnumber_noinflate.attrs['_FillValue'] = 0

    # Specify encoding list
    encodelist = { \
                    # 'time': {'zlib':True, 'units': 'seconds since 1970-01-01'}, \
                    # 'basetime': {'dtype':'float', 'zlib':True, 'units': 'seconds since 1970-01-01'}, \
                    'time': {'zlib':True}, \
                    'basetime': {'zlib':True, 'dtype':'float'}, \
                    'lon': {'zlib':True}, \
                    'lon': {'zlib':True}, \
                    # 'nclouds': {'zlib':True}, \
                    # 'filedate': {'dtype':'str', 'zlib':True}, \
                    # 'filetime': {'dtype':'str', 'zlib':True}, \
                    'longitude': {'zlib':True, '_FillValue':np.nan}, \
                    'latitude': {'zlib':True, '_FillValue':np.nan}, \
                    'comp_ref': {'zlib':True}, \
                    'conv_mask1': {'zlib':True}, \
                    'conv_mask2': {'zlib':True}, \
                    'convcold_cloudnumber': {'zlib':True, 'dtype':'int', }, \
                    'cloudnumber': {'zlib':True, 'dtype':'int'}, \
                    'cloudnumber_noinflate': {'zlib':True, 'dtype':'int'}, \
                    'nclouds': {'dtype':'int', 'zlib':True},  \
                    'ncorecoldpix': {'dtype':'int', 'zlib':True}, \
                    }

    # Write netCDF file
    # ds_out.to_netcdf(path=cloudid_outfile, mode='w', format='NETCDF4_CLASSIC', unlimited_dims='time', encoding=encodelist)
    ds_out.to_netcdf(path=cloudid_outfile, mode='w', format='NETCDF4_CLASSIC', encoding=encodelist)                

    # else:
    #     print(input_filename)
    #     print('No clouds')

    # import pdb; pdb.set_trace()

    return