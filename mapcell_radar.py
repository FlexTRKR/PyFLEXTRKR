# Purpose: Take the cell tracks identified in the previous steps and create pixel level maps of these cells. One netcdf file is create for each time step.

# Author: Zhe Feng (zhe.feng@pnnl.gov)

# def mapcell_radar(zipped_inputs):
def mapcell_radar(cloudid_filename, filebasetime, stats_path, statistics_filebase, \
                    startdate, enddate, out_path, out_filebase):
    # Inputs:
    # cloudid_filebase - file header of the cloudid file create in the first step
    # filebasetime - seconds since 1970-01-01 of the file being processed
    # statistics_filebase - file header for the all track statistics file generated in the trackstats step
    # out_path - directory where cell tracks maps generated in this step will be placed
    # stats_path - directory that contains the statistics files
    # startdate - starting date and time of the full dataset
    # enddate - ending date and time of the full dataset

    #######################################################################
    # Import modules
    import numpy as np
    import time
    import os
    import sys
    import xarray as xr
    import pandas as pd
    from netCDF4 import Dataset, num2date
    np.set_printoptions(threshold=np.inf)

    ######################################################################
    # define constants

    ###################################################################
    # Load track stats file
    statistics_file = stats_path + statistics_filebase + startdate + '_' + enddate + '.nc'

    allstatdata = Dataset(statistics_file, 'r')
    trackstat_basetime = allstatdata['basetime'][:] # Time of cloud in seconds since 01/01/1970 00:00
    trackstat_cloudnumber = allstatdata['cloudnumber'][:] # Number of the corresponding cloudid file
    trackstat_status = allstatdata['status'][:] # Flag indicating the status of the cloud
    trackstat_mergenumbers = allstatdata['mergenumbers'][:] # Track number that it merges into
    trackstat_splitnumbers = allstatdata['splitnumbers'][:]
    datasource = allstatdata.getncattr('source')
    datadescription = allstatdata.getncattr('description')
    allstatdata.close()
    
    #########################################################################
    # Get cloudid file associated with this time
    file_datetime = time.strftime("%Y%m%d_%H%M", time.gmtime(np.copy(filebasetime)))
    filedate = np.copy(file_datetime[0:8])
    filetime = np.copy(file_datetime[9:14])
    # print(('cloudid file: ' + cloudid_filename))
    
    # Load cloudid data
    cloudiddata = Dataset(cloudid_filename, 'r')
    cloudid_cloudnumber = cloudiddata['cloudnumber'][:]
    # cloudid_cloudnumber_noinflate = cloudiddata['cloudnumber_noinflate'][:]
    cloudid_basetime = cloudiddata['basetime'][:]
    basetime_units =  cloudiddata['basetime'].units
    # basetime_calendar = cloudiddata['basetime'].calendar
    longitude = cloudiddata['longitude'][:]
    latitude = cloudiddata['latitude'][:]
    nclouds = cloudiddata['nclouds'][:]
    comp_ref = cloudiddata['comp_ref'][:]
    conv_core = cloudiddata['conv_core'][:]
    conv_mask = cloudiddata['conv_mask'][:]
    # convcold_cloudnumber = cloudiddata['convcold_cloudnumber'][:]
    cloudiddata.close()
    
    # cloudid_cloudnumber = cloudid_cloudnumber.astype(np.int32)
    # cloudid_cloudnumber_noinflate = cloudid_cloudnumber_noinflate.astype(np.int32)
    comp_ref = comp_ref.data
    conv_core = conv_core.data
    conv_mask = conv_mask.data
    cloudid_cloudnumber = cloudid_cloudnumber.data
    # cloudid_cloudnumber_noinflate = cloudid_cloudnumber_noinflate.data
    
    # Create a binary conv_mask (remove the cell number)
    conv_mask_binary = conv_mask > 0

    # Get data dimensions
    [timeindex, ny, nx] = np.shape(cloudid_cloudnumber)
    
    ##############################################################
    # Intiailize track maps
    celltrackmap = np.zeros((1, ny, nx), dtype=int)
    celltrackmap_mergesplit = np.zeros((1, ny, nx), dtype=int)
        
    cellmergemap = np.zeros((1, ny, nx), dtype=int)
    cellsplitmap = np.zeros((1, ny, nx), dtype=int)
    
    ################################################################
    # Create map of status and track number for every feature in this file
    print('Create maps of all tracks')
    fillval = -9999
    statusmap = np.ones((1, ny, nx), dtype=int)*fillval
    trackmap = np.zeros((1, ny, nx), dtype=int)
    allmergemap = np.zeros((1, ny, nx), dtype=int)
    allsplitmap = np.zeros((1, ny, nx), dtype=int)

    # Find matching time from the trackstats_basetime
    itrack, itime = np.array(np.where(trackstat_basetime == cloudid_basetime))
    # If a match is found, that means there are tracked cells at this time
    # Proceed and lebel them
    ntimes = len(itime)
    if ntimes > 0:

        # Loop over each instance matching the trackstats time
        for jj in range(0, ntimes):
            # Get cloud number
            jjcloudnumber = trackstat_cloudnumber[itrack[jj], itime[jj]]
            jjstatus = trackstat_status[itrack[jj], itime[jj]]

            # Find pixels matching this cloud number
            jjcloudypixels, jjcloudxpixels = np.array(np.where(cloudid_cloudnumber[0, :, :] == jjcloudnumber))
            # Label this cloud with the track number. 
            # Need to add one to the cloud number since have the index number and we want the track number
            if len(jjcloudypixels) > 0:
                trackmap[0, jjcloudypixels, jjcloudxpixels] = itrack[jj] + 1
                statusmap[0, jjcloudypixels, jjcloudxpixels] = jjstatus                
                # import pdb; pdb.set_trace()
            else:
                sys.exit('Error: No matching cloud pixel found?!')

        # Get cloudnumbers and split cloudnumbers within this time
        jjcloudnumber = trackstat_cloudnumber[itrack, itime]
        jjallsplit = trackstat_splitnumbers[itrack, itime]
        # Count valid split cloudnumbers (> 0)
        splitpresent = np.count_nonzero(jjallsplit > 0)
        # splitpresent = len(np.array(np.where(np.isfinite(jjallsplit)))[0, :])
        if splitpresent > 0:
            # splittracks = np.copy(jjallsplit[np.where(np.isfinite(jjallsplit))])
            # splitcloudid = np.copy(jjcloudnumber[np.where(np.isfinite(jjallsplit))])
            # Find valid split cloudnumbers (> 0)
            splittracks = jjallsplit[np.where(jjallsplit > 0)]
            splitcloudid = jjcloudnumber[np.where(jjallsplit > 0)]
            if len(splittracks) > 0:
                for isplit in range(0, len(splittracks)):
                    splitypixels, splitxpixels = np.array(np.where(cloudid_cloudnumber[0, :, :] == splitcloudid[isplit]))
                    allsplitmap[0, splitypixels, splitxpixels] = splittracks[isplit]

        # Get cloudnumbers and merg cloudnumbers within this time
        jjallmerge = trackstat_mergenumbers[itrack, itime]
        # Count valid split cloudnumbers (> 0)
        mergepresent = np.count_nonzero(jjallmerge > 0)
        # mergepresent = len(np.array(np.where(np.isfinite(jjallmerge)))[0, :])
        if mergepresent > 0:
            # mergetracks = np.copy(jjallmerge[np.where(np.isfinite(jjallmerge))])
            # mergecloudid = np.copy(jjcloudnumber[np.where(np.isfinite(jjallmerge))])
            # Find valid merge cloudnumbers (> 0)
            mergetracks = jjallmerge[np.where(jjallmerge > 0)]
            mergecloudid = jjcloudnumber[np.where(jjallmerge > 0)]
            if len(mergetracks) > 0:
                for imerge in range(0, len(mergetracks)):
                    mergeypixels, mergexpixels = np.array(np.where(cloudid_cloudnumber[0, :, :] == mergecloudid[imerge]))
                    allmergemap[0, mergeypixels, mergexpixels] = mergetracks[imerge]

        trackmap = trackmap.astype(np.int32)
        allmergemap = allmergemap.astype(np.int32)
        allsplitmap = allsplitmap.astype(np.int32)

        # Multiply the tracknumber map with conv_mask to get the actual cell size without inflation
        # trackmap_cmask2 = (trackmap * conv_mask2).astype(np.int32)
        trackmap_cmask2 = (trackmap * conv_mask_binary).astype(np.int32)

    else:
        trackmap_cmask2 = trackmap

   
    #####################################################################
    # Output maps to netcdf file

    # Define output fileame
    celltrackmaps_outfile = out_path + out_filebase + str(filedate) + '_' + str(filetime) + '.nc'
    
    # Check if file already exists. If exists, delete
    if os.path.isfile(celltrackmaps_outfile):
        os.remove(celltrackmaps_outfile)
    
    # Define variable list
    varlist = {'basetime': (['time'], cloudid_basetime), \
                # 'basetime': (['time'], np.array([pd.to_datetime(num2date(cloudid_basetime, units=basetime_units, calendar=basetime_calendar))], dtype='datetime64[s]')[0, :]),  \
                'longitude': (['lat', 'lon'], longitude), \
                'latitude': (['lat', 'lon'], latitude), \
                'nclouds': (['time'], nclouds), \
                'comp_ref': (['time', 'lat', 'lon'], comp_ref), \
                'conv_core': (['time', 'lat', 'lon'], conv_core), \
                'conv_mask': (['time', 'lat', 'lon'], conv_mask), \
                'tracknumber': (['time', 'lat', 'lon'], trackmap), \
                'tracknumber_cmask2': (['time', 'lat', 'lon'], trackmap_cmask2), \
                'cloudstatus': (['time', 'lat', 'lon'], statusmap), \
                'cloudnumber': (['time', 'lat', 'lon'], cloudid_cloudnumber), \
                # 'cloudnumber_noinflate': (['time', 'lat', 'lon'], cloudid_cloudnumber_noinflate), \
                'mergecloudnumber': (['time', 'lat', 'lon'], allmergemap), \
                'splitcloudnumber': (['time', 'lat', 'lon'], allsplitmap), \
                # 'tracknumber': (['time', 'lat', 'lon'], trackmap_mergesplit), \
                # 'cellsplittracknumbers': (['time', 'lat', 'lon'], cellsplitmap), \
                # 'cellmergetracknumbers': (['time', 'lat', 'lon'], cellmergemap), \
                }
    
    # Define coordinate list
    coordlist = {'time': (['time'], cloudid_basetime), \
                    'lat': (['lat'], np.arange(0, ny)), \
                    'lon': (['lon'], np.arange(0, nx))}

    # Define global attributes
    gattrlist = {'title':'Pixel level of tracked cells', \
                    'source': datasource, \
                    'description': datadescription, \
                    # 'Main_cell_duration_hr': durationthresh, \
                    # 'Merger_duration_hr': mergethresh, \
                    # 'Split_duration_hr': splitthresh, \
                    'contact':'Zhe Feng, zhe.feng@pnnl.gov', \
                    'created_on':time.ctime(time.time())}
    
    # Define xarray dataset
    ds_out = xr.Dataset(varlist, coords=coordlist, attrs=gattrlist)
    
    # Specify variable attributes
    ds_out.time.attrs['long_name'] = 'epoch time (seconds since 01/01/1970 00:00) in epoch of file'
    ds_out.time.attrs['units'] = basetime_units

    ds_out.basetime.attrs['long_name'] = 'Epoch time (seconds since 01/01/1970 00:00) of this file'
    ds_out.basetime.attrs['units'] = basetime_units
    
    ds_out.longitude.attrs['long_name'] = 'Grid of longitude'
    ds_out.longitude.attrs['units'] = 'degrees'
    
    ds_out.latitude.attrs['long_name'] = 'Grid of latitude'
    ds_out.latitude.attrs['units'] = 'degrees'
    
    ds_out.nclouds.attrs['long_name'] = 'Number of cells identified in this file'
    ds_out.nclouds.attrs['units'] = 'unitless'
    
    ds_out.comp_ref.attrs['long_name'] = 'Composite reflectivity'
    ds_out.comp_ref.attrs['units'] = 'dBZ'
    
    ds_out.conv_core.attrs['long_name'] = 'Convective Core Mask After Reflectivity Threshold and Peakedness Steps'
    ds_out.conv_core.attrs['units'] = 'unitless'
    ds_out.conv_core.attrs['_FillValue'] = 0

    ds_out.conv_mask.attrs['long_name'] = 'Convective Region Mask After Reflectivity Threshold, Peakedness, and Expansion Steps'
    ds_out.conv_mask.attrs['units'] = 'unitless'
    ds_out.conv_mask.attrs['_FillValue'] = 0

    ds_out.tracknumber.attrs['long_name'] = 'Track number in this file at a given pixel'
    ds_out.tracknumber.attrs['units'] = 'unitless'
    ds_out.tracknumber.attrs['_FillValue'] = 0

    ds_out.tracknumber_cmask2.attrs['long_name'] = 'Track number (conv_mask) in this file at a given pixel'
    ds_out.tracknumber_cmask2.attrs['units'] = 'unitless'
    ds_out.tracknumber_cmask2.attrs['_FillValue'] = 0

    ds_out.cloudstatus.attrs['long_name'] = 'Flag indicating history of cloud'
    ds_out.cloudstatus.attrs['units'] = 'unitless'
    ds_out.cloudstatus.attrs['valid_min'] = 0
    ds_out.cloudstatus.attrs['valid_max'] = 65
    ds_out.cloudstatus.attrs['_FillValue'] = fillval

    ds_out.cloudnumber.attrs['long_name'] = 'Number associated with the cloud at a given pixel'
    ds_out.cloudnumber.attrs['units'] = 'unitless'
    ds_out.cloudnumber.attrs['_FillValue'] = 0
    # ds_out.cloudnumber.attrs['valid_min'] = 0
    # ds_out.cloudnumber.attrs['valid_max'] = np.nanmax(convcold_cloudnumber)

    # ds_out.cloudnumber_noinflate.attrs['long_name'] = 'Number associated with the cloud (no inflation) at a given pixel'
    # ds_out.cloudnumber_noinflate.attrs['units'] = 'unitless'
    # ds_out.cloudnumber_noinflate.attrs['_FillValue'] = 0
    
    ds_out.mergecloudnumber.attrs['long_name'] = 'Cloud number that this cloud merges into'
    ds_out.mergecloudnumber.attrs['units'] = 'unitless'
    ds_out.mergecloudnumber.attrs['_FillValue'] = 0
    # ds_out.mergecloudnumber.attrs['valid_min'] = 0
    # ds_out.mergecloudnumber.attrs['valid_max'] = np.nanmax(celltrackmap_mergesplit)

    ds_out.splitcloudnumber.attrs['long_name'] = 'Cloud number that this cloud splits from'
    ds_out.splitcloudnumber.attrs['units'] = 'unitless'
    ds_out.splitcloudnumber.attrs['_FillValue'] = 0
    # ds_out.splitcloudnumber.attrs['valid_min'] = 0
    # ds_out.splitcloudnumber.attrs['valid_max'] = np.nanmax(celltrackmap_mergesplit)
    
    # Write netcdf file
    print('Output celltracking file: ', celltrackmaps_outfile)
    # print('')

    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encodelist = {var: comp for var in ds_out.data_vars}

#    # Specify encoding list
#    encodelist = {'basetime': {'dtype':'float', 'zlib':True}, \
#                    'time': {'zlib':True}, \
#                    'lon': {'zlib':True, '_FillValue': np.nan}, \
#                    'lat': {'zlib':True, '_FillValue': np.nan}, \
#                    'longitude': {'zlib':True, '_FillValue': np.nan}, \
#                    'latitude': {'zlib':True, '_FillValue': np.nan}, \
#                    'nclouds': {'dtype': 'int64', 'zlib':True, '_FillValue': fillval}, \
#                    'comp_ref': {'zlib':True, '_FillValue': np.nan}, \
#                    'conv_core': {'zlib':True}, \
#                    'conv_mask': {'zlib':True}, \
#                    'tracknumber': {'zlib':True, 'dtype':'int32'}, \
#                    'tracknumber_cmask2': {'zlib':True, 'dtype':'int32'}, \
#                    'cloudstatus': {'zlib':True}, \
#                    'cloudnumber': {'dtype':'int32', 'zlib':True}, \
#                    # 'cloudnumber_noinflate': {'dtype':'int32', 'zlib':True}, \
#                    'mergecloudnumber': {'zlib':True,}, \
#                    'splitcloudnumber': {'zlib':True}, \
#                }

    # Write to netCDF file
    ds_out.to_netcdf(path=celltrackmaps_outfile, mode='w', format='NETCDF4_CLASSIC', unlimited_dims='time', encoding=encodelist)

    # import pdb; pdb.set_trace()
