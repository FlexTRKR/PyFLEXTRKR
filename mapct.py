# Purpose: Take the cell tracks identified in the previous steps and create pixel level maps of these storms. One netcdf file is create for each time step.

# Author: Original IDL code written by Zhe Feng (zhe.feng@pnnl.gov), Python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov)

def map_ct(zipped_inputs):
    # Inputs:
    # cloudid_filebase - file header of the cloudid file create in the first step
    # filebasetime - seconds since 1970-01-01 of the file being processed
    # cellstats_filebase - file header of the cell tracks statistics file generated in the identifycell step
    # statistics_filebase - file header for the all track statistics file generated in the trackstats step
    # celltracking_path - directory where cell tracks maps generated in this step will be placed
    # stats_path - directory that contains the statistics files
    # absolutelwp_thresh - range of valid liquid water paths 
    # startdate - starting date and time of the full dataset
    # enddate - ending date and time of the full dataset
    # showalltracks - flag indicating whether the output should include maps of all tracks (MCS and nonMCS). this greatly slows the code.

    # Output (One netcdf file of maps for each time step):
    # basetime - seconds since 1970-01-01 of the file being processed
    # lon - grid of analyzed longitudes
    # lat - grid of analyzed latitutdes
    # nclouds - total number of identified clouds, from the cloudid file
    # lwp - map of liquid water path
    # cloudtype - map of the cloud types identified in the idclouds step
    # cellsplittracknumbers - map of the clouds splitting from cells
    # cellmergetracknumbers - map of the clouds merging with cells
    # cloudnumber - map of cloud numbers associated with each cloud that were determined in the idclouds step
    # cloudtracknumbers_nomergesplit - map of MCS track numbers, excluding clouds that merge and split from the MCSs
    # cloudtracknumber -  map of MCS track numbers, includes clouds that merge and split from the MCSs
    # alltracknumber - map of all the identified tracks (MCS and nonMCS), optional
    # allsplittracknumbers - map of the clouds splitting from all tracks (MCS and nonMCS), optional
    # allmergetracknumbers - map of the clouds merging with all tracks (MCS and nonMCS), optional
    # cloudstatus - map the status that describes the evolution of the cloud, determine in the gettracks step (optional)

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

    # Separate inputs
    cloudid_filename = zipped_inputs[0]
    filebasetime = zipped_inputs[1]
    trackstats_filebase = zipped_inputs[2]
    rainaccumulation_filebase = zipped_inputs[3]
    cttracking_path = zipped_inputs[4]
    stats_path = zipped_inputs[5]
    rainaccumulation_path = zipped_inputs[6]
    startdate = zipped_inputs[7]
    enddate = zipped_inputs[8]
    showalltracks =  zipped_inputs[9]
      
    ###################################################################
    # Load all track stat file
    if showalltracks == 0:
        statistics_file = stats_path + trackstats_filebase + '_' + startdate + '_' + enddate + '.nc'

        allstatdata = Dataset(statistics_file, 'r')
        trackstat_basetime = allstatdata['basetime'][:] # Time of cloud in seconds since 01/01/1970 00:00
        trackstat_cloudnumber = allstatdata['cloudnumber'][:] # Number of the corresponding cloudid file
        trackstat_status = allstatdata['status'][:] # Flag indicating the status of the cloud
        trackstat_mergenumbers = allstatdata['mergenumbers'][:] # Track number that it merges into
        trackstat_splitnumbers = allstatdata['splitnumbers'][:]
        datadescription = allstatdata.getncattr('description')
        datasource = allstatdata.getncattr('source')
        allstatdata.close()

    #########################################################################
    # Get cloudid file associated with this time
    file_datetime = time.strftime("%Y%m%d_%H%M", time.gmtime(np.copy(filebasetime)))
    filedate = np.copy(file_datetime[0:8])
    filetime = np.copy(file_datetime[9:14])
    print(('cloudid file: ' + cloudid_filename))

    # Load cloudid data
    cloudiddata = Dataset(cloudid_filename, 'r')
    cloudid_cloudnumber = cloudiddata['convcold_cloudnumber'][:]
    cloudid_cloudtype = cloudiddata['original_cloudtype'][:]
    cloudid_basetime = cloudiddata['basetime'][:]
    basetime_units =  cloudiddata['basetime'].units
    basetime_calendar = cloudiddata['basetime'].calendar
    longitude = cloudiddata['longitude'][:]
    latitude = cloudiddata['latitude'][:]
    nclouds = cloudiddata['nclouds'][:]
    cloudtype = cloudiddata['original_cloudtype'][:]
    convcold_cloudnumber = cloudiddata['convcold_cloudnumber'][:]
    cloudiddata.close()

    cloudid_cloudnumber = cloudid_cloudnumber.astype(np.int32)
    cloudid_cloudtype = cloudid_cloudtype.astype(np.int32)
    
    # Get data dimensions
    [nt, nlat, nlon] = np.shape(cloudid_cloudnumber)

    ##############################################################
    # Intiailize track maps
    celltrackmap = np.zeros((1, nlat, nlon), dtype=int)
    celltrackmap_mergesplit = np.zeros((1, nlat, nlon), dtype=int)
        
    cellmergemap = np.zeros((1, nlat, nlon), dtype=int)
    cellsplitmap = np.zeros((1, nlat, nlon), dtype=int)

    ################################################################
    # Create map of status and track number for every feature in this file
    if showalltracks == 0:
        print('Create maps of all tracks')
        statusmap = np.ones((1,nlat, nlon), dtype=int)*-9999
        trackmap = np.zeros((1, nlat, nlon), dtype=int)
        allmergemap = np.zeros((1, nlat, nlon), dtype=int)
        allsplitmap = np.zeros((1, nlat, nlon), dtype=int)

        fulltrack, fulltime = np.array(np.where(trackstat_basetime == cloudid_basetime))
        for ifull in range(0, len(fulltime)):
            ffcloudnumber = trackstat_cloudnumber[fulltrack[ifull], fulltime[ifull]]
            ffstatus = trackstat_status[fulltrack[ifull], fulltime[ifull]]

            fullypixels, fullxpixels = np.array(np.where(cloudid_cloudnumber[0,:, :] == ffcloudnumber))
        
            statusmap[0, fullypixels, fullxpixels] = ffstatus
            trackmap[0, fullypixels, fullxpixels] = fulltrack[ifull] + 1

        ffcloudnumber = trackstat_cloudnumber[fulltrack, fulltime]
        ffallsplit = trackstat_splitnumbers[fulltrack, fulltime]
        splitpresent = len(np.array(np.where(np.isfinite(ffallsplit)))[0, :])
        if splitpresent > 0:
            splittracks = np.copy(ffallsplit[np.where(np.isfinite(ffallsplit))])
            splitcloudid = np.copy(ffcloudnumber[np.where(np.isfinite(ffallsplit))])

            for isplit in range(0, len(splittracks)):
                splitypixels, splitxpixels = np.array(np.where(cloudid_cloudnumber[0,:, :] == splitcloudid[isplit]))
                allsplitmap[0, splitypixels, splitxpixels] = splittracks[isplit]

        ffallmerge = trackstat_mergenumbers[fulltrack, fulltime]
        mergepresent = len(np.array(np.where(np.isfinite(ffallmerge)))[0, :])
        if mergepresent > 0:
            mergetracks = np.copy(ffallmerge[np.where(np.isfinite(ffallmerge))])
            mergecloudid = np.copy(ffcloudnumber[np.where(np.isfinite(ffallmerge))])

            for imerge in range(0, len(mergetracks)):
                mergeypixels, mergexpixels = np.array(np.where(cloudid_cloudnumber[0,:, :] == mergecloudid[imerge]))
                allmergemap[0, mergeypixels, mergexpixels] = mergetracks[imerge]

        trackmap = trackmap.astype(np.int32)
        allmergemap = allmergemap.astype(np.int32)
        allsplitmap = allsplitmap.astype(np.int32)
        
        trackmap = np.squeeze(trackmap,axis=0)
        allmergemap = np.squeeze(allmergemap,axis=0)
        allsplitmap = np.squeeze(allsplitmap,axis=0)

    #####################################################################
    # Output maps to netcdf file

    # Create output directories
    if not os.path.exists(cttracking_path):
        os.makedirs(cttracking_path)

    # Define output fileame
    celltrackmaps_outfile = cttracking_path + 'cttracks_' + str(filedate) + '_' + str(filetime) + '.nc'

    # Check if file already exists. If exists, delete
    if os.path.isfile(celltrackmaps_outfile):
        os.remove(celltrackmaps_outfile) 
        
    #print(nclouds)
    
    #nclouds_tot = np.max(nclouds)
    #print(nclouds_tot)
    
    # Define xarray dataset
    if showalltracks == 0:
        output_data = xr.Dataset({'basetime': (['time'], np.array([pd.to_datetime(num2date(cloudid_basetime, units=basetime_units, calendar=basetime_calendar))], dtype='datetime64[s]')[0, :]),  \
                                  'lon': (['nlat', 'nlon'], longitude), \
                                  'lat': (['nlat', 'nlon'], latitude), \
                                  'nclouds': (['clouds'], nclouds), \
                                  'cloudtype': (['time','nlat', 'nlon'], cloudtype), \
                                  'ctsplittracknumbers': (['time','nlat', 'nlon'], np.expand_dims(allsplitmap,axis=0)), \
                                  'ctmergetracknumbers': (['time','nlat', 'nlon'], np.expand_dims(allmergemap,axis=0)), \
                                  'cloudnumber': (['time','nlat', 'nlon'], convcold_cloudnumber), \
                                  'cttracknumber': (['time','nlat', 'nlon'], np.expand_dims(trackmap,axis=0))}, \
                                 coords={'time': (['time'], cloudid_basetime), \
                                         'clouds':(['clouds'], np.arange(0,len(nclouds))), \
                                         'nlat': (['nlat'], np.arange(0, nlat)), \
                                         'nlon': (['nlon'], np.arange(0, nlon))}, \
                                 attrs={'title':'Pixel level of tracked clouds', \
                                        'source': datasource, \
                                        'description': datadescription, \
                                        'contact':'Katelyn Barber: katelyn.barber@pnnl.gov', \
                                        'created_on':time.ctime(time.time())})
        
        # Specify variable attributes
        output_data.basetime.attrs['long_name'] = 'Epoch time (seconds since 01/01/1970 00:00) of this file'
        
        output_data.lon.attrs['long_name'] = 'Grid of longitude'
        output_data.lon.attrs['units'] = 'degrees'
        
        output_data.lat.attrs['long_name'] = 'Grid of latitude'
        output_data.lat.attrs['units'] = 'degrees'
        
        output_data.nclouds.attrs['long_name'] = 'Number of cells identified in this file'
        output_data.nclouds.attrs['units'] = 'unitless'
        
        output_data.cloudtype.attrs['long_name'] = 'flag indicating type cloud'
        output_data.cloudtype.attrs['units'] = 'unitless'
            
        output_data.ctmergetracknumbers.attrs['long_name'] = 'Number of the cell track that this cloud merges into'
        output_data.ctmergetracknumbers.attrs['units'] = 'unitless'

        output_data.ctsplittracknumbers.attrs['long_name'] = 'Number of the cell track that this cloud splits from'
        output_data.ctsplittracknumbers.attrs['units'] = 'unitless'
        
        output_data.cloudnumber.attrs['long_name'] = 'Number associated with the cloud at a given pixel'
        output_data.cloudnumber.attrs['units'] = 'unitless'
        
        output_data.cttracknumber.attrs['long_name'] = 'Number of the tracked cell associated with the cloud at a given pixel'
        output_data.cttracknumber.attrs['units'] = 'unitless'

        
        # Write netcdf file
        print(celltrackmaps_outfile)
        print('')

        output_data.to_netcdf(path=celltrackmaps_outfile, mode='w', format='NETCDF4_CLASSIC', unlimited_dims='time', \
                              encoding={'basetime': {'zlib':True, 'units': basetime_units, 'calendar': basetime_calendar}, \
                                        'time': {'dtype': 'int'}, \
                                        'lon': {'zlib':True, '_FillValue': np.nan}, \
                                        'lat': {'zlib':True, '_FillValue': np.nan}, \
                                        'nclouds': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                        'cloudtype': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                        'ctsplittracknumbers': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                        'ctmergetracknumbers': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                        'cloudnumber': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                        'cttracknumber': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}})

    else:
        output_data = xr.Dataset({'basetime': (['time'], np.array([pd.to_datetime(num2date(cloudid_basetime, units=basetime_units, calendar=basetime_calendar))], dtype='datetime64[s]')[0, :]),  \
                                  'lon': (['nlat', 'nlon'], longitude), \
                                  'lat': (['nlat', 'nlon'], latitude), \
                                  'nclouds': (['time'], nclouds), \
                                  'cloudtype': (['time', 'nlat', 'nlon'], cloudtype), \
                                  'cloudstatus': (['time', 'nlat', 'nlon'], statusmap), \
                                  'alltracknumbers': (['time', 'nlat', 'nlon'], trackmap), \
                                  'allsplittracknumbers': (['time', 'nlat', 'nlon'], allsplitmap), \
                                  'allmergetracknumbers': (['time', 'nlat', 'nlon'], allmergemap), \
                                  'cellsplittracknumbers': (['time', 'nlat', 'nlon'], cellsplitmap), \
                                  'cellmergetracknumbers': (['time', 'nlat', 'nlon'], cellmergemap), \
                                  'cloudnumber': (['time', 'nlat', 'nlon'], convcold_cloudnumber), \
                                  'celltracknumber_nomergesplit': (['time', 'nlat', 'nlon'], celltrackmap), \
                                  'celltracknumber': (['time', 'nlat', 'nlon'], celltrackmap_mergesplit)}, \
                                 coords={'time': (['time'], cloudid_basetime), \
                                         'nlat': (['nlat'], np.arange(0, nlat)), \
                                         'nlon': (['nlon'], np.arange(0, nlon))}, \
                                 attrs={'title':'Pixel level of tracked clouds and CELLs', \
                                        'source': datasource, \
                                        'description': datadescription, \
                                        'Main_cell_duration_hr': durationthresh, \
                                        'Merger_duration_hr': mergethresh, \
                                        'Split_duration_hr': splitthresh, \
                                        'contact':'Hannah C Barnes: hannah.barnes@pnnl.gov', \
                                        'created_on':time.ctime(time.time())})
        
        # Specify variable attributes
        output_data.basetime.attrs['long_name'] = 'Epoch time (seconds since 01/01/1970 00:00) of this file'
        
        output_data.lon.attrs['long_name'] = 'Grid of longitude'
        output_data.lon.attrs['units'] = 'degrees'
        
        output_data.lat.attrs['long_name'] = 'Grid of latitude'
        output_data.lat.attrs['units'] = 'degrees'
        
        output_data.nclouds.attrs['long_name'] = 'Number of cells identified in this file'
        output_data.nclouds.attrs['units'] = 'unitless'
        
        output_data.cloudstatus.attrs['long_name'] = 'flag indicating history of cloud'
        output_data.cloudstatus.attrs['units'] = 'unitless'
        output_data.cloudstatus.attrs['valid_min'] = 0
        output_data.cloudstatus.attrs['valid_max'] = 65
        
        output_data.alltracknumbers.attrs['long_name'] = 'Number of the cloud track associated with the cloud at a given pixel'
        output_data.alltracknumbers.attrs['units'] = 'unitless'
        output_data.alltracknumbers.attrs['valid_min'] = 0
        output_data.alltracknumbers.attrs['valid_max'] = np.nanmax(trackmap) 
        
        output_data.allmergetracknumbers.attrs['long_name'] = 'Number of the cloud track that this cloud merges into'
        output_data.allmergetracknumbers.attrs['units'] = 'unitless'
        output_data.allmergetracknumbers.attrs['valid_min'] = 0
        output_data.allmergetracknumbers.attrs['valid_max'] = np.nanmax(trackmap)  

        output_data.allsplittracknumbers.attrs['long_name'] = 'Number of the cloud track that this cloud splits from'
        output_data.allsplittracknumbers.attrs['units'] = 'unitless'
        output_data.allsplittracknumbers.attrs['valid_min'] = 0
        output_data.allsplittracknumbers.attrs['valid_max'] = np.nanmax(trackmap)

        output_data.cellmergetracknumbers.attrs['long_name'] = 'Number of the cell track that this cloud merges into'
        output_data.cellmergetracknumbers.attrs['units'] = 'unitless'
        output_data.cellmergetracknumbers.attrs['valid_min'] = 0
        output_data.cellmergetracknumbers.attrs['valid_max'] = np.nanmax(celltrackmap_mergesplit)

        output_data.cellsplittracknumbers.attrs['long_name'] = 'Number of the cell track that this cloud splits from'
        output_data.cellsplittracknumbers.attrs['units'] = 'unitless'
        output_data.cellsplittracknumbers.attrs['valid_min'] = 0
        output_data.cellsplittracknumbers.attrs['valid_max'] = np.nanmax(celltrackmap_mergesplit)
        
        output_data.cloudnumber.attrs['long_name'] = 'Number associated with the cloud at a given pixel'
        output_data.cloudnumber.attrs['comment'] = 'Extent of cloud system is defined using the warm anvil threshold'
        output_data.cloudnumber.attrs['units'] = 'unitless'
        output_data.cloudnumber.attrs['valid_min'] = 0
        output_data.cloudnumber.attrs['valid_max'] = np.nanmax(convcold_cloudnumber)
        
        output_data.celltracknumber_nomergesplit.attrs['long_name'] = 'Number of the tracked cell associated with the cloud at a given pixel'
        output_data.celltracknumber_nomergesplit.attrs['units'] = 'unitless'
        output_data.celltracknumber_nomergesplit.attrs['valid_min'] = 0
        output_data.celltracknumber_nomergesplit.attrs['valid_max'] = np.nanmax(celltrackmap)
        
        output_data.celltracknumber.attrs['long_name'] = 'Number of the tracked cell associated with the cloud at a given pixel'
        output_data.celltracknumber.attrs['comments'] = 'cell includes smaller merges and splits'
        output_data.celltracknumber.attrs['units'] = 'unitless'
        output_data.celltracknumber.attrs['valid_min'] = 0
        output_data.celltracknumber.attrs['valid_max'] = np.nanmax(celltrackmap_mergesplit)
        
        # Write netcdf file
        print(celltrackmaps_outfile)
        print('')

        output_data.to_netcdf(path=celltrackmaps_outfile, mode='w', format='NETCDF4_CLASSIC', unlimited_dims='time', \
                              encoding={'basetime': {'zlib':True, 'units': basetime_units, 'calendar': basetime_calendar}, \
                                        'time': {'dtype': 'int'}, \
                                        'lon': {'zlib':True, '_FillValue': np.nan}, \
                                        'lat': {'zlib':True, '_FillValue': np.nan}, \
                                        'nclouds': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                        'cloudtype': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                        'cloudstatus': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                        'allsplittracknumbers': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                        'allmergetracknumbers': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                        'cellsplittracknumbers': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                        'cellmergetracknumbers': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                        'alltracknumbers': {'zlib':True, '_FillValue': -9999}, \
                                        'cloudnumber': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                        'celltracknumber_nomergesplit': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                        'celltracknumber': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}})


