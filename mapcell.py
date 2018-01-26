# Author: Original IDL code written by Zhe Feng (zhe.feng@pnnl.gov), Python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov)

def mapcell_LES(zipped_inputs):
    # Purpose: Subset statistics file to keep only MCS. Uses brightness temperature statstics of cold cloud shield area, duration, and eccentricity base on Fritsch et al (1986) and Maddos (1980)

    #######################################################################
    # Import modules
    import numpy as np
    import time
    import os
    import sys
    import xarray as xr
    import pandas as pd
    import time, datetime, calendar
    np.set_printoptions(threshold=np.inf)

    # Separate inputs
    cloudid_filename = zipped_inputs[0]
    filebasetime = zipped_inputs[1]
    cellstats_filebase = zipped_inputs[2]
    statistics_filebase = zipped_inputs[3]
    celltracking_path = zipped_inputs[4]
    stats_path = zipped_inputs[5]
    absolutelwp_threshs = zipped_inputs[6]
    startdate = zipped_inputs[7]
    enddate = zipped_inputs[8]

    ######################################################################
    # define constants
    # minimum and maximum brightness temperature thresholds. data outside of this range is filtered
    minlwp_thresh = absolutelwp_threshs[0]    # k
    maxlwp_thresh = absolutelwp_threshs[1]    # k

    fillvalue = -9999

    ##################################################################
    # Load all track stat file
    statistics_file = stats_path + statistics_filebase + '_' + startdate + '_' + enddate + '.nc'

    allstatdata = xr.open_dataset(statistics_file, autoclose=True)
    trackstat_basetime = allstatdata['basetime'].data # Time of cloud in seconds since 01/01/1970 00:00
    trackstat_cloudnumber = allstatdata['cloudnumber'].data # Number of the corresponding cloudid file
    trackstat_status = allstatdata['status'].data # Flag indicating the status of the cloud
    trackstat_mergenumbers = allstatdata['mergenumbers'].data # Track number that it merges into
    trackstat_splitnumbers = allstatdata['splitnumbers'].data

    #######################################################################
    # Load cell track stat file
    cellstatistics_file = stats_path + cellstats_filebase + startdate + '_' + enddate + '.nc'
    print(cellstatistics_file)

    allcelldata = xr.open_dataset(cellstatistics_file, autoclose=True)
    celltrackstat_basetime = allcelldata['cell_basetime'].data # basetime of each cloud in the tracked cell
    celltrackstat_status = allcelldata['cell_status'].data # flag indicating the status of each cloud in the tracked cell
    celltrackstat_cloudnumber = allcelldata['cell_cloudnumber'].data # number of cloud in the corresponding cloudid file for each cloud in the tracked cell
    celltrackstat_mergecloudnumber = allcelldata['cell_mergecloudnumber'].data # number of cloud in the corresponding cloud file that merges into the tracked cell
    celltrackstat_splitcloudnumber = allcelldata['cell_splitcloudnumber'].data # number of cloud in the corresponding cloud file that splits into the tracked cell

    #########################################################################
    # Get cloudid file associated with this time
    file_datetime = time.strftime("%Y%m%d_%H%M", time.gmtime(np.copy(filebasetime)))
    filedate = np.copy(file_datetime[0:8])
    filetime = np.copy(file_datetime[9:14])
    print('cloudid file: ' + cloudid_filename)

    # Load cloudid data
    cloudiddata = xr.open_dataset(cloudid_filename, autoclose=True)
    cloudid_cloudnumber = cloudiddata['convcold_cloudnumber'].data
    cloudid_cloudtype = cloudiddata['cloudtype'].data
    cloudid_basetime = cloudiddata['basetime'].data

    cloudid_cloudnumber = cloudid_cloudnumber.astype(np.int32)
    cloudid_cloudtype = cloudid_cloudtype.astype(np.int32)
    
    # Get data dimensions
    [timeindex, nlat, nlon] = np.shape(cloudid_cloudnumber)

    ##############################################################
    # Intiailize track maps
    celltrackmap = np.zeros((1, nlat, nlon), dtype=int)
    celltrackmap_mergesplit = np.zeros((1, nlat, nlon), dtype=int)
        
    statusmap = np.ones((1, nlat, nlon), dtype=int)*fillvalue
    trackmap = np.zeros((1, nlat, nlon), dtype=int)

    allmergemap = np.zeros((1, nlat, nlon), dtype=int)
    allsplitmap = np.zeros((1, nlat, nlon), dtype=int)
    cellmergemap = np.zeros((1, nlat, nlon), dtype=int)
    cellsplitmap = np.zeros((1, nlat, nlon), dtype=int)

    ###############################################################
    # Create map of status and track number for every feature in this file
    fulltrack, fulltime = np.array(np.where(trackstat_basetime == cloudid_basetime))
    for ifull in range(0, len(fulltime)):
        ffcloudnumber = trackstat_cloudnumber[fulltrack[ifull], fulltime[ifull]]
        ffstatus = trackstat_status[fulltrack[ifull], fulltime[ifull]]

        fullypixels, fullxpixels = np.array(np.where(cloudid_cloudnumber[0, :, :] == ffcloudnumber))
        
        statusmap[0, fullypixels, fullxpixels] = ffstatus
        trackmap[0, fullypixels, fullxpixels] = fulltrack[ifull] + 1

    ffcloudnumber = trackstat_cloudnumber[fulltrack, fulltime]
    ffallsplit = trackstat_splitnumbers[fulltrack, fulltime]
    splitpresent = len(np.array(np.where(np.isfinite(ffallsplit)))[0, :])
    if splitpresent > 0:
        splittracks = np.copy(ffallsplit[np.where(np.isfinite(ffallsplit))])
        splitcloudid = np.copy(ffcloudnumber[np.where(np.isfinite(ffallsplit))])

        for isplit in range(0, len(splittracks)):
            splitypixels, splitxpixels = np.array(np.where(cloudid_cloudnumber[0, :, :] == splitcloudid[isplit]))
            allsplitmap[0, splitypixels, splitxpixels] = splittracks[isplit]

    ffallmerge = trackstat_mergenumbers[fulltrack, fulltime]
    mergepresent = len(np.array(np.where(np.isfinite(ffallmerge)))[0, :])
    if mergepresent > 0:
        mergetracks = np.copy(ffallmerge[np.where(np.isfinite(ffallmerge))])
        mergecloudid = np.copy(ffcloudnumber[np.where(np.isfinite(ffallmerge))])

        for imerge in range(0, len(mergetracks)):
            mergeypixels, mergexpixels = np.array(np.where(cloudid_cloudnumber[0, :, :] == mergecloudid[imerge]))
            allmergemap[0, mergeypixels, mergexpixels] = mergetracks[imerge]

    trackmap = trackmap.astype(np.int32)
    allmergemap = allmergemap.astype(np.int32)
    allsplitmap = allsplitmap.astype(np.int32)

    #plt.figure()
    #im = plt.pcolormesh(np.ma.masked_invalid(np.atleast_2d(trackmap[0, :, :])))
    #plt.colorbar(im)

    #plt.figure()
    #im = plt.pcolormesh(np.ma.masked_invalid(np.atleast_2d(allmergemap[0, :, :])))
    #plt.colorbar(im)

    #plt.figure()
    #im = plt.pcolormesh(np.ma.masked_invalid(np.atleast_2d(allsplitmap[0, :, :])))
    #plt.colorbar(im)
    #plt.show()

    ###############################################################
    # Get tracks
    itrack, itime = np.array(np.where(celltrackstat_basetime == cloudid_basetime))
    ntimes = len(itime)
    if ntimes > 0:
        #timestatus = np.copy(celltrackstat_status[itrack,itime])
        
        ##############################################################
        # Loop over each cloud in this unique file
        for jj in range(0, ntimes):
            # Get cloud nummber
            jjcloudnumber = celltrackstat_cloudnumber[itrack[jj],itime[jj]].astype(np.int32)

            # Find pixels assigned to this cloud number
            jjcloudypixels, jjcloudxpixels = np.array(np.where(cloudid_cloudnumber[0, :, :] == jjcloudnumber))

            # Label this cloud with the track number. Need to add one to the cloud number since have the index number and we want the track number
            if len(jjcloudypixels) > 0:
                celltrackmap[0, jjcloudypixels, jjcloudxpixels] = itrack[jj] + 1
                celltrackmap_mergesplit[0, jjcloudypixels, jjcloudxpixels] = itrack[jj] + 1

                #statusmap[0, jjcloudypixels, jjcloudxpixels] = timestatus[jj] 
            else:
                sys.exit('Error: No matching cloud pixel found?!')

            ###########################################################
            # Find merging clouds
            jjmerge = np.array(np.where(celltrackstat_mergecloudnumber[itrack[jj], itime[jj],:] > 0))[0,:]

            # Loop through merging clouds if present
            if len(jjmerge) > 0:
                for imerge in jjmerge:
                    # Find cloud number asosicated with the merging cloud
                    jjmergeypixels, jjmergexpixels = np.array(np.where(cloudid_cloudnumber[0,:,:] == celltrackstat_mergecloudnumber[itrack[jj], itime[jj], imerge]))
                        
                    # Label this cloud with the track number. Need to add one to the cloud number since have the index number and we want the track number
                    if len(jjmergeypixels) > 0:
                        celltrackmap_mergesplit[0, jjmergeypixels, jjmergexpixels] = itrack[jj] + 1
                        #statusmap[0, jjmergeypixels, jjmergexpixels] = cellmergestatus[itrack[jj], itime[jj], imerge]
                        cellmergemap[0, jjmergeypixels, jjmergexpixels] = itrack[jj] + 1
                    else:
                        sys.exit('Error: No matching merging cloud pixel found?!')

            ###########################################################
            # Find splitting clouds
            jjsplit = np.array(np.where(celltrackstat_splitcloudnumber[itrack[jj], itime[jj],:] > 0))[0,:]
            
            # Loop through splitting clouds if present
            if len(jjsplit) > 0:
                for isplit in jjsplit:
                    # Find cloud number asosicated with the splitting cloud
                    jjsplitypixels, jjsplitxpixels = np.array(np.where(cloudid_cloudnumber[0,:,:] == celltrackstat_splitcloudnumber[itrack[jj], itime[jj], isplit]))

                    # Label this cloud with the track number. Need to add one to the cloud number since have the index number and we want the track number
                    if len(jjsplitypixels) > 0:
                        celltrackmap_mergesplit[0, jjsplitypixels, jjsplitxpixels] = itrack[jj] + 1
                        #statusmap[0, jjsplitypixels, jjsplitxpixels] = cellsplitstatus[itrack[jj], itime[jj], isplit]
                        cellsplitmap[0, jjsplitypixels, jjsplitxpixels] = itrack[jj] + 1
                    else:
                        sys.exit('Error: No matching splitting cloud pixel found?!')

    cellsplitmap = cellsplitmap.astype(np.int32)
    cellmergemap = cellmergemap.astype(np.int32)
    celltrackmap_mergesplit = celltrackmap_mergesplit.astype(np.int32)
    celltrackmap = celltrackmap.astype(np.int32)
    statusmap = statusmap.astype(np.int32)

    #####################################################################
    # Output maps to netcdf file

    # Create output directories
    if not os.path.exists(celltracking_path):
        os.makedirs(celltracking_path)

    # Define output fileame
    celltrackmaps_outfile = celltracking_path + 'celltracks_' + str(filedate) + '_' + str(filetime) + '.nc'

    # Check if file already exists. If exists, delete
    if os.path.isfile(celltrackmaps_outfile):
        os.remove(celltrackmaps_outfile)

    # Define xarray dataset
    output_data = xr.Dataset({'basetime': (['time'], np.array([pd.to_datetime(cloudiddata['basetime'].data, unit='s')], dtype='datetime64[s]')[0]),  \
                              'lon': (['nlat', 'nlon'], cloudiddata['longitude']), \
                              'lat': (['nlat', 'nlon'], cloudiddata['latitude']), \
                              'nclouds': (['time'], cloudiddata['nclouds'].data), \
                              'lwp': (['time', 'nlat', 'nlon'], cloudiddata['lwp'].data), \
                              'cloudtype': (['time', 'nlat', 'nlon'], cloudiddata['cloudtype'].data), \
                              'cloudstatus': (['time', 'nlat', 'nlon'], statusmap), \
                              'alltracknumbers': (['time', 'nlat', 'nlon'], trackmap), \
                              'allsplittracknumbers': (['time', 'nlat', 'nlon'], allsplitmap), \
                              'allmergetracknumbers': (['time', 'nlat', 'nlon'], allmergemap), \
                              'cellsplittracknumbers': (['time', 'nlat', 'nlon'], cellsplitmap), \
                              'cellmergetracknumbers': (['time', 'nlat', 'nlon'], cellmergemap), \
                              'cloudnumber': (['time', 'nlat', 'nlon'], cloudiddata['convcold_cloudnumber']), \
                              'celltracknumber_nomergesplit': (['time', 'nlat', 'nlon'], celltrackmap), \
                              'celltracknumber': (['time', 'nlat', 'nlon'], celltrackmap_mergesplit)}, \
                             coords={'time': (['time'], cloudiddata['basetime']), \
                                     'nlat': (['nlat'], np.arange(0, nlat)), \
                                     'nlon': (['nlon'], np.arange(0, nlon))}, \
                             attrs={'title':'Pixel level of tracked clouds and CELLs', \
                                    'source': allcelldata.attrs['source'], \
                                    'description': allcelldata.attrs['description'], \
                                    'Cloud_area_km2': cloudiddata.attrs['minimum_cloud_area'], \
                                    'Main_cell_duration_hr': allcelldata.attrs['Main_cell_duration_hr'], \
                                    'Merger_duration_hr': allcelldata.attrs['Merge_duration_hr'], \
                                    'Split_duration_hr': allcelldata.attrs['Split_duration_hr'], \
                                    'contact':'Hannah C Barnes: hannah.barnes@pnnl.gov', \
                                    'created_on':time.ctime(time.time())})

    # Specify variable attributes
    #output_data.time.attrs['long_name'] = 'Number of times in this file'
    #output_data.time.attrs['units'] = 'unitless'
    
    #output_data.nlat.attrs['long_name'] = 'Number of latitude grid points in this file'
    #output_data.nlat.attrs['units'] = 'unitless'
    
    #output_data.nlon.attrs['long_name'] = 'Number of longitude grid points in this file'
    #output_data.nlon.attrs['units'] = 'unitless'
    
    output_data.basetime.attrs['long_name'] = 'Epoch time (seconds since 01/01/1970 00:00) of this file'
    
    output_data.lon.attrs['long_name'] = 'Grid of longitude'
    output_data.lon.attrs['units'] = 'degrees'
    
    output_data.lat.attrs['long_name'] = 'Grid of latitude'
    output_data.lat.attrs['units'] = 'degrees'
    
    output_data.nclouds.attrs['long_name'] = 'Number of cells identified in this file'
    output_data.nclouds.attrs['units'] = 'unitless'
    
    output_data.lwp.attrs['long_name'] = 'brightness temperature'
    output_data.lwp.attrs['min_value'] =  minlwp_thresh
    output_data.lwp.attrs['max_value'] = maxlwp_thresh
    output_data.lwp.attrs['units'] = 'K'
    
    output_data.cloudtype.attrs['long_name'] = 'flag indicating type of ir data'
    output_data.cloudtype.attrs['units'] = 'unitless'
    output_data.cloudtype.attrs['valid_min'] = 1
    output_data.cloudtype.attrs['valid_max'] = 5

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
    output_data.cloudnumber.attrs['valid_max'] = np.nanmax(cloudiddata['convcold_cloudnumber'].data)
    
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
                          encoding={'basetime': {'dtype': 'int64', 'zlib':True, '_FillValue': fillvalue, 'units': 'seconds since 1970-01-01'}, \
                                    'time': {'units': 'seconds since 1970-01-01'}, \
                                    'lon': {'zlib':True, '_FillValue': fillvalue}, \
                                    'lat': {'zlib':True, '_FillValue': fillvalue}, \
                                    'nclouds': {'zlib':True, '_FillValue': fillvalue}, \
                                    'lwp': {'zlib':True, '_FillValue': fillvalue}, \
                                    'cloudtype': {'zlib':True, '_FillValue': fillvalue}, \
                                    'cloudstatus': {'zlib':True, '_FillValue': fillvalue}, \
                                    'allsplittracknumbers': {'zlib':True, '_FillValue': fillvalue}, \
                                    'allmergetracknumbers': {'zlib':True, '_FillValue': fillvalue}, \
                                    'cellsplittracknumbers': {'zlib':True, '_FillValue': fillvalue}, \
                                    'cellmergetracknumbers': {'zlib':True, '_FillValue': fillvalue}, \
                                    'alltracknumbers': {'zlib':True, '_FillValue': fillvalue}, \
                                    'cloudnumber': {'dtype': 'int', 'zlib':True, '_FillValue': fillvalue}, \
                                    'celltracknumber_nomergesplit': {'zlib':True, '_FillValue': fillvalue}, \
                                    'celltracknumber': {'zlib':True, '_FillValue': fillvalue}})


