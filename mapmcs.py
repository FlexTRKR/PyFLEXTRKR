# Author: Original IDL code written by Zhe Feng (zhe.feng@pnnl.gov), Python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov)

def mapmcs_pf(zipped_inputs):
    # Purpose: Subset statistics file to keep only MCS. Uses brightness temperature statstics of cold cloud shield area, duration, and eccentricity base on Fritsch et al (1986) and Maddos (1980)

    #######################################################################
    # Import modules
    import numpy as np
    import time
    import os
    import sys
    import xarray as xr
    np.set_printoptions(threshold=np.inf)

    # Separate inputs
    cloudid_filename = zipped_inputs[0]
    filebasetime = zipped_inputs[1]
    mcsstats_filebase = zipped_inputs[2]
    statistics_filebase = zipped_inputs[3]
    pfdata_filebase = zipped_inputs[4]
    rainaccumulation_filebase = zipped_inputs[5]
    mcstracking_path = zipped_inputs[6]
    stats_path = zipped_inputs[7]
    pfdata_path = zipped_inputs[8]
    rainaccumulation_path = zipped_inputs[9]
    pcp_thresh = zipped_inputs[10]
    nmaxpf = zipped_inputs[11]
    absolutetb_threshs = zipped_inputs[12]
    startdate = zipped_inputs[13]
    enddate = zipped_inputs[14]

    ######################################################################
    # define constants
    # minimum and maximum brightness temperature thresholds. data outside of this range is filtered
    mintb_thresh = absolutetb_threshs[0]    # k
    maxtb_thresh = absolutetb_threshs[1]    # k

    fillvalue = -9999

    ##################################################################
    # Load all track stat file
    statistics_file = stats_path + statistics_filebase + '_' + startdate + '_' + enddate + '.nc'

    allstatdata = xr.open_dataset(statistics_file, autoclose=True, decode_times=False)
    trackstat_basetime = allstatdata['basetime'].data # Time of cloud in seconds since 01/01/1970 00:00
    trackstat_cloudnumber = allstatdata['cloudnumber'].data # Number of the corresponding cloudid file
    trackstat_status = allstatdata['status'].data # Flag indicating the status of the cloud
    trackstat_mergenumbers = allstatdata['mergenumbers'].data # Track number that it merges into
    trackstat_splitnumbers = allstatdata['splitnumbers'].data

    #######################################################################
    # Load MCS track stat file
    mcsstatistics_file = stats_path + mcsstats_filebase + startdate + '_' + enddate + '.nc'
    print(mcsstatistics_file)

    allmcsdata = xr.open_dataset(mcsstatistics_file, autoclose=True, decode_times=False)
    mcstrackstat_basetime = allmcsdata['base_time'].data # basetime of each cloud in the tracked mcs
    mcstrackstat_status = allmcsdata['status'].data # flag indicating the status of each cloud in the tracked mcs
    mcstrackstat_cloudnumber = allmcsdata['cloudnumber'].data # number of cloud in the corresponding cloudid file for each cloud in the tracked mcs
    mcstrackstat_mergecloudnumber = allmcsdata['mergecloudnumber'].data # number of cloud in the corresponding cloud file that merges into the tracked mcs
    mcstrackstat_splitcloudnumber = allmcsdata['splitcloudnumber'].data # number of cloud in the corresponding cloud file that splits into the tracked mcs

    #########################################################################
    # Get cloudid file associated with this time
    file_datetime = time.strftime("%Y%m%d_%H%M", time.gmtime(np.copy(filebasetime)))
    filedate = np.copy(file_datetime[0:8])
    filetime = np.copy(file_datetime[9:14])
    ipffile = pfdata_path + pfdata_filebase + str(filedate) + '-' + str(filetime) + '00.nc'
    irainaccumulationfile = rainaccumulation_path + rainaccumulation_filebase + str(filedate) + '.' + str(filetime) + '00.nc'
    print('cloudid file: ' + cloudid_filename)
    print('pf file: ' + ipffile)
    print('rain accumulation file: ' + irainaccumulationfile)

    # Load cloudid data
    cloudiddata = xr.open_dataset(cloudid_filename, autoclose=True, decode_times=False)
    cloudid_cloudnumber = cloudiddata['convcold_cloudnumber'].data
    cloudid_cloudtype = cloudiddata['cloudtype'].data

    cloudid_cloudnumber = cloudid_cloudnumber.astype(np.int32)
    cloudid_cloudtype = cloudid_cloudtype.astype(np.int32)
    
    # Get data dimensions
    [timeindex, nlat, nlon] = np.shape(cloudid_cloudnumber)

    if os.path.isfile(ipffile):
        # Load NMQ data
        pfdata = xr.open_dataset(ipffile, autoclose=True, decode_times=False)
        pf_reflectivity = pfdata['dbz_convsf'].data # radar reflectivity at convective-stratiform level
        pf_convstrat = pfdata['csa'].data # Steiner convective-stratiform-anvil classifications
        pf_dbz0height = pfdata['dbz0_height'].data # Maximum height of 0 dBZ echo
        pf_dbz10height = pfdata['dbz10_height'].data # Maximum height of 10 dBZ echo
        pf_dbz20height = pfdata['dbz20_height'].data # Maximum height of 20 dBZ echo
        pf_dbz30height = pfdata['dbz30_height'].data # Maximum height of 30 dBZ echo
        pf_dbz40height = pfdata['dbz40_height'].data # Maximum height of 40 dBZ echo
        pf_number = pfdata['pf_number'].data # number of associated precipitation feature at each pixel
        pf_area = pfdata['pf_area'].data # Area of precipitation feature
        pf_mask = pfdata['mask'].data # Flag showing where valid radar data present
        pfpresent = 'Yes'
    else:
        pf_reflectivity = np.ones((1, nlat, nlon), dtype=float)*fillvalue
        pf_convstrat = np.ones((1, nlat, nlon), dtype=float)*fillvalue
        pf_number = np.ones((1, nlat, nlon), dtype=float)*fillvalue
        pf_dbz0height = np.ones((1, nlat, nlon), dtype=float)*fillvalue
        pf_dbz10height = np.ones((1, nlat, nlon), dtype=float)*fillvalue
        pf_dbz20height = np.ones((1, nlat, nlon), dtype=float)*fillvalue
        pf_dbz30height = np.ones((1, nlat, nlon), dtype=float)*fillvalue
        pf_dbz40height = np.ones((1, nlat, nlon), dtype=float)*fillvalue
        pf_mask = np.ones((1, nlat, nlon), dtype=float)*fillvalue
        pfpresent = 'No'
                

    if os.path.isfile(irainaccumulationfile):
        # Load Q2 data
        rainaccumulationdata = xr.open_dataset(irainaccumulationfile, autoclose=True, decode_times=False)
        ra_precipitation = rainaccumulationdata['precipitation'].data # hourly accumulated rainfall
        rapresent = 'Yes'
    else:
        ra_precipitation = np.ones((nlat, nlon), dtype=float)*fillvalue
        rapresent = 'No'
        
    ##############################################################
    # Intiailize track maps
    mcstrackmap = np.zeros((1, nlat, nlon), dtype=int)
    mcstrackmap_mergesplit = np.zeros((1, nlat, nlon), dtype=int)
        
    mcspfnumbermap_mergesplit = np.zeros((1, nlat, nlon), dtype=int)
    mcsramap_mergesplit = np.zeros((1, nlat, nlon), dtype=int)
        
    statusmap = np.ones((1, nlat, nlon), dtype=int)*fillvalue
    trackmap = np.zeros((1, nlat, nlon), dtype=int)

    allmergemap = np.zeros((1, nlat, nlon), dtype=int)
    allsplitmap = np.zeros((1, nlat, nlon), dtype=int)
    mcsmergemap = np.zeros((1, nlat, nlon), dtype=int)
    mcssplitmap = np.zeros((1, nlat, nlon), dtype=int)

    ###############################################################
    # Create map of status and track number for every feature in this file
    fulltrack, fulltime = np.array(np.where(trackstat_basetime == float(filebasetime)))
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
    itrack, itime = np.array(np.where(mcstrackstat_basetime == float(filebasetime)))
    ntimes = len(itime)
    if ntimes > 0:
        #timestatus = np.copy(mcstrackstat_status[itrack,itime])
        
        ##############################################################
        # Loop over each cloud in this unique file
        for jj in range(0, ntimes):
            # Get cloud nummber
            jjcloudnumber = mcstrackstat_cloudnumber[itrack[jj],itime[jj]].astype(np.int32)

            # Find pixels assigned to this cloud number
            jjcloudypixels, jjcloudxpixels = np.array(np.where(cloudid_cloudnumber[0, :, :] == jjcloudnumber))

            # Label this cloud with the track number. Need to add one to the cloud number since have the index number and we want the track number
            if len(jjcloudypixels) > 0:
                mcstrackmap[0, jjcloudypixels, jjcloudxpixels] = itrack[jj] + 1
                mcstrackmap_mergesplit[0, jjcloudypixels, jjcloudxpixels] = itrack[jj] + 1

                #statusmap[0, jjcloudypixels, jjcloudxpixels] = timestatus[jj] 
            else:
                sys.exit('Error: No matching cloud pixel found?!')

            ###########################################################
            # Find merging clouds
            jjmerge = np.array(np.where(mcstrackstat_mergecloudnumber[itrack[jj], itime[jj],:] > 0))[0,:]

            # Loop through merging clouds if present
            if len(jjmerge) > 0:
                for imerge in jjmerge:
                    # Find cloud number asosicated with the merging cloud
                    jjmergeypixels, jjmergexpixels = np.array(np.where(cloudid_cloudnumber[0,:,:] == mcstrackstat_mergecloudnumber[itrack[jj], itime[jj], imerge]))
                        
                    # Label this cloud with the track number. Need to add one to the cloud number since have the index number and we want the track number
                    if len(jjmergeypixels) > 0:
                        mcstrackmap_mergesplit[0, jjmergeypixels, jjmergexpixels] = itrack[jj] + 1
                        #statusmap[0, jjmergeypixels, jjmergexpixels] = mcsmergestatus[itrack[jj], itime[jj], imerge]
                        mcsmergemap[0, jjmergeypixels, jjmergexpixels] = itrack[jj] + 1
                    else:
                        sys.exit('Error: No matching merging cloud pixel found?!')

            ###########################################################
            # Find splitting clouds
            jjsplit = np.array(np.where(mcstrackstat_splitcloudnumber[itrack[jj], itime[jj],:] > 0))[0,:]
            
            # Loop through splitting clouds if present
            if len(jjsplit) > 0:
                for isplit in jjsplit:
                    # Find cloud number asosicated with the splitting cloud
                    jjsplitypixels, jjsplitxpixels = np.array(np.where(cloudid_cloudnumber[0,:,:] == mcstrackstat_splitcloudnumber[itrack[jj], itime[jj], isplit]))

                    # Label this cloud with the track number. Need to add one to the cloud number since have the index number and we want the track number
                    if len(jjsplitypixels) > 0:
                        mcstrackmap_mergesplit[0, jjsplitypixels, jjsplitxpixels] = itrack[jj] + 1
                        #statusmap[0, jjsplitypixels, jjsplitxpixels] = mcssplitstatus[itrack[jj], itime[jj], isplit]
                        mcssplitmap[0, jjsplitypixels, jjsplitxpixels] = itrack[jj] + 1
                    else:
                        sys.exit('Error: No matching splitting cloud pixel found?!')

        ####################################################################
        # Create pf and rain accumulation masks
        extra, iymcs, ixmcs = np.array(np.where(mcstrackmap_mergesplit > 0))
        nmcs = len(iymcs)
        if nmcs > 0:
            # Find unique track numbers
            allmcsnumbers = np.copy(mcstrackmap_mergesplit[extra, iymcs, ixmcs])
            uniquemcsnumber = np.unique(allmcsnumbers)
            
            # Loop over each mcs track
            for imcs in uniquemcsnumber:
                # Find the cloud shield of the mcs
                extra, iymcscloud, ixmcscloud = np.array(np.where(mcstrackmap_mergesplit == imcs))
                nmcscloud = len(iymcscloud)

                if nmcscloud > 0:
                    #######################################################
                    # Label accumulated rain associated with each track

                    # Get accumulated rain under the cloud shield
                    tempra = np.ones((nlat, nlon), dtype=float)*fillvalue
                    tempra[iymcscloud, ixmcscloud] = np.copy(ra_precipitation[extra, iymcscloud, ixmcscloud])
                    
                    # Label "significant" precipitation with track number
                    iymcsra, ixmcsra = np.array(np.where(tempra > pcp_thresh))
                    nmcsra = len(iymcsra)
                    if nmcsra > 0:
                        mcsramap_mergesplit[0, iymcsra, ixmcsra] = np.copy(imcs)

                    #######################################################
                    # Label precipitation features associated with each track
                
                    # Get accumulated rain under the cloud shield
                    temppfnumber = np.ones((nlat, nlon), dtype=float)*fillvalue
                    temppfnumber[iymcscloud, ixmcscloud] = np.copy(pf_number[extra, iymcscloud, ixmcscloud])
                
                    # Label precpitation features with track number
                    iymcspf, ixmcspf = np.array(np.where(temppfnumber > 0))
                    nmcspf = len(iymcspf)

                    if nmcspf > 0:
                        # Find unique precpitation features
                        allpfnumber = np.copy(temppfnumber[iymcspf, ixmcspf])
                        uniquepfnumber = np.unique(allpfnumber)
                        npfs = len(uniquepfnumber)

                        # Count number of pixels in each precipitaiton feature
                        pfnpix = np.ones(npfs, dtype=float)*fillvalue
                        for pfstep, ipf in enumerate(uniquepfnumber):
                            iypf, ixpf = np.array(np.where(temppfnumber == ipf))
                            pfnpix[pfstep] = len(iypf)
                        
                        # Sort by size
                        order = np.argsort(pfnpix)
                        order = order[::-1]
                        pfnpix = pfnpix[order]
                    
                        # Determine if number of precipitation features exceeds present maximum. If that is true only label as many specified by nmaxpf.
                        if npfs < nmaxpf:
                            nlabel = np.copy(npfs)
                        else:
                            nlabel = np.copy(nmaxpf)
                            
                        # Loop over each precipitation feature and label it with track number
                        for ilabel in range(0, nlabel):
                            iylabel, ixlabel = np.array(np.where(temppfnumber == uniquepfnumber[order[ilabel]]))
                            nlabelpix = len(ixlabel)
                            if nlabelpix > 0:
                                mcspfnumbermap_mergesplit[0, iylabel, ixlabel] = np.copy(imcs)

    mcssplitmap = mcssplitmap.astype(np.int32)
    mcsmergemap = mcsmergemap.astype(np.int32)
    mcsramap_mergesplit = mcsramap_mergesplit.astype(np.int32)
    mcspfnumbermap_mergesplit = mcspfnumbermap_mergesplit.astype(np.int32)
    mcstrackmap_mergesplit = mcstrackmap_mergesplit.astype(np.int32)
    mcstrackmap = mcstrackmap.astype(np.int32)
    statusmap = statusmap.astype(np.int32)

    #####################################################################
    # Output maps to netcdf file

    # Create output directories
    if not os.path.exists(mcstracking_path):
        os.makedirs(mcstracking_path)

    # Define output fileame
    mcstrackmaps_outfile = mcstracking_path + 'mcstracks_' + str(filedate) + '_' + str(filetime) + '.nc'

    # Check if file already exists. If exists, delete
    if os.path.isfile(mcstrackmaps_outfile):
        os.remove(mcstrackmaps_outfile)

    # Define xarray dataset
    output_data = xr.Dataset({'basetime': (['time'], cloudiddata['basetime'].data), \
                              'lon': (['nlat', 'nlon'], cloudiddata['longitude']), \
                              'lat': (['nlat', 'nlon'], cloudiddata['latitude']), \
                              'nclouds': (['time'], cloudiddata['nclouds'].data), \
                              'tb': (['time', 'nlat', 'nlon'], cloudiddata['tb'].data), \
                              'reflectivity': (['time', 'nlat', 'nlon'], pf_reflectivity), \
                              'csa': (['time', 'nlat', 'nlon'], pf_convstrat), \
                              'dbz0height': (['time', 'nlat', 'nlon'], pf_dbz0height), \
                              'dbz10height': (['time', 'nlat', 'nlon'], pf_dbz10height), \
                              'dbz20height': (['time', 'nlat', 'nlon'], pf_dbz20height), \
                              'dbz30height': (['time', 'nlat', 'nlon'], pf_dbz30height), \
                              'dbz40height': (['time', 'nlat', 'nlon'], pf_dbz40height), \
                              'precipitation': (['time', 'nlat', 'nlon'], ra_precipitation), \
                              'mask': (['time', 'nlat', 'nlon'], pf_mask), \
                              'cloudtype': (['time', 'nlat', 'nlon'], cloudiddata['cloudtype'].data), \
                              'cloudstatus': (['time', 'nlat', 'nlon'], statusmap), \
                              'alltracknumbers': (['time', 'nlat', 'nlon'], trackmap), \
                              'allsplittracknumbers': (['time', 'nlat', 'nlon'], allsplitmap), \
                              'allmergetracknumbers': (['time', 'nlat', 'nlon'], allmergemap), \
                              'mcssplittracknumbers': (['time', 'nlat', 'nlon'], mcssplitmap), \
                              'mcsmergetracknumbers': (['time', 'nlat', 'nlon'], mcsmergemap), \
                              'cloudnumber': (['time', 'nlat', 'nlon'], cloudiddata['convcold_cloudnumber'].data), \
                              'cloudtracknumber_nomergesplit': (['time', 'nlat', 'nlon'], mcstrackmap), \
                              'cloudtracknumber': (['time', 'nlat', 'nlon'], mcstrackmap_mergesplit), \
                              'pftracknumber': (['time', 'nlat', 'nlon'], mcspfnumbermap_mergesplit), \
                              'pcptracknumber': (['time', 'nlat', 'nlon'], mcsramap_mergesplit)}, \
                             coords={'time': (['time'], cloudiddata['basetime']), \
                                     'nlat': (['nlat'], np.arange(0, nlat)), \
                                     'nlon': (['nlon'], np.arange(0, nlon))}, \
                             attrs={'title':'Pixel level of tracked clouds and MCSs', \
                                    'source1': allmcsdata.attrs['source1'], \
                                    'source2': allmcsdata.attrs['source2'], \
                                    'description': allmcsdata.attrs['description'], \
                                    'Radar_Data_Present': pfpresent, \
                                    'Rain_Acccumulation_Data_Present': rapresent, \
                                    'MCS_IR_area_km2': allmcsdata.attrs['MCS_IR_area_km2'], \
                                    'MCS_IR_duration_hr': allmcsdata.attrs['MCS_IR_duration_hr'], \
                                    'MCS_IR_eccentricity': allmcsdata.attrs['MCS_IR_eccentricity'], \
                                    'MCS_PF_majoraxis_km': allmcsdata.attrs['MCS_PF_majoraxis_km'], \
                                    'MCS_PF_duration_hr': allmcsdata.attrs['MCS_PF_duration_hr'], \
                                    'MCS_core_aspectratio': allmcsdata.attrs['MCS_core_aspectratio'], \
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
    output_data.basetime.attrs['units'] = 'seconds'
    
    output_data.lon.attrs['long_name'] = 'Grid of longitude'
    output_data.lon.attrs['units'] = 'degrees'
    
    output_data.lat.attrs['long_name'] = 'Grid of latitude'
    output_data.lat.attrs['units'] = 'degrees'
    
    output_data.nclouds.attrs['long_name'] = 'Number of MCSs identified in this file'
    output_data.nclouds.attrs['units'] = 'unitless'
    
    output_data.tb.attrs['long_name'] = 'brightness temperature'
    output_data.tb.attrs['min_value'] =  mintb_thresh
    output_data.tb.attrs['max_value'] = maxtb_thresh
    output_data.tb.attrs['units'] = 'K'
    
    output_data.reflectivity.attrs['long_name'] = 'Radar reflectivity'
    output_data.reflectivity.attrs['units'] = 'dBZ'
    
    output_data.csa.attrs['long_name'] = 'Convective-stratiform classification'
    output_data.csa.attrs['values'] = '0=NAN, 1=LowCloud, 2=MidCloud, 3=ShallowCumulus, 4=IsolateConvective, 5=Stratiform, 6=Convective, 7=TransitionalAnvil, 8=MixAnvil, 9=IceAnvil'
    output_data.csa.attrs['units'] ='unitless'
    
    output_data.dbz0height.attrs['long_name'] = 'Maximum height of 0 dBZ contour'
    output_data.dbz0height.attrs['units'] = 'km'
    
    output_data.dbz10height.attrs['long_name'] = 'Maximum height of 10 dBZ contour'
    output_data.dbz10height.attrs['units'] = 'km'
    
    output_data.dbz20height.attrs['long_name'] = 'Maximum height of 20 dBZ contour'
    output_data.dbz20height.attrs['units'] = 'km'
    
    output_data.dbz30height.attrs['long_name'] = 'Maximum height of 30 dBZ contour'
    output_data.dbz30height.attrs['units'] = 'km'
    
    output_data.dbz40height.attrs['long_name'] = 'Maximum height of 40 dBZ contour'
    output_data.dbz40height.attrs['units'] = 'km'
    
    output_data.mask.attrs['long_name'] = 'Radar reflectivity mask'
    output_data.mask.attrs['values'] = '0=NoData, 1=Data Present'
    output_data.mask.attrs['units'] = 'unitless'
    
    output_data.precipitation.attrs['long_name'] = 'NMQ hourly rainfall accumulation (gauge bias removed)'
    output_data.precipitation.attrs['units'] = 'mm'

    output_data.cloudtype.attrs['long_name'] = 'flag indicating type of ir data'
    output_data.cloudtype.attrs['units'] = 'unitless'

    output_data.cloudstatus.attrs['long_name'] = 'flag indicating history of cloud'
    output_data.cloudstatus.attrs['units'] = 'unitless'

    output_data.alltracknumbers.attrs['long_name'] = 'Number of the cloud track associated with the cloud at a given pixel'
    output_data.alltracknumbers.attrs['units'] = 'unitless'

    output_data.allmergetracknumbers.attrs['long_name'] = 'Number of the cloud track that this cloud merges into'
    output_data.allmergetracknumbers.attrs['units'] = 'unitless'

    output_data.allsplittracknumbers.attrs['long_name'] = 'Number of the cloud track that this cloud splits from'
    output_data.allsplittracknumbers.attrs['units'] = 'unitless'

    output_data.mcsmergetracknumbers.attrs['long_name'] = 'Number of the mcs track that this cloud merges into'
    output_data.mcsmergetracknumbers.attrs['units'] = 'unitless'

    output_data.mcssplittracknumbers.attrs['long_name'] = 'Number of the mcs track that this cloud splits from'
    output_data.mcssplittracknumbers.attrs['units'] = 'unitless'

    output_data.cloudnumber.attrs['long_name'] = 'Number associated with the cloud at a given pixel'
    output_data.cloudnumber.attrs['comment'] = 'Extent of cloud system is defined using the warm anvil threshold'
    output_data.cloudnumber.attrs['units'] = 'unitless'
    
    output_data.cloudtracknumber_nomergesplit.attrs['long_name'] = 'Number of the tracked mcs associated with the cloud at a given pixel'
    output_data.cloudtracknumber_nomergesplit.attrs['units'] = 'unitless'
    
    output_data.cloudtracknumber.attrs['long_name'] = 'Number of the tracked mcs associated with the cloud at a given pixel'
    output_data.cloudtracknumber.attrs['comments'] = 'mcs includes smaller merges and splits'
    output_data.cloudtracknumber.attrs['units'] = 'unitless'
    
    output_data.pftracknumber.attrs['long_name'] = 'Number of the tracked mcs associated with the precipitation feature at a given pixel'
    output_data.pftracknumber.attrs['comments'] = 'mcs includes smaller merges and splits'
    output_data.pftracknumber.attrs['units'] = 'unitless'
    
    output_data.pcptracknumber.attrs['long_name'] = 'Number of the tracked mcs associated with the accumulated precipitation at a given pixel'
    output_data.pcptracknumber.attrs['comments'] = 'mcs includes smaller merges and splits'
    output_data.pcptracknumber.attrs['units'] = 'unitless'
    
    # Write netcdf file
    print(mcstrackmaps_outfile)
    print('')

    output_data.to_netcdf(path=mcstrackmaps_outfile, mode='w', format='NETCDF4_CLASSIC', unlimited_dims='time', \
                          encoding={'basetime': {'dtype': 'int', 'zlib':True, '_FillValue': fillvalue}, \
                                    'lon': {'zlib':True, '_FillValue': fillvalue}, \
                                    'lat': {'zlib':True, '_FillValue': fillvalue}, \
                                    'nclouds': {'zlib':True, '_FillValue': fillvalue}, \
                                    'tb': {'zlib':True, '_FillValue': fillvalue}, \
                                    'reflectivity': {'zlib':True, '_FillValue': fillvalue}, \
                                    'csa': {'dtype': 'int16', 'zlib':True, '_FillValue': fillvalue}, \
                                    'dbz0height': {'zlib':True, '_FillValue': fillvalue}, \
                                    'dbz10height': {'zlib':True, '_FillValue': fillvalue}, \
                                    'dbz20height': {'zlib':True, '_FillValue': fillvalue}, \
                                    'dbz30height': {'zlib':True, '_FillValue': fillvalue}, \
                                    'dbz40height': {'zlib':True, '_FillValue': fillvalue}, \
                                    'mask': {'zlib':True, '_FillValue': fillvalue}, \
                                    'precipitation': {'zlib':True, '_FillValue': fillvalue},
                                    'cloudtype': {'zlib':True, '_FillValue': fillvalue}, \
                                    'cloudstatus': {'zlib':True, '_FillValue': fillvalue}, \
                                    'allsplittracknumbers': {'zlib':True, '_FillValue': fillvalue}, \
                                    'allmergetracknumbers': {'zlib':True, '_FillValue': fillvalue}, \
                                    'mcssplittracknumbers': {'zlib':True, '_FillValue': fillvalue}, \
                                    'mcsmergetracknumbers': {'zlib':True, '_FillValue': fillvalue}, \
                                    'alltracknumbers': {'zlib':True, '_FillValue': fillvalue}, \
                                    'cloudnumber': {'dtype': 'int', 'zlib':True, '_FillValue': fillvalue}, \
                                    'cloudtracknumber_nomergesplit': {'zlib':True, '_FillValue': fillvalue}, \
                                    'cloudtracknumber': {'zlib':True, '_FillValue': fillvalue}, \
                                    'pftracknumber': {'zlib':True, '_FillValue': fillvalue}, \
                                    'pcptracknumber': {'zlib':True, '_FillValue': fillvalue}})



def mapmcs_mergedir(zipped_inputs):
    # Purpose: Subset statistics file to keep only MCS. Uses brightness temperature statstics of cold cloud shield area, duration, and eccentricity base on Fritsch et al (1986) and Maddos (1980)

    #######################################################################
    # Import modules
    import numpy as np
    from netCDF4 import Dataset
    import time
    import os
    import sys

    #####################################################################
    # Separate inputs
    filebasetime = zipped_inputs[0]
    mcsstats_filebase = zipped_inputs[1]
    statistics_filebase = zipped_inputs[2]
    mcstracking_path = zipped_inputs[3]
    stats_path = zipped_inputs[4]
    tracking_path = zipped_inputs[5]
    cloudid_filebase = zipped_inputs[6]
    absolutetb_threshs = zipped_inputs[7]
    startdate = zipped_inputs[8]
    enddate = zipped_inputs[9]

    ######################################################################
    # define constants:
    # minimum and maximum brightness temperature thresholds. data outside of this range is filtered
    mintb_thresh = absolutetb_threshs[0]    # k
    maxtb_thresh = absolutetb_threshs[1]    # k

    fillvalue = -9999

    ##################################################################
    # Load all track stat file
    statistics_file = stats_path + statistics_filebase + '_' + startdate + '_' + enddate + '.nc'
    print(statistics_file)

    allstatdata = Dataset(statistics_file, 'r')
    trackstat_basetime = allstatdata.variables['basetime'][:] # Time of cloud in seconds since 01/01/1970 00:00
    trackstat_cloudnumber = allstatdata.variables['cloudnumber'][:] # Number of the corresponding cloudid file
    trackstat_status = allstatdata.variables['status'][:] # Flag indicating the status of the cloud
    allstatdata.close()

    #######################################################################
    # Load MCS track stat file
    mcsstatistics_file = stats_path + mcsstats_filebase + startdate + '_' + enddate + '.nc'
    print(mcsstatistics_file)

    allmcsdata = Dataset(mcsstatistics_file, 'r')
    mcstrackstat_basetime = allmcsdata.variables['mcs_basetime'][:] # basetime of each cloud in the tracked mcs
    mcstrackstat_status = allmcsdata.variables['mcs_status'][:] # flag indicating the status of each cloud in the tracked mcs
    mcstrackstat_cloudnumber = allmcsdata.variables['mcs_cloudnumber'][:] # number of cloud in the corresponding cloudid file for each cloud in the tracked mcs
    mcstrackstat_mergecloudnumber = allmcsdata.variables['mcs_mergecloudnumber'][:] # number of cloud in the corresponding cloud file that merges into the tracked mcs
    mcstrackstat_splitcloudnumber = allmcsdata.variables['mcs_splitcloudnumber'][:] # number of cloud in the corresponding cloud file that splits into the tracked mcs
    source = str(Dataset.getncattr(allstatdata, 'source'))
    description = str(Dataset.getncattr(allstatdata, 'description'))
    pixel_radius = str(Dataset.getncattr(allstatdata, 'pixel_radius_km'))
    area_thresh = str(Dataset.getncattr(allstatdata, 'MCS_area_km**2'))
    duration_thresh = str(Dataset.getncattr(allstatdata, 'MCS_duration_hour'))
    eccentricity_thresh = str(Dataset.getncattr(allstatdata, 'MCS_eccentricity'))
    allmcsdata.close()

    #########################################################################
    # Get tracks and times associated with this time
    itrack, itime = np.array(np.where(mcstrackstat_basetime == filebasetime))
    timestatus = np.copy(mcstrackstat_status[itrack,itime])
    ntimes = len(itime)

    if ntimes > 0:
        # Get cloudid file associated with this time
        file_datetime = time.strftime("%Y%m%d_%H%M", time.gmtime(np.copy(filebasetime)))
        filedate = np.copy(file_datetime[0:8])
        filetime = np.copy(file_datetime[9:14])
        ifile = tracking_path + cloudid_filebase + file_datetime + '.nc'
        print(ifile)

        if os.path.isfile(ifile):
            # Load cloudid data
            cloudiddata = Dataset(ifile, 'r')
            cloudid_basetime = cloudiddata.variables['basetime'][:]
            cloudid_latitude = cloudiddata.variables['latitude'][:]
            cloudid_longitude = cloudiddata.variables['longitude'][:]
            cloudid_tb = cloudiddata.variables['tb'][:]
            cloudid_cloudnumber = cloudiddata.variables['cloudnumber'][:]
            cloudid_cloudtype = cloudiddata.variables['cloudtype'][:]
            cloudid_nclouds = cloudiddata.variables['nclouds'][:]
            cloudiddata.close()

            # Get data dimensions
            [timeindex, nlat, nlon] = np.shape(cloudid_cloudnumber)
                    
            # Intiailize track maps
            mcstrackmap = np.ones((nlat,nlon), dtype=int)*fillvalue
            mcstrackmap_mergesplit = np.ones((nlat,nlon), dtype=int)*fillvalue
            statusmap = np.ones((nlat,nlon), dtype=int)*fillvalue
            trackmap = np.ones((nlat,nlon), dtype=int)*fillvalue
            #mcstrackmap = np.zeros((nlat,nlon), dtype=int)
            #mcstrackmap_mergesplit = np.zeros((nlat,nlon), dtype=int)

            ###############################################################
            # Create map of status and track number for every feature in this file
            fulltrack, fulltime = np.array(np.where(trackstat_basetime == filebasetime))
            for ifull in range(0,len(fulltime)):
                ffcloudnumber = trackstat_cloudnumber[fulltrack[ifull], fulltime[ifull]]
                ffstatus = trackstat_status[fulltrack[ifull], fulltime[ifull]]
                
                fullypixels, fullxpixels = np.array(np.where(cloudid_cloudnumber[0,:,:] == ffcloudnumber))

                statusmap[fullypixels, fullxpixels] = ffstatus
                trackmap[fullypixels, fullxpixels] = fulltrack[ifull] + 1

            ##############################################################
            # Loop over each cloud in this unique file
            for jj in range(0,ntimes):
                print('JJ: ' + str(jj))
                # Get cloud nummber
                jjcloudnumber = mcstrackstat_cloudnumber[itrack[jj],itime[jj]]

                # Find pixels assigned to this cloud number
                jjcloudypixels, jjcloudxpixels = np.array(np.where(cloudid_cloudnumber[0,:,:] == jjcloudnumber))

                # Label this cloud with the track number. Need to add one to the cloud number since have the index number and we want the track number
                if len(jjcloudypixels) > 0:
                    mcstrackmap[jjcloudypixels, jjcloudxpixels] = itrack[jj] + 1
                    mcstrackmap_mergesplit[jjcloudypixels, jjcloudxpixels] = itrack[jj] + 1
                    print('All')
                    print(itrack)
                    print(itrack[jj])

                    #statusmap[jjcloudypixels, jjcloudxpixels] = timestatus[jj] 
                else:
                    sys.exit('Error: No matching cloud pixel found?!')

                ###########################################################
                # Find merging clouds
                jjmerge = np.array(np.where(mcstrackstat_mergecloudnumber[itrack[jj], itime[jj],:] > 0))[0,:]

                # Loop through merging clouds if present
                if len(jjmerge) > 0:
                    for imerge in jjmerge:
                        # Find cloud number asosicated with the merging cloud
                        jjmergeypixels, jjmergexpixels = np.array(np.where(cloudid_cloudnumber[0,:,:] == mcstrackstat_mergecloudnumber[itrack[jj], itime[jj], imerge]))
                        
                        # Label this cloud with the track number. Need to add one to the cloud number since have the index number and we want the track number
                        if len(jjmergeypixels) > 0:
                            mcstrackmap_mergesplit[jjmergeypixels, jjmergexpixels] = itrack[jj] + 1
                            print('Merge')
                            print(itrack)
                            print(itrack[jj])
                            #statusmap[jjmergeypixels, jjmergexpixels] = mcsmergestatus[itrack[jj], itime[jj], imerge]
                        else:
                            sys.exit('Error: No matching merging cloud pixel found?!')

                ###########################################################
                # Find splitting clouds
                jjsplit = np.array(np.where(mcstrackstat_splitcloudnumber[itrack[jj], itime[jj],:] > 0))[0,:]
                print(jjsplit)

                # Loop through splitting clouds if present
                if len(jjsplit) > 0:
                    for isplit in jjsplit:
                        print(isplit)
                        # Find cloud number asosicated with the splitting cloud
                        jjsplitypixels, jjsplitxpixels = np.array(np.where(cloudid_cloudnumber[0,:,:] == mcstrackstat_splitcloudnumber[itrack[jj], itime[jj], isplit]))
                                
                        # Label this cloud with the track number. Need to add one to the cloud number since have the index number and we want the track number
                        if len(jjsplitypixels) > 0:
                            mcstrackmap_mergesplit[jjsplitypixels, jjsplitxpixels] = itrack[jj] + 1
                            print('Split')
                            print(itrack)
                            print(itrack[jj])
                            #statusmap[jjsplitypixels, jjsplitxpixels] = mcssplitstatus[itrack[jj], itime[jj], isplit]
                        else:
                            sys.exit('Error: No matching splitting cloud pixel found?!')
            print('Stop')

            #####################################################################
            # Output maps to netcdf file

            # Create output directories
            if not os.path.exists(mcstracking_path + '/' + startdate + '_' + enddate + '/'):
                os.makedirs(mcstracking_path)

            # Create file
            mcsmcstrackmaps_outfile = mcstracking_path + 'mcstracks_' + str(filedate) + '_' + str(filetime) + '.nc'
            filesave = Dataset(mcsmcstrackmaps_outfile, 'w', format='NETCDF4_CLASSIC')

            # Set global attributes
            filesave.Convenctions = 'CF-1.6'
            filesave.title = 'Pixel level of tracked clouds and MCSs'
            filesave.institution = 'Pacific Northwest National Laboratory'
            filesave.setncattr('Contact', 'Hannah C Barnes: hannah.barnes@pnnl.gov')
            filesave.history = 'Created ' + time.ctime(time.time())
            filesave.setncattr('source', source)
            filesave.setncattr('description', description)
            filesave.setncattr('pixel_radius_km', pixel_radius)
            filesave.setncattr('MCS_area_km^2', area_thresh)
            filesave.setncattr('MCS_duration_hour', duration_thresh)
            filesave.setncattr('MCS_eccentricity', eccentricity_thresh)
                
            # Create dimensions
            filesave.createDimension('time', None)
            filesave.createDimension('lat', nlat)
            filesave.createDimension('lon', nlon)
            filesave.createDimension('ndatetimechars', 13)
            
            # Define variables
            basetime = filesave.createVariable('mcs_basetime', 'i4', ('time'), zlib=True, complevel=5, fill_value=fillvalue)
            basetime.standard_name = 'time'
            basetime.long_name = 'epoch time'
            basetime.description = 'basetime of clouds in this file'
            basetime.units = 'seconds since 01/01/1970 00:00'
            basetime.fill_value = fillvalue

            latitude = filesave.createVariable('latitude', 'f4', ('lat', 'lon'), zlib=True, complevel=5, fill_value=fillvalue)
            latitude.long_name = 'y-coordinate in Cartesian system'
            latitude.valid_min = np.nanmin(np.nanmin(cloudid_latitude))
            latitude.valid_max = np.nanmax(np.nanmax(cloudid_latitude))
            latitude.axis = 'Y'
            latitude.units = 'degrees_north'
            latitude.standard_name = 'latitude'
                    
            longitude = filesave.createVariable('longitude', 'f4', ('lat', 'lon'), zlib=True, complevel=5, fill_value=fillvalue)
            longitude.valid_min = np.nanmin(np.nanmin(cloudid_longitude))
            longitude.valid_max = np.nanmax(np.nanmax(cloudid_longitude))
            longitude.axis = 'X'
            longitude.long_name = 'x-coordinate in Cartesian system'
            longitude.units = 'degrees_east'
            longitude.standard_name = 'longitude'
                    
            nclouds = filesave.createVariable('nclouds', 'i4', 'time', zlib=True, complevel=5, fill_value=fillvalue)
            nclouds.long_name = 'number of distict convective cores identified in file'
            nclouds.units = 'unitless'
            
            tb = filesave.createVariable('tb', 'f4', ('time', 'lat', 'lon'), zlib=True, complevel=5, fill_value=fillvalue)
            tb.long_name = 'brightness temperature'
            tb.units = 'K'
            tb.valid_min = mintb_thresh
            tb.valid_max = maxtb_thresh
            tb.standard_name = 'brightness_temperature'
            tb.fill_value = fillvalue
            
            cloudnumber = filesave.createVariable('cloudnumber', 'i4', ('time', 'lat', 'lon'), zlib=True, complevel=5, fill_value=0)
            cloudnumber.long_name = 'number of cloud system that a given pixel belongs to'
            cloudnumber.units = 'unitless'
            cloudnumber.comment = 'the extend of the cloud system is defined using the warm anvil threshold'
            cloudnumber.fillvalue = 0
                    
            cloudstatus = filesave.createVariable('cloudstatus', 'i4', ('time', 'lat', 'lon'), zlib=True, complevel=5, fill_value=fillvalue)
            cloudstatus.long_name = 'flag indicating status of the flag'
            cloudstatus.values = "-9999=missing cloud or cloud removed due to short track, 0=track ends here, 1=cloud continues as one cloud in next file, 2=Biggest cloud in merger, 21=Smaller cloud(s) in merger, 13=Cloud that splits, 3=Biggest cloud from a split that stops after the split, 31=Smaller cloud(s) from a split that stop after the split. The last seven classifications are added together in different combinations to describe situations."
            cloudstatus.units = 'unitless'
            cloudstatus.comment = 'the extend of the cloud system is defined using the warm anvil threshold'
            cloudstatus.fillvalue = fillvalue 

            tracknumber = filesave.createVariable('tracknumber', 'f4', ('time', 'lat', 'lon'), zlib=True, complevel=5, fill_value=fillvalue)
            tracknumber.long_name = 'track number that a given pixel belongs to'
            tracknumber.units = 'unitless'
            tracknumber.comment = 'the extend of the cloud system is defined using the warm anvil threshold'
            tracknumber.fillvalue = fillvalue

            mcstracknumber = filesave.createVariable('mcstracknumber', 'f4', ('time', 'lat', 'lon'), zlib=True, complevel=5, fill_value=fillvalue)
            mcstracknumber.long_name = 'mcs track number that a given pixel belongs to'
            mcstracknumber.units = 'unitless'
            mcstracknumber.comment = 'the extend of the cloud system is defined using the warm anvil threshold'
                    
            mcstracknumber_mergesplit = filesave.createVariable('mcstracknumber_mergesplit', 'i4', ('time', 'lat', 'lon'), zlib=True, complevel=5, fill_value=fillvalue)
            mcstracknumber_mergesplit.long_name = 'mcs track number that a given pixel belongs to, includes clouds that merge into and split from each mcs'
            mcstracknumber_mergesplit.units = 'unitless'
            mcstracknumber_mergesplit.comment = 'the extend of the cloud system is defined using the warm anvil threshold'

            # Fill variables
            basetime[:] = cloudid_basetime
            longitude[:,:] = cloudid_longitude
            latitude[:,:] = cloudid_latitude
            nclouds[:] = cloudid_nclouds
            tb[0,:,:] = cloudid_tb
            cloudnumber[0,:,:] = cloudid_cloudnumber[:,:]
            cloudstatus[0,:,:] = statusmap[:,:]
            tracknumber[0,:,:] = trackmap[:,:]
            mcstracknumber[0,:,:] = mcstrackmap[:,:]
            mcstracknumber_mergesplit[0,:,:] = mcstrackmap_mergesplit[:,:]
                
            # Close and save file
            filesave.close()
                
        else:
            sys.exit(ifile + ' does not exist?!"')
    else:
        sys.exit('No MCSs')

