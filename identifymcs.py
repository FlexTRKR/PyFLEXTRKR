# Purpose: Subset statistics file to keep only MCS. Uses brightness temperature statstics of cold cloud shield area, duration, and eccentricity base on Fritsch et al (1986) and Maddos (1980)

# Author: Original IDL code written by Zhe Feng (zhe.feng@pnnl.gov), Python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov)

def mergedir(statistics_filebase, datasource, datadescription, mcstracking_path, tracking_path, cloudid_filebase, stats_path, startdate, enddate, time_resolution, area_thresh, duration_thresh, eccentricity_thresh, split_duration, merge_duration, absolutetb_threshs, nmaxmerge):
    #######################################################################
    # Import modules
    import numpy as np
    from netCDF4 import Dataset
    import time
    import os
    import sys

    ######################################################################
    # define constants:
    # minimum and maximum brightness temperature thresholds. data outside of this range is filtered
    mintb_thresh = absolutetb_threshs[0]    # k
    maxtb_thresh = absolutetb_threshs[1]    # k

    ##########################################################################
    # Load statistics file
    statistics_file = stats_path + statistics_filebase + '_' + startdate + '_' + enddate + '.nc'
    print(statistics_file)

    allstatdata = Dataset(statistics_file, 'r')
    ntracks_all = len(allstatdata.dimensions['ntracks']) # Total number of tracked features
    nmaxlength = len(allstatdata.dimensions['nmaxlength']) # Maximum number of features in a given track
    trackstat_length = allstatdata.variables['lifetime'][:] # Duration of each track
    trackstat_basetime = allstatdata.variables['basetime'][:] # Time of cloud in seconds since 01/01/1970 00:00
    trackstat_datetime = allstatdata.variables['datetimestrings'][:]
    trackstat_cloudnumber = allstatdata.variables['cloudnumber'][:] # Number of the corresponding cloudid file
    trackstat_status = allstatdata.variables['status'][:] # Flag indicating the status of the cloud
    trackstat_startstatus = allstatdata.variables['startstatus'][:] # Flag indicating the status of the first feature in each track
    trackstat_endstatus = allstatdata.variables['endstatus'][:] # Flag indicating the status of the last feature in each track 
    trackstat_mergenumbers = allstatdata.variables['mergenumbers'][:] # Number of a small feature that merges onto a bigger feature
    trackstat_splitnumbers = allstatdata.variables['splitnumbers'][:] # Number of a small feature that splits onto a bigger feature
    trackstat_eccentricity = allstatdata.variables['eccentricity'][:] # Eccentricity of the core and cold anvil
    trackstat_npix_core = allstatdata.variables['nconv'][:] # Number of pixels in the core
    trackstat_npix_corecold = allstatdata.variables['ncoldanvil'][:] # Number of pixels in the cold anvil
    trackstat_meanlat = allstatdata.variables['meanlat'][:] # Mean latitude of the core and cold anvil
    trackstat_meanlon = allstatdata.variables['meanlon'][:] # Mean longitude of the core and cold anvil
    tb_coldanvil = Dataset.getncattr(allstatdata, 'tb_coldavil') # Brightness temperature threshold for cold anvil
    pixel_radius = Dataset.getncattr(allstatdata, 'pixel_radisu_km') # Radius of one pixel in dataset
    source = str(Dataset.getncattr(allstatdata, 'source'))
    description = str(Dataset.getncattr(allstatdata, 'description'))
    track_version = str(Dataset.getncattr(allstatdata,'track_version'))
    tracknumbers_version = str(Dataset.getncattr(allstatdata, 'tracknumbers_version'))

    trackstat_latmin = allstatdata.variables['meanlat'].getncattr('min_value')
    trackstat_latmax = allstatdata.variables['meanlat'].getncattr('max_value')
    trackstat_lonmin = allstatdata.variables['meanlon'].getncattr('min_value')
    trackstat_lonmax = allstatdata.variables['meanlon'].getncattr('max_value')
    allstatdata.close()

    fillvalue = -9999

    ####################################################################
    # Set up thresholds

    # Cold Cloud Shield (CCS) area
    trackstat_corearea = trackstat_npix_core * pixel_radius**2
    trackstat_ccsarea = trackstat_npix_corecold * pixel_radius**2

    # Convert path duration to time
    trackstat_duration = trackstat_length * time_resolution

    ##################################################################
    # Initialize matrices
    trackid_mcs = []
    trackid_sql= []
    trackid_nonmcs = []

    mcstype = np.zeros(ntracks_all, dtype=int)
    mcsstatus = np.ones((ntracks_all, nmaxlength), dtype=float)*fillvalue

    ###################################################################
    # Identify MCSs
    for nt in range(0,ntracks_all):
        # Get data for a given track
        track_corearea = np.copy(trackstat_corearea[nt,:])
        track_ccsarea = np.copy(trackstat_ccsarea[nt,:])
        track_eccentricity = np.copy(trackstat_eccentricity[nt,:])

        # Remove fill values
        track_corearea = track_corearea[(track_corearea != fillvalue) & (track_corearea != 0)]
        track_ccsarea = track_ccsarea[track_ccsarea != fillvalue]
        track_eccentricity = track_eccentricity[track_eccentricity != fillvalue]

        # Must have a cold core
        if np.shape(track_corearea)[0] !=0 and np.nanmax(track_corearea > 0):

            # Cold cloud shield area requirement
            iccs = np.array(np.where(track_ccsarea > area_thresh))[0,:]
            nccs = len(iccs)

            # Find continuous times
            groups = np.split(iccs, np.where(np.diff(iccs) != 1)[0]+1)
            nbreaks = len(groups)

            # System may have multiple periods satisfying area and duration requirements. Loop over each period
            if iccs != []:
                for t in range(0,nbreaks):
                    # Duration requirement
                    if np.multiply(len(groups[t][:]), time_resolution) > duration_thresh:

                        # Isolate area and eccentricity for the subperiod
                        subtrack_ccsarea = track_ccsarea[groups[t][:]]
                        subtrack_eccentricity = track_eccentricity[groups[t][:]]

                        # Get eccentricity when the feature is the largest
                        subtrack_imax_ccsarea = np.nanargmax(subtrack_ccsarea)
                        subtrack_maxccsarea_eccentricity = subtrack_eccentricity[subtrack_imax_ccsarea]

                        # Apply eccentricity requirement
                        if subtrack_maxccsarea_eccentricity > eccentricity_thresh:
                            # Label as MCS
                            mcstype[nt] = 1
                            mcsstatus[nt,groups[t][:]] = 1
                        else:
                            # Label as squall line
                            mcstype[nt] = 2
                            mcsstatus[nt,groups[t][:]] = 2
                            trackid_sql = np.append(trackid_sql, nt)
                        trackid_mcs = np.append(trackid_mcs, nt)
                    else:
                        # Size requirement met but too short of a period
                        trackid_nonmcs = np.append(trackid_nonmcs, nt)
                        
            else:
                # Size requirement not met
                trackid_nonmcs = np.append(trackid_nonmcs, nt)

    ################################################################
    # Subset MCS / Squall track index
    trackid = np.array(np.where(mcstype > 0))[0,:]
    nmcs = len(trackid)
    print(nmcs)

    if nmcs > 0:
        mcsstatus = mcsstatus[trackid,:]
        mcstype = mcstype[trackid]

        mcslength = np.ones(len(mcstype), dtype=float)*fillvalue
        for imcs in range(0,nmcs):
            mcslength[imcs] = len(np.array(np.where(mcsstatus[imcs,:] != fillvalue))[0,:])

    # trackid_mcs is the index number, want the track number so add one
    mcstracknumbers = np.copy(trackid) + 1

    ###############################################################
    # Find small merging and spliting louds and add to MCS
    mcsmergecloudnumber = np.ones((nmcs, nmaxlength, nmaxmerge), dtype=int)*fillvalue
    mcsmergestatus = np.ones((nmcs, nmaxlength, nmaxmerge), dtype=int)*fillvalue
    mcssplitcloudnumber = np.ones((nmcs, nmaxlength, nmaxmerge), dtype=int)*fillvalue
    mcssplitstatus = np.ones((nmcs, nmaxlength, nmaxmerge), dtype=int)*fillvalue

    # Loop through each MCS and link small clouds merging in
    for imcs in np.arange(0,nmcs):

        ###################################################################################
        # Find mergers
        [mergefile, mergefeature] = np.array(np.where(trackstat_mergenumbers == mcstracknumbers[imcs]))

        # Loop through all merging tracks, if present
        if len(mergefile) > 0:
            # Isolate merging cases that have short duration
            mergefeature = mergefeature[trackstat_duration[mergefile] < merge_duration]
            mergefile = mergefile[trackstat_duration[mergefile] < merge_duration]

            # Make sure the merger itself is not an MCS
            mergingmcs = np.intersect1d(mergefile, mcstracknumbers)
            if len(mergingmcs) > 0:
                for iremove in np.arange(0,len(mergingmcs)):
                    removemerges = np.array(np.where(mergefile == mergingmcs[iremove]))[0,:]
                    mergefile[removemerges] = fillvalue
                    mergefeature[removemerges] = fillvalue
                mergefile = mergefile[mergefile != fillvalue]
                mergefeature = mergefeature[mergefeature != fillvalue]

            # Continue if mergers satisfy duration and MCS restriction
            if len(mergefile) > 0:

                # Get data about merging tracks
                mergingcloudnumber = np.copy(trackstat_cloudnumber[mergefile,mergefeature])
                mergingbasetime = np.copy(trackstat_basetime[mergefile,mergefeature])
                mergingstatus = np.copy(trackstat_status[mergefile,mergefeature])
                mergingdatetime = np.copy(trackstat_datetime[mergefile,mergefeature])

                # Get data about MCS track
                mcsbasetime = np.copy(trackstat_basetime[int(mcstracknumbers[imcs])-1,0:int(mcslength[imcs])])

                # Loop through each timestep in the MCS track
                for t in np.arange(0,mcslength[imcs]):

                    # Find merging cloud times that match current mcs track time
                    timematch = np.array(np.where(np.absolute(mergingbasetime - mcsbasetime[int(t)])<0.001)).astype(int)

                    if np.shape(timematch)[1] > 0:

                        # save cloud number of small mergers
                        nmergers = np.shape(timematch)[1]
                        mcsmergecloudnumber[imcs, int(t), 0:nmergers] = mergingcloudnumber[timematch[0,:]]
                        mcsmergestatus[imcs, int(t), 0:nmergers] = mergingstatus[timematch[0,:]]

                        #print('merge')
                        #print(mergingdatetime[timematch[0,:]])
                        #print(mcsmergestatus[imcs, int(t), 0:nmergers])
                        #print(mcsmergecloudnumber[imcs, int(t), 0:nmergers])
                        #raw_input('Waiting for user')

        ############################################################
        # Find splits
        [splitfile, splitfeature] = np.array(np.where(trackstat_splitnumbers == mcstracknumbers[imcs]))

        # Loop through all splitting tracks, if present
        if len(splitfile) > 0:
            # Isolate splitting cases that have short duration
            splitfeature = splitfeature[trackstat_duration[splitfile] < split_duration]
            splitfile = splitfile[trackstat_duration[splitfile] < split_duration]

            # Make sure the spliter itself is not an MCS
            splittingmcs = np.intersect1d(splitfile, mcstracknumbers)
            if len(splittingmcs) > 0:
                for iremove in np.arange(0,len(splittingmcs)):
                    removesplits = np.array(np.where(splitfile == splittingmcs[iremove]))[0,:]
                    splitfile[removesplits] = fillvalue
                    splitfeature[removesplits] = fillvalue
                splitfile = splitfile[splitfile != fillvalue]
                splitfeature = splitfeature[splitfeature != fillvalue]

            # Continue if spliters satisfy duration and MCS restriction
            if len(splitfile) > 0:

                # Get data about splitting tracks
                splittingcloudnumber = np.copy(trackstat_cloudnumber[splitfile, splitfeature])
                splittingbasetime = np.copy(trackstat_basetime[splitfile, splitfeature])
                splittingstatus = np.copy(trackstat_status[splitfile, splitfeature])
                splittingdatetime = np.copy(trackstat_datetime[splitfile, splitfeature])

                # Get data about MCS track
                mcsbasetime = np.copy(trackstat_basetime[int(mcstracknumbers[imcs])-1,0:int(mcslength[imcs])])

                # Loop through each timestep in the MCS track
                for t in np.arange(0,mcslength[imcs]):

                    # Find splitting cloud times that match current mcs track time
                    timematch = np.array(np.where(np.absolute(splittingbasetime - mcsbasetime[int(t)])<0.001)).astype(int)
                    if np.shape(timematch)[1] > 0:

                        # save cloud number of small splitrs
                        nspliters = np.shape(timematch)[1]
                        mcssplitcloudnumber[imcs, int(t), 0:nspliters] = splittingcloudnumber[timematch[0,:]]
                        mcssplitstatus[imcs, int(t), 0:nspliters] = splittingstatus[timematch[0,:]]

                        print('Split')
                        print(splittingdatetime[timematch[0,:]])
                        print(mcssplitstatus[imcs, int(t), 0:nspliters])
                        print(mcssplitcloudnumber[imcs, int(t), 0:nspliters])
                        raw_input('Waiting for user')

    ########################################################################
    # Subset keeping just MCS tracks
    trackid = trackid.astype(int)
    trackstat_duration = trackstat_duration[trackid]
    trackstat_basetime = trackstat_basetime[trackid,:]
    trackstat_datetime = trackstat_datetime[trackid,:]
    trackstat_cloudnumber = trackstat_cloudnumber[trackid,:]
    trackstat_status = trackstat_status[trackid,:]
    trackstat_corearea = trackstat_corearea[trackid,:]
    trackstat_meanlat = trackstat_meanlat[trackid,:]
    trackstat_meanlon = trackstat_meanlon[trackid,:] 
    trackstat_ccsarea = trackstat_ccsarea[trackid,:]
    trackstat_eccentricity = trackstat_eccentricity[trackid,:]
    trackstat_startstatus = trackstat_startstatus[trackid]
    trackstat_endstatus = trackstat_endstatus[trackid]

    ###########################################################################
    # Write statistics to netcdf file

    # Create file
    mcstrackstatistics_outfile = stats_path + 'mcs_tracks_' + startdate + '_' + enddate + '.nc'
    filesave = Dataset(mcstrackstatistics_outfile, 'w', format='NETCDF4_CLASSIC')

    # Set global attributes
    filesave.Convenctions = 'CF-1.6'
    filesave.title = 'File containing statistics for each track'
    filesave.institution = 'Pacific Northwest National Laboratory'
    filesave.setncattr('Contact', 'Hannah C Barnes: hannah.barnes@pnnl.gov')
    filesave.history = 'Created ' + time.ctime(time.time())
    filesave.setncattr('source', datasource)
    filesave.setncattr('description', datadescription)
    filesave.setncattr('startdate', startdate)
    filesave.setncattr('enddate', enddate)
    filesave.setncattr('time_resolution_hour', time_resolution)
    filesave.setncattr('pixel_radius_km', pixel_radius)
    filesave.setncattr('MCS_area_km^2', area_thresh)
    filesave.setncattr('MCS_duration_hour', duration_thresh)
    filesave.setncattr('MCS_eccentricity', eccentricity_thresh)

    # Create dimensions
    filesave.createDimension('ntracks', None)
    filesave.createDimension('ntimes', nmaxlength)
    filesave.createDimension('nmergers', nmaxmerge)
    filesave.createDimension('ndatetimechars', 13)

    # Define variables
    mcs_basetime = filesave.createVariable('mcs_basetime', 'i4', ('ntracks', 'ntimes'), zlib=True, complevel=5, fill_value=fillvalue)
    mcs_basetime.standard_name = 'time'
    mcs_basetime.long_name = 'epoch time'
    mcs_basetime.description = 'basetime of cloud at the given time'
    mcs_basetime.units = 'seconds since 01/01/1970 00:00'
    mcs_basetime.fill_value = fillvalue

    mcs_datetimestring = filesave.createVariable('mcs_datetimestring', 'S1', ('ntracks', 'ntimes', 'ndatetimechars'), zlib=True, complevel=5, fill_value=fillvalue)
    mcs_datetimestring.long_name = 'date-time'
    mcs_datetimestring.description = 'date_time for each cloud in the mcs'

    mcs_length = filesave.createVariable('mcs_length', 'i4', 'ntracks', zlib=True, complevel=5, fill_value=fillvalue)
    mcs_length.long_name = 'track duration'
    mcs_length.description = 'length of each MCS'
    mcs_length.units = 'hours'
    mcs_length.fill_value = fillvalue

    mcs_type = filesave.createVariable('mcs_type', 'i4', 'ntracks', zlib=True, complevel=5, fill_value=fillvalue)
    mcs_type.description = 'Type of MCS'
    mcs_type.values = '1=MCS, 2=Squall Line'
    mcs_type.fill_value = fillvalue
    mcs_type.units = 'unitless'

    mcs_status = filesave.createVariable('mcs_status', 'i4', ('ntracks', 'ntimes'), zlib=True, complevel=5, fill_value=fillvalue)
    mcs_status.long_name ='flag indicating the status of each feature in MCS'
    mcs_status.description = 'Numbers in each row describe how the clouds in that track evolve over time'
    mcs_status.values = '-9999=missing cloud or cloud removed due to short track, 0=track ends here, 1=cloud continues as one cloud in next file, 2=Biggest cloud in merger, 21=Smaller cloud(s) in merger, 13=Cloud that splits, 3=Biggest cloud from a split that stops after the split, 31=Smaller cloud(s) from a split that stop after the split. The last seven classifications are added together in different combinations to describe situations.'
    mcs_status.min_value = 0
    mcs_status.max_value = 52
    mcs_status.fill_value = fillvalue
    mcs_status.units = 'unitless'

    mcs_startstatus = filesave.createVariable('mcs_startstatus', 'i4', 'ntracks', zlib=True, complevel=5, fill_value=fillvalue)
    mcs_startstatus.description = 'flag indicating the status of the first cloud in MCSs'
    mcs_startstatus.values = '-9999=missing cloud or cloud removed due to short track, 0=track ends here, 1=cloud continues as one cloud in next file, 2=Biggest cloud in merger, 21=Smaller cloud(s) in merger, 13=Cloud that splits, 3=Biggest cloud from a split that stops after the split, 31=Smaller cloud(s) from a split that stop after the split. The last seven classifications are added together in different combinations to describe situations.'
    mcs_startstatus.min_value = 0
    mcs_startstatus.max_value = 52
    mcs_startstatus.fill_value = fillvalue
    mcs_startstatus.units = 'unitless'
    
    mcs_endstatus = filesave.createVariable('mcs_endstatus', 'i4', 'ntracks', zlib=True, complevel=5, fill_value=fillvalue)
    mcs_endstatus.description = 'flag indicating the status of the last cloud in MCSs'
    mcs_endstatus.values = '-9999=missing cloud or cloud removed due to short track, 0=track ends here, 1=cloud continues as one cloud in next file, 2=Biggest cloud in merger, 21=Smaller cloud(s) in merger, 13=Cloud that splits, 3=Biggest cloud from a split that stops after the split, 31=Smaller cloud(s) from a split that stop after the split. The last seven classifications are added together in different combinations to describe situations.'
    mcs_endstatus.min_value = 0
    mcs_endstatus.max_value = 52
    mcs_endstatus.fill_value = fillvalue
    mcs_endstatus.units = 'unitless'

    mcs_meanlat = filesave.createVariable('mcs_meanlat', 'f4', ('ntracks', 'ntimes'), zlib=True, complevel=5, fill_value=fillvalue)
    mcs_meanlat.standard_name = 'latitude'
    mcs_meanlat.description = 'Mean latitude of the core + cold anvil for each feature in the MCS'
    mcs_meanlat.valid_min = 'trackstat_latmin'
    mcs_meanlat.valid_max = 'trackstat_latmax'
    mcs_meanlat.fill_value = fillvalue
    mcs_meanlat.units = 'degrees'

    mcs_meanlon = filesave.createVariable('mcs_meanlon', 'f4', ('ntracks', 'ntimes'), zlib=True, complevel=5, fill_value=fillvalue)
    mcs_meanlon.standard_name = 'lonitude'
    mcs_meanlon.description = 'Mean longitude of the core + cold anvil for each feature at the given time'
    mcs_meanlon.valid_min = 'trackstat_lonmin'
    mcs_meanlon.valid_max = 'trackstat_lonmax'
    mcs_meanlon.fill_value = fillvalue
    mcs_meanlon.units = 'degrees'

    mcs_corearea = filesave.createVariable('mcs_corearea', 'f4', ('ntracks', 'ntimes'), zlib=True, complevel=5, fill_value=fillvalue)
    mcs_corearea.description = 'area of the cold core at the given time'
    mcs_corearea.fill_value = fillvalue
    mcs_corearea.units = 'km^2'

    mcs_ccsarea = filesave.createVariable('mcs_ccsarea', 'f4', ('ntracks', 'ntimes'), zlib=True, complevel=5, fill_value=fillvalue)
    mcs_ccsarea.description = 'are of cold core and cold anvil at the given time'
    mcs_ccsarea.fill_value = fillvalue
    mcs_ccsarea.units = 'km^2'

    mcs_cloudnumber = filesave.createVariable('mcs_cloudnumber', 'i4', ('ntracks', 'ntimes'), zlib=True, complevel=5, fill_value=fillvalue)
    mcs_cloudnumber.description = 'cloud number in the corresponding cloudid file of clouds in the mcs'
    mcs_cloudnumber.usage = 'To link this tracking statistics file with pixel-level cloudid files, use the cloudidfile and cloudnumber together to identify which cloud this current track and time is associated with'
    mcs_cloudnumber.fill_value = fillvalue
    mcs_cloudnumber.units = 'unitless'

    mcs_mergecloudnumber = filesave.createVariable('mcs_mergecloudnumber', 'i4', ('ntracks', 'ntimes', 'nmergers'), zlib=True, complevel=5, fill_value=fillvalue)
    mcs_mergecloudnumber.long_name = 'cloud number of small, short-lived clouds merging into the MCS'
    mcs_mergecloudnumber.fill_value = fillvalue
    mcs_mergecloudnumber.units = 'unitless'

    mcs_splitcloudnumber = filesave.createVariable('mcs_splitcloudnumber', 'i4', ('ntracks', 'ntimes', 'nmergers'), zlib=True, complevel=5, fill_value=fillvalue)
    mcs_splitcloudnumber.long_name = 'cloud number of small, short-lived clouds splitting from the MCS'
    mcs_splitcloudnumber.fill_value = fillvalue
    mcs_splitcloudnumber.units = 'unitless'

    # Fill variables
    mcs_basetime[:,:] = trackstat_basetime
    mcs_datetimestring[:,:,:] = trackstat_datetime
    mcs_length[:] = trackstat_duration
    mcs_type[:] = mcstype
    mcs_status[:,:] = trackstat_status
    mcs_startstatus[:] = trackstat_startstatus
    mcs_endstatus[:] = trackstat_endstatus
    mcs_meanlat[:,:] = trackstat_meanlat
    mcs_meanlon[:,:] = trackstat_meanlon
    mcs_corearea[:,:] = trackstat_corearea
    mcs_ccsarea[:,:] = trackstat_ccsarea
    mcs_cloudnumber[:,:] = trackstat_cloudnumber
    mcs_mergecloudnumber[:,:,:] = mcsmergecloudnumber
    mcs_splitcloudnumber[:,:,:] = mcssplitcloudnumber

    # Close and save file
    filesave.close()

    ##################################################################################
    # Create maps of all tracked MCSs
    if nmcs > 0:
        # Set default time range
        startbasetime = np.nanmin(trackstat_basetime)
        endbasetime = np.nanmax(trackstat_basetime)

        # Find unique times
        uniquebasetime = np.unique(trackstat_basetime)
        uniquebasetime = uniquebasetime[0:-1]
        nuniquebasetime = len(uniquebasetime)

        #########################################################################
        # Loop over each unique time
        for ub in uniquebasetime:
            # Get tracks and times associated with this time
            itrack, itime = np.array(np.where(trackstat_basetime == ub))
            timestatus = np.copy(trackstat_status[itrack,itime])
            ntimes = len(itime)
            
            if ntimes > 0:
                # Get cloudid file associated with this time
                file_datetime = time.strftime("%Y%m%d_%H%M", time.gmtime(np.copy(ub)))
                filedate = np.copy(file_datetime[0:8])
                filetime = np.copy(file_datetime[9:14])
                ifile = tracking_path + cloudid_filebase + file_datetime + '.nc'

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
                    trackmap = np.ones((nlat,nlon), dtype=int)*fillvalue
                    trackmap_mergesplit = np.ones((nlat,nlon), dtype=int)*fillvalue
                    statusmap = np.ones((nlat,nlon), dtype=int)*fillvalue 
                    #trackmap = np.zeros((nlat,nlon), dtype=int)
                    #trackmap_mergesplit = np.zeros((nlat,nlon), dtype=int)

                    ##############################################################
                    # Loop over each cloud in this unique file
                    for jj in range(0,ntimes):
                        # Get cloud nummber
                        jjcloudnumber = trackstat_cloudnumber[itrack[jj],itime[jj]]

                        # Find pixels assigned to this cloud number
                        jjcloudypixels, jjcloudxpixels = np.array(np.where(cloudid_cloudnumber[0,:,:] == jjcloudnumber))

                        # Label this cloud with the track number. Need to add one to the cloud number since have the index number and we want the track number
                        if len(jjcloudypixels) > 0:
                            trackmap[jjcloudypixels, jjcloudxpixels] = itrack[jj] + 1
                            trackmap_mergesplit[jjcloudypixels, jjcloudxpixels] = itrack[jj] + 1

                            statusmap[jjcloudypixels, jjcloudxpixels] = timestatus[jj] 
                        else:
                            sys.exit('Error: No matching cloud pixel found?!')

                        ###########################################################
                        # Find merging clouds
                        jjmerge = np.array(np.where(mcsmergecloudnumber[itrack[jj], itime[jj],:] > 0))[0,:]

                        # Loop through merging clouds if present
                        if len(jjmerge) > 0:
                            for imerge in jjmerge:
                                # Find cloud number asosicated with the merging cloud
                                jjmergeypixels, jjmergexpixels = np.array(np.where(cloudid_cloudnumber[0,:,:] == mcsmergecloudnumber[itrack[jj], itime[jj], imerge]))

                                # Label this cloud with the track number. Need to add one to the cloud number since have the index number and we want the track number
                                if len(jjmergeypixels) > 0:
                                    trackmap_mergesplit[jjmergeypixels, jjmergexpixels] = itrack[jj] + 1
                                    statusmap[jjmergeypixels, jjmergexpixels] = mcsmergestatus[itrack[jj], itime[jj], imerge]
                                else:
                                    sys.exit('Error: No matching merging cloud pixel found?!')

                        ###########################################################
                        # Find splitting clouds
                        jjsplit = np.array(np.where(mcssplitcloudnumber[itrack[jj], itime[jj],:] > 0))[0,:]

                        # Loop through splitting clouds if present
                        if len(jjsplit) > 0:
                            for isplit in jjsplit:
                                # Find cloud number asosicated with the splitting cloud
                                jjsplitypixels, jjsplitxpixels = np.array(np.where(cloudid_cloudnumber[0,:,:] == mcssplitcloudnumber[itrack[jj], itime[jj], isplit]))
                                
                                # Label this cloud with the track number. Need to add one to the cloud number since have the index number and we want the track number
                                if len(jjsplitypixels) > 0:
                                    trackmap_mergesplit[jjsplitypixels, jjsplitxpixels] = itrack[jj] + 1
                                    statusmap[jjsplitypixels, jjsplitxpixels] = mcssplitstatus[itrack[jj], itime[jj], isplit]  
                                else:
                                    sys.exit('Error: No matching splitting cloud pixel found?!')

                    #####################################################################
                    # Output maps to netcdf file

                    # Create output directories
                    if not os.path.exists(mcstracking_path):
                        os.makedirs(mcstracking_path)

                    # Create file
                    mcstrackmaps_outfile = mcstracking_path + 'mcstracks_' + str(filedate) + '_' + str(filetime) + '.nc'
                    filesave = Dataset(mcstrackmaps_outfile, 'w', format='NETCDF4_CLASSIC')

                    # Set global attributes
                    filesave.Convenctions = 'CF-1.6'
                    filesave.title = 'Pixel level of tracked clouds and MCSs'
                    filesave.institution = 'Pacific Northwest National Laboratory'
                    filesave.setncattr('Contact', 'Hannah C Barnes: hannah.barnes@pnnl.gov')
                    filesave.history = 'Created ' + time.ctime(time.time())
                    filesave.setncattr('source', datasource)
                    filesave.setncattr('description', datadescription)
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
                    tracknumber.long_name = 'mcs track number that a given pixel belongs to'
                    tracknumber.units = 'unitless'
                    tracknumber.comment = 'the extend of the cloud system is defined using the warm anvil threshold'
                    tracknumber.fillvalue = fillvalue

                    tracknumber_mergesplit = filesave.createVariable('tracknumber_mergesplit', 'i4', ('time', 'lat', 'lon'), zlib=True, complevel=5, fill_value=fillvalue)
                    tracknumber_mergesplit.long_name = 'mcs track number that a given pixel belongs to, includes clouds that merge into and split from each mcs'
                    tracknumber_mergesplit.units = 'unitless'
                    tracknumber_mergesplit.comment = 'the extend of the cloud system is defined using the warm anvil threshold'
                    tracknumber_mergesplit.fillvalue = fillvalue

                    # Fill variables
                    basetime[:] = cloudid_basetime
                    longitude[:,:] = cloudid_longitude
                    latitude[:,:] = cloudid_latitude
                    nclouds[:] = cloudid_nclouds
                    tb[0,:,:] = cloudid_tb
                    cloudnumber[0,:,:] = cloudid_cloudnumber[:,:]
                    cloudstatus[0,:,:] = statusmap[:,:]
                    tracknumber[0,:,:] = trackmap[:,:]
                    tracknumber_mergesplit[0,:,:] = trackmap_mergesplit[:,:]

                    # Close and save file
                    filesave.close()

                else:
                    sys.exit(ifile + ' does not exist?!"')

