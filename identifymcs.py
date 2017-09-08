# Purpose: Subset statistics file to keep only MCS. Uses brightness temperature statstics of cold cloud shield area, duration, and eccentricity base on Fritsch et al (1986) and Maddos (1980)

# Author: Original IDL code written by Zhe Feng (zhe.feng@pnnl.gov), Python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov)

def identifymcs_mergedir(statistics_filebase, stats_path, startdate, enddate, time_resolution, area_thresh, duration_thresh, eccentricity_thresh, split_duration, merge_duration, nmaxmerge):
    #######################################################################
    # Import modules
    import numpy as np
    from netCDF4 import Dataset
    import time
    import os
    import sys

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
    trackstat_boundary = allstatdata.variables['boundary'][:] # Flag indicating whether the core + cold anvil touches one of the domain edges. 0 = away from edge. 1= touches edge.
    trackstat_trackinterruptions = allstatdata.variables['trackinterruptions'][:] # Numbers in each row indicate if the track started and ended naturally or if the start or end of the track was artifically cut short by data availability
    trackstat_eccentricity = allstatdata.variables['eccentricity'][:] # Eccentricity of the core and cold anvil
    trackstat_npix_core = allstatdata.variables['nconv'][:] # Number of pixels in the core
    trackstat_npix_cold = allstatdata.variables['ncoldanvil'][:] # Number of pixels in the cold anvil
    trackstat_meanlat = allstatdata.variables['meanlat'][:] # Mean latitude of the core and cold anvil
    trackstat_meanlon = allstatdata.variables['meanlon'][:] # Mean longitude of the core and cold anvil
    tb_coldanvil = Dataset.getncattr(allstatdata, 'tb_coldavil') # Brightness temperature threshold for cold anvil
    pixel_radius = Dataset.getncattr(allstatdata, 'pixel_radisu_km') # Radius of one pixel in dataset
    source = str(Dataset.getncattr(allstatdata, 'source'))
    description = str(Dataset.getncattr(allstatdata, 'description'))

    trackstat_latmin = allstatdata.variables['meanlat'].getncattr('min_value')
    trackstat_latmax = allstatdata.variables['meanlat'].getncattr('max_value')
    trackstat_lonmin = allstatdata.variables['meanlon'].getncattr('min_value')
    trackstat_lonmax = allstatdata.variables['meanlon'].getncattr('max_value')
    allstatdata.close()

    fillvalue = -9999
    
    #for itest in range(0, ntracks_all):
    #    print(trackstat_basetime[itest, 0:20])
    #    testy, testx = np.where(trackstat_mergenumbers == itest + 1)
    #    print(trackstat_basetime[testy, testx])
    #    if len(testy) > 0:
    #        for ii in range(0, len(testy)):
    #            print(np.where(trackstat_basetime[itest, :] == trackstat_basetime[testy[ii], testx[ii]]))
    #    raw_input('waiting')

    ####################################################################
    # Set up thresholds

    # Cold Cloud Shield (CCS) area
    trackstat_corearea = trackstat_npix_core * pixel_radius**2
    trackstat_ccsarea = (trackstat_npix_core + trackstat_npix_cold) * pixel_radius**2

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
                mergingcloudnumber = np.copy(trackstat_cloudnumber[mergefile, :])
                mergingbasetime = np.copy(trackstat_basetime[mergefile, :])
                mergingstatus = np.copy(trackstat_status[mergefile, :])
                mergingdatetime = np.copy(trackstat_datetime[mergefile, :, :])

                mergingstatus = mergingstatus[mergingcloudnumber != fillvalue]
                mergingdatetime = mergingdatetime[mergingcloudnumber != fillvalue]
                mergingbasetime = mergingbasetime[mergingcloudnumber != fillvalue]
                mergingcloudnumber = mergingcloudnumber[mergingcloudnumber != fillvalue]

                # Get data about MCS track
                mcsbasetime = np.copy(trackstat_basetime[int(mcstracknumbers[imcs])-1,0:int(mcslength[imcs])])

                #print(mcsbasetime)
                #print(mergingbasetime)
                #for itest in range(0, len(mergingbasetime)):
                #    print(np.where(mcsbasetime == mergingbasetime[itest]))
                #    print(np.where(trackstat_basetime[int(mcstracknumbers[imcs])-1,:] == mergingbasetime[itest]))
                #raw_input('waiting')

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
                splittingcloudnumber = np.copy(trackstat_cloudnumber[splitfile, :])
                splittingbasetime = np.copy(trackstat_basetime[splitfile, :])
                splittingstatus = np.copy(trackstat_status[splitfile, :])
                splittingdatetime = np.copy(trackstat_datetime[splitfile, :, :])

                splittingstatus = splittingstatus[splittingcloudnumber != fillvalue]
                splittingdatetime = splittingdatetime[splittingcloudnumber != fillvalue]
                splittingbasetime = splittingbasetime[splittingcloudnumber != fillvalue]
                splittingcloudnumber = splittingcloudnumber[splittingcloudnumber != fillvalue]

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

                        #print('Split')
                        #print(splittingdatetime[timematch[0,:]])
                        #print(mcssplitstatus[imcs, int(t), 0:nspliters])
                        #print(mcssplitcloudnumber[imcs, int(t), 0:nspliters])
                        #raw_input('Waiting for user')

    ########################################################################
    # Subset keeping just MCS tracks
    trackid = np.copy(trackid.astype(int))
    mcstrackstat_duration = np.copy(trackstat_duration[trackid])
    mcstrackstat_basetime = np.copy(trackstat_basetime[trackid,:])
    mcstrackstat_datetime = np.copy(trackstat_datetime[trackid,:])
    mcstrackstat_cloudnumber = np.copy(trackstat_cloudnumber[trackid,:])
    mcstrackstat_status = np.copy(trackstat_status[trackid,:])
    mcstrackstat_corearea = np.copy(trackstat_corearea[trackid,:])
    mcstrackstat_meanlat = np.copy(trackstat_meanlat[trackid,:])
    mcstrackstat_meanlon = np.copy(trackstat_meanlon[trackid,:]) 
    mcstrackstat_ccsarea = np.copy(trackstat_ccsarea[trackid,:])
    mcstrackstat_eccentricity = np.copy(trackstat_eccentricity[trackid,:])
    mcstrackstat_startstatus = np.copy(trackstat_startstatus[trackid])
    mcstrackstat_endstatus = np.copy(trackstat_endstatus[trackid])
    mcstrackstat_boundary = np.copy(trackstat_boundary[trackid])
    mcstrackstat_trackinterruptions = np.copy(trackstat_trackinterruptions[trackid])

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
    filesave.setncattr('source', source)
    filesave.setncattr('description', description)
    filesave.setncattr('startdate', startdate)
    filesave.setncattr('enddate', enddate)
    filesave.setncattr('time_resolution_hour', time_resolution)
    filesave.setncattr('pixel_radius_km', pixel_radius)
    filesave.setncattr('MCS_area_km**2', area_thresh)
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

    track_length = filesave.createVariable('track_length', 'i4', 'ntracks', zlib=True, complevel=5, fill_value=fillvalue)
    track_length.long_name = 'track duration'
    track_length.description = 'complete length of each track containing an mcs'
    track_length.units = 'hours'
    track_length.fill_value = fillvalue

    mcs_length = filesave.createVariable('mcs_length', 'i4', 'ntracks', zlib=True, complevel=5, fill_value=fillvalue)
    mcs_length.long_name = 'mcs duration'
    mcs_length.description = 'length of each MCS in each track'
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

    mcs_boundary = filesave.createVariable('mcs_boundary', 'i4', 'ntracks', zlib=True, complevel=5, fill_value=fillvalue)
    mcs_boundary.description = 'Flag indicating whether the core + cold anvil touches one of the domain edges. 0 = away from edge. 1= touches edge.'
    mcs_boundary.min_value = 0
    mcs_boundary.max_value = 1
    mcs_boundary.fill_value = fillvalue
    mcs_boundary.units = 'unitless'

    mcs_trackinterruptions = filesave.createVariable('mcs_trackinterruptions', 'i4', 'ntracks', zlib=True, complevel=5, fill_value=fillvalue)
    mcs_trackinterruptions.long_name = 'flag indication if track interrupted'
    mcs_trackinterruptions.description = 'Numbers in each row indicate if the track started and ended naturally or if the start or end of the track was artifically cut short by data availability'
    mcs_trackinterruptions.values = '0 = full track available, good data. 1 = track starts at first file, track cut short by data availability. 2 = track ends at last file, track cut short by data availability'
    mcs_trackinterruptions.min_value = 0
    mcs_trackinterruptions.max_value = 2
    mcs_trackinterruptions.fill_value = fillvalue
    mcs_trackinterruptions.units = 'unitless'

    mcs_meanlat = filesave.createVariable('mcs_meanlat', 'f4', ('ntracks', 'ntimes'), zlib=True, complevel=5, fill_value=fillvalue)
    mcs_meanlat.standard_name = 'latitude'
    mcs_meanlat.description = 'Mean latitude of the core + cold anvil for each feature in the MCS'
    mcs_meanlat.valid_min = trackstat_latmin
    mcs_meanlat.valid_max = trackstat_latmax
    mcs_meanlat.fill_value = fillvalue
    mcs_meanlat.units = 'degrees'

    mcs_meanlon = filesave.createVariable('mcs_meanlon', 'f4', ('ntracks', 'ntimes'), zlib=True, complevel=5, fill_value=fillvalue)
    mcs_meanlon.standard_name = 'lonitude'
    mcs_meanlon.description = 'Mean longitude of the core + cold anvil for each feature at the given time'
    mcs_meanlon.valid_min = trackstat_lonmin
    mcs_meanlon.valid_max = trackstat_lonmax
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
    mcs_basetime[:,:] = mcstrackstat_basetime
    mcs_datetimestring[:,:,:] = mcstrackstat_datetime
    track_length[:] = mcstrackstat_duration
    mcs_length[:] = mcslength
    mcs_type[:] = mcstype
    mcs_status[:,:] = mcstrackstat_status
    mcs_startstatus[:] = mcstrackstat_startstatus
    mcs_endstatus[:] = mcstrackstat_endstatus
    mcs_boundary[:] = mcstrackstat_boundary
    mcs_trackinterruptions[:] = mcstrackstat_trackinterruptions
    mcs_meanlat[:,:] = mcstrackstat_meanlat
    mcs_meanlon[:,:] = mcstrackstat_meanlon
    mcs_corearea[:,:] = mcstrackstat_corearea
    mcs_ccsarea[:,:] = mcstrackstat_ccsarea
    mcs_cloudnumber[:,:] = mcstrackstat_cloudnumber
    mcs_mergecloudnumber[:,:,:] = mcsmergecloudnumber
    mcs_splitcloudnumber[:,:,:] = mcssplitcloudnumber

    # Close and save file
    filesave.close()
