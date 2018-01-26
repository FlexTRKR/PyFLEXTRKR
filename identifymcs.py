# Purpose: Subset statistics file to keep only MCS. Uses brightness temperature statstics of cold cloud shield area, duration, and eccentricity base on Fritsch et al (1986) and Maddos (1980)

# Author: Original IDL code written by Zhe Feng (zhe.feng@pnnl.gov), Python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov)

def identifymcs_mergedir(statistics_filebase, stats_path, startdate, enddate, geolimits, time_resolution, area_thresh, duration_thresh, eccentricity_thresh, split_duration, merge_duration, nmaxmerge):
    #######################################################################
    # Import modules
    import numpy as np
    from netCDF4 import Dataset
    import time
    import os
    import sys
    import xarray as xr
    import pandas as pd
    import time, datetime, calendar
    np.set_printoptions(threshold=np.inf)

    ##########################################################################
    # Load statistics file
    statistics_file = stats_path + statistics_filebase + '_' + startdate + '_' + enddate + '.nc'
    print(statistics_file)

    allstatdata = xr.open_dataset(statistics_file, autoclose=True)
    ntracks_all = (np.nanmax(allstatdata['ntracks'].data) + 1).astype(int) # Total number of tracked features
    nmaxlength = (np.nanmax(allstatdata['nmaxlength'].data) + 1).astype(int) # Maximum number of features in a given track

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
    trackstat_corearea = np.copy(allstatdata['nconv'].data) * np.copy(allstatdata.attrs['pixel_radius_km'])**2
    trackstat_ccsarea = (np.copy(allstatdata['nconv'].data) + np.copy(allstatdata['ncoldanvil'].data)) * np.copy(allstatdata.attrs['pixel_radius_km'])**2

    # Convert path duration to time
    trackstat_duration = np.copy(allstatdata['lifetime'].data) * time_resolution
    trackstat_duration = trackstat_duration.astype(np.int32)

    ##################################################################
    # Initialize matrices
    trackid_mcs = []
    trackid_sql= []
    trackid_nonmcs = []

    mcstype = np.zeros(ntracks_all, dtype=np.int32)
    mcsstatus = np.ones((ntracks_all, nmaxlength), dtype=np.int32)*fillvalue

    ###################################################################
    # Identify MCSs
    print(ntracks_all)
    for nt in range(0, ntracks_all):
        # Get data for a given track
        track_corearea = np.copy(trackstat_corearea[nt,:])
        track_ccsarea = np.copy(trackstat_ccsarea[nt,:])
        track_eccentricity = np.copy(allstatdata['eccentricity'].data)

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

        mcslength = np.ones(len(mcstype), dtype=np.int32)*fillvalue
        for imcs in range(0,nmcs):
            mcslength[imcs] = len(np.array(np.where(mcsstatus[imcs,:] != fillvalue))[0,:])

    # trackid_mcs is the index number, want the track number so add one
    mcstracknumbers = np.copy(trackid) + 1

    ###############################################################
    # Find small merging and spliting louds and add to MCS
    mcsmergecloudnumber = np.ones((nmcs, nmaxlength, nmaxmerge), dtype=np.int32)*fillvalue
    mcsmergestatus = np.ones((nmcs, nmaxlength, nmaxmerge), dtype=np.int32)*fillvalue
    mcssplitcloudnumber = np.ones((nmcs, nmaxlength, nmaxmerge), dtype=np.int32)*fillvalue
    mcssplitstatus = np.ones((nmcs, nmaxlength, nmaxmerge), dtype=np.int32)*fillvalue

    # Loop through each MCS and link small clouds merging in
    for imcs in np.arange(0,nmcs):

        ###################################################################################
        # Isolate basetime data
        print('')

        if imcs == 0:
            mcsbasetime = np.array([pd.to_datetime(allstatdata['basetime'][trackid[imcs], :].data, unit='s')])
        else:
            mcsbasetime = np.concatenate((mcsbasetime, np.array([pd.to_datetime(allstatdata['basetime'][trackid[imcs], :].data, unit='s')])), axis=0)

        ###################################################################################
        # Find mergers
        [mergefile, mergefeature] = np.array(np.where(allstatdata['mergenumbers'].data == mcstracknumbers[imcs]))

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
                mergingcloudnumber = np.copy(allstatdata['cloudnumber'][mergefile, :])
                mergingbasetime = np.copy(allstatdata['basetime'][mergefile, :])
                mergingstatus = np.copy(allstatdata['status'][mergefile, :])
                mergingdatetime = np.copy(allstatdata['datetimestrings'][mergefile, :, :])

                #mergingstatus = mergingstatus[mergingcloudnumber != fillvalue]
                #mergingdatetime = mergingdatetime[mergingcloudnumber != fillvalue]
                #mergingbasetime = mergingbasetime[mergingcloudnumber != fillvalue]
                #mergingcloudnumber = mergingcloudnumber[mergingcloudnumber != fillvalue]

                # Get data about MCS track
                imcsbasetime = np.copy(allstatdata['basetime'][int(mcstracknumbers[imcs])-1,0:int(mcslength[imcs])])

                #print(mcsbasetime)
                #print(mergingbasetime)
                #for itest in range(0, len(mergingbasetime)):
                #    print(np.where(mcsbasetime == mergingbasetime[itest]))
                #    print(np.where(trackstat_basetime[int(mcstracknumbers[imcs])-1,:] == mergingbasetime[itest]))
                #raw_input('waiting')

                # Loop through each timestep in the MCS track
                for t in np.arange(0,mcslength[imcs]):

                    # Find merging cloud times that match current mcs track time
                    timematch = np.where(mergingbasetime == imcsbasetime[int(t)])

                    if np.shape(timematch)[1] > 0:

                        # save cloud number of small mergers
                        nmergers = np.shape(timematch)[1]
                        mcsmergecloudnumber[imcs, int(t), 0:nmergers] = mergingcloudnumber[timematch]
                        mcsmergestatus[imcs, int(t), 0:nmergers] = mergingstatus[timematch]

                        #print('merge')
                        #print(mergingdatetime[timematch[0,:]])
                        #print(mcsmergestatus[imcs, int(t), 0:nmergers])
                        #print(mcsmergecloudnumber[imcs, int(t), 0:nmergers])
                        #raw_input('Waiting for user')

        ############################################################
        # Find splits
        [splitfile, splitfeature] = np.array(np.where(allstatdata['splitnumbers'].data == mcstracknumbers[imcs]))

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
                splittingcloudnumber = np.copy(allstatdata['cloudnumber'][splitfile, :])
                splittingbasetime = np.copy(allstatdata['basetime'][splitfile, :])
                splittingstatus = np.copy(allstatdata['status'][splitfile, :])
                splittingdatetime = np.copy(allstatdata['datetimestrings'][splitfile, :, :])

                #splittingstatus = splittingstatus[splittingcloudnumber != fillvalue]
                #splittingdatetime = splittingdatetime[splittingcloudnumber != fillvalue]
                #splittingbasetime = splittingbasetime[splittingcloudnumber != fillvalue]
                #splittingcloudnumber = splittingcloudnumber[splittingcloudnumber != fillvalue]

                # Get data about MCS track
                imcsbasetime = np.copy(allstatdata['basetime'][int(mcstracknumbers[imcs])-1,0:int(mcslength[imcs])])

                # Loop through each timestep in the MCS track
                for t in np.arange(0,mcslength[imcs]):

                    # Find splitting cloud times that match current mcs track time
                    timematch = np.where(splittingbasetime == imcsbasetime[int(t)])
                    if np.shape(timematch)[1] > 0:

                        # save cloud number of small splitrs
                        nspliters = np.shape(timematch)[1]
                        mcssplitcloudnumber[imcs, int(t), 0:nspliters] = splittingcloudnumber[timematch]
                        mcssplitstatus[imcs, int(t), 0:nspliters] = splittingstatus[timematch]

                        #print('Split')
                        #print(splittingdatetime[timematch[0,:]])
                        #print(mcssplitstatus[imcs, int(t), 0:nspliters])
                        #print(mcssplitcloudnumber[imcs, int(t), 0:nspliters])
                        #raw_input('Waiting for user')

    mcsmergecloudnumber = mcsmergecloudnumber.astype(np.int32)
    mcssplitcloudnumber = mcssplitcloudnumber.astype(np.int32)

    ###########################################################################
    # Write statistics to netcdf file

    # Create file
    mcstrackstatistics_outfile = stats_path + 'mcs_tracks_' + startdate + '_' + enddate + '.nc'

        # Check if file already exists. If exists, delete
    if os.path.isfile(mcstrackstatistics_outfile):
        os.remove(mcstrackstatistics_outfile)

    # Defie xarray dataset
    output_data = xr.Dataset({'mcs_basetime': (['ntracks', 'ntimes'], mcsbasetime), \
                              'mcs_datetimestring': (['ntracks', 'ntimes', 'ndatetimechars'], allstatdata['datetimestrings'][trackid, :, :]), \
                              'track_length': (['ntracks'], trackstat_duration[trackid]), \
                              'mcs_length': (['ntracks'],  mcslength), \
                              'mcs_type': (['ntracks'], mcstype), \
                              'mcs_status': (['ntracks', 'ntimes'], allstatdata['status'][trackid, :]), \
                              'mcs_startstatus': (['ntracks'], allstatdata['startstatus'][trackid]), \
                              'mcs_endstatus': (['ntracks'], allstatdata['endstatus'][trackid]), \
                              'mcs_boundary': (['ntracks'], allstatdata['boundary'][trackid]), \
                              'mcs_trackinterruptions': (['ntracks'], allstatdata['trackinterruptions'][trackid]), \
                              'mcs_meanlat': (['ntracks', 'ntimes'], allstatdata['meanlat'][trackid, :]), \
                              'mcs_meanlon': (['ntracks', 'ntimes'], allstatdata['meanlon'][trackid, :]), \
                              'mcs_corearea': (['ntracks', 'ntimes'], trackstat_corearea[trackid,:]), \
                              'mcs_ccsarea': (['ntracks', 'ntimes'], trackstat_ccsarea[trackid, :]), \
                              'mcs_cloudnumber': (['ntracks', 'ntimes'], allstatdata['cloudnumber'][trackid, :]), \
                              'mcs_mergecloudnumber': (['ntracks', 'ntimes', 'nmergers'], mcsmergecloudnumber), \
                              'mcs_splitcloudnumber': (['ntracks', 'ntimes', 'nmergers'], mcssplitcloudnumber)}, \
                             coords={'ntracks': (['ntracks'], np.arange(0, nmcs)), \
                                     'ntimes': (['ntimes'], np.arange(0, nmaxlength)), \
                                     'nmergers': (['nmergers'], np.arange(0, nmaxmerge)), \
                                     'ndatetimechars': (['ndatetimechars'], np.arange(0, 13))}, \
                             attrs={'title': 'File containing statistics for each mcs track', \
                                    'Conventions': 'CF-1.6', \
                                    'Institution': 'Pacific Northwest National Laboratory', \
                                    'Contact': 'Hannah C Barnes: hannah.barnes@pnnl.gov', \
                                    'Created_on': time.ctime(time.time()), \
                                    'source': allstatdata.attrs['source'], \
                                    'description': allstatdata.attrs['description'], \
                                    'startdate': startdate, \
                                    'enddate': enddate, \
                                    'time_resolution_hour': time_resolution, \
                                    'pixel_radius_km': allstatdata.attrs['pixel_radius_km'], \
                                    'MCS_duration_hr': duration_thresh, \
                                    'MCS_area_km^2': area_thresh, \
                                    'MCS_eccentricity': eccentricity_thresh})

    # Specify variable attributes
    output_data.ntracks.attrs['long_name'] = 'Number of mcss tracked'
    output_data.ntracks.attrs['units'] = 'unitless'

    output_data.ntimes.attrs['long_name'] = 'Maximum number of clouds in a mcs track'
    output_data.ntimes.attrs['units'] = 'unitless'

    output_data.nmergers.attrs['long_name'] = 'Maximum number of allowed mergers/splits into one cloud'
    output_data.nmergers.attrs['units'] = 'unitless'

    output_data.mcs_basetime.attrs['long_name'] = 'epoch time (seconds since 01/01/1970 00:00) of each cloud in a mcs track'
    output_data.mcs_basetime.attrs['standard_name'] = 'time'

    output_data.mcs_datetimestring.attrs['long_name'] = 'date_time for each cloud in a mcs track'
    output_data.mcs_datetimestring.attrs['units'] = 'unitless'

    output_data.track_length.attrs['long_name'] = 'Complete length/duration of each track containing an mcs each mcs track'
    output_data.track_length.attrs['units'] = 'hr'

    output_data.mcs_length.attrs['long_name'] = 'Length/duration of each mcs track'
    output_data.mcs_length.attrs['units'] = 'hr'

    output_data.mcs_type.attrs['long_name'] = 'Type of MCS'
    output_data.mcs_type.attrs['usage'] = '1=MCS, 2=Squall Line'
    output_data.mcs_type.attrs['units'] = 'unitless'

    output_data.mcs_status.attrs['long_name'] = 'flag indicating the status of each cloud in mcs track'
    output_data.mcs_status.attrs['units'] = 'unitless'
    output_data.mcs_status.attrs['valid_min'] = 0
    output_data.mcs_status.attrs['valid_max'] = 65

    output_data.mcs_startstatus.attrs['long_name'] = 'flag indicating the status of the first cloud in each mcs track'
    output_data.mcs_startstatus.attrs['units'] = 'unitless'
    output_data.mcs_startstatus.attrs['valid_min'] = 0
    output_data.mcs_startstatus.attrs['valid_max'] = 65
    
    output_data.mcs_endstatus.attrs['long_name'] =  'flag indicating the status of the last cloud in each mcs track'
    output_data.mcs_endstatus.attrs['units'] = 'unitless'
    output_data.mcs_endstatus.attrs['valid_min'] = 0
    output_data.mcs_endstatus.attrs['valid_max'] = 65
    
    output_data.mcs_boundary.attrs['long_name'] = 'Flag indicating whether the core + cold anvil touches one of the domain edges.'
    output_data.mcs_boundary.attrs['values'] = '0 = away from edge. 1= touches edge.'
    output_data.mcs_boundary.attrs['units'] = 'unitless'
    output_data.mcs_boundary.attrs['valid_min'] = 0
    output_data.mcs_boundary.attrs['valid_min'] = 1

    output_data.mcs_trackinterruptions.attrs['long_name'] = 'Flag indiciate if the track started and ended naturally or if the start or end of the track was artifically cut short by data availability'
    output_data.mcs_trackinterruptions.attrs['values'] = '0 = full track available, good data. 1 = track starts at first file, track cut short by data availability. 2 = track ends at last file, track cut short by data availability'
    output_data.mcs_trackinterruptions.attrs['units'] = 'unitless'
    output_data.mcs_trackinterruptions.attrs['valid_min'] = 0
    output_data.mcs_trackinterruptions.attrs['valid_min'] = 2

    output_data.mcs_meanlat.attrs['long_name'] = 'Mean latitude of the core + cold anvil for each feature in a mcs track'
    output_data.mcs_meanlat.attrs['standard_name'] = 'latitude'
    output_data.mcs_meanlat.attrs['units'] = 'degrees_north'
    output_data.mcs_meanlat.attrs['valid_min'] = geolimits[0]
    output_data.mcs_meanlat.attrs['valid_max'] = geolimits[2]

    output_data.mcs_meanlon.attrs['long_name'] = 'Mean longitude of the core + cold anvil for each feature in a mcs track'
    output_data.mcs_meanlon.attrs['standard_name'] = 'latitude'
    output_data.mcs_meanlon.attrs['units'] = 'degrees_north'
    output_data.mcs_meanlon.attrs['valid_min'] = geolimits[1]
    output_data.mcs_meanlon.attrs['valid_max'] = geolimits[3]

    output_data.mcs_corearea.attrs['long_name'] = 'Area of the cold core for each feature in a mcs track'
    output_data.mcs_corearea.attrs['units'] = 'km^2'

    output_data.mcs_corearea.attrs['long_name'] = 'Area of the cold core and cold anvil for each feature in a mcs track'
    output_data.mcs_corearea.attrs['units'] = 'km^2'

    output_data.mcs_cloudnumber.attrs['long_name'] = 'Number of each cloud in a track that cooresponds to the cloudid map'
    output_data.mcs_cloudnumber.attrs['units'] = 'unitless'
    output_data.mcs_cloudnumber.attrs['usuage'] = 'To link this tracking statistics file with pixel-level cloudid files, use the cloudidfile and cloudnumber together to identify which cloud this current track and time is associated with'

    output_data.mcs_mergecloudnumber.attrs['long_name'] = 'cloud number of small, short-lived feature merging into a mcs track'
    output_data.mcs_mergecloudnumber.attrs['units'] = 'unitless'

    output_data.mcs_splitcloudnumber.attrs['long_name'] = 'cloud number of small, short-lived feature splitting into a mcs track'
    output_data.mcs_splitcloudnumber.attrs['units'] = 'unitless'

    # Write netcdf file
    print(mcstrackstatistics_outfile)
    print('')

    output_data.to_netcdf(path=mcstrackstatistics_outfile, mode='w', format='NETCDF4_CLASSIC', unlimited_dims='ntracks', \
                          encoding={'mcs_basetime': {'zlib':True, '_FillValue': fillvalue, 'units': 'seconds since 1970-01-01'}, \
                                    'mcs_datetimestring': {'zlib':True, '_FillValue': fillvalue}, \
                                    'track_length': {'zlib':True, '_FillValue': fillvalue}, \
                                    'mcs_length': {'dtype':'int', 'zlib':True, '_FillValue': fillvalue}, \
                                    'mcs_type': {'dtype':'int', 'zlib':True, '_FillValue': fillvalue}, \
                                    'mcs_status': {'zlib':True, '_FillValue': fillvalue}, \
                                    'mcs_startstatus': {'zlib':True, '_FillValue': fillvalue}, \
                                    'mcs_endstatus': {'zlib':True, '_FillValue': fillvalue}, \
                                    'mcs_boundary': {'zlib':True, '_FillValue': fillvalue}, \
                                    'mcs_trackinterruptions': {'zlib':True, '_FillValue': fillvalue}, \
                                    'mcs_meanlat': {'zlib':True, '_FillValue': fillvalue}, \
                                    'mcs_meanlon': {'zlib':True, '_FillValue': fillvalue}, \
                                    'mcs_corearea': {'zlib':True, '_FillValue': fillvalue}, \
                                    'mcs_ccsarea': {'zlib':True, '_FillValue': fillvalue}, \
                                    'mcs_cloudnumber': {'zlib':True, '_FillValue': fillvalue}, \
                                    'mcs_mergecloudnumber': {'dtype':'int', 'zlib':True, '_FillValue': fillvalue}, \
                                    'mcs_splitcloudnumber': {'dtype':'int', 'zlib':True, '_FillValue': fillvalue}})

