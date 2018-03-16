# Purpose: Subset statistics file to keep only MCS. Uses brightness temperature statstics of cold cloud shield area, duration, and eccentricity base on Fritsch et al (1986) and Maddos (1980)

# Author: Original IDL code written by Zhe Feng (zhe.feng@pnnl.gov), Python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov)

def mergedir(statistics_filebase, datasource, datadescription, stats_path, startdate, enddate, time_resolution, area_thresh, duration_thresh, eccentricity_thresh, split_duration, merge_duration, nmaxmerge):
    #######################################################################
    # Import modules
    import numpy as np
    from netCDF4 import Dataset
    import time

    ######################################################################
    # Load statistics file
    statistics_file = stats_path + statistics_filebase + '_' + startdate + '_' + enddate + '.nc'
    print(statistics_file)

    allstatdata = Dataset(statistics_file, 'r')
    ntracks_all = len(allstatdata.dimensions['ntracks']) # Total number of tracked features
    nmaxlength = len(allstatdata.dimensions['nmaxlength']) # Maximum number of features in a given track
    length = allstatdata.variables['lifetime'][:] # Duration of each track
    basetime = allstatdata.variables['basetime'][:] # Time of cloud in seconds since 01/01/1970 00:00
    datetime = allstatdata.variables['datetimestrings'][:]
    meanlat = allstatdata.variables['meanlat'][:] # Mean latitude of the core and cold anvil
    meanlon = allstatdata.variables['meanlon'][:] # Mean longitude of the core and cold anvil
    cloudnumber = allstatdata.variables['cloudnumber'][:] # Number of the corresponding cloudid file
    status = allstatdata.variables['status'][:] # Flag indicating the status of the cloud
    startstatus = allstatdata.variables['startstatus'][:] # Flag indicating the status of the first feature in each track
    endstatus = allstatdata.variables['endstatus'][:] # Flag indicating the status of the last feature in each track 
    mergenumbers = allstatdata.variables['mergenumbers'][:] # Number of a small feature that merges onto a bigger feature
    splitnumbers = allstatdata.variables['splitnumbers'][:] # Number of a small feature that splits onto a bigger feature
    npix_corecold = allstatdata.variables['npix'][:] # Number of pixels in the core and cold anvil
    npix_core = allstatdata.variables['nconv'][:] # Number of pixels in the core
    npix_cold = allstatdata.variables['ncoldanvil'][:] # Number of pixels in the cold anvil
    majoraxis = allstatdata.variables['majoraxis'][:] # Length of the major axis of the core and cold anvil
    eccentricity = allstatdata.variables['eccentricity'][:] # Eccentricity of the core and cold anvil
    tb_coldanvil = Dataset.getncattr(allstatdata, 'tb_coldavil') # Brightness temperature threshold for cold anvil
    pixel_radius = Dataset.getncattr(allstatdata, 'pixel_radisu_km') # Radius of one pixel in dataset
    source = str(Dataset.getncattr(allstatdata, 'source'))
    description = str(Dataset.getncattr(allstatdata, 'description'))
    track_version = str(Dataset.getncattr(allstatdata,'track_version'))
    tracknumbers_version = str(Dataset.getncattr(allstatdata, 'tracknumbers_version'))
    allstatdata.close()

    fillvalue = -9999

    ####################################################################
    # Set up thresholds

    # Cold Cloud Shield (CCS) area
    corearea = npix_core * pixel_radius**2
    ccsarea = npix_corecold * pixel_radius**2

    # Convert path duration to time
    duration = length * time_resolution

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
        track_corearea = np.copy(corearea[nt,:])
        track_ccsarea = np.copy(ccsarea[nt,:])
        track_eccentricity = np.copy(eccentricity[nt,:])

        # Remove fill values
        track_corearea = track_corearea[(track_corearea != fillvalue) & (track_corearea != 0)]
        track_ccsarea = track_ccsarea[track_ccsarea != fillvalue]
        track_eccentricity = track_eccentricity[track_eccentricity != fillvalue]

        # Must have a cold core
        if np.shape(track_corearea)[0] !=0 and np.nanmax(track_corearea > 0):
            # Cold cloud shield area requirement
            iccs = np.array(np.where(track_ccsarea > area_thresh))[0,:]
            nccs = len(iccs)

            # Duration requirement
            if np.multiply(nccs, time_resolution) > duration_thresh:
                # Find continuous times
                groups = np.split(iccs, np.where(np.diff(iccs) != 1)[0]+1)
                [nbreaks,ntimes] = np.shape(groups)

                if nbreaks > 0:
                    # System may have multiple periods satisfying area and duration requirements. Loop over each period
                    for t in np.arange(0,nbreaks):
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
                            mscstatus[nt,groups[t][:]] = 2
                            trackid_sql = np.append(trackid_sql, nt)
                        trackid_mcs = np.append(trackid_mcs, nt)
            else:
                # Size requirement met but for too short of a period
                trackid_nonmcs = np.append(trackid_nonmcs, nt)
                        
        else:
            # Size requirement not met
            trackid_nonmcs = np.append(trackid_nonmcs, nt)

    ################################################################
    # Subset MCS / Squall track index
    trackid = np.array(np.where(mcstype > 0))[0,:]
    nmcs = len(trackid)

    if nmcs > 0:
        mcsstatus = mcsstatus[trackid,:]
        mcstype = mcstype[trackid]

        mcslength = np.ones(len(mcstype), dtype=float)*fillvalue
        for imcs in range(0,nmcs):
            mcslength[imcs] = len(np.array(np.where(mcsstatus[imcs,:] != fillvalue))[0,:])

    # trackid_mcs is the index number, want the track number so add one
    mcstracknumbers = np.copy(trackid_mcs) + 1

    ###############################################################
    # Find small merging and spliting louds and add to MCS
    mergecloudnumber = np.ones((nmcs, nmaxlength, nmaxmerge), dtype=int)*fillvalue
    splitcloudnumber = np.ones((nmcs, nmaxlength, nmaxmerge), dtype=int)*fillvalue

    # Loop through each MCS and link small clouds merging in
    for imcs in np.arange(0,len(mcslength)):

        ###################################################################################
        # Find mergers
        [mergefile, mergefeature] = np.array(np.where(mergenumbers == mcstracknumbers[imcs]))

        # Loop through all merging tracks, if present
        if len(mergefile) > 0:
            # Isolate merging cases that have short duration
            mergefile = mergefile[duration[mergefile] < merge_duration]
            mergefeature = mergefeature[duration[mergefile] < merge_duration]
            # Make sure the merger itself is not an MCS
            mergingmcs = np.intersect1d(mergefile, mcstracknumbers)
            if len(mergingmcs) > 0:
                for iremove in np.arange(0,len(mergingmcs)):
                    removemerges = np.array(np.where(mergefile == mergingmcs[iremove]))[0,:]
                    mergefile[removemerges] = np.nan
                    mergefeature[removemerges] = np.nan
                mergefile = mergefile[np.isfinite(mergefile)]
                mergefeature = mergefeature[np.isfinite(mergefeature)]

            # Continue if mergers satisfy duration and MCS restriction
            if len(mergefile) > 0:

                # Get data about merging tracks
                mergingcloudnumber = np.copy(cloudnumber[mergefile,:])
                mergingbasetime = np.copy(basetime[mergefile,:])

                # Get data about MCS track
                mcsbasetime = np.copy(basetime[int(mcstracknumbers[imcs])-1,0:int(mcslength[imcs])])

                # Loop through each timestep in the MCS track
                for t in np.arange(0,mcslength[imcs]):

                    # Find merging cloud times that match current mcs track time
                    timematch_rows, timematch_columns = np.array(np.where(np.absolute(mergingbasetime - mcsbasetime[int(t)])<0.001)).astype(int)
                    if len(timematch_rows) > 0:

                        # save cloud number of small mergers
                        nmergers = len(timematch_rows)
                        mergecloudnumber[imcs, int(t), 0:nmergers] = mergingcloudnumber[timematch_rows, timematch_columns]

        ############################################################
        # Find splits
        [splitfile, splitfeature] = np.array(np.where(splitnumbers == mcstracknumbers[imcs]))

        # Loop through all splitting tracks, if present
        if len(splitfile) > 0:
            # Isolate splitting cases that have short duration
            splitfile = splitfile[duration[splitfile] < split_duration]
            splitfeature = splitfeature[duration[splitfile] < split_duration]

            # Make sure the split itself is not an MCS
            splittingmcs = np.intersect1d(splitfile, mcstracknumbers)
            if len(splittingmcs) > 0:
                for iremove in np.arange(0,len(splittingmcs)):
                    removesplits = np.array(np.where(splitfile == splittingmcs[iremove]))[0,:]
                    splitfile[removesplits] = np.nan
                    splitfeature[removesplits] = np.nan
                splitfile = splitfile[np.isfinite(splitfile)]
                splitfeature = splitfeature[np.isfinite(splitfeature)]

            # Continue if spliters satisfy duration and MCS restriction
            if len(splitfile) > 0:

                # Get data about splitting tracks
                splittingcloudnumber = np.copy(cloudnumber[splitfile,:])
                splittingbasetime = np.copy(basetime[splitfile,:])

                # Get data about MCS track
                mcsbasetime = np.copy(basetime[int(mcstracknumbers[imcs])-1,0:int(mcslength[imcs])])
                
                # Loop through each timestep in the MCS track
                for t in np.arange(0,mcslength[imcs]):

                    # Find splitting cloud times that match current mcs track time
                    timematch_rows, timematch_columns = np.array(np.where(np.absolute(splittingbasetime - mcsbasetime[int(t)])<0.001)).astype(int)
                    if len(timematch_rows) > 0:

                        # save cloud number of small splitrs
                        nsplitrs = len(timematch_rows)
                        splitcloudnumber[imcs, int(t), 0:nsplitrs] = splittingcloudnumber[timematch_rows, timematch_columns]

    ########################################################################
    # Subset keeping just MCS tracks
    trackid_mcs = trackid_mcs.astype(int)
    duration = duration[trackid_mcs]
    basetime = basetime[trackid_mcs,:]
    datetime = datetime[trackid_mcs,:]
    meanlat = meanlat[trackid_mcs,:]
    meanlon = meanlon[trackid_mcs,:]
    cloudnumber = cloudnumber[trackid_mcs,:]
    status = status[trackid_mcs,:]
    corearea = corearea[trackid_mcs,:]
    ccsarea = ccsarea[trackid_mcs,:]
    majoraxis = majoraxis[trackid_mcs,:]
    eccentricity = eccentricity[trackid_mcs,:]
    startstatus = startstatus[trackid_mcs]
    endstatus = endstatus[trackid_mcs]

    ###########################################################################
    # Write to netcdf file

    # Create file
    mcstracks_outfile = stats_path + 'mcs_tracks_' + startdate + '_' + enddate + '.nc'
    filesave = Dataset(mcstracks_outfile, 'w', format='NETCDF4_CLASSIC')

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
    ntracks = filesave.createVariable('ntracks', 'i4', zlib=True, complevel=5, fill_value=fillvalue)
    ntracks.description = 'Number of MCS tracks'

    ntimes = filesave.createVariable('ntimes', 'i4', zlib=True, complevel=5, fill_value=fillvalue)
    ntimes.description = 'Maximum length of tracks, predefined'

    nmergers = filesave.createVariable('nmergers','i4', zlib=True, complevel=5, fill_value=fillvalue)
    nmergers.description = 'Maximum number of features that can merge at any given time, predefined'

    ndatetimechars = filesave.createVariable('ndatetimechars', 'i4', zlib=True, complevel=5, fill_value=fillvalue)
    ndatetimechars.description = 'Length of date string'

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
    mcs_meanlat.fill_value = fillvalue
    mcs_meanlat.units = 'degrees'

    mcs_meanlon = filesave.createVariable('mcs_meanlon', 'f4', ('ntracks', 'ntimes'), zlib=True, complevel=5, fill_value=fillvalue)
    mcs_meanlon.standard_name = 'lonitude'
    mcs_meanlon.description = 'Mean longitude of the core + cold anvil for each feature at the given time'
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

    mcs_mergecloudnumber = filesave.createVariable('mcs_mergecloudnumber', 'i4', ('ntracks', 'ntimes', 'nmergers'), zlib=True, complevel=5, fill_value=fillvalue)
    mcs_mergecloudnumber.long_name = 'cloud number of small, short-lived clouds merging into the MCS'
    mcs_mergecloudnumber.fill_value = fillvalue
    mcs_mergecloudnumber.units = 'unitless'

    mcs_splitcloudnumber = filesave.createVariable('mcs_splitcloudnumber', 'i4', ('ntracks', 'ntimes', 'nmergers'), zlib=True, complevel=5, fill_value=fillvalue)
    mcs_splitcloudnumber.long_name = 'cloud number of small, short-lived clouds splitting from the MCS'
    mcs_splitcloudnumber.fill_value = fillvalue
    mcs_splitcloudnumber.units = 'unitless'

    # Fill variables
    ntracks[:] = nmcs
    ntimes[:] = nmaxlength
    nmergers[:] = nmaxmerge
    ndatetimechars[:] = 13
    mcs_basetime[:,:] = basetime
    mcs_datetimestring[:,:,:] = datetime
    mcs_length[:] = duration
    mcs_type[:] = mcstype
    mcs_status[:,:] = status
    mcs_startstatus[:] = startstatus
    mcs_endstatus[:] = endstatus
    mcs_meanlat[:,:] = meanlat
    mcs_meanlon[:,:] = meanlon
    mcs_corearea[:,:] = corearea
    mcs_ccsarea[:,:] = ccsarea
    mcs_mergecloudnumber[:,:,:] = mergecloudnumber
    mcs_splitcloudnumber[:,:,:] = splitcloudnumber














