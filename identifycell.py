# Purpose: Subset statistics file to keep only MCS. Uses brightness temperature statstics of cold cloud shield area, duration, and eccentricity base on Fritsch et al (1986) and Maddos (1980)

# Author: Original IDL code written by Zhe Feng (zhe.feng@pnnl.gov), Python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov)

def identifycell_LES(statistics_filebase, stats_path, startdate, enddate, time_resolution, geolimits, maincloud_duration, merge_duration, split_duration, nmaxmerge):
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
    ntracks_all = np.nanmax(allstatdata['ntracks'].data) + 1 # Total number of tracked features
    nmaxlength = np.nanmax(allstatdata['nmaxlength'].data) + 1 # Maximum number of features in a given track
    #trackstat_length = allstatdata.variables['lifetime'][:] # Duration of each track
    #trackstat_basetime = allstatdata.variables['basetime'][:] # Time of cloud in seconds since 01/01/1970 00:00
    #trackstat_datetime = allstatdata.variables['datetimestrings'][:]
    #trackstat_cloudnumber = allstatdata.variables['cloudnumber'][:] # Number of the corresponding cloudid file
    #trackstat_status = allstatdata.variables['status'][:] # Flag indicating the status of the cloud
    #trackstat_startstatus = allstatdata.variables['startstatus'][:] # Flag indicating the status of the first feature in each track
    #trackstat_endstatus = allstatdata.variables['endstatus'][:] # Flag indicating the status of the last feature in each track 
    #trackstat_mergenumbers = allstatdata.variables['mergenumbers'][:] # Number of a small feature that merges onto a bigger feature
    #trackstat_splitnumbers = allstatdata.variables['splitnumbers'][:] # Number of a small feature that splits onto a bigger feature
    #trackstat_boundary = allstatdata.variables['boundary'][:] # Flag indicating whether the core + cold anvil touches one of the domain edges. 0 = away from edge. 1= touches edge.
    #trackstat_trackinterruptions = allstatdata.variables['trackinterruptions'][:] # Numbers in each row indicate if the track started and ended naturally or if the start or end of the track was artifically cut short by data availability
    #trackstat_eccentricity = allstatdata.variables['eccentricity'][:] # Eccentricity of the core and cold anvil
    #trackstat_npix_core = allstatdata.variables['nconv'][:] # Number of pixels in the core
    #trackstat_npix_cold = allstatdata.variables['ncoldanvil'][:] # Number of pixels in the cold anvil
    #trackstat_meanlat = allstatdata.variables['meanlat'][:] # Mean latitude of the core and cold anvil
    #trackstat_meanlon = allstatdata.variables['meanlon'][:] # Mean longitude of the core and cold anvil

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
    # Set up duration and area variables

    # Cold Cloud Shield (CCS) area
    trackstat_corearea = np.copy(allstatdata['nconv'].data) * np.copy(allstatdata.attrs['pixel_radius_km'])**2
    trackstat_ccsarea = (np.copy(allstatdata['nconv'].data) + np.copy(allstatdata['ncoldanvil'].data)) * np.copy(allstatdata.attrs['pixel_radius_km'])**2

    # Convert path duration to time
    trackstat_duration = np.copy(allstatdata['lifetime'].data) * time_resolution

    ##################################################################
    # Initialize matrices
    trackid_cell = []

    ###################################################################
    # Identify main cells
    maincelltrackid = np.array(np.where(trackstat_duration >= maincloud_duration))[0, :]
    nmaincell = len(maincelltrackid)
    maincelltracknumbers = np.add(maincelltrackid, 1)

    maincellcloudnumbers = np.copy(allstatdata['cloudnumber'][maincelltrackid, :])
    maincelllength = np.copy(allstatdata['lifetime'][maincelltrackid])

    ###############################################################
    # Find small merging and spliting louds and add to MCS
    cellmergecloudnumber = np.ones((nmaincell, int(nmaxlength), nmaxmerge), dtype=int)*fillvalue
    cellmergestatus = np.ones((nmaincell, int(nmaxlength), nmaxmerge), dtype=int)*fillvalue
    cellsplitcloudnumber = np.ones((nmaincell, int(nmaxlength), nmaxmerge), dtype=int)*fillvalue
    cellsplitstatus = np.ones((nmaincell, int(nmaxlength), nmaxmerge), dtype=int)*fillvalue

    # Loop through each MCS and link small clouds merging in
    print('Number of main cells: ' + str(int(nmaincell)))
    for icell in np.arange(0, nmaincell):
        print('')
        print(icell)

        ###################################################################################
        # Isolate basetime data

        if icell == 0:
            cellbasetime = np.array([pd.to_datetime(allstatdata['basetime'][maincelltrackid[icell], :].data, unit='s')])
        else:
            cellbasetime = np.concatenate((cellbasetime, np.array([pd.to_datetime(allstatdata['basetime'][maincelltrackid[icell], :].data, unit='s')])), axis=0)

        print('Determining Base Time')

        ###################################################################################
        # Find mergers
        print('Determining mergers')
        [mergefile, mergefeature] = np.array(np.where(np.copy(allstatdata['mergenumbers'].data) == maincelltracknumbers[icell]))

        # Loop through all merging tracks, if present
        print(len(mergefile))
        if len(mergefile) > 0:
            # Isolate merging cases that have short duration
            mergefeature = mergefeature[trackstat_duration[mergefile] < merge_duration]
            mergefile = mergefile[trackstat_duration[mergefile] < merge_duration]

            # Make sure the merger itself is not a cell
            mergingcell = np.intersect1d(mergefile, maincelltracknumbers)
            if len(mergingcell) > 0:
                for iremove in np.arange(0,len(mergingcell)):
                    removemerges = np.array(np.where(mergefile == mergingcell[iremove]))[0,:]
                    mergefile[removemerges] = fillvalue
                    mergefeature[removemerges] = fillvalue
                mergefile = mergefile[mergefile != fillvalue]
                mergefeature = mergefeature[mergefeature != fillvalue]

            # Continue if mergers satisfy duration and CELL restriction
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

                # Get data about CELL track
                maincellbasetime = np.copy(allstatdata['basetime'][int(maincelltracknumbers[icell])-1, 0:int(maincelllength[icell])])

                #print(cellbasetime)
                #print(mergingbasetime)
                #for itest in range(0, len(mergingbasetime)):
                #    print(np.where(cellbasetime == mergingbasetime[itest]))
                #    print(np.where(trackstat_basetime[int(celltracknumbers[icell])-1,:] == mergingbasetime[itest]))
                #raw_input('waiting')

                # Loop through each timestep in the CELL track
                for t in np.arange(0, maincelllength[icell]):

                    # Find merging cloud times that match current cell track time
                    timematch = np.where(mergingbasetime == maincellbasetime[int(t)])

                    if np.shape(timematch)[1] > 0:

                        # save cloud number of small mergers
                        nmergers = np.shape(timematch)[1]
                        cellmergecloudnumber[icell, int(t), 0:nmergers] = mergingcloudnumber[timematch]
                        cellmergestatus[icell, int(t), 0:nmergers] = mergingstatus[timematch]

                        #print('merge')
                        #print(mergingdatetime[timematch[0,:]])
                        #print(cellmergestatus[icell, int(t), 0:nmergers])
                        #print(cellmergecloudnumber[icell, int(t), 0:nmergers])
                        #raw_input('Waiting for user')

        ############################################################
        # Find splits
        print('Determining splits')
        [splitfile, splitfeature] = np.array(np.where(np.copy(allstatdata['splitnumbers'].data) == maincelltracknumbers[icell]))

        # Loop through all splitting tracks, if present
        print(len(splitfile))
        if len(splitfile) > 0:
            # Isolate splitting cases that have short duration
            splitfeature = splitfeature[trackstat_duration[splitfile] < split_duration]
            splitfile = splitfile[trackstat_duration[splitfile] < split_duration]

            # Make sure the spliter itself is not an CELL
            splittingcell = np.intersect1d(splitfile, maincelltracknumbers)
            if len(splittingcell) > 0:
                for iremove in np.arange(0, len(splittingcell)):
                    removesplits = np.array(np.where(splitfile == splittingcell[iremove]))[0,:]
                    splitfile[removesplits] = fillvalue
                    splitfeature[removesplits] = fillvalue
                splitfile = splitfile[splitfile != fillvalue]
                splitfeature = splitfeature[splitfeature != fillvalue]
                
            # Continue if spliters satisfy duration and CELL restriction
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

                # Get data about CELL track
                maincellbasetime = np.copy(allstatdata['basetime'][int(maincelltracknumbers[icell])-1,0:int(maincelllength[icell])])

                # Loop through each timestep in the CELL track
                for t in np.arange(0, maincelllength[icell]):

                    # Find splitting cloud times that match current cell track time
                    timematch = np.where(splittingbasetime == maincellbasetime[int(t)])
                    if np.shape(timematch)[1] > 0:

                        # save cloud number of small splitrs
                        nspliters = np.shape(timematch)[1]
                        cellsplitcloudnumber[icell, int(t), 0:nspliters] = splittingcloudnumber[timematch]
                        cellsplitstatus[icell, int(t), 0:nspliters] = splittingstatus[timematch]

                        #print('Split')
                        #print(splittingdatetime[timematch[0,:]])
                        #print(cellsplitstatus[icell, int(t), 0:nspliters])
                        #print(cellsplitcloudnumber[icell, int(t), 0:nspliters])
                        #raw_input('Waiting for user')

    ########################################################################
    # Subset keeping just CELL tracks
    maincelltrackid = np.copy(maincelltrackid.astype(int))

    ###########################################################################
    # Write statistics to netcdf file

    # Create file
    celltrackstatistics_outfile = stats_path + 'cell_tracks_' + startdate + '_' + enddate + '.nc'

    # Check if file already exists. If exists, delete
    if os.path.isfile(celltrackstatistics_outfile):
        os.remove(celltrackstatistics_outfile)

    # Defie xarray dataset
    output_data = xr.Dataset({'cell_basetime': (['ntracks', 'ntimes'], cellbasetime), \
                              'cell_datetimestring': (['ntracks', 'ntimes', 'ndatetimechars'], allstatdata['datetimestrings'][maincelltrackid,:]), \
                              'cell_length': (['ntracks'], trackstat_duration[maincelltrackid]), \
                              'cell_status': (['ntracks', 'ntimes'], allstatdata['status'][maincelltrackid, :]), \
                              'cell_startstatus': (['ntracks'], allstatdata['startstatus'][maincelltrackid]), \
                              'cell_endstatus': (['ntracks'], allstatdata['endstatus'][maincelltrackid]), \
                              'cell_boundary': (['ntracks'], allstatdata['boundary'][maincelltrackid]), \
                              'cell_trackinterruptions': (['ntracks'], allstatdata['trackinterruptions'][maincelltrackid]), \
                              'cell_meanlat': (['ntracks', 'ntimes'], allstatdata['meanlat'][maincelltrackid, :]), \
                              'cell_meanlon': (['ntracks', 'ntimes'], allstatdata['meanlon'][maincelltrackid, :]), \
                              'cell_corearea': (['ntracks', 'ntimes'], trackstat_corearea[maincelltrackid]), \
                              'cell_ccsarea': (['ntracks', 'ntimes'], trackstat_ccsarea[maincelltrackid]), \
                              'cell_cloudnumber': (['ntracks', 'ntimes'], allstatdata['cloudnumber'][maincelltrackid, :]), \
                              'cell_mergecloudnumber': (['ntracks', 'ntimes', 'nmergers'], cellmergecloudnumber), \
                              'cell_splitcloudnumber': (['ntracks', 'ntimes', 'nmergers'], cellsplitcloudnumber)}, \
                             coords={'ntracks': (['ntracks'], np.arange(0, nmaincell)), \
                                     'ntimes': (['ntimes'], np.arange(0, nmaxlength)), \
                                     'nmergers': (['nmergers'], np.arange(0, nmaxmerge)), \
                                     'ndatetimechars': (['ndatetimechars'], np.arange(0, 13))}, \
                             attrs={'title': 'File containing statistics for each cell track', \
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
                                    'Main_cell_duration_hr': str(maincloud_duration), \
                                    'Merge_duration_hr': str(merge_duration), \
                                    'Split_duration_hr': str(split_duration)})

    # Specify variable attributes
    output_data.ntracks.attrs['long_name'] = 'Number of cells tracked'
    output_data.ntracks.attrs['units'] = 'unitless'

    output_data.ntimes.attrs['long_name'] = 'Maximum number of clouds in a cell track'
    output_data.ntimes.attrs['units'] = 'unitless'

    output_data.nmergers.attrs['long_name'] = 'Maximum number of allowed mergers/splits into one cloud'
    output_data.nmergers.attrs['units'] = 'unitless'

    output_data.cell_basetime.attrs['long_name'] = 'epoch time (seconds since 01/01/1970 00:00) of each cloud in a cell track'
    output_data.cell_basetime.attrs['standard_name'] = 'time'

    output_data.cell_datetimestring.attrs['long_name'] = 'date_time for each cloud in a cell track'
    output_data.cell_datetimestring.attrs['units'] = 'unitless'

    output_data.cell_length.attrs['long_name'] = 'Length/duration of each cell track'
    output_data.cell_length.attrs['units'] = 'hr'

    output_data.cell_status.attrs['long_name'] = 'flag indicating the status of each cloud in cell track'
    output_data.cell_status.attrs['units'] = 'unitless'
    output_data.cell_status.attrs['valid_min'] = 0
    output_data.cell_status.attrs['valid_max'] = 65

    output_data.cell_startstatus.attrs['long_name'] = 'flag indicating the status of the first cloud in each cell track'
    output_data.cell_startstatus.attrs['units'] = 'unitless'
    output_data.cell_startstatus.attrs['valid_min'] = 0
    output_data.cell_startstatus.attrs['valid_max'] = 65
    
    output_data.cell_endstatus.attrs['long_name'] =  'flag indicating the status of the last cloud in each cell track'
    output_data.cell_endstatus.attrs['units'] = 'unitless'
    output_data.cell_endstatus.attrs['valid_min'] = 0
    output_data.cell_endstatus.attrs['valid_max'] = 65
    
    output_data.cell_boundary.attrs['long_name'] = 'Flag indicating whether the core + cold anvil touches one of the domain edges.'
    output_data.cell_boundary.attrs['values'] = '0 = away from edge. 1= touches edge.'
    output_data.cell_boundary.attrs['units'] = 'unitless'
    output_data.cell_boundary.attrs['valid_min'] = 0
    output_data.cell_boundary.attrs['valid_min'] = 1

    output_data.cell_trackinterruptions.attrs['long_name'] = 'Flag indiciate if the track started and ended naturally or if the start or end of the track was artifically cut short by data availability'
    output_data.cell_trackinterruptions.attrs['values'] = '0 = full track available, good data. 1 = track starts at first file, track cut short by data availability. 2 = track ends at last file, track cut short by data availability'
    output_data.cell_trackinterruptions.attrs['units'] = 'unitless'
    output_data.cell_trackinterruptions.attrs['valid_min'] = 0
    output_data.cell_trackinterruptions.attrs['valid_min'] = 2

    output_data.cell_meanlat.attrs['long_name'] = 'Mean latitude of the core + cold anvil for each feature in a cell track'
    output_data.cell_meanlat.attrs['standard_name'] = 'latitude'
    output_data.cell_meanlat.attrs['units'] = 'degrees_north'
    output_data.cell_meanlat.attrs['valid_min'] = geolimits[0]
    output_data.cell_meanlat.attrs['valid_max'] = geolimits[2]

    output_data.cell_meanlon.attrs['long_name'] = 'Mean longitude of the core + cold anvil for each feature in a cell track'
    output_data.cell_meanlon.attrs['standard_name'] = 'latitude'
    output_data.cell_meanlon.attrs['units'] = 'degrees_north'
    output_data.cell_meanlon.attrs['valid_min'] = geolimits[1]
    output_data.cell_meanlon.attrs['valid_max'] = geolimits[3]

    output_data.cell_corearea.attrs['long_name'] = 'Area of the cold core for each feature in a cell track'
    output_data.cell_corearea.attrs['units'] = 'km^2'

    output_data.cell_corearea.attrs['long_name'] = 'Area of the cold core and cold anvil for each feature in a cell track'
    output_data.cell_corearea.attrs['units'] = 'km^2'

    output_data.cell_cloudnumber.attrs['long_name'] = 'Number of each cloud in a track that cooresponds to the cloudid map'
    output_data.cell_cloudnumber.attrs['units'] = 'unitless'
    output_data.cell_cloudnumber.attrs['usuage'] = 'To link this tracking statistics file with pixel-level cloudid files, use the cloudidfile and cloudnumber together to identify which cloud this current track and time is associated with'

    output_data.cell_mergecloudnumber.attrs['long_name'] = 'cloud number of small, short-lived feature merging into a cell track'
    output_data.cell_mergecloudnumber.attrs['units'] = 'unitless'

    output_data.cell_splitcloudnumber.attrs['long_name'] = 'cloud number of small, short-lived feature splitting into a cell track'
    output_data.cell_splitcloudnumber.attrs['units'] = 'unitless'

    # Write netcdf file
    print(celltrackstatistics_outfile)
    print('')

    output_data.to_netcdf(path=celltrackstatistics_outfile, mode='w', format='NETCDF4_CLASSIC', unlimited_dims='ntracks', \
                          encoding={'cell_basetime': {'dtype': 'int64', 'zlib':True, '_FillValue': fillvalue, 'units': 'seconds since 1970-01-01'}, \
                                    'cell_datetimestring': {'zlib':True, '_FillValue': fillvalue}, \
                                    'cell_length': {'zlib':True, '_FillValue': fillvalue}, \
                                    'cell_status': {'dtype':'int', 'zlib':True, '_FillValue': fillvalue}, \
                                    'cell_startstatus': {'dtype':'int', 'zlib':True, '_FillValue': fillvalue}, \
                                    'cell_endstatus': {'dtype':'int', 'zlib':True, '_FillValue': fillvalue}, \
                                    'cell_boundary': {'dtype':'int', 'zlib':True, '_FillValue': fillvalue}, \
                                    'cell_trackinterruptions': {'dtype':'int', 'zlib':True, '_FillValue': fillvalue}, \
                                    'cell_meanlat': {'zlib':True, '_FillValue': fillvalue}, \
                                    'cell_meanlon': {'zlib':True, '_FillValue': fillvalue}, \
                                    'cell_corearea': {'zlib':True, '_FillValue': fillvalue}, \
                                    'cell_ccsarea': {'zlib':True, '_FillValue': fillvalue}, \
                                    'cell_cloudnumber': {'dtype':'int', 'zlib':True, '_FillValue': fillvalue}, \
                                    'cell_mergecloudnumber': {'dtype':'int', 'zlib':True, '_FillValue': fillvalue}, \
                                    'cell_splitcloudnumber': {'dtype':'int', 'zlib':True, '_FillValue': fillvalue}})

