# Purpose: Subset statistics file to keep only MCS. Uses brightness temperature statstics of cold cloud shield area, duration, and eccentricity base on Fritsch et al (1986) and Maddos (1980)

# Author: Original IDL code written by Zhe Feng (zhe.feng@pnnl.gov), Python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov)

def identifycell_LES_xarray(statistics_filebase, stats_path, startdate, enddate, time_resolution, geolimits, maincloud_duration, merge_duration, split_duration, nmaxmerge):
    # Inputs:
    # statistics_filebase - header of the track statistics file that was generated in the previous code
    # stats_path - location of the track data. also the location where the data from this code will be saved
    # startdate - starting date and time of the data
    # enddate - ending date and time of the data
    # geolimits - 4-element array with plotting boundaries [lat_min, lon_min, lat_max, lon_max]
    # time_resolution - time resolution of the raw data
    # maincloud_duration - minimum time a cloud must be tracked to be considered a tracked cell
    # split_duration - splitting tracks with durations less than this will be label as a splitting track. tracks longer that this will be labeled as their own MCS
    # merge_duration - merging tracks with durations less than this will be label as a merging track. tracks longer that this will be labeled as their own MCS
    # nmaxmerge - maximum number of clouds that can split or merge into one cloud

    # Output: (One netcdf with statisitics about each liquid water path defined cell track in each row)
    # cell_basetime - seconds since 1970-01-01 for each cloud in a cell
    # cell_datetimestring - string of date and time of each cloud in a cell
    # cell_length - duration of each cell
    # cell_status - flag indicating the evolution of each cloud in a cell
    # cell_startstatus - flag indicating how a cell starts
    # cell_endstatus - flag indicating how a cell ends
    # cell_boundary - flag indicating if a cell touches the edge of the domain
    # cell_trackinterruptions - flag indicating if the data used to identify this cell is incomplete
    # cell_meanlat - mean latitude of the cell
    # cell_meanlon - mean longitude of the cell
    # cell_corearea - area of the core of cell
    # cell_ccsarea - area of the core and cold anvil of the cell
    # cell_cloudnumber - numbers indicating which clouds in the cloudid files are associated with a cell
    # cell_mergenumber - numbers indicating which clouds in the cloudid files merge into this track
    # cell_splitnumber - numbers indicating which clouds in the cloudid files split from this track

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
    nconv = allstatdata['nconv'].data
    ncoldanvil = allstatdata['ncoldanvil'].data

    ####################################################################
    # Set up duration and area variables

    # Cold Cloud Shield (CCS) area
    nconv[np.where(~np.isfinite(nconv))] = 0
    ncoldanvil[np.where(~np.isfinite(ncoldanvil))] = 0
    trackstat_corearea = nconv * np.copy(allstatdata.attrs['pixel_radius_km'])**2
    trackstat_ccsarea = (nconv + ncoldanvil) * np.copy(allstatdata.attrs['pixel_radius_km'])**2

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
    # Find small merging and spliting louds and add to cell
    cellmergecloudnumber = np.ones((nmaincell, int(nmaxlength), nmaxmerge), dtype=int)*-9999
    cellmergestatus = np.ones((nmaincell, int(nmaxlength), nmaxmerge), dtype=int)*-9999
    cellsplitcloudnumber = np.ones((nmaincell, int(nmaxlength), nmaxmerge), dtype=int)*-9999
    cellsplitstatus = np.ones((nmaincell, int(nmaxlength), nmaxmerge), dtype=int)*-9999

    # Loop through each cell and link small clouds merging in
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
                    mergefile[removemerges] = -9999
                    mergefeature[removemerges] = -9999
                mergefile = mergefile[mergefile != -9999]
                mergefeature = mergefeature[mergefeature != -9999]

            # Continue if mergers satisfy duration and cell restriction
            if len(mergefile) > 0:

                # Get data about merging tracks
                mergingcloudnumber = np.copy(allstatdata['cloudnumber'][mergefile, :])
                mergingbasetime = np.copy(allstatdata['basetime'][mergefile, :])
                mergingstatus = np.copy(allstatdata['status'][mergefile, :])
                mergingdatetime = np.copy(allstatdata['datetimestrings'][mergefile, :, :])

                # Get data about cell track
                maincellbasetime = np.copy(allstatdata['basetime'][int(maincelltracknumbers[icell])-1, 0:int(maincelllength[icell])])

                # Loop through each timestep in the cell track
                for t in np.arange(0, maincelllength[icell]):

                    # Find merging cloud times that match current cell track time
                    timematch = np.where(mergingbasetime == maincellbasetime[int(t)])

                    if np.shape(timematch)[1] > 0:

                        # save cloud number of small mergers
                        nmergers = np.shape(timematch)[1]
                        cellmergecloudnumber[icell, int(t), 0:nmergers] = mergingcloudnumber[timematch]
                        cellmergestatus[icell, int(t), 0:nmergers] = mergingstatus[timematch]

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

            # Make sure the spliter itself is not an cell
            splittingcell = np.intersect1d(splitfile, maincelltracknumbers)
            if len(splittingcell) > 0:
                for iremove in np.arange(0, len(splittingcell)):
                    removesplits = np.array(np.where(splitfile == splittingcell[iremove]))[0,:]
                    splitfile[removesplits] = -9999
                    splitfeature[removesplits] = -9999
                splitfile = splitfile[splitfile != -9999]
                splitfeature = splitfeature[splitfeature != -9999]
                
            # Continue if spliters satisfy duration and cell restriction
            if len(splitfile) > 0:

                # Get data about splitting tracks
                splittingcloudnumber = np.copy(allstatdata['cloudnumber'][splitfile, :])
                splittingbasetime = np.copy(allstatdata['basetime'][splitfile, :])
                splittingstatus = np.copy(allstatdata['status'][splitfile, :])
                splittingdatetime = np.copy(allstatdata['datetimestrings'][splitfile, :, :])

                # Get data about cell track
                maincellbasetime = np.copy(allstatdata['basetime'][int(maincelltracknumbers[icell])-1,0:int(maincelllength[icell])])

                # Loop through each timestep in the cell track
                for t in np.arange(0, maincelllength[icell]):

                    # Find splitting cloud times that match current cell track time
                    timematch = np.where(splittingbasetime == maincellbasetime[int(t)])
                    if np.shape(timematch)[1] > 0:

                        # save cloud number of small splitrs
                        nspliters = np.shape(timematch)[1]
                        cellsplitcloudnumber[icell, int(t), 0:nspliters] = splittingcloudnumber[timematch]
                        cellsplitstatus[icell, int(t), 0:nspliters] = splittingstatus[timematch]

    ########################################################################
    # Subset keeping just cell tracks
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
                          encoding={'cell_basetime': {'zlib':True, 'units': 'seconds since 1970-01-01'}, \
                                    'cell_datetimestring': {'zlib':True}, \
                                    'cell_length': {'zlib':True, '_FillValue': -9999}, \
                                    'cell_status': {'dtype':'int', 'zlib':True, '_FillValue': -9999}, \
                                    'cell_startstatus': {'dtype':'int', 'zlib':True, '_FillValue': -9999}, \
                                    'cell_endstatus': {'dtype':'int', 'zlib':True, '_FillValue': -9999}, \
                                    'cell_boundary': {'dtype':'int', 'zlib':True, '_FillValue': -9999}, \
                                    'cell_trackinterruptions': {'dtype':'int', 'zlib':True, '_FillValue': -9999}, \
                                    'cell_meanlat': {'zlib':True, '_FillValue': np.nan}, \
                                    'cell_meanlon': {'zlib':True, '_FillValue': np.nan}, \
                                    'cell_corearea': {'zlib':True, '_FillValue': np.nan}, \
                                    'cell_ccsarea': {'zlib':True, '_FillValue': np.nan}, \
                                    'cell_cloudnumber': {'dtype':'int', 'zlib':True, '_FillValue': -9999}, \
                                    'cell_mergecloudnumber': {'dtype':'int', 'zlib':True, '_FillValue': -9999}, \
                                    'cell_splitcloudnumber': {'dtype':'int', 'zlib':True, '_FillValue': -9999}})

def identifycell_LES_netcdf4(statistics_filebase, stats_path, startdate, enddate, time_resolution, geolimits, maincloud_duration, merge_duration, split_duration, nmaxmerge):
    # Inputs:
    # statistics_filebase - header of the track statistics file that was generated in the previous code
    # stats_path - location of the track data. also the location where the data from this code will be saved
    # startdate - starting date and time of the data
    # enddate - ending date and time of the data
    # geolimits - 4-element array with plotting boundaries [lat_min, lon_min, lat_max, lon_max]
    # time_resolution - time resolution of the raw data
    # maincloud_duration - minimum time a cloud must be tracked to be considered a tracked cell
    # split_duration - splitting tracks with durations less than this will be label as a splitting track. tracks longer that this will be labeled as their own MCS
    # merge_duration - merging tracks with durations less than this will be label as a merging track. tracks longer that this will be labeled as their own MCS
    # nmaxmerge - maximum number of clouds that can split or merge into one cloud

    # Output: (One netcdf with statisitics about each liquid water path defined cell track in each row)
    # cell_basetime - seconds since 1970-01-01 for each cloud in a cell
    # cell_datetimestring - string of date and time of each cloud in a cell
    # cell_length - duration of each cell
    # cell_status - flag indicating the evolution of each cloud in a cell
    # cell_startstatus - flag indicating how a cell starts
    # cell_endstatus - flag indicating how a cell ends
    # cell_boundary - flag indicating if a cell touches the edge of the domain
    # cell_trackinterruptions - flag indicating if the data used to identify this cell is incomplete
    # cell_meanlat - mean latitude of the cell
    # cell_meanlon - mean longitude of the cell
    # cell_corearea - area of the core of cell
    # cell_ccsarea - area of the core and cold anvil of the cell
    # cell_cloudnumber - numbers indicating which clouds in the cloudid files are associated with a cell
    # cell_mergenumber - numbers indicating which clouds in the cloudid files merge into this track
    # cell_splitnumber - numbers indicating which clouds in the cloudid files split from this track

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

    allstatdata = Dataset(statistics_file, 'r')
    ntracks_all = np.nanmax(allstatdata['ntracks'][:]) + 1
    nmaxlength = np.nanmax(allstatdata['nmaxlength'][:]) + 1
    nconv = allstatdata['nconv'][:]
    ncoldanvil = allstatdata['ncoldanvil'][:]
    lifetime = allstatdata['lifetime'][:]
    eccentricity = allstatdata['eccentricity'][:]
    basetime = allstatdata['basetime'][:]
    mergenumbers = allstatdata['mergenumbers'][:]
    cloudnumber = allstatdata['cloudnumber'][:]
    status = allstatdata['status'][:]
    datetimestrings = allstatdata['datetimestrings'][:]
    splitnumbers = allstatdata['splitnumbers'][:]
    startstatus = allstatdata['startstatus'][:]
    endstatus = allstatdata['endstatus'][:]
    boundary = allstatdata['boundary'][:]
    trackinterruptions = allstatdata['trackinterruptions'][:]
    meanlat = allstatdata['meanlat'][:]
    meanlon = allstatdata['meanlon'][:]
    datasource = str(allstatdata.source)
    datadescription = str(allstatdata.description)
    pixelradius = float(allstatdata.pixel_radius_km)
    allstatdata.close()

    ####################################################################
    # Set up duration and area variables

    # Cold Cloud Shield (CCS) area
    nconv[np.where(~np.isfinite(nconv))] = 0
    ncoldanvil[np.where(~np.isfinite(ncoldanvil))] = 0
    trackstat_corearea = np.multiply(nconv, pixelradius**2)
    trackstat_ccsarea = np.multiply((nconv + ncoldanvil), np.copy(pixelradius)**2)

    # Convert path duration to time
    trackstat_duration = np.multiply(lifetime, time_resolution)

    ##################################################################
    # Initialize matrices
    trackid_cell = []

    ###################################################################
    # Identify main cells
    maincelltrackid = np.array(np.where(trackstat_duration >= maincloud_duration))[0, :]
    nmaincell = len(maincelltrackid)
    maincelltracknumbers = np.add(maincelltrackid, 1)

    maincellcloudnumbers = np.copy(cloudnumber[maincelltrackid, :])
    maincelllength = np.copy(lifetime[maincelltrackid])

    ###############################################################
    # Find small merging and spliting louds and add to cell
    cellmergecloudnumber = np.ones((nmaincell, int(nmaxlength), nmaxmerge), dtype=int)*-9999
    cellmergestatus = np.ones((nmaincell, int(nmaxlength), nmaxmerge), dtype=int)*-9999
    cellsplitcloudnumber = np.ones((nmaincell, int(nmaxlength), nmaxmerge), dtype=int)*-9999
    cellsplitstatus = np.ones((nmaincell, int(nmaxlength), nmaxmerge), dtype=int)*-9999

    # Loop through each cell and link small clouds merging in
    print('Number of main cells: ' + str(int(nmaincell)))
    for icell in np.arange(0, nmaincell):
        print('')
        print(icell)

        ###################################################################################
        # Isolate basetime data

        if icell == 0:
            cellbasetime = np.array([pd.to_datetime(basetime[maincelltrackid[icell], :].data, unit='s')])
        else:
            cellbasetime = np.concatenate((cellbasetime, np.array([pd.to_datetime(basetime[maincelltrackid[icell], :].data, unit='s')])), axis=0)

        print('Determining Base Time')

        ###################################################################################
        # Find mergers
        print('Determining mergers')
        [mergefile, mergefeature] = np.array(np.where(np.copy(mergenumbers) == maincelltracknumbers[icell]))

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
                    mergefile[removemerges] = -9999
                    mergefeature[removemerges] = -9999
                mergefile = mergefile[mergefile != -9999].astype(int)
                mergefeature = mergefeature[mergefeature != -9999].astype(int)

            # Continue if mergers satisfy duration and cell restriction
            if len(mergefile) > 0:

                # Get data about merging tracks
                mergingcloudnumber = np.copy(cloudnumber[mergefile, :])
                mergingbasetime = np.copy(basetime[mergefile, :])
                mergingstatus = np.copy(status[mergefile, :])
                mergingdatetime = np.copy(datetimestrings[mergefile, :, :])

                # Get data about cell track
                maincellbasetime = np.copy(basetime[int(maincelltracknumbers[icell])-1, 0:int(maincelllength[icell])])

                # Loop through each timestep in the cell track
                for t in np.arange(0, maincelllength[icell]):

                    # Find merging cloud times that match current cell track time
                    timematch = np.where(mergingbasetime == maincellbasetime[int(t)])

                    if np.shape(timematch)[1] > 0:

                        # save cloud number of small mergers
                        nmergers = np.shape(timematch)[1]
                        cellmergecloudnumber[icell, int(t), 0:nmergers] = mergingcloudnumber[timematch]
                        cellmergestatus[icell, int(t), 0:nmergers] = mergingstatus[timematch]

        ############################################################
        # Find splits
        print('Determining splits')
        [splitfile, splitfeature] = np.array(np.where(np.copy(splitnumbers == maincelltracknumbers[icell])))

        # Loop through all splitting tracks, if present
        print(len(splitfile))
        if len(splitfile) > 0:
            # Isolate splitting cases that have short duration
            splitfeature = splitfeature[trackstat_duration[splitfile] < split_duration]
            splitfile = splitfile[trackstat_duration[splitfile] < split_duration]

            # Make sure the spliter itself is not an cell
            splittingcell = np.intersect1d(splitfile, maincelltracknumbers)
            if len(splittingcell) > 0:
                for iremove in np.arange(0, len(splittingcell)):
                    removesplits = np.array(np.where(splitfile == splittingcell[iremove]))[0,:]
                    splitfile[removesplits] = -9999
                    splitfeature[removesplits] = -9999
                splitfile = splitfile[splitfile != -9999]
                splitfeature = splitfeature[splitfeature != -9999]
                
            # Continue if spliters satisfy duration and cell restriction
            if len(splitfile) > 0:

                # Get data about splitting tracks
                splittingcloudnumber = np.copy(cloudnumber[splitfile, :])
                splittingbasetime = np.copy(basetime[splitfile, :])
                splittingstatus = np.copy(status[splitfile, :])
                splittingdatetime = np.copy(datetimestrings[splitfile, :, :])

                # Get data about cell track
                maincellbasetime = np.copy(basetime[int(maincelltracknumbers[icell])-1,0:int(maincelllength[icell])])

                # Loop through each timestep in the cell track
                for t in np.arange(0, maincelllength[icell]):

                    # Find splitting cloud times that match current cell track time
                    timematch = np.where(splittingbasetime == maincellbasetime[int(t)])
                    if np.shape(timematch)[1] > 0:

                        # save cloud number of small splitrs
                        nspliters = np.shape(timematch)[1]
                        cellsplitcloudnumber[icell, int(t), 0:nspliters] = splittingcloudnumber[timematch]
                        cellsplitstatus[icell, int(t), 0:nspliters] = splittingstatus[timematch]

    ########################################################################
    # Subset keeping just cell tracks
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
                              'cell_datetimestring': (['ntracks', 'ntimes', 'ndatetimechars'], datetimestrings[maincelltrackid,:]), \
                              'cell_length': (['ntracks'], trackstat_duration[maincelltrackid]), \
                              'cell_status': (['ntracks', 'ntimes'], status[maincelltrackid, :]), \
                              'cell_startstatus': (['ntracks'], startstatus[maincelltrackid]), \
                              'cell_endstatus': (['ntracks'], endstatus[maincelltrackid]), \
                              'cell_boundary': (['ntracks'], boundary[maincelltrackid]), \
                              'cell_trackinterruptions': (['ntracks'], trackinterruptions[maincelltrackid]), \
                              'cell_meanlat': (['ntracks', 'ntimes'], meanlat[maincelltrackid, :]), \
                              'cell_meanlon': (['ntracks', 'ntimes'], meanlon[maincelltrackid, :]), \
                              'cell_corearea': (['ntracks', 'ntimes'], trackstat_corearea[maincelltrackid]), \
                              'cell_ccsarea': (['ntracks', 'ntimes'], trackstat_ccsarea[maincelltrackid]), \
                              'cell_cloudnumber': (['ntracks', 'ntimes'], cloudnumber[maincelltrackid, :]), \
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
                                    'source': datasource, \
                                    'description': datadescription, \
                                    'startdate': startdate, \
                                    'enddate': enddate, \
                                    'time_resolution_hour': time_resolution, \
                                    'pixel_radius_km': pixelradius, \
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
                          encoding={'cell_basetime': {'dtype': 'int64', 'zlib':True, 'units': 'seconds since 1970-01-01'}, \
                                    'cell_datetimestring': {'zlib':True}, \
                                    'cell_length': {'zlib':True, '_FillValue': -9999}, \
                                    'cell_status': {'dtype':'int', 'zlib':True, '_FillValue': -9999}, \
                                    'cell_startstatus': {'dtype':'int', 'zlib':True, '_FillValue': -9999}, \
                                    'cell_endstatus': {'dtype':'int', 'zlib':True, '_FillValue': -9999}, \
                                    'cell_boundary': {'dtype':'int', 'zlib':True, '_FillValue': -9999}, \
                                    'cell_trackinterruptions': {'dtype':'int', 'zlib':True, '_FillValue': -9999}, \
                                    'cell_meanlat': {'zlib':True, '_FillValue': np.nan}, \
                                    'cell_meanlon': {'zlib':True, '_FillValue': np.nan}, \
                                    'cell_corearea': {'zlib':True, '_FillValue': np.nan}, \
                                    'cell_ccsarea': {'zlib':True, '_FillValue': np.nan}, \
                                    'cell_cloudnumber': {'dtype':'int', 'zlib':True, '_FillValue': -9999}, \
                                    'cell_mergecloudnumber': {'dtype':'int', 'zlib':True, '_FillValue': -9999}, \
                                    'cell_splitcloudnumber': {'dtype':'int', 'zlib':True, '_FillValue': -9999}})

