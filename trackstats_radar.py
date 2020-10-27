# Purpose: This gets statistics about each track from the radar data. 

# Define function that calculates track statistics for satellite data
def trackstats_radar(datasource, datadescription, pixel_radius, datatimeresolution, geolimits, areathresh, \
                    startdate, enddate, timegap, cloudid_filebase, tracking_inpath, stats_path, \
                    track_version, tracknumbers_version, tracknumbers_filebase, terrain_file, lengthrange=[2,120]):

    import numpy as np
    from netCDF4 import Dataset, num2date, chartostring
    import os, fnmatch
    import sys
    from math import pi
    from skimage.measure import regionprops
    import time
    import gc
    import datetime
    import xarray as xr
    import pandas as pd
    import netcdf_io_trackstats as net
    np.set_printoptions(threshold=np.inf)

    # Set output filename
    trackstats_outfile = stats_path + 'stats_' + tracknumbers_filebase + '_' + startdate + '_' + enddate + '.nc'

    # Read terrain file to get range mask
    dster = Dataset(terrain_file, 'r')
    rangemask = dster['mask110'][:]
    dster.close()

    # Load track data
    print('Loading gettracks data')
    print((time.ctime()))
    cloudtrack_file = stats_path + tracknumbers_filebase + '_' + startdate + '_' + enddate + '.nc'
    
    cloudtrackdata = Dataset(cloudtrack_file, 'r')
    numtracks = cloudtrackdata['ntracks'][:]
    cloudidfiles = cloudtrackdata['cloudid_files'][:]
    nfiles = cloudtrackdata.dimensions['nfiles'].size
    tracknumbers = cloudtrackdata['track_numbers'][:]
    trackreset = cloudtrackdata['track_reset'][:]
    tracksplit = cloudtrackdata['track_splitnumbers'][:]
    trackmerge = cloudtrackdata['track_mergenumbers'][:]
    trackstatus = cloudtrackdata['track_status'][:]
    cloudtrackdata.close()

    tmpfname = ''.join(chartostring(cloudidfiles[0]))
    numcharfilename = len(list(tmpfname))

    # Load latitude and longitude grid from any cloudidfile since they all have the map of latitude and longitude saved
    latlondata = Dataset(tracking_inpath + tmpfname, 'r')
    longitude = latlondata.variables['longitude'][:]
    latitude = latlondata.variables['latitude'][:]
    x_coord = latlondata['x'][:]/1000.  # convert unit to [km]
    y_coord = latlondata['y'][:]/1000.  # convert unit to [km]
    latlondata.close()

    # Determine dimensions of data
    # nfiles = len(cloudidfiles_list)
    ny, nx = np.shape(latitude)

    # xcoord2d, ycoord2d = np.meshgrid(x_coord, y_coord)

    ############################################################################
    # Initialize grids
    nmaxclouds = int(max(lengthrange))
    maxtracklength = int(max(lengthrange))
    numtracks = int(numtracks)
    fillval = -9999

    ###############################################################
    # to calculate the statistic after having the number of tracks with cells
    print('Initiailizinng matrices')
    print((time.ctime()))

    finaltrack_tracklength = np.zeros(numtracks, dtype=np.int32)
    finaltrack_cloudnumber = np.full((numtracks, maxtracklength), fillval, dtype=np.int32)
    finaltrack_basetime = np.full((numtracks, maxtracklength), fillval, dtype=np.float)
    finaltrack_core_meanlat = np.full((numtracks, maxtracklength), np.nan, dtype=np.float)
    finaltrack_core_meanlon = np.full((numtracks, maxtracklength), np.nan, dtype=np.float)
    finaltrack_core_mean_x = np.full((numtracks, maxtracklength), np.nan, dtype=np.float)
    finaltrack_core_mean_y = np.full((numtracks, maxtracklength), np.nan, dtype=np.float)

    finaltrack_cell_meanlat = np.full((numtracks, maxtracklength), np.nan, dtype=np.float)
    finaltrack_cell_meanlon = np.full((numtracks, maxtracklength), np.nan, dtype=np.float)
    finaltrack_cell_mean_x = np.full((numtracks, maxtracklength), np.nan, dtype=np.float)
    finaltrack_cell_mean_y = np.full((numtracks, maxtracklength), np.nan, dtype=np.float)

    finaltrack_cell_minlat = np.full((numtracks, maxtracklength), np.nan, dtype=np.float)
    finaltrack_cell_maxlat = np.full((numtracks, maxtracklength), np.nan, dtype=np.float)
    finaltrack_cell_minlon = np.full((numtracks, maxtracklength), np.nan, dtype=np.float)
    finaltrack_cell_maxlon = np.full((numtracks, maxtracklength), np.nan, dtype=np.float)
    finaltrack_cell_min_y = np.full((numtracks, maxtracklength), np.nan, dtype=np.float)
    finaltrack_cell_max_y = np.full((numtracks, maxtracklength), np.nan, dtype=np.float)
    finaltrack_cell_min_x = np.full((numtracks, maxtracklength), np.nan, dtype=np.float)
    finaltrack_cell_max_x = np.full((numtracks, maxtracklength), np.nan, dtype=np.float)
    
    finaltrack_dilatecell_meanlat = np.full((numtracks, maxtracklength), np.nan, dtype=np.float)
    finaltrack_dilatecell_meanlon = np.full((numtracks, maxtracklength), np.nan, dtype=np.float)
    finaltrack_dilatecell_mean_x = np.full((numtracks, maxtracklength), np.nan, dtype=np.float)
    finaltrack_dilatecell_mean_y = np.full((numtracks, maxtracklength), np.nan, dtype=np.float)

    finaltrack_cell_maxdbz = np.full((numtracks, maxtracklength), np.nan, dtype=float)

    finaltrack_cell_maxETH10dbz = np.full((numtracks, maxtracklength), np.nan, dtype=np.float)
    finaltrack_cell_maxETH20dbz = np.full((numtracks, maxtracklength), np.nan, dtype=np.float)
    finaltrack_cell_maxETH30dbz = np.full((numtracks, maxtracklength), np.nan, dtype=np.float)
    finaltrack_cell_maxETH40dbz = np.full((numtracks, maxtracklength), np.nan, dtype=np.float)
    finaltrack_cell_maxETH50dbz = np.full((numtracks, maxtracklength), np.nan, dtype=np.float)

    finaltrack_core_area = np.full((numtracks, maxtracklength), np.nan, dtype=float)
    finaltrack_cell_area = np.full((numtracks, maxtracklength), np.nan, dtype=float)
    
    finaltrack_status = np.full((numtracks, maxtracklength), fillval, dtype=np.int32)
    finaltrack_trackinterruptions = np.full(numtracks, fillval, dtype=np.int32)
    finaltrack_mergenumber = np.full((numtracks, maxtracklength), fillval, dtype=np.int32)
    finaltrack_splitnumber = np.full((numtracks, maxtracklength), fillval, dtype=np.int32)
    finaltrack_cloudidfile = np.chararray((numtracks, maxtracklength, int(numcharfilename)))

    finaltrack_cell_rangeflag = np.full((numtracks, maxtracklength), fillval, dtype=np.int)

    #########################################################################################
    # loop over files. Calculate statistics and organize matrices by tracknumber and cloud
    print('Looping over files and calculating statistics for each file')
    print((time.ctime()))
    for nf in range(0, nfiles):

        # Get all track numbers in the file
        file_tracknumbers = tracknumbers[0, nf, :]
        # Get idcloud filename
        fname = ''.join(chartostring(cloudidfiles[nf]))

        # Only process file if that file contains a track
        if np.nanmax(file_tracknumbers) > 0:

            # fname = ''.join(chartostring(cloudidfiles[nf]))
            print(nf, fname)

            # Load cloudid file
            cloudid_file = tracking_inpath + fname
            # print(cloudid_file)

            file_cloudiddata = Dataset(cloudid_file, 'r')
            file_dbz = file_cloudiddata['comp_ref'][:]
            file_all_cloudnumber = file_cloudiddata['cloudnumber'][:]
            file_corecold_cloudnumber = file_cloudiddata['convcold_cloudnumber'][:]
            conv_core = file_cloudiddata['conv_core'][:]
            conv_mask = file_cloudiddata['conv_mask'][:]
            echotop10 = file_cloudiddata['echotop10'][:] / 1000.    # convert unit to [km]
            echotop20 = file_cloudiddata['echotop20'][:] / 1000.    # convert unit to [km]
            echotop30 = file_cloudiddata['echotop30'][:] / 1000.    # convert unit to [km]
            echotop40 = file_cloudiddata['echotop40'][:] / 1000.    # convert unit to [km]
            echotop50 = file_cloudiddata['echotop50'][:] / 1000.    # convert unit to [km]
            file_basetime = file_cloudiddata['basetime'][:]
            basetime_units = file_cloudiddata['basetime'].units
            # basetime_calendar = file_cloudiddata['basetime'].calendar
            file_cloudiddata.close()

            file_datetimestring = cloudid_file[len(tracking_inpath) + len(cloudid_filebase):-3]

            # Find unique track numbers
            uniquetracknumbers = np.unique(file_tracknumbers)
            uniquetracknumbers = uniquetracknumbers[np.isfinite(uniquetracknumbers)]
            uniquetracknumbers = uniquetracknumbers[uniquetracknumbers > 0].astype(int)
            
            # Loop over unique tracknumbers
            # print('Loop over tracks in file')
            for itrack in uniquetracknumbers:

                # Find cloud number that belongs to the current track in this file
                # Need to add one since tells index, which starts at 0, and we want the number, which starts at one
                cloudnumber = np.array(np.where(file_tracknumbers == itrack))[0, :] + 1
                cloudindex = cloudnumber - 1    # Index within the matrice of this cloud.

                # Should only be one cloud number. 
                # In mergers and split, the associated clouds should be listed in 
                # the file_splittracknumbers and file_mergetracknumbers
                if len(cloudnumber) == 1: 
                    # Find core in cloudid file associated with this track, and is a convective core (conv_core == 1)
                    corearea = np.array(np.where((file_corecold_cloudnumber[0,:,:] == cloudnumber) & (conv_core[0,:,:] == 1)))
                    ncorepix = np.shape(corearea)[1]

                    # Convective cell (conv_mask >= 1). conv_mask is sorted and numbered.
                    cellarea = np.array(np.where((file_corecold_cloudnumber[0,:,:] == cloudnumber) & (conv_mask[0,:,:] >= 1)))
                    ncellpix = np.shape(cellarea)[1]

                    # Dilated convective cell
                    dilatecellarea = np.array(np.where(file_corecold_cloudnumber[0,:,:] == cloudnumber))
                    ndilatecellpix = np.shape(dilatecellarea)[1]

                    # Record previous length of the track (initially all track lengths start at 0)
                    nc = finaltrack_tracklength[itrack-1]
                    # Add 1 to the current track length
                    finaltrack_tracklength[itrack-1] = nc+1

                    if nc < maxtracklength:
                        # Save information that links this cloud back to its raw pixel level data
                        finaltrack_cloudnumber[itrack-1, nc] = cloudnumber
                        finaltrack_basetime[itrack-1, nc] = np.array(file_basetime[0])
                        finaltrack_cloudidfile[itrack-1][nc][:] = fname

                        ###############################################################
                        # Calculate statistics for this cell
                        if (ncellpix > 0):
                            # Location of core
                            corelat = latitude[corearea[0], corearea[1]]
                            corelon = longitude[corearea[0], corearea[1]]
                            core_y = y_coord[corearea[0]]
                            core_x = x_coord[corearea[1]]

                            # Location of convective cell
                            celllat = latitude[cellarea[0], cellarea[1]]
                            celllon = longitude[cellarea[0], cellarea[1]]
                            cell_y = y_coord[cellarea[0]]
                            cell_x = x_coord[cellarea[1]]

                            # Location of dilated convective cell
                            dilatecelllat = latitude[dilatecellarea[0], dilatecellarea[1]]
                            dilatecelllon = longitude[dilatecellarea[0], dilatecellarea[1]]
                            dilatecell_y = y_coord[dilatecellarea[0]]
                            dilatecell_x = x_coord[dilatecellarea[1]]

                            # Core center location
                            finaltrack_core_meanlat[itrack-1, nc] = np.nanmean(corelat)
                            finaltrack_core_meanlon[itrack-1, nc] = np.nanmean(corelon)
                            finaltrack_core_mean_y[itrack-1, nc] = np.nanmean(core_y)
                            finaltrack_core_mean_x[itrack-1, nc] = np.nanmean(core_x)

                            # Cell center location
                            finaltrack_cell_meanlat[itrack-1, nc] = np.nanmean(celllat)
                            finaltrack_cell_meanlon[itrack-1, nc] = np.nanmean(celllon)
                            finaltrack_cell_mean_y[itrack-1, nc] = np.nanmean(cell_y)
                            finaltrack_cell_mean_x[itrack-1, nc] = np.nanmean(cell_x)

                            # Dilated cell center location
                            finaltrack_dilatecell_meanlat[itrack-1, nc] = np.nanmean(dilatecelllat)
                            finaltrack_dilatecell_meanlon[itrack-1, nc] = np.nanmean(dilatecelllon)
                            finaltrack_dilatecell_mean_y[itrack-1, nc] = np.nanmean(dilatecell_y)
                            finaltrack_dilatecell_mean_x[itrack-1, nc] = np.nanmean(dilatecell_x)
                            
                            # Cell min/max location (for its maximum spatial extent)
                            finaltrack_cell_minlat[itrack-1, nc] = np.nanmin(celllat)
                            finaltrack_cell_maxlat[itrack-1, nc] = np.nanmax(celllat)
                            finaltrack_cell_minlon[itrack-1, nc] = np.nanmin(celllon)
                            finaltrack_cell_maxlon[itrack-1, nc] = np.nanmax(celllon)
                            finaltrack_cell_min_y[itrack-1, nc] = np.nanmin(cell_y)
                            finaltrack_cell_max_y[itrack-1, nc] = np.nanmax(cell_y)
                            finaltrack_cell_min_x[itrack-1, nc] = np.nanmin(cell_x)
                            finaltrack_cell_max_x[itrack-1, nc] = np.nanmax(cell_x)

                            # Area of the cell
                            finaltrack_core_area[itrack-1, nc] = ncorepix * pixel_radius**2
                            finaltrack_cell_area[itrack-1, nc] = ncellpix * pixel_radius**2

                            # Reflectivity maximum
                            finaltrack_cell_maxdbz[itrack-1, nc] = np.nanmean(file_dbz[0,cellarea[0],cellarea[1]])

                            # Echo-top heights
                            finaltrack_cell_maxETH10dbz[itrack-1, nc] = np.nanmax(echotop10[0,cellarea[0],cellarea[1]])
                            finaltrack_cell_maxETH20dbz[itrack-1, nc] = np.nanmax(echotop20[0,cellarea[0],cellarea[1]])
                            finaltrack_cell_maxETH30dbz[itrack-1, nc] = np.nanmax(echotop30[0,cellarea[0],cellarea[1]])
                            finaltrack_cell_maxETH40dbz[itrack-1, nc] = np.nanmax(echotop40[0,cellarea[0],cellarea[1]])
                            finaltrack_cell_maxETH50dbz[itrack-1, nc] = np.nanmax(echotop50[0,cellarea[0],cellarea[1]])

                            # Save track information. Need to subtract one since cloudnumber gives the number of the cloud (which starts at one), but we are looking for its index (which starts at zero)
                            finaltrack_status[itrack-1, nc] = np.copy(trackstatus[0, nf, cloudindex])
                            finaltrack_mergenumber[itrack-1, nc] = np.copy(trackmerge[0, nf, cloudindex])
                            finaltrack_splitnumber[itrack-1, nc] = np.copy(tracksplit[0, nf, cloudindex])
                            finaltrack_trackinterruptions[itrack-1] = np.copy(trackreset[0, nf, cloudindex])

                            # The min range mask value within the dilatecellarea (1: cell completely within range mask, 0: some portion of the cell outside range mask)
                            finaltrack_cell_rangeflag[itrack-1, nc] = np.min(rangemask[dilatecellarea[0],dilatecellarea[1]])
                    else:
                        print(itrack, ' greater than max track length')
                else:
                    print(cloudnumber, ' cloudnumber not found in idcloud file')
        else:
            print(fname, ' has no tracks in idcloud file')
            
    ###############################################################
    ## Remove tracks that have no cells. These tracks are short.
    print('Removing tracks with no cells')
    print((time.ctime()))
    gc.collect()

    cloudindexpresent = np.array(np.where(finaltrack_tracklength > 0))[0, :]
    numtracks = len(cloudindexpresent)

    finaltrack_tracklength = finaltrack_tracklength[cloudindexpresent]
    finaltrack_basetime = finaltrack_basetime[cloudindexpresent, :]
    finaltrack_core_meanlat = finaltrack_core_meanlat[cloudindexpresent, :]
    finaltrack_core_meanlon = finaltrack_core_meanlon[cloudindexpresent, :]
    finaltrack_core_mean_y = finaltrack_core_mean_y[cloudindexpresent, :]
    finaltrack_core_mean_x = finaltrack_core_mean_x[cloudindexpresent, :]

    finaltrack_cell_meanlat = finaltrack_cell_meanlat[cloudindexpresent, :]
    finaltrack_cell_meanlon = finaltrack_cell_meanlon[cloudindexpresent, :]
    finaltrack_cell_mean_y = finaltrack_cell_mean_y[cloudindexpresent, :]
    finaltrack_cell_mean_x = finaltrack_cell_mean_x[cloudindexpresent, :]

    finaltrack_dilatecell_meanlat = finaltrack_dilatecell_meanlat[cloudindexpresent, :]
    finaltrack_dilatecell_meanlon = finaltrack_dilatecell_meanlon[cloudindexpresent, :]
    finaltrack_dilatecell_mean_y = finaltrack_dilatecell_mean_y[cloudindexpresent, :]
    finaltrack_dilatecell_mean_x = finaltrack_dilatecell_mean_x[cloudindexpresent, :]

    finaltrack_cell_minlat = finaltrack_cell_minlat[cloudindexpresent, :]
    finaltrack_cell_maxlat = finaltrack_cell_maxlat[cloudindexpresent, :]
    finaltrack_cell_minlon = finaltrack_cell_minlon[cloudindexpresent, :]
    finaltrack_cell_maxlon = finaltrack_cell_maxlon[cloudindexpresent, :]
    finaltrack_cell_min_y = finaltrack_cell_min_y[cloudindexpresent, :]
    finaltrack_cell_max_y = finaltrack_cell_max_y[cloudindexpresent, :]
    finaltrack_cell_min_x = finaltrack_cell_min_x[cloudindexpresent, :]
    finaltrack_cell_max_x = finaltrack_cell_max_x[cloudindexpresent, :]

    finaltrack_cell_maxdbz = finaltrack_cell_maxdbz[cloudindexpresent, :]
    finaltrack_cell_maxETH10dbz = finaltrack_cell_maxETH10dbz[cloudindexpresent, :]
    finaltrack_cell_maxETH20dbz = finaltrack_cell_maxETH20dbz[cloudindexpresent, :]
    finaltrack_cell_maxETH30dbz = finaltrack_cell_maxETH30dbz[cloudindexpresent, :]
    finaltrack_cell_maxETH40dbz = finaltrack_cell_maxETH40dbz[cloudindexpresent, :]
    finaltrack_cell_maxETH50dbz = finaltrack_cell_maxETH50dbz[cloudindexpresent, :]

    finaltrack_core_area = finaltrack_core_area[cloudindexpresent, :]
    finaltrack_cell_area = finaltrack_cell_area[cloudindexpresent, :]
    finaltrack_status = finaltrack_status[cloudindexpresent, :]
    finaltrack_trackinterruptions = finaltrack_trackinterruptions[cloudindexpresent]
    finaltrack_mergenumber = finaltrack_mergenumber[cloudindexpresent, :]
    finaltrack_splitnumber = finaltrack_splitnumber[cloudindexpresent, :]
    finaltrack_cloudnumber = finaltrack_cloudnumber[cloudindexpresent, :]
    finaltrack_cloudidfile = finaltrack_cloudidfile[cloudindexpresent, :, :]

    finaltrack_cell_rangeflag = finaltrack_cell_rangeflag[cloudindexpresent, :]

    # Calculate equivalent radius
    finaltrack_core_radius = np.sqrt(np.divide(finaltrack_core_area, pi))
    finaltrack_cell_radius = np.sqrt(np.divide(finaltrack_cell_area, pi))

    gc.collect()

    #######################################################
    # Correct merger and split track numbers
    print('Correcting mergers and splits')

    # Initialize adjusted matrices
    adjusted_finaltrack_mergenumber = np.ones(np.shape(finaltrack_mergenumber))*fillval
    adjusted_finaltrack_splitnumber = np.ones(np.shape(finaltrack_mergenumber))*fillval
    print(('total tracks: ' + str(numtracks)))

    # Create adjustor
    indexcloudnumber = np.copy(cloudindexpresent) + 1
    adjustor = np.arange(0, np.max(cloudindexpresent)+2)
    for it in range(0, numtracks):
        adjustor[indexcloudnumber[it]] = it+1
    adjustor = np.append(adjustor, fillval)

    # Adjust mergers
    temp_finaltrack_mergenumber = finaltrack_mergenumber.astype(int).ravel()
    temp_finaltrack_mergenumber[temp_finaltrack_mergenumber == fillval] = np.max(cloudindexpresent)+2
    adjusted_finaltrack_mergenumber = adjustor[temp_finaltrack_mergenumber]
    adjusted_finaltrack_mergenumber = np.reshape(adjusted_finaltrack_mergenumber, np.shape(finaltrack_mergenumber))

    # Adjust splitters
    temp_finaltrack_splitnumber = finaltrack_splitnumber.astype(int).ravel()
    temp_finaltrack_splitnumber[temp_finaltrack_splitnumber == fillval] = np.max(cloudindexpresent)+2
    adjusted_finaltrack_splitnumber = adjustor[temp_finaltrack_splitnumber]
    adjusted_finaltrack_splitnumber = np.reshape(adjusted_finaltrack_splitnumber, np.shape(finaltrack_splitnumber))

    print('Merge and split adjustments done')    

    #########################################################################
    # Record starting and ending status
    print('Isolating starting and ending status')

    # Starting status
    finaltrack_startstatus = np.copy(finaltrack_status[:,0])
    finaltrack_startbasetime = np.copy(finaltrack_basetime[:,0])
    finaltrack_startsplit_tracknumber = np.copy(adjusted_finaltrack_splitnumber[:,0])

    finaltrack_startsplit_timeindex = np.full(numtracks, fillval, dtype=np.int32)
    finaltrack_startsplit_cloudnumber = np.full(numtracks, fillval, dtype=np.int32)

    # Ending status
    finaltrack_endstatus = np.full(numtracks, fillval, dtype=np.int32)
    finaltrack_endbasetime = np.full(numtracks, fillval, dtype=type(finaltrack_basetime))
    finaltrack_endmerge_tracknumber = np.full(numtracks, fillval, dtype=np.int32)
    finaltrack_endmerge_timeindex = np.full(numtracks, fillval, dtype=np.int32)
    finaltrack_endmerge_cloudnumber = np.full(numtracks, fillval, dtype=np.int32)

    # Loop over each track
    for itrack in range(0, numtracks):

        # Make sure the track length is < maxtracklength so array access would not be out of bounds
        if ((finaltrack_tracklength[itrack] > 0) & (finaltrack_tracklength[itrack] < maxtracklength)):

            # Get the end basetime
            finaltrack_endbasetime[itrack] = finaltrack_basetime[itrack, finaltrack_tracklength[itrack]-1]
            # Get the status at the last time step of the track
            finaltrack_endstatus[itrack] = np.copy(finaltrack_status[itrack, finaltrack_tracklength[itrack]-1])
            finaltrack_endmerge_tracknumber[itrack] = np.copy(adjusted_finaltrack_mergenumber[itrack, finaltrack_tracklength[itrack]-1])

            # If end merge tracknumber exists, this track ends by merge
            if (finaltrack_endmerge_tracknumber[itrack] >= 0):
                # Get the track number if merges with, -1 convert to track index
                imerge_idx = finaltrack_endmerge_tracknumber[itrack]-1
                # Get all the basetime for the track it merges with
                ibasetime = finaltrack_basetime[imerge_idx, 0:finaltrack_tracklength[imerge_idx]]
                # Find the time index matching the time when merging occurs
                match_timeidx = np.where(ibasetime == finaltrack_endbasetime[itrack])[0]
                if (len(match_timeidx) == 1):
                    #  The time to connect to the track it merges with should be 1 time step after
                    if ((match_timeidx + 1) >= 0) & ((match_timeidx + 1) < maxtracklength):
                        finaltrack_endmerge_timeindex[itrack] = match_timeidx + 1
                        finaltrack_endmerge_cloudnumber[itrack] = finaltrack_cloudnumber[imerge_idx, match_timeidx+1]
                    else:
                        print(f'Merge time occur after track ends??')
                else:
                    print(f'Error: track {itrack} has no matching time in the track it merges with!')
                    # import pdb; pdb.set_trace()
                    sys.exit(itrack)

            # If start split tracknumber exists, this track starts from a split
            if (finaltrack_startsplit_tracknumber[itrack] >= 0):
                # Get the tracknumber it splits from, -1 to convert to track index
                isplit_idx = finaltrack_startsplit_tracknumber[itrack]-1
                # Get all the basetime for the track it splits from
                ibasetime = finaltrack_basetime[isplit_idx, 0:finaltrack_tracklength[isplit_idx]]
                # Find the time index matching the time when splitting occurs
                match_timeidx = np.where(ibasetime == finaltrack_startbasetime[itrack])[0]
                if (len(match_timeidx) == 1):
                    # The time to connect to the track it splits from should be 1 time step prior
                    if (match_timeidx - 1) >= 0:
                        finaltrack_startsplit_timeindex[itrack] = match_timeidx - 1
                        finaltrack_startsplit_cloudnumber[itrack] = finaltrack_cloudnumber[isplit_idx, match_timeidx-1]
                    else:
                        print(f'Split time occur before track starts??')
                else:
                    print(f'Error: track {itrack} has no matching time in the track it splits from!')
                    sys.exit(itrack)


    #######################################################################
    # Write to netcdf
    print('Writing trackstats netcdf')
    print((time.ctime()))
    print(trackstats_outfile)
    print('')

    # Check if file already exists. If exists, delete
    if os.path.isfile(trackstats_outfile):
        os.remove(trackstats_outfile)

    
    # Define output file dimension names
    trackdimname='tracks'
    timedimname='times'
    net.write_trackstats_radar(trackstats_outfile, numtracks, maxtracklength, numcharfilename, \
                               trackdimname, timedimname, \
                               datasource, datadescription, startdate, enddate, \
                               track_version, tracknumbers_version, timegap, basetime_units, \
                               pixel_radius, areathresh, datatimeresolution, fillval, \
                               finaltrack_tracklength, finaltrack_basetime, \
                               finaltrack_cloudidfile, finaltrack_cloudnumber, \
                               finaltrack_core_meanlat, finaltrack_core_meanlon, \
                               finaltrack_core_mean_y, finaltrack_core_mean_x, \
                               finaltrack_cell_meanlat, finaltrack_cell_meanlon, \
                               finaltrack_cell_mean_y, finaltrack_cell_mean_x, \
                               finaltrack_cell_minlat, finaltrack_cell_maxlat, \
                               finaltrack_cell_minlon, finaltrack_cell_maxlon, \
                               finaltrack_cell_min_y, finaltrack_cell_max_y, \
                               finaltrack_cell_min_x, finaltrack_cell_max_x, \
                               finaltrack_dilatecell_meanlat, finaltrack_dilatecell_meanlon, \
                               finaltrack_dilatecell_mean_y, finaltrack_dilatecell_mean_x, \
                               finaltrack_core_area, finaltrack_cell_area, \
                               finaltrack_core_radius, finaltrack_cell_radius, \
                               finaltrack_cell_maxdbz, \
                               finaltrack_cell_maxETH10dbz, finaltrack_cell_maxETH20dbz, finaltrack_cell_maxETH30dbz, \
                               finaltrack_cell_maxETH40dbz, finaltrack_cell_maxETH50dbz, \
                               finaltrack_status, finaltrack_startstatus, finaltrack_endstatus, \
                               finaltrack_startbasetime, finaltrack_endbasetime, \
                               finaltrack_startsplit_tracknumber, finaltrack_startsplit_timeindex, finaltrack_startsplit_cloudnumber, \
                               finaltrack_endmerge_tracknumber, finaltrack_endmerge_timeindex, finaltrack_endmerge_cloudnumber, \
                               finaltrack_trackinterruptions, \
                               adjusted_finaltrack_mergenumber, adjusted_finaltrack_splitnumber, \
                               finaltrack_cell_rangeflag, \
                              )

    # import pdb; pdb.set_trace()
