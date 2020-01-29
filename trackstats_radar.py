# Purpose: This gets statistics about each track from the radar data. 

# Define function that calculates track statistics for satellite data
def trackstats_radar(datasource, datadescription, pixel_radius, geolimits, areathresh, \
                    startdate, enddate, timegap, cloudid_filebase, tracking_inpath, stats_path, \
                    track_version, tracknumbers_version, tracknumbers_filebase, lengthrange=[2,120]):

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
    np.set_printoptions(threshold=np.inf)

    # Set output filename
    trackstats_outfile = stats_path + 'stats_' + tracknumbers_filebase + '_' + startdate + '_' + enddate + '.nc'

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
    latlondata.close()

    # Determine dimensions of data
    # nfiles = len(cloudidfiles_list)
    ny, nx = np.shape(latitude)

    ############################################################################
    # Initialize grids
    print('Initiailizinng matrices')
    print((time.ctime()))

    fillval = -9999
    # nmaxclouds = max(lengthrange)
    maxtracklength = max(lengthrange)
    # finaltrack_tracklength = np.full(int(numtracks), fillval, dtype=np.int32)
    finaltrack_tracklength = np.zeros(int(numtracks), dtype=np.int32)
    # finaltrack_cell_boundary = np.full(int(numtracks), fillval, dtype=np.int32)
    # finaltrack_basetime = np.empty((int(numtracks),int(maxtracklength)), dtype='datetime64[s]')
    finaltrack_basetime = np.full((int(numtracks),int(maxtracklength)), fillval, dtype=np.int32)
    finaltrack_core_meanlat = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=np.float)
    finaltrack_core_meanlon = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=np.float)
    finaltrack_cell_minlat = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=np.float)
    finaltrack_cell_maxlat = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=np.float)
    finaltrack_cell_minlon = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=np.float)
    finaltrack_cell_maxlon = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=np.float)
    finaltrack_cell_maxdbz = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=float)
    finaltrack_core_npix = np.full((int(numtracks),int(maxtracklength)), fillval, dtype=np.int32)
    finaltrack_cell_npix = np.full((int(numtracks),int(maxtracklength)), fillval, dtype=np.int32)
    finaltrack_status = np.full((int(numtracks),int(maxtracklength)), fillval, dtype=np.int32)
    finaltrack_trackinterruptions = np.full(int(numtracks), fillval, dtype=np.int32)
    finaltrack_mergenumber = np.full((int(numtracks),int(maxtracklength)), fillval, dtype=np.int32)
    finaltrack_splitnumber = np.full((int(numtracks),int(maxtracklength)), fillval, dtype=np.int32)
    finaltrack_cloudnumber = np.full((int(numtracks),int(maxtracklength)), fillval, dtype=np.int32)
    # finaltrack_datetimestring = [[['' for x in range(13)] for y in range(int(maxtracklength))] for z in range(int(numtracks))]
    finaltrack_cloudidfile = np.chararray((int(numtracks), int(maxtracklength), int(numcharfilename)))

    #########################################################################################
    # loop over files. Calculate statistics and organize matrices by tracknumber and cloud
    print('Looping over files and calculating statistics for each file')
    print((time.ctime()))
    for nf in range(0, nfiles):
    # for nf in range(0, 2):
        # print(('File #: ' + str(nf)))
        # print((time.ctime()))

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
            # file_cloudtype = file_cloudiddata['cloudtype'][:]
            file_all_cloudnumber = file_cloudiddata['cloudnumber'][:]
            file_corecold_cloudnumber = file_cloudiddata['convcold_cloudnumber'][:]
            conv_mask1 = file_cloudiddata['conv_mask1'][:]
            conv_mask2 = file_cloudiddata['conv_mask2'][:]
            file_basetime = file_cloudiddata['basetime'][:]
            basetime_units = file_cloudiddata['basetime'].units
            # basetime_calendar = file_cloudiddata['basetime'].calendar
            file_cloudiddata.close()

            file_datetimestring = cloudid_file[len(tracking_inpath) + len(cloudid_filebase):-3]

            # Find unique track numbers
            uniquetracknumbers = np.unique(file_tracknumbers)
            uniquetracknumbers = uniquetracknumbers[np.isfinite(uniquetracknumbers)]
            uniquetracknumbers = uniquetracknumbers[uniquetracknumbers > 0].astype(int)
            
            # if (len(uniquetracknumbers) > 1):
            #     import pdb; pdb.set_trace()

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
                    # Find cloud in cloudid file associated with this track, and convective core (conv_mask1 == 1)
                    corearea = np.array(np.where((file_corecold_cloudnumber[0,:,:] == cloudnumber) & (conv_mask1[0,:,:] == 1)))
                    ncorepix = np.shape(corearea)[1]

                    # Convective cell (conv_mask2 == 1)
                    cellarea = np.array(np.where((file_corecold_cloudnumber[0,:,:] == cloudnumber) & (conv_mask2[0,:,:] == 1)))
                    ncellpix = np.shape(cellarea)[1]

                    # Record previous length of the track (initially all track lengths start at 0)
                    nc = finaltrack_tracklength[itrack-1]
                    # Add 1 to the current track length
                    finaltrack_tracklength[itrack-1] = nc+1

                    if nc < maxtracklength:
                        # Save information that links this cloud back to its raw pixel level data
                        # finaltrack_basetime[itrack-1, nc] = np.array([pd.to_datetime(num2date(file_basetime, units=basetime_units, calendar=basetime_calendar))], dtype='datetime64[s]')[0, 0]
                        finaltrack_basetime[itrack-1, nc] = np.array(file_basetime[0])
                        finaltrack_cloudnumber[itrack-1,nc] = cloudnumber
                        finaltrack_cloudidfile[itrack-1][nc][:] = fname

                        ###############################################################
                        # Calculate statistics for this cell
                        if (ncellpix > 0):
                            # Location statistics of core+cold anvil (aka the convective system)
                            corelat = latitude[corearea[0], corearea[1]]
                            corelon = longitude[corearea[0], corearea[1]]

                            celllat = latitude[cellarea[0], cellarea[1]]
                            celllon = longitude[cellarea[0], cellarea[1]]

                            # Core lat/lon (center location)
                            finaltrack_core_meanlat[itrack-1, nc] = np.nanmean(corelat)
                            finaltrack_core_meanlon[itrack-1, nc] = np.nanmean(corelon)

                            # Cell min/max lat/lon (for its maximum spatial extent)
                            finaltrack_cell_minlat[itrack-1, nc] = np.nanmin(celllat)
                            finaltrack_cell_maxlat[itrack-1, nc] = np.nanmax(celllat)
                            finaltrack_cell_minlon[itrack-1, nc] = np.nanmin(celllon)
                            finaltrack_cell_maxlon[itrack-1, nc] = np.nanmax(celllon)

                            # Save number of pixels (metric for size)
                            finaltrack_core_npix[itrack-1, nc] = ncorepix
                            finaltrack_cell_npix[itrack-1, nc] = ncellpix

                            # Reflectivity maximum
                            finaltrack_cell_maxdbz[itrack-1, nc] = np.nanmean(file_dbz[0,cellarea[0],cellarea[1]])

                            # Save track information. Need to subtract one since cloudnumber gives the number of the cloud (which starts at one), but we are looking for its index (which starts at zero)
                            finaltrack_status[itrack-1, nc] = np.copy(trackstatus[0, nf, cloudindex])
                            finaltrack_mergenumber[itrack-1, nc] = np.copy(trackmerge[0, nf, cloudindex])
                            finaltrack_splitnumber[itrack-1, nc] = np.copy(tracksplit[0, nf, cloudindex])
                            finaltrack_trackinterruptions[itrack-1] = np.copy(trackreset[0, nf, cloudindex])

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

    # maxtracklength = np.nanmax(finaltrack_tracklength)

    finaltrack_tracklength = finaltrack_tracklength[cloudindexpresent]
    finaltrack_basetime = finaltrack_basetime[cloudindexpresent, :]
    finaltrack_core_meanlat = finaltrack_core_meanlat[cloudindexpresent, :]
    finaltrack_core_meanlon = finaltrack_core_meanlon[cloudindexpresent, :]
    finaltrack_cell_minlat = finaltrack_cell_minlat[cloudindexpresent, :]
    finaltrack_cell_maxlat = finaltrack_cell_maxlat[cloudindexpresent, :]
    finaltrack_cell_minlon = finaltrack_cell_minlon[cloudindexpresent, :]
    finaltrack_cell_maxlon = finaltrack_cell_maxlon[cloudindexpresent, :]
    finaltrack_cell_maxdbz = finaltrack_cell_maxdbz[cloudindexpresent, :]
    finaltrack_core_npix = finaltrack_core_npix[cloudindexpresent, :]
    finaltrack_cell_npix = finaltrack_cell_npix[cloudindexpresent, :]
    finaltrack_status = finaltrack_status[cloudindexpresent, :]
    finaltrack_trackinterruptions = finaltrack_trackinterruptions[cloudindexpresent]
    finaltrack_mergenumber = finaltrack_mergenumber[cloudindexpresent, :]
    finaltrack_splitnumber = finaltrack_splitnumber[cloudindexpresent, :]
    finaltrack_cloudnumber = finaltrack_cloudnumber[cloudindexpresent, :]
    finaltrack_cloudidfile = finaltrack_cloudidfile[cloudindexpresent, :, :]

    # Calculate equivalent radius
    finaltrack_core_radius = np.sqrt(np.divide(finaltrack_core_npix * pixel_radius**2, pi))
    finaltrack_cell_radius = np.sqrt(np.divide(finaltrack_cell_npix * pixel_radius**2, pi))

    gc.collect()

    ########################################################
    # # Correct merger and split cloud numbers
    # print('Correcting mergers and splits')

    # # Initialize adjusted matrices
    # adjusted_finaltrack_mergenumber = np.ones(np.shape(finaltrack_mergenumber))*fillval
    # adjusted_finaltrack_splitnumber = np.ones(np.shape(finaltrack_mergenumber))*fillval
    # print(('total tracks: ' + str(numtracks)))

    # # Create adjustor
    # indexcloudnumber = np.copy(cloudindexpresent) + 1
    # adjustor = np.arange(0, np.max(cloudindexpresent)+2)
    # for it in range(0, numtracks):
    #     adjustor[indexcloudnumber[it]] = it+1
    # adjustor = np.append(adjustor, fillval)

    # # Adjust mergers
    # temp_finaltrack_mergenumber = finaltrack_mergenumber.astype(int).ravel()
    # temp_finaltrack_mergenumber[temp_finaltrack_mergenumber == fillval] = np.max(cloudindexpresent)+2
    # adjusted_finaltrack_mergenumber = adjustor[temp_finaltrack_mergenumber]
    # adjusted_finaltrack_mergenumber = np.reshape(adjusted_finaltrack_mergenumber, np.shape(finaltrack_mergenumber))

    # # Adjust splitters
    # temp_finaltrack_splitnumber = finaltrack_splitnumber.astype(int).ravel()
    # temp_finaltrack_splitnumber[temp_finaltrack_splitnumber == fillval] = np.max(cloudindexpresent)+2
    # adjusted_finaltrack_splitnumber = adjustor[temp_finaltrack_splitnumber]
    # adjusted_finaltrack_splitnumber = np.reshape(adjusted_finaltrack_splitnumber, np.shape(finaltrack_splitnumber))

    # print('Adjustment done')

    #########################################################################
    # Record starting and ending status
    print('Isolating starting and ending status')

    # Starting status
    finaltrack_startstatus = np.copy(finaltrack_status[:,0])

    # Ending status
    finaltrack_endstatus = np.ones(len(finaltrack_startstatus))*fillval

    for trackstep in range(0, numtracks):
        if ((finaltrack_tracklength[trackstep] > 0) & (finaltrack_tracklength[trackstep] < maxtracklength)):
            finaltrack_endstatus[trackstep] = np.copy(finaltrack_status[trackstep,finaltrack_tracklength[trackstep] - 1])

    #######################################################################
    # Write to netcdf
    print('Writing trackstats netcdf')
    print((time.ctime()))
    print(trackstats_outfile)
    print('')

    # Check if file already exists. If exists, delete
    if os.path.isfile(trackstats_outfile):
        os.remove(trackstats_outfile)

    import netcdf_io_trackstats as net
    net.write_trackstats_radar(trackstats_outfile, numtracks, maxtracklength, numcharfilename, \
                               datasource, datadescription, startdate, enddate, \
                               track_version, tracknumbers_version, timegap, basetime_units, \
                               pixel_radius, areathresh, fillval, \
                               finaltrack_tracklength, finaltrack_basetime, \
                               finaltrack_cloudidfile, finaltrack_cloudnumber, \
                               finaltrack_core_meanlat, finaltrack_core_meanlon, \
                               finaltrack_cell_minlat, finaltrack_cell_maxlat, \
                               finaltrack_cell_minlon, finaltrack_cell_maxlon, \
                               finaltrack_core_npix, finaltrack_cell_npix, \
                               finaltrack_core_radius, finaltrack_cell_radius, \
                               finaltrack_cell_maxdbz, finaltrack_status, \
                               finaltrack_startstatus, finaltrack_endstatus, \
                               finaltrack_trackinterruptions, \
                               finaltrack_mergenumber, finaltrack_splitnumber, \
                            #    adjusted_finaltrack_mergenumber, adjusted_finaltrack_splitnumber, \
                              )

    # import pdb; pdb.set_trace()
