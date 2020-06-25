# Purpose: This gets statistics about each track from the radar data.
# This version uses multiprocessing for parallelization
# Author: Zhe Feng (zhe.feng@pnnl.gov)

def trackstats_radar(datasource, datadescription, pixel_radius, datatimeresolution, geolimits, areathresh, \
                    startdate, enddate, timegap, cloudid_filebase, tracking_inpath, stats_path, \
                    track_version, tracknumbers_version, tracknumbers_filebase, lengthrange, \
                    nprocesses):

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
    from multiprocessing import Pool
    import netcdf_io_trackstats as net
    from trackstats_radar_func import calc_stats_radar
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
    x_coord = latlondata['x'][:]/1000.  # convert unit to [km]
    y_coord = latlondata['y'][:]/1000.  # convert unit to [km]
    basetime_units = latlondata['basetime'].units
    latlondata.close()

    # Determine dimensions of data
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
    finaltrack_basetime = np.full((int(numtracks),int(maxtracklength)), fillval, dtype=np.float)
    finaltrack_core_meanlat = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=np.float)
    finaltrack_core_meanlon = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=np.float)
    finaltrack_core_mean_x = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=np.float)
    finaltrack_core_mean_y = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=np.float)

    finaltrack_cell_meanlat = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=np.float)
    finaltrack_cell_meanlon = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=np.float)
    finaltrack_cell_mean_x = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=np.float)
    finaltrack_cell_mean_y = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=np.float)

    finaltrack_cell_minlat = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=np.float)
    finaltrack_cell_maxlat = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=np.float)
    finaltrack_cell_minlon = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=np.float)
    finaltrack_cell_maxlon = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=np.float)
    finaltrack_cell_min_y = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=np.float)
    finaltrack_cell_max_y = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=np.float)
    finaltrack_cell_min_x = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=np.float)
    finaltrack_cell_max_x = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=np.float)
    
    finaltrack_dilatecell_meanlat = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=np.float)
    finaltrack_dilatecell_meanlon = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=np.float)
    finaltrack_dilatecell_mean_x = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=np.float)
    finaltrack_dilatecell_mean_y = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=np.float)

    finaltrack_cell_maxdbz = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=float)

    finaltrack_cell_maxETH10dbz = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=np.float)
    finaltrack_cell_maxETH20dbz = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=np.float)
    finaltrack_cell_maxETH30dbz = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=np.float)
    finaltrack_cell_maxETH40dbz = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=np.float)
    finaltrack_cell_maxETH50dbz = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=np.float)

    finaltrack_core_area = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=float)
    finaltrack_cell_area = np.full((int(numtracks),int(maxtracklength)), np.nan, dtype=float)
    
    finaltrack_trackinterruptions = np.full(int(numtracks), fillval, dtype=np.int32)
    finaltrack_status = np.full((int(numtracks),int(maxtracklength)), fillval, dtype=np.int32)
    finaltrack_mergenumber = np.full((int(numtracks),int(maxtracklength)), fillval, dtype=np.int32)
    finaltrack_splitnumber = np.full((int(numtracks),int(maxtracklength)), fillval, dtype=np.int32)
    finaltrack_cloudnumber = np.full((int(numtracks),int(maxtracklength)), fillval, dtype=np.int32)
    # finaltrack_datetimestring = [[['' for x in range(13)] for y in range(int(maxtracklength))] for z in range(int(numtracks))]
    finaltrack_cloudidfile = np.chararray((int(numtracks), int(maxtracklength), int(numcharfilename)))

    #########################################################################################
    # loop over files. Calculate statistics and organize matrices by tracknumber and cloud
    print('Looping over files and calculating statistics for each file')
    print((time.ctime()))

    # for nf in range(0, nfiles):
    #     Results = calc_stats_radar(tracknumbers[0, nf, :], cloudidfiles[nf], tracking_inpath, cloudid_filebase, \
    #                                 numcharfilename, latitude, longitude, x_coord, y_coord, \
    #                                 nx, ny, pixel_radius, trackstatus[0, nf, :], \
    #                                 trackmerge[0, nf, :], tracksplit[0, nf, :], trackreset[0, nf, :])
    #     import pdb; pdb.set_trace()

    # Parallel at each file
    with Pool(nprocesses) as pool:
        Results = pool.starmap(calc_stats_radar, [(tracknumbers[0, nf, :], cloudidfiles[nf], tracking_inpath, cloudid_filebase, \
                                numcharfilename, latitude, longitude, x_coord, y_coord, \
                                nx, ny, pixel_radius, trackstatus[0, nf, :], \
                                trackmerge[0, nf, :], tracksplit[0, nf, :], trackreset[0, nf, :]) for nf in range(0,nfiles)])
        pool.close()

    # Collect pool results
    # Loop over each file from the pool results
    for nf in range(0, nfiles):
        tmp = Results[nf]
        if (tmp is not None) :
            tracknumbertmp = tmp[0]-1   # unique tracknumbers in the current file
            numtrackstmp = tmp[1]       # number of tracks in the current file

            # Record the current length of the track by adding 1
            finaltrack_tracklength[tracknumbertmp] = finaltrack_tracklength[tracknumbertmp] + 1

            # Loop over each track in the current file
            for iitrack in range(numtrackstmp):
                # Make sure the track length does not exceed maxtracklength
                if (finaltrack_tracklength[tracknumbertmp[iitrack]] <= maxtracklength):
                    # Note the variables must be in the exact same order as they are returned in "tmp"
                    finaltrack_basetime[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[2][iitrack]
                    finaltrack_cloudnumber[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[3][iitrack]
                    finaltrack_cloudidfile[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1, :] = tmp[4][iitrack,:]
                    finaltrack_core_meanlat[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[5][iitrack]
                    finaltrack_core_meanlon[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[6][iitrack]
                    finaltrack_core_mean_x[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[7][iitrack]
                    finaltrack_core_mean_y[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[8][iitrack]
                    finaltrack_cell_meanlat[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[9][iitrack]
                    finaltrack_cell_meanlon[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[10][iitrack]
                    finaltrack_cell_mean_x[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[11][iitrack]
                    finaltrack_cell_mean_y[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[12][iitrack]
                    finaltrack_cell_minlat[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[13][iitrack]
                    finaltrack_cell_maxlat[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[14][iitrack]
                    finaltrack_cell_minlon[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[15][iitrack]
                    finaltrack_cell_maxlon[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[16][iitrack]
                    finaltrack_cell_min_y[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[17][iitrack]
                    finaltrack_cell_max_y[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[18][iitrack]
                    finaltrack_cell_min_x[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[19][iitrack]
                    finaltrack_cell_max_x[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[20][iitrack]
                    finaltrack_dilatecell_meanlat[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[21][iitrack]
                    finaltrack_dilatecell_meanlon[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[22][iitrack]
                    finaltrack_dilatecell_mean_x[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[23][iitrack]
                    finaltrack_dilatecell_mean_y[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[24][iitrack]
                    finaltrack_cell_maxdbz[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[25][iitrack]
                    finaltrack_cell_maxETH10dbz[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[26][iitrack]
                    finaltrack_cell_maxETH20dbz[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[27][iitrack]
                    finaltrack_cell_maxETH30dbz[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[28][iitrack]
                    finaltrack_cell_maxETH40dbz[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[29][iitrack]
                    finaltrack_cell_maxETH50dbz[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[30][iitrack]
                    finaltrack_core_area[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[31][iitrack]
                    finaltrack_cell_area[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[32][iitrack]
                    finaltrack_status[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[33][iitrack]
                    finaltrack_trackinterruptions[tracknumbertmp[iitrack]] = tmp[34][iitrack]
                    finaltrack_mergenumber[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[35][iitrack]
                    finaltrack_splitnumber[tracknumbertmp[iitrack], finaltrack_tracklength[tracknumbertmp[iitrack]]-1] = tmp[36][iitrack]

            
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

    # Calculate equivalent radius
    finaltrack_core_radius = np.sqrt(np.divide(finaltrack_core_area, pi))
    finaltrack_cell_radius = np.sqrt(np.divide(finaltrack_cell_area, pi))

    gc.collect()

    # import pdb; pdb.set_trace()

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

    
    # Define output file dimension names
    trackdimname = 'tracks'
    timedimname = 'times'
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
                               finaltrack_trackinterruptions, \
                               finaltrack_mergenumber, finaltrack_splitnumber, \
                            #    adjusted_finaltrack_mergenumber, adjusted_finaltrack_splitnumber, \
                              )

    # import pdb; pdb.set_trace()
