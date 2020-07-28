# Purpose: This gets statistics about each track from the satellite data. 

# Author: Orginial IDL version written by Sally A. McFarline (sally.mcfarlane@pnnl.gov) and modified for Zhe Feng (zhe.feng@pnnl.gov). Python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov)

# Define function that calculates track statistics for satellite data
def trackstats_ct(datasource, datadescription, pixel_radius, geolimits, areathresh, cloudtb_threshs, absolutetb_threshs, startdate, enddate, timegap, cloudid_filebase, tracking_inpath, stats_path, track_version, tracknumbers_version, tracknumbers_filebase, nprocesses, lengthrange=[2,120]):
    # Inputs:
    # datasource - source of the data
    # datadescription - description of data source, included in all output file names
    # pixel_radius - radius of pixels in km
    # latlon_file - filename of the file that contains the latitude and longitude data
    # geolimits - 4-element array with plotting boundaries [lat_min, lon_min, lat_max, lon_max]
    # areathresh - minimum core + cold anvil area of a tracked cloud
    # cloudtb_threshs - brightness temperature thresholds for convective classification
    # absolutetb_threshs - brightness temperature thresholds defining the valid data range
    # startdate - starting date and time of the data
    # enddate - ending date and time of the data
    # cloudid_filebase - header of the cloudid data files
    # tracking_inpath - location of the cloudid and single track data
    # stats_path - location of the track data. also the location where the data from this code will be saved
    # track_version - Version of track single cloud files
    # tracknumbers_version - Verison of the complete track files
    # tracknumbers_filebase - header of the tracking matrix generated in the previous code. 
    # cloudid_filebase - 
    # lengthrange - Optional. Set this keyword to a vector [minlength,maxlength] to specify the lifetime range for the tracks.Fdef

    # Outputs: (One netcdf file with with each track represented as a row):
    # lifetime - duration of each track
    # basetime - seconds since 1970-01-01 for each cloud in a track
    # cloudidfiles - cloudid filename associated with each cloud in a track
    # meanlat - mean latitude of each cloud in a track of the core and cold anvil
    # meanlon - mean longitude of each cloud in a track of the core and cold anvil
    # minlat - minimum latitude of each cloud in a track of the core and cold anvil
    # minlon - minimum longitude of each cloud in a track of the core and cold anvil
    # maxlat - maximum latitude of each cloud in a track of the core and cold anvil
    # maxlon - maximum longitude of each cloud in a track of the core and cold anvil
    # radius - equivalent radius of each cloud in a track of the core and cold anvil
    # radius_warmanvil - equivalent radius of core, cold anvil, and warm anvil
    # npix - number of pixels in the core and cold anvil
    # nconv - number of pixels in the core
    # ncoldanvil - number of pixels in the cold anvil
    # nwarmanvil - number of pixels in the warm anvil
    # cloudnumber - number that corresponds to this cloud in the cloudid file
    # status - flag indicating how a cloud evolves over time
    # startstatus - flag indicating how this track started
    # endstatus - flag indicating how this track ends
    # mergenumbers - number indicating which track this cloud merges into
    # splitnumbers - number indicating which track this cloud split from
    # trackinterruptions - flag indicating if this track has incomplete data
    # boundary - flag indicating whether the track intersects the edge of the data
    # mintb - minimum brightness temperature of the core and cold anvil
    # meantb - mean brightness temperature of the core and cold anvil
    # meantb_conv - mean brightness temperature of the core
    # histtb - histogram of the brightness temperatures in the core and cold anvil
    # majoraxis - length of the major axis of the core and cold anvil
    # orientation - angular position of the core and cold anvil
    # eccentricity - eccentricity of the core and cold anvil
    # perimeter - approximate size of the perimeter in the core and cold anvil
    # xcenter - x-coordinate of the geometric center
    # ycenter - y-coordinate of the geometric center
    # xcenter_weighted - x-coordinate of the brightness temperature weighted center
    # ycenter_weighted - y-coordinate of the brightness temperature weighted center

    ###################################################################################
    # Initialize modules
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
    np.set_printoptions(threshold=np.inf)

    #############################################################################
    # Set constants

    # Set output filename
    trackstats_outfile = stats_path + 'stats_' + tracknumbers_filebase + '_' + startdate + '_' + enddate + '.nc'

    ###################################################################################
    # # Load latitude and longitude grid. These were created in subroutine_idclouds and is saved in each file.
    # print('Determining which files will be processed')
    # print((time.ctime()))

    # # Find filenames of idcloud files
    # temp_cloudidfiles = fnmatch.filter(os.listdir(tracking_inpath), cloudid_filebase +'*')
    # cloudidfiles_list = temp_cloudidfiles  # KB ADDED
    
    # # Sort the files by date and time   # KB added
    # def fdatetime(x):
    #     return(x[-11:])
    # cloudidfiles_list = sorted(cloudidfiles_list, key = fdatetime)
    
    # # Select one file. Any file is fine since they all have the map of latitude and longitude saved.
    # temp_cloudidfiles = temp_cloudidfiles[0]

    # # Load latitude and longitude grid
    # latlondata = Dataset(tracking_inpath + temp_cloudidfiles, 'r')
    # longitude = latlondata.variables['longitude'][:]
    # latitude = latlondata.variables['latitude'][:]
    # latlondata.close()

    #############################################################################
    # Load track data
    print('Loading gettracks data')
    print((time.ctime()))
    cloudtrack_file = stats_path + tracknumbers_filebase + '_' + startdate + '_' + enddate + '.nc'
    print(cloudtrack_file)
    
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

    # Convert filenames and timegap to string
    # numcharfilename = len(list(cloudidfiles_list[0]))
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

    nmaxclouds = max(lengthrange)

    finaltrack_tracklength = np.zeros(int(numtracks), dtype=np.int32)
    finaltrack_corecold_boundary = np.ones((int(numtracks), int(nmaxclouds)), dtype=np.int32)*-9999
    finaltrack_basetime = np.empty((int(numtracks),int(nmaxclouds)), dtype='datetime64[s]')
    finaltrack_corecold_radius = np.ones((int(numtracks),int(nmaxclouds)), dtype=float)*np.nan
    finaltrack_corecold_meanlat = np.ones((int(numtracks),int(nmaxclouds)), dtype=float)*np.nan
    finaltrack_corecold_meanlon = np.ones((int(numtracks),int(nmaxclouds)), dtype=float)*np.nan
    finaltrack_corecold_maxlon = np.ones((int(numtracks),int(nmaxclouds)), dtype=float)*np.nan
    finaltrack_corecold_maxlat = np.ones((int(numtracks),int(nmaxclouds)), dtype=float)*np.nan
    finaltrack_ncorecoldpix = np.ones((int(numtracks),int(nmaxclouds)), dtype=np.int32)*-9999
    finaltrack_corecold_minlon = np.ones((int(numtracks),int(nmaxclouds)), dtype=float)*np.nan
    finaltrack_corecold_minlat = np.ones((int(numtracks),int(nmaxclouds)), dtype=float)*np.nan
    finaltrack_ncorepix = np.ones((int(numtracks),int(nmaxclouds)), dtype=np.int32)*-9999
    finaltrack_ncoldpix = np.ones((int(numtracks),int(nmaxclouds)), dtype=np.int32)*-9999
    finaltrack_corecold_status = np.ones((int(numtracks),int(nmaxclouds)), dtype=np.int32)*-9999
    finaltrack_corecold_trackinterruptions = np.ones(int(numtracks), dtype=np.int32)*-9999
    finaltrack_corecold_mergenumber = np.ones((int(numtracks),int(nmaxclouds)), dtype=np.int32)*-9999
    finaltrack_corecold_splitnumber = np.ones((int(numtracks),int(nmaxclouds)), dtype=np.int32)*-9999
    finaltrack_corecold_cloudnumber = np.ones((int(numtracks),int(nmaxclouds)), dtype=np.int32)*-9999
    finaltrack_cloudtype_low = np.ones((int(numtracks),int(nmaxclouds)), dtype=np.int32)*-9999
    finaltrack_cloudtype_conglow = np.ones((int(numtracks),int(nmaxclouds)), dtype=np.int32)*-9999
    finaltrack_cloudtype_conghigh = np.ones((int(numtracks),int(nmaxclouds)), dtype=np.int32)*-9999
    finaltrack_cloudtype_deep = np.ones((int(numtracks),int(nmaxclouds)), dtype=np.int32)*-9999
    finaltrack_datetimestring = [[['' for x in range(13)] for y in range(int(nmaxclouds))] for z in range(int(numtracks))]
    finaltrack_cloudidfile = np.chararray((int(numtracks), int(nmaxclouds), int(numcharfilename)))
    finaltrack_corecold_majoraxis = np.ones((int(numtracks),int(nmaxclouds)), dtype=float)*np.nan
    finaltrack_corecold_orientation = np.ones((int(numtracks),int(nmaxclouds)), dtype=float)*np.nan 
    finaltrack_corecold_eccentricity = np.ones((int(numtracks),int(nmaxclouds)), dtype=float)*np.nan
    finaltrack_corecold_perimeter = np.ones((int(numtracks),int(nmaxclouds)), dtype=float)*np.nan
    finaltrack_corecold_xcenter = np.ones((int(numtracks),int(nmaxclouds)), dtype=float)*-9999
    finaltrack_corecold_ycenter = np.ones((int(numtracks),int(nmaxclouds)), dtype=float)*-9999
    finaltrack_corecold_xweightedcenter = np.ones((int(numtracks),int(nmaxclouds)), dtype=float)*-9999
    finaltrack_corecold_yweightedcenter = np.ones((int(numtracks),int(nmaxclouds)), dtype=float)*-9999

    #########################################################################################
    # loop over files. Calculate statistics and organize matrices by tracknumber and cloud
    print('Looping over files and calculating statistics for each file')
    print((time.ctime()))
    #parallel here, by Jianfeng Li
    from trackstats_ct_single import calc_stats_single
    with Pool(nprocesses) as pool:
        Results=pool.starmap(calc_stats_single,[(tracknumbers[0, nf, :],cloudidfiles[nf],tracking_inpath,cloudid_filebase, \
                numcharfilename, latitude, longitude, geolimits, nx, ny, pixel_radius, trackstatus[0, nf, :], \
                trackmerge[0, nf, :], tracksplit[0, nf, :], trackreset[0, nf, :]) for nf in range(0,nfiles)])
        pool.close()
        
    #collect pool results
    for nf in range(0, nfiles):
        tmp=Results[nf]
        if (tmp is not None) :
            tracknumbertmp=tmp[0]-1
            numtrackstmp=tmp[1]
            finaltrack_tracklength[tracknumbertmp]=finaltrack_tracklength[tracknumbertmp]+1
            for iitrack in range(numtrackstmp):
                if (finaltrack_tracklength[tracknumbertmp[iitrack]] <= nmaxclouds):
                    finaltrack_basetime[tracknumbertmp[iitrack],finaltrack_tracklength[tracknumbertmp[iitrack]]-1]=tmp[2][iitrack]
                    finaltrack_corecold_cloudnumber[tracknumbertmp[iitrack],finaltrack_tracklength[tracknumbertmp[iitrack]]-1]=tmp[3][iitrack]
                    finaltrack_cloudidfile[tracknumbertmp[iitrack],finaltrack_tracklength[tracknumbertmp[iitrack]]-1,:]=tmp[4][iitrack,:]
                    finaltrack_datetimestring[tracknumbertmp[iitrack]][finaltrack_tracklength[tracknumbertmp[iitrack]]-1][:]=tmp[5][iitrack][:]
                    finaltrack_corecold_meanlat[tracknumbertmp[iitrack],finaltrack_tracklength[tracknumbertmp[iitrack]]-1]=tmp[6][iitrack]
                    finaltrack_corecold_meanlon[tracknumbertmp[iitrack],finaltrack_tracklength[tracknumbertmp[iitrack]]-1]=tmp[7][iitrack]
                    finaltrack_corecold_minlat[tracknumbertmp[iitrack],finaltrack_tracklength[tracknumbertmp[iitrack]]-1]=tmp[8][iitrack]
                    finaltrack_corecold_minlon[tracknumbertmp[iitrack],finaltrack_tracklength[tracknumbertmp[iitrack]]-1]=tmp[9][iitrack]
                    finaltrack_corecold_maxlat[tracknumbertmp[iitrack],finaltrack_tracklength[tracknumbertmp[iitrack]]-1]=tmp[10][iitrack]
                    finaltrack_corecold_maxlon[tracknumbertmp[iitrack],finaltrack_tracklength[tracknumbertmp[iitrack]]-1]=tmp[11][iitrack]
                    finaltrack_corecold_boundary[tracknumbertmp[iitrack],finaltrack_tracklength[tracknumbertmp[iitrack]]-1]=tmp[12][iitrack]
                    finaltrack_ncorecoldpix[tracknumbertmp[iitrack],finaltrack_tracklength[tracknumbertmp[iitrack]]-1]=tmp[13][iitrack]
                    finaltrack_ncorepix[tracknumbertmp[iitrack],finaltrack_tracklength[tracknumbertmp[iitrack]]-1]=tmp[14][iitrack]
                    finaltrack_ncoldpix[tracknumbertmp[iitrack],finaltrack_tracklength[tracknumbertmp[iitrack]]-1]=tmp[15][iitrack]
                    finaltrack_cloudtype_low[tracknumbertmp[iitrack],finaltrack_tracklength[tracknumbertmp[iitrack]]-1]=tmp[16][iitrack]
                    finaltrack_cloudtype_conglow[tracknumbertmp[iitrack],finaltrack_tracklength[tracknumbertmp[iitrack]]-1]=tmp[17][iitrack]
                    finaltrack_cloudtype_conghigh[tracknumbertmp[iitrack],finaltrack_tracklength[tracknumbertmp[iitrack]]-1]=tmp[18][iitrack]
                    finaltrack_cloudtype_deep[tracknumbertmp[iitrack],finaltrack_tracklength[tracknumbertmp[iitrack]]-1]=tmp[19][iitrack]
                    #finaltrack_corecold_eccentricity[tracknumbertmp[iitrack],finaltrack_tracklength[tracknumbertmp[iitrack]]-1]=tmp[16][iitrack]
                    #finaltrack_corecold_majoraxis[tracknumbertmp[iitrack],finaltrack_tracklength[tracknumbertmp[iitrack]]-1]=tmp[17][iitrack]
                    #finaltrack_corecold_orientation[tracknumbertmp[iitrack],finaltrack_tracklength[tracknumbertmp[iitrack]]-1]=tmp[18][iitrack]
                    #finaltrack_corecold_perimeter[tracknumbertmp[iitrack],finaltrack_tracklength[tracknumbertmp[iitrack]]-1]=tmp[19][iitrack]
                    #finaltrack_corecold_ycenter[tracknumbertmp[iitrack],finaltrack_tracklength[tracknumbertmp[iitrack]]-1]=tmp[20][iitrack]
                    #finaltrack_corecold_xcenter[tracknumbertmp[iitrack],finaltrack_tracklength[tracknumbertmp[iitrack]]-1]=tmp[21][iitrack]
                    #finaltrack_corecold_yweightedcenter[tracknumbertmp[iitrack],finaltrack_tracklength[tracknumbertmp[iitrack]]-1]=tmp[22][iitrack]
                    #finaltrack_corecold_xweightedcenter[tracknumbertmp[iitrack],finaltrack_tracklength[tracknumbertmp[iitrack]]-1]=tmp[23][iitrack]
                    finaltrack_corecold_radius[tracknumbertmp[iitrack],finaltrack_tracklength[tracknumbertmp[iitrack]]-1]=tmp[20][iitrack]
                    finaltrack_corecold_status[tracknumbertmp[iitrack],finaltrack_tracklength[tracknumbertmp[iitrack]]-1]=tmp[21][iitrack]
                    finaltrack_corecold_mergenumber[tracknumbertmp[iitrack],finaltrack_tracklength[tracknumbertmp[iitrack]]-1]=tmp[22][iitrack]
                    finaltrack_corecold_splitnumber[tracknumbertmp[iitrack],finaltrack_tracklength[tracknumbertmp[iitrack]]-1]=tmp[23][iitrack]
                    finaltrack_corecold_trackinterruptions[tracknumbertmp[iitrack]]=tmp[24][iitrack]
                    basetime_units=tmp[25]
                    basetime_calendar=tmp[26]
                
    ###############################################################
    ## Remove tracks that have no cells. These tracks are short.
    print('Removing tracks with no cells')
    print((time.ctime()))
    gc.collect()

    #cloudindexpresent = np.array(np.where(finaltrack_tracklength != 0))[0, :]
    cloudindexpresent = np.array(np.where(finaltrack_tracklength >= 6))[0, :]
    
    numtracks = len(cloudindexpresent)

    maxtracklength = np.nanmax(finaltrack_tracklength)


    finaltrack_tracklength = finaltrack_tracklength[cloudindexpresent]
    finaltrack_corecold_boundary = finaltrack_corecold_boundary[cloudindexpresent, 0:maxtracklength]
    finaltrack_basetime = finaltrack_basetime[cloudindexpresent, 0:maxtracklength]
    finaltrack_corecold_radius = finaltrack_corecold_radius[cloudindexpresent, 0:maxtracklength]
    finaltrack_corecold_meanlat = finaltrack_corecold_meanlat[cloudindexpresent, 0:maxtracklength]
    finaltrack_corecold_meanlon = finaltrack_corecold_meanlon[cloudindexpresent, 0:maxtracklength]
    finaltrack_corecold_maxlon = finaltrack_corecold_maxlon[cloudindexpresent, 0:maxtracklength]
    finaltrack_corecold_maxlat = finaltrack_corecold_maxlat[cloudindexpresent, 0:maxtracklength]
    finaltrack_corecold_minlon = finaltrack_corecold_minlon[cloudindexpresent, 0:maxtracklength]
    finaltrack_corecold_minlat = finaltrack_corecold_minlat[cloudindexpresent, 0:maxtracklength]
    finaltrack_ncorecoldpix = finaltrack_ncorecoldpix[cloudindexpresent, 0:maxtracklength]
    finaltrack_ncorepix = finaltrack_ncorepix[cloudindexpresent, 0:maxtracklength]
    finaltrack_ncoldpix = finaltrack_ncoldpix[cloudindexpresent, 0:maxtracklength]
    finaltrack_corecold_status = finaltrack_corecold_status[cloudindexpresent, 0:maxtracklength]
    finaltrack_corecold_trackinterruptions = finaltrack_corecold_trackinterruptions[cloudindexpresent]
    finaltrack_corecold_mergenumber = finaltrack_corecold_mergenumber[cloudindexpresent, 0:maxtracklength]
    finaltrack_corecold_splitnumber = finaltrack_corecold_splitnumber[cloudindexpresent, 0:maxtracklength]
    finaltrack_corecold_cloudnumber = finaltrack_corecold_cloudnumber[cloudindexpresent, 0:maxtracklength]
    finaltrack_datetimestring = list(finaltrack_datetimestring[i][0:maxtracklength][:] for i in cloudindexpresent)
    finaltrack_cloudidfile = finaltrack_cloudidfile[cloudindexpresent, 0:maxtracklength, :]
    #finaltrack_corecold_majoraxis = finaltrack_corecold_majoraxis[cloudindexpresent, 0:maxtracklength]
    #finaltrack_corecold_orientation = finaltrack_corecold_orientation[cloudindexpresent, 0:maxtracklength] 
    #finaltrack_corecold_eccentricity = finaltrack_corecold_eccentricity[cloudindexpresent, 0:maxtracklength]
    #finaltrack_corecold_perimeter = finaltrack_corecold_perimeter[cloudindexpresent, 0:maxtracklength]
    #finaltrack_corecold_xcenter = finaltrack_corecold_xcenter[cloudindexpresent, 0:maxtracklength]
    #finaltrack_corecold_ycenter = finaltrack_corecold_ycenter[cloudindexpresent, 0:maxtracklength]
    #finaltrack_corecold_xweightedcenter = finaltrack_corecold_xweightedcenter[cloudindexpresent, 0:maxtracklength]
    #finaltrack_corecold_yweightedcenter = finaltrack_corecold_yweightedcenter[cloudindexpresent, 0:maxtracklength]
    finaltrack_cloudtype_low =  finaltrack_cloudtype_low[cloudindexpresent, 0:maxtracklength]
    finaltrack_cloudtype_conglow =  finaltrack_cloudtype_conglow[cloudindexpresent, 0:maxtracklength]
    finaltrack_cloudtype_conghigh =  finaltrack_cloudtype_conghigh[cloudindexpresent, 0:maxtracklength]
    print('finaltrack_cloudtype_conghigh.shape: ', finaltrack_cloudtype_conghigh.shape)
    finaltrack_cloudtype_deep =  finaltrack_cloudtype_deep[cloudindexpresent, 0:maxtracklength]

    gc.collect()

    ########################################################
    # Correct merger and split cloud numbers

    # Initialize adjusted matrices
    adjusted_finaltrack_corecold_mergenumber = np.ones(np.shape(finaltrack_corecold_mergenumber))*-9999
    adjusted_finaltrack_corecold_splitnumber = np.ones(np.shape(finaltrack_corecold_mergenumber))*-9999
    print(('total tracks: ' + str(numtracks)))
    print('Correcting mergers and splits')
    print((time.ctime()))

    # Create adjustor
    indexcloudnumber = np.copy(cloudindexpresent) + 1
    adjustor = np.arange(0, np.max(cloudindexpresent)+2)
    for it in range(0, numtracks):
        adjustor[indexcloudnumber[it]] = it+1
    adjustor = np.append(adjustor, -9999)

    # Adjust mergers
    temp_finaltrack_corecold_mergenumber = finaltrack_corecold_mergenumber.astype(int).ravel()
    temp_finaltrack_corecold_mergenumber[temp_finaltrack_corecold_mergenumber == -9999] = np.max(cloudindexpresent)+2
    adjusted_finaltrack_corecold_mergenumber = adjustor[temp_finaltrack_corecold_mergenumber]
    adjusted_finaltrack_corecold_mergenumber = np.reshape(adjusted_finaltrack_corecold_mergenumber, np.shape(finaltrack_corecold_mergenumber))

    # Adjust splitters
    temp_finaltrack_corecold_splitnumber = finaltrack_corecold_splitnumber.astype(int).ravel()
    temp_finaltrack_corecold_splitnumber[temp_finaltrack_corecold_splitnumber == -9999] = np.max(cloudindexpresent)+2
    adjusted_finaltrack_corecold_splitnumber = adjustor[temp_finaltrack_corecold_splitnumber]
    adjusted_finaltrack_corecold_splitnumber = np.reshape(adjusted_finaltrack_corecold_splitnumber, np.shape(finaltrack_corecold_splitnumber))

            
    #########################################################################
    # Record starting and ending status
    print('Determine starting and ending status')
    print((time.ctime()))

    # Starting status
    finaltrack_corecold_startstatus = finaltrack_corecold_status[:,0]

    # Ending status
    finaltrack_corecold_endstatus = np.ones(len(finaltrack_corecold_startstatus))*-9999
    for trackstep in range(0,numtracks):
        if finaltrack_tracklength[trackstep] > 0:
            finaltrack_corecold_endstatus[trackstep] = finaltrack_corecold_status[trackstep,finaltrack_tracklength[trackstep] - 1]

    #######################################################################
    # Write to netcdf
    print('Writing trackstat netcdf')
    print((time.ctime()))
    print(trackstats_outfile)
    print('')
    
    # Check if file already exists. If exists, delete
    if os.path.isfile(trackstats_outfile):
        os.remove(trackstats_outfile) 
        
    import netcdf_io as net 
    net.write_trackstats_ct(trackstats_outfile, numtracks, maxtracklength, numcharfilename, \
                            datasource, datadescription, startdate, enddate, \
                            track_version, tracknumbers_version, timegap, \
                            pixel_radius, geolimits, areathresh, \
                            basetime_units, basetime_calendar, \
                            finaltrack_tracklength, finaltrack_basetime, finaltrack_cloudidfile, finaltrack_datetimestring, \
                            finaltrack_corecold_meanlat, finaltrack_corecold_meanlon, \
                            finaltrack_corecold_minlat, finaltrack_corecold_minlon, \
                            finaltrack_corecold_maxlat, finaltrack_corecold_maxlon, \
                            finaltrack_corecold_radius, \
                            finaltrack_ncorecoldpix, finaltrack_ncorepix, finaltrack_ncoldpix, \
                            finaltrack_corecold_cloudnumber, finaltrack_corecold_status, \
                            finaltrack_corecold_startstatus, finaltrack_corecold_endstatus, \
                            adjusted_finaltrack_corecold_mergenumber, adjusted_finaltrack_corecold_splitnumber, \
                            finaltrack_corecold_trackinterruptions, finaltrack_corecold_boundary, \
                            finaltrack_cloudtype_low, finaltrack_cloudtype_conglow, finaltrack_cloudtype_conghigh, finaltrack_cloudtype_deep)
                            #finaltrack_corecold_majoraxis, finaltrack_corecold_orientation, finaltrack_corecold_eccentricity, \
                            #finaltrack_corecold_perimeter, finaltrack_corecold_xcenter, finaltrack_corecold_ycenter, \
                            #finaltrack_corecold_xweightedcenter, finaltrack_corecold_yweightedcenter)
