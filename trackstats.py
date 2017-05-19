import numpy as np
from netCDF4 import Dataset, chartostring, stringtochar
import os, fnmatch
import sys
import matplotlib.pyplot as plt
from math import pi
from skimage.measure import regionprops
from matplotlib.patches import Ellipse
import time

# Define function that calculates track statistics for satellite data
def trackstats_sat(datasource, datadescription, pixel_radius, latlon_file, geolimits, areathresh, cloudtb_threshs, absolutetb_threshs, startdate, enddate, cloudid_filebase, tracking_inpath, stats_path, track_version, tracknumbers_version, tracknumbers_filebase, lengthrange=[2,120], landsea=0):

    # Purpose: Final step is to renumber the track numbers, which must be done since the previous step removed short tracks, and calculate statistics about the tracks. This gets statistics that can only be calculaed from the satellite, pixel-level data. Any statistic that is a function of these pixel-level statistics should be calculated in a separate routine.

    # Author: Orginial IDL version written by Sally A. McFarline (sally.mcfarlane@pnnl.gov) and modified for Zhe Feng (she.feng@pnnl.gov). Python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov)

    # Inputs:
    # datasource - source of the data
    # datadescription - description of data source, included in all output file names
    # pixel_radius - radius of pixels in km
    # geolimits - 4-element array with plotting boundaries [lat_min, lon_min, lat_max, lon_max]
    # tb_threshs - brightness temperature thresholds 
    # startdate - data to start processing in YYYYMMDD format
    # enddate - data to stop processing in YYYYMMDD format
    # latlon_file - filename of the file that contains the latitude and longitude dat
    # path - location of track data and location where stats will be saved
    # track_version - Version of track single cloud files
    # tracknumbers_version - Verison of the complete track files
    # lengthrange - Optional. Set this keyword to a vector [minlength,maxlength] to specify the lifetime range for the tracks.

    ###################################################################################
    # Set constants

    # Isolate core and cold anvil brightness temperature thresholds
    thresh_core = cloudtb_threshs[0]
    thresh_cold = cloudtb_threshs[1]

    # Set output filename
    trackstats_outfile = stats_path + 'stats_' + tracknumbers_filebase + '_' + startdate + '_' + enddate + '.nc'

    ###################################################################################
    # Load latitude and longitude grid. These were created in subroutine_idclouds and is saved in each file.

    # Find filenames of idcloud files
    temp_cloudidfiles = fnmatch.filter(os.listdir(tracking_inpath), cloudid_filebase +'*')

    # Select one file. Any file is fine since they all havel the map of latitued and longitude saved.
    temp_cloudidfiles = temp_cloudidfiles[0]

    # Load latitude and longitude grid
    latlondata = Dataset(tracking_inpath + temp_cloudidfiles, 'r')
    lon = latlondata.variables['longitude'][:]
    lat = latlondata.variables['latitude'][:]
    latlondata.close()

    # Load landmask. Optional setting. Often used with model data.
    if landsea == 1:
        landmassfile = Dataset(latlon_file, 'r')
        landmask = landmassfile.variables['landmask_sat'][:]

        landlocation = np.array(np.where(landmask > 0))
        if np.shape(landlocation)[1] > 0:
            landmask[landlocation[0,:], landlocation[1,:]] = 1
        landmask.astype(int)

    #############################################################################
    # Load track data
    cloudtrackdata = Dataset(stats_path + tracknumbers_filebase + '_' + startdate + '_' + enddate + '.nc', 'r')
    numtracks = cloudtrackdata.variables['ntracks'][:]
    cloudidfiles = cloudtrackdata.variables['cloudid_files'][:]
    tracknumbers = cloudtrackdata.variables['track_numbers'][:]
    trackstatus = cloudtrackdata.variables['track_status'][:]
    trackmergenumber = cloudtrackdata.variables['track_mergenumbers'][:]
    tracksplitnumber = cloudtrackdata.variables['track_splitnumbers'][:]
    trackstartend = cloudtrackdata.variables['track_reset'][:]
    timegap = cloudtrackdata.timegap
    cloudtrackdata.close()

    # Convert filenames and timegap to string
    numcharfilename = np.array(np.shape(cloudidfiles))[1]
    numcharfilename = numcharfilename.astype(int)
    cloudidfiles = chartostring(cloudidfiles)
    timegap = timegap[0:3]

    # Determine dimensions of data
    nfiles = len(cloudidfiles)
    ny, nx = np.shape(lat)
    print(nfiles)

    ############################################################################
    # Initialize grids
    nmaxclouds = max(lengthrange)

    mintb_thresh = absolutetb_threshs[0]
    maxtb_thresh = absolutetb_threshs[1]
    tbinterval = 2
    tbbins = np.arange(mintb_thresh,maxtb_thresh+tbinterval,tbinterval)
    nbintb = len(tbbins)

    fillvalue = -9999

    temp_finaltrack_tracklength = np.ones(numtracks, dtype=int)*fillvalue
    temp_finaltrack_corecold_boundary = np.ones(numtracks, dtype=int)*fillvalue
    temp_finaltrack_basetime = np.ones((numtracks,nmaxclouds), dtype=int)*fillvalue
    temp_finaltrack_corecold_mintb = np.ones((numtracks,nmaxclouds), dtype=float)*fillvalue
    temp_finaltrack_corecold_meantb = np.ones((numtracks,nmaxclouds), dtype=float)*fillvalue
    temp_finaltrack_core_meantb = np.ones((numtracks,nmaxclouds), dtype=float)*fillvalue
    temp_finaltrack_corecold_histtb = np.zeros((numtracks,nmaxclouds, nbintb-1))
    temp_finaltrack_corecold_radius = np.ones((numtracks,nmaxclouds), dtype=float)*fillvalue
    temp_finaltrack_corecoldwarm_radius = np.ones((numtracks,nmaxclouds), dtype=float)*fillvalue
    temp_finaltrack_corecold_meanlat = np.ones((numtracks,nmaxclouds), dtype=float)*fillvalue
    temp_finaltrack_corecold_meanlon = np.ones((numtracks,nmaxclouds), dtype=float)*fillvalue
    temp_finaltrack_corecold_maxlon = np.ones((numtracks,nmaxclouds), dtype=float)*fillvalue
    temp_finaltrack_corecold_maxlat = np.ones((numtracks,nmaxclouds), dtype=float)*fillvalue
    temp_finaltrack_ncorecoldpix = np.ones((numtracks,nmaxclouds), dtype=int)*fillvalue
    temp_finaltrack_corecold_minlon = np.ones((numtracks,nmaxclouds), dtype=float)*fillvalue
    temp_finaltrack_corecold_minlat = np.ones((numtracks,nmaxclouds), dtype=float)*fillvalue
    temp_finaltrack_ncorepix = np.ones((numtracks,nmaxclouds), dtype=int)*fillvalue
    temp_finaltrack_ncoldpix = np.ones((numtracks,nmaxclouds), dtype=int)*fillvalue
    temp_finaltrack_nwarmpix = np.ones((numtracks,nmaxclouds), dtype=int)*fillvalue
    temp_finaltrack_corecold_status = np.ones((numtracks,nmaxclouds), dtype=float)*fillvalue
    temp_finaltrack_corecold_trackinterruptions = np.ones(numtracks, dtype=int)*fillvalue
    temp_finaltrack_corecold_mergenumber = np.ones((numtracks,nmaxclouds), dtype=float)*fillvalue
    temp_finaltrack_corecold_splitnumber = np.ones((numtracks,nmaxclouds), dtype=float)*fillvalue
    temp_finaltrack_corecold_cloudnumber = np.ones((numtracks,nmaxclouds), dtype=float)*fillvalue
    temp_finaltrack_datetimestring = np.ones((numtracks,nmaxclouds,13), dtype=str)
    temp_finaltrack_cloudidfile = np.ones((numtracks,nmaxclouds,numcharfilename), dtype=str)
    temp_finaltrack_corecold_majoraxis = np.ones((numtracks,nmaxclouds), dtype=float)*fillvalue
    temp_finaltrack_corecold_orientation = np.ones((numtracks,nmaxclouds), dtype=float)*fillvalue 
    temp_finaltrack_corecold_eccentricity = np.ones((numtracks,nmaxclouds), dtype=float)*fillvalue
    temp_finaltrack_corecold_perimeter = np.ones((numtracks,nmaxclouds), dtype=float)*fillvalue
    temp_finaltrack_corecold_xcenter = np.ones((numtracks,nmaxclouds), dtype=float)*fillvalue
    temp_finaltrack_corecold_ycenter = np.ones((numtracks,nmaxclouds), dtype=float)*fillvalue
    temp_finaltrack_corecold_xweightedcenter = np.ones((numtracks,nmaxclouds), dtype=float)*fillvalue
    temp_finaltrack_corecold_yweightedcenter = np.ones((numtracks,nmaxclouds), dtype=float)*fillvalue
    if landsea == 1:
        temp_finaltrack_corecold_landfrac = np.ones((numtracks,nmaxclouds), dtype=float)*fillvalue
        temp_finaltrack_core_landfrac = np.ones((numtracks,nmaxclouds), dtype=float)*fillvalue

    #########################################################################################
    # loop over files. Calculate statistics and organize matrices by tracknumber and cloud
    for nf in range(0,nfiles):
        file_tracknumbers = tracknumbers[0,nf,:]

        # Only process file if that file contains a track
        if np.nanmax(file_tracknumbers) > 0:
            print(cloudidfiles[nf])

            # Load cloudid file
            file_cloudiddata = Dataset(tracking_inpath + cloudidfiles[nf], 'r')
            file_basetime = file_cloudiddata.variables['basetime'][:]
            file_datestring = chartostring(file_cloudiddata.variables['filedate'][:])
            file_timestring = chartostring(file_cloudiddata.variables['filetime'][:])
            file_tb = file_cloudiddata.variables['tb'][:]
            file_cloudtype = file_cloudiddata.variables['cloudtype'][:]
            file_all_cloudnumber = file_cloudiddata.variables['cloudnumber'][:]
            file_corecold_cloudnumber = file_cloudiddata.variables['convcold_cloudnumber'][:]
            latitude = file_cloudiddata.variables['latitude'][:]
            longitude = file_cloudiddata.variables['longitude'][:]
            file_cloudiddata.close()

            file_datetimestring = file_datestring[0] + '_' + file_timestring[0]

            # Find unique track numbers
            uniquetracknumbers = np.unique(file_tracknumbers)
            uniquetracknumbers = uniquetracknumbers[uniquetracknumbers > 0]

            # Loop over unique tracknumbers
            for itrack in uniquetracknumbers:

                # Find cloud number that belongs to the current track in this file
                cloudnumber = np.array(np.where(file_tracknumbers == itrack)) + 1 # Finds cloud numbers associated with that track. Need to add one since tells index, which starts at 0, and we want the number, which starts at one
                cloudindex = cloudnumber - 1 # Index within the matrice of this cloud. 

                if len(cloudnumber) == 1: # Should only be one cloud number. In mergers and split, the associated clouds should be listed in the file_splittracknumbers and file_mergetracknumbers
                    # Find cloud in cloudid file associated with this track
                    corearea = np.array(np.where((file_corecold_cloudnumber[0,:,:] == cloudnumber) & (file_cloudtype[0,:,:] == 1)))
                    ncorepix = np.shape(corearea)[1]

                    coldarea = np.array(np.where((file_corecold_cloudnumber[0,:,:] == cloudnumber) & (file_cloudtype[0,:,:] == 2)))
                    ncoldpix = np.shape(coldarea)[1]

                    warmarea = np.array(np.where((file_all_cloudnumber[0,:,:] == cloudnumber) & (file_cloudtype[0,:,:] == 3)))
                    nwarmpix = np.shape(warmarea)[1]

                    corecoldarea = np.array(np.where((file_corecold_cloudnumber[0,:,:] == cloudnumber) & (file_cloudtype[0,:,:] >= 1) & (file_cloudtype[0,:,:] <= 2)))
                    ncorecoldpix = np.shape(corecoldarea)[1]

                    # Find current length of the track. Use for indexing purposes. Also, record the current length the given track.
                    lengthindex = np.array(np.where(temp_finaltrack_corecold_cloudnumber[itrack-1,:] > 0))
                    if np.shape(lengthindex)[1] > 0:
                        nc = np.nanmax(lengthindex) + 1
                    else:
                        nc = 0
                    temp_finaltrack_tracklength[itrack-1] = nc+1 # Need to add one since array index starts at 0

                    if nc < nmaxclouds:
                        # Save information that links this cloud back to its raw pixel level data
                        temp_finaltrack_basetime[itrack-1,nc] = file_basetime
                        temp_finaltrack_corecold_cloudnumber[itrack-1,nc] = cloudnumber
                        temp_finaltrack_cloudidfile[itrack-1,nc,:] = stringtochar(np.array(cloudidfiles[nf]))
                        temp_finaltrack_datetimestring[itrack-1,nc,:] = stringtochar(np.array(file_datetimestring))

                        ###############################################################
                        # Calculate statistics about this cloud system

                        #############
                        # Location statistics of core+cold anvil (aka the convective system)
                        corecoldlat = latitude[corecoldarea[0], corecoldarea[1]]
                        corecoldlon = longitude[corecoldarea[0], corecoldarea[1]]

                        temp_finaltrack_corecold_meanlat[itrack-1,nc] = np.nanmean(corecoldlat)
                        temp_finaltrack_corecold_meanlon[itrack-1,nc] = np.nanmean(corecoldlon)

                        temp_finaltrack_corecold_minlat[itrack-1,nc] = np.nanmin(corecoldlat)
                        temp_finaltrack_corecold_minlon[itrack-1,nc] = np.nanmin(corecoldlon)

                        temp_finaltrack_corecold_maxlat[itrack-1,nc] = np.nanmax(corecoldlat)
                        temp_finaltrack_corecold_maxlon[itrack-1,nc] = np.nanmax(corecoldlon)

                        # Determine if core+cold touches of the boundaries of the domain
                        if np.absolute(temp_finaltrack_corecold_minlat[itrack-1,nc]-geolimits[0]) < 0.1 or np.absolute(temp_finaltrack_corecold_maxlat[itrack-1,nc]-geolimits[2]) < 0.1 or np.absolute(temp_finaltrack_corecold_minlon[itrack-1,nc]-geolimits[1]) < 0.1 or np.absolute(temp_finaltrack_corecold_maxlon[itrack-1,nc]-geolimits[3]) < 0.1:
                            temp_finaltrack_corecold_boundary[itrack-1] = 1

                        ############
                        # Save number of pixels (metric for size)
                        temp_finaltrack_ncorecoldpix[itrack-1,nc] = ncorecoldpix
                        temp_finaltrack_ncorepix[itrack-1,nc] = ncorepix
                        temp_finaltrack_ncoldpix[itrack-1,nc] = ncoldpix
                        temp_finaltrack_nwarmpix[itrack-1,nc] = nwarmpix

                        #############
                        # Calculate physical characteristics associated with cloud system

                        # Create a padded region around the cloud.
                        pad = 5

                        if np.nanmin(corecoldarea[0]) - pad > 0:
                            minyindex = np.nanmin(corecoldarea[0]) - pad
                        else:
                            minyindex = 0

                        if np.nanmax(corecoldarea[0]) + pad < ny - 1:
                            maxyindex = np.nanmax(corecoldarea[0]) + pad + 1
                        else:
                            maxyindex = nx

                        if np.nanmin(corecoldarea[1]) - pad > 0:
                            minxindex = np.nanmin(corecoldarea[1]) - pad
                        else:
                            minxindex = 0

                        if np.nanmax(corecoldarea[1]) + pad < nx - 1:
                            maxxindex = np.nanmax(corecoldarea[1]) + pad + 1
                        else:
                            maxxindex = nx

                        # Isolate the region around the cloud using the padded region
                        isolatedcloudnumber = np.copy(file_corecold_cloudnumber[0, minyindex:maxyindex, minxindex:maxxindex])
                        isolatedtb = np.copy(file_tb[0, minyindex:maxyindex, minxindex:maxxindex])

                        # Remove brightness temperatures outside core + cold anvil
                        isolatedtb[isolatedcloudnumber != cloudnumber] = 0

                        # Turn cloud map to binary
                        isolatedcloudnumber[isolatedcloudnumber != cloudnumber] = 0
                        isolatedcloudnumber[isolatedcloudnumber == cloudnumber] = 1

                        # Calculate major axis, orientation, eccentricity
                        cloudproperities = regionprops(isolatedcloudnumber, intensity_image=isolatedtb)
                    
                        temp_finaltrack_corecold_eccentricity[itrack-1,nc] = cloudproperities[0].eccentricity
                        temp_finaltrack_corecold_majoraxis[itrack-1,nc] = cloudproperities[0].major_axis_length
                        temp_finaltrack_corecold_orientation[itrack-1,nc] = (cloudproperities[0].orientation)*(180/float(pi))
                        temp_finaltrack_corecold_perimeter[itrack-1,nc] = cloudproperities[0].perimeter*pixel_radius
                        [temp_ycenter, temp_xcenter] = cloudproperities[0].centroid
                        [temp_finaltrack_corecold_ycenter[itrack-1,nc], temp_finaltrack_corecold_xcenter[itrack-1,nc]] = np.add([temp_ycenter,temp_xcenter], [minyindex, minxindex]).astype(int)
                        [temp_yweightedcenter, temp_xweightedcenter] = cloudproperities[0].weighted_centroid
                        [temp_finaltrack_corecold_yweightedcenter[itrack-1,nc], temp_finaltrack_corecold_xweightedcenter[itrack-1,nc]] = np.add([temp_yweightedcenter, temp_xweightedcenter], [minyindex, minxindex]).astype(int)

                        #minoraxislength = cloudproperities[0].minor_axis_length
                        #[ycenter,xcenter] = cloudproperities[0].centroid
                        #ellipse = Ellipse(xy=(xcenter,ycenter), width=minoraxislength, height=majoraxislength, angle=90-orientation, facecolor='None', linewidth=3, edgecolor='y')
                        #fig, ax = plt.subplots(1,2)
                        #ax[0].pcolor(isolatedcloudnumber)
                        #ax[0].scatter(xcenter,ycenter, s=30, color='y')
                        #ax[0].add_patch(ellipse)
                        #plt.show()
                        #tbplot = np.ma.masked_invalid(np.atleast_2d(isolatedtb))
                        #plt.figure()
                        #plt.pcolor(tbplot)
                        #plt.scatter(temp_xcenter, temp_ycenter, marker='o', s=80, color='k')
                        #plt.scatter(temp_xweightedcenter, temp_yweightedcenter, marker='*', s=80, color='r')
                        #tbplot = np.ma.masked_invalid(np.atleast_2d(file_tb[0,:,:]))
                        #plt.figure()
                        #plt.pcolor(tbplot)
                        #plt.scatter(temp_finaltrack_corecold_xcenter[itrack-1,nc], temp_finaltrack_corecold_ycenter[itrack-1,nc], marker='o', s=80, color='k')
                        #plt.scatter(temp_finaltrack_corecold_xweightedcenter[itrack-1,nc], temp_finaltrack_corecold_yweightedcenter[itrack-1,nc], marker='*', s=80, color='r')
                        #plt.show()

                        # Determine equivalent radius of core+cold. Assuming circular area = (number pixels)*(pixel radius)^2, equivalent radius = sqrt(Area / pi)
                        temp_finaltrack_corecold_radius[itrack-1,nc] = np.sqrt(np.divide(ncorecoldpix*(np.square(pixel_radius)), pi))

                        # Determine equivalent radius of core+cold+warm. Assuming circular area = (number pixels)*(pixel radius)^2, equivalent radius = sqrt(Area / pi)
                        temp_finaltrack_corecoldwarm_radius[itrack-1,nc] = np.sqrt(np.divide((ncorepix + ncoldpix + nwarmpix)*(np.square(pixel_radius)), pi))

                        ##############################################################
                        # Calculate brightness temperature statistics of core+cold anvil
                        corecoldtb = np.copy(file_tb[0,corecoldarea[0], corecoldarea[1]])

                        temp_finaltrack_corecold_mintb[itrack-1,nc] = np.nanmin(corecoldtb)
                        temp_finaltrack_corecold_meantb[itrack-1,nc] = np.nanmean(corecoldtb)

                        ################################################################
                        # Histogram of brightness temperature for core+cold anvil
                        temp_finaltrack_corecold_histtb[itrack-1,nc,:], usedtbbins = np.histogram(corecoldtb, range=(mintb_thresh, maxtb_thresh), bins=tbbins)

                        # Save track information. Need to subtract one since cloudnumber gives the number of the cloud (which starts at one), but we are looking for its index (which starts at zero)
                        temp_finaltrack_corecold_status[itrack-1,nc] = np.copy(trackstatus[0,nf,cloudindex])

                        temp_finaltrack_corecold_mergenumber[itrack-1,nc] = np.copy(trackmergenumber[0,nf,cloudindex])
                        temp_finaltrack_corecold_splitnumber[itrack-1,nc] = np.copy(tracksplitnumber[0,nf,cloudindex])

                        if trackstartend[0,nf,cloudindex] > temp_finaltrack_corecold_trackinterruptions[itrack-1]:
                            temp_finaltrack_corecold_trackinterruptions[itrack-1] = trackstartend[0,nf,cloudindex]

                        ####################################################################
                        # Calculate mean brightness temperature for core
                        coretb = np.copy(file_tb[0, coldarea[0], coldarea[1]])

                        temp_finaltrack_core_meantb[itrack-1,nc] = np.nanmean(coretb)

                        #############
                        # Count number of pixels over land, if applicable
                        if landsea == 1:
                            corecoldland = np.array(np.where(landmask[corecoldarea[0], corecolodarea[1]]))
                            ncorecoldland = np.shape(corecoldland)[1]
                            temp_finaltrack_corecold_landfrac[itrack-1,nc] = np.divide(ncorecoldland, ncorecold)

                            coreland = np.array(np.where(landmask[corearea[0], corearea[1]]))
                            ncoreland = np.shape(coreland)[1]
                            temp_finaltrack_core_landfrac[itrack-1,nc] = np.divide(ncoreland, ncore)
                    
                    else:
                        sys.exit(str(nc) + ' greater than maximum allowed number clouds')

                elif len(cloudnumber) > 1:
                    sys.exit(str(cloudnumbers) + ' clouds linked to one track. Each track should only be linked to one cloud in each file in the track_number array. The track_number variable only tracks the largest cell in mergers and splits. The small clouds in tracks and mergers should only be listed in the track_splitnumbers and track_mergenumbers arrays.')

    ##############################################################
    # Remove tracks that have no cells. These tracks are short.
    cloudindexpresent = np.array(np.where(temp_finaltrack_tracklength != fillvalue))[0,:]
    numtracks = len(cloudindexpresent)

    finaltrack_tracklength = temp_finaltrack_tracklength[cloudindexpresent]
    finaltrack_corecold_boundary = temp_finaltrack_corecold_boundary[cloudindexpresent]
    finaltrack_basetime = temp_finaltrack_basetime[cloudindexpresent,:]
    finaltrack_corecold_mintb = temp_finaltrack_corecold_mintb[cloudindexpresent,:]
    finaltrack_corecold_meantb = temp_finaltrack_corecold_meantb[cloudindexpresent,:]
    finaltrack_core_meantb = temp_finaltrack_corecold_meantb[cloudindexpresent,:]
    finaltrack_corecold_histtb = temp_finaltrack_corecold_histtb[cloudindexpresent,:,:]
    finaltrack_corecold_radius = temp_finaltrack_corecold_radius[cloudindexpresent,:]
    finaltrack_corecoldwarm_radius = temp_finaltrack_corecoldwarm_radius[cloudindexpresent,:]
    finaltrack_corecold_meanlat = temp_finaltrack_corecold_meanlat[cloudindexpresent,:]
    finaltrack_corecold_meanlon = temp_finaltrack_corecold_meanlon[cloudindexpresent,:]
    finaltrack_corecold_maxlon = temp_finaltrack_corecold_maxlon[cloudindexpresent,:]
    finaltrack_corecold_maxlat = temp_finaltrack_corecold_maxlat[cloudindexpresent,:]
    finaltrack_corecold_minlon = temp_finaltrack_corecold_minlon[cloudindexpresent,:]
    finaltrack_corecold_minlat = temp_finaltrack_corecold_minlat[cloudindexpresent,:]
    finaltrack_ncorecoldpix = temp_finaltrack_ncorecoldpix[cloudindexpresent,:]
    finaltrack_ncorepix = temp_finaltrack_ncorepix[cloudindexpresent,:]
    finaltrack_ncoldpix = temp_finaltrack_ncoldpix[cloudindexpresent,:]
    finaltrack_nwarmpix = temp_finaltrack_nwarmpix[cloudindexpresent,:]
    finaltrack_corecold_status = temp_finaltrack_corecold_status[cloudindexpresent,:]
    finaltrack_corecold_trackinterruptions = temp_finaltrack_corecold_trackinterruptions[cloudindexpresent]
    finaltrack_corecold_mergenumber = temp_finaltrack_corecold_mergenumber[cloudindexpresent,:]
    finaltrack_corecold_splitnumber = temp_finaltrack_corecold_splitnumber[cloudindexpresent,:]
    finaltrack_corecold_cloudnumber = temp_finaltrack_corecold_cloudnumber[cloudindexpresent,:]
    finaltrack_datetimestring = temp_finaltrack_datetimestring[cloudindexpresent,:]
    finaltrack_cloudidfile = temp_finaltrack_cloudidfile[cloudindexpresent,:]
    finaltrack_corecold_majoraxis = temp_finaltrack_corecold_majoraxis[cloudindexpresent,:]
    finaltrack_corecold_orientation = temp_finaltrack_corecold_orientation[cloudindexpresent,:] 
    finaltrack_corecold_eccentricity = temp_finaltrack_corecold_eccentricity[cloudindexpresent,:]
    finaltrack_corecold_perimeter = temp_finaltrack_corecold_perimeter[cloudindexpresent,:]
    finaltrack_corecold_xcenter = temp_finaltrack_corecold_xcenter[cloudindexpresent,:]
    finaltrack_corecold_ycenter = temp_finaltrack_corecold_ycenter[cloudindexpresent,:]
    finaltrack_corecold_xweightedcenter = temp_finaltrack_corecold_xweightedcenter[cloudindexpresent,:]
    finaltrack_corecold_yweightedcenter = temp_finaltrack_corecold_yweightedcenter[cloudindexpresent,:]
    if landsea == 1:
        finaltrack_corecold_landfrac = temp_finaltrack_corecold_landfrac[cloudindexpresent,:]
        finaltrack_core_landfrac = temp_finaltrack_core_landfrac[cloudindexpresent,:]

    #########################################################################
    # Record starting and ending status

    # Starting status
    finaltrack_corecold_startstatus = finaltrack_corecold_status[:,0]

    # Ending status
    finaltrack_corecold_endstatus = np.ones(len(finaltrack_corecold_startstatus))*fillvalue
    for trackstep in range(0,numtracks):
        finaltrack_corecold_endstatus[trackstep] = finaltrack_corecold_status[trackstep,finaltrack_tracklength[trackstep] - 1]

    #for itrack in range(0,numtracks):
    #    print(finaltrack_corecold_status[itrack,0:15])
    #    print(finaltrack_corecold_startstatus[itrack])
    #    print(finaltrack_corecold_endstatus[itrack])
    #    print(finaltrack_corecold_trackinterruptions[itrack])
    #    print(finaltrack_corecold_mergenumber[itrack,0:15])
    #    print(finaltrack_corecold_splitnumber[itrack,0:15])
    #    raw_input('Waiting for User')

    #######################################################################
    # Write to netcdf

    # create file
    filesave = Dataset(trackstats_outfile, 'w', format='NETCDF4_CLASSIC')

    # set global attributes
    filesave.Convenctions = 'CF-1.6'
    filesave.title = 'File containing statistics for each track'
    filesave.institution = 'Pacific Northwest National Laboratory'
    filesave.setncattr('Contact', 'Hannah C Barnes: hannah.barnes@pnnl.gov')
    filesave.history = 'Created ' + time.ctime(time.time())
    filesave.setncattr('source', datasource)
    filesave.setncattr('description', datadescription)
    filesave.setncattr('startdate', startdate)
    filesave.setncattr('enddate', enddate)
    filesave.setncattr('track_version', track_version)
    filesave.setncattr('tracknumbers_version', tracknumbers_version)
    filesave.setncattr('timegap', str(timegap)+'-hr')
    filesave.setncattr('tb_core', thresh_core)
    filesave.setncattr('tb_coldavil', thresh_cold)
    filesave.setncattr('pixel_radisu_km', pixel_radius)

    # create netcdf dimensions
    filesave.createDimension('ntracks', None)
    filesave.createDimension('nmaxlength', nmaxclouds)
    filesave.createDimension('nbins', nbintb-1)
    filesave.createDimension('ndatetimechars', 13)
    filesave.createDimension('nfilechars', numcharfilename)

    # define variable
    lifetime = filesave.createVariable('lifetime', 'i4', 'ntracks', zlib=True, fill_value=fillvalue)
    lifetime.long_name = 'track duration'
    lifetime.description = 'Lifetime of each tracked system'
    lifetime.units = 'temporal resolution of original data'
    lifetime.fill_value = fillvalue
    lifetime.min_value = 2
    lifetime.max_value = nmaxclouds

    basetime = filesave.createVariable('basetime', 'i4', ('ntracks', 'nmaxlength'), zlib=True, fill_value=fillvalue)
    basetime.long_name = 'epoch time'
    basetime.description = 'basetime of cloud at the given time'
    basetime.units = 'seconds since 01/01/1970 00:00'
    basetime.fill_value = fillvalue
    basetime.standard_name = 'time'

    cloudidfiles = filesave.createVariable('cloudidfiles', 'S1', ('ntracks', 'nmaxlength', 'nfilechars'), zlib=True, complevel=5)
    cloudidfiles.long_name = 'file name'
    cloudidfiles.description = 'Name of the corresponding cloudid file for each cloud in each track'

    datetimestrings = filesave.createVariable('datetimestrings', 'S1', ('ntracks', 'nmaxlength', 'ndatetimechars'), zlib=True, complevel=5)
    datetimestrings.long_name = 'date-time'
    datetimestrings.description = 'date_time for for each cloud in each track'

    meanlat = filesave.createVariable('meanlat', 'f4', ('ntracks', 'nmaxlength'), zlib=True, complevel=5, fill_value=fillvalue)
    meanlat.standard_name = 'Latitude'
    meanlat.description = 'Mean latitude of the core + cold anvil at the given time'
    meanlat.fill_value = fillvalue
    meanlat.units = 'degrees'
    meanlat.min_value = geolimits[1]
    meanlat.max_value = geolimits[3]

    meanlon = filesave.createVariable('meanlon', 'f4', ('ntracks', 'nmaxlength'), zlib=True, complevel=5, fill_value=fillvalue)
    meanlon.standard_name = 'Lonitude'
    meanlon.description = 'Mean longitude of the core + cold anvil at the given time'
    meanlon.fill_value = fillvalue
    meanlon.units = 'degrees'
    meanlon.min_value = geolimits[0]
    meanlon.max_value = geolimits[2]

    minlat = filesave.createVariable('minlat', 'f4', ('ntracks', 'nmaxlength'), zlib=True, complevel=5, fill_value=fillvalue)
    minlat.standard_name = 'Latitude'
    minlat.description = 'Minimum latitude of the core + cold anvil at the given time'
    minlat.fill_value = fillvalue
    minlat.units = 'degrees'
    minlat.min_value = geolimits[1]
    minlat.max_value = geolimits[3]

    minlon = filesave.createVariable('minlon', 'f4', ('ntracks', 'nmaxlength'), zlib=True, complevel=5, fill_value=fillvalue)
    minlon.standard_name = 'Lonitude'
    minlon.description = 'Minimum longitude of the core + cold anvil at the given time'
    minlon.fill_value = fillvalue
    minlon.units = 'degrees'
    minlon.min_value = geolimits[0]
    minlon.max_value = geolimits[2]

    maxlat = filesave.createVariable('maxlat', 'f4', ('ntracks', 'nmaxlength'), zlib=True, complevel=5, fill_value=fillvalue)
    maxlat.standard_name = 'Latitude'
    maxlat.description = 'Maximum latitude of the core + cold anvil at the given time'
    maxlat.fill_value = fillvalue
    maxlat.units = 'degrees'
    maxlat.min_value = geolimits[1]
    maxlat.max_value = geolimits[3]

    maxlon = filesave.createVariable('maxlon', 'f4', ('ntracks', 'nmaxlength'), zlib=True, complevel=5, fill_value=fillvalue)
    maxlon.standard_name = 'Lonitude'
    maxlon.description = 'Maximum longitude of the core + cold anvil at the given time'
    maxlon.fill_value = fillvalue
    maxlon.units = 'degrees'
    maxlon.min_value = geolimits[0]
    maxlon.max_value = geolimits[2]

    radius = filesave.createVariable('radius', 'f4', ('ntracks', 'nmaxlength'), zlib=True, complevel=5, fill_value=fillvalue)
    radius.standard_name = 'Equivalent radius'
    radius.description = 'Equivalent radius of the core + cold anvil at the given time'
    radius.fill_value = fillvalue
    radius.units = 'km'
    radius.min_value = areathresh

    radius_warmanvil = filesave.createVariable('radius_warmanvil', 'f4', ('ntracks', 'nmaxlength'), zlib=True, complevel=5, fill_value=fillvalue)
    radius_warmanvil.standard_name = 'Equivalent radius'
    radius_warmanvil.description = 'Equivalent radius of the core + cold anvil + warm anvil at the given time'
    radius_warmanvil.fill_value = fillvalue
    radius_warmanvil.units = 'km'
    radius_warmanvil.min_value = areathresh

    npix = filesave.createVariable('npix', 'i4', ('ntracks', 'nmaxlength'), zlib=True, complevel=5, fill_value=fillvalue)
    npix.long_name = 'Number of pixels'
    npix.description = 'Number of pixels in the core + cold anvil at the given time'
    npix.fill_value = fillvalue
    npix.units = 'unitless'
    npix.min_value = int(areathresh/float(np.square(pixel_radius)))

    nconv = filesave.createVariable('nconv', 'i4', ('ntracks', 'nmaxlength'), zlib=True, complevel=5, fill_value=fillvalue)
    nconv.long_name = 'Number of pixels'
    nconv.description = 'Number of pixels in the core at the given time'
    nconv.fill_value = fillvalue
    nconv.units = 'unitless'
    nconv.min_value = int(areathresh/float(np.square(pixel_radius)))

    ncoldanvil = filesave.createVariable('ncoldanvil', 'i4', ('ntracks', 'nmaxlength'), zlib=True, complevel=5, fill_value=fillvalue)
    ncoldanvil.long_name = 'Number of pixels'
    ncoldanvil.description = 'Number of pixels in the cold anvil at the given time'
    ncoldanvil.fill_value = fillvalue
    ncoldanvil.units = 'unitless'
    ncoldanvil.min_value = int(areathresh/float(np.square(pixel_radius)))

    nwarmanvil = filesave.createVariable('nwarmanvil', 'i4', ('ntracks', 'nmaxlength'), zlib=True, complevel=5, fill_value=fillvalue)
    nwarmanvil.long_name = 'Number of pixels'
    nwarmanvil.description = 'Number of pixels in the warm anvil at the given time'
    nwarmanvil.fill_value = fillvalue
    nwarmanvil.units = 'unitless'
    nwarmanvil.min_value = int(areathresh/float(np.square(pixel_radius)))

    cloudnumber = filesave.createVariable('cloudnumber', 'i4', ('ntracks', 'nmaxlength'), zlib=True, complevel=5, fill_value=fillvalue)
    cloudnumber.description = 'Number of the cloud in the corresponding cloudid file'
    cloudnumber.usage = 'To link this tracking statistics file with corresponding pixel-level cloudid files, use the cloudidfile and cloudnumber together to identify which file and cloud this track is associated with at this time'
    cloudnumber.fill_value = fillvalue

    status = filesave.createVariable('status', 'i4', ('ntracks', 'nmaxlength'), zlib=True, complevel=5, fill_value=fillvalue)
    status.long_name = 'flag indicating the status of cloud'
    status.description = 'Numbers in each row describe how the clouds in that track evolve over time'
    status.values = '-9999=missing cloud or cloud removed due to short track, 0=track ends here, 1=cloud continues as one cloud in next file, 2=Biggest cloud in merger, 21=Smaller cloud(s) in merger, 13=Cloud that splits, 3=Biggest cloud from a split that stops after the split, 31=Smaller cloud(s) from a split that stop after the split. The last seven classifications are added together in different combinations to describe situations.'
    status.min_value = 0
    status.max_value = 52
    status.fill_value = fillvalue
    status.units = 'unitless'

    startstatus = filesave.createVariable('startstatus', 'i4', 'ntracks', zlib=True, complevel=5, fill_value=fillvalue)
    startstatus.long_name = 'flag indicating the status of the first cloud in track'
    startstatus.description = 'Numbers in each row describe the status of the first cloud in that track'
    startstatus.values = '-9999=missing cloud or cloud removed due to short track, 0=track ends here, 1=cloud continues as one cloud in next file, 2=Biggest cloud in merger, 21=Smaller cloud(s) in merger, 13=Cloud that splits, 3=Biggest cloud from a split that stops after the split, 31=Smaller cloud(s) from a split that stop after the split. The last seven classifications are added together in different combinations to describe situations.'
    startstatus.min_value = 0
    startstatus.max_value = 52
    startstatus.fill_value = fillvalue
    startstatus.units = 'unitless'

    trackinterruptions = filesave.createVariable('trackinterruptions', 'i4', 'ntracks', zlib=True, complevel=5, fill_value=fillvalue)
    trackinterruptions.long_name = 'flag indication if track interrupted'
    trackinterruptions.description = 'Numbers in each row indicate if the track started and ended naturally or if the start or end of the track was artifically cut short by data availability'
    trackinterruptions.values = '0 = full track available, good data. 1 = track starts at first file, track cut short by data availability. 2 = track ends at last file, track cut short by data availability'
    trackinterruptions.min_value = 0
    trackinterruptions.max_value = 2
    trackinterruptions.fill_value = fillvalue
    trackinterruptions.units = 'unitless'

    endstatus = filesave.createVariable('endstatus', 'i4', 'ntracks', zlib=True, complevel=5, fill_value=fillvalue)
    endstatus.long_name = 'flag indicating the status of the last cloud in track'
    endstatus.description = 'Numbers in each row describe the status of the last cloud in that track'
    endstatus.values = '-9999=missing cloud or cloud removed due to short track, 0=track ends here, 1=cloud continues as one cloud in next file, 2=Biggest cloud in merger, 21=Smaller cloud(s) in merger, 13=Cloud that splits, 3=Biggest cloud from a split that stops after the split, 31=Smaller cloud(s) from a split that stop after the split. The last seven classifications are added together in different combinations to describe situations.'
    endstatus.min_value = 0
    endstatus.max_value = 52
    endstatus.fill_value = fillvalue
    endstatus.units = 'unitless'

    mergenumbers = filesave.createVariable('mergenumbers', 'i4', ('ntracks', 'nmaxlength'), zlib=True, complevel=5, fill_value=fillvalue)
    mergenumbers.long_name = 'cloud track number'
    mergenumbers.description = 'Each row represents a cloudid file. Each column represets a cloud in that file. Numbers give the track number associated with the small clouds in mergers.'
    mergenumbers.fill_value = fillvalue
    mergenumbers.min_value = 1
    mergenumbers.max_value = numtracks
    mergenumbers.units = 'unitless'

    splitnumbers = filesave.createVariable('splitnumbers', 'i4', ('ntracks', 'nmaxlength'), zlib=True, complevel=5, fill_value=fillvalue)
    splitnumbers.long_name = 'cloud track number'
    splitnumbers.description = 'Each row represents a cloudid file. Each column represets a cloud in that file. Numbers give the track number associated with the small clouds in splits.'
    splitnumbers.fill_value = fillvalue
    splitnumbers.min_value = 1
    splitnumbers.max_value = numtracks
    splitnumbers.units = 'unitless'

    boundary = filesave.createVariable('boundary', 'i4', 'ntracks', zlib=True, complevel=5, fill_value=fillvalue)
    boundary.description = 'Flag indicating whether the core + cold anvil touches one of the domain edges. 0 = away from edge. 1= touches edge.'
    boundary.min_value = 0
    boundary.max_value = 1
    boundary.fill_value = fillvalue
    boundary.units = 'unitless'

    mintb = filesave.createVariable('mintb', 'f4', ('ntracks', 'nmaxlength'), zlib=True, complevel=5, fill_value=fillvalue)
    mintb.long_name = 'Minimum brightness temperature'
    mintb.standard_name = 'brightness temperature'
    mintb.description = 'Minimum brightness temperature of the core + cold anvil at the given time.'
    mintb.fill_value = fillvalue
    mintb.min_value = mintb_thresh
    mintb.max_value = maxtb_thresh
    mintb.units = 'K'

    meantb = filesave.createVariable('meantb', 'f4', ('ntracks', 'nmaxlength'), zlib=True, complevel=5, fill_value=fillvalue)
    meantb.long_name = 'Mean brightness temperature'
    meantb.standard_name = 'brightness temperature'
    meantb.description = 'Mean brightness temperature of the core + cold anvil at the given time.'
    meantb.fill_value = fillvalue
    meantb.mean_value = mintb_thresh
    meantb.max_value = maxtb_thresh
    meantb.units = 'K'

    meantb_conv = filesave.createVariable('meantb_conv', 'f4', ('ntracks', 'nmaxlength'), zlib=True, complevel=5, fill_value=fillvalue)
    meantb_conv.long_name = 'Mean brightness temperature'
    meantb_conv.standard_name = 'brightness temperature'
    meantb_conv.description = 'Mean brightness temperature of the core at the given time.'
    meantb_conv.fill_value = fillvalue
    meantb_conv.mean_value = mintb_thresh
    meantb_conv.max_value = maxtb_thresh
    meantb_conv.units = 'K'

    histtb = filesave.createVariable('histtb', 'i4', ('ntracks', 'nmaxlength', 'nbins'), zlib=True, complevel=5, fill_value=fillvalue)
    histtb.long_name = 'Histogram of brightness temperature'
    histtb.standard_name = 'brightness temperature'
    histtb.description = 'Histogram of brightess of the core + cold anvil at the given time.'
    histtb.fill_value = fillvalue
    histtb.hist_value = mintb_thresh
    histtb.max_value = maxtb_thresh
    histtb.units = 'K'

    orientation = filesave.createVariable('orientation', 'f4', ('ntracks', 'nmaxlength'), zlib=True, complevel=5, fill_value=fillvalue)
    orientation.description = 'Orientation of the major axis of the core + cold anvil at the given time'
    orientation.units = 'Degrees clockwise from vertical'
    orientation.min_value = 0
    orientation.max_value = 360
    orientation.fill_value = fillvalue

    eccentricity = filesave.createVariable('eccentricity', 'f4', ('ntracks', 'nmaxlength'), zlib=True, complevel=5, fill_value=fillvalue)
    eccentricity.description = 'Eccentricity of the major axis of the core + cold anvil at the given time'
    eccentricity.units = 'unitless'
    eccentricity.min_value = 0
    eccentricity.max_value = 1
    eccentricity.fill_value = fillvalue

    majoraxis = filesave.createVariable('majoraxis', 'f4', ('ntracks', 'nmaxlength'), zlib=True, complevel=5, fill_value=fillvalue)
    majoraxis.long_name = 'Major axis length'
    majoraxis.description = 'Length of the major axis of the core + cold anvil at the given time'
    majoraxis.units = 'km'
    majoraxis.fill_value = fillvalue

    perimeter = filesave.createVariable('perimeter', 'f4', ('ntracks', 'nmaxlength'), zlib=True, complevel=5, fill_value=fillvalue)
    perimeter.description = 'Approximnate circumference of the core + cold anvil at the given time'
    perimeter.units = 'km'
    perimeter.fill_value = fillvalue

    xcenter = filesave.createVariable('xcenter', 'f4', ('ntracks', 'nmaxlength'), zlib=True, complevel=5, fill_value=fillvalue)
    xcenter.long_name = 'X-index of centroid'
    xcenter.description = 'X index of the geometric center of the cloud feature at the given time'
    xcenter.units = 'unitless'
    xcenter.fill_value = fillvalue

    ycenter = filesave.createVariable('ycenter', 'f4', ('ntracks', 'nmaxlength'), zlib=True, complevel=5, fill_value=fillvalue)
    ycenter.long_name = 'Y-index of centroid'
    ycenter.description = 'Y index of the geometric center of the cloud feature at the given time'
    ycenter.units = 'unitless'
    ycenter.fill_value = fillvalue

    xcenter_weighted = filesave.createVariable('xcenter_weighted', 'f4', ('ntracks', 'nmaxlength'), zlib=True, complevel=5, fill_value=fillvalue)
    xcenter_weighted.long_name = 'X-index of centroid'
    xcenter_weighted.description = 'X index of the brightness temperature weighted center of the cloud feature at the given time'
    xcenter_weighted.units = 'unitless'
    xcenter_weighted.fill_value = fillvalue

    ycenter_weighted = filesave.createVariable('ycenter_weighted', 'f4', ('ntracks', 'nmaxlength'), zlib=True, complevel=5, fill_value=fillvalue)
    ycenter_weighted.long_name = 'Y-index of centroid'
    ycenter_weighted.description = 'Y index of the brightness temperature weighted center of the cloud feature at the given time'
    ycenter_weighted.units = 'unitless'
    ycenter_weighted.fill_value = fillvalue


    if landsea == 1:
        landfrac = filesave.createVariable('landfrac', 'f4', ('ntracks', 'nmaxlength'), zlib=True, complevel=5, fill_value=fillvalue)
        landfrac.long_name= 'Fraction of cloud over land'
        landfrac.description = 'Fraction of the core + cold anvil over land'
        landfrac.min_value = 0
        landfrac.max_value = 1
        landfrac.file_value = fillvalue 
        landfrac.units = 'unitless'

        landfrac_conv = filesave.createVariable('landfrac_conv', 'f4', ('ntracks', 'nmaxlength'), zlib=True, complevel=5, fill_value=fillvalue)
        landfrac_conv.long_name= 'Fraction of cloud over land'
        landfrac_conv.description = 'Fraction of the core over land'
        landfrac_conv.min_value = 0
        landfrac_conv.max_value = 1
        landfrac_conv.file_value = fillvalue 
        landfrac_conv.units = 'unitless'

    # Fill variables
    lifetime[:] = finaltrack_tracklength
    basetime[:,:] = finaltrack_basetime
    cloudidfiles[:,:,:] = finaltrack_cloudidfile
    datetimestrings[:,:,:] = finaltrack_datetimestring
    meanlat[:,:] = finaltrack_corecold_meanlat
    meanlon[:,:] = finaltrack_corecold_meanlon
    minlat[:,:] = finaltrack_corecold_minlat
    minlon[:,:] = finaltrack_corecold_minlon
    maxlat[:,:] = finaltrack_corecold_maxlat
    maxlon[:,:] = finaltrack_corecold_maxlon
    radius[:,:] = finaltrack_corecold_radius
    boundary[:] = finaltrack_corecold_boundary
    radius_warmanvil[:,:] = finaltrack_corecoldwarm_radius
    npix[:,:] = finaltrack_ncorecoldpix
    nconv[:,:] = finaltrack_ncorepix
    ncoldanvil[:,:] = finaltrack_ncoldpix
    nwarmanvil[:,:] = finaltrack_nwarmpix
    cloudnumber[:,:] = finaltrack_corecold_cloudnumber
    status[:,:] = finaltrack_corecold_status
    startstatus[:] = finaltrack_corecold_startstatus
    endstatus[:] = finaltrack_corecold_endstatus
    mergenumbers[:,:] = finaltrack_corecold_mergenumber
    splitnumbers[:,:] = finaltrack_corecold_splitnumber
    trackinterruptions[:] = finaltrack_corecold_trackinterruptions
    mintb[:,:] = finaltrack_corecold_mintb
    meantb[:,:] = finaltrack_corecold_meantb
    meantb_conv[:,:] = finaltrack_core_meantb
    histtb[:,:,:]= finaltrack_corecold_histtb
    majoraxis[:,:] = finaltrack_corecold_majoraxis
    orientation[:,:] = finaltrack_corecold_orientation
    eccentricity[:,:] = finaltrack_corecold_eccentricity
    perimeter[:,:] = finaltrack_corecold_perimeter
    xcenter[:,:] = finaltrack_corecold_xcenter
    ycenter[:,:] = finaltrack_corecold_ycenter
    xcenter_weighted[:,:] = finaltrack_corecold_xweightedcenter
    ycenter_weighted[:,:] = finaltrack_corecold_yweightedcenter
    if landsea == 1:
        landfrac[:,:] = finaltrack_corecold_landfrac
        landfrac_conv[:,:] = finaltrack_core_landfrac

    # Close and save netcdf
    filesave.close()









