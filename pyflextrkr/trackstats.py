# Purpose: This gets statistics about each track from the satellite data.

# Author: Orginial IDL version written by Sally A. McFarline (sally.mcfarlane@pnnl.gov) and modified for Zhe Feng (zhe.feng@pnnl.gov). Python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov)

# Define function that calculates track statistics for satellite data
def trackstats_tb(
    datasource,
    datadescription,
    pixel_radius,
    geolimits,
    areathresh,
    cloudtb_threshs,
    absolutetb_threshs,
    startdate,
    enddate,
    timegap,
    cloudid_filebase,
    tracking_inpath,
    stats_path,
    track_version,
    tracknumbers_version,
    tracknumbers_filebase,
    lengthrange=[2, 120],
):
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
    import os
    import sys
    from math import pi
    from skimage.measure import regionprops
    import time
    import gc
    import pandas as pd
    import logging
    
    logger = logging.getLogger(__name__)


    np.set_printoptions(threshold=np.inf)

    #############################################################################
    # Set constants

    # Isolate core and cold anvil brightness temperature thresholds
    thresh_core = cloudtb_threshs[0]
    thresh_cold = cloudtb_threshs[1]

    # Set output filename
    trackstats_outfile = (
        stats_path
        + "stats_"
        + tracknumbers_filebase
        + "_"
        + startdate
        + "_"
        + enddate
        + ".nc"
    )

    ###################################################################################
    # # Load latitude and longitude grid. These were created in subroutine_idclouds and is saved in each file.
    # logger.info('Determining which files will be processed')
    # logger.info((time.ctime()))

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
    logger.info("Loading gettracks data")
    logger.info((time.ctime()))
    cloudtrack_file = (
        stats_path + tracknumbers_filebase + "_" + startdate + "_" + enddate + ".nc"
    )

    cloudtrackdata = Dataset(cloudtrack_file, "r")
    numtracks = cloudtrackdata["ntracks"][:]
    cloudidfiles = cloudtrackdata["cloudid_files"][:]
    nfiles = cloudtrackdata.dimensions["nfiles"].size
    tracknumbers = cloudtrackdata["track_numbers"][:]
    trackreset = cloudtrackdata["track_reset"][:]
    tracksplit = cloudtrackdata["track_splitnumbers"][:]
    trackmerge = cloudtrackdata["track_mergenumbers"][:]
    trackstatus = cloudtrackdata["track_status"][:]
    cloudtrackdata.close()

    # Convert filenames and timegap to string
    # numcharfilename = len(list(cloudidfiles_list[0]))
    tmpfname = "".join(chartostring(cloudidfiles[0]))
    numcharfilename = len(list(tmpfname))

    # Load latitude and longitude grid from any cloudidfile since they all have the map of latitude and longitude saved
    latlondata = Dataset(tracking_inpath + tmpfname, "r")
    longitude = latlondata.variables["longitude"][:]
    latitude = latlondata.variables["latitude"][:]
    latlondata.close()

    # Determine dimensions of data
    # nfiles = len(cloudidfiles_list)
    ny, nx = np.shape(latitude)

    ############################################################################
    # Initialize grids
    logger.info("Initiailizinng matrices")
    logger.info((time.ctime()))

    nmaxclouds = max(lengthrange)

    mintb_thresh = absolutetb_threshs[0]
    maxtb_thresh = absolutetb_threshs[1]
    tbinterval = 2
    tbbins = np.arange(mintb_thresh, maxtb_thresh + tbinterval, tbinterval)
    nbintb = len(tbbins)

    finaltrack_tracklength = np.ones(int(numtracks), dtype=np.int32) * -9999
    finaltrack_corecold_boundary = np.ones(int(numtracks), dtype=np.int32) * -9999
    finaltrack_basetime = np.empty(
        (int(numtracks), int(nmaxclouds)), dtype="datetime64[s]"
    )
    finaltrack_corecold_mintb = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_meantb = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_core_meantb = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_histtb = np.zeros((int(numtracks), int(nmaxclouds), nbintb - 1))
    finaltrack_corecold_radius = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecoldwarm_radius = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_meanlat = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_meanlon = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_maxlon = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_maxlat = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_ncorecoldpix = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=np.int32) * -9999
    )
    finaltrack_corecold_minlon = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_minlat = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_ncorepix = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=np.int32) * -9999
    )
    finaltrack_ncoldpix = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=np.int32) * -9999
    )
    finaltrack_nwarmpix = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=np.int32) * -9999
    )
    finaltrack_corecold_status = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=np.int32) * -9999
    )
    finaltrack_corecold_trackinterruptions = (
        np.ones(int(numtracks), dtype=np.int32) * -9999
    )
    finaltrack_corecold_mergenumber = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=np.int32) * -9999
    )
    finaltrack_corecold_splitnumber = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=np.int32) * -9999
    )
    finaltrack_corecold_cloudnumber = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=np.int32) * -9999
    )
    finaltrack_datetimestring = [
        [["" for x in range(13)] for y in range(int(nmaxclouds))]
        for z in range(int(numtracks))
    ]
    finaltrack_cloudidfile = np.chararray(
        (int(numtracks), int(nmaxclouds), int(numcharfilename))
    )
    finaltrack_corecold_majoraxis = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_orientation = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_eccentricity = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_perimeter = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_xcenter = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * -9999
    )
    finaltrack_corecold_ycenter = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * -9999
    )
    finaltrack_corecold_xweightedcenter = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * -9999
    )
    finaltrack_corecold_yweightedcenter = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * -9999
    )

    #########################################################################################
    # loop over files. Calculate statistics and organize matrices by tracknumber and cloud
    logger.info("Looping over files and calculating statistics for each file")
    logger.info((time.ctime()))
    for nf in range(0, nfiles - 1):
        # for nf in range(0, 2):
        # logger.info(('File #: ' + str(nf)))
        # logger.info((time.ctime()))

        file_tracknumbers = tracknumbers[0, nf, :]

        # Only process file if that file contains a track
        if np.nanmax(file_tracknumbers) > 0:

            fname = "".join(chartostring(cloudidfiles[nf]))
            logger.info(nf, fname)

            # Load cloudid file
            cloudid_file = tracking_inpath + fname
            # logger.info(cloudid_file)

            file_cloudiddata = Dataset(cloudid_file, "r")
            file_tb = file_cloudiddata["tb"][:]
            file_cloudtype = file_cloudiddata["cloudtype"][:]
            file_all_cloudnumber = file_cloudiddata["cloudnumber"][:]
            file_corecold_cloudnumber = file_cloudiddata["convcold_cloudnumber"][:]
            file_basetime = file_cloudiddata["basetime"][:]
            basetime_units = file_cloudiddata["basetime"].units
            basetime_calendar = file_cloudiddata["basetime"].calendar
            file_cloudiddata.close()

            file_datetimestring = cloudid_file[
                len(tracking_inpath) + len(cloudid_filebase) : -3
            ]

            # Find unique track numbers
            uniquetracknumbers = np.unique(file_tracknumbers)
            uniquetracknumbers = uniquetracknumbers[np.isfinite(uniquetracknumbers)]
            uniquetracknumbers = uniquetracknumbers[uniquetracknumbers > 0].astype(int)

            # Loop over unique tracknumbers
            # logger.info('Loop over tracks in file')
            for itrack in uniquetracknumbers:
                # logger.info(('Unique track number: ' + str(itrack)))
                # logger.info('itrack: ', itrack)

                # Find cloud number that belongs to the current track in this file
                cloudnumber = (
                    np.array(np.where(file_tracknumbers == itrack))[0, :] + 1
                )  # Finds cloud numbers associated with that track. Need to add one since tells index, which starts at 0, and we want the number, which starts at one
                cloudindex = cloudnumber - 1  # Index within the matrice of this cloud.

                # if itrack == 4771:
                # logger.info('itrack = 4771 and cloud number = ', cloudnumber)
                # logger.info('length of cloud number: ', len(cloudnumber))
                # logger.info('count of file core cold cloudnumber equal to cloud number: ', np.count_nonzero(file_corecold_cloudnumber == cloudnumber))
                # if len(cloudnumber) == 1:
                # logger.info('THE LENGTH OF CLOUD NUMBER IS 1')
                # else:
                # logger.info('THE LENGTH OF CLOUD NUMBER IS NOT 1')
                # if len(cloudnumber) == 1:
                # logger.info('THE IF STATEMENT IS SATISFIED')
                # else:
                # logger.info('THE IF STATEMENT FAILED')

                if (
                    len(cloudnumber) == 1
                ):  # Should only be one cloud number. In mergers and split, the associated clouds should be listed in the file_splittracknumbers and file_mergetracknumbers
                    # Find cloud in cloudid file associated with this track
                    corearea = np.array(
                        np.where(
                            (file_corecold_cloudnumber[0, :, :] == cloudnumber)
                            & (file_cloudtype[0, :, :] == 1)
                        )
                    )
                    # if itrack == 4771:
                    # logger.info('itrack = 4771 and corearea = ', corearea)
                    ncorepix = np.shape(corearea)[1]

                    coldarea = np.array(
                        np.where(
                            (file_corecold_cloudnumber[0, :, :] == cloudnumber)
                            & (file_cloudtype[0, :, :] == 2)
                        )
                    )
                    ncoldpix = np.shape(coldarea)[1]

                    warmarea = np.array(
                        np.where(
                            (file_all_cloudnumber[0, :, :] == cloudnumber)
                            & (file_cloudtype[0, :, :] == 3)
                        )
                    )
                    nwarmpix = np.shape(warmarea)[1]

                    corecoldarea = np.array(
                        np.where(
                            (file_corecold_cloudnumber[0, :, :] == cloudnumber)
                            & (
                                (file_cloudtype[0, :, :] == 1)
                                | (file_cloudtype[0, :, :] == 2)
                            )
                        )
                    )
                    ncorecoldpix = np.shape(corecoldarea)[1]

                    # Find current length of the track. Use for indexing purposes. Also, record the current length the given track.
                    lengthindex = np.array(
                        np.where(finaltrack_corecold_cloudnumber[itrack - 1, :] > 0)
                    )
                    if np.shape(lengthindex)[1] > 0:
                        nc = np.nanmax(lengthindex) + 1
                    else:
                        nc = 0
                    finaltrack_tracklength[itrack - 1] = (
                        nc + 1
                    )  # Need to add one since array index starts at 0

                    if nc < nmaxclouds:
                        # Save information that links this cloud back to its raw pixel level data
                        finaltrack_basetime[itrack - 1, nc] = np.array(
                            [
                                pd.to_datetime(
                                    num2date(
                                        file_basetime,
                                        units=basetime_units,
                                        calendar=basetime_calendar,
                                    )
                                )
                            ],
                            dtype="datetime64[s]",
                        )[0, 0]
                        finaltrack_corecold_cloudnumber[itrack - 1, nc] = cloudnumber
                        # finaltrack_cloudidfile[itrack-1][nc][:] = list(cloudidfiles_list[nf])
                        finaltrack_cloudidfile[itrack - 1][nc][:] = fname
                        finaltrack_datetimestring[int(itrack - 1)][int(nc)][
                            :
                        ] = file_datetimestring
                        # if (nf == 6) & (itrack == 713):
                        #     import pdb; pdb.set_trace()
                        ###############################################################
                        # Calculate statistics about this cloud system
                        # 11/21/2019 - Make sure this cloud exists
                        if ncorecoldpix > 0:
                            # Location statistics of core+cold anvil (aka the convective system)
                            corecoldlat = latitude[corecoldarea[0], corecoldarea[1]]
                            corecoldlon = longitude[corecoldarea[0], corecoldarea[1]]

                            finaltrack_corecold_meanlat[itrack - 1, nc] = np.nanmean(
                                corecoldlat
                            )
                            finaltrack_corecold_meanlon[itrack - 1, nc] = np.nanmean(
                                corecoldlon
                            )

                            finaltrack_corecold_minlat[itrack - 1, nc] = np.nanmin(
                                corecoldlat
                            )
                            finaltrack_corecold_minlon[itrack - 1, nc] = np.nanmin(
                                corecoldlon
                            )

                            finaltrack_corecold_maxlat[itrack - 1, nc] = np.nanmax(
                                corecoldlat
                            )
                            finaltrack_corecold_maxlon[itrack - 1, nc] = np.nanmax(
                                corecoldlon
                            )

                            # Determine if core+cold touches of the boundaries of the domain
                            if (
                                np.absolute(
                                    finaltrack_corecold_minlat[itrack - 1, nc]
                                    - geolimits[0]
                                )
                                < 0.1
                                or np.absolute(
                                    finaltrack_corecold_maxlat[itrack - 1, nc]
                                    - geolimits[2]
                                )
                                < 0.1
                                or np.absolute(
                                    finaltrack_corecold_minlon[itrack - 1, nc]
                                    - geolimits[1]
                                )
                                < 0.1
                                or np.absolute(
                                    finaltrack_corecold_maxlon[itrack - 1, nc]
                                    - geolimits[3]
                                )
                                < 0.1
                            ):
                                finaltrack_corecold_boundary[itrack - 1] = 1

                            # Save number of pixels (metric for size)
                            finaltrack_ncorecoldpix[itrack - 1, nc] = ncorecoldpix
                            finaltrack_ncorepix[itrack - 1, nc] = ncorepix
                            finaltrack_ncoldpix[itrack - 1, nc] = ncoldpix
                            finaltrack_nwarmpix[itrack - 1, nc] = nwarmpix

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
                                maxyindex = ny

                            if np.nanmin(corecoldarea[1]) - pad > 0:
                                minxindex = np.nanmin(corecoldarea[1]) - pad
                            else:
                                minxindex = 0

                            if np.nanmax(corecoldarea[1]) + pad < nx - 1:
                                maxxindex = np.nanmax(corecoldarea[1]) + pad + 1
                            else:
                                maxxindex = nx

                            # Isolate the region around the cloud using the padded region
                            isolatedcloudnumber = np.copy(
                                file_corecold_cloudnumber[
                                    0, minyindex:maxyindex, minxindex:maxxindex
                                ]
                            ).astype(int)
                            isolatedtb = np.copy(
                                file_tb[0, minyindex:maxyindex, minxindex:maxxindex]
                            )

                            # Remove brightness temperatures outside core + cold anvil
                            isolatedtb[isolatedcloudnumber != cloudnumber] = 0

                            # Turn cloud map to binary
                            isolatedcloudnumber[isolatedcloudnumber != cloudnumber] = 0
                            isolatedcloudnumber[isolatedcloudnumber == cloudnumber] = 1

                            # Calculate major axis, orientation, eccentricity
                            cloudproperities = regionprops(
                                isolatedcloudnumber, intensity_image=isolatedtb
                            )

                            finaltrack_corecold_eccentricity[
                                itrack - 1, nc
                            ] = cloudproperities[0].eccentricity
                            finaltrack_corecold_majoraxis[itrack - 1, nc] = (
                                cloudproperities[0].major_axis_length * pixel_radius
                            )
                            finaltrack_corecold_orientation[itrack - 1, nc] = (
                                cloudproperities[0].orientation
                            ) * (180 / float(pi))
                            finaltrack_corecold_perimeter[itrack - 1, nc] = (
                                cloudproperities[0].perimeter * pixel_radius
                            )
                            [temp_ycenter, temp_xcenter] = cloudproperities[0].centroid
                            [
                                finaltrack_corecold_ycenter[itrack - 1, nc],
                                finaltrack_corecold_xcenter[itrack - 1, nc],
                            ] = np.add(
                                [temp_ycenter, temp_xcenter], [minyindex, minxindex]
                            ).astype(
                                int
                            )
                            [
                                temp_yweightedcenter,
                                temp_xweightedcenter,
                            ] = cloudproperities[0].weighted_centroid
                            [
                                finaltrack_corecold_yweightedcenter[itrack - 1, nc],
                                finaltrack_corecold_xweightedcenter[itrack - 1, nc],
                            ] = np.add(
                                [temp_yweightedcenter, temp_xweightedcenter],
                                [minyindex, minxindex],
                            ).astype(
                                int
                            )

                            # Determine equivalent radius of core+cold. Assuming circular area = (number pixels)*(pixel radius)^2, equivalent radius = sqrt(Area / pi)
                            finaltrack_corecold_radius[itrack - 1, nc] = np.sqrt(
                                np.divide(ncorecoldpix * (np.square(pixel_radius)), pi)
                            )

                            # Determine equivalent radius of core+cold+warm. Assuming circular area = (number pixels)*(pixel radius)^2, equivalent radius = sqrt(Area / pi)
                            finaltrack_corecoldwarm_radius[itrack - 1, nc] = np.sqrt(
                                np.divide(
                                    (ncorepix + ncoldpix + nwarmpix)
                                    * (np.square(pixel_radius)),
                                    pi,
                                )
                            )

                            ##############################################################
                            # Calculate brightness temperature statistics of core+cold anvil
                            corecoldtb = np.copy(
                                file_tb[0, corecoldarea[0], corecoldarea[1]]
                            )

                            finaltrack_corecold_mintb[itrack - 1, nc] = np.nanmin(
                                corecoldtb
                            )
                            finaltrack_corecold_meantb[itrack - 1, nc] = np.nanmean(
                                corecoldtb
                            )

                            ################################################################
                            # Histogram of brightness temperature for core+cold anvil
                            (
                                finaltrack_corecold_histtb[itrack - 1, nc, :],
                                usedtbbins,
                            ) = np.histogram(
                                corecoldtb,
                                range=(mintb_thresh, maxtb_thresh),
                                bins=tbbins,
                            )

                            # Save track information. Need to subtract one since cloudnumber gives the number of the cloud (which starts at one), but we are looking for its index (which starts at zero)
                            finaltrack_corecold_status[itrack - 1, nc] = np.copy(
                                trackstatus[0, nf, cloudindex]
                            )
                            finaltrack_corecold_mergenumber[itrack - 1, nc] = np.copy(
                                trackmerge[0, nf, cloudindex]
                            )
                            finaltrack_corecold_splitnumber[itrack - 1, nc] = np.copy(
                                tracksplit[0, nf, cloudindex]
                            )
                            finaltrack_corecold_trackinterruptions[
                                itrack - 1
                            ] = np.copy(trackreset[0, nf, cloudindex])

                            # logger.info('shape of finaltrack_corecold_status: ', finaltrack_corecold_status.shape)

                            ####################################################################
                            # Calculate mean brightness temperature for core
                            coretb = np.copy(file_tb[0, coldarea[0], coldarea[1]])

                            finaltrack_core_meantb[itrack - 1, nc] = np.nanmean(coretb)

                    else:
                        sys.exit(
                            str(nc)
                            + " greater than maximum allowed number clouds, "
                            + str(nmaxclouds)
                        )

                elif len(cloudnumber) > 1:
                    sys.exit(
                        str(cloudnumber)
                        + " clouds linked to one track. Each track should only be linked to one cloud in each file in the track_number array. The track_number variable only tracks the largest cell in mergers and splits. The small clouds in tracks and mergers should only be listed in the track_splitnumbers and track_mergenumbers arrays."
                    )

    ###############################################################
    ## Remove tracks that have no cells. These tracks are short.
    logger.info("Removing tracks with no cells")
    logger.info((time.ctime()))
    gc.collect()

    # logger.info('finaltrack_tracklength shape at line 385: ', finaltrack_tracklength.shape)
    # logger.info('finaltrack_tracklength(4771): ', finaltrack_tracklength[4770])
    cloudindexpresent = np.array(np.where(finaltrack_tracklength != -9999))[0, :]
    numtracks = len(cloudindexpresent)
    # logger.info('length of cloudindex present: ', len(cloudindexpresent))

    maxtracklength = np.nanmax(finaltrack_tracklength)
    # logger.info('maxtracklength: ', maxtracklength)

    finaltrack_tracklength = finaltrack_tracklength[cloudindexpresent]
    finaltrack_corecold_boundary = finaltrack_corecold_boundary[cloudindexpresent]
    finaltrack_basetime = finaltrack_basetime[cloudindexpresent, 0:maxtracklength]
    finaltrack_corecold_mintb = finaltrack_corecold_mintb[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_meantb = finaltrack_corecold_meantb[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_core_meantb = finaltrack_core_meantb[cloudindexpresent, 0:maxtracklength]
    finaltrack_corecold_histtb = finaltrack_corecold_histtb[
        cloudindexpresent, 0:maxtracklength, :
    ]
    finaltrack_corecold_radius = finaltrack_corecold_radius[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecoldwarm_radius = finaltrack_corecoldwarm_radius[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_meanlat = finaltrack_corecold_meanlat[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_meanlon = finaltrack_corecold_meanlon[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_maxlon = finaltrack_corecold_maxlon[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_maxlat = finaltrack_corecold_maxlat[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_minlon = finaltrack_corecold_minlon[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_minlat = finaltrack_corecold_minlat[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_ncorecoldpix = finaltrack_ncorecoldpix[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_ncorepix = finaltrack_ncorepix[cloudindexpresent, 0:maxtracklength]
    finaltrack_ncoldpix = finaltrack_ncoldpix[cloudindexpresent, 0:maxtracklength]
    finaltrack_nwarmpix = finaltrack_nwarmpix[cloudindexpresent, 0:maxtracklength]
    finaltrack_corecold_status = finaltrack_corecold_status[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_trackinterruptions = finaltrack_corecold_trackinterruptions[
        cloudindexpresent
    ]
    finaltrack_corecold_mergenumber = finaltrack_corecold_mergenumber[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_splitnumber = finaltrack_corecold_splitnumber[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_cloudnumber = finaltrack_corecold_cloudnumber[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_datetimestring = list(
        finaltrack_datetimestring[i][0:maxtracklength][:] for i in cloudindexpresent
    )
    finaltrack_cloudidfile = finaltrack_cloudidfile[
        cloudindexpresent, 0:maxtracklength, :
    ]
    finaltrack_corecold_majoraxis = finaltrack_corecold_majoraxis[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_orientation = finaltrack_corecold_orientation[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_eccentricity = finaltrack_corecold_eccentricity[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_perimeter = finaltrack_corecold_perimeter[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_xcenter = finaltrack_corecold_xcenter[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_ycenter = finaltrack_corecold_ycenter[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_xweightedcenter = finaltrack_corecold_xweightedcenter[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_yweightedcenter = finaltrack_corecold_yweightedcenter[
        cloudindexpresent, 0:maxtracklength
    ]

    gc.collect()

    ########################################################
    # Correct merger and split cloud numbers

    # Initialize adjusted matrices
    adjusted_finaltrack_corecold_mergenumber = (
        np.ones(np.shape(finaltrack_corecold_mergenumber)) * -9999
    )
    adjusted_finaltrack_corecold_splitnumber = (
        np.ones(np.shape(finaltrack_corecold_mergenumber)) * -9999
    )
    logger.info(("total tracks: " + str(numtracks)))
    logger.info("Correcting mergers and splits")
    logger.info((time.ctime()))

    # Create adjustor
    indexcloudnumber = np.copy(cloudindexpresent) + 1
    adjustor = np.arange(0, np.max(cloudindexpresent) + 2)
    for it in range(0, numtracks):
        adjustor[indexcloudnumber[it]] = it + 1
    adjustor = np.append(adjustor, -9999)

    # Adjust mergers
    temp_finaltrack_corecold_mergenumber = finaltrack_corecold_mergenumber.astype(
        int
    ).ravel()
    temp_finaltrack_corecold_mergenumber[
        temp_finaltrack_corecold_mergenumber == -9999
    ] = (np.max(cloudindexpresent) + 2)
    adjusted_finaltrack_corecold_mergenumber = adjustor[
        temp_finaltrack_corecold_mergenumber
    ]
    adjusted_finaltrack_corecold_mergenumber = np.reshape(
        adjusted_finaltrack_corecold_mergenumber,
        np.shape(finaltrack_corecold_mergenumber),
    )

    # Adjust splitters
    temp_finaltrack_corecold_splitnumber = finaltrack_corecold_splitnumber.astype(
        int
    ).ravel()
    temp_finaltrack_corecold_splitnumber[
        temp_finaltrack_corecold_splitnumber == -9999
    ] = (np.max(cloudindexpresent) + 2)
    adjusted_finaltrack_corecold_splitnumber = adjustor[
        temp_finaltrack_corecold_splitnumber
    ]
    adjusted_finaltrack_corecold_splitnumber = np.reshape(
        adjusted_finaltrack_corecold_splitnumber,
        np.shape(finaltrack_corecold_splitnumber),
    )

    #########################################################################
    # Record starting and ending status
    logger.info("Determine starting and ending status")
    logger.info((time.ctime()))

    # Starting status
    finaltrack_corecold_startstatus = finaltrack_corecold_status[:, 0]

    # Ending status
    finaltrack_corecold_endstatus = (
        np.ones(len(finaltrack_corecold_startstatus)) * -9999
    )
    for trackstep in range(0, numtracks):
        if finaltrack_tracklength[trackstep] > 0:
            finaltrack_corecold_endstatus[trackstep] = finaltrack_corecold_status[
                trackstep, finaltrack_tracklength[trackstep] - 1
            ]

    #######################################################################
    # Write to netcdf
    logger.info("Writing trackstat netcdf")
    logger.info((time.ctime()))
    logger.info(trackstats_outfile)
    logger.info("")

    # Check if file already exists. If exists, delete
    if os.path.isfile(trackstats_outfile):
        os.remove(trackstats_outfile)

    from pyflextrkr import netcdf_io as net

    net.write_trackstats_tb(
        trackstats_outfile,
        numtracks,
        maxtracklength,
        nbintb,
        numcharfilename,
        datasource,
        datadescription,
        startdate,
        enddate,
        track_version,
        tracknumbers_version,
        timegap,
        thresh_core,
        thresh_cold,
        pixel_radius,
        geolimits,
        areathresh,
        mintb_thresh,
        maxtb_thresh,
        basetime_units,
        basetime_calendar,
        finaltrack_tracklength,
        finaltrack_basetime,
        finaltrack_cloudidfile,
        finaltrack_datetimestring,
        finaltrack_corecold_meanlat,
        finaltrack_corecold_meanlon,
        finaltrack_corecold_minlat,
        finaltrack_corecold_minlon,
        finaltrack_corecold_maxlat,
        finaltrack_corecold_maxlon,
        finaltrack_corecold_radius,
        finaltrack_corecoldwarm_radius,
        finaltrack_ncorecoldpix,
        finaltrack_ncorepix,
        finaltrack_ncoldpix,
        finaltrack_nwarmpix,
        finaltrack_corecold_cloudnumber,
        finaltrack_corecold_status,
        finaltrack_corecold_startstatus,
        finaltrack_corecold_endstatus,
        adjusted_finaltrack_corecold_mergenumber,
        adjusted_finaltrack_corecold_splitnumber,
        finaltrack_corecold_trackinterruptions,
        finaltrack_corecold_boundary,
        finaltrack_corecold_mintb,
        finaltrack_corecold_meantb,
        finaltrack_core_meantb,
        finaltrack_corecold_histtb,
        finaltrack_corecold_majoraxis,
        finaltrack_corecold_orientation,
        finaltrack_corecold_eccentricity,
        finaltrack_corecold_perimeter,
        finaltrack_corecold_xcenter,
        finaltrack_corecold_ycenter,
        finaltrack_corecold_xweightedcenter,
        finaltrack_corecold_yweightedcenter,
    )


# Define function that calculates track statistics for LES data
def trackstats_LES(
    datasource,
    datadescription,
    pixel_radius,
    latlon_file,
    geolimits,
    areathresh,
    cloudlwp_threshs,
    absolutelwp_threshs,
    startdate,
    enddate,
    timegap,
    cloudid_filebase,
    tracking_inpath,
    stats_path,
    track_version,
    tracknumbers_version,
    tracknumbers_filebase,
    lengthrange=[5, 60],
):
    # Inputs:
    # datasource - source of the data
    # datadescription - description of data source, included in all output file names
    # pixel_radius - radius of pixels in km
    # latlon_file - filename of the file that contains the latitude and longitude data
    # geolimits - 4-element array with plotting boundaries [lat_min, lon_min, lat_max, lon_max]
    # areathresh - minimum core + cold anvil area of a tracked cloud
    # cloudlwp_threshs - brightness temperature thresholds for convective classification
    # absolutelwp_threshs - brightness temperature thresholds defining the valid data range
    # startdate - starting date and time of the data
    # enddate - ending date and time of the data
    # cloudid_filebase - header of the cloudid data files
    # tracking_inpath - location of the cloudid and single track data
    # stats_path - location of the track data. also the location where the data from this code will be saved
    # track_version - Version of track single cloud files
    # tracknumbers_version - Verison of the complete track files
    # tracknumbers_filebase - header of the tracking matrix generated in the previous code.
    # lengthrange - Optional. Set this keyword to a vector [minlength,maxlength] to specify the lifetime range for the tracks.

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
    # minlwp - minimum brightness temperature of the core and cold anvil
    # meanlwp - mean brightness temperature of the core and cold anvil
    # meanlwp_conv - mean brightness temperature of the core
    # histlwp - histogram of the brightness temperatures in the core and cold anvil
    # majoraxis - length of the major axis of the core and cold anvil
    # orientation - angular position of the core and cold anvil
    # eccentricity - eccentricity of the core and cold anvil
    # perimeter - approximate size of the perimeter in the core and cold anvil
    # xcenter - x-coordinate of the geometric center
    # ycenter - y-coordinate of the geometric center
    # xcenter_weighted - x-coordinate of the liquid water path weighted center
    # ycenter_weighted - y-coordinate of the liquid water path weighted center

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
    import xarray as xr
    import pandas as pd

    np.set_printoptions(threshold=np.inf)

    #############################################################################
    # Set constants

    # Isolate core and cold anvil brightness temperature thresholds
    thresh_core = cloudlwp_threshs[0]
    thresh_cold = cloudlwp_threshs[1]

    # Set output filename
    trackstats_outfile = (
        stats_path
        + "stats_"
        + tracknumbers_filebase
        + "_"
        + startdate
        + "_"
        + enddate
        + ".nc"
    )

    ###################################################################################
    # Load latitude and longitude grid. These were created in subroutine_idclouds and is saved in each file.

    # Find filenames of idcloud files
    temp_cloudidfiles = fnmatch.filter(
        os.listdir(tracking_inpath), cloudid_filebase + "*"
    )

    # Select one file. Any file is fine since they all havel the map of latitued and longitude saved.
    temp_cloudidfiles = temp_cloudidfiles[0]

    # Load latitude and longitude grid
    latlondata = Dataset(tracking_inpath + temp_cloudidfiles, "r")
    longitude = latlondata.variables["longitude"][:]
    latitude = latlondata.variables["latitude"][:]
    latlondata.close()

    #############################################################################
    # Load track data
    cloudtrack_file = (
        stats_path + tracknumbers_filebase + "_" + startdate + "_" + enddate + ".nc"
    )

    cloudtrackdata = Dataset(cloudtrack_file, "r")
    numtracks = cloudtrackdata["ntracks"][:]
    cloudidfiles = cloudtrackdata["cloudid_files"][:]
    tracknumbers = cloudtrackdata["track_numbers"][:]
    trackreset = cloudtrackdata["track_reset"][:]
    tracksplit = cloudtrackdata["track_splitnumbers"][:]
    trackmerge = cloudtrackdata["track_mergenumbers"][:]
    trackstatus = cloudtrackdata["track_status"][:]
    cloudtrackdata.close()

    # Convert filenames and timegap to string
    numcharfilename = len(list(cloudidfiles[0]))

    # Determine dimensions of data
    nfiles = len(cloudidfiles)
    ny, nx = np.shape(latitude)
    logger.info(f"nfiles_cloudid:{nfiles}")

    ############################################################################
    # Initialize grids
    nmaxclouds = max(lengthrange)

    minlwp_thresh = absolutelwp_threshs[0]
    maxlwp_thresh = absolutelwp_threshs[1]
    lwpinterval = 0.1
    lwpbins = np.arange(minlwp_thresh, maxlwp_thresh + lwpinterval, lwpinterval)
    # lwpbins = np.logspace(np.log10(mminlwp_thresh), np.log10(maxlwp_thresh)*1.001,100)
    nbinlwp = len(lwpbins)

    finaltrack_tracklength = np.ones(int(numtracks), dtype=int) * -9999
    finaltrack_corecold_cloudnumber = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=int) * -9999
    )

    #########################################################################################
    # loop over files. Calculate the tracknumbers of cell tracks

    logger.info(f"numtracks from previous step:{numtracks}")
    numtracks_old = numtracks

    for nf in range(0, nfiles):

        file_tracknumbers = tracknumbers[0, nf, :]

        if np.nanmax(file_tracknumbers) > 0:

            fname = "".join(chartostring(cloudidfiles[nf]))
            logger.info(nf, fname)

            # Find unique track numbers
            uniquetracknumbers = np.unique(file_tracknumbers)
            uniquetracknumbers = uniquetracknumbers[uniquetracknumbers > 0].astype(int)
            # logger.info(f'uniquetracknumbers:{uniquetracknumbers}')

            # Loop over unique tracknumbers
            for itrack in uniquetracknumbers:

                # Find cloud number that belongs to the current track in this file
                cloudnumber = (
                    np.array(np.where(file_tracknumbers == itrack))[0, :] + 1
                )  # Finds cloud numbers associated with that track. Need to add one since tells index, which starts at 0, and we want the number, which starts at one

                if (
                    len(cloudnumber) == 1
                ):  # Should only be one cloud number. In mergers and split, the associated clouds should be listed in the file_splittracknumbers and file_mergetracknumbers

                    # Find current length of the track. Use for indexing purposes. Also, record the current length the given track.
                    lengthindex = np.array(
                        np.where(finaltrack_corecold_cloudnumber[itrack - 1, :] > 0)
                    )
                    # logger.info(f'lengthindex:{np.shape(lengthindex)[1]}')

                    if np.shape(lengthindex)[1] > 0:
                        nc = np.nanmax(lengthindex) + 1
                    else:
                        nc = 0
                    finaltrack_tracklength[itrack - 1] = (
                        nc + 1
                    )  # Need to add one since array index starts at 0

                    if nc < nmaxclouds:
                        finaltrack_corecold_cloudnumber[itrack - 1, nc] = cloudnumber
                    else:
                        logger.info(
                            (
                                "Track: "
                                + str(itrack)
                                + "; "
                                + str(nc)
                                + " greater than maximum allowed number clouds, "
                                + str(nmaxclouds)
                            )
                        )
                        nc = nc + 1

                elif len(cloudnumber) > 1:
                    sys.exit(
                        str(cloudnumber)
                        + " clouds linked to one track. Each track should only be linked to one cloud in each file in the track_number array. The track_number variable only tracks the largest cell in mergers and splits. The small clouds in tracks and mergers should only be listed in the track_splitnumbers and track_mergenumbers arrays."
                    )

    ###############################################################
    ## Remove tracks that have no cells. These tracks are short.

    gc.collect()

    cloudindexpresent = np.array(np.where(finaltrack_tracklength != -9999))[0, :]
    numtracks = len(cloudindexpresent)
    maxtracklength = np.nanmin([np.nanmax(finaltrack_tracklength), int(nmaxclouds)])
    if maxtracklength < nmaxclouds:
        nmaxclouds = maxtracklength
        finaltrack_corecold_cloudnumber = finaltrack_corecold_cloudnumber[
            :, :nmaxclouds
        ]

    logger.info(("Number of Tracks:" + str(int(len(finaltrack_tracklength)))))
    logger.info(f"numtracks:      {numtracks}")
    logger.info(f"maxtracklength: {maxtracklength}")
    # logger.info(f'cloudindexpresent: {cloudindexpresent}')
    logger.info("Tracks with no cells NOT included")

    ###############################################################
    # to calculate the statistic after having the number of tracks with cells
    finaltrack_corecold_boundary = np.ones(int(numtracks)) * -9999
    finaltrack_basetime = np.zeros(
        (int(numtracks), int(nmaxclouds)), dtype="datetime64[s]"
    )
    finaltrack_corecold_minlwp = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_meanlwp = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_core_meanlwp = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    # finaltrack_corecold_histlwp = np.zeros((int(numtracks),int(nmaxclouds), nbinlwp-1), dtype=float)
    finaltrack_corecold_radius = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecoldwarm_radius = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_meanlat = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_meanlon = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_maxlon = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_maxlat = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_ncorecoldpix = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=int) * -9999
    )
    finaltrack_corecold_minlon = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_minlat = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_ncorepix = np.ones((int(numtracks), int(nmaxclouds)), dtype=int) * -9999
    finaltrack_ncoldpix = np.ones((int(numtracks), int(nmaxclouds)), dtype=int) * -9999
    finaltrack_nwarmpix = np.ones((int(numtracks), int(nmaxclouds)), dtype=int) * -9999
    finaltrack_corecold_status = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=int) * -9999
    )
    finaltrack_corecold_trackinterruptions = np.ones(int(numtracks), dtype=int) * -9999
    finaltrack_corecold_mergenumber = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=int) * -9999
    )
    finaltrack_corecold_splitnumber = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=int) * -9999
    )
    finaltrack_datetimestring = [
        [["" for x in range(13)] for y in range(int(nmaxclouds))]
        for z in range(int(numtracks))
    ]
    finaltrack_cloudidfile = np.chararray(
        (int(numtracks), int(nmaxclouds), int(numcharfilename))
    )
    finaltrack_corecold_majoraxis = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_orientation = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_eccentricity = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_perimeter = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_xcenter = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * -9999
    )
    finaltrack_corecold_ycenter = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * -9999
    )
    finaltrack_corecold_xweightedcenter = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * -9999
    )
    finaltrack_corecold_yweightedcenter = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * -9999
    )

    finaltrack_tracklength = np.ones(int(numtracks_old), dtype=int) * -9999
    finaltrack_corecold_cloudnumber = (
        np.ones((int(numtracks_old), int(nmaxclouds)), dtype=int) * -9999
    )

    #########################################################################################
    # loop over files. Calculate statistics and organize matrices by tracknumber and cloud
    for nf in range(0, nfiles):

        file_tracknumbers = tracknumbers[0, nf, :]

        # Only process file if that file contains a track
        if np.nanmax(file_tracknumbers) > 0:

            fname = "".join(chartostring(cloudidfiles[nf]))
            logger.info(nf, fname)

            # Load cloudid file
            cloudid_file = tracking_inpath + fname

            file_cloudiddata = Dataset(cloudid_file, "r")
            file_lwp = file_cloudiddata["lwp"][:]
            file_cloudtype = file_cloudiddata["cloudtype"][:]
            file_all_cloudnumber = file_cloudiddata["cloudnumber"][:]
            file_corecold_cloudnumber = file_cloudiddata["convcold_cloudnumber"][:]
            file_basetime = file_cloudiddata["basetime"][:]
            basetime_units = file_cloudiddata["basetime"].units
            basetime_calendar = file_cloudiddata["basetime"].calendar
            file_cloudiddata.close()

            file_datetimestring = cloudid_file[
                len(tracking_inpath) + len(cloudid_filebase) : -3
            ]

            # Find unique track numbers
            uniquetracknumbers = np.unique(file_tracknumbers)
            uniquetracknumbers = uniquetracknumbers[uniquetracknumbers > 0].astype(int)

            # Loop over unique tracknumbers
            for itrack in uniquetracknumbers:

                # define the index in shortened array
                icelltrack = np.where(cloudindexpresent == itrack - 1)[0][0]
                # logger.info(f'itrack:{itrack}')
                # logger.info(f'icelltrack:{icelltrack}')
                if not np.size(icelltrack):
                    continue

                # Find cloud number that belongs to the current track in this file
                cloudnumber = (
                    np.array(np.where(file_tracknumbers == itrack))[0, :] + 1
                )  # Finds cloud numbers associated with that track. Need to add one since tells index, which starts at 0, and we want the number, which starts at one
                cloudindex = cloudnumber - 1  # Index within the matrice of this cloud.

                if (
                    len(cloudnumber) == 1
                ):  # Should only be one cloud number. In mergers and split, the associated clouds should be listed in the file_splittracknumbers and file_mergetracknumbers
                    # Find cloud in cloudid file associated with this track
                    corearea = np.array(
                        np.where(
                            (file_corecold_cloudnumber[0, :, :] == cloudnumber)
                            & (file_cloudtype[0, :, :] == 1)
                        )
                    )
                    ncorepix = np.shape(corearea)[1]

                    coldarea = np.array(
                        np.where(
                            (file_corecold_cloudnumber[0, :, :] == cloudnumber)
                            & (file_cloudtype[0, :, :] == 2)
                        )
                    )
                    ncoldpix = np.shape(coldarea)[1]

                    warmarea = np.array(
                        np.where(
                            (file_all_cloudnumber[0, :, :] == cloudnumber)
                            & (file_cloudtype[0, :, :] == 3)
                        )
                    )
                    nwarmpix = np.shape(warmarea)[1]

                    corecoldarea = np.array(
                        np.where(
                            (file_corecold_cloudnumber[0, :, :] == cloudnumber)
                            & (file_cloudtype[0, :, :] >= 1)
                            & (file_cloudtype[0, :, :] <= 2)
                        )
                    )
                    ncorecoldpix = np.shape(corecoldarea)[1]

                    # Find current length of the track. Use for indexing purposes. Also, record the current length the given track.
                    lengthindex = np.array(
                        np.where(finaltrack_corecold_cloudnumber[itrack - 1, :] > 0)
                    )
                    if np.shape(lengthindex)[1] > 0:
                        nc = np.nanmax(lengthindex) + 1
                    else:
                        nc = 0
                    finaltrack_tracklength[itrack - 1] = (
                        nc + 1
                    )  # Need to add one since array index starts at 0

                    if nc < nmaxclouds:

                        # Save information that links this cloud back to its raw pixel level data
                        finaltrack_basetime[icelltrack, nc] = np.array(
                            [
                                pd.to_datetime(
                                    num2date(
                                        file_basetime,
                                        units=basetime_units,
                                        calendar=basetime_calendar,
                                    )
                                )
                            ],
                            dtype="datetime64[s]",
                        )[0, 0]
                        # finaltrack_basetime[icelltrack, nc] = np.datetime64(pd.to_datetime(file_cloudiddata['basetime'].data)[0])
                        finaltrack_corecold_cloudnumber[itrack - 1, nc] = cloudnumber
                        # logger.info(f'nc:{nc}')
                        finaltrack_cloudidfile[icelltrack][nc][:] = fname
                        #                        finaltrack_cloudidfile[icelltrack][nc][:] = list(cloudidfiles[nf])
                        finaltrack_datetimestring[int(icelltrack)][int(nc)][
                            :
                        ] = file_datetimestring

                        ###############################################################
                        # Calculate statistics about this cloud system

                        #############
                        # Location statistics of core+cold anvil (aka the convective system)
                        corecoldlat = latitude[corecoldarea[0], corecoldarea[1]]
                        corecoldlon = longitude[corecoldarea[0], corecoldarea[1]]

                        finaltrack_corecold_meanlat[icelltrack, nc] = np.nanmean(
                            corecoldlat
                        )
                        finaltrack_corecold_meanlon[icelltrack, nc] = np.nanmean(
                            corecoldlon
                        )

                        finaltrack_corecold_minlat[icelltrack, nc] = np.nanmin(
                            corecoldlat
                        )
                        finaltrack_corecold_minlon[icelltrack, nc] = np.nanmin(
                            corecoldlon
                        )

                        finaltrack_corecold_maxlat[icelltrack, nc] = np.nanmax(
                            corecoldlat
                        )
                        finaltrack_corecold_maxlon[icelltrack, nc] = np.nanmax(
                            corecoldlon
                        )

                        # Determine if core+cold touches of the boundaries of the domain
                        if (
                            np.absolute(
                                finaltrack_corecold_minlat[icelltrack, nc]
                                - geolimits[0]
                            )
                            < 0.1
                            or np.absolute(
                                finaltrack_corecold_maxlat[icelltrack, nc]
                                - geolimits[2]
                            )
                            < 0.1
                            or np.absolute(
                                finaltrack_corecold_minlon[icelltrack, nc]
                                - geolimits[1]
                            )
                            < 0.1
                            or np.absolute(
                                finaltrack_corecold_maxlon[icelltrack, nc]
                                - geolimits[3]
                            )
                            < 0.1
                        ):
                            finaltrack_corecold_boundary[icelltrack] = 1

                        ############
                        # Save number of pixels (metric for size)
                        finaltrack_ncorecoldpix[icelltrack, nc] = ncorecoldpix
                        finaltrack_ncorepix[icelltrack, nc] = ncorepix
                        finaltrack_ncoldpix[icelltrack, nc] = ncoldpix
                        finaltrack_nwarmpix[icelltrack, nc] = nwarmpix

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
                            maxyindex = ny

                        if np.nanmin(corecoldarea[1]) - pad > 0:
                            minxindex = np.nanmin(corecoldarea[1]) - pad
                        else:
                            minxindex = 0

                        if np.nanmax(corecoldarea[1]) + pad < nx - 1:
                            maxxindex = np.nanmax(corecoldarea[1]) + pad + 1
                        else:
                            maxxindex = nx

                        # Isolate the region around the cloud using the padded region
                        isolatedcloudnumber = np.copy(
                            file_corecold_cloudnumber[
                                0, minyindex:maxyindex, minxindex:maxxindex
                            ]
                        ).astype(int)
                        isolatedlwp = np.copy(
                            file_lwp[0, minyindex:maxyindex, minxindex:maxxindex]
                        )

                        # Remove brightness temperatures outside core + cold anvil
                        isolatedlwp[isolatedcloudnumber != cloudnumber] = 0

                        # Turn cloud map to binary
                        isolatedcloudnumber[isolatedcloudnumber != cloudnumber] = 0
                        isolatedcloudnumber[isolatedcloudnumber == cloudnumber] = 1

                        # Calculate major axis, orientation, eccentricity
                        cloudproperities = regionprops(
                            isolatedcloudnumber, intensity_image=isolatedlwp
                        )

                        finaltrack_corecold_eccentricity[
                            icelltrack, nc
                        ] = cloudproperities[0].eccentricity
                        finaltrack_corecold_majoraxis[icelltrack, nc] = (
                            cloudproperities[0].major_axis_length * pixel_radius
                        )
                        finaltrack_corecold_orientation[icelltrack, nc] = (
                            cloudproperities[0].orientation
                        ) * (180 / float(pi))
                        finaltrack_corecold_perimeter[icelltrack, nc] = (
                            cloudproperities[0].perimeter * pixel_radius
                        )
                        [temp_ycenter, temp_xcenter] = cloudproperities[0].centroid
                        [
                            finaltrack_corecold_ycenter[icelltrack, nc],
                            finaltrack_corecold_xcenter[icelltrack, nc],
                        ] = np.add(
                            [temp_ycenter, temp_xcenter], [minyindex, minxindex]
                        ).astype(
                            int
                        )
                        [temp_yweightedcenter, temp_xweightedcenter] = cloudproperities[
                            0
                        ].weighted_centroid
                        [
                            finaltrack_corecold_yweightedcenter[icelltrack, nc],
                            finaltrack_corecold_xweightedcenter[icelltrack, nc],
                        ] = np.add(
                            [temp_yweightedcenter, temp_xweightedcenter],
                            [minyindex, minxindex],
                        ).astype(
                            int
                        )

                        # Determine equivalent radius of core+cold. Assuming circular area = (number pixels)*(pixel radius)^2, equivalent radius = sqrt(Area / pi)
                        finaltrack_corecold_radius[icelltrack, nc] = np.sqrt(
                            np.divide(ncorecoldpix * (np.square(pixel_radius)), pi)
                        )

                        # Determine equivalent radius of core+cold+warm. Assuming circular area = (number pixels)*(pixel radius)^2, equivalent radius = sqrt(Area / pi)
                        finaltrack_corecoldwarm_radius[icelltrack, nc] = np.sqrt(
                            np.divide(
                                (ncorepix + ncoldpix + nwarmpix)
                                * (np.square(pixel_radius)),
                                pi,
                            )
                        )

                        ##############################################################
                        # Calculate brightness temperature statistics of core+cold anvil
                        corecoldlwp = np.copy(
                            file_lwp[0, corecoldarea[0], corecoldarea[1]]
                        )

                        finaltrack_corecold_minlwp[icelltrack, nc] = np.nanmin(
                            corecoldlwp
                        )
                        finaltrack_corecold_meanlwp[icelltrack, nc] = np.nanmean(
                            corecoldlwp
                        )

                        ################################################################
                        # Histogram of brightness temperature for core+cold anvil
                        # finaltrack_corecold_histlwp[icelltrack,nc,:], usedlwpbins = np.histogram(corecoldlwp, range=(minlwp_thresh, maxlwp_thresh), bins=lwpbins)

                        # Save track information. Need to subtract one since cloudnumber gives the number of the cloud (which starts at one), but we are looking for its index (which starts at zero)
                        finaltrack_corecold_status[icelltrack, nc] = np.copy(
                            trackstatus[0, nf, cloudindex]
                        )
                        finaltrack_corecold_mergenumber[icelltrack, nc] = np.copy(
                            trackmerge[0, nf, cloudindex]
                        )
                        finaltrack_corecold_splitnumber[icelltrack, nc] = np.copy(
                            tracksplit[0, nf, cloudindex]
                        )
                        finaltrack_corecold_trackinterruptions[icelltrack] = np.copy(
                            trackreset[0, nf, cloudindex]
                        )

                        ####################################################################
                        # Calculate mean brightness temperature for core
                        corelwp = np.copy(file_lwp[0, coldarea[0], coldarea[1]])

                        finaltrack_core_meanlwp[icelltrack, nc] = np.nanmean(corelwp)

                    else:
                        logger.info(
                            (
                                "Track: "
                                + str(itrack)
                                + "; "
                                + str(nc)
                                + " greater than maximum allowed number clouds, "
                                + str(nmaxclouds)
                            )
                        )
                        nc = nc + 1

                elif len(cloudnumber) > 1:
                    sys.exit(
                        str(cloudnumber)
                        + " clouds linked to one track. Each track should only be linked to one cloud in each file in the track_number array. The track_number variable only tracks the largest cell in mergers and splits. The small clouds in tracks and mergers should only be listed in the track_splitnumbers and track_mergenumbers arrays."
                    )

    # remove tracks without cell for those two variables
    finaltrack_tracklength = finaltrack_tracklength[cloudindexpresent]
    finaltrack_corecold_cloudnumber = finaltrack_corecold_cloudnumber[
        cloudindexpresent, 0:maxtracklength
    ]

    ########################################################
    # Correct merger and split cloud numbers
    logger.info("Correcting merger and split numbers")

    # Initialize adjusted matrices
    adjusted_finaltrack_corecold_mergenumber = (
        np.ones(np.shape(finaltrack_corecold_mergenumber)) * -9999
    )
    adjusted_finaltrack_corecold_splitnumber = (
        np.ones(np.shape(finaltrack_corecold_mergenumber)) * -9999
    )
    logger.info(("total tracks: " + str(numtracks)))

    # Create adjustor
    indexcloudnumber = np.copy(cloudindexpresent) + 1
    adjustor = np.arange(0, np.max(cloudindexpresent) + 2)
    for it in range(0, numtracks):
        adjustor[indexcloudnumber[it]] = it + 1
    adjustor = np.append(adjustor, -9999)

    # Adjust mergers
    temp_finaltrack_corecold_mergenumber = finaltrack_corecold_mergenumber.astype(
        int
    ).ravel()
    temp_finaltrack_corecold_mergenumber[
        temp_finaltrack_corecold_mergenumber == -9999
    ] = (np.max(cloudindexpresent) + 2)
    adjusted_finaltrack_corecold_mergenumber = adjustor[
        temp_finaltrack_corecold_mergenumber
    ]
    adjusted_finaltrack_corecold_mergenumber = np.reshape(
        adjusted_finaltrack_corecold_mergenumber,
        np.shape(finaltrack_corecold_mergenumber),
    )

    # Adjust splitters
    temp_finaltrack_corecold_splitnumber = finaltrack_corecold_splitnumber.astype(
        int
    ).ravel()
    temp_finaltrack_corecold_splitnumber[
        temp_finaltrack_corecold_splitnumber == -9999
    ] = (np.max(cloudindexpresent) + 2)
    adjusted_finaltrack_corecold_splitnumber = adjustor[
        temp_finaltrack_corecold_splitnumber
    ]
    adjusted_finaltrack_corecold_splitnumber = np.reshape(
        adjusted_finaltrack_corecold_splitnumber,
        np.shape(finaltrack_corecold_splitnumber),
    )

    logger.info("Adjustment done")

    #########################################################################
    # Record starting and ending status
    logger.info("Isolating starting and ending status")

    # Starting status
    finaltrack_corecold_startstatus = np.copy(finaltrack_corecold_status[:, 0])

    # Ending status
    finaltrack_corecold_endstatus = (
        np.ones(len(finaltrack_corecold_startstatus)) * -9999
    )
    #    for trackstep in range(0, maxtracklength):
    for trackstep in range(0, numtracks):
        if (finaltrack_tracklength[trackstep] > 0) & (
            finaltrack_tracklength[trackstep] < maxtracklength
        ):
            finaltrack_corecold_endstatus[trackstep] = np.copy(
                finaltrack_corecold_status[
                    trackstep, finaltrack_tracklength[trackstep] - 1
                ]
            )

    #######################################################################
    # Write to netcdf
    gc.collect()

    logger.info("Writing trackstat netcdf")
    logger.info(trackstats_outfile)
    logger.info("")

    # Check if file already exists. If exists, delete
    if os.path.isfile(trackstats_outfile):
        os.remove(trackstats_outfile)

    # Define xarray dataset
    output_data = xr.Dataset(
        {
            "lifetime": (["ntracks"], finaltrack_tracklength),
            "basetime": (["ntracks", "nmaxlength"], finaltrack_basetime),
            "cloudidfiles": (
                ["ntracks", "nmaxlength", "nfilenamechars"],
                finaltrack_cloudidfile,
            ),
            "datetimestrings": (
                ["ntracks", "nmaxlength", "ndatetimechars"],
                finaltrack_datetimestring,
            ),
            "meanlat": (["ntracks", "nmaxlength"], finaltrack_corecold_meanlat),
            "meanlon": (["ntracks", "nmaxlength"], finaltrack_corecold_meanlon),
            "minlat": (["ntracks", "nmaxlength"], finaltrack_corecold_minlat),
            "minlon": (["ntracks", "nmaxlength"], finaltrack_corecold_minlon),
            "maxlat": (["ntracks", "nmaxlength"], finaltrack_corecold_maxlat),
            "maxlon": (["ntracks", "nmaxlength"], finaltrack_corecold_maxlon),
            "radius": (["ntracks", "nmaxlength"], finaltrack_corecold_radius),
            "radius_warmanvil": (
                ["ntracks", "nmaxlength"],
                finaltrack_corecoldwarm_radius,
            ),
            "npix": (["ntracks", "nmaxlength"], finaltrack_ncorecoldpix),
            "nconv": (["ntracks", "nmaxlength"], finaltrack_ncorepix),
            "ncoldanvil": (["ntracks", "nmaxlength"], finaltrack_ncoldpix),
            "nwarmanvil": (["ntracks", "nmaxlength"], finaltrack_nwarmpix),
            "cloudnumber": (["ntracks", "nmaxlength"], finaltrack_corecold_cloudnumber),
            "status": (["ntracks", "nmaxlength"], finaltrack_corecold_status),
            "startstatus": (["ntracks"], finaltrack_corecold_startstatus),
            "endstatus": (["ntracks"], finaltrack_corecold_endstatus),
            "mergenumbers": (
                ["ntracks", "nmaxlength"],
                adjusted_finaltrack_corecold_mergenumber,
            ),
            "splitnumbers": (
                ["ntracks", "nmaxlength"],
                adjusted_finaltrack_corecold_splitnumber,
            ),
            "trackinterruptions": (["ntracks"], finaltrack_corecold_trackinterruptions),
            "boundary": (["ntracks"], finaltrack_corecold_boundary),
            "minlwp": (["ntracks", "nmaxlength"], finaltrack_corecold_minlwp),
            "meanlwp": (["ntracks", "nmaxlength"], finaltrack_corecold_meanlwp),
            "meanlwp_conv": (
                ["ntracks", "nmaxlength"],
                finaltrack_core_meanlwp,
            ),  #'histlwp': (['ntracks', 'nmaxlength', 'nbins'], finaltrack_corecold_histlwp), \
            "majoraxis": (["ntracks", "nmaxlength"], finaltrack_corecold_majoraxis),
            "orientation": (["ntracks", "nmaxlength"], finaltrack_corecold_orientation),
            "eccentricity": (
                ["ntracks", "nmaxlength"],
                finaltrack_corecold_eccentricity,
            ),
            "perimeter": (["ntracks", "nmaxlength"], finaltrack_corecold_perimeter),
            "xcenter": (["ntracks", "nmaxlength"], finaltrack_corecold_xcenter),
            "ycenter": (["ntracks", "nmaxlength"], finaltrack_corecold_ycenter),
            "xcenter_weighted": (
                ["ntracks", "nmaxlength"],
                finaltrack_corecold_xweightedcenter,
            ),
            "ycenter_weighted": (
                ["ntracks", "nmaxlength"],
                finaltrack_corecold_yweightedcenter,
            ),
        },
        coords={
            "ntracks": (["ntracks"], np.arange(0, numtracks)),
            "nmaxlength": (["nmaxlength"], np.arange(0, maxtracklength)),
            "nbins": (["nbins"], np.arange(0, nbinlwp - 1)),
            "nfilenamechars": (["nfilenamechars"], np.arange(0, numcharfilename)),
            "ndatetimechars": (["ndatetimechars"], np.arange(0, 13)),
        },
        attrs={
            "title": "File containing statistics for each track",
            "Conventions": "CF-1.6",
            "Institution": "Pacific Northwest National Laboratoy",
            "Contact": "Hannah C Barnes: hannah.barnes@pnnl.gov",
            "Created_on": time.ctime(time.time()),
            "source": datasource,
            "description": datadescription,
            "startdate": startdate,
            "enddate": enddate,
            "track_version": track_version,
            "tracknumbers_version": tracknumbers_version,
            "timegap": str(timegap) + "-hr",
            "lwp_core": thresh_core,
            "lwp_coldanvil": thresh_cold,
            "pixel_radius_km": pixel_radius,
        },
    )

    # Specify variable attributes
    output_data.ntracks.attrs["long_name"] = "Total number of cloud tracks"
    output_data.ntracks.attrs["units"] = "unitless"

    output_data.nmaxlength.attrs["long_name"] = "Maximum length of a cloud track"
    output_data.nmaxlength.attrs["units"] = "unitless"

    output_data.lifetime.attrs["long_name"] = "duration of each track"
    output_data.lifetime.attrs["units"] = "Temporal resolution of data"
    output_data.lifetime.attrs["valid_min"] = lengthrange[0]

    output_data.basetime.attrs[
        "long_name"
    ] = "epoch time (seconds since 01/01/1970 00:00) of each cloud in a track"
    output_data.basetime.attrs["standard_name"] = "time"

    output_data.cloudidfiles.attrs[
        "long_name"
    ] = "File name for each cloud in each track"

    output_data.datetimestrings.attrs[
        "long_name"
    ] = "date_time for for each cloud in each track"

    output_data.meanlat.attrs[
        "long_name"
    ] = "Mean latitude of the core + cold anvil for each cloud in a track"
    output_data.meanlat.attrs["standard_name"] = "latitude"
    output_data.meanlat.attrs["units"] = "degrees_north"
    output_data.meanlat.attrs["valid_min"] = geolimits[1]
    output_data.meanlat.attrs["valid_max"] = geolimits[3]

    output_data.meanlon.attrs[
        "long_name"
    ] = "Mean longitude of the core + cold anvil for each cloud in a track"
    output_data.meanlon.attrs["standard_name"] = "longitude"
    output_data.meanlon.attrs["units"] = "degrees_east"
    output_data.meanlon.attrs["valid_min"] = geolimits[0]
    output_data.meanlon.attrs["valid_max"] = geolimits[2]

    output_data.minlat.attrs[
        "long_name"
    ] = "Minimum latitude of the core + cold anvil for each cloud in a track"
    output_data.minlat.attrs["standard_name"] = "latitude"
    output_data.minlat.attrs["units"] = "degrees_north"
    output_data.minlat.attrs["valid_min"] = geolimits[1]
    output_data.minlat.attrs["valid_max"] = geolimits[3]

    output_data.minlon.attrs[
        "long_name"
    ] = "Minimum longitude of the core + cold anvil for each cloud in a track"
    output_data.minlon.attrs["standard_name"] = "longitude"
    output_data.minlon.attrs["units"] = "degrees_east"
    output_data.minlon.attrs["valid_min"] = geolimits[0]
    output_data.minlon.attrs["valid_max"] = geolimits[2]

    output_data.maxlat.attrs[
        "long_name"
    ] = "Maximum latitude of the core + cold anvil for each cloud in a track"
    output_data.maxlat.attrs["standard_name"] = "latitude"
    output_data.maxlat.attrs["units"] = "degrees_north"
    output_data.maxlat.attrs["valid_min"] = geolimits[1]
    output_data.maxlat.attrs["valid_max"] = geolimits[3]

    output_data.maxlon.attrs[
        "long_name"
    ] = "Maximum longitude of the core + cold anvil for each cloud in a track"
    output_data.maxlon.attrs["standard_name"] = "longitude"
    output_data.maxlon.attrs["units"] = "degrees_east"
    output_data.maxlon.attrs["valid_min"] = geolimits[0]
    output_data.maxlon.attrs["valid_max"] = geolimits[2]

    output_data.radius.attrs[
        "long_name"
    ] = "Equivalent radius of the core + cold anvil for each cloud in a track"
    output_data.radius.attrs["standard_name"] = "Equivalent radius"
    output_data.radius.attrs["units"] = "km"
    output_data.radius.attrs["valid_min"] = areathresh

    output_data.radius_warmanvil.attrs[
        "long_name"
    ] = "Equivalent radius of the core + cold anvil  + warm anvil for each cloud in a track"
    output_data.radius_warmanvil.attrs["standard_name"] = "Equivalent radius"
    output_data.radius_warmanvil.attrs["units"] = "km"
    output_data.radius_warmanvil.attrs["valid_min"] = areathresh

    output_data.npix.attrs[
        "long_name"
    ] = "Number of pixels in the core + cold anvil for each cloud in a track"
    output_data.npix.attrs["units"] = "unitless"
    output_data.npix.attrs["valid_min"] = int(
        areathresh / float(np.square(pixel_radius))
    )

    output_data.nconv.attrs[
        "long_name"
    ] = "Number of pixels in the core for each cloud in a track"
    output_data.nconv.attrs["units"] = "unitless"
    output_data.nconv.attrs["valid_min"] = int(
        areathresh / float(np.square(pixel_radius))
    )

    output_data.ncoldanvil.attrs[
        "long_name"
    ] = "Number of pixels in the cold anvil for each cloud in a track"
    output_data.ncoldanvil.attrs["units"] = "unitless"
    output_data.ncoldanvil.attrs["valid_min"] = int(
        areathresh / float(np.square(pixel_radius))
    )

    output_data.nwarmanvil.attrs[
        "long_name"
    ] = "Number of pixels in the warm anvil for each cloud in a track"
    output_data.nwarmanvil.attrs["units"] = "unitless"
    output_data.nwarmanvil.attrs["valid_min"] = int(
        areathresh / float(np.square(pixel_radius))
    )

    output_data.cloudnumber.attrs[
        "long_name"
    ] = "Ccorresponding cloud identification number in cloudid file for each cloud in a track"
    output_data.cloudnumber.attrs["units"] = "unitless"
    output_data.cloudnumber.attrs[
        "usage"
    ] = "To link this tracking statistics file with corresponding pixel-level cloudid files, use the cloudidfile and cloudnumber together to identify which file and cloud this track is associated with at this time"

    output_data.status.attrs[
        "long_name"
    ] = "Flag indicating evolution / behavior for each cloud in a track"
    output_data.status.attrs["units"] = "unitless"
    output_data.status.attrs["valid_min"] = 0
    output_data.status.attrs["valid_max"] = 65

    output_data.startstatus.attrs[
        "long_name"
    ] = "Flag indicating how the first cloud in a track starts"
    output_data.startstatus.attrs["units"] = "unitless"
    output_data.startstatus.attrs["valid_min"] = 0
    output_data.startstatus.attrs["valid_max"] = 65

    output_data.endstatus.attrs[
        "long_name"
    ] = "Flag indicating how the last cloud in a track ends"
    output_data.endstatus.attrs["units"] = "unitless"
    output_data.endstatus.attrs["valid_min"] = 0
    output_data.endstatus.attrs["valid_max"] = 65

    output_data.trackinterruptions.attrs[
        "long_name"
    ] = "Flag indicating if track started or ended naturally or artifically due to data availability"
    output_data.trackinterruptions.attrs[
        "values"
    ] = "0 = full track available, good data. 1 = track starts at first file, track cut short by data availability. 2 = track ends at last file, track cut short by data availability"
    output_data.trackinterruptions.attrs["valid_min"] = 0
    output_data.trackinterruptions.attrs["valid_max"] = 2
    output_data.trackinterruptions.attrs["units"] = "unitless"

    output_data.mergenumbers.attrs[
        "long_name"
    ] = "Number of the track that this small cloud merges into"
    output_data.mergenumbers.attrs[
        "usuage"
    ] = "Each row represents a track. Each column represets a cloud in that track. Numbers give the track number that this small cloud mergesinto."
    output_data.mergenumbers.attrs["units"] = "unitless"
    output_data.mergenumbers.attrs["valid_min"] = 1
    output_data.mergenumbers.attrs["valid_max"] = numtracks

    output_data.splitnumbers.attrs[
        "long_name"
    ] = "Number of the track that this small cloud splits from"
    output_data.splitnumbers.attrs[
        "usuage"
    ] = "Each row represents a track. Each column represets a cloud in that track. Numbers give the track number that this small cloud splits from."
    output_data.splitnumbers.attrs["units"] = "unitless"
    output_data.splitnumbers.attrs["valid_min"] = 1
    output_data.splitnumbers.attrs["valid_max"] = numtracks

    output_data.boundary.attrs[
        "long_name"
    ] = "Flag indicating whether the core + cold anvil touches one of the domain edges."
    output_data.boundary.attrs["usuage"] = " 0 = away from edge. 1= touches edge."
    output_data.boundary.attrs["units"] = "unitless"
    output_data.boundary.attrs["valid_min"] = 0
    output_data.boundary.attrs["valid_max"] = 1

    output_data.minlwp.attrs[
        "long_name"
    ] = "Minimum liquid water path for each core + cold anvil in a track"
    output_data.minlwp.attrs["standard_name"] = "liquid water path"
    output_data.minlwp.attrs["units"] = "K"
    output_data.minlwp.attrs["valid_min"] = minlwp_thresh
    output_data.minlwp.attrs["valid_max"] = maxlwp_thresh

    output_data.meanlwp.attrs[
        "long_name"
    ] = "Mean liquid water path for each core + cold anvil in a track"
    output_data.meanlwp.attrs["standard_name"] = "liquid water path"
    output_data.meanlwp.attrs["units"] = "K"
    output_data.meanlwp.attrs["valid_min"] = minlwp_thresh
    output_data.meanlwp.attrs["valid_max"] = maxlwp_thresh

    output_data.meanlwp_conv.attrs[
        "long_name"
    ] = "Mean liquid water path for each core in a track"
    output_data.meanlwp_conv.attrs["standard_name"] = "liquid water path"
    output_data.meanlwp_conv.attrs["units"] = "K"
    output_data.meanlwp_conv.attrs["valid_min"] = minlwp_thresh
    output_data.meanlwp_conv.attrs["valid_max"] = maxlwp_thresh

    # output_data.histlwp.attrs['long_name'] = 'Histogram of the liquid water path of the core + cold anvil for each cloud in a track.'
    # output_data.histlwp.attrs['standard_name'] = 'liquid water path'
    # output_data.histlwp.attrs['hist_value'] = minlwp_thresh
    # output_data.histlwp.attrs['valid_max'] =  maxlwp_thresh
    # output_data.histlwp.attrs['units'] = 'K'

    output_data.orientation.attrs[
        "long_name"
    ] = "Orientation of the major axis of the core + cold anvil for each cloud in a track"
    output_data.orientation.attrs["units"] = "Degrees clockwise from vertical"
    output_data.orientation.attrs["valid_min"] = 0
    output_data.orientation.attrs["valid_max"] = 360

    output_data.eccentricity.attrs[
        "long_name"
    ] = "Eccentricity of the major axis of the core + cold anvil for each cloud in a track"
    output_data.eccentricity.attrs["units"] = "unitless"
    output_data.eccentricity.attrs["valid_min"] = 0
    output_data.eccentricity.attrs["valid_max"] = 1

    output_data.majoraxis.attrs[
        "long_name"
    ] = "Length of the major axis of the core + cold anvil for each cloud in a track"
    output_data.majoraxis.attrs["units"] = "km"

    output_data.perimeter.attrs[
        "long_name"
    ] = "Approximnate circumference of the core + cold anvil for each cloud in a track"
    output_data.perimeter.attrs["units"] = "km"

    output_data.xcenter.attrs[
        "long_name"
    ] = "X index of the geometric center of the cloud feature for each cloud in a track"
    output_data.xcenter.attrs["units"] = "unitless"

    output_data.ycenter.attrs[
        "long_name"
    ] = "Y index of the geometric center of the cloud feature for each cloud in a track"
    output_data.ycenter.attrs["units"] = "unitless"

    output_data.xcenter_weighted.attrs[
        "long_name"
    ] = "X index of the brightness temperature weighted center of the cloud feature for each cloud in a track"
    output_data.xcenter_weighted.attrs["units"] = "unitless"

    output_data.ycenter_weighted.attrs[
        "long_name"
    ] = "Y index of the brightness temperature weighted center of the cloud feature for each cloud in a track"
    output_data.ycenter_weighted.attrs["units"] = "unitless"

    # Write netcdf file
    output_data.to_netcdf(
        path=trackstats_outfile,
        mode="w",
        format="NETCDF4_CLASSIC",
        unlimited_dims="ntracks",
        encoding={
            "lifetime": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "basetime": {"zlib": True, "units": "seconds since 1970-01-01"},
            "ntracks": {"dtype": "int", "zlib": True},
            "nmaxlength": {"dtype": "int", "zlib": True},
            "cloudidfiles": {"zlib": True},
            "datetimestrings": {"zlib": True},
            "meanlat": {"zlib": True, "_FillValue": np.nan},
            "meanlon": {"zlib": True, "_FillValue": np.nan},
            "minlat": {"zlib": True, "_FillValue": np.nan},
            "minlon": {"zlib": True, "_FillValue": np.nan},
            "maxlat": {"zlib": True, "_FillValue": np.nan},
            "maxlon": {"zlib": True, "_FillValue": np.nan},
            "radius": {"zlib": True, "_FillValue": np.nan},
            "radius_warmanvil": {"zlib": True, "_FillValue": np.nan},
            "boundary": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "npix": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "nconv": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "ncoldanvil": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "nwarmanvil": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "cloudnumber": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "mergenumbers": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "splitnumbers": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "status": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "startstatus": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "endstatus": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "trackinterruptions": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "minlwp": {"zlib": True, "_FillValue": np.nan},
            "meanlwp": {"zlib": True, "_FillValue": np.nan},
            "meanlwp_conv": {
                "zlib": True,
                "_FillValue": np.nan,
            },  #'histlwp': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
            "majoraxis": {"zlib": True, "_FillValue": np.nan},
            "orientation": {"zlib": True, "_FillValue": np.nan},
            "eccentricity": {"zlib": True, "_FillValue": np.nan},
            "perimeter": {"zlib": True, "_FillValue": np.nan},
            "xcenter": {"zlib": True, "_FillValue": -9999},
            "ycenter": {"zlib": True, "_FillValue": -9999},
            "xcenter_weighted": {"zlib": True, "_FillValue": -9999},
            "ycenter_weighted": {"zlib": True, "_FillValue": -9999},
        },
    )


# Define function that calculates track statistics for WRF data
def trackstats_WRF(
    datasource,
    datadescription,
    pixel_radius,
    latlon_file,
    geolimits,
    areathresh,
    cloudtb_threshs,
    absolutetb_threshs,
    startdate,
    enddate,
    timegap,
    cloudid_filebase,
    tracking_inpath,
    stats_path,
    track_version,
    tracknumbers_version,
    tracknumbers_filebase,
    lengthrange=[5, 60],
):
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
    # lengthrange - Optional. Set this keyword to a vector [minlength,maxlength] to specify the lifetime range for the tracks.

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
    # xcenter_weighted - x-coordinate of the liquid water path weighted center
    # ycenter_weighted - y-coordinate of the liquid water path weighted center

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
    import xarray as xr
    import pandas as pd

    np.set_printoptions(threshold=np.inf)

    #############################################################################
    # Set constants

    # Isolate core and cold anvil brightness temperature thresholds
    thresh_core = cloudtb_threshs[0]
    thresh_cold = cloudtb_threshs[1]

    # Set output filename
    trackstats_outfile = (
        stats_path
        + "stats_"
        + tracknumbers_filebase
        + "_"
        + startdate
        + "_"
        + enddate
        + ".nc"
    )

    ###################################################################################
    # Load latitude and longitude grid. These were created in subroutine_idclouds and is saved in each file.

    # Find filenames of idcloud files
    temp_cloudidfiles = fnmatch.filter(
        os.listdir(tracking_inpath), cloudid_filebase + "*"
    )

    # Select one file. Any file is fine since they all havel the map of latitued and longitude saved.
    temp_cloudidfiles = temp_cloudidfiles[0]

    # Load latitude and longitude grid
    latlondata = Dataset(tracking_inpath + temp_cloudidfiles, "r")
    longitude = latlondata.variables["longitude"][:]
    latitude = latlondata.variables["latitude"][:]
    latlondata.close()

    #############################################################################
    # Load track data
    cloudtrack_file = (
        stats_path + tracknumbers_filebase + "_" + startdate + "_" + enddate + ".nc"
    )

    cloudtrackdata = Dataset(cloudtrack_file, "r")
    numtracks = cloudtrackdata["ntracks"][:]
    cloudidfiles = cloudtrackdata["cloudid_files"][:]
    #    import pdb; pdb.set_trace()
    tracknumbers = cloudtrackdata["track_numbers"][:]
    trackreset = cloudtrackdata["track_reset"][:]
    tracksplit = cloudtrackdata["track_splitnumbers"][:]
    trackmerge = cloudtrackdata["track_mergenumbers"][:]
    trackstatus = cloudtrackdata["track_status"][:]
    cloudtrackdata.close()

    # Convert filenames and timegap to string
    numcharfilename = len(list(cloudidfiles[0]))

    # Determine dimensions of data
    nfiles = len(cloudidfiles)
    ny, nx = np.shape(latitude)
    logger.info(nfiles)

    ############################################################################
    # Initialize grids
    nmaxclouds = max(lengthrange)

    minlwp_thresh = absolutelwp_threshs[0]
    maxlwp_thresh = absolutelwp_threshs[1]
    tbinterval = 2
    tbbins = np.arange(mintb_thresh, maxtb_thresh + tbinterval, tbinterval)
    nbintb = len(tbbins)

    finaltrack_tracklength = np.ones(int(numtracks), dtype=int) * -9999
    finaltrack_corecold_boundary = np.ones(int(numtracks)) * -9999
    finaltrack_basetime = np.zeros(
        (int(numtracks), int(nmaxclouds)), dtype="datetime64[s]"
    )
    finaltrack_corecold_mintb = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_meantb = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_core_meantb = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_histtb = np.zeros((int(numtracks), int(nmaxclouds), nbintb - 1))
    finaltrack_corecold_radius = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecoldwarm_radius = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_meanlat = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_meanlon = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_maxlon = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_maxlat = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_ncorecoldpix = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=int) * -9999
    )
    finaltrack_corecold_minlon = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_minlat = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_ncorepix = np.ones((int(numtracks), int(nmaxclouds)), dtype=int) * -9999
    finaltrack_ncoldpix = np.ones((int(numtracks), int(nmaxclouds)), dtype=int) * -9999
    finaltrack_nwarmpix = np.ones((int(numtracks), int(nmaxclouds)), dtype=int) * -9999
    finaltrack_corecold_status = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=int) * -9999
    )
    finaltrack_corecold_trackinterruptions = np.ones(int(numtracks), dtype=int) * -9999
    finaltrack_corecold_mergenumber = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=int) * -9999
    )
    finaltrack_corecold_splitnumber = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=int) * -9999
    )
    finaltrack_corecold_cloudnumber = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=int) * -9999
    )
    finaltrack_datetimestring = [
        [["" for x in range(13)] for y in range(int(nmaxclouds))]
        for z in range(int(numtracks))
    ]
    finaltrack_cloudidfile = np.chararray(
        (int(numtracks), int(nmaxclouds), int(numcharfilename))
    ) #TODO: JOE: Refactor all of this
    finaltrack_corecold_majoraxis = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_orientation = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_eccentricity = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_perimeter = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_xcenter = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * -9999
    )
    finaltrack_corecold_ycenter = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * -9999
    )
    finaltrack_corecold_xweightedcenter = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * -9999
    )
    finaltrack_corecold_yweightedcenter = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * -9999
    )

    #########################################################################################
    # loop over files. Calculate statistics and organize matrices by tracknumber and cloud
    for nf in range(0, nfiles):
        file_tracknumbers = tracknumbers[0, nf, :]
        # file_tracknumbers = cloudtrackdata['track_numbers'][0, nf, :]
        #        import pdb; pdb.set_trace()
        # Only process file if that file contains a track
        if np.nanmax(file_tracknumbers) > 0:
            fname = "".join(chartostring(cloudidfiles[nf]))
            #            logger.info((''.join(cloudidfiles[nf])))
            logger.info(nf, fname)

            # Load cloudid file
            #            cloudid_file = tracking_inpath + ''.join(cloudidfiles[nf])
            #            cloudid_file = tracking_inpath + ''.join(chartostring(cloudidfiles[nf]))
            cloudid_file = tracking_inpath + fname

            file_cloudiddata = Dataset(cloudid_file, "r")
            file_tb = file_cloudiddata["tb"][:]
            file_cloudtype = file_cloudiddata["cloudtype"][:]
            file_all_cloudnumber = file_cloudiddata["cloudnumber"][:]
            file_corecold_cloudnumber = file_cloudiddata["convcold_cloudnumber"][:]
            file_basetime = file_cloudiddata["basetime"][:]
            basetime_units = file_cloudiddata["basetime"].units
            basetime_calendar = file_cloudiddata["basetime"].calendar
            file_cloudiddata.close()

            file_datetimestring = cloudid_file[
                len(tracking_inpath) + len(cloudid_filebase) : -3
            ]

            # Find unique track numbers
            uniquetracknumbers = np.unique(file_tracknumbers)
            uniquetracknumbers = uniquetracknumbers[uniquetracknumbers > 0].astype(int)

            # Loop over unique tracknumbers
            for itrack in uniquetracknumbers:

                # Find cloud number that belongs to the current track in this file
                cloudnumber = (
                    np.array(np.where(file_tracknumbers == itrack))[0, :] + 1
                )  # Finds cloud numbers associated with that track. Need to add one since tells index, which starts at 0, and we want the number, which starts at one
                cloudindex = cloudnumber - 1  # Index within the matrice of this cloud.

                if (
                    len(cloudnumber) == 1
                ):  # Should only be one cloud number. In mergers and split, the associated clouds should be listed in the file_splittracknumbers and file_mergetracknumbers
                    # Find cloud in cloudid file associated with this track
                    corearea = np.array(
                        np.where(
                            (file_corecold_cloudnumber[0, :, :] == cloudnumber)
                            & (file_cloudtype[0, :, :] == 1)
                        )
                    )
                    ncorepix = np.shape(corearea)[1]

                    coldarea = np.array(
                        np.where(
                            (file_corecold_cloudnumber[0, :, :] == cloudnumber)
                            & (file_cloudtype[0, :, :] == 2)
                        )
                    )
                    ncoldpix = np.shape(coldarea)[1]

                    warmarea = np.array(
                        np.where(
                            (file_all_cloudnumber[0, :, :] == cloudnumber)
                            & (file_cloudtype[0, :, :] == 3)
                        )
                    )
                    nwarmpix = np.shape(warmarea)[1]

                    corecoldarea = np.array(
                        np.where(
                            (file_corecold_cloudnumber[0, :, :] == cloudnumber)
                            & (file_cloudtype[0, :, :] >= 1)
                            & (file_cloudtype[0, :, :] <= 2)
                        )
                    )
                    ncorecoldpix = np.shape(corecoldarea)[1]

                    # Find current length of the track. Use for indexing purposes. Also, record the current length the given track.
                    lengthindex = np.array(
                        np.where(finaltrack_corecold_cloudnumber[itrack - 1, :] > 0)
                    )
                    if np.shape(lengthindex)[1] > 0:
                        nc = np.nanmax(lengthindex) + 1
                    else:
                        nc = 0
                    finaltrack_tracklength[itrack - 1] = (
                        nc + 1
                    )  # Need to add one since array index starts at 0

                    if nc < nmaxclouds:
                        # Save information that links this cloud back to its raw pixel level data
                        finaltrack_basetime[itrack - 1, nc] = np.array(
                            [
                                pd.to_datetime(
                                    num2date(
                                        file_basetime,
                                        units=basetime_units,
                                        calendar=basetime_calendar,
                                    )
                                )
                            ],
                            dtype="datetime64[s]",
                        )[0, 0]
                        # finaltrack_basetime[itrack-1, nc] = np.datetime64(pd.to_datetime(file_cloudiddata['basetime'].data)[0])
                        finaltrack_corecold_cloudnumber[itrack - 1, nc] = cloudnumber
                        finaltrack_cloudidfile[itrack - 1][nc][:] = fname
                        #                        finaltrack_cloudidfile[itrack-1][nc][:] = list(cloudidfiles[nf])
                        finaltrack_datetimestring[int(itrack - 1)][int(nc)][
                            :
                        ] = file_datetimestring

                        ###############################################################
                        # Calculate statistics about this cloud system

                        #############
                        # Location statistics of core+cold anvil (aka the convective system)
                        corecoldlat = latitude[corecoldarea[0], corecoldarea[1]]
                        corecoldlon = longitude[corecoldarea[0], corecoldarea[1]]

                        finaltrack_corecold_meanlat[itrack - 1, nc] = np.nanmean(
                            corecoldlat
                        )
                        finaltrack_corecold_meanlon[itrack - 1, nc] = np.nanmean(
                            corecoldlon
                        )

                        finaltrack_corecold_minlat[itrack - 1, nc] = np.nanmin(
                            corecoldlat
                        )
                        finaltrack_corecold_minlon[itrack - 1, nc] = np.nanmin(
                            corecoldlon
                        )

                        finaltrack_corecold_maxlat[itrack - 1, nc] = np.nanmax(
                            corecoldlat
                        )
                        finaltrack_corecold_maxlon[itrack - 1, nc] = np.nanmax(
                            corecoldlon
                        )

                        # Determine if core+cold touches of the boundaries of the domain
                        if (
                            np.absolute(
                                finaltrack_corecold_minlat[itrack - 1, nc]
                                - geolimits[0]
                            )
                            < 0.1
                            or np.absolute(
                                finaltrack_corecold_maxlat[itrack - 1, nc]
                                - geolimits[2]
                            )
                            < 0.1
                            or np.absolute(
                                finaltrack_corecold_minlon[itrack - 1, nc]
                                - geolimits[1]
                            )
                            < 0.1
                            or np.absolute(
                                finaltrack_corecold_maxlon[itrack - 1, nc]
                                - geolimits[3]
                            )
                            < 0.1
                        ):
                            finaltrack_corecold_boundary[itrack - 1] = 1

                        ############
                        # Save number of pixels (metric for size)
                        finaltrack_ncorecoldpix[itrack - 1, nc] = ncorecoldpix
                        finaltrack_ncorepix[itrack - 1, nc] = ncorepix
                        finaltrack_ncoldpix[itrack - 1, nc] = ncoldpix
                        finaltrack_nwarmpix[itrack - 1, nc] = nwarmpix

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
                            maxyindex = ny

                        if np.nanmin(corecoldarea[1]) - pad > 0:
                            minxindex = np.nanmin(corecoldarea[1]) - pad
                        else:
                            minxindex = 0

                        if np.nanmax(corecoldarea[1]) + pad < nx - 1:
                            maxxindex = np.nanmax(corecoldarea[1]) + pad + 1
                        else:
                            maxxindex = nx

                        # Isolate the region around the cloud using the padded region
                        isolatedcloudnumber = np.copy(
                            file_corecold_cloudnumber[
                                0, minyindex:maxyindex, minxindex:maxxindex
                            ]
                        ).astype(int)
                        isolatedtb = np.copy(
                            file_lwp[0, minyindex:maxyindex, minxindex:maxxindex]
                        )

                        # Remove brightness temperatures outside core + cold anvil
                        isolatedtb[isolatedcloudnumber != cloudnumber] = 0

                        # Turn cloud map to binary
                        isolatedcloudnumber[isolatedcloudnumber != cloudnumber] = 0
                        isolatedcloudnumber[isolatedcloudnumber == cloudnumber] = 1

                        # Calculate major axis, orientation, eccentricity
                        cloudproperities = regionprops(
                            isolatedcloudnumber, intensity_image=isolatedlwp
                        )

                        finaltrack_corecold_eccentricity[
                            itrack - 1, nc
                        ] = cloudproperities[0].eccentricity
                        finaltrack_corecold_majoraxis[itrack - 1, nc] = (
                            cloudproperities[0].major_axis_length * pixel_radius
                        )
                        finaltrack_corecold_orientation[itrack - 1, nc] = (
                            cloudproperities[0].orientation
                        ) * (180 / float(pi))
                        finaltrack_corecold_perimeter[itrack - 1, nc] = (
                            cloudproperities[0].perimeter * pixel_radius
                        )
                        [temp_ycenter, temp_xcenter] = cloudproperities[0].centroid
                        [
                            finaltrack_corecold_ycenter[itrack - 1, nc],
                            finaltrack_corecold_xcenter[itrack - 1, nc],
                        ] = np.add(
                            [temp_ycenter, temp_xcenter], [minyindex, minxindex]
                        ).astype(
                            int
                        )
                        [temp_yweightedcenter, temp_xweightedcenter] = cloudproperities[
                            0
                        ].weighted_centroid
                        [
                            finaltrack_corecold_yweightedcenter[itrack - 1, nc],
                            finaltrack_corecold_xweightedcenter[itrack - 1, nc],
                        ] = np.add(
                            [temp_yweightedcenter, temp_xweightedcenter],
                            [minyindex, minxindex],
                        ).astype(
                            int
                        )

                        # Determine equivalent radius of core+cold. Assuming circular area = (number pixels)*(pixel radius)^2, equivalent radius = sqrt(Area / pi)
                        finaltrack_corecold_radius[itrack - 1, nc] = np.sqrt(
                            np.divide(ncorecoldpix * (np.square(pixel_radius)), pi)
                        )

                        # Determine equivalent radius of core+cold+warm. Assuming circular area = (number pixels)*(pixel radius)^2, equivalent radius = sqrt(Area / pi)
                        finaltrack_corecoldwarm_radius[itrack - 1, nc] = np.sqrt(
                            np.divide(
                                (ncorepix + ncoldpix + nwarmpix)
                                * (np.square(pixel_radius)),
                                pi,
                            )
                        )

                        ##############################################################
                        # Calculate brightness temperature statistics of core+cold anvil
                        corecoldtb = np.copy(
                            file_tb[0, corecoldarea[0], corecoldarea[1]]
                        )

                        finaltrack_corecold_mintb[itrack - 1, nc] = np.nanmin(
                            corecoldtb
                        )
                        finaltrack_corecold_meantb[itrack - 1, nc] = np.nanmean(
                            corecoldtb
                        )

                        ################################################################
                        # Histogram of brightness temperature for core+cold anvil
                        (
                            finaltrack_corecold_histtb[itrack - 1, nc, :],
                            usedtbbins,
                        ) = np.histogram(
                            corecoldtb, range=(mintb_thresh, maxtb_thresh), bins=tbbins
                        )

                        # Save track information. Need to subtract one since cloudnumber gives the number of the cloud (which starts at one), but we are looking for its index (which starts at zero)
                        finaltrack_corecold_status[itrack - 1, nc] = np.copy(
                            trackstatus[0, nf, cloudindex]
                        )
                        finaltrack_corecold_mergenumber[itrack - 1, nc] = np.copy(
                            trackmerge[0, nf, cloudindex]
                        )
                        finaltrack_corecold_splitnumber[itrack - 1, nc] = np.copy(
                            tracksplit[0, nf, cloudindex]
                        )
                        finaltrack_corecold_trackinterruptions[itrack - 1] = np.copy(
                            trackreset[0, nf, cloudindex]
                        )

                        ####################################################################
                        # Calculate mean brightness temperature for core
                        coretb = np.copy(file_tb[0, coldarea[0], coldarea[1]])

                        finaltrack_core_meantb[itrack - 1, nc] = np.nanmean(coretb)

                    else:
                        logger.info(
                            (
                                "Track: "
                                + str(itrack)
                                + "; "
                                + str(nc)
                                + " greater than maximum allowed number clouds, "
                                + str(nmaxclouds)
                            )
                        )
                        nc = nc + 1

                elif len(cloudnumber) > 1:
                    sys.exit(
                        str(cloudnumber)
                        + " clouds linked to one track. Each track should only be linked to one cloud in each file in the track_number array. The track_number variable only tracks the largest cell in mergers and splits. The small clouds in tracks and mergers should only be listed in the track_splitnumbers and track_mergenumbers arrays."
                    )

    ###############################################################
    ## Remove tracks that have no cells. These tracks are short.

    logger.info("Removing tracks with no cells")
    gc.collect()

    cloudindexpresent = np.array(np.where(finaltrack_tracklength != -9999))[0, :]
    numtracks = len(cloudindexpresent)

    maxtracklength = np.nanmin([np.nanmax(finaltrack_tracklength), int(nmaxclouds)])

    finaltrack_tracklength = finaltrack_tracklength[cloudindexpresent]
    finaltrack_corecold_boundary = finaltrack_corecold_boundary[cloudindexpresent]
    finaltrack_basetime = finaltrack_basetime[cloudindexpresent, 0:maxtracklength]
    finaltrack_corecold_mintb = finaltrack_corecold_mintb[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_meantb = finaltrack_corecold_meantb[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_core_meantb = finaltrack_core_meantb[cloudindexpresent, 0:maxtracklength]
    finaltrack_corecold_histtb = finaltrack_corecold_histtb[
        cloudindexpresent, 0:maxtracklength, :
    ]
    finaltrack_corecold_radius = finaltrack_corecold_radius[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecoldwarm_radius = finaltrack_corecoldwarm_radius[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_meanlat = finaltrack_corecold_meanlat[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_meanlon = finaltrack_corecold_meanlon[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_maxlon = finaltrack_corecold_maxlon[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_maxlat = finaltrack_corecold_maxlat[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_minlon = finaltrack_corecold_minlon[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_minlat = finaltrack_corecold_minlat[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_ncorecoldpix = finaltrack_ncorecoldpix[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_ncorepix = finaltrack_ncorepix[cloudindexpresent, 0:maxtracklength]
    finaltrack_ncoldpix = finaltrack_ncoldpix[cloudindexpresent, 0:maxtracklength]
    finaltrack_nwarmpix = finaltrack_nwarmpix[cloudindexpresent, 0:maxtracklength]
    finaltrack_corecold_status = finaltrack_corecold_status[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_trackinterruptions = finaltrack_corecold_trackinterruptions[
        cloudindexpresent
    ]
    finaltrack_corecold_mergenumber = finaltrack_corecold_mergenumber[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_splitnumber = finaltrack_corecold_splitnumber[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_cloudnumber = finaltrack_corecold_cloudnumber[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_datetimestring = list(
        finaltrack_datetimestring[i][0:maxtracklength][:] for i in cloudindexpresent
    )
    finaltrack_cloudidfile = finaltrack_cloudidfile[
        cloudindexpresent, 0:maxtracklength, :
    ]
    finaltrack_corecold_majoraxis = finaltrack_corecold_majoraxis[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_orientation = finaltrack_corecold_orientation[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_eccentricity = finaltrack_corecold_eccentricity[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_perimeter = finaltrack_corecold_perimeter[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_xcenter = finaltrack_corecold_xcenter[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_ycenter = finaltrack_corecold_ycenter[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_xweightedcenter = finaltrack_corecold_xweightedcenter[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_yweightedcenter = finaltrack_corecold_yweightedcenter[
        cloudindexpresent, 0:maxtracklength
    ]

    gc.collect()
    logger.info("Tracks with no cells removed")
    logger.info(("Number of Tracks:" + str(int(len(finaltrack_tracklength)))))

    ########################################################
    # Correct merger and split cloud numbers
    logger.info("Correcting merger and split numbers")

    # Initialize adjusted matrices
    adjusted_finaltrack_corecold_mergenumber = (
        np.ones(np.shape(finaltrack_corecold_mergenumber)) * -9999
    )
    adjusted_finaltrack_corecold_splitnumber = (
        np.ones(np.shape(finaltrack_corecold_mergenumber)) * -9999
    )
    logger.info(("total tracks: " + str(numtracks)))

    # Create adjustor
    indexcloudnumber = np.copy(cloudindexpresent) + 1
    adjustor = np.arange(0, np.max(cloudindexpresent) + 2)
    for it in range(0, numtracks):
        adjustor[indexcloudnumber[it]] = it + 1
    adjustor = np.append(adjustor, -9999)

    # Adjust mergers
    temp_finaltrack_corecold_mergenumber = finaltrack_corecold_mergenumber.astype(
        int
    ).ravel()
    temp_finaltrack_corecold_mergenumber[
        temp_finaltrack_corecold_mergenumber == -9999
    ] = (np.max(cloudindexpresent) + 2)
    adjusted_finaltrack_corecold_mergenumber = adjustor[
        temp_finaltrack_corecold_mergenumber
    ]
    adjusted_finaltrack_corecold_mergenumber = np.reshape(
        adjusted_finaltrack_corecold_mergenumber,
        np.shape(finaltrack_corecold_mergenumber),
    )

    # Adjust splitters
    temp_finaltrack_corecold_splitnumber = finaltrack_corecold_splitnumber.astype(
        int
    ).ravel()
    temp_finaltrack_corecold_splitnumber[
        temp_finaltrack_corecold_splitnumber == -9999
    ] = (np.max(cloudindexpresent) + 2)
    adjusted_finaltrack_corecold_splitnumber = adjustor[
        temp_finaltrack_corecold_splitnumber
    ]
    adjusted_finaltrack_corecold_splitnumber = np.reshape(
        adjusted_finaltrack_corecold_splitnumber,
        np.shape(finaltrack_corecold_splitnumber),
    )

    logger.info("Adjustment done")

    #########################################################################
    # Record starting and ending status
    logger.info("Isolating starting and ending status")

    # Starting status
    finaltrack_corecold_startstatus = np.copy(finaltrack_corecold_status[:, 0])

    # Ending status
    finaltrack_corecold_endstatus = (
        np.ones(len(finaltrack_corecold_startstatus)) * -9999
    )
    #    for trackstep in range(0, maxtracklength):
    for trackstep in range(0, numtracks):
        if (finaltrack_tracklength[trackstep] > 0) & (
            finaltrack_tracklength[trackstep] < maxtracklength
        ):
            finaltrack_corecold_endstatus[trackstep] = np.copy(
                finaltrack_corecold_status[
                    trackstep, finaltrack_tracklength[trackstep] - 1
                ]
            )

    #######################################################################
    # Write to netcdf
    gc.collect()

    logger.info("Writing trackstat netcdf")
    logger.info(trackstats_outfile)
    logger.info("")

    # Check if file already exists. If exists, delete
    if os.path.isfile(trackstats_outfile):
        os.remove(trackstats_outfile)

    # Define xarray dataset
    output_data = xr.Dataset(
        {
            "lifetime": (["ntracks"], finaltrack_tracklength),
            "basetime": (["ntracks", "nmaxlength"], finaltrack_basetime),
            "cloudidfiles": (
                ["ntracks", "nmaxlength", "nfilenamechars"],
                finaltrack_cloudidfile,
            ),
            "datetimestrings": (
                ["ntracks", "nmaxlength", "ndatetimechars"],
                finaltrack_datetimestring,
            ),
            "meanlat": (["ntracks", "nmaxlength"], finaltrack_corecold_meanlat),
            "meanlon": (["ntracks", "nmaxlength"], finaltrack_corecold_meanlon),
            "minlat": (["ntracks", "nmaxlength"], finaltrack_corecold_minlat),
            "minlon": (["ntracks", "nmaxlength"], finaltrack_corecold_minlon),
            "maxlat": (["ntracks", "nmaxlength"], finaltrack_corecold_maxlat),
            "maxlon": (["ntracks", "nmaxlength"], finaltrack_corecold_maxlon),
            "radius": (["ntracks", "nmaxlength"], finaltrack_corecold_radius),
            "radius_warmanvil": (
                ["ntracks", "nmaxlength"],
                finaltrack_corecoldwarm_radius,
            ),
            "npix": (["ntracks", "nmaxlength"], finaltrack_ncorecoldpix),
            "nconv": (["ntracks", "nmaxlength"], finaltrack_ncorepix),
            "ncoldanvil": (["ntracks", "nmaxlength"], finaltrack_ncoldpix),
            "nwarmanvil": (["ntracks", "nmaxlength"], finaltrack_nwarmpix),
            "cloudnumber": (["ntracks", "nmaxlength"], finaltrack_corecold_cloudnumber),
            "status": (["ntracks", "nmaxlength"], finaltrack_corecold_status),
            "startstatus": (["ntracks"], finaltrack_corecold_startstatus),
            "endstatus": (["ntracks"], finaltrack_corecold_endstatus),
            "mergenumbers": (
                ["ntracks", "nmaxlength"],
                adjusted_finaltrack_corecold_mergenumber,
            ),
            "splitnumbers": (
                ["ntracks", "nmaxlength"],
                adjusted_finaltrack_corecold_splitnumber,
            ),
            "trackinterruptions": (["ntracks"], finaltrack_corecold_trackinterruptions),
            "boundary": (["ntracks"], finaltrack_corecold_boundary),
            "mintb": (["ntracks", "nmaxlength"], finaltrack_corecold_mintb),
            "meantb": (["ntracks", "nmaxlength"], finaltrack_corecold_meantb),
            "meantb_conv": (["ntracks", "nmaxlength"], finaltrack_core_meantb),
            "histtb": (["ntracks", "nmaxlength", "nbins"], finaltrack_corecold_histtb),
            "majoraxis": (["ntracks", "nmaxlength"], finaltrack_corecold_majoraxis),
            "orientation": (["ntracks", "nmaxlength"], finaltrack_corecold_orientation),
            "eccentricity": (
                ["ntracks", "nmaxlength"],
                finaltrack_corecold_eccentricity,
            ),
            "perimeter": (["ntracks", "nmaxlength"], finaltrack_corecold_perimeter),
            "xcenter": (["ntracks", "nmaxlength"], finaltrack_corecold_xcenter),
            "ycenter": (["ntracks", "nmaxlength"], finaltrack_corecold_ycenter),
            "xcenter_weighted": (
                ["ntracks", "nmaxlength"],
                finaltrack_corecold_xweightedcenter,
            ),
            "ycenter_weighted": (
                ["ntracks", "nmaxlength"],
                finaltrack_corecold_yweightedcenter,
            ),
        },
        coords={
            "ntracks": (["ntracks"], np.arange(0, numtracks)),
            "nmaxlength": (["nmaxlength"], np.arange(0, maxtracklength)),
            "nbins": (["nbins"], np.arange(0, nbinlwp - 1)),
            "nfilenamechars": (["nfilenamechars"], np.arange(0, numcharfilename)),
            "ndatetimechars": (["ndatetimechars"], np.arange(0, 13)),
        },
        attrs={
            "title": "File containing statistics for each track",
            "Conventions": "CF-1.6",
            "Institution": "Pacific Northwest National Laboratoy",
            "Contact": "Katelyn Barber: katelyn.barber@pnnl.gov",
            "Created_on": time.ctime(time.time()),
            "source": datasource,
            "description": datadescription,
            "startdate": startdate,
            "enddate": enddate,
            "track_version": track_version,
            "tracknumbers_version": tracknumbers_version,
            "timegap": str(timegap) + "-hr",
            "tb_core": thresh_core,
            "tb_coldanvil": thresh_cold,
            "pixel_radius_km": pixel_radius,
        },
    )

    # Specify variable attributes
    output_data.ntracks.attrs["long_name"] = "Total number of cloud tracks"
    output_data.ntracks.attrs["units"] = "unitless"

    output_data.nmaxlength.attrs["long_name"] = "Maximum length of a cloud track"
    output_data.nmaxlength.attrs["units"] = "unitless"

    output_data.lifetime.attrs["long_name"] = "duration of each track"
    output_data.lifetime.attrs["units"] = "Temporal resolution of data"
    output_data.lifetime.attrs["valid_min"] = lengthrange[0]

    output_data.basetime.attrs[
        "long_name"
    ] = "epoch time (seconds since 01/01/1970 00:00) of each cloud in a track"
    output_data.basetime.attrs["standard_name"] = "time"

    output_data.cloudidfiles.attrs[
        "long_name"
    ] = "File name for each cloud in each track"

    output_data.datetimestrings.attrs[
        "long_name"
    ] = "date_time for for each cloud in each track"

    output_data.meanlat.attrs[
        "long_name"
    ] = "Mean latitude of the core + cold anvil for each cloud in a track"
    output_data.meanlat.attrs["standard_name"] = "latitude"
    output_data.meanlat.attrs["units"] = "degrees_north"
    output_data.meanlat.attrs["valid_min"] = geolimits[1]
    output_data.meanlat.attrs["valid_max"] = geolimits[3]

    output_data.meanlon.attrs[
        "long_name"
    ] = "Mean longitude of the core + cold anvil for each cloud in a track"
    output_data.meanlon.attrs["standard_name"] = "longitude"
    output_data.meanlon.attrs["units"] = "degrees_east"
    output_data.meanlon.attrs["valid_min"] = geolimits[0]
    output_data.meanlon.attrs["valid_max"] = geolimits[2]

    output_data.minlat.attrs[
        "long_name"
    ] = "Minimum latitude of the core + cold anvil for each cloud in a track"
    output_data.minlat.attrs["standard_name"] = "latitude"
    output_data.minlat.attrs["units"] = "degrees_north"
    output_data.minlat.attrs["valid_min"] = geolimits[1]
    output_data.minlat.attrs["valid_max"] = geolimits[3]

    output_data.minlon.attrs[
        "long_name"
    ] = "Minimum longitude of the core + cold anvil for each cloud in a track"
    output_data.minlon.attrs["standard_name"] = "longitude"
    output_data.minlon.attrs["units"] = "degrees_east"
    output_data.minlon.attrs["valid_min"] = geolimits[0]
    output_data.minlon.attrs["valid_max"] = geolimits[2]

    output_data.maxlat.attrs[
        "long_name"
    ] = "Maximum latitude of the core + cold anvil for each cloud in a track"
    output_data.maxlat.attrs["standard_name"] = "latitude"
    output_data.maxlat.attrs["units"] = "degrees_north"
    output_data.maxlat.attrs["valid_min"] = geolimits[1]
    output_data.maxlat.attrs["valid_max"] = geolimits[3]

    output_data.maxlon.attrs[
        "long_name"
    ] = "Maximum longitude of the core + cold anvil for each cloud in a track"
    output_data.maxlon.attrs["standard_name"] = "longitude"
    output_data.maxlon.attrs["units"] = "degrees_east"
    output_data.maxlon.attrs["valid_min"] = geolimits[0]
    output_data.maxlon.attrs["valid_max"] = geolimits[2]

    output_data.radius.attrs[
        "long_name"
    ] = "Equivalent radius of the core + cold anvil for each cloud in a track"
    output_data.radius.attrs["standard_name"] = "Equivalent radius"
    output_data.radius.attrs["units"] = "km"
    output_data.radius.attrs["valid_min"] = areathresh

    output_data.radius_warmanvil.attrs[
        "long_name"
    ] = "Equivalent radius of the core + cold anvil  + warm anvil for each cloud in a track"
    output_data.radius_warmanvil.attrs["standard_name"] = "Equivalent radius"
    output_data.radius_warmanvil.attrs["units"] = "km"
    output_data.radius_warmanvil.attrs["valid_min"] = areathresh

    output_data.npix.attrs[
        "long_name"
    ] = "Number of pixels in the core + cold anvil for each cloud in a track"
    output_data.npix.attrs["units"] = "unitless"
    output_data.npix.attrs["valid_min"] = int(
        areathresh / float(np.square(pixel_radius))
    )

    output_data.nconv.attrs[
        "long_name"
    ] = "Number of pixels in the core for each cloud in a track"
    output_data.nconv.attrs["units"] = "unitless"
    output_data.nconv.attrs["valid_min"] = int(
        areathresh / float(np.square(pixel_radius))
    )

    output_data.ncoldanvil.attrs[
        "long_name"
    ] = "Number of pixels in the cold anvil for each cloud in a track"
    output_data.ncoldanvil.attrs["units"] = "unitless"
    output_data.ncoldanvil.attrs["valid_min"] = int(
        areathresh / float(np.square(pixel_radius))
    )

    output_data.nwarmanvil.attrs[
        "long_name"
    ] = "Number of pixels in the warm anvil for each cloud in a track"
    output_data.nwarmanvil.attrs["units"] = "unitless"
    output_data.nwarmanvil.attrs["valid_min"] = int(
        areathresh / float(np.square(pixel_radius))
    )

    output_data.cloudnumber.attrs[
        "long_name"
    ] = "Corresponding cloud identification number in cloudid file for each cloud in a track"
    output_data.cloudnumber.attrs["units"] = "unitless"
    output_data.cloudnumber.attrs[
        "usage"
    ] = "To link this tracking statistics file with corresponding pixel-level cloudid files, use the cloudidfile and cloudnumber together to identify which file and cloud this track is associated with at this time"

    output_data.status.attrs[
        "long_name"
    ] = "Flag indicating evolution / behavior for each cloud in a track"
    output_data.status.attrs["units"] = "unitless"
    output_data.status.attrs["valid_min"] = 0
    output_data.status.attrs["valid_max"] = 65

    output_data.startstatus.attrs[
        "long_name"
    ] = "Flag indicating how the first cloud in a track starts"
    output_data.startstatus.attrs["units"] = "unitless"
    output_data.startstatus.attrs["valid_min"] = 0
    output_data.startstatus.attrs["valid_max"] = 65

    output_data.endstatus.attrs[
        "long_name"
    ] = "Flag indicating how the last cloud in a track ends"
    output_data.endstatus.attrs["units"] = "unitless"
    output_data.endstatus.attrs["valid_min"] = 0
    output_data.endstatus.attrs["valid_max"] = 65

    output_data.trackinterruptions.attrs[
        "long_name"
    ] = "Flag indicating if track started or ended naturally or artifically due to data availability"
    output_data.trackinterruptions.attrs[
        "values"
    ] = "0 = full track available, good data. 1 = track starts at first file, track cut short by data availability. 2 = track ends at last file, track cut short by data availability"
    output_data.trackinterruptions.attrs["valid_min"] = 0
    output_data.trackinterruptions.attrs["valid_max"] = 2
    output_data.trackinterruptions.attrs["units"] = "unitless"

    output_data.mergenumbers.attrs[
        "long_name"
    ] = "Number of the track that this small cloud merges into"
    output_data.mergenumbers.attrs[
        "usuage"
    ] = "Each row represents a track. Each column represets a cloud in that track. Numbers give the track number that this small cloud mergesinto."
    output_data.mergenumbers.attrs["units"] = "unitless"
    output_data.mergenumbers.attrs["valid_min"] = 1
    output_data.mergenumbers.attrs["valid_max"] = numtracks

    output_data.splitnumbers.attrs[
        "long_name"
    ] = "Number of the track that this small cloud splits from"
    output_data.splitnumbers.attrs[
        "usuage"
    ] = "Each row represents a track. Each column represets a cloud in that track. Numbers give the track number that this small cloud splits from."
    output_data.splitnumbers.attrs["units"] = "unitless"
    output_data.splitnumbers.attrs["valid_min"] = 1
    output_data.splitnumbers.attrs["valid_max"] = numtracks

    output_data.boundary.attrs[
        "long_name"
    ] = "Flag indicating whether the core + cold anvil touches one of the domain edges."
    output_data.boundary.attrs["usuage"] = " 0 = away from edge. 1= touches edge."
    output_data.boundary.attrs["units"] = "unitless"
    output_data.boundary.attrs["valid_min"] = 0
    output_data.boundary.attrs["valid_max"] = 1

    output_data.minlwp.attrs[
        "long_name"
    ] = "Minimum brightness temperature for each core + cold anvil in a track"
    output_data.minlwp.attrs["standard_name"] = "brightness temperature"
    output_data.minlwp.attrs["units"] = "K"
    output_data.minlwp.attrs["valid_min"] = mintb_thresh
    output_data.minlwp.attrs["valid_max"] = maxtb_thresh

    output_data.meanlwp.attrs[
        "long_name"
    ] = "Mean brightness temperature for each core + cold anvil in a track"
    output_data.meanlwp.attrs["standard_name"] = "brightness temperature"
    output_data.meanlwp.attrs["units"] = "K"
    output_data.meanlwp.attrs["valid_min"] = mintb_thresh
    output_data.meanlwp.attrs["valid_max"] = maxtb_thresh

    output_data.meanlwp_conv.attrs[
        "long_name"
    ] = "Mean brightness temperature for each core in a track"
    output_data.meanlwp_conv.attrs["standard_name"] = "brightness temperature"
    output_data.meanlwp_conv.attrs["units"] = "K"
    output_data.meanlwp_conv.attrs["valid_min"] = mintb_thresh
    output_data.meanlwp_conv.attrs["valid_max"] = maxtb_thresh

    output_data.histlwp.attrs[
        "long_name"
    ] = "Histogram of the brightness temperature of the core + cold anvil for each cloud in a track."
    output_data.histlwp.attrs["standard_name"] = "brightness temperature"
    output_data.histlwp.attrs["hist_value"] = mintb_thresh
    output_data.histlwp.attrs["valid_max"] = maxtb_thresh
    output_data.histlwp.attrs["units"] = "K"

    output_data.orientation.attrs[
        "long_name"
    ] = "Orientation of the major axis of the core + cold anvil for each cloud in a track"
    output_data.orientation.attrs["units"] = "Degrees clockwise from vertical"
    output_data.orientation.attrs["valid_min"] = 0
    output_data.orientation.attrs["valid_max"] = 360

    output_data.eccentricity.attrs[
        "long_name"
    ] = "Eccentricity of the major axis of the core + cold anvil for each cloud in a track"
    output_data.eccentricity.attrs["units"] = "unitless"
    output_data.eccentricity.attrs["valid_min"] = 0
    output_data.eccentricity.attrs["valid_max"] = 1

    output_data.majoraxis.attrs[
        "long_name"
    ] = "Length of the major axis of the core + cold anvil for each cloud in a track"
    output_data.majoraxis.attrs["units"] = "km"

    output_data.perimeter.attrs[
        "long_name"
    ] = "Approximnate circumference of the core + cold anvil for each cloud in a track"
    output_data.perimeter.attrs["units"] = "km"

    output_data.xcenter.attrs[
        "long_name"
    ] = "X index of the geometric center of the cloud feature for each cloud in a track"
    output_data.xcenter.attrs["units"] = "unitless"

    output_data.ycenter.attrs[
        "long_name"
    ] = "Y index of the geometric center of the cloud feature for each cloud in a track"
    output_data.ycenter.attrs["units"] = "unitless"

    output_data.xcenter_weighted.attrs[
        "long_name"
    ] = "X index of the brightness temperature weighted center of the cloud feature for each cloud in a track"
    output_data.xcenter_weighted.attrs["units"] = "unitless"

    output_data.ycenter_weighted.attrs[
        "long_name"
    ] = "Y index of the brightness temperature weighted center of the cloud feature for each cloud in a track"
    output_data.ycenter_weighted.attrs["units"] = "unitless"

    # Write netcdf file
    output_data.to_netcdf(
        path=trackstats_outfile,
        mode="w",
        format="NETCDF4_CLASSIC",
        unlimited_dims="ntracks",
        encoding={
            "lifetime": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "basetime": {"zlib": True, "units": "seconds since 1970-01-01"},
            "ntracks": {"dtype": "int", "zlib": True},
            "nmaxlength": {"dtype": "int", "zlib": True},
            "cloudidfiles": {"zlib": True},
            "datetimestrings": {"zlib": True},
            "meanlat": {"zlib": True, "_FillValue": np.nan},
            "meanlon": {"zlib": True, "_FillValue": np.nan},
            "minlat": {"zlib": True, "_FillValue": np.nan},
            "minlon": {"zlib": True, "_FillValue": np.nan},
            "maxlat": {"zlib": True, "_FillValue": np.nan},
            "maxlon": {"zlib": True, "_FillValue": np.nan},
            "radius": {"zlib": True, "_FillValue": np.nan},
            "radius_warmanvil": {"zlib": True, "_FillValue": np.nan},
            "boundary": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "npix": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "nconv": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "ncoldanvil": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "nwarmanvil": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "cloudnumber": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "mergenumbers": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "splitnumbers": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "status": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "startstatus": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "endstatus": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "trackinterruptions": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "mintb": {"zlib": True, "_FillValue": np.nan},
            "meantb": {"zlib": True, "_FillValue": np.nan},
            "meantb_conv": {"zlib": True, "_FillValue": np.nan},
            "histtb": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "majoraxis": {"zlib": True, "_FillValue": np.nan},
            "orientation": {"zlib": True, "_FillValue": np.nan},
            "eccentricity": {"zlib": True, "_FillValue": np.nan},
            "perimeter": {"zlib": True, "_FillValue": np.nan},
            "xcenter": {"zlib": True, "_FillValue": -9999},
            "ycenter": {"zlib": True, "_FillValue": -9999},
            "xcenter_weighted": {"zlib": True, "_FillValue": -9999},
            "ycenter_weighted": {"zlib": True, "_FillValue": -9999},
        },
    )


# Purpose: This gets statistics about each track from the satellite data.

# Author: Orginial IDL version written by Sally A. McFarline (sally.mcfarlane@pnnl.gov) and modified for Zhe Feng (zhe.feng@pnnl.gov). Python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov)

# Define function that calculates track statistics for satellite data
def trackstats_ct(
    datasource,
    datadescription,
    pixel_radius,
    geolimits,
    areathresh,
    cloudtb_threshs,
    absolutetb_threshs,
    startdate,
    enddate,
    timegap,
    cloudid_filebase,
    tracking_inpath,
    stats_path,
    track_version,
    tracknumbers_version,
    tracknumbers_filebase,
    lengthrange=[2, 120],
):
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
    import os
    import sys
    from math import pi
    from skimage.measure import regionprops
    import time
    import gc
    import pandas as pd

    np.set_printoptions(threshold=np.inf)

    #############################################################################
    # Set constants

    # Set output filename
    trackstats_outfile = (
        stats_path
        + "stats_"
        + tracknumbers_filebase
        + "_"
        + startdate
        + "_"
        + enddate
        + ".nc"
    )

    ###################################################################################
    # # Load latitude and longitude grid. These were created in subroutine_idclouds and is saved in each file.
    # logger.info('Determining which files will be processed')
    # logger.info((time.ctime()))

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
    logger.info("Loading gettracks data")
    logger.info((time.ctime()))
    cloudtrack_file = (
        stats_path + tracknumbers_filebase + "_" + startdate + "_" + enddate + ".nc"
    )

    cloudtrackdata = Dataset(cloudtrack_file, "r")
    numtracks = cloudtrackdata["ntracks"][:]
    cloudidfiles = cloudtrackdata["cloudid_files"][:]
    nfiles = cloudtrackdata.dimensions["nfiles"].size
    tracknumbers = cloudtrackdata["track_numbers"][:]
    trackreset = cloudtrackdata["track_reset"][:]
    tracksplit = cloudtrackdata["track_splitnumbers"][:]
    trackmerge = cloudtrackdata["track_mergenumbers"][:]
    trackstatus = cloudtrackdata["track_status"][:]
    cloudtrackdata.close()

    # Convert filenames and timegap to string
    # numcharfilename = len(list(cloudidfiles_list[0]))
    tmpfname = "".join(chartostring(cloudidfiles[0]))
    numcharfilename = len(list(tmpfname))

    # Load latitude and longitude grid from any cloudidfile since they all have the map of latitude and longitude saved
    latlondata = Dataset(tracking_inpath + tmpfname, "r")
    longitude = latlondata.variables["longitude"][:]
    latitude = latlondata.variables["latitude"][:]
    latlondata.close()

    # Determine dimensions of data
    # nfiles = len(cloudidfiles_list)
    ny, nx = np.shape(latitude)

    ############################################################################
    # Initialize grids
    logger.info("Initiailizinng matrices")
    logger.info((time.ctime()))

    nmaxclouds = max(lengthrange)

    finaltrack_tracklength = np.ones(int(numtracks), dtype=np.int32) * -9999
    finaltrack_corecold_boundary = np.ones(int(numtracks), dtype=np.int32) * -9999
    finaltrack_basetime = np.empty(
        (int(numtracks), int(nmaxclouds)), dtype="datetime64[s]"
    )

    finaltrack_corecold_radius = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )

    finaltrack_corecold_meanlat = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_meanlon = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_maxlon = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_maxlat = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_ncorecoldpix = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=np.int32) * -9999
    )
    finaltrack_corecold_minlon = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_minlat = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_ncorepix = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=np.int32) * -9999
    )
    finaltrack_ncoldpix = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=np.int32) * -9999
    )

    finaltrack_corecold_status = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=np.int32) * -9999
    )
    finaltrack_corecold_trackinterruptions = (
        np.ones(int(numtracks), dtype=np.int32) * -9999
    )
    finaltrack_corecold_mergenumber = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=np.int32) * -9999
    )
    finaltrack_corecold_splitnumber = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=np.int32) * -9999
    )
    finaltrack_corecold_cloudnumber = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=np.int32) * -9999
    )
    # finaltrack_cloudtype = np.ones((int(numtracks),int(nmaxclouds)), dtype=np.int32)*-9999
    finaltrack_cloudtype_low = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=np.int32) * -9999
    )
    finaltrack_cloudtype_conglow = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=np.int32) * -9999
    )
    finaltrack_cloudtype_conghigh = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=np.int32) * -9999
    )
    finaltrack_cloudtype_deep = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=np.int32) * -9999
    )
    finaltrack_datetimestring = [
        [["" for x in range(13)] for y in range(int(nmaxclouds))]
        for z in range(int(numtracks))
    ]
    finaltrack_cloudidfile = np.chararray(
        (int(numtracks), int(nmaxclouds), int(numcharfilename))
    )
    finaltrack_corecold_majoraxis = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_orientation = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_eccentricity = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_perimeter = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * np.nan
    )
    finaltrack_corecold_xcenter = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * -9999
    )
    finaltrack_corecold_ycenter = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * -9999
    )
    finaltrack_corecold_xweightedcenter = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * -9999
    )
    finaltrack_corecold_yweightedcenter = (
        np.ones((int(numtracks), int(nmaxclouds)), dtype=float) * -9999
    )

    #########################################################################################
    # loop over files. Calculate statistics and organize matrices by tracknumber and cloud
    logger.info("Looping over files and calculating statistics for each file")
    logger.info((time.ctime()))
    for nf in range(0, nfiles - 1):
        # for nf in range(0, 2):
        # logger.info(('File #: ' + str(nf)))
        # logger.info((time.ctime()))

        file_tracknumbers = tracknumbers[0, nf, :]

        # Only process file if that file contains a track
        if np.nanmax(file_tracknumbers) > 0:

            fname = "".join(chartostring(cloudidfiles[nf]))
            logger.info(nf, fname)

            # Load cloudid file
            cloudid_file = tracking_inpath + fname
            # logger.info(cloudid_file)

            file_cloudiddata = Dataset(cloudid_file, "r")
            file_ct = file_cloudiddata["ct"][:]
            file_cloudtype = file_cloudiddata["original_cloudtype"][:]
            file_corecold_cloudnumber = file_cloudiddata["convcold_cloudnumber"][:]
            file_basetime = file_cloudiddata["basetime"][:]
            basetime_units = file_cloudiddata["basetime"].units
            basetime_calendar = file_cloudiddata["basetime"].calendar
            file_cloudiddata.close()

            file_datetimestring = cloudid_file[
                len(tracking_inpath) + len(cloudid_filebase) : -3
            ]

            # Find unique track numbers
            uniquetracknumbers = np.unique(file_tracknumbers)
            uniquetracknumbers = uniquetracknumbers[np.isfinite(uniquetracknumbers)]
            uniquetracknumbers = uniquetracknumbers[uniquetracknumbers > 0].astype(int)

            # Loop over unique tracknumbers
            # logger.info('Loop over tracks in file')
            for itrack in uniquetracknumbers:
                # logger.info(('Unique track number: ' + str(itrack)))
                # logger.info('itrack: ', itrack)

                # Find cloud number that belongs to the current track in this file
                cloudnumber = (
                    np.array(np.where(file_tracknumbers == itrack))[0, :] + 1
                )  # Finds cloud numbers associated with that track. Need to add one since tells index, which starts at 0, and we want the number, which starts at one
                cloudindex = cloudnumber - 1  # Index within the matrice of this cloud.

                if (
                    len(cloudnumber) == 1
                ):  # Should only be one cloud number. In mergers and split, the associated clouds should be listed in the file_splittracknumbers and file_mergetracknumbers
                    # Find cloud in cloudid file associated with this track
                    corearea = np.array(
                        np.where(
                            (file_corecold_cloudnumber[0, :, :] == cloudnumber)
                            & (file_cloudtype[0, :, :] == 4)
                        )
                    )
                    ncorepix = np.shape(corearea)[1]

                    coldarea = np.array(
                        np.where(
                            (file_corecold_cloudnumber[0, :, :] == cloudnumber)
                            & (file_cloudtype[0, :, :] == 3)
                        )
                    )
                    ncoldpix = np.shape(coldarea)[1]

                    corecoldarea = np.array(
                        np.where(
                            (file_corecold_cloudnumber[0, :, :] == cloudnumber)
                            & (
                                (file_cloudtype[0, :, :] == 4)
                                | (file_cloudtype[0, :, :] == 3)
                            )
                        )
                    )
                    ncorecoldpix = np.shape(corecoldarea)[1]

                    # Find cloud type that belongs to the current track in this file
                    cn_wherej, cn_wherei = np.array(
                        np.where(file_corecold_cloudnumber[0, :, :] == cloudnumber)
                    )
                    cloudtype_track = file_cloudtype[:, cn_wherej, cn_wherei]
                    n_lowpix = np.count_nonzero(cloudtype_track == 1)
                    n_conglowpix = np.count_nonzero(cloudtype_track == 2)
                    n_conghighpix = np.count_nonzero(cloudtype_track == 3)
                    n_deeppix = np.count_nonzero(cloudtype_track == 4)

                    # Find current length of the track. Use for indexing purposes. Also, record the current length the given track.
                    lengthindex = np.array(
                        np.where(finaltrack_corecold_cloudnumber[itrack - 1, :] > 0)
                    )
                    if np.shape(lengthindex)[1] > 0:
                        nc = np.nanmax(lengthindex) + 1
                    else:
                        nc = 0
                    finaltrack_tracklength[itrack - 1] = (
                        nc + 1
                    )  # Need to add one since array index starts at 0

                    if nc < nmaxclouds:
                        # Save information that links this cloud back to its raw pixel level data
                        finaltrack_basetime[itrack - 1, nc] = np.array(
                            [
                                pd.to_datetime(
                                    num2date(
                                        file_basetime,
                                        units=basetime_units,
                                        calendar=basetime_calendar,
                                    )
                                )
                            ],
                            dtype="datetime64[s]",
                        )[0, 0]
                        finaltrack_corecold_cloudnumber[itrack - 1, nc] = cloudnumber
                        # finaltrack_cloudidfile[itrack-1][nc][:] = list(cloudidfiles_list[nf])
                        finaltrack_cloudidfile[itrack - 1][nc][:] = fname
                        finaltrack_datetimestring[int(itrack - 1)][int(nc)][
                            :
                        ] = file_datetimestring
                        # finaltrack_cloudtype[itrack-1,nc] = cloudtype_track
                        # if (nf == 6) & (itrack == 713):
                        #     import pdb; pdb.set_trace()
                        ###############################################################
                        # Calculate statistics about this cloud system
                        # 11/21/2019 - Make sure this cloud exists
                        if ncorecoldpix > 0:
                            # Location statistics of core+cold anvil (aka the convective system)
                            corecoldlat = latitude[corecoldarea[0], corecoldarea[1]]
                            corecoldlon = longitude[corecoldarea[0], corecoldarea[1]]

                            finaltrack_corecold_meanlat[itrack - 1, nc] = np.nanmean(
                                corecoldlat
                            )
                            finaltrack_corecold_meanlon[itrack - 1, nc] = np.nanmean(
                                corecoldlon
                            )

                            finaltrack_corecold_minlat[itrack - 1, nc] = np.nanmin(
                                corecoldlat
                            )
                            finaltrack_corecold_minlon[itrack - 1, nc] = np.nanmin(
                                corecoldlon
                            )

                            finaltrack_corecold_maxlat[itrack - 1, nc] = np.nanmax(
                                corecoldlat
                            )
                            finaltrack_corecold_maxlon[itrack - 1, nc] = np.nanmax(
                                corecoldlon
                            )

                            # Determine if core+cold touches of the boundaries of the domain
                            if (
                                np.absolute(
                                    finaltrack_corecold_minlat[itrack - 1, nc]
                                    - geolimits[0]
                                )
                                < 0.1
                                or np.absolute(
                                    finaltrack_corecold_maxlat[itrack - 1, nc]
                                    - geolimits[2]
                                )
                                < 0.1
                                or np.absolute(
                                    finaltrack_corecold_minlon[itrack - 1, nc]
                                    - geolimits[1]
                                )
                                < 0.1
                                or np.absolute(
                                    finaltrack_corecold_maxlon[itrack - 1, nc]
                                    - geolimits[3]
                                )
                                < 0.1
                            ):
                                finaltrack_corecold_boundary[itrack - 1] = 1

                            # Save number of pixels (metric for size)
                            finaltrack_ncorecoldpix[itrack - 1, nc] = ncorecoldpix
                            finaltrack_ncorepix[itrack - 1, nc] = ncorepix
                            finaltrack_ncoldpix[itrack - 1, nc] = ncoldpix
                            finaltrack_cloudtype_low[itrack - 1, nc] = n_lowpix
                            finaltrack_cloudtype_conglow[itrack - 1, nc] = n_conglowpix
                            finaltrack_cloudtype_conghigh[
                                itrack - 1, nc
                            ] = n_conghighpix
                            finaltrack_cloudtype_deep[itrack - 1, nc] = n_deeppix

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
                                maxyindex = ny

                            if np.nanmin(corecoldarea[1]) - pad > 0:
                                minxindex = np.nanmin(corecoldarea[1]) - pad
                            else:
                                minxindex = 0

                            if np.nanmax(corecoldarea[1]) + pad < nx - 1:
                                maxxindex = np.nanmax(corecoldarea[1]) + pad + 1
                            else:
                                maxxindex = nx

                            # Isolate the region around the cloud using the padded region
                            isolatedcloudnumber = np.copy(
                                file_corecold_cloudnumber[
                                    0, minyindex:maxyindex, minxindex:maxxindex
                                ]
                            ).astype(int)
                            isolatedtb = np.copy(
                                file_ct[0, minyindex:maxyindex, minxindex:maxxindex]
                            )

                            # Remove brightness temperatures outside core + cold anvil
                            isolatedtb[isolatedcloudnumber != cloudnumber] = 0

                            # Turn cloud map to binary
                            isolatedcloudnumber[isolatedcloudnumber != cloudnumber] = 0
                            isolatedcloudnumber[isolatedcloudnumber == cloudnumber] = 1

                            # Calculate major axis, orientation, eccentricity
                            cloudproperities = regionprops(
                                isolatedcloudnumber, intensity_image=isolatedtb
                            )

                            finaltrack_corecold_eccentricity[
                                itrack - 1, nc
                            ] = cloudproperities[0].eccentricity
                            finaltrack_corecold_majoraxis[itrack - 1, nc] = (
                                cloudproperities[0].major_axis_length * pixel_radius
                            )
                            finaltrack_corecold_orientation[itrack - 1, nc] = (
                                cloudproperities[0].orientation
                            ) * (180 / float(pi))
                            finaltrack_corecold_perimeter[itrack - 1, nc] = (
                                cloudproperities[0].perimeter * pixel_radius
                            )
                            [temp_ycenter, temp_xcenter] = cloudproperities[0].centroid
                            [
                                finaltrack_corecold_ycenter[itrack - 1, nc],
                                finaltrack_corecold_xcenter[itrack - 1, nc],
                            ] = np.add(
                                [temp_ycenter, temp_xcenter], [minyindex, minxindex]
                            ).astype(
                                int
                            )
                            [
                                temp_yweightedcenter,
                                temp_xweightedcenter,
                            ] = cloudproperities[0].weighted_centroid
                            [
                                finaltrack_corecold_yweightedcenter[itrack - 1, nc],
                                finaltrack_corecold_xweightedcenter[itrack - 1, nc],
                            ] = np.add(
                                [temp_yweightedcenter, temp_xweightedcenter],
                                [minyindex, minxindex],
                            ).astype(
                                int
                            )

                            # Determine equivalent radius of core+cold. Assuming circular area = (number pixels)*(pixel radius)^2, equivalent radius = sqrt(Area / pi)
                            finaltrack_corecold_radius[itrack - 1, nc] = np.sqrt(
                                np.divide(ncorecoldpix * (np.square(pixel_radius)), pi)
                            )

                            ################################################################
                            # Save track information. Need to subtract one since cloudnumber gives the number of the cloud (which starts at one), but we are looking for its index (which starts at zero)
                            finaltrack_corecold_status[itrack - 1, nc] = np.copy(
                                trackstatus[0, nf, cloudindex]
                            )
                            finaltrack_corecold_mergenumber[itrack - 1, nc] = np.copy(
                                trackmerge[0, nf, cloudindex]
                            )
                            finaltrack_corecold_splitnumber[itrack - 1, nc] = np.copy(
                                tracksplit[0, nf, cloudindex]
                            )
                            finaltrack_corecold_trackinterruptions[
                                itrack - 1
                            ] = np.copy(trackreset[0, nf, cloudindex])

                            # logger.info('shape of finaltrack_corecold_status: ', finaltrack_corecold_status.shape)

                    else:
                        sys.exit(
                            str(nc)
                            + " greater than maximum allowed number clouds, "
                            + str(nmaxclouds)
                        )

                elif len(cloudnumber) > 1:
                    sys.exit(
                        str(cloudnumber)
                        + " clouds linked to one track. Each track should only be linked to one cloud in each file in the track_number array. The track_number variable only tracks the largest cell in mergers and splits. The small clouds in tracks and mergers should only be listed in the track_splitnumbers and track_mergenumbers arrays."
                    )

    ###############################################################
    ## Remove tracks that have no cells. These tracks are short.
    logger.info("Removing tracks with no cells")
    logger.info((time.ctime()))
    gc.collect()

    # logger.info('finaltrack_tracklength shape at line 385: ', finaltrack_tracklength.shape)
    # logger.info('finaltrack_tracklength(4771): ', finaltrack_tracklength[4770])
    cloudindexpresent = np.array(np.where(finaltrack_tracklength != -9999))[0, :]
    numtracks = len(cloudindexpresent)
    # logger.info('length of cloudindex present: ', len(cloudindexpresent))

    maxtracklength = np.nanmax(finaltrack_tracklength)
    # logger.info('maxtracklength: ', maxtracklength)

    finaltrack_tracklength = finaltrack_tracklength[cloudindexpresent]
    finaltrack_corecold_boundary = finaltrack_corecold_boundary[cloudindexpresent]
    finaltrack_basetime = finaltrack_basetime[cloudindexpresent, 0:maxtracklength]
    finaltrack_corecold_radius = finaltrack_corecold_radius[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_meanlat = finaltrack_corecold_meanlat[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_meanlon = finaltrack_corecold_meanlon[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_maxlon = finaltrack_corecold_maxlon[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_maxlat = finaltrack_corecold_maxlat[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_minlon = finaltrack_corecold_minlon[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_minlat = finaltrack_corecold_minlat[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_ncorecoldpix = finaltrack_ncorecoldpix[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_ncorepix = finaltrack_ncorepix[cloudindexpresent, 0:maxtracklength]
    finaltrack_ncoldpix = finaltrack_ncoldpix[cloudindexpresent, 0:maxtracklength]
    finaltrack_corecold_status = finaltrack_corecold_status[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_trackinterruptions = finaltrack_corecold_trackinterruptions[
        cloudindexpresent
    ]
    finaltrack_corecold_mergenumber = finaltrack_corecold_mergenumber[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_splitnumber = finaltrack_corecold_splitnumber[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_cloudnumber = finaltrack_corecold_cloudnumber[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_datetimestring = list(
        finaltrack_datetimestring[i][0:maxtracklength][:] for i in cloudindexpresent
    )
    finaltrack_cloudidfile = finaltrack_cloudidfile[
        cloudindexpresent, 0:maxtracklength, :
    ]
    finaltrack_corecold_majoraxis = finaltrack_corecold_majoraxis[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_orientation = finaltrack_corecold_orientation[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_eccentricity = finaltrack_corecold_eccentricity[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_perimeter = finaltrack_corecold_perimeter[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_xcenter = finaltrack_corecold_xcenter[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_ycenter = finaltrack_corecold_ycenter[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_xweightedcenter = finaltrack_corecold_xweightedcenter[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_corecold_yweightedcenter = finaltrack_corecold_yweightedcenter[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_cloudtype_low = finaltrack_cloudtype_low[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_cloudtype_conglow = finaltrack_cloudtype_conglow[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_cloudtype_conghigh = finaltrack_cloudtype_conghigh[
        cloudindexpresent, 0:maxtracklength
    ]
    finaltrack_cloudtype_deep = finaltrack_cloudtype_deep[
        cloudindexpresent, 0:maxtracklength
    ]

    gc.collect()

    ########################################################
    # Correct merger and split cloud numbers

    # Initialize adjusted matrices
    adjusted_finaltrack_corecold_mergenumber = (
        np.ones(np.shape(finaltrack_corecold_mergenumber)) * -9999
    )
    adjusted_finaltrack_corecold_splitnumber = (
        np.ones(np.shape(finaltrack_corecold_mergenumber)) * -9999
    )
    logger.info(("total tracks: " + str(numtracks)))
    logger.info("Correcting mergers and splits")
    logger.info((time.ctime()))

    # Create adjustor
    indexcloudnumber = np.copy(cloudindexpresent) + 1
    adjustor = np.arange(0, np.max(cloudindexpresent) + 2)
    for it in range(0, numtracks):
        adjustor[indexcloudnumber[it]] = it + 1
    adjustor = np.append(adjustor, -9999)

    # Adjust mergers
    temp_finaltrack_corecold_mergenumber = finaltrack_corecold_mergenumber.astype(
        int
    ).ravel()
    temp_finaltrack_corecold_mergenumber[
        temp_finaltrack_corecold_mergenumber == -9999
    ] = (np.max(cloudindexpresent) + 2)
    adjusted_finaltrack_corecold_mergenumber = adjustor[
        temp_finaltrack_corecold_mergenumber
    ]
    adjusted_finaltrack_corecold_mergenumber = np.reshape(
        adjusted_finaltrack_corecold_mergenumber,
        np.shape(finaltrack_corecold_mergenumber),
    )

    # Adjust splitters
    temp_finaltrack_corecold_splitnumber = finaltrack_corecold_splitnumber.astype(
        int
    ).ravel()
    temp_finaltrack_corecold_splitnumber[
        temp_finaltrack_corecold_splitnumber == -9999
    ] = (np.max(cloudindexpresent) + 2)
    adjusted_finaltrack_corecold_splitnumber = adjustor[
        temp_finaltrack_corecold_splitnumber
    ]
    adjusted_finaltrack_corecold_splitnumber = np.reshape(
        adjusted_finaltrack_corecold_splitnumber,
        np.shape(finaltrack_corecold_splitnumber),
    )

    #########################################################################
    # Record starting and ending status
    logger.info("Determine starting and ending status")
    logger.info((time.ctime()))

    # Starting status
    finaltrack_corecold_startstatus = finaltrack_corecold_status[:, 0]

    # Ending status
    finaltrack_corecold_endstatus = (
        np.ones(len(finaltrack_corecold_startstatus)) * -9999
    )
    for trackstep in range(0, numtracks):
        if finaltrack_tracklength[trackstep] > 0:
            finaltrack_corecold_endstatus[trackstep] = finaltrack_corecold_status[
                trackstep, finaltrack_tracklength[trackstep] - 1
            ]

    #######################################################################
    # Write to netcdf
    logger.info("Writing trackstat netcdf")
    logger.info((time.ctime()))
    logger.info(trackstats_outfile)
    logger.info("")

    # Check if file already exists. If exists, delete
    if os.path.isfile(trackstats_outfile):
        os.remove(trackstats_outfile)

    from pyflextrkr import netcdf_io as net

    net.write_trackstats_ct(
        trackstats_outfile,
        numtracks,
        maxtracklength,
        numcharfilename,
        datasource,
        datadescription,
        startdate,
        enddate,
        track_version,
        tracknumbers_version,
        timegap,
        pixel_radius,
        geolimits,
        areathresh,
        basetime_units,
        basetime_calendar,
        finaltrack_tracklength,
        finaltrack_basetime,
        finaltrack_cloudidfile,
        finaltrack_datetimestring,
        finaltrack_corecold_meanlat,
        finaltrack_corecold_meanlon,
        finaltrack_corecold_minlat,
        finaltrack_corecold_minlon,
        finaltrack_corecold_maxlat,
        finaltrack_corecold_maxlon,
        finaltrack_corecold_radius,
        finaltrack_ncorecoldpix,
        finaltrack_ncorepix,
        finaltrack_ncoldpix,
        finaltrack_corecold_cloudnumber,
        finaltrack_corecold_status,
        finaltrack_corecold_startstatus,
        finaltrack_corecold_endstatus,
        adjusted_finaltrack_corecold_mergenumber,
        adjusted_finaltrack_corecold_splitnumber,
        finaltrack_corecold_trackinterruptions,
        finaltrack_corecold_boundary,
        finaltrack_corecold_majoraxis,
        finaltrack_corecold_orientation,
        finaltrack_corecold_eccentricity,
        finaltrack_corecold_perimeter,
        finaltrack_corecold_xcenter,
        finaltrack_corecold_ycenter,
        finaltrack_corecold_xweightedcenter,
        finaltrack_corecold_yweightedcenter,
        finaltrack_cloudtype_low,
        finaltrack_cloudtype_conglow,
        finaltrack_cloudtype_conghigh,
        finaltrack_cloudtype_deep,
    )
