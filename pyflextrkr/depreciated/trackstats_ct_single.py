def calc_stats_single(
    tracknumbers,
    cloudidfiles,
    tracking_inpath,
    cloudid_filebase,
    numcharfilename,
    latitude,
    longitude,
    geolimits,
    nx,
    ny,
    pixel_radius,
    trackstatus,
    trackmerge,
    tracksplit,
    trackreset,
):
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
    import logging

    logger = logging.getLogger(__name__)

    file_tracknumbers = tracknumbers

    # Only process file if that file contains a track
    if np.nanmax(file_tracknumbers) > 0:

        fname = "".join(chartostring(cloudidfiles))
        logger.info(fname)

        # Load cloudid file
        cloudid_file = tracking_inpath + fname
        # logger.info(cloudid_file)

        file_cloudiddata = Dataset(cloudid_file, "r")
        # file_tb = file_cloudiddata['tb'][:]
        file_cloudtype = file_cloudiddata["ct"][:]
        logger.info("shape file_cloudtype: ", file_cloudtype.shape)
        # file_all_cloudnumber = file_cloudiddata['cloudnumber'][:]
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

        numtracks = len(uniquetracknumbers)
        finaltrack_corecold_boundary = np.ones(int(numtracks), dtype=np.int32) * -9999
        finaltrack_basetime = np.empty(int(numtracks), dtype="datetime64[s]")
        finaltrack_corecold_radius = np.ones(int(numtracks), dtype=float) * np.nan
        finaltrack_corecold_meanlat = np.ones(int(numtracks), dtype=float) * np.nan
        finaltrack_corecold_meanlon = np.ones(int(numtracks), dtype=float) * np.nan
        finaltrack_corecold_maxlon = np.ones(int(numtracks), dtype=float) * np.nan
        finaltrack_corecold_maxlat = np.ones(int(numtracks), dtype=float) * np.nan
        finaltrack_ncorecoldpix = np.ones(int(numtracks), dtype=np.int32) * -9999
        finaltrack_corecold_minlon = np.ones(int(numtracks), dtype=float) * np.nan
        finaltrack_corecold_minlat = np.ones(int(numtracks), dtype=float) * np.nan
        finaltrack_ncorepix = np.ones(int(numtracks), dtype=np.int32) * -9999
        finaltrack_ncoldpix = np.ones(int(numtracks), dtype=np.int32) * -9999
        finaltrack_corecold_status = np.ones(int(numtracks), dtype=np.int32) * -9999
        finaltrack_corecold_trackinterruptions = (
            np.ones(int(numtracks), dtype=np.int32) * -9999
        )
        finaltrack_corecold_mergenumber = (
            np.ones(int(numtracks), dtype=np.int32) * -9999
        )
        finaltrack_corecold_splitnumber = (
            np.ones(int(numtracks), dtype=np.int32) * -9999
        )
        finaltrack_corecold_cloudnumber = (
            np.ones(int(numtracks), dtype=np.int32) * -9999
        )
        finaltrack_datetimestring = [
            ["" for x in range(13)] for z in range(int(numtracks))
        ]
        finaltrack_cloudidfile = np.chararray((int(numtracks), int(numcharfilename)))
        finaltrack_corecold_majoraxis = np.ones(int(numtracks), dtype=float) * np.nan
        finaltrack_corecold_orientation = np.ones(int(numtracks), dtype=float) * np.nan
        finaltrack_corecold_eccentricity = np.ones(int(numtracks), dtype=float) * np.nan
        finaltrack_corecold_perimeter = np.ones(int(numtracks), dtype=float) * np.nan
        finaltrack_corecold_xcenter = np.ones(int(numtracks), dtype=float) * -9999
        finaltrack_corecold_ycenter = np.ones(int(numtracks), dtype=float) * -9999
        finaltrack_corecold_xweightedcenter = (
            np.ones(int(numtracks), dtype=float) * -9999
        )
        finaltrack_corecold_yweightedcenter = (
            np.ones(int(numtracks), dtype=float) * -9999
        )
        finaltrack_cloudtype_low = np.ones(int(numtracks), dtype=np.int32) * -9999
        finaltrack_cloudtype_conglow = np.ones(int(numtracks), dtype=np.int32) * -9999
        finaltrack_cloudtype_conghigh = np.ones(int(numtracks), dtype=np.int32) * -9999
        finaltrack_cloudtype_deep = np.ones(int(numtracks), dtype=np.int32) * -9999

        # Loop over unique tracknumbers
        # logger.info('Loop over tracks in file')
        # for itrack in uniquetracknumbers:
        for itrack in range(numtracks):
            # logger.info(('Unique track number: ' + str(itrack)))
            # logger.info('itrack: ', itrack)

            # Find cloud number that belongs to the current track in this file
            cloudnumber = (
                np.array(np.where(file_tracknumbers == uniquetracknumbers[itrack]))[
                    0, :
                ]
                + 1
            )  # Finds cloud numbers associated with that track. Need to add one since tells index, which starts at 0, and we want the number, which starts at one
            cloudindex = cloudnumber - 1  # Index within the matrice of this cloud.

            if (
                len(cloudnumber) == 1
            ):  # Should only be one cloud number. In mergers and split, the associated clouds should be listed in the file_splittracknumbers and file_mergetracknumbers
                # Find cloud in cloudid file associated with this track
                corearea = np.array(
                    np.where(
                        (file_corecold_cloudnumber[:, :] == cloudnumber)
                        & (file_cloudtype[:, :] == 4)
                    )
                )
                ncorepix = np.shape(corearea)[1]

                coldarea = np.array(
                    np.where(
                        (file_corecold_cloudnumber[:, :] == cloudnumber)
                        & (file_cloudtype[:, :] == 3)
                    )
                )
                ncoldpix = np.shape(coldarea)[1]

                # corecoldarea = np.array(np.where((file_corecold_cloudnumber[:,:] == cloudnumber) & ((file_cloudtype[:,:] == 4) | (file_cloudtype[0,:,:] == 3))))
                corecoldarea = np.array(
                    np.where(
                        (file_corecold_cloudnumber[:, :] == cloudnumber)
                        & ((file_cloudtype[:, :] == 4) | (file_cloudtype[:, :] == 3))
                    )
                )
                # corecoldarea = np.array(np.where((file_corecold_cloudnumber[:,:] == cloudnumber) & ((file_cloudtype[:,:] >=3))))
                # corecoldarea = np.array(np.where((file_corecold_cloudnumber[:,:] == cloudnumber)))
                ncorecoldpix = np.shape(corecoldarea)[1]

                # Find cloud type that belongs to the current track in this file
                cn_wheret, cn_wherej, cn_wherei = np.array(
                    np.where(file_corecold_cloudnumber[:, :] == cloudnumber)
                )
                cloudtype_track = file_cloudtype[:, cn_wherej, cn_wherei]
                n_lowpix = np.count_nonzero(cloudtype_track == 1)
                n_conglowpix = np.count_nonzero(cloudtype_track == 2)
                n_conghighpix = np.count_nonzero(cloudtype_track == 3)
                n_deeppix = np.count_nonzero(cloudtype_track == 4)

                ## Find current length of the track. Use for indexing purposes. Also, record the current length the given track.
                # logger.info('finaltrack_corecold_cloudnumber.shape: ',finaltrack_corecold_cloudnumber.shape)
                # lengthindex = np.array(np.where(finaltrack_corecold_cloudnumber[itrack-1] > 0))
                # if np.shape(lengthindex)[1] > 0:
                # nc = np.nanmax(lengthindex) + 1
                # else:
                # nc = 0
                ##finaltrack_tracklength[itrack-1] = nc+1 # Need to add one since array index starts at 0

                # if nc < nmaxclouds:
                # Save information that links this cloud back to its raw pixel level data
                finaltrack_basetime[itrack] = np.array(
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
                finaltrack_corecold_cloudnumber[itrack] = cloudnumber
                finaltrack_cloudidfile[itrack][:] = fname
                finaltrack_datetimestring[itrack][:] = file_datetimestring
                ###############################################################
                # Calculate statistics about this cloud system
                # 11/21/2019 - Make sure this cloud exists
                if ncorecoldpix > 0:
                    # Location statistics of core+cold anvil (aka the convective system)
                    corecoldlat = latitude[corecoldarea[1], corecoldarea[2]]
                    corecoldlon = longitude[corecoldarea[1], corecoldarea[2]]

                    finaltrack_corecold_meanlat[itrack] = np.nanmean(corecoldlat)
                    finaltrack_corecold_meanlon[itrack] = np.nanmean(corecoldlon)

                    finaltrack_corecold_minlat[itrack] = np.nanmin(corecoldlat)
                    finaltrack_corecold_minlon[itrack] = np.nanmin(corecoldlon)

                    finaltrack_corecold_maxlat[itrack] = np.nanmax(corecoldlat)
                    finaltrack_corecold_maxlon[itrack] = np.nanmax(corecoldlon)

                    # Determine if core+cold touches of the boundaries of the domain
                    if (
                        np.absolute(finaltrack_corecold_minlat[itrack] - geolimits[0])
                        < 0.1
                        or np.absolute(
                            finaltrack_corecold_maxlat[itrack] - geolimits[2]
                        )
                        < 0.1
                        or np.absolute(
                            finaltrack_corecold_minlon[itrack] - geolimits[1]
                        )
                        < 0.1
                        or np.absolute(
                            finaltrack_corecold_maxlon[itrack] - geolimits[3]
                        )
                        < 0.1
                    ):
                        finaltrack_corecold_boundary[itrack] = 1

                    # Save number of pixels (metric for size)
                    finaltrack_ncorecoldpix[itrack] = ncorecoldpix
                    finaltrack_ncorepix[itrack] = ncorepix
                    finaltrack_ncoldpix[itrack] = ncoldpix
                    finaltrack_cloudtype_low[itrack - 1] = n_lowpix
                    finaltrack_cloudtype_conglow[itrack - 1] = n_conglowpix
                    finaltrack_cloudtype_conghigh[itrack - 1] = n_conghighpix
                    finaltrack_cloudtype_deep[itrack - 1] = n_deeppix

                    # Calculate physical characteristics associated with cloud system
                    # Create a padded region around the cloud.
                    pad = 5

                    if np.nanmin(cn_wherej) - pad > 0:
                        minyindex = np.nanmin(cn_wherej) - pad
                    else:
                        minyindex = 0

                    if np.nanmax(cn_wherej) + pad < ny - 1:
                        maxyindex = np.nanmax(cn_wherej) + pad + 1
                    else:
                        maxyindex = ny

                    if np.nanmin(cn_wherei) - pad > 0:
                        minxindex = np.nanmin(cn_wherei) - pad
                    else:
                        minxindex = 0

                    if np.nanmax(cn_wherei) + pad < nx - 1:
                        maxxindex = np.nanmax(cn_wherei) + pad + 1
                    else:
                        maxxindex = nx

                    # Isolate the region around the cloud using the padded region
                    isolatedcloudnumber = np.copy(
                        file_corecold_cloudnumber[
                            minyindex:maxyindex, minxindex:maxxindex
                        ]
                    ).astype(int)

                    # isolatedtb = np.copy(file_tb[0, minyindex:maxyindex, minxindex:maxxindex])

                    # Turn cloud map to binary
                    isolatedcloudnumber[isolatedcloudnumber != cloudnumber] = 0
                    isolatedcloudnumber[isolatedcloudnumber == cloudnumber] = 1
                    # logger.info(isolatedcloudnumber)

                    # Calculate major axis, orientation, eccentricity
                    # cloudproperities = regionprops(isolatedcloudnumber, intensity_image=None)

                    # finaltrack_corecold_eccentricity[itrack] = cloudproperities[0].eccentricity
                    # finaltrack_corecold_majoraxis[itrack] = cloudproperities[0].major_axis_length*pixel_radius
                    # finaltrack_corecold_orientation[itrack] = (cloudproperities[0].orientation)*(180/float(pi))
                    # finaltrack_corecold_perimeter[itrack] = cloudproperities[0].perimeter*pixel_radius
                    # [temp_ycenter, temp_xcenter] = cloudproperities[0].centroid
                    # [finaltrack_corecold_ycenter[itrack], finaltrack_corecold_xcenter[itrack]] = np.add([temp_ycenter,temp_xcenter], [minyindex, minxindex]).astype(int)
                    # [temp_yweightedcenter, temp_xweightedcenter] = cloudproperities[0].weighted_centroid
                    # [finaltrack_corecold_yweightedcenter[itrack], finaltrack_corecold_xweightedcenter[itrack]] = np.add([temp_yweightedcenter, temp_xweightedcenter], [minyindex, minxindex]).astype(int)

                    # Determine equivalent radius of core+cold. Assuming circular area = (number pixels)*(pixel radius)^2, equivalent radius = sqrt(Area / pi)
                    finaltrack_corecold_radius[itrack] = np.sqrt(
                        np.divide(ncorecoldpix * (np.square(pixel_radius)), pi)
                    )

                    # Save track information. Need to subtract one since cloudnumber gives the number of the cloud (which starts at one), but we are looking for its index (which starts at zero)
                    finaltrack_corecold_status[itrack] = np.copy(
                        trackstatus[cloudindex]
                    )
                    # finaltrack_corecold_status[itrack] = trackstatus[cloudindex]
                    finaltrack_corecold_mergenumber[itrack] = np.copy(
                        trackmerge[cloudindex]
                    )
                    finaltrack_corecold_splitnumber[itrack] = np.copy(
                        tracksplit[cloudindex]
                    )
                    finaltrack_corecold_trackinterruptions[itrack] = np.copy(
                        trackreset[cloudindex]
                    )

                    # logger.info('shape of finaltrack_corecold_status: ', finaltrack_corecold_status.shape)

            elif len(cloudnumber) > 1:
                sys.exit(
                    str(cloudnumber)
                    + " clouds linked to one track. Each track should only be linked to one cloud in each file in the track_number array. The track_number variable only tracks the largest cell in mergers and splits. The small clouds in tracks and mergers should only be listed in the track_splitnumbers and track_mergenumbers arrays."
                )
        return (
            uniquetracknumbers,
            numtracks,
            finaltrack_basetime,
            finaltrack_corecold_cloudnumber,
            finaltrack_cloudidfile,
            finaltrack_datetimestring,
            finaltrack_corecold_meanlat,
            finaltrack_corecold_meanlon,
            finaltrack_corecold_minlat,
            finaltrack_corecold_minlon,
            finaltrack_corecold_maxlat,
            finaltrack_corecold_maxlon,
            finaltrack_corecold_boundary,
            finaltrack_ncorecoldpix,
            finaltrack_ncorepix,
            finaltrack_ncoldpix,
            finaltrack_cloudtype_low,
            finaltrack_cloudtype_conglow,
            finaltrack_cloudtype_conghigh,
            finaltrack_cloudtype_deep,
            finaltrack_corecold_radius,
            finaltrack_corecold_status,
            finaltrack_corecold_mergenumber,
            finaltrack_corecold_splitnumber,
            finaltrack_corecold_trackinterruptions,
            basetime_units,
            basetime_calendar,
        )
        # finaltrack_corecold_eccentricity, \
        # finaltrack_corecold_majoraxis, finaltrack_corecold_orientation, finaltrack_corecold_perimeter, \
        # finaltrack_corecold_ycenter, finaltrack_corecold_xcenter, finaltrack_corecold_yweightedcenter, \
        # finaltrack_corecold_xweightedcenter, finaltrack_corecold_radius, \
