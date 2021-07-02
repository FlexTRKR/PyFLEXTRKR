def calc_stats_single(
    tracknumbers,
    cloudidfiles,
    tracking_inpath,
    cloudid_filebase,
    nbintb,
    numcharfilename,
    latitude,
    longitude,
    geolimits,
    nx,
    ny,
    mintb_thresh,
    maxtb_thresh,
    tbbins,
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

    file_tracknumbers = tracknumbers
    logger = logging.getLogger(__name__)

    # Only process file if that file contains a track
    if np.nanmax(file_tracknumbers) <= 0:
        return None

    fname = "".join(chartostring(cloudidfiles))
    logger.debug(fname)

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
    # basetime_calendar = file_cloudiddata['basetime'].calendar
    file_cloudiddata.close()

    file_datetimestring = cloudid_file[
        len(tracking_inpath) + len(cloudid_filebase) : -3
    ]

    # Find unique track numbers
    uniquetracknumbers = np.unique(file_tracknumbers)
    uniquetracknumbers = uniquetracknumbers[np.isfinite(uniquetracknumbers)]
    uniquetracknumbers = uniquetracknumbers[uniquetracknumbers > 0].astype(int)

    numtracks = len(uniquetracknumbers)
    # finaltrack_corecold_boundary = np.zeros(int(numtracks), dtype=np.int32) # Kb messing around
    finaltrack_corecold_boundary = np.ones(int(numtracks), dtype=np.int32) * -9999
    # finaltrack_basetime = np.empty(int(numtracks), dtype='datetime64[s]')
    finaltrack_basetime = np.ones(int(numtracks), dtype=float) * np.nan
    finaltrack_corecold_mintb = np.ones(int(numtracks), dtype=float) * np.nan
    finaltrack_corecold_meantb = np.ones(int(numtracks), dtype=float) * np.nan
    finaltrack_core_meantb = np.ones(int(numtracks), dtype=float) * np.nan
    # finaltrack_corecold_histtb = np.zeros((int(numtracks), nbintb-1))
    finaltrack_corecold_radius = np.ones(int(numtracks), dtype=float) * np.nan
    finaltrack_corecoldwarm_radius = np.ones(int(numtracks), dtype=float) * np.nan
    finaltrack_corecold_meanlat = np.ones(int(numtracks), dtype=float) * np.nan
    finaltrack_corecold_meanlon = np.ones(int(numtracks), dtype=float) * np.nan
    finaltrack_corecold_maxlon = np.ones(int(numtracks), dtype=float) * np.nan
    finaltrack_corecold_maxlat = np.ones(int(numtracks), dtype=float) * np.nan
    finaltrack_ncorecoldpix = np.ones(int(numtracks), dtype=np.int32) * -9999
    finaltrack_corecold_minlon = np.ones(int(numtracks), dtype=float) * np.nan
    finaltrack_corecold_minlat = np.ones(int(numtracks), dtype=float) * np.nan
    finaltrack_ncorepix = np.ones(int(numtracks), dtype=np.int32) * -9999
    finaltrack_ncoldpix = np.ones(int(numtracks), dtype=np.int32) * -9999
    finaltrack_nwarmpix = np.ones(int(numtracks), dtype=np.int32) * -9999
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


    cloudnumber_map = np.array(np.zeros(numtracks))

    for itrack in range(numtracks):
        cloudnumber_map[itrack] =( # TODO: Joe: I really want to rewrite this later.
            np.array(np.where(file_tracknumbers == uniquetracknumbers[itrack]))[
                0, :
            ]
            + 1
        )
    cloudindex_map = cloudnumber_map - 1

    corearea_m, corearea_v = np.unique(file_corecold_cloudnumber * (file_cloudtype[0, :, :] ==1),
                                       return_counts=True)

    ncorepix_m = np.zeros(numtracks, dtype=int)
    coldarea_m, coldarea_v = np.unique(file_corecold_cloudnumber * (file_cloudtype[0, :, :] == 2),
                                       return_counts=True)
    ncoldpix_m = np.zeros(numtracks, dtype=int)
    coretb = np.zeros(numtracks)

    ast_coldarea = np.argsort(file_corecold_cloudnumber * (file_cloudtype[0, :, :] == 2), axis=None)
    cumulative_counts_coldarea = np.cumsum(coldarea_v)

    warmarea_m, warmarea_v = np.unique(file_corecold_cloudnumber * (file_cloudtype[0, :, :] == 3),
                                       return_counts=True)
    nwarmpix_m = np.zeros(numtracks, dtype=int)

    fct_is_1_or_2 = (file_cloudtype[0, :, :] == 1) | (file_cloudtype[0, :, :] == 2)
    corecoldarea_m, corecoldarea_v = np.unique(file_corecold_cloudnumber[0, :, :] * fct_is_1_or_2, return_counts=True)
    ast_corecoldarea = np.argsort(file_corecold_cloudnumber[0, :, :] * fct_is_1_or_2, axis=None)
    cumulative_counts_corecoldarea = np.cumsum(corecoldarea_v)
    ncorecoldpix_m = np.zeros(numtracks, dtype=int)

    for tracknum in range(numtracks):
        # Mostly just selecting the right alignment of cloudnumber and track number now.
        # We'll abuse idx variable as temporary throwaway to track alignment.
        idx = np.where(cloudnumber_map[tracknum] == corearea_m)[0]
        if len(idx > 0):
            ncorepix_m[tracknum] = corearea_v[idx]

        idx = np.where(cloudnumber_map[tracknum] == coldarea_m)[0]
        if len(idx > 0):
            ncoldpix_m[tracknum] = coldarea_v[idx]

            # Handle meantb while we're here.
            # We use this to know where to index into the sorted list
            if idx>0:
                indices = np.unravel_index(ast_coldarea[cumulative_counts_coldarea[idx-1][0]:cumulative_counts_coldarea[idx][0]], file_corecold_cloudnumber.shape)
            else:
                indices = np.unravel_index(ast_coldarea[0:cumulative_counts_coldarea[idx][0]], file_corecold_cloudnumber.shape)


            coretb[tracknum] = np.nanmean(file_tb[indices[0], indices[1], indices[2]])

        idx = np.where(cloudnumber_map[tracknum] == warmarea_m)[0]
        if len(idx > 0):
            nwarmpix_m[tracknum] = warmarea_v[idx]

        idx = np.where(cloudnumber_map[tracknum] == corecoldarea_m)[0]
        if len(idx > 0):
            ncorecoldpix_m[tracknum] = corecoldarea_v[idx]

    finaltrack_core_meantb = coretb
    finaltrack_ncoldpix = ncoldpix_m
    finaltrack_ncorepix = ncorepix_m
    finaltrack_nwarmpix = nwarmpix_m
    finaltrack_corecold_cloudnumber = cloudnumber_map

    for itrack in range(numtracks):

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

            idx = np.where(cloudnumber_map[itrack] == corecoldarea_m)[0]
            if idx >0:
                corecoldarea = np.array(np.unravel_index(
                    ast_corecoldarea[cumulative_counts_corecoldarea[idx-1][0]:cumulative_counts_corecoldarea[idx ][0]],
                    file_corecold_cloudnumber.shape[1:]))
            else:
                corecoldarea = np.array(np.unravel_index(
                    ast_corecoldarea[0:cumulative_counts_corecoldarea[idx][0]],
                    file_corecold_cloudnumber.shape[1:]))

            ncorecoldpix = np.shape(corecoldarea)[1]


            # Find current length of the track. Use for indexing purposes. Also, record the current length the given track.
            # lengthindex = np.array(np.where(finaltrack_corecold_cloudnumber[itrack-1,:] > 0))
            # if np.shape(lengthindex)[1] > 0:
            #    nc = np.nanmax(lengthindex) + 1
            # else:
            #    nc = 0
            # finaltrack_tracklength[itrack-1] = nc+1 # Need to add one since array index starts at 0

            # if nc < nmaxclouds:
            # Save information that links this cloud back to its raw pixel level data
            # finaltrack_basetime[itrack] = np.array([pd.to_datetime(num2date(file_basetime, units=basetime_units, calendar=basetime_calendar))], dtype='datetime64[s]')[0, 0]
            finaltrack_basetime[itrack] = file_basetime.data
            finaltrack_cloudidfile[itrack][:] = fname
            finaltrack_datetimestring[itrack][:] = file_datetimestring
            ###############################################################
            # Calculate statistics about this cloud system
            # 11/21/2019 - Make sure this cloud exists
            if ncorecoldpix > 0:
                # Location statistics of core+cold anvil (aka the convective system)
                corecoldlat = latitude[corecoldarea[0], corecoldarea[1]]
                corecoldlon = longitude[corecoldarea[0], corecoldarea[1]]

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
                # finaltrack_ncorepix[itrack] = ncorepix
                # finaltrack_ncoldpix[itrack] = ncoldpix
                # finaltrack_nwarmpix[itrack] = nwarmpix

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
                cloudproperties = regionprops(
                    isolatedcloudnumber, intensity_image=isolatedtb
                )

                finaltrack_corecold_eccentricity[itrack] = cloudproperties[
                    0
                ].eccentricity
                finaltrack_corecold_majoraxis[itrack] = (
                    cloudproperties[0].major_axis_length * pixel_radius
                )
                finaltrack_corecold_orientation[itrack] = (
                    cloudproperties[0].orientation
                ) * (180 / float(pi))
                finaltrack_corecold_perimeter[itrack] = (
                    cloudproperties[0].perimeter * pixel_radius
                )
                [temp_ycenter, temp_xcenter] = cloudproperties[0].centroid
                [
                    finaltrack_corecold_ycenter[itrack],
                    finaltrack_corecold_xcenter[itrack],
                ] = np.add(
                    [temp_ycenter, temp_xcenter], [minyindex, minxindex]
                ).astype(
                    int
                )
                [temp_yweightedcenter, temp_xweightedcenter] = cloudproperties[
                    0
                ].weighted_centroid
                [
                    finaltrack_corecold_yweightedcenter[itrack],
                    finaltrack_corecold_xweightedcenter[itrack],
                ] = np.add(
                    [temp_yweightedcenter, temp_xweightedcenter],
                    [minyindex, minxindex],
                ).astype(
                    int
                )

                # Determine equivalent radius of core+cold. Assuming circular area = (number pixels)*(pixel radius)^2, equivalent radius = sqrt(Area / pi)
                finaltrack_corecold_radius[itrack] = np.sqrt(
                    np.divide(ncorecoldpix * (np.square(pixel_radius)), pi)
                )

                # Determine equivalent radius of core+cold+warm. Assuming circular area = (number pixels)*(pixel radius)^2, equivalent radius = sqrt(Area / pi)
                finaltrack_corecoldwarm_radius[itrack] = np.sqrt(
                    np.divide(
                        (ncorepix_m[itrack] + ncoldpix_m[itrack] + nwarmpix_m[itrack])
                        * (np.square(pixel_radius)),
                        pi,
                    )
                )

                ##############################################################
                # Calculate brightness temperature statistics of core+cold anvil
                corecoldtb = np.copy(file_tb[0, corecoldarea[0], corecoldarea[1]])

                finaltrack_corecold_mintb[itrack] = np.nanmin(corecoldtb)
                finaltrack_corecold_meantb[itrack] = np.nanmean(corecoldtb)

                ################################################################
                # Histogram of brightness temperature for core+cold anvil
                # finaltrack_corecold_histtb[itrack,:], usedtbbins = np.histogram(corecoldtb, range=(mintb_thresh, maxtb_thresh), bins=tbbins)

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

                ####################################################################
                # Calculate mean brightness temperature for core
                #TODO: JOE: Uncomment out next few lines
                # coretb = np.copy(file_tb[0, coldarea[0], coldarea[1]])
                #
                # finaltrack_core_meantb[itrack] = np.nanmean(coretb)

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
        finaltrack_nwarmpix,
        finaltrack_corecold_eccentricity,
        finaltrack_corecold_majoraxis,
        finaltrack_corecold_orientation,
        finaltrack_corecold_perimeter,
        finaltrack_corecold_ycenter,
        finaltrack_corecold_xcenter,
        finaltrack_corecold_yweightedcenter,
        finaltrack_corecold_xweightedcenter,
        finaltrack_corecold_radius,
        finaltrack_corecoldwarm_radius,
        finaltrack_corecold_mintb,
        finaltrack_corecold_meantb,
        finaltrack_corecold_status,
        finaltrack_corecold_mergenumber,
        finaltrack_corecold_splitnumber,
        finaltrack_corecold_trackinterruptions,
        finaltrack_core_meantb,
        basetime_units,
    )  # , basetime_calendar
