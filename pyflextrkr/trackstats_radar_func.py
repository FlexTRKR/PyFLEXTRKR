# Purpose: Core function to calculate radar cell statistics for a given cloudid file
# Author: Zhe Feng (zhe.feng@pnnl.gov)


def calc_stats_radar(
    tracknumbers,
    cloudidfiles,
    tracking_inpath,
    cloudid_filebase,
    numcharfilename,
    latitude,
    longitude,
    x_coord,
    y_coord,
    nx,
    ny,
    pixel_radius,
    rangemask,
    trackstatus,
    trackmerge,
    tracksplit,
    trackreset,
):
    import numpy as np
    from netCDF4 import Dataset, num2date, chartostring
    import os
    import sys
    from math import pi
    from skimage.measure import regionprops
    import time
    import gc
    import datetime
    import logging

    logger = logging.getLogger(__name__)

    # import xarray as xr
    import pandas as pd

    file_tracknumbers = tracknumbers

    # Only process file if that file contains a track
    if np.nanmax(file_tracknumbers) > 0:

        fname = "".join(chartostring(cloudidfiles))
        logger.info(fname)

        # Load cloudid file
        cloudid_file = tracking_inpath + fname
        # logger.info(cloudid_file)

        file_cloudiddata = Dataset(cloudid_file, "r")
        file_dbz = file_cloudiddata["comp_ref"][:]
        file_all_cloudnumber = file_cloudiddata["cloudnumber"][:]
        file_corecold_cloudnumber = file_cloudiddata["convcold_cloudnumber"][:]
        conv_core = file_cloudiddata["conv_core"][:]
        conv_mask = file_cloudiddata["conv_mask"][:]
        echotop10 = file_cloudiddata["echotop10"][:] / 1000.0  # convert unit to [km]
        echotop20 = file_cloudiddata["echotop20"][:] / 1000.0  # convert unit to [km]
        echotop30 = file_cloudiddata["echotop30"][:] / 1000.0  # convert unit to [km]
        echotop40 = file_cloudiddata["echotop40"][:] / 1000.0  # convert unit to [km]
        echotop50 = file_cloudiddata["echotop50"][:] / 1000.0  # convert unit to [km]
        file_basetime = file_cloudiddata["basetime"][:]
        basetime_units = file_cloudiddata["basetime"].units
        file_cloudiddata.close()

        file_datetimestring = cloudid_file[
            len(tracking_inpath) + len(cloudid_filebase) : -3
        ]

        # Find unique track numbers
        uniquetracknumbers = np.unique(file_tracknumbers)
        uniquetracknumbers = uniquetracknumbers[np.isfinite(uniquetracknumbers)]
        uniquetracknumbers = uniquetracknumbers[uniquetracknumbers > 0].astype(int)

        # Create output variables
        fillval = -9999
        numtracks = int(len(uniquetracknumbers))
        finaltrack_basetime = np.full(numtracks, fillval, dtype=np.float)
        finaltrack_core_meanlat = np.full(numtracks, np.nan, dtype=np.float)
        finaltrack_core_meanlon = np.full(numtracks, np.nan, dtype=np.float)
        finaltrack_core_mean_x = np.full(numtracks, np.nan, dtype=np.float)
        finaltrack_core_mean_y = np.full(numtracks, np.nan, dtype=np.float)

        finaltrack_cell_meanlat = np.full(numtracks, np.nan, dtype=np.float)
        finaltrack_cell_meanlon = np.full(numtracks, np.nan, dtype=np.float)
        finaltrack_cell_mean_x = np.full(numtracks, np.nan, dtype=np.float)
        finaltrack_cell_mean_y = np.full(numtracks, np.nan, dtype=np.float)

        finaltrack_cell_minlat = np.full(numtracks, np.nan, dtype=np.float)
        finaltrack_cell_maxlat = np.full(numtracks, np.nan, dtype=np.float)
        finaltrack_cell_minlon = np.full(numtracks, np.nan, dtype=np.float)
        finaltrack_cell_maxlon = np.full(numtracks, np.nan, dtype=np.float)
        finaltrack_cell_min_y = np.full(numtracks, np.nan, dtype=np.float)
        finaltrack_cell_max_y = np.full(numtracks, np.nan, dtype=np.float)
        finaltrack_cell_min_x = np.full(numtracks, np.nan, dtype=np.float)
        finaltrack_cell_max_x = np.full(numtracks, np.nan, dtype=np.float)

        finaltrack_dilatecell_meanlat = np.full(numtracks, np.nan, dtype=np.float)
        finaltrack_dilatecell_meanlon = np.full(numtracks, np.nan, dtype=np.float)
        finaltrack_dilatecell_mean_x = np.full(numtracks, np.nan, dtype=np.float)
        finaltrack_dilatecell_mean_y = np.full(numtracks, np.nan, dtype=np.float)

        finaltrack_cell_maxdbz = np.full(numtracks, np.nan, dtype=float)

        finaltrack_cell_maxETH10dbz = np.full(numtracks, np.nan, dtype=np.float)
        finaltrack_cell_maxETH20dbz = np.full(numtracks, np.nan, dtype=np.float)
        finaltrack_cell_maxETH30dbz = np.full(numtracks, np.nan, dtype=np.float)
        finaltrack_cell_maxETH40dbz = np.full(numtracks, np.nan, dtype=np.float)
        finaltrack_cell_maxETH50dbz = np.full(numtracks, np.nan, dtype=np.float)

        finaltrack_core_area = np.full(numtracks, np.nan, dtype=float)
        finaltrack_cell_area = np.full(numtracks, np.nan, dtype=float)

        finaltrack_status = np.full(numtracks, fillval, dtype=np.int32)
        finaltrack_trackinterruptions = np.full(numtracks, fillval, dtype=np.int32)
        finaltrack_mergenumber = np.full(numtracks, fillval, dtype=np.int32)
        finaltrack_splitnumber = np.full(numtracks, fillval, dtype=np.int32)
        finaltrack_cloudnumber = np.full(numtracks, fillval, dtype=np.int32)
        finaltrack_cloudidfile = np.chararray((numtracks, int(numcharfilename)))

        finaltrack_cell_rangeflag = np.full(numtracks, fillval, dtype=np.int)

        # Loop over unique tracknumbers
        # logger.info('Loop over tracks in file')
        for itrack in range(numtracks):

            # Find cloud number that belongs to the current track in this file
            # Need to add one since tells index, which starts at 0, and we want the number, which starts at one
            cloudnumber = (
                np.array(np.where(file_tracknumbers == uniquetracknumbers[itrack]))[
                    0, :
                ]
                + 1
            )
            cloudindex = cloudnumber - 1  # Index within the matrice of this cloud

            # Should only be one cloud number.
            # In mergers and split, the associated clouds should be listed in
            # the file_splittracknumbers and file_mergetracknumbers
            if len(cloudnumber) == 1:
                # Find core in cloudid file associated with this track, and is a convective core (conv_core == 1)
                corearea = np.array(
                    np.where(
                        (file_corecold_cloudnumber[0, :, :] == cloudnumber)
                        & (conv_core[0, :, :] == 1)
                    )
                )
                ncorepix = np.shape(corearea)[1]

                # Convective cell (conv_mask >= 1). conv_mask is sorted and numbered.
                cellarea = np.array(
                    np.where(
                        (file_corecold_cloudnumber[0, :, :] == cloudnumber)
                        & (conv_mask[0, :, :] >= 1)
                    )
                )
                ncellpix = np.shape(cellarea)[1]

                # Dilated convective cell
                dilatecellarea = np.array(
                    np.where(file_corecold_cloudnumber[0, :, :] == cloudnumber)
                )
                ndilatecellpix = np.shape(dilatecellarea)[1]

                # Save information that links this cloud back to its raw pixel level data
                finaltrack_basetime[itrack] = np.array(file_basetime[0])
                finaltrack_cloudnumber[itrack] = cloudnumber
                finaltrack_cloudidfile[itrack][:] = fname

                ###############################################################
                # Calculate statistics for this cell
                if ncellpix > 0:
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
                    finaltrack_core_meanlat[itrack] = np.nanmean(corelat)
                    finaltrack_core_meanlon[itrack] = np.nanmean(corelon)
                    finaltrack_core_mean_y[itrack] = np.nanmean(core_y)
                    finaltrack_core_mean_x[itrack] = np.nanmean(core_x)

                    # Cell center location
                    finaltrack_cell_meanlat[itrack] = np.nanmean(celllat)
                    finaltrack_cell_meanlon[itrack] = np.nanmean(celllon)
                    finaltrack_cell_mean_y[itrack] = np.nanmean(cell_y)
                    finaltrack_cell_mean_x[itrack] = np.nanmean(cell_x)

                    # Dilated cell center location
                    finaltrack_dilatecell_meanlat[itrack] = np.nanmean(dilatecelllat)
                    finaltrack_dilatecell_meanlon[itrack] = np.nanmean(dilatecelllon)
                    finaltrack_dilatecell_mean_y[itrack] = np.nanmean(dilatecell_y)
                    finaltrack_dilatecell_mean_x[itrack] = np.nanmean(dilatecell_x)

                    # Cell min/max location (for its maximum spatial extent)
                    finaltrack_cell_minlat[itrack] = np.nanmin(celllat)
                    finaltrack_cell_maxlat[itrack] = np.nanmax(celllat)
                    finaltrack_cell_minlon[itrack] = np.nanmin(celllon)
                    finaltrack_cell_maxlon[itrack] = np.nanmax(celllon)
                    finaltrack_cell_min_y[itrack] = np.nanmin(cell_y)
                    finaltrack_cell_max_y[itrack] = np.nanmax(cell_y)
                    finaltrack_cell_min_x[itrack] = np.nanmin(cell_x)
                    finaltrack_cell_max_x[itrack] = np.nanmax(cell_x)

                    # Area of the cell
                    finaltrack_core_area[itrack] = ncorepix * pixel_radius ** 2
                    finaltrack_cell_area[itrack] = ncellpix * pixel_radius ** 2

                    # Reflectivity maximum
                    finaltrack_cell_maxdbz[itrack] = np.nanmax(
                        file_dbz[0, cellarea[0], cellarea[1]]
                    )

                    # Echo-top heights
                    finaltrack_cell_maxETH10dbz[itrack] = np.nanmax(
                        echotop10[0, cellarea[0], cellarea[1]]
                    )
                    finaltrack_cell_maxETH20dbz[itrack] = np.nanmax(
                        echotop20[0, cellarea[0], cellarea[1]]
                    )
                    finaltrack_cell_maxETH30dbz[itrack] = np.nanmax(
                        echotop30[0, cellarea[0], cellarea[1]]
                    )
                    finaltrack_cell_maxETH40dbz[itrack] = np.nanmax(
                        echotop40[0, cellarea[0], cellarea[1]]
                    )
                    finaltrack_cell_maxETH50dbz[itrack] = np.nanmax(
                        echotop50[0, cellarea[0], cellarea[1]]
                    )

                    # Save track status, merge/split information
                    finaltrack_status[itrack] = np.copy(trackstatus[cloudindex])
                    finaltrack_mergenumber[itrack] = np.copy(trackmerge[cloudindex])
                    finaltrack_splitnumber[itrack] = np.copy(tracksplit[cloudindex])
                    finaltrack_trackinterruptions[itrack] = np.copy(
                        trackreset[cloudindex]
                    )

                    # The min range mask value within the dilatecellarea (1: cell completely within range mask, 0: some portion of the cell outside range mask)
                    finaltrack_cell_rangeflag[itrack] = np.min(
                        rangemask[dilatecellarea[0], dilatecellarea[1]]
                    )

            elif len(cloudnumber) > 1:
                logger.info(
                    "Error: cloudnumber "
                    + str(cloudnumber)
                    + " clouds linked to more than one track!"
                )
                logger.info(
                    "Each track should only be linked to one cloud in each file in the track_number array. "
                    + "The track_number variable only tracks the largest cell in mergers and splits. "
                    + "The small clouds in tracks and mergers should only be listed in the "
                    + "track_splitnumbers and track_mergenumbers arrays."
                )
                sys.exit(str(cloudnumber))

        return (
            uniquetracknumbers,
            numtracks,
            finaltrack_basetime,
            finaltrack_cloudnumber,
            finaltrack_cloudidfile,
            finaltrack_core_meanlat,
            finaltrack_core_meanlon,
            finaltrack_core_mean_x,
            finaltrack_core_mean_y,
            finaltrack_cell_meanlat,
            finaltrack_cell_meanlon,
            finaltrack_cell_mean_x,
            finaltrack_cell_mean_y,
            finaltrack_cell_minlat,
            finaltrack_cell_maxlat,
            finaltrack_cell_minlon,
            finaltrack_cell_maxlon,
            finaltrack_cell_min_y,
            finaltrack_cell_max_y,
            finaltrack_cell_min_x,
            finaltrack_cell_max_x,
            finaltrack_dilatecell_meanlat,
            finaltrack_dilatecell_meanlon,
            finaltrack_dilatecell_mean_x,
            finaltrack_dilatecell_mean_y,
            finaltrack_cell_maxdbz,
            finaltrack_cell_maxETH10dbz,
            finaltrack_cell_maxETH20dbz,
            finaltrack_cell_maxETH30dbz,
            finaltrack_cell_maxETH40dbz,
            finaltrack_cell_maxETH50dbz,
            finaltrack_core_area,
            finaltrack_cell_area,
            finaltrack_status,
            finaltrack_trackinterruptions,
            finaltrack_mergenumber,
            finaltrack_splitnumber,
            finaltrack_cell_rangeflag,
        )
