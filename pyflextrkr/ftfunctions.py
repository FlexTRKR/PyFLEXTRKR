import numpy as np
import os, fnmatch, sys, glob
import datetime, calendar
from pytz import utc
import logging
import xarray as xr

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def get_basetime_from_string(datestring):
    """
    Calculate base time (Epoch time) from a string.

    Args:
        datestring: string (yyyymodd.hhmm)
            String containing date & time.

    Returns:
        base_time: int
            Epoch time in seconds.
    """
    TEMP_starttime = datetime.datetime(int(datestring[0:4]),
                                       int(datestring[4:6]),
                                       int(datestring[6:8]),
                                       int(datestring[9:11]),
                                       int(datestring[11:]),
                                       0, tzinfo=utc)
    base_time = calendar.timegm(TEMP_starttime.timetuple())
    return base_time

def get_basetime_from_filename(
    data_path,
    data_basename,
    time_format="yyyymodd_hhmm",
):
    """
    Calculate base time (Epoch time) from filenames.

    Args:
        data_path: string
            Data directory name.
        data_basename: string
            Data base name.
        time_format: string (optional, default="yyyymodd_hhmm")
            Specify file time format to extract date/time.
    Returns:
        data_filenames: list
            List of data filenames.
        files_basetime: numpy array
            Array of file base time.
        files_datestring: list
            List of file date string.
        files_timestring: list
            List of file time string.

    """
    # Isolate all possible files
    filenames = sorted(fnmatch.filter(os.listdir(data_path), data_basename + '*.nc'))

    # Loop through files, identifying files within the startdate - enddate interval
    nleadingchar = len(data_basename)

    data_filenames = [None]*len(filenames)
    files_timestring = [None]*len(filenames)
    files_datestring = [None]*len(filenames)
    files_basetime = np.full(len(filenames), -9999, dtype=int)

    yyyy_idx = time_format.find("yyyy")
    mo_idx = time_format.find("mo")
    dd_idx = time_format.find("dd")
    hh_idx = time_format.find("hh")
    mm_idx = time_format.find("mm")
    ss_idx = time_format.find("ss")

    # Add basetime character counts to get the actual date/time string positions
    yyyy_idx = nleadingchar + yyyy_idx
    mo_idx = nleadingchar + mo_idx
    dd_idx = nleadingchar + dd_idx
    hh_idx = nleadingchar + hh_idx
    mm_idx = nleadingchar + mm_idx if (mm_idx != -1) else None
    ss_idx = nleadingchar + ss_idx if (ss_idx != -1) else None

    # Loop over each file
    for ii, ifile in enumerate(filenames):
        year = ifile[yyyy_idx:yyyy_idx+4]
        month = ifile[mo_idx:mo_idx+2]
        day = ifile[dd_idx:dd_idx+2]
        hour = ifile[hh_idx:hh_idx+2]
        # If minute, second is not in time_format, assume 0
        minute = ifile[mm_idx:mm_idx+2] if (mm_idx is not None) else 0
        second = ifile[ss_idx:ss_idx+2] if (ss_idx is not None) else 0

        TEMP_filetime = datetime.datetime(int(year), int(month), int(day),
                                          int(hour), int(minute), int(second), tzinfo=utc)
        files_basetime[ii] = calendar.timegm(TEMP_filetime.timetuple())
        files_datestring[ii] = year + month + day
        files_timestring[ii] = hour + minute
        data_filenames[ii] = data_path + ifile
    return (
        data_filenames,
        files_basetime,
        files_datestring,
        files_timestring,
    )

def subset_files_timerange(
    data_path,
    data_basename,
    start_basetime,
    end_basetime,
    time_format="yyyymodd_hhmm",
):
    """
    Subset files within given start and end time.

    Args:
        data_path: string
            Data directory name.
        data_basename: string
            Data base name.
        start_basetime: int
            Start base time (Epoch time).
        end_basetime: int
            End base time (Epoch time).
        time_format: string (optional, default="yyyymodd_hhmm")
            Specify file time format to extract date/time.

    Returns:
        data_filenames: list
            List of data file names with full path.
        files_basetime: numpy array
            Array of file base time.
        files_datestring: list
            List of file date string.
        files_timestring: list
            List of file time string.
    """
    # Get basetime for all files
    data_filenames, files_basetime, \
    files_datestring, files_timestring = get_basetime_from_filename(
        data_path, data_basename, time_format=time_format,
    )

    # Find basetime within the given range
    fidx = np.where((files_basetime >= start_basetime) & (files_basetime <= end_basetime))[0]
    # Subset filenames, dates, times
    data_filenames = np.array(data_filenames)[fidx].tolist()
    files_basetime = files_basetime[fidx]
    files_datestring = np.array(files_datestring)[fidx].tolist()
    files_timestring = np.array(files_timestring)[fidx].tolist()
    return (
        data_filenames,
        files_basetime,
        files_datestring,
        files_timestring,
    )

def match_drift_times(
    cloudidfiles_datestring,
    cloudidfiles_timestring,
    driftfile=None
):
    """
    Match drift file times with cloudid file times.

    Args:
        cloudidfiles_datestring: list
            List of cloudid files date string.
        cloudidfiles_timestring: list
            List of cloudid files time string.
        driftfile: string (optional)
            Drift (advection) file name.

    Returns:
        datetime_drift_match: numpy array
            Matched drift file date time strings.
        xdrifts_match: numpy array
            Matched drift distance in the x-direction.
        ydrifts_match: numpy array
            Matched drift distance in the y-direction.

    """
    # Create drift variables that match number of reference cloudid files
    # Number of reference cloudid files (1 less than total cloudid files)
    ncloudidfiles = len(cloudidfiles_timestring) - 1
    datetime_drift_match = np.empty(ncloudidfiles, dtype='<U13')
    xdrifts_match = np.zeros(ncloudidfiles, dtype=int)
    ydrifts_match = np.zeros(ncloudidfiles, dtype=int)
    # Test if driftfile is defined
    try:
        driftfile
    except NameError:
        logger.info(f"Drift file is not defined. Regular tracksingle procedure is used.")
    else:
        logger.info(f"Drift file used: {driftfile}")

        # Read the drift file
        ds_drift = xr.open_dataset(driftfile)
        bt_drift = ds_drift.basetime
        xdrifts = ds_drift.x.values
        ydrifts = ds_drift.y.values

        # Convert dateime64 objects to string array
        datetime_drift = bt_drift.dt.strftime("%Y%m%d_%H%M").values

        # Loop over each cloudid file time to find matching drfit data
        for itime in range(0, len(cloudidfiles_timestring) - 1):
            cloudid_datetime = cloudidfiles_datestring[itime] + '_' + cloudidfiles_timestring[itime]
            idx = np.where(datetime_drift == cloudid_datetime)[0]
            if (len(idx) == 1):
                datetime_drift_match[itime] = datetime_drift[idx[0]]
                xdrifts_match[itime] = xdrifts[idx]
                ydrifts_match[itime] = ydrifts[idx]
    return (
        datetime_drift_match,
        xdrifts_match,
        ydrifts_match,
    )

def sort_renumber(
    labelcell_number2d,
    min_cellpix,
):
    """
    Sorts 2D labeled cells by size, and removes cells smaller than min_cellpix.

    Args:
        labelcell_number2d: np.ndarray()
            Labeled cell number array in 2D.
        min_cellpix: float
            Minimum number of pixel to count as a cell.

    Returns:
        sortedlabelcell_number2d: np.ndarray(int)
            Sorted labeled cell number array in 2D.
        sortedcell_npix: np.ndarray(int)
            Number of pixels for each labeled cell in 1D.
    """

    # Create output arrays
    sortedlabelcell_number2d = np.zeros(np.shape(labelcell_number2d), dtype=int)

    # Get number of labeled cells
    nlabelcells = np.nanmax(labelcell_number2d)

    # Check if there is any cells identified
    if nlabelcells > 0:

        labelcell_npix = np.full(nlabelcells, -999, dtype=int)

        # Loop over each labeled cell
        for ilabelcell in range(1, nlabelcells + 1):
            # Count number of pixels for the cell
            ilabelcell_npix = np.count_nonzero(labelcell_number2d == ilabelcell)
            # Check if cell satisfies size threshold
            if ilabelcell_npix > min_cellpix:
                labelcell_npix[ilabelcell - 1] = ilabelcell_npix

        # Check if any of the cells passes the size threshold test
        ivalidcells = np.array(np.where(labelcell_npix > 0))[0, :]
        ncells = len(ivalidcells)

        if ncells > 0:
            # Isolate cells that satisfy size threshold
            # Add one since label numbers start at 1 and indices, which validcells reports starts at 0
            labelcell_number1d = np.copy(ivalidcells) + 1
            labelcell_npix = labelcell_npix[ivalidcells]

            # Sort cells from largest to smallest and get the sorted index
            order = np.argsort(labelcell_npix)
            order = order[::-1]  # Reverses the order

            # Sort the cells by size
            sortedcell_npix = np.copy(labelcell_npix[order])
            sortedcell_number1d = np.copy(labelcell_number1d[order])

            # Loop over the 2D cell number to re-number them by size
            cellstep = 0
            for icell in range(0, ncells):
                # Find 2D indices that match the cell number
                sortedcell_indices = np.where(
                    labelcell_number2d == sortedcell_number1d[icell]
                )
                # Get one of the dimensions from the 2D indices to count the size
                #             nsortedcellindices = np.shape(sortedcell_indices)[1]
                nsortedcellindices = len(sortedcell_indices[1])
                # Check if the size matches the sorted cell size
                if nsortedcellindices == sortedcell_npix[icell]:
                    # Renumber the cell in 2D
                    cellstep += 1
                    sortedlabelcell_number2d[sortedcell_indices] = np.copy(cellstep)

        else:
            # Return an empty array
            sortedcell_npix = np.zeros(0)
    else:
        # Return an empty array
        sortedcell_npix = np.zeros(0)

    return (
        sortedlabelcell_number2d,
        sortedcell_npix,
    )


def sort_renumber2vars(
    labelcell_number2d,
    labelcell2_number2d,
    min_cellpix,
):
    """
    Sorts 2D labeled cells by size, and removes cells smaller than min_cellpix.
    This version renumbers two variables using the same size sorting from labelcell_number2d.

    Args:
        labelcell_number2d: np.ndarray()
            Labeled cell number array in 2D.
        labelcell2_number2d: np.ndarray()
            Labeled cell number array2 in 2D.
        min_cellpix: float
            Minimum number of pixel to count as a cell.

    Returns:
        sortedlabelcell_number2d: np.ndarray(int)
            Sorted labeled cell number array in 2D.
        sortedlabelcell2_number2d: np.ndarray(int)
            Sorted labeled cell number array2 in 2D.
        sortedcell_npix: np.ndarray(int)
            Number of pixels for each labeled cell in 1D.
    """

    # Create output arrays
    sortedlabelcell_number2d = np.zeros(np.shape(labelcell_number2d), dtype=int)
    sortedlabelcell2_number2d = np.zeros(np.shape(labelcell_number2d), dtype=int)

    # Get number of labeled cells
    nlabelcells = np.nanmax(labelcell_number2d)

    # Check if there is any cells identified
    if nlabelcells > 0:

        labelcell_npix = np.full(nlabelcells, -999, dtype=int)

        # Loop over each labeled cell
        for ilabelcell in range(1, nlabelcells + 1):
            # Count number of pixels for the cell
            ilabelcell_npix = np.count_nonzero(labelcell_number2d == ilabelcell)
            # Check if cell satisfies size threshold
            if ilabelcell_npix > min_cellpix:
                labelcell_npix[ilabelcell - 1] = ilabelcell_npix

        # Check if any of the cells passes the size threshold test
        ivalidcells = np.array(np.where(labelcell_npix > 0))[0, :]
        ncells = len(ivalidcells)

        if ncells > 0:
            # Isolate cells that satisfy size threshold
            # Add one since label numbers start at 1 and indices, which validcells reports starts at 0
            labelcell_number1d = np.copy(ivalidcells) + 1
            labelcell_npix = labelcell_npix[ivalidcells]

            # Sort cells from largest to smallest and get the sorted index
            order = np.argsort(labelcell_npix)
            order = order[::-1]  # Reverses the order

            # Sort the cells by size
            sortedcell_npix = np.copy(labelcell_npix[order])
            sortedcell_number1d = np.copy(labelcell_number1d[order])

            # Loop over the 2D cell number to re-number them by size
            cellstep = 0
            for icell in range(0, ncells):
                # Find 2D indices that match the cell number
                # Use the same sorted index to label labelcell2_number2d
                sortedcell_indices = np.where(
                    labelcell_number2d == sortedcell_number1d[icell]
                )
                sortedcell2_indices = np.where(
                    labelcell2_number2d == sortedcell_number1d[icell]
                )
                # Get one of the dimensions from the 2D indices to count the size
                nsortedcellindices = len(sortedcell_indices[1])
                # Check if the size matches the sorted cell size
                if nsortedcellindices == sortedcell_npix[icell]:
                    # Renumber the cell in 2D
                    cellstep += 1
                    sortedlabelcell_number2d[sortedcell_indices] = np.copy(cellstep)
                    sortedlabelcell2_number2d[sortedcell2_indices] = np.copy(cellstep)

        else:
            # Return an empty array
            sortedcell_npix = np.zeros(0)
    else:
        # Return an empty array
        sortedcell_npix = np.zeros(0)

    return (
        sortedlabelcell_number2d,
        sortedlabelcell2_number2d,
        sortedcell_npix,
    )


def link_pf_tb(
    convcold_cloudnumber,
    cloudnumber,
    pf_number,
    tb,
    tb_thresh,
):
    """
    Renumbers separated clouds over the same PF to one cloud, using the largest cloud number.

    Args:
        convcold_cloudnumber: np.ndarray(int)
            Convective-coldanvil cloud number
        cloudnumber: np.ndarray(int)
            Cloud number
        pf_number: np.ndarray(int)
            PF number
        tb: np.ndarray(float)
            Brightness temperature
        tb_thresh: float
            Temperature threshold to label PFs that have not been labeled in pf_number.
            Currently this threshold is NOT used.

    Returns:
        pf_convcold_cloudnumber: np.ndarray(int)
            Renumbered convective-coldanvil cloud number
        pf_cloudnumber: np.ndarray(int)
            Renumbered cloud number
    """

    # Get number of PFs
    npf = np.nanmax(pf_number)

    # Make a copy of the input arrays
    pf_convcold_cloudnumber = np.copy(convcold_cloudnumber)
    pf_cloudnumber = np.copy(cloudnumber)

    # Create a 2D index array with the same shape as the full image
    # This index array is used to map the indices of indices back to the full image
    arrayindex2d = np.reshape(np.arange(tb.size), tb.shape)

    # If number of PF > 0, proceed
    if npf > 0:

        # Initiallize masks to keep track of which clouds have been renumbered
        pf_convcold_mask = np.zeros(tb.shape, dtype=int)
        pf_cloud_mask = np.zeros(tb.shape, dtype=int)

        # Loop over each PF
        for ipf in range(1, npf):

            # Find pixel index for this PF
            pfidx = np.where(pf_number == ipf)
            npix_pf = len(pfidx[0])

            if npix_pf > 0:
                # Get unique cloud number defined within this PF
                # cn_uniq = np.unique(convcold_cloudnumber[pfidx])
                cn_uniq = np.unique(pf_convcold_cloudnumber[pfidx])

                # Find actual clouds (cloudnumber > 0)
                cn_uniq = cn_uniq[np.where(cn_uniq > 0)]
                nclouds_uniq = len(cn_uniq)
                # If there is at least 1 cloud, proceed
                if nclouds_uniq >= 1:

                    # Loop over each cloudnumber and get the size
                    npix_uniq = np.zeros(nclouds_uniq, dtype=np.int64)
                    for ic in range(0, nclouds_uniq):
                        # Find pixels for each cloud, save the size
                        # npix_uniq[ic] = len(np.where(convcold_cloudnumber == cn_uniq[ic])[0])
                        npix_uniq[ic] = len(
                            np.where(pf_convcold_cloudnumber == cn_uniq[ic])[0]
                        )

                    # Find cloud number that has maximum size
                    cn_max = cn_uniq[np.argmax(npix_uniq)]

                    # Loop over each cloudnumber again
                    for ic in range(0, nclouds_uniq):

                        # Find pixel locations within each cloud, and mask = 0 (cloud has not been renumbered)
                        # idx_convcold = np.where((convcold_cloudnumber == cn_uniq[ic]) & (pf_convcold_mask == 0))
                        idx_convcold = np.where(
                            (pf_convcold_cloudnumber == cn_uniq[ic])
                            & (pf_convcold_mask == 0)
                        )
                        # idx_cloud = np.where((cloudnumber == cn_uniq[ic]) & (pf_cloud_mask == 0))
                        idx_cloud = np.where(
                            (pf_cloudnumber == cn_uniq[ic]) & (pf_cloud_mask == 0)
                        )
                        if len(idx_convcold[0]) > 0:
                            # Renumber the cloud to the largest cloud number (that overlaps with this PF)
                            pf_convcold_cloudnumber[idx_convcold] = cn_max
                            pf_convcold_mask[idx_convcold] = 1
                        if len(idx_cloud[0]) > 0:
                            # Renumber the cloud to the largest cloud number (that overlaps with this PF)
                            pf_cloudnumber[idx_cloud] = cn_max
                            pf_cloud_mask[idx_cloud] = 1

                    # Find area within the PF that has no cloudnumber, Tb < warm threshold, and has not been labeled yet
                    #                     idx_nocloud = np.asarray((pf_convcold_cloudnumber[pfidx] == 0) & (tb[pfidx] < tb_thresh) & (pf_convcold_mask[pfidx] == 0)).nonzero()
                    idx_nocloud = np.asarray(
                        (pf_convcold_cloudnumber[pfidx] == 0)
                        & (pf_convcold_mask[pfidx] == 0)
                    ).nonzero()
                    if np.count_nonzero(idx_nocloud) > 0:
                        # At this point, idx_nocloud is a 1D index referring to the subset within pfidx
                        # Applying idx_nocloud of pfidx to the 2D full image index array gets the 1D indices referring to the full image,
                        # then unravel_index converts those 1D indices back to 2D, which can then be applied to the 2D full image
                        idx_loc = np.unravel_index(
                            arrayindex2d[pfidx][idx_nocloud], tb.shape
                        )
                        # Label the no cloud area using the largest cloud number
                        pf_convcold_cloudnumber[idx_loc] = cn_max
                        pf_convcold_mask[idx_loc] = 1

                    # Find area within the PF that has no cloudnumber, Tb < warm threshold, and has not been labeled yet
                    #                     idx_nocloud = np.asarray((pf_cloudnumber[pfidx] == 0) & (tb[pfidx] < tb_thresh) & (pf_cloud_mask[pfidx] == 0)).nonzero()
                    idx_nocloud = np.asarray(
                        (pf_cloudnumber[pfidx] == 0) & (pf_cloud_mask[pfidx] == 0)
                    ).nonzero()
                    if np.count_nonzero(idx_nocloud) > 0:
                        idx_loc = np.unravel_index(
                            arrayindex2d[pfidx][idx_nocloud], tb.shape
                        )
                        # Label the no cloud area using the largest cloud number
                        pf_cloudnumber[idx_loc] = cn_max
                        pf_cloud_mask[idx_loc] = 1

    else:
        # Pass input variables to output if no PFs are defined
        pf_convcold_cloudnumber = np.copy(convcold_cloudnumber)
        pf_cloudnumber = np.copy(cloudnumber)

    return (
        pf_convcold_cloudnumber,
        pf_cloudnumber,
    )
