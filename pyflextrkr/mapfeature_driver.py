import os
import sys
import logging
import numpy as np
import xarray as xr
import dask
from dask.distributed import wait
from pyflextrkr.ft_utilities import subset_files_timerange
from pyflextrkr.mapfeature_func import map_feature

def mapfeature_driver(
        config,
        trackstats_filebase="trackstats_",
        outpath_basename=None,
        outfile_basename=None,
):
    """
    Map tracked features to pixel-level files.

    Args:
        config: dictionary
            Dictionary containing config parameters.
        trackstats_filebase: string, default="trackstats_"
            Track statistics file basename.
        outpath_basename: string, default=None
            Output path basename for pixel-level files.
            If None, pixeltracking_outpath defaults to config["pixeltracking_outpath"].
            Otherwise, pixeltracking_outpath is constructed using the config file:
                f'{config["root_path"]}/{outpath_basename}/{config["startdate"]}_{config["enddate"]}/'
        outfile_basename: string, default=None
            Output pixel-level file basename.
            If None, defaults to use config["pixeltracking_filebase"].

    Returns:
        None.
    """
    logger = logging.getLogger(__name__)
    logger.info('Mapping tracked features to pixel-level files')

    stats_path = config["stats_outpath"]
    if outpath_basename is None:
        pixeltracking_outpath = config["pixeltracking_outpath"]
    else:
        pixeltracking_outpath = f'{config["root_path"]}/{outpath_basename}/{config["startdate"]}_{config["enddate"]}/'
    if outfile_basename is None:
        pixeltracking_filebase = config["pixeltracking_filebase"]
    else:
        pixeltracking_filebase = outfile_basename
    tracking_outpath = config["tracking_outpath"]
    cloudid_filebase = config["cloudid_filebase"]
    startdate = config["startdate"]
    enddate = config["enddate"]
    start_basetime = config["start_basetime"]
    end_basetime = config["end_basetime"]
    # Minimum time difference threshold [second] to match track stats and cloudid pixel files
    match_pixel_dt_thresh = config["match_pixel_dt_thresh"]
    run_parallel = config["run_parallel"]
    # feature_type = config["feature_type"]
    nmaxlinks = config["nmaxlinks"]
    tracks_dimname = config.get("tracks_dimname", "tracks")
    times_dimname = config.get("times_dimname", "times")
    fillval = config.get("fillval", -9999)

    #########################################################################################
    # Read track stats
    trackstats_file = f"{stats_path}{trackstats_filebase}{startdate}_{enddate}.nc"
    ds = xr.open_dataset(
        trackstats_file,
        mask_and_scale=False,
        decode_times=False,
    ).compute()
    # Get track stats variable names
    stats_varnames = list(ds.data_vars)
    # Get track stats dimensions
    ntracks = ds.sizes[tracks_dimname]
    ntimes = ds.sizes[times_dimname]
    # Get track variables
    stats_basetime = ds["base_time"].data
    stats_cloudnumber = ds["cloudnumber"].data
    stats_trackstatus = ds["track_status"].data
    trackstats_comments = ds["track_status"].comments
    ds.close()

    # Put merge/split tracknumbers & cloudnumbers in a list
    ms_tracknumber = ["merge_tracknumbers", "split_tracknumbers"]
    ms_cloudnumber = ["merge_cloudnumber", "split_cloudnumber"]

    # Check if tracknumber are in the stats dataset
    if (set(ms_tracknumber).issubset(set(stats_varnames))):
        stats_mergetracknumber = ds["merge_tracknumbers"].data
        stats_splittracknumber = ds["split_tracknumbers"].data
    else:
        stats_mergetracknumber = np.full((ntracks, ntimes), fillval, dtype=int)
        stats_splittracknumber = np.full((ntracks, ntimes), fillval, dtype=int)

    # Check if cloudnumber are in the stats dataset
    if (set(ms_cloudnumber).issubset(set(stats_varnames))):
        stats_mergecloudnumber = ds["merge_cloudnumber"].data
        stats_splitcloudnumber = ds["split_cloudnumber"].data
    else:
        stats_mergecloudnumber = np.full((ntracks, ntimes, nmaxlinks), fillval, dtype=int)
        stats_splitcloudnumber = np.full((ntracks, ntimes, nmaxlinks), fillval, dtype=int)

    #########################################################################################
    # Identify files to process
    # Create pixel tracking file output directory
    os.makedirs(pixeltracking_outpath, exist_ok=True)
    cloudidfiles, \
    cloudidfiles_basetime, \
    cloudidfiles_datestring, \
    cloudidfiles_timestring = subset_files_timerange(tracking_outpath,
                                                     cloudid_filebase,
                                                     start_basetime,
                                                     end_basetime)
    nfiles = len(cloudidfiles)
    logger.info(f"Total number of files to process: {nfiles}")

    results = []
    # Loop over each pixel file
    for ifile in range(0, nfiles):
        # Find all matching time indices from stats file to the current cloudid file
        itrack, itime = np.array(
            np.where(
                np.abs(stats_basetime - cloudidfiles_basetime[ifile]) < match_pixel_dt_thresh)
        )

        # Get cloudnumbers for this time (file)
        file_trackindex = itrack
        file_cloudnumber = stats_cloudnumber[itrack, itime]
        file_trackstatus = stats_trackstatus[itrack, itime]

        # Cloudnumbers for merge/split
        file_mergecloudnumber = stats_mergecloudnumber[itrack, itime, :]
        file_splitcloudnumber = stats_splitcloudnumber[itrack, itime, :]
        if (file_mergecloudnumber.size > 0) & (file_splitcloudnumber.size > 0):
            # Get number of max merge/split for all clouds at this time (file)
            max_merge = np.sum(file_mergecloudnumber > 0, axis=1).max()
            max_split = np.sum(file_splitcloudnumber > 0, axis=1).max()
            # Subset arrays containing useful data to reduce array size
            file_mergecloudnumber = file_mergecloudnumber[:, :max_merge]
            file_splitcloudnumber = file_splitcloudnumber[:, :max_split]

        # General merge/split tracknumber
        file_mergetracknumber = stats_mergetracknumber[itrack, itime]
        file_splittracknumber = stats_splittracknumber[itrack, itime]

        # Serial
        if run_parallel == 0:
            result = map_feature(
                cloudidfiles[ifile],
                cloudidfiles_basetime[ifile],
                file_trackindex,
                file_cloudnumber,
                file_trackstatus,
                file_mergetracknumber,
                file_splittracknumber,
                file_mergecloudnumber,
                file_splitcloudnumber,
                trackstats_comments,
                config,
                pixeltracking_outpath,
                pixeltracking_filebase,
            )
        # Parallel
        elif run_parallel >= 1:
            result = dask.delayed(map_feature)(
                cloudidfiles[ifile],
                cloudidfiles_basetime[ifile],
                file_trackindex,
                file_cloudnumber,
                file_trackstatus,
                file_mergetracknumber,
                file_splittracknumber,
                file_mergecloudnumber,
                file_splitcloudnumber,
                trackstats_comments,
                config,
                pixeltracking_outpath,
                pixeltracking_filebase,
            )
            results.append(result)
        else:
            sys.exit('Valid parallelization flag not provided.')

    if run_parallel >= 1:
        # Trigger dask computation
        final_result = dask.compute(*results)
        wait(final_result)

    logger.info('Done with mapping features to pixel-level files')
    return