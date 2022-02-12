import os
import sys
import logging
import numpy as np
import xarray as xr
import dask
from dask.distributed import wait
from pyflextrkr.ft_utilities import subset_files_timerange

def mapfeature_driver(config):
    """
    Map tracked features to pixel level files.

    Args:
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        None.
    """
    logger = logging.getLogger(__name__)
    logger.info('Mapping tracked features to pixel-level files')

    stats_path = config["stats_outpath"]
    pixeltracking_outpath = config["pixeltracking_outpath"]
    tracking_outpath = config["tracking_outpath"]
    cloudid_filebase = config["cloudid_filebase"]
    startdate = config["startdate"]
    enddate = config["enddate"]
    start_basetime = config["start_basetime"]
    end_basetime = config["end_basetime"]
    # Minimum time difference threshold [second] to match track stats and cloudid files
    match_tbpf_dt_thresh = config.get("match_tbpf_dt_thresh", 60)
    run_parallel = config["run_parallel"]
    feature_type = config["feature_type"]
    # Load function depending on feature_type
    if feature_type == "vorticity":
        from pyflextrkr.mapgeneric import map_generic as map_feature
        trackstats_filebase = config["trackstats_filebase"]
    elif feature_type == "radar_cells":
        from pyflextrkr.mapcell_radar import mapcell_radar as map_feature
        trackstats_filebase = config["trackstats_filebase"]
    elif feature_type == "tb_pf":
        from pyflextrkr.mapmcspf import mapmcs_tb_pf as map_feature
        trackstats_filebase = config["mcsrobust_filebase"]
    else:
        logger.critical(f"ERROR: Unknown feature_type: {feature_type}")
        logger.critical("Tracking will now exit.")
        sys.exit()

    #########################################################################################
    # Read track stats
    trackstats_file = f"{stats_path}{trackstats_filebase}{startdate}_{enddate}.nc"
    ds = xr.open_dataset(
        trackstats_file,
        mask_and_scale=False,
        decode_times=False,
    ).compute()
    stats_basetime = ds["base_time"].data
    stats_cloudnumber = ds["cloudnumber"].data
    stats_trackstatus = ds["track_status"].data
    trackstats_comments = ds["track_status"].comments
    if feature_type == "tb_pf":
        stats_mergecloudnumber = ds["merge_cloudnumber"].data
        stats_splitcloudnumber = ds["split_cloudnumber"].data
    else:
        stats_mergecloudnumber = ds["merge_tracknumbers"].data
        stats_splitcloudnumber = ds["split_tracknumbers"].data
    ds.close()

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
    # Serial
    # if run_parallel == 0:
    for ifile in range(0, nfiles):
        # Find all matching time indices from stats file to the current cloudid file
        itrack, itime = np.array(
            np.where(
                np.abs(stats_basetime - cloudidfiles_basetime[ifile]) < match_tbpf_dt_thresh)
        )

        # Get cloudnumbers for this time (file)
        file_trackindex = itrack
        file_cloudnumber = stats_cloudnumber[itrack, itime]
        file_trackstatus = stats_trackstatus[itrack, itime]

        # MCS-specific merge/split cloudnumbers
        if feature_type == "tb_pf":
            file_mergecloudnumber = stats_mergecloudnumber[itrack, itime, :]
            file_splitcloudnumber = stats_splitcloudnumber[itrack, itime, :]
            if (file_mergecloudnumber.size > 0) & (file_splitcloudnumber.size > 0):
                # Get number of max merge/split for all clouds at this time (file)
                max_merge = np.sum(file_mergecloudnumber > 0, axis=1).max()
                max_split = np.sum(file_splitcloudnumber > 0, axis=1).max()
                # Subset arrays containing useful data to reduce array size
                file_mergecloudnumber = file_mergecloudnumber[:, :max_merge]
                file_splitcloudnumber = file_splitcloudnumber[:, :max_split]
        else:
            file_mergecloudnumber = stats_mergecloudnumber[itrack, itime]
            file_splitcloudnumber = stats_splitcloudnumber[itrack, itime]

        # Serial
        if run_parallel == 0:
            result = map_feature(
                cloudidfiles[ifile],
                cloudidfiles_basetime[ifile],
                file_trackindex,
                file_cloudnumber,
                file_trackstatus,
                file_mergecloudnumber,
                file_splitcloudnumber,
                trackstats_comments,
                config,
            )
        # Parallel
        elif run_parallel >= 1:
            result = dask.delayed(map_feature)(
                cloudidfiles[ifile],
                cloudidfiles_basetime[ifile],
                file_trackindex,
                file_cloudnumber,
                file_trackstatus,
                file_mergecloudnumber,
                file_splitcloudnumber,
                trackstats_comments,
                config,
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