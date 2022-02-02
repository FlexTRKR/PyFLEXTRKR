import os
import sys
import logging
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

    pixeltracking_outpath = config["pixeltracking_outpath"]
    tracking_outpath = config["tracking_outpath"]
    cloudid_filebase = config["cloudid_filebase"]
    start_basetime = config["start_basetime"]
    end_basetime = config["end_basetime"]
    run_parallel = config["run_parallel"]
    feature_type = config["feature_type"]
    # Load function depending on feature_type
    if feature_type == "vorticity":
        from pyflextrkr.mapgeneric import map_generic as map_feature
    elif feature_type == "radar_cells":
        from pyflextrkr.mapcell_radar import mapcell_radar as map_feature
    elif feature_type == "tb_pf":
        from pyflextrkr.mapmcspf import mapmcs_tb_pf as map_feature
    else:
        logger.critical(f"ERROR: Unknown feature_type: {feature_type}")
        logger.critical("Tracking will now exit.")
        sys.exit()

    # Create pixel tracking file output directory
    os.makedirs(pixeltracking_outpath, exist_ok=True)
    # Identify files to process
    cloudidfiles, \
    cloudidfiles_basetime, \
    cloudidfiles_datestring, \
    cloudidfiles_timestring = subset_files_timerange(tracking_outpath,
                                                     cloudid_filebase,
                                                     start_basetime,
                                                     end_basetime)
    nfiles = len(cloudidfiles)
    logger.info(f"Total number of files to process: {nfiles}")

    # Serial
    if run_parallel == 0:
        for ifile in range(0, nfiles):
            result = map_feature(
                cloudidfiles[ifile],
                cloudidfiles_basetime[ifile],
                config,
            )
    # Parallel
    elif run_parallel >= 1:
        results = []
        for ifile in range(0, nfiles):
            result = dask.delayed(map_feature)(
                cloudidfiles[ifile],
                cloudidfiles_basetime[ifile],
                config,
            )
            results.append(result)
        final_result = dask.compute(*results)
        wait(final_result)
    else:
        sys.exit('Valid parallelization flag not provided.')

    logger.info('Done with mapping features to pixel-level files')
    return