import sys
import os
import logging
import dask
from dask.distributed import wait
from pyflextrkr.ft_utilities import subset_files_timerange
from pyflextrkr.mapcell_radar import mapcell_radar

def mapfeature_driver(config):
    logger = logging.getLogger(__name__)

    pixeltracking_outpath = config["pixeltracking_outpath"]
    tracking_outpath = config["tracking_outpath"]
    cloudid_filebase = config["cloudid_filebase"]
    start_basetime = config["start_basetime"]
    end_basetime = config["end_basetime"]
    run_parallel = config["run_parallel"]
    feature_type = config["feature_type"]

    logger.info('Identifying which pixel level maps to generate for the cell tracks')

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

    #######################################################################################
    # Radar convective cells
    if feature_type == "radar_cells":

        # Call function
        if run_parallel == 0:
            # Serial version
            for ifile in range(0, len(cloudidfiles)):
                result = mapcell_radar(
                    cloudidfiles[ifile],
                    cloudidfiles_basetime[ifile],
                    config,
                )
        elif run_parallel == 1:
            # Parallel version
            results = []
            for ifile in range(0, len(cloudidfiles)):
                result = dask.delayed(mapcell_radar)(
                    cloudidfiles[ifile],
                    cloudidfiles_basetime[ifile],
                    config,
                )
                results.append(result)
            final_result = dask.compute(*results)
            # wait(final_result)

    return