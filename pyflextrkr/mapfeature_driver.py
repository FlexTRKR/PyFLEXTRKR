import os
import logging
import dask
from dask.distributed import wait
from pyflextrkr.ft_utilities import subset_files_timerange
from pyflextrkr.mapcell_radar import mapcell_radar
from pyflextrkr.mapmcspf import mapmcs_tb_pf

def mapfeature_driver(config):
    logger = logging.getLogger(__name__)

    pixeltracking_outpath = config["pixeltracking_outpath"]
    tracking_outpath = config["tracking_outpath"]
    cloudid_filebase = config["cloudid_filebase"]
    start_basetime = config["start_basetime"]
    end_basetime = config["end_basetime"]
    run_parallel = config["run_parallel"]
    feature_type = config["feature_type"]

    logger.info('Mapping tracked features to pixel-level files')

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

    #######################################################################################
    # Satellite IR temperature & precipitation
    if feature_type == "tb_pf":
        # Serial
        if run_parallel == 0:
            for ifile in range(0, nfiles):
                result = mapmcs_tb_pf(
                    cloudidfiles[ifile],
                    cloudidfiles_basetime[ifile],
                    config,
                )

        # Parallel
        elif run_parallel == 1:
            results = []
            for ifile in range(0, nfiles):
                result = dask.delayed(mapmcs_tb_pf)(
                    cloudidfiles[ifile],
                    cloudidfiles_basetime[ifile],
                    config,
                )
                results.append(result)
            final_result = dask.compute(*results)


    #######################################################################################
    # Radar convective cells
    if feature_type == "radar_cells":

        # Serial
        if run_parallel == 0:
            # Serial version
            for ifile in range(0, nfiles):
                result = mapcell_radar(
                    cloudidfiles[ifile],
                    cloudidfiles_basetime[ifile],
                    config,
                )

        # Parallel
        elif run_parallel == 1:
            results = []
            for ifile in range(0, nfiles):
                result = dask.delayed(mapcell_radar)(
                    cloudidfiles[ifile],
                    cloudidfiles_basetime[ifile],
                    config,
                )
                results.append(result)
            final_result = dask.compute(*results)
            # wait(final_result)

    logger.info('Done with mapping features to pixel-level files')
    return