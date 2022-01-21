import sys
import logging
import dask
from dask.distributed import wait
from pyflextrkr.ft_utilities import subset_files_timerange
from pyflextrkr.idcells_radar import idcell_csapr
from pyflextrkr.idclouds_sat import idclouds_gpmmergir

def idfeature_driver(config):
    """
    Driver for feature identification.

    Args:
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        Feature identification are written to netCDF files.
    """

    logger = logging.getLogger(__name__)
    logger.info('Identifying features from raw data')

    clouddata_path = config["clouddata_path"]
    databasename = config["databasename"]
    start_basetime = config["start_basetime"]
    end_basetime = config["end_basetime"]
    time_format = config["time_format"]
    run_parallel = config["run_parallel"]
    feature_type = config["feature_type"]

    # Identify files to process
    infiles_info = subset_files_timerange(
        clouddata_path,
        databasename,
        start_basetime,
        end_basetime,
        time_format=time_format,
    )
    # Get file list
    rawdatafiles = infiles_info[0]
    nfiles = len(rawdatafiles)
    logger.info(f"Total number of files to process: {nfiles}")

    #######################################################################################
    # Satellite IR temperature & precipitation
    if feature_type == "tb_pf":
        # Serial version
        if run_parallel == 0:
            for ifile in range(0, nfiles):
                idclouds_gpmmergir(rawdatafiles[ifile], config)
        # Parallel version
        elif run_parallel == 1:
            results = []
            for ifile in range(0, nfiles):
                result = dask.delayed(idclouds_gpmmergir)(rawdatafiles[ifile], config)
                results.append(result)
            final_result = dask.compute(*results)
            wait(final_result)
        else:
            sys.exit('Valid parallelization flag not provided')

    #######################################################################################
    # Radar convective cells
    if feature_type == "radar_cells":

        # Serial version
        if run_parallel == 0:
            for ifile in range(0, nfiles):
                idcell_csapr(rawdatafiles[ifile], config)
        # Parallel version
        elif run_parallel == 1:
            results = []
            for ifile in range(0, nfiles):
                result = dask.delayed(idcell_csapr)(rawdatafiles[ifile], config)
                results.append(result)
            final_result = dask.compute(*results)
            wait(final_result)
        else:
            sys.exit('Valid parallelization flag not provided')

    logger.info('Done with features from raw data.')
    return