import sys
import logging
import dask
from dask.distributed import wait
from pyflextrkr.ft_utilities import subset_files_timerange

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
    # Load function depending on feature_type
    if feature_type == "generic":
        from pyflextrkr.idfeature_generic import idfeature_generic as id_feature
    elif feature_type == "radar_cells":
        from pyflextrkr.idcells_reflectivity import idcells_reflectivity as id_feature
    elif "tb_pf" in feature_type:
        from pyflextrkr.idclouds_tbpf import idclouds_tbpf as id_feature
    else:
        logger.critical(f"ERROR: Unknown feature_type: {feature_type}")
        logger.critical("Tracking will now exit.")
        sys.exit()

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

    # Serial
    if run_parallel == 0:
        for ifile in range(0, nfiles):
            id_feature(rawdatafiles[ifile], config)
    # Parallel
    elif run_parallel >= 1:
        results = []
        for ifile in range(0, nfiles):
            result = dask.delayed(id_feature)(rawdatafiles[ifile], config)
            results.append(result)
        final_result = dask.compute(*results)
        wait(final_result)
    else:
        sys.exit('Valid parallelization flag not provided')

    logger.info('Done with features from raw data.')
    return