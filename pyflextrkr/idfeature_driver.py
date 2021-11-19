import sys
import os
import logging
import dask
from dask.distributed import wait
from pyflextrkr.ft_utilities import subset_files_timerange
from pyflextrkr.idcells_radar import idcell_csapr

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

    clouddata_path = config["clouddata_path"]
    databasename = config["databasename"]
    start_basetime = config["start_basetime"]
    end_basetime = config["end_basetime"]
    time_format = config["time_format"]
    run_parallel = config["run_parallel"]
    feature_type = config["feature_type"]

    # Identify files to process
    logger.info('Identifying raw data files to process.')
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

    return