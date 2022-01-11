import sys
import logging
import dask
from dask.distributed import wait
from pyflextrkr.ft_utilities import subset_files_timerange, match_drift_times
from pyflextrkr.tracksingle_drift import trackclouds as trackclouds_drift
from pyflextrkr.tracksingle import trackclouds as trackclouds

def tracksingle_driver(config):
    """
    Driver for tracking successive pairs of idfeature files.

    Args:
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        Track data are written to netCDF files.
    """

    logger = logging.getLogger(__name__)

    tracking_outpath = config["tracking_outpath"]
    cloudid_filebase = config["cloudid_filebase"]
    start_basetime = config["start_basetime"]
    end_basetime = config["end_basetime"]
    run_parallel = config["run_parallel"]
    if 'driftfile' in config:
        driftfile = config["driftfile"]
    else:
        driftfile = None

    # Identify files to process
    logger.info('Identifying cloudid files to process')
    cloudidfiles, \
    cloudidfiles_basetime, \
    cloudidfiles_datestring, \
    cloudidfiles_timestring = subset_files_timerange(tracking_outpath,
                                                     cloudid_filebase,
                                                     start_basetime,
                                                     end_basetime)
    cloudidfilestep = len(cloudidfiles)

    # Match advection data times with cloudid times
    if driftfile is not None:
        datetime_drift_match, \
        xdrifts_match, \
        ydrifts_match = match_drift_times(cloudidfiles_datestring,
                                          cloudidfiles_timestring,
                                          driftfile=driftfile)
        # Create matching triplets of drift data
        drift_data = list(zip(datetime_drift_match, xdrifts_match, ydrifts_match))

    # Call function
    logger.info('Tracking clouds between single files')

    # Create pairs of input filenames and times
    cloudid_filepairs = list(zip(cloudidfiles[0:-1], cloudidfiles[1::]))
    cloudid_basetimepairs = list(zip(cloudidfiles_basetime[0:-1], cloudidfiles_basetime[1::]))

    # Serial version
    if run_parallel == 0:
        for ifile in range(0, cloudidfilestep - 1):
            if driftfile is not None:
                trackclouds_drift(cloudid_filepairs[ifile],
                                  cloudid_basetimepairs[ifile],
                                  drift_data[ifile],
                                  config)
            else:
                trackclouds(cloudid_filepairs[ifile],
                            cloudid_basetimepairs[ifile],
                            config)
    # Parallel version
    elif run_parallel == 1:
        results = []
        for ifile in range(0, cloudidfilestep - 1):
            if driftfile is not None:
                result = dask.delayed(trackclouds_drift)(
                    cloudid_filepairs[ifile],
                    cloudid_basetimepairs[ifile],
                    drift_data[ifile],
                    config,
                )
            else:
                result = dask.delayed(trackclouds)(
                    cloudid_filepairs[ifile],
                    cloudid_basetimepairs[ifile],
                    config,
                )
            results.append(result)
        final_result = dask.compute(*results)
        wait(final_result)
    else:
        sys.exit('Valid parallelization flag not provided.')

    return