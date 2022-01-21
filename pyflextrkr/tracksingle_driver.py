import sys
import logging
import dask
from dask.distributed import wait
from pyflextrkr.ft_utilities import subset_files_timerange, match_drift_times
from pyflextrkr.tracksingle_drift import trackclouds

def tracksingle_driver(config):
    """
    Driver for tracking sequential pairs of idfeature files.

    Args:
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        Track data are written to netCDF files.
    """

    logger = logging.getLogger(__name__)
    logger.info('Tracking sequential pairs of idfeature files')

    tracking_outpath = config["tracking_outpath"]
    cloudid_filebase = config["cloudid_filebase"]
    start_basetime = config["start_basetime"]
    end_basetime = config["end_basetime"]
    run_parallel = config["run_parallel"]
    driftfile = config.get("driftfile", None)

    # Identify files to process
    cloudidfiles, \
    cloudidfiles_basetime, \
    cloudidfiles_datestring, \
    cloudidfiles_timestring = subset_files_timerange(tracking_outpath,
                                                     cloudid_filebase,
                                                     start_basetime,
                                                     end_basetime)
    cloudidfilestep = len(cloudidfiles)
    logger.info(f"Total number of files to process: {cloudidfilestep}")

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
    logger.debug('Looping over pairs of single files to track')

    # Create pairs of input filenames and times
    cloudid_filepairs = list(zip(cloudidfiles[0:-1], cloudidfiles[1::]))
    cloudid_basetimepairs = list(zip(cloudidfiles_basetime[0:-1], cloudidfiles_basetime[1::]))

    # Serial version
    if run_parallel == 0:
        for ifile in range(0, cloudidfilestep - 1):
            if driftfile is not None:
                trackclouds(
                    cloudid_filepairs[ifile],
                    cloudid_basetimepairs[ifile],
                    config,
                    drift_data=drift_data[ifile]
                )
            else:
                trackclouds(
                    cloudid_filepairs[ifile],
                    cloudid_basetimepairs[ifile],
                    config
                )

    # Parallel version
    elif run_parallel == 1:
        results = []
        for ifile in range(0, cloudidfilestep - 1):
            if driftfile is not None:
                result = dask.delayed(trackclouds)(
                    cloudid_filepairs[ifile],
                    cloudid_basetimepairs[ifile],
                    config,
                    drift_data=drift_data[ifile],
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

    logger.info('Done with tracking sequential pairs of idfeature files')
    return