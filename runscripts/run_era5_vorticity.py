import os
import sys
import logging
import dask
from dask.distributed import Client, LocalCluster
from pyflextrkr.ft_utilities import load_config, setup_logging
from pyflextrkr.idfeature_driver import idfeature_driver
from pyflextrkr.tracksingle_driver import tracksingle_driver
from pyflextrkr.gettracks import gettracknumbers
from pyflextrkr.trackstats_driver import trackstats_driver
from pyflextrkr.mapfeature_driver import mapfeature_driver

# Name: run_era5_vorticity.py
# Purpose: Main script for tracking vorticity from ERA5 data
# Author: Zhe Feng (zhe.feng@pnnl.gov)

if __name__ == '__main__':

    # Set the logging message level
    setup_logging()
    logger = logging.getLogger(__name__)

    # Load configuration file
    config_file = sys.argv[1]
    config = load_config(config_file)

    ################################################################################################
    # Parallel processing options
    if config['run_parallel'] == 1:
        # Set Dask temporary directory for workers
        dask_tmp_dir = config.get("dask_tmp_dir", "./")
        dask.config.set({'temporary-directory': dask_tmp_dir})
        # Local cluster
        cluster = LocalCluster(n_workers=config['nprocesses'], threads_per_worker=1, silence_logs=False)
        client = Client(cluster)
        client.run(setup_logging)
    elif config['run_parallel'] == 2:
        # Dask scheduler
        # Get the scheduler filename from input argument
        scheduler_file = sys.argv[2]
        timeout = config.get("timeout", 120)
        client = Client(scheduler_file=scheduler_file)
        # client.wait_for_workers(n_workers=n_workers, timeout=timeout)
        client.run(setup_logging)
    else:
        logger.info(f"Running in serial.")

    # Step 1 - Identify features
    if config['run_idfeature']:
        idfeature_driver(config)

    # Step 2 - Link features in time adjacent files
    if config['run_tracksingle']:
        tracksingle_driver(config)

    # Step 3 - Track features through the entire dataset
    if config['run_gettracks']:
        tracknumbers_filename = gettracknumbers(config)

    # Step 4 - Calculate track statistics
    if config['run_trackstats']:
        trackstats_filename = trackstats_driver(config)

    # Step 5 - Map tracking to pixel files
    if config['run_mapfeature']:
        mapfeature_driver(config)

