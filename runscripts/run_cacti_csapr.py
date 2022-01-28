import os
import sys
import logging
from dask.distributed import Client, LocalCluster
from pyflextrkr.ft_utilities import load_config
from pyflextrkr.advection_radar import calc_mean_advection
from pyflextrkr.idfeature_driver import idfeature_driver
from pyflextrkr.tracksingle_driver import tracksingle_driver
from pyflextrkr.gettracks import gettracknumbers
from pyflextrkr.trackstats_driver import trackstats_driver
from pyflextrkr.mapfeature_driver import mapfeature_driver

# Name: run_cacti_csapr.py
# Purpose: Main script for tracking convective cells from CACTI CSAPR data
# Author: Zhe Feng (zhe.feng@pnnl.gov)

if __name__ == '__main__':

    # Set the logging message level
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load configuration file
    config_file = sys.argv[1]
    config = load_config(config_file)

    ################################################################################################
    # Initiate a local cluster for parallel processing
    if config['run_parallel'] == 1:
        cluster = LocalCluster(n_workers=config['nprocesses'], threads_per_worker=1)
        client = Client(cluster)
    elif config['run_parallel'] == 2:
        scheduler_file = os.path.join(os.environ["SCRATCH"], "scheduler.json")
        client = Client(scheduler_file=scheduler_file)

    # Step 0 - Run advection calculation
    if config['run_advection']:
        logger.info('Calculating domain mean advection.')
        driftfile = calc_mean_advection(config)
    else:
        driftfile = f'{config["stats_outpath"]}advection_' + \
                    f'{config["startdate"]}_{config["enddate"]}.nc'
    config.update({"driftfile": driftfile})

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

