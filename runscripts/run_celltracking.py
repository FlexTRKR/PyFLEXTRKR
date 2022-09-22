import os
import sys
import logging
import dask
from dask.distributed import Client, LocalCluster
from pyflextrkr.ft_utilities import load_config, setup_logging
from pyflextrkr.idfeature_driver import idfeature_driver
from pyflextrkr.advection_tiles import calc_mean_advection
from pyflextrkr.tracksingle_driver import tracksingle_driver
from pyflextrkr.gettracks import gettracknumbers
from pyflextrkr.trackstats_driver import trackstats_driver
from pyflextrkr.mapfeature_driver import mapfeature_driver

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
        cluster = LocalCluster(n_workers=config['nprocesses'], threads_per_worker=1)
        client = Client(cluster)
        client.run(setup_logging)
    elif config['run_parallel'] == 2:
        # Dask-MPI
        scheduler_file = os.path.join(os.environ["SCRATCH"], "scheduler.json")
        client = Client(scheduler_file=scheduler_file)
        client.run(setup_logging)
    else:
        logger.info(f"Running in serial.")

    # Step 1 - Identify features
    if config['run_idfeature']:
        idfeature_driver(config)

    # Step 2 - Run advection calculation
    if config['run_advection']:
        logger.info('Calculating domain mean advection.')
        driftfile = calc_mean_advection(config)
    else:
        driftfile = f'{config["stats_outpath"]}advection_' + \
                    f'{config["startdate"]}_{config["enddate"]}.nc'
    config.update({"driftfile": driftfile})

    # Step 3 - Link features in time adjacent files
    if config['run_tracksingle']:
        tracksingle_driver(config)

    # Step 4 - Track features through the entire dataset
    if config['run_gettracks']:
        tracknumbers_filename = gettracknumbers(config)

    # Step 5 - Calculate track statistics
    if config['run_trackstats']:
        trackstats_filename = trackstats_driver(config)

    # Step 6 - Map tracking to pixel files
    if config['run_mapfeature']:
        mapfeature_driver(config)