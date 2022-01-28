import os
import sys
import logging
from dask.distributed import Client, LocalCluster
from pyflextrkr.ft_utilities import load_config
from pyflextrkr.idfeature_driver import idfeature_driver
from pyflextrkr.tracksingle_driver import tracksingle_driver
from pyflextrkr.gettracks import gettracknumbers
from pyflextrkr.trackstats_driver import trackstats_driver
from pyflextrkr.identifymcs import identifymcs_tb
from pyflextrkr.matchtbpf_driver import match_tbpf_tracks
from pyflextrkr.robustmcspf import define_robust_mcs_pf
from pyflextrkr.mapfeature_driver import mapfeature_driver
from pyflextrkr.movement_speed import movement_speed

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

    # Step 5 - Identify MCS using Tb
    if config['run_identifymcs']:
        mcsstats_filename = identifymcs_tb(config)

    # Step 6 - Match PF to MCS
    if config['run_matchpf']:
        pfstats_filename = match_tbpf_tracks(config)

    # Step 7 - Identify robust MCS
    if config['run_robustmcs']:
        robustmcsstats_filename = define_robust_mcs_pf(config)

    # Step 8 - Map tracking to pixel files
    if config['run_mapfeature']:
        mapfeature_driver(config)

    # Step 9 - Movement speed calculation
    if config['run_speed']:
        movement_speed(config)