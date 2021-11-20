import sys
import logging
from dask.distributed import Client, LocalCluster
from pyflextrkr.ft_utilities import load_config
from pyflextrkr.idfeature_driver import idfeature_driver


if __name__ == '__main__':

    # Set the logging message level
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Load configuration file
    config_file = sys.argv[1]
    config = load_config(config_file)

    ################################################################################################
    # Initiate a local cluster for parallel processing
    if config['run_parallel'] == 1:
        cluster = LocalCluster(n_workers=config['nprocesses'], threads_per_worker=1)
        client = Client(cluster)

    # Step 1 - Identify features
    if config['run_idfeature']:
        idfeature_driver(config)

