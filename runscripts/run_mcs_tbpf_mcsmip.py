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
from pyflextrkr.identifymcs import identifymcs_tb
from pyflextrkr.matchtbpf_driver import match_tbpf_tracks
from pyflextrkr.robustmcspf_saag import define_robust_mcs_pf
from pyflextrkr.mapfeature_driver import mapfeature_driver
from pyflextrkr.movement_speed import movement_speed
from pyflextrkr.remap_healpix_zarr import remap_to_healpix_zarr

if __name__ == '__main__':

    # Set the logging message level
    setup_logging()
    logger = logging.getLogger(__name__)

    # Load configuration file
    config_file = sys.argv[1]
    config = load_config(config_file)

    # Specify track statistics file basename and pixel-level output directory
    # for mapping track numbers to pixel files
    trackstats_filebase = config['trackstats_filebase']  # All Tb tracks
    mcstbstats_filebase = config['mcstbstats_filebase']  # MCS tracks defined by Tb-only
    mcsrobust_filebase = config['mcsrobust_filebase']   # MCS tracks defined by Tb+PF
    mcstbmap_outpath = 'mcstracking_tb'     # Output directory for Tb-only MCS
    alltrackmap_outpath = 'ccstracking'     # Output directory for all Tb tracks

    ################################################################################################
    # Parallel processing options
    run_parallel = config.get('run_parallel', 0)
    if run_parallel == 1:
        # Set Dask temporary directory for workers
        dask_tmp_dir = config.get("dask_tmp_dir", "./")
        dask.config.set({'temporary-directory': dask_tmp_dir})
        # Local cluster
        cluster = LocalCluster(n_workers=config['nprocesses'], threads_per_worker=1)
        client = Client(cluster)
        client.run(setup_logging)
    elif run_parallel == 2:
        # Dask scheduler
        # Get the scheduler filename from input argument
        scheduler_file = sys.argv[2]
        timeout = config.get("timeout", 600)
        logger.info(f"Connecting to Dask scheduler at {scheduler_file} with timeout {timeout}")
        try:
            client = Client(scheduler_file=scheduler_file, timeout=timeout)
            client.run(setup_logging)
            logger.info("Successfully connected to Dask scheduler")
        except Exception as e:
            logger.error(f"Failed to connect to Dask scheduler: {e}")
            sys.exit(1)
    else:
        logger.info(f"Running in serial.")

    # Step 1 - Identify features
    if config.get('run_idfeature', False):
        idfeature_driver(config)

    # Step 2 - Link features in time adjacent files
    if config.get('run_tracksingle', False):
        tracksingle_driver(config)

    # Step 3 - Track features through the entire dataset
    if config.get('run_gettracks', False):
        tracknumbers_filename = gettracknumbers(config)

    # Step 4 - Calculate track statistics
    if config.get('run_trackstats', False):
        trackstats_filename = trackstats_driver(config)

    # Step 5 - Identify MCS using Tb
    if config.get('run_identifymcs', False):
        mcsstats_filename = identifymcs_tb(config)

    # Step 6 - Match PF to MCS
    if config.get('run_matchpf', False):
        pfstats_filename = match_tbpf_tracks(config)

    # Step 7 - Identify robust MCS
    if config.get('run_robustmcs', False):
        robustmcsstats_filename = define_robust_mcs_pf(config)

    # Step 8 - Map tracking to pixel files
    if config.get('run_mapfeature', False):
        # Map robust MCS track numbers to pixel files (default)
        mapfeature_driver(config, trackstats_filebase=mcsrobust_filebase)
        # Map Tb-only MCS track numbers to pixel files (provide outpath_basename keyword)
        # mapfeature_driver(config, trackstats_filebase=mcstbstats_filebase, outpath_basename=mcstbmap_outpath)
        # Map all Tb track numbers to pixel level files (provide outpath_basename keyword)
        # mapfeature_driver(config, trackstats_filebase=trackstats_filebase, outpath_basename=alltrackmap_outpath)

    # Step 9 - Movement speed calculation
    if config.get('run_speed', False):
        movement_speed(config)

    # Step 10 - Remap MCS mask to HEALPix grid
    if config.get('run_remap_healpix', False):
        remap_to_healpix_zarr(config)

    # Clean up resources
    if run_parallel >= 1:
        logger.info("Shutting down Dask client...")
        client.close()
        logger.info("Dask client shutdown complete.")