import os
import sys
import logging
import dask
from dask.distributed import Client, LocalCluster
from pyflextrkr.ft_utilities import load_config, setup_logging
from pyflextrkr.preprocess_wrf_tb_rainrate_reflectivity import preprocess_wrf
from pyflextrkr.idfeature_driver import idfeature_driver
from pyflextrkr.tracksingle_driver import tracksingle_driver
from pyflextrkr.gettracks import gettracknumbers
from pyflextrkr.trackstats_driver import trackstats_driver
from pyflextrkr.identifymcs import identifymcs_tb
from pyflextrkr.matchtbpf_driver import match_tbpf_tracks
from pyflextrkr.robustmcs_radar import define_robust_mcs_radar
from pyflextrkr.mapfeature_driver import mapfeature_driver
from pyflextrkr.movement_speed import movement_speed


import subprocess
import time
FLUSH_MEM = os.environ.get("FLUSH_MEM")
INVALID_OS_CACHE = os.environ.get("INVALID_OS_CACHE")

def invalidate_all_files(logger,directory):
    target_dir = directory[:-1]
    logger.info(f"Executing command \"touch {target_dir}/*\"")
    command = f"touch {target_dir}/*"
    if 'input' not in target_dir:
        command = f"touch {target_dir}/*/*"
    subprocess.call(command, shell=True)


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

    # Step 0 - Preprocess wrfout files to get Tb, rainrate, reflectivity
    if config['run_preprocess']:
        preprocess_wrf(config)
        if INVALID_OS_CACHE == "TRUE":
            invalidate_all_files(logger,config['wrfout_path'])
            invalidate_all_files(logger,config['root_path'])
            
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
    
    file_invalid_time = []
    
    # Step 1 - Identify features
    if config['run_idfeature']:
        os.environ['CURR_TASK'] = 'run_idfeature'
        idfeature_driver(config)
        if INVALID_OS_CACHE == "TRUE":
            start_time = time.perf_counter()
            invalidate_all_files(logger,config['wrfout_path'])
            invalidate_all_files(logger,config['root_path'])
            elapsed_time = (time.perf_counter() - start_time) * 1000
            

    # Step 2 - Link features in time adjacent files
    if config['run_tracksingle']:
        if FLUSH_MEM == "TRUE":
            client.shutdown()
            dask_tmp_dir = config.get("dask_tmp_dir", "./")
            dask.config.set({'temporary-directory': dask_tmp_dir})
            # Local cluster
            cluster = LocalCluster(n_workers=config['nprocesses'], threads_per_worker=1)
            client = Client(cluster)
        if INVALID_OS_CACHE == "TRUE":
            start_time = time.perf_counter()
            invalidate_all_files(logger,config['wrfout_path'])
            invalidate_all_files(logger,config['root_path'])
            file_invalid_time.append((time.perf_counter() - start_time) * 1000)
        
        os.environ['CURR_TASK'] = 'run_tracksingle'
        tracksingle_driver(config)

    # Step 3 - Track features through the entire dataset
    if config['run_gettracks']:
        if FLUSH_MEM == "TRUE":
            client.shutdown()
            cluster = LocalCluster(n_workers=config['nprocesses'], threads_per_worker=1)
            client = Client(cluster)
        if INVALID_OS_CACHE == "TRUE":
            start_time = time.perf_counter()
            invalidate_all_files(logger,config['wrfout_path'])
            invalidate_all_files(logger,config['root_path'])
            file_invalid_time.append((time.perf_counter() - start_time) * 1000)
        
        os.environ['CURR_TASK'] = 'run_gettracks'
        tracknumbers_filename = gettracknumbers(config)

    # Step 4 - Calculate track statistics
    if config['run_trackstats']:
        if FLUSH_MEM == "TRUE":
            client.shutdown()
            cluster = LocalCluster(n_workers=config['nprocesses'], threads_per_worker=1)
            client = Client(cluster)
        if INVALID_OS_CACHE == "TRUE":
            start_time = time.perf_counter()
            invalidate_all_files(logger,config['wrfout_path'])
            invalidate_all_files(logger,config['root_path'])
            file_invalid_time.append((time.perf_counter() - start_time) * 1000)
        
        os.environ['CURR_TASK'] = 'run_trackstats'
        trackstats_filename = trackstats_driver(config)

    # Step 5 - Identify MCS using Tb
    if config['run_identifymcs']:
        if FLUSH_MEM == "TRUE":
            client.shutdown()
            cluster = LocalCluster(n_workers=config['nprocesses'], threads_per_worker=1)
            client = Client(cluster)
        if INVALID_OS_CACHE == "TRUE":
            start_time = time.perf_counter()
            invalidate_all_files(logger,config['wrfout_path'])
            invalidate_all_files(logger,config['root_path'])
            file_invalid_time.append((time.perf_counter() - start_time) * 1000)
            
        os.environ['CURR_TASK'] = 'run_identifymcs'
        mcsstats_filename = identifymcs_tb(config)

    # Step 6 - Match PF to MCS
    if config['run_matchpf']:
        if FLUSH_MEM == "TRUE":
            client.shutdown()
            cluster = LocalCluster(n_workers=config['nprocesses'], threads_per_worker=1)
            client = Client(cluster)
        if INVALID_OS_CACHE == "TRUE":
            start_time = time.perf_counter()
            invalidate_all_files(logger,config['wrfout_path'])
            invalidate_all_files(logger,config['root_path'])
            file_invalid_time.append((time.perf_counter() - start_time) * 1000)
        
        os.environ['CURR_TASK'] = 'run_matchpf'    
        pfstats_filename = match_tbpf_tracks(config)

    # Step 7 - Identify robust MCS
    if config['run_robustmcs']:
        if FLUSH_MEM == "TRUE":
            client.shutdown()
            cluster = LocalCluster(n_workers=config['nprocesses'], threads_per_worker=1)
            client = Client(cluster)
        if INVALID_OS_CACHE == "TRUE":
            start_time = time.perf_counter()
            invalidate_all_files(logger,config['wrfout_path'])
            invalidate_all_files(logger,config['root_path'])
            file_invalid_time.append((time.perf_counter() - start_time) * 1000)
        
        os.environ['CURR_TASK'] = 'run_robustmcs'
        robustmcsstats_filename = define_robust_mcs_radar(config)

    # Step 8 - Map tracking to pixel files
    if config['run_mapfeature']:
        if FLUSH_MEM == "TRUE":
            client.shutdown()
            cluster = LocalCluster(n_workers=config['nprocesses'], threads_per_worker=1)
            client = Client(cluster)
        if INVALID_OS_CACHE == "TRUE":
            start_time = time.perf_counter()
            invalidate_all_files(logger,config['wrfout_path'])
            invalidate_all_files(logger,config['root_path'])
            file_invalid_time.append((time.perf_counter() - start_time) * 1000)
        
        # Map robust MCS track numbers to pixel files (default)
        os.environ['CURR_TASK'] =  'run_mapfeature'
        mapfeature_driver(config, trackstats_filebase=mcsrobust_filebase)
        # # Map Tb-only MCS track numbers to pixel files (provide outpath_basename keyword)
        # mapfeature_driver(config, trackstats_filebase=mcstbstats_filebase, outpath_basename=mcstbmap_outpath)
        # # Map all Tb track numbers to pixel level files (provide outpath_basename keyword)
        # mapfeature_driver(config, trackstats_filebase=trackstats_filebase, outpath_basename=alltrackmap_outpath)

    # Step 9 - Movement speed calculation
    if config['run_speed']:
        if FLUSH_MEM == "TRUE":
            client.shutdown()
            cluster = LocalCluster(n_workers=config['nprocesses'], threads_per_worker=1)
            client = Client(cluster)
        if INVALID_OS_CACHE == "TRUE":
            start_time = time.perf_counter()
            invalidate_all_files(logger,config['wrfout_path'])
            invalidate_all_files(logger,config['root_path'])
            file_invalid_time.append((time.perf_counter() - start_time) * 1000)
        
        os.environ['CURR_TASK'] = 'run_speed'
        movement_speed(config)
    
    if INVALID_OS_CACHE == "TRUE":
        logger.info("File invaliation overhead : {:.2f} milliseconds".format(sum(file_invalid_time)))