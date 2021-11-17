import os, sys
import yaml
from dask.distributed import Client, LocalCluster
from pyflextrkr.ftfunctions import get_basetime_from_string
from pyflextrkr.advection_radar import calc_mean_advection
from pyflextrkr.idfeature_driver import idfeature_driver
from pyflextrkr.tracksingle_driver import tracksingle_driver
from pyflextrkr.gettracks import gettracknumbers
from pyflextrkr.trackstats_driver import trackstats_driver
from pyflextrkr.mapfeature_driver import mapfeature_driver
import logging

# Name: run_cacti_csapr.py

# Purpose: Master script for tracking convective cells from CACTI CSAPR data

# Author: Zhe Feng (zhe.feng@pnnl.gov)

if __name__ == '__main__':

    # Get configuration file name from input
    config_file = sys.argv[1]
    # Read configuration from yaml file
    stream = open(config_file, 'r')
    config = yaml.full_load(stream)

    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.CRITICAL)
    logger = logging.getLogger(__name__)

    ################################################################################################
    # Get tracking set up from config file
    ################################################################################################
    startdate = config['startdate']
    enddate = config['enddate']
    # Set up tracking output file locations
    tracking_outpath = config["root_path"] + "/tracking/"
    stats_outpath = config["root_path"] + "/stats/"
    pixeltracking_outpath = config['root_path'] + '/celltracking/' + startdate + '_' + enddate + '/'
    cloudid_filebase = config["datasource"] + '_' + config["datadescription"] + '_cloudid_'
    singletrack_filebase = 'track_'
    tracknumbers_filebase = 'tracknumbers_'
    trackstats_filebase = 'trackstats_'

    # Create output directories
    os.makedirs(tracking_outpath, exist_ok=True)
    os.makedirs(stats_outpath, exist_ok=True)
    os.makedirs(pixeltracking_outpath, exist_ok=True)

    # Calculate basetime for start and end date
    start_basetime = get_basetime_from_string(startdate)
    end_basetime = get_basetime_from_string(enddate)
    # Add newly defined variables to config
    config.update(
        {
            "tracking_outpath": tracking_outpath,
            "stats_outpath": stats_outpath,
            "pixeltracking_outpath": pixeltracking_outpath,
            "cloudid_filebase": cloudid_filebase,
            "singletrack_filebase": singletrack_filebase,
            "tracknumbers_filebase": tracknumbers_filebase,
            "trackstats_filebase": trackstats_filebase,
            "start_basetime": start_basetime,
            "end_basetime": end_basetime,
        }
    )

    ################################################################################################
    # Initiate a local cluster for parallel processing
    if config['run_parallel'] == 1:
        cluster = LocalCluster(n_workers=config['nprocesses'], threads_per_worker=1)
        client = Client(cluster)

    ################################################################################################
    # Step 0 - Run advection calculation
    if config['run_advection']:
        logger.info('Calculating domain mean advection.')
        driftfile = calc_mean_advection(config)
    else:
        driftfile = f'{config["stats_outpath"]}{config["datasource"]}_advection_{startdate}_{enddate}.nc'
    config.update({"driftfile": driftfile})

    ################################################################################################
    # Step 1 - Identify features
    if config['run_idfeature']:
        idfeature_driver(config)

    ################################################################################################
    # Step 2 - Link features in time adjacent files (single file tracking)
    if config['run_tracksingle']:
        tracksingle_driver(config)

    ################################################################################################
    # Step 3 - Track features through the entire dataset
    if config['run_gettracks']:
        tracknumbers_filename = gettracknumbers(config)
    else:
        tracknumbers_filename = f'{tracknumbers_filebase}{startdate}_{enddate}.nc'

    ################################################################################################
    # Step 4 - Calculate track statistics
    if config['run_trackstats']:
        trackstats_filename = trackstats_driver(config)
    else:
        trackstats_filename = f'{trackstats_filebase}{startdate}_{enddate}.nc'

    ################################################################################################
    # Step 5 - Map tracking to pixel files
    if config['run_mapfeature']:
        mapfeature_driver(config)

