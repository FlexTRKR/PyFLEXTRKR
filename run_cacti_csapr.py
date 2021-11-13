import os, sys
import datetime, calendar
from pytz import utc
import yaml
import dask
from dask.distributed import Client, LocalCluster, wait
from pyflextrkr.ftfunctions import subset_files_timerange, match_drift_times
from pyflextrkr.advection_radar import calc_mean_advection
from pyflextrkr.idcells_radar import idcell_csapr
from pyflextrkr.tracksingle_drift import trackclouds
from pyflextrkr.gettracks import gettracknumbers
from pyflextrkr.trackstats_driver import trackstats_driver
from pyflextrkr.mapcell_radar import mapcell_radar
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
    run_parallel = config['run_parallel']
    nprocesses = config['nprocesses']
    databasename = config['databasename']
    datasource = config['datasource']
    datadescription = config['datadescription']
    clouddata_path = config['clouddata_path']

    # Set up tracking output file locations
    tracking_outpath = config['tracking_outpath']
    stats_outpath = config['stats_outpath']
    pixeltracking_outpath = config['pixeltracking_outpath']+ '/' + startdate + '_' + enddate + '/'

    cloudid_filebase = datasource + '_' + datadescription + '_cloudid_'
    singletrack_filebase = 'track_'
    tracknumbers_filebase = 'tracknumbers_'
    trackstats_filebase = 'trackstats_'


    ################################################################################################
    # Execute tracking scripts

    # Create output directories
    os.makedirs(tracking_outpath, exist_ok=True)
    os.makedirs(stats_outpath, exist_ok=True)
    # os.makedirs(pixeltracking_outpath, exist_ok=True)

    ########################################################################
    # Calculate basetime of start and end date
    TEMP_starttime = datetime.datetime(int(startdate[0:4]), int(startdate[4:6]), int(startdate[6:8]), \
                                        int(startdate[9:11]), int(startdate[11:]), 0, tzinfo=utc)
    start_basetime = calendar.timegm(TEMP_starttime.timetuple())
    TEMP_endtime = datetime.datetime(int(enddate[0:4]), int(enddate[4:6]), int(enddate[6:8]), \
                                        int(enddate[9:11]), int(enddate[11:]), 0, tzinfo=utc)
    end_basetime = calendar.timegm(TEMP_endtime.timetuple())

    # Add newly defined variables to config
    config.update(
        {
            "cloudid_filebase": cloudid_filebase,
            "singletrack_filebase": singletrack_filebase,
            "tracknumbers_filebase": tracknumbers_filebase,
            "trackstats_filebase": trackstats_filebase,
            "pixeltracking_outpath": pixeltracking_outpath,
            "start_basetime": start_basetime,
            "end_basetime": end_basetime,
        }
    )

    # Initiate a local cluster for parallel processing
    if run_parallel == 1:
        cluster = LocalCluster(n_workers=nprocesses, threads_per_worker=1)
        client = Client(cluster)

    ################################################################################################
    # Step 0 - Run advection calculation
    if config['run_advection']:
        logger.info('Calculating domain mean advection.')
        driftfile = calc_mean_advection(config)
    else:
        driftfile = f'{stats_outpath}{datasource}_advection_{startdate}_{enddate}.nc'

    ################################################################################################
    # Step 1 - Identify clouds / features in the data
    if config['run_idclouds']:
        # Identify files to process
        logger.info('Identifying raw data files to process.')
        infiles_info = subset_files_timerange(
            clouddata_path,
            databasename,
            start_basetime,
            end_basetime,
            time_format=config["time_format"]
        )
        rawdatafiles = infiles_info[0]
        nfiles = len(rawdatafiles)

        # Call function
        if run_parallel == 0:
            # Serial version
            for ifile in range(0, nfiles):
                idcell_csapr(rawdatafiles[ifile], config)
        elif run_parallel == 1:
            # Parallel version
            results = []
            for ifile in range(0, nfiles):
                result = dask.delayed(idcell_csapr)(rawdatafiles[ifile], config)
                results.append(result)
            final_result = dask.compute(*results)
            wait(final_result)
        else:
            sys.exit('Valid parallelization flag not provided')

    ################################################################################################
    # Step 2 - Link clouds / features in time adjacent files (single file tracking)
    if config['run_tracksingle']:
        ################################################################
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
        datetime_drift_match, \
        xdrifts_match, \
        ydrifts_match = match_drift_times(cloudidfiles_datestring,
                                          cloudidfiles_timestring,
                                          driftfile=driftfile)

        # Call function
        logger.info('Tracking clouds between single files')

        # Create pairs of input filenames and times
        cloudid_filepairs = list(zip(cloudidfiles[0:-1], cloudidfiles[1::]))
        cloudid_basetimepairs = list(zip(cloudidfiles_basetime[0:-1], cloudidfiles_basetime[1::]))
        # Create matching triplets of drift data
        drift_data = list(zip(datetime_drift_match, xdrifts_match, ydrifts_match))

        if run_parallel == 0:
            # Serial version
            for ifile in range(0, cloudidfilestep-1):
                trackclouds(cloudid_filepairs[ifile],
                            cloudid_basetimepairs[ifile],
                            drift_data[ifile],
                            config)
        elif run_parallel == 1:
            # Parallel version
            results = []
            for ifile in range(0, cloudidfilestep-1):
                result = dask.delayed(trackclouds)(
                    cloudid_filepairs[ifile],
                    cloudid_basetimepairs[ifile],
                    drift_data[ifile],
                    config,
                )
                results.append(result)
            final_result = dask.compute(*results)
            wait(final_result)
        else:
            sys.exit('Valid parallelization flag not provided.')

    ################################################################################################
    # Step 3 - Track clouds / features through the entire dataset
    if config['run_gettracks']:
        # Call function
        logger.info('Getting track numbers')
        logger.info('tracking_outpath:' + tracking_outpath)
        tracknumbers_filename = gettracknumbers(config)
        logger.info('Get track numbers done.')
    else:
        tracknumbers_filename = f'{tracknumbers_filebase}{startdate}_{enddate}.nc'

    ################################################################################################
    # Step 4 - Calculate track statistics
    if config['run_finalstats']:
        logger.info('Calculating cell statistics')
        trackstats_filename = trackstats_driver(config)
    else:
        trackstats_filename = f'{trackstats_filebase}{startdate}_{enddate}.nc'

    ################################################################################################
    # Step 5 - Create pixel files with cell tracks
    if config['run_labelcell']:
        logger.info('Identifying which pixel level maps to generate for the cell tracks')

        # Create pixel tracking file output directory
        os.makedirs(pixeltracking_outpath, exist_ok=True)
        # Identify files to process
        cloudidfiles, \
        cloudidfiles_basetime, \
        cloudidfiles_datestring, \
        cloudidfiles_timestring = subset_files_timerange(tracking_outpath,
                                                         cloudid_filebase,
                                                         start_basetime,
                                                         end_basetime)

        # Call function
        if run_parallel == 0:
            # Serial version
            for ifile in range(0, len(cloudidfiles)):
                result = mapcell_radar(
                    cloudidfiles[ifile],
                    cloudidfiles_basetime[ifile],
                    config,
                )
        elif run_parallel == 1:
            # Parallel version
            results = []
            for ifile in range(0, len(cloudidfiles)):
                result = dask.delayed(mapcell_radar)(
                    cloudidfiles[ifile],
                    cloudidfiles_basetime[ifile],
                    config,
                )
                results.append(result)
            final_result = dask.compute(*results)

