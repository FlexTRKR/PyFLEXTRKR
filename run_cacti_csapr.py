import numpy as np
import os, sys
import datetime, calendar
from pytz import utc
import yaml
from multiprocessing import Pool
from itertools import repeat
from pyflextrkr.ftfunctions import subset_files_timerange, match_drift_times
from pyflextrkr.advection_radar import calc_mean_advection
from pyflextrkr.idcells_radar import idcell_csapr
from pyflextrkr.tracksingle_drift import trackclouds
from pyflextrkr.gettracks import gettracknumbers
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

    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)

    ################################################################################################
    # Get tracking set up from config file
    ################################################################################################
    run_advection = config['run_advection']
    run_idclouds = config['run_idclouds']
    run_tracksingle = config['run_tracksingle']
    run_gettracks = config['run_gettracks']
    run_finalstats = config['run_finalstats']
    run_labelcell = config['run_labelcell']
    startdate = config['startdate']
    enddate = config['enddate']
    run_parallel = config['run_parallel']
    nprocesses = config['nprocesses']
    databasename = config['databasename']
    datasource = config['datasource']
    datadescription = config['datadescription']
    root_path = config['root_path']
    clouddata_path = config['clouddata_path']
    terrain_file = config['terrain_file']
    if "driftfile" in config:
        driftfile = config['driftfile']
        if os.path.isfile(driftfile):
            logger.info(f'Drift file already exist: {driftfile}')
            logger.info('Will be overwritten.')
    DBZ_THRESHOLD = config['DBZ_THRESHOLD']
    MED_FILT_LEN = config['MED_FILT_LEN']
    MAX_MOVEMENT_MPS = config['MAX_MOVEMENT_MPS']
    keep_singlemergesplit = config['keep_singlemergesplit']
    cloudid_version = config['cloudid_version']
    track_version = config['track_version']
    tracknumber_version = config['tracknumber_version']
    curr_id_version = config['curr_id_version']
    curr_track_version = config['curr_track_version']
    curr_tracknumbers_version = config['curr_tracknumbers_version']
    geolimits = np.array(config['geolimits'])
    pixel_radius = config['pixel_radius']
    timegap = config['timegap']
    area_thresh = config['area_thresh']
    miss_thresh = config['miss_thresh']
    othresh = config['othresh']
    lengthrange = config['lengthrange']
    maxnclouds = config['maxnclouds']
    nmaxlinks = config['nmaxlinks']
    # maincloud_duration = config['maincloud_duration']
    # merge_duration = config['merge_duration']
    # split_duration = config['split_duration']
    datatimeresolution = config['datatimeresolution']
    dimname = config['dimname']
    numbername = config['numbername']
    typename = config['typename']
    npxname = config['npxname']
    celltracking_filebase = config['celltracking_filebase']
    # Set up tracking output file locations
    tracking_outpath = root_path + 'tracking/'
    stats_outpath = root_path + 'stats/'
    celltracking_outpath = root_path + 'celltracking/' + startdate + '_' + enddate + '/'


    ################################################################################################
    # Execute tracking scripts

    # Create output directories
    os.makedirs(tracking_outpath, exist_ok=True)
    os.makedirs(stats_outpath, exist_ok=True)
    # os.makedirs(celltracking_outpath, exist_ok=True)

    # Set default driftfile if not specified in config file
    if "driftfile" not in config:
        driftfile = f'{stats_outpath}{datasource}_advection_all.nc'

    ########################################################################
    # Calculate basetime of start and end date
    TEMP_starttime = datetime.datetime(int(startdate[0:4]), int(startdate[4:6]), int(startdate[6:8]), \
                                        int(startdate[9:11]), int(startdate[11:]), 0, tzinfo=utc)
    start_basetime = calendar.timegm(TEMP_starttime.timetuple())

    TEMP_endtime = datetime.datetime(int(enddate[0:4]), int(enddate[4:6]), int(enddate[6:8]), \
                                        int(enddate[9:11]), int(enddate[11:]), 0, tzinfo=utc)
    end_basetime = calendar.timegm(TEMP_endtime.timetuple())

    ################################################################################################
    # Step 0 - Run advection calculation
    if run_advection == 1:
        logger.info('Calculating domain mean advection.')

        status = calc_mean_advection(
                    clouddata_path,
                    driftfile,
                    DBZ_THRESHOLD=DBZ_THRESHOLD,
                    dx=pixel_radius,
                    dy=pixel_radius,
                    MED_FILT_LEN=MED_FILT_LEN,
                    MAX_MOVEMENT_MPS=MAX_MOVEMENT_MPS,
                    datatimeresolution=datatimeresolution,
                    nprocesses=nprocesses,
                    )

    ################################################################################################
    # Step 1 - Identify clouds / features in the data
    if run_idclouds == 1:
        # Identify files to process
        logger.info('Identifying raw data files to process.')
        rawdatafiles, \
        files_basetime, \
        files_datestring, \
        files_timestring = subset_files_timerange(clouddata_path,
                                                    databasename,
                                                    start_basetime,
                                                    end_basetime)

        # Generate input lists
        idclouds_input = zip(rawdatafiles, files_datestring, files_timestring, files_basetime, \
                            repeat(datasource), repeat(datadescription), repeat(cloudid_version), \
                            repeat(tracking_outpath), repeat(startdate), repeat(enddate), \
                            repeat(pixel_radius), repeat(area_thresh), repeat(miss_thresh))

        ## Call function
        if run_parallel == 0:
            # Serial version
            for ifile in range(0, filestep):
                idcell_csapr(rawdatafiles[ifile], files_datestring[ifile], files_timestring[ifile], files_basetime[ifile], \
                                datasource, datadescription, cloudid_version, \
                                tracking_outpath, startdate, enddate, \
                                pixel_radius, area_thresh, miss_thresh)
        elif run_parallel == 1:
            # Parallel version
            logger.info('Identifying clouds')
            pool = Pool(nprocesses)
            pool.starmap(idcell_csapr, idclouds_input)
            pool.close()
            pool.join()
        else:
            sys.exit('Valid parallelization flag not provided')

        cloudid_filebase = datasource + '_' + datadescription + '_cloudid' + cloudid_version + '_'

    ################################################################################################
    # Step 2 - Link clouds / features in time adjacent files (single file tracking)

    # Determine if identification portion of the code run.
    # If not, set the version name and filename using names specified in the constants section
    if run_idclouds == 0:
        cloudid_filebase =  datasource + '_' + datadescription + '_cloudid' + curr_id_version + '_'

    # Call function
    if run_tracksingle == 1:
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

        # Generate input lists
        list_trackingoutpath = [tracking_outpath]*(cloudidfilestep-1)
        list_trackversion = [track_version]*(cloudidfilestep-1)
        list_timegap = np.ones(cloudidfilestep-1)*timegap
        list_nmaxlinks = np.ones(cloudidfilestep-1)*nmaxlinks
        list_othresh = np.ones(cloudidfilestep-1)*othresh
        list_startdate = [startdate]*(cloudidfilestep-1)
        list_enddate = [enddate]*(cloudidfilestep-1)

        # Call function
        logger.info('Tracking clouds between single files')

        trackclouds_input = list(zip(cloudidfiles[0:-1], cloudidfiles[1::], \
                                cloudidfiles_datestring[0:-1], cloudidfiles_datestring[1::], \
                                cloudidfiles_timestring[0:-1], cloudidfiles_timestring[1::], \
                                cloudidfiles_basetime[0:-1], cloudidfiles_basetime[1::], \
                                list_trackingoutpath, list_trackversion, list_timegap, \
                                list_nmaxlinks, list_othresh, list_startdate, list_enddate, \
                                datetime_drift_match, xdrifts_match, ydrifts_match))

        if run_parallel == 0:
            # Serial version
            for ifile in range(0, cloudidfilestep-1):
                trackclouds(trackclouds_input[ifile])
        elif run_parallel == 1:
            # parallelize version
            pool = Pool(nprocesses)
            pool.map(trackclouds, trackclouds_input)
            pool.close()
            pool.join()
        else:
            sys.exit('Valid parallelization flag not provided.')

        singletrack_filebase = 'track' + track_version + '_'

    ################################################################################################
    # Step 3 - Track clouds / features through the entire dataset

    # Determine if single file tracking code ran.
    # If not, set the version name and filename using names specified in the constants section
    if run_tracksingle == 0:
        singletrack_filebase = 'track' + curr_track_version + '_'

    # Call function
    if run_gettracks == 1:

        # Call function
        logger.info('Getting track numbers')
        logger.info('tracking_out:' + tracking_outpath)
        gettracknumbers(datasource, datadescription, tracking_outpath, stats_outpath, startdate, enddate, \
                        timegap, maxnclouds, cloudid_filebase, npxname, tracknumber_version, singletrack_filebase, \
                        keepsingletrack=keep_singlemergesplit, removestartendtracks=1)
        tracknumbers_filebase = 'tracknumbers' + tracknumber_version
        logger.info('tracking_out done')

    ################################################################################################
    # Step 4 - Calculate track statistics

    # Determine if the tracking portion of the code ran.
    # If not, set teh version name and filename using those specified in the constants section
    if run_gettracks == 0:
        tracknumbers_filebase = 'tracknumbers' + curr_tracknumbers_version

    # Call function
    if run_finalstats == 1:
        logger.info('Calculating cell statistics')

        #
        if run_parallel == 0:
            from pyflextrkr.trackstats_radar import trackstats_radar
            # Call serial version of trackstats
            trackstats_radar(datasource, datadescription, pixel_radius, datatimeresolution, geolimits, area_thresh, \
                            startdate, enddate, timegap, cloudid_filebase, tracking_outpath, stats_outpath, \
                            track_version, tracknumber_version, tracknumbers_filebase, terrain_file, lengthrange=lengthrange)

        elif run_parallel == 1:
            from pyflextrkr.trackstats_radar_parallel import trackstats_radar
            # Call parallel version of trackstats
            trackstats_radar(datasource, datadescription, pixel_radius, datatimeresolution, geolimits, area_thresh, \
                            startdate, enddate, timegap, cloudid_filebase, tracking_outpath, stats_outpath, \
                            track_version, tracknumber_version, tracknumbers_filebase, terrain_file, lengthrange, \
                            nprocesses=nprocesses)

        else:
            sys.ext('Valid parallelization flag not provided')

        trackstats_filebase = 'stats_tracknumbers' + tracknumber_version + '_'


    ################################################################################################
    # Step 5 - Create pixel files with cell tracks

    # Determine if final statistics portion ran.
    # If not, set the version name and filename using those specified in the constants section
    if run_finalstats == 0:
        trackstats_filebase = 'stats_tracknumbers' + curr_tracknumbers_version + '_'

    if run_labelcell == 1:
        logger.info('Identifying which pixel level maps to generate for the cell tracks')

        ###########################################################
        # Identify files to process
        ################################################################
        # Create labelcell output directory
        os.makedirs(celltracking_outpath, exist_ok=True)
        cloudidfiles, \
        cloudidfiles_basetime, \
        cloudidfiles_datestring, \
        cloudidfiles_timestring = subset_files_timerange(tracking_outpath,
                                                         cloudid_filebase,
                                                         start_basetime,
                                                         end_basetime)

        cellmap_input = zip(cloudidfiles, cloudidfiles_basetime, repeat(stats_outpath), repeat(trackstats_filebase), \
                            repeat(startdate), repeat(enddate), repeat(celltracking_outpath), repeat(celltracking_filebase))

        ## Call function
        if run_parallel == 0:
            # Call function
            for iunique in range(0, len(cloudidfiles)):
                # mapcell_radar(cellmap_input[iunique])
                mapcell_radar(cloudidfiles[iunique], cloudidfiles_basetime[iunique], stats_outpath, trackstats_filebase, \
                            startdate, enddate, celltracking_outpath, celltracking_filebase)
        elif run_parallel == 1:
            logger.info('Creating maps of tracked cells')
            pool = Pool(nprocesses)
            pool.starmap(mapcell_radar, cellmap_input)
            pool.close()
            pool.join()
        else:
            sys.ext('Valid parallelization flag not provided')
