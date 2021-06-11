import calendar
import datetime
import fnmatch
import os
import sys
import time
from itertools import repeat
from multiprocessing import Pool

import numpy as np
from pytz import utc

from pyflextrkr.gettracks import gettracknumbers
from pyflextrkr.idclouds_sat import idclouds_gpmmergir
from pyflextrkr.identifymcs import identifymcs_tb
from pyflextrkr.mapmcspf import mapmcs_tb_pf
from pyflextrkr.matchtbpf_parallel import match_tbpf_tracks
from pyflextrkr.robustmcspf import define_robust_mcs_pf
from pyflextrkr.tracksingle import trackclouds
from pyflextrkr.trackstats_parallel import trackstats_tb

# Purpose: Master script for tracking MCS using collocated IR brightness temperature (Tb) and GPM IMERG precipitation data.

# Comments:
# Features are tracked using 8 sets of code (idclouds, tracksingle, gettracks, trackstats, identifymcs, matchpf, robustmcs, mapmcs).
# The code does not need to run through each step each time. The code can be run starting at any step, as long as those previous codes have been run and their output is availiable. 

# Author: Orginial IDL version written by Sally McFarlane and Zhe Feng (zhe.feng@pnnl.gov). Adapted to Python by Hannah Barnes (hannah.barnes@pnnl.gov)

print('Code Started')
print((time.ctime()))
if __name__ == '__main__':
##################################################################################################
    # Set variables describing data, file structure, and tracking thresholds

    # Specify which sets of code to run. (1 = run code, 0 = don't run code)
    run_idclouds = 1        # Segment and identify cloud systems
    run_tracksingle = 1     # Track single consecutive cloudid files
    run_gettracks = 1       # Run trackig for all files
    run_finalstats = 1      # Calculate final statistics
    run_identifymcs = 1     # Isolate MCSs
    run_matchpf = 1         # Identify precipitation features with MCSs
    run_robustmcs = 1       # Filter potential mcs cases using nmq radar variables
    run_labelmcs = 1        # Create maps of MCSs

    # Set version ofcode
    cloudidmethod = 'futyan4'   # Option: futyan3 = identify cores and cold anvils and expand to get warm anvil, futyan4=identify core and expand for cold and warm anvils
    keep_singlemergesplit = 1   # Options: 0=All short tracks are removed, 1=Only short tracks without mergers or splits are removed
    show_alltracks = 0          # Options: 0=Maps of all tracks are not created, 1=Maps of all tracks are created (much slower!)
    run_parallel = 0            # Options: 0-run serially, 1-run parallel (uses Pool from Multiprocessing)
    nprocesses = 3             # Number of processors to use if run_parallel is set to 1
    idclouds_hourly = 1         # 0 = No, 1 = Yes
    idclouds_minute = 0         # 00 = 00min, 30 = 30min

    # Specify version of code using
    cloudid_version = 'v1.0'
    track_version = 'v1.0'
    tracknumber_version = 'v1.0'

    # Specify default code version
    curr_id_version = 'v1.0'
    curr_track_version = 'v1.0'
    curr_tracknumbers_version = 'v1.0'

    # Specify days to run, (YYYYMMDD.hhmm)
    startdate = '20150000.0000'
    enddate = '20160001.2300'

    # Specify cloud tracking parameters
    geolimits = np.array([-90, -360, 90, 360])  # 4-element array with plotting boundaries [lat_min, lon_min, lat_max, lon_max]
    pixel_radius = 10.0                         # km
    timegap = 3.1                              # hour
    area_thresh = 800                          # km^2
    miss_thresh = 0.2                          # Missing data threshold. If missing data in the domain from this file is greater than this value, this file is considered corrupt and is ignored. (0.1 = 10%)
    cloudtb_core = 225.                        # K
    cloudtb_cold = 241.                        # K
    cloudtb_warm = 261.                        # K
    cloudtb_cloud = 261.                       # K
    othresh = 0.5                              # overlap percentage threshold
    lengthrange = np.array([2,200])            # A vector [minlength,maxlength] to specify the lifetime range for the tracks
    nmaxlinks = 24                             # Maximum number of clouds that any single cloud can be linked to
    nmaxclouds = 1000                          # Maximum number of clouds allowed to be in one track
    absolutetb_threshs = np.array([160, 330])   # k A vector [min, max] brightness temperature allowed. Brightness temperatures outside this range are ignored.
    warmanvilexpansion = 0                     # If this is set to one, then the cold anvil is spread laterally until it exceeds the warm anvil threshold
    mincoldcorepix = 4                         # Minimum number of pixels for the cold core, needed for futyan version 4 cloud identification code. Not used if use futyan version 3.
    smoothwindowdimensions = 10                # Dimension of the boxcar filter used for futyan version 4. Not used in futyan version 3
    # medfiltsize = 5                            # Window size to perform medfilt2d to fill missing IR pixels, must be an odd number

    # Specify MCS parameters
    mcs_mergedir_areathresh = 4e4              # IR area threshold [km^2]
    mcs_mergedir_durationthresh = 4            # IR minimum length of a mcs [hr]
    mcs_mergedir_eccentricitythresh = 0.7      # IR eccentricity at time of maximum extent
    mcs_mergedir_splitduration = 12             # IR tracks smaller or equal to this length will be included with the MCS splits from
    mcs_mergedir_mergeduration = 12             # IR tracks smaller or equal to this length will be included with the MCS merges into

    mcs_pf_majoraxisthresh = 100               # MCS PF major axis length lower limit [km]
    max_pf_majoraxis_thresh = 1800             # MCS PF major axis length upper limit [km]
    mcs_pf_durationthresh = 4                  # PF minimum length of mcs [hr]
    mcs_pf_aspectratiothresh = 4               # PF aspect ragio require to define a squall lines
    mcs_pf_lifecyclethresh = 4                 # Minimum MCS lifetime required to classify life stages
    mcs_pf_lengththresh = 20                   # Minimum size required to classify life stages [km]
    mcs_pf_gap = 1                             # Allowable gap in data for subMCS characteristics [hr]

    # Specify rain rate parameters
    pf_rr_thres = 2.0                          # Rain rate threshold [mm/hr]
    nmaxpf = 3                                 # Maximum number of precipitation features that can be within a cloud feature
    nmaxcore = 20                              # Maximum number of convective cores that can be within a cloud feature
    pcp_thresh = 1.0                           # Pixels with hourly precipitation larger than this will be labeled with track number
    heavy_rainrate_thresh = 10.0               # Heavy rain rate threshold [mm/hr]

    # Specific parameters to link cloud objects using PF
    linkpf = 1                                 # Set to 1 to turn on linkpf option; default: 0
    pf_smooth_window = 5                       # Smoothing window for identifying PF
    pf_dbz_thresh = 3                          # [dBZ] for reflectivity, or [mm/h] for rainrate
    pf_link_area_thresh = 648.0                 # [km^2]

    # MCS PF parameter coefficients [intercept, slope]
    # These parameters are derived with pf_rr_thres = 2 mm/h
    # coefs_area = [1962.11, -14.598]      # 1%
    # coefs_area = [1962.11, 0]           # 1% [changed slope to 0: independent of lifetime]
    # coefs_area = [2119.02, 61.143]      # 3%
    coefs_area = [2874.05, 89.825]      # 5% [recommended]
    # coefs_area = [4160.82, 93.077]      # 7%
    # coefs_area = [4988.15, 138.172]      # 10%

    # coefs_rr = [2.72873, 0.0008317]      # 1%
    # coefs_rr = [2.81982, 0.0135463]      # 3%
    coefs_rr = [3.01657, 0.0144461]      # 5% [recommended]
    # coefs_rr = [3.14895, 0.0150174]      # 7%
    # coefs_rr = [3.34859, 0.0172043]      # 10%

    # coefs_skew = [0.036384, 0.0022199]      # 1%
    # coefs_skew = [0.072809, 0.0104444]      # 3%
    coefs_skew = [0.194462, 0.0100072]      # 5% [recommended]
    # coefs_skew = [0.256639, 0.0106527]      # 7%
    # coefs_skew = [0.376142, 0.0095545]      # 10%

    # coefs_heavyratio = [0.750260, 0.4133300]  # 5%
    coefs_heavyratio = [3.419024, 0.4387090]  # 10% [recommended]
    # coefs_heavyratio = [4.753215, 0.4886454]  # 15%
    # coefs_heavyratio = [4.592209, 0.6107371]  # 20%
    # coefs_heavyratio = [8.389616, 0.5079337]  # 25%

    # Specify filenames and locations
    datavariablename = 'Tb'
    irdatasource = 'gpmirimerg'
    pfdatasource = 'imerg'
    datadescription = 'EUS'
    databasename = 'merg_'
    label_filebase = 'cloudtrack_'
    pfdata_filebase = 'merg_'
    rainaccumulation_filebase = 'merg_'
    root_path = os.environ['FLEXTRKR_BASE_DATA_PATH']

    print(f'ROOT PATH IS {root_path}')

    clouddata_path = root_path + 'input/'
    pfdata_path = root_path + 'input/'
    print(f'Clouddatapath: {clouddata_path}, pfdata_path: {pfdata_path}')

    rainaccumulation_path = pfdata_path
    landmask_file = root_path+'map_data/IMERG_landmask_eus.nc'

    # Specify data structure
    datatimeresolution = 1     # hours
    dimname = 'nclouds'
    numbername = 'convcold_cloudnumber'
    typename = 'cloudtype'
    npxname = 'ncorecoldpix'
    tdimname = 'time'
    xdimname = 'lat'
    ydimname = 'lon'
    pfvarname = 'precipitationCal'
    pcpvarname = 'precipitation'
    landvarname = 'landseamask'
    landfrac_thresh = 90    # define threshold for land

    ######################################################################
    # Generate additional settings

    # Isolate year
    year = startdate[0:4]  #change 0:5 to 0:4

    # Concatonate thresholds into one variable
    cloudtb_threshs = np.hstack((cloudtb_core, cloudtb_cold, cloudtb_warm, cloudtb_cloud))

    # Specify additional file locations
    tracking_outpath = root_path + 'tracking/'       # Data on individual features being tracked
    stats_outpath = root_path + 'stats/'             # Data on track statistics
    mcstracking_outpath = root_path + 'mcstracking/' + startdate + '_' + enddate + '/' # Pixel level data for MCSs
    print(f'TRACKING PATH IS {tracking_outpath}, {stats_outpath}, {mcstracking_outpath}')
    ####################################################################
    # Execute tracking scripts

    # Create output directories
    if not os.path.exists(tracking_outpath):
        os.makedirs(tracking_outpath)

    if not os.path.exists(stats_outpath):
        os.makedirs(stats_outpath)

    ########################################################################
    # Calculate basetime of start and end date
    TEMP_starttime = datetime.datetime(int(startdate[0:4]), int(startdate[4:6]), int(startdate[6:8]), int(startdate[9:11]), int(startdate[11:13]), 0, tzinfo=utc)
    start_basetime = calendar.timegm(TEMP_starttime.timetuple())

    TEMP_endtime = datetime.datetime(int(enddate[0:4]), int(enddate[4:6]), int(enddate[6:8]), int(enddate[9:11]), int(enddate[11:13]), 0, tzinfo=utc)
    end_basetime = calendar.timegm(TEMP_endtime.timetuple())

    ##########################################################################
    # Step 1. Identify clouds
    if run_idclouds == 1:
        ######################################################################
        # Identify files to process
        print('Identifying raw data files to process.')
        print((time.ctime()))

        # Isolate all possible files
        allrawdatafiles = fnmatch.filter(os.listdir(clouddata_path), databasename+'*')

        # Sort the files by date and time. Filename exmaple: merg_2017033104_4km-pixel.nc
        def fdatetime(x):
            return(x[5:15])
        allrawdatafiles = sorted(allrawdatafiles, key = fdatetime)

        # Loop through files, identifying files within the startdate - enddate interval
        nleadingchar = np.array(len(databasename)).astype(int)
        rawdatafiles = [None]*len(allrawdatafiles)

        filestep = 0
        for ifile in allrawdatafiles:
            TEMP_filetime = datetime.datetime(int(ifile[nleadingchar:nleadingchar+4]),
                                                int(ifile[nleadingchar+4:nleadingchar+6]),
                                                int(ifile[nleadingchar+6:nleadingchar+8]),
                                                int(ifile[nleadingchar+8:nleadingchar+10]), 0, 0, tzinfo=utc)
            TEMP_filebasetime = calendar.timegm(TEMP_filetime.timetuple())

            if start_basetime <= TEMP_filebasetime <= end_basetime:
                rawdatafiles[filestep] = clouddata_path + ifile
                filestep = filestep + 1

        # Remove extra rows
        rawdatafiles = rawdatafiles[0:filestep]

        ##########################################################################
        # Process files

        # Generate input lists
        idclouds_input = zip(rawdatafiles, repeat(irdatasource), repeat(datadescription), repeat(datavariablename), repeat(cloudid_version), \
                            repeat(tracking_outpath), repeat(landmask_file), repeat(geolimits), repeat(startdate), repeat(enddate), \
                            repeat(pixel_radius), repeat(area_thresh), repeat(cloudtb_threshs), repeat(absolutetb_threshs), repeat(miss_thresh), \
                            repeat(cloudidmethod), repeat(mincoldcorepix), repeat(smoothwindowdimensions), repeat(warmanvilexpansion), \
                            repeat(idclouds_hourly), repeat(idclouds_minute), \
                            repeat(linkpf), repeat(pf_smooth_window), repeat(pf_dbz_thresh), repeat(pf_link_area_thresh), repeat(pfvarname), \
                            )
        ## Call function
        if run_parallel == 0:
            # Serial version
            for ifile in range(0, filestep):
                # idclouds_gpmmergir(idclouds_input[ifile])
                idclouds_gpmmergir(rawdatafiles[ifile], irdatasource, datadescription, datavariablename, cloudid_version, \
                                    tracking_outpath, landmask_file, geolimits, startdate, enddate, \
                                    pixel_radius, area_thresh, cloudtb_threshs, absolutetb_threshs, miss_thresh, \
                                    cloudidmethod, mincoldcorepix, smoothwindowdimensions, warmanvilexpansion, \
                                    idclouds_hourly, idclouds_minute, \
                                    linkpf, pf_smooth_window, pf_dbz_thresh, pf_link_area_thresh, pfvarname, \
                                    )
        elif run_parallel == 1:
            # Parallel version
            print('Identifying clouds')
            print((time.ctime()))
            pool = Pool(nprocesses)
            # pool.map(idclouds_gpmmergir, idclouds_input)
            pool.starmap(idclouds_gpmmergir, idclouds_input)
            pool.close()
            pool.join()
        else:
            sys.exit('Valid parallelization flag not provided')

        cloudid_filebase = irdatasource + '_' + datadescription + '_cloudid' + cloudid_version + '_'
    ###################################################################
    # Step 2. Link clouds in time adjacent files (single file tracking)

    # Determine if identification portion of the code run. If not, set the version name and filename using names specified in the constants section
    if run_idclouds == 0:
        print('Cloud already identified')
        cloudid_filebase =  irdatasource + '_' + datadescription + '_cloudid' + curr_id_version + '_'

    # Call function
    if run_tracksingle == 1:
        ################################################################
        # Identify files to process
        print('Identifying cloudid files to process')
        print((time.ctime()))

        # Isolate all possible files
        allcloudidfiles = fnmatch.filter(os.listdir(tracking_outpath), cloudid_filebase +'*')

        # Put in temporal order
        allcloudidfiles = sorted(allcloudidfiles)

        # Loop through files, identifying files within the startdate - enddate interval
        nleadingchar = np.array(len(cloudid_filebase)).astype(int)

        cloudidfiles = [None]*len(allcloudidfiles)
        cloudidfiles_timestring = [None]*len(allcloudidfiles)
        cloudidfiles_datestring = [None]*len(allcloudidfiles)
        cloudidfiles_basetime = [None]*len(allcloudidfiles)
        cloudidfilestep = 0
        for icloudidfile in allcloudidfiles:
            TEMP_cloudidtime = datetime.datetime(int(icloudidfile[nleadingchar:nleadingchar+4]), \
                                                int(icloudidfile[nleadingchar+4:nleadingchar+6]), \
                                                int(icloudidfile[nleadingchar+6:nleadingchar+8]), \
                                                int(icloudidfile[nleadingchar+9:nleadingchar+11]), \
                                                int(icloudidfile[nleadingchar+11:nleadingchar+13]), 0, tzinfo=utc)
            TEMP_cloudidbasetime = calendar.timegm(TEMP_cloudidtime.timetuple())

            if TEMP_cloudidbasetime >= start_basetime and TEMP_cloudidbasetime <= end_basetime:
                cloudidfiles[cloudidfilestep] = tracking_outpath + icloudidfile
                cloudidfiles_timestring[cloudidfilestep] = icloudidfile[nleadingchar+9:nleadingchar+11] + icloudidfile[nleadingchar+11:nleadingchar+13]
                cloudidfiles_datestring[cloudidfilestep] = icloudidfile[nleadingchar:nleadingchar+4] + icloudidfile[nleadingchar+4:nleadingchar+6] + icloudidfile[nleadingchar+6:nleadingchar+8]
                cloudidfiles_basetime[cloudidfilestep] = np.copy(TEMP_cloudidbasetime)
                cloudidfilestep = cloudidfilestep + 1

        # Remove extra rows
        cloudidfiles = cloudidfiles[0:cloudidfilestep]
        cloudidfiles_timestring = cloudidfiles_timestring[0:cloudidfilestep]
        cloudidfiles_datestring = cloudidfiles_datestring[0:cloudidfilestep]
        cloudidfiles_basetime = cloudidfiles_basetime[:cloudidfilestep]

        ################################################################
        # Process files

        # Generate input lists
        list_trackingoutpath = [tracking_outpath]*(cloudidfilestep-1)
        list_trackversion = [track_version]*(cloudidfilestep-1)
        list_timegap = np.ones(cloudidfilestep-1)*timegap
        list_nmaxlinks = np.ones(cloudidfilestep-1)*nmaxlinks
        list_othresh = np.ones(cloudidfilestep-1)*othresh
        list_startdate = [startdate]*(cloudidfilestep-1)
        list_enddate = [enddate]*(cloudidfilestep-1)

        # Call function
        print('Tracking clouds between single files')
        print((time.ctime()))

        trackclouds_input = list(zip(cloudidfiles[0:-1], cloudidfiles[1::], cloudidfiles_datestring[0:-1], cloudidfiles_datestring[1::], \
                                cloudidfiles_timestring[0:-1], cloudidfiles_timestring[1::], cloudidfiles_basetime[0:-1], cloudidfiles_basetime[1::], \
                                list_trackingoutpath, list_trackversion, list_timegap, list_nmaxlinks, list_othresh, list_startdate, list_enddate))

        if run_parallel == 0:
            # Serial version
            for ifile in range(0, cloudidfilestep-1):
                trackclouds(trackclouds_input[ifile])
        elif run_parallel == 1:
            # parallelize version
            if __name__ == '__main__':
                pool = Pool(nprocesses)
                pool.map(trackclouds, trackclouds_input)
                pool.close()
                pool.join()
        else:
            sys.exit('Valid parallelization flag not provided.')

        singletrack_filebase = 'track' + track_version + '_'

    ###########################################################
    # Step 3. Track clouds through the entire dataset

    # Determine if single file tracking code ran. If not, set the version name and filename using names specified in the constants section
    if run_tracksingle == 0:
        print('Single file tracks already determined')
        singletrack_filebase = 'track' + curr_track_version + '_'

    # Call function
    if run_gettracks == 1:
        # Call function
        print('Getting track numbers')
        print(time.ctime())
        gettracknumbers(irdatasource, datadescription, tracking_outpath, stats_outpath, startdate, enddate, \
                        timegap, nmaxclouds, cloudid_filebase, npxname, tracknumber_version, singletrack_filebase, \
                        keepsingletrack=keep_singlemergesplit, removestartendtracks=1)
        tracknumbers_filebase = 'tracknumbers' + tracknumber_version

    ############################################################
    # Step 4. Calculate cloud statistics

    # Determine if the tracking portion of the code ran. If not, set teh version name and filename using those specified in the constants section
    if run_gettracks == 0:
        print('Cloud tracks already determined')
        tracknumbers_filebase = 'tracknumbers' + curr_tracknumbers_version

    # Call function
    if run_finalstats == 1:
        # Call satellite version of function
        print('Calculating track statistics')
        print(time.ctime())
        trackstats_tb(irdatasource, datadescription, pixel_radius, geolimits, area_thresh, \
                        cloudtb_threshs, absolutetb_threshs, startdate, enddate, timegap, cloudid_filebase, \
                        tracking_outpath, stats_outpath, track_version, tracknumber_version, tracknumbers_filebase, \
                        nprocesses, lengthrange=lengthrange)
        trackstats_filebase = 'stats_tracknumbers' + tracknumber_version

    ##############################################################
    # Step 5. Identify MCS candidates based on IR Tb data

    # Determine if final statistics portion ran. If not, set the version name and filename using those specified in the constants section
    if run_finalstats == 0:
        print('Track stats already done')
        trackstats_filebase = 'stats_tracknumbers' + curr_tracknumbers_version

    if run_identifymcs == 1:
        print('Identifying MCSs')
        # Call satellite version of function
        print((time.ctime()))
        identifymcs_tb(trackstats_filebase, stats_outpath, startdate, enddate, geolimits, datatimeresolution, \
                        mcs_mergedir_areathresh, mcs_mergedir_durationthresh, mcs_mergedir_eccentricitythresh, \
                        mcs_mergedir_splitduration, mcs_mergedir_mergeduration, nmaxlinks, timegap=1)
        mcsstats_filebase =  'mcs_tracks_'

    #############################################################
    # Step 6. Match preciptation features with MCS cloud shields

    # Determine if identify mcs portion of code ran. If not set file name
    if run_identifymcs == 0:
        print('MCSs already identified')
        mcsstats_filebase = 'mcs_tracks_'

    if run_matchpf == 1:
        print('Identifying Precipitation Features in MCSs')
        # Call function
        print((time.ctime()))
        match_tbpf_tracks(mcsstats_filebase, cloudid_filebase, \
                            stats_outpath, tracking_outpath, startdate, enddate, \
                            geolimits, nmaxpf, nmaxcore, nmaxclouds, pf_rr_thres, pixel_radius, \
                            irdatasource, pfdatasource, datadescription, datatimeresolution, \
                            mcs_mergedir_areathresh, mcs_mergedir_durationthresh, mcs_mergedir_eccentricitythresh, \
                            pf_link_area_thresh, heavy_rainrate_thresh, nprocesses, \
                            landmask_file=landmask_file, landvarname=landvarname, landfrac_thresh=landfrac_thresh)
        pfstats_filebase = 'mcs_tracks_pf_'

    ##############################################################
    # Step 7. Identify robust MCS using precipitation feature statistics

    # Determine if identify precipitation feature portion of code ran. If not set file name
    if run_matchpf == 0:
        print('MCSs already linked to precipitation data')
        pfstats_filebase = 'mcs_tracks_pf_'

    # Run code to identify robust MCS
    if run_robustmcs == 1:
        print('Identifying robust MCSs using precipitation features')
        # Call function
        print((time.ctime()))
        define_robust_mcs_pf(stats_outpath, pfstats_filebase, startdate, enddate, datatimeresolution, \
                                geolimits, mcs_pf_majoraxisthresh, mcs_pf_durationthresh, mcs_pf_aspectratiothresh, \
                                mcs_pf_lifecyclethresh, mcs_pf_lengththresh, mcs_pf_gap, \
                                coefs_area, coefs_rr, coefs_skew, coefs_heavyratio, \
                                max_pf_majoraxis_thresh=max_pf_majoraxis_thresh)
        robustmcs_filebase = 'robust_mcs_tracks_'

    ############################################################
    # Step 8. Create pixel files with MCS tracks

    # Determine if the mcs identification and statistic generation step ran. If not, set the filename using those specified in the constants section
    if run_robustmcs == 0: #TODO: This seems a bit weird.
        print('Robust MCSs already determined')
        robustmcs_filebase =  'robust_mcs_tracks_'

    if run_labelmcs == 1:
        print('Identifying which pixel level maps to generate for the MCS tracks')

        ###########################################################
        # Identify files to process
        if run_tracksingle == 0:
            ################################################################
            # Identify files to process
            print('Identifying cloudid files to process')

            # Isolate all possible files
            allcloudidfiles = fnmatch.filter(os.listdir(tracking_outpath), cloudid_filebase +'*')

            # Put in temporal order
            allcloudidfiles = sorted(allcloudidfiles)

            # Loop through files, identifying files within the startdate - enddate interval
            nleadingchar = np.array(len(cloudid_filebase)).astype(int)

            cloudidfiles = [None]*len(allcloudidfiles)
            cloudidfiles_basetime = [None]*len(allcloudidfiles)
            cloudidfilestep = 0
            for icloudidfile in allcloudidfiles:
                TEMP_cloudidtime = datetime.datetime(int(icloudidfile[nleadingchar:nleadingchar+4]), \
                                                    int(icloudidfile[nleadingchar+4:nleadingchar+6]), \
                                                    int(icloudidfile[nleadingchar+6:nleadingchar+8]), \
                                                    int(icloudidfile[nleadingchar+9:nleadingchar+11]), \
                                                    int(icloudidfile[nleadingchar+11:nleadingchar+13]), 0, tzinfo=utc)
                TEMP_cloudidbasetime = calendar.timegm(TEMP_cloudidtime.timetuple())

                if TEMP_cloudidbasetime >= start_basetime and TEMP_cloudidbasetime <= end_basetime:
                    cloudidfiles[cloudidfilestep] = tracking_outpath + icloudidfile
                    cloudidfiles_basetime[cloudidfilestep] = np.copy(TEMP_cloudidbasetime)
                    cloudidfilestep = cloudidfilestep + 1

            # Remove extra rows
            cloudidfiles = cloudidfiles[0:cloudidfilestep]
            cloudidfiles_basetime = cloudidfiles_basetime[:cloudidfilestep]

        #############################################################
        # Process files

        # Generate input list
        robustmcsmap_input = zip(cloudidfiles, cloudidfiles_basetime, repeat(robustmcs_filebase), repeat(trackstats_filebase), repeat(rainaccumulation_filebase), \
                            repeat(mcstracking_outpath), repeat(stats_outpath), repeat(pfdata_path), repeat(pcp_thresh), repeat(nmaxpf), repeat(absolutetb_threshs), \
                            repeat(startdate), repeat(enddate), repeat(show_alltracks))

        if run_parallel == 0:
            # Call function
            for iunique in range(0, cloudidfilestep-1):
                # mapmcs_tb_pf(robustmcsmap_input[iunique])
                mapmcs_tb_pf(cloudidfiles[iunique], cloudidfiles_basetime[iunique], robustmcs_filebase, trackstats_filebase, rainaccumulation_filebase, \
                            mcstracking_outpath, stats_outpath, pfdata_path, pcp_thresh, nmaxpf, absolutetb_threshs, \
                            startdate, enddate, show_alltracks)

            #cProfile.run('mapmcs_pf(robustmcsmap_input[200])')
        elif run_parallel == 1:
            print('Creating maps of tracked MCSs')
            print((time.ctime()))
            pool = Pool(nprocesses)
            pool.starmap(mapmcs_tb_pf, robustmcsmap_input)
            pool.close()
            pool.join()
        else:
            sys.ext('Valid parallelization flag not provided')
