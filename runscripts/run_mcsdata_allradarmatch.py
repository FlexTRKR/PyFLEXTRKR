import numpy as np
import os, fnmatch
import time, datetime, calendar
from pytz import utc
from multiprocessing import Pool

#import cProfile

# Purpose: Master script for tracking MCS identified using IR satellite and radar data in the central and eastern USA. 

# Comments:
# Features are tracked using 8 sets of code (idclouds, tracksingle, gettracks, trackstats, identifymcs, matchpf, robustmcs, mapmcs).
# idclouds, tracksingle, and mapcs are run in parallel using the multiprocessing module. All other sets of code are run serially.
# The code does not need to run through each step each time. The code can be run starting at any step, as long as those previous codes have been run and their output is availiable. 

# Author: Orginial IDL version written by Sally McFarlane and Zhe Feng (zhe.feng@pnnl.gov). Adapted to Python by Hannah Barnes (hannah.barnes@pnnl.gov)

print('Code Started')
print((time.ctime()))

##################################################################################################
# Set variables describing data, file structure, and tracking thresholds

# Specify which sets of code to run. (1 = run code, 0 = don't run code)
run_idclouds = 0        # Segment and identify cloud systems
run_tracksingle = 0     # Track single consecutive cloudid files
run_gettracks = 0       # Run trackig for all files
run_finalstats = 0      # Calculate final statistics
run_identifymcs = 0     # Isolate MCSs
run_matchpf = 0         # Identify precipitation features with MCSs
run_robustmcs = 0       # Filter potential mcs cases using nmq radar variables
run_labelmcs = 1        # Create maps of MCSs

# Set version ofcode
cloudidmethod = 'futyan4' # Option: futyan3 = identify cores and cold anvils and expand to get warm anvil, futyan4=identify core and expand for cold and warm anvils
keep_singlemergesplit = 1 # Options: 0=All short tracks are removed, 1=Only short tracks without mergers or splits are removed
show_alltracks = 0 # Options: 0=Maps of all tracks are not created, 1=Maps of all tracks are created (much slower!)
run_parallel = 1 # Options: 0-run serially, 1-run parallel (uses Pool from Multiprocessing)
process_halfhours = 0                       # 0 = No, 1 = Yes

# Specify version of code using
cloudid_version = 'v1.0'
track_version = 'v1.0'
tracknumber_version = 'v1.0'

# Specify default code version
curr_id_version = 'v1.0'
curr_track_version = 'v1.0'
curr_tracknumbers_version = 'v1.0'

# Specify days to run, (YYYYMMDD.hhmm)
startdate = '20110401.0000'
enddate = '20110401.2300'

# Specify cloud tracking parameters
geolimits = np.array([25, -110, 51, -70])  # 4-element array with plotting boundaries [lat_min, lon_min, lat_max, lon_max]
pixel_radius = 4.0                         # km
timegap = 3.1                              # hour
area_thresh = 64                           # km^2
miss_thresh = 0.2                          # Missing data threshold. If missing data in the domain from this file is greater than this value, this file is considered corrupt and is ignored. (0.1 = 10%)
cloudtb_core = 225.                        # K
cloudtb_cold = 241.                        # K
cloudtb_warm = 261.                        # K
cloudtb_cloud = 261.                       # K
othresh = 0.5                              # overlap percentage threshold
lengthrange = np.array([2,120])            # A vector [minlength,maxlength] to specify the lifetime range for the tracks
nmaxlinks = 24                             # Maximum number of clouds that any single cloud can be linked to
nmaxclouds = 1000                          # Maximum number of clouds allowed to be in one track
absolutetb_threshs = np.array([160, 330])   # k A vector [min, max] brightness temperature allowed. Brightness temperatures outside this range are ignored.
warmanvilexpansion = 0                     # If this is set to one, then the cold anvil is spread laterally until it exceeds the warm anvil threshold
mincoldcorepix = 4                         # Minimum number of pixels for the cold core, needed for futyan version 4 cloud identification code. Not used if use futyan version 3.
smoothwindowdimensions = 10                # Dimension of the boxcar filter used for futyan version 4. Not used in futyan version 3

# Specify MCS parameters
mcs_mergedir_areathresh = 6e4              # IR area threshold [km^2]
mcs_mergedir_durationthresh = 6            # IR minimum length of a mcs [hr]
mcs_mergedir_eccentricitythresh = 0.7      # IR eccentricity at time of maximum extent
mcs_mergedir_splitduration = 6             # IR tracks smaller or equal to this length will be included with the MCS splits from 
mcs_mergedir_mergeduration = 6             # IR tracks smaller or equal to this length will be included with the MCS merges into 

mcs_pf_majoraxisthresh = 100               # PF major axis MCS threshold [km]
mcs_pf_durationthresh = 5                  # PF minimum length of mcs [hr]
mcs_pf_aspectratiothresh = 4               # PF aspect ragio require to define a squall lines 
mcs_pf_lifecyclethresh = 8                 # Minimum MCS lifetime required to classify life stages
mcs_pf_lengththresh = 20                   # Minimum size requiredto classify life stages
mcs_pf_gap = 1                             # Allowable gap in data for subMCS characteristics [hr]

# Specify rain rate parameters
rr_min = 1.0                               # Rain rate threshold [mm/hr]
nmaxpf = 10                                # Maximum number of precipitation features that can be within a cloud feature
nmaxcore = 20                              # Maximum number of convective cores that can be within a cloud feature
pcp_thresh = 1                             # Pixels with hourly precipitation larger than this will be labeled with track number

# Specify filenames and locations
datavariablename = 'Tb'
irdatasource = 'gpmmergedir'
nmqdatasource = 'nmq'
datadescription = 'EUS'
databasename = 'eus_merg_'
label_filebase = 'cloudtrack_'
pfdata_filebase = 'csa4km_'
rainaccumulation_filebase = 'regrid_q2_'

root_path = '/global/homes/h/hcbarnes/Tracking/SatelliteRadar/'
clouddata_path = '/global/homes/h/hcbarnes/Tracking/SatelliteRadar/data/2011/'
#root_path = '/global/homes/h/hcbarnes/Tracking/SatelliteRadar/'
#root_path = '/scratch3/scratchdirs/feng045/usa/mergir/netcdf/'
#clouddata_path = '/global/project/projectdirs/m1867/zfeng/usa/mergedir/Netcdf/2011/'
#clouddata_path = '/scratch3/scratchdirs/feng045/usa/mergir/netcdf/2011/'
pfdata_path = '/global/project/projectdirs/m1867/zfeng/usa/nmq/csa4km/2011/'
rainaccumulation_path = '/global/project/projectdirs/m1867/zfeng/usa/nmq/q2/regrid/2011/'
scratchpath = './'

# Specify data structure
datatimeresolution = 1                     # hours
dimname = 'nclouds'
numbername = 'convcold_cloudnumber'
typename = 'cloudtype'
npxname = 'ncorecoldpix'
tdimname = 'time'
xdimname = 'lat'
ydimname = 'lon'

######################################################################
# Generate additional settings

# Isolate year
year = startdate[0:5]

# Concatonate thresholds into one variable
cloudtb_threshs = np.hstack((cloudtb_core, cloudtb_cold, cloudtb_warm, cloudtb_cloud))

# Specify additional file locations
tracking_outpath = root_path + 'tracking/'       # Data on individual features being tracked
stats_outpath = root_path + 'stats/'             # Data on track statistics
mcstracking_outpath = root_path + 'mcstracking/' + startdate + '_' + enddate + '/' # Pixel level data for MCSs

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
# Identify clouds / features in the data, if neccesary
if run_idclouds == 1:
    ######################################################################
    # Identify files to process
    print('Identifying raw data files to process.')
    print((time.ctime()))

    # Isolate all possible files
    allrawdatafiles = fnmatch.filter(os.listdir(clouddata_path), databasename+'*')

    # Loop through files, identifying files within the startdate - enddate interval
    nleadingchar = np.array(len(databasename)).astype(int)

    rawdatafiles = [None]*len(allrawdatafiles)
    filestep = 0
    for ifile in allrawdatafiles:
        TEMP_filetime = datetime.datetime(int(ifile[nleadingchar:nleadingchar+4]), int(ifile[nleadingchar+4:nleadingchar+6]), int(ifile[nleadingchar+6:nleadingchar+8]), int(ifile[nleadingchar+8:nleadingchar+10]), 0, 0, tzinfo=utc)
        TEMP_filebasetime = calendar.timegm(TEMP_filetime.timetuple())

        if TEMP_filebasetime >= start_basetime and TEMP_filebasetime <= end_basetime:
            rawdatafiles[filestep] = clouddata_path + ifile
            filestep = filestep + 1

    # Remove extra rows
    rawdatafiles = rawdatafiles[0:filestep]

    ##########################################################################
    # Process files
    # Load function
    from pyflextrkr.depreciated.idclouds import idclouds_gpmmergedir

    # Generate input lists
    list_irdatasource = [irdatasource]*(filestep)
    list_datadescription = [datadescription]*(filestep)
    list_datavariablename = [datavariablename]*(filestep)
    list_cloudidversion = [cloudid_version]*(filestep)
    list_trackingoutpath = [tracking_outpath]*(filestep)
    list_latlonfile = [latlon_file]*(filestep)
    list_latname = [xdimname]*(filestep)
    list_lonname = [ydimname]*(filestep)
    list_geolimits = np.ones(((filestep), 4))*geolimits
    list_startdate = [startdate]*(filestep)
    list_enddate = [enddate]*(filestep)
    list_pixelradius = np.ones(filestep)*pixel_radius
    list_areathresh = np.ones(filestep)*area_thresh
    list_cloudtbthreshs = np.ones((filestep,4))*cloudtb_threshs
    list_absolutetbthreshs = np.ones(((filestep), 2))*absolutetb_threshs
    list_missthresh = np.ones(filestep)*miss_thresh
    list_cloudidmethod = [cloudidmethod]*(filestep)
    list_warmanvilexpansion = np.ones(filestep)*warmanvilexpansion
    list_coldcorethresh = np.ones(filestep)*mincoldcorepix
    list_smoothsize = [smoothwindowdimensions]*(filestep)
    list_processhalfhour = [process_halfhours]*(filestep)

    idclouds_input = list(zip(rawdatafiles, list_irdatasource, list_datadescription, list_datavariablename, list_cloudidversion, list_trackingoutpath, list_latlonfile, list_latname, list_lonname, list_geolimits, list_startdate, list_enddate, list_pixelradius, list_areathresh, list_cloudtbthreshs, list_absolutetbthreshs, list_missthresh, list_cloudidmethod, list_coldcorethresh, list_smoothsize, list_warmanvilexpansion, list_processhalfhour))

    ## Call function
    if run_parallel == 0:
        # Serial version
        for ifile in range(0, filestep):
            idclouds_gpmmergedir(idclouds_input[ifile])
    elif run_parallel == 1:
        # Parallel version
        if __name__ == '__main__':
            print('Identifying clouds')
            print((time.ctime()))
            pool = Pool(10)
            pool.map(idclouds_gpmmergedir, idclouds_input)
            pool.close()
            pool.join()
    else:
        sys.exit('Valid parallelization flag not provided')

    cloudid_filebase = irdatasource + '_' + datadescription + '_cloudid' + cloudid_version + '_'

###################################################################
# Link clouds/ features in time adjacent files (single file tracking), if necessary

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
        TEMP_cloudidtime = datetime.datetime(int(icloudidfile[nleadingchar:nleadingchar+4]), int(icloudidfile[nleadingchar+4:nleadingchar+6]), int(icloudidfile[nleadingchar+6:nleadingchar+8]), int(icloudidfile[nleadingchar+9:nleadingchar+11]), int(icloudidfile[nleadingchar+11:nleadingchar+13]), 0, tzinfo=utc)
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
    # Load function
    from pyflextrkr.tracksingle import trackclouds_mergedir

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

    trackclouds_input = list(zip(cloudidfiles[0:-1], cloudidfiles[1::], cloudidfiles_datestring[0:-1], cloudidfiles_datestring[1::], cloudidfiles_timestring[0:-1], cloudidfiles_timestring[1::], cloudidfiles_basetime[0:-1], cloudidfiles_basetime[1::], list_trackingoutpath, list_trackversion, list_timegap, list_nmaxlinks, list_othresh, list_startdate, list_enddate))

    if run_parallel == 0:
        # Serial version
        for ifile in range(0, cloudidfilestep-1):
            trackclouds_mergedir(trackclouds_input[ifile])
    elif run_parallel == 1:
        # parallelize version
        if __name__ == '__main__':
            pool = Pool(10)
            pool.map(trackclouds_mergedir, trackclouds_input)
            pool.close()
            pool.join()
    else:
        sys.exit('Valid parallelization flag not provided.')

    singletrack_filebase = 'track' + track_version + '_'

###########################################################
# Track clouds / features through the entire dataset

# Determine if single file tracking code ran. If not, set the version name and filename using names specified in the constants section
if run_tracksingle == 0:
    print('Single file tracks already determined')
    singletrack_filebase = 'track' + curr_track_version + '_'

# Call function
if run_gettracks == 1:
    # Load function
    from pyflextrkr.gettracks import gettracknumbers

    # Call function
    print('Getting track numbers')
    print(time.ctime())
    gettracknumbers(irdatasource, datadescription, tracking_outpath, stats_outpath, startdate, enddate, timegap, nmaxclouds, cloudid_filebase, npxname, tracknumber_version, singletrack_filebase, keepsingletrack=keep_singlemergesplit, removestartendtracks=1)
    tracknumbers_filebase = 'tracknumbers' + tracknumber_version

############################################################
# Calculate final statistics

# Determine if the tracking portion of the code ran. If not, set teh version name and filename using those specified in the constants section
if run_gettracks == 0:
    print('Cloud tracks already determined')
    tracknumbers_filebase = 'tracknumbers' + curr_tracknumbers_version

# Call function
if run_finalstats == 1:
    # Load function
    from pyflextrkr.depreciated.trackstats import trackstats_sat

    # Call satellite version of function
    print('Calculating track statistics')
    print(time.ctime())
    trackstats_sat(irdatasource, datadescription, pixel_radius, geolimits, area_thresh, cloudtb_threshs, absolutetb_threshs, startdate, enddate, timegap, cloudid_filebase, tracking_outpath, stats_outpath, track_version, tracknumber_version, tracknumbers_filebase, lengthrange=lengthrange)
    trackstats_filebase = 'stats_tracknumbers' + tracknumber_version

##############################################################
# Identify MCS candidates

# Determine if final statistics portion ran. If not, set the version name and filename using those specified in the constants section
if run_finalstats == 0:
    print('Cloud tracks already done')
    trackstats_filebase = 'stats_tracknumbers' + curr_tracknumbers_version

if run_identifymcs == 1:
    print('Identifying MCSs')

    # Load function
    from pyflextrkr.identifymcs import identifymcs_mergedir_netcdf4

    # Call satellite version of function
    print((time.ctime()))
    identifymcs_mergedir_netcdf4(trackstats_filebase, stats_outpath, startdate, enddate, geolimits, datatimeresolution, mcs_mergedir_areathresh, mcs_mergedir_durationthresh, mcs_mergedir_eccentricitythresh, mcs_mergedir_splitduration, mcs_mergedir_mergeduration, nmaxlinks)
    mcsstats_filebase =  'mcs_tracks_'

#############################################################
# Identify preciptation features within MCSs

# Determine if identify mcs portion of code ran. If not set file name
if run_identifymcs == 0:
    print('MCSs already identified')
    mcsstats_filebase = 'mcs_tracks_'

if run_matchpf == 1:
    print('Identifying Precipitation Features in MCSs')

    # Load function
    from pyflextrkr.depreciated.matchpf import identifypf_mergedir_nmq

    # Call function
    print((time.ctime()))
    identifypf_mergedir_nmq(mcsstats_filebase, cloudid_filebase, pfdata_filebase, rainaccumulation_filebase, stats_outpath, tracking_outpath, pfdata_path, rainaccumulation_path, startdate, enddate, geolimits, nmaxpf, nmaxcore, nmaxclouds, rr_min, pixel_radius, irdatasource, nmqdatasource, datadescription, datatimeresolution, mcs_mergedir_areathresh, mcs_mergedir_durationthresh, mcs_mergedir_eccentricitythresh)
    pfstats_filebase = 'mcs_tracks_'  + nmqdatasource + '_' 

##############################################################
# Identify robust MCS using precipitation feature statistics

# Determine if identify precipitation feature portion of code ran. If not set file name
if run_matchpf == 0:
    print('MCSs already linked to precipitation data')
    pfstats_filebase = 'mcs_tracks_'  + nmqdatasource + '_' 

# Run code to identify robust MCS
if run_robustmcs == 1:
    print('Identifying robust MCSs using precipitation features')

    # Load function
    from pyflextrkr.robustmcs import filtermcs_mergedir_nmq

    # Call function
    print((time.ctime()))
    filtermcs_mergedir_nmq(stats_outpath, pfstats_filebase, startdate, enddate, datatimeresolution, geolimits, mcs_pf_majoraxisthresh, mcs_pf_durationthresh, mcs_pf_aspectratiothresh, mcs_pf_lifecyclethresh, mcs_pf_lengththresh, mcs_pf_gap)
    robustmcs_filebase = 'robust_mcs_tracks_nmq_'

############################################################
# Create pixel files with MCS tracks

# Determine if the mcs identification and statistic generation step ran. If not, set the filename using those specified in the constants section
if run_robustmcs == 0:
    print('Robust MCSs already determined')
    robustmcs_filebase =  'robust_mcs_tracks_nmq_'

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
            TEMP_cloudidtime = datetime.datetime(int(icloudidfile[nleadingchar:nleadingchar+4]), int(icloudidfile[nleadingchar+4:nleadingchar+6]), int(icloudidfile[nleadingchar+6:nleadingchar+8]), int(icloudidfile[nleadingchar+9:nleadingchar+11]), int(icloudidfile[nleadingchar+11:nleadingchar+13]), 0, tzinfo=utc)
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

    # Load function 
    from pyflextrkr.depreciated.mapmcs import mapmcs_pf

    # Generate input list
    list_robustmcsstat_filebase = [robustmcs_filebase]*(cloudidfilestep-1)
    list_trackstat_filebase = [trackstats_filebase]*(cloudidfilestep-1)
    list_pfdata_filebase = [pfdata_filebase]*(cloudidfilestep-1)
    list_rainaccumulation_filebase = [rainaccumulation_filebase]*(cloudidfilestep-1)
    list_mcstracking_path = [mcstracking_outpath]*(cloudidfilestep-1)
    list_stats_path = [stats_outpath]*(cloudidfilestep-1)
    list_pfdata_path = [pfdata_path]*(cloudidfilestep-1)
    list_rainaccumulation_path = [rainaccumulation_path]*(cloudidfilestep-1)
    list_cloudid_filebase = [cloudid_filebase]*(cloudidfilestep-1)
    list_pcp_thresh = np.ones(cloudidfilestep-1)*pcp_thresh
    list_nmaxpf = np.ones(cloudidfilestep-1)*nmaxpf
    list_absolutetb_threshs = np.ones((cloudidfilestep-1, 2))*absolutetb_threshs
    list_startdate = [startdate]*(cloudidfilestep-1)
    list_enddate = [enddate]*(cloudidfilestep-1)
    list_showalltracks = [show_alltracks]*(cloudidfilestep-1)

    robustmcsmap_input = list(zip(cloudidfiles, cloudidfiles_basetime, list_robustmcsstat_filebase, list_trackstat_filebase, list_pfdata_filebase, list_rainaccumulation_filebase, list_mcstracking_path, list_stats_path, list_pfdata_path, list_rainaccumulation_path, list_pcp_thresh, list_nmaxpf, list_absolutetb_threshs, list_startdate, list_enddate, list_showalltracks))

    if run_parallel == 0:
        # Call function
        for iunique in range(0, cloudidfilestep-1):
            mapmcs_pf(robustmcsmap_input[iunique])

        #cProfile.run('mapmcs_pf(robustmcsmap_input[200])')
    elif run_parallel == 1:
        if __name__ == '__main__':
            print('Creating maps of tracked MCSs')
            print((time.ctime()))
            pool = Pool(24)
            pool.map(mapmcs_pf, robustmcsmap_input)
            pool.close()
            pool.join()
    else:
        sys.ext('Valid parallelization flag not provided')

