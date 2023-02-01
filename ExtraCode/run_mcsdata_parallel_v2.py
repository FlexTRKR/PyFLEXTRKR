import numpy as np
import os, fnmatch
import datetime, calendar
from pytz import utc
from ipyparallel import Client
from netCDF4 import Dataset

# Name: Run_TestData.py

# Purpose: Master script for trackig synthetic IR satellite data

# Comments:
# Features are tracked using 5 sets of code (idclouds, trackclouds_singlefile, get_tracknumbers, calc_sat_trackstats, label_celltrack).
# This scrithon/ExtraCode/pt controls which pieces of code are run.
# Eventually, idclouds and trackclouds_singlefile will be able to run in parallel
# Iplt.annotate('Top = ' + str(np.round(SeparatedMeanReflectivityMeanVelocity_Slope[5], 2)) + ' * Bottom + ' + str(np.round(SeparatedMeanReflectivityMeanVelocity_Intercept[5], 2)), xy=(0.6, 0.3), xycoords='axes fraction', fontsize=12, color='green')f trackclouds_singlefile is run in of tracksistringtochar(np.array(cloudidfiles[nf]))ngle between 12/20/2009 - 12/31/2009, make two copies of this script, and set startdate - enddate (ex: 20091220 - 20091225, 20091225 - 20091231).
# This is because the first time will not have a tracksingle file produced, overlapping the date makes sure every cloudid file is used.
# The idclouds and trackClouds_singlefile only need to be run once and can be run on portions of the data a time.
# However, get_tracknumbers, calc_set_tracks, and label_celltrack must be run for the entire dataset.

# Author: Orginial IDL version written by Zhe Feng (zhe.feng@pnnl.gov). Adapted to Python by Hannah Barnes (hannah.barnes@pnnl.gov)

##################################################################################################
# Set variables describing data, file structure, and tracking thresholds

# Specify which sets of code to run. (1 = run code, 0 = don't run code)
run_idclouds = 0        # Segment and identify cloud systems
run_tracksingle = 0     # Track single consecutive cloudid files
run_gettracks = 0       # Run tracking for all files
run_finalstats = 0      # Calculate final statistics
run_identifymcs = 1     # Isolate MCSs 
run_labelmcs = 1        # Create maps of MCSs

# Specify version of code using
cloudid_version = 'v1.0'
track_version = 'v1.0'
tracknumber_version = 'v1.0'

# Specify default code version
curr_id_version = 'v1.0'
curr_track_version = 'v1.0'
curr_tracknumbers_version = 'v1.0'

# Specify days to run
startdate = '20110517'
enddate = '20110527'

# Specify tracking parameters
geolimits = np.array([25,-110,51,-70]) # 4-element array with plotting boundaries [lat_min, lon_min, lat_max, lon_max]
pixel_radius = 4.0      # km
timegap = 1.6           # hour
area_thresh = 64.       # km^2
miss_thresh = 0.2       # Missing data threshold. If missing data in the domain from this file is greater than this value, this file is considered corrupt and is ignored. (0.1 = 10%)
cloudtb_core = 225.          # K
cloudtb_cold = 241.          # K
cloudtb_warm = 261.          # K
cloudtb_cloud = 261.         # K
othresh = 0.5                     # overlap percentage threshold
lengthrange = np.array([2,120])    # A vector [minlength,maxlength] to specify the lifetime range for the tracks
nmaxlinks = 10                    # Maximum number of clouds that any single cloud can be linked to
nmaxclouds = 3000                 # Maximum number of clouds allowed to be in one track
absolutetb_threshs = np.array([160,330])        # k A vector [min, max] brightness temperature allowed. Brightness temperatures outside this range are ignored.
warmanvilexpansion = 0            # If this is set to one, then the cold anvil is spread laterally until it exceeds the warm anvil threshold

# Specify MCS parameters
mcs_areathresh = 6e4              # area threshold [km^2]
mcs_durationthresh = 6            # Minimum length of a mcs [hr]
mcs_eccentricitythresh = 0.7      # eccentricity at time of maximum extent
mcs_splitduration = 6            # Tracks smaller or equal to this length will be included with the MCS is it relinks with the MCS
mcs_mergeduration = 6            # Tracks smaller or equal to this length will be included with the MCS is it relinks with the MCS

# Specify filenames and locations
datavariablename = 'IRBT'
datasource = 'mergedir'
datadescription = 'EUS'
databasename = 'EUS_IR_Subset_'
label_filebase = 'cloudtrack_'

root_path = '/global/homes/h/hcbarnes/Tracking/Satellite/'
data_path = '/global/project/projectdirs/m1867/zfeng/usa/mergedir/Netcdf/2011/'
scratchpath = './'
latlon_file = '/global/project/projectdirs/m1867/zfeng/usa/mergedir/Geolocation/EUS_Geolocation_Data.nc'

# Specify data structure
datatimeresolution = 0.5 # hours
dimname = 'nclouds'
numbername = 'convcold_cloudnumber'
typename = 'cloudtype'
npxname = 'ncorecoldpix'
tdimname = 'time'
xdimname = 'Lat_Grid'
ydimname = 'Lon_Grid'

######################################################################
# Generate additional settings

# Isolate year
year = startdate[0:5]

# Concatonate thresholds into one variable
cloudtb_threshs = np.hstack((cloudtb_core, cloudtb_cold, cloudtb_warm, cloudtb_cloud))

# Specify additional file locations
#datapath = root_path                            # Location of raw data
tracking_outpath = root_path + 'tracking/'         # Data on individual features being tracked
stats_outpath = root_path + 'stats/'      # Data on track statistics
mcstracking_outpath = root_path + 'mcstracking/' # Pixel level data for MCSs

####################################################################
# Create client, for parallelization
rc = Client()
dview = rc[:]

######################################################################
# Execute tracking scripts

# Create output directories
if not os.path.exists(tracking_outpath):
    os.makedirs(tracking_outpath)

if not os.path.exists(stats_outpath):
    os.makedirs(stats_outpath)

########################################################################
# Calculate basetime of start and end date
TEMP_starttime = datetime.datetime(int(startdate[0:4]), int(startdate[4:6]), int(startdate[6:8]), 0, 0, 0, tzinfo=utc)
start_basetime = calendar.timegm(TEMP_starttime.timetuple())

TEMP_endtime = datetime.datetime(int(enddate[0:4]), int(enddate[4:6]), int(enddate[6:8]), 23, 0, 0, tzinfo=utc)
end_basetime = calendar.timegm(TEMP_endtime.timetuple())

##########################################################################
# Identify clouds / features in the data, if neccesary
if run_idclouds == 1:
    ######################################################################
    # Identify files to process
    print('Identifying raw data files to process.')

    # Isolate all possible files
    allrawdatafiles = fnmatch.filter(os.listdir(data_path), databasename+'*')

    # Loop through files, identifying files within the startdate - enddate interval
    nleadingchar = np.array(len(databasename)).astype(int)

    rawdatafiles = [None]*len(allrawdatafiles)
    files_timestring = [None]*len(allrawdatafiles) 
    files_datestring = [None]*len(allrawdatafiles)
    files_basetime = np.ones(len(allrawdatafiles), dtype=int)*-9999
    filestep = 0
    for ifile in allrawdatafiles:
        TEMP_filetime = datetime.datetime(int(ifile[nleadingchar:nleadingchar+4]), int(ifile[nleadingchar+4:nleadingchar+6]), int(ifile[nleadingchar+6:nleadingchar+8]), int(ifile[nleadingchar+9:nleadingchar+11]), int(ifile[nleadingchar+11:nleadingchar+13]), 0, tzinfo=utc)
        TEMP_filebasetime = calendar.timegm(TEMP_filetime.timetuple())

        if TEMP_filebasetime >= start_basetime and TEMP_filebasetime <= end_basetime:
            rawdatafiles[filestep] = data_path + ifile
            files_timestring[filestep] = ifile[nleadingchar+9:nleadingchar+11] + ifile[nleadingchar+11:nleadingchar+13]
            files_datestring[filestep] = ifile[nleadingchar:nleadingchar+4] + ifile[nleadingchar+4:nleadingchar+6] + ifile[nleadingchar+6:nleadingchar+8]
            files_basetime[filestep] = np.copy(TEMP_filebasetime)
            filestep = filestep + 1

    # Remove extra rows
    rawdatafiles = rawdatafiles[0:filestep]
    files_timestring = files_timestring[0:filestep]
    files_datestring = files_datestring[0:filestep]
    files_basetime = files_basetime[:filestep]

    ##########################################################################
    # Process files
    # Load function
    #from idclouds import idclouds_mergedir

    # Generate input lists
    list_datasource = [datasource]*(filestep)
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
    list_warmanvilexpansion = np.ones(filestep)*warmanvilexpansion

    idclouds_input = zip(rawdatafiles, files_datestring, files_timestring, files_basetime, list_datasource, list_datadescription, list_datavariablename, list_cloudidversion, list_trackingoutpath, list_latlonfile, list_latname, list_lonname, list_geolimits, list_startdate, list_enddate, list_pixelradius, list_areathresh, list_cloudtbthreshs, list_absolutetbthreshs, list_missthresh, list_warmanvilexpansion)

    # Call function
    print('Identifying clouds')

    from pyflextrkr.depreciated.idclouds import idclouds_mergedir

    dview.map_sync(idclouds_mergedir, idclouds_input)

    cloudid_filebase = datasource + '_' + datadescription + '_cloudid' + cloudid_version + '_'

###################################################################
# Link clouds/ features in time adjacent files (single file tracking), if necessary

# Determine if identification portion of the code run. If not, set the version name and filename using names specified in the constants section
if run_idclouds == 0:
    cloudid_filebase =  datasource + '_' + datadescription + '_cloudid' + curr_id_version + '_'

# Call function
if run_tracksingle == 1:
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
    cloudidfiles_timestring = [None]*len(allcloudidfiles)
    cloudidfiles_datestring = [None]*len(allcloudidfiles)
    cloudidfiles_basetime = np.ones(len(allcloudidfiles), dtype=float)*-9999
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

    trackclouds_input = zip(cloudidfiles[0:-1], cloudidfiles[1::], cloudidfiles_datestring[0:-1], cloudidfiles_datestring[1::], cloudidfiles_timestring[0:-1], cloudidfiles_timestring[1::], cloudidfiles_basetime[0:-1], cloudidfiles_basetime[1::], list_trackingoutpath, list_trackversion, list_timegap, list_nmaxlinks, list_othresh, list_startdate, list_enddate)

    # Parallelize version
    dview.map_sync(trackclouds_mergedir, trackclouds_input)

    singletrack_filebase = 'track' + track_version + '_'

###########################################################
# Track clouds / features through the entire dataset

# Determine if single file tracking code ran. If not, set the version name and filename using names specified in the constants section
if run_tracksingle == 0:
    singletrack_filebase = 'track' + curr_track_version + '_'

# Call function
if run_gettracks == 1:
    # Load function
    from pyflextrkr.gettracks import gettracknumbers_mergedir

    # Call function
    print('Getting track numbers')
    gettracknumbers_mergedir(datasource, datadescription, tracking_outpath, stats_outpath, startdate, enddate, timegap, nmaxclouds, cloudid_filebase, npxname, tracknumber_version, singletrack_filebase, keepsingletrack=1, removestartendtracks=1)
    tracknumbers_filebase = 'tracknumbers' + tracknumber_version

############################################################
# Calculate final statistics

# Determine if the tracking portion of the code ran. If not, set teh version name and filename using those specified in the constants section
if run_gettracks == 0:
    tracknumbers_filebase = 'tracknumbers' + curr_tracknumbers_version

# Call function
if run_finalstats == 1:
    # Load function
    from pyflextrkr.depreciated.trackstats import trackstats_sat

    # Call satellite version of function
    print('Calculating track statistics')
    trackstats_sat(datasource, datadescription, pixel_radius, latlon_file, geolimits, area_thresh, cloudtb_threshs, absolutetb_threshs, startdate, enddate, cloudid_filebase, tracking_outpath, stats_outpath, track_version, tracknumber_version, tracknumbers_filebase, lengthrange=lengthrange)
    trackstats_filebase = 'stats_tracknumbers' + tracknumber_version

##############################################################
# Identify MCS

# Determine if final statistics portion ran. If not, set the version name and filename using those specified in the constants section
if run_finalstats == 0:
    trackstats_filebase = 'stats_tracknumbers' + curr_tracknumbers_version

if run_identifymcs == 1:
    print('Identifying MCSs')
    # Load function
    from pyflextrkr.identifymcs import identifymcs_mergedir

    # Call satellite version of function
    identifymcs_mergedir(trackstats_filebase, stats_outpath, startdate, enddate, datatimeresolution, mcs_areathresh, mcs_durationthresh, mcs_eccentricitythresh, mcs_splitduration, mcs_mergeduration, nmaxclouds)
    mcsstats_filebase =  'mcs_tracks_'

############################################################
# Create pixel files with MCS tracks

# Determine if the mcs identification and statistic generation step ran. If not, set the filename using those specified in the constants section
if run_identifymcs == 0:
    mcsstats_filebase =  'mcs_tracks_'

if run_labelmcs == 1:
    print('Identifying which pixel level maps to generate for the MCS tracks')

    ###########################################################
    # Identify files to process

    # Load MCS track stat file
    mcsstatistics_file = stats_outpath + mcsstats_filebase + startdate + '_' + enddate + '.nc'
    print(mcsstatistics_file)

    allmcsdata = Dataset(mcsstatistics_file, 'r')
    nmcs = len(allmcsdata.dimensions['ntracks']) # Total number of tracked mcss
    mcstrackstat_basetime = allmcsdata.variables['mcs_basetime'][:] # basetime of each cloud in the tracked mcs
    allmcsdata.close() 

    # Determine times that need to be processed
    if nmcs > 0:
        # Set default time range
        startbasetime = np.nanmin(mcstrackstat_basetime)
        endbasetime = np.nanmax(mcstrackstat_basetime)

        # Find unique times
        uniquebasetime = np.unique(mcstrackstat_basetime)
        uniquebasetime = uniquebasetime[0:-1]
        nuniquebasetime = len(uniquebasetime)

        #############################################################
        # Process files

        # Load function 
        from pyflextrkr.depreciated.mapmcs import mapmcs_mergedir

        # Generate input list
        list_mcstrackstat_filebase = [mcsstats_filebase]*nuniquebasetime
        list_trackstat_filebase = [trackstats_filebase]*nuniquebasetime
        list_mcstracking_path = [mcstracking_outpath]*nuniquebasetime
        list_stats_path = [stats_outpath]*nuniquebasetime
        list_tracking_path = [tracking_outpath]*nuniquebasetime
        list_cloudid_filebase = [cloudid_filebase]*nuniquebasetime
        list_absolutetb_threshs = np.ones(((nuniquebasetime), 2))*absolutetb_threshs
        list_startdate = [startdate]*(nuniquebasetime)
        list_enddate = [enddate]*(nuniquebasetime)

        mcsmap_input = zip(uniquebasetime, list_mcstrackstat_filebase, list_trackstat_filebase, list_mcstracking_path, list_stats_path, list_tracking_path, list_cloudid_filebase, list_absolutetb_threshs, list_startdate, list_enddate)

        # Call function
        dview.map_sync(mapmcs_mergedir, mcsmap_input)

    else:
        print('No MCSs to process ?!')
