import numpy as np
import os, fnmatch
import time, datetime, calendar
from pytz import timezone, utc

# Name: Run_TestData.py

# Purpose: Master script for trackig synthetic IR satellite data

# Comments:
# Features are tracked using 5 sets of code (idclouds, trackclouds_singlefile, get_tracknumbers, calc_sat_trackstats, label_celltrack).
# This script controls which pieces of code are run.
# Eventually, idclouds and trackclouds_singlefile will be able to run in parallel.
# If trackclouds_singlefile is run in of tracksingle between 12/20/2009 - 12/31/2009, make two copies of this script, and set startdate - enddate (ex: 20091220 - 20091225, 20091225 - 20091231).
# This is because the first time will not have a tracksingle file produced, overlapping the date makes sure every cloudid file is used.
# The idclouds and trackClouds_singlefile only need to be run once and can be run on portions of the data a time.
# However, get_tracknumbers, calc_set_tracks, and label_celltrack must be run for the entire dataset.

# Author: Orginial IDL version written by Zhe Feng (zhe.feng@pnnl.gov). Adapted to Python by Hannah Barnes (hannah.barnes@pnnl.gov)

##################################################################################################
# Set variables describing data, file structure, and tracking thresholds

# Specify which sets of code to run. (1 = run code, 0 = don't run code)
run_idclouds = 1        # Segment and identify cloud systems
run_tracksingle = 1     # Track single consecutive cloudid files
run_gettracks = 1       # Run trackig for all files
run_finalstats = 1      # Calculate final statistics
run_labelcloud = 1      # Create maps with all events in a tracking having the same number

# Specify version of code using
cloudid_version = 'v1.0'
track_version = 'v1.0'
tracknumber_version = 'v1.0'

# Specify default code version
curr_id_version = 'v1.0'
curr_track_version = 'v1.0'
curr_tracknumbers_version = 'v1.0'

# Specify days to run
startdate = '20110520'
enddate = '20110520'

# Specify tracking parameters
geolimits = np.array([25,-110,51,-70]) # 4-element array with plotting boundaries [lat_min, lon_min, lat_max, lon_max]
pixel_radius = 4.0      # km
timegap = 1.1           # hour
area_thresh = 1000 #64.       # km^2
miss_thresh = 0.2       # Missing data threshold. If missing data in the domain from this file is greater than this value, this file is considered corrupt and is ignored. (0.1 = 10%)
cloudtb_core = 225.          # K
cloudtb_cold = 241.          # K
cloudtb_warm = 261.          # K
cloudtb_cloud = 261.         # K
othresh = 0.5                     # overlap percentage threshold
lengthrange = np.array([2,30])    # A vector [minlength,maxlength] to specify the lifetime range for the tracks
nmaxlinks = 10                    # Maximum number of clouds that any single cloud can be linked to
nmaxclouds = 3000                 # Maximum number of clouds allowed to be in one track
absolutetb_threshs = np.array([160,330])        # k A vector [min, max] brightness temperature allowed. Brightness temperatures outside this range are ignored.
warmanvilexpansion = 1            # If this is set to one, then the cold anvil is spread laterally until it exceeds the warm anvil threshold

# Specify filenames and locations
datasource = 'mergedir'
datadescription = 'eus'
databasename = 'mcstrack_'
label_filebase = 'cloudtrack_'

root_path = '/global/homes/h/hcbarnes/Tracking/MCS/'
data_path = '/global/project/projectdirs/m1867/zfeng/usa/mergedir/mcstracking/2011/'
scratchpath = './'
#latlon_file = root_path + 'irdata_20000101_0000.nc'
latlon_file = data_path + 'mcstrack_20110520_0000.nc'

# Character number of dates and times in filename
fileyearindices = [9,13]
filemonthindices = [13,15]
filedayindices = [15,17]
filehourindices = [18,20]
fileminuteindices = [20,22]

# Specify data structure
dimname = 'nclouds'
numbername = 'convcold_cloudnumber'
typename = 'cloudtype'
npxname = 'ncorecoldpix'
tdimname = 'time'
xdimname = 'lon'
ydimname = 'lat'

######################################################################
# Generate additional settings

# Isolate year
year = startdate[0:5]

# Concatonate thresholds into one variable
cloudtb_threshs = np.hstack((cloudtb_core, cloudtb_cold, cloudtb_warm, cloudtb_cloud))

# Specify additional file locations
#datapath = root_path                            # Location of raw data
tracking_outpath = root_path + 'MCStracking/'         # Data on individual features being tracked
stats_outpath = root_path + 'MCSstats/'      # Data on track statistics

######################################################################
# Execute tracking scripts

# Create output directories
if not os.path.exists(tracking_outpath):
    os.makedirs(tracking_outpath)

if not os.path.exists(stats_outpath):
    os.makedirs(stats_outpath)

##########################################################################
# Identify clouds / features in the data, if neccesary
if run_idclouds == 1:
    ######################################################################
    # Identify files to process
    print('Identifying raw data files to process.')

    # Isolate all possible files
    allrawdatafiles = fnmatch.filter(os.listdir(data_path), databasename+'*')

    # Put start and endtimes in basetime
    TEMP_starttime = datetime.datetime(int(startdate[0:4]), int(startdate[4:6]), int(startdate[6:8]), 0, 0, 0, tzinfo=utc)
    start_basetime = calendar.timegm(TEMP_starttime.timetuple())

    TEMP_endtime = datetime.datetime(int(enddate[0:4]), int(enddate[4:6]), int(enddate[6:8]), 23, 0, 0, tzinfo=utc)
    end_basetime = calendar.timegm(TEMP_endtime.timetuple())

    # Loop through files, identifying files within the startdate - enddate interval
    rawdatafiles = [None]*len(allrawdatafiles)
    files_timestring = [None]*len(allrawdatafiles) 
    files_datestring = [None]*len(allrawdatafiles)
    files_basetime = np.ones(len(allrawdatafiles), dtype=int)*-9999
    filestep = 0
    for ifile in allrawdatafiles:
        TEMP_filetime = datetime.datetime(int(ifile[fileyearindices[0]:fileyearindices[1]]), int(ifile[filemonthindices[0]:filemonthindices[1]]), int(ifile[filedayindices[0]:filedayindices[1]]), int(ifile[filehourindices[0]:filehourindices[1]]), int(ifile[fileminuteindices[0]:fileminuteindices[1]]), 0, tzinfo=utc)
        TEMP_filebasetime = calendar.timegm(TEMP_filetime.timetuple())

        if TEMP_filebasetime >= start_basetime and TEMP_filebasetime <= end_basetime:
            rawdatafiles[filestep] = data_path + ifile
            files_timestring[filestep] = ifile[filehourindices[0]:filehourindices[1]] + ifile[fileminuteindices[0]:fileminuteindices[1]]
            files_datestring[filestep] = ifile[fileyearindices[0]:fileyearindices[1]] + ifile[filemonthindices[0]:filemonthindices[1]] + ifile[filedayindices[0]:filedayindices[1]]
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
    from idclouds import idclouds_mergedir

    # Call function
    print('Identifying Clouds')
    for filestep, ifile in enumerate(rawdatafiles):
        idclouds_mergedir(ifile, files_datestring[filestep], files_timestring[filestep], files_basetime[filestep], datasource, datadescription, cloudid_version, tracking_outpath, latlon_file, geolimits, startdate, enddate, pixel_radius, area_thresh, cloudtb_threshs, absolutetb_threshs, miss_thresh, warmanvilexpansion)
    cloudid_filebase = datasource + '_' + datadescription + '_cloudid' + cloudid_version + '_' 

###################################################################
# Link clouds/ features in time adjacent files (single file tracking), if necessary

# Determine if identification portion of the code run. If not, set the version name and filename using names specified in the constants section
if run_idclouds == 0:
    cloudid_filebase =  datasource + '_' + datadescription + '_cloudid' + curr_id_version + '_'

# Call function
if run_tracksingle == 1:
    # Load function
    from tracksingle import trackclouds_mergedir

    # Call function
    print('Tracking clouds between single files')
    trackclouds_mergedir(tracking_outpath, datasource, datadescription, track_version, timegap, nmaxlinks, othresh, startdate, enddate, tracking_outpath, cloudid_filebase)
    singletrack_filebase = 'track' + track_version + '_' 

###########################################################
# Track clouds / features through the entire dataset

# Determine if single file tracking code ran. If not, set the version name and filename using names specified in the constants section
if run_tracksingle == 0:
    singletrack_filebase = 'track' + curr_track_version + '_'

# Call function
if run_gettracks == 1:
    # Load function
    from gettracks import gettracknumbers_mergedir

    # Call function
    print('Getting track numbers')
    gettracknumbers_mergedir(datasource, datadescription, tracking_outpath, stats_outpath, startdate, enddate, timegap, nmaxclouds, cloudid_filebase, npxname, tracknumber_version, singletrack_filebase, keepsingletrack=1, removestartendtracks=1)
    tracknumbers_filebase = 'tracknumbers' + tracknumber_version

############################################################
# Calculate final statistics

# Determine if the tracking portion of the code ran. If not, set teh version name and filename using those specified in the constants section
if run_gettracks == 0:
    track_filebase = 'tracknumbers' + curr_tracknumbers_version

# Call function
if run_finalstats == 1:
    # Load function
    from trackstats import trackstats_sat

    # Call satellite version of function
    print('Calculating track statistics')
    trackstats_sat(datasource, datadescription, pixel_radius, latlon_file, geolimits, area_thresh, cloudtb_threshs, absolutetb_threshs, startdate, enddate, cloudid_filebase, tracking_outpath, stats_outpath, track_version, tracknumber_version, tracknumbers_filebase, lengthrange=lengthrange)






