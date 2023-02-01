import numpy as np
import os, fnmatch, sys
import datetime, calendar
from pytz import utc
from multiprocessing import Pool
import json

# Name: Run_LESData.py

# Purpose: Master script for tracking shallow clouds in LES data

# Author: Orginial IDL version written by Zhe Feng (zhe.feng@pnnl.gov). Adapted to Python by Hannah Barnes (hannah.barnes@pnnl.gov)

# Get configuration file name from input
config_file = sys.argv[1]
# Read configuration from json file
with open(config_file, encoding='utf-8') as data_file:
    config = json.load(data_file)

run_idclouds = config['run_idclouds']        # Segment and identify cloud systems
run_tracksingle = config['run_tracksingle']     # Track single consecutive cloudid files
run_gettracks = config['run_gettracks']       # Run trackig for all files
run_finalstats = config['run_finalstats']      # Calculate final statistics
run_identifycell = config['run_identifycell']    # Isolate cells
run_labelcell = config['run_labelcell']        # Create maps of MCSs
startdate = config['startdate']
enddate = config['enddate']
run_parallel = config['run_parallel']
nprocesses = config['nprocesses']
root_path = config['root_path']
clouddata_path = config['clouddata_path']

#run_idclouds = int(sys.argv[1])        # Segment and identify cloud systems
#run_tracksingle = int(sys.argv[2])     # Track single consecutive cloudid files
#run_gettracks = int(sys.argv[3])       # Run trackig for all files
#run_finalstats = int(sys.argv[4])      # Calculate final statistics
#run_identifycell = int(sys.argv[5])    # Isolate cells
#run_labelcell = int(sys.argv[6])       # Create maps of MCSs
#startdate = sys.argv[7]           # Start date/time for tracking
#enddate = sys.argv[8]             # End date/time for tracking
#nprocesses = int(sys.argv[9])          # Number of processes to run if run_parallel is set to 1

################################################################################################
# Set variables describing data, file structure, and tracking thresholds

# Specify which sets of code to run. (1 = run code, 0 = don't run code)
#run_idclouds = 0        # Segment and identify cloud systems
#run_tracksingle = 0     # Track single consecutive cloudid files
#run_gettracks = 0       # Run trackig for all files
#run_finalstats = 1      # Calculate final statistics
#run_identifycell = 0    # Isolate cells
#run_labelcell = 0        # Create maps of MCSs
#
## Specify days to run
#startdate = '20160830.1800'
#enddate = '20160830.2000'

# Set version of cloudid code
cloudidmethod = 'futyan4'
keep_singlemergesplit = 1 # 0=all short tracks removed, 1=only short tracks that are not mergers or splits are removed
show_alltracks = 0 # 0=do not create maps of all tracks in map stage, 1=create maps of all tracks (greatly slows the code)
#run_parallel = 1 # Options: 0-run serially, 1-run parallel (uses Pool from Multiprocessing)
#nprocesses = 4   # Number of processes to run if run_parallel is set to 1

# Specify version of code using
cloudid_version = 'v1.0'
track_version = 'v1.0'
tracknumber_version = 'v1.0'

# Specify default code version
curr_id_version = 'v1.0'
curr_track_version = 'v1.0'
curr_tracknumbers_version = 'v1.0'

# Specify domain size
ny = int(1200)
nx = int(1200)

# Specify cloud tracking parameters
#geolimits = np.array([36.05, -98.12, 37.15, -96.79])  # 4-element array with plotting boundaries [lat_min, lon_min, lat_max, lon_max]
geolimits = np.array([-90., -180., 90., 180.])  # 4-element array with plotting boundaries [lat_min, lon_min, lat_max, lon_max]
pixel_radius = 0.1                         # km
timegap = 1.1/float(60)                    # hour
area_thresh = 0.09                         # km^2
miss_thresh = 0.2                          # Missing data threshold. If missing data in the domain from this file is greater than this value, this file is considered corrupt and is ignored. (0.1 = 10%)
cloudlwp_core = 0.75                       # K
cloudlwp_cold = 0.2                       # K
cloudlwp_warm = 0.1                       # K
cloudlwp_cloud = 0.1                      # K
othresh = 0.5                              # overlap percentage threshold
lengthrange = np.array([5, 250])            # A vector [minlength,maxlength] to specify the lifetime range for the tracks
maxnclouds = 6000                          # Maximum clouds in one file
nmaxlinks = 10                             # Maximum number of clouds that any single cloud can be linked to
absolutelwp_threshs = np.array([-1.0e-6, 20])    # k A vector [min, max] brightness temperature allowed. Brightness temperatures outside this range are ignored.
warmanvilexpansion = 0                     # If this is set to one, then the cold anvil is spread laterally until it exceeds the warm anvil threshold
mincoldcorepix = 4                         # Minimum number of pixels for the cold core, needed for futyan version 4 cloud identification code. Not used if use futyan version 3.
smoothwindowdimensions = 10                # Dimension of the boxcar filter used for futyan version 4. Not used in futyan version 3

## Specify cell track parameters
maincloud_duration = 4/float(60)                      # Natural time resolution of data
merge_duration = 4/float(60)                          # Track shorter than this will be labeled as merger
split_duration = 4/float(60)                         # Track shorter than this will be labeled as merger

# Specify filenames and locations
#datavariablename = 'IRBT'
datasource = 'LES'
datadescription = 'SGP'
databasename = 'outmet_d02_'
label_filebase = 'cloudtrack_'

#root_path = '/scratch2/scratchdirs/hcbarnes/LES/'
#clouddata_path = '/scratch2/scratchdirs/hcbarnes/LES/data/'
#root_path = '/global/cscratch1/sd/feng045/hiscale/les/control_d02/'
#clouddata_path = f'{root_path}control_d02/data/'
#root_path = '/global/cscratch1/sd/feng045/hiscale/les/sitivity5_d02/'
#clouddata_path = f'{root_path}/lwp_d02_new/'
#scratchpath = './'
latlon_file = clouddata_path + 'coordinates_d02_big.dat'

# Specify data structure
datatimeresolution = 1/float(60)            # hours
dimname = 'nclouds'
numbername = 'convcold_cloudnumber'
typename = 'cloudtype'
npxname = 'ncorecoldpix'
#tdimname = 'time'
#xdimname = 'Lat_Grid'
#ydimname = 'Lon_Grid'

######################################################################
# Generate additional settings

# Isolate year
year = startdate[0:5]

# Concatonate thresholds into one variable
cloudlwp_threshs = np.hstack((cloudlwp_core, cloudlwp_cold, cloudlwp_warm, cloudlwp_cloud))

# Specify additional file locations
#datapath = root_path                            # Location of raw data
tracking_outpath = root_path + 'tracking/'       # Data on individual features being tracked
stats_outpath = root_path + 'stats/'             # Data on track statistics
celltracking_outpath = root_path + 'celltracking/' + startdate + '_' + enddate + '/' # Pixel level data for MCSs

####################################################################
# Execute tracking scripts

# Create output directories
if not os.path.exists(tracking_outpath):
    #os.makedirs(tracking_outpath)
    os.system('mkdir -p ' + tracking_outpath)

if not os.path.exists(stats_outpath):
    #os.makedirs(stats_outpath)
    os.system('mkdir -p ' + stats_outpath)

if not os.path.exists(celltracking_outpath):
    #os.makedirs(celltracking_outpath)
    os.system('mkdir -p ' + celltracking_outpath)

########################################################################
# Calculate basetime of start and end date
TEMP_starttime = datetime.datetime(int(startdate[0:4]), int(startdate[4:6]), int(startdate[6:8]), int(startdate[9:11]), int(startdate[11:]), 0, tzinfo=utc)
start_basetime = calendar.timegm(TEMP_starttime.timetuple())

TEMP_endtime = datetime.datetime(int(enddate[0:4]), int(enddate[4:6]), int(enddate[6:8]), int(enddate[9:11]), int(enddate[11:]), 0, tzinfo=utc)
end_basetime = calendar.timegm(TEMP_endtime.timetuple())

##########################################################################
# Identify clouds / features in the data, if neccesary
if run_idclouds == 1:
    ######################################################################
    # Identify files to process
    print('Identifying raw data files to process.')

    # Isolate all possible files
    allrawdatafiles = fnmatch.filter(os.listdir(clouddata_path), databasename+'*')

    # Loop through files, identifying files within the startdate - enddate interval
    nleadingchar = np.array(len(databasename)).astype(int)

    rawdatafiles = [None]*len(allrawdatafiles)
    files_timestring = [None]*len(allrawdatafiles) 
    files_datestring = [None]*len(allrawdatafiles)
    files_basetime = np.ones(len(allrawdatafiles), dtype=int)*-9999
    filestep = 0
    for ifile in allrawdatafiles:
        TEMP_filetime = datetime.datetime(int(ifile[nleadingchar:nleadingchar+4]), int(ifile[nleadingchar+5:nleadingchar+7]), int(ifile[nleadingchar+8:nleadingchar+10]), int(ifile[nleadingchar+11:nleadingchar+13]), int(ifile[nleadingchar+14:nleadingchar+16]), 0, tzinfo=utc)
        TEMP_filebasetime = calendar.timegm(TEMP_filetime.timetuple())

        if TEMP_filebasetime >= start_basetime and TEMP_filebasetime <= end_basetime: # and int(ifile[nleadingchar+14:nleadingchar+16]) == 0:

            rawdatafiles[filestep] = clouddata_path + ifile
            files_timestring[filestep] = ifile[nleadingchar+11:nleadingchar+13] + ifile[nleadingchar+14:nleadingchar+16]
            files_datestring[filestep] = ifile[nleadingchar:nleadingchar+4] + ifile[nleadingchar+5:nleadingchar+7] + ifile[nleadingchar+8:nleadingchar+10]
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
    from pyflextrkr.depreciated.idclouds import idclouds_LES

    # Generate input lists
    list_datasource = [datasource]*(filestep)
    list_datadescription = [datadescription]*(filestep)
    list_cloudidversion = [cloudid_version]*(filestep)
    list_trackingoutpath = [tracking_outpath]*(filestep)
    list_latlonfile = [latlon_file]*(filestep)
    list_geolimits = np.ones(((filestep), 4))*geolimits
    list_startdate = [startdate]*(filestep)
    list_enddate = [enddate]*(filestep)
    list_pixelradius = np.ones(filestep)*pixel_radius
    list_areathresh = np.ones(filestep)*area_thresh
    list_cloudlwpthreshs = np.ones((filestep,4))*cloudlwp_threshs
    list_absolutelwpthreshs = np.ones(((filestep), 2))*absolutelwp_threshs
    list_missthresh = np.ones(filestep)*miss_thresh
    list_cloudidmethod = [cloudidmethod]*(filestep)
    list_warmanvilexpansion = np.ones(filestep)*warmanvilexpansion
    list_coldcorethresh = np.ones(filestep)*mincoldcorepix
    list_smoothsize = [smoothwindowdimensions]*(filestep)
    list_xsize = np.ones(filestep)*nx
    list_ysize = np.ones(filestep)*ny

    idclouds_input = list(zip(rawdatafiles, files_datestring, files_timestring, files_basetime, list_datasource, list_datadescription, list_cloudidversion, list_trackingoutpath, list_latlonfile, list_geolimits, list_xsize, list_ysize, list_startdate, list_enddate, list_pixelradius, list_areathresh, list_cloudlwpthreshs, list_absolutelwpthreshs, list_missthresh, list_cloudidmethod, list_coldcorethresh, list_smoothsize, list_warmanvilexpansion))

    ## Call function
    if run_parallel == 0:
        # Serial version
        for ifile in range(0, filestep):
            idclouds_LES(idclouds_input[ifile])
    elif run_parallel == 1:
        # Parallel version
        if __name__ == '__main__':
            print('Identifying clouds')
            pool = Pool(nprocesses)
            pool.map(idclouds_LES, idclouds_input)
            pool.close()
            pool.join()
    else:
        sys.exit('Valid parallelization flag not provided')

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

    trackclouds_input = list(zip(cloudidfiles[0:-1], cloudidfiles[1::], cloudidfiles_datestring[0:-1], cloudidfiles_datestring[1::], cloudidfiles_timestring[0:-1], cloudidfiles_timestring[1::], cloudidfiles_basetime[0:-1], cloudidfiles_basetime[1::], list_trackingoutpath, list_trackversion, list_timegap, list_nmaxlinks, list_othresh, list_startdate, list_enddate))

    if run_parallel == 0:
        # Serial version
        for ifile in range(0, cloudidfilestep-1):
            trackclouds_mergedir(trackclouds_input[ifile])
    elif run_parallel == 1:
        # parallelize version
        if __name__ == '__main__':
            pool = Pool(nprocesses)
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
    singletrack_filebase = 'track' + curr_track_version + '_'

# Call function
if run_gettracks == 1:
    # Load function
    from pyflextrkr.gettracks import gettracknumbers

    # Call function
    print('Getting track numbers')
    print('tracking_out:' + tracking_outpath)
    gettracknumbers(datasource, datadescription, tracking_outpath, stats_outpath, startdate, enddate, timegap, maxnclouds, cloudid_filebase, npxname, tracknumber_version, singletrack_filebase, keepsingletrack=keep_singlemergesplit, removestartendtracks=1)
    tracknumbers_filebase = 'tracknumbers' + tracknumber_version
    print('tracking_out done')

############################################################
# Calculate final statistics

# Determine if the tracking portion of the code ran. If not, set teh version name and filename using those specified in the constants section
if run_gettracks == 0:
    tracknumbers_filebase = 'tracknumbers' + curr_tracknumbers_version

# Call function
if run_finalstats == 1:
    # Load function
    from pyflextrkr.depreciated.trackstats import trackstats_LES

    # Call satellite version of function
    print('Calculating cell statistics')
    trackstats_LES(datasource, datadescription, pixel_radius, latlon_file, geolimits, area_thresh, cloudlwp_threshs, absolutelwp_threshs, startdate, enddate, timegap, cloudid_filebase, tracking_outpath, stats_outpath, track_version, tracknumber_version, tracknumbers_filebase, lengthrange=lengthrange)
    trackstats_filebase = 'stats_tracknumbers' + tracknumber_version

##############################################################
# Identify cell candidates

# Determine if final statistics portion ran. If not, set the version name and filename using those specified in the constants section
if run_finalstats == 0:
    trackstats_filebase = 'stats_tracknumbers' + curr_tracknumbers_version

if run_identifycell == 1:
    print('Identifying Cells')

    # Load function
    from pyflextrkr.depreciated.identifycell import identifycell_LES_xarray

    # Call satellite version of function
    identifycell_LES_xarray(trackstats_filebase, stats_outpath, startdate, enddate, datatimeresolution, geolimits, maincloud_duration, merge_duration, split_duration, lengthrange[1])
    cellstats_filebase =  'cell_tracks_'

############################################################
# Create pixel files with MCS tracks

# Determine if the mcs identification and statistic generation step ran. If not, set the filename using those specified in the constants section
if run_identifycell == 0:
    cellstats_filebase =  'cell_tracks_'

if run_labelcell == 1:
    print('Identifying which pixel level maps to generate for the cell tracks')

    ###########################################################
    # Identify files to process
    if run_tracksingle == 0:
        ################################################################
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
    from pyflextrkr.depreciated.mapcell import mapcell_LES

    # Generate input list
    list_cellstat_filebase = [cellstats_filebase]*(cloudidfilestep-1)
    list_trackstat_filebase = [trackstats_filebase]*(cloudidfilestep-1)
    list_celltracking_path = [celltracking_outpath]*(cloudidfilestep-1)
    list_stats_path = [stats_outpath]*(cloudidfilestep-1)
    list_cloudid_filebase = [cloudid_filebase]*(cloudidfilestep-1)
    list_absolutelwp_threshs = np.ones((cloudidfilestep-1, 2))*absolutelwp_threshs
    list_startdate = [startdate]*(cloudidfilestep-1)
    list_enddate = [enddate]*(cloudidfilestep-1)
    list_showalltracks = [show_alltracks]*(cloudidfilestep-1)

    cellmap_input = list(zip(cloudidfiles, cloudidfiles_basetime, list_cellstat_filebase, list_trackstat_filebase, list_celltracking_path, list_stats_path, list_absolutelwp_threshs, list_startdate, list_enddate,list_showalltracks))

    ## Call function
    if run_parallel == 0:
        # Call function
        for iunique in range(0, cloudidfilestep-1):
            mapcell_LES(cellmap_input[iunique])
    elif run_parallel == 1:
        if __name__ == '__main__':
            print('Creating maps of tracked MCSs')
            pool = Pool(nprocesses)
            pool.map(mapcell_LES, cellmap_input)
            pool.close()
            pool.join()
    else:
        sys.ext('Valid parallelization flag not provided')
