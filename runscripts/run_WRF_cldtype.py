import numpy as np
import os, fnmatch
import time, datetime, calendar
from pytz import utc
from multiprocessing import Pool

#import cProfile

# Purpose: Master script for tracking MCS identified using IR satellite and radar data in the central and eastern USA. 

# Comments:
# Features are tracked using 8 sets of code (idclouds, tracksingle, gettracks, 
# trackstats, identifymcs, matchpf, robustmcs, mapmcs).
# idclouds, tracksingle, and mapcs are run in parallel using the multiprocessing module. 
# All other sets of code are run serially.
# The code does not need to run through each step each time. The code can be run 
# starting at any step, as long as those previous codes have been run and their output is availiable. 

# Author: Orginial IDL version written by Sally McFarlane and Zhe Feng (zhe.feng@pnnl.gov). 
# Adapted to Python by Hannah Barnes (hannah.barnes@pnnl.gov). Adapted for WRF output by Katelyn Barber (katelyn.barber@pnnl.gov)

print('Code Started')
print((time.ctime()))

##################################################################################################
# Set variables describing data, file structure, and tracking thresholds

# Specify which sets of code to run. (1 = run code, 0 = don't run code)
run_idclouds = 1        # Segment and identify cloud systems
run_tracksingle = 1     # Track single consecutive cloudid files
run_gettracks = 1       # Run trackig for all files
run_finalstats = 1      # Calculate final statistics
run_identifymcs = 0     # Isolate MCSs
run_matchpf = 0         # Identify precipitation features with MCSs
run_matchtbpf = 0       # Match brightness temperature tracking defined MCSs with precipitation files from WRF
use_wrf_rainrate = 0    # Using wrf rainrate- pfstats file will have 'WRF' identification
run_robustmcs = 0       # Filter potential mcs cases using nmq radar variables
run_robustmcspf = 0     # Filter potential mcs cases using precipitation features (NOT REFLECTIVITY)
run_labelmcs = 0        # Create pixel maps of MCSs
run_labelmcspf = 0      # Create pixel maps of MCSs from WRF precipitation features (NOT REFLECTIVITY)
run_labelct = 1         # Create pixel maps of cloud type objects that were tracked   

file_rr_tb = 1          # Input brightness temperature and rainrate from WRF are in the same file (0- they are in separate files)

# Set version ofcode
cloudidmethod = 'futyan3'   # Option: futyan3 = identify cores and cold anvils and expand to get warm anvil, futyan4=identify core and expand for cold and warm anvils
keep_singlemergesplit = 1   # Options: 0=All short tracks are removed, 1=Only short tracks without mergers or splits are removed
show_alltracks = 0          # Options: 0=Maps of all tracks are not created, 1=Maps of all tracks are created (much slower!)
run_parallel = 1            # Options: 0-run serially, 1-run parallel (uses Pool from Multiprocessing)
nprocesses = 32             # Number of processors to use if run_parallel is set to 1
process_halfhours = 0       # 0 = No, 1 = Yes

# Specify version of code using
cloudid_version = 'ct.0'
track_version = 'ct.0'
tracknumber_version = 'ct.0'

# Specify default code version
curr_id_version = 'ct.0'
curr_track_version = 'ct.0'
curr_tracknumbers_version = 'ct.0'

# Specify days to run, (YYYYMMDD.hhmm)
startdate = '20150302.0000'
enddate = '20150303.0000'

# Specify cloud tracking parameters
geolimits = np.array([-90, -180, 90, 180])  # 4-element array with plotting boundaries [lat_min, lon_min, lat_max, lon_max]
pixel_radius = 2.0                         # km
timegap = 1                                # hour
area_thresh = 64                           # km^2
miss_thresh = 0.2                          # Missing data threshold. If missing data in the domain from this file is greater than this value, this file is considered corrupt and is ignored. (0.1 = 10%)
cloudtb_core = 220.  #220                      # K # Vant-Hull et al. 2016 (220)
cloudtb_cold = 245.  #245                      # K # Vant-Hull et al. 2016
cloudtb_warm = 261.                        # K
cloudtb_cloud = 261.                       # K
othresh = 0.5                             # overlap percentage threshold
lengthrange = np.array([2,200])            # A vector [minlength,maxlength] to specify the lifetime range for the tracks
nmaxlinks = 50                             # Maximum number of clouds that any single cloud can be linked to
nmaxclouds = 3000                          # Maximum number of clouds allowed to be in one track
absolutetb_threshs = np.array([160, 330])  # k A vector [min, max] brightness temperature allowed. Brightness temperatures outside this range are ignored.
warmanvilexpansion = 1                     # If this is set to one, then the cold anvil is spread laterally until it exceeds the warm anvil threshold
mincoldcorepix = 2                         # Minimum number of pixels for the cold core, needed for futyan version 4 cloud identification code. Not used if use futyan version 3.
smoothwindowdimensions = 5                 # Dimension of the boxcar filter used for futyan version 4. Not used in futyan version 3

# Specify MCS parameters
mcs_mergedir_areathresh = 10000            # IR area threshold [km^2]
mcs_mergedir_durationthresh = 4            # IR minimum length of a mcs [hr]
mcs_mergedir_eccentricitythresh = 0.7      # IR eccentricity at time of maximum extent
mcs_mergedir_splitduration = 4             # IR tracks smaller or equal to this length will be included with the MCS splits from 
mcs_mergedir_mergeduration = 4             # IR tracks smaller or equal to this length will be included with the MCS merges into 
mcs_mergedir_timegap = 3                   # Number of times MCS area is allowed to be below area threshold and still considered an MCS, variant on time resolution of input 

mcs_pf_majoraxisthresh = 100               # PF major axis MCS threshold [km]
mcs_pf_durationthresh = 4                  # PF minimum length of mcs [hr]
mcs_pf_aspectratiothresh = 4               # PF aspect ratio require to define a squall lines 
mcs_pf_lifecyclethresh = 8                 # Minimum MCS lifetime required to classify life stages
mcs_pf_lengththresh = 20                   # Minimum size required to classify life stages
mcs_pf_gap = 3                             # Allowable gap in data for subMCS characteristics [hr] # techincally number of times in a row

# Specify rain rate parameters
rr_min = 1                                 # Rain rate threshold [mm/hr]
nmaxpf = 3                                 # Maximum number of precipitation features that can be within a cloud feature
nmaxcore = 20                              # Maximum number of convective cores that can be within a cloud feature
pcp_thresh = 1                             # Pixels with hourly precipitation larger than this will be labeled with track number

# Specific parameters to link cloud objects using PF
linkpf = 1                                 # Set to 1 to turn on linkpf option; default: 0
pf_smooth_window = 5                       # Smoothing window for identifying PF
pf_dbz_thresh = 3                          # [dBZ] for reflectivity, or [mm/h] for rainrate
pf_link_area_thresh = 648.0                 # [km^2]

# Specify filenames and locations
datavariablename = 'tb'
irdatasource = 'WRF'
nmqdatasource = 'nmq'
precipdatasource = 'WRF'
datadescription = 'WRF_Output'
if file_rr_tb == 1:
    databasename = 'wrfout_rainrate_tb'
    rainaccumulation_filebase = 'wrfout_rainrate_tb'
else:
    databasename = 'wrfout_tb'
    rainaccumulation_filebase = 'wrfout_rainrate_'+ startdate[0:4]
      
label_filebase = 'cloudtrack_'
pfdata_filebase = 'csa4km_'

############### TESTING DIRECTORIES ###########
#root_path = '/global/cscratch1/sd/feng045/goamazon/wrf/AMAZON_CONTROL01/'
#clouddata_path = '/global/cscratch1/sd/barb672/WRF381/AMAZON_CONTROL01/rr_tb/'
#pfdata_path = '/global/cscratch1/sd/barb672/WRF381/AMAZON_CONTROL01/rr_tb/'
#rainaccumulation_path = '/global/cscratch1/sd/barb672/WRF381/AMAZON_CONTROL01/rr_tb/'
#latlon_file = clouddata_path + databasename + '_' + startdate[0:4] + '-' + startdate[4:6] + '-' + startdate[6:8] + '_' + startdate[9:11] + ':' + startdate[11:13] + ':00.nc'

root_path = '/global/homes/b/barb672/Codes/Tracking/pyflextrkr/'
clouddata_path = '/global/cscratch1/sd/barb672/WRF4/AMAZON_EDMF/'
pfdata_path = '/global/cscratch1/sd/barb672/WRF4/AMAZON_EDMF/'
rainaccumulation_path = '/global/cscratch1/sd/barb672/WRF4/AMAZON_EDMF/'
scratchpath = '/global/cscratch1/sd/barb672/WRF4/AMAZON_EDMF/'
latlon_file = clouddata_path + databasename + '_' + startdate[0:4] + '-' + startdate[4:6] + '-' + startdate[6:8] + '_' + startdate[9:11] + ':' + startdate[11:13] + ':00.nc'

###############################################
# Specify data structure
datatimeresolution = 0.5                     # hours
dimname = 'nclouds'
numbername = 'convcold_cloudnumber'
typename = 'cloudtype'
npxname = 'ncorecoldpix'
tdimname = 'time'
xdimname = 'lat2d'
ydimname = 'lon2d'
pcpvarname = 'precipitation'
pfvarname = 'rainrate'

######################################################################
# Generate additional settings

# Isolate year
year = startdate[0:4]

# Concatenate thresholds into one variable
cloudtb_threshs = np.hstack((cloudtb_core, cloudtb_cold, cloudtb_warm, cloudtb_cloud))

# Specify additional file locations
#tracking_outpath = root_path + 'tracking/'
#stats_outpath = root_path + 'stats/'
#mcstracking_outpath = root_path + 'mcstracking/' + startdate + '_' + enddate + '/'

tracking_outpath = clouddata_path + 'cloudtype_tracking/tracking/'
stats_outpath = clouddata_path + 'cloudtype_tracking/stats/'
mcstracking_outpath = clouddata_path + 'mcstracking/' + startdate + '_' + enddate + '/'
cttracking_outpath = clouddata_path + 'cttracking/' + startdate + '_' + enddate + '/'

####################################################################
# Execute tracking scripts

# Create output directories
if not os.path.exists(tracking_outpath):
    os.makedirs(tracking_outpath)

if not os.path.exists(stats_outpath):
    os.makedirs(stats_outpath)

########################################################################
# Calculate basetime of start and end date
TEMP_starttime = datetime.datetime(int(startdate[0:4]), int(startdate[4:6]), int(startdate[6:8]), 
int(startdate[9:11]), int(startdate[11:13]), 0, tzinfo=utc)
start_basetime = calendar.timegm(TEMP_starttime.timetuple())

TEMP_endtime = datetime.datetime(int(enddate[0:4]), int(enddate[4:6]), int(enddate[6:8]),
int(enddate[9:11]), int(enddate[11:13]), 0, tzinfo=utc)
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
    
    # Sort the files by date and time
    def fdatetime(x):
        return(x[-22:])
    allrawdatafiles = sorted(allrawdatafiles, key = fdatetime)
    
    # Loop through files, identifying files within the startdate - enddate interval
    nleadingchar = np.array(len(databasename)).astype(int)
    rawdatafiles = [None]*len(allrawdatafiles)
    
    # KB changed to make minute string defined (otherwise 10 minute files were going past enddate)
    filestep = 0
    for ifile in allrawdatafiles:
        TEMP_filetime = datetime.datetime(int(ifile[nleadingchar+1:nleadingchar+5]),        
        int(ifile[nleadingchar+7:nleadingchar+8]), int(ifile[nleadingchar+9:nleadingchar+11]),
        int(ifile[nleadingchar+12:nleadingchar+14]), int(ifile[nleadingchar+15:nleadingchar+17]), 0, tzinfo=utc)
        TEMP_filebasetime = calendar.timegm(TEMP_filetime.timetuple())

        if TEMP_filebasetime >= start_basetime and TEMP_filebasetime <= end_basetime:
            rawdatafiles[filestep] = clouddata_path + ifile
            filestep = filestep + 1    

#    filestep = 0
#    for ifile in allrawdatafiles:
#        TEMP_filetime = datetime.datetime(int(ifile[nleadingchar+1:nleadingchar+5]),        
#        int(ifile[nleadingchar+7:nleadingchar+8]), int(ifile[nleadingchar+9:nleadingchar+11]),
#        int(ifile[nleadingchar+12:nleadingchar+14]), 0, 0, tzinfo=utc)
#        TEMP_filebasetime = calendar.timegm(TEMP_filetime.timetuple())

#        if TEMP_filebasetime >= start_basetime and TEMP_filebasetime <= end_basetime:
#            rawdatafiles[filestep] = clouddata_path + ifile
#            filestep = filestep + 1
            
    # Remove extra rows
    rawdatafiles = rawdatafiles[0:filestep]
    
    ##########################################################################
    # Process files
    # Load function
    from pyflextrkr.depreciated.idclouds import idclouds_ct

    # Generate input lists
    list_irdatasource = [irdatasource]*(filestep)
    list_datadescription = [datadescription]*(filestep)
    list_datavariablename = [datavariablename]*(filestep)
    list_cloudidversion = [cloudid_version]*(filestep)
    list_trackingoutpath = [tracking_outpath]*(filestep)
    list_latlonfile = [latlon_file]*(filestep)
    #list_latname = [xdimname]*(filestep)
    #list_lonname = [ydimname]*(filestep)
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
    list_linkpf = [linkpf]*(filestep)
    list_pfsmoothwindow = [pf_smooth_window]*(filestep)
    list_pfdbzthresh = [pf_dbz_thresh]*(filestep)
    list_pflinkareathres = [pf_link_area_thresh]*(filestep)
    list_pfvarname = [pfvarname]*(filestep)

    idclouds_input = list(zip(rawdatafiles, list_irdatasource, list_datadescription,
                            list_datavariablename, list_cloudidversion, list_trackingoutpath, list_latlonfile, 
                            list_geolimits, list_startdate, list_enddate, list_pixelradius, list_areathresh, 
                            list_cloudtbthreshs, list_absolutetbthreshs, list_missthresh, list_cloudidmethod, 
                            list_coldcorethresh, list_smoothsize, list_warmanvilexpansion, list_processhalfhour,
                            list_linkpf, list_pfsmoothwindow, list_pfdbzthresh, list_pflinkareathres, list_pfvarname))

    ## Call function
    t = time.time()

    if run_parallel == 0:
        # Serial version
        for ifile in range(0, filestep):
            idclouds_ct(idclouds_input[ifile])        
    elif run_parallel == 1:
        # Parallel version
        if __name__ == '__main__':
            print('Identifying clouds')
            print((time.ctime()))
            pool = Pool(nprocesses)
            pool.map(idclouds_ct, idclouds_input)
            pool.close()
            pool.join()
            elapsed = time.time()-t
            print('Elapsed time: ',elapsed)
    else:
        sys.exit('Valid parallelization flag not provided')

    cloudid_filebase = irdatasource + '_' + datadescription + '_cloudid' + cloudid_version + '_'

###################################################################
# Link clouds/ features in time adjacent files (single file tracking), if necessary

# Determine if identification portion of the code run. If not, set the version name and filename using names specified in the constants section
if run_idclouds == 0:
    print('Cloud already identified in previous run')
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
        TEMP_cloudidtime = datetime.datetime(int(icloudidfile[nleadingchar:nleadingchar+4]), 
        int(icloudidfile[nleadingchar+4:nleadingchar+6]), int(icloudidfile[nleadingchar+6:nleadingchar+8]), 
        int(icloudidfile[nleadingchar+9:nleadingchar+11]), int(icloudidfile[nleadingchar+11:nleadingchar+13]), 0, tzinfo=utc)
        TEMP_cloudidbasetime = calendar.timegm(TEMP_cloudidtime.timetuple())

        if TEMP_cloudidbasetime >= start_basetime and TEMP_cloudidbasetime <= end_basetime:
            cloudidfiles[cloudidfilestep] = tracking_outpath + icloudidfile
            cloudidfiles_timestring[cloudidfilestep] = icloudidfile[nleadingchar+9:nleadingchar+11] + \
            icloudidfile[nleadingchar+11:nleadingchar+13]
            cloudidfiles_datestring[cloudidfilestep] = icloudidfile[nleadingchar:nleadingchar+4] + \
            icloudidfile[nleadingchar+4:nleadingchar+6] + icloudidfile[nleadingchar+6:nleadingchar+8] 
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

    from pyflextrkr.depreciated.tracksingle_ct import trackclouds

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

    trackclouds_input = list(zip(cloudidfiles[0:-1], cloudidfiles[1::], cloudidfiles_datestring[0:-1], 
    cloudidfiles_datestring[1::], cloudidfiles_timestring[0:-1], cloudidfiles_timestring[1::], 
    cloudidfiles_basetime[0:-1], cloudidfiles_basetime[1::], list_trackingoutpath, list_trackversion,
    list_timegap, list_nmaxlinks, list_othresh, list_startdate, list_enddate))

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
# Track clouds / features through the entire dataset

# Determine if single file tracking code ran. If not, set the version name and filename
# using names specified in the constants section
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
    gettracknumbers(irdatasource, datadescription, tracking_outpath, stats_outpath, startdate, enddate,
                    timegap, nmaxclouds, cloudid_filebase, npxname, tracknumber_version, singletrack_filebase,
                    keepsingletrack=keep_singlemergesplit, removestartendtracks=1)
    tracknumbers_filebase = 'tracknumbers' + tracknumber_version

############################################################
# Calculate final statistics

# Determine if the tracking portion of the code ran. If not, set teh version name and
# filename using those specified in the constants section
if run_gettracks == 0:
    print('Cloud tracks already determined')
    tracknumbers_filebase = 'tracknumbers' + curr_tracknumbers_version

# Call function
if run_finalstats == 1 and run_parallel == 0:
    # Load function
    from pyflextrkr.depreciated.trackstats import trackstats_ct

    # Call satellite version of function
    print('Calculating track statistics')
    print(time.ctime())
    trackstats_ct(irdatasource, datadescription, pixel_radius, geolimits, area_thresh, 
                   cloudtb_threshs, absolutetb_threshs, startdate, enddate, timegap, cloudid_filebase,
                   tracking_outpath, stats_outpath, track_version, tracknumber_version,
                   tracknumbers_filebase, lengthrange=lengthrange)
    trackstats_filebase = 'stats_tracknumbers' + tracknumber_version

if run_finalstats == 1 and run_parallel == 1:
   # Load function
    from pyflextrkr.depreciated.trackstats_ct_parallel import trackstats_ct

    # Call satellite version of function
    print('Calculating track statistics')
    print(time.ctime())
    trackstats_ct(irdatasource, datadescription, pixel_radius, geolimits, area_thresh,
                   cloudtb_threshs, absolutetb_threshs, startdate, enddate, timegap, cloudid_filebase,
                   tracking_outpath, stats_outpath, track_version, tracknumber_version,
                   tracknumbers_filebase, nprocesses,lengthrange=lengthrange)
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
    from pyflextrkr.identifymcs import identifymcs_tb

    # Call wrf version of function
    print((time.ctime()))
    identifymcs_tb(trackstats_filebase, stats_outpath, startdate, enddate,
                            geolimits, datatimeresolution, mcs_mergedir_areathresh, 
                            mcs_mergedir_durationthresh, mcs_mergedir_eccentricitythresh, 
                            mcs_mergedir_splitduration, mcs_mergedir_mergeduration, nmaxlinks, mcs_mergedir_timegap)
    mcsstats_filebase =  'mcs_tracks_'

##############################################################
## Identify preciptation features within MCSs

## Determine if identify mcs portion of code ran. If not set file name
#if run_identifymcs == 0:
#    print('MCSs already identified')
#    mcsstats_filebase = 'mcs_tracks_'

#if run_matchpf == 1:
#    print('Identifying Precipitation Features in MCSs')

#    # Load function
#    from matchpf import identifypf_wrf_rain

#    # Call function
#    print((time.ctime()))
#    identifypf_wrf_rain(mcsstats_filebase, cloudid_filebase, pfdata_filebase, 
#                        rainaccumulation_filebase, stats_outpath, tracking_outpath, pfdata_path,
#                       rainaccumulation_path, startdate, enddate, geolimits, nmaxpf, nmaxcore, 
#                        nmaxclouds, rr_min, pixel_radius, irdatasource, nmqdatasource, datadescription, 
#                        datatimeresolution, mcs_mergedir_areathresh, mcs_mergedir_durationthresh,
#                        mcs_mergedir_eccentricitythresh)
#    pfstats_filebase = 'mcs_tracks_'  + nmqdatasource + '_' 

##############################################################
# Identify precipitation features within MCS
# Match brightness temperature tracking defined MCSs with WRF precipitation
    
# Determine if identify mcs portion of code ran. If not set file name
if run_identifymcs == 0:
    print('MCSs already identified')
    mcsstats_filebase = 'mcs_tracks_'
    
if run_matchtbpf == 1:
    print('Identifying precipitation features in MCSs')
    
    # Load function
    from pyflextrkr.depreciated.matchtbpf import identifypf_wrf_rain
    
    # Call function
    print((time.ctime()))
    identifypf_wrf_rain(mcsstats_filebase, cloudid_filebase,
                        rainaccumulation_filebase, stats_outpath, tracking_outpath, 
                        rainaccumulation_path, startdate, enddate, 
                        geolimits, nmaxpf, nmaxcore, nmaxclouds, rr_min, pixel_radius, 
                        irdatasource, precipdatasource, datadescription, datatimeresolution,
                        mcs_mergedir_areathresh, mcs_mergedir_durationthresh,
                        mcs_mergedir_eccentricitythresh,pf_link_area_thresh)

    pfstats_filebase = 'mcs_tracks_' + precipdatasource + '_'
##############################################################
## Identify robust MCS using precipitation feature statistics (NMQ with reflectivity)

## Determine if identify precipitation feature portion of code ran. If not set file name
#if run_matchpf == 0: 
#    print('MCSs already linked to precipitation data')
#    pfstats_filebase = 'robust_mcs_tracks_'
    
## Run code to identify robust MCS
#if run_robustmcs == 1:
#    print('Identifying robust MCSs using precipitation features')

#    # Load function
#    from robustmcs import filtermcs_wrf_rain

#    # Call function
#    print((time.ctime()))
#    filtermcs_wrf_rain(stats_outpath, pfstats_filebase, startdate, enddate,
#                       datatimeresolution, geolimits, mcs_pf_majoraxisthresh,
#                       mcs_pf_durationthresh, mcs_pf_aspectratiothresh,
#                       mcs_pf_lifecyclethresh, mcs_pf_lengththresh, mcs_pf_gap)
#    robustmcs_filebase = 'robust_mcs_tracks_nmq_'

############################################################
# Identify robust MCS using precipitation feature statistics (WRF with rainrate)
# Determine if identify precipitation feature portion of code ran. If not set file name
if run_matchtbpf == 0: 
    print('MCSs already linked to precipitation data')
    if use_wrf_rainrate == 1:
        pfstats_filebase = 'mcs_tracks_' + precipdatasource + '_'
        
# Run code to identify robust MCS
if run_robustmcspf == 1:
    print('Identifying robust MCSs using precipitation features')

    # Load function
    from pyflextrkr.robustmcspf import filtermcs_wrf_rain

    # Call function
    print((time.ctime()))
    filtermcs_wrf_rain(stats_outpath, pfstats_filebase, startdate, enddate, datatimeresolution,
                       geolimits, mcs_pf_majoraxisthresh, mcs_pf_durationthresh,
                       mcs_pf_aspectratiothresh, mcs_pf_lifecyclethresh, 
                       mcs_pf_lengththresh, mcs_pf_gap)
    robustmcs_filebase = 'robust_mcs_tracks_wrf_'  
    
#############################################################    
## Create pixel files with MCS tracks from NMQ data source

## Determine if the mcs identification and statistic generation step ran. If not, 
## set the filename using those specified in the constants section
#if run_robustmcs == 0:
#    print('Robust MCSs already determined')
#    robustmcs_filebase =  'robust_mcs_tracks_nmq_'

#if run_labelmcs == 1:
#    print('Identifying which pixel level maps to generate for the MCS tracks')

#    ###########################################################
#    # Identify files to process
#    if run_tracksingle == 0:
        ################################################################
#        # Identify files to process
#        print('Identifying cloudid files to process')

#        # Isolate all possible files
#       allcloudidfiles = fnmatch.filter(os.listdir(tracking_outpath), cloudid_filebase +'*')

#        # Put in temporal order
#        allcloudidfiles = sorted(allcloudidfiles)

#        # Loop through files, identifying files within the startdate - enddate interval
#        nleadingchar = np.array(len(cloudid_filebase)).astype(int)

#        cloudidfiles = [None]*len(allcloudidfiles)
#        cloudidfiles_basetime = [None]*len(allcloudidfiles)
#        cloudidfilestep = 0
#        for icloudidfile in allcloudidfiles:
#            TEMP_cloudidtime = datetime.datetime(int(icloudidfile[nleadingchar:nleadingchar+4]),
#            int(icloudidfile[nleadingchar+4:nleadingchar+6]), int(icloudidfile[nleadingchar+6:nleadingchar+8]),
#            int(icloudidfile[nleadingchar+9:nleadingchar+11]), int(icloudidfile[nleadingchar+11:nleadingchar+13]), 0, tzinfo=utc)
#            TEMP_cloudidbasetime = calendar.timegm(TEMP_cloudidtime.timetuple())

#            if TEMP_cloudidbasetime >= start_basetime and TEMP_cloudidbasetime <= end_basetime:
#                cloudidfiles[cloudidfilestep] = tracking_outpath + icloudidfile
#                cloudidfiles_basetime[cloudidfilestep] = np.copy(TEMP_cloudidbasetime)
#                cloudidfilestep = cloudidfilestep + 1

#        # Remove extra rows
#        cloudidfiles = cloudidfiles[0:cloudidfilestep]
#        cloudidfiles_basetime = cloudidfiles_basetime[:cloudidfilestep]

#    #############################################################
#    # Process files

#    # Load function 
#    from mapmcs import mapmcs_mergedir

#    # Generate input list
#    list_robustmcsstat_filebase = [robustmcs_filebase]*(cloudidfilestep-1)
#    list_trackstat_filebase = [trackstats_filebase]*(cloudidfilestep-1)
#    list_pfdata_filebase = [pfdata_filebase]*(cloudidfilestep-1)
#    list_rainaccumulation_filebase = [rainaccumulation_filebase]*(cloudidfilestep-1)
#    list_mcstracking_path = [mcstracking_outpath]*(cloudidfilestep-1)
#    list_stats_path = [stats_outpath]*(cloudidfilestep-1)
#    list_pfdata_path = [pfdata_path]*(cloudidfilestep-1)
#    list_rainaccumulation_path = [rainaccumulation_path]*(cloudidfilestep-1)
#    list_cloudid_filebase = [cloudid_filebase]*(cloudidfilestep-1)
#    list_pcp_thresh = np.ones(cloudidfilestep-1)*pcp_thresh
#    list_nmaxpf = np.ones(cloudidfilestep-1)*nmaxpf
#    list_absolutetb_threshs = np.ones((cloudidfilestep-1, 2))*absolutetb_threshs
#    list_startdate = [startdate]*(cloudidfilestep-1)
#    list_enddate = [enddate]*(cloudidfilestep-1)
#    list_showalltracks = [show_alltracks]*(cloudidfilestep-1)

#    robustmcsmap_input = list(zip(cloudidfiles, cloudidfiles_basetime, list_robustmcsstat_filebase,
#    list_trackstat_filebase, list_pfdata_filebase, list_rainaccumulation_filebase, 
#    list_mcstracking_path, list_stats_path, list_pfdata_path, list_rainaccumulation_path,
#    list_pcp_thresh, list_nmaxpf, list_absolutetb_threshs, list_startdate, list_enddate, list_showalltracks))

#    if run_parallel == 0:
#        # Call function
#        for iunique in range(0, cloudidfilestep-1):
#            mapmcs_mergedir(robustmcsmap_input[iunique])

#        #cProfile.run('mapmcs_pf(robustmcsmap_input[200])')
#    elif run_parallel == 1:
#        if __name__ == '__main__':
#            print('Creating maps of tracked MCSs')
#           print((time.ctime()))
#            pool = Pool(24)
#            pool.map(mapmcs_mergedir, robustmcsmap_input)
#            pool.close()
#            pool.join()
#    else:
#        sys.ext('Valid parallelization flag not provided')

############################################################    
# Create pixel files with MCS tracks from WRF precip (not reflectivity) data source

# Determine if the mcs identification and statistic generation step ran. If not, 
# set the filename using those specified in the constants section
if run_robustmcspf == 0:
    print('Robust MCSs already determined')
    robustmcs_filebase = 'robust_mcs_tracks_' + precipdatasource + '_'

if run_labelmcspf == 1:
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
            TEMP_cloudidtime = datetime.datetime(int(icloudidfile[nleadingchar:nleadingchar+4]),
            int(icloudidfile[nleadingchar+4:nleadingchar+6]), int(icloudidfile[nleadingchar+6:nleadingchar+8]),
            int(icloudidfile[nleadingchar+9:nleadingchar+11]), int(icloudidfile[nleadingchar+11:nleadingchar+13]), 0, tzinfo=utc)
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
    from pyflextrkr.depreciated.mapmcspf import mapmcs_wrf_pf

    # Generate input list
    list_robustmcsstat_filebase = [robustmcs_filebase]*(cloudidfilestep-1)
    list_trackstat_filebase = [trackstats_filebase]*(cloudidfilestep-1)
    list_rainaccumulation_filebase = [rainaccumulation_filebase]*(cloudidfilestep-1)
    list_mcstracking_path = [mcstracking_outpath]*(cloudidfilestep-1)
    list_stats_path = [stats_outpath]*(cloudidfilestep-1)
    list_rainaccumulation_path = [rainaccumulation_path]*(cloudidfilestep-1)
    list_cloudid_filebase = [cloudid_filebase]*(cloudidfilestep-1)
    list_pcp_thresh = np.ones(cloudidfilestep-1)*pcp_thresh
    list_nmaxpf = np.ones(cloudidfilestep-1)*nmaxpf
    list_absolutetb_threshs = np.ones((cloudidfilestep-1, 2))*absolutetb_threshs
    list_startdate = [startdate]*(cloudidfilestep-1)
    list_enddate = [enddate]*(cloudidfilestep-1)
    list_showalltracks = [show_alltracks]*(cloudidfilestep-1)

    robustmcsmap_input = list(zip(cloudidfiles, cloudidfiles_basetime, list_robustmcsstat_filebase,
    list_trackstat_filebase, list_rainaccumulation_filebase, 
    list_mcstracking_path, list_stats_path, list_rainaccumulation_path,
    list_pcp_thresh, list_nmaxpf, list_absolutetb_threshs,
    list_startdate, list_enddate, list_showalltracks))

    if run_parallel == 0:
        # Call function
        for iunique in range(0, cloudidfilestep-1):
            mapmcs_wrf_pf(robustmcsmap_input[iunique])

        #cProfile.run('mapmcs_pf(robustmcsmap_input[200])')
    elif run_parallel == 1:
        if __name__ == '__main__':
            print('Creating maps of tracked MCSs')
            print((time.ctime()))
            pool = Pool(nprocesses)
            pool.map(mapmcs_wrf_pf, robustmcsmap_input)
            pool.close()
            pool.join()
    else:
        sys.ext('Valid parallelization flag not provided')

############################################################    
# Create pixel files with cloud type tracks 
if run_labelct == 1:
    print('Identifying which pixel level maps to generate for the cloud type tracks')    
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
            TEMP_cloudidtime = datetime.datetime(int(icloudidfile[nleadingchar:nleadingchar+4]),
            int(icloudidfile[nleadingchar+4:nleadingchar+6]), int(icloudidfile[nleadingchar+6:nleadingchar+8]),
            int(icloudidfile[nleadingchar+9:nleadingchar+11]), int(icloudidfile[nleadingchar+11:nleadingchar+13]), 0, tzinfo=utc)
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
    from pyflextrkr.depreciated.mapct import map_ct

    # Generate input list
    list_trackstat_filebase = [trackstats_filebase]*(cloudidfilestep-1)
    list_rainaccumulation_filebase = [rainaccumulation_filebase]*(cloudidfilestep-1)
    list_stats_path = [stats_outpath]*(cloudidfilestep-1)
    list_tracking_path = [cttracking_outpath]*(cloudidfilestep-1)
    list_rainaccumulation_path = [rainaccumulation_path]*(cloudidfilestep-1)
    list_cloudid_filebase = [cloudid_filebase]*(cloudidfilestep-1)
    list_startdate = [startdate]*(cloudidfilestep-1)
    list_enddate = [enddate]*(cloudidfilestep-1)
    list_showalltracks = [show_alltracks]*(cloudidfilestep-1)

    map_input = list(zip(cloudidfiles, cloudidfiles_basetime,
    list_trackstat_filebase, list_rainaccumulation_filebase, 
    list_tracking_path, list_stats_path, list_rainaccumulation_path,
    list_startdate, list_enddate, list_showalltracks))

    if run_parallel == 0:
        # Call function
        for iunique in range(0, cloudidfilestep-1):
            map_ct(map_input[iunique])

        #cProfile.run('mapmcs_pf(robustmcsmap_input[200])')
    elif run_parallel == 1:
        if __name__ == '__main__':
            print('Creating maps of tracked MCSs')
            print((time.ctime()))
            pool = Pool(nprocesses)
            pool.map(map_ct, map_input)
            pool.close()
            pool.join()
    else:
        sys.ext('Valid parallelization flag not provided')
