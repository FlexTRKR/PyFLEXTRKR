import numpy as np
import os, fnmatch, sys, glob
import time, datetime, calendar, pytz
from pytz import timezone, utc
from multiprocessing import Pool
from itertools import repeat
from netCDF4 import Dataset
import xarray as xr
import json

# Name: run_cacti_csapr.py

# Purpose: Master script for tracking convective cells from CACTI CSAPR data

# Author: Zhe Feng (zhe.feng@pnnl.gov)

# Get configuration file name from input
config_file = sys.argv[1]
# Read configuration from json file
with open(config_file, encoding='utf-8') as data_file:
    config = json.load(data_file)

run_idclouds = config['run_idclouds']        # Segment and identify cloud systems
run_tracksingle = config['run_tracksingle']     # Track single consecutive cloudid files
run_gettracks = config['run_gettracks']       # Run trackig for all files
run_finalstats = config['run_finalstats']      # Calculate final statistics
# run_identifycell = config['run_identifycell']    # Isolate cells
run_labelcell = config['run_labelcell']        # Create maps of MCSs
startdate = config['startdate']
enddate = config['enddate']
run_parallel = config['run_parallel']
nprocesses = config['nprocesses']
root_path = config['root_path']
clouddata_path = config['clouddata_path']
if "driftfile" in config:
    driftfile = config['driftfile']



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

keep_singlemergesplit = 1 # 0=all short tracks removed, 1=only short tracks that are not mergers or splits are removed
# show_alltracks = 0 # 0=do not create maps of all tracks in map stage, 1=create maps of all tracks (greatly slows the code)
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

# Specify cloud tracking parameters
geolimits = np.array([-90., -180., 90., 180.])  # 4-element array with plotting boundaries [lat_min, lon_min, lat_max, lon_max]
pixel_radius = 1                         # km
timegap = 30/float(60)                    # hour
area_thresh = 4                              # km^2
miss_thresh = 0.2                          # Missing data threshold. (0.1 = 10%)
othresh = 0.3                              # overlap percentage threshold
lengthrange = np.array([2, 60])            # A vector [minlength,maxlength] to specify the lifetime range for the tracks
maxnclouds = 1000                          # Maximum clouds in one file
nmaxlinks = 10                             # Maximum number of clouds that any single cloud can be linked to
mincellpix = 4                             # Minimum number of pixels for a cell.

## Specify cell track parameters
maincloud_duration = 30/float(60)                      # Natural time resolution of data
merge_duration = 30/float(60)                          # Track shorter than this will be labeled as merger
split_duration = 30/float(60)                         # Track shorter than this will be labeled as merger

# Specify filenames and locations
datasource = 'CSAPR'
datadescription = 'COR'
# databasename = 'CSAPR2_Taranis_Gridded_1000m.Conv_Mask.'
databasename = 'csa_csapr2_'
label_filebase = 'cloudtrack_'

# latlon_file = clouddata_path + 'coordinates_d02_big.dat'

# Specify data structure
datatimeresolution = 15/float(60)            # hours
dimname = 'nclouds'
numbername = 'convcold_cloudnumber'
typename = 'cloudtype'
npxname = 'ncorecoldpix'
celltracking_filebase = 'celltracks_'
# datavariablename = 'comp_ref'
#tdimname = 'time'
#xdimname = 'Lat_Grid'
#ydimname = 'Lon_Grid'

######################################################################
# Specify additional file locations
#datapath = root_path                            # Location of raw data
tracking_outpath = root_path + 'tracking/'       # Data on individual features being tracked
stats_outpath = root_path + 'stats/'             # Data on track statistics
celltracking_outpath = root_path + 'celltracking/' + startdate + '_' + enddate + '/' # Pixel level data for MCSs

####################################################################
# Execute tracking scripts

# Create output directories
os.makedirs(tracking_outpath, exist_ok=True)
os.makedirs(stats_outpath, exist_ok=True)
# os.makedirs(celltracking_outpath, exist_ok=True)

########################################################################
# Calculate basetime of start and end date
TEMP_starttime = datetime.datetime(int(startdate[0:4]), int(startdate[4:6]), int(startdate[6:8]), \
                                    int(startdate[9:11]), int(startdate[11:]), 0, tzinfo=utc)
start_basetime = calendar.timegm(TEMP_starttime.timetuple())

TEMP_endtime = datetime.datetime(int(enddate[0:4]), int(enddate[4:6]), int(enddate[6:8]), \
                                    int(enddate[9:11]), int(enddate[11:]), 0, tzinfo=utc)
end_basetime = calendar.timegm(TEMP_endtime.timetuple())

##########################################################################
# Identify clouds / features in the data, if neccesary
if run_idclouds == 1:
    ######################################################################
    # Identify files to process
    print('Identifying raw data files to process.')

    # Isolate all possible files
    allrawdatafiles = sorted(fnmatch.filter(os.listdir(clouddata_path), databasename+'*.nc'))

    # Loop through files, identifying files within the startdate - enddate interval
    nleadingchar = np.array(len(databasename)).astype(int)

    rawdatafiles = [None]*len(allrawdatafiles)
    files_timestring = [None]*len(allrawdatafiles) 
    files_datestring = [None]*len(allrawdatafiles)
    files_basetime = np.ones(len(allrawdatafiles), dtype=int)*-9999
    filestep = 0
    for ifile in allrawdatafiles:
        TEMP_filetime = datetime.datetime(int(ifile[nleadingchar:nleadingchar+4]), 
                                            int(ifile[nleadingchar+4:nleadingchar+6]), 
                                            int(ifile[nleadingchar+6:nleadingchar+8]), 
                                            int(ifile[nleadingchar+9:nleadingchar+11]), 
                                            int(ifile[nleadingchar+11:nleadingchar+13]), 0, tzinfo=utc)
        TEMP_filebasetime = calendar.timegm(TEMP_filetime.timetuple())

        if TEMP_filebasetime >= start_basetime and TEMP_filebasetime <= end_basetime:

            rawdatafiles[filestep] = clouddata_path + ifile
            files_datestring[filestep] = ifile[nleadingchar:nleadingchar+4] + \
                                            ifile[nleadingchar+4:nleadingchar+6] + \
                                            ifile[nleadingchar+6:nleadingchar+8]
            files_timestring[filestep] = ifile[nleadingchar+9:nleadingchar+11] + \
                                            ifile[nleadingchar+11:nleadingchar+13]
            files_basetime[filestep] = np.copy(TEMP_filebasetime)
            filestep = filestep + 1
    
    # Remove extra rows
    rawdatafiles = rawdatafiles[0:filestep]
    files_datestring = files_datestring[0:filestep]
    files_timestring = files_timestring[0:filestep]
    files_basetime = files_basetime[:filestep]
    
    ##########################################################################
    # Process files
    # Load function
    from idcells_radar import idcell_csapr

    # Generate input lists
    idclouds_input = zip(rawdatafiles, files_datestring, files_timestring, files_basetime, \
                        repeat(datasource), repeat(datadescription), repeat(cloudid_version), \
                        repeat(tracking_outpath), repeat(startdate), repeat(enddate), \
                        repeat(pixel_radius), repeat(area_thresh), repeat(miss_thresh), repeat(mincellpix))
    
    ## Call function
    if run_parallel == 0:
        # Serial version
        for ifile in range(0, filestep):
            idcell_csapr(rawdatafiles[ifile], files_datestring[ifile], files_timestring[ifile], files_basetime[ifile], \
                            datasource, datadescription, cloudid_version, \
                            tracking_outpath, startdate, enddate, \
                            pixel_radius, area_thresh, miss_thresh, mincellpix)
    elif run_parallel == 1:
        # Parallel version
        if __name__ == '__main__':
            print('Identifying clouds')
            pool = Pool(nprocesses)
            pool.starmap(idcell_csapr, idclouds_input)
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
    allcloudidfiles = sorted(fnmatch.filter(os.listdir(tracking_outpath), cloudid_filebase +'*'))

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
            cloudidfiles_timestring[cloudidfilestep] = icloudidfile[nleadingchar+9:nleadingchar+11] + \
                                                        icloudidfile[nleadingchar+11:nleadingchar+13]
            cloudidfiles_datestring[cloudidfilestep] = icloudidfile[nleadingchar:nleadingchar+4] + \
                                                        icloudidfile[nleadingchar+4:nleadingchar+6] + \
                                                        icloudidfile[nleadingchar+6:nleadingchar+8] 
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
    from tracksingle_drift import trackclouds

    # Create draft variables that match number of reference cloudid files
    # Number of reference cloudid files (1 less than total cloudid files)
    ncloudidfiles = len(cloudidfiles_timestring)-1
    datetime_drift_match = np.empty(ncloudidfiles, dtype='<U13')
    xdrifts_match = np.zeros(ncloudidfiles, dtype=int)
    ydrifts_match = np.zeros(ncloudidfiles, dtype=int)

    # Test if driftfile is defined
    try:
        driftfile
    except NameError:
        print(f"Drift file is not defined. Regular tracksingle procedure is used.")
    else:
        print(f"Drift file used: {driftfile}")

        # Read the drift file
        ds_drift = xr.open_dataset(driftfile)
        bt_drift = ds_drift.basetime
        xdrifts = ds_drift.x.values
        ydrifts = ds_drift.y.values

        # Convert dateime64 objects to string array
        datetime_drift = bt_drift.dt.strftime("%Y%m%d_%H%M").values

#        # Read the drift file
#        drift_data = np.loadtxt(driftfile, usecols=(0, 1, 2, 3), dtype=str, delimiter=' ')
#        datestr = drift_data[:,0]
#        timestr = drift_data[:,1]
#        xdrifts_str = drift_data[:,2]
#        ydrifts_str = drift_data[:,3]
#
#        nt_drift = len(datestr)
#        bt_drift = np.full(nt_drift, -999, dtype=float)
#        xdrifts = np.full(nt_drift, -999, dtype=int)
#        ydrifts = np.full(nt_drift, -999, dtype=int)
#        datetime_drift = []
#
#        # Loop over each time/line
#        for itime in range(0, nt_drift):
#            iyear = datestr[itime].split('-')[0]
#            imonth = datestr[itime].split('-')[1]
#            iday = datestr[itime].split('-')[2]
#            ihour = timestr[itime].split('+')[0].split(':')[0]
#            iminute = timestr[itime].split('+')[0].split(':')[1]
#            isecond = timestr[itime].split('+')[0].split(':')[2]
#            datetime_drift.append(f'{iyear}{imonth}{iday}_{ihour}{iminute}')
#            # Calculate basetime
#            bt_drift[itime] = calendar.timegm(datetime.datetime(int(iyear), int(imonth), int(iday), int(ihour), int(iminute), int(isecond), tzinfo=pytz.UTC).timetuple())
#            # Convert string to int
#            xdrifts[itime] = int(float(xdrifts_str[itime][:-1]))  # Remove the , after the number
#            ydrifts[itime] = int(float(ydrifts_str[itime][:]))
#
#        # Convert list to array    
#        datetime_drift = np.array(datetime_drift)

        # Loop over each cloudid file time to find matching drfit data
        for itime in range(0, len(cloudidfiles_timestring)-1):
            cloudid_datetime = cloudidfiles_datestring[itime] + '_' + cloudidfiles_timestring[itime]
            idx = np.where(datetime_drift == cloudid_datetime)[0]
            if (len(idx) == 1):
                datetime_drift_match[itime] = datetime_drift[idx[0]]
                xdrifts_match[itime] = xdrifts[idx]
                ydrifts_match[itime] = ydrifts[idx]
    
    # import pdb; pdb.set_trace()

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

# Determine if single file tracking code ran. If not, set the version name and filename using names specified in the constants section
if run_tracksingle == 0:
    singletrack_filebase = 'track' + curr_track_version + '_'

# Call function
if run_gettracks == 1:
    # Load function
    from gettracks import gettracknumbers

    # Call function
    print('Getting track numbers')
    print('tracking_out:' + tracking_outpath)
    gettracknumbers(datasource, datadescription, tracking_outpath, stats_outpath, startdate, enddate, \
                    timegap, maxnclouds, cloudid_filebase, npxname, tracknumber_version, singletrack_filebase, \
                    keepsingletrack=keep_singlemergesplit, removestartendtracks=1)
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
    from trackstats_radar import trackstats_radar

    # Call satellite version of function
    print('Calculating cell statistics')
    trackstats_radar(datasource, datadescription, pixel_radius, geolimits, area_thresh, \
                    startdate, enddate, timegap, cloudid_filebase, tracking_outpath, stats_outpath, \
                    track_version, tracknumber_version, tracknumbers_filebase, lengthrange=lengthrange)
    trackstats_filebase = 'stats_tracknumbers' + tracknumber_version + '_'

##############################################################
# # Identify cell candidates

# # Determine if final statistics portion ran. If not, set the version name and filename using those specified in the constants section
# if run_finalstats == 0:
#     trackstats_filebase = 'stats_tracknumbers' + curr_tracknumbers_version + '_'

# if run_identifycell == 1:
#     print('Identifying Cells')

#     # Load function
#     from identifycell import identifycell_LES_xarray

#     # Call satellite version of function
#     identifycell_LES_xarray(trackstats_filebase, stats_outpath, startdate, enddate, datatimeresolution, geolimits, maincloud_duration, merge_duration, split_duration, lengthrange[1])
#     cellstats_filebase =  'cell_tracks_'

############################################################
# Create pixel files with cell tracks

# # Determine if the mcs identification and statistic generation step ran. If not, set the filename using those specified in the constants section
# if run_identifycell == 0:
#     # cellstats_filebase =  'cell_tracks_'
#     cellstats_filebase = 'stats_tracknumbers' + tracknumber_version

# Determine if final statistics portion ran. If not, set the version name and filename using those specified in the constants section
if run_finalstats == 0:
    trackstats_filebase = 'stats_tracknumbers' + curr_tracknumbers_version + '_'

if run_labelcell == 1:
    print('Identifying which pixel level maps to generate for the cell tracks')

    ###########################################################
    # Identify files to process
    # if run_tracksingle == 0:
    ################################################################
    # Create labelcell output directory
    os.makedirs(celltracking_outpath, exist_ok=True)

    # Isolate all possible files
    allcloudidfiles = sorted(fnmatch.filter(os.listdir(tracking_outpath), cloudid_filebase +'*'))

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

    # Load function 
    from mapcell_radar import mapcell_radar

    # Generate input list
    # list_cellstat_filebase = [cellstats_filebase]*(cloudidfilestep-1)
    # list_trackstat_filebase = [trackstats_filebase]*(cloudidfilestep-1)
    # list_celltracking_path = [celltracking_outpath]*(cloudidfilestep-1)
    # list_stats_path = [stats_outpath]*(cloudidfilestep-1)
    # list_cloudid_filebase = [cloudid_filebase]*(cloudidfilestep-1)
    # list_absolutelwp_threshs = np.ones((cloudidfilestep-1, 2))*absolutelwp_threshs
    # list_startdate = [startdate]*(cloudidfilestep-1)
    # list_enddate = [enddate]*(cloudidfilestep-1)
    # list_showalltracks = [show_alltracks]*(cloudidfilestep-1)

    # cellmap_input = list(zip(cloudidfiles, cloudidfiles_basetime, list_cellstat_filebase, list_trackstat_filebase, \
    #                     list_celltracking_path, list_stats_path, list_startdate, list_enddate, list_showalltracks))
    cellmap_input = zip(cloudidfiles, cloudidfiles_basetime, repeat(stats_outpath), repeat(trackstats_filebase), \
                        repeat(startdate), repeat(enddate), repeat(celltracking_outpath), repeat(celltracking_filebase))

    ## Call function
    if run_parallel == 0:
        # Call function
        # for iunique in range(0, cloudidfilestep-1):
        for iunique in range(0, cloudidfilestep):
            # mapcell_radar(cellmap_input[iunique])
            mapcell_radar(cloudidfiles[iunique], cloudidfiles_basetime[iunique], stats_outpath, trackstats_filebase, \
                        startdate, enddate, celltracking_outpath, celltracking_filebase)
    elif run_parallel == 1:
        if __name__ == '__main__':
            print('Creating maps of tracked cells')
            pool = Pool(nprocesses)
            pool.starmap(mapcell_radar, cellmap_input)
            pool.close()
            pool.join()
    else:
        sys.ext('Valid parallelization flag not provided')
