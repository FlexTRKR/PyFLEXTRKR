#!/usr/bin/env python

""" Storm tracking code via FFT Correlation
"""

from __future__ import division, print_function

import glob
import sys
import os
import json

from datetime import datetime, timedelta

from multiprocessing import Pool

import numpy as np
from netCDF4 import Dataset
from scipy.signal import fftconvolve

#NUM_PROCESSES=60
NUM_PROCESSES=60

# This seems to alternate between cloudtracknumber and pftracknumber
def get_pixel_size_of_clouds(dataset, total_tracks, track_variable='pcptracknumber'):

    """ Calculate pixel size of each identified cloud in the file.

    Parameters:
    -----------
    dataset: Dataset
        netcdf Dataset
    track_variable: string
        variable that contains pixel level values.

    Returns:
    --------
    counts: array_like
        Pixel size of every cloud in file. Cloud 0 is stored at 0.
    """
    storm_sizes = np.zeros(total_tracks+1)

    track, counts = np.unique(dataset.variables[track_variable][:], return_counts=True)
    storm_sizes[track] = counts
    storm_sizes[0] = 0
    return storm_sizes


def movement_of_storm_fft(filename_1,  
                            track_variable='pcptracknumber', min_size_threshold=20,
                            optimize_sub_array=True, storm_buffer=None,
                            tracked_field='precipitation'):

    """ Calculate Movement of first labeled storm

    Parameters
    ----------
    filename_1: str
        Filename of source storm
    filename_2: str
        Filename of target storm
    """
    print("Starting Storm File: %s" % filename_1[0])
    sys.stdout.flush()

    dset1 = Dataset(filename_1[0], 'r')
    dset2 = Dataset(filename_1[1], 'r')
    # base_field = processing_results['track_variable']
    base_field = track_variable
    # total_tracks = processing_results['total_tracks']
    total_tracks = filename_1[2]
    y_lag = np.zeros(total_tracks)
    x_lag = np.zeros(total_tracks)

    min_cloud_size = np.minimum(get_pixel_size_of_clouds(dset1, total_tracks),
                            get_pixel_size_of_clouds(dset2, total_tracks)) #I think new stats files have this already. 

    conv_number_1 = dset1.variables[base_field][:].squeeze()
    conv_number_2 = dset2.variables[base_field][:].squeeze()

    conv_dbz_1 = dset1.variables[tracked_field][:].squeeze()
    conv_dbz_2 = dset2.variables[tracked_field][:].squeeze()
    for track_number in np.arange(0, total_tracks):
        if min_cloud_size[track_number] < min_size_threshold:
            y_lag[track_number] = np.nan
            x_lag[track_number] = np.nan
        else:
            if optimize_sub_array:
                # Calculate size of bounding box
                xmin, xmax, ymin, ymax = get_bounding_box_for_fft(conv_number_1, conv_number_2, track_number)
                masked_dbz_1 = conv_dbz_1[xmin:xmax, ymin:ymax].copy()
                masked_dbz_2 = conv_dbz_2[xmin:xmax, ymin:ymax].copy()

                masked_dbz_1[conv_number_1[xmin:xmax, ymin:ymax] != track_number] = 0
                masked_dbz_1[np.isnan(masked_dbz_1)] = 0

                masked_dbz_2[conv_number_2[xmin:xmax, ymin:ymax] != track_number] = 0
                masked_dbz_2[np.isnan(masked_dbz_2)] = 0
            else:
                masked_dbz_1 = conv_dbz_1.copy()
                masked_dbz_2 = conv_dbz_2.copy()

                masked_dbz_1[conv_number_1 != track_number] = 0
                masked_dbz_1[np.isnan(masked_dbz_1)] = 0

                masked_dbz_2[conv_number_2 != track_number] = 0
                masked_dbz_2[np.isnan(masked_dbz_2)] = 0

            result = fftconvolve(masked_dbz_1, masked_dbz_2[::-1, ::-1], mode='same')
            y_step, x_step = np.unravel_index(np.argmax(result), result.shape)
            y_dim, x_dim = np.shape(masked_dbz_1)

            y_lag[track_number] = np.floor(y_dim/2) - y_step
            x_lag[track_number] = np.floor(x_dim/2) - x_step

    time_lag = dset2.variables['base_time'][0] - dset1.variables['base_time'][0]
    base_time = dset1.variables['base_time'][0].copy()
    dset1.close()
    dset2.close()
    return y_lag, x_lag, time_lag, base_time

def get_bounding_box_for_fft(in1, in2, track_number):
    ''' Given two masks and a track number, calculate the maximum bounding box to fit both
    
    Parameters
    ----------
    in1: np.array
        first mask array
    in2: np.array
        secondm ask array
    '''

    a = in1 == track_number
    b = in2 == track_number

    rows = np.any(a, axis=1)
    cols = np.any(a, axis=0)
    rmin1, rmax1 = np.where(rows)[0][[0, -1]]
    cmin1, cmax1 = np.where(cols)[0][[0, -1]]

    rows = np.any(b, axis=1)
    cols = np.any(b, axis=0)
    rmin2, rmax2 = np.where(rows)[0][[0, -1]]
    cmin2, cmax2 = np.where(cols)[0][[0, -1]]

    return min(rmin1, rmin2), max(rmax1, rmax2), min(cmin1, cmin2), max(cmax1, cmax2)

def offset_to_speed(x, y, time_lag):
    """ Return normalized speed assuming uniform grid.
    """
    mag_movement = np.sqrt(x**2 + y**2)
    mag_dir = np.arctan2(x, y)*180/np.pi
    mag_movement_mps = np.array([mag_movement_i / (time_lag) * 1000.0 for mag_movement_i in mag_movement.T]).T
    return mag_movement, mag_dir, mag_movement_mps

def track_case(processing_results):
    """ Driver script to process a set of labeled tracks

        Parameters
        ----------
        case_configuration: dict
            Dictionary containing configuration for case to be run

    """


    pool = Pool(processes=NUM_PROCESSES)
    filelist = []
    if type(processing_results['case']) is list:
        for case in processing_results['case']:
            for path in processing_results['data_path']:
                filelist.extend(glob.glob(path % case))
                print(path % case)
    else:
        filelist.extend(glob.glob(processing_results['data_path'] % processing_results['case']))
    filelist.sort()

    lag = processing_results['lag']

    processing_results['num_files'] = len(filelist)
    print("Found %d files" % processing_results['num_files'])

    # Open stats file to get maximum number of storms to track.
    stats_dset = Dataset(processing_results['stats_file_path'], 'a')
    #total_tracks = len(stats_dset.dimensions['ntracks'])
    total_tracks = len(stats_dset.dimensions['tracks'])
    max_length = len(stats_dset.dimensions['times'])
    # max_length = len(stats_dset.dimensions['nmaxlength'])

    # Open a file for testing

    processing_results['total_tracks'] = total_tracks
    print("Found a total of %d storms to track." % processing_results['total_tracks'])

    filename_packet = zip(filelist[0:-lag], filelist[lag:], np.repeat(total_tracks, processing_results['num_files']))

    if processing_results['from_results_file']:
        res = np.load(processing_results['checkpoint_file'])
        r = res['r']
        r_mps = res['r_mps']
        theta = res['theta']
        storm_x = res['storm_x']
        storm_y = res['storm_y']
        base_time = res['base_time']
        time_lag = res['time_lag']

    else:
        # result = pool.map(movement_lambda, list(filename_packet)[0:40])
        result = pool.map(movement_of_storm_fft, filename_packet)
        storm_y, storm_x, time_lag, base_time= zip(*result)

        storm_y = np.array(storm_y)
        storm_x = np.array(storm_x)

        (r, theta, r_mps) = offset_to_speed(storm_y, storm_x, time_lag) # These files are reversed axis
        np.savez(processing_results['checkpoint_file'], r=r, theta=theta, r_mps=r_mps, storm_y = storm_y, storm_x = storm_x, time_lag=time_lag, base_time=base_time)


    #Time to write these out to the netcdf file.
    #trackname = 'ntracks'
    trackname = 'tracks'

    #timename = 'nmaxlength'
    timename = 'times'
    try:
        nc_r = stats_dset.createVariable('movement_r', 'f4', (trackname, timename))
        nc_r.units = "km"
        nc_r.long_name = "Movement along angle movement_theta relative to lag."
        nc_r.comments = "This is the total movement along the angle theta between lag estimates. Use movement_r_meters_per_second for a normalized parameter."

        nc_r_mps = stats_dset.createVariable('movement_r_meters_per_second', 'f4', (trackname, timename))
        nc_r_mps.units = "m/s"
        nc_r_mps.long_name = "Movement along unit vector given by angle movement_theta"

        nc_theta = stats_dset.createVariable('movement_theta', 'f4', (trackname, timename))
        nc_theta.units = 'degrees'
        nc_theta.long_name = "Direction of movement"

        nc_storm_x = stats_dset.createVariable('movement_storm_x', 'f4', (trackname, timename))
        nc_storm_x.units = "km"
        nc_storm_x.long_name = "East-West component of movement."

        nc_storm_y = stats_dset.createVariable('movement_storm_y', 'f4', (trackname, timename))
        nc_storm_y.units = "km"
        nc_storm_y.long_name = "North-South component of movement"

        nc_storm_time_lag = stats_dset.createVariable('movement_time_lag', 'f4', (trackname, timename))
        nc_storm_time_lag.units = "s"
        nc_storm_time_lag.long_name = "Time Lag between consecutive advection estimates."
        nc_storm_time_lag.comments = "This is the lag between files multiplied by order of the lag estimator."

        stats_dset.movement_estimate_lag_order = lag
    except Exception as e:
        nc_r = stats_dset.variables['movement_r']
        nc_r_mps = stats_dset.variables['movement_r_meters_per_second']
        nc_theta = stats_dset.variables['movement_theta']
        nc_storm_x = stats_dset.variables['movement_storm_x']
        nc_storm_y = stats_dset.variables['movement_storm_y']
        nc_storm_time_lag = stats_dset.variables['movement_time_lag']

    nc_r[:] = np.nan
    nc_r_mps[:] = np.nan
    nc_theta[:] = np.nan
    nc_storm_x[:] = np.nan
    nc_storm_y[:] = np.nan
    nc_storm_time_lag = np.nan

    # We need to take each storm and find the start date, and then line it back up.
    for track_number in np.arange(0, total_tracks-1):
        start_time = stats_dset.variables['base_time'][track_number, 0]
        # if processing_results['case_name'] == 'obs_2011':
        #     start_time = datetime.strptime(start_time, '%Y-%m-%d_%H:%M:') # Obs timestrings
        # else:
        #     start_time = datetime.strptime(start_time, '%Y-%m-%d_%H:%M:%S') # Model timestrings

        # base_time_diff = np.abs(np.array(base_time) - (start_time - datetime(1970, 1, 1)).total_seconds())

        base_time_diff = np.abs(np.array(base_time) - start_time)
        start_idx = np.nanargmin(base_time_diff)
        valid_indices = np.where(np.isfinite(r_mps[:, track_number+1]))[0]

        if len(valid_indices) < 1:
            continue
        else:
            end_idx = valid_indices[-1]


        duration = np.min([max_length, end_idx - start_idx])
        #duration = np.min([60, end_idx - start_idx])


        # Now we know where Zhe's file starts, let's set find which time entry matches with this.
        nc_r[track_number, 0:duration] = processing_results['y_step_size']*r[start_idx:start_idx+duration, 1+track_number]/lag
        nc_r_mps[track_number, 0:duration] = processing_results['y_step_size']*r_mps[start_idx:start_idx+duration, 1+track_number]
        nc_theta[track_number, 0:duration] = theta[start_idx:start_idx+duration, 1+track_number]
        nc_storm_x[track_number, 0:duration] = processing_results['y_step_size']/lag*storm_x[start_idx:start_idx+duration, 1+track_number]
        nc_storm_y[track_number, 0:duration] = processing_results['y_step_size']/lag*storm_y[start_idx:start_idx+duration, 1+track_number]

    #print(nc_storm_time_lag)
    #print(lag)

    stats_dset.close()

def run_template_case(template, processing_results):
    """ Run a templated case
    """
    template_4 = "%s%s%s" % (template[0], template[1], template[2])
    processing_results['case_name'] = processing_results['case_name'] % template_4
    processing_results['stats_file_path'] = processing_results['stats_file_path'] % (template[0], template[1], template[2], template_4, template_4)
    processing_results['data_path'] = processing_results['data_path'] %(template[0], template[1], template[2], '%s') 
    processing_results['checkpoint_file'] = processing_results['checkpoint_file'] % (template[0], template[1], template[2])
    # print(processing_results)

    track_case(processing_results)

def run_case_list(processing_results):
    """ Run entire caselist
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    case_list = open(dir_path + '/case_list.txt')

    for case in case_list:
        if case[0] == '#':
            continue
        radar, year, month, day, hour_start, hour_end = case.rstrip('\n').split(',')
        year = "%04d" % int(year)
        month = "%02d" % int(month)
        day = "%02d" % int(day)
        # print(year, month, day)
        template = (year, month, day)
        try:
            run_template_case(template, processing_results.copy())
        except Exception as e:
            print("Failure on case:", template)
            print(processing_results)

def run_gridrad_case(template, processing_results, gridrad_case):
    """ Run a gridrad case
    """
    # First we enumerate the statistics files. We get the location from the template.
    stats_filelist = glob.glob(processing_results['stats_file_path']+'robust*.nc')
    stats_filelist.sort()

    # Then we pick our gridrad_case from the list. 
    stats_filename = stats_filelist[gridrad_case]
    dname, fname = os.path.split(stats_filename)
    processing_results['stats_file_path'] = stats_filename
    print(processing_results['stats_file_path'])

    data_parts = fname.split('_')
    data_addon_path = data_parts[-2] + '_' + data_parts[-1][0:-3]
    processing_results['data_path'][0] = processing_results['data_path'][0] +\
                                    data_addon_path + '/%s.nc'


    processing_results['checkpoint_file'] = processing_results['stats_file_path'][0:-3] +'.npz'


    track_case(processing_results)

def run_sipam_case(template, processing_results, gridrad_case):
    """ Run a gridrad case
    """
    # First we enumerate the statistics files. We get the location from the template.
    stats_filelist = glob.glob(processing_results['stats_file_path']+'*.nc')
    stats_filelist.sort()

    # Then we pick our gridrad_case from the list. 
    stats_filename = stats_filelist[gridrad_case]
    dname, fname = os.path.split(stats_filename)
    processing_results['stats_file_path'] = stats_filename
    print(processing_results['stats_file_path'])

    data_parts = fname.split('_')
    year = data_parts[2][0:4]
    month = data_parts[2][4:6]
    day = data_parts[2][6:8]
    data_addon_path = year + '/' + month + '/' + day + '/'

    processing_results['data_path'][0] = processing_results['data_path'][0] + data_addon_path + "tracking/%s*.nc" 
    processing_results['checkpoint_file'] = processing_results['stats_file_path'][0:-3] +'.npz'

    track_case(processing_results)



if __name__ == '__main__':
    use_template = False
    use_case_list = False
    use_gridrad = False
    use_sipam = False

    template = ("2014", "06", "03")
    
    if len(sys.argv) == 2:
        case_name = sys.argv[1]
    elif len(sys.argv) == 3: # For now this is the gridrad case
        if sys.argv[1] == 'gridrad_template':
            use_gridrad = True
            gridrad_case = int(sys.argv[2])
            case_name = 'gridrad_template'
        elif sys.argv[1] == 'sipam':
            use_sipam = True
            print("SIPAM Enabled\n*Bon Dia!")
            sipam_case = int(sys.argv[2])
            case_name = 'sipam_template'
        elif sys.argv[1] == 'gpm_template':
            use_gridrad = True
            gridrad_case = int(sys.argv[2])
            case_name = 'gpm_template'
        elif sys.argv[1] == 'cacti_wrf':
            use_gridrad = True
            gridrad_case = int(sys.argv[2])
            case_name = 'cacti_wrf'
    elif use_template:
        case_name="template"
    else:
        case_name = "KVNX_20120615"
    
    print("Case:" + case_name)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path + "/config_list.json") as json_file:
        case_config_list = json.load(json_file)

    processing_results = case_config_list[case_name]

    optimize_sub_array = processing_results.get('optimize_sub_array', True)
    os.environ['optimize_sub_array'] = str(int(optimize_sub_array))

    if "NUM_PROCESSES" in processing_results:
        NUM_PROCESSES = int(processing_results['NUM_PROCESSES'])

    if use_case_list:
        run_case_list(processing_results)
    elif use_template:
        run_template_case(template, processing_results)
    elif use_gridrad:
        run_gridrad_case(template, processing_results, gridrad_case)
    elif use_sipam:
        run_sipam_case(template, processing_results, sipam_case)
    else:
        track_case(processing_results)
