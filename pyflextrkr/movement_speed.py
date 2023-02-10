from __future__ import division, print_function
import sys
import os
import time
import logging
import numpy as np
from netCDF4 import Dataset
import xarray as xr
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
import dask
from dask.distributed import wait
from pyflextrkr.ft_utilities import subset_files_timerange

def movement_speed(
        config,
        trackstats_filebase=None,
        trackstats_outfilebase=None,
        pixelpath_basename=None,
        pixeltracking_filebase=None,
):
    """
    Calculate movement speed using pixel level tracked feature.

    Args:
        config: dictionary
            Dictionary containing config parameters.
        trackstats_filebase: string, default=None
            Input track statistics file basename.
        trackstats_outfilebase: string, default=None
            Output track statistics file basename.
        pixelpath_basename: string, default=None,
            Pixel-level files base path name.
            If None, pixeltracking_outpath defaults to config["pixeltracking_outpath"].
            Otherwise, pixeltracking_outpath is constructed using the config info:
                f'{config["root_path"]}/{pixelpath_basename}/{config["startdate"]}_{config["enddate"]}/'
        pixeltracking_filebase=None, default=None
            Pixel-level file basename.
            If None, defaults to config["pixeltracking_filebase"].

    Returns:
        statistics_outfile: string
            MCS track statistics file name.
    """

    stats_outpath = config["stats_outpath"]
    startdate = config["startdate"]
    enddate = config["enddate"]
    start_basetime = config["start_basetime"]
    end_basetime = config["end_basetime"]
    tracks_dimname = config["tracks_dimname"]
    times_dimname = config["times_dimname"]
    run_parallel = config["run_parallel"]
    feature_type = config["feature_type"]
    pixel_radius = config["pixel_radius"]
    lag = config["lag_for_speed"]
    max_speed_thresh = config["max_speed_thresh"]

    logger = logging.getLogger(__name__)
    logger.info('Calculating movement speed using pixel-level tracked feature')

    # Set trackstats file basenames
    if trackstats_filebase is None:
        trackstats_filebase = config["mcsrobust_filebase"]
    if trackstats_outfilebase is None:
        trackstats_outfilebase = config["mcsfinal_filebase"]
    # Set pixel file path and basename
    if pixelpath_basename is None:
        pixeltracking_outpath = config["pixeltracking_outpath"]
    else:
        pixeltracking_outpath = f'{config["root_path"]}/{pixelpath_basename}/{config["startdate"]}_{config["enddate"]}/'
    if pixeltracking_filebase is None:
        pixeltracking_filebase = config["pixeltracking_filebase"]

    # Stats file name
    if 'tb_pf' in feature_type:
        # Robust MCS track stats filename
        statistics_file = f"{stats_outpath}{trackstats_filebase}{startdate}_{enddate}.nc"
        # Output MCS track stats filename
        statistics_outfile = f"{stats_outpath}{trackstats_outfilebase}{startdate}_{enddate}.nc"

    # Identify pixel files to process
    filelist, \
    files_basetime, \
    files_datestring, \
    files_timestring = subset_files_timerange(pixeltracking_outpath,
                                              pixeltracking_filebase,
                                              start_basetime,
                                              end_basetime)
    nfiles = len(filelist)
    logger.info(f"Total number of files to process: {nfiles}")

    # Open stats file to get maximum number of storms to track.
    ds_stats = xr.open_dataset(statistics_file,
                               mask_and_scale=False,
                               decode_times=False)
    ntracks = ds_stats.dims[tracks_dimname]
    ntimes = ds_stats.dims[times_dimname]
    tracks_coord = ds_stats.coords[tracks_dimname].data
    times_coord = ds_stats.coords[times_dimname].data
    stats_basetime = ds_stats.variables['base_time'].values
    ds_stats.close()

    # Make file pairs
    filepairs = list(zip(filelist[0:-lag], filelist[lag::]))


    results = []
    # Serial
    if run_parallel == 0:
        for ifile in range(0, nfiles-1):
            result = movement_of_feature_fft(
                filepairs[ifile], ntracks,
                config,
            )
            results.append(result)
        final_result = results
    # Parallel
    elif run_parallel >= 1:
        for ifile in range(0, nfiles-1):
            result = dask.delayed(movement_of_feature_fft)(
                filepairs[ifile], ntracks,
                config,
            )
            results.append(result)
        final_result = dask.compute(*results)
        wait(final_result)
    else:
        sys.exit('Valid parallelization flag not provided')

    move_y, move_x, time_lag, base_time = zip(*final_result)
    move_y = np.array(move_y)
    move_x = np.array(move_x)
    base_time = np.asarray(base_time, dtype=float)

    # Compute movement speed, direction
    (r_mag, r_dir, r_speed) = offset_to_speed(move_x, move_y, time_lag)

    # Convert distance to physical units
    # Movement magnitude [km]
    movement_mag = r_mag * pixel_radius / lag
    movement_x = move_x * pixel_radius / lag
    movement_y = move_y * pixel_radius / lag
    # Movement speed [m/s]
    movement_speed = r_speed * pixel_radius * 1000.
    # Movement direction
    # theta is not the same with traditional direction definition in meteorology
    # TODO: convert the direction to 0 deg = North
    movement_dir = r_dir

    # Put the movement variables in an Xarray Dataset
    # consistent with track statistics
    ds_vars = define_movement_dataset(base_time, lag, movement_dir, movement_mag, movement_speed, movement_x,
                                      movement_y, ntimes, ntracks, stats_basetime, times_coord, times_dimname,
                                      tracks_coord, tracks_dimname)

    # Run filter to interpolate over high movement speeds
    ds_vars_filt = filter_interp_speed(ds_vars, config)

    # Merge Datasets
    dsout = xr.merge([ds_stats, ds_vars_filt], compat="override", combine_attrs="no_conflicts")

    # Update global attributes
    dsout.attrs["Created_on"] = time.ctime(time.time())
    dsout.attrs["max_speed_thresh"] = max_speed_thresh

    ###########################################################################
    # Write statistics to netcdf file

    # Delete file if it already exists
    if os.path.isfile(statistics_outfile):
        os.remove(statistics_outfile)

    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in dsout.data_vars}

    # Write to netcdf file
    dsout.to_netcdf(path=statistics_outfile, mode="w",
                    format="NETCDF4", unlimited_dims=tracks_dimname, encoding=encoding)
    logger.info(f"{statistics_outfile}")

    return statistics_outfile



def movement_of_feature_fft(
        filepairs,
        ntracks,
        config,
        optimize_sub_array=True,
):
    """
    Calculate movement of tracked features.

    Args:
        filepairs: tuple
            Pairs of pixel file names.
        ntracks: int
            Number of tracks.
        config: dictionary
            Dictionary containing config parameters.
        optimize_sub_array: boolean
            Flag to subset each tracked feature from the full image.

    Returns:
        y_lag: np.array
            Movement magnitude in y-direction.
        x_lag: np.array
            Movement magnitude in x-direction.
        time_lag: float
            Time difference between two pixel files.
        base_time: float
            Base time for the first pixel file.
    """

    tracknumber = config["track_number_for_speed"]
    track_field = config["track_field_for_speed"]
    min_size_thresh = config["min_size_thresh_for_speed"]
    # storm_buffer = None

    logger = logging.getLogger(__name__)
    logger.debug("Starting Storm File: %s" % filepairs[0])
    sys.stdout.flush()

    dset1 = Dataset(filepairs[0], 'r')
    dset2 = Dataset(filepairs[1], 'r')
    y_lag = np.zeros(ntracks)
    x_lag = np.zeros(ntracks)

    # Get minimum size of feature from pixel files
    min_cloud_size = np.minimum(get_pixel_size_of_clouds(dset1, ntracks, tracknumber),
                                get_pixel_size_of_clouds(dset2, ntracks, tracknumber))
    # Get tracknumber and field values
    tracknumber_1 = dset1.variables[tracknumber][:].squeeze()
    tracknumber_2 = dset2.variables[tracknumber][:].squeeze()
    field_1 = dset1.variables[track_field][:].squeeze()
    field_2 = dset2.variables[track_field][:].squeeze()

    # Loop over each track number
    for track_number in np.arange(0, ntracks):
        if min_cloud_size[track_number] < min_size_thresh:
            y_lag[track_number] = np.nan
            x_lag[track_number] = np.nan
        else:
            if optimize_sub_array:
                # Calculate size of bounding box
                ymin, ymax, xmin, xmax = get_bounding_box_for_fft(tracknumber_1, tracknumber_2, track_number)
                masked_field_1 = field_1[ymin:ymax, xmin:xmax].copy()
                masked_field_2 = field_2[ymin:ymax, xmin:xmax].copy()

                masked_field_1[tracknumber_1[ymin:ymax, xmin:xmax] != track_number] = 0
                masked_field_1[np.isnan(masked_field_1)] = 0

                masked_field_2[tracknumber_2[ymin:ymax, xmin:xmax] != track_number] = 0
                masked_field_2[np.isnan(masked_field_2)] = 0
            else:
                masked_field_1 = field_1.copy()
                masked_field_2 = field_2.copy()

                masked_field_1[tracknumber_1 != track_number] = 0
                masked_field_1[np.isnan(masked_field_1)] = 0

                masked_field_2[tracknumber_2 != track_number] = 0
                masked_field_2[np.isnan(masked_field_2)] = 0

            # Flip the second image, do an FFT convolution
            result = fftconvolve(masked_field_1, masked_field_2[::-1, ::-1], mode='same')
            # Get the index with max value (highest correlation)
            # then reshape it to 2D to get x, y index
            y_step, x_step = np.unravel_index(np.argmax(result), result.shape)
            y_dim, x_dim = np.shape(masked_field_1)
            # Get the relative position from the center of the image
            # This is the movement in x, y direction
            y_lag[track_number] = np.floor(y_dim/2) - y_step
            x_lag[track_number] = np.floor(x_dim/2) - x_step

    # Get time difference between the file pair
    time_lag = dset2.variables['time'][0] - dset1.variables['time'][0]
    base_time = dset1.variables['time'][0].copy()

    dset1.close()
    dset2.close()
    return y_lag, x_lag, time_lag, base_time


def get_pixel_size_of_clouds(
        dataset,
        ntracks,
        tracknumber,
):
    """
    Calculate pixel size of each identified cloud in the file.

    Args:
        dataset: Dataset
            netcdf Dataset
        tracknumber: string
            variable that contains pixel level values.

    Returns:
        counts: array_like
            Pixel size of every cloud in file. Cloud 0 is stored at 0.
    """
    storm_sizes = np.zeros(ntracks + 1)

    track, counts = np.unique(dataset.variables[tracknumber][:], return_counts=True)
    storm_sizes[track] = counts
    storm_sizes[0] = 0
    return storm_sizes


def get_bounding_box_for_fft(in1, in2, track_number):
    """
    Given two masks and a track number, calculate the maximum bounding box to fit both

    Args:
        in1: np.array
            First mask array
        in2: np.array
            Second mask array
        track_number: int
            Track number for masking.

    Returns:
        ymin, ymax, xmin, xmax: int
            Bounding box x, y indices.
    """

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

    ymin = min(rmin1, rmin2)
    ymax = max(rmax1, rmax2)
    xmin = min(cmin1, cmin2)
    xmax = max(cmax1, cmax2)
    return ymin, ymax, xmin, xmax

def offset_to_speed(x, y, time_lag):
    """
    Return normalized speed assuming uniform grid.

    Args:
        x: np.array
            Movement in x-direction.
        y: np.array
            Movement in y-direction.
        time_lag: np.array
            Time lag for each movement.

    Returns:
        r_mag: np.array
            Movement magnitude.
        r_dir: np.array
            Movement direction.
        r_speed: np.array
            Movement speed.
    """
    # Movement in grid point units
    r_mag = np.sqrt(x**2 + y**2)
    # Movement direction
    r_dir = np.arctan2(y, x)*180/np.pi
    # Movement speed [n_grid / second]
    r_speed = np.array([r_mag_i / (time_lag) for r_mag_i in r_mag.T]).T
    return r_mag, r_dir, r_speed

def define_movement_dataset(
        base_time,
        lag,
        movement_dir,
        movement_mag,
        movement_speed,
        movement_x,
        movement_y,
        ntimes,
        ntracks,
        stats_basetime,
        times_coord,
        times_dimname,
        tracks_coord,
        tracks_dimname,
):
    # Create arrays to match track stats structure
    fillval_f = np.nan
    tracks_movement_mag = np.full((ntracks, ntimes), fillval_f, dtype=np.float32)
    tracks_movement_speed = np.full((ntracks, ntimes), fillval_f, dtype=np.float32)
    tracks_movement_dir = np.full((ntracks, ntimes), fillval_f, dtype=np.float32)
    tracks_movement_x = np.full((ntracks, ntimes), fillval_f, dtype=np.float32)
    tracks_movement_y = np.full((ntracks, ntimes), fillval_f, dtype=np.float32)
    # Loop over each track to align the data
    for track_number in np.arange(0, ntracks - 1):
        # Find matching track start base_time
        start_time = stats_basetime[track_number, 0]
        # base_time_diff = np.abs(base_time - np.array(start_time))
        base_time_diff = np.abs(base_time - start_time)
        start_idx = np.nanargmin(base_time_diff)
        # Find valid movement value indices
        valid_indices = np.where(np.isfinite(movement_speed[:, track_number + 1]))[0]

        if len(valid_indices) < 1:
            continue
        else:
            end_idx = valid_indices[-1]

        duration = np.min([ntimes, end_idx - start_idx + 1])

        tracks_movement_mag[track_number, 0:duration] = \
            movement_mag[start_idx:start_idx + duration, track_number + 1]
        tracks_movement_speed[track_number, 0:duration] = \
            movement_speed[start_idx:start_idx + duration, track_number + 1]
        tracks_movement_dir[track_number, 0:duration] = \
            movement_dir[start_idx:start_idx + duration, track_number + 1]
        tracks_movement_x[track_number, 0:duration] = \
            movement_x[start_idx:start_idx + duration, track_number + 1]
        tracks_movement_y[track_number, 0:duration] = \
            movement_y[start_idx:start_idx + duration, track_number + 1]

    # Define new variables dictionary
    var_dict = {
        "movement_distance": tracks_movement_mag,
        "movement_speed": tracks_movement_speed,
        "movement_theta": tracks_movement_dir,
        "movement_distance_x": tracks_movement_x,
        "movement_distance_y": tracks_movement_y,
    }
    var_attrs = {
        "movement_distance": {
            "long_name": "Movement distance along angle movement_theta",
            "units": "km",
            "_FillValue": fillval_f,
            "lag": lag,
            "comments": "This is the total movement along the angle theta between lag estimates.",
        },
        "movement_speed": {
            "long_name": "Movement speed along angle movement_theta",
            "units": "m/s",
            "_FillValue": fillval_f,
        },
        "movement_theta": {
            "long_name": "Movement direction",
            "units": "degrees",
            "_FillValue": fillval_f,
            # "comments": "This is the total movement along the angle theta between lag estimates.",
        },
        "movement_distance_x": {
            "long_name": "East-West component of movement distance",
            "units": "km",
            "_FillValue": fillval_f,
        },
        "movement_distance_y": {
            "long_name": "North-South component of movement distance",
            "units": "km",
            "_FillValue": fillval_f,
        },
    }
    # Define output variable dictionary
    varlist = {}
    for key, value in var_dict.items():
        if value.ndim == 2:
            varlist[key] = ([tracks_dimname, times_dimname], value, var_attrs[key])
    # Define coordinate list
    coordlist = {
        tracks_dimname: ([tracks_dimname], tracks_coord),
        times_dimname: ([times_dimname], times_coord),
    }
    # Define Dataset
    ds_vars = xr.Dataset(varlist, coords=coordlist)
    return ds_vars

def filter_interp_speed(dset, config):
    """
    Filter and interpolate high values of movement_speed.

    Args:
        dset: Xarray Dataset
            Dataset containing movement variables.
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        dset: Xarray Dataset
            Update Dataset with filtered values.
    """

    tracks_dimname = config["tracks_dimname"]
    times_dimname = config["times_dimname"]
    max_speed_thresh = config["max_speed_thresh"]

    logger = logging.getLogger(__name__)

    move_r = dset['movement_speed']
    mask_r = move_r < max_speed_thresh
    mask_nan = np.logical_not(np.isnan(move_r))
    total_mask = np.logical_and(mask_r, mask_nan)

    fillval_f = np.nan
    m_mag = dset['movement_distance'].values
    m_speed = dset['movement_speed'].values
    m_theta = dset['movement_theta'].values
    m_x = dset['movement_distance_x'].values
    m_y = dset['movement_distance_y'].values
    m_mag_attrs = dset['movement_distance'].attrs
    m_speed_attrs = dset['movement_speed'].attrs
    m_theta_attrs = dset['movement_theta'].attrs
    m_x_attrs = dset['movement_distance_x'].attrs
    m_y_attrs = dset['movement_distance_y'].attrs

    for track in np.arange(0, len(dset[tracks_dimname])):
        if np.count_nonzero(dset['movement_speed'][track] < max_speed_thresh) < 3:
            logger.debug('Not enough values in Track %d' % track)
            m_mag[track] = fillval_f
            m_speed[track] = fillval_f
            m_theta[track] = fillval_f
            m_x[track] = fillval_f
            m_y[track] = fillval_f
        else:
            x = dset[times_dimname][total_mask[track]]
            xm = dset[times_dimname][mask_nan[track]]
            spd = dset['movement_speed'][track][total_mask[track]]
            theta = dset['movement_theta'][track][total_mask[track]]
            mag = dset['movement_distance'][track][total_mask[track]]
            intp_r = interp1d(x, spd, kind='quadratic', fill_value=fillval_f, bounds_error=False)
            intp_theta = interp1d(x, theta, kind='quadratic', fill_value=fillval_f, bounds_error=False)
            intp_mag = interp1d(x, mag, kind='quadratic', fill_value=fillval_f, bounds_error=False)

            # Original formula from Joe
            # mov_x = 3.6 * intp_r(dset[times_dimname][mask_nan[track]]) * np.cos(
            #     np.pi / 180.0 * intp_theta(dset[times_dimname][mask_nan[track]]))
            # mov_y = 3.6 * intp_r(dset[times_dimname][mask_nan[track]]) * np.sin(
            #     np.pi / 180.0 * intp_theta(dset[times_dimname][mask_nan[track]]))
            # TODO: need to double check the following formula
            mov_x = intp_mag(xm) * np.cos(np.pi / 180.0 * intp_theta(xm))
            mov_y = intp_mag(xm) * np.sin(np.pi / 180.0 * intp_theta(xm))

            # Interpolate values
            m_mag[track][mask_nan[track]] = intp_mag(xm)
            m_speed[track][mask_nan[track]] = intp_r(xm)
            m_theta[track][mask_nan[track]] = intp_theta(xm)
            m_x[track][mask_nan[track]] = mov_x
            m_y[track][mask_nan[track]] = mov_y

    # Update variables in dataset
    dset['movement_distance'] = ((tracks_dimname, times_dimname), m_mag, m_mag_attrs)
    dset['movement_speed'] = ((tracks_dimname, times_dimname), m_speed, m_speed_attrs)
    dset['movement_theta'] = ((tracks_dimname, times_dimname), m_theta, m_theta_attrs)
    dset['movement_distance_x'] = ((tracks_dimname, times_dimname), m_x, m_x_attrs)
    dset['movement_distance_y'] = ((tracks_dimname, times_dimname), m_y, m_y_attrs)

    return dset