import sys
import numpy as np
import xarray as xr
from netCDF4 import Dataset
from scipy.signal import medfilt
from skimage.registration import phase_cross_correlation
from scipy import ndimage as ndi
import logging
import dask
from pyflextrkr.ft_utilities import subset_files_timerange


def offset_to_speed(x, y, time_lag, dx, dy):
    """
    Return normalized speed assuming uniform grid.

    Args:
        x: np.array
            Movement distance in x-direction [km]
        y: np.array
            Movement distance in y-direction [km]
        time_lag: float
            Time elapsed for movement
        dx: float
            Grid spacing in x-direction [km]
        dy: float
            Grid spacing in y-direction [km]
    Returns:
        mag_movement: np.array
            Movement distance
        mag_dir: np.array
            Movement direction
        mag_movement_mps: np.array
            Movement speed [m/s]
    """
    mag_movement = np.sqrt((dx * x) ** 2 + (dy * y) ** 2)
    mag_dir = np.arctan2(x * dx, y * dy) * 180 / np.pi
    mag_movement_mps = np.array(
        [mag_movement_i / (time_lag) * 1000.0 for mag_movement_i in mag_movement.T]
    ).T
    return mag_movement, mag_dir, mag_movement_mps


def get_pixel_size_of_clouds(dataset, total_tracks, track_variable="pcptracknumber"):
    """
    Calculate pixel size of each identified cloud in the file.

    Args:
        dataset: Dataset
            netcdf Dataset
        total_tracks: integer
            Total number of tracks.
        track_variable: tring
            variable that contains pixel level values.

    Returns:
        counts: array_like
            Pixel size of every cloud in file. Cloud 0 is stored at 0.
    """
    storm_sizes = np.zeros(total_tracks + 1)

    track, counts = np.unique(dataset.variables[track_variable][:], return_counts=True)
    storm_sizes[track] = counts
    storm_sizes[0] = 0
    return storm_sizes


def movement_of_storm_fft(
        dset_1,
        dset_2,
        dx,
        dy,
        config,
        plot_subplots=False,
):
    """
    Calculate Movement of labeled storm.

    Args:
        dset_1: Xarray DataSet
            Dataset at current time (t=0)
        dset_2: Xarray DataSet
            Dataset at next time (t=1)
        dx: float
            Grid spacing in x-direction [km]
        dy: float
            Grid spacing in y-direction [km]
        config: dictionary
            Dictionary containing config parameters

    Returns:
        y_lag: int
            Advection in x-direction [number of grids]
        x_lag: int
            Advection in y-direction [number of grids]
    """
    logger = logging.getLogger(__name__)

    ref_varname = config['ref_varname']
    field_threshold = config['advection_field_threshold']
    datatimeresolution = config["datatimeresolution"]
    advection_mask_method = config.get('advection_mask_method', 'greater')
    buffer = config.get('advection_buffer', 30)
    size_threshold = config.get('advection_size_threshold', 10)
    tiles = config.get('advection_tiles', [1,1])
    advection_max_movement_mps = config.get('advection_max_movement_mps', 60)

    # Convert data time resolution from [hour] to [second]
    TIME_RES_SECOND = datatimeresolution * 3600

    field_1 = np.squeeze(dset_1[ref_varname].values)
    field_2 = np.squeeze(dset_2[ref_varname].values)

    # Make arrays for advection
    tiles_y, tiles_x = tiles[0], tiles[1]
    y_lag = np.zeros((tiles_y, tiles_x), dtype=np.float32)
    x_lag = np.zeros((tiles_y, tiles_x), dtype=np.float32)

    # Mask data by thresholds
    if advection_mask_method == 'greater':
        mask_1 = field_1 > field_threshold
        mask_2 = field_2 > field_threshold
    elif advection_mask_method == 'smaller':
        mask_1 = field_1 < field_threshold
        mask_2 = field_2 < field_threshold
    else:
        logger.error(f'Error: Undefined advection_mask_method: {advection_mask_method}')
        logger.error("Tracking will now exit.")
        sys.exit()

    dimensions = field_1.shape
    row_skip = int(dimensions[0] / tiles_y)
    col_skip = int(dimensions[1] / tiles_x)

    # Mask each region into a cut
    for col in range(0, tiles_x):
        for row in range(0, tiles_y):
            # Buffer the edge
            mask = np.zeros(field_1.shape)
            mask[
                buffer + row * row_skip : (row + 1) * row_skip - buffer,
                buffer + col * col_skip : (col + 1) * col_skip - buffer,
            ] = 1
            # Separate mask by tiles
            mask_1t = mask * mask_1
            mask_2t = mask * mask_2
            # Set mask to 0 for areas not used
            if advection_mask_method == 'greater':
                mask_1t[field_1 < field_threshold] = 0
                mask_2t[field_2 < field_threshold] = 0
            elif advection_mask_method == 'smaller':
                mask_1t[field_1 > field_threshold] = 0
                mask_2t[field_2 > field_threshold] = 0
            # mask[field_1 < -100] = 0
            # mask_2[field_2 < -100] = 0

            num_points = np.sum(mask > 0)
            # Updated to use phase_cross_correlation to replace masked_register_translation
            y, x = -1 * phase_cross_correlation(
                field_1, field_2, reference_mask=mask_1t, moving_mask=mask_2t, overlap_ratio=0.7
            )[0]

            # plot_subplots = True
            if plot_subplots:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.pcolormesh(field_1 * mask_1t, vmin=0, vmax=50, cmap="gist_ncar")
                plt.colorbar()
                plt.arrow(100, 100, x, y, head_width=5)
                plt.subplot(1, 2, 2)
                plt.pcolormesh(field_2 * mask_2t, vmin=0, vmax=50, cmap="gist_ncar")
                plt.colorbar()

                plt.figure(figsize=(10, 10))
                plt.pcolormesh(field_2 * mask_2t, vmin=0, vmax=50, cmap="gist_ncar")
                plt.colorbar()
                shifted_field_1 = ndi.shift(mask_1t, [int(y), int(x)])
                plt.contour(shifted_field_1, vmin=-1, vmax=1, cmap="seismic", levels=3)
                plt.contour(-1 * mask_1t, vmin=-1, vmax=1, cmap="seismic", levels=3)
                plt.arrow(100, 100, x, y, head_width=15)
                plt.show()

            if num_points < size_threshold:
                x_lag[row, col] = np.nan
                y_lag[row, col] = np.nan
                continue
            # Save movement values
            y_lag[row, col] = y
            x_lag[row, col] = x

    # Calculate movement speed
    mag_movement, mag_dir, mag_movement_mps = offset_to_speed(
        x_lag, y_lag, TIME_RES_SECOND, dx, dy,
    )
    # Remove movement values larger than max speed allowed
    x_lag[mag_movement_mps > advection_max_movement_mps] = np.nan
    y_lag[mag_movement_mps > advection_max_movement_mps] = np.nan
    # Replace NaN values with 0
    x_lag[np.isnan(x_lag)] = np.nanmedian(0)
    y_lag[np.isnan(y_lag)] = np.nanmedian(0)

    return y_lag, x_lag
    # return y_lag[0, 0], x_lag[0, 0]

def movement_of_storm_fft_l(
    filenames, dx, dy, config,
):
    """This just exists to make parallelism easier"""
    dset_1 = xr.open_dataset(filenames[0])
    dset_2 = xr.open_dataset(filenames[1])

    y1, x1 = movement_of_storm_fft(
        dset_1,
        dset_2,
        dx=dx,
        dy=dy,
        config=config,
    )
    return y1, x1


def calc_mean_advection(config):
    """
    Calculate domain mean advection.

    Args:
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        output_filename: string
            Advection file name.
    """
    logger = logging.getLogger(__name__)

    clouddata_path = config["tracking_outpath"]
    field_threshold = config["advection_field_threshold"]
    dx = config["pixel_radius"]
    dy = config["pixel_radius"]
    advection_tiles = config["advection_tiles"]
    advection_med_filt_len = config["advection_med_filt_len"]
    advection_max_movement_mps = config["advection_max_movement_mps"]
    datatimeresolution = config["datatimeresolution"]
    run_parallel = config["run_parallel"]

    output_filename = (
        config["stats_outpath"] +
        "advection_" +
        config["startdate"] + "_" +
        config["enddate"] + ".nc"
    )

    # Find files within start/end time
    filelist, \
    files_basetime, \
    files_datestring, \
    files_timestring = subset_files_timerange(
        clouddata_path,
        config["cloudid_filebase"],
        config["start_basetime"],
        config["end_basetime"],
        # time_format=config["time_format"]
    )
    logger.info(f"Found {len(filelist)} files.")

    # Convert data time resolution from [hour] to [second]
    TIME_RES_SECOND = datatimeresolution * 3600
    # Number of tiles in y, x direction
    tiles_y, tiles_x = advection_tiles[0], advection_tiles[1]

    # Run advection calculation
    if run_parallel == 0:
        # Serial version
        final_results = []
        for ii in zip(filelist[:-1], filelist[1:]):
            x_y = movement_of_storm_fft_l(
                ii,
                dx=dx,
                dy=dy,
                config=config,
            )
            final_results.append(x_y)

    elif run_parallel >= 1:
        # Parallel version
        results = []
        for ii in zip(filelist[:-1], filelist[1:]):
            x_y = dask.delayed(movement_of_storm_fft_l)(
                ii,
                dx=dx,
                dy=dy,
                config=config,
            )
            results.append(x_y)
        final_results = dask.compute(*results)
        dask.distributed.wait(final_results)

    else:
        sys.exit('Valid parallelization flag not provided')

    # Zip the (x, y) and convert them into numpy array
    x_and_y = np.array(tuple(zip(*final_results)))
    y = x_and_y[0]
    x = x_and_y[1]

    # Calculate movement distance [km]
    mag = np.sqrt((x * dx) ** 2 + (y * dy) ** 2)
    # Calculate movement speed [m/s]
    mag_mps = 1000 * mag / TIME_RES_SECOND
    # Calculate movement direction [degree from North]
    angle = 90 - np.arctan2(y, x) * 180 / np.pi

    # Make a median filter window only applied on time dimension (time,y,x)
    advection_med_filt_window = (advection_med_filt_len, 1, 1)
    # Perform median filter on time dimension to remove outliers
    mag_med = medfilt(mag, advection_med_filt_window)
    mag_mps_med = medfilt(mag_mps, advection_med_filt_window)
    angle_med = medfilt(angle, advection_med_filt_window)
    # Back out movement x, y from filtered magnitude & angle
    med_x = np.round((1 / dx) * mag_med * np.cos(np.pi / 180 * (90 - angle_med)))
    med_y = np.round((1 / dy) * mag_med * np.sin(np.pi / 180 * (90 - angle_med)))

    # Remove the last time to match the advection variables
    match_basetime = files_basetime[:-1]

    # Write output to netCDF file
    var_dict = {
        'base_time': (['time'], match_basetime),
        'x': (['time', 'tile_y', 'tile_x'], med_x),
        'y': (['time', 'tile_y', 'tile_x'], med_y),
        'magnitude': (['time', 'tile_y', 'tile_x'], mag_med),
        'direction': (['time', 'tile_y', 'tile_x'], angle_med),
        'speed': (['time', 'tile_y', 'tile_x'], mag_mps_med),
    }
    coord_dict = {
        'time': (['time'], match_basetime),
        'tile_y': (['tile_y'], np.arange(0, tiles_y)),
        'tile_x': (['tile_x'], np.arange(0, tiles_x)),
    }
    gattr_dict = {
        'field_threshold': float(field_threshold),
        'advection_max_movement_mps': float(advection_max_movement_mps),
        'dx': dx,
        'dy': dy,
    }
    # Define xarray dataset
    ds_out = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)
    # Define variable attributes
    ds_out['base_time'].attrs['units'] = 'Seconds since 1970-1-1 0:00:00 0:00'
    ds_out['time'].attrs['units'] = 'Seconds since 1970-1-1 0:00:00 0:00'
    ds_out['tile_y'].attrs['long_name'] = 'Tile numbers in y-direction'
    ds_out['tile_x'].attrs['long_name'] = 'Tile numbers in x-direction'
    ds_out['x'].attrs['long_name'] = 'Advection in x-direction'
    ds_out['x'].attrs['units'] = 'Number of grids'
    ds_out['y'].attrs['long_name'] = 'Advection in y-direction'
    ds_out['y'].attrs['units'] = 'Number of grids'
    ds_out['magnitude'].attrs['long_name'] = 'Advection distance magnitude'
    ds_out['magnitude'].attrs['units'] = 'km'
    ds_out['direction'].attrs['long_name'] = 'Advection direction'
    ds_out['direction'].attrs['units'] = 'degree'
    ds_out['speed'].attrs['long_name'] = 'Advection speed'
    ds_out['speed'].attrs['units'] = 'm/s'
    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in ds_out.data_vars}
    # Write to netcdf file
    ds_out.to_netcdf(
        path=output_filename, mode='w', format='NETCDF4', unlimited_dims='time', encoding=encoding,
    )
    logger.info(f"Advection file saved: {output_filename}")

    # Update config to add drift filename
    config.update({"advection_filename": output_filename})

    return output_filename
