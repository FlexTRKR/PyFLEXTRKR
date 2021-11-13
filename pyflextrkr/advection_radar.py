import numpy as np
import xarray as xr
from netCDF4 import Dataset
from scipy.signal import medfilt
from skimage.registration import phase_cross_correlation
from scipy import ndimage as ndi
import logging
import dask
from pyflextrkr.ftfunctions import subset_files_timerange


def offset_to_speed(x, y, time_lag, dx, dy):
    """Return normalized speed assuming uniform grid."""
    mag_movement = np.sqrt((dx * x) ** 2 + (dy * y) ** 2)
    mag_dir = np.arctan2(x * dx, y * dy) * 180 / np.pi
    mag_movement_mps = np.array(
        [mag_movement_i / (time_lag) * 1000.0 for mag_movement_i in mag_movement.T]
    ).T
    return mag_movement, mag_dir, mag_movement_mps


def get_pixel_size_of_clouds(dataset, total_tracks, track_variable="pcptracknumber"):

    """Calculate pixel size of each identified cloud in the file.
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
    cuts=1,
    times=None,
    threshold=30,
    plot_subplots=False,
    buffer=30,
    size_threshold=10,
    TIME_RES_SECOND=15 * 60,
    MAX_MOVEMENT_MPS=60,
):

    """Calculate Movement of first labeled storm
    Parameters
    ----------
    dset_1: Xarray DataSet
        Dataset at current time (t=0)
    dset_2: Xarray DataSet
        Dataset at next time (t=1)
    dx: float
        Grid spacing in x-direction [km]
    dy: float
        Grid spacing in y-direction [km]
    cuts: int
        Number of tiles to cut the domain into for calculating advection
    times: ?
        Unknown.
    threshold: float
        Threshold value for filtering the data
    buffer: int
        Number of grids to add buffer to the field
    size_threshold: int
        Number of valid points to calculate advection.
    TIME_RES_SECOND: float
        Time resolution of data [seconds]
    MAX_MOVEMENT_MPS: float
        Maximum movement speed allowed [m/s]
    Returns:
    --------
    y_lag: int
        Advection in x-direction [number of grids]
    x_lag: int
        Advection in y-direction [number of grids]
    """
    field_1 = np.squeeze(dset_1["dbz_comp"].values)
    field_2 = np.squeeze(dset_2["dbz_comp"].values)

    y_lag = np.zeros((cuts, cuts))
    x_lag = np.zeros((cuts, cuts))

    mask_1 = field_1 > threshold
    mask_2 = field_2 > threshold

    dimensions = field_1.shape
    x_skip = int(dimensions[0] / cuts)
    y_skip = int(dimensions[1] / cuts)

    # Mask each region into a cut
    for col in range(0, cuts):
        for row in range(0, cuts):
            mask = np.zeros(field_1.shape)
            mask[
                buffer + row * x_skip : (row + 1) * x_skip - buffer,
                buffer + col * y_skip : (col + 1) * y_skip - buffer,
            ] = 1
            mask = mask * mask_1
            mask[field_1 < -100] = 0
            mask_2[field_2 < -100] = 0

            num_points = np.sum(mask > 0)
            # Updated to use phase_cross_correlation to replace masked_register_translation
            y, x = -1 * phase_cross_correlation(
                field_1, field_2, reference_mask=mask_1, moving_mask=mask_2, overlap_ratio=0.7
            )

            if plot_subplots:
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.pcolormesh(field_1 * mask, vmin=-20, vmax=50, cmap="jet")
                plt.colorbar()
                plt.arrow(100, 100, x, y, head_width=5)
                plt.subplot(1, 2, 2)
                plt.pcolormesh(field_2 * mask_2, vmin=-20, vmax=50, cmap="jet")
                plt.colorbar()

                plt.figure(figsize=(10, 10))
                plt.pcolormesh(field_2 * mask_2, vmin=-20, vmax=50, cmap="jet")
                shifted_field_1 = ndi.shift(mask_1, [int(y), int(x)])
                plt.contour(shifted_field_1, vmin=-1, vmax=1, cmap="seismic", levels=3)
                plt.contour(-1 * mask_1, vmin=-1, vmax=1, cmap="seismic", levels=3)

                plt.arrow(100, 100, x, y, head_width=15)

                plt.colorbar()

            if num_points < size_threshold:
                x_lag[row, col] = np.nan
                y_lag[row, col] = np.nan
                continue

            y_lag[row, col] = y
            x_lag[row, col] = x

    # mag_movement, mag_dir, mag_movement_mps = offset_to_speed(x_lag, y_lag, 15 * 60)
    mag_movement, mag_dir, mag_movement_mps = offset_to_speed(
        x_lag, y_lag, TIME_RES_SECOND, dx, dy
    )

    x_lag[mag_movement_mps > MAX_MOVEMENT_MPS] = np.nan
    y_lag[mag_movement_mps > MAX_MOVEMENT_MPS] = np.nan

    x_lag[np.isnan(x_lag)] = np.nanmedian(0)
    y_lag[np.isnan(y_lag)] = np.nanmedian(0)

    return y_lag[0, 0], x_lag[0, 0]

def movement_of_storm_fft_l(
    filenames, dx, dy, DBZ_THRESHOLD, TIME_RES_SECOND, MAX_MOVEMENT_MPS
):
    """This just exists to make parallelism easier"""
    dset_1 = xr.open_dataset(filenames[0])
    dset_2 = xr.open_dataset(filenames[1])

    y1, x1 = movement_of_storm_fft(
        dset_1,
        dset_2,
        dx=dx,
        dy=dy,
        threshold=DBZ_THRESHOLD,
        TIME_RES_SECOND=TIME_RES_SECOND,
        MAX_MOVEMENT_MPS=MAX_MOVEMENT_MPS,
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

    clouddata_path = config["clouddata_path"]
    DBZ_THRESHOLD = config["DBZ_THRESHOLD"]
    dx = config["pixel_radius"]
    dy = config["pixel_radius"]
    MED_FILT_LEN = config["MED_FILT_LEN"]
    MAX_MOVEMENT_MPS = config["MAX_MOVEMENT_MPS"]
    datatimeresolution = config["datatimeresolution"]
    run_parallel = config["run_parallel"]

    output_filename = (
        config["stats_outpath"] +
        config["datasource"] +
        "_advection_" +
        config["startdate"] + "_" +
        config["enddate"] + ".nc"
    )

    # Find files within start/end time
    # filelist = sorted(glob.glob(f"{clouddata_path}*.nc"))
    filelist, \
    files_basetime, \
    files_datestring, \
    files_timestring = subset_files_timerange(
        clouddata_path,
        config["databasename"],
        config["start_basetime"],
        config["end_basetime"],
        time_format=config["time_format"]
    )
    logger.info(f"Found {len(filelist)} files.")

    # Convert data time resolution from [hour] to [second]
    TIME_RES_SECOND = datatimeresolution * 3600

    # Run advection calculation
    if run_parallel == 0:
        # Serial version
        final_results = []
        for ii in zip(filelist[:-1], filelist[1:]):
            x_y = movement_of_storm_fft_l(
                ii,
                dx=dx,
                dy=dy,
                DBZ_THRESHOLD=DBZ_THRESHOLD,
                TIME_RES_SECOND=TIME_RES_SECOND,
                MAX_MOVEMENT_MPS=MAX_MOVEMENT_MPS,
            )
            final_results.append(x_y)

    if run_parallel == 1:
        # Parallel version
        results = []
        for ii in zip(filelist[:-1], filelist[1:]):
            x_y = dask.delayed(movement_of_storm_fft_l)(
                ii,
                dx=dx,
                dy=dy,
                DBZ_THRESHOLD=DBZ_THRESHOLD,
                TIME_RES_SECOND=TIME_RES_SECOND,
                MAX_MOVEMENT_MPS=MAX_MOVEMENT_MPS,
            )
            results.append(x_y)
        final_results = dask.compute(*results)
        dask.distributed.wait(final_results)

    # Zip the (x, y) and convert them into numpy array
    x_and_y = np.array(tuple(zip(*final_results)))
    y = x_and_y[0]
    x = x_and_y[1]

    mag = np.sqrt((x * dx) ** 2 + (y * dy) ** 2)
    mag_mps = 1000 / (60 * 15) * mag

    angle = 90 - np.arctan2(y, x) * 180 / np.pi
    mag_med = medfilt(np.squeeze(mag), MED_FILT_LEN)
    mag_mps_med = medfilt(np.squeeze(mag_mps), MED_FILT_LEN)
    angle_med = medfilt(np.squeeze(angle), MED_FILT_LEN)

    med_x = (1 / dx) * mag_med * np.cos(np.pi / 180 * (90 - angle_med))
    med_y = (1 / dy) * mag_med * np.sin(np.pi / 180 * (90 - angle_med))

    corrections = zip(files_basetime, med_x, med_y, mag_med, angle_med)
    corrections = list(corrections)
    # basetime = np.array([t[0][0] for t in corrections])
    # basedate = [t[0][1] for t in corrections]

    rootgrp = Dataset(output_filename, "w", format="NETCDF4")
    d_time = rootgrp.createDimension("time", len(corrections))
    v_time = rootgrp.createVariable("time", "i8", ("time",))
    v_basetime = rootgrp.createVariable("basetime", "i8", ("time",))
    v_x = rootgrp.createVariable("x", "f8", ("time",))
    v_y = rootgrp.createVariable("y", "f8", ("time",))
    v_mag = rootgrp.createVariable("magnitude", "f8", ("time",))
    v_dir = rootgrp.createVariable("direction", "f8", ("time",))

    v_time[:] = [t[0] for t in corrections]
    v_time.units = "Seconds since 1970-1-1 0:00:00 0:00"
    v_basetime[:] = [t[0] for t in corrections]
    v_basetime.units = "Seconds since 1970-1-1 0:00:00 0:00"
    v_x[:] = [np.round(t[1]) for t in corrections]
    v_x.long_name = "Advection in x-direction"
    v_x.units = "Number of grids"
    v_y[:] = [np.round(t[2]) for t in corrections]
    v_y.long_name = "Advection in y-direction"
    v_y.units = "Number of grids"
    v_mag[:] = [t[3] for t in corrections]
    v_mag.long_name = "Advection speed magnitude"
    v_mag.units = "m/s"
    v_dir[:] = [t[4] for t in corrections]
    v_dir.long_name = "Advection direction"
    v_dir.units = "degree"

    # import pdb;
    # pdb.set_trace()

    rootgrp.dbz_threshold = float(DBZ_THRESHOLD)
    rootgrp.MAX_MOVEMENT_MPS = float(MAX_MOVEMENT_MPS)
    rootgrp.dx = dx
    rootgrp.dy = dy
    rootgrp.close()
    logger.info(f"Advection file saved: {output_filename}")

    status = 1
    return output_filename
