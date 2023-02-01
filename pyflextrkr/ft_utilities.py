import numpy as np
import os, fnmatch, sys, glob
import datetime, calendar, time
from pytz import utc
import yaml
import xarray as xr
import logging
from scipy.sparse import csr_matrix

def setup_logging():
    """
    Set the logging message level

    Args:
        None.

    Returns:
        None.
    """
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

def load_config(config_file):
    """
    Load configuration file, set paths and update the configuration dictionary.

    Args:
        config_file: string
            Path to a config file.

    Returns:
        config: dictionary
            Dictionary containing config parameters.
    """
    logger = logging.getLogger(__name__)
    # Read configuration from yaml file
    stream = open(config_file, "r")
    config = yaml.full_load(stream)

    startdate = config["startdate"]
    enddate = config["enddate"]
    # Set up tracking output file locations
    tracking_outpath = config["root_path"] + "/" + config["tracking_path_name"] + "/"
    stats_outpath = config["root_path"] + "/" + config["stats_path_name"] + "/"
    pixeltracking_outpath = config["root_path"] + "/" + config["pixel_path_name"] + "/" + \
                            config["startdate"] + "_" + config["enddate"] + "/"
    cloudid_filebase = "cloudid_"
    singletrack_filebase = "track_"
    tracknumbers_filebase = "tracknumbers_"
    trackstats_filebase = "trackstats_"
    trackstats_sparse_filebase = "trackstats_sparse_"

    # Optional parameters (default values if not in config file)
    trackstats_dense_netcdf = config.get("trackstats_dense_netcdf", 0)
    geolimits = config.get("geolimits", [-90., -360., 90., 360.])

    # Create output directories
    os.makedirs(tracking_outpath, exist_ok=True)
    os.makedirs(stats_outpath, exist_ok=True)
    os.makedirs(pixeltracking_outpath, exist_ok=True)

    # Calculate basetime for start and end date
    start_basetime = get_basetime_from_string(startdate)
    end_basetime = get_basetime_from_string(enddate)
    # Add newly defined variables to config
    config.update(
        {
            "tracking_outpath": tracking_outpath,
            "stats_outpath": stats_outpath,
            "pixeltracking_outpath": pixeltracking_outpath,
            "cloudid_filebase": cloudid_filebase,
            "singletrack_filebase": singletrack_filebase,
            "tracknumbers_filebase": tracknumbers_filebase,
            "trackstats_filebase": trackstats_filebase,
            "trackstats_sparse_filebase": trackstats_sparse_filebase,
            "trackstats_dense_netcdf": trackstats_dense_netcdf,
            "start_basetime": start_basetime,
            "end_basetime": end_basetime,
            "geolimits": geolimits,
        }
    )
    return config

def get_basetime_from_string(datestring):
    """
    Calculate base time (Epoch time) from a string.

    Args:
        datestring: string (yyyymodd.hhmm)
            String containing date & time.

    Returns:
        base_time: int
            Epoch time in seconds.
    """
    logger = logging.getLogger(__name__)
    TEMP_starttime = datetime.datetime(int(datestring[0:4]),
                                       int(datestring[4:6]),
                                       int(datestring[6:8]),
                                       int(datestring[9:11]),
                                       int(datestring[11:]),
                                       0, tzinfo=utc)
    base_time = calendar.timegm(TEMP_starttime.timetuple())
    return base_time

def get_basetime_from_filename(
    data_path,
    data_basename,
    time_format="yyyymodd_hhmm",
):
    """
    Calculate base time (Epoch time) from filenames.

    Args:
        data_path: string
            Data directory name.
        data_basename: string
            Data base name.
        time_format: string (optional, default="yyyymodd_hhmm")
            Specify file time format to extract date/time.
    Returns:
        data_filenames: list
            List of data filenames.
        files_basetime: numpy array
            Array of file base time.
        files_datestring: list
            List of file date string.
        files_timestring: list
            List of file time string.

    """
    logger = logging.getLogger(__name__)
    # Isolate all possible files
    filenames = sorted(fnmatch.filter(os.listdir(data_path), data_basename + '*'))

    # Loop through files, identifying files within the startdate - enddate interval
    nleadingchar = len(data_basename)

    data_filenames = [None]*len(filenames)
    files_timestring = [None]*len(filenames)
    files_datestring = [None]*len(filenames)
    files_basetime = np.full(len(filenames), -9999, dtype=int)

    yyyy_idx = time_format.find("yyyy")
    mo_idx = time_format.find("mo")
    dd_idx = time_format.find("dd")
    hh_idx = time_format.find("hh")
    mm_idx = time_format.find("mm")
    ss_idx = time_format.find("ss")

    # Add basetime character counts to get the actual date/time string positions
    yyyy_idx = nleadingchar + yyyy_idx
    mo_idx = nleadingchar + mo_idx
    dd_idx = nleadingchar + dd_idx
    hh_idx = nleadingchar + hh_idx if (hh_idx != -1) else None
    mm_idx = nleadingchar + mm_idx if (mm_idx != -1) else None
    ss_idx = nleadingchar + ss_idx if (ss_idx != -1) else None

    # Loop over each file
    for ii, ifile in enumerate(filenames):
        year = ifile[yyyy_idx:yyyy_idx+4]
        month = ifile[mo_idx:mo_idx+2]
        day = ifile[dd_idx:dd_idx+2]
        # If hour, minute, second is not in time_format, assume 0
        hour = ifile[hh_idx:hh_idx+2] if (hh_idx is not None) else '00'
        minute = ifile[mm_idx:mm_idx+2] if (mm_idx is not None) else '00'
        second = ifile[ss_idx:ss_idx+2] if (ss_idx is not None) else '00'

        # Check month, day, hour, minute, second valid values
        if (0 <= int(month) <= 12) & (0 <= int(day) <= 31) & \
           (0 <= int(hour) <= 23) & (0 <= int(minute) <= 59) & (0 <= int(second) <= 59):
            TEMP_filetime = datetime.datetime(int(year), int(month), int(day),
                                            int(hour), int(minute), int(second), tzinfo=utc)
            files_basetime[ii] = calendar.timegm(TEMP_filetime.timetuple())
        else:
            logger.warning(f'File has invalid date/time, will not be included in processing: {ifile}')
            pass
        files_datestring[ii] = year + month + day
        files_timestring[ii] = hour + minute
        data_filenames[ii] = data_path + ifile
    return (
        data_filenames,
        files_basetime,
        files_datestring,
        files_timestring,
    )

def subset_files_timerange(
    data_path,
    data_basename,
    start_basetime,
    end_basetime,
    time_format="yyyymodd_hhmm",
):
    """
    Subset files within given start and end time.

    Args:
        data_path: string
            Data directory name.
        data_basename: string
            Data base name.
        start_basetime: int
            Start base time (Epoch time).
        end_basetime: int
            End base time (Epoch time).
        time_format: string (optional, default="yyyymodd_hhmm")
            Specify file time format to extract date/time.

    Returns:
        data_filenames: list
            List of data file names with full path.
        files_basetime: numpy array
            Array of file base time.
        files_datestring: list
            List of file date string.
        files_timestring: list
            List of file time string.
    """
    logger = logging.getLogger(__name__)
    # Get basetime for all files
    data_filenames, files_basetime, \
    files_datestring, files_timestring = get_basetime_from_filename(
        data_path, data_basename, time_format=time_format,
    )

    # Find basetime within the given range
    fidx = np.where((files_basetime >= start_basetime) & (files_basetime <= end_basetime))[0]
    # Subset filenames, dates, times
    data_filenames = np.array(data_filenames)[fidx].tolist()
    files_basetime = files_basetime[fidx]
    files_datestring = np.array(files_datestring)[fidx].tolist()
    files_timestring = np.array(files_timestring)[fidx].tolist()

    return (
        data_filenames,
        files_basetime,
        files_datestring,
        files_timestring,
    )

def match_drift_times(
    cloudidfiles_datestring,
    cloudidfiles_timestring,
    driftfile=None
):
    """
    Match drift file times with cloudid file times.

    Args:
        cloudidfiles_datestring: list
            List of cloudid files date string.
        cloudidfiles_timestring: list
            List of cloudid files time string.
        driftfile: string (optional)
            Drift (advection) file name.

    Returns:
        datetime_drift_match: numpy array
            Matched drift file date time strings.
        xdrifts_match: numpy array
            Matched drift distance in the x-direction.
        ydrifts_match: numpy array
            Matched drift distance in the y-direction.

    """
    logger = logging.getLogger(__name__)
    # Create drift variables that match number of reference cloudid files
    # Number of reference cloudid files (1 less than total cloudid files)
    ncloudidfiles = len(cloudidfiles_timestring) - 1
    datetime_drift_match = np.empty(ncloudidfiles, dtype='<U13')
    xdrifts_match = np.zeros(ncloudidfiles, dtype=int)
    ydrifts_match = np.zeros(ncloudidfiles, dtype=int)
    # Test if driftfile is defined
    try:
        driftfile
    except NameError:
        logger.info(f"Drift file is not defined. Regular tracksingle procedure is used.")
    else:
        logger.info(f"Drift file used: {driftfile}")

        # Read the drift file
        ds_drift = xr.open_dataset(driftfile)
        bt_drift = ds_drift['time']
        xdrifts = ds_drift['x'].values.squeeze()
        ydrifts = ds_drift['y'].values.squeeze()

        # Convert dateime64 objects to string array
        datetime_drift = bt_drift.dt.strftime("%Y%m%d_%H%M").values

        # Loop over each cloudid file time to find matching drfit data
        for itime in range(0, len(cloudidfiles_timestring) - 1):
            cloudid_datetime = cloudidfiles_datestring[itime] + '_' + cloudidfiles_timestring[itime]
            idx = np.where(datetime_drift == cloudid_datetime)[0]
            if (len(idx) == 1):
                datetime_drift_match[itime] = datetime_drift[idx[0]]
                xdrifts_match[itime] = xdrifts[idx]
                ydrifts_match[itime] = ydrifts[idx]
    return (
        datetime_drift_match,
        xdrifts_match,
        ydrifts_match,
    )


def load_sparse_trackstats(
        max_trackduration,
        statistics_file,
        times_idx_varname,
        tracks_dimname,
        tracks_idx_varname,
):
    """
    Load sparse trackstats file and convert to sparse arrays.

    Args:
        max_trackduration:
        statistics_file:
        times_idx_varname:
        tracks_dimname:
        tracks_idx_varname:

    Returns:
        ds_1d: Xarray Dataset
            Dataset containing 1D track stats variables.
        sparse_attrs_dict: dictionary
            Dictionary containing sparse array attributes.
        sparse_dict: dictionary
            Dictionary containing sparse array variables.
    """
    # xr.set_options(keep_attrs=True)
    ds_all = xr.open_dataset(statistics_file,
                             mask_and_scale=False,
                             decode_times=False)
    # Get sparse array info
    sparse_dimname = 'sparse_index'
    nsparse_data = ds_all.dims[sparse_dimname]
    ntracks = ds_all.dims[tracks_dimname]
    # Sparse array indices
    tracks_idx = ds_all[tracks_idx_varname].values
    times_idx = ds_all[times_idx_varname].values
    row_col_ind = (tracks_idx, times_idx)
    # Sparse array shapes
    shape_2d = (ntracks, max_trackduration)
    # Convert sparse arrays and put in a dictionary
    sparse_dict = {}
    sparse_attrs_dict = {}
    for ivar in ds_all.data_vars.keys():
        # Check dimension name for sparse arrays
        if ds_all[ivar].dims[0] == sparse_dimname:
            # Convert to sparse array
            sparse_dict[ivar] = csr_matrix(
                (ds_all[ivar].values, row_col_ind), shape=shape_2d, dtype=ds_all[ivar].dtype,
            )
            # Collect variable attributes
            sparse_attrs_dict[ivar] = ds_all[ivar].attrs
    # Drop all sparse variables and dimension
    ds_1d = ds_all.drop_dims(sparse_dimname)
    ds_all.close()
    return ds_1d, sparse_attrs_dict, sparse_dict


def convert_trackstats_sparse2dense(
        filename_sparse,
        filename_dense,
        max_trackduration,
        tracks_idx_varname,
        times_idx_varname,
        tracks_dimname,
        times_dimname,
        fillval,
        fillval_f,
):
    """
    Convert sparse trackstats netCDF file to dense trackstats netCDF file.

    Args:
        filename_sparse: string
            Filename for sparse trackstats netCDF file.
        filename_dense: string
            Filename for dense trackstats netCDF file.
        max_trackduration: int
            Maximum track duration.
        tracks_idx_varname: string
            Tracks indices variable name.
        times_idx_varname: string
            Times indices variable name.
        tracks_dimname: string
            Tracks dimension name.
        times_dimname: string
            Times dimension name.
        fillval: int
            Missing value for int type variables.
        fillval_f: float
            Missing value for float type variables.

    Returns:
        True.
    """
    # Read sparse netCDF file
    ds_all = xr.open_dataset(
        filename_sparse,
        mask_and_scale=False,
        decode_times=False
    )
    # Get sparse array info
    sparse_dimname = 'sparse_index'
    nsparse_data = ds_all.dims[sparse_dimname]
    ntracks = ds_all.dims[tracks_dimname]
    # Sparse array indices
    tracks_idx = ds_all[tracks_idx_varname].values
    times_idx = ds_all[times_idx_varname].values
    row_col_ind = (tracks_idx, times_idx)
    # Sparse array shapes
    shape_2d = (ntracks, max_trackduration)

    # Create a dense mask for no feature
    mask = csr_matrix(
        (ds_all['base_time'].data, row_col_ind),
        shape=shape_2d,
        dtype=ds_all['base_time'].dtype,
    ).toarray() == 0

    # Create variable dictionary
    var_dict = {}
    for key, value in ds_all.data_vars.items():
        # Check dimension name for sparse arrays
        if ds_all[key].dims[0] == sparse_dimname:
            # Convert to sparse array, then to dense array
            var_dense = csr_matrix(
                (ds_all[key].data, row_col_ind), shape=shape_2d, dtype=ds_all[key].dtype,
            ).toarray()
            # Replace missing values based on variable type
            if isinstance(var_dense[0, 0], np.floating):
                var_dense[mask] = fillval_f
            else:
                var_dense[mask] = fillval
            var_dict[key] = ([tracks_dimname, times_dimname], var_dense, ds_all[key].attrs)
        else:
            var_dict[key] = ([tracks_dimname], value.data, ds_all[key].attrs)
    # Remove the tracks/times indices variables
    var_dict.pop(tracks_idx_varname, None)
    var_dict.pop(times_idx_varname, None)

    # Define coordinate dictionary
    coord_dict = {
        tracks_dimname: ([tracks_dimname], np.arange(0, ntracks)),
        times_dimname: ([times_dimname], np.arange(0, max_trackduration)),
    }

    # Update file creation time in global attribute
    gattr_dict = ds_all.attrs
    gattr_dict["Created_on"] = time.ctime(time.time())

    # Define output Xarray dataset
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)
    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in dsout.data_vars}
    # Write to netcdf file
    dsout.to_netcdf(
        path=filename_dense,
        mode='w',
        format='NETCDF4',
        unlimited_dims=tracks_dimname,
        encoding=encoding
    )
    return True