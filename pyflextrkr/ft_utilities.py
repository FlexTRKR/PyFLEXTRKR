import numpy as np
import os, fnmatch, sys, glob
import datetime, calendar, time, cftime
from pytz import utc
import yaml
import xarray as xr
import pandas as pd
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

    # Get start/end dates
    startdate = config.get("startdate", None)
    enddate = config.get("enddate", None)
    # Get start/end date from input files if not specified
    if startdate is None or enddate is None:
        # Get start/end Epoch time from all input files
        start_basetime, end_basetime = get_start_end_basetime_from_filenames(
            config["clouddata_path"],
            config["databasename"],
            time_format=config["time_format"],
        )
        # Update start/end date
        if startdate is None:
            startdate = pd.to_datetime(start_basetime, unit='s').strftime("%Y%m%d.%H%M")
        if enddate is None:
            enddate = pd.to_datetime(end_basetime, unit='s').strftime("%Y%m%d.%H%M")
        # Update config
        config.update(
            {
                "startdate": startdate,
                "enddate": enddate,
            }
        )
        logger.info(f"startdate/enddate not specified in config.")
        logger.info(f"startdate/enddate calculated from all input files, startdate: {startdate}, enddate: {enddate}")

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
    area_method = config.get("area_method", "fixed")

    # Create output directories
    os.makedirs(tracking_outpath, exist_ok=True)
    os.makedirs(stats_outpath, exist_ok=True)
    os.makedirs(pixeltracking_outpath, exist_ok=True)

    # Set grid_area_file path if area_method is "latlon"
    if area_method == "latlon":
        grid_area_file = config.get(
            "grid_area_file",
            stats_outpath + "grid_area_from_latlon.nc",
        )
    else:
        grid_area_file = None

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
            "area_method": area_method,
            "grid_area_file": grid_area_file,
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
    time_format="yyyymodd_hhmmss",
):
    """
    Calculate base time (Epoch time) from filenames.

    Args:
        data_path: string
            Data directory name.
        data_basename: string
            Data base name.
        time_format: string (optional, default="yyyymodd_hhmmss")
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
    mo_idx = nleadingchar + mo_idx if (mo_idx != -1) else None
    dd_idx = nleadingchar + dd_idx if (dd_idx != -1) else None
    hh_idx = nleadingchar + hh_idx if (hh_idx != -1) else None
    mm_idx = nleadingchar + mm_idx if (mm_idx != -1) else None
    ss_idx = nleadingchar + ss_idx if (ss_idx != -1) else None

    # Loop over each file
    for ii, ifile in enumerate(filenames):
        year = ifile[yyyy_idx:yyyy_idx+4]
        month = ifile[mo_idx:mo_idx+2] if (mo_idx is not None) else '01'
        day = ifile[dd_idx:dd_idx+2] if (dd_idx is not None) else '01'
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
        files_timestring[ii] = hour + minute + second
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
    time_format="yyyymodd_hhmmss",
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
        time_format: string (optional, default="yyyymodd_hhmmss")
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

def get_start_end_basetime_from_filenames(
    data_path,
    data_basename,
    time_format="yyyymodd_hhmmss",
):
    """
    Get start and end basetime from filenames.

    Args:
        data_path: string
            Data directory name.
        data_basename: string
            Data base name.
        time_format: string (optional, default="yyyymodd_hhmmss")
            Specify file time format to extract date/time.

    Returns:
        start_basetime: int
            Start base time (Epoch time).
        end_basetime: int
            End base time (Epoch time).
    """
    logger = logging.getLogger(__name__)
    # Get basetime for all files
    data_filenames, files_basetime, \
    files_datestring, files_timestring = get_basetime_from_filename(
        data_path, data_basename, time_format=time_format,
    )
    # Get min/max basetimes
    start_basetime = np.nanmin(files_basetime)
    end_basetime = np.nanmax(files_basetime)
    return (start_basetime, end_basetime)

def get_timestamp_from_filename_single(
    filename,
    data_basename,
    time_format="yyyymodd_hhmmss",
):
    """
    Calculate Timestamp from a filename.

    Args:
        filename: string
            File name.
        data_basename: string
            Data base name.
        time_format: string (optional, default="yyyymodd_hhmmss")
            Specify file time format to extract date/time.
    Returns:
        file_timestamp: Pandas Timestamp
            Timestamp of the file.
    """
    logger = logging.getLogger(__name__)
    nleadingchar = len(data_basename)

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

    # Remove path from filename
    ifile = os.path.basename(filename)
    year = ifile[yyyy_idx:yyyy_idx + 4]
    month = ifile[mo_idx:mo_idx + 2]
    day = ifile[dd_idx:dd_idx + 2]
    # If hour, minute, second is not in time_format, assume 0
    hour = ifile[hh_idx:hh_idx + 2] if (hh_idx is not None) else '00'
    minute = ifile[mm_idx:mm_idx + 2] if (mm_idx is not None) else '00'
    second = ifile[ss_idx:ss_idx + 2] if (ss_idx is not None) else '00'

    # Check month, day, hour, minute, second valid values
    if (0 <= int(month) <= 12) & (0 <= int(day) <= 31) & \
            (0 <= int(hour) <= 23) & (0 <= int(minute) <= 59) & (0 <= int(second) <= 59):
        file_timestamp = pd.Timestamp(f"{year}-{month}-{day}T{hour}:{minute}:{second}")
    else:
        logger.warning(f'File has invalid date/time, will return timestampe as NaN: {filename}')
        file_timestamp = np.nan
    return file_timestamp

def convert_to_cftime(datetime, calendar):
    """
    Convert a pandas.Timestamp object to a cftime object based on the calendar type.
    Return pandas.Timestamp for standard calendars that don't need conversion.

    Args:
        datetime: pandas.Timestamp
            Timestamp object to convert.
        calendar: str
            Calendar type.

    Returns:
        cftime object.
    """
    # Standard/Gregorian/Proleptic Gregorian calendars can often use pandas Timestamps directly
    if calendar in ['proleptic_gregorian', 'gregorian', 'standard']:
        return datetime
    if calendar == 'noleap':
        return cftime.DatetimeNoLeap(datetime.year, datetime.month, datetime.day, datetime.hour, datetime.minute)
    elif calendar == '360_day':
        return cftime.Datetime360Day(datetime.year, datetime.month, datetime.day, datetime.hour, datetime.minute)
    else:
        raise ValueError(f"Unsupported calendar type: {calendar}")
    
def convert_cftime_to_standard(cftime_times):
    """
    Convert cftime objects to pandas Timestamps (proleptic_gregorian calendar)
    
    Args:
        cftime_times: cftime object or array-like
            Single cftime datetime object or array of cftime datetime objects
    
    Returns:
        pandas.DatetimeIndex or pandas.Timestamp: 
            DatetimeIndex with proleptic_gregorian calendar if input is array-like,
            or a single Timestamp if input is a single cftime object
    """
    # Check if input is a single cftime object (has year attribute directly)
    is_single_object = hasattr(cftime_times, 'year')
    
    # If single object, convert it to a list with one element
    if is_single_object:
        cftime_list = [cftime_times]
    else:
        cftime_list = cftime_times
    
    # Extract date components from cftime objects
    timestamps = []
    for t in cftime_list:
        # Extract time components from the cftime object
        dt_components = {
            'year': t.year,
            'month': t.month,
            'day': t.day,
            'hour': t.hour if hasattr(t, 'hour') else 0,
            'minute': t.minute if hasattr(t, 'minute') else 0,
            'second': t.second if hasattr(t, 'second') else 0
        }
        # Create a pandas timestamp with the same components (proleptic_gregorian)
        timestamps.append(pd.Timestamp(**dt_components))
    
    # Return either a single Timestamp or a DatetimeIndex based on input type
    if is_single_object:
        return timestamps[0]
    else:
        return pd.DatetimeIndex(timestamps)
    
def subset_ds_geolimit(
        ds_in,
        config,
        x_coordname=None,
        y_coordname=None,
        x_dimname=None,
        y_dimname=None,
):
    """
    Subset Xarray DataSet by lat/lon boundary.

    Args:
        ds_in: Xarray DataSet
            Input Xarray DataSet.
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        ds_out: Xarray DataSet
            Subsetted Xarray DataSset.
    """
    logger = logging.getLogger(__name__)
    # Get coordinate, dimension names from config if not supplied
    if x_coordname is None: x_coordname = config.get('x_coordname')
    if y_coordname is None: y_coordname = config.get('y_coordname')
    if x_dimname is None: x_dimname = config.get('x_dimname')
    if y_dimname is None: y_dimname = config.get('y_dimname')
    geolimits = config.get('geolimits')

    # Get coordinate variables
    lat = ds_in[y_coordname].data.squeeze()
    lon = ds_in[x_coordname].data.squeeze()
    # Check coordinate dimensions
    if (lat.ndim == 1) | (lon.ndim == 1):
        # Mesh 1D coordinate into 2D
        in_lon, in_lat = np.meshgrid(lon, lat)
    elif (lat.ndim == 2) | (lon.ndim == 2):
        in_lon = lon
        in_lat = lat
    else:
        logger.critical("ERROR in subset_ds_geolimit func: Unexpected input data x, y coordinate dimensions.")
        logger.critical(f"{x_coordname} dimension: {lon.ndim}")
        logger.critical(f"{y_coordname} dimension: {lat.ndim}")
        logger.critical("Tracking will now exit.")
        sys.exit()

    # Subset input dataset within geolimits
    # Find indices within lat/lon range set by geolimits
    indicesy, indicesx = np.array(
        np.where(
            (in_lat >= geolimits[0])
            & (in_lat <= geolimits[2])
            & (in_lon >= geolimits[1])
            & (in_lon <= geolimits[3])
        )
    )
    ymin, ymax = np.nanmin(indicesy), np.nanmax(indicesy) + 1
    xmin, xmax = np.nanmin(indicesx), np.nanmax(indicesx) + 1
    # Create a dictionary for dataset subset
    subset_dict = {
        y_dimname: slice(ymin, ymax),
        x_dimname: slice(xmin, xmax),
    }
    # Subset dataset
    ds_out = ds_in[subset_dict]
    # transpose coordinates to ensure LAT is always the 1st dimension
    # Use '...' to preserve any extra dimensions (e.g. nbnd)
    ds_out = ds_out.transpose(y_dimname, x_dimname, ...)
    return ds_out

def find_maxnclouds(config):
    """
    Find maximum number of clouds across track files.
    
    This function scans track_*.nc files and finds the maximum values of
    nclouds_ref and nclouds_new dimensions to determine the appropriate
    maxnclouds setting for tracking.
    
    Args:
        config: dictionary
            Dictionary containing config parameters.
    
    Returns:
        max_nclouds: int
            Maximum number of clouds found across all track files.
    """
    from netCDF4 import Dataset
    
    logger = logging.getLogger(__name__)
    
    # Get parameters from config
    tracking_outpath = config["tracking_outpath"]
    start_basetime = config["start_basetime"]
    end_basetime = config["end_basetime"]
    
    # Track file base name
    trackfile_filebase = "track_"
    
    logger.info("Scanning track files to find maximum number of clouds...")
    
    # Get list of track files within time range
    trackfiles, \
    trackfiles_basetime, \
    trackfiles_datestring, \
    trackfiles_timestring = subset_files_timerange(tracking_outpath,
                                                    trackfile_filebase,
                                                    start_basetime,
                                                    end_basetime)
    
    nfiles = len(trackfiles)
    logger.info(f"Scanning {nfiles} track files...")
    
    if nfiles == 0:
        logger.warning("No track files found")
        return 0
    
    # Initialize max values
    max_nclouds_ref = 0
    max_nclouds_new = 0
    
    # Loop through all files
    for i, filepath in enumerate(trackfiles):
        try:
            # Open the NetCDF file
            with Dataset(filepath, 'r') as ds:
                # Get dimension sizes
                nclouds_ref = len(ds.dimensions['nclouds_ref'])
                nclouds_new = len(ds.dimensions['nclouds_new'])
                
                # Update maximum values
                if nclouds_ref > max_nclouds_ref:
                    max_nclouds_ref = nclouds_ref
                
                if nclouds_new > max_nclouds_new:
                    max_nclouds_new = nclouds_new
            
            # Print progress every 500 files
            if (i + 1) % 500 == 0:
                logger.info(f"Processed {i + 1}/{nfiles} files...")
                
        except Exception as e:
            logger.warning(f"Error processing {filepath}: {e}")
            continue
    
    # Return the maximum of the two
    max_nclouds = max(max_nclouds_ref, max_nclouds_new)
    logger.info(f"Maximum nclouds_ref: {max_nclouds_ref}")
    logger.info(f"Maximum nclouds_new: {max_nclouds_new}")
    logger.info(f"Maximum nclouds across all files: {max_nclouds}")
    
    return max_nclouds


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
    datetime_drift_match = np.empty(ncloudidfiles, dtype='<U15')
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
        datetime_drift = bt_drift.dt.strftime("%Y%m%d_%H%M%S").values

        # Loop over each cloudid file time to find matching drfit data
        for itime in range(0, len(cloudidfiles_timestring) - 1):
            cloudid_datetime = cloudidfiles_datestring[itime] + '_' + cloudidfiles_timestring[itime]
            idx = np.where(datetime_drift == cloudid_datetime)[0]
            if (len(idx) == 1):
                datetime_drift_match[itime] = datetime_drift[idx[0]]
                xdrifts_match[itime] = xdrifts[idx[0]]
                ydrifts_match[itime] = ydrifts[idx[0]]
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
    nsparse_data = ds_all.sizes[sparse_dimname]
    ntracks = ds_all.sizes[tracks_dimname]
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
    nsparse_data = ds_all.sizes[sparse_dimname]
    ntracks = ds_all.sizes[tracks_dimname]
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


def compute_grid_area(lat, lon):
    """
    Compute the area of each grid cell (in km^2) for a 2D lat/lon grid
    using great-circle (haversine) distances between grid cell edges.

    Works for both regular and curvilinear 2D lat/lon grids.
    Input 1D lat/lon arrays are meshed to 2D internally.

    Args:
        lat: np.ndarray
            Latitude array, either 1D (ny,) or 2D (ny, nx), in degrees.
        lon: np.ndarray
            Longitude array, either 1D (nx,) or 2D (ny, nx), in degrees.

    Returns:
        grid_area: np.ndarray
            2D array (ny, nx) of grid cell areas in km^2.
    """
    R_earth = 6371.0  # Earth radius in km

    # Ensure 2D arrays
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    elif lat.ndim == 2 and lon.ndim == 2:
        lat2d = lat
        lon2d = lon
    else:
        raise ValueError(
            f"lat and lon must both be 1D or both be 2D. "
            f"Got lat.ndim={lat.ndim}, lon.ndim={lon.ndim}"
        )

    ny, nx = lat2d.shape

    # Compute cell edge coordinates as midpoints between grid centers
    # For interior edges, use midpoints; for boundary edges, extrapolate
    # Latitude edges: shape (ny+1, nx)
    lat_edges_y = np.empty((ny + 1, nx), dtype=np.float64)
    if ny > 1:
        lat_edges_y[1:-1, :] = 0.5 * (lat2d[:-1, :] + lat2d[1:, :])
        lat_edges_y[0, :] = lat2d[0, :] - 0.5 * (lat2d[1, :] - lat2d[0, :])
        lat_edges_y[-1, :] = lat2d[-1, :] + 0.5 * (lat2d[-1, :] - lat2d[-2, :])
    else:
        # Single row: assume symmetric half-cell around center
        # Use a default ~1 degree half-width if only 1 row
        half_dlat = 0.5 if nx == 1 else 0.5
        lat_edges_y[0, :] = lat2d[0, :] - half_dlat
        lat_edges_y[1, :] = lat2d[0, :] + half_dlat

    # Longitude edges: shape (ny, nx+1)
    lon_edges_x = np.empty((ny, nx + 1), dtype=np.float64)
    if nx > 1:
        lon_edges_x[:, 1:-1] = 0.5 * (lon2d[:, :-1] + lon2d[:, 1:])
        lon_edges_x[:, 0] = lon2d[:, 0] - 0.5 * (lon2d[:, 1] - lon2d[:, 0])
        lon_edges_x[:, -1] = lon2d[:, -1] + 0.5 * (lon2d[:, -1] - lon2d[:, -2])
    else:
        # Single column: assume symmetric half-cell around center
        half_dlon = 0.5
        lon_edges_x[:, 0] = lon2d[:, 0] - half_dlon
        lon_edges_x[:, 1] = lon2d[:, 0] + half_dlon

    # Compute dy (north-south distance) using haversine between top/bottom edges
    # at the cell center longitude
    # dy[j, i] = distance from (lat_edges_y[j], lon[j,i]) to (lat_edges_y[j+1], lon[j,i])
    lat_s = np.deg2rad(lat_edges_y[:-1, :])  # (ny, nx)
    lat_n = np.deg2rad(lat_edges_y[1:, :])    # (ny, nx)
    dlat = lat_n - lat_s
    dy = R_earth * np.abs(dlat)  # For pure N-S displacement, haversine simplifies

    # Compute dx (east-west distance) using haversine between left/right edges
    # at the cell center latitude
    # dx[j, i] = distance from (lat[j,i], lon_edges_x[j,i]) to (lat[j,i], lon_edges_x[j,i+1])
    lat_c = np.deg2rad(lat2d)  # (ny, nx)
    lon_w = np.deg2rad(lon_edges_x[:, :-1])  # (ny, nx)
    lon_e = np.deg2rad(lon_edges_x[:, 1:])    # (ny, nx)
    dlon = lon_e - lon_w
    # Haversine for pure E-W displacement: a = cos(lat)^2 * sin(dlon/2)^2
    a = np.cos(lat_c) ** 2 * np.sin(dlon / 2.0) ** 2
    dx = R_earth * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    grid_area = dx * dy
    return grid_area


def save_grid_area(grid_area, lat2d, lon2d, filepath):
    """
    Save grid_area and 2D lat/lon arrays to a netCDF file.

    Args:
        grid_area: np.ndarray
            2D array (ny, nx) of grid cell areas in km^2.
        lat2d: np.ndarray
            2D latitude array (ny, nx) in degrees.
        lon2d: np.ndarray
            2D longitude array (ny, nx) in degrees.
        filepath: str
            Output netCDF file path.

    Returns:
        filepath: str
            Path to the saved file.
    """
    logger = logging.getLogger(__name__)
    ny, nx = grid_area.shape
    ds = xr.Dataset(
        {
            "grid_area": (
                ["lat", "lon"],
                grid_area.astype(np.float32),
                {
                    "long_name": "Grid cell area",
                    "units": "km^2",
                },
            ),
            "latitude": (
                ["lat", "lon"],
                lat2d.astype(np.float32),
                {
                    "long_name": "Latitude",
                    "units": "degrees_north",
                },
            ),
            "longitude": (
                ["lat", "lon"],
                lon2d.astype(np.float32),
                {
                    "long_name": "Longitude",
                    "units": "degrees_east",
                },
            ),
        },
        coords={
            "lat": (["lat"], np.arange(ny)),
            "lon": (["lon"], np.arange(nx)),
        },
        attrs={
            "Title": "Grid cell area computed from latitude-dependent spacing",
            "Created_on": time.ctime(time.time()),
        },
    )
    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in ds.data_vars}
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    ds.to_netcdf(path=filepath, mode="w", format="NETCDF4", encoding=encoding)
    logger.info(f"Grid area file saved: {filepath}")
    return filepath


def load_grid_area(filepath):
    """
    Load grid_area from a netCDF file.

    Args:
        filepath: str
            Path to the grid area netCDF file.

    Returns:
        grid_area: np.ndarray
            2D array (ny, nx) of grid cell areas in km^2.
    """
    ds = xr.open_dataset(filepath)
    grid_area = ds["grid_area"].values
    ds.close()
    return grid_area


def get_pixel_area(config, latitude=None, longitude=None):
    """
    Get pixel area based on config settings.

    When area_method is "latlon", computes or loads latitude-dependent
    2D grid cell areas. Otherwise returns a scalar pixel_radius^2.

    Args:
        config: dict
            Configuration dictionary. Must contain 'pixel_radius'.
            Optionally contains 'area_method' and 'grid_area_file'.
        latitude: np.ndarray, optional
            Latitude array (1D or 2D). Required if area_method is "latlon"
            and grid_area file does not exist yet.
        longitude: np.ndarray, optional
            Longitude array (1D or 2D). Required if area_method is "latlon"
            and grid_area file does not exist yet.

    Returns:
        pixel_area: float or np.ndarray
            Scalar pixel_radius^2 (km^2) if area_method is "fixed",
            or 2D array (ny, nx) of grid cell areas (km^2) if "latlon".
    """
    logger = logging.getLogger(__name__)
    area_method = config.get("area_method", "fixed")
    pixel_radius = config["pixel_radius"]

    if area_method == "latlon":
        grid_area_file = config.get("grid_area_file", None)
        # If the file already exists, load it
        if grid_area_file is not None and os.path.isfile(grid_area_file):
            pixel_area = load_grid_area(grid_area_file)
            logger.debug(f"Loaded grid area from: {grid_area_file}")
        else:
            # Compute from lat/lon
            if latitude is None or longitude is None:
                raise ValueError(
                    "area_method is 'latlon' but no grid_area_file found and "
                    "latitude/longitude arrays were not provided."
                )
            pixel_area = compute_grid_area(latitude, longitude)
            # Save to file if grid_area_file path is defined
            if grid_area_file is not None:
                # Ensure 2D lat/lon for saving
                if latitude.ndim == 1 and longitude.ndim == 1:
                    lon2d, lat2d = np.meshgrid(longitude, latitude)
                else:
                    lat2d, lon2d = latitude, longitude
                save_grid_area(pixel_area, lat2d, lon2d, grid_area_file)
            logger.info(f"Computed grid area from lat/lon arrays.")
        return pixel_area
    else:
        # Fixed pixel area (backward compatible)
        return pixel_radius ** 2


def get_feature_area(pixel_area, feature_mask_indices):
    """
    Compute the total area of a feature from its pixel indices.

    Works with both scalar pixel_area (fixed) and 2D pixel_area (latlon).

    Args:
        pixel_area: float or np.ndarray
            Scalar pixel_radius^2 or 2D grid_area array (ny, nx) in km^2.
        feature_mask_indices: tuple of np.ndarray
            Tuple of (y_indices, x_indices) for the feature pixels.

    Returns:
        area: float
            Total area of the feature in km^2.
    """
    npix = len(feature_mask_indices[0])
    if np.ndim(pixel_area) == 0 or (isinstance(pixel_area, np.ndarray) and pixel_area.ndim == 0):
        # Scalar pixel area
        return npix * float(pixel_area)
    else:
        # 2D pixel area
        return np.sum(pixel_area[feature_mask_indices[0], feature_mask_indices[1]])


def get_mean_pixel_length(pixel_area, feature_mask_indices=None):
    """
    Get the representative pixel length scale (km) for a feature,
    used to convert regionprops pixel-unit lengths to physical units.

    For scalar pixel_area, returns sqrt(pixel_area).
    For 2D pixel_area, returns sqrt(mean(pixel_area over feature pixels)).

    Args:
        pixel_area: float or np.ndarray
            Scalar pixel_radius^2 or 2D grid_area array (ny, nx) in km^2.
        feature_mask_indices: tuple of np.ndarray, optional
            Tuple of (y_indices, x_indices) for the feature pixels.
            Required when pixel_area is 2D.

    Returns:
        pixel_length: float
            Representative pixel length scale in km.
    """
    if np.ndim(pixel_area) == 0 or (isinstance(pixel_area, np.ndarray) and pixel_area.ndim == 0):
        return np.sqrt(float(pixel_area))
    else:
        if feature_mask_indices is None:
            raise ValueError("feature_mask_indices required for 2D pixel_area")
        mean_area = np.mean(pixel_area[feature_mask_indices[0], feature_mask_indices[1]])
        return np.sqrt(mean_area)


def precompute_grid_area(config, first_file=None):
    """
    Pre-compute and save the grid area file for area_method='latlon'.

    Should be called once before parallel processing to avoid race conditions.
    Does nothing if area_method is not 'latlon' or if the grid area file
    already exists.

    Args:
        config: dict
            Configuration dictionary.
        first_file: str, optional
            Path to the first input data file (for NetCDF format).
            Not needed for Zarr format when landmask_filename is in config.
    """
    area_method = config.get("area_method", "fixed")
    if area_method != "latlon":
        return

    grid_area_file = config.get("grid_area_file", None)
    if grid_area_file is not None and os.path.isfile(grid_area_file):
        return

    logger = logging.getLogger(__name__)
    input_format = config.get("input_format", "netcdf")
    x_coordname = config.get("x_coordname", "lon")
    y_coordname = config.get("y_coordname", "lat")
    geolimits = config.get("geolimits", None)

    # Read lat/lon coordinates
    if input_format.lower() == "zarr":
        # For HEALPix/Zarr, coordinates come from the landmask file
        landmask_filename = config.get("landmask_filename", None)
        landmask_x_coordname = config.get("landmask_x_coordname", x_coordname)
        landmask_y_coordname = config.get("landmask_y_coordname", y_coordname)
        if landmask_filename is not None:
            ds = xr.open_dataset(landmask_filename)
            lat = ds[landmask_y_coordname].values
            lon = ds[landmask_x_coordname].values
            ds.close()
        else:
            raise ValueError(
                "area_method='latlon' with Zarr format requires "
                "'landmask_filename' in config to read lat/lon."
            )
    else:
        # NetCDF: open the first data file
        if first_file is None:
            raise ValueError(
                "first_file must be provided for NetCDF format "
                "to read lat/lon for grid area computation."
            )
        ds = xr.open_dataset(first_file, decode_timedelta=False)
        lat = ds[y_coordname].values
        lon = ds[x_coordname].values
        ds.close()

    # Mesh 1D to 2D if needed
    if lat.ndim == 1 or lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    else:
        lat2d, lon2d = lat, lon

    # Apply geolimit subsetting if defined
    if geolimits is not None:
        indicesy, indicesx = np.array(np.where(
            (lat2d >= geolimits[0]) & (lat2d <= geolimits[2]) &
            (lon2d >= geolimits[1]) & (lon2d <= geolimits[3])
        ))
        if len(indicesy) > 0 and len(indicesx) > 0:
            ymin = np.nanmin(indicesy)
            ymax = np.nanmax(indicesy) + 1
            xmin = np.nanmin(indicesx)
            xmax = np.nanmax(indicesx) + 1
            lat2d = lat2d[ymin:ymax, xmin:xmax]
            lon2d = lon2d[ymin:ymax, xmin:xmax]

    # Compute and save grid area
    get_pixel_area(config, latitude=lat2d, longitude=lon2d)
    logger.info(f"Pre-computed grid area file: {grid_area_file}")
