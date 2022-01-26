import numpy as np
import os, fnmatch, sys, glob
import datetime, calendar
from pytz import utc
import yaml
import xarray as xr
import logging

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
    geolimits = config.get("geolimits", [-90., -180., 90., 180.])

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
    filenames = sorted(fnmatch.filter(os.listdir(data_path), data_basename + '*.nc'))

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
    hh_idx = nleadingchar + hh_idx
    mm_idx = nleadingchar + mm_idx if (mm_idx != -1) else None
    ss_idx = nleadingchar + ss_idx if (ss_idx != -1) else None

    # Loop over each file
    for ii, ifile in enumerate(filenames):
        year = ifile[yyyy_idx:yyyy_idx+4]
        month = ifile[mo_idx:mo_idx+2]
        day = ifile[dd_idx:dd_idx+2]
        hour = ifile[hh_idx:hh_idx+2]
        # If minute, second is not in time_format, assume 0
        minute = ifile[mm_idx:mm_idx+2] if (mm_idx is not None) else '00'
        second = ifile[ss_idx:ss_idx+2] if (ss_idx is not None) else '00'

        TEMP_filetime = datetime.datetime(int(year), int(month), int(day),
                                          int(hour), int(minute), int(second), tzinfo=utc)
        files_basetime[ii] = calendar.timegm(TEMP_filetime.timetuple())
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
        bt_drift = ds_drift.basetime
        xdrifts = ds_drift.x.values
        ydrifts = ds_drift.y.values

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