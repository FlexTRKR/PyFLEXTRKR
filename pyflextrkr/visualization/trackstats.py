"""
Track statistics of a period as dictionaries, for the plot functions: the tracks that were initiated between a start and an
end time, with the arrays that the plot needs (times, positions, lifetime). The arrays stay lazily loaded from the file
(xarray), so the dictionary is cheap to send to Dask workers.

One function per kind of track statistics file:
    get_mcs_track_stats      MCS final statistics (cloud-system center, precipitation feature (PF) center and diameter)
    get_cell_track_stats     convective cell tracks (radar_cells tracking)
    get_generic_track_stats  generic feature tracks (meanlon / meanlat)
    get_cartesian_track_stats  tracks of a domain in x/y (meters or km, optionally scaled)

Called by: mcs_cli, cell_cli, generic_cli, cartesian_cli. Calls: xarray, numpy.
"""
import numpy as np
import xarray as xr

#-----------------------------------------------------------------------
def _select_tracks(dss, start_datetime, end_datetime):
    """
    Index of the tracks that were initiated within start/end datetime, and the time resolution [hour] of the file.
    """
    stats_starttime = dss['base_time'].isel(times=0)
    # Convert input datetime to np.datetime64
    stime = np.datetime64(start_datetime)
    etime = np.datetime64(end_datetime)
    time_res = dss.attrs['time_resolution_hour']

    # Find tracks initiated within the time window
    idx = np.where((stats_starttime >= stime) & (stats_starttime <= etime))[0]
    ntracks = len(idx)
    print(f'Number of tracks within input period: {ntracks}')
    if ntracks == 0:
        # An empty index array together with an integer index (nmaxpf=0) fails in the lazy indexing of xarray, an empty slice does not
        idx = slice(0, 0)
    return idx, ntracks, time_res

#-----------------------------------------------------------------------
def get_mcs_track_stats(trackstats_file, start_datetime, end_datetime, dt_thres, use_smooth=True):
    """
    Subset tracks statistics data within start/end datetime

    Args:
        trackstats_file: string
            Track statistics file name.
        start_datetime: string
            Start datetime to subset tracks.
        end_datetime: dstring
            End datetime to subset tracks.
        dt_thres: timedelta
            A timedelta threshold to retain tracks.
        use_smooth: bool
            Use the smoothed track center (meanlon_smooth, meanlat_smooth) if the file has it (default True).

    Returns:
        track_dict: dictionary
            Dictionary containing track stats data.
    """
    # Read track stats file
    dss = xr.open_dataset(trackstats_file)
    idx, ntracks, time_res = _select_tracks(dss, start_datetime, end_datetime)

    # Check if smoothed coordinates exist, otherwise use regular coordinates
    if use_smooth and 'meanlon_smooth' in dss.variables and 'meanlat_smooth' in dss.variables:
        track_ccs_lon = dss['meanlon_smooth'].isel(tracks=idx)
        track_ccs_lat = dss['meanlat_smooth'].isel(tracks=idx)
    else:
        track_ccs_lon = dss['meanlon'].isel(tracks=idx)
        track_ccs_lat = dss['meanlat'].isel(tracks=idx)

    # Check if PF centroid coordinates exist, otherwise use PF mean lat/lon coordinates
    if 'pf_lon_centroid' in dss.variables and 'pf_lat_centroid' in dss.variables:
        track_pf_lon = dss['pf_lon_centroid'].isel(tracks=idx, nmaxpf=0)
        track_pf_lat = dss['pf_lat_centroid'].isel(tracks=idx, nmaxpf=0)
        track_pf_diam = 2 * np.sqrt(dss['pf_area'].isel(tracks=idx, nmaxpf=0) / np.pi)
    elif 'pf_lon' in dss.variables:
        track_pf_lon = dss['pf_lon'].isel(tracks=idx, nmaxpf=0)
        track_pf_lat = dss['pf_lat'].isel(tracks=idx, nmaxpf=0)
        track_pf_diam = 2 * np.sqrt(dss['pf_area'].isel(tracks=idx, nmaxpf=0) / np.pi)
    else:
        # If PF variables do not exist (Tb-only or Tb + radar MCS tracking), use CCS variables
        track_pf_lon = track_ccs_lon
        track_pf_lat = track_ccs_lat
        # Replace all valid values as NaN (PF diameter circles will not be plotted)
        track_pf_diam = track_pf_lon.where(np.isnan(track_pf_lon), other=np.nan)

    # Subset these tracks and put in a dictionary
    track_dict = {
        'ntracks': ntracks,
        'lifetime': dss['track_duration'].isel(tracks=idx) * time_res,
        'track_bt': dss['base_time'].isel(tracks=idx),
        'track_ccs_lon': track_ccs_lon,
        'track_ccs_lat': track_ccs_lat,
        'track_pf_lon': track_pf_lon,
        'track_pf_lat': track_pf_lat,
        'track_pf_diam': track_pf_diam,
        'dt_thres': dt_thres,
        'time_res': time_res,
    }

    return track_dict

#-----------------------------------------------------------------------
def get_cell_track_stats(trackstats_file, start_datetime, end_datetime, dt_thres, min_hours=0.0):
    """
    Subset tracks statistics data within start/end datetime

    Args:
        trackstats_file: string
            Track statistics file name.
        start_datetime: string
            Start datetime to subset tracks.
        end_datetime: dstring
            End datetime to subset tracks.
        dt_thres: timedelta
            A timedelta threshold to retain tracks.
        min_hours: float, optional
            Only the tracks that live at least this long [hour] (0: all). The track numbers (track index + 1 in the
            whole file) of these tracks are returned as 'tracknumbers_all' (None: all tracks), to take the other
            tracks out of the pixel-level masks.

    Returns:
        track_dict: dictionary
            Dictionary containing track stats data.
    """
    # Read track stats file
    dss = xr.open_dataset(trackstats_file)
    idx, ntracks, time_res = _select_tracks(dss, start_datetime, end_datetime)

    # Only the tracks that live at least min_hours
    tracknumbers_all = None
    if min_hours and min_hours > 0:
        tracknumbers_all = np.where(dss['track_duration'].values * time_res >= min_hours)[0] + 1
        if ntracks > 0:
            idx = idx[np.isin(idx + 1, tracknumbers_all)]
            ntracks = len(idx)
            if ntracks == 0:
                idx = slice(0, 0)
        print(f'Number of tracks that live at least {min_hours} h: {ntracks}')

    # Calculate cell lifetime
    lifetime = dss.track_duration.isel(tracks=idx) * time_res

    # # Select long-lived tracks
    # idx_long = np.where(lifetime > lifetime_longtracks)[0]
    # ntracks_long = len(idx_long)
    # lifetime_long = lifetime.isel(tracks=idx_long)
    # print(f'Number of long tracks within input period: {ntracks_long}')

    # Subset these tracks and put in a dictionary
    track_dict = {
        'ntracks' : ntracks,
        'tracknumbers_all' : tracknumbers_all,
        'lifetime' : lifetime,
        'cell_bt' : dss['base_time'].isel(tracks=idx),
        'cell_lon' : dss['cell_meanlon'].isel(tracks=idx),
        'cell_lat' : dss['cell_meanlat'].isel(tracks=idx),
        'start_split_tracknumber' : dss['start_split_tracknumber'].isel(tracks=idx),
        'end_merge_tracknumber' : dss['end_merge_tracknumber'].isel(tracks=idx),
        'dt_thres': dt_thres,
        'time_res': time_res,
    }

    return track_dict

#-----------------------------------------------------------------------
def get_generic_track_stats(trackstats_file, start_datetime, end_datetime, dt_thres):
    """
    Subset tracks statistics data within start/end datetime

    Args:
        trackstats_file: string
            Track statistics file name.
        start_datetime: string
            Start datetime to subset tracks.
        end_datetime: dstring
            End datetime to subset tracks.
        dt_thres: timedelta
            A timedelta threshold to retain tracks.

    Returns:
        track_dict: dictionary
            Dictionary containing track stats data.
    """
    # Read track stats file
    dss = xr.open_dataset(trackstats_file)
    idx, ntracks, time_res = _select_tracks(dss, start_datetime, end_datetime)

    # Calculate track lifetime
    lifetime = dss.track_duration.isel(tracks=idx) * time_res

    # Subset these tracks and put in a dictionary
    track_dict = {
        'ntracks': ntracks,
        'lifetime': lifetime,
        'base_time': dss['base_time'].isel(tracks=idx),
        'meanlon': dss['meanlon'].isel(tracks=idx),
        'meanlat': dss['meanlat'].isel(tracks=idx),
        # 'start_split_tracknumber': dss['start_split_tracknumber'].isel(tracks=idx),
        # 'end_merge_tracknumber': dss['end_merge_tracknumber'].isel(tracks=idx),
        'dt_thres': dt_thres,
        'time_res': time_res,
    }

    return track_dict

#-----------------------------------------------------------------------
def get_cartesian_track_stats(trackstats_file, start_datetime, end_datetime, dt_thres, xscale=1, yscale=1, with_pf=True):
    """
    Subset tracks statistics data within start/end datetime for a domain in x/y (no map): the track positions are
    meanlon / meanlat (and the PF positions) times xscale / yscale (e.g. 1e-3 for meters -> kilometers).

    Args:
        trackstats_file, start_datetime, end_datetime, dt_thres: as in get_mcs_track_stats
        xscale, yscale: float, scale factors of the x and y positions
        with_pf: bool, also return the positions of the first precipitation feature (pf_lon, pf_lat)

    Returns:
        track_dict: dictionary
            Dictionary containing track stats data.
    """
    # Read track stats file
    dss = xr.open_dataset(trackstats_file)
    idx, ntracks, time_res = _select_tracks(dss, start_datetime, end_datetime)

    # Calculate track lifetime
    lifetime = dss.track_duration.isel(tracks=idx) * time_res

    # Subset these tracks and put in a dictionary
    track_dict = {
        'ntracks': ntracks,
        'lifetime': lifetime,
        'track_bt': dss['base_time'].isel(tracks=idx),
        'track_meanlon': dss['meanlon'].isel(tracks=idx) * xscale,
        'track_meanlat': dss['meanlat'].isel(tracks=idx) * yscale,
        'dt_thres': dt_thres,
        'time_res': time_res,
    }
    if with_pf:
        track_dict['track_pflon'] = dss['pf_lon'].isel(tracks=idx, nmaxpf=0) * xscale
        track_dict['track_pflat'] = dss['pf_lat'].isel(tracks=idx, nmaxpf=0) * yscale

    return track_dict
