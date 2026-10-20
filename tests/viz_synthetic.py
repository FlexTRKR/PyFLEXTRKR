"""
Synthetic data for the visualization tests (tests/test_viz_*.py): pixel-level files and track statistics on a small
lon/lat grid, and the config dictionary that the command lines of the plot families read (the tests replace
load_config by it, so no config file and no tracking run are needed).

Two tracks cross a 60 x 40 grid (100-130E, 10S-10N): track 1 lives 3 hourly frames, track 2 lives 2; the fourth frame
has no object at all. A blob of tracknumber / Tb / precipitation moves 3 pixels east per frame. A few KB of files.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr

NY, NX = 40, 60
DLON = DLAT = 0.5
LON0, LAT0 = 100.0, -10.0
NFRAMES = 4
TIMES = pd.date_range('2025-06-01T00:30', periods=NFRAMES, freq='h')
START, END = '2025-06-01T00', '2025-06-01T04'      # the period of the command lines (all frames)
EXTENT = [100.0, 130.0, -10.0, 10.0]               # the lon/lat box of the grid
RADIUS = 4                                         # pixels
# track number -> row and column of the center at the first frame, number of frames the track lives
TRACKS = {1: dict(row=15, col=15, nframes=3), 2: dict(row=28, col=40, nframes=2)}
FILEBASE = 'mcstrack_'

#-----------------------------------------------------------------------
def _grid():
    lon = LON0 + DLON * np.arange(NX)
    lat = LAT0 + DLAT * np.arange(NY)
    lon2d, lat2d = np.meshgrid(lon, lat)
    return lon, lat, lon2d, lat2d

def _blob(track, frame):
    """Boolean (NY, NX) mask of a track at a frame (empty after the track has ended)."""
    t = TRACKS[track]
    if frame >= t['nframes']:
        return np.zeros((NY, NX), dtype=bool)
    iy, ix = np.ogrid[:NY, :NX]
    return (iy - t['row'])**2 + (ix - (t['col'] + 3 * frame))**2 <= RADIUS**2

def _centers():
    """Center (lon, lat) of the tracks, arrays (tracks, times) with NaN after the end of a track."""
    lon = np.full((len(TRACKS), NFRAMES), np.nan)
    lat = np.full((len(TRACKS), NFRAMES), np.nan)
    for i, t in enumerate(TRACKS.values()):
        for f in range(t['nframes']):
            lon[i, f] = LON0 + DLON * (t['col'] + 3 * f)
            lat[i, f] = LAT0 + DLAT * t['row']
    return lon, lat

def _base_time():
    bt = np.full((len(TRACKS), NFRAMES), np.datetime64('NaT'), dtype='datetime64[ns]')
    for i, t in enumerate(TRACKS.values()):
        for f in range(t['nframes']):
            bt[i, f] = TIMES[f].to_datetime64()
    return bt

#-----------------------------------------------------------------------
def write_pixel_files(pixel_dir, filebase=FILEBASE):
    """
    Pixel-level files of the MCS, generic and x/y families (one time each): tb, precipitation, reflectivity_comp,
    cloudtracknumber (NaN outside the objects) on a 2D longitude / latitude grid. Returns the list of files.
    """
    os.makedirs(pixel_dir, exist_ok=True)
    _, _, lon2d, lat2d = _grid()
    files = []
    for f, t in enumerate(TIMES):
        tn = np.full((NY, NX), np.nan, dtype='float32')
        tb = np.full((NY, NX), 285.0, dtype='float32')
        pcp = np.zeros((NY, NX), dtype='float32')
        dbz = np.full((NY, NX), np.nan, dtype='float32')
        for track in TRACKS:
            m = _blob(track, f)
            tn[m] = track
            tb[m] = 200.0 + 8.0 * track
            pcp[m] = 4.0 + 4.0 * track
            dbz[m] = 25.0 + 10.0 * track
        ds = xr.Dataset(
            {
                'tb': (('time', 'y', 'x'), tb[None]),
                'precipitation': (('time', 'y', 'x'), pcp[None]),
                'reflectivity_comp': (('time', 'y', 'x'), dbz[None]),
                'cloudtracknumber': (('time', 'y', 'x'), tn[None]),
                'longitude': (('y', 'x'), lon2d),
                'latitude': (('y', 'x'), lat2d),
            },
            coords={'time': [t.to_datetime64()]},
        )
        path = f"{pixel_dir}/{filebase}{t:%Y%m%d_%H%M%S}.nc"
        ds.to_netcdf(path)
        files.append(path)
    return files

def write_cell_pixel_files(pixel_dir, filebase='celltracking_'):
    """
    Pixel-level files of the convective cell family: tracknumber, conv_mask, dbz_comp and the x/y coordinates (km) and
    longitude / latitude of the grid. Returns the list of files.
    """
    os.makedirs(pixel_dir, exist_ok=True)
    lon, lat, lon2d, lat2d = _grid()
    xx = (lon - lon.mean()) * 111.0     # km from the center, only used for the cell centers
    yy = (lat - lat.mean()) * 111.0
    files = []
    for f, t in enumerate(TIMES):
        tn = np.full((NY, NX), np.nan, dtype='float32')
        conv = np.zeros((NY, NX), dtype='float32')
        dbz = np.full((NY, NX), 5.0, dtype='float32')
        for track in TRACKS:
            m = _blob(track, f)
            tn[m] = track
            conv[m] = 1
            dbz[m] = 30.0 + 10.0 * track
        ds = xr.Dataset(
            {
                'tracknumber': (('time', 'y', 'x'), tn[None]),
                'conv_mask': (('time', 'y', 'x'), conv[None]),
                'dbz_comp': (('time', 'y', 'x'), dbz[None]),
                'lon': (('x',), xx),
                'lat': (('y',), yy),
                'longitude': (('y', 'x'), lon2d),
                'latitude': (('y', 'x'), lat2d),
            },
            coords={'time': [t.to_datetime64()]},
        )
        path = f"{pixel_dir}/{filebase}{t:%Y%m%d_%H%M%S}.nc"
        ds.to_netcdf(path)
        files.append(path)
    return files

def write_terrain_file(path):
    """Terrain height [m] on the grid of the cell pixel files (0 at the west edge to 3000 m at the east edge), with the
    variable names of the CSAPR terrain file: hgt, longitude, latitude."""
    _, _, lon2d, lat2d = _grid()
    hgt = 3000.0 * (lon2d - LON0) / (NX * DLON)
    xr.Dataset({'hgt': (('y', 'x'), hgt.astype('float32')), 'longitude': (('y', 'x'), lon2d),
                'latitude': (('y', 'x'), lat2d)}).to_netcdf(path)
    return path

#-----------------------------------------------------------------------
def write_track_stats(path, mcs=True):
    """
    Track statistics file: the final MCS statistics (mcs=True: meanlon / meanlat / pf_* ...) or the cell tracks
    (mcs=False: cell_meanlon / cell_meanlat / start_split_tracknumber / end_merge_tracknumber).
    """
    lon, lat = _centers()
    ntracks = len(TRACKS)
    data = {
        'base_time': (('tracks', 'times'), _base_time()),
        'track_duration': (('tracks',), np.array([t['nframes'] for t in TRACKS.values()])),
    }
    if mcs:
        area = np.where(np.isnan(lon), np.nan, np.pi * 10.0**2)     # a precipitation feature of 20 km diameter
        data.update({
            'meanlon': (('tracks', 'times'), lon),
            'meanlat': (('tracks', 'times'), lat),
            'pf_lon_centroid': (('tracks', 'times', 'nmaxpf'), lon[..., None]),
            'pf_lat_centroid': (('tracks', 'times', 'nmaxpf'), lat[..., None]),
            'pf_lon': (('tracks', 'times', 'nmaxpf'), lon[..., None]),
            'pf_lat': (('tracks', 'times', 'nmaxpf'), lat[..., None]),
            'pf_area': (('tracks', 'times', 'nmaxpf'), area[..., None]),
        })
    else:
        data.update({
            'cell_meanlon': (('tracks', 'times'), lon),
            'cell_meanlat': (('tracks', 'times'), lat),
            'start_split_tracknumber': (('tracks',), np.full(ntracks, np.nan)),
            'end_merge_tracknumber': (('tracks',), np.full(ntracks, np.nan)),
        })
    xr.Dataset(data, attrs={'time_resolution_hour': 1.0}).to_netcdf(path)
    return path

#-----------------------------------------------------------------------
def build(root):
    """
    All the synthetic inputs under the directory root, and the config dictionary that points to them.

    Returns:
        configs: dictionaries with the keys that the command lines read from load_config(): 'mcs' (also used for the
            generic and x/y families) and 'cell'
        paths: dictionary with the file names: mcs_stats, cell_stats, mcs_pixel_dir, cell_pixel_dir
    """
    root = str(root)
    stats_dir = f"{root}/stats/"
    os.makedirs(stats_dir, exist_ok=True)
    mcs_stats = write_track_stats(f"{stats_dir}mcs_tracks_final_20250601.0000_20250601.0400.nc", mcs=True)
    cell_stats = write_track_stats(f"{stats_dir}trackstats_final_20250601.0000_20250601.0400.nc", mcs=False)
    mcs_pixel_dir = f"{root}/mcs_pixel/"
    cell_pixel_dir = f"{root}/cell_pixel/"
    write_pixel_files(mcs_pixel_dir)
    write_cell_pixel_files(cell_pixel_dir)
    # the Tb-only MCS tracking has its own statistics and pixel-level folder (named from the config)
    write_track_stats(f"{stats_dir}mcs_tracks_tb_20250601.0000_20250601.0400.nc", mcs=True)
    write_pixel_files(f"{root}/mcstracking_tb/20250601.0000_20250601.0400/")
    mcs_config = {
        'root_path': f"{root}/",
        'startdate': '20250601.0000',
        'enddate': '20250601.0400',
        'nprocesses': 2,
        'stats_outpath': stats_dir,
        'pixeltracking_outpath': mcs_pixel_dir,
        'pixeltracking_filebase': FILEBASE,
        'trackstats_filebase': 'mcs_tracks_final_',
        'finalstats_filebase': 'mcs_tracks_final_',
        'mcsfinal_filebase': 'mcs_tracks_final_',
        'mcstbstats_filebase': 'mcs_tracks_tb_',
        'pixel_radius': 10.0,
        'datatimeresolution': 1.0,
        'field_varname': 'tb',
    }
    terrain_file = write_terrain_file(f"{root}/terrain.nc")
    cell_config = dict(mcs_config, pixeltracking_outpath=cell_pixel_dir, pixeltracking_filebase='celltracking_',
                       trackstats_filebase='trackstats_final_', terrain_file=terrain_file, elev_varname='hgt')
    paths = {'mcs_stats': mcs_stats, 'cell_stats': cell_stats, 'mcs_pixel_dir': mcs_pixel_dir, 'cell_pixel_dir': cell_pixel_dir,
             'terrain_file': terrain_file}
    return {'mcs': mcs_config, 'cell': cell_config}, paths
