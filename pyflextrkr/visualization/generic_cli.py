r"""
Command line of the generic feature track maps (visualization/plot_generic_tracks.py): arguments, the plotting information,
the tracks and files of the period, and the loop over the pixel-level files.

Plots the tracks of generic features (feature_type 'generic': Z500 anomaly, stream function, vorticity, ...) on snapshots of the
tracked 2D field. The field is `field_varname` of the tracking config; levels, colormap and label are options (the defaults
are for the Z500 anomaly demo).

Usage:
    python plot_generic_tracks.py -s STARTDATE -e ENDDATE -c CONFIG.yml
    python plot_generic_tracks.py -s STARTDATE -e ENDDATE -c CONFIG.yml --cmap RdBu_r --levels -4 4 0.2 --cbticks -4 4 1 \
        --cblabel 'Z500 Anomaly (m$^{2}$ s$^{-1}$)'

Called by: visualization/plot_generic_tracks.py, and the stubs in Analysis/ that keep the old plot_subset_generic_tracks_demo.py
and plot_subset_psi_tracks_demo.py commands working (main(legacy_name=...)).
Calls: cli_common (arguments), trackstats.get_generic_track_stats, runner.run_frames(generic_map.work_for_time_loop).
"""
import argparse
import datetime
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import warnings
from pyflextrkr.ft_utilities import load_config, subset_files_timerange
from pyflextrkr.visualization import cli_common, legacy
from pyflextrkr.visualization.mapcore import (
    projection_key, panel_fractions, figsize_for_aspect, check_lat_range, auto_map_domain, min_figure_height,
    map_feature_styles,
)
from pyflextrkr.visualization.runner import run_frames
from pyflextrkr.visualization.trackstats import get_generic_track_stats
from pyflextrkr.visualization import generic_map

#-----------------------------------------------------------------------
def _range(values):
    """Array of the 3 numbers vmin vmax step (vmax is included, as np.arange(vmin, vmax + step/2, step))."""
    vmin, vmax, step = values
    return np.arange(vmin, vmax + step / 2., step)

#-----------------------------------------------------------------------
def parse_cmd_args(argv=None, legacy_name=None):
    # Define and retrieve the command-line arguments...
    parser = argparse.ArgumentParser(
        description="Plot tracks on input data snapshots."
    )
    cli_common.add_common_arguments(parser)
    parser.set_defaults(workers=None)
    parser.add_argument("--field_varname", default=None, help="variable of the pixel-level files to plot (default: field_varname of the config)")
    parser.add_argument("--cmap", default=None, help="colormap of the field (matplotlib or colorcet name, default RdBu_r)")
    parser.add_argument("--levels", type=float, nargs=3, default=None, metavar=('VMIN', 'VMAX', 'STEP'),
                        help="shading levels: first, last and step (default: -4 4 0.2)")
    parser.add_argument("--cbticks", type=float, nargs=3, default=None, metavar=('VMIN', 'VMAX', 'STEP'),
                        help="colorbar ticks: first, last and step (default: -4 4 1)")
    parser.add_argument("--cblabel", default=None, help="colorbar label (default: Z500 Anomaly)")
    parser.add_argument("--remove_oob_low", type=int, default=0, help="mask the values below the lowest level (0:no, 1:yes)")
    parser.add_argument("--remove_oob_high", type=int, default=0, help="mask the values above the highest level (0:no, 1:yes)")
    parser.add_argument("--perim_width", type=int, default=1, help="width of the outline of the tracked features in pixels")
    # Used by the defaults of the old stream-function script only
    parser.add_argument("--track_number_source", default='track', choices=['track', 'mask'], help=argparse.SUPPRESS)
    parser.add_argument("--show_symbols", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--time_label_format", default="%Y-%m-%d %H:%M:%S UTC", help=argparse.SUPPRESS)
    parser.add_argument("--figname_time_format", default="%Y%m%d_%H%M%S", help=argparse.SUPPRESS)
    if legacy_name is not None:
        # Defaults of the old script that is being replaced
        parser.set_defaults(**legacy.defaults(legacy_name))
    args = parser.parse_args(argv)
    return cli_common.args_to_dict(args)

#-----------------------------------------------------------------------
def main(argv=None, legacy_name=None):
    """
    Make the generic track maps.

    Args:
        argv: list of string, command-line arguments (default: sys.argv[1:])
        legacy_name: string, name of an old Analysis script whose defaults are used (the stubs in Analysis/)
    """

    # Non-interactive matplotlib backend
    plt.switch_backend('agg')
    warnings.filterwarnings("ignore")

    # Get the command-line arguments...
    if legacy_name is not None:
        legacy.announce(legacy_name)
    args_dict = parse_cmd_args(argv, legacy_name)
    start_datetime = args_dict.get('start_datetime')
    end_datetime = args_dict.get('end_datetime')
    run_parallel = args_dict.get('run_parallel')
    n_workers = args_dict.get('workers')
    config_file = args_dict.get('config_file')
    map_extent = args_dict.get('extent')
    subset = args_dict.get('subset')
    figsize = args_dict.get('figsize')
    figsize_x = args_dict.get('figsize_x', 10)
    out_dir = args_dict.get('out_dir')
    figbasename = args_dict.get('figbasename')
    figname_type = args_dict.get('figname_type')
    trackstats_file = args_dict.get('trackstats_file')
    pixeltracking_path = args_dict.get('pixeltracking_path')
    time_format = args_dict.get('time_format')
    projection = args_dict.get('projection')
    lat_range = args_dict.get('lat_range')
    map_aspect = args_dict.get('map_aspect') or 'equal'
    check_lat_range(projection, lat_range)
    draw_land = args_dict.get('draw_land') if args_dict.get('draw_land') is not None else 1
    draw_border = args_dict.get('draw_border') if args_dict.get('draw_border') is not None else 1
    draw_state = args_dict.get('draw_state') if args_dict.get('draw_state') is not None else 0
    draw_river = args_dict.get('draw_river', 0)
    lookback_days = args_dict.get('track_lookback_days') if args_dict.get('track_lookback_days') is not None else 4

    if time_format is None: time_format = "yyyymodd_hhmmss"

    # Determine the figsize (map_aspect 'auto': the size of the old script for PlateCarree; otherwise the shape of the map
    # decides it, below, after the data are known)
    if (figsize is None):
        if projection_key(projection) == 'platecarree' and map_aspect == 'auto':
            figsize = [8, 7]
        else:
            # The figure has the shape of the map (set below, after the data are known)
            pass

    # Specify plotting info
    cmap = args_dict.get('cmap') if args_dict.get('cmap') is not None else 'RdBu_r'   # colormap
    levels = _range(args_dict['levels']) if args_dict.get('levels') is not None else np.arange(-4, 4.01, 0.2)  # shading levels
    cbticks = _range(args_dict['cbticks']) if args_dict.get('cbticks') is not None else np.arange(-4, 4.01, 1)  # colorbar ticks
    cblabels = args_dict.get('cblabel') if args_dict.get('cblabel') is not None else 'Z500 Anomaly (m$^{2}$ s$^{-1}$)'  # colorbar label
    # cmap = 'gist_ncar'   # colormap
    # levels = np.arange(0, 70.01, 5)  # shading levels
    # cbticks = np.arange(0, 70.01, 10)  # colorbar ticks
    # cblabels = 'Reflectivity (dBZ)'  # colorbar label
    plot_info = {
        'fontsize': 13,     # plot font size
        'dpi': 200,
        'cmap': cmap,
        'levels': levels,
        'cbticks': cbticks,
        'cblabels': cblabels,
        'field_varname': args_dict.get('field_varname'),
        'remove_oob_low': bool(args_dict.get('remove_oob_low')),   # mask out-of-bounds low values (< min(levels))
        'remove_oob_high': bool(args_dict.get('remove_oob_high')),  # mask out-of-bounds high values (> max(levels))
        'marker_size': 10,   # track symbol marker size
        'tracknumber_fontsize': 10,
        'perim_width': args_dict.get('perim_width', 1),  # width of the track feature perimeter (number of pixels)
        'trackpath_linewidth': 1.5, # track path line width
        'trackpath_color': 'blueviolet',    # track path color
        'map_resolution': args_dict.get('map_resolution'),   # background map resolution ('110m', '50m', 10m')
        'figsize': figsize,
        'figbasename': figbasename,
        'figname_type': figname_type,
        'map_aspect': map_aspect,
        'track_number_source': args_dict.get('track_number_source'),
        'show_symbols': bool(args_dict.get('show_symbols')),
        'time_label_format': args_dict.get('time_label_format'),
        'figname_time_format': args_dict.get('figname_time_format'),
    }
    if args_dict.get('dpi') is not None:
        plot_info['dpi'] = args_dict['dpi']
    if args_dict.get('fontsize') is not None:
        plot_info['fontsize'] = args_dict['fontsize']
    cli_common.apply_track_style_args(plot_info, args_dict)

    # Customize lat/lon labels
    lonv = args_dict.get('lon_ticks')
    latv = args_dict.get('lat_ticks')
    # Put map info in a dictionary
    map_info = {
        'map_extent': map_extent,
        'subset': subset,
        'lonv': lonv,
        'latv': latv,
        'draw_land': bool(draw_land),
        'draw_border': bool(draw_border),
        'draw_state': bool(draw_state),
        'draw_river': bool(draw_river),
        'map_feature_styles': map_feature_styles(args_dict),
        'projection': projection,
        'central_longitude': args_dict.get('central_longitude'),
        'central_latitude': args_dict.get('central_latitude'),
        'standard_parallels': args_dict.get('standard_parallels'),
        'lat_range': lat_range,
    }
    if args_dict.get('grid_linewidth') is not None:
        map_info['grid_linewidth'] = args_dict['grid_linewidth']
    if args_dict.get('grid_alpha') is not None:
        map_info['grid_alpha'] = args_dict['grid_alpha']

    # Track stats file
    config = load_config(config_file, make_dirs=False)      # reading a config creates no folders
    stats_path = config["stats_outpath"]
    if pixeltracking_path is None:
        pixeltracking_path = config["pixeltracking_outpath"]
    pixeltracking_filebase = config["pixeltracking_filebase"]
    trackstats_filebase = config["trackstats_filebase"]
    finalstats_filebase = config.get("finalstats_filebase", None)
    startdate = config["startdate"]
    enddate = config["enddate"]
    if n_workers is None:
        n_workers = config["nprocesses"]
    datatimeresolution = config["datatimeresolution"]  # hour

    # Tracks that end longer than this threshold from the current pixel-level frame are not plotted
    # This treshold controls the time window to retain previous tracks
    track_retain_time_min = (datatimeresolution * 60)

    # Create a timedelta threshold in minutes
    dt_thres = datetime.timedelta(minutes=track_retain_time_min)

    # If trackstats_file is not specified
    if trackstats_file is None:
        # If finalstats_filebase is present, use it (links merge/split tracks)
        if finalstats_filebase is None:
            trackstats_file = f"{stats_path}{trackstats_filebase}{startdate}_{enddate}.nc"
        else:
            trackstats_file = f"{stats_path}{finalstats_filebase}{startdate}_{enddate}.nc"

    # Output figure directory
    if out_dir is None:
        figdir = f'{pixeltracking_path}quicklooks_trackpaths/'
    else:
        figdir = out_dir if out_dir.endswith('/') else f'{out_dir}/'
    os.makedirs(figdir, exist_ok=True)
    # Add to plot_info dictionary
    plot_info['figdir'] = figdir

    # Convert datetime string to Epoch time (base time)
    start_basetime = pd.to_datetime(start_datetime).timestamp()
    end_basetime = pd.to_datetime(end_datetime).timestamp()
    # Subtract start_datetime by TimeDelta to include tracks
    # that start before the start_datetime but may not have ended yet
    TimeDelta = pd.Timedelta(days=lookback_days)
    start_datetime_4stats = (pd.to_datetime(start_datetime) - TimeDelta).strftime('%Y-%m-%dT%H:%M:%S')

    # Find all pixel-level files that match the input datetime
    datafiles, \
    datafiles_basetime, \
    datafiles_datestring, \
    datafiles_timestring = subset_files_timerange(
        pixeltracking_path,
        pixeltracking_filebase,
        start_basetime,
        end_basetime,
        time_format=time_format,
    )
    print(f'Number of pixel files: {len(datafiles)}')
    if len(datafiles) == 0:
        sys.exit(f'No pixel-level files {pixeltracking_filebase}* found in {pixeltracking_path} for the period')

    # The shape of the map decides the figure size when it was not given. Without --extent and with a projection other than
    # PlateCarree the domain of the map is the domain of the data in the coordinates of the projection
    if figsize is None or (map_extent is None and projection_key(projection) != 'platecarree'):
        with xr.open_dataset(datafiles[0]) as ds0:
            data_lon, data_lat = ds0['longitude'].values, ds0['latitude'].values
        extent_xy, aspect_yx = auto_map_domain(map_extent, projection, map_info['central_longitude'], map_info['central_latitude'],
                                               map_info['standard_parallels'], lat_range, data_lon, data_lat)
        if extent_xy is not None:
            map_info['map_extent_xy'] = extent_xy
    if figsize is None:
        panel_w, panel_h = panel_fractions(**generic_map.GRID_MAP['gridspec'], **generic_map.GRID_MAP['update'])
        figsize = figsize_for_aspect(figsize_x, aspect_yx, panel_w, panel_h)
        # (not shorter than the titles need)
        figsize[1] = max(figsize[1], round(min_figure_height(plot_info['fontsize'], 1 - generic_map.GRID_MAP['update']['top']), 2))
        plot_info['figsize'] = figsize
    print(f'Figure size (width, height) in inches: {figsize}')

    # Get track stats data
    track_dict = get_generic_track_stats(trackstats_file, start_datetime_4stats, end_datetime, dt_thres)

    # Serial option (run_parallel=0) or parallel option (run_parallel=1)
    run_frames(generic_map.work_for_time_loop, datafiles, track_dict, map_info, plot_info, config,
               run_parallel=run_parallel, n_workers=n_workers)
