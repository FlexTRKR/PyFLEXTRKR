"""
Command line of the convective cell track maps (visualization/plot_cell_tracks.py): arguments, the plotting information,
the tracks and files of the period, and the loop over the pixel-level files.

Plots cell tracks (paths and symbols colored by lifetime, cell outlines and numbers) on radar reflectivity snapshots
(--product refl, default) or on echo-top height snapshots (--product eth) for a single radar domain.

Usage:
>python plot_cell_tracks.py -s STARTDATE -e ENDDATE -c CONFIG.yml --radar_lat LAT --radar_lon LON

Called by: visualization/plot_cell_tracks.py, visualization/make_cell_animation.py (main(argv=[...])), and the stubs in
Analysis/ that keep the old plot_subset_cell_tracks_*.py commands working (main(legacy_name=...)).
Calls: cli_common (arguments), trackstats.get_cell_track_stats, runner.run_frames(cell_map.work_for_time_loop).
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
    projection_key, extent_aspect, panel_fractions, figsize_for_aspect, check_lat_range, auto_map_domain, min_figure_height,
)
from pyflextrkr.visualization.runner import run_frames
from pyflextrkr.visualization.trackstats import get_cell_track_stats
from pyflextrkr.visualization import cell_map

#-----------------------------------------------------------------------
def parse_cmd_args(argv=None, legacy_name=None):
    # Define and retrieve the command-line arguments...
    parser = argparse.ArgumentParser(
        description="Plot cell tracks on radar reflectivity snapshots for a user-defined subset domain."
    )
    cli_common.add_common_arguments(parser)
    parser.add_argument("--radar_lat", help="radar latitude (optional, defaults to domain center)", type=float, default=None)
    parser.add_argument("--radar_lon", help="radar longitude (optional, defaults to domain center)", type=float, default=None)
    parser.add_argument("--product", default='refl', choices=['refl', 'eth'],
                        help="field: 'refl' composite reflectivity, 'eth' 10 dBZ echo-top height (km)")
    parser.add_argument("--varname", help="Variable name for plotting in pixel files (default: dbz_comp, echotop10 for eth)", default=None)
    parser.add_argument("--var_scale", type=float, default=None, help="scale factor of the variable (default 1, 1e-3 for eth: m -> km)")
    parser.add_argument("--show_rangecircle", type=int, help="Draw radar range circles (0:no, 1:yes)", default=1)
    parser.add_argument("--show_azimuth", type=int, help="Draw azimuth lines (0:no, 1:yes)", default=1)
    parser.add_argument("--show_tracks", type=int, help="Show track paths (0:no, 1:yes)", default=None)
    parser.add_argument("--show_paths", type=int, help="Show track path lines (0:no, 1:yes, None:auto from show_tracks)", default=None)
    parser.add_argument("--show_symbols", type=int, help="Show track centroid symbols (0:no, 1:yes, None:auto from show_tracks)", default=None)
    parser.add_argument("--fontsize_tracks", type=float, help="Font size for track numbers (default: fontsize*0.8)", default=None)
    parser.add_argument("--perim_color", default=None, help="color of the cell outlines (default: black)")
    parser.add_argument("--perim_alpha", type=float, default=None, help="transparency of the cell outlines (default 0.6)")
    parser.add_argument("--perim_width", type=int, default=1, help="cell outline width in pixels")
    if legacy_name is not None:
        # Defaults of the old script that is being replaced
        parser.set_defaults(**legacy.defaults(legacy_name))
    args = parser.parse_args(argv)
    return cli_common.args_to_dict(args)

#-----------------------------------------------------------------------
def main(argv=None, legacy_name=None):
    """
    Make the cell track maps.

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
    radar_lat = args_dict.get('radar_lat')
    radar_lon = args_dict.get('radar_lon')
    map_extent = args_dict.get('extent')
    subset = args_dict.get('subset')
    figbasename = args_dict.get('figbasename')
    figname_type = args_dict.get('figname_type')
    figsize = args_dict.get('figsize')
    figsize_x = args_dict.get('figsize_x', 10)
    out_dir = args_dict.get('out_dir')
    trackstats_file = args_dict.get('trackstats_file')
    pixeltracking_path = args_dict.get('pixeltracking_path')
    time_format = args_dict.get('time_format')
    product = args_dict.get('product')
    varname_fill = args_dict.get('varname')
    projection = args_dict.get('projection')
    lat_range = args_dict.get('lat_range')
    map_aspect = args_dict.get('map_aspect') or 'equal'
    check_lat_range(projection, lat_range)
    draw_land = args_dict.get('draw_land') if args_dict.get('draw_land') is not None else 0
    draw_border = args_dict.get('draw_border') if args_dict.get('draw_border') is not None else 0
    draw_state = args_dict.get('draw_state') if args_dict.get('draw_state') is not None else 0
    draw_river = args_dict.get('draw_river', 0)
    show_rangecircle = args_dict.get('show_rangecircle', 1)
    show_azimuth = args_dict.get('show_azimuth', 1)
    title_prefix = args_dict.get('title_prefix', '')
    fontsize = args_dict.get('fontsize') if args_dict.get('fontsize') is not None else 13
    fontsize_tracks_arg = args_dict.get('fontsize_tracks', None)
    map_resolution = args_dict.get('map_resolution', '50m')
    lookback_days = args_dict.get('track_lookback_days') if args_dict.get('track_lookback_days') is not None else 0

    if time_format is None: time_format = "yyyymodd_hhmmss"

    # Product: the field to plot, its levels, colormap and label (edit them here)
    if product == 'eth':
        # Echo-top height
        if varname_fill is None: varname_fill = 'echotop10'
        var_scale = 1e-3    # scale factor for the variable
        # Colorfill levels
        levels = [1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,9,10,12,14,16,18,20]
        cbticks = levels
        cblabels = '10 dBZ ETH (km)'
        cmaps = 'nipy_spectral'     # Echo-top Height
        shade_alpha = 1.0    # transparancy alpha for shading
        show_tracks_default = 0
        marker_size = [30, 30, 30]  # track centroid marker size (short, medium, long lived)
        lw_centroid = [3, 3, 3]     # track path line width
    else:
        # Composite reflectivity
        if varname_fill is None: varname_fill = 'dbz_comp'
        var_scale = 1     # scale factor for the variable
        # Colorfill levels
        # levels = np.arange(-10, 60.1, 5)
        levels = np.arange(-10, 70.1, 5)
        # levels = [1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,9,10,12,14,16,18,20]
        # Colorbar ticks & labels
        # cbticks = np.arange(-10, 60.1, 5)
        cbticks = levels
        cblabels = 'Composite Reflectivity (dBZ)'
        cmaps = 'gist_ncar'     # Reflectivity
        shade_alpha = 0.85   # transparancy alpha for shading
        show_tracks_default = 1
        marker_size = [10, 10, 10]  # track centroid marker size (short, medium, long lived)
        lw_centroid = [1, 1, 1]     # track path line width
    if args_dict.get('var_scale') is not None:
        var_scale = args_dict['var_scale']
    lev_lifetime = np.arange(0.5, 4.01, 0.5)
    cblabel_tracks = 'Lifetime (hour)'
    cbticks_tracks = [1,2,3,4]
    cmap_tracks = 'Spectral_r'  # Lifetime

    # Set show_paths and show_symbols: if None, default to show_tracks value
    show_tracks = args_dict.get('show_tracks') if args_dict.get('show_tracks') is not None else show_tracks_default
    show_paths_arg = args_dict.get('show_paths', None)
    show_symbols_arg = args_dict.get('show_symbols', None)
    show_paths = show_paths_arg if show_paths_arg is not None else show_tracks
    show_symbols = show_symbols_arg if show_symbols_arg is not None else show_tracks
    # Set fontsize_tracks: if None, default to fontsize * 0.8
    fontsize_tracks = fontsize_tracks_arg if fontsize_tracks_arg is not None else fontsize * 0.8

    # Determine the figsize based on lat/lon ratio (map_aspect 'auto': the formula of the old scripts for PlateCarree;
    # otherwise the shape of the map decides it, below, after the data are known)
    if (figsize is None):
        if projection_key(projection) == 'platecarree' and map_aspect == 'auto':
            if (map_extent is not None):
                # Calculate aspect ratio from map extent
                aspect_ratio = extent_aspect(map_extent)
                # Use simple lat/lon ratio (works reasonably well for mid-latitudes)
                figsize_y = figsize_x * aspect_ratio
                figsize = [figsize_x, figsize_y]
            else:
                # Default figsize if map_extent not provided
                figsize = [figsize_x, figsize_x * 0.9]
        else:
            # The figure has the shape of the map (set below, after the data are known)
            pass

    # Put plot specifications in a dictionary
    plot_info = {
        'varname_fill': varname_fill,
        'var_scale': var_scale,
        'levels': levels,
        'lev_lifetime': lev_lifetime,
        'cbticks': cbticks,
        'cbticks_tracks': cbticks_tracks,
        'cblabels': cblabels,
        'cblabel_tracks': cblabel_tracks,
        'fontsize': fontsize,
        'fontsize_tracks': fontsize_tracks,
        'shade_alpha': shade_alpha,    # transparancy alpha for shading
        'mask_alpha': 0.6,   # transparancy alpha for cell perimeter mask
        'cmaps': cmaps,
        'cmap_tracks': cmap_tracks,
        'show_tracks': bool(show_tracks),
        'show_paths': bool(show_paths),
        'show_symbols': bool(show_symbols),
        'marker_size': marker_size,    # track centroid marker size (short, medium, long lived)
        'lw_centroid': lw_centroid,    # track path line width
        'radii': np.arange(20,101,20),  # radii for the radar range rings [km]
        'azimuths': np.arange(0,361,90),   # azimuth angles for HSRHI scans [degree]
        'map_edgecolor': 'k',
        'map_resolution': map_resolution,
        'figbasename': figbasename,
        'figname_type': figname_type,
        'figsize': figsize,
        'radar_lon': radar_lon,
        'radar_lat': radar_lat,
        'show_rangecircle': bool(show_rangecircle),
        'show_azimuth': bool(show_azimuth),
        'perim_width': args_dict.get('perim_width', 1),
        'skip_empty': bool(args_dict.get('skip_empty') or 0),
        'map_aspect': map_aspect,
    }
    plot_info['title_prefix'] = title_prefix
    if args_dict.get('dpi') is not None:
        plot_info['dpi'] = args_dict['dpi']
    if args_dict.get('land_linewidth') is not None:
        plot_info['land_linewidth'] = args_dict['land_linewidth']
    if args_dict.get('perim_color') is not None:
        plot_info['perim_color'] = args_dict['perim_color']
    if args_dict.get('perim_alpha') is not None:
        plot_info['mask_alpha'] = args_dict['perim_alpha']
    # Track style: the path line width and the symbol size are one value per lifetime class in these maps
    cli_common.apply_track_style_args(plot_info, args_dict)
    if args_dict.get('track_lw') is not None:
        plot_info['lw_centroid'] = [args_dict['track_lw']] * 3
        plot_info.pop('trackpath_linewidth', None)
    if args_dict.get('track_marker_size') is not None:
        plot_info['marker_size'] = [args_dict['track_marker_size']] * 3
    if args_dict.get('track_text_size') is not None:
        plot_info['fontsize_tracks'] = args_dict['track_text_size']

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

    # Tracks that end longer than this threshold from the current pixel-level frame are not plotted
    # This treshold controls the time window to retain previous tracks
    track_retain_time_min = 15

    # Create a timedelta threshold in minutes
    dt_thres = datetime.timedelta(minutes=track_retain_time_min)

    # Track stats file
    config = load_config(config_file, make_dirs=False)      # reading a config creates no folders
    stats_path = config["stats_outpath"]
    if pixeltracking_path is None:
        pixeltracking_path = config["pixeltracking_outpath"]
    pixeltracking_filebase = config["pixeltracking_filebase"]
    trackstats_filebase = config["trackstats_filebase"]
    startdate = config["startdate"]
    enddate = config["enddate"]
    radar_sensitivity = config.get("radar_sensitivity", 0)
    plot_info['radar_sensitivity'] = radar_sensitivity

    if trackstats_file is None:
        trackstats_file = f"{stats_path}{trackstats_filebase}{startdate}_{enddate}.nc"
    # Use n_workers from command-line if provided, otherwise use config file
    if n_workers is None:
        n_workers = config["nprocesses"]

    # Convert datetime string to Epoch time (base time)
    start_basetime = pd.to_datetime(start_datetime).timestamp()
    end_basetime = pd.to_datetime(end_datetime).timestamp()

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

    # Output figure directory
    if out_dir is None:
        figdir = f'{pixeltracking_path}quicklooks_trackpaths/'
    else:
        figdir = out_dir if out_dir.endswith('/') else f'{out_dir}/'
    os.makedirs(figdir, exist_ok=True)
    # Add to plot_info dictionary
    plot_info['figdir'] = figdir

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
        panel_w, panel_h = panel_fractions(**cell_map.GRID_MAP['gridspec'], **cell_map.GRID_MAP['update'])
        figsize = figsize_for_aspect(figsize_x, aspect_yx, panel_w, panel_h)
        # (not shorter than the titles need)
        figsize[1] = max(figsize[1], round(min_figure_height(plot_info['fontsize'], 1 - cell_map.GRID_MAP['update']['top']), 2))
        plot_info['figsize'] = figsize
    print(f'Figure size (width, height) in inches: {figsize}')

    # Get track stats data (cell tracks initiated within the period, or lookback_days before it)
    start_datetime_4stats = (pd.to_datetime(start_datetime) - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%dT%H:%M:%S')
    track_dict = get_cell_track_stats(trackstats_file, start_datetime_4stats, end_datetime, dt_thres)

    # Serial option (run_parallel=0) or parallel option (run_parallel=1)
    run_frames(cell_map.work_for_time_loop, datafiles, track_dict, map_info, plot_info, config,
               run_parallel=run_parallel, n_workers=n_workers)
