"""
Command line of the MCS track maps (visualization/plot_mcs_tracks.py): arguments, the plotting information of each product
and layout, the tracks and files of the period, and the loop over the pixel-level files.

Plots MCS tracks on Tb snapshots with a second field (precipitation 'tbpf', radar reflectivity 'tbze') or without ('tb'),
on 2 panels (--layout top_bottom: panel (a) above panel (b); --layout left_right: left + right) or 1 panel (--panels 1).

Usage:
>python plot_mcs_tracks.py -s STARTDATE -e ENDDATE -c CONFIG.yaml --layout top_bottom
>python plot_mcs_tracks.py -s STARTDATE -e ENDDATE -c CONFIG.yaml --product tbze --panels 1

Called by: visualization/plot_mcs_tracks.py, visualization/make_mcs_animation.py (main(argv=[...])), and the stubs in
Analysis/ that keep the old plot_subset_tb*_mcs_tracks_*.py commands working (main(legacy=...)).
Calls: cli_common (arguments), trackstats.get_mcs_track_stats, runner.run_frames(mcs_map.work_for_time_loop).
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
from pyflextrkr.visualization.colors import truncate_colormap, concat_cmaps
from pyflextrkr.visualization.mapcore import (
    projection_key, extent_aspect, panel_fractions, figsize_for_aspect, check_lat_range, auto_map_domain, figure_margins,
)
from pyflextrkr.visualization.runner import run_frames
from pyflextrkr.visualization.trackstats import get_mcs_track_stats
from pyflextrkr.visualization import mcs_map

#-----------------------------------------------------------------------
def parse_cmd_args(argv=None, legacy_name=None):
    # Define and retrieve the command-line arguments...
    parser = argparse.ArgumentParser(
        description="Plot MCS tracks on Tb & precipitation (or radar reflectivity) snapshots for a user-defined subset domain."
    )
    cli_common.add_common_arguments(parser)
    cli_common.add_layout_argument(parser)
    parser.add_argument("--product", default='tbpf', choices=['tbpf', 'tbze', 'tb'],
                        help="fields: 'tbpf' Tb + precipitation, 'tbze' Tb + radar reflectivity, 'tb' Tb only")
    parser.add_argument("--panels", type=int, default=2, choices=[1, 2],
                        help="1 panel (Tb and the second field on one map, MCS outlined by track) or 2 panels (Tb | second field with the MCS shaded)")
    parser.add_argument("--tb_varname", default='tb', help="Tb variable name in the pixel-level files")
    parser.add_argument("--var2_varname", "--dbz_varname", dest="var2_varname", default=None,
                        help="variable name of the second field in the pixel-level files (default: precipitation for tbpf, "
                             "reflectivity_comp for tbze; 'reflectivity_lowlevel' is the low-level reflectivity)")
    parser.add_argument("--track_smooth", type=int, default=1,
                        help="use the smoothed track center (meanlon_smooth, meanlat_smooth) if the stats file has it (0:no, 1:yes)")
    parser.add_argument("--perim_thick", type=float, default=None, help="MCS outline width [km] (default 10, 6 for 1-panel tbze)")
    parser.add_argument("--perim_width", type=int, default=None, help="MCS outline width in pixels (replaces --perim_thick)")
    parser.add_argument("--perim_color", default=None, help="draw the MCS outlines in this color (default: magenta on Tb, a color per track on 1 panel)")
    parser.add_argument("--perim_alpha", type=float, default=None, help="transparency of the MCS outlines when --perim_color is given")
    parser.add_argument("--periodic_perimeter", default='none', choices=['none', 'x', 'y', 'both', 'auto'],
                        help="the MCS outlines continue across the edges of a periodic domain ('auto': from pbc_direction of the file; "
                             "not with --subset 1)")
    parser.add_argument("--pfdiam_color", default=None, help="color of the PF diameter circles on 2 panels")
    parser.add_argument("--pfdiam_lw", type=float, default=None, help="line width of the PF diameter circles")
    parser.add_argument("--pfdiam_scale", type=float, default=None, help="scale of the PF diameter circles (1: the PF diameter)")
    parser.add_argument("--show_pfdiam", type=int, default=1,
                        help="draw the PF diameter circles on 2 panels (0:no, 1:yes); 0 also takes them out of the legend")
    parser.add_argument("--cbar_thickness", type=float, default=None,
                        help="how thin the colorbars are: their thickness as a fraction of the map width (colorbars at the right of the map) "
                             "or of the map height (colorbars below it) (default 0.02 for 2 panels, 1/70 for 1 panel with 2 colorbars, 0.03 for Tb only)")
    parser.add_argument("--cbar_spacing", type=float, default=None,
                        help="1 panel with 2 colorbars: the gap between them as a fraction of the map width (default 5/70, "
                             "widened if the text of the first colorbar needs more room)")
    if legacy_name is not None:
        # Defaults of the old script that is being replaced
        parser.set_defaults(**legacy.defaults(legacy_name))
    args = parser.parse_args(argv)
    return cli_common.args_to_dict(args)

#-----------------------------------------------------------------------
def main(argv=None, legacy_name=None):
    """
    Make the MCS track maps.

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
    panel_layout = args_dict.get('layout')
    product = args_dict.get('product')
    panels = args_dict.get('panels')
    map_extent = args_dict.get('extent')
    subset = args_dict.get('subset')
    figsize = args_dict.get('figsize')
    out_dir = args_dict.get('out_dir')
    figbasename = args_dict.get('figbasename')
    figname_type = args_dict.get('figname_type')
    title_prefix = args_dict.get('title_prefix')
    trackstats_file = args_dict.get('trackstats_file')
    pixeltracking_path = args_dict.get('pixeltracking_path')
    time_format = args_dict.get('time_format')
    figsize_x = args_dict.get('figsize_x', 10)
    projection = args_dict.get('projection')
    lat_range = args_dict.get('lat_range')
    map_aspect = args_dict.get('map_aspect') or 'equal'
    check_lat_range(projection, lat_range)
    draw_land = args_dict.get('draw_land') if args_dict.get('draw_land') is not None else 1
    draw_border = args_dict.get('draw_border') if args_dict.get('draw_border') is not None else 0
    draw_state = args_dict.get('draw_state') if args_dict.get('draw_state') is not None else 0
    draw_river = args_dict.get('draw_river', 0)
    var2_varname = args_dict.get('var2_varname')
    skip_empty = args_dict.get('skip_empty') if args_dict.get('skip_empty') is not None else 0
    lookback_days = args_dict.get('track_lookback_days') if args_dict.get('track_lookback_days') is not None else 4

    if time_format is None: time_format = "yyyymodd_hhmmss"
    if product == 'tb':
        panels = 1
        var2_varname = None
    elif var2_varname is None:
        var2_varname = 'precipitation' if product == 'tbpf' else 'reflectivity_comp'

    # Specify plotting info
    # Each product and layout has its own levels, colormaps, labels and titles (edit them here)
    if product == 'tbpf' and panels == 2:
        import colormaps
        # Precipitation color levels
        var2_levels = [2, 3, 4, 5, 6, 8, 10, 15, 20]
        var2_ticks = var2_levels
        # Tb color levels
        # tb_levels = np.arange(200, 320.1, 5)
        # tb_ticks = np.arange(200, 320.1, 10)
        tb_levels = np.arange(180, 315.1, 1)
        tb_ticks = np.arange(180, 310.1, 10)
        cblabels = {'tb_label': 'Tb (K)', 'var2_label': 'Precipitation (mm h$^{-1}$)'}
        # Colormap
        # Create a customed Tb colormap by combining two colormaps
        c1 = colormaps.GMT_wysiwygcont.reversed()
        c2 = colormaps.gray.reversed()
        # tb_cmap = concat_cmaps([c1, c2], [0.45, 0.55], discrete=135, trim_left=0.1, trim_right=0.1)
        tb_cmap = concat_cmaps([c1, c2], [0.45, 0.55], trim_left=0.1, trim_right=0.1)
        # tb_cmap = truncate_colormap(plt.get_cmap('turbo'), minval=0.05, maxval=0.95)
        var2_cmap = truncate_colormap(plt.get_cmap('viridis'), minval=0.2, maxval=1.0)
        tn_cmap = colormaps.cet_g_bw_minc_minl
        # tn_cmap = cc.cm["glasbey_light"]
        # tn_cmap = cc.cm["glasbey"]
        titles = {'tb_title': '(a) IR Brightness Temperature', 'var2_title': '(b) Precipitation (Tracked MCSs Shaded)'}
        extra = {'dpi': 200, 'fontsize': 10, 'tracknumber_fontsize': 12, 'perim_thick': 10, 'map_edgecolor': 'k'}
    elif product == 'tbze' and panels == 2:
        import colormaps
        # Reflectivity color levels
        var2_levels = np.arange(0, 70.1, 5)
        var2_ticks = var2_levels
        # Tb color levels
        # tb_levels = np.arange(200, 320.1, 10)
        # tb_ticks = np.arange(200, 320.1, 20)
        tb_levels = np.arange(180, 315.1, 1)
        tb_ticks = np.arange(180, 310.1, 10)
        cblabels = {'tb_label': 'Tb (K)', 'var2_label': 'Reflectivity (dBZ)'}
        # Colormaps
        # Create a customed Tb colormap by combining two colormaps
        c1 = colormaps.GMT_wysiwygcont.reversed()
        c2 = colormaps.gray.reversed()
        tb_cmap = concat_cmaps([c1, c2], [0.45, 0.55], trim_left=0.1, trim_right=0.1)
        # tb_cmap = truncate_colormap(plt.get_cmap('jet'), minval=0.05, maxval=0.95)
        var2_cmap = truncate_colormap(plt.get_cmap('gist_ncar'), minval=0.0, maxval=1.0)
        # tn_cmap = cc.cm["glasbey_light"]
        tn_cmap = colormaps.cet_g_bw_minc_minl
        if var2_varname == 'reflectivity_lowlevel':
            title_b = '(b) Low-level Reflectivity (Tracked MCSs Shaded)'
        elif var2_varname == 'reflectivity_comp':
            title_b = '(b) Composite Reflectivity (Tracked MCSs Shaded)'
        else:
            title_b = f'(b) Reflectivity {var2_varname} (Tracked MCSs Shaded)'
        titles = {'tb_title': '(a) IR Brightness Temperature', 'var2_title': title_b}
        # (gridlines below the data, coastline of the second panel 1 point, track-number shading of one color per level, colorbar minor ticks)
        extra = {'dpi': 200, 'fontsize': 10, 'tracknumber_fontsize': 12, 'perim_thick': 10, 'map_edgecolor': 'k',
                 'var2_mask_le': True, 'grid_zorder': 1, 'land_linewidth2': 1, 'tn_norm': 'boundary', 'cb_minorticks_off': False}
    elif product == 'tbpf' and panels == 1:
        import colorcet as cc
        # Precipitation color levels
        var2_levels = [2, 3, 4, 5, 6, 8, 10, 15, 20, 30]
        var2_ticks = var2_levels
        # Tb color levels
        tb_levels = np.arange(180, 300.1, 2)
        tb_ticks = np.arange(180, 300.1, 20)
        cblabels = {'tb_label': 'Tb (K)', 'var2_label': 'Precipitation (mm h$^{-1}$)'}
        # Colormaps
        tb_cmap = truncate_colormap(plt.get_cmap("Greys"), minval=0.01, maxval=0.99)
        var2_cmap = truncate_colormap(plt.get_cmap(cc.cm["CET_R1"]), minval=0, maxval=1.0)
        # tn_cmap = cc.cm["glasbey_light"]
        tn_cmap = cc.cm["glasbey_dark"]
        # tn_cmap = cc.cm["glasbey"]
        titles = {'tb_title': 'IR Brightness Temperature, Precipitation, Tracked MCS (Outline)'}
        extra = {'dpi': 200, 'fontsize': 10, 'tracknumber_fontsize': 12, 'perim_thick': 10, 'map_edgecolor': 'k',
                 'tb_alpha': 0.7, 'var2_alpha': 0.9}
    elif product == 'tbze' and panels == 1:
        import colorcet as cc
        # Reflectivity color levels
        # dbz_levels = np.arange(0, 60.1, 5)
        var2_levels = np.arange(10, 70.1, 5)
        var2_ticks = var2_levels
        # Tb color levels
        tb_levels = np.arange(170, 300.1, 5)
        tb_ticks = np.arange(170, 300.1, 10)
        cblabels = {'tb_label': 'Tb (K)', 'var2_label': 'Reflectivity (dBZ)'}
        # Colormaps
        tb_cmap = truncate_colormap(plt.get_cmap('Greys'), minval=0.01, maxval=0.99)
        var2_cmap = truncate_colormap(plt.get_cmap('gist_ncar'), minval=0.0, maxval=1.0)
        tn_cmap = cc.cm["glasbey_dark"]
        if var2_varname == 'reflectivity_lowlevel':
            titles = {'tb_title': 'Tb, Low-level Reflectivity (Tracked MCSs)'}
        else:
            titles = {'tb_title': 'Tb, Composite Reflectivity (Tracked MCSs)'}
        extra = {'dpi': 200, 'fontsize': 10, 'tracknumber_fontsize': 10, 'perim_thick': 6, 'map_edgecolor': 'k',
                 'tb_alpha': 0.7, 'var2_alpha': 0.9, 'var2_mask_le': True, 'var2_zorder': 3, 'grid_top': 0.92,
                 'map_zorder': 4, 'grid_zorder': 1, 'cb_minorticks_off': False}
    else:
        import colorcet as cc
        # Tb only
        var2_levels = None
        var2_ticks = None
        # Tb color levels
        tb_levels = np.arange(200, 320.1, 5)
        tb_ticks = np.arange(200, 320.1, 10)
        cblabels = {'tb_label': 'T$_{b}$ (K)'}
        # Colormap
        tb_cmap = truncate_colormap(plt.get_cmap('turbo'), minval=0.05, maxval=0.95)
        var2_cmap = None
        # tn_cmap = cc.cm["glasbey_light"]
        tn_cmap = cc.cm["glasbey"]
        titles = {'tb_title': 'IR Brightness Temperature (Tracked MCS Outline)'}
        extra = {'dpi': 200, 'fontsize': 13, 'tracknumber_fontsize': 10, 'perim_width': 6, 'map_edgecolor': 'dimgray',
                 'tb_alpha': 0.7, 'land_linewidth': 1, 'grid_zorder': 1, 'cb_minorticks_off': False, 'title_style': 'plain',
                 'perim_zorder': 2}
    if var2_varname is not None:
        levels = {'tb_levels': tb_levels, 'var2_levels': var2_levels}
        cbticks = {'tb_ticks': tb_ticks, 'var2_ticks': var2_ticks}
        cmaps = {'tb_cmap': tb_cmap, 'var2_cmap': var2_cmap, 'tn_cmap': tn_cmap}
    else:
        levels = {'tb_levels': tb_levels}
        cbticks = {'tb_ticks': tb_ticks}
        cmaps = {'tb_cmap': tb_cmap, 'tn_cmap': tn_cmap}
    # Determine the figsize based on lat/lon ratio (map_aspect 'auto': the formula of the old scripts for PlateCarree;
    # otherwise the shape of the map decides it, below, after the data are known)
    if (figsize is None):
        if projection_key(projection) == 'platecarree' and map_aspect == 'auto':
            # If map_extent is specified, calculate aspect ratio
            if (map_extent is not None):
                # Get map aspect ratio from map_extent (minlon, maxlon, minlat, maxlat)
                # Use modulo to compute lon_span correctly for any convention
                # including 0-360 extents and regions crossing the dateline.
                fig_ratio_yx = extent_aspect(map_extent)

                # figsize_x = 10
                figsize_y = figsize_x * fig_ratio_yx * (2.1 if panels == 2 else 1)
                figsize_y = float("{:.2f}".format(figsize_y))  # round to 2 decimal digits
                figsize = [figsize_x, figsize_y]
            else:
                figsize = [10, 10]
        else:
            # The figure has the shape of the map (set below, after the data are known)
            pass

    # The look of the product (fontsize, dpi, ...) and the information that is common to all products
    plot_info = dict(extra)
    plot_info.update({
        'levels': levels,
        'cmaps': cmaps,
        'titles': titles,
        'cbticks': cbticks,
        'cblabels': cblabels,
        'tb_varname': args_dict.get('tb_varname'),
        'var2_varname': var2_varname,
        'panels': panels,
        'marker_size': 10,
        'trackpath_linewidth': 1.5,
        'trackpath_color': 'purple',
        'mcsperim_color': 'magenta',
        'pfdiam_linewidth': 1,
        'pfdiam_scale': 1,  # scaling ratio for PF diameter circle (1: no scaling)
        'pfdiam_color': 'magenta',
        'map_resolution': args_dict.get('map_resolution'),
        'map_central_lon': 180,
        'figsize': figsize,
        'figbasename': figbasename,
        'figname_type': figname_type,
        'panel_layout': panel_layout,
        'cbar_thickness': args_dict.get('cbar_thickness'),
        'cbar_spacing': args_dict.get('cbar_spacing'),
        'title_prefix': title_prefix,
        'skip_empty': bool(skip_empty),
        'map_aspect': map_aspect,
        'periodic_perimeter': args_dict.get('periodic_perimeter'),
        'show_legend': bool(args_dict.get('show_legend')),
        'legend_loc': args_dict.get('legend_loc'),
        'show_pfdiam': bool(args_dict.get('show_pfdiam')),
    })
    # Options of the command line replace the defaults
    for option, key in [('dpi', 'dpi'), ('fontsize', 'fontsize'),
                        ('perim_thick', 'perim_thick'), ('perim_width', 'perim_width'), ('perim_color', 'perim_color'),
                        ('perim_alpha', 'perim_alpha'), ('pfdiam_color', 'pfdiam_color'),
                        ('pfdiam_lw', 'pfdiam_linewidth'), ('pfdiam_scale', 'pfdiam_scale')]:
        if args_dict.get(option) is not None:
            plot_info[key] = args_dict[option]
    if args_dict.get('land_linewidth') is not None:
        plot_info['land_linewidth'] = args_dict['land_linewidth']
        plot_info.pop('land_linewidth2', None)
    if args_dict.get('perim_width') is not None:
        plot_info.pop('perim_thick', None)
    elif args_dict.get('perim_thick') is not None:
        plot_info.pop('perim_width', None)
    cli_common.apply_track_style_args(plot_info, args_dict, npanels=panels)

    # Customize lat/lon labels
    lonv = args_dict.get('lon_ticks')
    latv = args_dict.get('lat_ticks')
    # Box corner lat/lon (list: [min, max])
    box = args_dict.get('box')
    box_lon = None if box is None else [box[0], box[1]]
    box_lat = None if box is None else [box[2], box[3]]
    # Put map info in a dictionary
    map_info = {
        'map_extent': map_extent,
        'subset': subset,
        'lonv': lonv,
        'latv': latv,
        'panel_layout': panel_layout,
        'draw_land': bool(draw_land),
        'draw_border': bool(draw_border),
        'draw_state': bool(draw_state),
        'draw_river': bool(draw_river),
        'box_lon': box_lon,
        'box_lat': box_lat,
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

    # Create a timedelta threshold
    # Tracks that end longer than this threshold from the current pixel-level frame are not plotted
    # This treshold controls the time window to retain previous tracks
    dt_thres = datetime.timedelta(hours=1)

    # Track stats file
    config = load_config(config_file, make_dirs=False)      # reading a config creates no folders
    stats_path = config["stats_outpath"]
    startdate = config["startdate"]
    enddate = config["enddate"]
    if trackstats_file is None:
        if product == 'tb':
            # Tb-only MCS tracking
            trackstats_file = f"{stats_path}{config['mcstbstats_filebase']}{startdate}_{enddate}.nc"
        else:
            trackstats_file = f"{stats_path}{config['mcsfinal_filebase']}{startdate}_{enddate}.nc"
    if pixeltracking_path is None:
        if product == 'tb':
            pixeltracking_path = f"{config['root_path']}mcstracking_tb/{startdate}_{enddate}/"
        else:
            pixeltracking_path = config["pixeltracking_outpath"]
    pixeltracking_filebase = config["pixeltracking_filebase"]
    if n_workers is None:
        n_workers = config["nprocesses"]
    pixel_radius = config.get("pixel_radius")

    # Output figure directory
    if out_dir is None:
        figdir = f'{pixeltracking_path}quicklooks_trackpaths/'
    else:
        figdir = f'{out_dir}/'
    os.makedirs(figdir, exist_ok=True)
    # Add to plot_info dictionary
    plot_info['figdir'] = figdir
    plot_info['pixel_radius'] = pixel_radius

    # Convert datetime string to Epoch time (base time)
    # These are for searching pixel-level files
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
    # PlateCarree the domain of the map is the domain of the data in the coordinates of the projection (e.g. WRF data on a
    # Lambert conformal grid shown with that projection: no empty space around the data)
    if figsize is None or (map_extent is None and projection_key(projection) != 'platecarree'):
        with xr.open_dataset(datafiles[0]) as ds0:
            data_lon, data_lat = ds0['longitude'].values, ds0['latitude'].values
        extent_xy, aspect_yx = auto_map_domain(map_extent, projection, map_info['central_longitude'], map_info['central_latitude'],
                                               map_info['standard_parallels'], lat_range, data_lon, data_lat)
        if extent_xy is not None:
            map_info['map_extent_xy'] = extent_xy
    if figsize is None:
        layout = mcs_map.figure_layout(panels, var2_varname is not None, panel_layout, args_dict.get('cbar_thickness'),
                                       args_dict.get('cbar_spacing'), plot_info.get('grid_top'))
        panel_w, panel_h = panel_fractions(**layout['gridspec'], **layout['update'])
        figsize = figsize_for_aspect(figsize_x, aspect_yx, panel_w, panel_h)
        # a short figure (a map much wider than high) gets margins in inches for the titles and labels
        figsize, margins_in = figure_margins(figsize, layout['update'], plot_info['fontsize'])
        plot_info['figsize'] = figsize
        plot_info['margins_in'] = margins_in
    print(f'Figure size (width, height) in inches: {figsize}')

    # Get track stats data
    track_dict = get_mcs_track_stats(trackstats_file, start_datetime_4stats, end_datetime, dt_thres,
                                     use_smooth=bool(args_dict.get('track_smooth', 1)))

    # Serial option (run_parallel=0) or parallel option (run_parallel=1)
    run_frames(mcs_map.work_for_time_loop, datafiles, track_dict, map_info, plot_info, config,
               run_parallel=run_parallel, n_workers=n_workers)
