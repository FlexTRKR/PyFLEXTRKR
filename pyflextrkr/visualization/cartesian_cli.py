"""
Command line of the track maps of a domain in x/y (visualization/plot_cartesian_tracks.py): arguments, the plotting
information of each product, the tracks and files of the period, and the loop over the pixel-level files.

Plots the tracks of an idealized or cloud-resolving simulation domain without a map: Tb + precipitation with the MCS outlined
(--product tbpf, periodic domain: --pbc 1 wraps the tracks at the edges), or one field, e.g. cold pool intensity
(--product field).

Usage:
>python plot_cartesian_tracks.py -s STARTDATE -e ENDDATE -c CONFIG.yml --extent "0. 1500 0. 1500"
>python plot_cartesian_tracks.py -s STARTDATE -e ENDDATE -c CONFIG.yml --product field --field_varname cp_intensity

Called by: visualization/plot_cartesian_tracks.py, and the stubs in Analysis/ that keep the old plot_subset_tbpf_demo_pbc.py,
plot_subset_tbpf_tracks_pinacles_pbc.py and plot_subset_coldpool_tracks_nomap.py commands working (main(legacy_name=...)).
Calls: cli_common (arguments), trackstats.get_cartesian_track_stats, runner.run_frames(cartesian_map.work_for_time_loop).
"""
import argparse
import datetime
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pyflextrkr.ft_utilities import load_config, subset_files_timerange
from pyflextrkr.visualization import cli_common, legacy
from pyflextrkr.visualization.runner import run_frames
from pyflextrkr.visualization.trackstats import get_cartesian_track_stats
from pyflextrkr.visualization import cartesian_map

#-----------------------------------------------------------------------
def _range(values):
    """Array of the 3 numbers vmin vmax step (vmax is included, as np.arange(vmin, vmax + step/2, step))."""
    vmin, vmax, step = values
    return np.arange(vmin, vmax + step / 2., step)

#-----------------------------------------------------------------------
def parse_cmd_args(argv=None, legacy_name=None):
    # Define and retrieve the command-line arguments...
    parser = argparse.ArgumentParser(
        description="Plot tracks on snapshots of a domain in x/y (idealized simulations, cold pools) without a map."
    )
    cli_common.add_common_arguments(parser, cartesian=True)
    parser.set_defaults(figsize_x=12, workers=None)
    parser.add_argument("--product", default='tbpf', choices=['tbpf', 'field'],
                        help="'tbpf': Tb + precipitation with the MCS outlined, 'field': one field (--field_varname)")
    parser.add_argument("--tb_varname", default='tb', help="Tb variable name (tbpf)")
    parser.add_argument("--var2_varname", default='precipitation', help="precipitation variable name (tbpf)")
    parser.add_argument("--field_varname", default=None, help="variable of the field (field; default: field_varname of the config)")
    parser.add_argument("--cmap", default=None, help="colormap of the field (field)")
    parser.add_argument("--levels", type=float, nargs=3, default=None, metavar=('VMIN', 'VMAX', 'STEP'), help="shading levels of the field (field)")
    parser.add_argument("--cbticks", type=float, nargs=3, default=None, metavar=('VMIN', 'VMAX', 'STEP'), help="colorbar ticks of the field (field)")
    parser.add_argument("--cblabel", default=None, help="colorbar label of the field (field)")
    parser.add_argument("--pbc", type=int, default=None, help="periodic domain: wrap the tracks at the edges (0:no, 1:yes; default: 1 for tbpf, 0 for field)")
    parser.add_argument("--show_pf_markers", type=int, default=None, help="red triangles at the positions of the precipitation features (0:no, 1:yes; default 1 for tbpf with --pbc 1)")
    parser.add_argument("--xscale", type=float, default=None, help="scale factor of the x coordinates, e.g. 1e-3: m -> km (default: 1 for tbpf, 1e-3 for field)")
    parser.add_argument("--yscale", type=float, default=None, help="scale factor of the y coordinates (default: as --xscale)")
    parser.add_argument("--xlabel", default='X (km)', help="x axis label")
    parser.add_argument("--ylabel", default='Y (km)', help="y axis label")
    parser.add_argument("--perim_plot", default=None, choices=['pcolormesh', 'contour'],
                        help="outline of the tracked features: pixels ('pcolormesh') or a line ('contour'); default: contour for tbpf, pcolormesh for field")
    parser.add_argument("--perim_linewidth", type=float, default=1.5, help="line width of the contour outline")
    parser.add_argument("--perim_color", default=None, help="color of the contour outline (default: orange for tbpf)")
    parser.add_argument("--perim_alpha", type=float, default=0.6, help="transparency of the pixel outline")
    parser.add_argument("--perim_thick", type=float, default=2, help="outline width [km] (pixel_radius of the config)")
    # Used by the defaults of the old scripts only
    parser.add_argument("--stats_key", default=None, choices=['mcsfinal', 'finalstats', 'trackstats'], help=argparse.SUPPRESS)
    if legacy_name is not None:
        # Defaults of the old script that is being replaced
        parser.set_defaults(**legacy.defaults(legacy_name))
    args = parser.parse_args(argv)
    return cli_common.args_to_dict(args)

#-----------------------------------------------------------------------
def main(argv=None, legacy_name=None):
    """
    Make the track maps of a domain in x/y.

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
    figsize_x = args_dict.get('figsize_x')
    out_dir = args_dict.get('out_dir')
    figbasename = args_dict.get('figbasename')
    figname_type = args_dict.get('figname_type')
    trackstats_file = args_dict.get('trackstats_file')
    pixeltracking_path = args_dict.get('pixeltracking_path')
    time_format = args_dict.get('time_format')
    product = args_dict.get('product')
    lookback_days = args_dict.get('track_lookback_days') if args_dict.get('track_lookback_days') is not None else 30

    if time_format is None: time_format = "yyyymodd_hhmmss"

    # Determine the figsize based on the ratio of the domain
    if (figsize is None):
        # If map_extent is specified, calculate aspect ratio
        if (map_extent is not None):
            # Get aspect ratio from the extent (minx, maxx, miny, maxy)
            lon_span = map_extent[1] - map_extent[0]
            lat_span = map_extent[3] - map_extent[2]
            fig_ratio_yx = lat_span / lon_span

            figsize_y = figsize_x * fig_ratio_yx
            figsize_y = float("{:.2f}".format(figsize_y))  # round to 2 decimal digits
            figsize = [figsize_x, figsize_y]
        else:
            figsize = [10, 10]

    # Specify plotting info
    # Each product has its own levels, colormaps, labels and titles (edit them here)
    if product == 'tbpf':
        # Precipitation color levels
        pcp_levels = [2, 3, 4, 5, 6, 8, 10, 15, 20, 30]
        pcp_ticks = pcp_levels
        # Tb color levels
        tb_levels = np.arange(180, 300.1, 2)
        tb_ticks = np.arange(180, 300.1, 20)
        levels = {'var1_levels': tb_levels, 'var2_levels': pcp_levels}
        # Colorbar ticks & labels
        cbticks = {'var1_ticks': tb_ticks, 'var2_ticks': pcp_ticks}
        cblabels = {'var1_label': 'Tb (K)', 'var2_label': 'Precipitation (mm h$^{-1}$)'}
        # Colormaps
        cmaps = {'var1_cmap': 'Greys', 'var2_cmap': 'YlGnBu'}
        titles = {'var1_title': 'IR Brightness Temperature, Precipitation, Tracked MCS (Outline)'}
        var1_varname = args_dict.get('tb_varname')
        var2_varname = args_dict.get('var2_varname')
        pbc, xscale, perim_plot, show_pf_markers = 1, 1, 'contour', 1
        extra = {'tracknumber_color': 'r', 'perim_color': 'orange'}
    else:
        cmap = args_dict.get('cmap') if args_dict.get('cmap') is not None else 'YlGnBu'   # colormap
        field_levels = _range(args_dict['levels']) if args_dict.get('levels') is not None else np.arange(0.1, 12.01, 0.1)  # shading levels
        field_ticks = _range(args_dict['cbticks']) if args_dict.get('cbticks') is not None else np.arange(0, 12.01, 1)  # colorbar ticks
        field_label = args_dict.get('cblabel') if args_dict.get('cblabel') is not None else 'Cold Pool Intensity (m s$^{-1}$)'  # colorbar label
        levels = {'var1_levels': field_levels}
        cbticks = {'var1_ticks': field_ticks}
        cblabels = {'var1_label': field_label}
        cmaps = {'var1_cmap': cmap}
        titles = None
        var1_varname = args_dict.get('field_varname')
        var2_varname = None
        pbc, xscale, perim_plot, show_pf_markers = 0, 1e-3, 'pcolormesh', 0
        extra = {}
    # Options of the command line replace the defaults of the product
    if args_dict.get('pbc') is not None: pbc = args_dict['pbc']
    if args_dict.get('xscale') is not None: xscale = args_dict['xscale']
    yscale = args_dict['yscale'] if args_dict.get('yscale') is not None else xscale
    if args_dict.get('perim_plot') is not None: perim_plot = args_dict['perim_plot']
    if args_dict.get('show_pf_markers') is not None: show_pf_markers = args_dict['show_pf_markers']
    if args_dict.get('perim_color') is not None: extra['perim_color'] = args_dict['perim_color']

    plot_info = {
        'fontsize': args_dict.get('fontsize') if args_dict.get('fontsize') is not None else 14,     # plot font size
        'dpi': args_dict.get('dpi') if args_dict.get('dpi') is not None else 200,
        'levels': levels,
        'cmaps': cmaps,
        'cbticks': cbticks,
        'cblabels': cblabels,
        'titles': titles,
        'var1_varname': var1_varname,
        'var2_varname': var2_varname,
        'remove_oob_low': True,   # mask out-of-bounds low values (< min(levels))
        'remove_oob_high': False,  # mask out-of-bounds high values (> max(levels))
        'mask_alpha': args_dict.get('perim_alpha'),   # transparancy alpha for perimeter mask
        'marker_size': 10,   # track symbol marker size
        'tracknumber_fontsize': 14,
        'perim_plot': perim_plot,  # method to plot tracked feature perimeter ('contour', 'pcolormesh')
        'perim_linewidth': args_dict.get('perim_linewidth'),  # perimeter line width for 'contour' method
        'perim_thick': args_dict.get('perim_thick'),  # width of the tracked feature perimeter [km]
        'trackpath_linewidth': 1.5, # track path line width
        'trackpath_color': 'blueviolet',    # track path color
        'xlabel': args_dict.get('xlabel'),
        'ylabel': args_dict.get('ylabel'),
        'xscale': xscale,
        'yscale': yscale,
        'pbc': bool(pbc),
        'show_pf_markers': bool(show_pf_markers) and bool(pbc),
        'figsize': figsize,
        'figbasename': figbasename,
        'figname_type': figname_type,
    }
    plot_info.update(extra)
    cli_common.apply_track_style_args(plot_info, args_dict)

    # Put map info in a dictionary
    map_info = {
        'map_extent': map_extent,
        'subset': subset,
    }

    # Track stats file
    config = load_config(config_file, make_dirs=False)      # reading a config creates no folders
    stats_path = config["stats_outpath"]
    if pixeltracking_path is None:
        pixeltracking_path = config["pixeltracking_outpath"]
    pixeltracking_filebase = config["pixeltracking_filebase"]
    trackstats_filebase = config["trackstats_filebase"]
    finalstats_filebase = config.get("finalstats_filebase", None)
    mcsfinal_filebase = config.get("mcsfinal_filebase", None)
    startdate = config["startdate"]
    enddate = config["enddate"]
    if n_workers is None:
        n_workers = config["nprocesses"]
    datatimeresolution = config["datatimeresolution"]  # hour
    plot_info['pixel_radius'] = config["pixel_radius"]
    if var1_varname is None:
        var1_varname = config["field_varname"]
        plot_info['var1_varname'] = var1_varname

    # Tracks that end longer than this threshold from the current pixel-level frame are not plotted
    # This treshold controls the time window to retain previous tracks
    track_retain_time_min = (datatimeresolution * 60)

    # Create a timedelta threshold in minutes
    dt_thres = datetime.timedelta(minutes=track_retain_time_min)

    # If trackstats_file is not specified
    if trackstats_file is None:
        # If the final stats of the MCS or generic tracking is present, use it (links merge/split tracks)
        stats_key = args_dict.get('stats_key')
        if stats_key is None:
            stats_key = 'mcsfinal' if mcsfinal_filebase is not None else ('finalstats' if finalstats_filebase is not None else 'trackstats')
        filebase = {'mcsfinal': mcsfinal_filebase, 'finalstats': finalstats_filebase, 'trackstats': trackstats_filebase}[stats_key]
        if filebase is None:
            filebase = trackstats_filebase
        trackstats_file = f"{stats_path}{filebase}{startdate}_{enddate}.nc"

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
    print(f'Figure size (width, height) in inches: {figsize}')

    # Get track stats data
    track_dict = get_cartesian_track_stats(trackstats_file, start_datetime_4stats, end_datetime, dt_thres,
                                           xscale=xscale, yscale=yscale, with_pf=bool(plot_info['show_pf_markers']))

    # Serial option (run_parallel=0) or parallel option (run_parallel=1)
    run_frames(cartesian_map.work_for_time_loop, datafiles, track_dict, map_info, plot_info, config,
               run_parallel=run_parallel, n_workers=n_workers)
