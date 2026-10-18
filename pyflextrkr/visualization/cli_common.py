"""
The command-line arguments that all plot families share, and the conversion of the parsed arguments to the dictionary
(`args_dict`) that the main() of each family reads. A family adds its own arguments to the same parser (mcs_cli.py,
cell_cli.py, generic_cli.py, cartesian_cli.py).

Options whose default depends on the family (dpi, fontsize, draw_state, ...) are None here and are resolved in the main()
of the family, next to the plotting information of that family.

Called by: mcs_cli, cell_cli, generic_cli, cartesian_cli (parse_cmd_args, main). Calls: argparse.
"""
import argparse
import sys

# Layout of 2 panels = where panel (b) is. The old option -o/--orientation called them 'horizontal' (top + bottom) and
# 'vertical' (left + right), which is easy to mix up: the old values still work and print a deprecation warning
LAYOUT_NAMES = ('top_bottom', 'left_right')
OLD_LAYOUT_NAMES = {'horizontal': 'top_bottom', 'vertical': 'left_right'}

#-----------------------------------------------------------------------
def layout_name(value):
    """
    argparse type of --layout: the layout name. An old orientation name (horizontal, vertical) is converted to the new name
    and a deprecation warning is printed.
    """
    if value in OLD_LAYOUT_NAMES:
        new = OLD_LAYOUT_NAMES[value]
        print(f"DeprecationWarning: the panel layout '{value}' is now called '{new}' (use --layout {new}; -o and --orientation "
              f"are the old names of --layout; horizontal -> top_bottom, vertical -> left_right).", file=sys.stderr)
        return new
    if value not in LAYOUT_NAMES:
        raise argparse.ArgumentTypeError(f"invalid layout '{value}' (choose from {', '.join(LAYOUT_NAMES)})")
    return value

#-----------------------------------------------------------------------
def add_layout_argument(parser):
    """The layout of 2 panels: --layout top_bottom|left_right (-o and --orientation horizontal|vertical are the old names)."""
    parser.add_argument("-o", "--layout", "--orientation", dest="layout", type=layout_name, default='top_bottom', metavar="LAYOUT",
                        help="2 panels: 'top_bottom' (default: panel (a) above panel (b)) or 'left_right' (side by side). "
                             "-o and --orientation are the old names; their values horizontal (= top_bottom) and vertical "
                             "(= left_right) still work, with a deprecation warning")

#-----------------------------------------------------------------------
def parse_extent(values):
    """
    Map extent [lonmin, lonmax latmin, latmax] from the tokens of --extent: four numbers, or one quoted string with four
    numbers (--extent "0. 1500 0. 1500", as the scripts for periodic domains were called).
    """
    if values is None:
        return None
    tokens = " ".join(str(v) for v in values).replace(",", " ").split()
    if len(tokens) != 4:
        raise argparse.ArgumentTypeError("--extent needs 4 numbers: lonmin lonmax latmin latmax")
    return [float(t) for t in tokens]

#-----------------------------------------------------------------------
def add_common_arguments(parser, cartesian=False):
    """
    Add the arguments that all plot families share to an argparse parser.

    cartesian: True for a domain in x/y without a map (cartesian_cli): the options of the map (projection, coastlines,
    gridlines, ...) are not added.
    """

    # Period, files, parallel
    parser.add_argument("-s", "--start", help="first time in time series to plot, format=YYYY-mm-ddTHH:MM:SS", required=True)
    parser.add_argument("-e", "--end", help="last time in time series to plot, format=YYYY-mm-ddTHH:MM:SS", required=True)
    parser.add_argument("-c", "--config", help="yaml config file for tracking", required=True)
    parser.add_argument("-p", "--parallel", help="flag to run in parallel (0:serial, 1:parallel)", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of Dask workers for parallel processing (default: 4; the generic and cartesian maps: nprocesses of the config)")
    parser.add_argument("--trackstats_file", help="track stats file name (default: the final stats file of the config)", default=None)
    parser.add_argument("--pixel_path", help="Pixel-level tracknumber mask files directory (default: from the config)", default=None)
    parser.add_argument("--time_format", help="Pixel-level file datetime format", default=None)
    parser.add_argument("--track_lookback_days", type=float, default=None,
                        help="tracks that were initiated up to this many days before --start are drawn too (default: as the "
                             "original script of the family: 4 days for MCS and generic maps, 30 for x/y maps, 0 for cell maps)")

    # Output
    parser.add_argument("--output", help="output directory (default: quicklooks_trackpaths/ in the pixel-level directory)", default=None)
    parser.add_argument("--figbasename", help="output figure base name", default="")
    parser.add_argument("--figname_type", help="output figure name type ('date_time' or 'sequence')", default="date_time")
    parser.add_argument("--title_prefix", help="Prefix string to add to figure title", default="")
    parser.add_argument("--figsize", nargs='+', help="figure size (width, height) in inches", type=float, default=None)
    parser.add_argument("--figsize_x", type=float, help="figure size width in inches (the height follows from the map extent)", default=10)
    parser.add_argument("--dpi", type=int, help="figure resolution (default: 200; 300 for the cell maps)", default=None)
    parser.add_argument("--fontsize", type=float, help="font size of labels and text (default: 10 for the MCS maps with a second field; 13 for the Tb-only, cell and generic maps; 14 for the x/y maps)", default=None)
    parser.add_argument("--skip_empty", type=int, default=None,
                        help="skip the times without a tracked object (0:no, 1:yes; default: no, except for the old 1-panel and Tb-only scripts)")

    # Map domain, projection, background
    parser.add_argument("--extent", nargs='+', help="map extent: lonmin lonmax latmin latmax (4 numbers or one quoted string)", default=None)
    parser.add_argument("--subset", help="flag to subset data (0:no, 1:yes)", type=int, default=0)
    if not cartesian:
        parser.add_argument("--projection", default="PlateCarree",
                            help="map projection: PlateCarree (default; central longitude 180 if the extent crosses the dateline), "
                                 "Robinson, Mollweide, EqualEarth, Mercator, LambertConformal, AlbersEqualArea, Orthographic, "
                                 "NorthPolarStereo, SouthPolarStereo")
        parser.add_argument("--central_longitude", type=float, default=None, help="central longitude of the projection [degree east]")
        parser.add_argument("--central_latitude", type=float, default=None, help="central latitude of the projection (LambertConformal, AlbersEqualArea, Orthographic)")
        parser.add_argument("--standard_parallels", type=float, nargs='+', default=None, help="1 or 2 standard parallels (LambertConformal, AlbersEqualArea)")
        parser.add_argument("--lat_range", type=float, nargs=2, default=None, metavar=('LAT_MIN', 'LAT_MAX'),
                            help="crop the map to this latitude band, e.g. -60 60 for a global map of data within 60S-60N")
        parser.add_argument("--lon_ticks", type=float, nargs='+', default=None, help="longitudes of the gridlines and labels")
        parser.add_argument("--lat_ticks", type=float, nargs='+', default=None, help="latitudes of the gridlines and labels")
        parser.add_argument("--grid_linewidth", type=float, default=None, help="line width of the lat/lon gridlines (0: labels only)")
        parser.add_argument("--grid_alpha", type=float, default=None, help="transparency of the lat/lon gridlines")
        parser.add_argument("--map_aspect", default='equal', choices=['equal', 'auto'],
                            help="shape of the map: 'equal' (default) keeps the true shape of the domain (a map that does not fill its panel is "
                                 "centered in it and the colorbars follow it); 'auto' stretches the map to fill its panel (the old scripts)")
        parser.add_argument("--map_resolution", help="Map resolution for Natural Earth features ('10m', '50m', '110m')", default='50m')
        parser.add_argument("--draw_land", type=int, default=None, help="Draw coastlines (0:no, 1:yes; default 1, 0 for the cell maps)")
        parser.add_argument("--land_linewidth", type=float, default=None, help="line width of the coastlines [points] (default: 2 for MCS maps, 1 for the others)")
        parser.add_argument("--draw_border", type=int, help="Draw country borders (0:no, 1:yes; default 0, 1 for the generic maps)", default=None)
        parser.add_argument("--draw_state", type=int, help="Draw state/province borders (0:no, 1:yes; default 0)", default=None)
        parser.add_argument("--draw_river", type=int, help="Draw rivers (0:no, 1:yes)", default=0)
        parser.add_argument("--box", type=float, nargs=4, default=None, metavar=('LON0', 'LON1', 'LAT0', 'LAT1'),
                            help="draw a rectangle (e.g. the domain of a regional plot on a global map)")
        parser.add_argument("--show_legend", type=int, default=1, help="Show the legend (0:no, 1:yes)")
        parser.add_argument("--legend_loc", default="lower right", help="Legend location (matplotlib loc string)")

    # Track style
    parser.add_argument("--show_track_numbers", type=int, nargs='+', default=None,
                        help="Write the track numbers (0:no, 1:yes); the 2-panel MCS maps take two values, one for each panel: "
                             "1 0 writes them on panel (a) only (default: yes)")
    parser.add_argument("--track_color", default=None, help="track path color")
    parser.add_argument("--track_lw", type=float, default=None, help="track path line width [points]")
    parser.add_argument("--track_ls", default=None, help="track path line style: solid, dashed, dotted, dashdot (or '-', ':')")
    parser.add_argument("--track_alpha", type=float, default=None, help="track path transparency (1: opaque)")
    parser.add_argument("--track_marker", default=None, help="symbol at the initiation location (matplotlib marker: o, ^, *, s, ...)")
    parser.add_argument("--track_marker_size", type=float, default=None, help="size of the initiation symbol (scatter size is twice this)")
    parser.add_argument("--track_marker_color", default=None, help="color of the initiation symbol (default: the track color)")
    parser.add_argument("--track_text_size", type=float, default=None, help="font size of the track numbers")
    parser.add_argument("--track_text_color", default=None, help="color of the track numbers")
    parser.add_argument("--track_text_weight", default=None, help="font weight of the track numbers (normal, bold)")
    parser.add_argument("--track_text_alpha", type=float, default=None, help="transparency of the track numbers")
    parser.add_argument("--track_halo_color", default=None,
                        help="color of a halo around track paths, symbols and numbers (default: no halo)")
    parser.add_argument("--track_halo_width", type=float, default=None, help="halo width outside the edge [points] (default 2)")
    parser.add_argument("--track_halo_alpha", type=float, default=None, help="halo transparency (1: opaque, default 1)")

#-----------------------------------------------------------------------
def options_to_argv(options):
    """
    Command-line arguments from a dictionary of options, for the wrapper scripts that call main(argv) of a family:
    {'start': '2025-06-01T00', 'extent': [140., 220., -20., 20.], 'draw_border': True, 'track_color': None} ->
    ['--start', '2025-06-01T00', '--extent', '140.0', '220.0', '-20.0', '20.0', '--draw_border', '1'].
    The keys are the long option names without the dashes; None values are left out (the default of the option applies);
    a list of lists repeats the option: {'city': [['A', 1, 2], ['B', 3, 4]]} -> ['--city', 'A', '1', '2', '--city', 'B', '3', '4'].
    """
    argv = []
    for key, value in options.items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, (list, tuple)) and value and all(isinstance(v, (list, tuple)) for v in value):
            for item in value:
                argv += [f'--{key}'] + [str(v) for v in item]
        elif isinstance(value, (list, tuple)):
            argv += [f'--{key}'] + [str(int(v)) if isinstance(v, bool) else str(v) for v in value]
        elif isinstance(value, str) and value.startswith('-'):
            argv.append(f'--{key}={value}')
        else:
            argv += [f'--{key}', str(value)]
    return argv

#-----------------------------------------------------------------------
def args_to_dict(args):
    """
    Parsed arguments as a dictionary; the options of the original scripts keep the key names the original main() used
    (start_datetime, end_datetime, run_parallel, config_file, out_dir, pixeltracking_path).
    """
    args_dict = dict(vars(args))
    args_dict['start_datetime'] = args_dict.pop('start')
    args_dict['end_datetime'] = args_dict.pop('end')
    args_dict['run_parallel'] = args_dict.pop('parallel')
    args_dict['config_file'] = args_dict.pop('config')
    args_dict['out_dir'] = args_dict.pop('output')
    args_dict['pixeltracking_path'] = args_dict.pop('pixel_path')
    args_dict['extent'] = parse_extent(args_dict['extent'])
    return args_dict

#-----------------------------------------------------------------------
def apply_track_style_args(plot_info, args_dict, npanels=1):
    """
    Put the track style options that were given on the command line into plot_info (the others stay as the family
    set them). The keys are the ones that the plot functions read. npanels: the number of panels that can have their
    own --show_track_numbers value (2 for the 2-panel MCS maps).
    """
    keys = {
        'track_color': 'trackpath_color',
        'track_lw': 'trackpath_linewidth',
        'track_ls': 'trackpath_linestyle',
        'track_alpha': 'trackpath_alpha',
        'track_marker': 'track_marker',
        'track_marker_size': 'marker_size',
        'track_marker_color': 'track_marker_color',
        'track_text_size': 'tracknumber_fontsize',
        'track_text_color': 'tracknumber_color',
        'track_text_weight': 'tracknumber_weight',
        'track_text_alpha': 'tracknumber_alpha',
        'track_halo_color': 'track_halo_color',
        'track_halo_width': 'track_halo_width',
        'track_halo_alpha': 'track_halo_alpha',
    }
    for option, key in keys.items():
        if args_dict.get(option) is not None:
            plot_info[key] = args_dict[option]
    values = args_dict.get('show_track_numbers')
    if values is not None:
        if len(values) > npanels:
            raise ValueError(f"--show_track_numbers takes {npanels} value(s) for this map (one per panel; 0:no, 1:yes), "
                             f"not {len(values)}")
        plot_info['show_tracknumbers'] = bool(values[0])
        if len(values) > 1:
            plot_info['show_tracknumbers2'] = bool(values[1])        # the second panel
    return plot_info
