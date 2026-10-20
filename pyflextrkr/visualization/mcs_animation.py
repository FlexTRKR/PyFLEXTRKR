"""
MCS tracking animation from a set of parameters: the PNG frames (mcs_cli.main, the options of plot_mcs_tracks.py), the mp4 and
optionally a GIF (animation.py).

The wrapper scripts keep the parameters as variables at the top of the file, so you can edit them or set them on the command
line (--set NAME=VALUE), and call main(globals()):
    visualization/examples/make_mcs_animation.py                  all parameters, the global Robinson map as default
    visualization/examples/make_mcs_animation_wrf_tbze.py         WRF Tb + radar reflectivity on a Lambert conformal map
    visualization/examples/make_mcs_animation_imerg_1panel.py     IMERG over South America, 1 panel
    visualization/examples/make_mcs_animation_himawari_tb.py      Himawari Tb only
A wrapper sets the parameters it needs; DEFAULTS below lists all of them (a parameter left out has the value listed there).
The PNG files and the animation go to {root_path}/quicklooks_trackpaths/ of the tracking config file (the folder of the demo scripts)
unless fig_root, or figdir and animation_dir, are set.

Called by: the wrapper scripts above. Calls: mcs_cli.main (the PNG frames), animation.make_animation (mp4 and GIF with ffmpeg).
"""
import os
import sys
from pyflextrkr.visualization.cli_common import options_to_argv, layout_name
from pyflextrkr.visualization.mcs_cli import main as plot_mcs_tracks
from pyflextrkr.visualization.animation import frames_in_range, make_animation
from pyflextrkr.visualization import wrapper_common
from pyflextrkr.visualization.wrapper_common import to_plain

# All parameters and their values when a wrapper does not set them (None: the default of plot_mcs_tracks.py)
DEFAULTS = dict(
    # What to plot: plot_type '1panel' (Tb and the second field on one map) or '2panel' (two maps);
    # data_type 'tbpf' (Tb + precipitation), 'tbze' (Tb + reflectivity) or 'tb' (Tb only, 1 panel)
    plot_type='2panel', data_type='tbpf',
    start_date=None, end_date=None,                       # 'YYYY-MM-DDTHH' (or with :MM)
    # Map extent: lon_min, lon_max, lat_min, lat_max (None: the whole domain of the data); subset 1 reads only the pixels of the extent
    lon_min=None, lon_max=None, lat_min=None, lat_max=None, subset=0,
    # 2 panels: 'top_bottom' (panel (a) above panel (b)) or 'left_right'
    layout='top_bottom',
    # Tracking config file; trackstats_file and pixel_path: None = from the config file
    config_file=None, trackstats_file=None, pixel_path=None,
    track_lookback_days=None,                             # days before start_date from which the tracks are drawn (None: 4)
    time_format='yyyymodd_hhmmss',                        # time format in the pixel-level file names
    parallel_mode=1, n_workers=64,
    figsize=None, figsize_x=14, dpi=None,                 # figsize [width, height] in inches, or figsize_x (the height follows the map)
    title_prefix='',
    dbz_varname='reflectivity_lowlevel',                  # only for data_type='tbze': 'reflectivity_comp' (composite) or 'reflectivity_lowlevel'
    fontsize=None, show_track_numbers=None,               # show_track_numbers: [panel (a), panel (b)], e.g. [True, False]
    # Map projection (None: PlateCarree) and its parameters
    projection=None, central_longitude=None, central_latitude=None, standard_parallels=None, lat_range=None,
    periodic_perimeter=None,                              # 'x': the MCS outlines continue across the edges of a global domain
    map_aspect=None,                                      # 'equal' (default) or 'auto'
    draw_border=None, draw_state=None, draw_river=None, map_resolution=None,       # None: the default of plot_mcs_tracks.py (no)
    # Style of each map feature (None: gray, 0.8 points, opaque, the same in every map): color, line width [points], transparency
    land_color=None, land_linewidth=None, land_alpha=None, border_color=None, border_linewidth=None, border_alpha=None,
    state_color=None, state_linewidth=None, state_alpha=None, river_color=None, river_linewidth=None, river_alpha=None,
    # Style of the tracks (None: the default look)
    track_color=None, track_lw=None, track_ls=None, track_alpha=None, track_marker=None, track_marker_size=None,
    track_text_size=None, track_text_color=None, track_text_alpha=None,
    track_halo_color=None, track_halo_width=None, track_halo_alpha=None,
    show_pfdiam=None,                                     # False: no PF diameter circles on the second panel (and not in its legend)
    skip_empty=None,                                      # True: no frame for times without a tracked MCS
    cbar_thickness=None, cbar_spacing=None,               # how thin the colorbars are; the gap between the 2 colorbars of a 1-panel map
    # What to run
    run_plotting=True,                                    # False: use the PNG files that exist
    run_ffmpeg=True,                                      # False: only the PNG files
    # mp4 (ffmpeg), see pyflextrkr/visualization/animation.py: PNG files per second (the speed), use every Nth PNG file, frame rate of the
    # video file (None: playback_fps repeated to at least 10 per second), quality (CRF, lower = better and larger), width in pixels, x264 preset
    playback_fps=2, frame_stride=1, video_fps=None, video_quality=24, output_width=1920, video_preset=None,
    # GIF from the mp4 (for READMEs): width in pixels (None: no GIF), frames per second (None: playback_fps), the part of the mp4
    # (start and duration in seconds, None: to the end), colors, dither
    gif_width=None, gif_fps=None, gif_start=0.0, gif_duration=None, gif_colors=256, gif_dither='none',
    # Output: the PNG files are in figdir, the animation in animation_dir. Default (all None): both in {root_path}/quicklooks_trackpaths/ of the
    # config file (the folder of the demo scripts). fig_root: the PNG files in {fig_root}/quicklooks_{plot_type}_{data_type}/{year}/, the
    # animation in {fig_root}/animations_{plot_type}_{data_type}/ (for runs whose folder is read-only or that you keep apart)
    fig_root=None, figdir=None, animation_dir=None, fig_basename=None,
)

# Old parameter names -> the names that replace them. orientation -> layout (horizontal -> top_bottom, vertical -> left_right);
# input_framerate -> playback_fps (PNG files per second); output_framerate -> video_fps (frame rate of the video file; now automatic if None)
DEPRECATED_NAMES = {'orientation': 'layout', 'input_framerate': 'playback_fps', 'output_framerate': 'video_fps'}

#-----------------------------------------------------------------------
def apply_overrides(params, argv=None, description=None):
    """Set parameters from the command line, --set NAME=VALUE (wrapper_common.apply_overrides); the old names in DEPRECATED_NAMES still work."""
    return wrapper_common.apply_overrides(params, DEFAULTS, 'pyflextrkr.visualization.mcs_animation.DEFAULTS', argv, description, DEPRECATED_NAMES)

#-----------------------------------------------------------------------
def complete_parameters(params):
    """The parameters of a wrapper with the defaults filled in, checked, and the names that depend on them."""
    params = dict(params)
    for old_name, new_name in DEPRECATED_NAMES.items():
        if old_name in params:
            print(f"DeprecationWarning: the parameter {old_name} is now called {new_name}")
            params[new_name] = params.pop(old_name)
    unknown = [k for k in params if k not in DEFAULTS]
    if unknown:
        print(f"WARNING: not a parameter of the animation (ignored): {', '.join(unknown)}")
    p = {**DEFAULTS, **{k: v for k, v in params.items() if k in DEFAULTS}}
    p['layout'] = layout_name(p['layout'])               # a deprecation warning for the old names
    if p['plot_type'] not in ("1panel", "2panel"):
        raise ValueError(f"Invalid plot_type: {p['plot_type']}. Must be one of ['1panel', '2panel']")
    if p['data_type'] not in ("tbpf", "tbze", "tb"):
        raise ValueError(f"Invalid data_type: {p['data_type']}. Must be one of ['tbpf', 'tbze', 'tb']")
    for key in ('start_date', 'end_date', 'config_file'):
        if p[key] is None:
            raise ValueError(f"The parameter {key} is not set")
    p['start_date'], p['end_date'] = str(to_plain(p['start_date'])), str(to_plain(p['end_date']))
    p['panels'] = 1 if (p['plot_type'] == "1panel" or p['data_type'] == "tb") else 2
    p['start_year'] = p['start_date'].split('-')[0]
    if p['fig_basename'] is None:
        p['fig_basename'] = f"mcs_{p['data_type']}_"
    if p['figdir'] is None or p['animation_dir'] is None:
        if p['fig_root'] is not None:
            root = p['fig_root'].rstrip('/') + '/'
            if p['figdir'] is None:
                p['figdir'] = f"{root}quicklooks_{p['plot_type']}_{p['data_type']}/{p['start_year']}/"
            if p['animation_dir'] is None:
                p['animation_dir'] = f"{root}animations_{p['plot_type']}_{p['data_type']}/"
        else:
            # next to the tracking output of the config file, like the demo scripts
            folder = wrapper_common.default_output_folder(p['config_file'])
            wrapper_common.check_writable(folder)
            p['figdir'] = p['figdir'] or folder
            p['animation_dir'] = p['animation_dir'] or folder
    return p

#-----------------------------------------------------------------------
def plot_options(p):
    """The options of plot_mcs_tracks.py for these parameters (the long option names without the dashes; None: the default)."""
    extent = [p['lon_min'], p['lon_max'], p['lat_min'], p['lat_max']]
    two_panels = p['panels'] == 2
    show_numbers = p['show_track_numbers']
    return dict(
        start=p['start_date'],
        end=p['end_date'],
        config=p['config_file'],
        trackstats_file=p['trackstats_file'],
        pixel_path=p['pixel_path'],
        track_lookback_days=p['track_lookback_days'],
        product=p['data_type'],
        panels=p['panels'],
        extent=extent if all(v is not None for v in extent) else None,
        subset=p['subset'],
        time_format=p['time_format'],
        parallel=p['parallel_mode'],
        workers=p['n_workers'],
        output=p['figdir'],
        title_prefix=p['title_prefix'],
        figsize=p['figsize'],
        figsize_x=p['figsize_x'],
        dpi=p['dpi'],
        figbasename=p['fig_basename'],
        fontsize=p['fontsize'],
        show_track_numbers=None if show_numbers is None else list(show_numbers)[:2 if two_panels else 1],
        projection=p['projection'],
        central_longitude=p['central_longitude'],
        central_latitude=p['central_latitude'],
        standard_parallels=p['standard_parallels'],
        lat_range=p['lat_range'],
        periodic_perimeter=p['periodic_perimeter'],
        map_aspect=p['map_aspect'],
        map_resolution=p['map_resolution'],
        land_color=p['land_color'], land_linewidth=p['land_linewidth'], land_alpha=p['land_alpha'],
        border_color=p['border_color'], border_linewidth=p['border_linewidth'], border_alpha=p['border_alpha'],
        state_color=p['state_color'], state_linewidth=p['state_linewidth'], state_alpha=p['state_alpha'],
        river_color=p['river_color'], river_linewidth=p['river_linewidth'], river_alpha=p['river_alpha'],
        track_color=p['track_color'],
        track_lw=p['track_lw'],
        track_ls=p['track_ls'],
        track_alpha=p['track_alpha'],
        track_marker=p['track_marker'],
        track_marker_size=p['track_marker_size'],
        track_text_size=p['track_text_size'],
        track_text_color=p['track_text_color'],
        track_text_alpha=p['track_text_alpha'],
        track_halo_color=p['track_halo_color'],
        track_halo_width=p['track_halo_width'],
        track_halo_alpha=p['track_halo_alpha'],
        show_pfdiam=p['show_pfdiam'] if two_panels else None,
        skip_empty=p['skip_empty'],
        cbar_thickness=p['cbar_thickness'],
        cbar_spacing=p['cbar_spacing'],
        # the layout of 2 panels (a 1-panel map does not use it)
        layout=p['layout'] if two_panels else None,
        # the variable of the second field: the reflectivity (only for data_type='tbze')
        var2_varname=p['dbz_varname'] if p['data_type'] == "tbze" else None,
        draw_border=p['draw_border'],
        draw_state=p['draw_state'],
        draw_river=p['draw_river'] if p['data_type'] == "tbpf" else None,
    )

#-----------------------------------------------------------------------
def main(params, argv=None, description=None):
    """
    Make the PNG frames, the mp4 and the GIF of the parameters of a wrapper script.

    Args:
        params: the parameters of the wrapper, i.e. its globals() (only the names of DEFAULTS are used)
        argv: command line (default: sys.argv[1:]), --set NAME=VALUE changes a parameter
        description: text of --help (the docstring of the wrapper)
    """
    values = wrapper_common.parameters_of(params)
    apply_overrides(values, argv, description)
    p = complete_parameters(values)

    start_date_str = p['start_date'].split('T')[0]  # YYYY-MM-DD
    end_date_str = p['end_date'].split('T')[0]
    animation_name = f"{p['animation_dir']}{p['fig_basename']}{start_date_str}_{end_date_str}"

    print("Make MCS tracking animation")
    print(f"Plot type: {p['plot_type']}")
    print(f"Data type: {p['data_type']}")
    print(f"Plotting: pyflextrkr.visualization.mcs_cli.main (visualization/plot_mcs_tracks.py)")
    print(f"Date range: {p['start_date']} to {p['end_date']}")
    print(f"Execution mode: Plotting={'✅' if p['run_plotting'] else '❌'}, FFmpeg={'✅' if p['run_ffmpeg'] else '❌'}")

    # Create directories if they don't exist
    if p['run_ffmpeg']:
        os.makedirs(p['animation_dir'], exist_ok=True)
    if p['run_plotting']:
        os.makedirs(p['figdir'], exist_ok=True)

    #########################################################
    # Plot the PNG frames
    #########################################################
    if p['run_plotting']:
        print(f"📊 Running plotting script with {p['n_workers']} workers...")
        argv_plot = options_to_argv(plot_options(p))
        print(f"Options: {' '.join(argv_plot)}")
        try:
            plot_mcs_tracks(argv_plot)
        except SystemExit as err:
            if err.code not in (None, 0):
                print(f"❌ Error: Plotting failed: {err.code}")
                sys.exit(1)
        print("✅ Plotting completed successfully!")
    else:
        print("⏭️  Skipping plotting - using existing PNG files")

    #########################################################
    # mp4 (and GIF) with ffmpeg
    #########################################################
    if not p['run_ffmpeg']:
        print("⏭️  Skipping animation creation - PNG files ready for manual processing")
        return

    print("🎬 Creating animation from PNG files...")
    # All PNG files of the figure name within the date range
    expected_files = frames_in_range(p['figdir'], p['fig_basename'], p['start_date'], p['end_date'])
    print(f"Found {len(expected_files)} PNG files within date range")
    print(f"Time range: {p['start_date']} to {p['end_date']}")
    if len(expected_files) == 0:
        print("❌ No PNG files found within the specified date range!")
        print(f"   Check directory: {p['figdir']}")
        print(f"   Expected pattern: {p['fig_basename']}YYYYMMDD_HHMMSS.png")
        sys.exit(1)

    mp4_file = f"{animation_name}.mp4"
    returncode = make_animation(expected_files, mp4_file, playback_fps=p['playback_fps'], frame_stride=p['frame_stride'],
                                video_fps=p['video_fps'], video_quality=p['video_quality'], output_width=p['output_width'],
                                video_preset=p['video_preset'], gif_width=p['gif_width'], gif_fps=p['gif_fps'],
                                gif_start=p['gif_start'], gif_duration=p['gif_duration'], gif_colors=p['gif_colors'],
                                gif_dither=p['gif_dither'])
    if returncode != 0:
        sys.exit(1)
    print(f"🎬 View animation here: {mp4_file}")
