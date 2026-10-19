"""
Convective cell tracking animation from a set of parameters: the PNG frames (cell_cli.main, the options of plot_cell_tracks.py), the mp4
and optionally a GIF (animation.py).

The wrapper scripts keep the parameters as variables at the top of the file, so you can edit them or set them on the command
line (--set NAME=VALUE), and call main(globals()):
    visualization/examples/make_cell_animation_nexrad.py          the NEXRAD demo (KHGX) on a square map around the radar
A wrapper sets the parameters it needs; DEFAULTS below lists all of them (a parameter left out has the value listed there).
The PNG files and the animation go to {root_path}/quicklooks_trackpaths/ of the tracking config file (the folder of the demo scripts)
unless fig_root, or figdir and animation_dir, are set.

Called by: the wrapper scripts above. Calls: cell_cli.main (the PNG frames), animation.make_animation (mp4 and GIF with ffmpeg).
"""
import math
import os
import sys
from pyflextrkr.visualization.cli_common import options_to_argv
from pyflextrkr.visualization.cell_cli import main as plot_cell_tracks
from pyflextrkr.visualization.animation import frames_in_range, make_animation
from pyflextrkr.visualization import wrapper_common
from pyflextrkr.visualization.wrapper_common import to_plain

# All parameters and their values when a wrapper does not set them (None: the default of plot_cell_tracks.py)
DEFAULTS = dict(
    start_date=None, end_date=None,                       # 'YYYY-MM-DDTHH' (or with :MM)
    config_file=None,                                     # tracking config file
    product='refl',                                       # the composite reflectivity ('refl') or the 10 dBZ echo-top height ('eth')
    varname=None,                                         # variable in the pixel-level files (None: dbz_comp, echotop10 for 'eth')
    # Radar location (None: the pixel closest to the center of the domain)
    radar_lat=None, radar_lon=None,
    # Map extent lon_min, lon_max, lat_min, lat_max (None: the whole domain of the data); subset 1 reads only the pixels of the extent.
    # map_half_width_km: a square of 2 * map_half_width_km around the radar instead (needs radar_lat, radar_lon); with a LambertConformal or
    # AlbersEqualArea projection that has no center of its own, the map is centered on the radar, so the range circles are round
    lon_min=None, lon_max=None, lat_min=None, lat_max=None, subset=0, map_half_width_km=None,
    time_format='yyyymodd_hhmmss',                        # time format in the pixel-level file names
    track_lookback_days=None,                             # days before start_date from which the tracks are drawn (None: 0)
    parallel_mode=1, n_workers=64,
    figsize=None, figsize_x=10, dpi=None,                 # figsize [width, height] in inches, or figsize_x (the height follows the map)
    title_prefix='',                                      # the title is "{title_prefix} | {time}"
    fontsize=None, fontsize_tracks=None,
    # Map projection (None: PlateCarree) and its parameters, the lat/lon labels and gridlines
    projection=None, central_longitude=None, central_latitude=None, standard_parallels=None,
    map_aspect=None, map_resolution=None, lon_ticks=None, lat_ticks=None,
    grid_linewidth=None, grid_linestyle=None, grid_alpha=None,
    # Map features to draw and the style of each one (None: gray, 0.8 points, opaque, the same in every map): color, line width, transparency
    draw_land=None, draw_border=None, draw_state=None, draw_river=None,
    land_color=None, land_linewidth=None, land_alpha=None, border_color=None, border_linewidth=None, border_alpha=None,
    state_color=None, state_linewidth=None, state_alpha=None, river_color=None, river_linewidth=None, river_alpha=None,
    # What to draw on the radar map
    show_rangecircle=None, show_azimuth=None, show_tracks=None, show_paths=None, show_symbols=None,
    show_terrain=False,                                   # terrain height under the reflectivity (terrain_file and elev_varname of the config file)
    # Cell outlines: color, transparency, width [pixels]; one color per track (a colormap name such as 'glasbey_dark'); a ring just outside
    # them; only the tracks that live at least min_hours [hour]
    perim_color=None, perim_alpha=None, perim_width=None, perim_cmap=None, perim_halo=None, min_hours=0.0,
    # Cities and other radars: [['name', lon, lat], ...]; a radar gets a range circle of radar_site_radius [km] (0: none)
    cities=None, radar_sites=None, radar_site_radius=None,
    # Style of the tracks (None: the default look)
    track_color=None, track_lw=None, track_ls=None, track_alpha=None, track_marker=None, track_marker_size=None,
    track_text_size=None, track_text_color=None, track_text_alpha=None,
    track_halo_color=None, track_halo_width=None, track_halo_alpha=None,
    skip_empty=None,                                      # True: no frame for times without a tracked cell
    # What to run
    run_plotting=True,                                    # False: use the PNG files that exist
    run_ffmpeg=True,                                      # False: only the PNG files
    # mp4 (ffmpeg), see pyflextrkr/visualization/animation.py: PNG files per second (the speed), use every Nth PNG file, frame rate of the
    # video file (None: playback_fps repeated to at least 10 per second), quality (CRF, lower = better and larger), width in pixels (None: the
    # width of the PNG files), x264 preset
    playback_fps=4, frame_stride=1, video_fps=None, video_quality=24, output_width=1920, video_preset=None,
    # GIF from the mp4 (for READMEs): width in pixels (None: no GIF), frames per second (None: playback_fps), the part of the mp4
    # (start and duration in seconds, None: to the end), colors, dither
    gif_width=None, gif_fps=None, gif_start=0.0, gif_duration=None, gif_colors=256, gif_dither='none',
    # Output: the PNG files are in figdir, the animation in animation_dir. Default (all None): both in {root_path}/quicklooks_trackpaths/ of the
    # config file (the folder of the demo scripts). fig_root: the PNG files in {fig_root}/quicklooks_cell/, the animation in
    # {fig_root}/animations_cell/ (for runs whose folder is read-only or that you keep apart)
    fig_root=None, figdir=None, animation_dir=None, fig_basename='cell_tracks_',
)

# The projections that are centered on the radar when map_half_width_km is set and they have no center of their own
CENTERED_PROJECTIONS = ('lambertconformal', 'albersequalarea')

#-----------------------------------------------------------------------
def square_extent(radar_lat, radar_lon, half_width_km):
    """[lon_min, lon_max, lat_min, lat_max] of a square of 2 * half_width_km around the radar (1 degree of latitude: 111.19 km)."""
    lat_half = half_width_km / 111.19
    lon_half = half_width_km / (111.19 * math.cos(math.radians(radar_lat)))
    return [round(radar_lon - lon_half, 3), round(radar_lon + lon_half, 3), round(radar_lat - lat_half, 3), round(radar_lat + lat_half, 3)]

#-----------------------------------------------------------------------
def complete_parameters(params):
    """The parameters of a wrapper with the defaults filled in, checked, and the names that depend on them."""
    params = dict(params)
    unknown = [k for k in params if k not in DEFAULTS]
    if unknown:
        print(f"WARNING: not a parameter of the animation (ignored): {', '.join(unknown)}")
    p = {**DEFAULTS, **{k: v for k, v in params.items() if k in DEFAULTS}}
    for key in ('start_date', 'end_date', 'config_file'):
        if p[key] is None:
            raise ValueError(f"The parameter {key} is not set")
    p['start_date'], p['end_date'] = str(to_plain(p['start_date'])), str(to_plain(p['end_date']))
    # A square map around the radar (and, for a projection without a center of its own, centered on it)
    if p['map_half_width_km'] is not None:
        if p['radar_lat'] is None or p['radar_lon'] is None:
            raise ValueError("map_half_width_km needs radar_lat and radar_lon")
        p['lon_min'], p['lon_max'], p['lat_min'], p['lat_max'] = square_extent(p['radar_lat'], p['radar_lon'], p['map_half_width_km'])
        if p['projection'] is not None and str(p['projection']).lower().replace('_', '') in CENTERED_PROJECTIONS:
            if p['central_longitude'] is None:
                p['central_longitude'] = p['radar_lon']
            if p['central_latitude'] is None:
                p['central_latitude'] = p['radar_lat']
            if p['standard_parallels'] is None:
                p['standard_parallels'] = [p['radar_lat']]
    if p['figdir'] is None or p['animation_dir'] is None:
        if p['fig_root'] is not None:
            root = p['fig_root'].rstrip('/') + '/'
            if p['figdir'] is None:
                p['figdir'] = f"{root}quicklooks_cell/"
            if p['animation_dir'] is None:
                p['animation_dir'] = f"{root}animations_cell/"
        else:
            # next to the tracking output of the config file, like the demo scripts
            folder = wrapper_common.default_output_folder(p['config_file'])
            wrapper_common.check_writable(folder)
            p['figdir'] = p['figdir'] or folder
            p['animation_dir'] = p['animation_dir'] or folder
    return p

#-----------------------------------------------------------------------
def plot_options(p):
    """The options of plot_cell_tracks.py for these parameters (the long option names without the dashes; None: the default)."""
    extent = [p['lon_min'], p['lon_max'], p['lat_min'], p['lat_max']]
    return dict(
        start=p['start_date'],
        end=p['end_date'],
        config=p['config_file'],
        product=p['product'],
        varname=p['varname'],
        time_format=p['time_format'],
        radar_lat=p['radar_lat'],
        radar_lon=p['radar_lon'],
        extent=extent if all(v is not None for v in extent) else None,
        subset=p['subset'],
        track_lookback_days=p['track_lookback_days'],
        parallel=p['parallel_mode'],
        workers=p['n_workers'],
        output=p['figdir'],
        figsize=p['figsize'],
        figsize_x=p['figsize_x'],
        dpi=p['dpi'],
        figbasename=p['fig_basename'],
        title_prefix=p['title_prefix'],
        fontsize=p['fontsize'],
        fontsize_tracks=p['fontsize_tracks'],
        projection=p['projection'],
        central_longitude=p['central_longitude'],
        central_latitude=p['central_latitude'],
        standard_parallels=p['standard_parallels'],
        map_aspect=p['map_aspect'],
        map_resolution=p['map_resolution'],
        lon_ticks=p['lon_ticks'],
        lat_ticks=p['lat_ticks'],
        grid_linewidth=p['grid_linewidth'],
        grid_linestyle=p['grid_linestyle'],
        grid_alpha=p['grid_alpha'],
        draw_land=p['draw_land'], draw_border=p['draw_border'], draw_state=p['draw_state'], draw_river=p['draw_river'],
        land_color=p['land_color'], land_linewidth=p['land_linewidth'], land_alpha=p['land_alpha'],
        border_color=p['border_color'], border_linewidth=p['border_linewidth'], border_alpha=p['border_alpha'],
        state_color=p['state_color'], state_linewidth=p['state_linewidth'], state_alpha=p['state_alpha'],
        river_color=p['river_color'], river_linewidth=p['river_linewidth'], river_alpha=p['river_alpha'],
        show_rangecircle=p['show_rangecircle'],
        show_azimuth=p['show_azimuth'],
        show_tracks=p['show_tracks'],
        show_paths=p['show_paths'],
        show_symbols=p['show_symbols'],
        show_terrain=p['show_terrain'],
        perim_color=p['perim_color'],
        perim_alpha=p['perim_alpha'],
        perim_width=p['perim_width'],
        perim_cmap=p['perim_cmap'],
        perim_halo=p['perim_halo'],
        min_hours=p['min_hours'],
        city=p['cities'],
        radar_site=p['radar_sites'],
        radar_site_radius=p['radar_site_radius'],
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
        skip_empty=p['skip_empty'],
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
    wrapper_common.apply_overrides(values, DEFAULTS, 'pyflextrkr.visualization.cell_animation.DEFAULTS', argv, description)
    p = complete_parameters(values)

    animation_name = f"{p['animation_dir']}{p['fig_basename']}{p['start_date'][:13]}_{p['end_date'][:13]}"
    print("Make cell tracking animation")
    print(f"Plotting: pyflextrkr.visualization.cell_cli.main (visualization/plot_cell_tracks.py)")
    print(f"Date range: {p['start_date']} to {p['end_date']}")
    print(f"Radar location: ({p['radar_lat']}, {p['radar_lon']})")
    print(f"Execution mode: Plotting={'yes' if p['run_plotting'] else 'no'}, FFmpeg={'yes' if p['run_ffmpeg'] else 'no'}")

    #########################################################
    # Plot the PNG frames
    #########################################################
    if p['run_plotting']:
        os.makedirs(p['figdir'], exist_ok=True)
        argv_plot = options_to_argv(plot_options(p))
        print(f"Options: {' '.join(argv_plot)}")
        try:
            plot_cell_tracks(argv_plot)
        except SystemExit as err:
            if err.code not in (None, 0):
                print(f"Error: plotting failed: {err.code}")
                sys.exit(1)
        print("Plotting completed")
    else:
        print("Skipping plotting: using the existing PNG files")

    #########################################################
    # mp4 (and GIF) with ffmpeg
    #########################################################
    if not p['run_ffmpeg']:
        print("Skipping the animation: the PNG files are ready")
        return
    os.makedirs(p['animation_dir'], exist_ok=True)
    png_files = frames_in_range(p['figdir'], p['fig_basename'], p['start_date'], p['end_date'])
    print(f"Found {len(png_files)} PNG files in {p['start_date']} - {p['end_date']}")
    if len(png_files) == 0:
        sys.exit(f"No PNG files {p['figdir']}{p['fig_basename']}YYYYMMDD_HHMMSS.png in the period")
    mp4_file = f"{animation_name}.mp4"
    returncode = make_animation(png_files, mp4_file, playback_fps=p['playback_fps'], frame_stride=p['frame_stride'],
                                video_fps=p['video_fps'], video_quality=p['video_quality'], output_width=p['output_width'],
                                video_preset=p['video_preset'], gif_width=p['gif_width'], gif_fps=p['gif_fps'],
                                gif_start=p['gif_start'], gif_duration=p['gif_duration'], gif_colors=p['gif_colors'],
                                gif_dither=p['gif_dither'])
    if returncode != 0:
        sys.exit(1)
    print(f"Animation: {mp4_file}")
