#!/usr/bin/env python
"""
Make cell tracking animation
- Plot the PNG frames: calls main() of pyflextrkr/visualization/cell_cli.py (the options of plot_cell_tracks.py)
- Create a video animation using FFmpeg (pyflextrkr/visualization/animation.py)

Edit the parameters below, or set them on the command line without editing the file:
>python make_cell_animation.py --set start_date=2020-06-14T00:00 --set end_date=2020-06-14T06:00 --set n_workers=32

The mp4 of PNG files that are already plotted, with ffmpeg alone (explained in visualization/make_mcs_animation.py; a directory
or a list of PNG files works with `python pngs_to_animation.py FRAMES movie.mp4 --playback_fps 12`):
>ffmpeg -framerate 12 -pattern_type glob -i '/global/cfs/cdirs/m1657/zfeng/SCREAMv1-cess2/cell_conus/quicklooks/cell_tracks_*.png' \\
>       -vf "scale='min(1920,trunc(iw/2)*2)':-2" -c:v libx264 -r 12 -crf 24 -pix_fmt yuv420p -y movie.mp4

Author: Zhe Feng | zhe.feng@pnnl.gov
"""

import os
import sys
import argparse
import inspect
import yaml
from pyflextrkr.visualization.cli_common import options_to_argv
from pyflextrkr.visualization.cell_cli import main as plot_cell_tracks
from pyflextrkr.visualization.animation import frames_in_range, make_animation

###############################################################################################
# Script parameters
###############################################################################################

# Script parameters
start_date = "2020-06-13T00:00"
end_date = "2020-06-16T00:00"

# Radar location (optional - if not provided, defaults to domain center)
radar_lat = 34.723
radar_lon = 273.428

# Define domain map extent (optional - if not provided, defaults to full domain)
lon_min = radar_lon - 6.0
lon_max = radar_lon + 6.0
lat_min = radar_lat - 4
lat_max = radar_lat + 4

# Get start year from start_date
start_year = str(start_date.split('-')[0])

# Tracking config file
config_file = f"/global/homes/f/feng045/program/scream/config/config_celltracking_3km_SCREAMv1-Cess2_CONUS.yaml"

# Execution parameters
parallel_mode = 1
n_workers = 128
figsize_x = 10  # Width in inches (height is auto-calculated to maintain aspect ratio)

# Tracking pixel-level file time format
time_format = 'yyyymodd_hhmmss'

# Variable name for plotting in pixel files
varname = 'dbz_comp'
# Plot the composite reflectivity ('refl') or the 10 dBZ echo-top height ('eth')
product = 'refl'

# Title prefix for plots (added before date/time, can be an empty string)
title_prefix = 'SCREAM Cess2'

# Control features to draw
draw_land = True  # Draw land features
draw_border = True  # Draw country borders
draw_state = True    # Draw state/province borders
show_rangecircle = False  # Draw radar range circles
show_azimuth = False      # Draw azimuth lines
show_tracks = True        # Show tracks
show_paths = True         # Show track path lines (None = auto from show_tracks)
show_symbols = False       # Show track centroid symbols (None = auto from show_tracks)
# Font sizes
fontsize = 13              # Main font size for labels and text
fontsize_tracks = 6        # Font size for track numbers (None = auto-calculate as fontsize*0.8)
# Map resolution for Natural Earth features (Cartopy options: '10m', '50m', '110m')
# '10m' = highest detail (slowest), '50m' = medium detail, '110m' = lowest detail (fastest)
map_resolution = '50m'
# Map projection (None: PlateCarree) and style of the tracks (None: the default look)
projection = None            # PlateCarree, LambertConformal, Robinson, ...
central_longitude = None
track_color = None           # path color (default black)
track_lw = None              # path line width
track_halo_color = None      # halo (outline) around paths and numbers (None: no halo)
track_halo_width = None
track_halo_alpha = None
# Cell outlines: one color per track (a colormap name, e.g. 'glasbey_dark'; None: all outlines in one color), a ring just outside
# them (a color; None: no ring), and only the cells of the tracks that live at least min_hours [hour] (0: all)
perim_cmap = None
perim_halo = None
min_hours = 0.0
# Terrain height under the reflectivity (terrain_file and elev_varname of the tracking config file)
show_terrain = False
# Cities and other radars: [['name', lon, lat], ...]; a radar gets a range circle of radar_site_radius [km] (0: none)
cities = None
radar_sites = None
radar_site_radius = 150
# Line style of the lat/lon gridlines (None: '--')
grid_linestyle = None
# Figure parameters
subset = 1  # Subset data before plotting (0:no, 1:yes)
fig_basename = "cell_tracks_"

# Output directories
figdir = f"/global/cfs/cdirs/m1657/zfeng/SCREAMv1-cess2/cell_conus/quicklooks/"
animation_dir = f"/global/cfs/cdirs/m1657/zfeng/SCREAMv1-cess2/cell_conus/animations/"

# Execution control options
run_plotting = True   # Set to False to skip plotting and use existing PNG files
run_ffmpeg = True     # Set to False to skip animation creation (plotting only)

# FFmpeg animation parameters (explained in visualization/make_mcs_animation.py, tested in docs/visualization/animations.md)
playback_fps = 12      # PNG files shown per second = the playback speed of the animation (larger = faster). 5-minute files: 12 means
                       # 1 hour of data per second, so 3 days (864 files) take 72 s; hourly files would use 2
frame_stride = 1       # use every Nth PNG file (2 = every second file): the other way to shorten a long period
video_fps = None       # frame rate of the video file. None = automatic: playback_fps repeated to at least 10 fps with a whole
                       # number of repeats per PNG file (12 -> 12, 2 -> 10). It does not change the speed; players handle very
                       # low frame rates badly. Keep it a whole multiple of playback_fps
video_quality = 24     # CRF value of libx264 (lower = better quality and larger file, range 0-51; 18-28 is the useful range).
                       # +6 CRF makes the file 33-55 % smaller. The CONUS MRMS maps (288 frames, 1920 px): CRF 20: 33.7 MB (SSIM 0.9952),
                       # 22: 28.2 MB (0.9935), 24: 23.5 MB (0.9911, -30 %, thin lines and markers look the same), 26: 19.2 MB (0.9878, -43 %)
output_width = 1920    # Output video width in pixels, never enlarged (height auto-calculated to maintain aspect ratio; None: the size
                       # of the PNG files)
video_preset = None    # x264 preset: ultrafast ... fast, medium (x264 default), slow, slower, veryslow; None = no -preset option (medium).
                       # A slower preset makes the file only about 1 % smaller, see docs/visualization/animations.md

###############################################################################################
# Main execution
###############################################################################################

# The parameters above (names that can be set with --set)
_PARAMETERS = [k for k, v in list(globals().items()) if not k.startswith('_') and not callable(v) and not inspect.ismodule(v)]

def apply_overrides():
    """Set the parameters above from the command line: --set NAME=VALUE (a value is read as yaml: numbers, true, null, lists)."""
    ap = argparse.ArgumentParser(description="Make cell tracking animation. The parameters are the variables at the top of the file.")
    ap.add_argument('--set', action='append', default=[], metavar='NAME=VALUE',
                    help='set a parameter, e.g. --set start_date=2020-06-14T00:00 --set n_workers=32 --set radar_lat=34.7')
    g = globals()
    for item in ap.parse_args().set:
        name, sep, value = item.partition('=')
        if not sep or name not in _PARAMETERS:
            sys.exit(f"ERROR: --set {item}: expected NAME=VALUE with NAME one of the parameters at the top of the file")
        g[name] = yaml.safe_load(value)

def main():

    apply_overrides()

    # Animation parameters
    start_date_str = start_date[:13]  # Extract YYYY-MM-DDTHH
    end_date_str = end_date[:13]      # Extract YYYY-MM-DDTHH
    animation_filename = f"{animation_dir}{fig_basename}{start_date_str}_{end_date_str}.mp4"

    print("Make cell tracking animation")
    print(f"Plotting: pyflextrkr.visualization.cell_cli.main (visualization/plot_cell_tracks.py)")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Radar location: ({radar_lat}, {radar_lon})")
    print(f"Execution mode: Plotting={'✅' if run_plotting else '❌'}, FFmpeg={'✅' if run_ffmpeg else '❌'}")

    # Create directories if they don't exist
    if run_ffmpeg:
        os.makedirs(animation_dir, exist_ok=True)
    if run_plotting:
        os.makedirs(figdir, exist_ok=True)

    #########################################################
    # Run plotting script
    #########################################################
    if run_plotting:
        print(f"📊 Running plotting script with {n_workers} workers...")
        # The options of plot_cell_tracks.py (the long option names without the dashes; None: the default)
        extent = [lon_min, lon_max, lat_min, lat_max]
        options = dict(
            start=start_date,
            end=end_date,
            config=config_file,
            product=product,
            # Add radar location if provided
            radar_lat=radar_lat,
            radar_lon=radar_lon,
            # Add extent if all values are provided
            extent=extent if all(v is not None for v in extent) else None,
            subset=subset,
            time_format=time_format,
            parallel=parallel_mode,
            workers=n_workers,
            output=figdir,
            figsize_x=figsize_x,
            figbasename=fig_basename,
            varname=varname,
            draw_land=draw_land,
            draw_border=draw_border,
            draw_state=draw_state,
            show_rangecircle=show_rangecircle,
            show_azimuth=show_azimuth,
            show_tracks=show_tracks,
            fontsize=fontsize,
            map_resolution=map_resolution,
            title_prefix=title_prefix,
            # Optional show_paths and show_symbols (not None)
            show_paths=show_paths,
            show_symbols=show_symbols,
            fontsize_tracks=fontsize_tracks,
            projection=projection,
            central_longitude=central_longitude,
            track_color=track_color,
            track_lw=track_lw,
            track_halo_color=track_halo_color,
            track_halo_width=track_halo_width,
            track_halo_alpha=track_halo_alpha,
            perim_cmap=perim_cmap,
            perim_halo=perim_halo,
            min_hours=min_hours,
            show_terrain=show_terrain,
            city=cities,
            radar_site=radar_sites,
            radar_site_radius=radar_site_radius,
            grid_linestyle=grid_linestyle,
        )
        argv = options_to_argv(options)

        print(f"Options: {' '.join(argv)}")
        try:
            plot_cell_tracks(argv)
        except SystemExit as err:
            if err.code not in (None, 0):
                print(f"❌ Error: Plotting failed: {err.code}")
                sys.exit(1)

        print("✅ Plotting completed successfully!")
    else:
        print("⏭️  Skipping plotting - using existing PNG files")

    #########################################################
    # Make animation using ffmpeg
    #########################################################
    if run_ffmpeg:
        print("🎬 Creating animation from PNG files...")

        # Get all PNG files matching the basename pattern within the date range
        expected_files = frames_in_range(figdir, fig_basename, start_date, end_date)

        print(f"Found {len(expected_files)} PNG files within date range")
        print(f"Time range: {start_date} to {end_date}")

        if len(expected_files) == 0:
            print("❌ No PNG files found within the specified date range!")
            print(f"   Check directory: {figdir}")
            print(f"   Expected pattern: {fig_basename}YYYYMMDD_HHMMSS.png")
            sys.exit(1)

        # The settings are printed, then the mp4 is made (animation.make_animation)
        returncode = make_animation(expected_files, animation_filename, playback_fps=playback_fps, frame_stride=frame_stride,
                                    video_fps=video_fps, video_quality=video_quality, output_width=output_width,
                                    video_preset=video_preset)
        if returncode == 0:
            print(f"🎬 View animation here: {animation_filename}")
        else:
            sys.exit(1)

    else:
        print("⏭️  Skipping animation creation - PNG files ready for manual processing")


# The Dask workers import this file: the work is under __main__
if __name__ == "__main__":
    main()
