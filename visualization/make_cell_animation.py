#!/usr/bin/env python
"""
Make cell tracking animation
- Plot the PNG frames: calls main() of pyflextrkr/visualization/cell_cli.py (the options of plot_cell_tracks.py)
- Create a video animation using FFmpeg (pyflextrkr/visualization/animation.py)

Edit the parameters below, or set them on the command line without editing the file:
>python make_cell_animation.py --set start_date=2020-06-14T00:00 --set end_date=2020-06-14T06:00 --set n_workers=32

Author: Zhe Feng | zhe.feng@pnnl.gov
"""

import os
import sys
import argparse
import inspect
import yaml
from pyflextrkr.visualization.cli_common import options_to_argv
from pyflextrkr.visualization.cell_cli import main as plot_cell_tracks
from pyflextrkr.visualization.animation import frames_in_range, make_mp4

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
# Figure parameters
subset = 1  # Subset data before plotting (0:no, 1:yes)
fig_basename = "cell_tracks_"

# Output directories
figdir = f"/global/cfs/cdirs/m1657/zfeng/SCREAMv1-cess2/cell_conus/quicklooks/"
animation_dir = f"/global/cfs/cdirs/m1657/zfeng/SCREAMv1-cess2/cell_conus/animations/"

# Execution control options
run_plotting = True   # Set to False to skip plotting and use existing PNG files
run_ffmpeg = True     # Set to False to skip animation creation (plotting only)

# FFmpeg animation parameters
input_framerate = 2    # (frames per second) - how fast to transition between frames
output_framerate = 10  # (frames per second) - video playback speed (lower values = smaller file size, e.g., 24 fps is cinema)
video_quality = 20     # CRF value (lower = better quality, range 0-51, 18-28 is good)
output_width = 1920    # Output video width in pixels (height auto-calculated to maintain aspect ratio, set to None to keep original size)
video_preset = None    # x264 preset (ultrafast ... medium ... veryslow); None = no -preset option (x264 default, medium). A slower preset
                       # makes the file only about 1 % smaller, see docs/visualization/animations.md

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

        print(f"🎬 Animation settings:")
        print(f"   Input framerate: {input_framerate} fps (PNG reading speed)")
        print(f"   Output framerate: {output_framerate} fps (video playback speed)")
        print(f"   Video quality (CRF): {video_quality} (lower=better), x264 preset {video_preset}")
        print(f"   Output width: {output_width if output_width else 'original'} pixels (height auto-scaled)")
        print(f"Using {len(expected_files)} PNG files from {expected_files[0]} to {expected_files[-1]}")

        returncode, ffmpeg_cmd = make_mp4(expected_files, animation_filename, input_framerate=input_framerate,
                                          output_framerate=output_framerate, crf=video_quality, width=output_width,
                                          preset=video_preset)
        print(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")

        if returncode == 0:
            print(f"✅ Animation created successfully!")
            print(f"🎬 View animation here: {animation_filename}")
        else:
            print(f"❌ Error: FFmpeg failed with exit code {returncode}")

    else:
        print("⏭️  Skipping animation creation - PNG files ready for manual processing")


# The Dask workers import this file: the work is under __main__
if __name__ == "__main__":
    main()
