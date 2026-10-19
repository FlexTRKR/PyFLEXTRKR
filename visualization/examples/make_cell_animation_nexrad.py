#!/usr/bin/env python
"""
Convective cell tracking animation of the NEXRAD demo (config/demo_cell_nexrad.sh): composite reflectivity of the KHGX radar (Houston,
7 August 2014, 12-15 UTC) with the cell outlines, track paths, cell numbers and the radar range rings, as an mp4 and a GIF.

Run the demo first (it makes the tracking outputs and config/config_cell_demo.yml), then
>python make_cell_animation_nexrad.py
or with the config file of your own run, another period or more workers, without editing the file:
>python make_cell_animation_nexrad.py --set config_file=/path/to/config.yml --set n_workers=32

The map is a square around the radar, with round range circles: a PlateCarree map (the default) keeps 1 degree of longitude as wide
as 1 degree of latitude, but at 29.5N 1 degree of longitude is only 0.87 times as long, so the circles come out 15 % too wide. A
LambertConformal map centered on the radar keeps the distances (the domain is half_width_km to each side of the radar) and the circles are round.

The parameters that are not set here (cell outlines, track style, ...) have the defaults of visualization/plot_cell_tracks.py;
visualization/make_cell_animation.py lists more of them (terrain, cities, other radars, ...).

The same frames with plot_cell_tracks.py alone (every option of the dictionary 'options' below is an option of the script):
>python ../plot_cell_tracks.py -s 2014-08-07T12:00 -e 2014-08-07T15:00 -c ../../config/config_cell_demo.yml \
        --radar_lat 29.4719 --radar_lon -95.0787 --extent -96.164 -93.994 28.528 30.416 \
        --projection LambertConformal --central_longitude -95.0787 --central_latitude 29.4719 --standard_parallels 29.4719 \
        --title_prefix "NEXRAD (KHGX)" --draw_land 1 --land_linewidth 2 --draw_border 1 --draw_state 1 --map_resolution 10m \
        --figsize 8 7.2 --lon_ticks -96 -95.5 -95 -94.5 -94 --lat_ticks 28.5 29 29.5 30 --output frames

The mp4 and the GIF of PNG files that are already plotted, with pngs_to_animation.py (--print_command shows the ffmpeg commands):
>python ../pngs_to_animation.py frames/ cell_nexrad.mp4 --playback_fps 4 --video_quality 18 --gif_width 800
"""

import os
import sys
import argparse
import inspect
import math
import yaml
from pyflextrkr.visualization.cli_common import options_to_argv
from pyflextrkr.visualization.cell_cli import main as plot_cell_tracks
from pyflextrkr.visualization.animation import frames_in_range, make_animation

###############################################################################################
# Script parameters
###############################################################################################

# Period of the demo (5-minute NEXRAD volumes)
start_date = "2014-08-07T12:00"
end_date = "2014-08-07T15:00"

# The config file that config/demo_cell_nexrad.sh writes
config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config', 'config_cell_demo.yml')
# Tracking pixel-level file time format
time_format = 'yyyymodd_hhmmss'

# Radar location (KHGX)
radar_lat = 29.4719
radar_lon = -95.0787

# Map: a square of 2 * half_width_km around the radar (the 100 km range circle is inside), as a LambertConformal map centered on the
# radar so that the range circles are round (projection = None: PlateCarree, where they are ellipses; see the top of the file)
half_width_km = 105.
projection = 'LambertConformal'
# Extent [lon_min, lon_max, lat_min, lat_max] of the square: 1 degree of latitude is 111.19 km, 1 degree of longitude is cos(lat) times that
lat_half = half_width_km / 111.19
lon_half = half_width_km / (111.19 * math.cos(math.radians(radar_lat)))
lon_min = round(radar_lon - lon_half, 3)
lon_max = round(radar_lon + lon_half, 3)
lat_min = round(radar_lat - lat_half, 3)
lat_max = round(radar_lat + lat_half, 3)

# Execution parameters
parallel_mode = 1
n_workers = 16
# Figure size [width, height] in inches (300 dpi: 2400 x 2160 pixels). The map panel of the layout is 0.757 of the width and 0.84 of the
# height of the figure, so 8 x 7.2 inches make a square panel for the square map (without figsize the height would follow from the
# shape of the map but not be shorter than the layout needs for 2 titles: 9.2 inches, with white space above and below the map)
figsize = [8, 7.2]
# Longitudes and latitudes of the tick labels (every 0.5 degree: the default ticks every 0.25 degree have labels like 30.25N that are
# cut off at the left edge of the figure)
lon_ticks = [-96, -95.5, -95, -94.5, -94]
lat_ticks = [28.5, 29, 29.5, 30]

# Plot the composite reflectivity ('refl') or the 10 dBZ echo-top height ('eth')
product = 'refl'
# Title prefix: the title is "NEXRAD (KHGX) | 2014-08-07 13:05:53 UTC" (can be an empty string)
title_prefix = 'NEXRAD (KHGX)'

# Features to draw: coastlines (land_linewidth is their line width, the borders and states have thinner lines)
draw_land = True
land_linewidth = 2
draw_border = True    # country borders
draw_state = True     # state borders
show_rangecircle = True   # range circles every 20 km up to 100 km
show_azimuth = True       # azimuth lines every 90 degrees
show_tracks = True
fontsize = 13
map_resolution = '10m'   # Natural Earth features: '10m' (most detail: the coastline at 2 points width needs it), '50m', '110m'

# Figure file names {figdir}{fig_basename}YYYYMMDD_HHMMSS.png
fig_basename = "cell_tracks_"
# PNG files go to {fig_root}quicklooks_cell/, the animation to {fig_root}animations_cell/
fig_root = os.path.join(os.environ.get('SCRATCH', os.getcwd()), 'pyflextrkr_animations', 'nexrad_cell') + os.sep
import pdb; pdb.set_trace()
# Execution control options
run_plotting = True   # False: skip the plotting and use the PNG files that exist
run_ffmpeg = True     # False: plotting only

# FFmpeg animation parameters (explained in visualization/make_mcs_animation.py, tested in docs/visualization/animations.md)
playback_fps = 4       # PNG files shown per second = the playback speed (larger = faster): 5-minute volumes, so 20 minutes of data per
                       # second, the 34 volumes take 8.5 s
frame_stride = 1       # use every Nth PNG file (2 = every second file)
video_fps = None       # frame rate of the video file, None = playback_fps repeated to at least 10 fps (4 -> 12); the speed is the same
video_quality = 18     # CRF of libx264 (lower = better quality and larger file; 18-28 is useful, 24 is the usual choice). The GIF is made
                       # from this mp4, so it is kept near lossless: a lower CRF makes the mp4 larger but the GIF only a few % larger
output_width = None    # video width in pixels, None = the width of the PNG files (never enlarged)
video_preset = None    # x264 preset, None = the x264 default (medium)
# GIF made from the mp4 (one palette of 256 colors, no dither): its size is about frames x width^2, so the width is the way to make
# it smaller. This example (34 frames): 800 pixels 6.2 MB (the README GIF), 1000 pixels 9.3 MB, 1200 pixels 11.4 MB
gif_width = 800
gif_duration = None    # length of the GIF [s], None = all the frames

###############################################################################################
# Main execution
###############################################################################################

# The parameters above (names that can be set with --set)
_PARAMETERS = [k for k, v in list(globals().items()) if not k.startswith('_') and not callable(v) and not inspect.ismodule(v)]

def apply_overrides():
    """Set the parameters above from the command line: --set NAME=VALUE (a value is read as yaml: numbers, true, null, lists)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--set', action='append', default=[], metavar='NAME=VALUE',
                    help='set a parameter, e.g. --set n_workers=32 --set half_width_km=80 --set gif_width=1000')
    g = globals()
    for item in ap.parse_args().set:
        name, sep, value = item.partition('=')
        if not sep or name not in _PARAMETERS:
            sys.exit(f"ERROR: --set {item}: expected NAME=VALUE with NAME one of the parameters at the top of the file")
        g[name] = yaml.safe_load(value)

def main():

    apply_overrides()
    figdir = f"{fig_root}quicklooks_cell/"
    animation_dir = f"{fig_root}animations_cell/"
    animation_filename = f"{animation_dir}{fig_basename}{start_date[:13]}_{end_date[:13]}.mp4"

    print("Make cell tracking animation")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Radar location: ({radar_lat}, {radar_lon}), map {projection or 'PlateCarree'}, extent {[lon_min, lon_max, lat_min, lat_max]}")
    print(f"Execution mode: Plotting={'yes' if run_plotting else 'no'}, FFmpeg={'yes' if run_ffmpeg else 'no'}")

    #########################################################
    # Plot the PNG files (the options of plot_cell_tracks.py)
    #########################################################
    if run_plotting:
        os.makedirs(figdir, exist_ok=True)
        options = dict(
            start=start_date,
            end=end_date,
            config=config_file,
            product=product,
            time_format=time_format,
            radar_lat=radar_lat,
            radar_lon=radar_lon,
            extent=[lon_min, lon_max, lat_min, lat_max],
            subset=1,
            projection=projection,
            # a LambertConformal map centered on the radar (one standard parallel, at the radar)
            central_longitude=radar_lon if projection else None,
            central_latitude=radar_lat if projection else None,
            standard_parallels=[radar_lat] if projection else None,
            parallel=parallel_mode,
            workers=n_workers,
            output=figdir,
            figsize=figsize,
            lon_ticks=lon_ticks,
            lat_ticks=lat_ticks,
            figbasename=fig_basename,
            title_prefix=title_prefix,
            draw_land=draw_land,
            land_linewidth=land_linewidth,
            draw_border=draw_border,
            draw_state=draw_state,
            show_rangecircle=show_rangecircle,
            show_azimuth=show_azimuth,
            show_tracks=show_tracks,
            fontsize=fontsize,
            map_resolution=map_resolution,
        )
        argv = options_to_argv(options)
        print(f"Options: {' '.join(argv)}")
        try:
            plot_cell_tracks(argv)
        except SystemExit as err:
            if err.code not in (None, 0):
                print(f"Error: plotting failed: {err.code}")
                sys.exit(1)
        print("Plotting completed")
    else:
        print("Skipping plotting: using the existing PNG files")

    #########################################################
    # mp4 and GIF with ffmpeg
    #########################################################
    if run_ffmpeg:
        os.makedirs(animation_dir, exist_ok=True)
        png_files = frames_in_range(figdir, fig_basename, start_date, end_date)
        print(f"Found {len(png_files)} PNG files in {start_date} - {end_date}")
        if len(png_files) == 0:
            sys.exit(f"No PNG files {figdir}{fig_basename}YYYYMMDD_HHMMSS.png in the period")
        returncode = make_animation(png_files, animation_filename, playback_fps=playback_fps, frame_stride=frame_stride,
                                    video_fps=video_fps, video_quality=video_quality, output_width=output_width,
                                    video_preset=video_preset, gif_width=gif_width, gif_duration=gif_duration)
        if returncode != 0:
            sys.exit(1)
        print(f"Animation: {animation_filename}")
    else:
        print("Skipping the animation: the PNG files are ready")


# The Dask workers import this file: the work is under __main__
if __name__ == "__main__":
    main()
