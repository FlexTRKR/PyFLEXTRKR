#!/usr/bin/env python
r"""
Convective cell tracking animation of the NEXRAD demo (config/demo_cell_nexrad.sh): composite reflectivity of the KHGX radar (Houston,
7 August 2014, 12-15 UTC) with the cell outlines, track paths, cell numbers, the radar range rings and azimuth lines, the coastline and the
state borders, as an mp4 (and a GIF with --set gif_width=...).

Run the demo first (it makes the tracking outputs and config/config_cell_demo.yml), then, from this folder,
    python make_cell_animation_nexrad.py
or with the config file of your own run, another period or more workers, without editing the file:
    python make_cell_animation_nexrad.py --set config_file=/path/to/config.yml --set n_workers=32 --set map_half_width_km=80

The PNG files and the animation go to {root_path}/quicklooks_trackpaths/ of the config file, the folder of the demo script; set fig_root
to put them somewhere else. The parameters that are not set here have the values of DEFAULTS in pyflextrkr/visualization/cell_animation.py,
where the work is done: terrain, one outline color per track, cities and other radars, the style of the tracks and of every map feature
(--set land_color=k, --set state_linewidth=0.3, ...) are parameters too.

The map is a square of 2 * map_half_width_km around the radar on a LambertConformal projection centered on it. A PlateCarree map (the
default) keeps 1 degree of longitude as wide as 1 degree of latitude, but at 29.5N a degree of longitude is 0.87 times as long, so the
range circles come out 15 % too wide; here they are round.

The same frames with plot_cell_tracks.py alone (these are the options of the demo script, and of the `Options:` line that this script prints):
    python ../plot_cell_tracks.py -s 2014-08-07T12:00 -e 2014-08-07T15:00 -c ../../config/config_cell_demo.yml \
        --radar_lat 29.4719 --radar_lon -95.0787 --extent -96.163 -93.994 28.528 30.416 --subset 1 \
        --projection LambertConformal --central_longitude -95.0787 --central_latitude 29.4719 --standard_parallels 29.4719 \
        --title_prefix "NEXRAD (KHGX)" --figsize 8 7.2 --lon_ticks -96 -95.5 -95 -94.5 -94 --lat_ticks 28.5 29 29.5 30 \
        --draw_land 1 --land_linewidth 1 --draw_border 1 --draw_state 1 --map_resolution 10m --figbasename cell_tracks_ \
        --output /path/of/the/demo/quicklooks_trackpaths/

The mp4 and the GIF of PNG files that are already plotted, with ffmpeg alone (-framerate 4: PNG files per second = playback_fps;
-r 12: frames per second of the file = video_fps, each PNG file is repeated 3 times; -crf 18: video_quality; the scale filter keeps the
size of the PNG files and makes it even, which yuv420p needs):
    ffmpeg -framerate 4 -pattern_type glob -i '/path/of/the/demo/quicklooks_trackpaths/cell_tracks_*.png' \
        -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -r 12 -crf 18 -pix_fmt yuv420p -y movie.mp4
    ffmpeg -i movie.mp4 -vf "fps=4,scale='min(800,iw)':-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];[b][p]paletteuse=dither=none" -loop 0 -y movie.gif
or with pngs_to_animation.py (--print_command shows the ffmpeg commands):
    python ../pngs_to_animation.py /path/of/the/demo/quicklooks_trackpaths/ movie.mp4 --playback_fps 4 --video_quality 18 --gif_width 800
"""
import os
from pyflextrkr.visualization.cell_animation import main

# Period of the demo (5-minute NEXRAD volumes)
start_date = "2014-08-07T12:00"
end_date = "2014-08-07T15:00"
# The config file that config/demo_cell_nexrad.sh writes
# Another config file without editing this script: --set config_file=/path/to/config.yml
config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config', 'config_cell_demo.yml')
time_format = 'yyyymodd_hhmmss'
parallel_mode = 1
n_workers = 16

# Radar (KHGX) and the map: a square of 2 * 105 km around it (the 100 km range circle is inside), as a LambertConformal map centered on the
# radar so that the range circles are round (the extent is made from the radar location and the half width)
radar_lat = 29.4719
radar_lon = -95.0787
map_half_width_km = 105
subset = 1      # read only the pixels of the map
projection = 'LambertConformal'
# Tick labels every 0.5 degree (the default ticks every 0.25 degree have labels like 30.25N that are cut off at the left edge)
lon_ticks = [-96, -95.5, -95, -94.5, -94]
lat_ticks = [28.5, 29, 29.5, 30]

# Figure [width, height] in inches (300 dpi by default: 2400 x 2160 pixels). The map panel of the layout is 0.757 of the width and 0.84 of the
# height of the figure, so 8 x 7.2 inches make a square panel for the square map (without figsize the height follows from the shape of the map
# but is not shorter than the layout needs for the titles: 9.2 inches, with white space above and below the map)
figsize = [8, 7.2]
# The title is "NEXRAD (KHGX) | 2014-08-07 13:05:53 UTC"
title_prefix = 'NEXRAD (KHGX)'

# Map features: coastlines (land_linewidth of 1 point), country and state borders; the style of the four features is gray, 0.8 points, opaque
# unless set (land_color, border_linewidth, state_alpha, ...). 10m Natural Earth: the most detailed coastline
draw_land = True
land_linewidth = 1
draw_border = True
draw_state = True
map_resolution = '10m'

# FFmpeg animation parameters (explained in visualization/examples/make_mcs_animation.py, tested in docs/visualization/animations.md)
playback_fps = 4       # PNG files shown per second = the playback speed (larger = faster): 5-minute volumes, so 20 minutes of data per
                       # second, the 34 volumes take 8.5 s
frame_stride = 1       # use every Nth PNG file (2 = every second file)
video_fps = None       # frame rate of the video file, None = playback_fps repeated to at least 10 fps (4 -> 12); the speed is the same
video_quality = 18     # CRF of libx264 (lower = better quality and larger file; 18-28 is useful, 24 is the usual choice). The GIF is made
                       # from this mp4, so it is kept near lossless: a lower CRF makes the mp4 larger but the GIF only a few % larger
output_width = None    # video width in pixels, None = the width of the PNG files (never enlarged)
video_preset = None    # x264 preset, None = the x264 default (medium)
# GIF: off by default, the animation is the mp4. For a README, --set gif_width=800 also makes a GIF of the mp4 (800 pixels wide, one palette of 256 colors, no dither);
# a GIF is big (its size is about frames x width^2), so keep it narrow (this example, 34 frames: 800 pixels 6.2 MB, 1000 pixels 9.3 MB, 1200 pixels 11.4 MB); --set gif_duration=5 keeps the first 5 seconds. --set gif_width=null turns it off again.
gif_width = None
gif_duration = None

# Where the PNG frames and the animation go. None (default): both in {root_path}/quicklooks_trackpaths/ of the tracking config file, the folder of the demo
# script. A path replaces that default: fig_root puts the frames in {fig_root}/quicklooks_.../ and the animation in {fig_root}/animations_.../ (subfolders named
# after the plot); figdir is the folder of the PNG frames only and animation_dir the folder of the mp4 / GIF only (each one wins over fig_root).
fig_root = None
figdir = None
animation_dir = None

# The Dask workers import this file: the work is under __main__
if __name__ == "__main__":
    main(globals(), description=__doc__)
