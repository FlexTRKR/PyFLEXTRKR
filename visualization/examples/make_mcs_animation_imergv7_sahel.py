#!/usr/bin/env python
r"""
MCS tracking animation of the Sahel demo (config/demo_mcs_imergv7_sahel.sh): GPM IMERG V07B Tb (top panel) and precipitation with the tracked
MCSs shaded (bottom panel) of 22-23 August 2025 over the Sahel (0-30N, 20W-60E), as an mp4 (and a GIF with --set gif_width=...). A white halo around the track paths, symbols
and numbers keeps them readable on both fields. The sample data are cropped to the map, so no --extent and no --subset are needed; the figure
size follows the shape of the map (no --figsize needed).

Run the demo first (it makes the tracking outputs and config/config_mcs_demo.yml; every MCS demo writes that file name, so run the demo of
this example last, or give config_file=), then, from this folder,
    python make_mcs_animation_imergv7_sahel.py
or with the config file of your own run, another period or more workers, without editing the file:
    python make_mcs_animation_imergv7_sahel.py --set config_file=/path/to/config.yml --set start_date=2025-08-23T00 --set n_workers=32

The PNG files and the animation go to {root_path}/quicklooks_trackpaths/ of the config file, the folder of the demo script; set fig_root to
put them somewhere else (the PNG files in {fig_root}/quicklooks_2panel_tbpf/{year}/, the animation in {fig_root}/animations_2panel_tbpf/).
The parameters that are not set here have the values of DEFAULTS in pyflextrkr/visualization/mcs_animation.py, where the work is done; every
parameter is listed with comments in make_mcs_animation.py next to this file.

The same frames with plot_mcs_tracks.py alone (these are the options of the demo script, and of the `Options:` line that this script prints):
    python ../plot_mcs_tracks.py -s 2025-08-22T00 -e 2025-08-24T00 -c ../../config/config_mcs_demo.yml \
        --product tbpf --panels 2 --layout top_bottom --title_prefix "GPM Tb+IMERGv07" --draw_border 1 \
        --track_halo_color white --track_halo_alpha 0.7 --track_text_alpha 0.7 --show_pfdiam 1 --map_resolution 50m \
        --figsize_x 10 --dpi 100 --fontsize 12 --figbasename mcs_tbpf_ --output /path/of/the/demo/quicklooks_trackpaths/

The mp4 and the GIF of PNG files that are already plotted, with ffmpeg alone (explained in make_mcs_animation.py):
    ffmpeg -framerate 2 -pattern_type glob -i '/path/of/the/demo/quicklooks_trackpaths/mcs_tbpf_*.png' \
        -vf "scale='min(1920,trunc(iw/2)*2)':-2" -c:v libx264 -r 10 -crf 24 -pix_fmt yuv420p -y movie.mp4
    ffmpeg -t 8 -i movie.mp4 -vf "fps=2,scale='min(480,iw)':-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];[b][p]paletteuse=dither=none" -loop 0 -y movie.gif
"""
import os
from pyflextrkr.visualization.mcs_animation import main

# Two panels (top and bottom): Tb, and precipitation with the tracked MCSs shaded
plot_type = "2panel"
data_type = "tbpf"
layout = 'top_bottom'
# The 2 days of the demo
start_date = "2025-08-22T00"
end_date = "2025-08-24T00"
# The config file that config/demo_mcs_imergv7_sahel.sh writes
# Another config file without editing this script: --set config_file=/path/to/config.yml
config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config', 'config_mcs_demo.yml')
time_format = 'yyyymodd_hhmmss'
parallel_mode = 1
n_workers = 16
# Figure: 10 inches wide, 100 dpi (1000 pixels); the height follows the shape of the map
figsize_x = 10
dpi = 100
fontsize = 12
title_prefix = 'GPM Tb+IMERGv07'   # the title is "GPM Tb+IMERGv07 | 2025-08-22 12:30:00 UTC"
# Map: country borders and Natural Earth coastlines at 50m (the style of the map features is black, 0.8 points, unless set: land_color, border_linewidth, ...)
draw_border = True
map_resolution = '50m'
# Tracks: a white halo (outline) around the paths, symbols and numbers, 70 % opaque, and faint numbers; the PF diameter circles of panel (b)
track_halo_color = 'white'
track_halo_alpha = 0.7
track_text_alpha = 0.7
show_pfdiam = True
# FFmpeg animation parameters (explained in visualization/examples/make_mcs_animation.py, tested in docs/visualization/animations.md)
playback_fps = 2       # PNG files shown per second = the playback speed (larger = faster): hourly files, so 2 hours of data per second,
                       # the 48 files of the 2 days take 24 s
frame_stride = 1       # use every Nth PNG file (2 = every second file)
video_fps = None       # frame rate of the video file, None = playback_fps repeated to at least 10 fps (2 -> 10); the speed is the same
video_quality = 24     # CRF of libx264 (lower = better quality and larger file; 18-28 is useful), +6 makes the file 33-55 % smaller.
                       # (the size of the mp4 of this example is printed when it is made)
output_width = 1920    # video width in pixels, never enlarged: these PNG files are 1000 pixels wide, so the mp4 is too
video_preset = None    # x264 preset, None = the x264 default (medium); a slower preset makes the file only about 1 % smaller
# GIF: off by default, the animation is the mp4. For a README, --set gif_width=800 also makes a GIF of the mp4 (800 pixels wide, one palette of 256 colors, no dither);
# a GIF is big (its size is about frames x width^2), so keep it narrow and short: --set gif_duration=8 keeps the first 8 seconds (16 PNG files, 16 hours); all 48 files at 800 pixels are about 7 MB. --set gif_width=null turns it off again.
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
