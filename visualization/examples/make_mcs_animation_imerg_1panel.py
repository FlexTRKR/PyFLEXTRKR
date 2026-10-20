#!/usr/bin/env python
r"""
MCS tracking animation of the IMERG demo (config/demo_mcs_imerg.sh): Tb + precipitation of 2 days over South America on ONE
panel (the MCS outlined by track), as an mp4 (and a GIF with --set gif_width=...). The figure size follows the shape of the map (no --extent and no --figsize
needed): the map keeps its true shape and the colorbars have its height.

Run the demo first (it makes the tracking outputs and config/config_mcs_demo.yml; every MCS demo writes that file name, so run the demo of
this example last, or give config_file=), then, from this folder,
    python make_mcs_animation_imerg_1panel.py
or with the config file of your own run, another period or more workers, without editing the file:
    python make_mcs_animation_imerg_1panel.py --set config_file=/path/to/config.yml --set start_date=2019-01-26T00 --set n_workers=32

The PNG files and the animation go to {root_path}/quicklooks_trackpaths/ of the config file, the folder of the demo script; set fig_root to
put them somewhere else (the PNG files in {fig_root}/quicklooks_1panel_tbpf/{year}/, the animation in {fig_root}/animations_1panel_tbpf/).
The parameters that are not set here have the values of DEFAULTS in pyflextrkr/visualization/mcs_animation.py, where the work is done; every
parameter is listed with comments in make_mcs_animation.py next to this file.

The same frames with plot_mcs_tracks.py alone (these are the options of the demo script, and of the `Options:` line that this script prints):
    python ../plot_mcs_tracks.py -s 2019-01-25T00 -e 2019-01-27T00 -c ../../config/config_mcs_demo.yml \
        --product tbpf --panels 1 --figsize_x 8 --dpi 100 --cbar_thickness 0.02 --cbar_spacing 0.1 \
        --figbasename mcs_tbpf_ --output /path/of/the/demo/quicklooks_trackpaths/

The mp4 and the GIF of PNG files that are already plotted, with ffmpeg alone (explained in make_mcs_animation.py):
    ffmpeg -framerate 2 -pattern_type glob -i '/path/of/the/demo/quicklooks_trackpaths/mcs_tbpf_*.png' \
        -vf "scale='min(1920,trunc(iw/2)*2)':-2" -c:v libx264 -r 10 -crf 24 -pix_fmt yuv420p -y movie.mp4
    ffmpeg -t 8 -i movie.mp4 -vf "fps=2,scale='min(480,iw)':-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];[b][p]paletteuse=dither=none" -loop 0 -y movie.gif
"""
import os
from pyflextrkr.visualization.mcs_animation import main

# One panel with Tb and precipitation
plot_type = "1panel"
data_type = "tbpf"
# The 2 days of the demo
start_date = "2019-01-25T00"
end_date = "2019-01-27T00"
# The config file that config/demo_mcs_imerg.sh writes
# Another config file without editing this script: --set config_file=/path/to/config.yml
config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config', 'config_mcs_demo.yml')
time_format = 'yyyymodd_hhmmss'
parallel_mode = 1
n_workers = 16
# Figure: 8 inches wide, 100 dpi (800 pixels); the height follows the shape of the map
figsize_x = 8
dpi = 100
# Colorbars: how thin they are, and the gap between the two (fractions of the map width)
cbar_thickness = 0.02
cbar_spacing = 0.1
# FFmpeg animation parameters (explained in visualization/examples/make_mcs_animation.py, tested in docs/visualization/animations.md)
playback_fps = 2       # PNG files shown per second = the playback speed (larger = faster): hourly files, so 2 hours of data per second,
                       # the 48 files of the 2 days take 24 s
frame_stride = 1       # use every Nth PNG file (2 = every second file)
video_fps = None       # frame rate of the video file, None = playback_fps repeated to at least 10 fps (2 -> 10); the speed is the same
video_quality = 24     # CRF of libx264 (lower = better quality and larger file; 18-28 is useful), +6 makes the file 33-55 % smaller.
                       # This example (48 PNG files, 800 x 1081): CRF 20: 4.7 MB, 24: 3.2 MB (-33 %), 28: 2.0 MB
output_width = 1920    # video width in pixels, never enlarged: these PNG files are 800 pixels wide, so the mp4 is too
video_preset = None    # x264 preset, None = the x264 default (medium); a slower preset makes the file only about 1 % smaller
# GIF: off by default, the animation is the mp4. For a README, --set gif_width=480 also makes a GIF of the mp4 (480 pixels wide, one palette of 256 colors, no dither);
# a GIF is big (its size is about frames x width^2), so keep it narrow and short: --set gif_duration=8 keeps the first 8 seconds (16 PNG files, 16 hours: 2.3 MB at 480 pixels; all 48 files 6.6 MB). --set gif_width=null turns it off again.
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
