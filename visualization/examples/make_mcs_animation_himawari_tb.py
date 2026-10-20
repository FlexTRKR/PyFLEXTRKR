#!/usr/bin/env python
r"""
MCS tracking animation of the Himawari demo (config/demo_mcs_himawari.sh): brightness temperature only (no second field) of
1 day on one panel, as an mp4 (and a GIF with --set gif_width=...).

Run the demo first (it makes the tracking outputs and config/config_mcs_demo.yml; every MCS demo writes that file name, so run the demo of
this example last, or give config_file=), then, from this folder,
    python make_mcs_animation_himawari_tb.py
or with the config file of your own run, another period or more workers, without editing the file:
    python make_mcs_animation_himawari_tb.py --set config_file=/path/to/config.yml --set start_date=2021-10-24T12 --set n_workers=32

The PNG files and the animation go to {root_path}/quicklooks_trackpaths/ of the config file, the folder of the demo script; set fig_root to
put them somewhere else (the PNG files in {fig_root}/quicklooks_1panel_tb/{year}/, the animation in {fig_root}/animations_1panel_tb/).
The parameters that are not set here have the values of DEFAULTS in pyflextrkr/visualization/mcs_animation.py, where the work is done; every
parameter is listed with comments in make_mcs_animation.py next to this file.

The same frames with plot_mcs_tracks.py alone (these are the options of the demo script, and of the `Options:` line that this script prints):
    python ../plot_mcs_tracks.py -s 2021-10-24T00 -e 2021-10-25T00 -c ../../config/config_mcs_demo.yml \
        --product tb --panels 1 --figsize 8 8 --dpi 100 \
        --figbasename mcs_tb_ --output /path/of/the/demo/quicklooks_trackpaths/

The mp4 and the GIF of PNG files that are already plotted, with ffmpeg alone (explained in make_mcs_animation.py):
    ffmpeg -framerate 2 -pattern_type glob -i '/path/of/the/demo/quicklooks_trackpaths/mcs_tb_*.png' \
        -vf "scale='min(1920,trunc(iw/2)*2)':-2" -c:v libx264 -r 10 -crf 24 -pix_fmt yuv420p -y movie.mp4
    ffmpeg -t 8 -i movie.mp4 -vf "fps=2,scale='min(480,iw)':-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];[b][p]paletteuse=dither=none" -loop 0 -y movie.gif
"""
import os
from pyflextrkr.visualization.mcs_animation import main

# Tb only: one panel, one colorbar
plot_type = "1panel"
data_type = "tb"
# The day of the demo
start_date = "2021-10-24T00"
end_date = "2021-10-25T00"
# The config file that config/demo_mcs_himawari.sh writes
# Another config file without editing this script: --set config_file=/path/to/config.yml
config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config', 'config_mcs_demo.yml')
time_format = 'yyyymodd_hhmmss'
parallel_mode = 1
n_workers = 16
# Figure: 8 x 8 inches, 100 dpi (800 pixels)
figsize = [8, 8]
dpi = 100
# FFmpeg animation parameters (explained in visualization/examples/make_mcs_animation.py, tested in docs/visualization/animations.md)
playback_fps = 2       # PNG files shown per second = the playback speed (larger = faster): hourly files, so 2 hours of data per second,
                       # the 24 files of the day take 12 s
frame_stride = 1       # use every Nth PNG file (2 = every second file)
video_fps = None       # frame rate of the video file, None = playback_fps repeated to at least 10 fps (2 -> 10); the speed is the same
video_quality = 24     # CRF of libx264 (lower = better quality and larger file; 18-28 is useful), +6 makes the file 33-55 % smaller.
                       # This example (24 PNG files, 800 x 800): CRF 20: 3.2 MB (SSIM 0.9924), 22: 2.7 MB (0.9884), 24: 2.2 MB (0.9823,
                       # -30 %), 28: 1.4 MB (0.9610). The Himawari IR map has a lot of fine detail: CRF 22 is the safer choice here
output_width = 1920    # video width in pixels, never enlarged: these PNG files are 800 pixels wide, so the mp4 is too
video_preset = None    # x264 preset, None = the x264 default (medium); a slower preset makes the file only about 1 % smaller
# GIF: off by default, the animation is the mp4. For a README, --set gif_width=480 also makes a GIF of the mp4 (480 pixels wide, one palette of 256 colors, no dither);
# a GIF is big (its size is about frames x width^2), so keep it narrow and short: --set gif_duration=8 keeps the first 8 seconds (16 PNG files, 16 hours: 1.6 MB at 480 pixels; all 24 files 2.4 MB). --set gif_width=null turns it off again.
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
