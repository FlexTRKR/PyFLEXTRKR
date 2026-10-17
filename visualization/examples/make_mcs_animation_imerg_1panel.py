#!/usr/bin/env python
"""
MCS tracking animation of the IMERG demo (config/demo_mcs_imerg.sh): Tb + precipitation of 2 days over South America on ONE
panel (the MCS outlined by track), as an mp4 and a GIF.

Run the demo first (it makes the tracking outputs and config/config_mcs_demo.yml), then
>python make_mcs_animation_imerg_1panel.py
or with the config file of your own run, another period or more workers, without editing the file:
>python make_mcs_animation_imerg_1panel.py --set config_file=/path/to/config.yml --set n_workers=32

The parameters that are not set here have the values of DEFAULTS in pyflextrkr/visualization/mcs_animation.py; every parameter
is listed with comments in visualization/make_mcs_animation.py. The figure size follows the shape of the map (no --extent and no
--figsize needed): the map keeps its true shape and the colorbars have its height.

The mp4 of PNG files that are already plotted, with ffmpeg alone (explained in visualization/make_mcs_animation.py):
>ffmpeg -framerate 2 -pattern_type glob -i "$SCRATCH/pyflextrkr_animations/imerg_1panel/quicklooks_1panel_tbpf/2019/mcs_tbpf_*.png" \\
>       -vf "scale='min(1920,trunc(iw/2)*2)':-2" -c:v libx264 -r 10 -crf 24 -pix_fmt yuv420p -y movie.mp4
and the GIF from it (480 pixels wide, the first 8 s):
>ffmpeg -t 8 -i movie.mp4 -vf "fps=2,scale='min(480,iw)':-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];[b][p]paletteuse=dither=none" -loop 0 -y movie.gif
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
# FFmpeg animation parameters (explained in visualization/make_mcs_animation.py, tested in docs/visualization/animations.md)
playback_fps = 2       # PNG files shown per second = the playback speed (larger = faster): hourly files, so 2 hours of data per second,
                       # the 48 files of the 2 days take 24 s
frame_stride = 1       # use every Nth PNG file (2 = every second file)
video_fps = None       # frame rate of the video file, None = playback_fps repeated to at least 10 fps (2 -> 10); the speed is the same
video_quality = 24     # CRF of libx264 (lower = better quality and larger file; 18-28 is useful), +6 makes the file 33-55 % smaller.
                       # This example (48 PNG files, 800 x 1081): CRF 20: 4.7 MB, 24: 3.2 MB (-33 %), 28: 2.0 MB
output_width = 1920    # video width in pixels, never enlarged: these PNG files are 800 pixels wide, so the mp4 is too
video_preset = None    # x264 preset, None = the x264 default (medium); a slower preset makes the file only about 1 % smaller
# GIF made from the mp4, for the README: a GIF is big (about 0.14 MB per frame at this width), so it is narrow and short:
# 480 pixels wide, the first 8 seconds (16 PNG files, 16 hours) are 2.3 MB; all 48 files would be 6.6 MB
gif_width = 480
gif_duration = 8
# PNG files go to {fig_root}quicklooks_1panel_tbpf/{year}/, the animation to {fig_root}animations_1panel_tbpf/
fig_root = os.path.join(os.environ.get('SCRATCH', os.getcwd()), 'pyflextrkr_animations', 'imerg_1panel')

# The Dask workers import this file: the work is under __main__
if __name__ == "__main__":
    main(globals(), description=__doc__)
