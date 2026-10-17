#!/usr/bin/env python
"""
MCS tracking animation of the WRF demo (config/demo_mcs_wrf_tbradar.sh): Tb and radar reflectivity of 2 days on 2 panels side
by side, as an mp4 and a GIF. The WRF data are on a Lambert conformal grid and are shown with that projection; the map is
the domain of the data (no --extent and no --subset needed).

Run the demo first (it makes the tracking outputs and config/config_mcs_demo.yml), then
>python make_mcs_animation_wrf_tbze.py
or with the config file of your own run, another period or more workers, without editing the file:
>python make_mcs_animation_wrf_tbze.py --set config_file=/path/to/config.yml --set start_date=2015-05-06T00 --set n_workers=32

The parameters that are not set here have the values of DEFAULTS in pyflextrkr/visualization/mcs_animation.py; every parameter
is listed with comments in visualization/make_mcs_animation.py.

The mp4 of PNG files that are already plotted, with ffmpeg alone (explained in visualization/make_mcs_animation.py):
>ffmpeg -framerate 2 -pattern_type glob -i "$SCRATCH/pyflextrkr_animations/wrf_tbze/quicklooks_2panel_tbze/2015/mcs_tbze_*.png" \\
>       -vf "scale='min(1920,trunc(iw/2)*2)':-2" -c:v libx264 -r 10 -crf 24 -pix_fmt yuv420p -y movie.mp4
and the GIF from it (720 pixels wide, the first 8 s):
>ffmpeg -t 8 -i movie.mp4 -vf "fps=2,scale='min(720,iw)':-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];[b][p]paletteuse=dither=none" -loop 0 -y movie.gif
"""
import os
from pyflextrkr.visualization.mcs_animation import main

# Tb and reflectivity on 2 panels: panel (a) Tb, panel (b) reflectivity with the MCS shaded
plot_type = "2panel"
data_type = "tbze"
layout = 'left_right'
# 2 of the 4 days of the demo, the time with the most MCSs
start_date = "2015-05-07T00"
end_date = "2015-05-08T23"
# The config file that config/demo_mcs_wrf_tbradar.sh writes
config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config', 'config_mcs_demo.yml')
time_format = 'yyyymodd_hhmmss'
parallel_mode = 1
n_workers = 16
# 'reflectivity_comp' (composite) or 'reflectivity_lowlevel'
dbz_varname = 'reflectivity_comp'
# The Lambert conformal grid of the WRF domain (STAND_LON, TRUELAT1 and TRUELAT2 of the WRF files)
projection = 'LambertConformal'
central_longitude = -97
standard_parallels = [31, 35]
draw_state = True
# Figure: 12 inches wide, 100 dpi (1200 pixels); the height follows the shape of the map
figsize_x = 12
dpi = 100
# FFmpeg animation parameters (explained in visualization/make_mcs_animation.py, tested in docs/visualization/animations.md)
playback_fps = 2       # PNG files shown per second = the playback speed (larger = faster): hourly files, so 2 hours of data per second,
                       # the 48 files of the 2 days take 24 s
frame_stride = 1       # use every Nth PNG file (2 = every second file)
video_fps = None       # frame rate of the video file, None = playback_fps repeated to at least 10 fps (2 -> 10); the speed is the same
video_quality = 24     # CRF of libx264 (lower = better quality and larger file; 18-28 is useful), +6 makes the file 33-55 % smaller.
                       # This example (48 PNG files, 1200 x 689): CRF 20: 6.9 MB, 24: 5.4 MB (-23 %), 28: 4.0 MB
output_width = 1920    # video width in pixels, never enlarged: these PNG files are 1200 pixels wide, so the mp4 is too
video_preset = None    # x264 preset, None = the x264 default (medium); a slower preset makes the file only about 1 % smaller
# GIF made from the mp4, for the README: a GIF is big (about 0.12 MB per frame at this width), so it is narrow and short:
# 720 pixels wide, the first 8 seconds (16 PNG files, 16 hours) are 1.9 MB; all 48 files would be 5.6 MB
gif_width = 720
gif_duration = 8
# PNG files go to {fig_root}quicklooks_2panel_tbze/{year}/, the animation to {fig_root}animations_2panel_tbze/
fig_root = os.path.join(os.environ.get('SCRATCH', os.getcwd()), 'pyflextrkr_animations', 'wrf_tbze')

# The Dask workers import this file: the work is under __main__
if __name__ == "__main__":
    main(globals(), description=__doc__)
