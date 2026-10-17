#!/usr/bin/env python
"""
Make MCS tracking animation
- Plot the PNG frames: calls main() of pyflextrkr/visualization/mcs_cli.py (the options of plot_mcs_tracks.py)
- Options to make 1-panel or 2-panel plots
- Options to plot Tb + Precipitation, Tb + Reflectivity or Tb only
- Options for the map projection (e.g. a global Robinson map) and the style of the tracks (colors, symbols, halo)
- Create a video animation using FFmpeg (pyflextrkr/visualization/animation.py), and a GIF of it if you set gif_width

The defaults below make the global MCS animation (V3 tracking, Robinson map, 2 days). Edit the parameters, or set them on the
command line without editing the file:
>python make_mcs_animation.py --set start_date=2025-01-05T00 --set end_date=2025-01-07T00 --set n_workers=32

Examples for other data (WRF, IMERG over South America, Himawari) are in examples/make_mcs_animation_*.py. Every parameter
that is left out of a script has the value in DEFAULTS of pyflextrkr/visualization/mcs_animation.py, where the work is done.

The mp4 of PNG files that are already plotted, with ffmpeg alone (the same file as run_plotting = False gives; a directory or a
list of PNG files works with `python pngs_to_animation.py FRAMES movie.mp4`, which also makes the GIF):
>ffmpeg -framerate 2 -pattern_type glob -i '/pscratch/sd/f/feng045/waccem/mcs_global/quicklooks_2panel_tbpf/2025/mcs_tbpf_*.png' \\
>       -vf "scale='min(1920,trunc(iw/2)*2)':-2" -c:v libx264 -r 10 -crf 24 -pix_fmt yuv420p -y movie.mp4
  -framerate 2 (before -i): PNG files per second = playback_fps. -vf scale: at most 1920 pixels wide (output_width), never enlarged,
  and even sizes, which yuv420p needs (a PNG file with an odd number of pixels fails without it). -r 10 (after the codec): frames per
  second of the file (video_fps), each PNG file is repeated 5 times. -crf 24: video_quality. -pix_fmt yuv420p: plays in every player.
The GIF from that mp4, 480 pixels wide, the first 8 s, 2 frames per second, one palette of 256 colors, no dither (gif_width = 480, gif_duration = 8):
>ffmpeg -t 8 -i movie.mp4 -vf "fps=2,scale='min(480,iw)':-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];[b][p]paletteuse=dither=none" -loop 0 -y movie.gif

Author: Zhe Feng | zhe.feng@pnnl.gov
"""

import os
from pyflextrkr.visualization.mcs_animation import main

###############################################################################################
# Script parameters
###############################################################################################

# Choose panel layout:
# "1panel": Tb and data overlaid in the same panel
# "2panel": Tb and data in separate panels
plot_type = "2panel"

# Choose data type:
# "tbpf": Brightness Temperature + Precipitation Features
# "tbze": Brightness Temperature + Reflectivity
# "tb": Brightness Temperature only (1 panel)
data_type = "tbpf"

# Period. The track numbers count the tracks from the start of the run (1 January): the first days have numbers of
# 1-3 digits, which keeps the labels small on a global map
start_date = "2025-01-01T00"
end_date = "2025-01-03T00"
# Define domain map extent (None: the whole domain of the data)
# Global domain
lon_min = None
lon_max = None
lat_min = None
lat_max = None
# Sahel domain
# lon_min = -20.0
# lon_max = 60.0
# lat_min = 0.0
# lat_max = 30.0
# CONUS domain
# lon_min = -110.0
# lon_max = -70.0
# lat_min = 24.0
# lat_max = 50.0
# Subset data before plotting (0:no, 1:yes): 1 reads only the pixels of the domain above (use it for a region, not a global map)
subset = 0
# 2-panel layout: 'top_bottom' (panel (a) above panel (b)) or 'left_right'
layout = 'top_bottom'
# Tracking config file: the global MCS tracking V3 (Feng21) of 2025, in /global/cfs/cdirs/wcm_shr/gsharing/MCS_Global/V3/Feng21/
# (read-only: the config only points to the outputs, it does not make folders). The track statistics file and the pixel-level
# files directory come from the config file (trackstats_file = None, pixel_path = None); give them to use others
config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'config_imerg_global_mcs_v3_2025.yml')
# config_file = "/global/homes/f/feng045/program/pyflex_config/config/config_imerg_mcs_tbpf_2025.yml"
trackstats_file = None
pixel_path = None
# Days before start_date from which the tracks are drawn (None: 4 days), for long-lived tracks
track_lookback_days = 7

parallel_mode = 1
n_workers = 64
figsize_x = 14  # Width in inches (height is auto-calculated to maintain aspect ratio)

# Tracking pixel-level file time format
time_format = 'yyyymodd_hhmmss'
# Title prefix for plots (added before date/time, can be an empty string)
title_prefix = 'GPM Tb+IMERGv07'
# Variable name for reflectivity to use in plotting (only for data_type='tbze')
# Options: 'reflectivity_comp' (composite), 'reflectivity_lowlevel' (low-level)
dbz_varname = 'reflectivity_lowlevel'

# Font size of the labels, titles and legends (the track numbers have their own size: track_text_size)
fontsize = 14
# Write the track numbers on panel (a) Tb and panel (b) the second field: [True, False] = on (a) only. One value for 1 panel
show_track_numbers = [True, False]

# Map projection (None: PlateCarree, the central longitude is 180 for a domain that crosses the dateline)
# Global map: Robinson centered on the Pacific, only the latitudes of the data (crop of the latitude band)
projection = 'Robinson'      # PlateCarree, Robinson, Mollweide, EqualEarth, LambertConformal, ...
central_longitude = 225
lat_range = [-60, 60]
# The MCS outlines continue across the edges of a periodic domain ('x': the edge of the data in longitude is a global edge)
periodic_perimeter = 'x'

# Map features to draw
draw_border = False  # Draw country borders
draw_state = False   # Draw state/province borders
draw_river = False   # Draw rivers (only for data_type='tbpf')
land_linewidth = 0.8   # Coastline width (None: 2)
map_resolution = '110m'   # Natural Earth resolution of the coastlines ('110m', '50m', '10m')

# Style of the tracks (None: the default look). On a global map the clouds matter most: no halo, small numbers that let the
# clouds show through, a small symbol at the initiation location, no PF diameter circles (too small to see)
track_color = None          # path color (default purple)
track_lw = None             # path line width
track_ls = None             # path line style ('solid', 'dashed', 'dotted')
track_alpha = None          # path transparency
track_marker = None         # symbol at the initiation location ('o', '^', '*', ...)
track_marker_size = 4       # symbol size (default 10)
track_text_size = 7         # font size of the track numbers
track_text_color = None     # color of the track numbers
track_text_alpha = 0.6      # transparency of the track numbers (1: opaque)
track_halo_color = None     # halo (outline) around paths, symbols and numbers (None: no halo)
track_halo_width = None     # halo width [points]
track_halo_alpha = None     # halo transparency
show_pfdiam = False         # PF diameter circles on panel (b), and their entry in the legend

# Colorbars (None: the layout of the old scripts)
cbar_thickness = None       # thickness as a fraction of the map width (colorbars at the right) or height (below it)
cbar_spacing = None         # 1 panel: the gap between the 2 colorbars as a fraction of the map width

# Execution control options
run_plotting = True   # Set to False to skip plotting and use existing PNG files
run_ffmpeg = True     # Set to False to skip animation creation (plotting only)

# FFmpeg animation parameters
playback_fps = 2       # PNG files shown per second = the playback speed of the animation (larger = faster). Hourly files: 2 means
                       # 2 hours of data per second, so the 48 files of 2 days take 24 s
frame_stride = 1       # use every Nth PNG file (2 = every second file): the other way to shorten a long period
video_fps = None       # frame rate of the video file. None = automatic: playback_fps repeated to at least 10 fps with a whole
                       # number of repeats per PNG file (2 -> 10). It does not change the speed; players handle very low frame
                       # rates badly, so every PNG file is repeated. Keep it a whole multiple of playback_fps (in between, the
                       # PNG files are repeated unevenly)
video_quality = 24     # CRF value of libx264 (lower = better quality and larger file, range 0-51; 18-28 is the useful range).
                       # +6 CRF makes the file 33-55 % smaller. Measured on the 48 PNG files of the global default (2800 x 2304
                       # pixels, scaled to 1920) against the source PNG files:
                       #   CRF 20: 16.3 MB (SSIM 0.9964)   CRF 22: 13.9 MB (0.9947)
                       #   CRF 24: 11.8 MB (0.9923, -28 %, thin lines and numbers look the same)
                       #   CRF 26: 9.8 MB (0.9891, -40 %)  CRF 28: 7.9 MB (0.9849, -51 %)
                       # Maps with fine detail (the Himawari example) lose more at the same CRF: use 22 for them
output_width = 1920    # Output video width in pixels, never enlarged: smaller PNG files keep their size (height auto-calculated to
                       # maintain aspect ratio; None: the size of the PNG files). At CRF 24 this global map is 11.8 MB at 1920 pixels,
                       # 6.7 MB at 1400 and 24.7 MB at its full 2800
video_preset = None    # x264 preset: ultrafast ... fast, medium (x264 default), slow, slower, veryslow; None = no -preset option (medium).
                       # 'slow' makes the file only about 1 % smaller; 'slower' 1-12 % and 'veryslow' 4-19 % smaller, for 1.1-2.4 and
                       # 1.2-3.6 times the encoding time (docs/visualization/animations.md). Use them only if the size matters more
# GIF made from the mp4 (for a README): width in pixels, None: no GIF. A GIF is big: its size is about proportional to the number of
# frames and of pixels, so make it short and narrow. gif_fps None: playback_fps (the same speed); the part of the mp4 to use:
# gif_start and gif_duration in seconds (None: to the end); gif_colors up to 256 (128 saves 13-23 %, but the gradient colorbars
# band); gif_dither 'none' (the maps are flat colors, dither only adds dots)
gif_width = None
gif_fps = None
gif_start = 0.0
gif_duration = None

# Output directories (set the paths for your data): the PNG files go to {fig_root}quicklooks_{plot_type}_{data_type}/{year}/ and the
# animation to {fig_root}animations_{plot_type}_{data_type}/
fig_root = "/pscratch/sd/f/feng045/waccem/mcs_global/"

###############################################################################################
# Main execution
###############################################################################################

# The Dask workers import this file: the work is under __main__
if __name__ == "__main__":
    main(globals(), description=__doc__)
