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
input_framerate = 2    # (frames per second) - how fast to transition between frames
output_framerate = 10  # (frames per second) - video playback speed (lower values = smaller file size, e.g., 24 fps is cinema)
video_quality = 20     # CRF value (lower = better quality, range 0-51, 18-28 is good)
output_width = 1920    # Output video width in pixels (height auto-calculated to maintain aspect ratio, set to None to keep original size)
video_preset = None    # x264 preset (ultrafast ... medium ... veryslow); None = no -preset option (x264 default, medium). A slower preset
                       # makes the file only about 1 % smaller, see docs/visualization/animations.md
# GIF made from the mp4 (for a README): width in pixels, None: no GIF; gif_fps None: input_framerate; the part of the mp4 to
# use: gif_start and gif_duration in seconds (None: to the end); a GIF has 256 colors at most (gif_colors)
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
