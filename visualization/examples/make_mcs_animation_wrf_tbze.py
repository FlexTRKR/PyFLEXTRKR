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
# Video: 1 PNG every 0.5 s; the mp4 is as wide as the PNG files. The GIF is made from the mp4: 800 pixels wide, the first 12 seconds
# (the first day, 24 PNG files) to keep it small
input_framerate = 2
output_framerate = 10
output_width = 1200
gif_width = 800
gif_duration = 12
# PNG files go to {fig_root}quicklooks_2panel_tbze/{year}/, the animation to {fig_root}animations_2panel_tbze/
fig_root = os.path.join(os.environ.get('SCRATCH', os.getcwd()), 'pyflextrkr_animations', 'wrf_tbze')

# The Dask workers import this file: the work is under __main__
if __name__ == "__main__":
    main(globals(), description=__doc__)
