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
# Video: 1 PNG every 0.5 s; the mp4 is as wide as the PNG files. The GIF is made from the mp4: 480 pixels wide, the first 12 seconds
# (the first day, 24 PNG files) to keep it small
input_framerate = 2
output_framerate = 10
output_width = 800
gif_width = 480
gif_duration = 12
# PNG files go to {fig_root}quicklooks_1panel_tbpf/{year}/, the animation to {fig_root}animations_1panel_tbpf/
fig_root = os.path.join(os.environ.get('SCRATCH', os.getcwd()), 'pyflextrkr_animations', 'imerg_1panel')

# The Dask workers import this file: the work is under __main__
if __name__ == "__main__":
    main(globals(), description=__doc__)
