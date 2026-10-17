#!/usr/bin/env python
"""
MCS tracking animation of the Himawari demo (config/demo_mcs_himawari.sh): brightness temperature only (no second field) of
1 day on one panel, as an mp4 and a GIF.

Run the demo first (it makes the tracking outputs and config/config_mcs_demo.yml), then
>python make_mcs_animation_himawari_tb.py
or with the config file of your own run, another period or more workers, without editing the file:
>python make_mcs_animation_himawari_tb.py --set config_file=/path/to/config.yml --set n_workers=32

The parameters that are not set here have the values of DEFAULTS in pyflextrkr/visualization/mcs_animation.py; every parameter
is listed with comments in visualization/make_mcs_animation.py.
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
config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config', 'config_mcs_demo.yml')
time_format = 'yyyymodd_hhmmss'
parallel_mode = 1
n_workers = 16
# Figure: 8 x 8 inches, 100 dpi (800 pixels)
figsize = [8, 8]
dpi = 100
# Video: 1 PNG every 0.5 s; the mp4 is as wide as the PNG files; the GIF is made from the mp4 (560 pixels wide)
input_framerate = 2
output_framerate = 10
output_width = 800
gif_width = 560
# PNG files go to {fig_root}quicklooks_1panel_tb/{year}/, the animation to {fig_root}animations_1panel_tb/
fig_root = os.path.join(os.environ.get('SCRATCH', os.getcwd()), 'pyflextrkr_animations', 'himawari_tb')

# The Dask workers import this file: the work is under __main__
if __name__ == "__main__":
    main(globals(), description=__doc__)
