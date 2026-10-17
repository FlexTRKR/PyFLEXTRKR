"""
MOVED: this script is now visualization/plot_mcs_tracks.py (--product tb --dpi 300 --fontsize 13 --figsize 10 10 --skip_empty 1 --track_smooth 0 --map_aspect auto).

This stub keeps the old command working: it takes the same arguments, with the defaults of the old script
(pyflextrkr/visualization/legacy.py), and calls the new code (pyflextrkr/visualization/mcs_cli.py, mcs_map.py).
"""
from pyflextrkr.visualization.mcs_cli import main

if __name__ == "__main__":
    main(legacy_name="plot_subset_tb_mcs_tracks_demo")
