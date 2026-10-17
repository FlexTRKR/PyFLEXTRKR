"""
MOVED: this script is now visualization/plot_mcs_tracks.py (--product tbze --panels 2 --dpi 300 --draw_state 1 --map_aspect auto).

This stub keeps the old command working: it takes the same arguments, with the defaults of the old script
(pyflextrkr/visualization/legacy.py), and calls the new code (pyflextrkr/visualization/mcs_cli.py, mcs_map.py).
"""
from pyflextrkr.visualization.mcs_cli import main

if __name__ == "__main__":
    main(legacy_name="plot_subset_tbze_mcs_tracks_demo")
