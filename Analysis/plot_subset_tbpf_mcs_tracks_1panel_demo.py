"""
MOVED: this script is now visualization/plot_mcs_tracks.py (--product tbpf --panels 1 --skip_empty 1 --map_aspect auto).

This stub keeps the old command working: it takes the same arguments, with the defaults of the old script
(pyflextrkr/visualization/legacy.py), and calls the new code (pyflextrkr/visualization/mcs_cli.py, mcs_map.py).
"""
from pyflextrkr.visualization.mcs_cli import main

if __name__ == "__main__":
    main(legacy_name="plot_subset_tbpf_mcs_tracks_1panel_demo")
