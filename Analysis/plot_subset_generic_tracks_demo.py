"""
MOVED: this script is now visualization/plot_generic_tracks.py (--map_aspect auto).

This stub keeps the old command working: it takes the same arguments, with the defaults of the old script
(pyflextrkr/visualization/legacy.py), and calls the new code (pyflextrkr/visualization/generic_cli.py, generic_map.py).
"""
from pyflextrkr.visualization.generic_cli import main

if __name__ == "__main__":
    main(legacy_name="plot_subset_generic_tracks_demo")
