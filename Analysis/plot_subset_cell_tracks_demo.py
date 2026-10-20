"""
MOVED: this script is now visualization/plot_cell_tracks.py (--product refl --map_aspect auto).

This stub keeps the old command working: it takes the same arguments, with the defaults of the old script
(pyflextrkr/visualization/legacy.py), and calls the new code (pyflextrkr/visualization/cell_cli.py, cell_map.py).
"""
from pyflextrkr.visualization.cell_cli import main

if __name__ == "__main__":
    main(legacy_name="plot_subset_cell_tracks_demo")
