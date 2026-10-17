"""
MOVED: this script is now visualization/plot_generic_tracks.py (--cmap RdBu_r --levels -3e7 3e7 0.5e6 --cbticks -3e7 3e7 1e7 --cblabel '600hPa PSI Anomaly' --fontsize 11 --map_resolution 110m --grid_linewidth 0 --track_marker_size 6 --draw_border 0 --map_aspect auto).

This stub keeps the old command working: it takes the same arguments, with the defaults of the old script
(pyflextrkr/visualization/legacy.py), and calls the new code (pyflextrkr/visualization/generic_cli.py, generic_map.py).
"""
from pyflextrkr.visualization.generic_cli import main

if __name__ == "__main__":
    main(legacy_name="plot_subset_psi_tracks_demo")
