"""
MOVED: this script is now visualization/plot_cartesian_tracks.py (--product tbpf).

This stub keeps the old command working: it takes the same arguments, with the defaults of the old script
(pyflextrkr/visualization/legacy.py), and calls the new code (pyflextrkr/visualization/cartesian_cli.py, cartesian_map.py).
"""
from pyflextrkr.visualization.cartesian_cli import main

if __name__ == "__main__":
    main(legacy_name="plot_subset_tbpf_demo_pbc")
