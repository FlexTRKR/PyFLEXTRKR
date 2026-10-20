#!/usr/bin/env python
"""
Plot tracks on snapshots of a domain in x/y without a map: idealized and cloud-resolving simulations in a doubly
periodic domain (Tb + precipitation with the MCS outlined; tracks wrapped at the edges of the domain) or a single field such as the
cold pool intensity: one PNG per pixel-level file.

Replaces Analysis/plot_subset_tbpf_demo_pbc.py, plot_subset_tbpf_tracks_pinacles_pbc.py and
plot_subset_coldpool_tracks_nomap.py (those names still work, see docs/visualization/inventory.md).

Usage:
    python plot_cartesian_tracks.py -s STARTDATE -e ENDDATE -c CONFIG.yml --extent "0. 1500 0. 1500"
    python plot_cartesian_tracks.py -s STARTDATE -e ENDDATE -c CONFIG.yml --product field --field_varname cp_intensity --xscale 1e-3

Required arguments:
-s, --start           Start time (format: YYYY-mm-ddTHH:MM:SS)
-e, --end             End time (format: YYYY-mm-ddTHH:MM:SS)
-c, --config          YAML config file for tracking

What is plotted:
--product             tbpf: Tb + precipitation (default), field: one field (--field_varname, --cmap, --levels, --cbticks, --cblabel)
--pbc                 1: periodic domain, the tracks are wrapped at the edges (default 1 for tbpf, 0 for field)
--show_pf_markers     red triangles at the PF positions (default 1 for tbpf with --pbc 1)
--xscale --yscale     scale factors of the coordinates, e.g. 1e-3 for meters -> km (default 1 for tbpf, 1e-3 for field)
--extent              xmin xmax ymin ymax in the scaled units, e.g. "0. 1500 0. 1500" (default: the data domain)
--xlabel --ylabel     axis labels (default X (km), Y (km))
--perim_plot          outline of the tracked features: 'contour' (line, default for tbpf) or 'pcolormesh' (pixels)
--perim_thick --perim_linewidth --perim_color --perim_alpha     outline width [km], line width, color, transparency

Period, files, parallel (-p 1 --workers N: Dask, use a compute node):
-p, --parallel        0: serial, 1: parallel (default 0)       --workers  number of Dask workers (default 4)
--trackstats_file     track statistics file (default: the final stats file of the config)
--pixel_path          pixel-level files directory (default: from the config)
--time_format         pixel-level file datetime format (default yyyymodd_hhmmss)
--track_lookback_days tracks initiated this many days before --start are drawn too (default 30)

Track style (all default to the original look):
--track_color --track_lw --track_ls --track_alpha --track_marker --track_marker_size --track_marker_color
--show_track_numbers --track_text_size --track_text_color --track_text_weight --track_text_alpha
--track_halo_color --track_halo_width --track_halo_alpha

Figure:
--output              output directory (default: quicklooks_trackpaths/ in the pixel-level directory)
--figbasename --figname_type --figsize W H --figsize_x (default 12) --dpi --fontsize

The code is in pyflextrkr/visualization: cartesian_cli.py (arguments, main), cartesian_map.py (the plot function and the loop body).
docs/visualization/cartesian_maps.md has the full option table and examples.

Zhe Feng, PNNL
contact: Zhe.Feng@pnnl.gov
"""
__author__ = "Zhe.Feng@pnnl.gov"
__created_date__ = "26-Jan-2023"

from pyflextrkr.visualization.cartesian_cli import main

if __name__ == "__main__":
    main()
