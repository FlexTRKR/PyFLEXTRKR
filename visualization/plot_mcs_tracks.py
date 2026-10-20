#!/usr/bin/env python
"""
Plot MCS tracks on brightness temperature (Tb) snapshots, with a second field (precipitation or radar reflectivity) or
without, for a user-defined domain and period: one PNG per pixel-level file.

Replaces Analysis/plot_subset_tbpf_mcs_tracks_demo.py, ..._1panel_demo.py, plot_subset_tbze_mcs_tracks_demo.py,
..._1panel_demo.py and plot_subset_tb_mcs_tracks_demo.py (those names still work, see docs/visualization/inventory.md).

Usage:
    python plot_mcs_tracks.py -s STARTDATE -e ENDDATE -c CONFIG.yaml --layout top_bottom
    python plot_mcs_tracks.py -s STARTDATE -e ENDDATE -c CONFIG.yaml --product tbze --panels 1

Required arguments:
-s, --start           Start time (format: YYYY-mm-ddTHH:MM:SS)
-e, --end             End time (format: YYYY-mm-ddTHH:MM:SS)
-c, --config          YAML config file for tracking

What is plotted:
--product             tbpf: Tb + precipitation (default), tbze: Tb + radar reflectivity, tb: Tb only
--panels              2: Tb | second field with the MCS shaded by track number (default), 1: one map, MCS outlined
--layout, -o          2 panels: 'top_bottom' (panel (a) above panel (b), default) or 'left_right'; the old names of the
                      option, --orientation horizontal|vertical, still work and print a deprecation warning
--var2_varname        variable of the second field (default: precipitation; reflectivity_comp for tbze;
                      --dbz_varname is the old name; reflectivity_lowlevel for the low-level reflectivity)

Period, files, parallel (-p 1 --workers N: Dask, use a compute node):
-p, --parallel        0: serial, 1: parallel (default 0)       --workers  number of Dask workers (default 4)
--trackstats_file     track statistics file (default: final MCS stats file of the config)
--pixel_path          pixel-level files directory (default: from the config)
--time_format         pixel-level file datetime format (default yyyymodd_hhmmss)
--track_lookback_days tracks initiated this many days before --start are drawn too (default 4)
--skip_empty          1: no figure for times without a tracked MCS

Map:
--extent              lonmin lonmax latmin latmax (default: the data domain); --subset 1 reads only that domain
--projection          PlateCarree (default), Robinson, Mollweide, EqualEarth, Mercator, LambertConformal, AlbersEqualArea, ...
--central_longitude   central longitude of the projection (e.g. 225 for a Pacific-centered Robinson map)
--lat_range           LAT_MIN LAT_MAX: crop the map to a latitude band (e.g. -60 60)
--lon_ticks --lat_ticks --grid_linewidth --draw_land --draw_border --draw_state --draw_river --map_resolution --box

Track style (path, initiation symbol, number, halo; all default to the original look):
--track_color --track_lw --track_ls --track_alpha
--track_marker --track_marker_size --track_marker_color
--show_track_numbers --track_text_size --track_text_color --track_text_weight --track_text_alpha
--track_halo_color --track_halo_width --track_halo_alpha         (halo: a contrasting outline, with its own alpha)
--perim_color --perim_alpha --perim_thick (km) --perim_width (pixels) --periodic_perimeter
--pfdiam_color --pfdiam_lw --pfdiam_scale --show_pfdiam --show_legend --legend_loc
--cbar_thickness (how thin the colorbars are) --cbar_spacing (gap between the 2 colorbars of a 1-panel map)

Figure:
--output              output directory (default: quicklooks_trackpaths/ in the pixel-level directory)
--figbasename --figname_type --title_prefix --figsize W H --figsize_x --dpi --fontsize

The code is in pyflextrkr/visualization: mcs_cli.py (arguments, main), mcs_map.py (the plot functions and the loop body).
docs/visualization/mcs_maps.md has the full option table and examples.

Zhe Feng, PNNL
contact: Zhe.Feng@pnnl.gov
"""
__author__ = "Zhe.Feng@pnnl.gov"

from pyflextrkr.visualization.mcs_cli import main

if __name__ == "__main__":
    main()
