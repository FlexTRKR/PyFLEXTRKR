#!/usr/bin/env python
"""
Plot convective cell tracks on radar reflectivity (or echo-top height) snapshots for a single radar domain: one PNG per
pixel-level file. Cells are outlined and numbered, track paths and symbols are colored by lifetime, range circles and
azimuth lines mark the radar.

Replaces Analysis/plot_subset_cell_tracks_demo.py and plot_subset_cell_tracks_ETH_demo.py (those names still work, see
docs/visualization/inventory.md).

Usage:
>python plot_cell_tracks.py -s STARTDATE -e ENDDATE -c CONFIG.yml --radar_lat LAT --radar_lon LON

Required arguments:
-s, --start           Start time (format: YYYY-mm-ddTHH:MM:SS)
-e, --end             End time (format: YYYY-mm-ddTHH:MM:SS)
-c, --config          YAML config file for tracking

What is plotted:
--product             refl: composite reflectivity (default), eth: 10 dBZ echo-top height [km]
--varname             variable of the pixel-level files (default: dbz_comp, echotop10 for eth)
--var_scale           scale factor of the variable (default 1, 1e-3 for eth)
--radar_lat --radar_lon   radar location (default: the domain center)
--show_rangecircle --show_azimuth    radar range circles and azimuth lines (default 1)
--show_tracks --show_paths --show_symbols    track paths and symbols (default 1; 0 for eth)

Period, files, parallel (-p 1 --workers N: Dask, use a compute node):
-p, --parallel        0: serial, 1: parallel (default 0)       --workers  number of Dask workers (default 4)
--trackstats_file     track statistics file (default: trackstats file of the config)
--pixel_path          pixel-level files directory (default: from the config)
--time_format         pixel-level file datetime format (default yyyymodd_hhmmss)
--track_lookback_days tracks initiated this many days before --start are drawn too (default 0)

Map:
--extent              lonmin lonmax latmin latmax (default: the data domain); --subset 1 reads only that domain
--projection --central_longitude --central_latitude --standard_parallels --lat_range
--lon_ticks --lat_ticks --grid_linewidth --draw_land --draw_border --draw_state --draw_river --map_resolution

Track style (all default to the original look; path color black, symbols colored by lifetime):
--track_color --track_lw --track_ls --track_alpha --track_marker --track_marker_size --track_marker_color
--show_track_numbers --fontsize_tracks (--track_text_size) --track_text_color --track_text_weight --track_text_alpha
--track_halo_color --track_halo_width --track_halo_alpha
--perim_color --perim_alpha --perim_width    cell outlines

Figure:
--output              output directory (default: quicklooks_trackpaths/ in the pixel-level directory)
--figbasename --figname_type --title_prefix --figsize W H --figsize_x --dpi --fontsize

The code is in pyflextrkr/visualization: cell_cli.py (arguments, main), cell_map.py (the plot function and the loop body).
docs/visualization/cell_maps.md has the full option table and examples.

Zhe Feng, PNNL
contact: Zhe.Feng@pnnl.gov
"""
__author__ = "Zhe.Feng@pnnl.gov"
__created_date__ = "08-Jun-2022"

from pyflextrkr.visualization.cell_cli import main

if __name__ == "__main__":
    main()
