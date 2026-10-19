#!/usr/bin/env python
"""
Plot generic feature tracks (Z500 anomaly, stream function, vorticity, reflectivity, ...) on snapshots of the tracked 2D
field: one PNG per pixel-level file. The features are outlined, track paths and numbers are overplotted.

Replaces Analysis/plot_subset_generic_tracks_demo.py and plot_subset_psi_tracks_demo.py (those names still work, see
docs/visualization/inventory.md).

Usage:
    python plot_generic_tracks.py -s STARTDATE -e ENDDATE -c CONFIG.yml
    python plot_generic_tracks.py -s STARTDATE -e ENDDATE -c CONFIG.yml --cmap gist_ncar --levels 0 70 5 --cbticks 0 70 10 --cblabel 'Reflectivity (dBZ)'

Required arguments:
-s, --start           Start time (format: YYYY-mm-ddTHH:MM:SS)
-e, --end             End time (format: YYYY-mm-ddTHH:MM:SS)
-c, --config          YAML config file for tracking

What is plotted (the defaults are for the Z500 anomaly demo):
--field_varname       variable of the pixel-level files (default: field_varname of the config)
--cmap                colormap (matplotlib or colorcet name, default RdBu_r)
--levels VMIN VMAX STEP   shading levels (default -4 4 0.2)       --cbticks VMIN VMAX STEP   colorbar ticks (default -4 4 1)
--cblabel             colorbar label        --remove_oob_low / --remove_oob_high   mask the values outside the levels
--perim_width         outline width of the features in pixels (default 1)

Period, files, parallel (-p 1 --workers N: Dask, use a compute node):
-p, --parallel        0: serial, 1: parallel (default 0)       --workers  number of Dask workers (default 4)
--trackstats_file     track statistics file (default: final stats file of the config, else trackstats)
--pixel_path          pixel-level files directory (default: from the config)
--time_format         pixel-level file datetime format (default yyyymodd_hhmmss)
--track_lookback_days tracks initiated this many days before --start are drawn too (default 4)

Map:
--extent              lonmin lonmax latmin latmax (default: the data domain); --subset 1 reads only that domain
--projection --central_longitude --central_latitude --standard_parallels --lat_range
--lon_ticks --lat_ticks --grid_linewidth --draw_land --draw_border --draw_state --draw_river --map_resolution

Track style (all default to the original look):
--track_color --track_lw --track_ls --track_alpha --track_marker --track_marker_size --track_marker_color
--show_track_numbers --track_text_size --track_text_color --track_text_weight --track_text_alpha
--track_halo_color --track_halo_width --track_halo_alpha

Figure:
--output              output directory (default: quicklooks_trackpaths/ in the pixel-level directory)
--figbasename --figname_type --title_prefix --figsize W H (default 8 7) --dpi --fontsize

The code is in pyflextrkr/visualization: generic_cli.py (arguments, main), generic_map.py (the plot function and the loop body).
docs/visualization/generic_maps.md has the full option table and examples.

Zhe Feng, PNNL
contact: Zhe.Feng@pnnl.gov
"""
__author__ = "Zhe.Feng@pnnl.gov"
__created_date__ = "26-Jan-2023"

from pyflextrkr.visualization.generic_cli import main

if __name__ == "__main__":
    main()
