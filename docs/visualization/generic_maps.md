# Generic feature maps: `plot_generic_tracks.py`

Any 2-D field of a generic tracking run (Z500 anomalies, vorticity, stream function, ...) with the outline of each tracked feature, its track
path, symbol and number. One PNG per pixel-level file. The field, its colormap, levels and colorbar are options (they were edits in `__main__`
of the old script).

```
python visualization/plot_generic_tracks.py -s START -e END -c CONFIG.yml [--field_varname VAR --cmap CMAP --levels VMIN VMAX STEP ...] [options]
```

![generic features](../figures/visualization/generic_z500.png)

Pixel-level files: `time`, `longitude`, `latitude`, the field, `cloudtracknumber`; track statistics: `meanlon`, `meanlat`, `base_time`,
`track_duration` of the `finalstats_*.nc` file (or `trackstats_*.nc`) of the config. Defaults of this family: `dpi` 200, font size 13, coastlines
and country borders on (gray, 0.8 points like in every map, see [options.md](options.md)), tracks of the previous 4 days, track path color blueviolet, `--workers` from `nprocesses` of the config. The map keeps
its true shape (`--map_aspect equal`, the colorbar follows it); the old stretched look is `--map_aspect auto`.

## Options of this family

The options of all families are in [options.md](options.md).

| option | default | meaning |
|---|---|---|
| `--field_varname` | — | variable of the pixel-level files to plot (default: field_varname of the config) |
| `--cmap` | — | colormap of the field (matplotlib or colorcet name, default RdBu_r) |
| `--levels` VMIN VMAX STEP | — | shading levels: first, last and step (default: -4 4 0.2) |
| `--cbticks` VMIN VMAX STEP | — | colorbar ticks: first, last and step (default: -4 4 1) |
| `--cblabel` | — | colorbar label (default: Z500 Anomaly) |
| `--remove_oob_low` | `0` | mask the values below the lowest level (0:no, 1:yes) |
| `--remove_oob_high` | `0` | mask the values above the highest level (0:no, 1:yes) |
| `--perim_width` | `1` | width of the outline of the tracked features in pixels |

The hidden options `--track_number_source mask`, `--show_symbols 1`, `--time_label_format` and `--figname_time_format` exist for the stream-function (PSI)
script, whose track number is read from the mask, whose figure names have no seconds and which has a different time label; its defaults are in
`legacy.py`.

## Examples

```bash
cd config                                   # after bash demo_generic_tracking.sh (ERA5 Z500 blocking)
python ../visualization/plot_generic_tracks.py -s 1979-06-01 -e 1979-08-31 -c config_z500_demo.yml --figsize 8 4 -p 1 --output $SCRATCH/quicklooks_z500/

# stream function anomaly (the settings of plot_subset_psi_tracks_demo.py)
python ../visualization/plot_generic_tracks.py -s 2012-06-01T00 -e 2012-07-01T00 -c config_psi_demo.yml --figsize 15 3 -p 1 \
    --cmap RdBu_r --levels -3e7 3e7 0.5e6 --cbticks -3e7 3e7 1e7 --cblabel '600hPa PSI Anomaly' \
    --fontsize 11 --map_resolution 110m --grid_linewidth 0 --draw_border 0 --track_marker_size 6

# any other field and projection: 500 hPa vorticity on a polar map
python ../visualization/plot_generic_tracks.py -s 2020-01-01 -e 2020-01-10 -c config.yml --field_varname vort500 --cmap RdBu_r \
    --levels -2e-4 2e-4 1e-5 --cbticks -2e-4 2e-4 1e-4 --cblabel 'Vorticity (s$^{-1}$)' \
    --projection NorthPolarStereo --central_longitude 0 --extent -180 180 20 90
```

Code map: [structure.md](structure.md).
