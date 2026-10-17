# Convective cell maps: `plot_cell_tracks.py`

Maps of radar convective cells: the composite reflectivity (or the 10 dBZ echo-top height) with the outline of each tracked cell, the cell
track paths (symbols colored by lifetime), the track numbers, the radar range rings and azimuth lines. One PNG per pixel-level file.

```
python visualization/plot_cell_tracks.py -s START -e END -c CONFIG.yml [--product refl|eth] [--radar_lat LAT --radar_lon LON] [options]
```

| `--product` | field | default look | replaces |
|---|---|---|---|
| `refl` (default) | `dbz_comp` composite reflectivity (dBZ), `gist_ncar`, alpha 0.85, tracks shown | paths 1 point, symbols 10 | `plot_subset_cell_tracks_demo.py` |
| `eth` | `echotop10` (m, scaled to km with `--var_scale`), `nipy_spectral`, tracks not shown (`--show_tracks 1` to show them) | paths 3 points, symbols 30 | `plot_subset_cell_tracks_ETH_demo.py` (also `--skip_empty 1`) |

![cell reflectivity](../figures/visualization/cell_refl.png)

The radar location (`--radar_lat`, `--radar_lon`) defaults to the pixel closest to the center of the domain. The cell center for the numbers is
the mean position of the pixels with the convective mask, the cell tracks are in the cell statistics file (`trackstats_*.nc`) with `cell_meanlon`,
`cell_meanlat`, `base_time`, `track_duration`. The pixel-level files have `tracknumber`, `conv_mask`, the field, `longitude`, `latitude` and the
radar-relative `lon` / `lat` (km).

## Options of this family

The options of all families are in [options.md](options.md); the track style options change the cell paths as follows: `--track_color` the path
color (the symbols stay colored by lifetime), `--track_lw` and `--track_marker_size` set one value for the three lifetime classes,
`--track_text_size` is `--fontsize_tracks`. The default `--dpi` is 300, `--draw_land` 0, `--track_lookback_days` 0. The map keeps its true shape (`--map_aspect equal`, the colorbar follows it);
the old stretched look is `--map_aspect auto`.

| option | default | meaning |
|---|---|---|
| `--radar_lat` | — | radar latitude (optional, defaults to domain center) |
| `--radar_lon` | — | radar longitude (optional, defaults to domain center) |
| `--product` | `refl` | field: 'refl' composite reflectivity, 'eth' 10 dBZ echo-top height (km) |
| `--varname` | — | Variable name for plotting in pixel files (default: dbz_comp, echotop10 for eth) |
| `--var_scale` | — | scale factor of the variable (default 1, 1e-3 for eth: m -> km) |
| `--show_rangecircle` | `1` | Draw radar range circles (0:no, 1:yes) |
| `--show_azimuth` | `1` | Draw azimuth lines (0:no, 1:yes) |
| `--show_tracks` | — | Show track paths (0:no, 1:yes) |
| `--show_paths` | — | Show track path lines (0:no, 1:yes, None:auto from show_tracks) |
| `--show_symbols` | — | Show track centroid symbols (0:no, 1:yes, None:auto from show_tracks) |
| `--fontsize_tracks` | — | Font size for track numbers (default: fontsize*0.8) |
| `--perim_color` | — | color of the cell outlines (default: black) |
| `--perim_alpha` | — | transparency of the cell outlines (default 0.6) |
| `--perim_width` | `1` | cell outline width in pixels |

## Examples

```bash
cd config                                   # after bash demo_cell_csapr.sh
python ../visualization/plot_cell_tracks.py -s 2019-01-25T12 -e 2019-01-26T00 -c config_csapr_demo.yml \
    --radar_lat -32.12641 --radar_lon -64.72837 -p 1 --figsize 8 7 --output $SCRATCH/quicklooks_cell/

# a window around the radar, with the state borders and a white halo around the paths and numbers
python ../visualization/plot_cell_tracks.py -s 2019-01-25T16 -e 2019-01-25T19 -c config_csapr_demo.yml \
    --extent -66 -63 -34 -31 --subset 1 --draw_land 1 --draw_border 1 --draw_state 1 --track_halo_color white --track_halo_alpha 0.6
```

## Animations

[`visualization/make_cell_animation.py`](../../visualization/make_cell_animation.py): parameter block (dates, config, radar location, domain around it, products, output
folders, ffmpeg) -> `cell_cli.main(argv)` -> `animation.make_mp4`; `--set NAME=VALUE` overrides a parameter. Map projection and track style
parameters are in the block too (`projection`, `central_longitude`, `track_color`, `track_halo_color`, ...). Code map: [structure.md](structure.md).
