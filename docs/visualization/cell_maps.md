# Convective cell maps: `plot_cell_tracks.py`

Maps of radar convective cells: the composite reflectivity (or the 10 dBZ echo-top height) with the outline of each tracked cell, the cell
track paths (symbols colored by lifetime), the track numbers, the radar range rings and azimuth lines, and optionally the terrain, one outline color per track, only the
cells of long-lived tracks, and cities and other radars (the functions of the cell part of `plotmap.py` of conus_tracking). One PNG per pixel-level file.

```
python visualization/plot_cell_tracks.py -s START -e END -c CONFIG.yml [--product refl|eth] [--radar_lat LAT --radar_lon LON] [options]
```

| `--product` | field | default look | replaces |
|---|---|---|---|
| `refl` (default) | `dbz_comp` composite reflectivity (dBZ), `gist_ncar`, alpha 0.85, tracks shown | paths 1 point, symbols 10 | `plot_subset_cell_tracks_demo.py` |
| `eth` | `echotop10` (m, scaled to km with `--var_scale`), `nipy_spectral`, tracks not shown (`--show_tracks 1` to show them) | paths 3 points, symbols 30 | `plot_subset_cell_tracks_ETH_demo.py` (also `--skip_empty 1`) |

![cell reflectivity](../figures/visualization/cell_refl.png)

The radar location (`--radar_lat`, `--radar_lon`) defaults to the pixel closest to the center of the domain. The range rings (20-100 km) and the
azimuth lines (every 90 degrees, 200 km) are geodesic lines on the WGS84 ellipsoid (`mapcore.geodesic_circle`, `geodesic_ray`) drawn with the transform of the data like
everything else on the map, so they are centered on the radar (in frames with echo all around the radar, the white hole of the blind zone above it is in the middle of the
rings). They are not made with `ax.tissot` and `transform=ccrs.Geodetic()` (the first version of these maps): with PROJ 9.8 (cartopy 0.25, the `pyflex26.3` environment)
the transform from Geodetic to PlateCarree moves a point north or south by up to 0.19 degree, so the radar at 32.13S was drawn at 31.94S, 20 km off (on a Lambert map
too); the older PROJ of the `pyflex-dev` environment does not do it. On a PlateCarree map (the default) the range rings are ellipses, because the map keeps 1 degree of longitude as wide as 1 degree of
latitude (at 29.5N a degree of longitude is only 0.87 of a degree of latitude): a map in a projection that keeps the distances, centered on the radar (`--projection LambertConformal
--central_longitude RADAR_LON --central_latitude RADAR_LAT --standard_parallels RADAR_LAT`) with an `--extent` of the same length in km to the east, west, north and south of the
radar, has round rings and a square domain; [`examples/make_cell_animation_nexrad.py`](../../visualization/examples/make_cell_animation_nexrad.py) is the NEXRAD demo this way. The cell center for the numbers is
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
| `--perim_cmap` (`--cell_perim_cmap`) | — | colormap of the cell outlines: one color per track (e.g. `glasbey_dark`; colorcet or matplotlib names), instead of `--perim_color` |
| `--perim_halo` (`--cell_halo`) | — | color of a ring just outside the cell outlines (default: none) |
| `--min_hours` (`--cell_min_hours`) | `0` | draw only the cells of tracks that live at least this long [hour] (0: all) |
| `--show_terrain` | `0` | draw the terrain height (gray) under the reflectivity, from `terrain_file` and `elev_varname` of the config (0:no, 1:yes) |
| `--grid_linestyle` | `--` | line style of the lat/lon gridlines (they are drawn with `--grid_linewidth`) |
| `--city NAME LON LAT` | — | mark a city: a yellow diamond and its name (repeat the option for more cities) |
| `--radar_site NAME LON LAT` | — | mark another radar: a cyan circle, its name and its range circle (repeat the option for more radars) |
| `--radar_site_radius` | `150` | radius of the range circle of the `--radar_site` radars [km] (0: no circle) |

These options are the functions of the cell part of `plotmap.py` (conus_tracking), with the names of the options above (`--cell_halo`, `--cell_perim_cmap`, `--cell_min_hours`
are the names there and work here too). `--min_hours` removes the cells of the shorter tracks from the whole picture (outline, number, path, symbol), the outline colors of
`--perim_cmap` are the same for a track in every frame, the ring (`--perim_halo`) is drawn under the outlines, and the terrain (`--show_terrain 1`, white below 500 m, then
gray getting darker to 4000 m) needs `terrain_file` and `elev_varname` in the config (the CACTI CSAPR configs have them). plotmap has the Arizona sites hard-coded; here the cities and radars
are given on the command line (or `cities` and `radar_sites` as lists of `[name, lon, lat]` in a wrapper such as `make_cell_animation_nexrad.py`). The names of the other radars are cyan with a thin black
outline and their range circle too, because the light cyan can not be read on the reflectivity otherwise.

![cell options](../figures/visualization/cell_options.png)

The four pictures are the CACTI CSAPR2 frame of 2019-01-25 18:45 UTC (the options at the top of each, on top of `-s 2019-01-25T18 -e 2019-01-25T19 -c config_csapr_demo.yml --radar_lat -32.12641 --radar_lon -64.72837`):
(a) none, (b) `--show_terrain 1 --min_hours 1`, (c) `--perim_cmap glasbey_dark --perim_halo white --perim_alpha 1`, (d) `--city Cordoba -64.19 -31.42 --city "Alta Gracia" -64.43 -31.65 --radar_site "Radar B" -64.0 -32.5 --radar_site_radius 40`.

## Examples

```bash
cd config                                   # after bash demo_cell_csapr.sh
python ../visualization/plot_cell_tracks.py -s 2019-01-25T12 -e 2019-01-26T00 -c config_csapr_demo.yml \
    --radar_lat -32.12641 --radar_lon -64.72837 -p 1 --figsize 8 7 --output $SCRATCH/quicklooks_cell/

# a window around the radar, with the state borders and a white halo around the paths and numbers
python ../visualization/plot_cell_tracks.py -s 2019-01-25T16 -e 2019-01-25T19 -c config_csapr_demo.yml \
    --extent -66 -63 -34 -31 --subset 1 --draw_land 1 --draw_border 1 --draw_state 1 --track_halo_color white --track_halo_alpha 0.6
```

## A square map around the radar (NEXRAD)

The NEXRAD demo (KHGX, Houston, 7 August 2014; `config/demo_cell_nexrad.sh`) on a square map of 2 x 105 km around the radar. A PlateCarree map keeps 1 degree of longitude as wide as 1 degree of latitude,
but at 29.5N a degree of longitude is 0.87 times as long, so the range circles would be 15 % too wide; on a LambertConformal map centered on the radar they are round. The frame is 13:42 UTC of the animation,
with the title prefix `NEXRAD (KHGX)`, the coastline and the state borders (1 point, gray: `--land_linewidth 1`, the cell maps draw them in gray, the MCS maps in black), and tick labels every 0.5 degree:

![NEXRAD cell tracking](../figures/visualization/cell_nexrad.png)

![NEXRAD cell tracking animation](../figures/readme/cell_nexrad_20140807.gif)

```bash
python ../visualization/plot_cell_tracks.py -s 2014-08-07T12:00 -e 2014-08-07T15:00 -c config_cell_demo.yml \
    --radar_lat 29.4719 --radar_lon -95.0787 --extent -96.163 -93.994 28.528 30.416 --subset 1 \
    --projection LambertConformal --central_longitude -95.0787 --central_latitude 29.4719 --standard_parallels 29.4719 \
    --title_prefix 'NEXRAD (KHGX)' --figsize 8 7.2 --lon_ticks -96 -95.5 -95 -94.5 -94 --lat_ticks 28.5 29 29.5 30 \
    --draw_land 1 --land_linewidth 1 --draw_border 1 --draw_state 1 --map_resolution 10m --figbasename cell_tracks_ --output quicklooks/
```

`--figsize 8 7.2` makes the map panel of the layout square (the automatic height is not shorter than 9.2 inches, with white space above and below a square map). The animation is made by the example below.

## Animations

[`visualization/examples/make_cell_animation_nexrad.py`](../../visualization/examples/make_cell_animation_nexrad.py) (the NEXRAD demo above; the animation is an mp4, `--set gif_width=800` also makes the GIF; `map_half_width_km` makes the square map from the radar location; the code is `pyflextrkr/visualization/cell_animation.py`): parameter block (dates, config, radar location, domain around it, products, output
folders, ffmpeg) -> `cell_cli.main(argv)` -> `animation.make_animation`; `--set NAME=VALUE` overrides a parameter. Map projection and track style
parameters are in the block too (`projection`, `central_longitude`, `track_color`, `track_halo_color`, ...), and so are the options of plotmap (`perim_cmap`, `perim_halo`, `min_hours`,
`show_terrain`, `grid_linestyle`, `cities` and `radar_sites` as lists of `[name, lon, lat]`, `radar_site_radius`). Code map: [structure.md](structure.md).
