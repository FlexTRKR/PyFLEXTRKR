# Options shared by all families

The four scripts (`plot_mcs_tracks.py`, `plot_cell_tracks.py`, `plot_generic_tracks.py`, `plot_cartesian_tracks.py`) share the options below
(the x / y family has no map options: no projection, coastlines or gridlines). The options of one family are in its page:
[mcs_maps.md](mcs_maps.md), [cell_maps.md](cell_maps.md), [generic_maps.md](generic_maps.md), [cartesian_maps.md](cartesian_maps.md).
`python visualization/<script> --help` lists everything for the script.

A `—` in the *default* column means "not set": the family default of the original script applies (listed in the *meaning* column or in the
family page), so that a command without these options makes the same figure as the old script. The options are defined once, in
[`cli_common.py`](../../pyflextrkr/visualization/cli_common.py).

- [All options](#all-options)
- [Which tracks are drawn](#which-tracks-are-drawn)
- [Map projection, central longitude, latitude band](#map-projection-central-longitude-latitude-band)
- [Shape of the map and the figure size](#shape-of-the-map-and-the-figure-size)
- [Track style: path, symbol, number, halo](#track-style-path-symbol-number-halo)

## All options

#### Period, files, parallel

| option | default | meaning |
|---|---|---|
| `-s`, `--start` | — | first time in time series to plot, format=YYYY-mm-ddTHH:MM:SS |
| `-e`, `--end` | — | last time in time series to plot, format=YYYY-mm-ddTHH:MM:SS |
| `-c`, `--config` | — | yaml config file for tracking |
| `-p`, `--parallel` | `0` | flag to run in parallel (0:serial, 1:parallel) |
| `--workers` | `4` | Number of Dask workers for parallel processing (default: 4; the generic and cartesian maps: nprocesses of the config) |
| `--trackstats_file` | — | track stats file name (default: the final stats file of the config) |
| `--pixel_path` | — | Pixel-level tracknumber mask files directory (default: from the config) |
| `--time_format` | — | Pixel-level file datetime format |
| `--track_lookback_days` | — | tracks that were initiated up to this many days before --start are drawn too (default: as the original script of the family: 4 days for MCS and generic maps, 30 for x/y maps, 0 for cell maps) |

#### Output and figure

| option | default | meaning |
|---|---|---|
| `--output` | — | output directory (default: quicklooks_trackpaths/ in the pixel-level directory) |
| `--figbasename` | `` | output figure base name |
| `--figname_type` | `date_time` | output figure name type ('date_time' or 'sequence') |
| `--title_prefix` | `` | Prefix string to add to figure title |
| `--figsize` VALUE ... | — | figure size (width, height) in inches |
| `--figsize_x` | `10` | figure size width in inches (the height follows from the map extent) |
| `--dpi` | — | figure resolution (default: 200; 300 for the cell maps) |
| `--fontsize` | — | font size of labels and text (default: 10 for the MCS maps with a second field; 13 for the Tb-only, cell and generic maps; 14 for the x/y maps) |
| `--skip_empty` | — | skip the times without a tracked object (0:no, 1:yes; default: no, except for the old 1-panel and Tb-only scripts) |

#### Map domain, projection, background

| option | default | meaning |
|---|---|---|
| `--extent` VALUE ... | — | map extent: lonmin lonmax latmin latmax (4 numbers or one quoted string) |
| `--subset` | `0` | flag to subset data (0:no, 1:yes) |
| `--projection` | `PlateCarree` | map projection: PlateCarree (default; central longitude 180 if the extent crosses the dateline), Robinson, Mollweide, EqualEarth, Mercator, LambertConformal, AlbersEqualArea, Orthographic, NorthPolarStereo, SouthPolarStereo |
| `--central_longitude` | — | central longitude of the projection [degree east] |
| `--central_latitude` | — | central latitude of the projection (LambertConformal, AlbersEqualArea, Orthographic) |
| `--standard_parallels` VALUE ... | — | 1 or 2 standard parallels (LambertConformal, AlbersEqualArea) |
| `--lat_range` LAT_MIN LAT_MAX | — | crop the map to this latitude band, e.g. -60 60 for a global map of data within 60S-60N |
| `--lon_ticks` VALUE ... | — | longitudes of the gridlines and labels |
| `--lat_ticks` VALUE ... | — | latitudes of the gridlines and labels |
| `--grid_linewidth` | — | line width of the lat/lon gridlines (0: labels only) |
| `--grid_alpha` | — | transparency of the lat/lon gridlines |
| `--map_aspect` | `equal` | shape of the map: 'equal' (default) keeps the true shape of the domain (a map that does not fill its panel is centered in it and the colorbars follow it); 'auto' stretches the map to fill its panel (the old scripts) |
| `--map_resolution` | `50m` | Map resolution for Natural Earth features ('10m', '50m', '110m') |
| `--draw_land` | — | Draw coastlines (0:no, 1:yes; default 1, 0 for the cell maps) |
| `--draw_border` | — | Draw country borders (0:no, 1:yes; default 0, 1 for the generic maps) |
| `--draw_state` | — | Draw state/province borders (0:no, 1:yes; default 0) |
| `--draw_river` | `0` | Draw rivers (0:no, 1:yes) |
| `--land_color`, `--land_linewidth`, `--land_alpha` | `gray` (MCS maps: `k`), `0.8`, `1` | color, line width [points] and transparency (1: opaque) of the coastlines |
| `--border_color`, `--border_linewidth`, `--border_alpha` | `gray` (MCS maps: `k`), `0.8`, `1` | the same for the country borders |
| `--state_color`, `--state_linewidth`, `--state_alpha` | `gray` (MCS maps: `k`), `0.8`, `1` | the same for the state / province borders |
| `--river_color`, `--river_linewidth`, `--river_alpha` | `gray` (MCS maps: `k`), `0.8`, `1` | the same for the rivers |

The four map features have the same default style (0.8 points, opaque; gray in the cell and generic maps, black in the MCS maps, where gray lines are hard to see on the Tb colormaps: contrast 1.4-3 : 1 against 7-16 : 1 for black) and are drawn in the same order, from top to bottom: coastlines,
country borders, state borders, rivers, all above the shading, the outlines, the track paths, symbols and numbers, the gridlines and the range circles
(only the markers of cities and radars of the cell maps are above them). The old `Analysis/plot_subset_*.py` commands keep the style they had
(black or gray lines, 2 or 1 points for the coastlines, 0.5 points for rivers and the states of the cell maps): [inventory.md](inventory.md).

| `--box` LON0 LON1 LAT0 LAT1 | — | draw a rectangle (e.g. the domain of a regional plot on a global map) |
| `--show_legend` | `1` | Show the legend (0:no, 1:yes) |
| `--legend_loc` | `lower right` | Legend location (matplotlib loc string) |

#### Track style

| option | default | meaning |
|---|---|---|
| `--show_track_numbers` VALUE ... | — | Write the track numbers (0:no, 1:yes); the 2-panel MCS maps take two values, one for each panel: 1 0 writes them on panel (a) only (default: yes) |
| `--track_color` | — | track path color |
| `--track_lw` | — | track path line width [points] |
| `--track_ls` | — | track path line style: solid, dashed, dotted, dashdot (or '-', ':') |
| `--track_alpha` | — | track path transparency (1: opaque) |
| `--track_marker` | — | symbol at the initiation location (matplotlib marker: o, ^, *, s, ...) |
| `--track_marker_size` | — | size of the initiation symbol (scatter size is twice this) |
| `--track_marker_color` | — | color of the initiation symbol (default: the track color) |
| `--track_text_size` | — | font size of the track numbers |
| `--track_text_color` | — | color of the track numbers |
| `--track_text_weight` | — | font weight of the track numbers (normal, bold) |
| `--track_text_alpha` | — | transparency of the track numbers |
| `--track_halo_color` | — | color of a halo around track paths, symbols and numbers (default: no halo) |
| `--track_halo_width` | — | halo width outside the edge [points] (default 2) |
| `--track_halo_alpha` | — | halo transparency (1: opaque, default 1) |


## Which tracks are drawn

`-s` / `-e` select the pixel-level files (one figure per file). The tracks come from the track statistics file of the config
(`--trackstats_file` to give another one); a track is drawn in a frame when

1. it was initiated between `--start` minus `--track_lookback_days` and `--end` (the lookback makes the tracks that began before the first
   frame visible; 4 days for the MCS and generic maps, 30 for the x / y maps, 0 for the cell maps, as the old scripts), and
2. it has not ended more than the retain time before the frame (MCS and generic maps: 1 hour; cell maps: 15 minutes; x / y maps: the time
   resolution of the data), so a track stays on the map for a short time after its end.

The path is drawn from the initiation up to the time of the frame, the initiation symbol at the first position, the track number (and the
precipitation feature circle of the MCS maps) at the position of the frame. `--skip_empty 1` skips the frames without any tracked object
(the old 1-panel and Tb-only scripts did that; the default of the new scripts is to make every frame, which keeps the numbering of an
animation regular).

## Map projection, central longitude, latitude band

`--projection` takes a name (case and `_` do not matter): `PlateCarree` (default), `Robinson`, `Mollweide`, `EqualEarth`, `Mercator`,
`LambertConformal`, `AlbersEqualArea`, `Orthographic`, `NorthPolarStereo`, `SouthPolarStereo`. The parameters that a projection does not
take (e.g. `--central_latitude` for Robinson) are an error, not silently ignored.

| option | projections | meaning |
|---|---|---|
| `--central_longitude` | all | the longitude in the middle of the map: `225` makes a Pacific-centered Robinson map; for `PlateCarree` the default is 180 when the extent crosses the dateline and 0 otherwise |
| `--central_latitude` | LambertConformal, AlbersEqualArea, Orthographic | latitude of the origin |
| `--standard_parallels` | LambertConformal, AlbersEqualArea | one or two latitudes |
| `--extent` | all | the lon / lat box of the data to show; a span of 360 degrees of longitude is the whole globe for Robinson, Mollweide, EqualEarth and Orthographic (the latitudes of the extent are shown for the first three); `140 -140 -20 20` crosses the dateline; a polar cap is `-180 180 20 90` with `NorthPolarStereo` |
| `--lat_range LAT_MIN LAT_MAX` | PlateCarree, Robinson, Mollweide, EqualEarth, Mercator | crop the visible latitude band after the extent is set, e.g. `--lat_range -60 60` for a global map of data that exist between 60S and 60N (the y limits are the projected `LAT_MIN` and `LAT_MAX`, `ax.set_ylim`). The other projections stop with an error (their y does not depend on the latitude only): use `--extent` |
| `--subset 1` | all | read only the pixels inside `--extent` (an index window with one pixel of margin; works across the dateline) |

```bash
# global MCS map, Robinson centered on the Pacific, the 60S-60N band only
python visualization/plot_mcs_tracks.py -s 2025-01-04T12 -e 2025-01-04T13 -c config.yml \
    --extent 0 360 -90 90 --projection Robinson --central_longitude 225 --lat_range -60 60 --map_resolution 110m

# a regional map in the Lambert conformal projection
python visualization/plot_mcs_tracks.py -s 2019-06-01T00 -e 2019-06-01T06 -c config.yml \
    --extent -125 -70 24 50 --projection LambertConformal --central_longitude -97 --standard_parallels 30 45
```

Longitudes of the data and of the tracks may be in -180..180, 0..360 or unwrapped (e.g. -195..216 or +-545 in the global V3 statistics):
they are normalized for the projections other than `PlateCarree`, which uses the data as they are (as the old scripts did).

**Without `--extent`** the map shows the domain of the data. For a projection other than PlateCarree and regional data this is the box of the data
in the coordinates of the projection: data on a Lambert conformal grid (e.g. WRF) shown with the same Lambert conformal projection
(`--projection LambertConformal --central_longitude -97 --standard_parallels 31 35` for the WRF CONUS domain) fill the map, with no empty corners
and no need for `--extent` or `--subset`. The projection parameters of a WRF domain are the global attributes `STAND_LON`, `TRUELAT1`, `TRUELAT2`
of the WRF files.

**Latitude labels** of a whole-world map (Robinson, Mollweide, EqualEarth) are written along the curved left edge of the map, one for every gridline
(`--lat_ticks` or the automatic ones: every 20 degrees, e.g. 60S ... 60N for `--lat_range -60 60`; a label is drawn for a latitude that is inside the
band, including its ends). The gridliner of cartopy writes a label where a gridline meets the rectangle of the axes, which leaves the middle latitudes
(+-40) without label for the figure sizes in which the gridline ends at the curved edge, inside the rectangle.

## Shape of the map and the figure size

`--map_aspect equal` (the default) keeps the true shape of the map: 1 degree of longitude is as long as 1 degree of latitude for PlateCarree, and the
projected shape for the other projections. If the panel that the layout gives to the map has another shape (a `--figsize` that does not fit), the map
is centered in its panel and **the colorbars follow the map**: they are as long as the map is wide (colorbars below) or high (colorbars at the right) and
at the same distance from it. Without `--figsize` the figure size is computed from the shape of the map (`--extent`, or the domain of the data) and the
layout of the panels (`--figsize_x` sets the width, 10 inches by default), so the map fills its panel.
`--map_aspect auto` is the look of the old scripts: the map is stretched to fill its panel, and the figure size of the old formula is used for a
PlateCarree map without `--figsize` (the stubs of `Analysis/` use it). A map of South America in a 10 x 8 inch figure looks squeezed with `auto`.
The MCS maps also keep the text of the colorbars inside the figure and have options for how thin the colorbars are ([mcs_maps.md](mcs_maps.md#colorbars)).

![projections](../figures/visualization/projections.png)

## Track style: path, symbol, number, halo

All defaults are the looks of the old scripts. The options below change one element at a time:

| element | options | notes |
|---|---|---|
| path (line) | `--track_color`, `--track_lw` [points], `--track_ls` (`solid`, `dashed`, `dotted`, `dashdot` or `-`, `--`, `:`), `--track_alpha` | defaults: MCS maps purple, 1.5 points; cell maps black (symbols colored by lifetime); generic and x / y maps blueviolet, 1.5 |
| initiation symbol | `--track_marker` (matplotlib marker: `o`, `^`, `*`, `s`, ...), `--track_marker_size`, `--track_marker_color` | default: a circle in the color of the path; the symbol is drawn at twice the size of the path points |
| track number | `--show_track_numbers 0/1` (the 2-panel MCS maps take two values, one per panel: `1 0` = on panel (a) only), `--track_text_size`, `--track_text_color`, `--track_text_weight`, `--track_text_alpha` | default: black, bold; 12 points (MCS maps with a second field), 10 (Tb only, generic maps), 0.8 x font size (cell maps), 14 (x / y maps; red for the x / y Tb + precipitation maps); `--fontsize` does not change them |
| halo | `--track_halo_color` (not set = no halo), `--track_halo_width` [points outside the edge, default 2], `--track_halo_alpha` [0-1, default 1] | an outline in a contrasting color around the path, the symbol and the number together (matplotlib path effects, `trackdraw.halo_effects`); `alpha` is the transparency of the halo only |
| outline of the tracked object | `--perim_color`, `--perim_alpha`, `--perim_thick` [km] / `--perim_width` [pixels] | the family pages |

`alpha` options are independent: `--track_alpha` makes the path (and the symbol) transparent, `--track_text_alpha` the number, `--track_halo_alpha` the halo.

```bash
# the four panels of the figure below: the Sahel on 22 August 2025 at 20:30 UTC (the global V3 tracking, config/config_imerg_global_mcs_v3_2025.yml)
BASE="-s 2025-08-22T20 -e 2025-08-22T21 -c config/config_imerg_global_mcs_v3_2025.yml --panels 1 --extent -20 60 0 30 --subset 1 --draw_border 1
      --figsize_x 9 --dpi 80 --fontsize 11 --land_linewidth 1 --track_lookback_days 3"
python visualization/plot_mcs_tracks.py $BASE --output a/                                                                    # (a) default
python visualization/plot_mcs_tracks.py $BASE --output b/ --track_halo_color white --track_halo_width 1.5                    # (b) white halo
python visualization/plot_mcs_tracks.py $BASE --output c/ --track_halo_color white --track_halo_width 2.5 --track_halo_alpha 0.4 \
    --track_text_size 9 --track_text_alpha 0.7                                                                               # (c) faint halo, small numbers
python visualization/plot_mcs_tracks.py $BASE --output d/ --track_color red --track_lw 2.5 --track_ls dashed \
    --track_marker '*' --track_marker_size 12 --track_text_color blue                                                        # (d) red dashed path, star, blue numbers
```

The frame was picked from the hourly frames of 18-31 August 2025 (two weeks of 168 frames each, `-p 1 --workers 32`: 30 seconds for a week): the MCSs of
the Sahel mostly move westward and live for a day or more, so their paths are long and do not turn back. Track 23730, the long path across the middle of the
map, has traveled about 3400 km (the length of its path) towards the west.

![track style options](../figures/visualization/track_style_options.png)
