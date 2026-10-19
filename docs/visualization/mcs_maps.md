# MCS maps: `plot_mcs_tracks.py`

Brightness temperature (Tb) maps with the tracked MCS: the outline of each MCS, the track path from the initiation to the time of the
frame, the initiation symbol, the track number and the precipitation feature (PF) circle. One PNG per pixel-level file.
One script for what were five: the products differ in the second field, the colors and the layout only.

```
python visualization/plot_mcs_tracks.py -s START -e END -c CONFIG.yml [--product tbpf|tbze|tb] [--panels 2|1] [--layout top_bottom|left_right] [options]
```

| `--product` | second field | `--panels 2` (default) | `--panels 1` | replaces |
|---|---|---|---|---|
| `tbpf` (default) | `precipitation` (mm h-1) | (a) Tb and (b) precipitation with the tracked MCS shaded by track number | Tb (gray) + precipitation on one map, MCS outlined in a color per track | `plot_subset_tbpf_mcs_tracks_demo.py`, `..._1panel_demo.py` |
| `tbze` | `reflectivity_comp` (dBZ; `--var2_varname reflectivity_lowlevel` for the low-level reflectivity) | (a) Tb and (b) reflectivity, MCS shaded | Tb + reflectivity on one map | `plot_subset_tbze_mcs_tracks_demo.py`, `..._1panel_demo.py` |
| `tb` | none (Tb only) | one panel (always) | Tb with the MCS outlined (`--perim_width` pixels) | `plot_subset_tb_mcs_tracks_demo.py` |

`--layout top_bottom` (default) puts panel (a) above panel (b), colorbars at the right; `--layout left_right` puts them side by side, colorbars below.
The old option was `-o` / `--orientation` with the values `horizontal` (= top_bottom) and `vertical` (= left_right), which are easy to mix up: they
still work and print a deprecation warning.

The figure size comes from the shape of the map (`--extent`, or the domain of the data) and the layout, so the maps are not squeezed
(`--figsize_x` sets the width, `--figsize W H` fixes both; a map that does not fill its panel is centered in it and the colorbars follow it). The old
stretched look is `--map_aspect auto`. See [options.md](options.md#shape-of-the-map-and-the-figure-size). A map much wider than high (the 60S-60N band of the
globe, the Sahel) gets margins in inches for the titles and the longitude labels, so the figure is not taller than the maps need.

![tbpf, 2 panels](../figures/visualization/mcs_tbpf_2panel.png)

### Colorbars

`--cbar_thickness` sets how thin the colorbars are, as a fraction of the map width (colorbars at the right of the map) or of the map height (colorbars below
it), so the same number means the same for both layouts (defaults: 0.02 for 2 panels, 1/70 for 1 panel with 2 colorbars, 0.03 for Tb only).
`--cbar_spacing` is the gap between the 2 colorbars of a 1-panel map, as a fraction of the map width (default 5/70).
When the map keeps its shape (`--map_aspect equal`, the default) the text of the colorbars (tick labels and label) is measured and kept inside the figure:
a colorbar that would overlap the text of the previous one moves away from it (unless you gave `--cbar_spacing`), and the maps shrink a little (all by the
same amount, keeping their shape) if the text would end outside the figure.

```bash
# thin colorbars with more room between them, large text
python ../visualization/plot_mcs_tracks.py -s 2019-01-25T13 -e 2019-01-25T14 -c config_mcs_demo.yml --panels 1 \
    --cbar_thickness 0.008 --cbar_spacing 0.12 --fontsize 14 --output $SCRATCH/quicklooks_cbar/
```

Pixel-level files: `time`, `longitude`, `latitude` (2-D), `tb` (`--tb_varname`), the second field, `cloudtracknumber`; track statistics: the final MCS
statistics file (`mcs_tracks_final_*.nc`, or `mcs_tracks_tb_*.nc` for `--product tb`) with `base_time`, `track_duration`, `meanlon`/`meanlat`
(`meanlon_smooth`/`meanlat_smooth` are used when present, `--track_smooth 0` uses the raw center) and the PF variables (`pf_lon_centroid`,
`pf_lat_centroid`, `pf_area`; without them no PF circle is drawn).

## Options of this family

The options of all families (period, files, output, map, projection, track style) are in [options.md](options.md).
`--show_track_numbers 1 0` writes the track numbers on panel (a) (Tb) and not on panel (b); `--fontsize` sets the size of all text except the track
numbers (`--track_text_size`).

| option | default | meaning |
|---|---|---|
| `-o`, `--layout`, `--orientation` | `top_bottom` | 2 panels: 'top_bottom' (default: panel (a) above panel (b), colorbars at the right) or 'left_right' (panel (a) left of panel (b), colorbars below). -o and --orientation are the old names of this option; their values horizontal (= top_bottom) and vertical (= left_right) still work but print a deprecation warning |
| `--product` | `tbpf` | fields: 'tbpf' Tb + precipitation, 'tbze' Tb + radar reflectivity, 'tb' Tb only |
| `--panels` | `2` | 1 panel (Tb and the second field on one map, MCS outlined by track) or 2 panels (Tb \| second field with the MCS shaded) |
| `--tb_varname` | `tb` | Tb variable name in the pixel-level files |
| `--var2_varname`, `--dbz_varname` | — | variable name of the second field in the pixel-level files (default: precipitation for tbpf, reflectivity_comp for tbze; 'reflectivity_lowlevel' is the low-level reflectivity) |
| `--track_smooth` | `1` | use the smoothed track center (meanlon_smooth, meanlat_smooth) if the stats file has it (0:no, 1:yes) |
| `--perim_thick` | — | MCS outline width [km] (default 10, 6 for 1-panel tbze) |
| `--perim_width` | — | MCS outline width in pixels (replaces --perim_thick) |
| `--perim_color` | — | draw the MCS outlines in this color (default: magenta on Tb, a color per track on 1 panel) |
| `--perim_alpha` | — | transparency of the MCS outlines when --perim_color is given |
| `--periodic_perimeter` | `none` | the MCS outlines continue across the edges of a periodic domain ('auto': from pbc_direction of the file; not with --subset 1) |
| `--pfdiam_color` | — | color of the PF diameter circles on 2 panels |
| `--pfdiam_lw` | — | line width of the PF diameter circles |
| `--pfdiam_scale` | — | scale of the PF diameter circles (1: the PF diameter) |
| `--show_pfdiam` | `1` | draw the PF diameter circles on 2 panels (0:no, 1:yes); 0 also takes them out of the legend |
| `--cbar_thickness` | — | thickness of the colorbars as a fraction of the map width (colorbars at the right of the map) or of the map height (colorbars below it): the same for any layout, a smaller value is a thinner colorbar (default: 0.02 for 2 panels, 0.0143 (1/70) for 1 panel with 2 colorbars, 0.03 for Tb only) |
| `--cbar_spacing` | — | 1 panel with 2 colorbars: the gap between them as a fraction of the map width (default: 0.0714 (5/70), widened if the text of the first colorbar needs more) |

The look of each product (levels, colormaps, titles, colorbar labels, outline width, `dpi`, font size) is set in the product block of `main()` in
[`mcs_cli.py`](../../pyflextrkr/visualization/mcs_cli.py); the drawing is in [`mcs_map.py`](../../pyflextrkr/visualization/mcs_map.py) (`plot_map_2panels`, `plot_map_1panel`).

| product / panels | `dpi` | font size | outline | notes |
|---|---|---|---|---|
| tbpf, 2 | 200 | 10 | 10 km, magenta | track numbers 12 points; gridlines above the data |
| tbze, 2 | 200 | 10 | 10 km | the second field is masked where it is below the lowest level |
| tbpf, 1 | 200 | 10 | 10 km, a color per track | Tb alpha 0.7, precipitation alpha 0.9 |
| tbze, 1 | 200 | 10 | 6 km, a color per track | |
| tb | 200 | 13 | 6 pixels, a color per track | plain title |

Coastlines, country and state borders are black and 0.8 points wide in these maps (gray lines are hard to see on the Tb colormaps); `--land_color`, `--border_linewidth`, `--state_alpha`, ... change each
feature on its own ([options.md](options.md)).

## Examples

```bash
cd config                                   # after bash demo_mcs_imerg.sh (it makes config_mcs_demo.yml and the tracking output)
# 2 panels, left + right, in parallel on 4 workers (a compute node), one PNG per hour
python ../visualization/plot_mcs_tracks.py -s 2019-01-25T00 -e 2019-01-27T00 -c config_mcs_demo.yml \
    --layout left_right -p 1 --workers 4 --figsize 10 8 --output $SCRATCH/quicklooks/

# a region, reading only that part of the pixel files, 1 panel, without the frames that have no MCS
python ../visualization/plot_mcs_tracks.py -s 2019-01-25T12 -e 2019-01-25T18 -c config_mcs_demo.yml \
    --panels 1 --extent -80 -35 -55 12 --subset 1 --skip_empty 1 --output $SCRATCH/quicklooks_1panel/

# Tb + radar reflectivity (WRF Tb + radar tracking). The data are on a Lambert conformal grid: show them in that projection
# (the parameters are STAND_LON, TRUELAT1, TRUELAT2 of the WRF files); without --extent the map is the domain of the data, no empty corners
python ../visualization/plot_mcs_tracks.py -s 2015-05-06T00 -e 2015-05-10T00 -c config_wrf_mcs_tbradar.yml \
    --product tbze --draw_state 1 --layout left_right -p 1 --figsize_x 12 \
    --projection LambertConformal --central_longitude -97 --standard_parallels 31 35 --output $SCRATCH/quicklooks_tbze/

# Tb only (Himawari)
python ../visualization/plot_mcs_tracks.py -s 2021-10-24T00 -e 2021-10-25T00 -c config_himawari.yml \
    --product tb -p 1 --figsize 10 10 --output $SCRATCH/quicklooks_tb/
```

New in the shared options (see [options.md](options.md)): `--projection` with `--central_longitude`, `--lat_range`, the track path / symbol / number
style and the halo. A global map of a Pacific-centered Robinson projection is in [global_mcs_animation.md](global_mcs_animation.md).

### Periodic domains and global grids: `--periodic_perimeter`

The MCS outline is the edge of each object. On a periodic domain (an idealized simulation with periodic boundaries, or a global grid
with the dateline at the edge of the array) an object that crosses the edge continues on the other side, so there should be no outline along the
edge of the array. `--periodic_perimeter x|y|both` wraps the array before the outlines are found; `auto` reads `pbc_direction` from the pixel-level
file. It applies to the whole domain (not with `--subset 1`). The default `none` is the old behavior.

## Animations

The wrapper scripts make the PNG files, the mp4 and, if you ask for it, a GIF in one go. Each has the parameters (dates, config, output folders,
products, map and track style, ffmpeg settings) as variables at the top of the file; `--set NAME=VALUE` overrides one without editing the file. The work is done in
[`mcs_animation.py`](../../pyflextrkr/visualization/mcs_animation.py) (`DEFAULTS` lists every parameter with its default):
`options_to_argv(...)` -> `mcs_cli.main(argv)` -> `animation.make_animation` (`make_mp4`, then `make_gif`).

| script | makes |
|---|---|
| [`visualization/examples/make_mcs_animation.py`](../../visualization/examples/make_mcs_animation.py) | all the parameters, with the defaults of the global Robinson map of [global_mcs_animation.md](global_mcs_animation.md) (2 days) |
| [`examples/make_mcs_animation_wrf_tbze.py`](../../visualization/examples/make_mcs_animation_wrf_tbze.py) | the WRF demo: Tb + radar reflectivity on the Lambert conformal grid of the data, 2 panels side by side |
| [`examples/make_mcs_animation_imerg_1panel.py`](../../visualization/examples/make_mcs_animation_imerg_1panel.py) | the IMERG demo over South America: Tb + precipitation on 1 panel |
| [`examples/make_mcs_animation_himawari_tb.py`](../../visualization/examples/make_mcs_animation_himawari_tb.py) | the Himawari demo: Tb only |
| [`examples/make_mcs_animation_imergv7_sahel.py`](../../visualization/examples/make_mcs_animation_imergv7_sahel.py) | the Sahel demo (IMERG V07B, 22-23 August 2025): Tb + precipitation on 2 panels, white track halo |

The demo wrappers read the config file that the demo script leaves in `config/` (`bash config/demo_mcs_imerg.sh`, ...); the animations of the three are in the
[README](../../visualization/README.md#more-mcs-looks). The video settings are those of the CONUS wrapper: `playback_fps = 2` (PNG files per second = the speed: hourly files, 2 hours of data per second,
2 days in 24 s), `video_quality = 24`, `output_width = 1920` (never enlarged), `video_fps = None` (automatic, 10 frames per second); `gif_width` (default `None`: mp4 only) makes a GIF from the mp4 (`--set gif_width=480 --set gif_duration=8` for README
GIFs of 1.6-2.3 MB). The settings, what they do, the test of size and quality and the ffmpeg commands for PNG files that already exist are in [animations.md](animations.md) (`pngs_to_animation.py` makes the mp4 from any PNG files).
Animations are about 2 days long: a longer run
is not needed to see how a style looks (to find a good time for an example, plot a week of frames with `-p 1 --workers 32` and look at them).

## Old commands

The five old scripts keep working through stubs in `Analysis/` that call `mcs_cli.main(legacy_name=...)`; the table with the new command of each is in
[inventory.md](inventory.md). Code map: [structure.md](structure.md).
