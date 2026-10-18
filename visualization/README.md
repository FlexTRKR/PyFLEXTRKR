# Visualization

Scripts that plot PyFLEXTRKR tracking results on maps: MCS (Tb with precipitation, with radar reflectivity, or Tb only), convective cells,
generic features (Z500 anomalies, vorticity, ...) and idealized x / y domains. One PNG per time, and animations (mp4, GIF). Each script is a thin
command line; the code they share is in [`../pyflextrkr/visualization`](../pyflextrkr/visualization) ([how it fits together](../docs/visualization/structure.md)).

Run them with the PyFLEXTRKR environment (`pip install -e .` in the repository; at NERSC `/global/common/software/m1867/python/pyflex26.3`) from any
directory. Parallel plotting (`-p 1`, `--workers N`) belongs on a compute node. The old `Analysis/plot_subset_*.py` commands still work (they call the same
code, with their old defaults): [old script -> new command](../docs/visualization/inventory.md).

## Catalog

| set | scripts | produces | example | details |
|---|---|---|---|---|
| **MCS maps** | `plot_mcs_tracks.py`, `make_mcs_animation.py` | Tb with the tracked MCS (outline, path, number, PF circle) and a second field: precipitation (`--product tbpf`), radar reflectivity (`tbze`) or none (`tb`), on 1 or 2 panels; mp4 | ![](../docs/figures/visualization/mcs_tbpf_2panel.png) | [mcs_maps.md](../docs/visualization/mcs_maps.md) |
| **Global MCS animation** | `make_mcs_animation.py` (its defaults) | the global MCS tracking on a Robinson map centered on the Pacific, 60S-60N | ![](../docs/figures/visualization/global_robinson.png) | [global_mcs_animation.md](../docs/visualization/global_mcs_animation.md) |
| **Convective cells** | `plot_cell_tracks.py`, `make_cell_animation.py` | composite reflectivity or echo-top height with the cell outlines, paths (colored by lifetime), numbers, radar range rings and azimuth lines; optionally the terrain, one outline color per track, only long-lived cells, cities and other radars; mp4 | ![](../docs/figures/visualization/cell_refl.png) | [cell_maps.md](../docs/visualization/cell_maps.md) |
| **Generic features** | `plot_generic_tracks.py` | any 2-D field (Z500 anomaly, vorticity, stream function) with the tracked features | ![](../docs/figures/visualization/generic_z500.png) | [generic_maps.md](../docs/visualization/generic_maps.md) |
| **x / y domains** | `plot_cartesian_tracks.py` | idealized simulations and cold pools without a map, periodic boundaries | ![](../docs/figures/visualization/cartesian_pbc.png) | [cartesian_maps.md](../docs/visualization/cartesian_maps.md) |
| **Batch examples** | `examples/` | shell, slurm and joblib examples that make many quicklooks | | [examples/](examples) |

## More MCS looks

Three more kinds of MCS maps, each as an animation with a wrapper script in [`examples/`](examples) (the animations are made from the demo outputs, `bash config/demo_*.sh`
makes the tracking and the config file that the wrapper reads). The parameters are variables at the top of each wrapper; `--set NAME=VALUE` changes one
([how the wrappers work](../docs/visualization/mcs_maps.md#animations); the video settings and the GIF sizes (each GIF here is 1.6-2.3 MB: 16 frames) are tested in [animations.md](../docs/visualization/animations.md#gif-for-the-readme)).

**Tb + radar reflectivity of a WRF run**, on the Lambert conformal grid of the data (`--product tbze --projection LambertConformal ...`: the map is the domain of the data, no `--extent` needed), 2 panels side by side (`--layout left_right`):
[`examples/make_mcs_animation_wrf_tbze.py`](examples/make_mcs_animation_wrf_tbze.py)

![WRF Tb and radar reflectivity](../docs/figures/visualization/mcs_wrf_tbze.gif)

**Tb + precipitation on 1 panel** (IMERG over South America; `--panels 1`: the figure size follows the shape of the map, no `--figsize` needed, the colorbars have the height of the map):
[`examples/make_mcs_animation_imerg_1panel.py`](examples/make_mcs_animation_imerg_1panel.py)

![IMERG 1 panel](../docs/figures/visualization/mcs_imerg_1panel.gif)

**Tb only** (Himawari; `--product tb`): [`examples/make_mcs_animation_himawari_tb.py`](examples/make_mcs_animation_himawari_tb.py)

![Himawari Tb only](../docs/figures/visualization/mcs_himawari_tb.gif)

## Options that all scripts share

Full table: [options.md](../docs/visualization/options.md). New in these scripts (the defaults keep the look of the old ones):

| | options |
|---|---|
| **map projection** | `--projection` (PlateCarree, Robinson, Mollweide, EqualEarth, Mercator, LambertConformal, AlbersEqualArea, Orthographic, North/SouthPolarStereo), `--central_longitude`, `--central_latitude`, `--standard_parallels`, `--lat_range LAT_MIN LAT_MAX` (crop to a latitude band), `--extent` (default: the domain of the data), `--subset`, `--lon_ticks`, `--lat_ticks`, `--grid_linewidth`, `--grid_alpha`, `--box` |
| **shape of the map** | `--map_aspect equal` (default: the map keeps its true shape, the colorbars follow it; the figure size follows the map unless `--figsize` is given) or `auto` (stretched to its panel, the old scripts) |
| **panels and colorbars** (MCS maps) | `--layout top_bottom` (default) or `left_right` for 2 panels (the old `-o horizontal` / `vertical`: still work, with a deprecation warning), `--cbar_thickness` (how thin the colorbars are), `--cbar_spacing` (gap between the 2 colorbars of a 1-panel map); the text of the colorbars stays inside the figure |
| **track path** | `--track_color`, `--track_lw`, `--track_ls`, `--track_alpha` |
| **initiation symbol** | `--track_marker`, `--track_marker_size`, `--track_marker_color` |
| **track number** | `--show_track_numbers` (two values for the 2 panels: `1 0` = on panel (a) only), `--track_text_size`, `--track_text_color`, `--track_text_weight`, `--track_text_alpha` |
| **halo** (path, symbol, number) | `--track_halo_color` (not set: no halo), `--track_halo_width`, `--track_halo_alpha` |
| **frames and tracks** | `--track_lookback_days`, `--skip_empty`, `--figsize`, `--figsize_x`, `--dpi`, `--fontsize` (all text except the track numbers), `--time_format`, `--trackstats_file`, `--pixel_path` |

![projections](../docs/figures/visualization/projections.png)
The track style on the Sahel (22 August 2025, 20:30 UTC; the commands are in [options.md](../docs/visualization/options.md#track-style-path-symbol-number-halo)):

![track style options](../docs/figures/visualization/track_style_options.png)

## Quick start

After a demo (`bash config/demo_mcs_imerg.sh` runs the tracking and makes the quicklooks; the plot command is its last step), or for any tracking output with its config:

```bash
cd <repository>/config                      # the demo scripts run from here; a config path from anywhere works

# MCS: Tb + precipitation, 2 panels (left + right), one PNG per hour, on 4 workers (compute node)
python ../visualization/plot_mcs_tracks.py -s 2019-01-25T00 -e 2019-01-27T00 -c config_mcs_demo.yml \
    --layout left_right -p 1 --workers 4 --figsize 10 8 --output $SCRATCH/quicklooks/

# the same with a Lambert conformal map, a white halo around the tracks and the numbers, bigger numbers
python ../visualization/plot_mcs_tracks.py -s 2019-01-25T12 -e 2019-01-25T14 -c config_mcs_demo.yml --panels 1 \
    --projection LambertConformal --central_longitude -60 --standard_parallels -40 -10 --extent -80 -35 -55 12 \
    --track_halo_color white --track_halo_alpha 0.7 --track_text_size 14 --output $SCRATCH/quicklooks_lcc/

# Tb + radar reflectivity of a WRF run: the data are on a Lambert conformal grid, so the map is too (STAND_LON, TRUELAT1/2 of the WRF files)
python ../visualization/plot_mcs_tracks.py -s 2015-05-07T00 -e 2015-05-07T06 -c config_wrf_mcs_tbradar.yml --product tbze --draw_state 1 \
    --layout left_right --figsize_x 12 --projection LambertConformal --central_longitude -97 --standard_parallels 31 35 --output $SCRATCH/quicklooks_tbze/

# convective cells, generic features, an x / y domain
python ../visualization/plot_cell_tracks.py -s 2019-01-25T12 -e 2019-01-26T00 -c config_csapr_demo.yml --radar_lat -32.12641 --radar_lon -64.72837 -p 1 --output $SCRATCH/quicklooks_cell/
python ../visualization/plot_generic_tracks.py -s 1979-06-01 -e 1979-08-31 -c config_z500_demo.yml --figsize 8 4 -p 1 --output $SCRATCH/quicklooks_z500/
python ../visualization/plot_cartesian_tracks.py -s 2020-01-01T00 -e 2020-01-02T00 -c config_demo_pbc.yml -p 1 --extent "0. 1500. 0. 1500." --output $SCRATCH/quicklooks_pbc/

# an animation (the parameters are the variables at the top of the file, or --set NAME=VALUE)
python make_mcs_animation.py --set start_date=2025-01-01T00 --set end_date=2025-01-03T00 --set n_workers=32
python examples/make_mcs_animation_imerg_1panel.py --set n_workers=32      # after bash config/demo_mcs_imerg.sh
python make_cell_animation.py --set start_date=2020-06-14T00:00 --set end_date=2020-06-14T06:00

# an mp4 (and a GIF) from PNG files that are already plotted (a directory, a quoted pattern or a list of files); --print_command shows the ffmpeg commands
python pngs_to_animation.py 'frames/mcs_tbpf_*.png' movie.mp4 --playback_fps 2 --gif_width 480 --gif_duration 8
```

Without `--output` the figures go to `quicklooks_trackpaths/` in the pixel-level directory of the tracking run.
`python <script> --help` lists the options of a script.

## Layout of this folder

| file | |
|---|---|
| `plot_mcs_tracks.py` | MCS track maps (`--product tbpf|tbze|tb`, `--panels 1|2`, `--layout top_bottom|left_right`) |
| `plot_cell_tracks.py` | convective cell track maps (`--product refl|eth`) |
| `plot_generic_tracks.py` | generic feature maps of any 2-D field |
| `plot_cartesian_tracks.py` | x / y domain maps (`--product tbpf|field`) |
| `make_mcs_animation.py`, `make_cell_animation.py` | wrappers: parameter block -> the plot command -> mp4 (`--set NAME=VALUE`); `make_mcs_animation.py` makes the global Robinson animation (and a GIF if you set `gif_width`) |
| `pngs_to_animation.py` | PNG files that are already plotted -> mp4 (and GIF) with the video settings of the wrappers (`playback_fps`, `video_quality`, ...); `--print_command` prints the plain ffmpeg commands |
| `examples/` | `make_mcs_animation_wrf_tbze.py`, `make_mcs_animation_imerg_1panel.py`, `make_mcs_animation_himawari_tb.py` (the animations above); batch examples: `plot_*quicklooks*.sh`, `slurm.submit_mcs_tbze_quicklook.sh` with the task lists, `make_mcs_quicklook_joblib.py`, `make_gridrad_mcstracks_animation.py`; `compare_x264_presets.py` and `compare_gif_settings.py` (size and quality of the mp4 for each x264 preset / CRF, and of the GIF for each palette / dither, on your frames) |

The documentation is in [`../docs/visualization`](../docs/visualization): [structure.md](../docs/visualization/structure.md) (call chains, where to change what),
[options.md](../docs/visualization/options.md), one page per family, [global_mcs_animation.md](../docs/visualization/global_mcs_animation.md),
[animations.md](../docs/visualization/animations.md) (playback speed and video settings: tests of CRF, file frame rate, x264 preset and the GIF size), [inventory.md](../docs/visualization/inventory.md) (old scripts -> new, what changed, how it was checked), [testing.md](../docs/visualization/testing.md).
