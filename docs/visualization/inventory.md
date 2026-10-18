# Inventory: old scripts -> new scripts, what changed, how it was checked

The 12 plotting scripts `Analysis/plot_subset_*.py` (9,239 lines, copies of one skeleton) and the two animation wrappers were re-organized into
four commands in [`visualization/`](../../visualization) and the shared code in [`pyflextrkr/visualization/`](../../pyflextrkr/visualization)
([structure.md](structure.md)). **The old file names still work**: each is now a stub of a few lines that calls the new code with the defaults of
the old script, so existing commands, batch scripts and notebooks keep making the same figures.

- [Old script -> new command](#old-script---new-command)
- [Other files that moved](#other-files-that-moved)
- [What is new](#what-is-new)
- [What differs from the old scripts](#what-differs-from-the-old-scripts)
- [How it was checked](#how-it-was-checked)
- [Not checked with real data](#not-checked-with-real-data)

## Old script -> new command

| old script (`Analysis/`) | new command (`visualization/`) | defaults restored by the stub (`legacy.py`) |
|---|---|---|
| `plot_subset_tbpf_mcs_tracks_demo.py` | `plot_mcs_tracks.py` (`--product tbpf --panels 2`) | |
| `plot_subset_tbpf_mcs_tracks_1panel_demo.py` | `plot_mcs_tracks.py --panels 1 --skip_empty 1` | skips the frames without MCS |
| `plot_subset_tbze_mcs_tracks_demo.py` | `plot_mcs_tracks.py --product tbze --dpi 300 --draw_state 1` | dpi 300, state borders |
| `plot_subset_tbze_mcs_tracks_1panel_demo.py` | `plot_mcs_tracks.py --product tbze --panels 1 --dpi 300 --draw_state 1` | dpi 300, state borders |
| `plot_subset_tb_mcs_tracks_demo.py` | `plot_mcs_tracks.py --product tb --dpi 300 --fontsize 13 --figsize 10 10 --skip_empty 1 --track_smooth 0` | Tb only, raw track center |
| `plot_subset_cell_tracks_demo.py` | `plot_cell_tracks.py` (`--product refl`) | |
| `plot_subset_cell_tracks_ETH_demo.py` | `plot_cell_tracks.py --product eth --skip_empty 1` | echo-top height, skips the frames without cells |
| `plot_subset_generic_tracks_demo.py` | `plot_generic_tracks.py` | workers from `nprocesses` of the config |
| `plot_subset_psi_tracks_demo.py` | `plot_generic_tracks.py --cmap RdBu_r --levels -3e7 3e7 0.5e6 --cbticks -3e7 3e7 1e7 --cblabel '600hPa PSI Anomaly' --fontsize 11 --map_resolution 110m --grid_linewidth 0 --track_marker_size 6 --draw_border 0` | track number from the mask, symbols on, time label without seconds |
| `plot_subset_tbpf_demo_pbc.py` | `plot_cartesian_tracks.py --product tbpf` | |
| `plot_subset_tbpf_tracks_pinacles_pbc.py` | `plot_cartesian_tracks.py --product tbpf --xscale 1e-3 --perim_plot pcolormesh --show_pf_markers 0` | PINACLES in meters |
| `plot_subset_coldpool_tracks_nomap.py` | `plot_cartesian_tracks.py --product field --field_varname cp_intensity` | |

The stubs of the MCS, cell and generic maps also restore `--map_aspect auto` (the old stretched map; the new commands keep the true shape). An option on the command line replaces the default of the old script. The stub prints one line to stderr (`NOTE: Analysis/... has moved to visualization/...`).
The 20 `config/demo_*.sh` scripts and the registry of `tests/run_demo_tests.py` call the new commands.

## Other files that moved

| old | new |
|---|---|
| `Analysis/make_mcs_animation.py`, `make_cell_animation.py` (called the plot script as a subprocess) | `visualization/make_mcs_animation.py`, `make_cell_animation.py` (call `main(argv)`, `--set NAME=VALUE`); the MCS wrapper defaults to the global Robinson map ([global_mcs_animation.md](global_mcs_animation.md)) |
| `Analysis/plot_*quicklooks*.sh`, `slurm.submit_mcs_tbze_quicklook.sh`, `tasklist_mcs_tbze_quicklook*.txt`, `make_mcs_quicklook_joblib.py`, `make_gridrad_mcstracks_animation.py` | `visualization/examples/` (site-specific batch examples; they call the new scripts; the `${extent}` typo of the shell examples is fixed) |
| `Analysis/plot_cell_mask_quicklook.py`, `plot_cell_tracking_quicklook.py`, `plot_csapr_tracking_quicklook.py`, `plot_csapr_tracks_demo.py`, `plot_saag_tbpf_mcs_tracks_demo.py` (2023, superseded, referenced nowhere) | `Analysis/ToBeDeleted/` |

## What is new

- Map projection by name (`--projection`) with `--central_longitude`, `--central_latitude`, `--standard_parallels`, a latitude-band crop (`--lat_range`),
  gridline ticks / width / alpha, and a box on the map: [options.md](options.md#map-projection-central-longitude-latitude-band).
- Track style options for the path, initiation symbol, number and a **halo with its own alpha**: [options.md](options.md#track-style-path-symbol-number-halo).
- One script for the five MCS variants, the generic field options on the command line, `--track_lookback_days`, `--skip_empty`, `--periodic_perimeter`,
  `--extent` as four numbers or one quoted string.
- The map keeps its true shape and the colorbars follow it (`--map_aspect equal`, the default; the figure size follows the shape of the map), and without `--extent`
  the map is the domain of the data in the coordinates of the projection (WRF data on a Lambert conformal grid shown with that projection).
- `--show_track_numbers` per panel (`1 0`: on panel (a) only), latitude labels on the curved edge of whole-world maps.
- The global MCS animation ([global_mcs_animation.md](global_mcs_animation.md)), and animations of three more kinds of MCS maps with a wrapper script each
  (WRF Tb + radar, IMERG 1 panel, Himawari Tb only: [visualization/examples](../../visualization/examples), GIFs in the [README](../../visualization/README.md#more-mcs-looks)).
- `--layout top_bottom|left_right` for 2 panels, `--cbar_thickness` and `--cbar_spacing` for the colorbars of the MCS maps, `--show_pfdiam 0` also takes the PF circles out of the legend.
- The convective cell maps have the functions of the cell part of `plotmap.py` (conus_tracking): terrain height, one outline color per track, a ring around the outlines, only the cells of tracks that live long enough, the line style of the gridlines, cities and other radars ([cell_maps.md](cell_maps.md#options-of-this-family)).
- Tests: `tests/test_viz_*.py` ([testing.md](testing.md)).

## What differs from the old scripts

Everything not listed here is the same figure (checked pixel by pixel, below).

| change | effect |
|---|---|
| `-o horizontal` / `-o vertical` (`--orientation`) is now `--layout top_bottom` / `left_right` | the old names were easy to mix up (horizontal = panels above each other, vertical = side by side). `-o` and `--orientation` are still accepted, also with the old values, which print a deprecation warning and mean the same as before (the `Analysis/` stubs, notebooks and shell scripts with `-o vertical` keep working); the wrapper parameter `orientation` is now `layout` (`--set orientation=vertical` still works, with the warning) |
| the text of the colorbars stays inside the figure (new commands only, `--map_aspect equal`) | in the old scripts, and in the stubs (`--map_aspect auto`), the label of the second colorbar of a 1-panel map could be cut at the edge of the figure and the text of the two colorbars could overlap; now a colorbar moves away from the text of the previous one, and the maps shrink a little if the text would end outside the figure |
| a figure for a map much wider than high (no `--figsize`, new commands) | the margins above and below the maps are in inches (the titles and the longitude labels need a fixed room) instead of fractions of a short figure, so the figure is not made taller than the maps need |
| the new commands keep the true shape of the map (`--map_aspect equal`), and the colorbars follow the map | the old scripts stretched the map to its panel (a South America map in a 10 x 8 inch figure looked squeezed). The stubs in `Analysis/` keep `--map_aspect auto`, so the old commands make the old pictures; the new demo commands have the true shape |
| without `--figsize` the figure size follows the shape of the map | the old scripts used a formula (width x lat span / lon span, x 2.1 for 2 panels) or a fixed size |
| the tbze demos (`demo_mcs_gridrad.sh`, `demo_mcs_wrf_tbradar.sh`) | `--draw_state 1` (the old tbze scripts drew the state borders by default; the new command does not unless asked); the WRF demo shows the data on their Lambert conformal grid (`--projection LambertConformal --central_longitude -97 --standard_parallels 31 35`) |
| `--subset 1` with an extent that covers all longitudes (`0 360`, `-180 180`) | the old extent normalization made the box a single meridian (one column of the data); now it is the whole width |
| `--time_format` is read by all scripts | the old tbze (2 panels), generic, PBC, PINACLES and cold pool scripts ignored it (always `yyyymodd_hhmmss`) |
| the PNG is written once (`fig.savefig`) | the old cell / tbze / generic / cold pool scripts wrote it twice (`canvas.print_png` and `savefig`) |
| `--subset 1` of the 1-panel, Tb-only and x / y scripts cuts an index window with one pixel of margin | the old scripts cut with `where(drop=True)` or without margin: 0.001-0.02 % of the pixels differ, in the edge column(s) of the map (see below). The 2-panel MCS, cell and generic maps were already like this |
| a period without any track starting in it | no longer stops in xarray (an empty index with `nmaxpf=0` fails in the lazy indexing); the frames are made without tracks |
| the MCS outline (`label_perimeter`) works on the bounding box of each object | same result (tested against the loop over the whole grid); one global 10-km frame (1200 x 3600, 89 MCS) takes 0.02 s instead of 2.0 s |
| the tracks are sent to the Dask workers once (`scatter`) | instead of with every frame; same figures |
| `--lat_range` with a projection whose y is not a function of the latitude only (conic, polar, orthographic) | an error with the reason, instead of a map cut at the wrong place |
| shell examples and demos | the `${extent}` typo, an uncommented prose line in `demo_mcs_gridrad.sh` and `demo_mcs_pinacles_tracking.sh`, absolute paths of another user in `demo_coldpool_tracking.sh` |
| the range rings and azimuth lines of the cell maps, the PF diameter circles of the MCS maps | geodesic lines drawn with the transform of the data (`mapcore.geodesic_circle`, `geodesic_ray`) instead of `ax.tissot` and `transform=ccrs.Geodetic()`. With PROJ 9.8 (`pyflex26.3`) the old circles were 20 km (0.18 degree of latitude) from the radar, with PROJ 9.6 they were right; `cell_refl.png`, `mcs_tbpf_2panel.png` and the WRF animation changed only at the circles and lines (CACTI frames, old and new code, pixel by pixel: identical without the circles) |
| `Analysis/make_*_animation.py` | call the plot code directly (no subprocess); the video settings follow the CONUS wrapper: `input_framerate` -> `playback_fps`, `output_framerate` -> `video_fps` (automatic), the default CRF 20 -> 24 (23-33 % smaller files), a width that is never enlarged, `frame_stride`, GIF settings ([animations.md](animations.md)) |

`load_config(config_file, make_dirs=True)` has a new keyword: the plot commands read the config with `make_dirs=False`, so reading a config no longer creates the folders
`tracking/`, `stats/` and `pixel...` below `root_path` (the tracking pipeline still creates them). For tracking outputs in a read-only folder (CFS) a config file with
the paths of the outputs is enough: [config/config_imerg_global_mcs_v3_2025.yml](../../config/config_imerg_global_mcs_v3_2025.yml) for the global V3 maps.

## How it was checked

A harness ran the **frozen old scripts**, the **stub** that has the old name and (where there is one) the **new command** with the equivalent new-style
options on the same inputs, and compared every PNG pixel by pixel. Demo data: IMERG Tb + precipitation over South America (regional) and global (2 days),
WRF Tb + radar, Himawari Tb, CSAPR cells, ERA5 Z500 blocking, idealized periodic domain.

| case | what it covers | result |
|---|---|---|
| `tbpf2_vert_imerg`, `tbpf2_horiz_ext_subset`, `tbpf2_horiz_ext_nosubset` | 2 panels, both layouts, extent with and without `--subset` | identical |
| `tbpf2_global_dateline`, `..._subset`, `tbpf2_global_full` | extent `140 -140 -20 20` across the dateline, global map | identical |
| `tbpf2_figname_seq` | `--figname_type sequence`, borders, states, rivers, title prefix | identical |
| `tbpf1_imerg`, `tbpf1_ext_nosubset` | 1 panel | identical |
| `tbpf1_ext_subset` | 1 panel, `--subset 1` | 42-45 pixels of 3.2 million differ (edge column) |
| `tbze2_wrf`, `tbze2_wrf_vert_ext`, `tbze2_wrf_lowlevel` | tbze 2 panels, composite and low-level reflectivity | identical |
| `tbze1_wrf`, `tbze1_wrf_ext_subset` | tbze 1 panel | identical; with `--subset 1` 500-700 pixels (0.01 %) differ |
| `tb_himawari`, `tb_himawari_ext` | Tb only | identical; with `--subset 1` 1,100-1,900 pixels (0.01-0.02 %) differ |
| `cell_csapr`, `cell_csapr_ext_subset`, `cell_csapr_opts`, `cell_eth_csapr` | cells, extent and subset, range rings / azimuths / lookback options, echo-top height | identical |
| `generic_z500`, `generic_z500_extent` | generic features | identical |
| `psi_on_z500` | the PSI options (colormap, levels, track number from the mask) on the Z500 data | identical |
| `pbc_idealized`, `pbc_idealized_subset` | x / y domain with periodic tracks | identical; with `--subset 1` at most 411 pixels (0.007 %) differ |
| `pinacles_on_pbc`, `coldpool_on_pbc_cp` | the PINACLES and cold pool options on the idealized data | identical |

The new commands (`direct`, with `--map_aspect auto`, the old look) gave the same pictures as the old scripts wherever they were run. Besides the pixel comparison:
`tests/test_viz_*.py` (unit tests and synthetic frames of each family, [testing.md](testing.md)); the plot call of every `config/demo_*.sh` was parsed by the real command line, and the 6 demos with
test data were run; the scripts of `visualization/` and the stubs have static imports only (a test).

## Not checked with real data

- **PSI** (`plot_subset_psi_tracks_demo.py`), **PINACLES** and **cold pool** inputs were not available (purged or not in the data portal): their commands were checked with the equivalent options on the Z500 and idealized periodic demo data (cases above), not on their own data.
- The Dask path (`-p 1`) of every family ran on a compute node in the harness; the unit test compares it with the serial run for the MCS maps.
