# Testing the visualization code

Three levels: unit tests with synthetic data (in `tests/`, run in seconds, no data download), a comparison with the old scripts on the demo data (done
for the re-organization, results in [inventory.md](inventory.md#how-it-was-checked)), and the demo scripts.

## Unit tests

```bash
# python with pytest, matplotlib and cartopy (e.g. the pyflex-dev environment); the full suite (613 tests, 89 skipped without data) takes about 4 minutes, the visualization tests (about 190) 1 minute
python -m pytest tests/test_viz_*.py
python -m pytest tests/ -m "not local and not demo"
```

| file | what it checks |
|---|---|
| `test_viz_mapcore.py` | projections by name and their parameters, the dateline set-up of PlateCarree, `set_map_domain` (the latitude-band crop of a global Robinson map, polar cap, orthographic, Mercator, the poles where the transform is NaN), the error for `--lat_range` on a projection where it would cut the map at the wrong place, the domain of data on a Lambert conformal grid in that projection (`data_extent_xy`, `auto_map_domain`), colorbars that follow a map that keeps its shape (`fit_colorbars_to_maps`), the latitude labels on the curved edge of a world map (60S ... 60N in a tall figure, the title above the 60N label), figure size from the domain, longitude helpers, the geodesic circles and lines (`geodesic_circle`: radius within 1 m on the ellipsoid, closed, centered, continuous at the dateline; `geodesic_ray`: azimuth and length), and no `tissot` or `Geodetic` call anywhere in the package (syntax tree) |
| `test_viz_masks.py` | `label_perimeter` (bounding boxes) equals the loop over the whole grid of the old scripts for random objects, perimeter widths, objects at the edges; the periodic wrap in x, y and both; object centers |
| `test_viz_trackdraw.py` | the halo: drawn for a path, a symbol and a character; `alpha` makes it lighter, alpha 0 leaves none, the width is outside the edge; the colour levels of the track numbers |
| `test_viz_trackstats.py` | the tracks of a period for the four families (start / end selection, smoothed center, no PF variables, empty period, x / y scaling); the index window of an extent (regional, dateline, 0-360 and -180..180 data, all longitudes, the whole grid is no subset); figure names |
| `test_viz_cli.py` | `--extent` as four numbers and as one string, `--layout` and its old names `-o` / `--orientation` (horizontal / vertical, with the deprecation warning), `options_to_argv`, the defaults of all 12 old scripts (`legacy.py`) are real options and are what the stub gets, the stubs in `Analysis/` call the right family with their own name, the scripts of `visualization/` and the stubs use static imports only (AST: no `sys.path`, `subprocess`, `importlib`, every imported name exists), every module imports |
| `test_viz_colorbars.py` | the GridSpec numbers of the MCS figures (`figure_layout`): the defaults are the layout of the old scripts, `--cbar_thickness` / `--cbar_spacing` are fractions of the map, margins in inches for a short figure; `keep_colorbar_text_inside`: the text of colorbars at the right and below stays inside the figure, a colorbar moves away from the text of the previous one (and keeps a gap that was asked for), nothing moves when the text fits |
| `test_viz_animation.py` | `frames_in_range` (inclusive, sorted, other files skipped), `list_png_files` (directory, pattern, list), the automatic file frame rate (`video_frame_rate`: 2 -> 10, 3 -> 12), the scale filter (never enlarged, even sizes), the `make_mp4` command (input and file rates, CRF, preset only when given, every Nth file, the list file removed), `make_mp4` with ffmpeg (odd sizes, never enlarged, repeats, stride, directory and pattern input), `make_animation` (settings printed, GIF only with `gif_width`), the command line of `pngs_to_animation.py` (the plain ffmpeg commands with `--print_command`, an mp4 and a GIF from a directory), the palette / dither / colors / part-of-the-video options of `make_gif` and a GIF made with ffmpeg |
| `test_viz_wrappers.py` | the animation wrappers (`make_mcs_animation.py`, the three `examples/make_mcs_animation_*.py`, `make_cell_animation.py`): the parameter block (and `--set NAME=VALUE`) gives the arguments of the plot command, and the real command line parses them (global Robinson defaults, the maps of the examples, 1 panel / Tb only / reflectivity, `layout`, the old `orientation` with a deprecation warning), the video settings of every wrapper (`playback_fps` 2, CRF 24, 1920 pixels; 12 for the 5-minute cell frames) and the old `input_framerate` / `output_framerate` as `playback_fps` / `video_fps` with a deprecation message, the mp4 and GIF steps get the right settings (ffmpeg replaced), every variable of a wrapper is a parameter, the global config file points to the outputs; `load_config(make_dirs=False)` makes no folders |
| `test_viz_frames.py` | whole runs of `main(argv)` on synthetic frames: all MCS products and layouts, the cell, generic and x / y families, 9 projections, the latitude band, every track style option changes the picture as intended (halo with alpha is lighter than the opaque halo, color, marker, numbers, ...), `--show_track_numbers 1 0` (panel (a) unchanged, panel (b) without numbers), the map keeps its shape and the colorbars have the extent of the map (and `--map_aspect auto` moves nothing), the figure size follows the shape of the map, the domain of the data in the projection, the range rings and azimuth lines of the cell maps (the radius around the radar, drawn with the transform of the data, on a PlateCarree and a Lambert map), `--track_lookback_days`, `--skip_empty`, a Dask run gives the same pictures as the serial run, an old command through its stub gives the pictures of the new command with the options of the old script (`--map_aspect auto`) |

`tests/viz_synthetic.py` makes the inputs: four hourly pixel-level files on a 60 x 40 grid (100-130E, 10S-10N) with two tracks (3 and 2 frames, the fourth frame
without an object), their statistics files (MCS and cells), and the config dictionaries that the command lines read (the tests replace `load_config`, so no
config file or tracking run is needed). Coastlines are switched off (`--draw_land 0`): nothing is downloaded.

The CI environment installs neither matplotlib nor cartopy, so the tests that need them are skipped there (`pytest.importorskip`); `test_viz_masks.py`,
`test_viz_trackstats.py` and the frame-filter part of `test_viz_animation.py` run everywhere.

## Comparison with the old scripts

For the re-organization the old scripts were frozen (`git archive` of the commit before the change) and run next to the new code on the same inputs
(IMERG over South America and global, WRF Tb + radar, Himawari, CSAPR cells, ERA5 Z500, the idealized periodic domain), 28 cases, every PNG compared pixel by pixel.
24 cases are identical, 4 (`--subset 1` of the 1-panel, Tb-only and periodic-domain maps) differ in 0.001-0.02 % of the pixels at the edge of the map
([inventory.md](inventory.md#how-it-was-checked)); the same result after the later changes (`--layout`, the colorbar options and the colorbar text fit, which only the new commands use). The harness is a scratch tool, not part of the repository; to repeat a comparison on one case, run the old script
(`git show <commit>:Analysis/plot_subset_...py`) and the stub with the same arguments and compare the PNG files (`np.array_equal` of the images).

## Demo scripts

`config/demo_*.sh` make the quicklooks with the new scripts (the plot lines of all 20 scripts were parsed by the real command lines, and those of the 6 demos with test data
(IMERG, WRF Tb + radar, Himawari, CSAPR cells, ERA5 Z500, idealized periodic domain) were run), and `tests/run_demo_tests.py` has the new script names in its
registry (its 11 plot entries were parsed, 6 of them run on the test data; the mp4 step needs an ffmpeg with libx264) (`python tests/run_demo_tests.py --demos demo_mcs_imerg --with-plots -n 4`, or `--make-plots-only` after a tracking run). `bash -n config/demo_*.sh` checks the syntax.
