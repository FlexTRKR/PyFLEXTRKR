# Global MCS animation (Robinson map, 60S-60N)

Hourly maps of the global MCS tracking (GPM Tb + IMERG, 10 km) on a Robinson projection centered on the Pacific (central longitude 225E), showing only
the 60S-60N band that the data cover: (a) IR brightness temperature with the track numbers, (b) precipitation with the tracked MCS shaded (no numbers).
On a global map the clouds matter more than the tracks, so the tracks are light: thin paths, small initiation symbols, small faint track numbers (they are there to show
that a track continues), no halo and no PF diameter circles (too small to see at this scale).

![global MCS map](../figures/visualization/global_robinson.png)

It is the 2-panel `tbpf` map of [mcs_maps.md](mcs_maps.md) with the map options of [options.md](options.md): `--projection Robinson --central_longitude 225
--lat_range -60 60`, the track style options (`--track_marker_size 4 --track_text_size 7 --track_text_alpha 0.6`, no `--track_halo_color`), `--show_pfdiam 0`,
`--show_track_numbers 1 0` (numbers on panel (a) only), a larger `--fontsize` and `--periodic_perimeter x`.
[`visualization/make_mcs_animation.py`](../../visualization/make_mcs_animation.py) has these settings as its defaults.

- [Data](#data)
- [The config file](#the-config-file)
- [A few frames to check the look](#a-few-frames-to-check-the-look)
- [The animation](#the-animation)
- [How the global map is made](#how-the-global-map-is-made)

## Data

Global MCS tracking V3, 2025 (read-only, on CFS): `/global/cfs/cdirs/wcm_shr/gsharing/MCS_Global/V3/Feng21/`

| | |
|---|---|
| pixel-level files | `mcstracking_f21/20250101.0000_20251001.0000/mcstrack_YYYYMMDD_HHMMSS.nc` (hourly at :30, 1200 x 3600, 10 km: `tb`, `precipitation`, `cloudtracknumber`, `longitude`, `latitude`) |
| track statistics | `stats_f21/mcs_tracks_final_20250101.0000_20251001.0000.nc` |
| config of the run | `config_imerg_mcs_tbpf.yml` (points to another user's scratch and to the dates of 2024: the copy in this repository, [`config/config_imerg_global_mcs_v3_2025.yml`](../../config/config_imerg_global_mcs_v3_2025.yml), points to these files) |

The track positions in the statistics are not wrapped at the dateline (longitudes from -195 to 216), and the precipitation feature centers reach +-545
degrees; the plot normalizes them for the Robinson projection.

**Track numbers.** The number of a track is its index in the statistics file, so it grows with time from the start of the run on 1 January: about 100 new
tracks start per day (100 by 1 January 12 UTC, 276 by 3 January 00 UTC, 1,500 by 15 January, 28,157 in the 9 months of the file). The first days have
numbers of 1-3 digits, which keeps the labels small on a global map (the default dates of the wrapper are 1-2 January 2025). Later in the year the numbers have 4-5 digits: use a smaller
`--track_text_size`, or `--show_track_numbers 0 0` / `--set show_track_numbers=[false,false]`.

## The config file

[`config/config_imerg_global_mcs_v3_2025.yml`](../../config/config_imerg_global_mcs_v3_2025.yml) is the config of the V3 run with the paths of the finished outputs: `root_path` is the
CFS folder above, `stats_path_name: stats_f21`, `pixel_path_name: mcstracking_f21` and the dates of the 2025 outputs (it differs from `config_imerg_mcs_tbpf.yml` of that folder in `startdate`,
`enddate`, `root_path`, `stats_path_name` and `pixel_path_name`). The plot scripts read the track statistics file and the pixel-level files from it, so no `--trackstats_file` or `--pixel_path`
is needed. Reading a config no longer creates the folders `tracking/`, `stats/`, `pixel...` below `root_path` (`load_config(config_file, make_dirs=False)` in the plot scripts), which would fail in the
read-only CFS folder. `pixel_radius` 10 km sets the width of the MCS outline. To run the tracking itself, change `root_path` to a folder you can write to.

## A few frames to check the look

On a compute node (the figure of one frame is 4.3 million cells per mesh through the projection: about 30-40 s of one core per frame):

```bash
python visualization/plot_mcs_tracks.py -s 2025-01-02T12 -e 2025-01-02T13 -c config/config_imerg_global_mcs_v3_2025.yml \
    --product tbpf --panels 2 --layout top_bottom --figsize_x 14 \
    --projection Robinson --central_longitude 225 --lat_range -60 60 --map_resolution 110m --land_linewidth 0.8 \
    --fontsize 14 --show_track_numbers 1 0 --track_text_size 7 --track_text_alpha 0.6 --track_marker_size 4 --show_pfdiam 0 \
    --track_lookback_days 7 --periodic_perimeter x --title_prefix 'GPM Tb+IMERG V07' --output $SCRATCH/pyflextrkr_viz/frames/
```

`--layout top_bottom` stacks the panels (14 x 11.5 inches, 2800 x 2300 pixels at 200 dpi); `--layout left_right --figsize 20 5.6` puts them side by side (a wide figure).
For the first look at another period make a few days of frames with `-p 1 --workers 32` (48 frames took 1.5 minutes together with the mp4).

Things to look at: the clouds under the tracks (`--track_text_size`, `--track_text_alpha`, `--track_marker_size`; `--track_halo_color white` if the tracks have to stand out), the label crowding (`--show_track_numbers`), the text size (`--fontsize`: the labels, titles, colorbars and
legends; the track numbers keep their own size), the width of the coastlines (`--land_linewidth`), the `--map_resolution` (`110m` is enough for the global map), the
title (`--title_prefix`), and the seam of the map at 45E (the edge of the Robinson map, where nothing should be drawn across).

## The animation

```bash
# a compute node: 32 workers, one frame per task (the frames are independent); the parameters are at the top of the file
python visualization/make_mcs_animation.py --set start_date=2025-01-01T00 --set end_date=2025-01-03T00 --set n_workers=32
```

The wrapper makes the PNG files `fig_root/quicklooks_2panel_tbpf/2025/mcs_tbpf_YYYYMMDD_HHMMSS.png` and the mp4
`fig_root/animations_2panel_tbpf/mcs_tbpf_2025-01-01_2025-01-03.mp4` (the video settings: `playback_fps = 2` PNG files per second, which is 2 hours of data per second, so the 2 days take 24 s;
written at 10 frames per second, `video_quality = 24`, at most 1920 pixels wide (the PNG files are 2800), x264 default preset: [animations.md](animations.md)). `--set run_ffmpeg=false` makes the PNG files only; `--set run_plotting=false`
makes the mp4 from existing PNG files; `--set gif_width=900 --set gif_duration=8` also makes a GIF of the first 8 seconds; `--set NAME=VALUE` changes any parameter of the block (`--set lat_range=[-30,30]`, `--set track_halo_color=white`,
`--set show_track_numbers=[true,true]`, `--set fontsize=16`, `--set show_pfdiam=true`, `--set data_type=tb`). The 2 days (48 frames, `n_workers=32`) took 1.5 minutes
on one node including the start of the Dask workers and the mp4 (11.8 MB at the default CRF 24, 16.3 MB at CRF 20).

## How the global map is made

| step | where | what |
|---|---|---|
| projection | `mapcore.get_projections` | `Robinson(central_longitude=225)` for the axes, `PlateCarree` for the data (longitudes of the files and of the statistics are normalized to -180..180 for non-PlateCarree projections) |
| domain | `mapcore.set_map_domain` | `set_global()` and then the y limits of the band: `y = proj.transform_point(0, lat, PlateCarree())` for `lat_min` and `lat_max` -> `ax.set_ylim` (the snippet of the cofeature maps; at the poles, where the transform is NaN, the y limit of the projection) |
| figure size | `mapcore.projected_aspect` (`auto_map_domain`), `panel_fractions`, `figsize_for_aspect` | the height from the shape of the projected band (0.374 of the width for 60S-60N) and the panel layout |
| latitude labels | `mapcore.lat_labels_on_boundary` | one label at each gridline (60S, 40S, ... 60N) along the curved left edge of the map; cartopy's own labels skip the lines (+-40, +-60) that end at the curved edge inside the rectangle of the axes. The panel titles are lifted above the 60N label |
| colorbars | `mapcore.fit_colorbars_to_maps`, `keep_colorbar_text_inside` | the map keeps its shape, each colorbar has the extent of its map, and the text of the colorbars stays inside the figure |
| outlines | `masks.label_perimeter(..., periodic=(True, False))` | the files are global in longitude, so an MCS that crosses the edge of the array (at the dateline) is one object: the outline is found on the array wrapped in x (`--periodic_perimeter x`), with no line along the dateline. (`--subset 1` with an extent that covers all longitudes is no subset: the window is the whole grid, so the wrap stays on) |
| tracks | `mcs_map.plot_map_2panels`, `trackdraw.halo_effects` | paths up to the frame time (7 days of history, `--track_lookback_days 7`), the initiation symbol, the numbers (panel (a) only; small and 60 % opaque); with `--show_pfdiam 1` the PF circles (`tissot` on the normalized longitude) and with `--track_halo_color` a halo around all of them |

Code map: [structure.md](structure.md).
