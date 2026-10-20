# Animations: playback speed, video quality and file size

The wrappers (`visualization/examples/make_mcs_animation*.py`, `make_cell_animation_nexrad.py`) make the PNG frames and then an mp4 with ffmpeg
(`animation.make_animation`, which calls `make_mp4`) and, only if `gif_width` is set (`--set gif_width=480`; the default is no GIF), a GIF of it for a README. The settings are the same in every wrapper and in the command
line [`visualization/pngs_to_animation.py`](../../visualization/pngs_to_animation.py), which makes the mp4 from PNG files you already have. The names and explanations
follow the wrapper of the CONUS tracking (`conus_tracking/visualizations/make_mcs_multicell_animation.py`).

**Where the files go.** Without `fig_root`, `figdir` and `animation_dir` the PNG files and the animation (`{fig_basename}{start}_{end}.mp4` and `.gif`) go to
`{root_path}/quicklooks_trackpaths/` of the tracking config file, next to the tracking output: the folder that the demo scripts (`config/demo_*.sh`) use for their
quicklooks, and the PNG file names are the same (`--figbasename` of the demo command = `fig_basename` of the wrapper), so a demo run and a wrapper run share the
folder. If that folder cannot be written (the read-only outputs of someone else's run, as with the global V3 config of `make_mcs_animation.py`), the wrapper stops and asks for
`fig_root`: `{fig_root}/quicklooks_{plot_type}_{data_type}/{year}/` for the PNG files of the MCS maps (`quicklooks_cell/` for the cell maps) and `{fig_root}/animations_.../` for the animation.

## Where the README and docs images are

The images and animations of the documentation (about 60 MB, the README animations alone 42 MB) are not in the repository: they are assets of the release
[`docs-assets`](https://github.com/FlexTRKR/PyFLEXTRKR/releases/tag/docs-assets), linked as
`https://github.com/FlexTRKR/PyFLEXTRKR/releases/download/docs-assets/<file name>`. This keeps the clone at the size of the code and the images still show on GitHub. A new or re-rendered file is added with
`gh release upload docs-assets <file>` (the same name with `--clobber` replaces a file, and every README that links it, also the ones of old versions, then shows the new picture: use a new name for a new version of a figure
that an old README should keep). `tests/test_docs.py` checks that the markdown files only link images of this release.

## The settings

| parameter | default | meaning |
|---|---|---|
| `playback_fps` | 2 | PNG files shown per second = **the playback speed** of the animation (larger = faster) |
| `frame_stride` | 1 | use every Nth PNG file (2 = every second file): the other way to make a long period shorter |
| `video_fps` | `None` | frame rate of the video **file**. `None` = automatic: `playback_fps` repeated to at least 10 frames per second, a whole number of repeats per PNG file (2 -> 10, 3 -> 12, 12 -> 12). It does not change the speed. Keep a value you give a whole multiple of `playback_fps` (in between, the PNG files are repeated unevenly) |
| `video_quality` | 24 | CRF of libx264: lower = better and larger (0-51, 18-28 is the useful range); +6 makes the file 33-55 % smaller (tests below) |
| `output_width` | 1920 | at most this many pixels wide, **never enlarged** (smaller PNG files keep their size), the height follows; `None`: the size of the PNG files |
| `video_preset` | `None` | x264 preset (`ultrafast` ... `medium` ... `veryslow`); `None` = no `-preset` option, the x264 default (`medium`) |
| `gif_width` | `None` | width of a GIF made of the mp4 (`None`: no GIF); `gif_fps` (`None`: `playback_fps`), `gif_start` and `gif_duration` [s] select the part of the mp4, `gif_colors` (256), `gif_dither` (`'none'`) |

The old `input_framerate` is `playback_fps`, and `output_framerate` is `video_fps` (now automatic): an old script still runs, with a message. The earlier default `video_quality = 20` is now
24: the mp4 files of the examples are 23-33 % smaller (set 20 to get the old files back).

**Recommended (the defaults of the wrappers):** hourly MCS files `playback_fps = 2`, `video_quality = 24`, `output_width = 1920`, `video_fps = None`, `video_preset = None`; CRF 22 for maps with a lot of fine detail
(the IR brightness temperature of Himawari). The GIF for a README (opt-in, `--set gif_width=...`): `gif_width` 480-720 pixels, `gif_duration = 8` seconds (16 frames), 256 colors, no dither: 1.6-2.3 MB each (below).

### What `playback_fps` means

The speed of the animation is `playback_fps` times the time step of the PNG files:

| PNG files | `playback_fps` | speed | length |
|---|---|---|---|
| hourly (the MCS examples), 2 days = 48 files | 2 | 2 hours of data per second | 24 s |
| hourly, 2 days | 4 | 4 hours per second | 12 s |
| 5-minute (the cell tracking of `make_cell_animation_nexrad.py`), 3 days = 864 files | 12 | 1 hour per second | 72 s |

## Make an mp4 or a GIF from PNG files you already have

With the script (a directory, a quoted glob pattern, or a list of files, in time order; `--print_command` only prints the ffmpeg commands):

```bash
python visualization/pngs_to_animation.py 'frames/mcs_tbpf_*.png' movie.mp4 --playback_fps 2 --video_quality 24 --output_width 1920
python visualization/pngs_to_animation.py frames/ movie.mp4 --gif_width 480 --gif_duration 8      # and a GIF of the first 8 seconds
```

or in Python (`from pyflextrkr.visualization.animation import make_mp4, make_gif, make_animation`): `make_mp4('frames/mcs_*.png', 'movie.mp4', playback_fps=2, crf=24)`.

With ffmpeg alone (this is what the script runs; the same file, tested):

```bash
ffmpeg -framerate 2 -pattern_type glob -i 'frames/mcs_tbpf_*.png' \
       -vf "scale='min(1920,trunc(iw/2)*2)':-2" -c:v libx264 -r 10 -crf 24 -pix_fmt yuv420p -y movie.mp4
```

- `-framerate 2` (before `-i`): PNG files read per second = `playback_fps`, the speed. `-pattern_type glob -i 'frames/mcs_tbpf_*.png'`: all files that match, sorted by name (quote the pattern, so the shell does not expand it).
- `-vf "scale='min(1920,trunc(iw/2)*2)':-2"`: at most 1920 pixels wide and never enlarged (`min(1920, iw)`), the height follows (`-2` keeps the shape and makes it even).
  **Both sizes have to be even**: `trunc(iw/2)*2` makes an odd width even, and without it ffmpeg fails on a PNG file with an odd number of pixels (tested: "Invalid argument", nothing is written).
- `-c:v libx264 -crf 24`: H.264 at the quality `video_quality`.
- `-r 10` (after the codec): frames per second of the file (`video_fps`); every PNG file is repeated 5 times, the speed does not change.
- `-pix_fmt yuv420p`: the pixel format that every player plays. Without it ffmpeg writes `yuv444p` for PNG input (tested: H.264 "High 4:4:4 Predictive"), which many players and browsers cannot play (a usual reason for "the video does not play").

The GIF from that mp4 (what `gif_width = 480`, `gif_duration = 8` make): 2 frames per second, one palette of 256 colors for all frames (`palettegen`, `stats_mode=full`), no dither:

```bash
ffmpeg -t 8 -i movie.mp4 -vf "fps=2,scale='min(480,iw)':-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];[b][p]paletteuse=dither=none" -loop 0 -y movie.gif
```

## Quality and size: CRF

Tested on the frames of the examples (the mp4 of the wrappers, `playback_fps = 2`, 10 frames per second in the file, `output_width = 1920`): size, **the size relative to CRF 20**, and the quality as the SSIM of the video
against the PNG files (so that a smaller file is not a worse video). The login node encodes 48 PNG files of 800-1200 pixels in about 0.7-1.3 s, the global frames (2800 pixels, read and scaled) in 5 s; the time does not depend on the CRF.

| CRF | IMERG 1 panel, 800 x 1081 px, 48 PNG files | WRF Tb + radar, 1200 x 689 px, 48 | Himawari Tb, 800 x 800 px, 24 | global map, 2800 x 2304 px (scaled to 1920), 48 |
|---|---|---|---|---|
| 20 | 4.73 MB, 0.9944 | 6.91 MB, 0.9958 | 3.21 MB, 0.9924 | 16.30 MB, 0.9964 |
| 22 | 3.91 MB, -17 %, 0.9920 | 6.11 MB, -12 %, 0.9939 | 2.71 MB, -16 %, 0.9884 | 13.94 MB, -14 %, 0.9947 |
| **24** | **3.19 MB, -33 %, 0.9888** | **5.35 MB, -23 %, 0.9913** | **2.23 MB, -31 %, 0.9822** | **11.75 MB, -28 %, 0.9923** |
| 26 | 2.57 MB, -46 %, 0.9847 | 4.64 MB, -33 %, 0.9878 | 1.78 MB, -45 %, 0.9732 | 9.75 MB, -40 %, 0.9891 |
| 28 | 2.04 MB, -57 %, 0.9796 | 4.00 MB, -42 %, 0.9833 | 1.36 MB, -58 %, 0.9610 | 7.94 MB, -51 %, 0.9849 |
| 30 | 1.62 MB, -66 %, 0.9728 | 3.42 MB, -51 %, 0.9773 | 1.01 MB, -69 %, 0.9465 | 6.35 MB, -61 %, 0.9794 |

- **+6 CRF makes the file 33-55 % smaller** on these sets (the x264 rule of thumb is "half"; the WRF set is at the low end).
- **CRF 24** (the default) against 20: 23-33 % smaller, SSIM lower by 0.004-0.010. Looked at in crops (2 times magnified) of the frames, CRF 20 and 24 look like the PNG file: thin outlines, track lines and numbers, the faint
  grid lines; at CRF 28 the thin outlines and the small colored dots get softer.
- The SSIM falls fastest for the Himawari IR frames (fine texture everywhere): 0.9610 at CRF 28 against 0.9796-0.9849 for the others. Use 22 there, or for any map with a lot of fine detail.
- **Width**: the global frames are 2800 pixels wide. At CRF 24, `output_width = 1920` makes 11.75 MB, 1400 pixels 6.66 MB and the full 2800 pixels 24.65 MB. The SSIM is measured against the PNG files scaled the same way (0.9920, 0.9923, 0.9925),
  so it does not show the detail that the smaller size loses: look at the frames. A width above the size of the PNG files changes nothing (never enlarged).

## Frame rate of the video file (`video_fps`)

The wrappers write 10 frames per second although a PNG file is shown for 0.5 s: `video_fps = None` repeats every PNG file 5 times, because players handle files with a very low frame rate badly (the experience of the
CONUS wrapper; not tested here). The repeats are not free, but cheap. The same PNG files as above at CRF 24, file frame rate 2 against 10 per second:

| set | file frame rate 2 per second (no repeats): size, SSIM | file frame rate 10 per second (the default): size, SSIM | the 2 per second file at the same SSIM as the 10 per second file |
|---|---|---|---|
| IMERG 1 panel, 800 x 1081 px, 48 PNG files | 3.69 MB, 0.9918 | 3.19 MB (-14 %), 0.9888 | 3.03 MB (the 10 per second file is +5 %) |
| WRF Tb + radar, 1200 x 689 px, 48 | 5.84 MB, 0.9940 | 5.35 MB (-8 %), 0.9913 | 5.07 MB (the 10 per second file is +6 %) |
| Himawari Tb, 800 x 800 px, 24 | 2.51 MB, 0.9873 | 2.23 MB (-11 %), 0.9822 | 2.14 MB (the 10 per second file is +4 %) |

- At the **same CRF** the 10 per second file is 8-14 % smaller than the 2 per second file, but x264 also gives it a lower quality (SSIM lower by 0.003-0.005).
  At the **same SSIM** it is 4-6 % larger. Keep `video_fps = None` for the players, and use CRF 24 (or 22 for fine detail) as in the tables above.
- A `video_fps` that is not a whole multiple of `playback_fps` (5 for 2) repeats the PNG files 2 and 3 times in turn: the speed becomes uneven.

## Does a slower preset make the file smaller?

Tested on **your example frames**: 120 consecutive PNG frames of three of the CONUS animations of `airscape/conus_tracking` (cells + MCS and cells in the
southeast, the MCS of the CONUS maps: 3000 px wide, scaled to 1920, CRF 24, 6 frames per second read, as the wrappers there) and the frames of the
global (12) and IMERG South America (24) MCS maps of this repository (CRF 20, 2 frames per second read). The same ffmpeg command with each preset
(`-threads 32`); the quality is the SSIM of the video against the source frames (so "smaller" is not "worse").

**Size of the mp4 relative to the default (no `-preset`):**

| frames | CRF | ultrafast | veryfast | fast | default (none) | slow | slower | veryslow |
|---|---|---|---|---|---|---|---|---|
| CONUS cells + MCS, southeast (3000 px), 120 | 24 | +159.6 % | -2.8 % | +2.0 % | **0** | -1.5 % | -12.4 % | -16.9 % |
| CONUS cells, southeast (3000 px), 120 | 24 | +147.5 % | -3.3 % | +2.1 % | **0** | -1.2 % | -11.5 % | -18.9 % |
| CONUS MCS (3000 px), 120 | 24 | +162.0 % | -12.8 % | +0.2 % | **0** | -0.6 % | -3.1 % | -8.4 % |
| global MCS, Robinson (2800 px), 12 | 20 | +148.5 % | -8.3 % | +0.5 % | **0** | -0.2 % | -2.1 % | -6.9 % |
| IMERG MCS, South America (2000 px), 24 | 20 | +165.4 % | -3.2 % | +0.4 % | **0** | +0.0 % | -1.4 % | -3.9 % |

**SSIM of the video against the PNG frames** (the quality; the PSNR follows the same pattern):

| frames | CRF | ultrafast | veryfast | fast | default (none) | slow | slower | veryslow |
|---|---|---|---|---|---|---|---|---|
| CONUS cells + MCS, southeast (3000 px), 120 | 24 | 0.99740 | 0.99443 | 0.99616 | 0.99621 | 0.99629 | 0.99621 | 0.99626 |
| CONUS cells, southeast (3000 px), 120 | 24 | 0.99708 | 0.99405 | 0.99567 | 0.99572 | 0.99579 | 0.99568 | 0.99556 |
| CONUS MCS (3000 px), 120 | 24 | 0.99737 | 0.99176 | 0.99536 | 0.99560 | 0.99584 | 0.99605 | 0.99584 |
| global MCS, Robinson (2800 px), 12 | 20 | 0.99747 | 0.99402 | 0.99635 | 0.99639 | 0.99647 | 0.99649 | 0.99630 |
| IMERG MCS, South America (2000 px), 24 | 20 | 0.99797 | 0.99548 | 0.99684 | 0.99687 | 0.99698 | 0.99698 | 0.99688 |

**Encoding time relative to the default** (the PNG files have to be read and decoded, which takes most of the time of the large frames):

| frames | CRF | ultrafast | veryfast | fast | default (none) | slow | slower | veryslow |
|---|---|---|---|---|---|---|---|---|
| CONUS cells + MCS, southeast (3000 px), 120 | 24 | 1.38 | 0.95 | 0.99 | 1.0 | 1.03 | 1.13 | 1.28 |
| CONUS cells, southeast (3000 px), 120 | 24 | 1.30 | 0.93 | 0.95 | 1.0 | 1.09 | 1.12 | 1.29 |
| CONUS MCS (3000 px), 120 | 24 | 1.49 | 0.94 | 0.94 | 1.0 | 1.26 | 1.11 | 1.18 |
| global MCS, Robinson (2800 px), 12 | 20 | 0.76 | 0.76 | 0.94 | 1.0 | 1.53 | 2.06 | 2.76 |
| IMERG MCS, South America (2000 px), 24 | 20 | 0.65 | 0.71 | 0.94 | 1.0 | 1.71 | 2.41 | 3.59 |

Size of the default encodes (MB):

| frames | CRF | ultrafast | veryfast | fast | default (none) | slow | slower | veryslow |
|---|---|---|---|---|---|---|---|---|
| CONUS cells + MCS, southeast (3000 px), 120 | 24 | 17.78 | 6.66 | 6.99 | 6.85 | 6.75 | 6.00 | 5.69 |
| CONUS cells, southeast (3000 px), 120 | 24 | 16.51 | 6.45 | 6.81 | 6.67 | 6.59 | 5.90 | 5.41 |
| CONUS MCS (3000 px), 120 | 24 | 14.28 | 4.75 | 5.46 | 5.45 | 5.42 | 5.28 | 4.99 |
| global MCS, Robinson (2800 px), 12 | 20 | 10.46 | 3.86 | 4.23 | 4.21 | 4.20 | 4.12 | 3.92 |
| IMERG MCS, South America (2000 px), 24 | 20 | 14.78 | 5.39 | 5.59 | 5.57 | 5.57 | 5.49 | 5.35 |

What the numbers say (computed from the table above):

- **`slow` is not worth it**: it changes the size by -1.5 to +0.0 % relative to the default on these sets (at equal SSIM, interpolated over CRF 20-28: -2.5 %, -1.2 %), the
  quality is the same (SSIM within 0.0002) and the encoding takes 1.0-1.7 times as long. The default `video_preset = None` is as good.
- `fast` changes the size by +0.2 to +2.1 %. `veryfast` by -12.8 to -2.8 %, but its SSIM is lower by 0.0014-0.0038: a worse video, not a better compression.
  `ultrafast` makes files 2.5-2.7 times as large.
- `slower` and `veryslow` make smaller files at the same SSIM: -12.4 to -1.4 % (`slower`) and -18.9 to -3.9 % (`veryslow`), for 1.2-3.6 times the encoding time of the default
  (`veryslow`). The gain is largest for the large frames, and the time is small compared with the plotting of the frames. For a long animation that is a choice,
  not a default: set `video_preset = 'slower'` or `'veryslow'` if the size matters more than the minutes.

**A change of CRF is a much larger lever than the preset**: CRF 24 -> 28 makes the file 29 % smaller. Size and SSIM of the default preset and `slow` for
CRF 20-28 on two of the sets (SSIM against the PNG frames):

| preset | CRF 20 | CRF 22 | CRF 24 | CRF 26 | CRF 28 |
|---|---|---|---|---|---|
| CONUS cells + MCS, southeast (3000 px), default | 9.63 MB, 0.9980 | 8.18 MB, 0.9972 | 6.85 MB, 0.9962 | 5.78 MB, 0.9949 | 4.83 MB, 0.9933 |
| CONUS cells + MCS, southeast (3000 px), slow | 9.51 MB, 0.9981 | 8.06 MB, 0.9973 | 6.75 MB, 0.9963 | 5.69 MB, 0.9950 | 4.75 MB, 0.9934 |
| global MCS, Robinson (2800 px), default | 4.21 MB, 0.9964 | 3.60 MB, 0.9947 | 3.03 MB, 0.9924 | 2.51 MB, 0.9895 | 2.04 MB, 0.9855 |
| global MCS, Robinson (2800 px), slow | 4.20 MB, 0.9965 | 3.59 MB, 0.9949 | 3.03 MB, 0.9926 | 2.51 MB, 0.9897 | 2.05 MB, 0.9858 |

- CONUS cells + MCS, southeast (3000 px): size at SSIM 0.99621 (the default preset at CRF 24): none 6.85 MB (+0.0 %), slow 6.68 MB (-2.5 %)
- global MCS, Robinson (2800 px): size at SSIM 0.99244 (the default preset at CRF 24): none 3.03 MB (+0.0 %), slow 3.00 MB (-1.2 %)

## Reproduce, or test your own frames

[`visualization/examples/compare_x264_presets.py`](../../visualization/examples/compare_x264_presets.py) makes the mp4 of the wrappers (`animation.make_mp4`) for each preset, CRF or file frame rate and prints the size, the encoding time and the SSIM
against the PNG frames (ffmpeg `ssim` filter); the login node is enough for 50-200 frames:

```bash
python visualization/examples/compare_x264_presets.py '/path/to/frames/mcs_tbpf_*.png' --n 48 --presets none --crf 20 22 24 26 28 30 --playback_fps 2
python visualization/examples/compare_x264_presets.py '/path/to/frames/*.png' --presets none slow --crf 20 22 24 26 28
python visualization/examples/compare_x264_presets.py '/path/to/frames/*.png' --crf 24 --playback_fps 2 --video_fps 2 10
```

## GIF for the README

`animation.make_gif` makes the GIF from the mp4 (`gif_width` pixels wide, `None`: no GIF), `gif_fps` frames per second (default `playback_fps`: one GIF frame per PNG file, the speed of the mp4), the part of the mp4 from `gif_start` for `gif_duration` seconds
(default: all of it). A GIF has at most 256 colors, so ffmpeg makes a palette of the frames (`palettegen`) and maps every pixel to it (`paletteuse`); the pixels can be dithered to imitate the colors in between.

**A GIF is big, and its size is set by the pixels and the number of frames** (3.2-3.6 bits per pixel here, 0.14 / 0.12 / 0.10 MB per frame for the three examples below).
Tested on the first 24 frames of the examples (the mp4 at CRF 24; SSIM of the GIF against the PNG files):

| GIF, 24 frames (12 s at 2 per second) | IMERG 1 panel (800 x 1081 px PNG) | WRF Tb + radar (1200 x 689) | Himawari Tb (800 x 800) |
|---|---|---|---|
| width: size | 360 px: 1.97 MB, 420 px: 2.66 MB, 480 px: 3.37 MB, 560 px: 4.60 MB | 560 px: 1.78 MB, 640 px: 2.27 MB, 720 px: 2.84 MB, 800 px: 3.57 MB | 400 px: 1.65 MB, 480 px: 2.44 MB, 560 px: 3.27 MB |

| GIF at the middle width | IMERG 1 panel (800 x 1081 px PNG), 480 px | WRF Tb + radar (1200 x 689), 720 px | Himawari Tb (800 x 800), 480 px |
|---|---|---|---|
| 24 frames (12 s) | 3.37 MB | 2.84 MB | 2.44 MB |
| 16 frames (8 s) | 2.28 MB | 1.88 MB | 1.62 MB |
| 12 frames (6 s) | 1.70 MB | 1.43 MB | 1.21 MB |
| 256 colors, 24 frames | 3.37 MB, SSIM 0.9881 | 2.84 MB, SSIM 0.9873 | 2.44 MB, SSIM 0.9854 |
| 128 colors, 24 frames | 2.93 MB, SSIM 0.9851 | 2.31 MB, SSIM 0.9830 | 1.89 MB, SSIM 0.9801 |
| 64 colors, 24 frames | 2.42 MB, SSIM 0.9770 | 1.84 MB, SSIM 0.9745 | 1.48 MB, SSIM 0.9719 |
| made from the PNG files, 24 frames | 3.28 MB, SSIM 0.9934 | 2.73 MB, SSIM 0.9844 | 2.48 MB, SSIM 0.9916 |
| from the mp4, CRF 20, 24 frames | 3.39 MB, SSIM 0.9901 | 2.84 MB, SSIM 0.9888 | 2.44 MB, SSIM 0.9881 |
| from the mp4, CRF 24 (the wrappers), 24 frames | 3.37 MB, SSIM 0.9881 | 2.84 MB, SSIM 0.9873 | 2.44 MB, SSIM 0.9854 |
| from the mp4, CRF 28, 24 frames | 3.38 MB, SSIM 0.9835 | 2.84 MB, SSIM 0.9847 | 2.37 MB, SSIM 0.9782 |

- **Frames and width are the levers**: the size is proportional to the number of frames (12 s -> 8 s: -33 %), and goes with the square of the width. For the README the GIFs are **16 frames (`gif_duration = 8`)** at 480 (IMERG, Himawari) and 720 pixels (WRF): 2.3, 1.9 and 1.6 MB, instead of 3.4, 3.5 and 3.3 MB for 24 frames at 480-800 pixels; the text of the figures (titles, colorbars, track numbers) stays readable.
- **256 colors are kept**: 128 colors make the file 13-23 % smaller and lower the SSIM by 0.003-0.005; the gray gradient of the Tb colorbar starts to show bands (viewed). 64 colors (28-39 % smaller) are clearly banded and change the track color.
- **The mp4 quality does not change the GIF size** (CRF 20, 24, 28: within 3 % for each set). A GIF made straight from the PNG files has about the same size (the GIF from the mp4 is -2 to +4 % of it)
  and no consistent gain in SSIM against the PNG files (-0.003 to +0.006), so the wrappers make it from the mp4: one conversion, and the GIF has the speed of the video. (An earlier test on 48 frames at CRF 20 had found the GIF from PNG files 13 % smaller: not reproduced.)

Palette and dither, on the final GIFs (16 frames; the SSIM of the GIF against the mp4):

| setting (palette / colors / dither) | IMERG 1 panel, 480 px | WRF Tb + radar, 720 px | Himawari Tb, 480 px |
|---|---|---|---|
| **global / 256 / none** (default) | **2.21 MB, SSIM 0.9925** | **1.87 MB, SSIM 0.9894** | **1.61 MB, SSIM 0.9891** |
| global / 256 / bayer, scale 5 | 2.23 MB, SSIM 0.9925 | 1.91 MB, SSIM 0.9894 | 1.63 MB, SSIM 0.9881 |
| global / 256 / bayer, scale 3 | 2.33 MB, SSIM 0.9846 | 1.97 MB, SSIM 0.9804 | 1.66 MB, SSIM 0.9755 |
| global / 256 / sierra2_4a | 2.26 MB, SSIM 0.9925 | 1.99 MB, SSIM 0.9890 | 1.80 MB, SSIM 0.9872 |
| global / 256 / floyd_steinberg | 2.24 MB, SSIM 0.9927 | 1.95 MB, SSIM 0.9893 | 1.75 MB, SSIM 0.9880 |
| global / 128 / sierra2_4a | 2.00 MB, SSIM 0.9896 | 1.62 MB, SSIM 0.9838 | 1.47 MB, SSIM 0.9795 |
| frame / 256 / sierra2_4a (a palette for each frame) | 2.75 MB, SSIM 0.9926 | 2.63 MB, SSIM 0.9892 | 2.12 MB, SSIM 0.9874 |
| frame / 256 / bayer, scale 3 | 2.96 MB, SSIM 0.9845 | 2.77 MB, SSIM 0.9781 | 2.12 MB, SSIM 0.9762 |

- **No dither is the best for these maps**: the fields are drawn with flat color levels (`BoundaryNorm`), so there is no gradient to imitate, and dithering only puts dots into the
  flat colors of the colorbars and the shading (error diffusion), or a visible pattern (`bayer`, with the lowest SSIM), and makes the files 1-12 % larger. The colorbars of the GIFs without dither are as clean as the PNG files.
- **One palette for all frames** (`global`) makes the files 20-29 % smaller than a palette for each frame, and the colors do not change from frame to frame. `diff_mode='rectangle'` (store only the changed rectangle) does not change the size, because the clouds change everywhere from one hour to the next.

Test your own animation: [`visualization/examples/compare_gif_settings.py`](../../visualization/examples/compare_gif_settings.py) makes the GIF for each setting and prints the size and the
SSIM against the mp4:

```bash
python visualization/examples/compare_gif_settings.py animation.mp4 --width 480 --fps 2
python visualization/examples/compare_gif_settings.py animation.mp4 --width 720 --settings global/256/none global/128/none global/256/sierra2_4a --keep
```
