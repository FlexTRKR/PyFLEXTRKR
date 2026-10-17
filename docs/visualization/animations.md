# Animations: mp4 settings (CRF, x264 preset) and test on the example frames

The wrappers (`visualization/make_mcs_animation.py`, `examples/make_mcs_animation_*.py`, `make_cell_animation.py`) make the PNG frames and then an mp4 with ffmpeg
(`animation.make_mp4`): the PNG files are read in time order (`input_framerate` frames per second, the speed of the animation), written as H.264 (`libx264`,
`yuv420p`) at `output_framerate` frames per second, `video_quality` (CRF), `output_width` pixels wide (the height follows), and **`video_preset`**.

| parameter | default | meaning |
|---|---|---|
| `input_framerate` | 2 | PNG files shown per second |
| `output_framerate` | 10 | frame rate of the video file (each PNG is repeated 5 times) |
| `video_quality` | 20 | CRF of x264: lower = better and larger; +6 is about half the size |
| `output_width` | 1920 | pixels (`None`: the size of the PNG files) |
| `video_preset` | `None` | x264 preset (`ultrafast` ... `medium` ... `veryslow`); `None` = no `-preset` option, the x264 default (`medium`) |

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

Reproduce, or test your own frames: [`visualization/examples/compare_x264_presets.py`](../../visualization/examples/compare_x264_presets.py) makes the mp4 of the wrappers
(`animation.make_mp4`) for each preset or CRF and prints the size, the encoding time and the SSIM against the PNG frames (ffmpeg `ssim` filter):

```bash
python visualization/examples/compare_x264_presets.py '/path/to/frames/mcs_tbpf_*.png' --start 100 --n 120 --crf 24 --in_fps 6 --out_fps 12
python visualization/examples/compare_x264_presets.py '/path/to/frames/*.png' --presets none slow --crf 20 22 24 26 28
```

## GIF: palette, colors, dither

`animation.make_gif` makes a small GIF from the mp4 (for the READMEs): `gif_width` pixels wide (`None`: no GIF), `gif_fps` frames per second (default `input_framerate`:
one GIF frame per PNG file), the part of the mp4 from `gif_start` for `gif_duration` seconds (default: all of it). A GIF has at most 256 colors, so ffmpeg makes a palette of the
frames (`palettegen`) and maps every pixel to it (`paletteuse`); the pixels can be dithered (error diffusion or a fixed pattern) to imitate the colors in between.
Tested on the three animations of the [README](../../visualization/README.md#more-mcs-looks) (the mp4 of the examples, GIF 480-800 pixels wide, 2 frames per second);
the quality is the SSIM of the GIF against the mp4:

| setting (palette / colors / dither) | IMERG 1 panel, 48 frames, 560 px | WRF Tb + radar, 49 frames, 800 px | Himawari Tb, 24 frames, 560 px |
|---|---|---|---|
| **global / 256 / none** (default) | **8.99 MB**, SSIM 0.9927 | **7.14 MB**, 0.9895 | **3.28 MB**, 0.9890 |
| global / 256 / bayer, scale 5 | 9.10 MB, 0.9926 | 7.26 MB, 0.9863 | 3.32 MB, 0.9880 |
| global / 256 / bayer, scale 3 | 9.51 MB, 0.9831 | 7.40 MB, 0.9768 | 3.34 MB, 0.9772 |
| global / 256 / sierra2_4a | 9.21 MB, 0.9927 | 7.53 MB, 0.9890 | 3.61 MB, 0.9870 |
| global / 256 / floyd_steinberg | 9.12 MB, 0.9928 | 7.39 MB, 0.9894 | 3.52 MB, 0.9878 |
| global / 128 / sierra2_4a | 7.73 MB, 0.9894 | 6.29 MB, 0.9841 | 2.93 MB, 0.9793 |
| frame / 256 / sierra2_4a (a palette for each frame) | 10.80 MB, 0.9927 | 9.69 MB, 0.9896 | 4.24 MB, 0.9876 |
| frame / 256 / bayer, scale 3 | 11.76 MB, 0.9833 | 10.27 MB, 0.9767 | 4.26 MB, 0.9764 |

- **No dither is the best for these maps**: the fields are drawn with flat color levels (`BoundaryNorm`), so there is no gradient to imitate, and dithering only puts dots into the
  flat colors of the colorbars and the shading (error diffusion), or a visible pattern (`bayer`, with the lowest SSIM), and makes the files 1-10 % larger. The colorbars of the
  GIFs without dither are as clean as the PNG files.
- **One palette for all frames** (`global`) makes the files 17-36 % smaller than a palette for each frame, and the colors do not change from frame to frame.
- 128 colors make the files 11-14 % smaller, but the colors of the colorbars start to merge: not used. `diff_mode='rectangle'` (store only the changed rectangle) does not change the size,
  because the clouds change everywhere from one hour to the next.
- A GIF made from the PNG files instead of the mp4 is about 13 % smaller (7.86 MB for the IMERG set) with the same quality; the wrappers make it from the mp4, so the
  GIF has the size and the speed of the video.
- **The size is set by the pixels and the number of frames** (about 3.5 bits per pixel here). The GIFs of the README are 3.3-3.5 MB: 24 frames (`gif_duration = 12` seconds at 2 frames per second = the first day)
  of 480 x 649 (IMERG), 800 x 460 (WRF) and 560 x 560 pixels (Himawari). For a smaller file use fewer frames or a smaller `gif_width`.

Test your own animation: [`visualization/examples/compare_gif_settings.py`](../../visualization/examples/compare_gif_settings.py) makes the GIF for each setting and prints the size and the
SSIM against the mp4:

```bash
python visualization/examples/compare_gif_settings.py animation.mp4 --width 560 --fps 2
python visualization/examples/compare_gif_settings.py animation.mp4 --width 800 --settings global/256/none global/256/sierra2_4a frame/256/none --keep
```
