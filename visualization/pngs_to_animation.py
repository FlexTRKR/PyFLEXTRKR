#!/usr/bin/env python
"""
Make an mp4 (and a GIF) from PNG files that are already plotted, with the video settings of the make_*_animation.py wrappers:
the speed of the animation, the frame rate of the file, the quality, even picture sizes (yuv420p needs them), a width that is never enlarged.

Usage (the PNG files in time order: a directory, a quoted glob pattern, or a list of files; the last argument is the mp4 file):
>python pngs_to_animation.py frames/ movie.mp4
>python pngs_to_animation.py 'frames/mcs_tbpf_*.png' movie.mp4 --playback_fps 2 --video_quality 24 --output_width 1920
>python pngs_to_animation.py 'frames/mcs_tbpf_*.png' movie.mp4 --gif_width 480 --gif_duration 12      # and a GIF of the first 12 s
>python pngs_to_animation.py 'frames/mcs_tbpf_*.png' movie.mp4 --gif_width 480 --print_command        # only show the ffmpeg commands

The settings (the same names as in the wrappers):
--playback_fps    PNG files shown per second = the speed of the animation (2: hourly PNG files, 2 hours of data per second)
--frame_stride    use every Nth PNG file (2: every second one), the other way to make a long period shorter
--video_fps       frame rate of the video file; default: playback_fps repeated to at least 10 frames per second, a whole number of
                  repeats per PNG (players handle very low frame rates badly); it does not change the speed
--video_quality   CRF of libx264: lower = better and larger (default 24; 18-28 is useful, +6 makes the file 33-55 % smaller)
--output_width    at most this many pixels wide, never enlarged, the height follows (default 1920)
--video_preset    x264 preset (default: none = the x264 default, medium)
--gif_width       also make a GIF of the mp4 (--gif_fps, --gif_start, --gif_duration in seconds, --gif_colors, --gif_dither)

The same by hand with ffmpeg alone (what --print_command shows), for the PNG files frames/mcs_*.png:
ffmpeg -framerate 2 -pattern_type glob -i 'frames/mcs_*.png' -vf "scale='min(1920,trunc(iw/2)*2)':-2" \\
       -c:v libx264 -r 10 -crf 24 -pix_fmt yuv420p -y movie.mp4
(-framerate before -i = PNG files per second; the scale filter makes both sizes even and keeps the width at 1920 or less; -r after the codec =
frames per second of the file, each PNG is repeated; -crf = quality; yuv420p = plays in every player). The code is in pyflextrkr/visualization/animation.py.

Author: Zhe Feng | zhe.feng@pnnl.gov
"""
import sys
from pyflextrkr.visualization.animation import main

if __name__ == "__main__":
    sys.exit(main())
