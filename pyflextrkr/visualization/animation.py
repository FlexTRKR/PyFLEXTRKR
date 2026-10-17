"""
Animations from the PNG files of the plot scripts with ffmpeg: mp4 (H.264) with the settings of the make_*_animation.py
wrappers, and small GIFs for documents.

Called by: visualization/make_mcs_animation.py, visualization/make_cell_animation.py. Calls: ffmpeg, pandas.
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import pandas as pd

#-----------------------------------------------------------------------
def find_ffmpeg():
    """ffmpeg next to the running python (the conda environment of PyFLEXTRKR), then the one on PATH."""
    env = os.path.join(os.path.dirname(sys.executable), 'ffmpeg')
    return env if os.path.isfile(env) else (shutil.which('ffmpeg') or 'ffmpeg')

#-----------------------------------------------------------------------
def frames_in_range(figdir, basename, start, end):
    """
    Sorted PNG files {figdir}{basename}YYYYMMDD_HHMMSS.png whose time is within [start, end].

    The time is read from the 15 characters after the base name (the figure names of figname_type='date_time'); other
    PNG files in the directory are skipped.
    """
    t0, t1 = pd.to_datetime(start), pd.to_datetime(end)
    all_png_files = sorted(glob.glob(f'{figdir}{basename}*.png'))
    out = []
    for png_file in all_png_files:
        # Extract the datetime string from filename (assumes format: basename_YYYYMMDD_HHMMSS.png)
        try:
            basename_len = len(basename)
            datetime_str = os.path.basename(png_file)[basename_len:basename_len+15]  # YYYYMMDD_HHMMSS
            file_dt = pd.to_datetime(datetime_str, format='%Y%m%d_%H%M%S')

            # Check if file is within the specified date range
            if t0 <= file_dt <= t1:
                out.append(png_file)
        except (ValueError, IndexError):
            # Skip files that don't match the expected datetime format
            continue
    return out

#-----------------------------------------------------------------------
def make_mp4(png_files, out_file, input_framerate=2, output_framerate=10, crf=20, width=1920, preset=None):
    """
    mp4 (H.264, yuv420p) from PNG files.

    Args:
        png_files: list of PNG paths in time order
        out_file: string, output file name
        input_framerate: PNG files shown per second (the speed of the animation)
        output_framerate: frame rate of the video file (lower values = smaller file)
        crf: quality (lower = better, 18-28 is good)
        width: output width in pixels, height follows from the aspect ratio (None = the size of the PNG files)
        preset: x264 speed / compression preset (ultrafast ... medium ... veryslow); None = no -preset option (x264 default, medium).
            A slower preset made the file only about 1 % smaller at the same CRF in the tests of docs/visualization/animations.md.

    Returns:
        return_code, command (list)
    """
    # Create a temporary file list for FFmpeg
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_filelist = f.name
        for png_file in sorted(png_files):
            f.write(f"file '{png_file}'\n")

    # Build video filter string based on width
    if width is not None:
        # Scale to specified width, maintaining aspect ratio, ensuring dimensions divisible by 2
        vf_scale = f'scale={width}:-2'
    else:
        # Just ensure dimensions are divisible by 2 (no scaling)
        vf_scale = 'scale=trunc(iw/2)*2:trunc(ih/2)*2'

    # Use FFmpeg concat demuxer with file list
    ffmpeg_cmd = [
        find_ffmpeg(),
        '-f', 'concat',
        '-safe', '0',
        '-r', str(input_framerate),  # Input framerate
        '-i', temp_filelist,
        '-vf', vf_scale,
        '-c:v', 'libx264',
        '-r', str(output_framerate),  # Output framerate
        '-crf', str(crf),
        '-pix_fmt', 'yuv420p',
    ]
    if preset:
        ffmpeg_cmd += ['-preset', str(preset)]
    ffmpeg_cmd += ['-y', out_file]
    try:
        result = subprocess.run(ffmpeg_cmd)
    finally:
        # Clean up temporary file
        os.unlink(temp_filelist)
    return result.returncode, ffmpeg_cmd

#-----------------------------------------------------------------------
def make_gif(source, out_file, start=0.0, duration=None, fps=2, width=800, colors=256, dither='none', palette='global',
             diff_mode=None):
    """
    GIF from an mp4 (or any video ffmpeg reads): the part from `start` [s] for `duration` [s] (None = to the end), `fps`
    frames per second, at most `width` pixels wide. A GIF has at most 256 colors, so the colors of the frames are
    reduced to a palette and the pixels are dithered.

    Args:
        colors: number of colors of the palette (up to 256)
        dither: 'none' (default: the maps are made of flat colors, so dithering only adds dots to the colorbars and makes the file
            bigger, see docs/visualization/animations.md), 'bayer:bayer_scale=N' (a fixed pattern), 'sierra2_4a' and 'floyd_steinberg'
            (error diffusion, for smooth color gradients), ... (the dither options of ffmpeg paletteuse)
        palette: 'global' (one palette for all frames, no flicker of colors) or 'frame' (a palette for each frame)
        diff_mode: None or 'rectangle' (paletteuse: only the changed rectangle of each frame is stored)

    Returns:
        return_code, command (list)
    """
    if palette == 'global':
        gen, use = f"palettegen=max_colors={int(colors)}:stats_mode=full", f"paletteuse=dither={dither}"
    elif palette == 'frame':
        gen, use = f"palettegen=max_colors={int(colors)}:stats_mode=single", f"paletteuse=dither={dither}:new=1"
    else:
        raise ValueError(f"palette must be 'global' or 'frame', not {palette}")
    if diff_mode:
        use += f":diff_mode={diff_mode}"
    vf = f"fps={fps},scale='min({int(width)},iw)':-1:flags=lanczos,split[a][b];[a]{gen}[p];[b][p]{use}"
    cmd = [find_ffmpeg(), '-hide_banner', '-loglevel', 'warning', '-ss', str(start)]
    if duration is not None:
        cmd += ['-t', str(duration)]
    cmd += ['-i', source, '-vf', vf, '-loop', '0', '-y', out_file]
    return subprocess.run(cmd).returncode, cmd
