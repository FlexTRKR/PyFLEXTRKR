"""
Animations from PNG files with ffmpeg: an mp4 (H.264) with the settings of the make_*_animation.py wrappers, and small GIFs
for documents. The functions work on any PNG files, not only on those of the plot scripts:

    make_mp4(png_files, out_file, ...)        PNG files (a list, a directory or a glob pattern) -> mp4
    make_gif(mp4_file, out_file, ...)         mp4 -> GIF
    make_animation(png_files, mp4_file, ...)  both, with the settings printed (what the wrappers call)
    main(argv)                                the command line of visualization/pngs_to_animation.py

The settings (the same names in the wrappers and in the command line):
    playback_fps   PNG files shown per second = the speed of the animation (2: hourly files, 2 hours of data per second)
    frame_stride   use every Nth PNG file (2: every second file), the other way to make a long period shorter
    video_fps      frame rate of the video FILE; None = playback_fps repeated to at least 10 frames per second, a whole number
                   of repeats per PNG (players handle very low frame rates badly); it does not change the speed
    video_quality  CRF of libx264: lower = better and larger (18-28 is the useful range, +6 makes the file 33-55 % smaller)
    output_width   at most this many pixels wide (never enlarged), the height follows; None = the size of the PNG files
    video_preset   x264 preset; None = no -preset option (the x264 default, medium)

The same mp4 by hand, for the PNG files frames/mcs_*.png (2 PNG files per second, written at 10 frames per second):
    ffmpeg -framerate 2 -pattern_type glob -i 'frames/mcs_*.png' -vf "scale='min(1920,trunc(iw/2)*2)':-2" \\
           -c:v libx264 -r 10 -crf 24 -pix_fmt yuv420p -y movie.mp4
(-framerate before -i = PNG files per second; the scale filter keeps the width at 1920 pixels or less and makes both sizes even,
which yuv420p needs; -r after the codec = frames per second of the file, each PNG is repeated; -crf = quality; yuv420p = plays everywhere)
and the GIF from it (2 frames per second, 480 pixels wide, one palette of 256 colors, no dither):
    ffmpeg -i movie.mp4 -vf "fps=2,scale='min(480,iw)':-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];[b][p]paletteuse=dither=none" \\
           -loop 0 -y movie.gif

Called by: visualization/make_mcs_animation.py and make_cell_animation.py (through mcs_animation.py), visualization/pngs_to_animation.py.
Calls: ffmpeg, pandas.
"""
import argparse
import glob
import math
import os
import shlex
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
def list_png_files(frames, pattern='*.png'):
    """
    PNG files from a directory (all pattern files in it, sorted), a glob pattern (sorted), a file, or a list of these: the
    files of a list are kept in the order given.
    """
    items = [frames] if isinstance(frames, (str, os.PathLike)) else list(frames)
    files = []
    for item in items:
        item = os.fspath(item)
        if os.path.isdir(item):
            files += sorted(glob.glob(os.path.join(item, pattern)))
        elif not os.path.exists(item) and glob.has_magic(item):
            files += sorted(glob.glob(item))
        else:
            files.append(item)
    return files

#-----------------------------------------------------------------------
def video_frame_rate(playback_fps, video_fps=None):
    """
    Frame rate of the video file. None: playback_fps repeated to at least 10 frames per second with a whole number of
    repeats for every PNG file (2 -> 10, 3 -> 12, 12 -> 12). Players handle very low frame rates badly, so each PNG is
    repeated; the repeats do not change the speed and cost almost no file size. Keep a given video_fps a whole multiple
    of playback_fps (below it, PNG files are dropped).
    """
    if video_fps is not None:
        return video_fps
    return playback_fps * max(1, math.ceil(10 / playback_fps))

#-----------------------------------------------------------------------
def scale_filter(width):
    """ffmpeg scale filter: at most `width` pixels wide (never enlarged), the height follows, both sizes even (yuv420p needs it)."""
    if width:
        return f"scale='min({int(width) // 2 * 2},trunc(iw/2)*2)':-2"
    return 'scale=trunc(iw/2)*2:trunc(ih/2)*2'

#-----------------------------------------------------------------------
def make_mp4(png_files, out_file, playback_fps=2, frame_stride=1, video_fps=None, crf=24, preset=None, width=1920):
    """
    mp4 (H.264, yuv420p) from PNG files, e.g. make_mp4('frames/', 'movie.mp4') or make_mp4('frames/mcs_*.png', 'movie.mp4').

    Args:
        png_files: a list of PNG files in time order, or a directory (all its PNG files), or a glob pattern (see list_png_files)
        out_file: string, output file name
        playback_fps: PNG files shown per second = the speed of the animation (2: one PNG every 0.5 s)
        frame_stride: use every Nth PNG file
        video_fps: frame rate of the file, None = playback_fps repeated to at least 10 frames per second (video_frame_rate)
        crf: quality of libx264 (lower = better and larger, 18-28 is useful; 24 is a compromise of quality and size)
        preset: x264 speed / compression preset (ultrafast ... medium ... veryslow); None = no -preset option (x264 default,
            medium). A slower preset made the file only about 1 % smaller at the same CRF in the tests of docs/visualization/animations.md.
        width: at most this many pixels wide, never enlarged, the height follows (None = the size of the PNG files)

    What the ffmpeg command has (see the module docstring for the same by hand): the PNG files are read from a list file with
    the input rate playback_fps (-r before -i), the video is scaled (scale_filter) to even sizes, coded with libx264 at
    the file frame rate video_frame_rate (-r after the codec: each PNG is repeated), CRF `crf`, pixel format yuv420p.

    Returns:
        return_code, command (list)
    """
    files = list_png_files(png_files)[::max(1, int(frame_stride))]
    if not files:
        raise ValueError(f"no PNG files in {png_files}")
    fps_out = video_frame_rate(playback_fps, video_fps)
    # Create a temporary file list for FFmpeg (the concat demuxer)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_filelist = f.name
        for png_file in files:
            f.write(f"file '{png_file}'\n")
    ffmpeg_cmd = [
        find_ffmpeg(), '-hide_banner', '-loglevel', 'error', '-stats',   # (the concat demuxer warns about the missing time stamps of PNG files; the frames are right)
        '-f', 'concat', '-safe', '0',
        '-r', f'{playback_fps:g}',          # input frame rate: PNG files per second
        '-i', temp_filelist,
        '-vf', scale_filter(width),
        '-c:v', 'libx264',
        '-r', f'{fps_out:g}',               # frame rate of the file
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
    cmd = [find_ffmpeg(), '-hide_banner', '-loglevel', 'error', '-ss', str(start)]    # (error level: palettegen warns "Duped color" for frames with fewer colors than the palette)
    if duration is not None:
        cmd += ['-t', str(duration)]
    cmd += ['-i', source, '-vf', vf, '-loop', '0', '-y', out_file]
    return subprocess.run(cmd).returncode, cmd

#-----------------------------------------------------------------------
def command_text(cmd):
    """A command (list) as one line that can be pasted into a shell (double quotes keep the single quotes of the ffmpeg filters readable)."""
    def quote(arg):
        arg = str(arg)
        return f'"{arg}"' if "'" in arg and not any(c in arg for c in '"$`\\!') else shlex.quote(arg)
    return ' '.join(quote(c) for c in cmd)

#-----------------------------------------------------------------------
def plain_mp4_command(frames_glob, out_file, playback_fps=2, video_fps=None, crf=24, preset=None, width=1920):
    """The ffmpeg command line (text) that makes the same mp4 from the PNG files `frames_glob` (a glob pattern), for reading and pasting."""
    cmd = (f"ffmpeg -framerate {playback_fps:g} -pattern_type glob -i {shlex.quote(frames_glob)} -vf \"{scale_filter(width)}\" "
           f"-c:v libx264 -r {video_frame_rate(playback_fps, video_fps):g} -crf {crf} -pix_fmt yuv420p")
    if preset:
        cmd += f" -preset {preset}"
    return cmd + f" -y {shlex.quote(out_file)}"

#-----------------------------------------------------------------------
def plain_gif_command(mp4_file, gif_file, start=0.0, duration=None, fps=2, width=800, colors=256, dither='none'):
    """The ffmpeg command line (text) that makes the same GIF from the mp4 (palette 'global', as make_gif)."""
    vf = (f"fps={fps:g},scale='min({int(width)},iw)':-1:flags=lanczos,split[a][b];[a]palettegen=max_colors={int(colors)}:stats_mode=full[p];"
          f"[b][p]paletteuse=dither={dither}")
    cut = (f" -ss {start:g}" if start else '') + (f" -t {duration:g}" if duration is not None else '')
    return f"ffmpeg{cut} -i {shlex.quote(mp4_file)} -vf \"{vf}\" -loop 0 -y {shlex.quote(gif_file)}"

#-----------------------------------------------------------------------
def make_animation(png_files, mp4_file, playback_fps=2, frame_stride=1, video_fps=None, video_quality=24, output_width=1920,
                   video_preset=None, gif_width=None, gif_fps=None, gif_start=0.0, gif_duration=None, gif_colors=256,
                   gif_dither='none'):
    """
    The mp4 of the PNG files and, if gif_width is given, a GIF of it next to it (same name, .gif), with the settings printed.
    This is what the wrappers (visualization/make_*_animation.py) and visualization/pngs_to_animation.py run. The arguments
    are the settings of the module docstring; gif_fps None = playback_fps; gif_start and gif_duration [s] select the part of
    the mp4 that becomes the GIF (the first gif_duration seconds are the first gif_duration * playback_fps PNG files).

    Returns:
        0 when everything was made, otherwise the return code of the failing ffmpeg command
    """
    files = list_png_files(png_files)[::max(1, int(frame_stride))]
    if not files:
        raise ValueError(f"no PNG files in {png_files}")
    fps_out = video_frame_rate(playback_fps, video_fps)
    print("🎬 Animation settings:")
    print(f"   Playback: {playback_fps:g} PNG files per second (the speed), every {frame_stride}th PNG: {len(files)} files, "
          f"duration {len(files) / playback_fps:.0f} s")
    print(f"   Video file: {fps_out:g} frames per second (each PNG repeated {fps_out / playback_fps:g} times, the speed is the same)")
    print(f"   Video quality (CRF): {video_quality} (lower = better and larger), x264 preset {video_preset}")
    print(f"   Output width: {'at most ' + str(output_width) if output_width else 'the size of the PNG files'} pixels (never enlarged, height auto-scaled)")
    print(f"Using {len(files)} PNG files from {files[0]} to {files[-1]}")
    returncode, ffmpeg_cmd = make_mp4(files, mp4_file, playback_fps=playback_fps, frame_stride=1, video_fps=video_fps,
                                      crf=video_quality, preset=video_preset, width=output_width)
    print(f"FFmpeg command: {command_text(ffmpeg_cmd)}")
    if returncode != 0:
        print(f"❌ Error: FFmpeg failed with exit code {returncode}")
        return returncode
    print(f"✅ Animation created successfully: {mp4_file} ({os.path.getsize(mp4_file) / 1e6:.1f} MB)")
    if gif_width:
        gif_file = os.path.splitext(mp4_file)[0] + '.gif'
        returncode, gif_cmd = make_gif(mp4_file, gif_file, start=gif_start, duration=gif_duration, fps=gif_fps or playback_fps,
                                       width=gif_width, colors=gif_colors, dither=gif_dither)
        if returncode != 0:
            print(f"❌ Error: FFmpeg failed to make the GIF with exit code {returncode}")
            return returncode
        print(f"✅ GIF created: {gif_file} ({os.path.getsize(gif_file) / 1e6:.1f} MB)")
    return 0

#-----------------------------------------------------------------------
def main(argv=None):
    """Command line of visualization/pngs_to_animation.py: PNG files -> mp4 (and GIF), the settings of the wrappers."""
    ap = argparse.ArgumentParser(
        description="Make an mp4 (and a GIF) from PNG files with the settings of the make_*_animation.py wrappers. "
                    "FRAMES is a directory (all its PNG files), a quoted glob pattern or a list of files, in time order; "
                    "the last argument is the mp4 file.",
        epilog="Example: pngs_to_animation.py 'frames/mcs_*.png' movie.mp4 --playback_fps 2 --video_quality 24 --gif_width 480 --gif_duration 12")
    ap.add_argument('paths', nargs='+', metavar='FRAMES... OUT.mp4', help="the PNG files (directory, quoted glob pattern, or files), then the mp4 file")
    ap.add_argument('--playback_fps', type=float, default=2, help="PNG files shown per second = the speed of the animation (default 2)")
    ap.add_argument('--frame_stride', type=int, default=1, help="use every Nth PNG file (default 1: all)")
    ap.add_argument('--video_fps', type=float, default=None,
                    help="frame rate of the video file (default: playback_fps repeated to at least 10 frames per second, a whole number of repeats)")
    ap.add_argument('--video_quality', type=int, default=24, help="CRF of libx264, lower = better and larger (default 24; 18-28 is useful)")
    ap.add_argument('--output_width', type=int, default=1920, help="at most this many pixels wide, never enlarged (default 1920; 0: the size of the PNG files)")
    ap.add_argument('--video_preset', default=None, help="x264 preset (default: none = the x264 default, medium)")
    ap.add_argument('--gif_width', type=int, default=None, help="also make a GIF of this width (next to the mp4, same name)")
    ap.add_argument('--gif_fps', type=float, default=None, help="frames per second of the GIF (default: playback_fps)")
    ap.add_argument('--gif_start', type=float, default=0.0, help="start of the GIF in the mp4 [s]")
    ap.add_argument('--gif_duration', type=float, default=None, help="length of the GIF [s] (default: to the end)")
    ap.add_argument('--gif_colors', type=int, default=256, help="colors of the GIF (default 256)")
    ap.add_argument('--gif_dither', default='none', help="dither of the GIF: none (default), bayer:bayer_scale=3, sierra2_4a, floyd_steinberg")
    ap.add_argument('--print_command', action='store_true', help="only print the plain ffmpeg commands (to run by hand), make nothing")
    args = ap.parse_args(argv)
    if len(args.paths) < 2 or not args.paths[-1].lower().endswith('.mp4'):
        ap.error("give the PNG files, then the name of the mp4 file (ending in .mp4) as the last argument")
    frames, mp4_file = args.paths[:-1], args.paths[-1]
    width = args.output_width or None
    if args.print_command:
        pattern = frames[0] if len(frames) == 1 else '<your PNG files>'
        pattern = os.path.join(pattern, '*.png') if os.path.isdir(pattern) else pattern
        print(plain_mp4_command(pattern, mp4_file, args.playback_fps, args.video_fps, args.video_quality, args.video_preset, width))
        if args.gif_width:
            print(plain_gif_command(mp4_file, os.path.splitext(mp4_file)[0] + '.gif', args.gif_start, args.gif_duration,
                                    args.gif_fps or args.playback_fps, args.gif_width, args.gif_colors, args.gif_dither))
        return 0
    files = list_png_files(frames)
    if not files:
        sys.exit(f"ERROR: no PNG files in {' '.join(frames)}")
    os.makedirs(os.path.dirname(os.path.abspath(mp4_file)), exist_ok=True)
    return make_animation(files, mp4_file, playback_fps=args.playback_fps, frame_stride=args.frame_stride, video_fps=args.video_fps,
                          video_quality=args.video_quality, output_width=width, video_preset=args.video_preset,
                          gif_width=args.gif_width, gif_fps=args.gif_fps, gif_start=args.gif_start, gif_duration=args.gif_duration,
                          gif_colors=args.gif_colors, gif_dither=args.gif_dither)
