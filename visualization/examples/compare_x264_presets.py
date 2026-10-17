#!/usr/bin/env python
"""
Compare the x264 presets (or CRF values) of the animation wrappers on your own PNG frames: for each setting the same mp4 as
make_mcs_animation.py makes (animation.make_mp4), its size, the encoding time and its SSIM against the source frames (so that
a smaller file is not a worse video).

Usage (the login node is enough for a few dozen frames; 100-200 frames are enough for a test):
>python compare_x264_presets.py '/path/to/frames/mcs_tbpf_*.png' --start 100 --n 120 --crf 24 --playback_fps 6 --width 1920
>python compare_x264_presets.py '/path/to/frames/*.png' --presets none slow --crf 20 22 24 26 28
>python compare_x264_presets.py '/path/to/frames/*.png' --crf 24 --playback_fps 2 --video_fps 2 10      # file frame rate 2 or 10

--presets: x264 presets; none = no -preset option (the x264 default, medium; video_preset = None in the wrappers).
The result of this test on the example frames of the CONUS and global MCS maps is in docs/visualization/animations.md.
"""
import argparse
import glob
import os
import re
import subprocess
import tempfile
import time

from pyflextrkr.visualization.animation import find_ffmpeg, make_mp4, scale_filter, video_frame_rate


def ssim_psnr(files, video, playback_fps, video_fps, width):
    """SSIM and PSNR of the video against the source frames, scaled (scale_filter of make_mp4) and held to the frame rate of the video the same way."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        listfile = f.name
        for p in files:
            f.write(f"file '{p}'\n")
    graph = (f"[1:v]{scale_filter(width)},format=yuv420p,fps={video_fps},split=2[r1][r2];"
             f"[0:v]split=2[a][b];[a][r1]ssim[s];[b][r2]psnr[p]")
    cmd = [find_ffmpeg(), '-hide_banner', '-nostats', '-i', video, '-f', 'concat', '-safe', '0', '-r', f'{playback_fps:g}', '-i', listfile,
           '-filter_complex', graph, '-map', '[s]', '-f', 'null', '-', '-map', '[p]', '-f', 'null', '-']
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(listfile)
    ssim = re.findall(r'SSIM Y:[\d.]+ \([\d.]+\) U:[\d.]+ \([\d.]+\) V:[\d.]+ \([\d.]+\) All:([\d.]+)', result.stderr)
    psnr = re.findall(r'PSNR y:[\d.]+ u:[\d.]+ v:[\d.]+ average:([\d.]+)', result.stderr)
    return (float(ssim[-1]) if ssim else float('nan')), (float(psnr[-1]) if psnr else float('nan'))


def main():
    parser = argparse.ArgumentParser(description="Size, encoding time and SSIM of the mp4 for several x264 presets / CRF values.")
    parser.add_argument('frames', help="PNG frames, a glob pattern in quotes (sorted by name)")
    parser.add_argument('--start', type=int, default=0, help="index of the first frame")
    parser.add_argument('--n', type=int, default=120, help="number of frames")
    parser.add_argument('--presets', nargs='+', default=['ultrafast', 'veryfast', 'fast', 'none', 'slow', 'slower', 'veryslow'])
    parser.add_argument('--crf', type=int, nargs='+', default=[20], help="CRF value(s)")
    parser.add_argument('--playback_fps', type=float, default=2, help="PNG files shown per second = the speed (playback_fps of the wrappers)")
    parser.add_argument('--video_fps', type=float, nargs='+', default=[None],
                        help="frame rate(s) of the video file; default: automatic (playback_fps repeated to at least 10 per second)")
    parser.add_argument('--width', type=int, default=1920, help="at most this many pixels wide, never enlarged (output_width of the wrappers)")
    parser.add_argument('--keep', action='store_true', help="keep the mp4 files (in the current directory)")
    args = parser.parse_args()

    files = sorted(glob.glob(args.frames))[args.start:args.start + args.n]
    if not files:
        raise SystemExit(f"no frames: {args.frames}")
    print(f"{len(files)} frames from {os.path.basename(files[0])} to {os.path.basename(files[-1])}, width at most {args.width}, "
          f"{args.playback_fps:g} PNG files per second: {len(files) / args.playback_fps:.0f} s")
    print(f"{'preset':10s} {'CRF':>4s} {'file fps':>8s} {'size MB':>9s} {'vs first':>9s} {'time s':>7s} {'SSIM':>8s} {'PSNR':>6s}")
    first = None
    for crf in args.crf:
        for video_fps in args.video_fps:
            fps_out = video_frame_rate(args.playback_fps, video_fps)
            for preset in args.presets:
                out_file = f"x264_{preset}_crf{crf}_fps{fps_out:g}.mp4"
                t0 = time.time()
                returncode, _ = make_mp4(files, out_file, playback_fps=args.playback_fps, video_fps=video_fps, crf=crf,
                                         width=args.width, preset=None if preset == 'none' else preset)
                elapsed = time.time() - t0
                if returncode != 0:
                    print(f"{preset:10s} {crf:4d} ffmpeg failed")
                    continue
                size = os.path.getsize(out_file) / 1e6
                ssim, psnr = ssim_psnr(files, out_file, args.playback_fps, fps_out, args.width)
                first = first or size
                print(f"{preset:10s} {crf:4d} {fps_out:8g} {size:9.2f} {100 * (size / first - 1):+8.1f}% {elapsed:7.1f} {ssim:8.5f} {psnr:6.2f}", flush=True)
                if not args.keep:
                    os.unlink(out_file)


if __name__ == "__main__":
    main()
