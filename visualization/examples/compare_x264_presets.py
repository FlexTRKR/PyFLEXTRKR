#!/usr/bin/env python
"""
Compare the x264 presets (or CRF values) of the animation wrappers on your own PNG frames: for each setting the same mp4 as
make_mcs_animation.py makes (animation.make_mp4), its size, the encoding time and its SSIM against the source frames (so that
a smaller file is not a worse video).

Usage (a compute node; 100-200 frames are enough):
>python compare_x264_presets.py '/path/to/frames/mcs_tbpf_*.png' --start 100 --n 120 --crf 24 --in_fps 6 --out_fps 12 --width 1920
>python compare_x264_presets.py '/path/to/frames/*.png' --presets none slow --crf 20 22 24 26 28

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

from pyflextrkr.visualization.animation import find_ffmpeg, make_mp4


def ssim_psnr(files, video, in_fps, out_fps, width):
    """SSIM and PSNR of the video against the source frames, scaled and held to the frame rate of the video the same way."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        listfile = f.name
        for p in files:
            f.write(f"file '{p}'\n")
    graph = (f"[1:v]scale={width}:-2,format=yuv420p,fps={out_fps},split=2[r1][r2];"
             f"[0:v]split=2[a][b];[a][r1]ssim[s];[b][r2]psnr[p]")
    cmd = [find_ffmpeg(), '-hide_banner', '-nostats', '-i', video, '-f', 'concat', '-safe', '0', '-r', str(in_fps), '-i', listfile,
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
    parser.add_argument('--in_fps', type=float, default=2, help="PNG frames shown per second (input_framerate of the wrappers)")
    parser.add_argument('--out_fps', type=float, default=10, help="frame rate of the video file (output_framerate)")
    parser.add_argument('--width', type=int, default=1920, help="output width in pixels (output_width)")
    parser.add_argument('--keep', action='store_true', help="keep the mp4 files (in the current directory)")
    args = parser.parse_args()

    files = sorted(glob.glob(args.frames))[args.start:args.start + args.n]
    if not files:
        raise SystemExit(f"no frames: {args.frames}")
    print(f"{len(files)} frames from {os.path.basename(files[0])} to {os.path.basename(files[-1])}, width {args.width}, "
          f"{args.in_fps:g} -> {args.out_fps:g} frames per second")
    print(f"{'preset':10s} {'CRF':>4s} {'size MB':>9s} {'vs first':>9s} {'time s':>7s} {'SSIM':>8s} {'PSNR':>6s}")
    first = None
    for crf in args.crf:
        for preset in args.presets:
            out_file = f"x264_{preset}_crf{crf}.mp4"
            t0 = time.time()
            returncode, _ = make_mp4(files, out_file, input_framerate=args.in_fps, output_framerate=args.out_fps, crf=crf,
                                     width=args.width, preset=None if preset == 'none' else preset)
            elapsed = time.time() - t0
            if returncode != 0:
                print(f"{preset:10s} {crf:4d} ffmpeg failed")
                continue
            size = os.path.getsize(out_file) / 1e6
            ssim, psnr = ssim_psnr(files, out_file, args.in_fps, args.out_fps, args.width)
            first = first or size
            print(f"{preset:10s} {crf:4d} {size:9.2f} {100 * (size / first - 1):+8.1f}% {elapsed:7.1f} {ssim:8.5f} {psnr:6.2f}", flush=True)
            if not args.keep:
                os.unlink(out_file)


if __name__ == "__main__":
    main()
