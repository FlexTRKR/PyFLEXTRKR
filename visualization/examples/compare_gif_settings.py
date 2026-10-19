#!/usr/bin/env python
"""
Compare GIF settings (palette, number of colors, dither) on your own animation: for each setting the GIF that make_mcs_animation
makes from the mp4 (animation.make_gif), its size and its SSIM / PSNR against the mp4 (so that a smaller file is not a worse
GIF). Look at the colorbars and the gray shades of the GIFs too (--keep): a GIF has 256 colors at most.

Usage (a compute node is not needed, the GIFs are small):
    python compare_gif_settings.py /path/to/animation.mp4 --width 560 --fps 2
    python compare_gif_settings.py animation.mp4 --width 800 --fps 2 --settings global/256/none global/256/sierra2_4a frame/256/bayer:bayer_scale=3 --keep

--settings: palette/colors/dither[/diff_mode], palette = global (one palette for all frames) or frame (a palette for each
frame); dither as in the ffmpeg filter paletteuse (none, bayer:bayer_scale=N, sierra2_4a, floyd_steinberg, ...); diff_mode
rectangle stores only the changed rectangle of each frame. The result of this test on the example animations is in
docs/visualization/animations.md.
"""
import argparse
import os
import re
import subprocess
import time

from pyflextrkr.visualization.animation import find_ffmpeg, make_gif

DEFAULT_SETTINGS = [
    'global/256/none', 'global/256/bayer:bayer_scale=3', 'global/256/bayer:bayer_scale=5', 'global/256/sierra2_4a',
    'global/256/floyd_steinberg', 'global/128/sierra2_4a', 'frame/256/sierra2_4a', 'frame/256/bayer:bayer_scale=3',
    'global/256/sierra2_4a/rectangle', 'global/256/bayer:bayer_scale=3/rectangle',
]


def ssim_psnr(gif_file, source, width, fps):
    """SSIM and PSNR of the GIF against the video, both at `width` pixels and `fps` frames per second."""
    graph = (f"[0:v]fps={fps},scale={width}:-2:flags=bicubic,format=yuv420p,split=2[g1][g2];"
             f"[1:v]fps={fps},scale={width}:-2:flags=lanczos,format=yuv420p,split=2[r1][r2];"
             f"[g1][r1]ssim[s];[g2][r2]psnr[p]")
    cmd = [find_ffmpeg(), '-hide_banner', '-nostats', '-i', gif_file, '-i', source, '-filter_complex', graph,
           '-map', '[s]', '-f', 'null', '-', '-map', '[p]', '-f', 'null', '-']
    result = subprocess.run(cmd, capture_output=True, text=True)
    ssim = re.findall(r'SSIM Y:[\d.]+ \([\d.]+\) U:[\d.]+ \([\d.]+\) V:[\d.]+ \([\d.]+\) All:([\d.]+)', result.stderr)
    psnr = re.findall(r'PSNR y:[\d.]+ u:[\d.]+ v:[\d.]+ average:([\d.]+)', result.stderr)
    return (float(ssim[-1]) if ssim else float('nan')), (float(psnr[-1]) if psnr else float('nan'))


def main():
    parser = argparse.ArgumentParser(description="Size, time and SSIM of the GIF for several palette / color / dither settings.")
    parser.add_argument('video', help="the mp4 file (or any video that ffmpeg reads)")
    parser.add_argument('--width', type=int, default=560, help="width of the GIF in pixels")
    parser.add_argument('--fps', type=float, default=2, help="frames per second of the GIF (the PNG files per second of the mp4)")
    parser.add_argument('--settings', nargs='+', default=DEFAULT_SETTINGS, help="palette/colors/dither[/diff_mode]")
    parser.add_argument('--keep', action='store_true', help="keep the GIF files (in the current directory)")
    args = parser.parse_args()

    print(f"{args.video}: width {args.width}, {args.fps:g} frames per second")
    print(f"{'setting':44s} {'size MB':>8s} {'vs first':>9s} {'time s':>7s} {'SSIM':>8s} {'PSNR':>6s}")
    first = None
    for setting in args.settings:
        parts = setting.split('/')
        palette, colors, dither = parts[0], int(parts[1]), parts[2]
        diff_mode = parts[3] if len(parts) > 3 else None
        out_file = 'gif_' + re.sub(r'[^A-Za-z0-9]+', '_', setting) + '.gif'
        t0 = time.time()
        returncode, _ = make_gif(args.video, out_file, fps=args.fps, width=args.width, colors=colors, dither=dither, palette=palette,
                                 diff_mode=diff_mode)
        elapsed = time.time() - t0
        if returncode != 0:
            print(f"{setting:44s} ffmpeg failed")
            continue
        size = os.path.getsize(out_file) / 1e6
        ssim, psnr = ssim_psnr(out_file, args.video, args.width, args.fps)
        first = first or size
        print(f"{setting:44s} {size:8.2f} {100 * (size / first - 1):+8.1f}% {elapsed:7.1f} {ssim:8.5f} {psnr:6.2f}", flush=True)
        if not args.keep:
            os.unlink(out_file)


if __name__ == "__main__":
    main()
