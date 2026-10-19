#!/usr/bin/env python
r"""
One mp4 per month of the GridRad MCS track quicklooks: Tb + radar reflectivity maps that plot_mcs_tracks.py made, one folder of PNG files
per year (the quicklook batch of make_mcs_quicklook_slurm_tasks.py writes {frames_dir}/{year}/mcs_YYYYMMDD_HHMMSS.png), so that a multi-year
record can be looked at month by month.

Usage (from this folder; frames in /path/to/gridrad/quicklooks/2018/, .../2019/, ...; the mp4 files go to the output folder):
    python make_gridrad_mcstracks_animation.py /path/to/gridrad/quicklooks/ /path/to/gridrad/animations/ --years 2018 2020
    python make_gridrad_mcstracks_animation.py /path/to/gridrad/quicklooks/ /path/to/gridrad/animations/ --years 2019 2019 --months 5 6 --playback_fps 4

--years first last and --months first last (default: April to August, the warm season of the GridRad record) choose the months; for each
one, {output_dir}/mcs_YYYYMM.mp4 is made from the PNG files {frames_dir}/{year}/{figbasename}YYYYMM*.png. A month without PNG files is
skipped, and an mp4 that exists is replaced. The video settings are those of the animation wrappers (pyflextrkr/visualization/animation.py):
--playback_fps PNG files per second (the speed: 2 is two hourly files per second), --video_quality the CRF of libx264 (lower = better and
larger), --output_width the width in pixels (never enlarged, 1200 here: these maps are large).

One month by hand with ffmpeg alone (what this script runs for 2018-04; -framerate 2: PNG files per second = playback_fps; -r 10: frames per
second of the file, each PNG is repeated 5 times; -crf 20: video_quality; the scale filter makes the width at most 1200 and both sizes even,
which yuv420p needs):
    ffmpeg -framerate 2 -pattern_type glob -i '/path/to/gridrad/quicklooks/2018/mcs_201804*.png' \
        -vf "scale='min(1200,trunc(iw/2)*2)':-2" -c:v libx264 -r 10 -crf 20 -pix_fmt yuv420p -y /path/to/gridrad/animations/mcs_201804.mp4
or with pngs_to_animation.py:
    python ../pngs_to_animation.py '/path/to/gridrad/quicklooks/2018/mcs_201804*.png' /path/to/gridrad/animations/mcs_201804.mp4 --output_width 1200 --video_quality 20
"""
import argparse
import calendar
import os
import sys
from pyflextrkr.visualization.animation import frames_in_range, make_mp4


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('frames_dir', help='folder with one subfolder per year of PNG files {figbasename}YYYYMMDD_HHMMSS.png')
    ap.add_argument('output_dir', help='folder for the mp4 files (made if it does not exist)')
    ap.add_argument('--years', type=int, nargs=2, required=True, metavar=('FIRST', 'LAST'), help='first and last year')
    ap.add_argument('--months', type=int, nargs=2, default=[4, 8], metavar=('FIRST', 'LAST'), help='first and last month (default: 4 8)')
    ap.add_argument('--figbasename', default='mcs_', help='file name prefix of the PNG files (default: mcs_)')
    ap.add_argument('--playback_fps', type=float, default=2, help='PNG files shown per second = the speed (default: 2)')
    ap.add_argument('--video_quality', type=int, default=20, help='CRF of libx264, lower = better and larger (default: 20)')
    ap.add_argument('--output_width', type=int, default=1200, help='at most this many pixels wide, never enlarged (default: 1200)')
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    n_made = 0
    for year in range(args.years[0], args.years[1] + 1):
        for month in range(args.months[0], args.months[1] + 1):
            first_day = f'{year}-{month:02d}-01T00:00:00'
            last_day = f'{year}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}T23:59:59'
            png_files = frames_in_range(os.path.join(args.frames_dir, str(year)) + os.sep, args.figbasename, first_day, last_day)
            if not png_files:
                print(f'{year}{month:02d}: no PNG files, skipped')
                continue
            mp4_file = os.path.join(args.output_dir, f'mcs_{year}{month:02d}.mp4')
            print(f'{year}{month:02d}: {len(png_files)} PNG files -> {mp4_file}')
            returncode, _ = make_mp4(png_files, mp4_file, playback_fps=args.playback_fps, crf=args.video_quality, width=args.output_width)
            if returncode != 0:
                sys.exit(f'ffmpeg failed for {year}{month:02d}')
            n_made += 1
    print(f'{n_made} mp4 files in {args.output_dir}')


if __name__ == "__main__":
    main()
