#!/usr/bin/env python
r"""
Make the task list and the Slurm job array script for MCS track quicklooks of several years: one task (one line, one job of the array) per
year, each a plot_mcs_tracks.py command for that year's tracking config file; the PNG files of a year go to {fig_dir}/{year}/. The
GridRad record is the use it was written for (Tb + radar reflectivity maps, product tbze, one config file per year), but any product and
any set of per-year config files work.

Usage (from this folder; {year} in the config pattern is replaced by each year):
    python make_mcs_quicklook_slurm_tasks.py '/path/to/config_gridrad_mcs_{year}.yml' /path/to/quicklooks/ --years 2018 2020 --months 4 8 \
        --product tbze --figsize 10 13 --account YOUR_ACCOUNT --qos regular --time 01:00:00 \
        --env_command 'conda activate /path/to/your/pyflextrkr/environment'
This writes tasklist_mcs_tbze_quicklook.txt and slurm.submit_mcs_tbze_quicklook.sh (in --output_dir, default: the current folder) and prints
the sbatch command; --submit runs it. Everything about the cluster is an option (nothing is assumed): without --account, --qos, --constraint,
--env_command or --mail_user the script has no such line.

The task of the year 2018 with the options above is
    python /path/to/PyFLEXTRKR/visualization/plot_mcs_tracks.py -s 2018-04-01T00 -e 2018-09-01T00 -c /path/to/config_gridrad_mcs_2018.yml \
        --product tbze --layout top_bottom -p 1 --figsize 10 13 --output /path/to/quicklooks/2018/ --figbasename mcs_
and the jobs run it with `sed -n "$SLURM_ARRAY_TASK_ID"p tasklist...` (the line of the task number). --months FIRST LAST are both months of the
period (4 8: April to August); the monthly mp4 files of the PNG files are made by make_gridrad_mcstracks_animation.py.
"""
import argparse
import os
import subprocess
import sys
import textwrap

PLOT_SCRIPT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plot_mcs_tracks.py'))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config_pattern', help="tracking config file of a year, with {year} in it, e.g. '/path/config_mcs_{year}.yml'")
    ap.add_argument('fig_dir', help='folder for the PNG files: the files of a year go to {fig_dir}/{year}/')
    ap.add_argument('--years', type=int, nargs=2, required=True, metavar=('FIRST', 'LAST'), help='first and last year')
    ap.add_argument('--months', type=int, nargs=2, default=[1, 12], metavar=('FIRST', 'LAST'),
                    help='first and last month of the period of each year, both included (default: 1 12)')
    ap.add_argument('--product', default='tbze', choices=['tbpf', 'tbze', 'tb'], help='plot_mcs_tracks.py --product (default: tbze)')
    ap.add_argument('--layout', default='top_bottom', choices=['top_bottom', 'left_right'], help='2 panels above each other or side by side')
    ap.add_argument('--figsize', type=float, nargs=2, default=[10, 13], metavar=('WIDTH', 'HEIGHT'), help='figure size in inches (default: 10 13)')
    ap.add_argument('--figbasename', default='mcs_', help='file name prefix of the PNG files (default: mcs_)')
    ap.add_argument('--output_dir', default='.', help='folder for the task list and the Slurm script (default: the current folder)')
    ap.add_argument('--name', default=None, help='job name and the name of the two files (default: mcs_{product}_quicklook)')
    # Slurm: only what is given is written
    ap.add_argument('--account', default=None, help='#SBATCH -A')
    ap.add_argument('--qos', default=None, help='#SBATCH -q (e.g. regular, debug)')
    ap.add_argument('--time', default='01:00:00', help='#SBATCH -t of one task (default: 01:00:00)')
    ap.add_argument('--constraint', default=None, help='#SBATCH -C (e.g. cpu on Perlmutter)')
    ap.add_argument('--cpus', type=int, default=128, help='#SBATCH -n of one task, one exclusive node (default: 128)')
    ap.add_argument('--mail_user', default=None, help='#SBATCH --mail-user (and --mail-type=END)')
    ap.add_argument('--env_command', default=None, help="command that activates the PyFLEXTRKR environment, e.g. 'conda activate /path/to/env'")
    ap.add_argument('--submit', action='store_true', help='run sbatch --array=1-N on the script')
    return ap.parse_args(argv)


def period_of(year, first_month, last_month):
    """Start and end (the first day of the month after the last month) of the quicklooks of a year."""
    end_year, end_month = (year + 1, 1) if last_month == 12 else (year, last_month + 1)
    return f'{year}-{first_month:02d}-01T00', f'{end_year}-{end_month:02d}-01T00'


def task_lines(args):
    """One plot_mcs_tracks.py command per year."""
    fig_dir = args.fig_dir.rstrip('/')
    lines = []
    for year in range(args.years[0], args.years[1] + 1):
        start, end = period_of(year, *args.months)
        lines.append(f"python {PLOT_SCRIPT} -s {start} -e {end} -c {args.config_pattern.format(year=year)} --product {args.product} "
                     f"--layout {args.layout} -p 1 --figsize {args.figsize[0]:g} {args.figsize[1]:g} --output {fig_dir}/{year}/ "
                     f"--figbasename {args.figbasename}")
    return lines


def slurm_script(name, task_file, ntasks, args):
    """The text of the job array script: the job with array index i runs line i of the task file."""
    header = [f'#SBATCH -J {name}']
    for flag, value in (('-A', args.account), ('-q', args.qos), ('-t', args.time), ('-C', args.constraint)):
        if value:
            header.append(f'#SBATCH {flag} {value}')
    header += ['#SBATCH -N 1', f'#SBATCH -n {args.cpus}', '#SBATCH --exclusive', f'#SBATCH --output=log_{name}_%A_%a.log']
    if args.mail_user:
        header += ['#SBATCH --mail-type=END', f'#SBATCH --mail-user={args.mail_user}']
    header.append(f'#SBATCH --array=1-{ntasks}')
    environment = args.env_command or '# activate the PyFLEXTRKR environment here (--env_command)'
    body = textwrap.dedent(f"""\

        date
        {environment}

        # The line of the task file with the number of this job of the array
        LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p {task_file})
        echo $LINE
        # Run the line as a command
        $LINE

        date
        """)
    return '#!/bin/bash\n' + '\n'.join(header) + '\n' + body


def main(argv=None):
    args = parse_args(argv)
    if '{year}' not in args.config_pattern:
        sys.exit("ERROR: the config pattern has no {year}")
    if args.years[0] > args.years[1] or not (1 <= args.months[0] <= args.months[1] <= 12):
        sys.exit("ERROR: --years and --months are first last, in order (months 1-12)")
    name = args.name or f'mcs_{args.product}_quicklook'
    os.makedirs(args.output_dir, exist_ok=True)
    task_file = os.path.abspath(os.path.join(args.output_dir, f'tasklist_{name}.txt'))
    slurm_file = os.path.abspath(os.path.join(args.output_dir, f'slurm.submit_{name}.sh'))
    lines = task_lines(args)
    with open(task_file, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(slurm_file, 'w') as f:
        f.write(slurm_script(name, task_file, len(lines), args))
    print(f'{len(lines)} tasks: {task_file}')
    print(f'Slurm script: {slurm_file}')
    command = ['sbatch', f'--array=1-{len(lines)}', slurm_file]
    print('Submit with: ' + ' '.join(command))
    if args.submit:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
