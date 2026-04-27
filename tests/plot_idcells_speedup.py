#!/usr/bin/env python
"""
Measure and visualise per-step speedups for idcells_reflectivity.

For each demo dataset (NEXRAD, CSAPR) this script:
  1. Times every major step with orig / fast / edt implementations
     across all available input files.
  2. Produces a two-panel bar-chart PNG:
       Top panel    : absolute timing (orig / fast / edt bars)
       Bottom panel : speedup vs orig on a log scale

Steps timed
-----------
  mod_dilate_conv_rad  : orig vs edt
  expand_conv_core     : orig vs fast vs edt
  echotop all-5        : orig vs fast (summed over 5 thresholds)
  echotop thresh=N     : orig vs fast per threshold (10, 20, 30, 40, 50 dBZ)

Usage
-----
  # Run on an existing tracking config directly (no demo data setup needed):
  python tests/plot_idcells_speedup.py \
      --config /path/to/config.yml --nfiles 10 --outdir /path/to/figures/

  # Run with default demo data paths, save figure to current directory:
  PYFLEXTRKR_TEST_DATA=~/data/demo python tests/plot_idcells_speedup.py

  # Override data root and output path:
  python tests/plot_idcells_speedup.py \
      --data_root /path/to/demo --outdir /path/to/figures

  # Limit to first 5 files per demo (quick test):
  python tests/plot_idcells_speedup.py --nfiles 5

Requirements
------------
  matplotlib, numpy  (available in the pyflextrkr environment)
  Demo data under $PYFLEXTRKR_TEST_DATA (or --data_root):
    cell_radar/nexrad/input/KHGX*.nc
    cell_radar/csapr/input/taranis_corcsapr2*.nc
  Download with:
    python tests/run_demo_tests.py --demos demo_cell_nexrad demo_cell_csapr -n 4
"""

import argparse
import os
import time
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pyflextrkr.echotop_func import echotop_height, echotop_height_fast
from pyflextrkr.ft_utilities import load_config, subset_files_timerange
from pyflextrkr.idcells_reflectivity import get_composite_reflectivity_generic
from pyflextrkr.steiner_func import (
    expand_conv_core,
    expand_conv_core_edt,
    expand_conv_core_fast,
    make_dilation_step_func,
    mod_dilate_conv_rad,
    mod_dilate_conv_rad_edt,
    mod_steiner_classification,
)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', default=None,
                    help='Path to an existing PyFLEXTRKR config YAML. '
                         'When specified, times that config directly (bypasses demo data setup).')
parser.add_argument('--data_root', default=os.environ.get('PYFLEXTRKR_TEST_DATA', ''),
                    help='Path to demo data root (default: $PYFLEXTRKR_TEST_DATA)')
parser.add_argument('--outdir', default='.',
                    help='Directory to save output figures (default: .)')
parser.add_argument('--nfiles', type=int, default=0,
                    help='Number of files per demo to time (0 = all)')
args = parser.parse_args()

DATA_ROOT = args.data_root
OUTDIR    = args.outdir
os.makedirs(OUTDIR, exist_ok=True)

if not DATA_ROOT and not args.config:
    raise SystemExit(
        "ERROR: set --data_root or export PYFLEXTRKR_TEST_DATA=/path/to/demo, "
        "or use --config to specify a config directly"
    )

# ── Demo specifications ───────────────────────────────────────────────────────
# Each entry needs:
#   name         – human-readable label for plots
#   template     – config template filename (has INPUT_DIR / TRACK_DIR placeholders)
#   data_subdir  – path under DATA_ROOT that contains the input/ folder
DEMOS = [
    {
        'name':       'NEXRAD (KHGX)',
        'template':   'config_nexrad500m_example.yml',
        'data_subdir': 'cell_radar/nexrad',
    },
    {
        'name':       'CSAPR-2 (CACTI)',
        'template':   'config_csapr500m_example.yml',
        'data_subdir': 'cell_radar/csapr',
    },
]

# ── Step labels used in plots ─────────────────────────────────────────────────
STEP_LABELS = [
    'mod_steiner',
    'expand_conv_core',
    'echotop all-5',
    'echotop thresh=10',
    'echotop thresh=20',
    'echotop thresh=30',
    'echotop thresh=40',
    'echotop thresh=50',
]


def make_config(template_name, data_subdir, data_root, outdir):
    """
    Generate a temporary config by substituting INPUT_DIR / TRACK_DIR in the
    template, exactly like run_demo_tests.py does.

    The generated file is written to the same config/ directory and named
    config_speedup_<template_name> so it doesn't collide with autotest configs.
    """
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
    template_path = os.path.join(config_dir, template_name)
    text = open(template_path).read()

    dir_input = os.path.join(data_root, data_subdir, 'input') + '/'
    dir_track = os.path.join(data_root, data_subdir, 'speedup_run') + '/'
    text = text.replace('INPUT_DIR/', dir_input)
    text = text.replace('INPUT_DIR',  dir_input)   # catch without trailing slash
    text = text.replace('TRACK_DIR/', dir_track)
    text = text.replace('TRACK_DIR',  dir_track)

    out_name = 'config_speedup_' + template_name
    out_path = os.path.join(config_dir, out_name)
    with open(out_path, 'w') as fh:
        fh.write(text)
    return out_path


def time_demo(config_path, nfiles):
    """
    Run all steps on demo data and return per-step timing lists.

    Returns
    -------
    timings: dict  {step_key: [elapsed_per_file, ...]}
    """
    config = load_config(config_path)
    all_files, _, _, _ = subset_files_timerange(
        config['clouddata_path'], config.get('databasename', ''),
        start_basetime=config.get('start_basetime'),
        end_basetime=config.get('end_basetime'),
        time_format=config['time_format'],
    )
    infiles = all_files[:nfiles] if nfiles > 0 else all_files
    print(f"  Timing {len(infiles)} file(s) …")

    timings = defaultdict(list)

    bkg_bin, conv_rad_bin = make_dilation_step_func(
        config['mindBZuse'], config['dBZforMaxConvRadius'],
        config['bkg_refl_increment'], config['conv_rad_increment'],
        config['conv_rad_start'], config['maxConvRadius'],
    )
    types_steiner = {'NO_SURF_ECHO': 1, 'WEAK_ECHO': 2, 'STRATIFORM': 3, 'CONVECTIVE': 4}
    dx = config['dx']
    dy = config['dy']
    radii_expand = np.array(config['radii_expand'])
    z_dimname = config.get('z_dimname', 'z')
    echotop_gap = config['echotop_gap']

    for input_file in infiles:
        comp_dict = get_composite_reflectivity_generic(input_file, config)
        refl            = comp_dict['refl']
        dbz3d_filt      = comp_dict['dbz3d_filt']
        height          = comp_dict['height']
        mask_goodvalues = comp_dict['mask_goodvalues']
        shape_2d        = refl.shape

        # mod_steiner with orig dilation for timing
        t0 = time.perf_counter()
        result = mod_steiner_classification(
            types_steiner, refl, mask_goodvalues, dx, dy,
            bkg_rad=config['bkgrndRadius'] * 1000,
            minZdiff=config['minZdiff'], absConvThres=config['absConvThres'],
            truncZconvThres=config['truncZconvThres'],
            weakEchoThres=config['weakEchoThres'],
            bkg_bin=bkg_bin, conv_rad_bin=conv_rad_bin,
            min_corearea=config.get('min_corearea', 0),
            min_cellarea=config.get('min_cellarea', 0),
            remove_smallcores=config.get('remove_smallcores', True),
            remove_smallcells=config.get('remove_smallcells', False),
            return_diag=False,
            convolve_method=config.get('convolve_method', 'fft'),
            dilate_method='orig',
        )
        timings['mod_steiner_orig'].append(time.perf_counter() - t0)
        timings['mod_steiner_fast'].append(timings['mod_steiner_orig'][-1])

        # Dilation sub-timing: get pre-dilation state via return_diag
        result_diag = mod_steiner_classification(
            types_steiner, refl, mask_goodvalues, dx, dy,
            bkg_rad=config['bkgrndRadius'] * 1000,
            minZdiff=config['minZdiff'], absConvThres=config['absConvThres'],
            truncZconvThres=config['truncZconvThres'],
            weakEchoThres=config['weakEchoThres'],
            bkg_bin=bkg_bin, conv_rad_bin=conv_rad_bin,
            min_corearea=config.get('min_corearea', 0),
            min_cellarea=config.get('min_cellarea', 0),
            remove_smallcores=config.get('remove_smallcores', True),
            remove_smallcells=config.get('remove_smallcells', False),
            return_diag=True,
            convolve_method=config.get('convolve_method', 'fft'),
            dilate_method='orig',
        )
        _refl_bkg   = result_diag['refl_bkg']
        _score_keep = result_diag['score']
        _score_orig = result_diag['score_orig']
        _sclass = np.zeros(refl.shape, dtype=int)
        _sclass[mask_goodvalues == 1] = types_steiner['STRATIFORM']
        _sclass[_score_orig == 1] = types_steiner['CONVECTIVE']
        _sclass[np.logical_and(refl > config['mindBZuse'],
                               refl < config['weakEchoThres'])] = types_steiner['WEAK_ECHO']
        _sclass[np.logical_and(mask_goodvalues == 1,
                               refl < config['mindBZuse'])] = types_steiner['NO_SURF_ECHO']

        t0 = time.perf_counter()
        mod_dilate_conv_rad(types_steiner, _refl_bkg, _sclass, _score_keep,
                            mask_goodvalues, dx, dy, bkg_bin, conv_rad_bin)
        timings['dilate_orig'].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        mod_dilate_conv_rad_edt(types_steiner, _refl_bkg, _sclass, _score_keep,
                                mask_goodvalues, dx, dy, bkg_bin, conv_rad_bin)
        timings['dilate_edt'].append(time.perf_counter() - t0)

        core_dilate = result['score_dilate']

        # expand_conv_core original
        t0 = time.perf_counter()
        expand_conv_core(core_dilate, radii_expand, dx, dy, min_corenpix=0)
        timings['expand_orig'].append(time.perf_counter() - t0)

        # expand_conv_core_fast
        t0 = time.perf_counter()
        expand_conv_core_fast(core_dilate, radii_expand, dx, dy, min_corenpix=0)
        timings['expand_fast'].append(time.perf_counter() - t0)

        # expand_conv_core_edt
        t0 = time.perf_counter()
        expand_conv_core_edt(core_dilate, radii_expand, dx, dy, min_corenpix=0)
        timings['expand_edt'].append(time.perf_counter() - t0)

        # echotop original
        for thresh in [10, 20, 30, 40, 50]:
            t0 = time.perf_counter()
            echotop_height(dbz3d_filt, height, z_dimname, shape_2d,
                           dbz_thresh=thresh, gap=echotop_gap, min_thick=0)
            timings[f'echotop_orig_{thresh}'].append(time.perf_counter() - t0)

        # echotop_fast
        for thresh in [10, 20, 30, 40, 50]:
            t0 = time.perf_counter()
            echotop_height_fast(dbz3d_filt, height, z_dimname, shape_2d,
                                dbz_thresh=thresh, gap=echotop_gap, min_thick=0)
            timings[f'echotop_fast_{thresh}'].append(time.perf_counter() - t0)

    # Aggregate echotop all-5
    n = len(infiles)
    timings['echotop_orig_all5'] = [
        sum(timings[f'echotop_orig_{t}'][i] for t in [10, 20, 30, 40, 50])
        for i in range(n)
    ]
    timings['echotop_fast_all5'] = [
        sum(timings[f'echotop_fast_{t}'][i] for t in [10, 20, 30, 40, 50])
        for i in range(n)
    ]

    return timings


def make_speedup_figure(demo_name, timings, outdir):
    """Create and save a two-panel bar chart: timing + log-scale speedup vs orig."""
    # (label, orig_key, fast_key_or_None, edt_key_or_None)
    steps = [
        ('mod_dilate',        'dilate_orig',       None,                'dilate_edt'),
        ('expand_conv_core',  'expand_orig',        'expand_fast',       'expand_edt'),
        ('echotop all-5',     'echotop_orig_all5',  'echotop_fast_all5', None),
        ('echotop thresh=10', 'echotop_orig_10',    'echotop_fast_10',   None),
        ('echotop thresh=20', 'echotop_orig_20',    'echotop_fast_20',   None),
        ('echotop thresh=30', 'echotop_orig_30',    'echotop_fast_30',   None),
        ('echotop thresh=40', 'echotop_orig_40',    'echotop_fast_40',   None),
        ('echotop thresh=50', 'echotop_orig_50',    'echotop_fast_50',   None),
    ]

    labels  = [s[0] for s in steps]
    n_steps = len(labels)
    x       = np.arange(n_steps)
    width   = 0.26

    means_o = [np.mean(timings[s[1]]) for s in steps]
    stds_o  = [np.std(timings[s[1]])  for s in steps]
    means_f = [np.mean(timings[s[2]]) if s[2] else None for s in steps]
    stds_f  = [np.std(timings[s[2]])  if s[2] else None for s in steps]
    means_e = [np.mean(timings[s[3]]) if s[3] else None for s in steps]
    stds_e  = [np.std(timings[s[3]])  if s[3] else None for s in steps]

    speedups_f = [o / max(f, 1e-9) if f is not None else None
                  for o, f in zip(means_o, means_f)]
    speedups_e = [o / max(e, 1e-9) if e is not None else None
                  for o, e in zip(means_o, means_e)]

    nf = len(timings['dilate_orig'])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)
    fig.suptitle(f'idcells_reflectivity performance — {demo_name} (n={nf} files)',
                 fontsize=13)

    # ── Top panel: absolute timing ────────────────────────────────────────────
    ax1.bar(x - width, means_o, width, yerr=stds_o,
            label='orig', color='steelblue', capsize=3)
    xf = [xi for xi, mf in zip(x, means_f) if mf is not None]
    if xf:
        ax1.bar(np.array(xf), [mf for mf in means_f if mf is not None], width,
                yerr=[sf for sf in stds_f if sf is not None],
                label='fast', color='darkorange', capsize=3)
    xe = [xi for xi, me in zip(x, means_e) if me is not None]
    if xe:
        ax1.bar(np.array(xe) + width, [me for me in means_e if me is not None], width,
                yerr=[se for se in stds_e if se is not None],
                label='edt', color='forestgreen', capsize=3)
    ax1.set_ylabel('Mean time per file (s)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha='right')
    ax1.legend()
    ax1.set_title('Absolute timing (s)')
    ax1.grid(axis='y', alpha=0.3)

    # ── Bottom panel: speedup vs orig (log scale) ─────────────────────────────
    for xi, sp in zip(x, speedups_f):
        if sp is not None:
            ax2.bar(xi, sp, width, color='darkorange', alpha=0.85)
            ax2.text(xi, sp * 1.15, f'{sp:.1f}×', ha='center', va='bottom',
                     fontsize=7, color='darkorange')
    for xi, sp in zip(x, speedups_e):
        if sp is not None:
            ax2.bar(xi + width, sp, width, color='forestgreen', alpha=0.85)
            lbl = f'{sp:.0f}×' if sp >= 10 else f'{sp:.1f}×'
            ax2.text(xi + width, sp * 1.15, lbl, ha='center', va='bottom',
                     fontsize=7, color='forestgreen')
    ax2.axhline(1, color='k', linewidth=0.8, linestyle='--')
    ax2.set_ylabel('Speedup vs orig (log scale)')
    ax2.set_yscale('log')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha='right')
    from matplotlib.patches import Patch
    ax2.legend(handles=[
        Patch(color='darkorange',  label='fast vs orig'),
        Patch(color='forestgreen', label='edt vs orig'),
    ], loc='upper right')
    ax2.set_title('Speedup ratio (orig / method, log scale)')
    ax2.grid(axis='y', alpha=0.3, which='both')

    safe_name = demo_name.replace(' ', '_').replace('(', '').replace(')', '')
    outfile = os.path.join(outdir, f'idcells_speedup_{safe_name}.png')
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outfile}")
    return outfile


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    saved = []

    def _print_summary(demo_name, timings):
        """Print per-step timing summary to stdout."""
        for step_label, orig_key, fast_key, edt_key in [
            ('mod_dilate',       'dilate_orig',       None,              'dilate_edt'),
            ('expand_conv_core', 'expand_orig',        'expand_fast',     'expand_edt'),
            ('echotop all-5',    'echotop_orig_all5',  'echotop_fast_all5', None),
        ]:
            o = np.mean(timings[orig_key])
            parts = [f"orig={o:.3f}s"]
            if fast_key:
                f = np.mean(timings[fast_key])
                parts.append(f"fast={f:.3f}s ({o/max(f,1e-9):.1f}x)")
            if edt_key:
                e = np.mean(timings[edt_key])
                sp = o / max(e, 1e-9)
                parts.append(f"edt={e:.3f}s ({sp:.0f}x)" if sp >= 10 else
                              f"edt={e:.3f}s ({sp:.1f}x)")
            print(f"  {step_label}: {', '.join(parts)}")

    if args.config:
        # ── Direct-config mode: time a single user-supplied config ──────────
        demo_name = os.path.splitext(os.path.basename(args.config))[0]
        print(f"\n{'='*60}")
        print(f"Config: {args.config}")

        timings = time_demo(args.config, args.nfiles)
        _print_summary(demo_name, timings)
        outfile = make_speedup_figure(demo_name, timings, OUTDIR)
        saved.append(outfile)

    else:
        # ── Demo mode: iterate over built-in NEXRAD / CSAPR demos ───────────
        for demo in DEMOS:
            print(f"\n{'='*60}")
            print(f"Demo: {demo['name']}")

            config_path = make_config(
                demo['template'], demo['data_subdir'], DATA_ROOT, OUTDIR
            )
            if not os.path.exists(config_path):
                print(f"  Config not found: {config_path} — skipping")
                continue

            timings = time_demo(config_path, args.nfiles)
            _print_summary(demo['name'], timings)
            outfile = make_speedup_figure(demo['name'], timings, OUTDIR)
            saved.append(outfile)

    print(f"\nFigures saved to: {OUTDIR}")
    for f in saved:
        print(f"  {f}")
