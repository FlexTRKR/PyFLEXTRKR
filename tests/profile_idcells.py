#!/usr/bin/env python
"""
Profile each major step in idcells_reflectivity across all (or --nfiles) input files.
Takes any PyFLEXTRKR config YAML — no special demo data setup required.

Steps timed per file
--------------------
  get_composite              : file I/O and 3D composite reflectivity
  mod_steiner                : full Steiner classification (FFT background + dilation)
    background_intensity     : FFT background reflectivity sub-step
    peakedness               : cosine peakedness sub-step
    mod_dilate_conv_rad      : original binary_dilation dilation sub-step
    mod_dilate_conv_rad_edt  : EDT-based dilation sub-step
  expand_conv_core (orig)    : per-core binary_dilation expansion
  expand_conv_core_fast      : grey_dilation expansion (~8-24x vs orig)
  expand_conv_core_edt       : EDT-based expansion (1000-3000x vs orig)
  echotop_height orig/fast   : echo-top heights at 5 dBZ thresholds (10-50 dBZ)

Usage
-----
  # All files, print summary table:
  python tests/profile_idcells.py config/config_autotest_demo_cell_csapr.yml

  # First 5 files only:
  python tests/profile_idcells.py config/config_autotest_demo_cell_csapr.yml --nfiles 5

  # LES 100m data (first file only for a quick test):
  python tests/profile_idcells.py \\
    lasso/cellgrowth/tracking/slurm/config_lasso_wrf100m_20181129_gefs09_base.yml --nfiles 1

  # Also dump cProfile top-25 table:
  python tests/profile_idcells.py config/config_autotest_demo_cell_csapr.yml --cprofile
"""
import argparse
import cProfile
import io
import pstats
import time
from collections import defaultdict

import numpy as np

from pyflextrkr.echotop_func import echotop_height, echotop_height_fast
from pyflextrkr.ft_utilities import load_config, subset_files_timerange
from pyflextrkr.idcells_reflectivity import (
    get_composite_reflectivity_generic,
    idcells_reflectivity,
)
from pyflextrkr.steiner_func import (
    background_intensity,
    expand_conv_core,
    expand_conv_core_edt,
    expand_conv_core_fast,
    make_dilation_step_func,
    mod_dilate_conv_rad,
    mod_dilate_conv_rad_edt,
    mod_steiner_classification,
    peakedness,
)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('config')
parser.add_argument('--nfiles', type=int, default=0,
                    help='Number of input files to profile (0 = all)')
parser.add_argument('--cprofile', action='store_true',
                    help='Run cProfile on every file and print top-25 table at the end')
args = parser.parse_args()

# ── Load config & file list ───────────────────────────────────────────────────
config = load_config(args.config)
all_files, _, _, _ = subset_files_timerange(
    config['clouddata_path'], config.get('databasename', ''),
    start_basetime=config.get('start_basetime'),
    end_basetime=config.get('end_basetime'),
    time_format=config['time_format'],
)
infiles = all_files[:args.nfiles] if args.nfiles > 0 else all_files
print(f"Profiling {len(infiles)} file(s) from {config['clouddata_path']}")

# ── Step timing accumulators ──────────────────────────────────────────────────
timings = defaultdict(list)

if args.cprofile:
    combined_pr = cProfile.Profile()

# ── Per-file loop ─────────────────────────────────────────────────────────────
for input_file in infiles:
    print(f"\n=== {input_file} ===")

    # Step 1: File I/O & 3D composite
    t0 = time.perf_counter()
    comp_dict = get_composite_reflectivity_generic(input_file, config)
    dt = time.perf_counter() - t0
    timings['get_composite'].append(dt)
    print(f"  get_composite_reflectivity_generic : {dt:7.3f}s")

    refl            = comp_dict['refl']
    dbz3d_filt      = comp_dict['dbz3d_filt']
    height          = comp_dict['height']
    mask_goodvalues = comp_dict['mask_goodvalues']
    dx = config['dx']
    dy = config['dy']

    bkg_bin, conv_rad_bin = make_dilation_step_func(
        config['mindBZuse'], config['dBZforMaxConvRadius'],
        config['bkg_refl_increment'], config['conv_rad_increment'],
        config['conv_rad_start'], config['maxConvRadius'],
    )
    types_steiner = {'NO_SURF_ECHO': 1, 'WEAK_ECHO': 2, 'STRATIFORM': 3, 'CONVECTIVE': 4}

    # Step 2: Steiner classification (includes background_intensity)
    t0 = time.perf_counter()
    steiner_result = mod_steiner_classification(
        types_steiner, refl, mask_goodvalues, dx, dy,
        bkg_rad=config['bkgrndRadius'] * 1000,
        minZdiff=config['minZdiff'],
        absConvThres=config['absConvThres'],
        truncZconvThres=config['truncZconvThres'],
        weakEchoThres=config['weakEchoThres'],
        bkg_bin=bkg_bin, conv_rad_bin=conv_rad_bin,
        min_corearea=config.get('min_corearea', 0),
        min_cellarea=config.get('min_cellarea', 0),
        remove_smallcores=config.get('remove_smallcores', True),
        remove_smallcells=config.get('remove_smallcells', False),
        return_diag=False,
        convolve_method=config.get('convolve_method', 'fft'),
    )
    dt = time.perf_counter() - t0
    timings['mod_steiner'].append(dt)
    print(f"  mod_steiner_classification         : {dt:7.3f}s")

    # ── Step 2 sub-timings ────────────────────────────────────────────────────
    bkg_rad_m = config['bkgrndRadius'] * 1000
    convolve_method = config.get('convolve_method', 'fft')

    t0 = time.perf_counter()
    refl_bkg = background_intensity(
        refl, mask_goodvalues, dx, dy, bkg_rad_m,
        convolve_method=convolve_method,
    )
    dt = time.perf_counter() - t0
    timings['background_intensity'].append(dt)
    print(f"    background_intensity             : {dt:7.3f}s")

    t0 = time.perf_counter()
    peak = peakedness(refl_bkg, mask_goodvalues,
                      config['minZdiff'], config['absConvThres'])
    dt = time.perf_counter() - t0
    timings['peakedness'].append(dt)
    print(f"    peakedness                       : {dt:7.3f}s")

    # Reconstruct score (mirrors logic inside mod_steiner_classification)
    from scipy import ndimage as _ndimage
    _score = np.zeros(refl.shape, dtype=int)
    _ind_core = np.logical_or(
        refl >= config['truncZconvThres'],
        (refl - refl_bkg) >= peak,
    )
    _score[_ind_core] = 1
    if config.get('remove_smallcores', True):
        _min_corenpix = int(config.get('min_corearea', 0) * (1000**2) / (dx * dy))
        _score_keep = np.copy(_score)
        _tmp, _nreg = _ndimage.label(_score_keep)
        for _rr in range(1, _nreg + 1):
            _rid = np.where(_tmp == _rr)
            if len(_rid[0]) < _min_corenpix:
                _score_keep[_rid] = 0
        _score = _score_keep
    _sclass = np.zeros(refl.shape, dtype=int)
    _sclass[mask_goodvalues == 1] = types_steiner['STRATIFORM']
    _sclass[_score == 1] = types_steiner['CONVECTIVE']
    _sclass[np.logical_and(refl > config['mindBZuse'],
                           refl < config['weakEchoThres'])] = types_steiner['WEAK_ECHO']
    _sclass[np.logical_and(mask_goodvalues == 1,
                           refl < config['mindBZuse'])] = types_steiner['NO_SURF_ECHO']

    t0 = time.perf_counter()
    mod_dilate_conv_rad(
        types_steiner, refl_bkg, _sclass, _score,
        mask_goodvalues, dx, dy, bkg_bin, conv_rad_bin,
    )
    dt = time.perf_counter() - t0
    timings['mod_dilate_conv_rad'].append(dt)
    print(f"    mod_dilate_conv_rad              : {dt:7.3f}s")

    t0 = time.perf_counter()
    mod_dilate_conv_rad_edt(
        types_steiner, refl_bkg, _sclass, _score,
        mask_goodvalues, dx, dy, bkg_bin, conv_rad_bin,
    )
    dt = time.perf_counter() - t0
    timings['mod_dilate_conv_rad_edt'].append(dt)
    print(f"    mod_dilate_conv_rad_edt          : {dt:7.3f}s")

    core_dilate = steiner_result['score_dilate']
    radii_expand = np.array(config['radii_expand'])

    # Step 3a: expand_conv_core (original)
    t0 = time.perf_counter()
    expand_conv_core(core_dilate, radii_expand, dx, dy, min_corenpix=0)
    dt = time.perf_counter() - t0
    timings['expand_conv_core_orig'].append(dt)
    print(f"  expand_conv_core (orig)            : {dt:7.3f}s")

    # Step 3b: expand_conv_core_fast
    t0 = time.perf_counter()
    expand_conv_core_fast(core_dilate, radii_expand, dx, dy, min_corenpix=0)
    dt = time.perf_counter() - t0
    timings['expand_conv_core_fast'].append(dt)
    print(f"  expand_conv_core_fast              : {dt:7.3f}s")

    # Step 3c: expand_conv_core_edt
    t0 = time.perf_counter()
    expand_conv_core_edt(core_dilate, radii_expand, dx, dy, min_corenpix=0)
    dt = time.perf_counter() - t0
    timings['expand_conv_core_edt'].append(dt)
    print(f"  expand_conv_core_edt               : {dt:7.3f}s")

    shape_2d = refl.shape
    z_dimname = config.get('z_dimname', 'z')
    echotop_gap = config['echotop_gap']

    # Step 4: echotop_height original (5 thresholds)
    for thresh in [10, 20, 30, 40, 50]:
        t0 = time.perf_counter()
        echotop_height(dbz3d_filt, height, z_dimname, shape_2d,
                       dbz_thresh=thresh, gap=echotop_gap, min_thick=0)
        dt = time.perf_counter() - t0
        timings[f'echotop_orig_{thresh}'].append(dt)
        print(f"  echotop_height orig(thresh={thresh:2d})     : {dt:7.3f}s")

    # Step 5: echotop_height_fast (5 thresholds)
    for thresh in [10, 20, 30, 40, 50]:
        t0 = time.perf_counter()
        echotop_height_fast(dbz3d_filt, height, z_dimname, shape_2d,
                            dbz_thresh=thresh, gap=echotop_gap, min_thick=0)
        dt = time.perf_counter() - t0
        timings[f'echotop_fast_{thresh}'].append(dt)
        print(f"  echotop_height_fast(thresh={thresh:2d})     : {dt:7.3f}s")

    # Optional cProfile
    if args.cprofile:
        combined_pr.enable()
        idcells_reflectivity(input_file, config)
        combined_pr.disable()


# ── Per-step summary table ─────────────────────────────────────────────────────
print("\n" + "=" * 72)
print(f"  {'Step':<40}  {'mean':>7}  {'std':>7}  {'min':>7}  {'max':>7}  {'total':>8}")
print("-" * 72)


def _row(label, vals):
    a = np.array(vals)
    print(f"  {label:<40}  {a.mean():7.3f}  {a.std():7.3f}  "
          f"{a.min():7.3f}  {a.max():7.3f}  {a.sum():8.3f}")


_row("get_composite",           timings['get_composite'])
_row("mod_steiner",               timings['mod_steiner'])
_row("  background_intensity",    timings['background_intensity'])
_row("  peakedness",              timings['peakedness'])
_row("  mod_dilate_conv_rad",     timings['mod_dilate_conv_rad'])
_row("  mod_dilate_conv_rad_edt", timings['mod_dilate_conv_rad_edt'])
_row("expand_conv_core_orig",     timings['expand_conv_core_orig'])
_row("expand_conv_core_fast",     timings['expand_conv_core_fast'])
_row("expand_conv_core_edt",      timings['expand_conv_core_edt'])

orig_echotop = [sum(timings[f'echotop_orig_{t}'][i] for t in [10, 20, 30, 40, 50])
                for i in range(len(timings['echotop_orig_10']))]
fast_echotop = [sum(timings[f'echotop_fast_{t}'][i] for t in [10, 20, 30, 40, 50])
                for i in range(len(timings['echotop_fast_10']))]
_row("echotop all-5 (orig)",    orig_echotop)
_row("echotop all-5 (fast)",    fast_echotop)

for thresh in [10, 20, 30, 40, 50]:
    _row(f"  echotop_orig  thresh={thresh}",  timings[f'echotop_orig_{thresh}'])
    _row(f"  echotop_fast  thresh={thresh}",  timings[f'echotop_fast_{thresh}'])

print("=" * 72)
print("\nSpeedups (mean orig / mean fast):")

o = np.mean(timings['expand_conv_core_orig'])
f = np.mean(timings['expand_conv_core_fast'])
e = np.mean(timings['expand_conv_core_edt'])
print(f"  expand_conv_core : fast={o / max(f, 1e-9):.1f}x  edt={o / max(e, 1e-9):.1f}x")
print(f"  mod_dilate_edt   : {np.mean(timings['mod_dilate_conv_rad']) / max(np.mean(timings['mod_dilate_conv_rad_edt']), 1e-9):.1f}x")

o_et = np.mean(orig_echotop)
f_et = np.mean(fast_echotop)
print(f"  echotop all-5    : {o_et / max(f_et, 1e-9):.1f}x")
for thresh in [10, 20, 30, 40, 50]:
    o = np.mean(timings[f'echotop_orig_{thresh}'])
    f = np.mean(timings[f'echotop_fast_{thresh}'])
    print(f"    thresh={thresh}     : {o / max(f, 1e-9):.1f}x")

# ── cProfile summary ──────────────────────────────────────────────────────────
if args.cprofile:
    print("\n=== cProfile (all files combined, top-25 by cumulative time) ===")
    s = io.StringIO()
    pstats.Stats(combined_pr, stream=s).sort_stats('cumulative').print_stats(25)
    print(s.getvalue())

