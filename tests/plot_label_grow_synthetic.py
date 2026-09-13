#!/usr/bin/env python
"""
Compare BFS vs EDT growth methods on synthetic Tb frames with randomized,
hand-built circular cold cores - complements test_label_grow_methods.py's
synthetic pytest fixtures (which use sharp step-function blobs to check
correctness invariants) with a visual stress test of the two growth
methods' boundary behavior on large, sharp-edged fields modeled on real
deep convection rather than smooth toy blobs.

Each core is a "flat-top" (super-Gaussian) temperature dip: T(r) =
background - depth * exp(-(r/R_plateau)^(2p)). Unlike a plain Gaussian,
this decouples cloud *size* (R_out, see R_OUT_KM_RANGE) from edge
*sharpness* (p, see SHARPNESS_P) - a plain Gaussian's single sigma
controls both at once, so making it sharp also makes it small. Sizing is
grounded in real IMERG data (see plot_label_grow_speedup.py output): a raw
transect there shows a 96 K drop in a single ~11 km pixel, and domain-wide
gradient statistics right at the 261 K threshold give a p90 of ~17 K/pixel
- the outermost cloud-to-clear-sky edge really is close to a step
function. Deliberately not a literal hard step (uniform-temperature
disk), though: that would give the whole cloud one flat temperature with
no core/secondary/tertiary substructure, collapsing
label_and_grow_features's core-detect-then-grow mechanism into a no-op
(core detection would already equal the final extent, so BFS and EDT
would trivially always agree) - defeating the point of a growth-method
comparison script.

That last point turned out to matter more than expected. A high p (~5,
tried first) reproduces the real sharp-edge measurement closely - but
also leaves almost no "growable" 225-261 K band for the two methods to
possibly disagree about (confirmed directly: 0 disagreement over dozens
of frames, tight multi-core clusters included). Real MCS complexes
*do* have that room - measuring distance-from-nearest-core-pixel in the
same IMERG file, the 225-241 K band's median is only 5 px from the
nearest core, but the 241-261 K band's is 35 px (p90 108 px): a small,
sharp-edged core sits inside a much broader, gradually-varying cold
shield, not a uniformly-thin sharp rim. A low p (SHARPNESS_P, currently
0.5) reproduces *that* instead: the profile stops being flat-topped and
becomes a gradually-decaying dome, trading the single-pixel edge
sharpness for a wide, gradual 225-261 K shield with real disagreement
possible across it. See SHARPNESS_P's own comment for the tradeoff and
the measured numbers; it's a tunable dial, not one fixed "correct" value,
and which end matters more depends on what a given run of this script is
for - matching the sharpest real edges, or stress-testing BFS vs EDT.

Cores are combined via the pixel-wise minimum across all of them (not a
sum - summing would artificially fuse nearby cores into one unrealistically
cold blob; minimum is how independent cold features actually combine).
Frames alternate between cores placed close together (stresses BFS/EDT
tie-breaking at contested, near-touching boundaries) and cores spread
apart (cleanly separated, a sanity baseline).

Uses the exact same tb thresholds/params as the idealized demo
(_IDEALIZED_PARAMS in tests/test_label_grow_methods.py and
config/config_mcs_idealized.yml), fake lat/lon near the equator at the
idealized demo's own apparent resolution, and calls
plot_label_grow_comparison.py's plot_comparison() directly (same plotting
code, not a reimplementation) for the actual figures.

This script is for visual inspection - run it manually to produce figures.
It has no pytest assertions; test_label_grow_methods.py covers correctness.

Usage
-----
  python tests/plot_label_grow_synthetic.py --outdir ~/data/demo/benchmark_results/synthetic/

Requirements
------------
  matplotlib, numpy, cartopy (available in pyflextrkr environment)
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np

# Sibling-module imports, not a package - same pattern already used by
# test_label_grow_methods.py, test_area_method_clouds.py, etc.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyflextrkr.label_and_grow_features import label_and_grow_features
from plot_label_grow_comparison import (
    plot_comparison,
    match_features_by_overlap,
    feature_matched_diff_mask,
)

# Same thresholds/params as the idealized demo - see _IDEALIZED_PARAMS in
# tests/test_label_grow_methods.py and config/config_mcs_idealized.yml.
THRESHOLDS = [225.0, 241.0, 261.0, 261.0]
PIXEL_RADIUS = 10.0
AREA_THRESH = 800.0
MIN_CORE_NPIX = 4
SMOOTH_SIZE = 5
EXPAND_TO_TERTIARY = 0
CONFIG = {"pbc_direction": "none"}

# Large enough to comfortably fit several 20-80-px-diameter cores (see
# R_OUT_KM_RANGE below) with room for both close-together and well-spread
# placement, without constant edge-clipping or forced overlap.
NY, NX = 320, 420
BACKGROUND_TB = 290.0
NOISE_STD = 0.7

# km per pixel for sizing cores from physical units - same value already
# used as label_and_grow_features's own pixel_radius (fixed-area mode) and
# to build the fake lat/lon's ~0.1 deg/pixel spacing, so "R_out in km" and
# "the grid this runs on" stay mutually consistent.
KM_PER_PIXEL = PIXEL_RADIUS

# Outer radius (where a core's flat-top profile crosses 261 K, the
# warm-anvil cutoff) in km, drawn per core from this range. Widening it
# (100-150 originally -> 100-400) does increase BFS/EDT disagreement, but
# mostly because it raises the *typical* size, not because of the range's
# width/diversity per se: isolated directly (same seeds, only R_OUT_KM_RANGE
# varied) - a *narrow* band at roughly the same mean as this wide one
# (200-400) produces nearly as much disagreement (8903 vs 10011 diff px
# over the same 6 test frames) as the actual wide range. Bigger clouds mean
# more contested-boundary pixels in absolute terms, simply because there's
# more boundary; a wider size range does add a smaller amount on top of
# that (asymmetric-size pairs contest their shared boundary differently
# under BFS's graph distance than EDT's Euclidean distance), it's just not
# the dominant effect.
R_OUT_KM_RANGE = (100.0, 400.0)
R_OUT_PX_RANGE = tuple(r / KM_PER_PIXEL for r in R_OUT_KM_RANGE)

# Sharpness exponent for the flat-top profile (higher = flatter interior +
# steeper edge, independent of size - see module docstring's tradeoff
# discussion for the full story). Verified directly at r_out=15px,
# core_tb=200K: p=5 crosses 225 K (core threshold) at r~13px - a sharp ~2px
# rim carries the whole 225-261 K range, ~20 K/px locally, matching real
# IMERG edges closely, but leaves BFS and EDT almost nothing to disagree
# about (confirmed: 0 diff pixels over dozens of test frames, including
# properly-clustered, shape-irregular ones). p=0.5 crosses 225 K at only
# r~4px instead - an 11+ px shield carries that same range, ~2 K/px
# locally (~10x softer than p=5, and softer than a plain Gaussian tail
# too - this is *not* a "sharp edges" setting). The tradeoff is real and
# deliberate: this value picks disagreement-testing over edge-realism.
# Raise it back toward 5 for a sharpness-accurate run with little/no
# disagreement; this repo has no single "correct" value for both at once.
SHARPNESS_P = 0.5

# Coldest-point temperature per core, drawn from this range. Capped at
# 210 K (not the original 220 K) for the same reason as SHARPNESS_P above:
# under the now-gradual (p=0.5) profile, how deep a core is directly sets
# how much of r_out is actually below the 225 K core threshold - verified
# directly at r_out=15px: core_tb=220K -> only 1px (7% of r_out) is real
# core, too small and shallow to reliably survive SMOOTH_SIZE's box-average
# and min_core_npix as its own detected core; 210K -> 3px (20%); 195K ->
# 4.5px (30%). A too-shallow core risks smoothing away entirely (silently
# merging into a neighbor, or vanishing) rather than surviving as its own
# separately-labeled feature with a real boundary to contest.
CORE_TB_RANGE = (195.0, 210.0)

# Cores per frame. More cores directly means more candidate pairs whose
# boundaries can end up contested, on top of the per-pair effects above.
N_CORES_RANGE = (10, 30)  # rng.integers high is exclusive -> 10 to 29 cores


def make_fake_latlon(ny=NY, nx=NX, lat0=-20.0, lon0=0.0, dlat=0.1, dlon=0.1):
    """
    2D near-equator lat/lon, at the idealized demo's own apparent resolution
    (~0.1 deg/pixel, i.e. pixel_radius=10 km) for visual consistency with
    the real idealized-demo comparison figures. Purely for display via the
    cartopy plotting in plot_comparison() - not used for any area
    calculation (pixel_radius stays a scalar, fixed-area mode, same as the
    idealized demo).
    """
    lat_1d = lat0 + dlat * np.arange(ny)
    lon_1d = lon0 + dlon * np.arange(nx)
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
    return lat2d, lon2d


def _r_plateau_for_target(r_out, p, depth, edge_thresh=261.0, background=BACKGROUND_TB):
    """
    Solve the flat-top profile's R_plateau so it crosses edge_thresh
    exactly at radius r_out, for a given sharpness p and core depth
    (background - core_tb). See module docstring for the profile shape.
    """
    frac = (background - edge_thresh) / depth
    # frac in (0, 1) as long as depth > background - edge_thresh (i.e. the
    # core is actually colder than edge_thresh, always true here since
    # CORE_TB_RANGE tops out well below 261 K) - guaranteed real, positive.
    k = -np.log(frac)
    return r_out / (k ** (1.0 / (2.0 * p)))


def _flat_top_dip(r, r_out, p, depth):
    """T(r) - background, i.e. how much colder than background at radius r."""
    r_plateau = _r_plateau_for_target(r_out, p, depth)
    return depth * np.exp(-(r / r_plateau) ** (2.0 * p))


def _place_cores(rng, n_cores, ny, nx, close_together, max_attempts=200):
    """
    Randomly place n_cores (center, r_out_px, core_tb) tuples.

    "spread" frames: every pair placed clearly apart (an all-pairs
    constraint that stays easy to satisfy simultaneously for well-separated
    points in a large domain).

    "close" frames: each new core is chained onto one *randomly chosen*
    already-placed core (not every existing core) at close range, growing
    organic, branching clusters - contested 3+-way boundaries, not just
    isolated pairs. Requiring closeness to *every* prior core instead
    (the first version of this script did) becomes geometrically
    impossible past 2-3 cores, silently degenerating into random placement
    for the rest and losing the clustering entirely; chaining to one anchor
    avoids that. Other, non-anchor cores only get a loose "don't sit
    exactly on top of" check, not the full close-range constraint.

    Falls back to accepting whatever spacing results if max_attempts is
    exhausted (keeps this from ever hanging, at the cost of an occasional
    less-ideal placement).
    """
    margin = R_OUT_PX_RANGE[1] + 10
    cores = []
    for icore in range(n_cores):
        r_out = rng.uniform(*R_OUT_PX_RANGE)
        core_tb = rng.uniform(*CORE_TB_RANGE)
        anchor = cores[rng.integers(len(cores))] if (close_together and cores) else None
        placed = False
        for _attempt in range(max_attempts):
            cy = rng.uniform(margin, ny - margin)
            cx = rng.uniform(margin, nx - margin)
            ok = True
            for (oy, ox, o_rout, _otb) in cores:
                dist = float(np.hypot(cy - oy, cx - ox))
                combined = r_out + o_rout
                is_anchor = anchor is not None and (oy, ox, o_rout, _otb) == anchor
                if close_together and is_anchor:
                    # Tight enough that this core's and the anchor's
                    # secondary-threshold anvils genuinely overlap/compete
                    # for boundary pixels (where BFS and EDT can legitimately
                    # disagree) - their combined 400-600 km cloud complex is
                    # the whole point of this mode.
                    in_range = 0.3 * combined <= dist <= 0.6 * combined
                elif close_together:
                    # Non-anchor core in the same cluster: just don't
                    # coincide almost exactly with it.
                    in_range = dist >= 0.3 * combined
                else:
                    in_range = dist >= 2.0 * combined
                if not in_range:
                    ok = False
                    break
            if ok:
                cores.append((cy, cx, r_out, core_tb))
                placed = True
                break
        if not placed:
            cy = rng.uniform(margin, ny - margin)
            cx = rng.uniform(margin, nx - margin)
            cores.append((cy, cx, r_out, core_tb))
    return cores


def _irregular_radius_scale(theta, rng, n_harmonics=3, base_amp=0.15):
    """
    A smooth, organic-looking multiplier on r_out as a function of angle
    theta - real cloud shields are lumpy/irregular, not perfect circles.

    This was originally the *only* thing that made BFS and EDT disagree at
    all: with the flat-top profile's original sharp setting (SHARPNESS_P
    ~5), two perfectly circular seeds meet at a clean perpendicular
    bisector under any reasonable distance metric, so BFS
    (graph/Chebyshev-ish distance) and EDT (Euclidean distance) agreed on
    the boundary even for close, differently-sized circular cores -
    confirmed directly, a tight 4-core circular cluster still produced
    exactly 0 disagreement without this. Since SHARPNESS_P dropped to 0.5
    (see its comment - a wide, gradually-decaying growable band matters far
    more than shape), this is no longer load-bearing: with base_amp=0
    (perfect circles), disagreement is actually slightly *higher* than with
    the current amplitude (9260 vs 8083 diff px over the same 6 test
    frames) - the size diversity from R_OUT_KM_RANGE and the wide growable
    band from a low SHARPNESS_P are doing the real work now. Kept anyway
    for visual realism (real cloud shields are lumpy), not because it's
    required for any disagreement to occur.

    Sum of a few random-phase cosine harmonics, amplitude decreasing with
    harmonic number (smoother, less noisy-looking than equal-weight); clipped
    well short of 0 so the shape never self-intersects.
    """
    scale = np.ones_like(theta)
    for h in range(1, n_harmonics + 1):
        amp = base_amp / h
        phase = rng.uniform(0.0, 2.0 * np.pi)
        scale = scale + amp * np.cos(h * theta + phase)
    return np.clip(scale, 0.55, 1.45)


def make_synthetic_frame(rng, close_together, ny=NY, nx=NX):
    """
    Build one synthetic Tb frame: warm background, one flat-top (sharp
    -edged), mildly irregular-shaped cold-core dip per core (combined via
    minimum), plus mild pixel noise.

    Returns (tb, cores) - cores is the list of (cy, cx, r_out_px, core_tb)
    tuples actually used, for the printed per-frame summary.
    """
    n_cores = int(rng.integers(*N_CORES_RANGE))
    cores = _place_cores(rng, n_cores, ny, nx, close_together)

    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    tb = np.full((ny, nx), BACKGROUND_TB, dtype=np.float64)
    for (cy, cx, r_out, core_tb) in cores:
        depth = BACKGROUND_TB - core_tb
        dy, dx = y_idx - cy, x_idx - cx
        r = np.hypot(dy, dx)
        theta = np.arctan2(dy, dx)
        shape_scale = _irregular_radius_scale(theta, rng)
        r_eff = r / shape_scale
        dip = _flat_top_dip(r_eff, r_out, SHARPNESS_P, depth)
        core_field = BACKGROUND_TB - dip
        tb = np.minimum(tb, core_field)

    tb += rng.normal(0.0, NOISE_STD, size=tb.shape)
    tb = np.clip(tb, 180.0, 310.0)
    return tb, cores


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--outdir", default=".",
        help="Directory to save cloudid arrays and figures (default: .)",
    )
    parser.add_argument(
        "--nframes", type=int, default=6,
        help="Number of synthetic frames to generate (default: 6)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base RNG seed - frame i uses seed+i, for reproducibility (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bfs_dir = os.path.join(args.outdir, "cloudid_bfs")
    edt_dir = os.path.join(args.outdir, "cloudid_edt")
    fig_dir = os.path.join(args.outdir, "figures")
    for d in (bfs_dir, edt_dir, fig_dir):
        os.makedirs(d, exist_ok=True)

    lat, lon = make_fake_latlon()

    print(f"\n{'='*60}")
    print("  Synthetic BFS vs EDT comparison")
    print(f"  Frames: {args.nframes}, domain: {NY}x{NX}, seed: {args.seed}")
    print(f"{'='*60}\n")

    total_diff = 0
    total_pixels = 0
    saved_figs = []

    for i in range(args.nframes):
        close_together = (i % 2 == 0)
        rng = np.random.default_rng(args.seed + i)
        tb, cores = make_synthetic_frame(rng, close_together)
        frame_name = f"synthetic_{i:02d}_{'close' if close_together else 'spread'}"

        result_bfs = label_and_grow_features(
            tb, PIXEL_RADIUS, THRESHOLDS, AREA_THRESH, MIN_CORE_NPIX,
            SMOOTH_SIZE, EXPAND_TO_TERTIARY, CONFIG,
            core_operator="lt", growth_method="bfs",
        )
        result_edt = label_and_grow_features(
            tb, PIXEL_RADIUS, THRESHOLDS, AREA_THRESH, MIN_CORE_NPIX,
            SMOOTH_SIZE, EXPAND_TO_TERTIARY, CONFIG,
            core_operator="lt", growth_method="edt",
        )

        def _cloudid_dict(result):
            return {
                "cloudnumber": result["final_Feature_Number"].astype(np.int32),
                "convcold_cloudnumber": result["final_CoreSecondary_Number"].astype(np.int32),
                "cloudtype": result["final_Feature_Type"].astype(np.int32),
            }

        bfs_data = _cloudid_dict(result_bfs)
        edt_data = _cloudid_dict(result_edt)

        # Save cloudid arrays too - same schema as plot_label_grow_speedup.py,
        # for consistency and so these frames can be re-plotted later without
        # regenerating them.
        np.savez_compressed(
            os.path.join(bfs_dir, f"{frame_name}.npz"),
            tb=tb.astype(np.float32), lat=lat.astype(np.float32),
            lon=lon.astype(np.float32), **bfs_data,
        )
        np.savez_compressed(
            os.path.join(edt_dir, f"{frame_name}.npz"),
            tb=tb.astype(np.float32), lat=lat.astype(np.float32),
            lon=lon.astype(np.float32), **edt_data,
        )

        outfile = plot_comparison(bfs_data, edt_data, tb, lat, lon, frame_name, fig_dir)
        saved_figs.append(outfile)

        bfs_labels = bfs_data["convcold_cloudnumber"]
        edt_labels = edt_data["convcold_cloudnumber"]
        bfs_to_edt, edt_to_bfs = match_features_by_overlap(bfs_labels, edt_labels)
        diff_mask = feature_matched_diff_mask(bfs_labels, edt_labels, bfs_to_edt, edt_to_bfs)
        n_diff = int(np.count_nonzero(diff_mask))
        n_total = bfs_labels.size
        total_diff += n_diff
        total_pixels += n_total

        n_bfs = int(result_bfs["final_nFeature"])
        n_edt = int(result_edt["final_nFeature"])
        print(
            f"  Frame {i:02d} ({'close' if close_together else 'spread'}, "
            f"{len(cores)} cores): BFS={n_bfs} features, EDT={n_edt} features, "
            f"diff={n_diff} px ({n_diff/n_total*100:.4f}%)"
        )

    print(f"\n  Summary:")
    print(f"    Frames: {args.nframes}")
    print(f"    Total pixels: {total_pixels:,}")
    print(f"    Total differing pixels: {total_diff:,}")
    print(f"    Overall agreement: {(1 - total_diff / total_pixels) * 100:.4f}%")
    print(f"\n  Cloudid arrays saved to: {bfs_dir} and {edt_dir}")
    print(f"  Figures saved to: {fig_dir}")
    for f in saved_figs:
        print(f"    {f}")


if __name__ == "__main__":
    main()
