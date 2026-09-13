#!/usr/bin/env python
"""
Measure and visualize performance of BFS vs EDT growth methods in label_and_grow_features.

For each demo dataset this script:
  1. Enumerates input Tb frames - one per *time index* in each file, not just the
     first, so e.g. the idealized demo's 26 hourly times per test file are all
     included, not only hour 0.
  2. For each frame: loads + preprocesses it (same as idclouds_tbpf), then times
     `label_and_grow_features` with growth_method='bfs' and 'edt' back-to-back,
     and saves both methods' cloudid arrays - one .npz per frame per method,
     including that frame's own tb/lat/lon so plot_label_grow_comparison.py can
     plot it directly without re-matching back to the original input file.
  3. Produces a two-panel bar-chart PNG:
       Top panel    : absolute timing (BFS / EDT bars)
       Bottom panel : speedup (BFS / EDT) on log scale

Demos supported
---------------
  demo_mcs_tbpf_idealized : Small idealized domain, fast (verification)
  demo_mcs_imerg          : IMERG South America subset (690x480, 55.95S-12.95N,
                             81.95W-34.05W, 0.1 deg) - moderate performance benchmark
  demo_mcs_imerg_mcsmip   : IMERG Global (1200x3600, 60S-60N, 0.1 deg) -
                             large-domain performance benchmark; use --n_workers

Parallelism
-----------
  --n_workers N (default 1 = serial) times frames concurrently via a Dask
  LocalCluster (same pattern as runscripts/run_mcs_tbpf.py: one worker process
  each, threads_per_worker=1). For a given frame, BFS and EDT always run
  back-to-back in the *same* worker, so the BFS/EDT speedup ratio for that
  frame stays a fair comparison; what gets noisier as n_workers grows is the
  *absolute* per-frame time, since concurrent workers compete for CPU. Useful
  for demo_mcs_imerg_mcsmip, where serial timing is slow.

Usage
-----
  # 1. Download demo data (only needed once)
  python tests/run_demo_tests.py --demos demo_mcs_tbpf_idealized demo_mcs_imerg -n 4

  # 2. Run benchmark on idealized (fast, multiple frames)
  python tests/plot_label_grow_speedup.py \\
      --demo demo_mcs_tbpf_idealized \\
      --data_root ~/data/demo \\
      --outdir ~/data/demo/benchmark_results/idealized/

  # 3. Run benchmark on IMERG South America (moderate domain, performance test)
  python tests/plot_label_grow_speedup.py \\
      --demo demo_mcs_imerg \\
      --data_root ~/data/demo \\
      --outdir ~/data/demo/benchmark_results/imerg/

  # 4. Run benchmark on IMERG Global (large domain - parallelize on a compute node)
  python tests/plot_label_grow_speedup.py \\
      --demo demo_mcs_imerg_mcsmip \\
      --data_root ~/data/demo \\
      --outdir ~/data/demo/benchmark_results/imerg_global/ \\
      --n_workers 12

  # 5. Both methods' cloudid outputs saved to:
  #    <outdir>/cloudid_bfs/
  #    <outdir>/cloudid_edt/

Requirements
------------
  matplotlib, numpy, scipy, xarray, dask[distributed] (available in pyflextrkr
  environment; dask is only imported when --n_workers > 1).
  Demo data under $PYFLEXTRKR_TEST_DATA or --data_root.
"""

import argparse
import glob
import os
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyflextrkr.label_and_grow_features import label_and_grow_features

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    "--demo",
    default="demo_mcs_tbpf_idealized",
    choices=["demo_mcs_tbpf_idealized", "demo_mcs_imerg", "demo_mcs_imerg_mcsmip"],
    help="Which demo to benchmark (default: demo_mcs_tbpf_idealized)",
)
parser.add_argument(
    "--data_root",
    default=os.environ.get("PYFLEXTRKR_TEST_DATA", os.path.expanduser("~/data/demo")),
    help="Path to demo data root (default: $PYFLEXTRKR_TEST_DATA or ~/data/demo)",
)
parser.add_argument(
    "--outdir",
    default=".",
    help="Directory to save output figures and cloudid arrays (default: .)",
)
parser.add_argument(
    "--nfiles",
    type=int,
    default=0,
    help="Number of input files to time (0 = all available)",
)
parser.add_argument(
    "--ntimes",
    type=int,
    default=0,
    help="Number of time indices per file to time (0 = all; each demo's input "
         "files hold multiple times - idealized has 26 hourly per test file, "
         "IMERG has 2 half-hourly per hourly file)",
)
parser.add_argument(
    "--n_workers",
    type=int,
    default=1,
    help="Number of Dask worker processes to time frames concurrently on "
         "(default: 1 = serial, matching all prior runs). BFS and EDT for a "
         "given frame always run back-to-back in the same worker so their "
         "ratio stays fair; only absolute per-frame times get noisier as "
         "this grows (worker contention). Recommended for demo_mcs_imerg_mcsmip.",
)
args = parser.parse_args()

# ── Demo specifications ───────────────────────────────────────────────────────
DEMOS = {
    "demo_mcs_tbpf_idealized": {
        "name": "MCS Idealized",
        "data_subdir": "mcs_tbpf/idealized/test4/input",
        "pattern": "*.nc",
        "tb_varname": "Tb",
        "pixel_radius": 10.0,
        "thresholds": [225.0, 241.0, 261.0, 261.0],
        "area_thresh": 800.0,
        "min_core_npix": 4,
        "smooth_size": 5,
        "expand_to_tertiary": 0,
        "config": {"pbc_direction": "none"},
    },
    "demo_mcs_imerg": {
        # This demo's input is a South America subset (690x480 @ 0.1 deg,
        # 55.95S-12.95N, 81.95W-34.05W - verified directly from
        # merg_2019012500_4km-pixel.nc's own lat/lon coords), NOT a 60S-60N
        # global domain (a prior version of this "name" wrongly claimed
        # that). The lat_range/lon_range for this demo in run_demo_tests.py's
        # DemoInfo are track-validation bounds (where tracks' mean lat/lon
        # must fall), not the input data's actual domain - easy to conflate,
        # but they answer a different question. The real 60S-60N global IMERG
        # domain is demo_mcs_imerg_mcsmip, below.
        "name": "IMERG South America (56S-13N, 82W-34W, 0.1°)",
        "file_label": "IMERG_SouthAmerica_0.1°",
        "data_subdir": "mcs_tbpf/imerg/input",
        "pattern": "merg_*.nc",
        "tb_varname": "Tb",
        "pixel_radius": 10.0,
        "thresholds": [225.0, 241.0, 261.0, 261.0],
        "area_thresh": 800.0,
        "min_core_npix": 4,
        # Matches smoothwindowdimensions in config/config_imerg_mcs_tbpf_example.yml
        # (the actual config this demo runs with) - not 5, which is only
        # correct for the idealized demo's config_mcs_idealized.yml.
        "smooth_size": 10,
        "expand_to_tertiary": 0,
        "config": {"pbc_direction": "none"},
    },
    "demo_mcs_imerg_mcsmip": {
        # The actual 60S-60N global IMERG domain (1200x3600 @ 0.1 deg,
        # -179.95 to 179.95 lon - verified from merg_2020010100_10km-pixel.nc).
        # Much larger than demo_mcs_imerg; use --n_workers (see module
        # docstring) rather than running this serially.
        "name": "IMERG Global (60S-60N, 0.1°)",
        "file_label": "IMERG_Global_60S-60N_0.1°",
        "data_subdir": "mcs_tbpf/imerg_global/input",
        "pattern": "merg_*.nc",
        "tb_varname": "Tb",
        "pixel_radius": 10.0,
        "thresholds": [225.0, 241.0, 261.0, 261.0],
        "area_thresh": 800.0,
        "min_core_npix": 4,
        # Matches config/config_mcs_tbpf_imergv7_mcsmip_demo.yml, the actual
        # config this demo runs with (including pbc_direction: x, unlike the
        # South America subset's "none").
        "smooth_size": 10,
        "expand_to_tertiary": 0,
        # pbc_extended_fraction must be set explicitly - pad_and_extend/
        # adjust_axis/call_adjust_axis (ftfunctions.py) all default to 1.0
        # (each side padded by a full domain width, i.e. 3x total width)
        # when a config dict omits it, which is NOT what the real demo
        # config uses (0.3) and is far more padding than this benchmark
        # needs. 0.2 -> 20% each side -> 1.4x width (1200x3600 -> 1200x5040).
        "config": {"pbc_direction": "x", "pbc_extended_fraction": 0.2},
    },
}


def enumerate_frames(demo_spec, data_root, nfiles, ntimes=0):
    """
    Enumerate Tb frames to time, without loading their data.

    Returns a list of (filepath, time_index, frame_name) tuples - one per
    *time index* within each file, not just the first (each demo's input
    files hold multiple times: idealized 26 hourly per test file, IMERG 2
    half-hourly per hourly file; earlier versions of this script silently
    dropped everything but index 0). frame_name is
    "<file_stem>_<YYYYMMDDTHHMM>", built from the file's own time coordinate.

    Deliberately does not load or preprocess the Tb data itself - actual
    loading happens per-frame in process_frame(), called lazily from
    time_methods() (serially, or one frame at a time per Dask worker). This
    keeps memory bounded regardless of how many frames there are, which
    matters for demo_mcs_imerg_mcsmip (a full run would otherwise need to
    hold ~192 frames x 1200x3600 in memory at once).
    """
    import xarray as xr

    input_dir = os.path.join(data_root, demo_spec["data_subdir"])
    files = sorted(glob.glob(os.path.join(input_dir, demo_spec["pattern"])))
    if not files:
        files = sorted(
            glob.glob(os.path.join(input_dir, "**", demo_spec["pattern"]), recursive=True)
        )
    if nfiles > 0:
        files = files[:nfiles]

    if not files:
        raise FileNotFoundError(
            f"No input files found in {input_dir} with pattern {demo_spec['pattern']}.\n"
            f"Download demo data first:\n"
            f"  python tests/run_demo_tests.py --demos {args.demo} -n 4"
        )

    print(f"  Found {len(files)} input file(s) in {input_dir}")

    frames = []
    for filepath in files:
        ds = xr.open_dataset(filepath, decode_timedelta=False)
        varname = demo_spec["tb_varname"]
        if varname not in ds:
            # Try common alternatives
            for alt in ["Tb", "tb", "brightness_temperature"]:
                if alt in ds:
                    varname = alt
                    break
        if varname not in ds:
            ds.close()
            continue

        ndim = ds[varname].ndim
        if ndim == 2:
            time_vals = [None]
        elif ndim == 3:
            # .shape is metadata-only on a lazily-opened DataArray - doesn't
            # trigger a data read.
            n_times = ds[varname].shape[0]
            time_vals = ds["time"].values if "time" in ds else [None] * n_times
        else:
            ds.close()
            continue

        file_stem = os.path.basename(filepath).rsplit(".nc", 1)[0]
        n_times = len(time_vals)
        time_indices = range(n_times) if ntimes <= 0 else range(min(ntimes, n_times))

        for it in time_indices:
            tval = time_vals[it]
            if tval is not None:
                ts = np.datetime_as_string(np.datetime64(tval), unit="m")
                ts = ts.replace("-", "").replace(":", "")
                frame_name = f"{file_stem}_{ts}"
            else:
                frame_name = file_stem
            frames.append((filepath, it, frame_name))

        ds.close()

    return frames


def process_frame(filepath, it, frame_name, demo_spec, outdir):
    """
    Load, preprocess, and time one frame; save both methods' cloudid arrays.

    Runs BFS then EDT back-to-back for this one frame, so their timing ratio
    is a fair comparison regardless of whether this call happens serially or
    concurrently on a Dask worker (concurrency only adds noise to *absolute*
    times, via CPU contention between workers).

    Saves each method's cloudid arrays directly from this function (rather
    than returning the full arrays to be saved by the caller), so the large
    per-frame arrays never have to round-trip through the Dask scheduler -
    only the small summary tuple below does.

    Returns (frame_name, ny, nx, n_features, t_bfs, t_edt), or None if the
    frame is skipped (missing Tb variable, or >=40% missing data).
    """
    import xarray as xr
    from scipy.signal import medfilt2d

    ds = xr.open_dataset(filepath, decode_timedelta=False)
    varname = demo_spec["tb_varname"]
    if varname not in ds:
        for alt in ["Tb", "tb", "brightness_temperature"]:
            if alt in ds:
                varname = alt
                break
    if varname not in ds:
        ds.close()
        return None

    tb_all = ds[varname].values
    lat = ds["lat"].values if "lat" in ds else None
    lon = ds["lon"].values if "lon" in ds else None
    ds.close()

    tb = tb_all[it, :, :] if tb_all.ndim == 3 else tb_all

    # Preprocess (same as idclouds_tbpf)
    tb_filt = medfilt2d(tb.astype(np.float64), kernel_size=5)
    out_tb = np.copy(tb)
    missmask = np.isnan(tb)
    out_tb[missmask] = tb_filt[missmask]
    out_tb[out_tb < 160] = np.nan
    out_tb[out_tb > 330] = np.nan

    # Skip frames with too much missing data
    ny, nx = out_tb.shape
    if np.count_nonzero(np.isnan(out_tb)) / (ny * nx) >= 0.4:
        return None

    # In production (idclouds_tbpf.py) `config` is the entire loaded YAML, so
    # it always carries pixel_radius/area_thresh alongside pbc_direction.
    # This script instead keeps those as separate top-level demo_spec keys
    # for clarity - but label_and_grow_features's own PBC-adjustment path
    # (adjust_axis -> get_pixel_area) reads pixel_radius (hard requirement)
    # and area_thresh (soft, but a missing value crashes one line later)
    # directly *from config*, only when pbc_direction != "none". Merge them
    # in here so that path sees the same config shape production gives it,
    # rather than duplicating the literals a second time in every DEMOS entry.
    config = dict(demo_spec["config"])
    config.setdefault("pixel_radius", demo_spec["pixel_radius"])
    config.setdefault("area_thresh", demo_spec["area_thresh"])

    params = dict(
        pixel_radius=demo_spec["pixel_radius"],
        thresholds=demo_spec["thresholds"],
        area_thresh=demo_spec["area_thresh"],
        min_core_npix=demo_spec["min_core_npix"],
        smooth_size=demo_spec["smooth_size"],
        expand_to_tertiary=demo_spec["expand_to_tertiary"],
        config=config,
    )

    t0 = time.perf_counter()
    result_bfs = label_and_grow_features(
        out_tb, **params, core_operator="lt", growth_method="bfs",
    )
    t_bfs = time.perf_counter() - t0

    t0 = time.perf_counter()
    result_edt = label_and_grow_features(
        out_tb, **params, core_operator="lt", growth_method="edt",
    )
    t_edt = time.perf_counter() - t0

    bfs_dir = os.path.join(outdir, "cloudid_bfs")
    edt_dir = os.path.join(outdir, "cloudid_edt")
    os.makedirs(bfs_dir, exist_ok=True)
    os.makedirs(edt_dir, exist_ok=True)
    _save_cloudid(bfs_dir, frame_name, result_bfs, out_tb, lat, lon)
    _save_cloudid(edt_dir, frame_name, result_edt, out_tb, lat, lon)

    return (frame_name, ny, nx, int(result_bfs["final_nFeature"]), t_bfs, t_edt)


def _save_cloudid(save_dir, fname, result, tb, lat, lon):
    """Save one method's cloudid arrays - one .npz per frame, including that
    frame's own tb/lat/lon so plot_label_grow_comparison.py can plot it
    directly without re-matching back to the original input file by filename.
    """
    kwargs = dict(
        # int32 is ample for label counts and halves array size vs the
        # int64 label_and_grow_features returns by default.
        cloudnumber=result["final_Feature_Number"].astype(np.int32),
        convcold_cloudnumber=result["final_CoreSecondary_Number"].astype(np.int32),
        cloudtype=result["final_Feature_Type"].astype(np.int32),
        tb=tb.astype(np.float32),
    )
    if lat is not None:
        kwargs["lat"] = lat.astype(np.float32)
    if lon is not None:
        kwargs["lon"] = lon.astype(np.float32)
    np.savez_compressed(os.path.join(save_dir, f"{fname}.npz"), **kwargs)


def time_methods(frames, demo_spec, outdir, n_workers=1):
    """
    Time BFS and EDT methods on all enumerated frames.

    n_workers=1 (default): plain serial loop, printing progress as each
    frame completes - identical behavior/output to every prior run of this
    script.

    n_workers>1: frames are distributed across a Dask LocalCluster (one
    worker process each, threads_per_worker=1, matching
    runscripts/run_mcs_tbpf.py's own LocalCluster usage). Per-frame results
    are only available after the whole batch completes (worker print()
    output isn't reliably visible from the driver process), so progress is
    printed all at once afterward, re-sorted by frame name for a
    deterministic, reproducible figure ordering.
    """
    timings = defaultdict(list)

    if n_workers <= 1:
        results = []
        for filepath, it, frame_name in frames:
            res = process_frame(filepath, it, frame_name, demo_spec, outdir)
            if res is None:
                continue
            results.append(res)
            fname, ny, nx, nfeat, t_bfs, t_edt = res
            speedup = t_bfs / max(t_edt, 1e-9)
            print(
                f"    {fname}: {ny}x{nx}, {nfeat} features, "
                f"BFS={t_bfs:.3f}s, EDT={t_edt:.3f}s, speedup={speedup:.1f}x"
            )
    else:
        from dask.distributed import Client, LocalCluster

        print(
            f"  Distributing {len(frames)} frame(s) across {n_workers} Dask "
            f"worker(s). Absolute per-frame times below reflect worker "
            f"contention and are less meaningful than a serial run; the "
            f"BFS/EDT ratio per frame stays fair since both always run "
            f"back-to-back in the same worker."
        )
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
        client = Client(cluster)
        try:
            futures = [
                client.submit(process_frame, filepath, it, frame_name, demo_spec, outdir)
                for filepath, it, frame_name in frames
            ]
            gathered = client.gather(futures)
        finally:
            client.close()
            cluster.close()

        results = sorted((r for r in gathered if r is not None), key=lambda r: r[0])
        for fname, ny, nx, nfeat, t_bfs, t_edt in results:
            speedup = t_bfs / max(t_edt, 1e-9)
            print(
                f"    {fname}: {ny}x{nx}, {nfeat} features, "
                f"BFS={t_bfs:.3f}s, EDT={t_edt:.3f}s, speedup={speedup:.1f}x"
            )

    for fname, ny, nx, nfeat, t_bfs, t_edt in results:
        timings["bfs"].append(t_bfs)
        timings["edt"].append(t_edt)

    bfs_dir = os.path.join(outdir, "cloudid_bfs")
    edt_dir = os.path.join(outdir, "cloudid_edt")
    print(f"  Saved cloudid arrays to {bfs_dir} and {edt_dir}")
    return timings


def make_figure(demo_spec, timings, outdir, n_workers=1):
    """Create two-panel bar chart: timing + speedup."""
    bfs_times = np.array(timings["bfs"])
    edt_times = np.array(timings["edt"])
    n = len(bfs_times)
    demo_name = demo_spec["name"]

    # Widen the figure for large frame counts (one bar-pair per timestep,
    # not per file, so idealized/IMERG/global can all run into the hundreds).
    fig_width = max(10, min(n * 0.15, 40))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, 7), constrained_layout=True)
    title = f"label_and_grow_features: BFS vs EDT — {demo_name} (n={n} frames)"
    if n_workers > 1:
        title += (
            f"\n[timed with {n_workers} concurrent Dask workers - absolute "
            f"times include worker contention]"
        )
    fig.suptitle(title, fontsize=12)

    x = np.arange(n)
    width = 0.35

    # Top panel: absolute timing
    ax1.bar(x - width / 2, bfs_times, width, label="BFS", color="steelblue")
    ax1.bar(x + width / 2, edt_times, width, label="EDT", color="forestgreen")
    ax1.set_ylabel("Time per frame (s)")
    ax1.set_xlabel("Frame index")
    ax1.legend()
    ax1.set_title("Absolute timing")
    ax1.grid(axis="y", alpha=0.3)

    # Add mean annotations
    mean_bfs = np.mean(bfs_times)
    mean_edt = np.mean(edt_times)
    ax1.axhline(mean_bfs, color="steelblue", linestyle="--", alpha=0.5)
    ax1.axhline(mean_edt, color="forestgreen", linestyle="--", alpha=0.5)
    ax1.text(
        n - 0.5, mean_bfs, f"mean={mean_bfs:.3f}s",
        color="steelblue", va="bottom", ha="right", fontsize=8,
    )
    ax1.text(
        n - 0.5, mean_edt, f"mean={mean_edt:.3f}s",
        color="forestgreen", va="bottom", ha="right", fontsize=8,
    )

    # Bottom panel: speedup ratio
    speedups = bfs_times / np.maximum(edt_times, 1e-9)
    ax2.bar(x, speedups, color="darkorange", alpha=0.85)
    # Per-bar labels get unreadable once there are many frames (all-timeframe
    # runs land in the 90-100+ range) - the mean in the title covers it then.
    if n <= 30:
        for xi, sp in zip(x, speedups):
            lbl = f"{sp:.0f}x" if sp >= 10 else f"{sp:.1f}x"
            ax2.text(xi, sp * 1.05, lbl, ha="center", va="bottom", fontsize=7)
    ax2.axhline(1, color="k", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Speedup (BFS / EDT)")
    ax2.set_xlabel("Frame index")
    ax2.set_yscale("log")
    ax2.set_title(f"Speedup ratio (mean = {np.mean(speedups):.1f}x)")
    ax2.grid(axis="y", alpha=0.3, which="both")

    file_label = demo_spec.get("file_label")
    if file_label is None:
        file_label = demo_name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    outfile = os.path.join(outdir, f"label_grow_speedup_{file_label}.png")
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {outfile}")
    return outfile


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo_spec = DEMOS[args.demo]
    os.makedirs(args.outdir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Benchmark: {demo_spec['name']}")
    print(f"  Data root: {args.data_root}")
    if args.n_workers > 1:
        print(f"  Workers:   {args.n_workers} (Dask LocalCluster)")
    print(f"{'='*60}")

    # Enumerate frames (cheap - metadata only; actual data is loaded lazily,
    # one frame at a time, inside time_methods()/process_frame()).
    frames = enumerate_frames(demo_spec, args.data_root, args.nfiles, args.ntimes)
    print(f"  Enumerated {len(frames)} frame(s) to time")

    if not frames:
        raise SystemExit("ERROR: No valid Tb frames found.")

    # Time methods
    print("\n  Timing BFS vs EDT:")
    timings = time_methods(frames, demo_spec, args.outdir, args.n_workers)

    if not timings["bfs"]:
        raise SystemExit("ERROR: No frames survived preprocessing (all skipped).")

    # Summary
    bfs_mean = np.mean(timings["bfs"])
    edt_mean = np.mean(timings["edt"])
    speedup_mean = bfs_mean / max(edt_mean, 1e-9)
    print(f"\n  Summary ({len(timings['bfs'])} frames):")
    print(f"    BFS mean: {bfs_mean:.3f}s")
    print(f"    EDT mean: {edt_mean:.3f}s")
    print(f"    Mean speedup: {speedup_mean:.1f}x")

    # Make figure
    outfile = make_figure(demo_spec, timings, args.outdir, args.n_workers)
    print(f"\n  Done. Output in: {args.outdir}")
