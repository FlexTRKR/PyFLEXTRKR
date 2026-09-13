#!/usr/bin/env python
"""
Visualize differences between BFS and EDT growth methods in label_and_grow_features.

Creates 3-panel figures for each time frame, plotted on a PlateCarree map
projection using each frame's own lat/lon:
  Panel 1: Input Tb field with BFS feature contours
  Panel 2: Input Tb field with EDT feature contours
  Panel 3: Difference map (pixels where labels differ)

Also supports comparing final MCS track statistics between two runs
(--compare-stats mode).

Usage
-----
  # After running plot_label_grow_speedup.py to generate cloudid outputs
  # (each .npz already carries its own tb/lat/lon - see that script):
  python tests/plot_label_grow_comparison.py \\
      --bfs_dir ~/data/demo/benchmark_results/idealized/cloudid_bfs/ \\
      --edt_dir ~/data/demo/benchmark_results/idealized/cloudid_edt/ \\
      --outdir ~/data/demo/benchmark_results/idealized/figures/

  # Compare final track stats between two runs:
  python tests/plot_label_grow_comparison.py --compare-stats \\
      --bfs_stats ~/data/demo/mcs_tbpf/idealized/test4/stats_bfs/ \\
      --edt_stats ~/data/demo/mcs_tbpf/idealized/test4/stats_edt/

Requirements
------------
  matplotlib, numpy, xarray, cartopy (available in pyflextrkr environment)
  Output from plot_label_grow_speedup.py (cloudid_bfs/, cloudid_edt/ directories)
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Land outline resolution - matches the convention already used by every
# other cartopy plot in this repo (Analysis/plot_subset_tbpf_mcs_tracks_demo.py
# and siblings).
_MAP_RESOLUTION = "50m"
_LAND = cfeature.NaturalEarthFeature("physical", "land", _MAP_RESOLUTION)


def match_features_by_overlap(labels_a, labels_b):
    """Build a bijective best-match mapping between two feature label arrays.

    For each label in ``labels_a``, finds the label in ``labels_b`` with the
    most overlapping pixels, and vice versa.  Only mutually-consistent pairs
    (a is b's best match AND b is a's best match) are returned.

    Parameters
    ----------
    labels_a, labels_b : np.ndarray of int
        2-D labeled arrays (0 = background/unlabeled).

    Returns
    -------
    a_to_b : dict  {label_in_a -> label_in_b}
    b_to_a : dict  {label_in_b -> label_in_a}
        Only mutually-consistent (bijective) matches are included.
    """
    mask = (labels_a > 0) | (labels_b > 0)
    a = labels_a[mask].astype(np.int64)
    b = labels_b[mask].astype(np.int64)

    # Encode (a_val, b_val) pair as a single integer key
    max_b = int(labels_b.max()) if labels_b.max() > 0 else 1
    keys = a * (max_b + 1) + b
    unique_keys, counts = np.unique(keys, return_counts=True)

    a_vals = (unique_keys // (max_b + 1)).astype(int)
    b_vals = (unique_keys % (max_b + 1)).astype(int)

    # For each label in A, find the B label with maximum overlap
    a_best = {}  # a_label -> (b_label, count)
    for av, bv, cnt in zip(a_vals, b_vals, counts):
        if av == 0:
            continue
        if av not in a_best or cnt > a_best[av][1]:
            a_best[av] = (bv, cnt)

    # For each label in B, find the A label with maximum overlap
    b_best = {}  # b_label -> (a_label, count)
    for av, bv, cnt in zip(a_vals, b_vals, counts):
        if bv == 0:
            continue
        if bv not in b_best or cnt > b_best[bv][1]:
            b_best[bv] = (av, cnt)

    # Retain only mutually-consistent (bijective) pairs
    a_to_b = {}
    b_to_a = {}
    for ai, (bi, _) in a_best.items():
        if bi > 0 and b_best.get(bi, (None,))[0] == ai:
            a_to_b[ai] = bi
            b_to_a[bi] = ai

    return a_to_b, b_to_a


def feature_matched_diff_mask(labels_a, labels_b, a_to_b, b_to_a):
    """Boolean mask of true spatial differences after feature-label matching.

    For each mutually-matched pair ``(i, j)``: pixels in
    ``(labels_a == i) XOR (labels_b == j)`` are true differences — one method
    claims the pixel for that feature, the other doesn't.
    All pixels belonging to unmatched features are also flagged as differences.

    Parameters
    ----------
    labels_a, labels_b : np.ndarray of int
    a_to_b : dict  from :func:`match_features_by_overlap`
    b_to_a : dict  from :func:`match_features_by_overlap`

    Returns
    -------
    diff_mask : np.ndarray of bool
    """
    diff_mask = np.zeros(labels_a.shape, dtype=bool)

    # Matched pairs: symmetric difference of their pixel sets
    for ai, bi in a_to_b.items():
        diff_mask |= (labels_a == ai) ^ (labels_b == bi)

    # Pixels belonging to unmatched features in A
    for ai in np.unique(labels_a[labels_a > 0]):
        if ai not in a_to_b:
            diff_mask |= labels_a == ai

    # Pixels belonging to unmatched features in B
    for bi in np.unique(labels_b[labels_b > 0]):
        if bi not in b_to_a:
            diff_mask |= labels_b == bi

    return diff_mask


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bfs_dir", help="Directory with BFS cloudid .npz files")
    parser.add_argument("--edt_dir", help="Directory with EDT cloudid .npz files")
    parser.add_argument(
        "--tb_dir", default="",
        help="Directory with input Tb .nc files, for background plotting - only "
             "needed as a fallback for older .npz files that don't already "
             "carry their own tb/lat/lon (current plot_label_grow_speedup.py "
             "output always does)",
    )
    parser.add_argument("--outdir", default=".", help="Output directory for figures")
    parser.add_argument(
        "--compare-stats", action="store_true",
        help="Compare track stats instead of cloudid arrays",
    )
    parser.add_argument("--bfs_stats", help="Directory with BFS stats files")
    parser.add_argument("--edt_stats", help="Directory with EDT stats files")
    parser.add_argument(
        "--nfiles", type=int, default=0,
        help="Number of files to plot (0 = all)",
    )
    return parser.parse_args()


def load_tb_for_frame(tb_dir, frame_name):
    """
    Legacy fallback only: try to load Tb data for a frame by matching the
    input .nc filename. Current plot_label_grow_speedup.py output always
    embeds tb (and lat/lon) directly in the .npz, so this path isn't
    exercised for anything generated by this version of that script - it
    only helps with older .npz files saved before that change, and even
    then only recovers tb, not lat/lon (so those frames fall back to plain
    pixel-index axes rather than the map projection - see plot_comparison).
    """
    if not tb_dir:
        return None
    # frame_name is like "MCS-test-4_20200101.npz" -> look for matching .nc
    nc_name = frame_name.replace(".npz", ".nc")
    nc_path = os.path.join(tb_dir, nc_name)
    if not os.path.exists(nc_path):
        # Try glob
        matches = glob.glob(os.path.join(tb_dir, "*" + frame_name[:8] + "*.nc"))
        if matches:
            nc_path = matches[0]
        else:
            return None

    try:
        import xarray as xr
        ds = xr.open_dataset(nc_path, decode_timedelta=False)
        for varname in ["Tb", "tb", "brightness_temperature"]:
            if varname in ds:
                tb = ds[varname].values
                ds.close()
                if tb.ndim == 3:
                    tb = tb[0, :, :]
                return tb
        ds.close()
    except Exception:
        pass
    return None


def _compute_figsize(lat, lon, n_panels=3, base_width=5.0, aspect_bounds=(0.3, 3.0)):
    """
    Size a figure of n_panels side-by-side PlateCarree panels so each one's
    box matches the domain's actual lat/lon aspect ratio - avoids both
    distortion (squishing the map to fit an arbitrary panel shape) and
    wasted letterboxing whitespace (a panel shaped very differently from the
    map it contains).
    """
    lat_span = float(np.max(lat) - np.min(lat))
    lon_span = float(np.max(lon) - np.min(lon))
    aspect = lat_span / lon_span if lon_span > 0 else 1.0
    aspect = float(np.clip(aspect, aspect_bounds[0], aspect_bounds[1]))
    panel_height = base_width * aspect
    return base_width * n_panels, panel_height


def _add_map_layers(ax, proj, lon, lat):
    """Land outline + bottom/left-only lat/lon labels with no grid lines."""
    ax.set_extent(
        [float(np.min(lon)), float(np.max(lon)), float(np.min(lat)), float(np.max(lat))],
        crs=proj,
    )
    # zorder above the data layers (tb ~1, label overlay ~2) so the outline
    # actually draws on top instead of being painted over by an opaque fill -
    # matches Analysis/plot_subset_tbpf_mcs_tracks_demo.py's zorder=3 exactly.
    ax.add_feature(_LAND, facecolor="none", edgecolor="lightgray", linewidth=1, zorder=3)
    gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=0, zorder=4)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = False
    gl.ylines = False


def plot_comparison(bfs_data, edt_data, tb, lat, lon, frame_name, outdir):
    """Create 3-panel comparison figure for one frame, on a PlateCarree map."""
    bfs_labels = bfs_data["convcold_cloudnumber"]
    edt_labels = edt_data["convcold_cloudnumber"]

    ny, nx = bfs_labels.shape

    # Compute statistics using feature-matched comparison: label-number swaps
    # for similarly-sized features (due to BFS vs EDT boundary differences) do
    # not count as real spatial differences.
    n_bfs = len(np.unique(bfs_labels[bfs_labels > 0]))
    n_edt = len(np.unique(edt_labels[edt_labels > 0]))
    bfs_to_edt, edt_to_bfs = match_features_by_overlap(bfs_labels, edt_labels)
    diff_mask = feature_matched_diff_mask(bfs_labels, edt_labels, bfs_to_edt, edt_to_bfs)
    n_diff = np.count_nonzero(diff_mask)
    n_total = bfs_labels.size
    agreement = 1.0 - n_diff / n_total

    have_latlon = lat is not None and lon is not None
    proj = ccrs.PlateCarree()

    if have_latlon:
        figsize = _compute_figsize(lat, lon, n_panels=3)
        fig, axes = plt.subplots(
            1, 3, figsize=figsize, constrained_layout=True,
            subplot_kw={"projection": proj},
        )
    else:
        # Legacy fallback (see load_tb_for_frame) - no lat/lon available,
        # plot on plain pixel-index axes instead of a map.
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    fig.suptitle(
        f"{frame_name}  |  BFS: {n_bfs} features, EDT: {n_edt} features, "
        f"Agreement: {agreement*100:.3f}%",
        fontsize=10,
    )

    # Colormap for labels
    n_labels = max(np.max(bfs_labels), np.max(edt_labels), 1)
    cmap_labels = plt.colormaps.get_cmap("tab20").resampled(n_labels)

    for ax, labels, title in [
        (axes[0], bfs_labels, "BFS"),
        (axes[1], edt_labels, "EDT"),
    ]:
        if have_latlon:
            if tb is not None:
                ax.pcolormesh(lon, lat, tb, cmap="gray_r", vmin=180, vmax=300,
                               transform=proj, shading="auto", zorder=1)
            if np.any(labels > 0):
                masked = np.ma.masked_where(labels == 0, labels)
                ax.pcolormesh(lon, lat, masked, cmap=cmap_labels, alpha=0.4,
                               vmin=1, vmax=n_labels, transform=proj,
                               shading="auto", zorder=2)
                ax.contour(lon, lat, labels > 0, colors="red", linewidths=0.5,
                           transform=proj, zorder=2.5)
            _add_map_layers(ax, proj, lon, lat)
        else:
            if tb is not None:
                ax.imshow(tb, cmap="gray_r", vmin=180, vmax=300, aspect="auto")
            if np.any(labels > 0):
                masked = np.ma.masked_where(labels == 0, labels)
                ax.imshow(masked, cmap=cmap_labels, alpha=0.4, aspect="auto",
                          vmin=1, vmax=n_labels)
                ax.contour(labels > 0, colors="red", linewidths=0.5)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        ax.set_title(title)

    # Panel 3: Difference map
    ax3 = axes[2]
    diff_cmap = mcolors.ListedColormap(["red"])
    if have_latlon:
        if tb is not None:
            ax3.pcolormesh(lon, lat, tb, cmap="gray_r", vmin=180, vmax=300,
                            transform=proj, shading="auto", alpha=0.5, zorder=1)
        if n_diff > 0:
            diff_display = np.ma.masked_where(~diff_mask, diff_mask.astype(float))
            ax3.pcolormesh(lon, lat, diff_display, cmap=diff_cmap, transform=proj,
                            shading="auto", alpha=0.8, zorder=2)
        _add_map_layers(ax3, proj, lon, lat)
    else:
        if tb is not None:
            ax3.imshow(tb, cmap="gray_r", vmin=180, vmax=300, aspect="auto", alpha=0.5)
        diff_display = np.zeros((ny, nx, 4))  # RGBA
        if n_diff > 0:
            diff_display[diff_mask, 0] = 1.0  # Red
            diff_display[diff_mask, 3] = 0.8  # Alpha
        ax3.imshow(diff_display, aspect="auto")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
    ax3.set_title(f"Feature-matched diffs ({n_diff} pixels, {n_diff/n_total*100:.4f}%)")

    # frame_name may or may not carry a .npz suffix (direct in-memory callers,
    # like tests/plot_label_grow_synthetic.py, don't have one).
    base_name = frame_name[:-4] if frame_name.endswith(".npz") else frame_name
    outfile = os.path.join(outdir, f"comparison_{base_name}.png")
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outfile


def compare_stats(bfs_stats_dir, edt_stats_dir):
    """Compare final track statistics between BFS and EDT runs."""
    import xarray as xr

    # Find stats files
    bfs_files = sorted(glob.glob(os.path.join(bfs_stats_dir, "mcs_tracks_final_*.nc")))
    edt_files = sorted(glob.glob(os.path.join(edt_stats_dir, "mcs_tracks_final_*.nc")))

    if not bfs_files:
        print(f"  No BFS stats files found in {bfs_stats_dir}")
        return
    if not edt_files:
        print(f"  No EDT stats files found in {edt_stats_dir}")
        return

    bfs_ds = xr.open_dataset(bfs_files[-1])
    edt_ds = xr.open_dataset(edt_files[-1])

    print("\n  Track Statistics Comparison:")
    print(f"  {'Metric':<30} {'BFS':>10} {'EDT':>10} {'Diff':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")

    # Number of tracks
    n_bfs = bfs_ds.dims.get("tracks", 0)
    n_edt = edt_ds.dims.get("tracks", 0)
    print(f"  {'Number of tracks':<30} {n_bfs:>10} {n_edt:>10} {n_edt - n_bfs:>+10}")

    # Compare common variables. Note: "movement_distance" (not
    # "movement_distance_total" - that variable doesn't exist in
    # mcs_tracks_final_*.nc, so it was silently skipped by the `if var in
    # ds` guard below rather than raising).
    for var in ["track_duration", "mcs_duration", "movement_distance"]:
        if var in bfs_ds and var in edt_ds:
            bfs_mean = float(bfs_ds[var].mean())
            edt_mean = float(edt_ds[var].mean())
            print(
                f"  {f'mean {var}':<30} {bfs_mean:>10.2f} {edt_mean:>10.2f} "
                f"{edt_mean - bfs_mean:>+10.2f}"
            )

    bfs_ds.close()
    edt_ds.close()


def main():
    args = parse_args()

    if args.compare_stats:
        if not args.bfs_stats or not args.edt_stats:
            raise SystemExit("--compare-stats requires --bfs_stats and --edt_stats")
        compare_stats(args.bfs_stats, args.edt_stats)
        return

    if not args.bfs_dir or not args.edt_dir:
        raise SystemExit("Provide --bfs_dir and --edt_dir (from plot_label_grow_speedup.py output)")

    os.makedirs(args.outdir, exist_ok=True)

    # Find matching files
    bfs_files = sorted(glob.glob(os.path.join(args.bfs_dir, "*.npz")))
    edt_files = sorted(glob.glob(os.path.join(args.edt_dir, "*.npz")))

    if not bfs_files:
        raise SystemExit(f"No .npz files found in {args.bfs_dir}")
    if not edt_files:
        raise SystemExit(f"No .npz files found in {args.edt_dir}")

    # Match by filename
    bfs_names = {os.path.basename(f): f for f in bfs_files}
    edt_names = {os.path.basename(f): f for f in edt_files}
    common = sorted(set(bfs_names.keys()) & set(edt_names.keys()))

    if not common:
        raise SystemExit("No matching filenames between BFS and EDT directories")

    if args.nfiles > 0:
        common = common[: args.nfiles]

    print(f"\n  Comparing {len(common)} frame(s)...")
    print(f"  Output: {args.outdir}")

    saved = []
    total_diff = 0
    total_pixels = 0

    for frame_name in common:
        bfs_data = np.load(bfs_names[frame_name])
        edt_data = np.load(edt_names[frame_name])

        # tb/lat/lon are embedded directly in the .npz by current
        # plot_label_grow_speedup.py - only fall back to --tb_dir file
        # matching (tb only, no lat/lon) for older .npz files without them.
        if "tb" in bfs_data:
            tb = bfs_data["tb"]
        else:
            tb = load_tb_for_frame(args.tb_dir, frame_name)
        lat = bfs_data["lat"] if "lat" in bfs_data else None
        lon = bfs_data["lon"] if "lon" in bfs_data else None

        outfile = plot_comparison(bfs_data, edt_data, tb, lat, lon, frame_name, args.outdir)
        saved.append(outfile)

        # Accumulate stats
        bfs_labels = bfs_data["convcold_cloudnumber"]
        edt_labels = edt_data["convcold_cloudnumber"]
        total_diff += np.count_nonzero(bfs_labels != edt_labels)
        total_pixels += bfs_labels.size

    # Summary
    print(f"\n  Summary:")
    print(f"    Frames compared: {len(common)}")
    print(f"    Total pixels: {total_pixels:,}")
    print(f"    Total differing pixels: {total_diff:,}")
    print(f"    Overall agreement: {(1 - total_diff / total_pixels) * 100:.4f}%")
    print(f"\n  Figures saved to: {args.outdir}")
    for f in saved[:5]:
        print(f"    {f}")
    if len(saved) > 5:
        print(f"    ... and {len(saved) - 5} more")


if __name__ == "__main__":
    main()
