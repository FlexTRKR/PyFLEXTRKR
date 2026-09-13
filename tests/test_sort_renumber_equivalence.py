#!/usr/bin/env python
"""
Differential test: prove the rewritten sort_renumber (pyflextrkr/ftfunctions.py)
produces IDENTICAL output to the original, pre-rewrite implementation.

Context
-------
sort_renumber counts pixels per label and renumbers labels by size, and is
called from nearly every feature-identification code path in PyFLEXTRKR
(label_and_grow_features, label_and_grow_cold_clouds, idclouds_tbpf,
matchtbpf_func, matchtbradar_func, idvorticity_era5, idcoldpool,
idfeature_generic). The original implementation rescanned the *entire* input
array once per label, twice over (once to count pixels/area per label, once
more to renumber each surviving label) - O(n_labels x domain_size), profiled
directly at ~500s/frame on a 1200x3600 global Tb field with ~1000+ core
labels (see the plot_label_grow_speedup.py --demo demo_mcs_imerg_mcsmip
1.1x-speedup investigation this test accompanies). The rewrite replaces both
scans with single-pass np.unique/np.bincount + lookup-table operations -
O(domain_size + n_labels).

Given how widely sort_renumber is used, this file proves output equivalence
rather than just checking that behavior "looks reasonable": _sort_renumber_reference
below is a verbatim, frozen copy of the pre-rewrite function body (copied
before the rewrite landed, comments included), and every test compares the
live sort_renumber against it on the same input, asserting *exact* equality
of both returned arrays.

Usage
-----
  python -m pytest tests/test_sort_renumber_equivalence.py -v
  # Real-data tests additionally need:
  export PYFLEXTRKR_TEST_DATA=~/data/demo
  python tests/run_demo_tests.py --demos demo_mcs_tbpf_idealized demo_mcs_imerg -n 4
"""
import os
import sys
import glob
import numpy as np
import pytest

# Needed for `from pyflextrkr...` imports to resolve under a bare
# `pytest tests/` invocation (no tests/__init__.py) - see
# test_label_grow_methods.py for the same pattern/explanation.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyflextrkr.ftfunctions import sort_renumber


# ---------------------------------------------------------------------------
# Frozen reference: verbatim copy of sort_renumber's pre-rewrite body, from
# pyflextrkr/ftfunctions.py before the np.unique/bincount + lookup-table
# rewrite. Deliberately kept exactly as it was (including its original
# comments) so this is a true differential test against the actual prior
# behavior, not a hand-written "expected" re-implementation that could share
# blind spots with the new one.
# ---------------------------------------------------------------------------
def _sort_renumber_reference(labelcell_number2d, min_size, grid_area=None):
    """Frozen pre-rewrite sort_renumber (see module docstring)."""
    sortedlabelcell_number2d = np.zeros(np.shape(labelcell_number2d), dtype=int)
    nlabelcells = np.nanmax(labelcell_number2d)

    if nlabelcells > 0:
        labelcell_npix = np.full(nlabelcells, -999, dtype=int)
        for ilabelcell in range(1, nlabelcells + 1):
            ilabelcell_npix = np.count_nonzero(labelcell_number2d == ilabelcell)
            if grid_area is None:
                if ilabelcell_npix > min_size:
                    labelcell_npix[ilabelcell - 1] = ilabelcell_npix
            else:
                ilabelcell_area = np.sum(grid_area[labelcell_number2d == ilabelcell])
                if ilabelcell_area > min_size:
                    labelcell_npix[ilabelcell - 1] = ilabelcell_npix

        ivalidcells = np.where(labelcell_npix > 0)[0]
        ncells = len(ivalidcells)

        if ncells > 0:
            labelcell_number1d = np.copy(ivalidcells) + 1
            labelcell_npix = labelcell_npix[ivalidcells]

            order = np.argsort(labelcell_npix)[::-1]

            sortedcell_npix = np.copy(labelcell_npix[order])
            sortedcell_number1d = np.copy(labelcell_number1d[order])

            cellstep = 0
            for icell in range(0, ncells):
                sortedcell_indices = np.where(
                    labelcell_number2d == sortedcell_number1d[icell]
                )
                nsortedcellindices = len(sortedcell_indices[1])
                if nsortedcellindices == sortedcell_npix[icell]:
                    cellstep += 1
                    sortedlabelcell_number2d[sortedcell_indices] = np.copy(cellstep)
        else:
            sortedcell_npix = np.zeros(0)
    else:
        sortedcell_npix = np.zeros(0)

    return (sortedlabelcell_number2d, sortedcell_npix)


def _assert_equivalent(labelcell_number2d, min_size, grid_area=None, label=""):
    """Run both implementations on the same input; assert exact equality."""
    new_2d, new_npix = sort_renumber(
        labelcell_number2d.copy(), min_size, grid_area=grid_area
    )
    ref_2d, ref_npix = _sort_renumber_reference(
        labelcell_number2d.copy(), min_size, grid_area=grid_area
    )
    assert np.array_equal(new_2d, ref_2d), (
        f"[{label}] 2D relabeled arrays differ:\n"
        f"  new sums/labels present: {sorted(np.unique(new_2d).tolist())}\n"
        f"  ref sums/labels present: {sorted(np.unique(ref_2d).tolist())}"
    )
    assert np.array_equal(new_npix, ref_npix), (
        f"[{label}] npix arrays differ: new={new_npix} ref={ref_npix}"
    )


# ---------------------------------------------------------------------------
# Synthetic edge cases
# ---------------------------------------------------------------------------

def test_no_cells():
    arr = np.zeros((10, 10), dtype=int)
    _assert_equivalent(arr, min_size=0, label="no_cells")


def test_single_cell():
    arr = np.zeros((10, 10), dtype=int)
    arr[2:5, 2:5] = 1
    _assert_equivalent(arr, min_size=0, label="single_cell")


def test_all_cells_below_threshold():
    arr = np.zeros((10, 10), dtype=int)
    arr[0, 0] = 1
    arr[5, 5] = 2
    _assert_equivalent(arr, min_size=100, label="all_below_threshold")


def test_label_gap_all_survive():
    """Labels {1, 3, 4} present, label 2 absent entirely - the scenario the
    reverted np.unique attempt (see the comment preserved in
    ftfunctions.py's history) got wrong."""
    arr = np.zeros((12, 12), dtype=int)
    arr[0:2, 0:2] = 1     # 4 px
    # label 2 deliberately never used
    arr[4:8, 4:8] = 3     # 16 px
    arr[9:10, 9:12] = 4   # 3 px
    _assert_equivalent(arr, min_size=0, label="label_gap_all_survive")


def test_label_gap_from_size_filter():
    """A mid-numbered label fails the size filter, creating a gap in the
    *surviving* labels used during renumbering (sortedcell_number1d)."""
    arr = np.zeros((12, 12), dtype=int)
    arr[0:3, 0:3] = 1      # 9 px, survives
    arr[5, 5] = 2          # 1 px, filtered out
    arr[7:11, 7:11] = 3    # 16 px, survives
    _assert_equivalent(arr, min_size=2, label="label_gap_from_size_filter")


def test_tied_sizes():
    """Multiple cells with the exact same pixel count - exercises argsort's
    tie-breaking, which must match exactly since the rewrite reuses the same
    argsort call unchanged, only feeding it differently-computed counts."""
    arr = np.zeros((12, 12), dtype=int)
    arr[0:2, 0:2] = 1    # 4 px
    arr[4:6, 4:6] = 2    # 4 px
    arr[8:10, 8:10] = 3  # 4 px
    arr[0:2, 8:10] = 4   # 4 px
    _assert_equivalent(arr, min_size=0, label="tied_sizes")


def test_with_grid_area():
    arr = np.zeros((10, 10), dtype=int)
    arr[0:2, 0:2] = 1    # 4 px
    arr[5:9, 5:9] = 2    # 16 px
    grid_area = np.ones((10, 10)) * 100.0
    grid_area[0:5, :] = 25.0  # smaller cells in the top half
    # Cell 1 (top half): area = 4*25=100; cell 2 (bottom half): area=16*100=1600
    _assert_equivalent(arr, min_size=500, grid_area=grid_area, label="grid_area_basic")


def test_grid_area_with_gap():
    """grid_area combined with a label gap created by the size filter."""
    arr = np.zeros((10, 10), dtype=int)
    arr[0:2, 0:2] = 1     # top-left, small area -> filtered
    arr[3, 3] = 2         # single pixel, mid area -> filtered
    arr[5:9, 5:9] = 3     # bottom-right, large area -> survives
    grid_area = np.ones((10, 10)) * 10.0
    grid_area[5:9, 5:9] = 200.0
    _assert_equivalent(arr, min_size=50, grid_area=grid_area, label="grid_area_with_gap")


def test_irregular_shapes():
    """Non-rectangular, disconnected-looking regions (diagonal stripe,
    L-shape, isolated single pixels) rather than clean rectangles."""
    arr = np.zeros((30, 30), dtype=int)
    for i in range(29):
        arr[i, i] = 1
        arr[i, i + 1] = 1
    arr[10:15, 20] = 2
    arr[14, 20:25] = 2
    arr[25, 3] = 3
    arr[26, 25] = 4
    _assert_equivalent(arr, min_size=1, label="irregular_shapes")


def test_float_dtype_with_nan():
    """Float input with NaN, matching what the function's own use of nanmax
    implies it should tolerate.

    Not a like-for-like equivalence check: the frozen reference (old code)
    passes nlabelcells straight from np.nanmax() into np.full(nlabelcells,
    ...) with no int() cast, and current numpy (2.x) hard-errors on a
    float64 shape argument there (TypeError, verified directly - this is a
    latent, pre-existing bug in the *old* implementation, unrelated to and
    predating this rewrite; real callers only ever pass int-dtype label
    arrays in practice, which is presumably why it was never hit). The
    rewrite happens to fix this as a side effect, since it casts nlabelcells
    to a plain int up front. Documented here rather than silently dropped,
    and checked in both directions so this doesn't bit-rot unnoticed."""
    arr = np.zeros((10, 10), dtype=float)
    arr[0:2, 0:2] = 1.0
    arr[5:8, 5:8] = 2.0
    arr[9, 9] = np.nan

    with pytest.raises(TypeError):
        _sort_renumber_reference(arr.copy(), min_size=0)

    new_2d, new_npix = sort_renumber(arr.copy(), min_size=0)
    assert new_2d.shape == arr.shape
    assert np.array_equal(np.unique(new_2d), [0, 1, 2])
    assert np.array_equal(np.sort(new_npix)[::-1], [9, 4])


def test_negative_labels():
    """Negative values must resolve to background (0), matching the frozen
    reference: its loop only ever tests `== ilabelcell` for ilabelcell in
    [1, nlabelcells], so a negative value never matches any positive label
    and is implicitly background.

    This is a real regression check, not just a hypothetical: the version
    of sort_renumber pushed earlier in this same PR (commit 64c49f9, before
    the shared _labels_as_int helper existed) fed the raw int array straight
    into np.bincount without clamping negatives first, and np.bincount
    raises ValueError on negative input - verified directly against that
    commit. _labels_as_int's negative-clamp (added alongside the
    sort_renumber2vars fix) restores parity with the true original
    behavior here."""
    arr = np.array([[1, -5, 2], [0, 1, 2]])
    _assert_equivalent(arr, min_size=0, label="negative_labels")


def test_large_label_count():
    """Many small, individually-labeled single-pixel cells - stresses the
    per-label bookkeeping (bincount/LUT sizing) at a larger label count than
    the other synthetic cases."""
    rng = np.random.default_rng(7)
    ny, nx = 80, 80
    arr = np.zeros((ny, nx), dtype=int)
    mask = rng.random((ny, nx)) < 0.08
    ys, xs = np.nonzero(mask)
    arr[ys, xs] = np.arange(1, len(ys) + 1)
    _assert_equivalent(arr, min_size=0, label="large_label_count")


# ---------------------------------------------------------------------------
# Randomized synthetic arrays
# ---------------------------------------------------------------------------

def test_randomized_arrays():
    """Many random labeled arrays (varying size, label count, and size
    distribution, with deliberate gaps from dropped labels), each checked
    for exact equivalence against the frozen reference."""
    from scipy.ndimage import label as ndi_label, binary_dilation

    rng = np.random.default_rng(12345)
    n_cases = 60
    n_run = 0
    for case in range(n_cases):
        ny = int(rng.integers(15, 60))
        nx = int(rng.integers(15, 60))
        n_seeds = int(rng.integers(1, 25))

        seed_mask = np.zeros((ny, nx), dtype=bool)
        ys = rng.integers(0, ny, size=n_seeds)
        xs = rng.integers(0, nx, size=n_seeds)
        seed_mask[ys, xs] = True

        n_dilate = int(rng.integers(0, 4))
        for _ in range(n_dilate):
            seed_mask = binary_dilation(seed_mask)

        labeled, nlbl = ndi_label(seed_mask)
        if nlbl == 0:
            continue

        # Randomly blank out some whole labels to force gaps in numbering -
        # the array keeps its original max label value (via nanmax in both
        # implementations) even though some intermediate labels are absent.
        if nlbl > 2:
            n_drop = int(rng.integers(0, nlbl // 2))
            if n_drop > 0:
                drop_labels = rng.choice(
                    np.arange(1, nlbl + 1), size=n_drop, replace=False
                )
                for dl in drop_labels:
                    labeled[labeled == dl] = 0

        min_size = float(rng.choice([0, 1, 2, 5]))
        grid_area = None
        if rng.random() < 0.3:
            grid_area = rng.uniform(0.5, 5.0, size=(ny, nx))
            min_size = float(rng.choice([0.0, 1.0, 3.0]))

        _assert_equivalent(labeled, min_size, grid_area, label=f"random_case_{case}")
        n_run += 1

    assert n_run > 30, f"Expected most of {n_cases} random cases to run, only {n_run} did"


# ---------------------------------------------------------------------------
# Real captured data: idealized + South America IMERG demos
# ---------------------------------------------------------------------------
DATA_ROOT = os.environ.get("PYFLEXTRKR_TEST_DATA", "")


def _demo_path(*parts):
    return os.path.join(DATA_ROOT, *parts)


def _find_input_files(input_dir, pattern="*.nc", n=0):
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        files = sorted(glob.glob(os.path.join(input_dir, "**", pattern), recursive=True))
    if n > 0:
        files = files[:n]
    return files


def _load_tb_frames(input_dir, pattern, tb_varname="Tb", n_files=0, n_times_per_file=2):
    """Yield preprocessed 2D Tb frames from real demo input files (same
    preprocessing idclouds_tbpf/plot_label_grow_speedup.py apply)."""
    import xarray as xr
    from scipy.signal import medfilt2d

    files = _find_input_files(input_dir, pattern, n_files)
    for filepath in files:
        ds = xr.open_dataset(filepath, decode_timedelta=False)
        varname = tb_varname if tb_varname in ds else None
        if varname is None:
            for alt in ["Tb", "tb"]:
                if alt in ds:
                    varname = alt
                    break
        if varname is None:
            ds.close()
            continue
        tb_all = ds[varname].values
        ds.close()
        if tb_all.ndim == 2:
            tb_all = tb_all[np.newaxis, :, :]
        n_times = (
            min(n_times_per_file, tb_all.shape[0])
            if n_times_per_file > 0 else tb_all.shape[0]
        )
        for it in range(n_times):
            tb = tb_all[it, :, :]
            tb_filt = medfilt2d(tb.astype(np.float64), kernel_size=5)
            out_tb = np.copy(tb)
            missmask = np.isnan(tb)
            out_tb[missmask] = tb_filt[missmask]
            out_tb[out_tb < 160] = np.nan
            out_tb[out_tb > 330] = np.nan
            yield out_tb


@pytest.mark.local
def test_real_idealized_core_labels():
    """Real labeled-core arrays from demo_mcs_tbpf_idealized, as
    sort_renumber actually receives them inside label_and_grow_features."""
    if not DATA_ROOT:
        pytest.skip("PYFLEXTRKR_TEST_DATA not set")
    from pyflextrkr.label_and_grow_features import smooth_field, find_and_label_cores

    input_dir = _demo_path("mcs_tbpf/idealized/test4/input")
    if not _find_input_files(input_dir, "*.nc"):
        pytest.skip(f"No idealized demo input files found under {input_dir}")

    n_checked = 0
    for tb in _load_tb_frames(input_dir, "*.nc", n_files=0, n_times_per_file=26):
        smoothed = smooth_field(tb, 5)
        labeled_cores, nlabelcores = find_and_label_cores(smoothed, 225.0, "lt")
        if nlabelcores == 0:
            continue
        _assert_equivalent(labeled_cores, 4, label=f"idealized_frame_{n_checked}")
        n_checked += 1
    assert n_checked > 0, "No idealized frames with cores were found to test against"


@pytest.mark.local
def test_real_south_america_core_labels():
    """Real labeled-core arrays from demo_mcs_imerg (South America), as
    sort_renumber actually receives them inside label_and_grow_features."""
    if not DATA_ROOT:
        pytest.skip("PYFLEXTRKR_TEST_DATA not set")
    from pyflextrkr.label_and_grow_features import smooth_field, find_and_label_cores

    input_dir = _demo_path("mcs_tbpf/imerg/input")
    if not _find_input_files(input_dir, "merg_*.nc"):
        pytest.skip(f"No South America IMERG demo input files found under {input_dir}")

    n_checked = 0
    for tb in _load_tb_frames(input_dir, "merg_*.nc", n_files=6, n_times_per_file=2):
        smoothed = smooth_field(tb, 10)
        labeled_cores, nlabelcores = find_and_label_cores(smoothed, 225.0, "lt")
        if nlabelcores == 0:
            continue
        _assert_equivalent(labeled_cores, 4, label=f"south_america_frame_{n_checked}")
        n_checked += 1
    assert n_checked > 0, "No South America frames with cores were found to test against"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
