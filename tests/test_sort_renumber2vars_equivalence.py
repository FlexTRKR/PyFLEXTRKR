#!/usr/bin/env python
"""
Differential test: prove the rewritten sort_renumber2vars (pyflextrkr/ftfunctions.py)
produces IDENTICAL output to the original, pre-rewrite implementation.

Context
-------
sort_renumber2vars is sort_renumber's two-array sibling: it counts/filters
cells in labelcell_number2d, then renumbers *both* labelcell_number2d and
labelcell2_number2d using that same size-based ordering. It has the identical
O(n_labels x domain_size) bug already fixed in sort_renumber
(test_sort_renumber_equivalence.py) - one full-domain rescan per label to
count pixels/area, then, per surviving cell, *two* more full-domain rescans
to renumber (one per array) - and is used by idclouds_tbpf.py's linkpf path,
reached by every demo config in this repo (all three set linkpf: 1).

Given how directly this mirrors sort_renumber, this file mirrors
test_sort_renumber_equivalence.py's structure and rigor:
_sort_renumber2vars_reference below is a verbatim, frozen copy of the
pre-rewrite function body, and every test compares the live
sort_renumber2vars against it on the same input, asserting *exact* equality
of all three returned values.

This file additionally covers what's specific to the two-array case: var1
and var2 are matched by label *value* (not spatial position) - the original
loop looked up sortedcell_number1d[icell] (a value from var1) in *both*
arrays - and var2 can carry label values var1 never does (link_pf_tb
renumbers the two arrays semi-independently), which the lookup-table
renumbering must size for explicitly (see the "var2 max label" tests below).

Usage
-----
  python -m pytest tests/test_sort_renumber2vars_equivalence.py -v
  # Real-data tests additionally need:
  export PYFLEXTRKR_TEST_DATA=~/data/demo
  python tests/run_demo_tests.py --demos demo_mcs_tbpf_idealized demo_mcs_imerg demo_mcs_imerg_mcsmip -n 8
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

from pyflextrkr.ftfunctions import sort_renumber2vars


# ---------------------------------------------------------------------------
# Frozen reference: verbatim copy of sort_renumber2vars's pre-rewrite body,
# from pyflextrkr/ftfunctions.py before the np.bincount + lookup-table
# rewrite. Deliberately kept exactly as it was (including its original
# comments) so this is a true differential test against the actual prior
# behavior, not a hand-written "expected" re-implementation that could share
# blind spots with the new one.
# ---------------------------------------------------------------------------
def _sort_renumber2vars_reference(
    labelcell_number2d, labelcell2_number2d, min_cellpix, grid_area=None
):
    """Frozen pre-rewrite sort_renumber2vars (see module docstring)."""
    sortedlabelcell_number2d = np.zeros(np.shape(labelcell_number2d), dtype=int)
    sortedlabelcell2_number2d = np.zeros(np.shape(labelcell_number2d), dtype=int)

    nlabelcells = np.nanmax(labelcell_number2d)

    if nlabelcells > 0:

        labelcell_npix = np.full(nlabelcells, -999, dtype=int)
        for ilabelcell in range(1, nlabelcells + 1):
            ilabelcell_npix = np.count_nonzero(labelcell_number2d == ilabelcell)
            if grid_area is None:
                if ilabelcell_npix > min_cellpix:
                    labelcell_npix[ilabelcell - 1] = ilabelcell_npix
            else:
                ilabelcell_area = np.sum(grid_area[labelcell_number2d == ilabelcell])
                if ilabelcell_area > min_cellpix:
                    labelcell_npix[ilabelcell - 1] = ilabelcell_npix

        ivalidcells = np.array(np.where(labelcell_npix > 0))[0, :]
        ncells = len(ivalidcells)

        if ncells > 0:
            labelcell_number1d = np.copy(ivalidcells) + 1
            labelcell_npix = labelcell_npix[ivalidcells]

            order = np.argsort(labelcell_npix)
            order = order[::-1]

            sortedcell_npix = np.copy(labelcell_npix[order])
            sortedcell_number1d = np.copy(labelcell_number1d[order])

            cellstep = 0
            for icell in range(0, ncells):
                sortedcell_indices = np.where(
                    labelcell_number2d == sortedcell_number1d[icell]
                )
                sortedcell2_indices = np.where(
                    labelcell2_number2d == sortedcell_number1d[icell]
                )
                nsortedcellindices = len(sortedcell_indices[1])
                if nsortedcellindices == sortedcell_npix[icell]:
                    cellstep += 1
                    sortedlabelcell_number2d[sortedcell_indices] = np.copy(cellstep)
                    sortedlabelcell2_number2d[sortedcell2_indices] = np.copy(cellstep)

        else:
            sortedcell_npix = np.zeros(0)
    else:
        sortedcell_npix = np.zeros(0)

    return (
        sortedlabelcell_number2d,
        sortedlabelcell2_number2d,
        sortedcell_npix,
    )


def _assert_equivalent(
    labelcell_number2d, labelcell2_number2d, min_cellpix, grid_area=None, label=""
):
    """Run both implementations on the same input; assert exact equality."""
    new_1, new_2, new_npix = sort_renumber2vars(
        labelcell_number2d.copy(), labelcell2_number2d.copy(), min_cellpix,
        grid_area=grid_area,
    )
    ref_1, ref_2, ref_npix = _sort_renumber2vars_reference(
        labelcell_number2d.copy(), labelcell2_number2d.copy(), min_cellpix,
        grid_area=grid_area,
    )
    assert np.array_equal(new_1, ref_1), (
        f"[{label}] var1 relabeled arrays differ:\n"
        f"  new labels present: {sorted(np.unique(new_1).tolist())}\n"
        f"  ref labels present: {sorted(np.unique(ref_1).tolist())}"
    )
    assert np.array_equal(new_2, ref_2), (
        f"[{label}] var2 relabeled arrays differ:\n"
        f"  new labels present: {sorted(np.unique(new_2).tolist())}\n"
        f"  ref labels present: {sorted(np.unique(ref_2).tolist())}"
    )
    assert np.array_equal(new_npix, ref_npix), (
        f"[{label}] npix arrays differ: new={new_npix} ref={ref_npix}"
    )
    return ref_1, ref_2, ref_npix


# ---------------------------------------------------------------------------
# Synthetic edge cases shared with sort_renumber (single-array behavior)
# ---------------------------------------------------------------------------

def test_no_cells():
    arr = np.zeros((10, 10), dtype=int)
    _assert_equivalent(arr, arr.copy(), min_cellpix=0, label="no_cells")


def test_single_cell():
    arr = np.zeros((10, 10), dtype=int)
    arr[2:5, 2:5] = 1
    _assert_equivalent(arr, arr.copy(), min_cellpix=0, label="single_cell")


def test_all_cells_below_threshold():
    arr = np.zeros((10, 10), dtype=int)
    arr[0, 0] = 1
    arr[5, 5] = 2
    _assert_equivalent(arr, arr.copy(), min_cellpix=100, label="all_below_threshold")


def test_label_gap_all_survive():
    """Labels {1, 3, 4} present, label 2 absent entirely."""
    arr = np.zeros((12, 12), dtype=int)
    arr[0:2, 0:2] = 1     # 4 px
    # label 2 deliberately never used
    arr[4:8, 4:8] = 3     # 16 px
    arr[9:10, 9:12] = 4   # 3 px
    _assert_equivalent(arr, arr.copy(), min_cellpix=0, label="label_gap_all_survive")


def test_label_gap_from_size_filter():
    """A mid-numbered label fails the size filter, creating a gap in the
    *surviving* labels used during renumbering."""
    arr = np.zeros((12, 12), dtype=int)
    arr[0:3, 0:3] = 1      # 9 px, survives
    arr[5, 5] = 2          # 1 px, filtered out
    arr[7:11, 7:11] = 3    # 16 px, survives
    _assert_equivalent(arr, arr.copy(), min_cellpix=2, label="label_gap_from_size_filter")


def test_tied_sizes():
    """Multiple cells with the exact same pixel count - exercises argsort's
    tie-breaking, which must match exactly since the rewrite reuses the same
    argsort call unchanged, only feeding it differently-computed counts."""
    arr = np.zeros((12, 12), dtype=int)
    arr[0:2, 0:2] = 1    # 4 px
    arr[4:6, 4:6] = 2    # 4 px
    arr[8:10, 8:10] = 3  # 4 px
    arr[0:2, 8:10] = 4   # 4 px
    _assert_equivalent(arr, arr.copy(), min_cellpix=0, label="tied_sizes")


def test_with_grid_area():
    arr = np.zeros((10, 10), dtype=int)
    arr[0:2, 0:2] = 1    # 4 px
    arr[5:9, 5:9] = 2    # 16 px
    grid_area = np.ones((10, 10)) * 100.0
    grid_area[0:5, :] = 25.0  # smaller cells in the top half
    _assert_equivalent(
        arr, arr.copy(), min_cellpix=500, grid_area=grid_area, label="grid_area_basic"
    )


def test_grid_area_with_gap():
    """grid_area combined with a label gap created by the size filter."""
    arr = np.zeros((10, 10), dtype=int)
    arr[0:2, 0:2] = 1     # top-left, small area -> filtered
    arr[3, 3] = 2         # single pixel, mid area -> filtered
    arr[5:9, 5:9] = 3     # bottom-right, large area -> survives
    grid_area = np.ones((10, 10)) * 10.0
    grid_area[5:9, 5:9] = 200.0
    _assert_equivalent(
        arr, arr.copy(), min_cellpix=50, grid_area=grid_area, label="grid_area_with_gap"
    )


def test_irregular_shapes():
    """Non-rectangular, disconnected-looking regions."""
    arr = np.zeros((30, 30), dtype=int)
    for i in range(29):
        arr[i, i] = 1
        arr[i, i + 1] = 1
    arr[10:15, 20] = 2
    arr[14, 20:25] = 2
    arr[25, 3] = 3
    arr[26, 25] = 4
    _assert_equivalent(arr, arr.copy(), min_cellpix=1, label="irregular_shapes")


def test_large_label_count():
    """Many small, individually-labeled single-pixel cells."""
    rng = np.random.default_rng(7)
    ny, nx = 80, 80
    arr = np.zeros((ny, nx), dtype=int)
    mask = rng.random((ny, nx)) < 0.08
    ys, xs = np.nonzero(mask)
    arr[ys, xs] = np.arange(1, len(ys) + 1)
    _assert_equivalent(arr, arr.copy(), min_cellpix=0, label="large_label_count")


def test_float_dtype_with_nan():
    """Float input with NaN, matching what the function's own use of nanmax
    implies it should tolerate.

    Not a like-for-like equivalence check: the frozen reference (old code)
    passes nlabelcells straight from np.nanmax() into np.full(nlabelcells,
    ...) with no int() cast, and current numpy (2.x) hard-errors on a
    float64 shape argument there (TypeError - same pre-existing latent bug
    documented for sort_renumber in test_sort_renumber_equivalence.py,
    unrelated to and predating this rewrite; real callers only ever pass
    int-dtype label arrays in practice - confirmed directly against the
    on-disk dtype of real demo cloudid_*.nc files, which is int64). The
    rewrite happens to fix this as a side effect via _labels_as_int's
    int(nlabelcells) cast."""
    arr = np.zeros((10, 10), dtype=float)
    arr[0:2, 0:2] = 1.0
    arr[5:8, 5:8] = 2.0
    arr[9, 9] = np.nan
    arr2 = arr.copy()

    with pytest.raises(TypeError):
        _sort_renumber2vars_reference(arr.copy(), arr2.copy(), min_cellpix=0)

    new_1, new_2, new_npix = sort_renumber2vars(arr.copy(), arr2.copy(), min_cellpix=0)
    assert new_1.shape == arr.shape
    assert np.array_equal(np.unique(new_1), [0, 1, 2])
    assert np.array_equal(new_1, new_2)
    assert np.array_equal(np.sort(new_npix)[::-1], [9, 4])


def test_negative_labels():
    """Negative values in either array must resolve to background (0),
    exactly as they did pre-rewrite: the old loop only ever tested
    `== ilabelcell` for ilabelcell in [1, nlabelcells], so a negative value
    never matched any positive label and was implicitly background. The
    lookup-table renumbering must clamp negatives explicitly, or an
    uncontrolled negative index would wrap around instead (see
    _labels_as_int's docstring in ftfunctions.py)."""
    arr = np.array([[1, -5, 2], [0, 1, 2]])
    arr2 = np.array([[1, -5, 2], [0, 1, -3]])
    _assert_equivalent(arr, arr2, min_cellpix=0, label="negative_labels")


# ---------------------------------------------------------------------------
# var2-specific cases - the two-array behavior this file exists to cover
# ---------------------------------------------------------------------------

def test_var2_identical_to_var1():
    arr = np.zeros((10, 10), dtype=int)
    arr[0:2, 0:2] = 1
    arr[5:9, 5:9] = 2
    _assert_equivalent(arr, arr.copy(), min_cellpix=0, label="var2_identical")


def test_var2_spatial_superset():
    """var2 shares var1's label values but covers more pixels per label -
    the realistic case (e.g. convcold_cloudnumber vs cloudnumber, where the
    core-only footprint is a subset of the grown footprint)."""
    var1 = np.zeros((10, 10), dtype=int)
    var1[2:4, 2:4] = 1
    var1[6:8, 6:8] = 2
    var2 = np.zeros((10, 10), dtype=int)
    var2[1:5, 1:5] = 1
    var2[5:9, 5:9] = 2
    _assert_equivalent(var1, var2, min_cellpix=0, label="var2_superset")


def test_var2_spatial_subset():
    var1 = np.zeros((10, 10), dtype=int)
    var1[1:6, 1:6] = 1
    var1[5:10, 5:10] = 2
    var2 = np.zeros((10, 10), dtype=int)
    var2[2:4, 2:4] = 1
    var2[7:9, 7:9] = 2
    _assert_equivalent(var1, var2, min_cellpix=0, label="var2_subset")


def test_var2_labels_absent_from_var1():
    """var2 has a label value that var1 never uses at all - those var2
    pixels must stay unlabeled (0) in the output, exactly as the original
    loop (which only ever looked up values from sortedcell_number1d, a
    subset of var1's values) never touched them."""
    var1 = np.zeros((10, 10), dtype=int)
    var1[0:2, 0:2] = 1
    var1[5:8, 5:8] = 2
    var2 = np.zeros((10, 10), dtype=int)
    var2[0:2, 0:2] = 1
    var2[8, 8] = 5  # value 5 never appears in var1
    _assert_equivalent(var1, var2, min_cellpix=0, label="var2_extra_label")


def test_var2_max_label_exceeds_var1_max():
    """var2's max label value is larger than var1's - the lookup table must
    be sized to cover it (see the lut_size comment in ftfunctions.py), or
    indexing lut[v2_int] would raise IndexError. The extra-valued pixels
    still resolve to 0 (background), matching the original loop, which
    never matched value 99 to any of var1's surviving labels either."""
    var1 = np.zeros((10, 10), dtype=int)
    var1[0:2, 0:2] = 1
    var1[5:8, 5:8] = 2
    var2 = np.zeros((10, 10), dtype=int)
    var2[0:2, 0:2] = 1
    var2[5:8, 5:8] = 2
    var2[9, 9] = 99  # far above var1's max label (2)
    _assert_equivalent(var1, var2, min_cellpix=0, label="var2_max_exceeds_var1")


def test_var2_all_zeros():
    var1 = np.zeros((10, 10), dtype=int)
    var1[0:2, 0:2] = 1
    var1[5:8, 5:8] = 2
    var2 = np.zeros((10, 10), dtype=int)
    _assert_equivalent(var1, var2, min_cellpix=0, label="var2_all_zeros")


def test_var2_spatially_disjoint():
    """var2 has the same label *values* as var1 but at entirely different
    pixel locations - by-value matching (not spatial overlap) must still
    apply identically."""
    var1 = np.zeros((10, 10), dtype=int)
    var1[0:2, 0:2] = 1
    var1[5:8, 5:8] = 2
    var2 = np.zeros((10, 10), dtype=int)
    var2[8:10, 0:2] = 1
    var2[0:2, 8:10] = 2
    _assert_equivalent(var1, var2, min_cellpix=0, label="var2_disjoint")


def test_var2_independent_gaps_with_grid_area():
    """var2 has its own gaps/extra labels, combined with the grid_area path
    (which only ever weights/filters var1)."""
    var1 = np.zeros((12, 12), dtype=int)
    var1[0:3, 0:3] = 1      # 9 px, survives
    var1[5, 5] = 2          # 1 px, filtered out by area
    var1[7:11, 7:11] = 3    # 16 px, survives
    var2 = np.zeros((12, 12), dtype=int)
    var2[0:3, 0:3] = 1
    var2[6, 6] = 7           # value absent from var1
    var2[7:11, 7:11] = 3
    grid_area = np.ones((12, 12)) * 10.0
    grid_area[7:11, 7:11] = 50.0
    _assert_equivalent(
        var1, var2, min_cellpix=50, grid_area=grid_area, label="var2_independent_gaps"
    )


def test_cellstep_guard_always_matches():
    """Explicit check of the reasoning in ftfunctions.py's comment: the
    original loop's `if nsortedcellindices == sortedcell_npix[icell]` guard
    is always true (nsortedcellindices and sortedcell_npix[icell] are both
    exactly np.count_nonzero(var1 == label) for the same label/array), so
    cellstep always equals icell + 1. Verified directly against the frozen
    reference's own internals across several of the cases above, rather
    than just assumed."""
    rng = np.random.default_rng(99)
    from scipy.ndimage import label as ndi_label, binary_dilation

    for trial in range(10):
        ny, nx = 40, 40
        seed_mask = np.zeros((ny, nx), dtype=bool)
        ys = rng.integers(0, ny, size=15)
        xs = rng.integers(0, nx, size=15)
        seed_mask[ys, xs] = True
        for _ in range(int(rng.integers(0, 3))):
            seed_mask = binary_dilation(seed_mask)
        var1, nlbl = ndi_label(seed_mask)
        if nlbl == 0:
            continue
        var2 = var1.copy()

        # Re-derive cellstep the way the frozen reference computes it, to
        # confirm cellstep == icell + 1 for every icell (i.e. the guard
        # never actually filters anything out).
        nlabelcells = int(np.nanmax(var1))
        labelcell_npix = np.full(nlabelcells, -999, dtype=int)
        for ilabelcell in range(1, nlabelcells + 1):
            n = np.count_nonzero(var1 == ilabelcell)
            if n > 0:
                labelcell_npix[ilabelcell - 1] = n
        ivalidcells = np.array(np.where(labelcell_npix > 0))[0, :]
        ncells = len(ivalidcells)
        if ncells == 0:
            continue
        labelcell_number1d = np.copy(ivalidcells) + 1
        labelcell_npix = labelcell_npix[ivalidcells]
        order = np.argsort(labelcell_npix)[::-1]
        sortedcell_npix = np.copy(labelcell_npix[order])
        sortedcell_number1d = np.copy(labelcell_number1d[order])
        cellstep = 0
        for icell in range(ncells):
            n = np.count_nonzero(var1 == sortedcell_number1d[icell])
            assert n == sortedcell_npix[icell], (
                f"trial {trial}: guard failed at icell={icell} "
                f"(n={n}, expected={sortedcell_npix[icell]})"
            )
            cellstep += 1
            assert cellstep == icell + 1

        _assert_equivalent(var1, var2, min_cellpix=0, label=f"cellstep_guard_trial{trial}")


# ---------------------------------------------------------------------------
# Randomized synthetic arrays
# ---------------------------------------------------------------------------

def test_randomized_arrays():
    """Many random (var1, var2) pairs (varying size, label count, size
    distribution, and independent gaps/extra labels between the two arrays),
    each checked for exact equivalence against the frozen reference."""
    from scipy.ndimage import label as ndi_label, binary_dilation, binary_erosion

    rng = np.random.default_rng(54321)
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

        var1, nlbl = ndi_label(seed_mask)
        if nlbl == 0:
            continue

        # Randomly blank out some whole labels in var1 to force gaps.
        if nlbl > 2:
            n_drop = int(rng.integers(0, nlbl // 2))
            if n_drop > 0:
                drop_labels = rng.choice(
                    np.arange(1, nlbl + 1), size=n_drop, replace=False
                )
                for dl in drop_labels:
                    var1[var1 == dl] = 0

        # Derive var2 from var1 with an independent perturbation: dilate,
        # erode, relabel from scratch, or inject an out-of-range extra
        # label - so var2 has its own gaps/extents rather than mirroring
        # var1 exactly.
        mode = rng.choice(["dilate", "erode", "relabel", "extra"])
        var1_mask = var1 > 0
        if mode == "dilate":
            var2_mask = binary_dilation(var1_mask)
            var2, _ = ndi_label(var2_mask)
        elif mode == "erode":
            var2_mask = binary_erosion(var1_mask)
            var2, _ = ndi_label(var2_mask)
        elif mode == "relabel":
            var2, _ = ndi_label(var1_mask)
            var2 = var2 * 3  # same footprint, different (still positive) values
        else:  # extra
            var2 = var1.copy()
            if ny > 2 and nx > 2:
                var2[0, 0] = int(rng.integers(50, 200))

        min_size = float(rng.choice([0, 1, 2, 5]))
        grid_area = None
        if rng.random() < 0.3:
            grid_area = rng.uniform(0.5, 5.0, size=(ny, nx))
            min_size = float(rng.choice([0.0, 1.0, 3.0]))

        _assert_equivalent(var1, var2, min_size, grid_area, label=f"random_case_{case}")
        n_run += 1

    assert n_run > 30, f"Expected most of {n_cases} random cases to run, only {n_run} did"


# ---------------------------------------------------------------------------
# Real captured data: idealized, South America IMERG, global MCSMIP
# ---------------------------------------------------------------------------
DATA_ROOT = os.environ.get("PYFLEXTRKR_TEST_DATA", "")


def _demo_path(*parts):
    return os.path.join(DATA_ROOT, *parts)


def _find_cloudid_files(tracking_dir, n=0):
    files = sorted(glob.glob(os.path.join(tracking_dir, "cloudid_*.nc")))
    if n > 0:
        files = files[:n]
    return files


def _iter_real_linkpf_inputs(tracking_dir, n_files):
    """Reconstruct the exact production inputs to sort_renumber2vars by
    reading real cloudid_*.nc output and running it back through
    link_pf_tb - literally what idclouds_tbpf.py:413-443 does immediately
    before calling sort_renumber2vars. Reads with mask_and_scale=False to
    recover the true on-disk int64 arrays (xarray's default decoding turns
    the _FillValue=0 int64 fields into float64+NaN, which is not what the
    real pipeline operates on in memory at this point)."""
    import xarray as xr
    from pyflextrkr.ftfunctions import link_pf_tb

    files = _find_cloudid_files(tracking_dir, n_files)
    for filepath in files:
        ds = xr.open_dataset(filepath, mask_and_scale=False, decode_timedelta=False)
        if not all(
            v in ds for v in
            ["convcold_cloudnumber_orig", "cloudnumber_orig", "pf_number", "tb"]
        ):
            ds.close()
            continue
        convcold_cn = np.squeeze(ds["convcold_cloudnumber_orig"].values)
        cn = np.squeeze(ds["cloudnumber_orig"].values)
        pf_number = np.squeeze(ds["pf_number"].values)
        tb = np.squeeze(ds["tb"].values)
        lat = ds["latitude"].values if "latitude" in ds else None
        lon = ds["longitude"].values if "longitude" in ds else None
        ds.close()

        if np.nanmax(pf_number) <= 0:
            continue

        pf_convcold_cloudnumber, pf_cloudnumber = link_pf_tb(
            convcold_cn, cn, pf_number, tb, tb_thresh=225.0,
        )
        yield filepath, pf_convcold_cloudnumber, pf_cloudnumber, lat, lon


def _run_real_data_case(tracking_dir, n_files, pixel_radius, area_thresh, label):
    if not DATA_ROOT:
        pytest.skip("PYFLEXTRKR_TEST_DATA not set")
    if not _find_cloudid_files(tracking_dir):
        pytest.skip(f"No cloudid_*.nc files found under {tracking_dir}")
    from pyflextrkr.ft_utilities import get_pixel_area

    n_checked = 0
    for filepath, v1, v2, lat, lon in _iter_real_linkpf_inputs(tracking_dir, n_files):
        # "fixed" branch: area_thresh converted to a pixel-count threshold.
        min_npix = np.ceil(area_thresh / (pixel_radius ** 2)).astype(int)
        _assert_equivalent(v1, v2, float(min_npix), label=f"{label}_fixed_{n_checked}")

        # "latlon" branch: area_thresh used directly, weighted by grid_area.
        if lat is not None and lon is not None:
            pixel_area = get_pixel_area(
                {"pixel_radius": pixel_radius, "area_method": "latlon"},
                latitude=lat, longitude=lon,
            )
            _assert_equivalent(
                v1, v2, float(area_thresh), grid_area=pixel_area,
                label=f"{label}_latlon_{n_checked}",
            )
        n_checked += 1
    assert n_checked > 0, f"No usable frames found under {tracking_dir}"


@pytest.mark.local
def test_real_idealized_linkpf():
    """Real linkpf inputs from demo_mcs_tbpf_idealized (2 cloud labels)."""
    _run_real_data_case(
        _demo_path("mcs_tbpf/idealized/test4/tracking"),
        n_files=8, pixel_radius=10.0, area_thresh=800.0,
        label="idealized",
    )


@pytest.mark.local
def test_real_south_america_linkpf():
    """Real linkpf inputs from demo_mcs_imerg (South America, up to 138
    labels)."""
    _run_real_data_case(
        _demo_path("mcs_tbpf/imerg/tracking"),
        n_files=12, pixel_radius=10.0, area_thresh=800.0,
        label="south_america",
    )


@pytest.mark.local
def test_real_global_mcsmip_linkpf():
    """Real linkpf inputs from demo_mcs_imerg_mcsmip (global, ~1200-1300
    labels) - the scale at which this bottleneck was actually found.

    n_files is deliberately small (unlike the idealized/South America
    cases): the frozen *reference* is the pre-fix O(n_labels x
    domain_size) code being proven equivalent here, and at this label
    count/domain size it is genuinely slow to run at all (~30s/frame/branch
    measured directly on this machine) - that slowness is exactly the bug
    this file exists to prove was fixed, not a test-quality shortcut."""
    _run_real_data_case(
        _demo_path("mcs_tbpf/imerg_global/tracking"),
        n_files=3, pixel_radius=10.0, area_thresh=800.0,
        label="global_mcsmip",
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
