"""
Verification tests for label_and_grow_features: BFS backward compatibility and EDT comparison.

This module tests the generalized `label_and_grow_features` function against the
original `label_and_grow_cold_clouds` to ensure:

1. **BFS backward compat** (TestLabelGrowBfsBackwardCompat):
   The new function with `growth_method='bfs'` produces bit-identical output
   to the original function across all input frames.

2. **EDT vs BFS** (TestLabelGrowEdtVsBfs):
   EDT growth produces the same number of features and >99% pixel-level
   agreement with BFS. Differences are only at contested boundaries.

3. **Operator 'gt'** (TestLabelGrowOperatorGt):
   Verifies `core_operator='gt'` works correctly with synthetic data where
   higher values define cores (e.g., radar reflectivity).

Prerequisites
-------------
Demo input files must be present under $PYFLEXTRKR_TEST_DATA.
Download with:
    python tests/run_demo_tests.py --demos demo_mcs_tbpf_idealized -n 4

Usage
-----
    # Download demo data (only needed once)
    python tests/run_demo_tests.py --demos demo_mcs_tbpf_idealized -n 4

    # Run verification tests
    export PYFLEXTRKR_TEST_DATA=~/data/demo
    pytest tests/test_label_grow_methods.py -m local -v -s

    # Run only the backward compatibility test
    pytest tests/test_label_grow_methods.py::TestLabelGrowBfsBackwardCompat -m local -v -s

    # Run only the EDT comparison test
    pytest tests/test_label_grow_methods.py::TestLabelGrowEdtVsBfs -m local -v -s
"""

import glob
import os
import sys

import numpy as np
import pytest

# CI runs bare `pytest tests/` (no `-m`), and tests/ has no __init__.py, so
# pytest's prepend import mode puts tests/ itself - not the repo root - on
# sys.path when collecting this file. Without this, `from reference....`
# below would only resolve by accident, if the active environment's editable
# install happens to also leak the repo root onto sys.path (older
# egg-link-style installs do; modern PEP 660 finder-based ones, used by CI
# and pyflex26.3, do not). Matches the same pattern already used by
# test_area_method_clouds.py, test_area_method_utils.py, and
# test_ftfunctions_link_pf.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyflextrkr.label_and_grow_features import label_and_grow_features

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = os.environ.get("PYFLEXTRKR_TEST_DATA", "")


def demo_path(*parts):
    return os.path.join(DATA_ROOT, *parts)


# Config parameters matching config_mcs_idealized.yml
_IDEALIZED_CONFIG = {
    "pbc_direction": "none",
}

_IDEALIZED_PARAMS = {
    "pixel_radius": 10.0,
    "thresholds": [225.0, 241.0, 261.0, 261.0],
    "area_thresh": 800.0,
    "min_core_npix": 4,
    "smooth_size": 5,
    "expand_to_tertiary": 0,
}


def _find_input_files(input_dir, pattern="*.nc", n=0):
    """Return sorted list of input files (n=0 means all)."""
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        files = sorted(
            glob.glob(os.path.join(input_dir, "**", pattern), recursive=True)
        )
    if n > 0:
        files = files[:n]
    return files


def _load_tb_from_file(filepath):
    """
    Load Tb data from a demo input file.

    Returns (tb_2d, config_dict) or None if file can't be loaded.
    """
    import xarray as xr
    from scipy.signal import medfilt2d

    ds = xr.open_dataset(filepath, decode_timedelta=False)

    # Get Tb variable
    if "Tb" in ds:
        tb = ds["Tb"].values
    elif "tb" in ds:
        tb = ds["tb"].values
    else:
        ds.close()
        return None

    ds.close()

    # Handle 3D (time, y, x) -> take first time
    if tb.ndim == 3:
        tb = tb[0, :, :]

    # Apply median filter to fill missing values (same as idclouds_tbpf)
    tb_filt = medfilt2d(tb, kernel_size=5)
    out_tb = np.copy(tb)
    missmask = np.isnan(tb)
    out_tb[missmask] = tb_filt[missmask]

    # Mask outside valid range
    out_tb[out_tb < 160] = np.nan
    out_tb[out_tb > 330] = np.nan

    return out_tb


def _contested_boundary_mask(labels):
    """
    Boolean mask for pixels that are 8-connected adjacent to a different label.

    These are the only pixels where EDT may assign a different label than BFS
    (equidistant Voronoi boundaries vs majority-voting tie-breaking).

    Parameters
    ----------
    labels : np.ndarray of int
        Labeled array (0 = background).

    Returns
    -------
    contested : np.ndarray of bool
    """
    from scipy.ndimage import maximum_filter, minimum_filter

    covered = labels > 0
    local_max = maximum_filter(labels, size=3, mode="constant", cval=0)
    sentinel = int(labels.max()) + 1
    labels_filled = np.where(covered, labels, sentinel)
    local_min_nz = minimum_filter(labels_filled, size=3, mode="constant", cval=sentinel)
    contested = covered & ((local_max > labels) | (local_min_nz < labels))
    return contested


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


# ---------------------------------------------------------------------------
# Test: BFS backward compatibility
# ---------------------------------------------------------------------------


@pytest.mark.local
class TestLabelGrowBfsBackwardCompat:
    """
    Verify that the live label_and_grow_cold_clouds wrapper (which calls the
    generalized label_and_grow_features with growth_method='bfs') produces
    bit-identical results to the original, pre-refactor algorithm.

    The ground truth here is tests/reference/label_and_grow_cold_clouds_reference.py,
    a frozen, independent copy of the pre-refactor implementation - not the
    live wrapper compared against itself, and not label_and_grow_features
    compared against the very function that calls it. Either of those would
    be tautological (cannot fail by construction); this comparison can.
    """

    INPUT_DIR = demo_path("mcs_tbpf", "idealized", "test4", "input")

    @pytest.fixture(scope="class")
    def input_files(self):
        if not DATA_ROOT:
            pytest.skip("PYFLEXTRKR_TEST_DATA not set")
        files = _find_input_files(self.INPUT_DIR)
        if not files:
            pytest.skip(f"No input files found in {self.INPUT_DIR}")
        return files

    def test_bfs_matches_original(self, input_files, capsys):
        """Live wrapper must produce bit-identical output to the frozen original across all frames."""
        from pyflextrkr.label_and_grow_cold_clouds import (
            label_and_grow_cold_clouds,
        )
        from tests.reference.label_and_grow_cold_clouds_reference import (
            label_and_grow_cold_clouds_reference,
        )

        n_tested = 0
        for filepath in input_files:
            tb = _load_tb_from_file(filepath)
            if tb is None:
                continue

            # Skip if too much missing data
            ny, nx = tb.shape
            if np.count_nonzero(np.isnan(tb)) / (ny * nx) >= 0.4:
                continue

            # Run the frozen pre-refactor reference implementation
            result_orig = label_and_grow_cold_clouds_reference(
                tb,
                _IDEALIZED_PARAMS["pixel_radius"],
                _IDEALIZED_PARAMS["thresholds"],
                _IDEALIZED_PARAMS["area_thresh"],
                _IDEALIZED_PARAMS["min_core_npix"],
                _IDEALIZED_PARAMS["smooth_size"],
                _IDEALIZED_PARAMS["expand_to_tertiary"],
                _IDEALIZED_CONFIG,
            )

            # Run the live backward-compatible wrapper
            result_new = label_and_grow_cold_clouds(
                tb,
                _IDEALIZED_PARAMS["pixel_radius"],
                _IDEALIZED_PARAMS["thresholds"],
                _IDEALIZED_PARAMS["area_thresh"],
                _IDEALIZED_PARAMS["min_core_npix"],
                _IDEALIZED_PARAMS["smooth_size"],
                _IDEALIZED_PARAMS["expand_to_tertiary"],
                _IDEALIZED_CONFIG,
            )

            # Both return the same legacy key set - compare all of them
            assert set(result_orig.keys()) == set(result_new.keys()), (
                f"Key set mismatch for {os.path.basename(filepath)}: "
                f"reference={sorted(result_orig.keys())}, "
                f"wrapper={sorted(result_new.keys())}"
            )
            for key in result_orig:
                orig_val = result_orig[key]
                new_val = result_new[key]
                if isinstance(orig_val, np.ndarray):
                    assert np.array_equal(orig_val, new_val), (
                        f"Mismatch in '{key}' for {os.path.basename(filepath)}: "
                        f"max diff = {np.max(np.abs(orig_val.astype(float) - new_val.astype(float)))}"
                    )
                else:
                    assert orig_val == new_val, (
                        f"Mismatch in '{key}' for {os.path.basename(filepath)}: "
                        f"orig={orig_val}, new={new_val}"
                    )

            n_tested += 1

        with capsys.disabled():
            print(f"\n  [BFS compat] Tested {n_tested} file(s) against frozen reference: all bit-identical")
        assert n_tested > 0, "No files were tested"


# ---------------------------------------------------------------------------
# Test: PBC+bfs deliberately diverges from the (buggy) frozen reference
# ---------------------------------------------------------------------------


def test_pbc_bfs_deliberately_diverges_from_reference():
    """
    Documents that bit-identical reproduction of the pre-refactor algorithm
    is NOT the goal when pbc_direction != 'none', because the pre-refactor
    algorithm itself (frozen in tests/reference/, and unchanged on
    public/main today) has two bugs in its PBC-crop path:

    1. Its returned final_nclouds is the stale pre-crop count
       (final_ncorecold), not the post-crop count it computes locally
       (final_nclouds = len(labels)) and then never uses in the return dict.
    2. Its returned 2D label arrays (final_cloudnumber,
       final_convcold_cloudnumber) are never renumbered after cropping, so
       they can hold an arbitrary sparse subset of the padded domain's
       label range (e.g. [1, 7, 9, 12, 14] instead of [1..5]) while the
       npix arrays are a dense 0-indexed array of length 5 - so a caller
       doing npix[label - 1] with the real (sparse) label value, the
       standard pattern used everywhere else in this codebase (see
       gettracks.py), gets a wrong count or an IndexError. Same crash
       class as issue #146, gated behind pbc_direction != 'none' plus a
       non-contiguous post-crop label set - reachable today by any config
       with pbc_direction set and growth_method left at its bfs default
       (e.g. config_mcs_pbc_idealized_demo.yml).

    label_and_grow_features's PBC-crop fix (LUT-densify both the label
    arrays and the npix arrays from the same post-crop `labels`) means the
    live bfs wrapper is *correct* where the frozen reference is buggy - so
    this test asserts the reference exhibits both bugs (so it flags loudly,
    for the right reason, if either is ever independently fixed upstream)
    and the live wrapper does not.
    """
    from pyflextrkr.label_and_grow_cold_clouds import label_and_grow_cold_clouds
    from reference.label_and_grow_cold_clouds_reference import (
        label_and_grow_cold_clouds_reference,
    )

    ny, nx = 40, 60
    ir = np.full((ny, nx), 280.0)
    # Core straddling the PBC-wrapped x edge (same physical feature under
    # periodic boundaries), plus several unrelated interior cores - pushes
    # the padded-domain label numbering high enough that the surviving
    # post-crop labels are a genuinely sparse subset.
    ir[10:16, 55:60] = 210.0
    ir[10:16, 0:3] = 210.0
    ir[5:9, 10:14] = 208.0
    ir[20:25, 20:26] = 207.0
    ir[30:35, 40:46] = 206.0
    ir[2:6, 45:50] = 209.0

    thresholds = [225.0, 241.0, 261.0, 261.0]
    config = {"pbc_direction": "x", "pixel_radius": 10.0, "area_thresh": 100.0}
    common_args = dict(
        pixel_radius=10.0, tb_threshs=thresholds, area_thresh=100.0,
        mincoldcorepix=4, smoothsize=3, warmanvilexpansion=0, config=config,
    )

    result_ref = label_and_grow_cold_clouds_reference(ir, **common_args)
    result_new = label_and_grow_cold_clouds(ir, **common_args)

    # --- Reference (frozen, pre-refactor) exhibits both known bugs ---
    ref_npix = result_ref["final_ncorecoldpix"]
    assert result_ref["final_nclouds"] != len(ref_npix), (
        "Frozen reference's final_nclouds now matches its npix array "
        "length - bug 1 appears fixed upstream; update this test's "
        "docstring/assertions (and consider un-skipping bit-identical "
        "comparison for PBC) rather than deleting this check."
    )
    ref_labels = np.unique(result_ref["final_cloudnumber"])
    ref_labels = ref_labels[ref_labels != 0]
    assert not np.array_equal(ref_labels, np.arange(1, len(ref_labels) + 1)), (
        "Frozen reference's final_cloudnumber is now contiguous - bug 2 "
        "appears fixed upstream; update this test rather than deleting it."
    )

    # --- Live wrapper (this branch) has neither bug ---
    new_nclouds = result_new["final_nclouds"]
    new_npix = result_new["final_ncorecoldpix"]
    assert new_nclouds == len(new_npix), (
        f"final_nclouds ({new_nclouds}) must match the npix array length "
        f"({len(new_npix)})"
    )
    new_labels = np.unique(result_new["final_cloudnumber"])
    new_labels = new_labels[new_labels != 0]
    assert np.array_equal(new_labels, np.arange(1, new_nclouds + 1)), (
        f"final_cloudnumber labels {new_labels} are not contiguous "
        f"1..{new_nclouds}"
    )
    for k in range(1, new_nclouds + 1):
        actual = np.count_nonzero(result_new["final_cloudnumber"] == k)
        assert new_npix[k - 1] == actual, (
            f"label {k}: final_ncorecoldpix reports {new_npix[k - 1]}, "
            f"actual pixel count is {actual}"
        )


# ---------------------------------------------------------------------------
# Test: EDT vs BFS comparison
# ---------------------------------------------------------------------------


@pytest.mark.local
class TestLabelGrowEdtVsBfs:
    """
    Verify that EDT growth produces nearly identical results to BFS.
    Expect same number of features and >99% pixel agreement.
    """

    INPUT_DIR = demo_path("mcs_tbpf", "idealized", "test4", "input")

    @pytest.fixture(scope="class")
    def input_files(self):
        if not DATA_ROOT:
            pytest.skip("PYFLEXTRKR_TEST_DATA not set")
        files = _find_input_files(self.INPUT_DIR)
        if not files:
            pytest.skip(f"No input files found in {self.INPUT_DIR}")
        return files

    def test_edt_vs_bfs_feature_count(self, input_files, capsys):
        """EDT must detect the same number of features as BFS."""
        n_tested = 0
        for filepath in input_files:
            tb = _load_tb_from_file(filepath)
            if tb is None:
                continue
            ny, nx = tb.shape
            if np.count_nonzero(np.isnan(tb)) / (ny * nx) >= 0.4:
                continue

            result_bfs = label_and_grow_features(
                tb,
                _IDEALIZED_PARAMS["pixel_radius"],
                _IDEALIZED_PARAMS["thresholds"],
                _IDEALIZED_PARAMS["area_thresh"],
                _IDEALIZED_PARAMS["min_core_npix"],
                _IDEALIZED_PARAMS["smooth_size"],
                _IDEALIZED_PARAMS["expand_to_tertiary"],
                _IDEALIZED_CONFIG,
                core_operator="lt",
                growth_method="bfs",
            )
            result_edt = label_and_grow_features(
                tb,
                _IDEALIZED_PARAMS["pixel_radius"],
                _IDEALIZED_PARAMS["thresholds"],
                _IDEALIZED_PARAMS["area_thresh"],
                _IDEALIZED_PARAMS["min_core_npix"],
                _IDEALIZED_PARAMS["smooth_size"],
                _IDEALIZED_PARAMS["expand_to_tertiary"],
                _IDEALIZED_CONFIG,
                core_operator="lt",
                growth_method="edt",
            )

            n_bfs = result_bfs["final_nFeature"]
            n_edt = result_edt["final_nFeature"]
            assert n_bfs == n_edt, (
                f"Feature count mismatch for {os.path.basename(filepath)}: "
                f"BFS={n_bfs}, EDT={n_edt}"
            )
            n_tested += 1

        with capsys.disabled():
            print(f"\n  [EDT vs BFS] Feature count: {n_tested} file(s) match")
        assert n_tested > 0

    def test_edt_vs_bfs_pixel_agreement(self, input_files, capsys):
        """EDT must have >99% pixel-level agreement with BFS."""
        min_agreement = 0.99
        n_tested = 0
        worst_agreement = 1.0

        for filepath in input_files:
            tb = _load_tb_from_file(filepath)
            if tb is None:
                continue
            ny, nx = tb.shape
            if np.count_nonzero(np.isnan(tb)) / (ny * nx) >= 0.4:
                continue

            result_bfs = label_and_grow_features(
                tb,
                _IDEALIZED_PARAMS["pixel_radius"],
                _IDEALIZED_PARAMS["thresholds"],
                _IDEALIZED_PARAMS["area_thresh"],
                _IDEALIZED_PARAMS["min_core_npix"],
                _IDEALIZED_PARAMS["smooth_size"],
                _IDEALIZED_PARAMS["expand_to_tertiary"],
                _IDEALIZED_CONFIG,
                core_operator="lt",
                growth_method="bfs",
            )
            result_edt = label_and_grow_features(
                tb,
                _IDEALIZED_PARAMS["pixel_radius"],
                _IDEALIZED_PARAMS["thresholds"],
                _IDEALIZED_PARAMS["area_thresh"],
                _IDEALIZED_PARAMS["min_core_npix"],
                _IDEALIZED_PARAMS["smooth_size"],
                _IDEALIZED_PARAMS["expand_to_tertiary"],
                _IDEALIZED_CONFIG,
                core_operator="lt",
                growth_method="edt",
            )

            bfs_labels = result_bfs["final_CoreSecondary_Number"]
            edt_labels = result_edt["final_CoreSecondary_Number"]

            if not np.any(bfs_labels > 0) and not np.any(edt_labels > 0):
                continue

            # Use feature-matched agreement: label-number swaps for similarly-
            # sized features (due to BFS vs EDT boundary differences) do not
            # count as real pixel differences.
            bfs_to_edt, edt_to_bfs = match_features_by_overlap(bfs_labels, edt_labels)
            diff_mask = feature_matched_diff_mask(
                bfs_labels, edt_labels, bfs_to_edt, edt_to_bfs
            )
            n_diff = np.count_nonzero(diff_mask)
            total = bfs_labels.size
            agreement = 1.0 - n_diff / total

            worst_agreement = min(worst_agreement, agreement)
            n_tested += 1

        with capsys.disabled():
            print(
                f"\n  [EDT vs BFS] Pixel agreement: {n_tested} file(s), "
                f"worst={worst_agreement:.6f}"
            )
        assert worst_agreement >= min_agreement, (
            f"Pixel agreement {worst_agreement:.4f} < {min_agreement}"
        )
        assert n_tested > 0

    def test_edt_diffs_at_boundaries_only(self, input_files, capsys):
        """Differences between EDT and BFS should only occur at feature boundaries."""
        n_tested = 0
        n_non_boundary_diffs = 0

        for filepath in input_files:
            tb = _load_tb_from_file(filepath)
            if tb is None:
                continue
            ny, nx = tb.shape
            if np.count_nonzero(np.isnan(tb)) / (ny * nx) >= 0.4:
                continue

            result_bfs = label_and_grow_features(
                tb,
                _IDEALIZED_PARAMS["pixel_radius"],
                _IDEALIZED_PARAMS["thresholds"],
                _IDEALIZED_PARAMS["area_thresh"],
                _IDEALIZED_PARAMS["min_core_npix"],
                _IDEALIZED_PARAMS["smooth_size"],
                _IDEALIZED_PARAMS["expand_to_tertiary"],
                _IDEALIZED_CONFIG,
                core_operator="lt",
                growth_method="bfs",
            )
            result_edt = label_and_grow_features(
                tb,
                _IDEALIZED_PARAMS["pixel_radius"],
                _IDEALIZED_PARAMS["thresholds"],
                _IDEALIZED_PARAMS["area_thresh"],
                _IDEALIZED_PARAMS["min_core_npix"],
                _IDEALIZED_PARAMS["smooth_size"],
                _IDEALIZED_PARAMS["expand_to_tertiary"],
                _IDEALIZED_CONFIG,
                core_operator="lt",
                growth_method="edt",
            )

            bfs_labels = result_bfs["final_CoreSecondary_Number"]
            edt_labels = result_edt["final_CoreSecondary_Number"]

            # Use feature-matched diff: label-number swaps do not count as
            # real spatial differences.
            bfs_to_edt, edt_to_bfs = match_features_by_overlap(bfs_labels, edt_labels)
            diff_mask = feature_matched_diff_mask(
                bfs_labels, edt_labels, bfs_to_edt, edt_to_bfs
            )

            if not np.any(diff_mask):
                n_tested += 1
                continue

            # Check if diffs are at boundaries (contested)
            boundary_bfs = _contested_boundary_mask(bfs_labels)
            boundary_edt = _contested_boundary_mask(edt_labels)
            boundary_combined = boundary_bfs | boundary_edt

            # Diffs that are NOT at boundaries
            non_boundary_diffs = diff_mask & ~boundary_combined
            n_non_boundary_diffs += np.count_nonzero(non_boundary_diffs)
            n_tested += 1

        with capsys.disabled():
            print(
                f"\n  [EDT vs BFS] Boundary check: {n_tested} file(s), "
                f"non-boundary diffs = {n_non_boundary_diffs}"
            )
        # Allow a small tolerance for non-boundary diffs (EDT may assign
        # differently in some edge cases due to discrete distance ties)
        assert n_non_boundary_diffs == 0 or n_tested == 0, (
            f"Found {n_non_boundary_diffs} non-boundary differences"
        )


# ---------------------------------------------------------------------------
# Test: core_operator='gt' with synthetic data
# ---------------------------------------------------------------------------


def _assert_labels_contiguous_and_npix_correct(
    result, mask_key="final_Feature_Number", npix_key="final_CoreSecondary_npix",
):
    """
    Shared invariant check for label_and_grow_features output, added as a
    regression guard for GitHub issue #146 and the related nclouds/npix and
    PBC-crop defects found in the same audit:

    - final_nFeature must equal the length of the returned npix array.
    - The labels actually present in the 2D mask must be exactly
      {1, ..., final_nFeature} - no gaps, nothing left sparse after a
      periodic-boundary crop.
    - Every npix value must equal the actual pixel count for that label -
      this is what the original cloud_sizes[index]-vs-cloud_sizes[i]
      confusion got wrong.
    """
    nfeature = result["final_nFeature"]
    npix = result[npix_key]
    mask = result[mask_key]

    assert len(npix) == nfeature, (
        f"{npix_key} length {len(npix)} != final_nFeature {nfeature}"
    )

    labels_present = np.unique(mask)
    labels_present = labels_present[labels_present != 0]
    assert np.array_equal(labels_present, np.arange(1, nfeature + 1)), (
        f"Labels in {mask_key} are not contiguous 1..{nfeature}: "
        f"{labels_present}"
    )

    for k in range(1, nfeature + 1):
        actual = np.count_nonzero(mask == k)
        assert npix[k - 1] == actual, (
            f"label {k}: {npix_key} reports {npix[k - 1]}, "
            f"actual pixel count is {actual}"
        )


class TestLabelGrowOperatorGt:
    """
    Verify that core_operator='gt' correctly labels features where higher
    values define cores (e.g., radar reflectivity).
    Uses synthetic data — no demo data download required.
    """

    def test_gt_operator_basic(self):
        """Higher values should form cores with core_operator='gt'."""
        np.random.seed(42)
        ny, nx = 100, 100

        # Create synthetic reflectivity field: background noise + 2 cores
        field = np.random.uniform(0, 5, (ny, nx)).astype(np.float32)
        # Core 1: centered at (30, 30), values 35-45 dBZ
        field[25:35, 25:35] = 40.0
        # Core 2: centered at (70, 70), values 35-45 dBZ
        field[65:75, 65:75] = 35.0
        # Secondary region around cores: 15-25 dBZ
        field[20:40, 20:40] = np.maximum(field[20:40, 20:40], 15.0)
        field[60:80, 60:80] = np.maximum(field[60:80, 60:80], 15.0)
        # Re-apply cores on top
        field[25:35, 25:35] = 40.0
        field[65:75, 65:75] = 35.0

        config = {"pbc_direction": "none"}
        # Thresholds: core > 30, secondary > 10, tertiary > 5, edge > 5
        thresholds = [30.0, 10.0, 5.0, 5.0]

        result = label_and_grow_features(
            field,
            pixel_radius=1.0,
            thresholds=thresholds,
            area_thresh=5.0,  # 5 km^2 (5 pixels at 1km)
            min_core_npix=4,
            smooth_size=3,
            expand_to_tertiary=0,
            config=config,
            core_operator="gt",
            growth_method="bfs",
        )

        # Should detect at least 2 features
        assert result["final_nFeature"] >= 2, (
            f"Expected >= 2 features, got {result['final_nFeature']}"
        )

        # Core pixels should be present
        assert np.sum(result["final_Core_npix"]) > 0, "No core pixels detected"

        # Labels should be non-zero where field is strong
        cloud_number = result["final_Feature_Number"]
        # The core regions should be labeled
        assert np.all(cloud_number[25:35, 25:35] > 0), "Core 1 not labeled"
        assert np.all(cloud_number[65:75, 65:75] > 0), "Core 2 not labeled"

        _assert_labels_contiguous_and_npix_correct(result)

    def test_gt_operator_edt(self):
        """EDT method should also work with core_operator='gt'."""
        np.random.seed(42)
        ny, nx = 100, 100

        field = np.random.uniform(0, 5, (ny, nx)).astype(np.float32)
        field[25:35, 25:35] = 40.0
        field[65:75, 65:75] = 35.0
        field[20:40, 20:40] = np.maximum(field[20:40, 20:40], 15.0)
        field[60:80, 60:80] = np.maximum(field[60:80, 60:80], 15.0)
        field[25:35, 25:35] = 40.0
        field[65:75, 65:75] = 35.0

        config = {"pbc_direction": "none"}
        thresholds = [30.0, 10.0, 5.0, 5.0]

        result_bfs = label_and_grow_features(
            field, 1.0, thresholds, 5.0, 4, 3, 0, config,
            core_operator="gt", growth_method="bfs",
        )
        result_edt = label_and_grow_features(
            field, 1.0, thresholds, 5.0, 4, 3, 0, config,
            core_operator="gt", growth_method="edt",
        )

        # Same number of features
        assert result_bfs["final_nFeature"] == result_edt["final_nFeature"], (
            f"Feature count: BFS={result_bfs['final_nFeature']}, "
            f"EDT={result_edt['final_nFeature']}"
        )

        # High pixel agreement
        bfs_labels = result_bfs["final_Feature_Number"]
        edt_labels = result_edt["final_Feature_Number"]
        agreement = np.count_nonzero(bfs_labels == edt_labels) / bfs_labels.size
        assert agreement > 0.95, f"Agreement only {agreement:.3f}"

        _assert_labels_contiguous_and_npix_correct(result_bfs)
        _assert_labels_contiguous_and_npix_correct(result_edt)

    def test_lt_vs_inverted_gt(self):
        """
        Inverting a Tb field and using 'gt' should produce equivalent labeling
        to using 'lt' on the original field.
        """
        np.random.seed(123)
        ny, nx = 80, 80

        # Create Tb-like field: warm background with cold features
        tb = np.full((ny, nx), 280.0, dtype=np.float32)
        # Cold core 1
        tb[20:30, 20:30] = 210.0
        # Cold anvil around it
        tb[15:35, 15:35] = np.minimum(tb[15:35, 15:35], 235.0)
        tb[20:30, 20:30] = 210.0

        config = {"pbc_direction": "none"}
        thresholds_lt = [225.0, 241.0, 261.0, 261.0]

        result_lt = label_and_grow_features(
            tb, 10.0, thresholds_lt, 100.0, 4, 3, 0, config,
            core_operator="lt", growth_method="bfs",
        )

        # Invert field: inverted = max_tb - tb
        max_tb = 330.0
        inverted = max_tb - tb
        # Inverted thresholds: core > (330-225)=105, secondary > (330-241)=89
        thresholds_gt = [
            max_tb - thresholds_lt[0],  # 105
            max_tb - thresholds_lt[1],  # 89
            max_tb - thresholds_lt[2],  # 69
            max_tb - thresholds_lt[3],  # 69
        ]

        result_gt = label_and_grow_features(
            inverted, 10.0, thresholds_gt, 100.0, 4, 3, 0, config,
            core_operator="gt", growth_method="bfs",
        )

        # Same number of features
        assert result_lt["final_nFeature"] == result_gt["final_nFeature"], (
            f"lt={result_lt['final_nFeature']}, gt={result_gt['final_nFeature']}"
        )

        # Same labeled pixels (labels might be in different order if sizes differ)
        lt_labeled = result_lt["final_Feature_Number"] > 0
        gt_labeled = result_gt["final_Feature_Number"] > 0
        assert np.array_equal(lt_labeled, gt_labeled), (
            "Labeled pixel masks differ between lt and inverted gt"
        )

        _assert_labels_contiguous_and_npix_correct(result_lt)
        _assert_labels_contiguous_and_npix_correct(result_gt)

    def test_all_cold_domain_no_background(self):
        """
        Regression test for GitHub issue #146.

        Reported traceback: label_and_grow_cold_clouds.py:170 (pre-refactor)
        / label_and_grow_features.py:225 (this module),
        IndexError: index N is out of bounds for axis 0 with size N.

        The issue's stated root cause ("two cold cores merge during growth")
        does not hold: grow_cells()/EDT growth cannot make a core label
        vanish by merging - they never overwrite an existing positive
        label. The actual trigger is a domain/tile with no pixel warmer
        than the secondary threshold, so background label 0 is absent from
        the grown label array - before the fix, cloud_sizes[index] assumed
        cloud_indices is exactly [0, 1, ..., N] (position == label value),
        which breaks as soon as 0 is missing. Tested for both growth
        methods, since both funnel through the same counting loop.
        """
        ny, nx = 30, 30
        tb = np.full((ny, nx), 230.0, dtype=np.float32)  # colder than secondary=241 everywhere
        tb[5:12, 5:12] = 210.0  # core 1
        tb[18:26, 18:26] = 208.0  # core 2

        thresholds = [225.0, 241.0, 261.0, 261.0]
        config = {"pbc_direction": "none"}

        for growth_method in ("bfs", "edt"):
            result = label_and_grow_features(
                tb, 10.0, thresholds, 100.0,
                min_core_npix=4, smooth_size=3, expand_to_tertiary=0,
                config=config, core_operator="lt", growth_method=growth_method,
            )
            assert result["final_nFeature"] == 2, (
                f"[{growth_method}] expected 2 features, got {result['final_nFeature']}"
            )
            _assert_labels_contiguous_and_npix_correct(result)

    def test_pbc_crop_produces_contiguous_labels(self):
        """
        Regression test for the PBC-crop label-contiguity defect found in
        the same audit as issue #146.

        Before the fix, label_and_grow_features's periodic-boundary crop
        path recomputed final_nFeature and the npix arrays correctly, but
        never renumbered the 2D label arrays themselves - so surviving
        labels could be an arbitrary sparse subset of the padded domain's
        label range (e.g. [8, 11] instead of [1, 2]), breaking every
        downstream `label - 1` positional read (gettracks.py,
        netcdf_io.py, tracksingle_drift.py).
        """
        np.random.seed(7)
        ny, nx = 40, 40
        tb = np.full((ny, nx), 280.0, dtype=np.float32)
        # Scatter several small cold cores, including near every edge, so
        # periodic-boundary padding/cropping produces a genuinely sparse
        # surviving label set.
        cores = [
            (2, 2), (2, 36), (36, 2), (36, 36), (18, 18),
            (5, 20), (20, 5), (30, 10), (10, 30),
        ]
        for (y, x) in cores:
            y0, y1 = max(0, y - 2), min(ny, y + 2)
            x0, x1 = max(0, x - 2), min(nx, x + 2)
            tb[y0:y1, x0:x1] = 210.0

        thresholds = [225.0, 241.0, 261.0, 261.0]
        config = {
            "pbc_direction": "both",
            "pbc_extended_fraction": 1.0,
            "pixel_radius": 10.0,
            "area_thresh": 100.0,
        }

        for growth_method in ("bfs", "edt"):
            result = label_and_grow_features(
                tb, 10.0, thresholds, 100.0,
                min_core_npix=1, smooth_size=1, expand_to_tertiary=0,
                config=config, core_operator="lt", growth_method=growth_method,
            )
            assert result["final_nFeature"] > 0, (
                f"[{growth_method}] expected at least 1 feature"
            )
            _assert_labels_contiguous_and_npix_correct(result)
            # final_CoreSecondary_Number must be renumbered the same way as
            # final_Feature_Number, not just final_Feature_Number.
            _assert_labels_contiguous_and_npix_correct(
                result, mask_key="final_CoreSecondary_Number",
            )

    def test_no_core_subthreshold_secondary_does_not_crash(self):
        """
        Regression test for an UnboundLocalError found while stress-testing
        with tests/plot_label_grow_synthetic.py's varied synthetic frames.

        Trigger: no pixel anywhere crosses the core threshold (nlabelcores
        == 0, so label_and_grow_features falls into its "no core" fallback
        branch, which instead labels connected secondary-threshold regions
        directly), and at least one such region exists but is smaller than
        area_thresh. The fallback branch's inner `if nFeature > 0:` (after
        re-purposing nFeature to mean "count that passed the area filter")
        only assigns sortedcore_npix/sortedSecondary_npix/sortedTertiary_npix
        inside that if - with no else, so when every candidate region is
        rejected by the area filter (nFeature reset to 0), the following
        `final_Secondary_npix = np.copy(sortedSecondary_npix)` raises
        UnboundLocalError: the top-of-function defaults use different
        casing (sortedsecondary_npix/sortedtertiary_npix) and are never
        consulted, so nothing else binds these names in that path.
        """
        ny, nx = 40, 40
        tb = np.full((ny, nx), 290.0)
        # Crosses secondary (241 K) but not core (225 K); 4 px < the
        # 8-pixel area threshold (area_thresh=800 / pixel_radius^2=100).
        tb[10:12, 10:12] = 235.0

        thresholds = [225.0, 241.0, 261.0, 261.0]
        config = {"pbc_direction": "none"}

        for growth_method in ("bfs", "edt"):
            result = label_and_grow_features(
                tb, 10.0, thresholds, 800.0,
                min_core_npix=4, smooth_size=5, expand_to_tertiary=0,
                config=config, core_operator="lt", growth_method=growth_method,
            )
            assert result["final_nFeature"] == 0, (
                f"[{growth_method}] expected 0 features (sub-threshold "
                f"secondary region only), got {result['final_nFeature']}"
            )
