#!/usr/bin/env python
"""
Differential test: prove the rewritten cache_label_positions/adjust_axis
(pyflextrkr/ftfunctions.py) produce IDENTICAL output to the original,
pre-rewrite implementation.

Context
-------
cache_label_positions built a dict mapping every non-zero label in a
labeled array to np.where(segments == label), computed eagerly for every
label up front - O(n_labels x domain_size). Its only caller, adjust_axis
(the PBC-crop logic in label_and_grow_features's `pbc_direction != 'none'`
path), only ever looks up a small subset of that label set: the labels
straddling the seam between the padded/extended region and the original
domain, plus whichever labels happen to appear in the single row/column
being examined at each step of its boundary-refinement search. Profiled
directly (after sort_renumber's identical pattern was already fixed):
cache_label_positions was 77% of an 81.6s single global frame, called 3
times (once each for final_feature_number, feature_type_map,
final_CoreSecondary_Number).

The rewrite makes the cache lazy (pyflextrkr.ftfunctions._LazyLabelPositions,
a dict subclass computing np.where(...) on first access via __missing__,
memoizing after), so only labels actually looked up ever get computed -
same values, none of the wasted work. adjust_axis's own code is completely
unchanged; only what cache_label_positions returns changes.

This file proves equivalence two ways:
1. Value-level: the new lazy cache returns identical np.where(...) tuples
   to a frozen eager reference, for every label that's ever looked up.
2. Behavior-level: a frozen copy of adjust_axis wired to the frozen eager
   cache builder is compared against the live (now-lazy) adjust_axis, on
   synthetic PBC-seam-crossing scenarios and on real global-demo data -
   asserting exact equality of both returned values (segments, adjusted).

Usage
-----
  python -m pytest tests/test_cache_label_positions_equivalence.py -v
  # Real-data test additionally needs:
  export PYFLEXTRKR_TEST_DATA=~/data/demo
  python tests/run_demo_tests.py --demos demo_mcs_imerg_mcsmip -n 4
"""
import os
import sys
import glob
import logging
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyflextrkr.ftfunctions import (
    cache_label_positions,
    adjust_axis,
    calc_extension,
    pad_and_extend,
)
from pyflextrkr.ft_utilities import get_pixel_area, get_mean_pixel_length


# ---------------------------------------------------------------------------
# Frozen references: verbatim copies of the pre-rewrite
# cache_label_positions and adjust_axis (adjust_axis's own logic is
# unchanged by the rewrite - it's copied here only so it can be wired to
# the frozen eager cache builder instead of the live lazy one, giving a
# true old-vs-new behavioral diff rather than just a value-level one).
# ---------------------------------------------------------------------------
def _cache_label_positions_reference(segments):
    """Frozen pre-rewrite cache_label_positions (see module docstring)."""
    label_positions_cache = {}
    unique_labels = np.unique(segments)
    for label in unique_labels:
        if label != 0:
            label_positions_cache[label] = np.where(segments == label)
    return label_positions_cache


def _adjust_axis_reference(segments, axis, original_shape, ext_frac, config):
    """Frozen pre-rewrite adjust_axis, wired to the frozen eager cache
    builder above instead of the live (lazy) cache_label_positions."""
    logger = logging.getLogger(__name__)
    pixel_area = get_pixel_area(config)
    pixel_length = get_mean_pixel_length(pixel_area)
    area_thresh = config.get('area_thresh')
    size_factor = 3
    width_thresh = size_factor * int(2 * np.sqrt(area_thresh / np.pi) / pixel_length)
    ext_size = calc_extension(original_shape[axis], ext_frac)
    adjusted = False
    label_positions_cache = _cache_label_positions_reference(segments)

    if axis == 1:
        left_slice = segments[:, :ext_size]
        middle_slice = segments[:, ext_size:ext_size + original_shape[1]]
        shared_labels = np.intersect1d(left_slice[:, -1], middle_slice[:, 0])
    elif axis == 0:
        top_slice = segments[:ext_size, :]
        middle_slice = segments[ext_size:ext_size + original_shape[0], :]
        shared_labels = np.intersect1d(top_slice[-1, :], middle_slice[0, :])
    shared_labels = shared_labels[shared_labels != 0]

    if shared_labels.size > 0 and not np.all(shared_labels == 0):
        for label in shared_labels:
            if np.all(middle_slice == label):
                logger.warning(f"Full-domain spanning feature detected in axis {axis} with label {label}.")
                continue
            adjusted = True
            min_pos = np.min(label_positions_cache[label][axis])

            while True:
                current_labels = segments[min_pos, :] if axis == 0 else segments[:, min_pos]
                non_zero_labels = current_labels[current_labels != 0]
                unique_labels, unique_npix = np.unique(non_zero_labels, return_counts=True)
                max_unique_npix = np.max(unique_npix)
                if (unique_labels.size > 1) and (max_unique_npix > width_thresh):
                    min_positions = [np.min(label_positions_cache[ul][axis]) for ul in unique_labels]
                    new_min_pos = min(min_positions)
                    if new_min_pos == min_pos:
                        break
                    min_pos = new_min_pos
                else:
                    break
        if axis == 1:
            dx = ext_size - min_pos
            segments = segments[:, min_pos:ext_size + original_shape[1] - dx]
            segments = np.roll(segments, shift=-dx, axis=1)
        elif axis == 0:
            dy = ext_size - min_pos
            segments = segments[min_pos:ext_size + original_shape[0] - dy, :]
            segments = np.roll(segments, shift=-dy, axis=0)
    else:
        logger.debug(f"No shared labels found in axis {axis}.")
    return segments, adjusted


# ---------------------------------------------------------------------------
# 1. Value-level: cache_label_positions itself
# ---------------------------------------------------------------------------

def _assert_cache_equivalent(segments, label=""):
    """Every non-zero label present in `segments` must map to the exact
    same np.where(...) tuple in both the new (lazy) and reference (eager)
    caches."""
    new_cache = cache_label_positions(segments)
    ref_cache = _cache_label_positions_reference(segments)

    present_labels = np.unique(segments)
    present_labels = present_labels[present_labels != 0]

    assert set(int(l) for l in present_labels) == set(int(k) for k in ref_cache.keys()), (
        f"[{label}] reference cache's keys don't match the array's own "
        f"non-zero labels - test setup issue, not a rewrite issue"
    )

    for lbl in present_labels:
        new_pos = new_cache[lbl]
        ref_pos = ref_cache[lbl]
        assert len(new_pos) == len(ref_pos) == segments.ndim
        for axis in range(segments.ndim):
            assert np.array_equal(new_pos[axis], ref_pos[axis]), (
                f"[{label}] label {lbl}, axis {axis}: positions differ"
            )

    # Label 0 (background) was never populated by the original eager
    # builder either - confirm the lazy version preserves that (KeyError),
    # not a silently-computed background mask.
    with pytest.raises(KeyError):
        _ = new_cache[0]


def test_cache_simple_labels():
    arr = np.zeros((10, 10), dtype=int)
    arr[0:2, 0:2] = 1
    arr[5:9, 5:9] = 2
    _assert_cache_equivalent(arr, label="simple")


def test_cache_with_gaps():
    arr = np.zeros((10, 10), dtype=int)
    arr[0:2, 0:2] = 1
    # label 2 absent
    arr[5:9, 5:9] = 3
    _assert_cache_equivalent(arr, label="with_gaps")


def test_cache_lazy_only_computes_what_is_looked_up():
    """The whole point of the rewrite: labels never accessed are never
    computed at all (not just cheaply - not at all)."""
    arr = np.zeros((10, 10), dtype=int)
    arr[0:2, 0:2] = 1
    arr[5:9, 5:9] = 2
    arr[3, 7] = 3

    cache = cache_label_positions(arr)
    assert len(cache) == 0, "Nothing looked up yet - cache should start empty"
    _ = cache[1]
    assert set(cache.keys()) == {1}, "Only the looked-up label should be cached"
    _ = cache[2]
    assert set(cache.keys()) == {1, 2}
    # label 3 never looked up - stays uncomputed
    assert 3 not in cache


def test_cache_no_labels():
    arr = np.zeros((10, 10), dtype=int)
    _assert_cache_equivalent(arr, label="no_labels")


# ---------------------------------------------------------------------------
# 2. Behavior-level: adjust_axis, synthetic PBC-seam scenarios
# ---------------------------------------------------------------------------

def _make_padded_labels(ny, nx, ext_frac, blobs_orig, pbc_direction="x", area_thresh=20.0):
    """
    Build a small labeled array the way label_and_grow_features actually
    produces one going into adjust_axis: start from an *unpadded* field of
    shape (ny, nx), place blobs (list of (y0,y1,x0,x1) slices, one integer
    label each starting at 1) in the *original* coordinate frame, pad via
    the real pad_and_extend (mode='wrap', matching production), and return
    the padded label array plus the config/original_shape adjust_axis needs.

    config always carries pixel_radius/area_thresh alongside pbc_direction,
    matching how production always shapes config (the entire loaded YAML) -
    adjust_axis's own get_pixel_area() call requires config['pixel_radius'].
    """
    field = np.zeros((ny, nx), dtype=int)
    for i, (y0, y1, x0, x1) in enumerate(blobs_orig, start=1):
        field[y0:y1, x0:x1] = i

    config = {
        "pbc_direction": pbc_direction, "pbc_extended_fraction": ext_frac,
        "pixel_radius": 10.0, "area_thresh": area_thresh,
    }
    padded, padded_x, padded_y = pad_and_extend(field, config)
    return padded, (ny, nx), config


def _assert_adjust_axis_equivalent(segments, axis, original_shape, ext_frac, config, label=""):
    new_segments, new_adjusted = adjust_axis(
        segments.copy(), axis, original_shape, ext_frac, config
    )
    ref_segments, ref_adjusted = _adjust_axis_reference(
        segments.copy(), axis, original_shape, ext_frac, config
    )
    assert new_adjusted == ref_adjusted, f"[{label}] 'adjusted' flag differs"
    assert np.array_equal(new_segments, ref_segments), (
        f"[{label}] output segments differ:\n"
        f"  new shape={new_segments.shape} ref shape={ref_segments.shape}"
    )


def test_adjust_axis_no_seam_crossing():
    """No feature touches the wrap seam - shared_labels empty, trivial
    pass-through for both implementations."""
    padded, orig_shape, config = _make_padded_labels(
        30, 40, ext_frac=0.5, blobs_orig=[(10, 15, 15, 20)],
    )
    _assert_adjust_axis_equivalent(
        padded, 1, orig_shape, 0.5, config, label="no_seam_crossing"
    )


def test_adjust_axis_single_seam_crossing_feature():
    """One feature straddling the x-wrap seam (right edge <-> left edge),
    plus unrelated interior features - the core scenario adjust_axis exists
    to handle (same construction pattern as
    test_pbc_bfs_deliberately_diverges_from_reference in
    test_label_grow_methods.py)."""
    ny, nx = 40, 60
    padded, orig_shape, config = _make_padded_labels(
        ny, nx, ext_frac=0.5,
        blobs_orig=[
            (10, 16, 55, 60),  # right edge - part of the seam-crossing feature
            (10, 16, 0, 3),    # left edge - other part of the same feature
        ],
    )
    # These two blobs are separate *labels* in the unpadded field (1 and 2)
    # but pad_and_extend's wrap makes them touch in the padded array -
    # relabel the padded array with scipy so they merge into one connected
    # component the way find_and_label_cores would actually produce, rather
    # than testing an artificial pre-labeled seam (which adjust_axis's own
    # `shared_labels = intersect1d(...)` logic assumes: a label is shared
    # only if the *same* label value appears on both sides of the seam).
    from scipy.ndimage import label as ndi_label
    relabeled, _ = ndi_label(padded > 0, structure=np.ones((3, 3), dtype=bool))
    _assert_adjust_axis_equivalent(
        relabeled, 1, orig_shape, 0.5, config, label="single_seam_crossing"
    )


def test_adjust_axis_multiple_seam_crossing_features():
    ny, nx = 50, 70
    padded, orig_shape, config = _make_padded_labels(
        ny, nx, ext_frac=0.5,
        blobs_orig=[
            (5, 10, 65, 70), (5, 10, 0, 3),     # seam-crosser 1
            (30, 36, 66, 70), (30, 36, 0, 4),   # seam-crosser 2
            (15, 20, 20, 26),                    # unrelated interior blob
        ],
    )
    from scipy.ndimage import label as ndi_label
    relabeled, _ = ndi_label(padded > 0, structure=np.ones((3, 3), dtype=bool))
    _assert_adjust_axis_equivalent(
        relabeled, 1, orig_shape, 0.5, config, label="multiple_seam_crossing"
    )


def test_adjust_axis_y_axis():
    """Same scenario, but adjusting along axis=0 (y) instead of axis=1 (x)."""
    ny, nx = 60, 40
    padded, orig_shape, config = _make_padded_labels(
        ny, nx, ext_frac=0.5,
        blobs_orig=[(55, 60, 10, 16), (0, 3, 10, 16)],
        pbc_direction="y",
    )
    from scipy.ndimage import label as ndi_label
    relabeled, _ = ndi_label(padded > 0, structure=np.ones((3, 3), dtype=bool))
    _assert_adjust_axis_equivalent(
        relabeled, 0, orig_shape, 0.5, config, label="y_axis"
    )


def test_adjust_axis_refinement_loop_iterates():
    """A wide seam-crossing feature plus several unrelated wide blobs
    packed close enough along the search axis that the while-loop's
    "multiple labels at this position, largest > width_thresh" branch
    actually fires more than once - exercises the *second* usage site of
    label_positions_cache (the `[np.min(label_positions_cache[ul][axis])
    for ul in unique_labels]` comprehension inside the loop), not just the
    first lookup before it."""
    ny, nx = 40, 80
    # Small area_thresh -> small width_thresh, so modest blobs are enough
    # to trigger "keep searching".
    config_area_thresh = 20.0
    field = np.zeros((ny, nx), dtype=int)
    # Seam-crossing feature (wide, low y)
    field[5:10, 76:80] = 1
    field[5:10, 0:2] = 1
    # A chain of wide blobs immediately inside the domain, forcing the
    # refinement search to keep walking left before settling on a crop point.
    field[5:10, 2:10] = 2
    field[5:10, 10:18] = 3
    field[12:18, 20:30] = 4

    config = {"pbc_direction": "x", "pbc_extended_fraction": 0.5,
              "pixel_radius": 10.0, "area_thresh": config_area_thresh}
    padded, padded_x, padded_y = pad_and_extend(field, config)
    from scipy.ndimage import label as ndi_label
    relabeled, _ = ndi_label(padded > 0, structure=np.ones((3, 3), dtype=bool))
    _assert_adjust_axis_equivalent(
        relabeled, 1, (ny, nx), 0.5, config, label="refinement_loop_iterates"
    )


def test_adjust_axis_full_domain_spanning_label():
    """A feature that spans the entire middle slice along the adjustment
    axis - exercises the `np.all(middle_slice == label)` / logger.warning
    branch, which `continue`s without ever touching label_positions_cache
    for that label.

    Note: when *every* label in shared_labels hits this branch, `min_pos`
    is never assigned inside the loop, yet is read unconditionally right
    after it (`dx = ext_size - min_pos`) - an UnboundLocalError. This is a
    genuine, pre-existing bug in adjust_axis's own logic (unchanged by the
    cache_label_positions rewrite - verified below that both the live and
    frozen-reference implementations raise identically), not something
    introduced here. Flagged to the user rather than fixed in this pass,
    which is scoped to cache_label_positions/adjust_axis's *caching*
    behavior, not this unrelated control-flow bug."""
    ny, nx = 20, 30
    field = np.ones((ny, nx), dtype=int)  # the entire domain is label 1
    config = {"pbc_direction": "x", "pbc_extended_fraction": 0.5,
              "pixel_radius": 10.0, "area_thresh": 20.0}
    padded, padded_x, padded_y = pad_and_extend(field, config)
    from scipy.ndimage import label as ndi_label
    relabeled, _ = ndi_label(padded > 0, structure=np.ones((3, 3), dtype=bool))

    with pytest.raises(UnboundLocalError):
        adjust_axis(relabeled.copy(), 1, (ny, nx), 0.5, config)
    with pytest.raises(UnboundLocalError):
        _adjust_axis_reference(relabeled.copy(), 1, (ny, nx), 0.5, config)


def test_adjust_axis_randomized():
    """Many random padded/labeled scenarios, some with seam-crossing
    features and some without, checked for exact old-vs-new equivalence."""
    from scipy.ndimage import label as ndi_label, binary_dilation

    rng = np.random.default_rng(2024)
    n_cases = 25
    n_run = 0
    for case in range(n_cases):
        ny = int(rng.integers(20, 50))
        nx = int(rng.integers(30, 70))
        n_seeds = int(rng.integers(2, 10))

        field = np.zeros((ny, nx), dtype=bool)
        # A mix of interior seeds and edge-touching seeds (the latter are
        # what create seam-crossers once wrapped).
        for _ in range(n_seeds):
            if rng.random() < 0.4:
                y = int(rng.integers(0, ny))
                x = 0 if rng.random() < 0.5 else nx - 1
            else:
                y = int(rng.integers(0, ny))
                x = int(rng.integers(0, nx))
            field[y, x] = True
        for _ in range(int(rng.integers(1, 4))):
            field = binary_dilation(field)

        ext_frac = float(rng.choice([0.2, 0.3, 0.5]))
        area_thresh = float(rng.choice([10.0, 50.0, 200.0]))
        config = {"pbc_direction": "x", "pbc_extended_fraction": ext_frac,
                  "pixel_radius": 10.0, "area_thresh": area_thresh}
        padded, _, _ = pad_and_extend(field.astype(int), config)
        relabeled, nlbl = ndi_label(padded > 0, structure=np.ones((3, 3), dtype=bool))
        if nlbl == 0:
            continue

        _assert_adjust_axis_equivalent(
            relabeled, 1, (ny, nx), ext_frac, config, label=f"random_case_{case}"
        )
        n_run += 1

    assert n_run > 15, f"Expected most of {n_cases} random cases to run, only {n_run} did"


# ---------------------------------------------------------------------------
# 3. Real data: global MCSMIP demo (the only local demo using pbc_direction
#    != 'none', and the exact scenario that surfaced this bottleneck)
# ---------------------------------------------------------------------------
DATA_ROOT = os.environ.get("PYFLEXTRKR_TEST_DATA", "")


@pytest.mark.local
def test_real_global_mcsmip_adjust_axis():
    """Run the real label_and_grow_features pipeline up to (but not
    including) the PBC-crop step on real global IMERG frames, then compare
    old-vs-new adjust_axis on the actual padded/labeled arrays it produces."""
    if not DATA_ROOT:
        pytest.skip("PYFLEXTRKR_TEST_DATA not set")
    import xarray as xr
    from scipy.signal import medfilt2d
    from pyflextrkr.label_and_grow_features import (
        classify_pixels_by_thresholds, smooth_field, find_and_label_cores,
        _grow_edt,
    )
    from pyflextrkr.ftfunctions import sort_renumber

    input_dir = os.path.join(DATA_ROOT, "mcs_tbpf/imerg_global/input")
    files = sorted(glob.glob(os.path.join(input_dir, "merg_*.nc")))
    if not files:
        pytest.skip(f"No global MCSMIP demo input files found under {input_dir}")

    config = {"pbc_direction": "x", "pbc_extended_fraction": 0.2,
              "pixel_radius": 10.0, "area_thresh": 800.0}

    n_checked = 0
    for filepath in files[:2]:
        ds = xr.open_dataset(filepath, decode_timedelta=False)
        tb_all = ds["Tb"].values
        ds.close()
        tb = tb_all[0, :, :]
        tb_filt = medfilt2d(tb.astype(np.float64), kernel_size=5)
        out_tb = np.copy(tb)
        missmask = np.isnan(tb)
        out_tb[missmask] = tb_filt[missmask]
        out_tb[out_tb < 160] = np.nan
        out_tb[out_tb > 330] = np.nan

        secondary_flag, core_flag, feature_type_map = classify_pixels_by_thresholds(
            out_tb, out_tb.shape[1], out_tb.shape[0], 261.0, 241.0, 225.0, 261.0, "lt",
        )
        padded_field, padded_x, padded_y = pad_and_extend(out_tb, config)
        padded_core, _, _ = pad_and_extend(core_flag, config)

        smoothed = smooth_field(padded_field, 10)
        labeled_cores, nlabelcores = find_and_label_cores(smoothed, 225.0, "lt")
        if nlabelcores == 0:
            continue
        sortedcore_number2d, sortedcore_npix = sort_renumber(labeled_cores, 4)
        if not np.any(sortedcore_npix > 0):
            continue

        grown = _grow_edt(sortedcore_number2d, padded_field, 261.0, "lt")

        original_shape = out_tb.shape
        _assert_adjust_axis_equivalent(
            grown, 1, original_shape, 0.2, config,
            label=f"real_global_frame_{n_checked}",
        )
        n_checked += 1

    assert n_checked > 0, "No real global frames with cores were found to test against"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
