"""
Tests comparing 'ndimage' and 'fft' convolution methods in background_intensity
and mod_steiner_classification, with emphasis on domain edge behaviour.

Background
----------
``ndimage.convolve(mode='constant', cval=0.0)`` pads inputs with zeros at
the boundary during direct spatial convolution.

``signal.fftconvolve(mode='same')`` zero-pads the input arrays to length
``(s1 + s2 - 1)`` along each axis before computing the FFT, which prevents
circular wrap-around artefacts.  Both approaches are mathematically equivalent
to zero-padded linear convolution, so their outputs should agree to within
floating-point precision everywhere — including at domain edges.

Purpose of these tests
----------------------
1. Confirm numerically that the two methods agree at interior pixels (where
   both methods are unambiguous).
2. Confirm that no circular-wrap artefact appears in the FFT method: a
   high-reflectivity patch near one edge must NOT raise the background near
   the opposite edge.
3. Quantify worst-case differences at realistic kernel sizes:
   - dx=500 m, bkg_rad=11 km  (kernel radius = 22 px)   ← NEXRAD / CSAPR demos
   - dx=100 m, bkg_rad=11 km  (kernel radius = 110 px)  ← LASSO WRF 100 m
4. Verify that tiny floating-point differences in background reflectivity do
   NOT propagate to different Steiner classifications (thresholds are ≥10 dBZ
   apart, so classification is robust).

All tests use synthetic arrays only — no external data required.
"""

import numpy as np
import pytest

from pyflextrkr.steiner_func import (
    background_intensity,
    make_dilation_step_func,
    mod_steiner_classification,
)

# ---------------------------------------------------------------------------
# Shared Steiner configuration (matches both NEXRAD and CSAPR demo YAMLs)
# ---------------------------------------------------------------------------

STEINER_TYPES = {
    'NO_SURF_ECHO': 1,
    'WEAK_ECHO':    2,
    'STRATIFORM':   3,
    'CONVECTIVE':   4,
}

BKG_BIN, CONV_RAD_BIN = make_dilation_step_func(
    mindBZuse=25,
    dBZforMaxConvRadius=60,
    bkg_refl_increment=5,
    conv_rad_increment=0.5,
    conv_rad_start=1.0,
    maxConvRadius=5,
)


def _bkg(refl, mask, dx, bkg_rad, method):
    """Thin wrapper to keep call sites concise."""
    return background_intensity(refl, mask, dx, dx, bkg_rad, convolve_method=method)


def _steiner(refl, mask, dx, bkg_rad, method):
    """Run full Steiner classification with return_diag=True."""
    return mod_steiner_classification(
        STEINER_TYPES, refl, mask, dx, dx,
        bkg_rad=bkg_rad,
        minZdiff=10.0,
        absConvThres=60.0,
        truncZconvThres=55.0,
        weakEchoThres=15.0,
        bkg_bin=BKG_BIN,
        conv_rad_bin=CONV_RAD_BIN,
        min_corearea=0,
        min_cellarea=0,
        remove_smallcores=False,
        remove_smallcells=False,
        return_diag=True,
        convolve_method=method,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _diff_stats(a, b):
    """Return max-abs-diff and its (row, col) location for two finite arrays."""
    diff = np.abs(a - b)
    valid = np.isfinite(diff)
    if not valid.any():
        return 0.0, (0, 0)
    max_diff = diff[valid].max()
    loc = np.unravel_index(np.nanargmax(np.where(valid, diff, np.nan)), diff.shape)
    return float(max_diff), loc


def _edge_mask(shape, width):
    """Boolean mask that is True within ``width`` pixels of any edge."""
    ny, nx = shape
    m = np.zeros(shape, dtype=bool)
    m[:width, :] = True
    m[-width:, :] = True
    m[:, :width] = True
    m[:, -width:] = True
    return m


# ---------------------------------------------------------------------------
# Part 1 – background_intensity: interior pixel agreement
# ---------------------------------------------------------------------------

class TestInteriorAgreement:
    """
    Methods must agree to within floating-point tolerance on interior pixels
    where both are unambiguously zero-padded linear convolution.
    """

    @pytest.mark.parametrize("dx,bkg_rad,ny,nx", [
        (500.0,  11_000.0, 200, 200),   # NEXRAD/CSAPR kernel: radius=22 px
        (100.0,  11_000.0, 400, 400),   # LASSO WRF kernel:    radius=110 px
    ])
    def test_interior_max_diff_below_threshold(self, dx, bkg_rad, ny, nx):
        """
        For a random realistic reflectivity field the max absolute difference
        between ndimage and fft should be < 1e-4 dBZ on interior pixels.
        """
        rng = np.random.default_rng(seed=42)
        refl = (rng.uniform(0, 1, (ny, nx)) * 50.0).astype(np.float32)
        mask = np.ones((ny, nx), dtype=int)

        bkg_ndimage = _bkg(refl, mask, dx, bkg_rad, 'ndimage')
        bkg_fft     = _bkg(refl, mask, dx, bkg_rad, 'fft')

        # Interior: more than kernel_radius pixels from any edge
        k = int(bkg_rad / dx) + 1
        interior = np.zeros((ny, nx), dtype=bool)
        interior[k:-k, k:-k] = True
        valid = interior & np.isfinite(bkg_ndimage) & np.isfinite(bkg_fft)

        max_diff, loc = _diff_stats(
            np.where(valid, bkg_ndimage, np.nan),
            np.where(valid, bkg_fft,     np.nan),
        )
        assert max_diff < 1e-4, (
            f"[dx={dx:.0f}m, bkg_rad={bkg_rad:.0f}m] "
            f"Interior max|ndimage-fft| = {max_diff:.2e} dBZ at {loc} "
            f"(threshold 1e-4 dBZ)"
        )

    def test_uniform_field_both_methods_return_same_value(self):
        """Uniform 30 dBZ interior should be ~30 dBZ with both methods."""
        ny, nx, dx, bkg_rad = 100, 100, 500.0, 11_000.0
        refl = np.full((ny, nx), 30.0, dtype=np.float32)
        mask = np.ones((ny, nx), dtype=int)

        k = int(bkg_rad / dx) + 1
        for method in ('ndimage', 'fft'):
            bkg = _bkg(refl, mask, dx, bkg_rad, method)
            interior = bkg[k:-k, k:-k]
            assert np.allclose(interior, 30.0, atol=0.5), (
                f"method={method}: uniform 30 dBZ interior should return ~30 dBZ, "
                f"got min={interior.min():.2f} max={interior.max():.2f}"
            )


# ---------------------------------------------------------------------------
# Part 2 – No circular-wrap artefact in FFT method
# ---------------------------------------------------------------------------

class TestNoCircularWrapArtefact:
    """
    A high-reflectivity patch near one edge must NOT raise the background
    near the opposite edge (which would happen with true circular convolution).
    """

    @pytest.mark.parametrize("corner", ['top_left', 'top_right', 'bottom_left', 'bottom_right'])
    def test_corner_signal_does_not_wrap_to_opposite_corner(self, corner):
        """
        Place a 50 dBZ 3×3-pixel patch in one corner; the opposite corner's
        background (computed by both methods) must remain near background level.
        """
        ny, nx   = 200, 200
        dx       = 500.0
        bkg_rad  = 11_000.0        # kernel radius = 22 px
        refl     = np.full((ny, nx), 10.0, dtype=np.float32)
        mask     = np.ones((ny, nx), dtype=int)

        # Place the high-dBZ patch
        offsets = {
            'top_left':     (slice(0, 4), slice(0, 4)),
            'top_right':    (slice(0, 4), slice(-4, None)),
            'bottom_left':  (slice(-4, None), slice(0, 4)),
            'bottom_right': (slice(-4, None), slice(-4, None)),
        }
        opposite = {
            'top_left':     (slice(-4, None), slice(-4, None)),
            'top_right':    (slice(-4, None), slice(0, 4)),
            'bottom_left':  (slice(0, 4), slice(-4, None)),
            'bottom_right': (slice(0, 4), slice(0, 4)),
        }
        refl[offsets[corner]] = 50.0

        bkg_ndimage = _bkg(refl, mask, dx, bkg_rad, 'ndimage')
        bkg_fft     = _bkg(refl, mask, dx, bkg_rad, 'fft')

        # The opposite corner should be unaffected (background ≈ 10 dBZ).
        # If FFT wraps, it would be elevated relative to ndimage.
        opp = opposite[corner]
        diff_opp = np.abs(bkg_fft[opp] - bkg_ndimage[opp])
        max_diff = float(np.nanmax(diff_opp))

        assert max_diff < 1.0, (
            f"[corner={corner}] Max |fft-ndimage| at opposite corner = "
            f"{max_diff:.3f} dBZ — suggests circular wrap-around in FFT method"
        )

    def test_left_edge_signal_does_not_bleed_to_right_edge(self):
        """
        A vertical band of 50 dBZ along the left 3 columns must not elevate
        background on the right 3 columns in the FFT result.
        """
        ny, nx   = 200, 200
        dx       = 500.0
        bkg_rad  = 11_000.0
        refl     = np.full((ny, nx), 10.0, dtype=np.float32)
        refl[:, :3] = 50.0
        mask     = np.ones((ny, nx), dtype=int)

        bkg_ndimage = _bkg(refl, mask, dx, bkg_rad, 'ndimage')
        bkg_fft     = _bkg(refl, mask, dx, bkg_rad, 'fft')

        right_diff = np.abs(bkg_fft[:, -3:] - bkg_ndimage[:, -3:])
        max_diff   = float(np.nanmax(right_diff))

        assert max_diff < 1.0, (
            f"Left-edge signal bled to right edge: max |fft-ndimage| = "
            f"{max_diff:.3f} dBZ — circular wrap-around detected"
        )


# ---------------------------------------------------------------------------
# Part 3 – Quantify edge vs interior differences
# ---------------------------------------------------------------------------

class TestEdgeDifferenceQuantification:
    """
    Measure and document the actual difference between methods at edge and
    interior pixels for realistic kernel sizes.  This lets the user understand
    the magnitude of any discrepancy and decide whether it is acceptable.
    """

    @pytest.mark.parametrize("dx,bkg_rad,ny,nx,label", [
        (500.0, 11_000.0, 300, 300, 'NEXRAD_500m'),
        (100.0, 11_000.0, 400, 400, 'LASSO_100m'),
    ])
    def test_edge_vs_interior_difference(self, dx, bkg_rad, ny, nx, label, capsys):
        """
        Compute and print the max |ndimage - fft| separately for edge pixels
        (within kernel_radius of boundary) and interior pixels.

        Asserts that neither region exceeds a conservative threshold of 1e-3 dBZ.
        This threshold is far below the Steiner classification thresholds
        (minimum 10 dBZ peakedness), so any differences cannot affect sclass.
        """
        rng = np.random.default_rng(seed=7)
        # Realistic field: mix of stratiform (20-30 dBZ) and convective (40-55 dBZ)
        refl = np.full((ny, nx), 20.0, dtype=np.float32)
        cy, cx = ny // 2, nx // 2
        rr = 15  # convective patch radius [pixels]
        yy, xx = np.ogrid[:ny, :nx]
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        refl[r < rr] = 50.0
        refl[(r >= rr) & (r < rr * 2)] = 30.0
        # Add a patch near the top-left corner to exercise edge behaviour
        refl[:10, :10] = 48.0
        mask = np.ones((ny, nx), dtype=int)

        bkg_ndimage = _bkg(refl, mask, dx, bkg_rad, 'ndimage')
        bkg_fft     = _bkg(refl, mask, dx, bkg_rad, 'fft')

        k = int(bkg_rad / dx)
        edge_m    = _edge_mask((ny, nx), k)
        valid     = np.isfinite(bkg_ndimage) & np.isfinite(bkg_fft)

        edge_diff     = np.abs(bkg_fft - bkg_ndimage)
        edge_diff_vals   = edge_diff[edge_m & valid]
        interior_diff_vals = edge_diff[~edge_m & valid]

        max_edge     = float(edge_diff_vals.max())     if edge_diff_vals.size     else 0.0
        max_interior = float(interior_diff_vals.max()) if interior_diff_vals.size else 0.0

        with capsys.disabled():
            print(
                f"\n[{label}] dx={dx:.0f}m, bkg_rad={bkg_rad:.0f}m, "
                f"kernel_radius={k}px, domain={ny}x{nx}\n"
                f"  Max |ndimage - fft| at edge pixels (within {k}px of boundary): "
                f"{max_edge:.3e} dBZ\n"
                f"  Max |ndimage - fft| at interior pixels: {max_interior:.3e} dBZ"
            )

        assert max_interior < 1e-3, (
            f"[{label}] Interior max diff = {max_interior:.3e} dBZ exceeds 1e-3 dBZ threshold"
        )
        assert max_edge < 1e-3, (
            f"[{label}] Edge max diff = {max_edge:.3e} dBZ exceeds 1e-3 dBZ threshold\n"
            f"  If this fails, the 'fft' method may need explicit zero-padding to match ndimage exactly."
        )


# ---------------------------------------------------------------------------
# Part 4 – Classification equivalence (sclass, score, score_dilate)
# ---------------------------------------------------------------------------

class TestClassificationEquivalence:
    """
    Even if background reflectivity differs slightly between methods, the
    final Steiner classification arrays must be identical because the
    classification thresholds (minZdiff=10 dBZ, absConvThres=60 dBZ, etc.)
    are far larger than any floating-point differences.
    """

    def _make_realistic_field(self, ny, nx):
        """Stratiform background with a convective core near an edge."""
        refl = np.full((ny, nx), 20.0, dtype=np.float32)
        # Convective core near top-left corner (tests edge classification)
        refl[1:8, 1:8] = 55.0
        # Convective core at centre
        cy, cx = ny // 2, nx // 2
        refl[cy-3:cy+4, cx-3:cx+4] = 52.0
        # Stratiform ring
        yy, xx = np.ogrid[:ny, :nx]
        r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        refl[(r >= 5) & (r < 15)] = 30.0
        mask = np.ones((ny, nx), dtype=int)
        return refl, mask

    @pytest.mark.parametrize("dx,bkg_rad,ny,nx,label", [
        (500.0, 11_000.0, 200, 200, 'NEXRAD_500m'),
        (100.0, 11_000.0, 400, 400, 'LASSO_100m'),
    ])
    def test_sclass_arrays_identical(self, dx, bkg_rad, ny, nx, label):
        """sclass (convective/stratiform mask) must be elementally identical."""
        refl, mask = self._make_realistic_field(ny, nx)
        r_ndimage = _steiner(refl, mask, dx, bkg_rad, 'ndimage')
        r_fft     = _steiner(refl, mask, dx, bkg_rad, 'fft')

        n_diff = int(np.sum(r_ndimage['sclass'] != r_fft['sclass']))
        assert n_diff == 0, (
            f"[{label}] sclass differs at {n_diff} pixel(s) between ndimage and fft"
        )

    @pytest.mark.parametrize("dx,bkg_rad,ny,nx,label", [
        (500.0, 11_000.0, 200, 200, 'NEXRAD_500m'),
        (100.0, 11_000.0, 400, 400, 'LASSO_100m'),
    ])
    def test_score_dilate_arrays_identical(self, dx, bkg_rad, ny, nx, label):
        """score_dilate (dilated convective core) must be elementally identical."""
        refl, mask = self._make_realistic_field(ny, nx)
        r_ndimage = _steiner(refl, mask, dx, bkg_rad, 'ndimage')
        r_fft     = _steiner(refl, mask, dx, bkg_rad, 'fft')

        n_diff = int(np.sum(r_ndimage['score_dilate'] != r_fft['score_dilate']))
        assert n_diff == 0, (
            f"[{label}] score_dilate differs at {n_diff} pixel(s) between ndimage and fft"
        )

    def test_convective_core_at_domain_edge_detected_by_both_methods(self):
        """
        A 55 dBZ patch placed within 3 pixels of the domain boundary exceeds
        truncZconvThres=55 dBZ, so it must be classified CONVECTIVE by both methods.
        """
        ny, nx, dx, bkg_rad = 100, 100, 500.0, 11_000.0
        refl = np.full((ny, nx), 20.0, dtype=np.float32)
        # Strong convective core touching the top edge
        refl[0:4, 40:60] = 56.0
        mask = np.ones((ny, nx), dtype=int)

        for method in ('ndimage', 'fft'):
            result = _steiner(refl, mask, dx, bkg_rad, method)
            # Pixels with refl > truncZconvThres must be CONVECTIVE
            core_pixels = (refl[0:4, 40:60] > 55.0)
            sclass_core = result['sclass'][0:4, 40:60][core_pixels]
            n_conv = int(np.sum(sclass_core == STEINER_TYPES['CONVECTIVE']))
            assert n_conv == core_pixels.sum(), (
                f"[method={method}] Edge convective core: only {n_conv}/"
                f"{core_pixels.sum()} pixels labelled CONVECTIVE"
            )

    def test_refl_bkg_difference_at_classification_boundary(self):
        """
        The background reflectivity difference between methods must be much
        smaller than the peakedness threshold (minZdiff=10 dBZ), ensuring no
        misclassification can occur.
        """
        ny, nx, dx, bkg_rad = 200, 200, 500.0, 11_000.0
        rng = np.random.default_rng(seed=99)
        refl = (rng.uniform(15, 55, (ny, nx))).astype(np.float32)
        mask = np.ones((ny, nx), dtype=int)

        r_ndimage = _steiner(refl, mask, dx, bkg_rad, 'ndimage')
        r_fft     = _steiner(refl, mask, dx, bkg_rad, 'fft')

        refl_bkg_diff = np.abs(
            r_ndimage['refl_bkg'] - r_fft['refl_bkg']
        )
        valid_diff = refl_bkg_diff[np.isfinite(refl_bkg_diff)]
        max_diff = float(valid_diff.max()) if valid_diff.size else 0.0

        # Must be far below minZdiff=10 dBZ — use 1.0 dBZ as conservative bound
        assert max_diff < 1.0, (
            f"Max |refl_bkg_ndimage - refl_bkg_fft| = {max_diff:.4f} dBZ "
            f"(threshold 1.0 dBZ, minZdiff=10 dBZ)"
        )
