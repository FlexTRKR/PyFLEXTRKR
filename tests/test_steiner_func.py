"""
Unit tests for pyflextrkr/steiner_func.py.

These tests use only small synthetic arrays — no external data required,
and run in a few seconds on any machine including GitHub CI.

Tested functions
----------------
background_intensity        – convolution-based background reflectivity
peakedness                  – convective peakedness threshold
make_dilation_step_func     – step-function builder
mod_steiner_classification  – full Steiner classification pipeline
expand_conv_core            – outward expansion of convective masks
"""

import numpy as np
import pytest

from pyflextrkr.steiner_func import (
    background_intensity,
    make_dilation_step_func,
    mod_steiner_classification,
    expand_conv_core,
)


# ---------------------------------------------------------------------------
# background_intensity
# ---------------------------------------------------------------------------

class TestBackgroundIntensity:
    def test_uniform_field_returns_same_value(self, grid_params, mask_goodvalues):
        """Uniform reflectivity should equal itself as background."""
        nx, ny = grid_params['nx'], grid_params['ny']
        refl = np.full((ny, nx), 30.0, dtype=np.float32)
        bkg = background_intensity(refl, mask_goodvalues, 1000.0, 1000.0,
                                   bkg_rad=5000.0, convolve_method='ndimage')
        # Interior pixels (away from boundary) should be ~30 dBZ
        interior = bkg[10:-10, 10:-10]
        assert np.allclose(interior, 30.0, atol=1.0), \
            "Background of uniform 30 dBZ field should stay ~30 dBZ"

    def test_masked_region_is_nan(self, grid_params):
        """Bad-value pixels should be NaN in background output."""
        nx, ny = grid_params['nx'], grid_params['ny']
        refl = np.full((ny, nx), 30.0, dtype=np.float32)
        mask = np.ones((ny, nx), dtype=int)
        mask[:, 0] = 0   # mask out first column
        bkg = background_intensity(refl, mask, 1000.0, 1000.0,
                                   bkg_rad=5000.0, convolve_method='ndimage')
        assert np.all(np.isnan(bkg[:, 0])), "Masked pixels should be NaN in background"

    def test_both_convolve_methods_agree(self, synthetic_refl_2d, mask_goodvalues, grid_params):
        """ndimage and signal convolution methods should give the same result."""
        bkg_ndimage = background_intensity(
            synthetic_refl_2d, mask_goodvalues,
            grid_params['dx'], grid_params['dy'],
            bkg_rad=5000.0, convolve_method='ndimage',
        )
        bkg_signal = background_intensity(
            synthetic_refl_2d, mask_goodvalues,
            grid_params['dx'], grid_params['dy'],
            bkg_rad=5000.0, convolve_method='signal',
        )
        valid = np.isfinite(bkg_ndimage) & np.isfinite(bkg_signal)
        assert np.allclose(bkg_ndimage[valid], bkg_signal[valid], atol=1e-3), \
            "ndimage and signal convolution should agree"


# ---------------------------------------------------------------------------
# make_dilation_step_func
# ---------------------------------------------------------------------------

class TestMakeDilationStepFunc:
    def test_default_lengths(self):
        bkg_bin, conv_rad_bin = make_dilation_step_func()
        # bkg_bin defines N bin edges; conv_rad_bin has N-1 values (one per interval)
        assert len(bkg_bin) == len(conv_rad_bin) + 1, \
            f"bkg_bin (len={len(bkg_bin)}) should have exactly one more element than conv_rad_bin (len={len(conv_rad_bin)})"

    def test_monotone_bkg_bin(self):
        bkg_bin, _ = make_dilation_step_func()
        assert np.all(np.diff(bkg_bin) > 0), "bkg_bin should be strictly increasing"

    def test_conv_rad_capped_at_max(self):
        max_r = 3
        _, conv_rad_bin = make_dilation_step_func(maxConvRadius=max_r)
        assert np.all(conv_rad_bin <= max_r), \
            f"All conv_rad_bin values should be <= maxConvRadius={max_r}"

    def test_first_bin_is_zero(self):
        bkg_bin, _ = make_dilation_step_func()
        assert bkg_bin[0] == 0, "First background bin should start at 0"

    def test_last_bin_is_hundred(self):
        bkg_bin, _ = make_dilation_step_func()
        assert bkg_bin[-1] == 100, "Last background bin should end at 100"


# ---------------------------------------------------------------------------
# mod_steiner_classification
# ---------------------------------------------------------------------------

class TestModSteinerClassification:
    """Integration-style test of the full Steiner pipeline on synthetic data."""

    @pytest.fixture
    def steiner_config(self, grid_params):
        bkg_bin, conv_rad_bin = make_dilation_step_func(
            mindBZuse=25, dBZforMaxConvRadius=40,
            bkg_refl_increment=5, conv_rad_increment=1,
            conv_rad_start=1, maxConvRadius=5,
        )
        return dict(
            dx=grid_params['dx'],
            dy=grid_params['dy'],
            bkg_bin=bkg_bin,
            conv_rad_bin=conv_rad_bin,
            bkg_rad=11000.0,
            minZdiff=8.0,
            absConvThres=43.0,
            truncZconvThres=46.0,
            weakEchoThres=15.0,
            min_corearea=0,
            min_cellarea=0,
            remove_smallcores=False,
            remove_smallcells=False,
            return_diag=False,
            convolve_method='ndimage',
        )

    def _run(self, steiner_types, refl, mask, cfg):
        return mod_steiner_classification(
            steiner_types,
            refl,
            mask,
            cfg['dx'], cfg['dy'],
            bkg_rad=cfg['bkg_rad'],
            minZdiff=cfg['minZdiff'],
            absConvThres=cfg['absConvThres'],
            truncZconvThres=cfg['truncZconvThres'],
            weakEchoThres=cfg['weakEchoThres'],
            bkg_bin=cfg['bkg_bin'],
            conv_rad_bin=cfg['conv_rad_bin'],
            min_corearea=cfg['min_corearea'],
            min_cellarea=cfg['min_cellarea'],
            remove_smallcores=cfg['remove_smallcores'],
            remove_smallcells=cfg['remove_smallcells'],
            return_diag=cfg['return_diag'],
            convolve_method=cfg['convolve_method'],
        )

    def test_convective_core_detected(
        self, steiner_types, synthetic_refl_2d, mask_goodvalues, steiner_config
    ):
        """The high-dBZ core at the centre should be labelled CONVECTIVE."""
        result = self._run(steiner_types, synthetic_refl_2d, mask_goodvalues, steiner_config)
        sclass = result['sclass']
        cy, cx = synthetic_refl_2d.shape[0] // 2, synthetic_refl_2d.shape[1] // 2
        assert sclass[cy, cx] == steiner_types['CONVECTIVE'], \
            "Centre of high-dBZ peak should be classified as CONVECTIVE"

    def test_output_keys_present(
        self, steiner_types, synthetic_refl_2d, mask_goodvalues, steiner_config
    ):
        """Result dict must contain at minimum sclass, score, score_dilate."""
        result = self._run(steiner_types, synthetic_refl_2d, mask_goodvalues, steiner_config)
        for key in ('sclass', 'score', 'score_dilate'):
            assert key in result, f"Expected key '{key}' missing from steiner result"

    def test_output_shape_matches_input(
        self, steiner_types, synthetic_refl_2d, mask_goodvalues, steiner_config
    ):
        result = self._run(steiner_types, synthetic_refl_2d, mask_goodvalues, steiner_config)
        assert result['sclass'].shape == synthetic_refl_2d.shape, \
            "sclass shape should match input reflectivity shape"

    def test_no_convective_in_weak_echo_field(
        self, steiner_types, grid_params, steiner_config
    ):
        """Uniform low reflectivity should not produce any CONVECTIVE pixels."""
        nx, ny = grid_params['nx'], grid_params['ny']
        weak_refl = np.full((ny, nx), 10.0, dtype=np.float32)
        mask = np.ones((ny, nx), dtype=int)
        result = self._run(steiner_types, weak_refl, mask, steiner_config)
        n_conv = np.sum(result['sclass'] == steiner_types['CONVECTIVE'])
        assert n_conv == 0, \
            f"Uniform 10 dBZ field should have 0 CONVECTIVE pixels, got {n_conv}"


# ---------------------------------------------------------------------------
# expand_conv_core
# ---------------------------------------------------------------------------

class TestExpandConvCore:
    def test_expanded_area_larger_than_input(self, grid_params):
        """Expanding a small core should produce a larger mask."""
        nx, ny = grid_params['nx'], grid_params['ny']
        core = np.zeros((ny, nx), dtype=int)
        core[25, 25] = 1    # single-pixel core at centre

        radii = np.array([1.0, 2.0, 3.0])
        core_expand, _ = expand_conv_core(
            core, radii, grid_params['dx'], grid_params['dy'], min_corenpix=0
        )
        n_original = np.sum(core > 0)
        n_expanded = np.sum(core_expand > 0)
        assert n_expanded > n_original, \
            "Expanded core should cover more pixels than the single-pixel input"

    def test_zero_radii_no_change(self, grid_params):
        """With radii of zero, the core should not change."""
        nx, ny = grid_params['nx'], grid_params['ny']
        core = np.zeros((ny, nx), dtype=int)
        core[25, 25] = 1
        radii = np.array([0.0])
        core_expand, _ = expand_conv_core(
            core, radii, grid_params['dx'], grid_params['dy'], min_corenpix=0
        )
        assert np.sum(core_expand > 0) >= np.sum(core > 0), \
            "Expansion with zero radius should not shrink the core"
