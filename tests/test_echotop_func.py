"""
Unit tests for pyflextrkr/echotop_func.py.

No external data required — uses synthetic xarray DataArrays.

Tested functions
----------------
calc_cloud_boundary   – splits a vertical cloud profile into layers
echotop_height        – 2D echo-top height from 3D reflectivity
"""

import numpy as np
import pytest
import xarray as xr

from pyflextrkr.echotop_func import calc_cloud_boundary, echotop_height


# ---------------------------------------------------------------------------
# calc_cloud_boundary
# ---------------------------------------------------------------------------

class TestCalcCloudBoundary:
    def test_single_continuous_layer(self):
        """One unbroken cloud layer: base=first idx, top=last idx."""
        height = np.array([0, 1, 2, 3, 4, 5], dtype=float) * 1000.0  # metres
        idxcld = np.array([1, 2, 3, 4])
        base, top = calc_cloud_boundary(height, idxcld, gap=1, min_thick=0.0)
        assert len(base) == 1 and len(top) == 1, "One layer expected"
        assert base[0] == height[1]
        assert top[0] == height[4]

    def test_two_separated_layers(self):
        """Gap > 1 in indices should split into two layers."""
        height = np.arange(10, dtype=float) * 1000.0
        idxcld = np.array([1, 2, 5, 6, 7])   # gap at index 3-4
        base, top = calc_cloud_boundary(height, idxcld, gap=1, min_thick=0.0)
        assert len(base) == 2, f"Expected 2 layers, got {len(base)}"
        assert base[0] == height[1] and top[0] == height[2]
        assert base[1] == height[5] and top[1] == height[7]

    def test_empty_cloud_returns_empty_arrays(self):
        height = np.arange(5, dtype=float) * 1000.0
        base, top = calc_cloud_boundary(height, np.array([]), gap=1, min_thick=0.0)
        assert len(base) == 0 and len(top) == 0


# ---------------------------------------------------------------------------
# echotop_height  (1D height)
# ---------------------------------------------------------------------------

class TestEchotopHeight1D:
    @pytest.fixture
    def simple_dbz3d(self):
        """
        5×5 grid, 10 vertical levels (0..9 km).
        Column at (2,2): 50 dBZ from 0..7 km, 0 dBZ above.
        All other columns: 0 dBZ.
        """
        nz, ny, nx = 10, 5, 5
        data = np.zeros((nz, ny, nx), dtype=np.float32)
        data[:8, 2, 2] = 50.0
        height_1d = np.arange(nz, dtype=np.float32) * 1000.0
        da = xr.DataArray(data, dims=['z', 'y', 'x'],
                          coords={'z': height_1d})
        return da, height_1d

    def test_echotop_at_correct_height(self, simple_dbz3d):
        da, height = simple_dbz3d
        echotop = echotop_height(da, height, 'z',
                                  shape_2d=(5, 5),
                                  dbz_thresh=35.0,
                                  gap=1, min_thick=0.0)
        # Column (2,2) has echo up to level index 7 → height 7000 m
        assert echotop[2, 2] == pytest.approx(7000.0), \
            f"Expected echotop 7000 m, got {echotop[2, 2]}"

    def test_no_echo_above_threshold_is_nan(self, simple_dbz3d):
        da, height = simple_dbz3d
        echotop = echotop_height(da, height, 'z',
                                  shape_2d=(5, 5),
                                  dbz_thresh=35.0,
                                  gap=1, min_thick=0.0)
        # All columns except (2,2) should be NaN
        assert np.isnan(echotop[0, 0]), "Column with no echo should be NaN"

    def test_higher_threshold_lowers_echotop(self, simple_dbz3d):
        """Using a threshold above 50 dBZ should give no echo-top anywhere."""
        da, height = simple_dbz3d
        echotop = echotop_height(da, height, 'z',
                                  shape_2d=(5, 5),
                                  dbz_thresh=55.0,
                                  gap=1, min_thick=0.0)
        assert np.all(np.isnan(echotop)), \
            "No pixel reaches 55 dBZ, so every echotop should be NaN"


# ---------------------------------------------------------------------------
# echotop_height  (3D height array – variable terrain)
# ---------------------------------------------------------------------------

class TestEchotopHeight3D:
    def test_3d_height_same_result_as_1d_for_flat_terrain(self):
        """3D height that is uniform in x/y should give same result as 1D."""
        nz, ny, nx = 8, 4, 4
        data = np.zeros((nz, ny, nx), dtype=np.float32)
        data[:5, 2, 2] = 50.0
        height_1d = np.arange(nz, dtype=np.float32) * 1000.0
        # Build 3D height by broadcasting 1D → [z, y, x]
        height_3d = np.broadcast_to(
            height_1d[:, None, None], (nz, ny, nx)
        ).astype(np.float32)

        da = xr.DataArray(data, dims=['z', 'y', 'x'],
                          coords={'z': height_1d})

        echotop_1d = echotop_height(da, height_1d, 'z',
                                     shape_2d=(ny, nx),
                                     dbz_thresh=35.0, gap=1, min_thick=0.0)
        echotop_3d = echotop_height(da, height_3d, 'z',
                                     shape_2d=(ny, nx),
                                     dbz_thresh=35.0, gap=1, min_thick=0.0)
        # Both should agree where defined
        valid = np.isfinite(echotop_1d) & np.isfinite(echotop_3d)
        assert np.allclose(echotop_1d[valid], echotop_3d[valid]), \
            "1D and flat-3D height arrays should give identical echo-top heights"
