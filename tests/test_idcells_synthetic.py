"""
Synthetic end-to-end test for the cell identification pipeline.

This test:
  1. Creates a minimal synthetic NetCDF file in memory (via a temp directory)
  2. Calls get_composite_reflectivity_generic() to exercise the reader
  3. Calls the Steiner classification core
  4. Calls expand_conv_core
  5. Calls echotop_height
  6. Verifies that the known convective cell in the synthetic data is detected

No real radar data is needed — everything is self-contained.
The test mirrors the code path used by idcells_reflectivity() without
needing valid output paths or the full run_celltracking pipeline.
"""

import os
import tempfile
import numpy as np
import pytest
import xarray as xr

from pyflextrkr.steiner_func import make_dilation_step_func, mod_steiner_classification, expand_conv_core
from pyflextrkr.echotop_func import echotop_height
from pyflextrkr.idcells_reflectivity import get_composite_reflectivity_generic


# ---------------------------------------------------------------------------
# Helpers to build synthetic data
# ---------------------------------------------------------------------------

def _make_synthetic_netcdf(tmpdir, nx=40, ny=40, nz=10, dx=1000.0):
    """
    Write a minimal synthetic 3D radar NetCDF file.

    Layout:
    - One convective cell (50 dBZ) in a 5×5 pixel block at the centre,
      extending from the surface up to level 7 (7 km).
    - All other pixels: 20 dBZ (uniform stratiform background).
    - Height coordinate: 0..9 km (1 km spacing).
    - Times: single timestamp.
    """
    height_1d = np.arange(nz, dtype=np.float32) * 1000.0   # 0..9 km
    x_1d = (np.arange(nx) - nx // 2) * dx                   # centred on 0
    y_1d = (np.arange(ny) - ny // 2) * dx

    # 3D reflectivity [time, z, y, x]
    refl = np.full((1, nz, ny, nx), 20.0, dtype=np.float32)
    cy, cx = ny // 2, nx // 2
    refl[0, :8, cy-2:cy+3, cx-2:cx+3] = 50.0  # convective core up to 7 km

    # Fake lat/lon (Cartesian approximate — OK for unit tests)
    x2d, y2d = np.meshgrid(x_1d, y_1d)
    lon2d = (x2d / 111_000.0).astype(np.float32)   # rough degrees
    lat2d = (y2d / 111_000.0).astype(np.float32)

    ds = xr.Dataset(
        {
            'reflectivity': xr.DataArray(
                refl,
                dims=['time', 'z', 'y', 'x'],
                attrs={'units': 'dBZ'},
            ),
            'point_longitude': xr.DataArray(
                lon2d[np.newaxis, np.newaxis, :, :].repeat(nz, axis=1),
                dims=['time', 'z', 'y', 'x'],
            ),
            'point_latitude': xr.DataArray(
                lat2d[np.newaxis, np.newaxis, :, :].repeat(nz, axis=1),
                dims=['time', 'z', 'y', 'x'],
            ),
        },
        coords={
            'time': xr.DataArray(
                np.array(['2014-08-07T12:00:00'], dtype='datetime64[ns]'),
                dims=['time'],
            ),
            'z':    xr.DataArray(height_1d, dims=['z'], attrs={'units': 'm'}),
            'x':    xr.DataArray(x_1d.astype(np.float32), dims=['x'], attrs={'units': 'm'}),
            'y':    xr.DataArray(y_1d.astype(np.float32), dims=['y'], attrs={'units': 'm'}),
        },
    )

    fpath = os.path.join(tmpdir, 'synthetic_radar.nc')
    ds.to_netcdf(fpath)
    return fpath, nx, ny, nz, cy, cx


def _minimal_config(fpath, dx=1000.0):
    """Return the minimum config dict needed by get_composite_reflectivity_generic."""
    return {
        'input_format': 'netcdf',
        'reflectivity_varname': 'reflectivity',
        'x_dimname': 'x',
        'y_dimname': 'y',
        'z_dimname': 'z',
        'time_dimname': 'time',
        'time_coordname': 'time',
        'x_coordname': 'x',
        'y_coordname': 'y',
        'z_coordname': 'z',
        'lon_coordname': 'point_longitude',
        'lat_coordname': 'point_latitude',
        'dx': dx,
        'dy': dx,
        'radar_sensitivity': 0.0,
        'sfc_dz_min': 0.0,
        'sfc_dz_max': 3000.0,
        'z_coord_type': 'height',
        'echotop_gap': 2,
        'fillval': -9999,
        'return_diag': False,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSyntheticCellIdentification:

    @pytest.fixture(scope='class')
    def synthetic_data(self, tmp_path_factory):
        tmpdir = str(tmp_path_factory.mktemp('radar_data'))
        fpath, nx, ny, nz, cy, cx = _make_synthetic_netcdf(tmpdir)
        config = _minimal_config(fpath)
        comp_dict = get_composite_reflectivity_generic(fpath, config)
        return comp_dict, config, nx, ny, cy, cx

    # ------------------------------------------------------------------
    def test_reader_returns_required_keys(self, synthetic_data):
        comp_dict, *_ = synthetic_data
        required = ['refl', 'dbz_comp', 'dbz3d_filt', 'dbz_lowlevel',
                    'grid_lat', 'grid_lon', 'height',
                    'mask_goodvalues', 'time_coords']
        for key in required:
            assert key in comp_dict, f"Missing key '{key}' in comp_dict"

    def test_reader_correct_shape(self, synthetic_data, grid_params=None):
        comp_dict, config, nx, ny, cy, cx = synthetic_data
        refl = comp_dict['refl']
        assert refl.shape == (ny, nx), \
            f"Expected refl shape ({ny}, {nx}), got {refl.shape}"

    def test_reader_composite_has_convective_peak(self, synthetic_data):
        comp_dict, config, nx, ny, cy, cx = synthetic_data
        dbz_comp = comp_dict['dbz_comp'].squeeze().values
        centre_dBZ = dbz_comp[cy, cx]
        assert centre_dBZ >= 50.0, \
            f"Composite dBZ at convective centre should be ~50, got {centre_dBZ}"

    # ------------------------------------------------------------------
    def test_steiner_detects_convective_cell(self, synthetic_data):
        comp_dict, config, nx, ny, cy, cx = synthetic_data

        types_steiner = {
            'NO_SURF_ECHO': 1, 'WEAK_ECHO': 2,
            'STRATIFORM': 3, 'CONVECTIVE': 4,
        }
        bkg_bin, conv_rad_bin = make_dilation_step_func(
            mindBZuse=25, dBZforMaxConvRadius=40, bkg_refl_increment=5,
            conv_rad_increment=1, conv_rad_start=1, maxConvRadius=5,
        )

        result = mod_steiner_classification(
            types_steiner,
            comp_dict['refl'],
            comp_dict['mask_goodvalues'],
            config['dx'], config['dy'],
            bkg_rad=11_000.0,
            minZdiff=8.0,
            absConvThres=43.0,
            truncZconvThres=46.0,
            weakEchoThres=15.0,
            bkg_bin=bkg_bin,
            conv_rad_bin=conv_rad_bin,
            min_corearea=0,
            min_cellarea=0,
            remove_smallcores=False,
            remove_smallcells=False,
            return_diag=False,
            convolve_method='ndimage',
        )

        sclass = result['sclass']
        assert sclass[cy, cx] == types_steiner['CONVECTIVE'], \
            f"Centre pixel should be CONVECTIVE, got class {sclass[cy, cx]}"

    # ------------------------------------------------------------------
    def test_echotop_height_at_convective_column(self, synthetic_data):
        comp_dict, config, nx, ny, cy, cx = synthetic_data
        dbz3d = comp_dict['dbz3d_filt']
        height = comp_dict['height']

        # height may be 1D or 3D depending on the reader; handle both
        if hasattr(height, 'ndim') and height.ndim > 1:
            h = height.squeeze()
            h = h[:, 0, 0] if h.ndim == 3 else h
        else:
            h = np.asarray(height).squeeze()

        echotop = echotop_height(
            dbz3d.squeeze(), h, config['z_dimname'],
            shape_2d=(ny, nx),
            dbz_thresh=35.0,
            gap=config['echotop_gap'],
            min_thick=0.0,
        )

        # The convective column has 50 dBZ up to level 7 → 7000 m
        et_centre = echotop[cy, cx]
        assert np.isfinite(et_centre), "Echo-top at convective centre should not be NaN"
        assert et_centre >= 5000.0, \
            f"Expected echo-top >= 5 km at convective centre, got {et_centre} m"

    # ------------------------------------------------------------------
    def test_expand_conv_core_produces_labelled_cells(self, synthetic_data):
        comp_dict, config, nx, ny, cy, cx = synthetic_data

        types_steiner = {
            'NO_SURF_ECHO': 1, 'WEAK_ECHO': 2,
            'STRATIFORM': 3, 'CONVECTIVE': 4,
        }
        bkg_bin, conv_rad_bin = make_dilation_step_func(
            mindBZuse=25, dBZforMaxConvRadius=40, bkg_refl_increment=5,
            conv_rad_increment=1, conv_rad_start=1, maxConvRadius=5,
        )
        result = mod_steiner_classification(
            types_steiner,
            comp_dict['refl'],
            comp_dict['mask_goodvalues'],
            config['dx'], config['dy'],
            bkg_rad=11_000.0,
            minZdiff=8.0,
            absConvThres=43.0,
            truncZconvThres=46.0,
            weakEchoThres=15.0,
            bkg_bin=bkg_bin,
            conv_rad_bin=conv_rad_bin,
            min_corearea=0,
            min_cellarea=0,
            remove_smallcores=False,
            remove_smallcells=False,
            return_diag=False,
            convolve_method='ndimage',
        )

        radii_expand = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        core_expand, core_sorted = expand_conv_core(
            result['score_dilate'], radii_expand,
            config['dx'], config['dy'],
            min_corenpix=0,
        )

        n_cells = np.nanmax(core_expand)
        assert n_cells >= 1, \
            f"Expected at least 1 labelled cell after expansion, got {n_cells}"
