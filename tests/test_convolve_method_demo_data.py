"""
Local integration tests for cell identification functions using real demo data
(demo_cell_nexrad: NEXRAD KHGX, demo_cell_csapr: CACTI CSAPR-2).

All tests are marked @pytest.mark.local and skipped unless PYFLEXTRKR_TEST_DATA is set.

Test classes
------------
Convolution method comparison (ndimage vs fft):
  TestNexradConvolveMethod, TestCsaprConvolveMethod
    - background_intensity agreement: interior < 1e-4 dBZ, edge < 1e-3 dBZ
    - full Steiner sclass and score_dilate must be elementally identical

Fast / EDT function validation:
  TestEchotopHeightFast
    - echotop_height_fast vs echotop_height for 5 dBZ thresholds (10-50)
    - assert max difference < 1 m (float32 rounding)
  TestLabelCellsFast
    - label_cells_fast vs label_cells: bit-identical output
  TestExpandConvCoreFast
    - expand_conv_core_fast vs expand_conv_core: bit-identical output
  TestExpandConvCoreEdt
    - expand_conv_core_edt vs orig and fast: <= 0.1% pixel difference
    - EDT uses nearest-core Voronoi; sub-pixel tie-breaking at equidistant
      boundaries may differ from the largest-core-wins rule in grey_dilation
  TestModDilateConvRadEdt
    - mod_dilate_conv_rad_edt vs mod_dilate_conv_rad: <= 0.1% pixel difference
    - EDT applies mask_goodvalues as output gate only, not as propagation barrier

Prerequisites
-------------
Demo input files must be present under $PYFLEXTRKR_TEST_DATA:
  cell_radar/nexrad/input/KHGX*.nc
  cell_radar/csapr/input/taranis_corcsapr2*.nc

Download automatically with:
  python tests/run_demo_tests.py --demos demo_cell_nexrad demo_cell_csapr -n 4

Or set the variable manually after placing data:
  export PYFLEXTRKR_TEST_DATA=~/data/demo

Usage
-----
  # Run all local tests with verbose output:
  conda run -n pyflextrkr bash -c \\
    "PYFLEXTRKR_TEST_DATA=~/data/demo \\
     pytest tests/test_convolve_method_demo_data.py -m local -v -s"
"""

import glob
import os
import time

import numpy as np
import pytest

from pyflextrkr.idcells_reflectivity import get_composite_reflectivity_generic
from pyflextrkr.steiner_func import (
    background_intensity,
    make_dilation_step_func,
    mod_steiner_classification,
)

# ---------------------------------------------------------------------------
# Paths and demo-specific configuration (must match the demo YAML files)
# ---------------------------------------------------------------------------

DATA_ROOT = os.environ.get("PYFLEXTRKR_TEST_DATA", "")


def demo_path(*parts):
    return os.path.join(DATA_ROOT, *parts)


# Steiner config that matches both demo YAML files (config_nexrad500m_example.yml
# and config_csapr500m_example.yml — parameters are identical between the two)
_BKG_BIN, _CONV_RAD_BIN = make_dilation_step_func(
    mindBZuse=25,
    dBZforMaxConvRadius=60,
    bkg_refl_increment=5,
    conv_rad_increment=0.5,
    conv_rad_start=1.0,
    maxConvRadius=5,
)

_STEINER_TYPES = {
    'NO_SURF_ECHO': 1,
    'WEAK_ECHO':    2,
    'STRATIFORM':   3,
    'CONVECTIVE':   4,
}


def _demo_steiner_params(dx=500.0, bkg_rad_km=11.0):
    """Return the Steiner parameter dict matching the demo YAML configs."""
    return dict(
        dx=dx, dy=dx,
        bkg_rad=bkg_rad_km * 1000.0,
        minZdiff=10.0,
        absConvThres=60.0,
        truncZconvThres=55.0,
        weakEchoThres=15.0,
        bkg_bin=_BKG_BIN,
        conv_rad_bin=_CONV_RAD_BIN,
        min_corearea=4.0,
        min_cellarea=0.0,
        remove_smallcores=True,
        remove_smallcells=False,
        return_diag=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_first_input_file(input_dir, pattern="*.nc"):
    """Return the first (alphabetically sorted) input NetCDF file."""
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        # Some demos nest files in subdirectories
        files = sorted(glob.glob(os.path.join(input_dir, "**", pattern), recursive=True))
    return files[0] if files else None


def _bkg(refl, mask, dx, bkg_rad, method):
    return background_intensity(refl, mask, dx, dx, bkg_rad, convolve_method=method)


def _run_steiner(refl, mask, params, method):
    return mod_steiner_classification(
        _STEINER_TYPES, refl, mask,
        params['dx'], params['dy'],
        bkg_rad=params['bkg_rad'],
        minZdiff=params['minZdiff'],
        absConvThres=params['absConvThres'],
        truncZconvThres=params['truncZconvThres'],
        weakEchoThres=params['weakEchoThres'],
        bkg_bin=params['bkg_bin'],
        conv_rad_bin=params['conv_rad_bin'],
        min_corearea=params['min_corearea'],
        min_cellarea=params['min_cellarea'],
        remove_smallcores=params['remove_smallcores'],
        remove_smallcells=params['remove_smallcells'],
        return_diag=params['return_diag'],
        convolve_method=method,
    )


def _edge_mask(shape, width):
    """Boolean mask that is True within ``width`` pixels of any edge."""
    ny, nx = shape
    m = np.zeros(shape, dtype=bool)
    m[:width, :] = True
    m[-width:, :] = True
    m[:, :width] = True
    m[:, -width:] = True
    return m


def _report_diff(label, field_ndimage, field_fft, kernel_radius, capsys):
    """Print a structured report of differences between two 2D fields."""
    diff = np.abs(field_fft - field_ndimage)
    valid = np.isfinite(diff)
    em = _edge_mask(diff.shape, kernel_radius)

    interior_diff = diff[~em & valid]
    edge_diff     = diff[em & valid]

    max_interior = float(interior_diff.max()) if interior_diff.size else 0.0
    max_edge     = float(edge_diff.max())     if edge_diff.size     else 0.0
    mean_interior = float(interior_diff.mean()) if interior_diff.size else 0.0
    mean_edge     = float(edge_diff.mean())     if edge_diff.size     else 0.0

    if valid.any():
        worst_loc = np.unravel_index(
            np.nanargmax(np.where(valid, diff, np.nan)), diff.shape
        )
    else:
        worst_loc = (-1, -1)

    with capsys.disabled():
        print(
            f"\n[{label}] domain={diff.shape[0]}×{diff.shape[1]}, "
            f"kernel_radius={kernel_radius}px\n"
            f"  refl_bkg |ndimage - fft|:\n"
            f"    Interior (>{kernel_radius}px from edge): "
            f"max={max_interior:.3e} dBZ, mean={mean_interior:.3e} dBZ\n"
            f"    Edge     (≤{kernel_radius}px from edge): "
            f"max={max_edge:.3e} dBZ, mean={mean_edge:.3e} dBZ\n"
            f"    Worst pixel: {worst_loc}"
        )
    return max_interior, max_edge


# ---------------------------------------------------------------------------
# Shared base class
# ---------------------------------------------------------------------------

class _ConvolveCompareBase:
    """
    Base class holding parametrised tests for ndimage vs fft comparison.
    Subclasses supply INPUT_DIR, INPUT_PATTERN, CONFIG, and LABEL.
    """

    INPUT_DIR = ""      # path to demo input directory
    INPUT_PATTERN = "*.nc"
    CONFIG = {}         # dict with get_composite_reflectivity_generic keys
    PARAMS = {}         # dict returned by _demo_steiner_params()
    LABEL = ""

    @pytest.fixture(scope="class")
    def comp_dict(self):
        """
        Load the first input file and return the composite reflectivity dict.
        Skips the test class if input data is not present.
        """
        first_file = _find_first_input_file(self.INPUT_DIR, self.INPUT_PATTERN)
        if first_file is None:
            pytest.skip(
                f"No input files found in {self.INPUT_DIR}\n"
                f"Run: python tests/run_demo_tests.py --demos {self.LABEL} -n 4"
            )
        return get_composite_reflectivity_generic(first_file, self.CONFIG)

    @pytest.fixture(scope="class")
    def refl_mask(self, comp_dict):
        """Extract a single-time composite reflectivity array and mask."""
        dbz_comp = comp_dict['dbz_comp']
        # dbz_comp may be an xarray.DataArray; convert to numpy
        refl = np.array(dbz_comp.values if hasattr(dbz_comp, 'values') else dbz_comp,
                        dtype=np.float32)
        # Replace NaN with radar_sensitivity so mask is consistent
        radar_sens = self.CONFIG.get('radar_sensitivity', 0.0)
        mask = np.ones(refl.shape, dtype=int)
        mask[~np.isfinite(refl)] = 0
        refl[mask == 0] = radar_sens
        # Squeeze any leading singleton dimensions (time)
        refl = refl.squeeze()
        mask = mask.squeeze()
        return refl, mask

    # ── Test 1: background reflectivity agreement (all pixels) ─────────────

    def test_background_reflectivity_interior_max_diff(self, refl_mask, capsys):
        """Max |ndimage - fft| at interior pixels must be < 1e-4 dBZ."""
        refl, mask = refl_mask
        dx  = self.PARAMS['dx']
        bkg_rad = self.PARAMS['bkg_rad']
        k   = int(bkg_rad / dx)

        t0 = time.time()
        bkg_ndimage = _bkg(refl, mask, dx, bkg_rad, 'ndimage')
        t_ndimage   = time.time() - t0
        t0 = time.time()
        bkg_fft     = _bkg(refl, mask, dx, bkg_rad, 'fft')
        t_fft       = time.time() - t0

        with capsys.disabled():
            print(
                f"\n[{self.LABEL}] Timing: ndimage={t_ndimage:.2f}s, "
                f"fft={t_fft:.2f}s, speedup={t_ndimage/t_fft:.1f}x"
            )

        max_interior, max_edge = _report_diff(
            self.LABEL, bkg_ndimage, bkg_fft, k, capsys
        )
        assert max_interior < 1e-4, (
            f"[{self.LABEL}] Interior max diff = {max_interior:.3e} dBZ (threshold 1e-4 dBZ)"
        )

    def test_background_reflectivity_edge_max_diff(self, refl_mask, capsys):
        """Max |ndimage - fft| at edge pixels (within kernel_radius) must be < 1e-3 dBZ."""
        refl, mask = refl_mask
        dx  = self.PARAMS['dx']
        bkg_rad = self.PARAMS['bkg_rad']
        k   = int(bkg_rad / dx)

        bkg_ndimage = _bkg(refl, mask, dx, bkg_rad, 'ndimage')
        bkg_fft     = _bkg(refl, mask, dx, bkg_rad, 'fft')

        em   = _edge_mask(refl.shape, k)
        diff = np.abs(bkg_fft - bkg_ndimage)
        valid_edge = em & np.isfinite(diff)
        max_edge = float(diff[valid_edge].max()) if valid_edge.any() else 0.0

        assert max_edge < 1e-3, (
            f"[{self.LABEL}] Edge max diff = {max_edge:.3e} dBZ (threshold 1e-3 dBZ)\n"
            "  If this fails, consider explicit zero-padding in background_intensity for 'fft' method."
        )

    # ── Test 2: Steiner classification agreement ────────────────────────────

    def test_sclass_identical_ndimage_vs_fft(self, refl_mask, capsys):
        """Full Steiner sclass array must be elementally identical."""
        refl, mask = refl_mask
        r_ndimage = _run_steiner(refl, mask, self.PARAMS, 'ndimage')
        r_fft     = _run_steiner(refl, mask, self.PARAMS, 'fft')

        diff_pixels = np.sum(r_ndimage['sclass'] != r_fft['sclass'])
        total = int(np.sum(mask > 0))

        with capsys.disabled():
            print(
                f"\n[{self.LABEL}] sclass differences: "
                f"{diff_pixels}/{total} pixels "
                f"({100.0*diff_pixels/max(total,1):.4f}%)"
            )
        assert diff_pixels == 0, (
            f"[{self.LABEL}] sclass differs at {diff_pixels} pixel(s). "
            f"Check refl_bkg differences above for root cause."
        )

    def test_score_dilate_identical_ndimage_vs_fft(self, refl_mask, capsys):
        """Dilated convective core (score_dilate) must be elementally identical."""
        refl, mask = refl_mask
        r_ndimage = _run_steiner(refl, mask, self.PARAMS, 'ndimage')
        r_fft     = _run_steiner(refl, mask, self.PARAMS, 'fft')

        diff_pixels = np.sum(r_ndimage['score_dilate'] != r_fft['score_dilate'])
        with capsys.disabled():
            print(
                f"\n[{self.LABEL}] score_dilate differences: {diff_pixels} pixel(s)"
            )
        assert diff_pixels == 0, (
            f"[{self.LABEL}] score_dilate differs at {diff_pixels} pixel(s)."
        )

    # ── Test 3: Edge-specific pixel comparison ──────────────────────────────

    def test_edge_pixel_background_vs_interior(self, refl_mask, capsys):
        """
        Measure and compare edge vs interior |ndimage - fft| differences.
        Reports exact values; asserts both stay below their respective thresholds.
        """
        refl, mask = refl_mask
        dx  = self.PARAMS['dx']
        bkg_rad = self.PARAMS['bkg_rad']
        k   = int(bkg_rad / dx)

        bkg_ndimage = _bkg(refl, mask, dx, bkg_rad, 'ndimage')
        bkg_fft     = _bkg(refl, mask, dx, bkg_rad, 'fft')

        diff  = np.abs(bkg_fft - bkg_ndimage)
        valid = np.isfinite(diff) & (mask > 0).reshape(diff.shape)
        em    = _edge_mask(refl.shape, k)

        interior_vals = diff[~em & valid]
        edge_vals     = diff[em & valid]

        max_interior = float(interior_vals.max()) if interior_vals.size else 0.0
        max_edge     = float(edge_vals.max())     if edge_vals.size     else 0.0

        # Find the edge pixel with the largest difference
        edge_diff = np.where(em & valid, diff, np.nan)
        if np.any(np.isfinite(edge_diff)):
            worst_edge_loc = np.unravel_index(np.nanargmax(edge_diff), diff.shape)
            worst_edge_refl = float(refl[worst_edge_loc])
            worst_edge_ndimage = float(bkg_ndimage[worst_edge_loc])
            worst_edge_fft     = float(bkg_fft[worst_edge_loc])
        else:
            worst_edge_loc = (-1, -1)
            worst_edge_refl = worst_edge_ndimage = worst_edge_fft = float('nan')

        with capsys.disabled():
            print(
                f"\n[{self.LABEL}] Edge pixel analysis (kernel_radius={k}px):\n"
                f"  Interior max diff : {max_interior:.3e} dBZ\n"
                f"  Edge     max diff : {max_edge:.3e} dBZ\n"
                f"  Worst edge pixel  : {worst_edge_loc}\n"
                f"    refl            : {worst_edge_refl:.2f} dBZ\n"
                f"    bkg_ndimage     : {worst_edge_ndimage:.4f} dBZ\n"
                f"    bkg_fft         : {worst_edge_fft:.4f} dBZ\n"
                f"    difference      : {max_edge:.4e} dBZ\n"
                f"  Note: classification thresholds are ≥10 dBZ, so differences\n"
                f"  below 1.0 dBZ cannot cause misclassification."
            )

        assert max_interior < 1e-4, (
            f"[{self.LABEL}] Interior diff = {max_interior:.3e} dBZ exceeds 1e-4 dBZ"
        )
        assert max_edge < 1e-3, (
            f"[{self.LABEL}] Edge diff = {max_edge:.3e} dBZ exceeds 1e-3 dBZ"
        )


# ---------------------------------------------------------------------------
# NEXRAD demo (demo_cell_nexrad)
# ---------------------------------------------------------------------------

@pytest.mark.local
class TestNexradConvolveMethod(_ConvolveCompareBase):
    """
    Compare ndimage vs fft convolution on real NEXRAD data (KHGX, 2014-08-07).

    Input files: KHGX*.nc in cell_radar/nexrad/input/
    Config:  config_nexrad500m_example.yml  (dx=500m, bkgrndRadius=11km)
    """

    INPUT_DIR     = demo_path('cell_radar', 'nexrad', 'input')
    INPUT_PATTERN = "KHGX*.nc"
    LABEL         = "demo_cell_nexrad"
    PARAMS        = _demo_steiner_params(dx=500.0, bkg_rad_km=11.0)

    CONFIG = {
        # Spatial grid
        'dx': 500, 'dy': 500,
        # Dimension and coordinate names (must match NEXRAD PyART grid files)
        'x_dimname': 'x', 'y_dimname': 'y', 'z_dimname': 'z',
        'time_dimname': 'time', 'time_coordname': 'time',
        'x_coordname': 'x', 'y_coordname': 'y', 'z_coordname': 'z',
        'lon_coordname': 'point_longitude', 'lat_coordname': 'point_latitude',
        'radar_lon_varname': 'origin_longitude',
        'radar_lat_varname': 'origin_latitude',
        'radar_alt_varname': 'alt',
        'reflectivity_varname': 'reflectivity',
        # Vertical range for composite reflectivity
        'radar_sensitivity': 0.0,
        'sfc_dz_min': 1000, 'sfc_dz_max': 3000,
        # Terrain
        'terrain_file': None,
        'elev_varname': None,
        'rangemask_varname': 'mask110',
        'z_coord_type': 'height',
        'input_format': 'netcdf',
    }


# ---------------------------------------------------------------------------
# CSAPR demo (demo_cell_csapr)
# ---------------------------------------------------------------------------

@pytest.mark.local
class TestCsaprConvolveMethod(_ConvolveCompareBase):
    """
    Compare ndimage vs fft convolution on real CSAPR-2 data (CACTI, 2019-01-25).

    Input files: taranis_corcsapr2*.nc in cell_radar/csapr/input/
    Config:  config_csapr500m_example.yml  (dx=500m, bkgrndRadius=11km)
    """

    INPUT_DIR     = demo_path('cell_radar', 'csapr', 'input')
    INPUT_PATTERN = "taranis_corcsapr2*.nc"
    LABEL         = "demo_cell_csapr"
    PARAMS        = _demo_steiner_params(dx=500.0, bkg_rad_km=11.0)

    CONFIG = {
        # Spatial grid
        'dx': 500, 'dy': 500,
        # Dimension and coordinate names (CSAPR PyART grid format)
        'x_dimname': 'x', 'y_dimname': 'y', 'z_dimname': 'z',
        'time_dimname': 'time', 'time_coordname': 'time',
        'x_coordname': 'x', 'y_coordname': 'y', 'z_coordname': 'z',
        'lon_coordname': 'point_longitude', 'lat_coordname': 'point_latitude',
        'radar_lon_varname': 'origin_longitude',
        'radar_lat_varname': 'origin_latitude',
        'radar_alt_varname': 'alt',
        'reflectivity_varname': 'taranis_attenuation_corrected_reflectivity',
        # Vertical range for composite reflectivity
        'radar_sensitivity': 0.0,
        'sfc_dz_min': 500, 'sfc_dz_max': 3000,
        # Terrain
        'terrain_file': None,
        'elev_varname': None,
        'rangemask_varname': 'mask110',
        'z_coord_type': 'height',
        'input_format': 'netcdf',
    }


# ---------------------------------------------------------------------------
# Validation tests for new fast functions
# ---------------------------------------------------------------------------

# Additional imports needed by the new tests
from pyflextrkr.echotop_func import echotop_height, echotop_height_fast
from pyflextrkr.steiner_func import (
    label_cells,
    label_cells_fast,
    expand_conv_core,
    expand_conv_core_fast,
)


# ── Shared fixture helpers ──────────────────────────────────────────────────

_NEXRAD_CONFIG = {
    'dx': 500, 'dy': 500,
    'x_dimname': 'x', 'y_dimname': 'y', 'z_dimname': 'z',
    'time_dimname': 'time', 'time_coordname': 'time',
    'x_coordname': 'x', 'y_coordname': 'y', 'z_coordname': 'z',
    'lon_coordname': 'point_longitude', 'lat_coordname': 'point_latitude',
    'radar_lon_varname': 'origin_longitude',
    'radar_lat_varname': 'origin_latitude',
    'radar_alt_varname': 'alt',
    'reflectivity_varname': 'reflectivity',
    'radar_sensitivity': 0.0,
    'sfc_dz_min': 1000, 'sfc_dz_max': 3000,
    'terrain_file': None, 'elev_varname': None,
    'rangemask_varname': 'mask110',
    'z_coord_type': 'height', 'input_format': 'netcdf',
}

_CSAPR_CONFIG = {
    'dx': 500, 'dy': 500,
    'x_dimname': 'x', 'y_dimname': 'y', 'z_dimname': 'z',
    'time_dimname': 'time', 'time_coordname': 'time',
    'x_coordname': 'x', 'y_coordname': 'y', 'z_coordname': 'z',
    'lon_coordname': 'point_longitude', 'lat_coordname': 'point_latitude',
    'radar_lon_varname': 'origin_longitude',
    'radar_lat_varname': 'origin_latitude',
    'radar_alt_varname': 'alt',
    'reflectivity_varname': 'taranis_attenuation_corrected_reflectivity',
    'radar_sensitivity': 0.0,
    'sfc_dz_min': 500, 'sfc_dz_max': 3000,
    'terrain_file': None, 'elev_varname': None,
    'rangemask_varname': 'mask110',
    'z_coord_type': 'height', 'input_format': 'netcdf',
}


def _load_demo_3d(input_dir, pattern, config):
    """Return (comp_dict, first_file) for the given demo directory."""
    first_file = _find_first_input_file(input_dir, pattern)
    if first_file is None:
        return None, None
    return get_composite_reflectivity_generic(first_file, config), first_file


def _steiner_result_for_expand(comp_dict):
    """Run Steiner classification and return (core_dilate, dx, radii_expand)."""
    from pyflextrkr.steiner_func import make_dilation_step_func, mod_steiner_classification
    dx = 500.0
    bkg_bin, conv_rad_bin = make_dilation_step_func(
        mindBZuse=25, dBZforMaxConvRadius=60,
        bkg_refl_increment=5, conv_rad_increment=0.5,
        conv_rad_start=1.0, maxConvRadius=5,
    )
    types_steiner = {'NO_SURF_ECHO': 1, 'WEAK_ECHO': 2, 'STRATIFORM': 3, 'CONVECTIVE': 4}
    refl = comp_dict['refl']
    mask = comp_dict['mask_goodvalues']
    result = mod_steiner_classification(
        types_steiner, refl, mask, dx, dx,
        bkg_rad=11000.0, minZdiff=10.0, absConvThres=60.0,
        truncZconvThres=55.0, weakEchoThres=15.0,
        bkg_bin=bkg_bin, conv_rad_bin=conv_rad_bin,
        min_corearea=4.0, min_cellarea=0.0,
        remove_smallcores=True, remove_smallcells=False,
        return_diag=False, convolve_method='fft',
    )
    radii_expand = np.array([1.0, 2.0, 3.0])
    return result['score_dilate'], dx, radii_expand


# ── TestEchotopHeightFast ───────────────────────────────────────────────────

@pytest.mark.local
class TestEchotopHeightFast:
    """
    Verify that echotop_height_fast produces output that agrees with
    echotop_height to within 1 m (float32 rounding) for both demo datasets.
    """

    @pytest.mark.parametrize("demo,input_dir,pattern,config", [
        ("nexrad", demo_path('cell_radar', 'nexrad', 'input'),
         "KHGX*.nc", _NEXRAD_CONFIG),
        ("csapr",  demo_path('cell_radar', 'csapr', 'input'),
         "taranis_corcsapr2*.nc", _CSAPR_CONFIG),
    ])
    @pytest.mark.parametrize("thresh", [10, 20, 30, 40, 50])
    def test_echotop_agrees(self, demo, input_dir, pattern, config, thresh, capsys):
        """echotop_height_fast == echotop_height within 1 m for every threshold."""
        comp_dict, first_file = _load_demo_3d(input_dir, pattern, config)
        if comp_dict is None:
            pytest.skip(f"No input files in {input_dir}")

        dbz3d  = comp_dict['dbz3d_filt']
        height = comp_dict['height']
        shape_2d = comp_dict['refl'].shape
        gap = 3  # default echotop_gap used in demo configs

        t0 = time.time()
        old = echotop_height(
            dbz3d, height, 'z', shape_2d, dbz_thresh=thresh, gap=gap, min_thick=0)
        t_old = time.time() - t0

        t0 = time.time()
        new = echotop_height_fast(
            dbz3d, height, 'z', shape_2d, dbz_thresh=thresh, gap=gap, min_thick=0)
        t_new = time.time() - t0

        nan_diff = int(np.sum(np.isnan(old) != np.isnan(new)))
        valid = np.isfinite(old) & np.isfinite(new)
        max_diff = float(np.abs(old[valid] - new[valid]).max()) if valid.any() else 0.0

        with capsys.disabled():
            print(
                f"\n[{demo} thresh={thresh}] "
                f"old={t_old:.3f}s  fast={t_new:.3f}s  "
                f"speedup={t_old/max(t_new, 1e-9):.1f}x  "
                f"max_diff={max_diff:.2e}m  nan_mismatch={nan_diff}"
            )

        assert nan_diff == 0, (
            f"[{demo} thresh={thresh}] NaN pattern differs: {nan_diff} pixel(s)"
        )
        assert np.allclose(old, new, equal_nan=True, atol=1.0), (
            f"[{demo} thresh={thresh}] max |old-fast| = {max_diff:.3e} m (threshold 1.0 m)"
        )


# ── TestLabelCellsFast ──────────────────────────────────────────────────────

@pytest.mark.local
class TestLabelCellsFast:
    """
    Verify that label_cells_fast returns exactly the same labeled array and
    pixel-count array as label_cells on both demo datasets.
    """

    @pytest.mark.parametrize("demo,input_dir,pattern,config", [
        ("nexrad", demo_path('cell_radar', 'nexrad', 'input'),
         "KHGX*.nc", _NEXRAD_CONFIG),
        ("csapr",  demo_path('cell_radar', 'csapr', 'input'),
         "taranis_corcsapr2*.nc", _CSAPR_CONFIG),
    ])
    def test_label_cells_identical(self, demo, input_dir, pattern, config, capsys):
        """label_cells_fast output must be array_equal to label_cells output."""
        comp_dict, first_file = _load_demo_3d(input_dir, pattern, config)
        if comp_dict is None:
            pytest.skip(f"No input files in {input_dir}")

        core_dilate, dx, radii_expand = _steiner_result_for_expand(comp_dict)
        convmask = (core_dilate > 0).astype(int)

        t0 = time.time()
        lbl_old, npix_old = label_cells(convmask, min_cellpix=0)
        t_old = time.time() - t0

        t0 = time.time()
        lbl_new, npix_new = label_cells_fast(convmask, min_cellpix=0)
        t_new = time.time() - t0

        with capsys.disabled():
            print(
                f"\n[{demo}] label_cells: old={t_old:.4f}s  fast={t_new:.4f}s  "
                f"speedup={t_old/max(t_new, 1e-9):.1f}x  ncells={len(npix_old)}"
            )

        assert np.array_equal(lbl_old, lbl_new), (
            f"[{demo}] label array differs: {np.sum(lbl_old != lbl_new)} pixel(s)"
        )
        assert np.array_equal(npix_old, npix_new), (
            f"[{demo}] npix array differs"
        )


# ── TestExpandConvCoreFast ──────────────────────────────────────────────────

@pytest.mark.local
class TestExpandConvCoreFast:
    """
    Verify that expand_conv_core_fast returns a bit-for-bit identical result
    to expand_conv_core for both demo datasets.
    """

    @pytest.mark.parametrize("demo,input_dir,pattern,config", [
        ("nexrad", demo_path('cell_radar', 'nexrad', 'input'),
         "KHGX*.nc", _NEXRAD_CONFIG),
        ("csapr",  demo_path('cell_radar', 'csapr', 'input'),
         "taranis_corcsapr2*.nc", _CSAPR_CONFIG),
    ])
    def test_expand_core_identical(self, demo, input_dir, pattern, config, capsys):
        """expand_conv_core_fast output must be array_equal to expand_conv_core."""
        comp_dict, first_file = _load_demo_3d(input_dir, pattern, config)
        if comp_dict is None:
            pytest.skip(f"No input files in {input_dir}")

        core_dilate, dx, radii_expand = _steiner_result_for_expand(comp_dict)

        t0 = time.time()
        exp_old, sorted_old = expand_conv_core(
            core_dilate, radii_expand, dx, dx, min_corenpix=0)
        t_old = time.time() - t0

        t0 = time.time()
        exp_new, sorted_new = expand_conv_core_fast(
            core_dilate, radii_expand, dx, dx, min_corenpix=0)
        t_new = time.time() - t0

        ncores = int(sorted_old.max()) if sorted_old.size else 0
        diff_expand = int(np.sum(exp_old != exp_new))
        diff_sorted = int(np.sum(sorted_old != sorted_new))

        with capsys.disabled():
            print(
                f"\n[{demo}] expand_conv_core: old={t_old:.3f}s  fast={t_new:.3f}s  "
                f"speedup={t_old/max(t_new, 1e-9):.1f}x  "
                f"ncores={ncores}  diff_expand={diff_expand}  diff_sorted={diff_sorted}"
            )

        assert diff_sorted == 0, (
            f"[{demo}] score_sorted differs: {diff_sorted} pixel(s)"
        )
        assert diff_expand == 0, (
            f"[{demo}] score_expand differs: {diff_expand} pixel(s)"
        )


# ---------------------------------------------------------------------------
# EDT validation tests
# ---------------------------------------------------------------------------

from pyflextrkr.steiner_func import (
    mod_dilate_conv_rad,
    mod_dilate_conv_rad_edt,
    expand_conv_core_edt,
)


def _steiner_result_for_dilate(comp_dict):
    """
    Run Steiner up to (but not including) the dilation step.
    Returns (refl_bkg, sclass, score_keep, mask_goodvalues, bkg_bin, conv_rad_bin, dx).
    """
    from scipy import ndimage as _ndimage
    from pyflextrkr.steiner_func import (
        background_intensity, peakedness, make_dilation_step_func,
    )
    dx = 500.0
    bkg_bin, conv_rad_bin = make_dilation_step_func(
        mindBZuse=25, dBZforMaxConvRadius=60,
        bkg_refl_increment=5, conv_rad_increment=0.5,
        conv_rad_start=1.0, maxConvRadius=5,
    )
    types_steiner = {'NO_SURF_ECHO': 1, 'WEAK_ECHO': 2, 'STRATIFORM': 3, 'CONVECTIVE': 4}

    refl = np.array(comp_dict['refl'])
    mask = np.array(comp_dict['mask_goodvalues'])

    refl_bkg = background_intensity(refl, mask, dx, dx, 11000.0, convolve_method='fft')
    peak = peakedness(refl_bkg, mask, minZdiff=10.0, absConvThres=60.0)

    score = np.zeros(refl.shape, dtype=int)
    sclass = np.zeros(refl.shape, dtype=int)
    sclass[mask == 1] = types_steiner['STRATIFORM']
    ind_core = np.logical_or(refl >= 55.0, (refl - refl_bkg) >= peak)
    score[ind_core] = 1
    sclass[ind_core] = types_steiner['CONVECTIVE']
    sclass[np.logical_and(refl > 25.0, refl < 15.0)] = types_steiner['WEAK_ECHO']
    sclass[np.logical_and(mask == 1, refl < 25.0)] = types_steiner['NO_SURF_ECHO']

    # Remove small cores (min_corearea=4 km^2 → 16 pixels at dx=500m)
    min_corenpix = int(4.0 * (1000**2) / (dx * dx))
    score_keep = np.copy(score)
    tmpregions, num_regions = _ndimage.label(score_keep)
    for rr in range(1, num_regions + 1):
        rid = np.where(tmpregions == rr)
        if len(rid[0]) < min_corenpix:
            score_keep[rid] = 0

    return refl_bkg, sclass, score_keep, mask, bkg_bin, conv_rad_bin, types_steiner, dx


# ── TestExpandConvCoreEdt ───────────────────────────────────────────────────

@pytest.mark.local
class TestExpandConvCoreEdt:
    """
    Verify that expand_conv_core_edt agrees with expand_conv_core (orig) and
    expand_conv_core_fast on both demo datasets.

    score_sorted must be identical (same label_cells_fast call).
    score_expand may differ at sub-pixel-wide equidistant boundaries (≤ 0.1%).
    """

    @pytest.mark.parametrize("demo,input_dir,pattern,config", [
        ("nexrad", demo_path('cell_radar', 'nexrad', 'input'),
         "KHGX*.nc", _NEXRAD_CONFIG),
        ("csapr",  demo_path('cell_radar', 'csapr', 'input'),
         "taranis_corcsapr2*.nc", _CSAPR_CONFIG),
    ])
    def test_expand_core_edt_vs_orig(self, demo, input_dir, pattern, config, capsys):
        """expand_conv_core_edt score_expand must agree with orig within 0.1% pixels."""
        comp_dict, _ = _load_demo_3d(input_dir, pattern, config)
        if comp_dict is None:
            pytest.skip(f"No input files in {input_dir}")

        core_dilate, dx, radii_expand = _steiner_result_for_expand(comp_dict)

        t0 = time.time()
        exp_orig, sorted_orig = expand_conv_core(
            core_dilate, radii_expand, dx, dx, min_corenpix=0)
        t_orig = time.time() - t0

        t0 = time.time()
        exp_fast, sorted_fast = expand_conv_core_fast(
            core_dilate, radii_expand, dx, dx, min_corenpix=0)
        t_fast = time.time() - t0

        t0 = time.time()
        exp_edt, sorted_edt = expand_conv_core_edt(
            core_dilate, radii_expand, dx, dx, min_corenpix=0)
        t_edt = time.time() - t0

        total_pix = int(np.sum(sorted_orig > 0))
        diff_sorted_orig  = int(np.sum(sorted_orig  != sorted_edt))
        diff_sorted_fast  = int(np.sum(sorted_fast  != sorted_edt))
        diff_expand_orig  = int(np.sum(exp_orig  != exp_edt))
        diff_expand_fast  = int(np.sum(exp_fast  != exp_edt))
        pct_orig = 100.0 * diff_expand_orig / max(total_pix, 1)
        pct_fast = 100.0 * diff_expand_fast / max(total_pix, 1)

        with capsys.disabled():
            print(
                f"\n[{demo}] expand_conv_core timing:\n"
                f"  orig={t_orig:.3f}s  fast={t_fast:.3f}s  edt={t_edt:.3f}s\n"
                f"  speedup vs orig: {t_orig/max(t_edt,1e-9):.1f}x\n"
                f"  speedup vs fast: {t_fast/max(t_edt,1e-9):.1f}x\n"
                f"  score_sorted diff (edt vs orig): {diff_sorted_orig} px\n"
                f"  score_sorted diff (edt vs fast): {diff_sorted_fast} px\n"
                f"  score_expand diff (edt vs orig): {diff_expand_orig} px ({pct_orig:.4f}%)\n"
                f"  score_expand diff (edt vs fast): {diff_expand_fast} px ({pct_fast:.4f}%)"
            )

        assert diff_sorted_orig == 0, (
            f"[{demo}] score_sorted differs (edt vs orig): {diff_sorted_orig} px"
        )
        assert pct_orig <= 0.1, (
            f"[{demo}] score_expand (edt vs orig): {diff_expand_orig} px = {pct_orig:.4f}% > 0.1%"
        )
        assert pct_fast <= 0.1, (
            f"[{demo}] score_expand (edt vs fast): {diff_expand_fast} px = {pct_fast:.4f}% > 0.1%"
        )


# ── TestModDilateConvRadEdt ─────────────────────────────────────────────────

@pytest.mark.local
class TestModDilateConvRadEdt:
    """
    Verify that mod_dilate_conv_rad_edt agrees with mod_dilate_conv_rad (orig)
    on both demo datasets within ≤ 0.1% pixel differences.
    """

    @pytest.mark.parametrize("demo,input_dir,pattern,config", [
        ("nexrad", demo_path('cell_radar', 'nexrad', 'input'),
         "KHGX*.nc", _NEXRAD_CONFIG),
        ("csapr",  demo_path('cell_radar', 'csapr', 'input'),
         "taranis_corcsapr2*.nc", _CSAPR_CONFIG),
    ])
    def test_mod_dilate_edt_vs_orig(self, demo, input_dir, pattern, config, capsys):
        """mod_dilate_conv_rad_edt sclass/score_dilate must agree with orig within 0.1%."""
        comp_dict, _ = _load_demo_3d(input_dir, pattern, config)
        if comp_dict is None:
            pytest.skip(f"No input files in {input_dir}")

        refl_bkg, sclass, score_keep, mask, bkg_bin, conv_rad_bin, types_steiner, dx = \
            _steiner_result_for_dilate(comp_dict)

        t0 = time.time()
        sclass_orig, sdilate_orig = mod_dilate_conv_rad(
            types_steiner, refl_bkg, sclass, score_keep,
            mask, dx, dx, bkg_bin, conv_rad_bin)
        t_orig = time.time() - t0

        t0 = time.time()
        sclass_edt, sdilate_edt = mod_dilate_conv_rad_edt(
            types_steiner, refl_bkg, sclass, score_keep,
            mask, dx, dx, bkg_bin, conv_rad_bin)
        t_edt = time.time() - t0

        total_valid = int(np.sum(mask == 1))
        diff_sclass  = int(np.sum(sclass_orig  != sclass_edt))
        diff_sdilate = int(np.sum(sdilate_orig != sdilate_edt))
        pct_sclass  = 100.0 * diff_sclass  / max(total_valid, 1)
        pct_sdilate = 100.0 * diff_sdilate / max(total_valid, 1)

        with capsys.disabled():
            print(
                f"\n[{demo}] mod_dilate_conv_rad timing:\n"
                f"  orig={t_orig:.3f}s  edt={t_edt:.3f}s  "
                f"speedup={t_orig/max(t_edt,1e-9):.1f}x\n"
                f"  sclass  diff (edt vs orig): {diff_sclass}  px ({pct_sclass:.4f}%)\n"
                f"  sdilate diff (edt vs orig): {diff_sdilate} px ({pct_sdilate:.4f}%)"
            )

        assert pct_sclass <= 0.1, (
            f"[{demo}] sclass (edt vs orig): {diff_sclass} px = {pct_sclass:.4f}% > 0.1%"
        )
        assert pct_sdilate <= 0.1, (
            f"[{demo}] score_dilate (edt vs orig): {diff_sdilate} px = {pct_sdilate:.4f}% > 0.1%"
        )

