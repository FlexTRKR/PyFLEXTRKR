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

def _find_input_files(input_dir, pattern="*.nc", n=1):
    """
    Return a sorted list of input NetCDF files.

    Parameters
    ----------
    n : int
        Maximum number of files to return.
        0 means all available files.
        Default is 1 (first file only, same as previous behaviour).
    """
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        # Some demos nest files in subdirectories
        files = sorted(glob.glob(os.path.join(input_dir, "**", pattern), recursive=True))
    if n > 0:
        files = files[:n]
    return files


def _find_first_input_file(input_dir, pattern="*.nc"):
    """Return the first (alphabetically sorted) input NetCDF file (legacy helper)."""
    result = _find_input_files(input_dir, pattern, n=1)
    return result[0] if result else None


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
    def comp_dicts(self, nfiles):
        """
        Load the first *nfiles* input files (0 = all) and return a list of
        ``(comp_dict, filepath)`` pairs.  Skips the test class if no data found.
        """
        pairs = _load_demo_3d_multi(
            self.INPUT_DIR, self.INPUT_PATTERN, self.CONFIG, n=nfiles
        )
        if not pairs:
            pytest.skip(
                f"No input files found in {self.INPUT_DIR}\n"
                f"Run: python tests/run_demo_tests.py --demos {self.LABEL} -n 4"
            )
        return pairs

    @pytest.fixture(scope="class")
    def refl_mask(self, comp_dicts):
        """
        Return a list of ``(refl, mask)`` tuples — one entry per loaded file.

        Each ``refl`` is a 2-D float32 array; ``mask`` is a 2-D int array
        (1 = valid, 0 = missing/below-sensitivity).
        """
        result = []
        for comp_dict, _ in comp_dicts:
            dbz_comp = comp_dict['dbz_comp']
            refl = np.array(
                dbz_comp.values if hasattr(dbz_comp, 'values') else dbz_comp,
                dtype=np.float32,
            )
            radar_sens = self.CONFIG.get('radar_sensitivity', 0.0)
            mask = np.ones(refl.shape, dtype=int)
            mask[~np.isfinite(refl)] = 0
            refl[mask == 0] = radar_sens
            refl = refl.squeeze()
            mask = mask.squeeze()
            result.append((refl, mask))
        return result

    # ── Test 1: background reflectivity agreement (all pixels) ─────────────

    def test_background_reflectivity_interior_max_diff(self, refl_mask, capsys):
        """Max |ndimage - fft| at interior pixels must be < 1e-4 dBZ (all files)."""
        dx      = self.PARAMS['dx']
        bkg_rad = self.PARAMS['bkg_rad']
        k       = int(bkg_rad / dx)

        agg_max_interior = 0.0
        for fi, (refl, mask) in enumerate(refl_mask):
            t0 = time.time()
            bkg_ndimage = _bkg(refl, mask, dx, bkg_rad, 'ndimage')
            t_ndimage   = time.time() - t0
            t0 = time.time()
            bkg_fft     = _bkg(refl, mask, dx, bkg_rad, 'fft')
            t_fft       = time.time() - t0

            with capsys.disabled():
                print(
                    f"\n[{self.LABEL} file={fi}] "
                    f"ndimage={t_ndimage:.2f}s  fft={t_fft:.2f}s  "
                    f"speedup={t_ndimage/t_fft:.1f}x"
                )

            max_interior, _ = _report_diff(
                f"{self.LABEL} file={fi}", bkg_ndimage, bkg_fft, k, capsys
            )
            agg_max_interior = max(agg_max_interior, max_interior)

        with capsys.disabled():
            print(
                f"\n[{self.LABEL}] AGGREGATE over {len(refl_mask)} file(s): "
                f"max_interior={agg_max_interior:.3e} dBZ"
            )
        assert agg_max_interior < 1e-4, (
            f"[{self.LABEL}] Interior max diff = {agg_max_interior:.3e} dBZ "
            f"(threshold 1e-4 dBZ, over {len(refl_mask)} file(s))"
        )

    def test_background_reflectivity_edge_max_diff(self, refl_mask, capsys):
        """Max |ndimage - fft| at edge pixels must be < 1e-3 dBZ (all files)."""
        dx      = self.PARAMS['dx']
        bkg_rad = self.PARAMS['bkg_rad']
        k       = int(bkg_rad / dx)

        agg_max_edge = 0.0
        for fi, (refl, mask) in enumerate(refl_mask):
            bkg_ndimage = _bkg(refl, mask, dx, bkg_rad, 'ndimage')
            bkg_fft     = _bkg(refl, mask, dx, bkg_rad, 'fft')

            em   = _edge_mask(refl.shape, k)
            diff = np.abs(bkg_fft - bkg_ndimage)
            valid_edge = em & np.isfinite(diff)
            max_edge = float(diff[valid_edge].max()) if valid_edge.any() else 0.0
            agg_max_edge = max(agg_max_edge, max_edge)

        assert agg_max_edge < 1e-3, (
            f"[{self.LABEL}] Edge max diff = {agg_max_edge:.3e} dBZ "
            f"(threshold 1e-3 dBZ, over {len(refl_mask)} file(s))\n"
            "  If this fails, consider explicit zero-padding in "
            "background_intensity for 'fft' method."
        )

    # ── Test 2: Steiner classification agreement ────────────────────────────

    def test_sclass_identical_ndimage_vs_fft(self, refl_mask, capsys):
        """Full Steiner sclass array must be elementally identical (all files)."""
        total_diff  = 0
        total_valid = 0
        for fi, (refl, mask) in enumerate(refl_mask):
            r_ndimage = _run_steiner(refl, mask, self.PARAMS, 'ndimage')
            r_fft     = _run_steiner(refl, mask, self.PARAMS, 'fft')

            diff_px = int(np.sum(r_ndimage['sclass'] != r_fft['sclass']))
            n_valid = int(np.sum(mask > 0))
            total_diff  += diff_px
            total_valid += n_valid

            with capsys.disabled():
                print(
                    f"\n[{self.LABEL} file={fi}] sclass diffs: "
                    f"{diff_px}/{n_valid} ({100.0*diff_px/max(n_valid,1):.4f}%)"
                )

        with capsys.disabled():
            print(
                f"\n[{self.LABEL}] AGGREGATE: sclass diffs "
                f"{total_diff}/{total_valid} over {len(refl_mask)} file(s)"
            )
        assert total_diff == 0, (
            f"[{self.LABEL}] sclass differs at {total_diff} pixel(s) total "
            f"over {len(refl_mask)} file(s)"
        )

    def test_score_dilate_identical_ndimage_vs_fft(self, refl_mask, capsys):
        """Dilated convective core (score_dilate) must be elementally identical (all files)."""
        total_diff = 0
        for fi, (refl, mask) in enumerate(refl_mask):
            r_ndimage = _run_steiner(refl, mask, self.PARAMS, 'ndimage')
            r_fft     = _run_steiner(refl, mask, self.PARAMS, 'fft')

            diff_px = int(np.sum(r_ndimage['score_dilate'] != r_fft['score_dilate']))
            total_diff += diff_px

            with capsys.disabled():
                print(
                    f"\n[{self.LABEL} file={fi}] score_dilate diffs: {diff_px} px"
                )

        assert total_diff == 0, (
            f"[{self.LABEL}] score_dilate differs at {total_diff} pixel(s) total "
            f"over {len(refl_mask)} file(s)"
        )

    # ── Test 3: Edge-specific pixel comparison ──────────────────────────────

    def test_edge_pixel_background_vs_interior(self, refl_mask, capsys):
        """
        Measure and compare edge vs interior |ndimage - fft| differences across
        all loaded files.  Asserts both stay below their respective thresholds.
        """
        dx      = self.PARAMS['dx']
        bkg_rad = self.PARAMS['bkg_rad']
        k       = int(bkg_rad / dx)

        agg_max_interior = 0.0
        agg_max_edge     = 0.0

        for fi, (refl, mask) in enumerate(refl_mask):
            bkg_ndimage = _bkg(refl, mask, dx, bkg_rad, 'ndimage')
            bkg_fft     = _bkg(refl, mask, dx, bkg_rad, 'fft')

            diff  = np.abs(bkg_fft - bkg_ndimage)
            valid = np.isfinite(diff) & (mask > 0).reshape(diff.shape)
            em    = _edge_mask(refl.shape, k)

            interior_vals = diff[~em & valid]
            edge_vals     = diff[em & valid]

            max_interior = float(interior_vals.max()) if interior_vals.size else 0.0
            max_edge     = float(edge_vals.max())     if edge_vals.size     else 0.0

            edge_diff = np.where(em & valid, diff, np.nan)
            if np.any(np.isfinite(edge_diff)):
                worst_edge_loc = np.unravel_index(np.nanargmax(edge_diff), diff.shape)
                worst_edge_ndimage = float(bkg_ndimage[worst_edge_loc])
                worst_edge_fft     = float(bkg_fft[worst_edge_loc])
            else:
                worst_edge_loc = (-1, -1)
                worst_edge_ndimage = worst_edge_fft = float('nan')

            agg_max_interior = max(agg_max_interior, max_interior)
            agg_max_edge     = max(agg_max_edge, max_edge)

            with capsys.disabled():
                print(
                    f"\n[{self.LABEL} file={fi}] Edge pixel analysis (k={k}px):\n"
                    f"  Interior max diff : {max_interior:.3e} dBZ\n"
                    f"  Edge     max diff : {max_edge:.3e} dBZ\n"
                    f"  Worst edge pixel  : {worst_edge_loc}  "
                    f"ndimage={worst_edge_ndimage:.4f}  fft={worst_edge_fft:.4f} dBZ"
                )

        with capsys.disabled():
            print(
                f"\n[{self.LABEL}] AGGREGATE over {len(refl_mask)} file(s): "
                f"max_interior={agg_max_interior:.3e}  max_edge={agg_max_edge:.3e} dBZ"
            )

        assert agg_max_interior < 1e-4, (
            f"[{self.LABEL}] Interior diff = {agg_max_interior:.3e} dBZ exceeds 1e-4 dBZ"
        )
        assert agg_max_edge < 1e-3, (
            f"[{self.LABEL}] Edge diff = {agg_max_edge:.3e} dBZ exceeds 1e-3 dBZ"
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


def _load_demo_3d_multi(input_dir, pattern, config, n=1):
    """
    Return a list of ``(comp_dict, filepath)`` pairs for the first *n* input files
    (``n=0`` loads all available files).

    Returns an empty list if no files are found.
    """
    files = _find_input_files(input_dir, pattern, n=n)
    result = []
    for f in files:
        try:
            cd = get_composite_reflectivity_generic(f, config)
            result.append((cd, f))
        except Exception as exc:  # pragma: no cover
            import warnings
            warnings.warn(f"Skipping {f}: {exc}")
    return result


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
    def test_echotop_agrees(self, demo, input_dir, pattern, config, thresh, nfiles, capsys):
        """echotop_height_fast == echotop_height within 1 m for every threshold (all files)."""
        pairs = _load_demo_3d_multi(input_dir, pattern, config, n=nfiles)
        if not pairs:
            pytest.skip(f"No input files in {input_dir}")

        gap = 3  # default echotop_gap used in demo configs
        agg_nan_diff = 0
        agg_max_diff = 0.0

        for fi, (comp_dict, fpath) in enumerate(pairs):
            dbz3d    = comp_dict['dbz3d_filt']
            height   = comp_dict['height']
            shape_2d = comp_dict['refl'].shape

            t0 = time.time()
            old = echotop_height(
                dbz3d, height, 'z', shape_2d, dbz_thresh=thresh, gap=gap, min_thick=0)
            t_old = time.time() - t0

            t0 = time.time()
            new = echotop_height_fast(
                dbz3d, height, 'z', shape_2d, dbz_thresh=thresh, gap=gap, min_thick=0)
            t_new = time.time() - t0

            nan_diff = int(np.sum(np.isnan(old) != np.isnan(new)))
            valid    = np.isfinite(old) & np.isfinite(new)
            max_diff = float(np.abs(old[valid] - new[valid]).max()) if valid.any() else 0.0

            agg_nan_diff += nan_diff
            agg_max_diff  = max(agg_max_diff, max_diff)

            with capsys.disabled():
                print(
                    f"\n[{demo} thresh={thresh} file={fi}] "
                    f"old={t_old:.3f}s  fast={t_new:.3f}s  "
                    f"speedup={t_old/max(t_new, 1e-9):.1f}x  "
                    f"max_diff={max_diff:.2e}m  nan_mismatch={nan_diff}"
                )

        with capsys.disabled():
            print(
                f"\n[{demo} thresh={thresh}] AGGREGATE over {len(pairs)} file(s): "
                f"max_diff={agg_max_diff:.2e}m  nan_mismatch={agg_nan_diff}"
            )

        assert agg_nan_diff == 0, (
            f"[{demo} thresh={thresh}] NaN pattern differs: "
            f"{agg_nan_diff} pixel(s) over {len(pairs)} file(s)"
        )
        assert agg_max_diff <= 1.0, (
            f"[{demo} thresh={thresh}] max |old-fast| = {agg_max_diff:.3e} m "
            f"(threshold 1.0 m, over {len(pairs)} file(s))"
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
    def test_label_cells_identical(self, demo, input_dir, pattern, config, nfiles, capsys):
        """label_cells_fast output must be array_equal to label_cells output (all files)."""
        pairs = _load_demo_3d_multi(input_dir, pattern, config, n=nfiles)
        if not pairs:
            pytest.skip(f"No input files in {input_dir}")

        total_diff_lbl  = 0
        total_diff_npix = 0

        for fi, (comp_dict, _) in enumerate(pairs):
            core_dilate, dx, radii_expand = _steiner_result_for_expand(comp_dict)
            convmask = (core_dilate > 0).astype(int)

            t0 = time.time()
            lbl_old, npix_old = label_cells(convmask, min_cellpix=0)
            t_old = time.time() - t0

            t0 = time.time()
            lbl_new, npix_new = label_cells_fast(convmask, min_cellpix=0)
            t_new = time.time() - t0

            diff_lbl  = int(np.sum(lbl_old  != lbl_new))
            diff_npix = int(np.sum(npix_old != npix_new))
            total_diff_lbl  += diff_lbl
            total_diff_npix += diff_npix

            with capsys.disabled():
                print(
                    f"\n[{demo} file={fi}] label_cells: "
                    f"old={t_old:.4f}s  fast={t_new:.4f}s  "
                    f"speedup={t_old/max(t_new,1e-9):.1f}x  "
                    f"ncells={len(npix_old)}  "
                    f"diff_lbl={diff_lbl}  diff_npix={diff_npix}"
                )

        assert total_diff_lbl == 0, (
            f"[{demo}] label array differs: "
            f"{total_diff_lbl} pixel(s) over {len(pairs)} file(s)"
        )
        assert total_diff_npix == 0, (
            f"[{demo}] npix array differs over {len(pairs)} file(s)"
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
    def test_expand_core_identical(self, demo, input_dir, pattern, config, nfiles, capsys):
        """expand_conv_core_fast output must be array_equal to expand_conv_core (all files)."""
        pairs = _load_demo_3d_multi(input_dir, pattern, config, n=nfiles)
        if not pairs:
            pytest.skip(f"No input files in {input_dir}")

        total_diff_expand = 0
        total_diff_sorted = 0

        for fi, (comp_dict, _) in enumerate(pairs):
            core_dilate, dx, radii_expand = _steiner_result_for_expand(comp_dict)

            t0 = time.time()
            exp_old, sorted_old = expand_conv_core(
                core_dilate, radii_expand, dx, dx, min_corenpix=0)
            t_old = time.time() - t0

            t0 = time.time()
            exp_new, sorted_new = expand_conv_core_fast(
                core_dilate, radii_expand, dx, dx, min_corenpix=0)
            t_new = time.time() - t0

            ncores     = int(sorted_old.max()) if sorted_old.size else 0
            diff_expand = int(np.sum(exp_old != exp_new))
            diff_sorted = int(np.sum(sorted_old != sorted_new))
            total_diff_expand += diff_expand
            total_diff_sorted += diff_sorted

            with capsys.disabled():
                print(
                    f"\n[{demo} file={fi}] expand_conv_core: "
                    f"old={t_old:.3f}s  fast={t_new:.3f}s  "
                    f"speedup={t_old/max(t_new,1e-9):.1f}x  "
                    f"ncores={ncores}  diff_expand={diff_expand}  diff_sorted={diff_sorted}"
                )

        assert total_diff_sorted == 0, (
            f"[{demo}] score_sorted differs: "
            f"{total_diff_sorted} pixel(s) over {len(pairs)} file(s)"
        )
        assert total_diff_expand == 0, (
            f"[{demo}] score_expand differs: "
            f"{total_diff_expand} pixel(s) over {len(pairs)} file(s)"
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

def _contested_boundary_mask(exp_orig):
    """
    Return a boolean mask that is True for every covered pixel that is
    8-connected adjacent to a covered pixel belonging to a **different** cell.

    These are the only pixels where expand_conv_core_edt is permitted to assign
    a different label than expand_conv_core: at equidistant Voronoi boundaries
    the two methods apply different tie-breaking rules (nearest-core EDT vs
    largest-core-wins sequential dilation).

    Parameters
    ----------
    exp_orig : np.ndarray of int
        Labeled expansion array from expand_conv_core (0 = background).

    Returns
    -------
    contested : np.ndarray of bool, same shape as exp_orig
    """
    from scipy.ndimage import maximum_filter, minimum_filter
    covered = exp_orig > 0
    # Local maximum label in the 3x3 neighbourhood
    local_max = maximum_filter(exp_orig, size=3, mode='constant', cval=0)
    # Local minimum *nonzero* label: replace background with a sentinel so that
    # minimum_filter returns the smallest real label rather than 0.
    sentinel = int(exp_orig.max()) + 1
    exp_filled = np.where(covered, exp_orig, sentinel)
    local_min_nz = minimum_filter(exp_filled, size=3, mode='constant', cval=sentinel)
    # A covered pixel is contested if the neighbourhood contains a strictly
    # larger label OR a strictly smaller nonzero label.
    contested = covered & ((local_max > exp_orig) | (local_min_nz < exp_orig))
    return contested


@pytest.mark.local
class TestExpandConvCoreEdt:
    """
    Verify that expand_conv_core_edt agrees with expand_conv_core (orig) and
    expand_conv_core_fast on both demo datasets.

    Assertions:
      1. Coverage identical: the set of covered pixels (label > 0) must be
         bit-identical between EDT and orig.  No pixel should be gained or
         lost — only label re-assignment at boundaries is permitted.
      2. score_sorted identical: both methods use the same label_cells_fast
         call, so sorted core indices must match exactly.
      3. All label mismatches at contested boundaries: the only pixels where
         EDT may assign a different label are those 8-connected adjacent to a
         covered pixel belonging to a different cell.  At equidistant Voronoi
         boundaries EDT uses nearest-core assignment while orig uses
         largest-core-wins; both choices are equally valid scientifically.
    """

    @pytest.mark.parametrize("demo,input_dir,pattern,config", [
        ("nexrad", demo_path('cell_radar', 'nexrad', 'input'),
         "KHGX*.nc", _NEXRAD_CONFIG),
        ("csapr",  demo_path('cell_radar', 'csapr', 'input'),
         "taranis_corcsapr2*.nc", _CSAPR_CONFIG),
    ])
    def test_expand_core_edt_vs_orig(self, demo, input_dir, pattern, config, nfiles, capsys):
        """
        Three structural assertions (all files):
          1. Coverage bit-identical   (zero tolerance)
          2. score_sorted identical   (zero tolerance)
          3. All label mismatches at contested multi-cell boundaries
        """
        pairs = _load_demo_3d_multi(input_dir, pattern, config, n=nfiles)
        if not pairs:
            pytest.skip(f"No input files in {input_dir}")

        total_sorted_diff      = 0
        total_cov_diff         = 0
        total_label_mismatch   = 0
        total_outside_boundary = 0

        for fi, (comp_dict, _) in enumerate(pairs):
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

            # Coverage diff: pixels where one method covers but the other does not
            cov_orig = exp_orig > 0
            cov_edt  = exp_edt  > 0
            cov_diff = int(np.sum(cov_orig != cov_edt))

            # Label mismatch: both covered, different cell ID
            label_mismatch_mask = cov_orig & cov_edt & (exp_orig != exp_edt)
            n_label_mismatch    = int(np.sum(label_mismatch_mask))

            # All label mismatches must lie at contested multi-cell boundaries
            if n_label_mismatch > 0:
                contested = _contested_boundary_mask(exp_orig)
                n_outside = int(np.sum(label_mismatch_mask & ~contested))
            else:
                n_outside = 0

            diff_sorted = int(np.sum(sorted_orig != sorted_edt))

            total_sorted_diff      += diff_sorted
            total_cov_diff         += cov_diff
            total_label_mismatch   += n_label_mismatch
            total_outside_boundary += n_outside

            with capsys.disabled():
                print(
                    f"\n[{demo} file={fi}] expand_conv_core timing:\n"
                    f"  orig={t_orig:.3f}s  fast={t_fast:.3f}s  edt={t_edt:.3f}s\n"
                    f"  speedup vs orig: {t_orig/max(t_edt,1e-9):.1f}x  "
                    f"vs fast: {t_fast/max(t_edt,1e-9):.1f}x\n"
                    f"  score_sorted diff         : {diff_sorted} px\n"
                    f"  coverage diff             : {cov_diff} px\n"
                    f"  label mismatch (at bdry)  : {n_label_mismatch} px\n"
                    f"  label mismatch (off bdry) : {n_outside} px"
                )

        with capsys.disabled():
            print(
                f"\n[{demo}] AGGREGATE over {len(pairs)} file(s):\n"
                f"  sorted_diff={total_sorted_diff}  "
                f"cov_diff={total_cov_diff}  "
                f"label_mismatch={total_label_mismatch}  "
                f"outside_boundary={total_outside_boundary}"
            )

        # Assert 1: coverage must be bit-identical
        assert total_cov_diff == 0, (
            f"[{demo}] coverage differs (edt vs orig): "
            f"{total_cov_diff} px over {len(pairs)} file(s).\n"
            "  EDT and orig must expand to the same set of covered pixels."
        )

        # Assert 2: score_sorted labels must be identical
        assert total_sorted_diff == 0, (
            f"[{demo}] score_sorted differs (edt vs orig): "
            f"{total_sorted_diff} px over {len(pairs)} file(s)"
        )

        # Assert 3: all label mismatches must be at contested boundaries.
        # EDT uses nearest-core Voronoi; orig uses largest-core-wins sequential
        # dilation.  At equidistant boundaries between adjacent cells the two
        # tie-breaking rules assign different labels — both are scientifically
        # valid.  Any mismatch *away* from such boundaries indicates a bug.
        assert total_outside_boundary == 0, (
            f"[{demo}] {total_outside_boundary} label-mismatch pixel(s) are NOT "
            f"at contested multi-cell boundaries over {len(pairs)} file(s).\n"
            "  EDT should only differ from orig at equidistant Voronoi boundaries."
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
    def test_mod_dilate_edt_vs_orig(self, demo, input_dir, pattern, config, nfiles, capsys):
        """mod_dilate_conv_rad_edt sclass/score_dilate must agree with orig within 0.1% (all files)."""
        pairs = _load_demo_3d_multi(input_dir, pattern, config, n=nfiles)
        if not pairs:
            pytest.skip(f"No input files in {input_dir}")

        total_diff_sclass  = 0
        total_diff_sdilate = 0
        total_valid        = 0

        for fi, (comp_dict, _) in enumerate(pairs):
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

            n_valid       = int(np.sum(mask == 1))
            diff_sclass   = int(np.sum(sclass_orig  != sclass_edt))
            diff_sdilate  = int(np.sum(sdilate_orig != sdilate_edt))
            total_diff_sclass  += diff_sclass
            total_diff_sdilate += diff_sdilate
            total_valid        += n_valid

            with capsys.disabled():
                print(
                    f"\n[{demo} file={fi}] mod_dilate_conv_rad: "
                    f"orig={t_orig:.3f}s  edt={t_edt:.3f}s  "
                    f"speedup={t_orig/max(t_edt,1e-9):.1f}x\n"
                    f"  sclass  diff: {diff_sclass} px "
                    f"({100.0*diff_sclass/max(n_valid,1):.4f}%)\n"
                    f"  sdilate diff: {diff_sdilate} px "
                    f"({100.0*diff_sdilate/max(n_valid,1):.4f}%)"
                )

        pct_sclass  = 100.0 * total_diff_sclass  / max(total_valid, 1)
        pct_sdilate = 100.0 * total_diff_sdilate / max(total_valid, 1)

        with capsys.disabled():
            print(
                f"\n[{demo}] AGGREGATE over {len(pairs)} file(s): "
                f"sclass_diff={total_diff_sclass} ({pct_sclass:.4f}%)  "
                f"sdilate_diff={total_diff_sdilate} ({pct_sdilate:.4f}%)"
            )

        assert pct_sclass <= 0.1, (
            f"[{demo}] sclass (edt vs orig): "
            f"{total_diff_sclass} px = {pct_sclass:.4f}% > 0.1% "
            f"over {len(pairs)} file(s)"
        )
        assert pct_sdilate <= 0.1, (
            f"[{demo}] score_dilate (edt vs orig): "
            f"{total_diff_sdilate} px = {pct_sdilate:.4f}% > 0.1% "
            f"over {len(pairs)} file(s)"
        )

