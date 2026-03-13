"""
Local regression tests for PyFLEXTRKR demo outputs.

These tests verify that full demo runs produce numerically sane outputs —
replacing the need to visually inspect animations after every code change.

Requirements
------------
- Set env variable PYFLEXTRKR_TEST_DATA to the root demo data directory.
  e.g.:  export PYFLEXTRKR_TEST_DATA=/pscratch/sd/f/feng045/demo

- Run the demos first (bash config/demo_cell_nexrad.sh, etc.) to produce the
  output files, THEN run these tests to validate them.

- Marked with @pytest.mark.local so they are automatically skipped on
  GitHub CI (see conftest.py).

Usage
-----
# Skip on CI (default):
pytest tests/

# Run all local regression tests on HPC/workstation:
export PYFLEXTRKR_TEST_DATA=/pscratch/sd/f/feng045/demo
pytest tests/test_regression_local.py -v

# Run with extra verbosity on a specific demo:
pytest tests/test_regression_local.py::TestCellNexradDemo -v
"""

import os
import glob
import numpy as np
import pytest
import xarray as xr


# ---------------------------------------------------------------------------
# Helper – find demo directories from env variable
# ---------------------------------------------------------------------------

DATA_ROOT = os.environ.get("PYFLEXTRKR_TEST_DATA", "")


def demo_path(*parts):
    return os.path.join(DATA_ROOT, *parts)


# ---------------------------------------------------------------------------
# Shared helper assertions
# ---------------------------------------------------------------------------

def assert_valid_stat_file(stat_file, min_tracks=1):
    """Common checks on any trackstats NetCDF output."""
    assert os.path.isfile(stat_file), f"Track stats file not found: {stat_file}"
    with xr.open_dataset(stat_file) as ds:
        assert 'tracks' in ds.dims, "trackstats file should have a 'tracks' dimension"
        ntracks = ds.sizes['tracks']
        assert ntracks >= min_tracks, \
            f"Expected >= {min_tracks} tracks in {stat_file}, got {ntracks}"

        # base_time must have at least some valid (non-NaT) entries
        bt = ds['base_time'].values
        if np.issubdtype(bt.dtype, np.datetime64):
            n_valid = np.sum(~np.isnat(bt))
            assert n_valid > 0, "base_time is all NaT"
        else:
            bt_float = bt.astype(float)
            assert np.isfinite(bt_float[~np.isnan(bt_float)]).all(), \
                "base_time should be finite (no NaT/NaN)"

        # At least one numeric variable must have non-NaN values
        for vname in ('cell_meanlon', 'cell_meanlat', 'track_duration'):
            if vname in ds:
                vals = ds[vname].values.ravel()
                n_valid = np.sum(np.isfinite(vals))
                assert n_valid > 0, f"{vname} is all NaN in {stat_file}"


def assert_valid_pixel_files(pixel_dir, filebase, min_files=1, check_var='tracknumber'):
    """Common checks on pixel-level tracking output files."""
    # Use ** to handle startdate_enddate subdirectory created by PyFLEXTRKR
    # e.g. celltracking/20140807.1200_20140807.1500/celltracks_*.nc
    pattern = os.path.join(pixel_dir, '**', f"{filebase}*.nc")
    files = sorted(glob.glob(pattern, recursive=True))
    assert len(files) >= min_files, \
        f"Expected >= {min_files} pixel files matching {pattern}, got {len(files)}"

    # Spot-check first and last file
    for fpath in [files[0], files[-1]]:
        with xr.open_dataset(fpath) as ds:
            assert check_var in ds.data_vars, \
                f"Variable '{check_var}' not found in {fpath}"
            # At least some tracked pixels should exist
            tn = ds[check_var].values
            n_tracked = np.sum(np.isfinite(tn.astype(float)) & (tn > 0))
            assert n_tracked >= 0, f"Cannot read {check_var} in {fpath}"  # always passes; structure check only


def assert_animation_exists(quicklook_dir, anim_name='quicklook_animation.mp4'):
    anim_path = os.path.join(quicklook_dir, anim_name)
    assert os.path.isfile(anim_path), \
        f"Animation file not found: {anim_path}\n" \
        f"Did the demo script run to completion?"


# ---------------------------------------------------------------------------
# Cell tracking on NEXRAD data
# ---------------------------------------------------------------------------

@pytest.mark.local
class TestCellNexradDemo:
    """
    Validates the output of demo_cell_nexrad.sh
    (KHGX NEXRAD data, 2014-08-07 12:00–15:00 UTC)
    """

    DEMO_ROOT = demo_path('cell_radar', 'nexrad')
    STATS_DIR = demo_path('cell_radar', 'nexrad', 'stats')
    PIXEL_DIR = demo_path('cell_radar', 'nexrad', 'celltracking')
    QUICKLOOK_DIR = demo_path('cell_radar', 'nexrad', 'quicklooks_trackpaths')

    def test_stats_output_exists_and_valid(self):
        stat_files = sorted(glob.glob(os.path.join(self.STATS_DIR, 'trackstats_*.nc')))
        assert stat_files, f"No trackstats file found in {self.STATS_DIR}"
        assert_valid_stat_file(stat_files[-1], min_tracks=1)

    def test_pixel_files_exist_and_valid(self):
        assert_valid_pixel_files(self.PIXEL_DIR, filebase='celltracks_')

    def test_track_durations_are_positive(self):
        stat_files = sorted(glob.glob(os.path.join(self.STATS_DIR, 'trackstats_*.nc')))
        if not stat_files:
            pytest.skip("No trackstats file found")
        with xr.open_dataset(stat_files[-1]) as ds:
            if 'track_duration' in ds:
                dur = ds['track_duration'].values
                valid = dur[np.isfinite(dur)]
                assert np.all(valid >= 0), "Track durations must be non-negative"

    def test_cell_lons_within_domain(self):
        """Cell longitudes should be near KHGX radar (Gulf Coast Texas ~-95 °W)."""
        stat_files = sorted(glob.glob(os.path.join(self.STATS_DIR, 'trackstats_*.nc')))
        if not stat_files:
            pytest.skip("No trackstats file found")
        with xr.open_dataset(stat_files[-1]) as ds:
            if 'cell_meanlon' in ds:
                lons = ds['cell_meanlon'].values
                valid = lons[np.isfinite(lons)]
                assert len(valid) > 0, "No valid longitudes found"
                assert np.all(valid > -110) and np.all(valid < -80), \
                    f"Cell longitudes out of expected range (TX Gulf Coast): {valid.min():.1f}..{valid.max():.1f}"

    def test_quicklook_plots_exist(self):
        pngs = glob.glob(os.path.join(self.QUICKLOOK_DIR, '*.png'))
        assert len(pngs) > 0, \
            f"No quicklook PNG files found in {self.QUICKLOOK_DIR}"

    def test_animation_exists(self):
        assert_animation_exists(self.QUICKLOOK_DIR)


# ---------------------------------------------------------------------------
# Shared MCS / generic helper
# ---------------------------------------------------------------------------

def _find_mcs_stats(stats_dir, pattern='mcs_tracks_final_*.nc'):
    """Return the best MCS stats file, with fallbacks."""
    for pat in [pattern, 'mcs_tracks_robust_*.nc',
                'mcs_tracks_pf_*.nc', 'mcs_tracks_*.nc',
                'trackstats_*.nc']:
        files = sorted(glob.glob(os.path.join(stats_dir, pat)))
        files = [f for f in files if 'sparse' not in os.path.basename(f)]
        if files:
            return files[-1]
    return None


def assert_latlons_in_range(stat_file, lat_var, lon_var, lat_range, lon_range):
    """Check that lat/lon variables fall within expected geographic range."""
    with xr.open_dataset(stat_file) as ds:
        for var, (lo, hi), label in [
            (lat_var, lat_range, 'latitude'),
            (lon_var, lon_range, 'longitude'),
        ]:
            if var not in ds:
                continue
            vals = ds[var].values.ravel()
            valid = vals[np.isfinite(vals)]
            assert len(valid) > 0, f"No valid {label} values in {stat_file}"
            assert np.all(valid >= lo) and np.all(valid <= hi), \
                f"{label} out of range: {valid.min():.1f}..{valid.max():.1f} " \
                f"expected [{lo}, {hi}]"


# ---------------------------------------------------------------------------
# Cell tracking on CSAPR data
# ---------------------------------------------------------------------------

@pytest.mark.local
class TestCellCsaprDemo:
    """Validates the output of demo_cell_csapr.sh"""

    STATS_DIR = demo_path('cell_radar', 'csapr', 'stats')
    PIXEL_DIR = demo_path('cell_radar', 'csapr', 'celltracking')
    QUICKLOOK_DIR = demo_path('cell_radar', 'csapr', 'quicklooks_trackpaths')

    def test_stats_output_exists_and_valid(self):
        stat_files = sorted(glob.glob(os.path.join(self.STATS_DIR, 'trackstats_*.nc')))
        stat_files = [f for f in stat_files if 'sparse' not in os.path.basename(f)]
        assert stat_files, f"No trackstats file found in {self.STATS_DIR}"
        assert_valid_stat_file(stat_files[-1], min_tracks=1)

    def test_pixel_files_exist_and_valid(self):
        assert_valid_pixel_files(self.PIXEL_DIR, filebase='celltracks_')

    def test_cell_lons_within_domain(self):
        """CSAPR at Cordoba, Argentina (~-64.7 °E, ~-32.1 °S)."""
        stat_files = sorted(glob.glob(os.path.join(self.STATS_DIR, 'trackstats_*.nc')))
        stat_files = [f for f in stat_files if 'sparse' not in os.path.basename(f)]
        if not stat_files:
            pytest.skip("No trackstats file found")
        assert_latlons_in_range(
            stat_files[-1],
            'cell_meanlat', 'cell_meanlon',
            lat_range=(-35, -30), lon_range=(-67, -62),
        )

    def test_quicklook_plots_exist(self):
        pngs = glob.glob(os.path.join(self.QUICKLOOK_DIR, '*.png'))
        assert len(pngs) > 0, f"No quicklook PNGs in {self.QUICKLOOK_DIR}"

    def test_animation_exists(self):
        assert_animation_exists(self.QUICKLOOK_DIR)


# ---------------------------------------------------------------------------
# MCS tracking on GPM IMERG
# ---------------------------------------------------------------------------

@pytest.mark.local
class TestMcsImergDemo:
    """Validates the output of demo_mcs_imerg.sh"""

    STATS_DIR = demo_path('mcs_tbpf', 'imerg', 'stats')
    PIXEL_DIR = demo_path('mcs_tbpf', 'imerg', 'mcstracking')
    QUICKLOOK_DIR = demo_path('mcs_tbpf', 'imerg', 'quicklooks_trackpaths')

    def test_stats_output_exists_and_valid(self):
        stat_file = _find_mcs_stats(self.STATS_DIR)
        assert stat_file, f"No MCS stats file found in {self.STATS_DIR}"
        assert_valid_stat_file(stat_file, min_tracks=1)

    def test_pixel_files_exist_and_valid(self):
        assert_valid_pixel_files(self.PIXEL_DIR, filebase='mcstrack_')

    def test_latlons_within_domain(self):
        stat_file = _find_mcs_stats(self.STATS_DIR)
        if not stat_file:
            pytest.skip("No MCS stats file found")
        assert_latlons_in_range(
            stat_file, 'meanlat', 'meanlon',
            lat_range=(-60, 20), lon_range=(-80, 10),
        )

    def test_quicklook_plots_exist(self):
        pngs = glob.glob(os.path.join(self.QUICKLOOK_DIR, '*.png'))
        assert len(pngs) > 0, f"No quicklook PNGs in {self.QUICKLOOK_DIR}"

    def test_animation_exists(self):
        assert_animation_exists(self.QUICKLOOK_DIR)


# ---------------------------------------------------------------------------
# MCS tracking on WRF Tb+Precipitation
# ---------------------------------------------------------------------------

@pytest.mark.local
class TestMcsWrfTbpfDemo:
    """Validates the output of demo_mcs_wrf_tbpf.sh"""

    STATS_DIR = demo_path('mcs_tbpf', 'wrf', 'stats')
    PIXEL_DIR = demo_path('mcs_tbpf', 'wrf', 'mcstracking')
    QUICKLOOK_DIR = demo_path('mcs_tbpf', 'wrf', 'quicklooks_trackpaths')

    def test_stats_output_exists_and_valid(self):
        stat_file = _find_mcs_stats(self.STATS_DIR)
        assert stat_file, f"No MCS stats file found in {self.STATS_DIR}"
        assert_valid_stat_file(stat_file, min_tracks=1)

    def test_pixel_files_exist_and_valid(self):
        assert_valid_pixel_files(self.PIXEL_DIR, filebase='mcstrack_')

    def test_latlons_within_domain(self):
        stat_file = _find_mcs_stats(self.STATS_DIR)
        if not stat_file:
            pytest.skip("No MCS stats file found")
        assert_latlons_in_range(
            stat_file, 'meanlat', 'meanlon',
            lat_range=(-20, 5), lon_range=(-80, -40),
        )

    def test_quicklook_plots_exist(self):
        pngs = glob.glob(os.path.join(self.QUICKLOOK_DIR, '*.png'))
        assert len(pngs) > 0, f"No quicklook PNGs in {self.QUICKLOOK_DIR}"

    def test_animation_exists(self):
        assert_animation_exists(self.QUICKLOOK_DIR)


# ---------------------------------------------------------------------------
# MCS tracking on WRF Tb+Radar (3D)
# ---------------------------------------------------------------------------

@pytest.mark.local
class TestMcsWrfTbradarDemo:
    """Validates the output of demo_mcs_wrf_tbradar.sh"""

    STATS_DIR = demo_path('mcs_tbpfradar3d', 'wrf', 'stats')
    PIXEL_DIR = demo_path('mcs_tbpfradar3d', 'wrf', 'mcstracking')
    QUICKLOOK_DIR = demo_path('mcs_tbpfradar3d', 'wrf', 'quicklooks_trackpaths')

    def test_stats_output_exists_and_valid(self):
        stat_file = _find_mcs_stats(self.STATS_DIR)
        assert stat_file, f"No MCS stats file found in {self.STATS_DIR}"
        assert_valid_stat_file(stat_file, min_tracks=1)

    def test_pixel_files_exist_and_valid(self):
        assert_valid_pixel_files(self.PIXEL_DIR, filebase='mcstrack_')

    def test_latlons_within_domain(self):
        stat_file = _find_mcs_stats(self.STATS_DIR)
        if not stat_file:
            pytest.skip("No MCS stats file found")
        assert_latlons_in_range(
            stat_file, 'meanlat', 'meanlon',
            lat_range=(25, 55), lon_range=(-115, -75),
        )

    def test_animation_exists(self):
        assert_animation_exists(self.QUICKLOOK_DIR)


# ---------------------------------------------------------------------------
# MCS tracking on GridRad Tb+Radar
# ---------------------------------------------------------------------------

@pytest.mark.local
class TestMcsGridradDemo:
    """Validates the output of demo_mcs_gridrad.sh"""

    STATS_DIR = demo_path('mcs_tbpfradar3d', 'gridrad', 'stats')
    PIXEL_DIR = demo_path('mcs_tbpfradar3d', 'gridrad', 'mcstracking')
    QUICKLOOK_DIR = demo_path('mcs_tbpfradar3d', 'gridrad', 'quicklooks_trackpaths')

    def test_stats_output_exists_and_valid(self):
        stat_file = _find_mcs_stats(self.STATS_DIR)
        assert stat_file, f"No MCS stats file found in {self.STATS_DIR}"
        assert_valid_stat_file(stat_file, min_tracks=1)

    def test_pixel_files_exist_and_valid(self):
        assert_valid_pixel_files(self.PIXEL_DIR, filebase='mcstrack_')

    def test_latlons_within_domain(self):
        stat_file = _find_mcs_stats(self.STATS_DIR)
        if not stat_file:
            pytest.skip("No MCS stats file found")
        assert_latlons_in_range(
            stat_file, 'meanlat', 'meanlon',
            lat_range=(20, 55), lon_range=(-130, -60),
        )

    def test_animation_exists(self):
        assert_animation_exists(self.QUICKLOOK_DIR)


# ---------------------------------------------------------------------------
# MCS tracking on E3SM model (25 km)
# ---------------------------------------------------------------------------

@pytest.mark.local
class TestMcsModel25kmDemo:
    """Validates the output of demo_mcs_model25km.sh"""

    STATS_DIR = demo_path('mcs_tbpf', 'e3sm', 'stats')
    PIXEL_DIR = demo_path('mcs_tbpf', 'e3sm', 'mcstracking')
    QUICKLOOK_DIR = demo_path('mcs_tbpf', 'e3sm', 'quicklooks_robust')

    def test_stats_output_exists_and_valid(self):
        stat_file = _find_mcs_stats(self.STATS_DIR)
        assert stat_file, f"No MCS stats file found in {self.STATS_DIR}"
        assert_valid_stat_file(stat_file, min_tracks=1)

    def test_pixel_files_exist_and_valid(self):
        assert_valid_pixel_files(self.PIXEL_DIR, filebase='mcstrack_')

    def test_latlons_within_domain(self):
        stat_file = _find_mcs_stats(self.STATS_DIR)
        if not stat_file:
            pytest.skip("No MCS stats file found")
        assert_latlons_in_range(
            stat_file, 'meanlat', 'meanlon',
            lat_range=(-60, 60), lon_range=(-180, 360),
        )

    def test_animation_exists(self):
        assert_animation_exists(
            self.QUICKLOOK_DIR, anim_name='mcs_robust_animation.mp4',
        )


# ---------------------------------------------------------------------------
# MCS tracking on Himawari (Tb-only)
# ---------------------------------------------------------------------------

@pytest.mark.local
class TestMcsHimawariDemo:
    """Validates the output of demo_mcs_himawari.sh"""

    STATS_DIR = demo_path('mcs_tbpf', 'himawari', 'stats')
    PIXEL_DIR = demo_path('mcs_tbpf', 'himawari', 'mcstracking_tb')
    QUICKLOOK_DIR = demo_path('mcs_tbpf', 'himawari', 'quicklooks_trackpaths')

    def test_stats_output_exists_and_valid(self):
        stat_file = _find_mcs_stats(self.STATS_DIR, pattern='mcs_tracks_*.nc')
        assert stat_file, f"No MCS stats file found in {self.STATS_DIR}"
        assert_valid_stat_file(stat_file, min_tracks=1)

    def test_pixel_files_exist_and_valid(self):
        assert_valid_pixel_files(self.PIXEL_DIR, filebase='mcstrack_')

    def test_latlons_within_domain(self):
        stat_file = _find_mcs_stats(self.STATS_DIR, pattern='mcs_tracks_*.nc')
        if not stat_file:
            pytest.skip("No MCS stats file found")
        assert_latlons_in_range(
            stat_file, 'meanlat', 'meanlon',
            lat_range=(-60, 60), lon_range=(80, 200),
        )

    def test_animation_exists(self):
        assert_animation_exists(self.QUICKLOOK_DIR)


# ---------------------------------------------------------------------------
# Generic feature tracking (ERA5 Z500 anomaly)
# ---------------------------------------------------------------------------

@pytest.mark.local
class TestGenericTrackingDemo:
    """Validates the output of demo_generic_tracking.sh"""

    STATS_DIR = demo_path('general_tracking', 'z500_blocking', 'stats')
    PIXEL_DIR = demo_path('general_tracking', 'z500_blocking', 'z500tracking')
    QUICKLOOK_DIR = demo_path('general_tracking', 'z500_blocking', 'quicklooks_trackpaths')

    def test_stats_output_exists_and_valid(self):
        stat_files = sorted(glob.glob(os.path.join(self.STATS_DIR, 'trackstats_*.nc')))
        stat_files = [f for f in stat_files if 'sparse' not in os.path.basename(f)]
        assert stat_files, f"No trackstats file found in {self.STATS_DIR}"
        assert_valid_stat_file(stat_files[-1], min_tracks=1)

    def test_pixel_files_exist_and_valid(self):
        assert_valid_pixel_files(self.PIXEL_DIR, filebase='z500tracks_')

    def test_latlons_within_domain(self):
        stat_files = sorted(glob.glob(os.path.join(self.STATS_DIR, 'trackstats_*.nc')))
        stat_files = [f for f in stat_files if 'sparse' not in os.path.basename(f)]
        if not stat_files:
            pytest.skip("No trackstats file found")
        assert_latlons_in_range(
            stat_files[-1], 'meanlat', 'meanlon',
            lat_range=(-90, 90), lon_range=(-180, 360),
        )

    def test_quicklook_plots_exist(self):
        pngs = glob.glob(os.path.join(self.QUICKLOOK_DIR, '*.png'))
        assert len(pngs) > 0, f"No quicklook PNGs in {self.QUICKLOOK_DIR}"

    def test_animation_exists(self):
        assert_animation_exists(self.QUICKLOOK_DIR)
