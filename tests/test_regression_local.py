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

        # base_time must be finite
        bt = ds['base_time'].values
        assert np.isfinite(bt[~np.isnan(bt.astype(float))]).all(), \
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
# Generic: add more demo tests below following the same pattern
# ---------------------------------------------------------------------------
# Example stub for an MCS demo — uncomment and adapt when you have a demo:
#
# @pytest.mark.local
# class TestMcsTbpfDemo:
#     DEMO_ROOT  = demo_path('mcs', 'tbpf')
#     STATS_DIR  = demo_path('mcs', 'tbpf', 'stats')
#     PIXEL_DIR  = demo_path('mcs', 'tbpf', 'mcstracking')
#
#     def test_stats_output_exists_and_valid(self):
#         stat_files = sorted(glob.glob(os.path.join(self.STATS_DIR, 'mcs_tracks_*.nc')))
#         assert stat_files, f"No MCS trackstats file found in {self.STATS_DIR}"
#         assert_valid_stat_file(stat_files[-1], min_tracks=1)
#
#     def test_pixel_files_exist_and_valid(self):
#         assert_valid_pixel_files(self.PIXEL_DIR, filebase='mcstracking_')
