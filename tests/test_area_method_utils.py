"""
Test Phase 2: precompute_grid_area, sort_renumber2vars with grid_area,
and pixel_area loading in idclouds_tbpf.
"""
import os
import sys
import tempfile
import shutil
import numpy as np

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyflextrkr.ft_utilities import (
    compute_grid_area,
    save_grid_area,
    load_grid_area,
    get_pixel_area,
    precompute_grid_area,
)
from pyflextrkr.ftfunctions import sort_renumber, sort_renumber2vars

def test_sort_renumber2vars_with_grid_area():
    """Test sort_renumber2vars with grid_area parameter."""
    print("=" * 60)
    print("Test: sort_renumber2vars with grid_area")
    print("=" * 60)

    # Create a simple labeled field (5x5)
    label2d = np.zeros((5, 5), dtype=int)
    label2d[0:2, 0:2] = 1  # 4 pixels
    label2d[2:4, 2:5] = 2  # 6 pixels
    label2d[4, 0:3] = 3    # 3 pixels

    label2d_var2 = np.copy(label2d)

    # Create a grid_area array (varying cell sizes)
    # Higher latitude = smaller cells
    grid_area = np.ones((5, 5)) * 100.0  # 100 km^2 per pixel
    grid_area[0, :] = 80.0   # smaller at "high lat"
    grid_area[4, :] = 120.0  # larger at "low lat"

    # Test 1: Without grid_area (legacy behavior)
    # min_cellpix = 4 means cells with <= 4 pixels are removed
    out1, out2, npix1 = sort_renumber2vars(label2d.copy(), label2d_var2.copy(), 4)
    ncells_legacy = np.nanmax(out1)
    print(f"  Legacy (min_cellpix=4): ncells={ncells_legacy}, npix={npix1}")
    assert ncells_legacy == 1, f"Expected 1 cell (only cell2 has >4 pix), got {ncells_legacy}"
    print("  PASS: Legacy sort_renumber2vars works correctly")

    # Test 2: With grid_area, area threshold = 500 km^2
    # Cell 1: 4 pix * ~80-100 km^2 = 360 km^2 (should be removed)
    # Cell 2: 6 pix * 100 km^2 = 600 km^2 (should pass)
    # Cell 3: 3 pix * 120 km^2 = 360 km^2 (should be removed)
    out1_ga, out2_ga, npix1_ga = sort_renumber2vars(
        label2d.copy(), label2d_var2.copy(), 500, grid_area=grid_area
    )
    ncells_ga = np.nanmax(out1_ga)
    print(f"  grid_area (area_thresh=500): ncells={ncells_ga}, npix={npix1_ga}")
    assert ncells_ga == 1, f"Expected 1 cell (only cell2 area>500), got {ncells_ga}"
    print("  PASS: sort_renumber2vars with grid_area works correctly")

    # Test 3: With grid_area, lower threshold = 300 km^2
    # Cell 1: area = 2*80 + 2*100 = 360 km^2 (should pass)
    # Cell 2: area = 6*100 = 600 km^2 (should pass)
    # Cell 3: area = 3*120 = 360 km^2 (should pass)
    out1_ga2, out2_ga2, npix1_ga2 = sort_renumber2vars(
        label2d.copy(), label2d_var2.copy(), 300, grid_area=grid_area
    )
    ncells_ga2 = np.nanmax(out1_ga2)
    print(f"  grid_area (area_thresh=300): ncells={ncells_ga2}, npix={npix1_ga2}")
    assert ncells_ga2 == 3, f"Expected 3 cells (all area>300), got {ncells_ga2}"
    print("  PASS: All cells pass with lower threshold")

    # Test 4: Verify both output variables are renumbered consistently
    assert np.array_equal(out1_ga2 > 0, out2_ga2 > 0), "var1 and var2 should have same nonzero mask"
    print("  PASS: Both variables renumbered consistently")
    print()


def test_precompute_grid_area():
    """Test precompute_grid_area function."""
    print("=" * 60)
    print("Test: precompute_grid_area")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp()
    try:
        # Create a mock NetCDF input file with lat/lon
        import xarray as xr
        lat = np.arange(30, 35, 0.5)  # 10 lat points
        lon = np.arange(-100, -95, 0.5)  # 10 lon points
        lon2d, lat2d = np.meshgrid(lon, lat)
        ds = xr.Dataset(
            {"tb": (["lat", "lon"], np.random.rand(len(lat), len(lon)) * 50 + 200)},
            coords={"lat": lat, "lon": lon},
        )
        input_file = os.path.join(tmpdir, "test_input.nc")
        ds.to_netcdf(input_file)

        grid_area_file = os.path.join(tmpdir, "grid_area_from_latlon.nc")

        # Config: fixed mode (should do nothing)
        config_fixed = {
            "pixel_radius": 10.0,
            "area_method": "fixed",
            "grid_area_file": grid_area_file,
            "input_format": "netcdf",
            "x_coordname": "lon",
            "y_coordname": "lat",
            "geolimits": [30, -100, 35, -95],
        }
        precompute_grid_area(config_fixed, first_file=input_file)
        assert not os.path.isfile(grid_area_file), "File should not be created for fixed mode"
        print("  PASS: precompute_grid_area does nothing for fixed mode")

        # Config: latlon mode
        config_latlon = {
            "pixel_radius": 10.0,
            "area_method": "latlon",
            "grid_area_file": grid_area_file,
            "input_format": "netcdf",
            "x_coordname": "lon",
            "y_coordname": "lat",
            "geolimits": [30, -100, 35, -95],
            "stats_outpath": tmpdir,
        }
        precompute_grid_area(config_latlon, first_file=input_file)
        assert os.path.isfile(grid_area_file), "grid_area_file should be created"
        print("  PASS: precompute_grid_area creates grid_area file for latlon mode")

        # Load and verify
        grid_area = load_grid_area(grid_area_file)
        print(f"  Grid area shape: {grid_area.shape}")
        print(f"  Grid area range: {grid_area.min():.1f} - {grid_area.max():.1f} km^2")
        assert grid_area.shape == (len(lat), len(lon)), f"Shape mismatch: {grid_area.shape}"
        print("  PASS: Grid area shape matches input domain")

        # Verify loading from existing file (should not recompute)
        mtime_before = os.path.getmtime(grid_area_file)
        precompute_grid_area(config_latlon, first_file=input_file)
        mtime_after = os.path.getmtime(grid_area_file)
        assert mtime_before == mtime_after, "File should not be rewritten if it exists"
        print("  PASS: precompute_grid_area skips if file already exists")

        # Test geolimit subsetting
        os.remove(grid_area_file)
        config_subset = config_latlon.copy()
        config_subset["geolimits"] = [31, -99, 34, -96]  # Smaller domain
        precompute_grid_area(config_subset, first_file=input_file)
        grid_area_sub = load_grid_area(grid_area_file)
        print(f"  Subsetted grid area shape: {grid_area_sub.shape}")
        assert grid_area_sub.shape[0] < len(lat), "Subsetted domain should be smaller"
        assert grid_area_sub.shape[1] < len(lon), "Subsetted domain should be smaller"
        print("  PASS: precompute_grid_area applies geolimit subsetting")

    finally:
        shutil.rmtree(tmpdir)
    print()


def test_get_pixel_area_integration():
    """Test get_pixel_area loading from pre-computed file."""
    print("=" * 60)
    print("Test: get_pixel_area integration with precomputed file")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp()
    try:
        import xarray as xr

        # Create mock input and precompute
        lat = np.arange(25, 30, 1.0)
        lon = np.arange(-90, -85, 1.0)
        ds = xr.Dataset(
            {"tb": (["lat", "lon"], np.random.rand(len(lat), len(lon)))},
            coords={"lat": lat, "lon": lon},
        )
        input_file = os.path.join(tmpdir, "input.nc")
        ds.to_netcdf(input_file)

        grid_area_file = os.path.join(tmpdir, "grid_area_from_latlon.nc")

        config = {
            "pixel_radius": 10.0,
            "area_method": "latlon",
            "grid_area_file": grid_area_file,
            "input_format": "netcdf",
            "x_coordname": "lon",
            "y_coordname": "lat",
            "geolimits": [25, -90, 30, -85],
            "stats_outpath": tmpdir,
        }

        # 1. Precompute
        precompute_grid_area(config, first_file=input_file)
        assert os.path.isfile(grid_area_file)

        # 2. Load via get_pixel_area (no lat/lon needed since file exists)
        pixel_area = get_pixel_area(config)
        assert isinstance(pixel_area, np.ndarray), "Should return 2D array for latlon"
        assert pixel_area.ndim == 2, f"Expected 2D, got {pixel_area.ndim}D"
        print(f"  pixel_area shape: {pixel_area.shape}")
        print(f"  pixel_area range: {pixel_area.min():.1f} - {pixel_area.max():.1f} km^2")
        print("  PASS: get_pixel_area loads from pre-computed file")

        # 3. Compare with fixed mode
        config_fixed = config.copy()
        config_fixed["area_method"] = "fixed"
        pixel_area_fixed = get_pixel_area(config_fixed)
        assert np.isscalar(pixel_area_fixed) or pixel_area_fixed.ndim == 0
        assert float(pixel_area_fixed) == 100.0  # 10.0 ** 2
        print(f"  pixel_area_fixed: {pixel_area_fixed} km^2")
        print("  PASS: get_pixel_area returns scalar for fixed mode")

    finally:
        shutil.rmtree(tmpdir)
    print()


def test_sort_renumber_with_grid_area():
    """Verify sort_renumber (already supports grid_area) works correctly."""
    print("=" * 60)
    print("Test: sort_renumber with grid_area (existing support)")
    print("=" * 60)

    label2d = np.zeros((4, 4), dtype=int)
    label2d[0, 0:2] = 1  # 2 pixels (high latitude, small area)
    label2d[2:4, 1:4] = 2  # 6 pixels (low latitude, large area)

    grid_area = np.ones((4, 4)) * 100.0
    grid_area[0, :] = 50.0  # smaller cells at high lat

    # Cell 1: area = 2 * 50 = 100 km^2
    # Cell 2: area = 6 * 100 = 600 km^2

    # With area threshold = 200 km^2, only cell 2 should survive
    out, npix = sort_renumber(label2d.copy(), 200, grid_area=grid_area)
    assert np.nanmax(out) == 1, "Only 1 cell should survive"
    assert npix[0] == 6, "Surviving cell should have 6 pixels"
    print("  PASS: sort_renumber with grid_area filters correctly")

    # Without grid_area, min_size=3 should keep only cell 2
    out2, npix2 = sort_renumber(label2d.copy(), 3)
    assert np.nanmax(out2) == 1, "Only 1 cell should survive (npix>3)"
    print("  PASS: sort_renumber without grid_area filters correctly")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Phase 2 Tests")
    print("=" * 60 + "\n")

    test_sort_renumber_with_grid_area()
    test_sort_renumber2vars_with_grid_area()
    test_precompute_grid_area()
    test_get_pixel_area_integration()

    print("=" * 60)
    print("All Phase 2 tests PASSED!")
    print("=" * 60)
