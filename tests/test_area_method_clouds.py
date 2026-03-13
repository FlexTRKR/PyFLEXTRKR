"""
Test Phase 3: label_and_grow_cold_clouds with pixel_area (grid_area) support.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyflextrkr.label_and_grow_cold_clouds import label_and_grow_cold_clouds


def make_ir_field(ny=50, nx=50):
    """Create a synthetic IR Tb field with two cold features."""
    # Background warm Tb
    ir = np.full((ny, nx), 280.0)
    # Feature 1: near row 10 (high lat, small grid cells)
    ir[8:14, 8:14] = 220.0  # core (6x6 = 36 pixels)
    ir[6:16, 6:16] = np.where(ir[6:16, 6:16] < 240, ir[6:16, 6:16], 235.0)  # cold anvil
    # Feature 2: near row 40 (low lat, large grid cells)
    ir[38:44, 38:44] = 218.0  # core (6x6 = 36 pixels)
    ir[36:46, 36:46] = np.where(ir[36:46, 36:46] < 240, ir[36:46, 36:46], 233.0)  # cold anvil
    return ir


def test_fixed_mode():
    """Test backward compatibility: fixed pixel_area (no pixel_area arg)."""
    print("=" * 60)
    print("Test: label_and_grow_cold_clouds (fixed mode, no pixel_area)")
    print("=" * 60)

    ir = make_ir_field()
    pixel_radius = 10.0  # 10 km
    tb_threshs = [225.0, 241.0, 261.0, 261.0]
    area_thresh = 800.0  # km^2
    config = {"pbc_direction": "none"}

    result = label_and_grow_cold_clouds(
        ir, pixel_radius, tb_threshs, area_thresh,
        mincoldcorepix=4, smoothsize=5, warmanvilexpansion=0,
        config=config,
    )
    nclouds = result["final_nclouds"]
    print(f"  nclouds (fixed, no pixel_area): {nclouds}")
    assert nclouds >= 2, f"Expected at least 2 clouds, got {nclouds}"
    print("  PASS")
    print()


def test_fixed_mode_with_scalar():
    """Test with explicit scalar pixel_area."""
    print("=" * 60)
    print("Test: label_and_grow_cold_clouds (fixed mode, scalar pixel_area)")
    print("=" * 60)

    ir = make_ir_field()
    pixel_radius = 10.0
    tb_threshs = [225.0, 241.0, 261.0, 261.0]
    area_thresh = 800.0
    config = {"pbc_direction": "none"}

    result = label_and_grow_cold_clouds(
        ir, pixel_radius, tb_threshs, area_thresh,
        mincoldcorepix=4, smoothsize=5, warmanvilexpansion=0,
        config=config, pixel_area=pixel_radius**2,
    )
    nclouds = result["final_nclouds"]
    print(f"  nclouds (scalar pixel_area): {nclouds}")
    assert nclouds >= 2, f"Expected at least 2 clouds, got {nclouds}"
    print("  PASS")
    print()


def make_ir_field_mixed(ny=50, nx=50):
    """Create a synthetic IR Tb field with one cored feature + one isolated cold anvil."""
    ir = np.full((ny, nx), 280.0)
    # Feature 1: has cold core (rows 8-14, high lat)
    ir[8:14, 8:14] = 220.0  # core
    ir[6:16, 6:16] = np.where(ir[6:16, 6:16] < 240, ir[6:16, 6:16], 235.0)
    # Feature 2: isolated cold anvil only (no core), at low lat rows 38-46
    ir[38:46, 38:46] = 235.0  # cold anvil only, no core pixels below 225K
    return ir


def test_latlon_mode():
    """Test with 2D pixel_area (latlon mode)."""
    print("=" * 60)
    print("Test: label_and_grow_cold_clouds (latlon mode, 2D pixel_area)")
    print("=" * 60)

    ny, nx = 50, 50
    ir = make_ir_field(ny, nx)
    pixel_radius = 10.0
    tb_threshs = [225.0, 241.0, 261.0, 261.0]
    config = {"pbc_direction": "none"}

    # Create a 2D grid_area
    grid_area = np.ones((ny, nx)) * 100.0
    for j in range(16):
        grid_area[j, :] = 20.0  # small cells at high lat

    area_thresh = 800.0
    result = label_and_grow_cold_clouds(
        ir, pixel_radius, tb_threshs, area_thresh,
        mincoldcorepix=4, smoothsize=5, warmanvilexpansion=0,
        config=config, pixel_area=grid_area,
    )
    nclouds = result["final_nclouds"]
    print(f"  nclouds (2D pixel_area, thresh=800): {nclouds}")
    assert nclouds >= 2, f"Expected at least 2 clouds, got {nclouds}"
    print("  PASS: Both features pass area threshold")

    # Test isolated cold anvil filtering with area threshold
    # Use a field with one cored feature + one isolated cold anvil
    ir2 = make_ir_field_mixed(ny, nx)
    # grid_area: high-lat rows have 20 km^2/pixel, low-lat rows have 100 km^2/pixel
    # Isolated cold anvil at rows 38-46: 8*8=64 pixels * 100 km^2 = 6400 km^2
    # With area_thresh=800, it should pass
    result_low = label_and_grow_cold_clouds(
        ir2, pixel_radius, tb_threshs, 800.0,
        mincoldcorepix=4, smoothsize=5, warmanvilexpansion=0,
        config=config, pixel_area=grid_area,
    )
    nclouds_low = result_low["final_nclouds"]
    print(f"  nclouds (mixed field, thresh=800): {nclouds_low}")

    # With area_thresh=7000, isolated cold anvil (6400 km^2) should be filtered out
    result_high = label_and_grow_cold_clouds(
        ir2, pixel_radius, tb_threshs, 7000.0,
        mincoldcorepix=4, smoothsize=5, warmanvilexpansion=0,
        config=config, pixel_area=grid_area,
    )
    nclouds_high = result_high["final_nclouds"]
    print(f"  nclouds (mixed field, thresh=7000): {nclouds_high}")
    assert nclouds_high < nclouds_low, (
        f"Expected fewer clouds with higher threshold: {nclouds_high} vs {nclouds_low}"
    )
    print("  PASS: Higher area threshold filters out small isolated cold anvil")
    print()


def test_consistency():
    """Verify fixed mode and latlon mode with uniform grid give same results."""
    print("=" * 60)
    print("Test: Consistency between fixed and latlon with uniform grid")
    print("=" * 60)

    ny, nx = 50, 50
    ir = make_ir_field(ny, nx)
    pixel_radius = 10.0
    tb_threshs = [225.0, 241.0, 261.0, 261.0]
    area_thresh = 800.0
    config = {"pbc_direction": "none"}

    # Fixed mode
    result_fixed = label_and_grow_cold_clouds(
        ir, pixel_radius, tb_threshs, area_thresh,
        mincoldcorepix=4, smoothsize=5, warmanvilexpansion=0,
        config=config,
    )

    # Latlon mode with uniform grid_area = pixel_radius^2
    uniform_area = np.full((ny, nx), pixel_radius**2)
    result_latlon = label_and_grow_cold_clouds(
        ir, pixel_radius, tb_threshs, area_thresh,
        mincoldcorepix=4, smoothsize=5, warmanvilexpansion=0,
        config=config, pixel_area=uniform_area,
    )

    nclouds_f = result_fixed["final_nclouds"]
    nclouds_l = result_latlon["final_nclouds"]
    print(f"  nclouds fixed: {nclouds_f}, nclouds latlon (uniform): {nclouds_l}")
    assert nclouds_f == nclouds_l, f"Mismatch: {nclouds_f} vs {nclouds_l}"
    print("  PASS: Same results for uniform latlon and fixed mode")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Phase 3 Tests: label_and_grow_cold_clouds")
    print("=" * 60 + "\n")

    test_fixed_mode()
    test_fixed_mode_with_scalar()
    test_latlon_mode()
    test_consistency()

    print("=" * 60)
    print("All Phase 3 tests PASSED!")
    print("=" * 60)
