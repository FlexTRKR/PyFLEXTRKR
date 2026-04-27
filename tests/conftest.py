"""
conftest.py – shared fixtures and custom pytest markers.

Markers
-------
local   Tests that require locally-available data (HPC / workstation only).
        Skip automatically on GitHub CI where no data is mounted.
demo    Full end-to-end demo run (slow, needs data + multiple CPUs).

Usage
-----
Run everything:           pytest tests/
Skip local/demo tests:    pytest tests/ -m "not local and not demo"
Run only local tests:     pytest tests/ -m local
"""

import os
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Register custom markers so pytest does not emit warnings
# ---------------------------------------------------------------------------
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "local: test requires locally mounted data (skip on GitHub CI)"
    )
    config.addinivalue_line(
        "markers",
        "demo: full end-to-end demo run (slow, needs data and multiple workers)"
    )


# ---------------------------------------------------------------------------
# Custom CLI options
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--nfiles",
        type=int,
        default=1,
        help=(
            "Number of demo input files to use in @pytest.mark.local tests. "
            "1 = first file only (default, fast). "
            "0 = all available files (most thorough)."
        ),
    )


@pytest.fixture(scope="session")
def nfiles(request):
    """Number of demo input files to process in local integration tests."""
    return request.config.getoption("--nfiles")


# ---------------------------------------------------------------------------
# Auto-skip local/demo tests when no data root is found
# ---------------------------------------------------------------------------
DATA_ROOT_ENV = "PYFLEXTRKR_TEST_DATA"  # set this env-var on HPC/workstation


def pytest_collection_modifyitems(config, items):
    data_available = os.environ.get(DATA_ROOT_ENV) is not None
    skip_local = pytest.mark.skip(
        reason=f"Requires local data: set {DATA_ROOT_ENV} env variable to enable."
    )
    for item in items:
        if "local" in item.keywords and not data_available:
            item.add_marker(skip_local)
        if "demo" in item.keywords and not data_available:
            item.add_marker(skip_local)


# ---------------------------------------------------------------------------
# Shared synthetic-data fixtures (usable by all test files)
# ---------------------------------------------------------------------------

@pytest.fixture
def grid_params():
    """Standard small grid used across tests: 50x50 at 1 km resolution."""
    return dict(nx=50, ny=50, dx=1000.0, dy=1000.0)


@pytest.fixture
def synthetic_refl_2d(grid_params):
    """
    Simple 2D composite reflectivity field with one obvious convective cell.

    Layout (50x50 grid, 1 km spacing):
    - Background: 20 dBZ everywhere
    - Stratiform ring (r=5..10 km around centre): 30 dBZ
    - Convective core (r<5 km around centre):     50 dBZ
    """
    nx, ny = grid_params['nx'], grid_params['ny']
    cx, cy = nx // 2, ny // 2

    yy, xx = np.ogrid[:ny, :nx]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    refl = np.full((ny, nx), 20.0, dtype=np.float32)
    refl[(r >= 5) & (r < 10)] = 30.0
    refl[r < 5] = 50.0
    return refl


@pytest.fixture
def mask_goodvalues(grid_params):
    """All-valid mask matching the default 50x50 grid."""
    return np.ones((grid_params['ny'], grid_params['nx']), dtype=int)


@pytest.fixture
def synthetic_dbz3d(grid_params):
    """
    Minimal 3D reflectivity DataArray [z, y, x] for echo-top tests.

    10 height levels (0..9 km); convective column above the grid centre
    has 50 dBZ up to 8 km, 0 dBZ elsewhere.
    """
    import xarray as xr

    nx, ny = grid_params['nx'], grid_params['ny']
    nz = 10
    height_1d = np.arange(nz, dtype=np.float32) * 1000.0  # 0..9 km in m
    cx, cy = nx // 2, ny // 2

    data = np.zeros((nz, ny, nx), dtype=np.float32)
    # Convective column: 50 dBZ below 8 km
    data[:8, cy - 2:cy + 3, cx - 2:cx + 3] = 50.0

    da = xr.DataArray(
        data,
        dims=['z', 'y', 'x'],
        coords={'z': height_1d},
    )
    return da, height_1d


@pytest.fixture
def steiner_types():
    return {
        'NO_SURF_ECHO': 1,
        'WEAK_ECHO': 2,
        'STRATIFORM': 3,
        'CONVECTIVE': 4,
    }
