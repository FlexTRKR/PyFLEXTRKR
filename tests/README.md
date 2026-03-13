# PyFLEXTRKR Automated Tests

This directory contains automated tests for PyFLEXTRKR.  
Tests are written with [pytest](https://docs.pytest.org/) and are split into three tiers:

| Tier | Files | Runs on | Needs data? |
|------|-------|---------|-------------|
| Unit / synthetic | `test_steiner_func.py`, `test_echotop_func.py`, `test_vertical_coordinate.py`, `test_idcells_synthetic.py`, `test_area_method_utils.py`, `test_area_method_clouds.py`, `test_example.py` | GitHub CI + local | No |
| Local regression | `test_regression_local.py` | Local / HPC only | Yes (demo outputs) |
| End-to-end demos | `run_demo_tests.py` | Local / HPC only | Auto-downloads |

---

## Quick start

Make sure pytest is installed (once per environment):
```bash
mamba install pytest        # or: pip install pytest
```

Run all CI-safe tests from the repo root:
```bash
cd /global/homes/f/feng045/program/PyFLEXTRKR-dev
python -m pytest tests/ -v
```

You should see output like:
```
tests/test_echotop_func.py .......
tests/test_steiner_func.py ..............
tests/test_vertical_coordinate.py ...................
...
49 passed, 6 skipped
```

The 6 `s` (skipped) tests are the local regression tests — they are automatically
skipped unless you set the `PYFLEXTRKR_TEST_DATA` environment variable (see below).

---

## Running a specific test file

```bash
python -m pytest tests/test_steiner_func.py -v
python -m pytest tests/test_echotop_func.py -v
python -m pytest tests/test_vertical_coordinate.py -v
python -m pytest tests/test_idcells_synthetic.py -v
```

## Running a single test by name

Use the `-k` flag with any part of the test name:
```bash
python -m pytest tests/ -v -k "convective"     # all tests with 'convective' in the name
python -m pytest tests/ -v -k "echotop"
python -m pytest tests/ -v -k "scale_factor"
```

## Stopping at the first failure

```bash
python -m pytest tests/ -v -x
```

## Show the full error traceback (more detail than default)

```bash
python -m pytest tests/ -v --tb=long
```

---

## Understanding the output symbols

| Symbol | Meaning |
|--------|---------|
| `.`    | Test passed |
| `F`    | Test failed |
| `s`    | Test skipped (e.g., local data not available) |
| `E`    | Error during test setup (not a test failure itself) |

---

## Test files explained

### `test_steiner_func.py`
Unit tests for the Steiner convective/stratiform classification functions
(`background_intensity`, `make_dilation_step_func`, `mod_steiner_classification`,
`expand_conv_core`). Uses a small synthetic 50×50 km reflectivity field with one
obvious convective core at the centre — no real radar data needed.

### `test_echotop_func.py`
Unit tests for echo-top height calculation (`calc_cloud_boundary`, `echotop_height`).
Tests both 1D and 3D height coordinate inputs.

### `test_vertical_coordinate.py`
Unit tests for `standardize_vertical_coordinate()` — the function that converts
vertical coordinates to standard units (metres for height, hPa for pressure).
Covers: metres, kilometres, Pascals, hPa, `scale_factor` override, `units_override`,
missing units, and error cases (geopotential height, unknown units).

### `test_idcells_synthetic.py`
End-to-end integration test. Creates a minimal synthetic NetCDF radar file in a
temporary directory, then runs the full cell-identification pipeline:
1. `get_composite_reflectivity_generic` (file reader)
2. `mod_steiner_classification` (convective cell detection)
3. `echotop_height` (echo-top calculation)
4. `expand_conv_core` (cell labelling)

Verifies that the known convective cell in the synthetic data is detected correctly.
No real data needed.

### `test_area_method_utils.py`
Unit tests for the `area_method: 'latlon'` utility functions: `compute_grid_area`,
`save_grid_area`, `load_grid_area`, `get_pixel_area`, and `precompute_grid_area`.
Also tests `sort_renumber2vars` with a 2D `grid_area` array to verify that
feature areas are computed correctly with variable-size pixels.
Uses synthetic lat/lon grids and temporary files — no real data needed.

### `test_area_method_clouds.py`
Unit tests for `label_and_grow_cold_clouds` with 2D `pixel_area` (grid area)
support. Verifies that cloud labelling, area thresholding, and feature
renumbering work correctly when pixel areas vary across the domain (as they
do for latitude-dependent grids). Tests both the fixed (scalar) and
latlon (2D array) modes to ensure backward compatibility.

### `test_regression_local.py`
Validates the output of full demo runs (e.g., `demo_cell_nexrad.sh`).
These tests check that track statistics files contain reasonable values
(correct geographic region, positive durations, finite coordinates) and that
quicklook plots and animations were produced.

**These tests are skipped automatically unless you set `PYFLEXTRKR_TEST_DATA`:**
```bash
# Point to the root directory of your demo output data
export PYFLEXTRKR_TEST_DATA=/pscratch/sd/f/feng045/demo

# Run the demo first, then validate its outputs:
bash config/demo_cell_nexrad.sh
python -m pytest tests/test_regression_local.py -v
```

---

## Workflow: running tests after making code changes

```bash
# 1. Make your code changes
# 2. Run all CI-safe tests — should take < 10 seconds
python -m pytest tests/ -v

# 3. If you have local demo data, run regression tests too
export PYFLEXTRKR_TEST_DATA=/pscratch/sd/f/feng045/demo
python -m pytest tests/ -v
```

---

## End-to-end demo tests (`run_demo_tests.py`)

The demo test runner automates the full workflow: download sample data →
run tracking → validate outputs. It replaces the manual process of editing
`dir_demo` in shell scripts and visually inspecting results.

### List available demos
```bash
python tests/run_demo_tests.py --list
```

### Run specific demos
```bash
# Run two demos with 4 workers
python tests/run_demo_tests.py --demos demo_mcs_imerg demo_cell_nexrad -n 4

# Run all portable demos with 8 workers
python tests/run_demo_tests.py --all -n 8
```

### Validate outputs without re-running
```bash
python tests/run_demo_tests.py --demos demo_mcs_imerg --validate-only
```

### Custom data root directory
```bash
python tests/run_demo_tests.py --demos demo_mcs_imerg --data-root /tmp/demo -n 4
```

### Include plotting and animation
```bash
python tests/run_demo_tests.py --demos demo_mcs_imerg --with-plots -n 4
```

### Force re-download and backup existing data
```bash
python tests/run_demo_tests.py --demos demo_mcs_imerg --fresh --backup -n 4
```

### Available demos

| Demo name | Type | Description |
|-----------|------|-------------|
| `demo_mcs_imerg` | MCS Tb+PF | GPM IMERG Tb+Precipitation |
| `demo_mcs_wrf_tbpf` | MCS Tb+PF | WRF Tb+Precipitation |
| `demo_mcs_model25km` | MCS Tb+PF | E3SM OLR+Precipitation (25 km) |
| `demo_mcs_tbpf_idealized` | MCS Tb+PF | Idealized Tb+Precipitation |
| `demo_mcs_wrf_tbradar` | MCS Tb+Radar | WRF Tb+Radar (3D) |
| `demo_mcs_gridrad` | MCS Tb+Radar | GridRad Tb+Radar |
| `demo_mcs_himawari` | MCS Tb-only | Himawari Tb-only |
| `demo_cell_nexrad` | Cell | NEXRAD radar (KHGX) |
| `demo_cell_csapr` | Cell | CACTI CSAPR2 radar |
| `demo_generic_tracking` | Generic | ERA5 Z500 anomaly |

### What the runner does (for each demo)

1. **Clean** — Removes previous output directories (stats, tracking, pixel files).
   Input data is preserved unless `--fresh` is specified.
   Use `--backup` to move outputs to `*.bak/` instead of deleting.
2. **Download** — Downloads sample data via `wget` (skipped if input already exists).
3. **Configure** — Generates a config YAML from the template, overriding
   `nprocesses` with the value from `-n`.
4. **Track** — Runs the Python tracking pipeline.
5. **Plot** (optional) — Runs quicklook plotting and `ffmpeg` animation (with `--with-plots`).
6. **Validate** — Checks that:
   - Stats file exists with ≥ 1 track
   - Lat/lon values fall within expected geographic range
   - Track durations are non-negative
   - Pixel-level tracking files exist with `tracknumber` variable
   - (If `--with-plots`) Quicklook PNGs and animation file exist

### Linking to pytest regression tests

After running demos, you can also run the pytest-based regression tests:
```bash
export PYFLEXTRKR_TEST_DATA=~/data/demo
python -m pytest tests/test_regression_local.py -v
```

The regression test classes (`TestMcsImergDemo`, `TestCellNexradDemo`, etc.)
provide finer-grained assertions on the same output files.

If any test fails after your changes, the output will show exactly which assertion
failed and what values were produced vs. expected, making it easy to diagnose
whether the change broke existing behaviour or whether the test needs updating.

---

## Adding new tests

Each test file follows the same pattern:

```python
import numpy as np
import pytest
from pyflextrkr.some_module import some_function

def test_my_function_does_something():
    # Arrange: set up inputs
    my_input = np.array([1, 2, 3])
    # Act: call the function
    result = some_function(my_input)
    # Assert: check the result
    assert result == 6, f"Expected 6, got {result}"
```

Group related tests in a class for clarity:
```python
class TestSomeFunction:
    def test_normal_case(self): ...
    def test_edge_case(self): ...
    def test_raises_on_bad_input(self): ...
```

To mark a test as local-only (skipped on CI):
```python
@pytest.mark.local
def test_something_needing_data():
    ...
```
