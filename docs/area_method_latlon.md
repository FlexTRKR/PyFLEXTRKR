# Latitude-Dependent Area Calculation (`area_method: 'latlon'`)

## Overview

This feature adds support for latitude-dependent pixel area calculation in PyFLEXTRKR.
Instead of using a fixed `pixel_radius` (uniform grid assumption), the code can now
compute per-pixel areas using the Haversine formula on lat/lon coordinates, producing
a 2D array of pixel areas (km²) that accounts for the convergence of meridians at
higher latitudes.

This is controlled by a single config parameter:

```yaml
area_method: 'latlon'   # Use latitude-dependent area (new)
# area_method: 'fixed'  # Use uniform pixel_radius² (default, backward compatible)
```

When `area_method` is not specified, it defaults to `'fixed'`, preserving full
backward compatibility.

## Config Parameter

| Parameter     | Type   | Default   | Description |
|---------------|--------|-----------|-------------|
| `area_method` | string | `'fixed'` | `'fixed'`: use `pixel_radius²` for all areas. `'latlon'`: compute per-pixel area from lat/lon using the Haversine formula. |

When `area_method: 'latlon'`, a standalone grid area file is pre-computed and saved to:
```
{stats_outpath}/grid_area_from_latlon.nc
```
This file is computed once at the start of tracking and reused by all downstream steps.

## Modified Files

### Utility Functions — `pyflextrkr/ft_utilities.py`

New functions added:

- **`compute_grid_area(lon, lat)`** — Computes a 2D pixel area array (km²) from lon/lat
  coordinates using the Haversine formula. Handles single-row edge cases.
- **`save_grid_area(grid_area, outfile)`** — Saves grid area to a NetCDF file.
- **`load_grid_area(grid_area_file)`** — Loads grid area from a NetCDF file.
- **`get_pixel_area(config)`** — Returns the 2D pixel area array (cached via
  `load_grid_area`) or `None` if `area_method != 'latlon'`.
- **`get_feature_area(pixel_area, feature_mask, feature_labels)`** — Computes area of
  labeled features using per-pixel areas.
- **`get_mean_pixel_length(pixel_area, feature_mask_indices=None)`** — Returns mean
  pixel side length (km), optionally restricted to specific indices.
- **`precompute_grid_area(config, first_file=None)`** — Pre-computes and saves the
  grid area file at the start of tracking. Reads lat/lon from the first data file
  (NetCDF) or landmask file (Zarr).

Updated function:
- **`load_config()`** — Reads `area_method` from config, sets `grid_area_file` path.

### Feature Identification — Driver & Cloud ID

- **`pyflextrkr/idfeature_driver.py`** — Calls `precompute_grid_area()` before
  parallel feature identification loops.
- **`pyflextrkr/idclouds_tbpf.py`** — Loads `pixel_area` via `get_pixel_area()`.
  Passes it to `sort_renumber2vars()` and `label_and_grow_cold_clouds()`.
- **`pyflextrkr/idclouds_sat.py`** — Same pattern as `idclouds_tbpf.py`.
- **`pyflextrkr/ftfunctions.py`** — `sort_renumber2vars()` accepts optional
  `grid_area` parameter. When provided, `min_cellpix` is treated as an area
  threshold and converted to pixel count using mean pixel area.

### Cloud Labeling — `pyflextrkr/label_and_grow_cold_clouds.py`

- Accepts optional `pixel_area` parameter.
- Uses area-based thresholding for `sort_renumber` and the no-core branch.
- Handles periodic boundary conditions (PBC) by extending `pixel_area` via
  `pad_and_extend`.

### Track Statistics — `pyflextrkr/trackstats_func.py`

- Loads `pixel_area` from the grid area file.
- Five area calculations updated (corecold area, core area, cold area for Tb;
  core area, cell area for radar) to use `np.nansum(pixel_area[indices])` when
  `area_method == 'latlon'`.

### Track Statistics (deprecated) — `pyflextrkr/trackstats_single.py`

- Accepts optional `config` parameter.
- Uses per-pixel area for equivalent radius and regionprops scaling.

### Tb+PF Matching — `pyflextrkr/matchtbpf_func.py`

- Loads `pixel_area` once before the cloud loop (efficiency improvement).
- Subsets to `sub_pixel_area` per cloud for PF area calculations.
- Uses `mean_pixelength` (from domain-mean pixel area) for regionprops scaling
  (major/minor axis, perimeter).
- PF area computed via `np.nansum(sub_pixel_area[pf_indices])`.
- Computes `total_volrain` and `total_heavyvolrain` as per-pixel area-weighted
  sums: `np.nansum(rainrate * pixel_area)`.

### Tb+Radar Matching — `pyflextrkr/matchtbradar_func.py`

- Same pattern as `matchtbpf_func.py`: loads `pixel_area` once, subsets per cloud.
- Convective core stats (`calc_cc_stats`) and PF stats (`calc_pf_stats`) updated
  with per-pixel area arrays.
- All area-related outputs (ccarea, pfarea, sfarea, echotop areas) use per-pixel
  summation.
- Computes `total_volrain` and `total_heavyvolrain`.

### Movement Speed — `pyflextrkr/movement_speed.py`

- Computes **per-feature** mean pixel length within each tracked feature's mask area
  (union of both time steps), rather than a single domain-wide average.
- `movement_of_feature_fft()` returns a 5th output: `pixel_length_per_feature`.
- Distance/speed conversions (movement_mag, movement_x, movement_y, movement_speed)
  use per-feature pixel length for element-wise scaling.

### Robust MCS Classification (SAAG & MCSMIP) — `pyflextrkr/robustmcspf_saag.py`

- Uses pre-computed `total_volrain` directly from the stats file instead of
  `total_rain * pixel_radius²`.
- Removed `pixel_radius` config extraction (no longer needed).

## New Output Variables

Two new variables are added to the Tb+PF and Tb+Radar matching output files:

| Variable            | Units       | Description |
|---------------------|-------------|-------------|
| `total_volrain`     | mm/h km²   | Total volumetric precipitation under cold cloud shield |
| `total_heavyvolrain`| mm/h km²   | Total heavy volumetric precipitation under cold cloud shield |

These are computed as `sum(rainrate_i × area_i)` per pixel, which is physically
correct for non-uniform grids. For `area_method: 'fixed'`, they equal
`total_rain × pixel_radius²` (backward compatible).

## Usage Patterns Addressed

| Pattern | Description | Example | Solution |
|---------|-------------|---------|----------|
| A | `npix × pixel_radius²` → area | Cloud/PF area | `np.nansum(pixel_area[indices])` |
| B | `area_thresh / pixel_radius²` → min pixels | Min feature size | `area_thresh / np.nanmean(sub_pixel_area)` |
| C | `regionprops × pixel_radius` → km | Major axis, perimeter | `regionprops × mean_pixelength` |
| D | Movement displacement × pixel_radius | Speed calculation | Per-feature `sqrt(mean(pixel_area))` |
| E | Rain rate sum × pixel_radius² → vol rain | Volumetric rain | `sum(rainrate × pixel_area)` per pixel |
