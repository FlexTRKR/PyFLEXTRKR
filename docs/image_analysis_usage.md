# Image-Analysis Package Usage

This document catalogs every call to a third-party image-analysis library (`scipy.ndimage`, `scipy.signal`, `astropy.convolution`, `scikit-image`) reached from three feature-identification drivers, including calls made inside the helper functions each driver invokes. Pure-NumPy logic (indexing, masking, arithmetic on already-labeled arrays) is noted as containing no image-analysis calls, even where it implements similar-sounding logic by hand.

Config flags gate several alternative code paths (e.g. `cloudidmethod`, `growth_method`, `convolve_method`, `dilate_method`, `expand_method`, `label_method`). Each branch is documented separately rather than implying one single linear call sequence — only one branch executes per run, depending on the YAML config.

## `pyflextrkr/idclouds_tbpf.py`

Identifies convective cloud objects from infrared brightness temperature (+ optional precipitation linking, + optional 3D radar via SL3D).

### Direct call

| # | Function | Called from | Purpose |
|---|----------|-------------|---------|
| 1 | `scipy.signal.medfilt2d` | `idclouds_tbpf` (line 275) | Median-filters the raw Tb field and uses the filtered values to fill in missing/NaN pixels before thresholding. |

### Branch: `cloudidmethod == "label_grow"` → `label_and_grow_features()` (`pyflextrkr/label_and_grow_features.py`)

| # | Function | Called from | Purpose |
|---|----------|-------------|---------|
| 2 | `astropy.convolution.Box2DKernel` | `smooth_field` helper (line 816) | Builds a 2D boxcar smoothing kernel of configurable width. |
| 3 | `astropy.convolution.convolve` | `smooth_field` helper (lines 817-820) | NaN-aware convolution of the input field with the box kernel, smoothing it before core detection. |
| 4 | `scipy.ndimage.label` | `find_and_label_cores` helper (line 799) | Labels connected components of the binary core mask (pixels beyond `core_thresh`) to identify individual cores. |
| 5 | `scipy.ndimage.label` | `_grow_edt` helper, only if `growth_method == "edt"` (line 589) | Labels 8-connected components of the "valid" mask so distance-based growth stays confined per region. |
| 6 | `scipy.ndimage.distance_transform_edt` | `_grow_edt` helper, only if `growth_method == "edt"` (lines 626-628) | Euclidean distance transform with `return_indices=True` from non-seed pixels to the nearest core seed — a Voronoi-style nearest-seed growth from cores to the secondary threshold. |
| 7 | `scipy.ndimage.label` | `label_and_grow_features` main function (line 241) | Labels connected components of isolated secondary/core pixels not already grown from a core, as standalone candidate features. |
| 8 | `scipy.ndimage.label` | `label_and_grow_features` main function, fallback when no cores exist (line 318) | Labels connected components of the raw secondary mask directly. |
| 9 | `scipy.ndimage.generate_binary_structure` | `_expand_tertiary` helper, only if `expand_to_tertiary == 1` (line 724) | Builds a cross-shaped, one-connectivity structuring element for single-pixel dilation. |
| 10 | `scipy.ndimage.binary_dilation` | `_expand_tertiary` helper, only if `expand_to_tertiary == 1` (lines 725-727) | Iteratively dilates each feature mask by one pixel per loop to grow toward the tertiary (warm-anvil) threshold. |

If `growth_method == "bfs"` (the default), growth instead uses the custom `grow_cells` function (`pyflextrkr.ftfunctions`) — pyflextrkr's own code, not a third-party image-analysis call.

Confirmed **no** image-analysis calls in `sort_renumber`, `sort_renumber2vars`, `link_pf_tb`, `pad_and_extend`, `call_adjust_axis`, `olr_to_tb` — all pure NumPy operations on already-labeled arrays or scalar formulas.

### Branch: `cloudidmethod == "futyan3"` → `futyan3()` (`pyflextrkr/futyan3.py`)

| # | Function | Called from | Purpose |
|---|----------|-------------|---------|
| 11 | `scipy.ndimage.label` | `futyan3` (line 38) | Labels connected components of the cold-anvil-threshold mask to identify candidate cloud features (core + cold anvil). |
| 12 | `scipy.ndimage.generate_binary_structure` | `futyan3`, only if `warmanvilexpansion == 1` (lines 147-149) | Builds a cross-shaped, one-connectivity structuring element. |
| 13 | `scipy.ndimage.binary_dilation` | `futyan3`, only if `warmanvilexpansion == 1` (lines 151-153) | Iteratively dilates each feature by one pixel per loop, expanding core+cold-anvil into the warm-anvil region. |

### Branch: `linkpf == 1` (precipitation-feature linking, runs after either cloud-ID method above)

| # | Function | Called from | Purpose |
|---|----------|-------------|---------|
| 14 | `astropy.convolution.Box2DKernel` | `idclouds_tbpf` (lines 376/387) | Boxcar kernel to smooth the precipitation field (PBC and non-PBC branches use the same call). |
| 15 | `astropy.convolution.convolve` | `idclouds_tbpf` (lines 377-380/388-391) | Smooths the precipitation field, handling NaNs, before thresholding. |
| 16 | `scipy.ndimage.label` | `idclouds_tbpf` (lines 382/398) | Labels precipitation features exceeding the dBZ threshold. |

### Branch: `"radar3d" in feature_type` → `run_sl3d()` (`pyflextrkr/sl3d_func.py`)

| # | Function | Called from | Purpose |
|---|----------|-------------|---------|
| 17 | `scipy.ndimage.median_filter` | `gridrad_sl3d` helper (line 364) | 2D median filter over each low-level vertical layer to compute local background reflectivity; the difference from the raw field becomes the "peakedness" convective indicator. |
| 18 | `scipy.ndimage.uniform_filter` | `gridrad_sl3d` helper (line 394) | 3×3 neighborhood-mean of the binary convective mask, used to prune isolated single-pixel convective classifications. |
| 19 | `scipy.ndimage.uniform_filter` | `gridrad_sl3d` helper (line 402) | 3×3 neighborhood-mean recomputed after pruning, to expand convection to adjacent similarly-intense pixels. |
| 20 | `scipy.ndimage.uniform_filter` | `gridrad_sl3d` helper (line 439) | 3D neighborhood-mean (`[1,3,3]` kernel) of the finite-echo mask, a coverage-fraction test used when detecting convective updraft weak-echo regions. |
| 21 | `scipy.ndimage.uniform_filter` | `gridrad_sl3d` helper (line 451) | 3×3 neighborhood-mean of the binary updraft mask, used to prune isolated single-pixel updraft classifications. |
| 22 | `scipy.ndimage.median_filter` | `gridrad_sl3d` helper (line 457) | 3×3 median filter over the full classification array, used to reclassify pruned pixels to their local-majority class. |

`run_sl3d` also calls `echotop_height` (`pyflextrkr/echotop_func.py`) — see the "no calls found" note under Section 2 below.

---

## `pyflextrkr/idcells_reflectivity.py`

Identifies convective cells from composite radar reflectivity using a modified Steiner classification.

The driver itself has no direct image-analysis imports; every call happens inside helpers.

### `mod_steiner_classification()` (`pyflextrkr/steiner_func.py`)

| # | Function | Called from | Purpose |
|---|----------|-------------|---------|
| 1 | `scipy.ndimage.convolve` (×2) | `background_intensity` helper, only if `convolve_method == "ndimage"` (lines 44-45) | Direct spatial convolution of linear reflectivity (and the good-value mask, for normalization) with a circular disk footprint to compute local background reflectivity. |
| 2 | `scipy.signal.fftconvolve` (×2) | `background_intensity` helper, only if `convolve_method == "fft"` — **the driver's default** (lines 52-53) | FFT-based convolution of linear reflectivity and the good-value mask with the disk footprint; faster than direct convolution for large kernels. |
| 3 | `scipy.signal.convolve` (`method='fft'`, ×2) | `background_intensity` helper, only if `convolve_method == "signal"` (lines 56-57) | Alternate explicit-FFT convolution path via `scipy.signal.convolve`. |
| 4 | `scipy.ndimage.label` | `mod_steiner_classification`, only if `remove_smallcores == True` (line 870) | Connected-component labeling of the convective-core binary mask, to filter cores below `min_corearea`. |
| 5 | `scipy.ndimage.binary_dilation` | `mod_dilate_conv_rad` helper, only if `dilate_method == "orig"` — **the default** (line 290) | Dilates each convective-core radius bin outward with a circular structuring element, masked by good-value pixels. |
| 6 | `scipy.ndimage.distance_transform_edt` | `mod_dilate_conv_rad_edt` helper, only if `dilate_method == "edt"` (line 356) | Euclidean distance transform (pixel-size-aware `sampling=(dy,dx)`) from core pixels, an O(npix) alternative to repeated dilation for radius-based expansion. |
| 7 | `scipy.ndimage.label` | `mod_steiner_classification`, only if `remove_smallcells == True` (line 913) | Connected-component labeling of the dilated convective-cell mask, to filter cells below `min_cellarea`. |

### Convective-core expansion (branches on `expand_method`)

| # | Function | Called from | Purpose |
|---|----------|-------------|---------|
| 8 | `scipy.ndimage.label` | `label_cells` helper, via `expand_conv_core()` — **the default** (line 387) | Labels/sorts convective cores by size before sequential radius expansion. |
| 9 | `scipy.ndimage.binary_dilation` | `expand_conv_core()` (line 657) | Per-core, per-radius loop dilating each core outward with a circular structuring element into the still-unclaimed region. |
| 10 | `scipy.ndimage.label` | `label_cells_fast` helper, via `expand_conv_core_fast()`, only if `expand_method == "fast"` (line 455) | Labels/sorts convective cores by size (vectorized variant). |
| 11 | `scipy.ndimage.grey_dilation` | `expand_conv_core_fast()`, only if `expand_method == "fast"` (line 543) | Grayscale dilation with a circular footprint over inverted core-label values — a single vectorized call per radius replacing the per-core dilation loop. |
| 12 | `scipy.ndimage.label` | `label_cells_fast` helper, via `expand_conv_core_edt()`, only if `expand_method == "edt"` (line 455) | Labels/sorts convective cores by size. |
| 13 | `scipy.ndimage.distance_transform_edt` | `expand_conv_core_edt()`, only if `expand_method == "edt"` (line 590) | Distance transform with `return_indices=True`, assigning every pixel to its nearest labeled core within `max_radius` (Voronoi-style, replacing the sequential dilation loop). |

### Echo-top height and HEALPix remap — confirmed no image-analysis calls

- `echotop_height` / `echotop_height_fast` (`pyflextrkr/echotop_func.py`): pure NumPy (boolean masking, `np.where`/`np.diff`/`np.split` over vertical levels) — no scipy/skimage/cv2/astropy calls anywhere in the file.
- `remap_healpix_to_latlon_grid` (`pyflextrkr/hp_utilities.py`, zarr/HEALPix input path only): uses `healpix.ang2pix` (nearest-cell index lookup) and xarray ops — an index/regridding lookup, not an image-processing function.

---

## `pyflextrkr/idfeature_generic.py`

Identifies generic 2D features (e.g. Z500, vorticity) via simple thresholding or watershed.

| # | Function | Called from | Purpose |
|---|----------|-------------|---------|
| 1 | `scipy.ndimage.label` | `idfeature_generic`, only if `label_method == "ndimage.label"` — **the default** (line 150) | Labels connected components of pixels within `[field_thresh_min, field_thresh_max]` as candidate features. |
| 2 | `skimage.feature.peak_local_max` | `skimage_watershed()` (`pyflextrkr/ftfunctions.py`), only if `label_method == "skimage.watershed"` (line 510) | Finds grid indices of local maxima in the field, using configured `min_distance`/`exclude_border`/`threshold_abs`, to seed the watershed. |
| 3 | `skimage.segmentation.watershed` | `skimage_watershed()`, only if `label_method == "skimage.watershed"` (line 524) | Runs watershed segmentation on the inverted field from the peak-local-max markers, masked by threshold, with `watershed_line=True`, to delineate individual features. |

Confirmed **no** image-analysis calls in `sort_renumber` (pure NumPy relabeling/size filtering of an already-labeled array).

---

## Library usage summary

| Library | Functions used |
|---|---|
| `scipy.ndimage` | `label`, `binary_dilation`, `generate_binary_structure`, `distance_transform_edt`, `convolve`, `grey_dilation`, `median_filter`, `uniform_filter` |
| `scipy.signal` | `medfilt2d`, `fftconvolve`, `convolve` (FFT method) |
| `astropy.convolution` | `Box2DKernel`, `convolve` |
| `skimage.feature` | `peak_local_max` |
| `skimage.segmentation` | `watershed` |

No OpenCV (`cv2`) usage exists in any of the three drivers or their helper call chains.
