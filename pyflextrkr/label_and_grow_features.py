"""
Generalized feature labeling and growth for 2D scalar fields.

This module provides `label_and_grow_features`, a generalized version of
`label_and_grow_cold_clouds` that works with any 2D field (e.g., brightness
temperature, radar reflectivity). It supports:

- **core_operator='lt'**: Lower values define cores (e.g., Tb cold cores).
- **core_operator='gt'**: Higher values define cores (e.g., Ze convective cores).
- **growth_method='bfs'**: Original BFS-based growth via `grow_cells` (backward
  compatible, slower for large domains).
- **growth_method='edt'**: Voronoi-based growth via scipy `distance_transform_edt`
  (faster, recommended for new applications). Boundary assignments may differ
  by 1-2 pixels from BFS at equidistant boundaries.

Threshold ordering convention
-----------------------------
Thresholds are always ordered from "most intense" (core) to "least intense"
(edge), regardless of the operator direction:

- For Tb (core_operator='lt'): [225, 241, 261, 261] means core < 225 K,
  secondary < 241 K, tertiary < 261 K, edge < 261 K.
- For Ze (core_operator='gt'): [30, 10, 5, 5] means core > 30 dBZ,
  secondary > 10 dBZ, tertiary > 5 dBZ, edge > 5 dBZ.
"""

import logging
import numpy as np
from scipy.ndimage import label, binary_dilation, generate_binary_structure
from astropy.convolution import Box2DKernel, convolve
from pyflextrkr.ftfunctions import sort_renumber, grow_cells, pad_and_extend, call_adjust_axis


def label_and_grow_features(
    field,
    pixel_radius,
    thresholds,
    area_thresh,
    min_core_npix,
    smooth_size,
    expand_to_tertiary,
    config,
    pixel_area=None,
    core_operator="lt",
    growth_method="bfs",
):
    """
    Label and grow features in a 2D scalar field.

    Identifies intense cores, grows them outward to a secondary threshold,
    labels remaining isolated secondary-threshold regions, combines and sorts
    all features by size, and optionally expands to a tertiary threshold.

    Args:
        field: np.ndarray
            2D input data array (e.g., Tb in K, or reflectivity in dBZ).
        pixel_radius: float
            Pixel size in km.
        thresholds: list or np.ndarray
            4-element list of thresholds ordered from most intense to least:
            [core_thresh, secondary_thresh, tertiary_thresh, edge_thresh].
        area_thresh: float
            Minimum area to define a feature [km^2].
        min_core_npix: int
            Minimum number of pixels to define a core.
        smooth_size: int
            Window size for Box2DKernel smoothing before core detection.
        expand_to_tertiary: int
            Flag (0 or 1) to expand features to include tertiary region.
        config: dict
            Dictionary containing config parameters (for PBC settings).
        pixel_area: float or np.ndarray, optional
            Scalar pixel_radius^2 or 2D grid_area array (ny, nx) in km^2.
            When 2D, area-based thresholds use actual grid cell areas.
        core_operator: str, optional
            Threshold comparison operator for defining cores.
            'lt': field < threshold defines cores (e.g., Tb).
            'gt': field > threshold defines cores (e.g., reflectivity).
            Default: 'lt'.
        growth_method: str, optional
            Method for growing labeled cores outward to secondary threshold.
            'bfs': Breadth-first search via grow_cells (original, exact backward
                   compatibility with label_and_grow_cold_clouds).
            'edt': Euclidean distance transform (Voronoi assignment, faster,
                   recommended for new applications).
            Default: 'bfs'.

    Returns:
        dict: Dictionary containing:
            - final_nFeature (int): Number of labeled features.
            - final_Core_npix (np.ndarray): Core pixel count per feature.
            - final_Secondary_npix (np.ndarray): Secondary region pixel count per feature.
            - final_CoreSecondary_npix (np.ndarray): Core + secondary pixel count.
            - final_Tertiary_npix (np.ndarray): Tertiary region pixel count per feature.
            - final_Feature_Number (np.ndarray): 2D labeled feature array (with
              tertiary expansion if enabled).
            - final_Feature_Type (np.ndarray): 2D pixel classification array.
            - final_CoreSecondary_Number (np.ndarray): 2D labeled feature array
              (core + secondary only, no tertiary).
    """
    logger = logging.getLogger(__name__)

    # Validate inputs
    if core_operator not in ("lt", "gt"):
        raise ValueError(f"core_operator must be 'lt' or 'gt', got '{core_operator}'")
    if growth_method not in ("bfs", "edt"):
        raise ValueError(f"growth_method must be 'bfs' or 'edt', got '{growth_method}'")

    # Periodic boundary conditions
    pbc_direction = config.get("pbc_direction", "none")

    # Separate thresholds
    core_thresh = thresholds[0]       # Most intense (core)
    secondary_thresh = thresholds[1]  # Secondary expansion
    tertiary_thresh = thresholds[2]   # Tertiary expansion
    edge_thresh = thresholds[3]       # Outermost boundary

    # Determine dimensions
    ny, nx = np.shape(field)

    # Set pixel_area if not provided (backward compatible)
    if pixel_area is None:
        pixel_area = pixel_radius ** 2

    # Check if pixel_area is a 2D array (latlon mode)
    use_grid_area = isinstance(pixel_area, np.ndarray) and pixel_area.ndim == 2

    # Calculate minimum number of pixels based on area threshold
    if use_grid_area:
        nthresh = area_thresh
    else:
        nthresh = area_thresh / pixel_area

    ######################################################################
    # Classify pixels by thresholds
    (
        secondary_flag,
        core_flag,
        feature_type_map,
    ) = classify_pixels_by_thresholds(
        field, nx, ny, edge_thresh, secondary_thresh, core_thresh,
        tertiary_thresh, core_operator,
    )

    #################################################################
    # Handle periodic boundary conditions
    if pbc_direction != "none":
        # Save original data
        field_orig = np.copy(field)
        core_flag_orig = np.copy(core_flag)
        secondary_flag_orig = np.copy(secondary_flag)
        feature_type_map_orig = np.copy(feature_type_map)

        # Step 1: Extend and pad data
        field, padded_x, padded_y = pad_and_extend(field, config)
        core_flag, _, _ = pad_and_extend(core_flag, config)
        secondary_flag, _, _ = pad_and_extend(secondary_flag, config)
        feature_type_map, _, _ = pad_and_extend(feature_type_map, config)
        # Extend pixel_area if it's 2D
        if use_grid_area:
            pixel_area, _, _ = pad_and_extend(pixel_area, config)

        # Update dimensions after padding
        ny, nx = field.shape
    else:
        # If PBC is not applied, keep original data
        field_orig = field
        core_flag_orig = core_flag
        secondary_flag_orig = secondary_flag
        feature_type_map_orig = feature_type_map

    # Smooth field data
    smoothed_field = smooth_field(field, smooth_size)
    # Label cores
    labeled_cores, nlabelcores = find_and_label_cores(
        smoothed_field, core_thresh, core_operator,
    )

    # Create empty arrays
    labeled_core_secondary = np.zeros((ny, nx), dtype=int)
    sorted_features = np.zeros((ny, nx), dtype=int)
    final_feature_number = np.zeros((ny, nx), dtype=int)
    labeled_core_secondary_npix = []
    sortedcore_npix = []
    sortedsecondary_npix = []
    sortedtertiary_npix = []

    # Check if any cores have been identified
    if nlabelcores > 0:

        # Sort cores by size and remove small cores
        sortedcore_number2d, sortedcore_npix = sort_renumber(
            labeled_cores, min_core_npix,
        )
        # Check if any of the cores passed the size threshold test
        ivalidcores = np.array(np.where(sortedcore_npix > 0))[0]
        ncores = len(ivalidcores)

        # Check if cores satisfy size threshold
        if ncores > 0:

            #####################################################
            # Grow cores outward until reaching secondary threshold.
            labeled_core_secondary = np.copy(sortedcore_number2d)
            labeled_core_secondary_npix = np.copy(sortedcore_npix)

            if growth_method == "bfs":
                # BFS growth (original method)
                labeled_core_secondary = _grow_bfs(
                    labeled_core_secondary, field, secondary_thresh, core_operator,
                )
            elif growth_method == "edt":
                # EDT/Voronoi growth
                labeled_core_secondary = _grow_edt(
                    labeled_core_secondary, field, secondary_thresh, core_operator,
                )

            # Update the cloud sizes
            cloud_indices, cloud_sizes = np.unique(
                labeled_core_secondary, return_counts=True,
            )
            for i, index in enumerate(cloud_indices):
                if index == 0:
                    continue
                labeled_core_secondary_npix[index - 1] = cloud_sizes[i]

        ############################################################
        # Label secondary regions that do not have a core

        # Find indices that satisfy secondary threshold or core threshold
        # and are not labeled
        isolated_flag = np.zeros((ny, nx), dtype=int)
        isolated_indices = np.where(
            (labeled_core_secondary == 0)
            & ((secondary_flag > 0) | (core_flag > 0))
        )
        nisolated = np.shape(isolated_indices)[1]
        if nisolated > 0:
            isolated_flag[isolated_indices] = 1

        labelisolated_number2d, nlabelisolated = label(isolated_flag)
        # Sort isolated regions by size and remove small ones
        if use_grid_area:
            sortedisolated_number2d, sortedisolated_npix = sort_renumber(
                labelisolated_number2d, nthresh, grid_area=pixel_area,
            )
        else:
            sortedisolated_number2d, sortedisolated_npix = sort_renumber(
                labelisolated_number2d, nthresh,
            )

        ##############################################################
        # Combine cores+secondary with isolated secondary regions

        # Add isolated features with numbers after the valid cores
        labelcombined_number2d = np.copy(labeled_core_secondary)

        sortedisolated_indices = np.where(sortedisolated_number2d > 0)
        nsortedisolatedindices = np.shape(sortedisolated_indices)[1]
        if nsortedisolatedindices > 0:
            labelcombined_number2d[sortedisolated_indices] = np.copy(
                sortedisolated_number2d[sortedisolated_indices]
            ) + np.copy(ncores)

        # Combine the npix data
        labelcombined_npix = np.hstack(
            (labeled_core_secondary_npix, sortedisolated_npix)
        )
        ncombined = len(labelcombined_npix)

        # Initialize cloud numbers
        labelcombined_number1d = np.arange(1, ncombined + 1)

        # Sort clouds by size
        order = np.argsort(labelcombined_npix)
        order = order[::-1]
        sortedcombined_npix = np.copy(labelcombined_npix[order])
        sortedcombined_number1d = np.copy(labelcombined_number1d[order])

        # Re-number features
        sortedcombined_number2d = np.zeros((ny, nx), dtype=int)
        final_Core_npix = np.ones(ncombined, dtype=int) * -9999
        final_Secondary_npix = np.ones(ncombined, dtype=int) * -9999
        final_Tertiary_npix = np.ones(ncombined, dtype=int) * -9999
        featurecount = 0
        for ifeature in range(0, ncombined):
            # Find pixels that have matching number
            feature_indices = (
                labelcombined_number2d == sortedcombined_number1d[ifeature]
            )
            nfeatureindices = np.count_nonzero(feature_indices)

            if nfeatureindices == sortedcombined_npix[ifeature]:
                featurecount = featurecount + 1
                sortedcombined_number2d[feature_indices] = featurecount

                final_Core_npix[featurecount - 1] = np.nansum(
                    core_flag[feature_indices]
                )
                final_Secondary_npix[featurecount - 1] = np.nansum(
                    secondary_flag[feature_indices]
                )

        ##############################################
        # Save final matrices
        final_CoreSecondary_Number = np.copy(sortedcombined_number2d)
        # Use featurecount (post size-consistency-filter), not ncombined
        # (pre-filter): a rejected feature at the check above means
        # ncombined can exceed featurecount, which previously made
        # final_nFeature bigger than the (already trimmed) npix arrays
        # below - and, further downstream, bigger than the label values
        # actually present in final_CoreSecondary_Number.
        final_nFeature = np.copy(featurecount)

        final_Core_npix = final_Core_npix[0:featurecount]
        final_Secondary_npix = final_Secondary_npix[0:featurecount]

        final_CoreSecondary_npix = final_Core_npix + final_Secondary_npix

    ######################################################################
    # If no core is found, use secondary threshold to identify features
    else:
        # Label connected secondary-threshold regions
        feature_number2d, nFeature = label(secondary_flag_orig)

        ##########################################################
        # Loop through features and only keep those exceeding area threshold
        if nFeature > 0:
            labeled_core_secondary = np.zeros((ny, nx), dtype=int)
            labelcore_npix = np.ones(nFeature, dtype=int) * -9999
            labelSecondary_npix = np.ones(nFeature, dtype=int) * -9999
            labelTertiary_npix = np.ones(nFeature, dtype=int) * -9999
            featurecount = 0

            for ifeature in range(1, nFeature + 1):
                feature_indices = np.where(feature_number2d == ifeature)
                nfeatureindices = np.shape(feature_indices)[1]

                if nfeatureindices > 0:
                    temp_core = np.copy(core_flag[feature_indices])
                    temp_corenpix = np.nansum(temp_core)

                    temp_secondary = np.copy(secondary_flag[feature_indices])
                    temp_secondary_npix = np.nansum(temp_secondary)

                    # Check if feature exceeds area threshold
                    if use_grid_area:
                        feature_area = np.sum(pixel_area[feature_indices])
                        passes_thresh = feature_area >= area_thresh
                    else:
                        passes_thresh = temp_corenpix + temp_secondary_npix >= nthresh
                    if passes_thresh:
                        featurecount = featurecount + 1

                        labeled_core_secondary[feature_indices] = np.copy(
                            featurecount
                        )
                        labelcore_npix[featurecount - 1] = np.copy(temp_corenpix)
                        labelSecondary_npix[featurecount - 1] = np.copy(temp_secondary_npix)

            ###############################
            # Update feature count
            nFeature = np.copy(featurecount)
            labelFeature_number1d = (
                np.array(np.where(labelcore_npix + labelSecondary_npix > 0))[0, :] + 1
            )

            ###########################################################
            # Reduce size of final arrays so only as long as number of valid features
            if nFeature > 0:
                labelcore_npix = labelcore_npix[0:nFeature]
                labelSecondary_npix = labelSecondary_npix[0:nFeature]
                labelTertiary_npix = labelTertiary_npix[0:nFeature]

                ##########################################################
                # Reorder based on size, largest to smallest
                labelFeature_npix = labelcore_npix + labelSecondary_npix + labelTertiary_npix
                order = np.argsort(labelFeature_npix)
                order = order[::-1]
                sortedcore_npix = np.copy(labelcore_npix[order])
                sortedSecondary_npix = np.copy(labelSecondary_npix[order])
                sortedTertiary_npix = np.copy(labelTertiary_npix[order])

                sortedFeature_npix = np.add(sortedcore_npix, sortedSecondary_npix)

                # Re-number features
                sortedFeature_number1d = np.copy(labelFeature_number1d[order])

                sorted_features = np.zeros((ny, nx), dtype=int)
                featureStep = 0
                for isortedFeature in range(0, nFeature):
                    sortedFeature_indices = np.where(
                        labeled_core_secondary
                        == sortedFeature_number1d[isortedFeature]
                    )
                    nsortedFeatureIndices = np.shape(sortedFeature_indices)[1]
                    if nsortedFeatureIndices == sortedFeature_npix[isortedFeature]:
                        featureStep = featureStep + 1
                        sorted_features[sortedFeature_indices] = np.copy(
                            featureStep
                        )
            else:
                # label() found >=1 connected secondary-threshold component
                # (outer nFeature was >0 on entry to this branch), but none
                # of them passed the area threshold, so featurecount stayed
                # 0 and nFeature was reset to 0 just above. sortedcore_npix/
                # sortedSecondary_npix/sortedTertiary_npix are only assigned
                # inside this if - without this else they're unbound here
                # (crashing with UnboundLocalError on the very next lines),
                # since the top-of-function defaults use different casing
                # (sortedsecondary_npix/sortedtertiary_npix) and are never
                # consulted. Empty-features case - same convention as the
                # "outer nFeature == 0 from the start" branch below.
                sortedcore_npix = np.zeros((1,), dtype=int)
                sortedSecondary_npix = np.zeros((1,), dtype=int)
                sortedTertiary_npix = np.zeros((1,), dtype=int)

            ##############################################
            # Save final matrices
            final_CoreSecondary_Number = np.copy(sorted_features)
            final_nFeature = np.copy(nFeature)
            final_Core_npix = np.copy(sortedcore_npix)
            final_Secondary_npix = np.copy(sortedSecondary_npix)
            final_Tertiary_npix = np.copy(sortedTertiary_npix)
            final_CoreSecondary_npix = final_Core_npix + final_Secondary_npix
        else:
            final_CoreSecondary_Number = np.zeros((ny, nx), dtype=int)
            final_feature_number = np.zeros((ny, nx), dtype=int)
            final_nFeature = 0
            final_Core_npix = np.zeros((1,), dtype=int)
            final_Secondary_npix = np.zeros((1,), dtype=int)
            final_Tertiary_npix = np.zeros((1,), dtype=int)
            final_CoreSecondary_npix = np.zeros((1,), dtype=int)

    ###################################################
    # Get tertiary expansion, if applicable
    if final_nFeature > 0:
        if expand_to_tertiary == 1:
            final_feature_number = _expand_tertiary(
                final_CoreSecondary_Number, final_CoreSecondary_npix, final_nFeature,
                field, tertiary_thresh, core_operator, ny, nx,
            )
            # Compute tertiary pixel counts
            final_Tertiary_npix = np.zeros(len(final_CoreSecondary_npix), dtype=int)
            for ifeature in range(1, final_nFeature + 1):
                idx = ifeature - 1
                if idx < len(final_Tertiary_npix):
                    total = np.count_nonzero(final_feature_number == ifeature)
                    final_Tertiary_npix[idx] = total - final_CoreSecondary_npix[idx]

        #######################################################################
        # If not expanding to tertiary, just copy core+secondary data
        else:
            final_feature_number = np.copy(final_CoreSecondary_Number)

    ##################################################################
    # Adjust axes back to original shape if PBC was applied
    if pbc_direction != "none":
        # Adjust labeled arrays back to original dimensions
        final_feature_number = call_adjust_axis(
            final_feature_number, field_orig, config, padded_x, padded_y,
        )
        feature_type_map = call_adjust_axis(
            feature_type_map, field_orig, config, padded_x, padded_y,
        )
        final_CoreSecondary_Number = call_adjust_axis(
            final_CoreSecondary_Number, field_orig, config, padded_x, padded_y,
        )

        # Update dimensions back to original
        ny, nx = field_orig.shape

        # Recalculate feature counts based on adjusted labels.
        # The cropped label set can be sparse (e.g. [8, 11] instead of
        # [1, 2]) since only a subset of the padded domain's labels survive
        # the crop. Renumber both label arrays to contiguous 1..N *before*
        # counting, so every downstream `label - 1` positional read (in
        # gettracks.py, netcdf_io.py, tracksingle_drift.py) stays valid -
        # previously only final_nFeature/the npix arrays were recomputed
        # here, while the label arrays themselves kept their sparse values.
        labels = np.unique(
            np.concatenate(
                [final_feature_number.ravel(), final_CoreSecondary_Number.ravel()]
            )
        )
        labels = labels[labels != 0]  # Exclude background label 0
        final_nFeature = len(labels)

        label_to_index = {lbl: idx for idx, lbl in enumerate(labels)}
        lut = np.zeros(
            int(labels.max()) + 1 if final_nFeature > 0 else 1,
            dtype=final_feature_number.dtype,
        )
        for lbl in labels:
            lut[lbl] = label_to_index[lbl] + 1

        def _densify(arr):
            out = np.zeros_like(arr)
            nz = arr > 0
            out[nz] = lut[arr[nz]]
            return out

        final_feature_number = _densify(final_feature_number)
        final_CoreSecondary_Number = _densify(final_CoreSecondary_Number)

        # Initialize arrays to hold counts
        final_Core_npix = np.zeros(final_nFeature, dtype=int)
        final_Secondary_npix = np.zeros(final_nFeature, dtype=int)
        final_Tertiary_npix = np.zeros(final_nFeature, dtype=int)
        final_CoreSecondary_npix = np.zeros(final_nFeature, dtype=int)

        # Use the original (unpadded) flag arrays for counting, now against
        # the renumbered final_feature_number so idx lines up with
        # new_label - 1 directly.
        for new_label in range(1, final_nFeature + 1):
            idx = new_label - 1
            label_mask = final_feature_number == new_label

            core_pixels = np.sum(core_flag_orig[label_mask])
            cold_pixels = np.sum(secondary_flag_orig[label_mask])
            total_pixels = np.sum(label_mask)
            warm_pixels = total_pixels - core_pixels - cold_pixels

            final_Core_npix[idx] = core_pixels
            final_Secondary_npix[idx] = cold_pixels
            final_Tertiary_npix[idx] = warm_pixels
            final_CoreSecondary_npix[idx] = core_pixels + cold_pixels
    else:
        # No adjustment needed
        final_nFeature = final_nFeature

    ###################################################################
    # Output data
    return {
        "final_nFeature": final_nFeature,
        "final_Core_npix": final_Core_npix,
        "final_Secondary_npix": final_Secondary_npix,
        "final_CoreSecondary_npix": final_CoreSecondary_npix,
        "final_Tertiary_npix": final_Tertiary_npix,
        "final_Feature_Number": final_feature_number,
        "final_Feature_Type": feature_type_map,
        "final_CoreSecondary_Number": final_CoreSecondary_Number,
    }


# ---------------------------------------------------------------------------
# Growth methods
# ---------------------------------------------------------------------------


def _grow_bfs(labeled_cores, field, secondary_thresh, core_operator):
    """
    Grow labeled cores outward to secondary threshold using BFS (grow_cells).

    This is the original growth method. Pixels beyond the secondary threshold
    (or NaN) are marked as excluded (-1), and grow_cells expands seeds into
    the remaining unlabeled (0) pixels using breadth-first search with
    majority-voting tie-breaking.

    Args:
        labeled_cores: np.ndarray
            2D array with labeled core regions (>0), unlabeled (0).
        field: np.ndarray
            2D input field.
        secondary_thresh: float
            Threshold for secondary region.
        core_operator: str
            'lt' or 'gt'.

    Returns:
        np.ndarray: 2D array with grown labels.
    """
    result = np.copy(labeled_cores)

    # Mark excluded pixels (beyond secondary threshold or NaN)
    if core_operator == "lt":
        excluded = np.logical_or(field > secondary_thresh, np.isnan(field))
    else:
        excluded = np.logical_or(field < secondary_thresh, np.isnan(field))

    temp_storage = result[excluded]
    result[excluded] = -1

    # Grow seeds outward
    result = grow_cells(result)

    # Restore excluded pixels
    result[excluded] = temp_storage

    return result


def _grow_edt(labeled_cores, field, secondary_thresh, core_operator):
    """
    Grow labeled cores outward to secondary threshold using EDT (Voronoi).

    Each unlabeled pixel within the secondary threshold is assigned the label
    of its nearest core (Euclidean distance), but only within the same
    connected component of valid pixels. Disconnected secondary-threshold
    regions without a core remain unlabeled (0), preserving BFS connectivity
    semantics so that the downstream isolated-feature pipeline can filter
    them by size.

    Args:
        labeled_cores: np.ndarray
            2D array with labeled core regions (>0), unlabeled (0).
        field: np.ndarray
            2D input field.
        secondary_thresh: float
            Threshold for secondary region.
        core_operator: str
            'lt' or 'gt'.

    Returns:
        np.ndarray: 2D array with grown labels.
    """
    from scipy.ndimage import distance_transform_edt

    result = np.copy(labeled_cores)

    # Determine which pixels are valid for expansion (within secondary threshold)
    if core_operator == "lt":
        valid_mask = np.logical_and(field <= secondary_thresh, ~np.isnan(field))
    else:
        valid_mask = np.logical_and(field >= secondary_thresh, ~np.isnan(field))

    # Pixels that are seeds (already labeled)
    seed_mask = labeled_cores > 0

    # If no seeds, return as-is
    if not np.any(seed_mask):
        return result

    # Label connected components of valid_mask to respect connectivity.
    # Only grow within 8-connected components that contain at least one seed.
    struct_8conn = np.ones((3, 3), dtype=bool)
    valid_components, n_components = label(valid_mask, structure=struct_8conn)

    # Find which connected components contain seeds
    seeded_component_labels = np.unique(valid_components[seed_mask])
    # Remove background (0) if present
    seeded_component_labels = seeded_component_labels[seeded_component_labels > 0]

    if len(seeded_component_labels) == 0:
        return result

    # Process each seeded connected component independently.
    # For each component, EDT finds the nearest seed WITHIN that component,
    # ensuring growth never crosses invalid-pixel gaps.
    for comp_lbl in seeded_component_labels:
        comp_mask = valid_components == comp_lbl
        comp_seeds = seed_mask & comp_mask
        grow_pixels = comp_mask & ~comp_seeds

        if not np.any(grow_pixels):
            continue

        # Find bounding box for this component (performance optimization)
        rows = np.any(comp_mask, axis=1)
        cols = np.any(comp_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Crop to bounding box
        sub_seeds = comp_seeds[rmin:rmax + 1, cmin:cmax + 1]
        sub_grow = grow_pixels[rmin:rmax + 1, cmin:cmax + 1]
        sub_cores = labeled_cores[rmin:rmax + 1, cmin:cmax + 1]

        # EDT input: only this component's seeds are sources (False).
        # All other pixels (including non-component pixels within the bbox)
        # are True. The EDT finds the nearest seed in this component for
        # every pixel in the sub-array.
        edt_input = ~sub_seeds
        _, sub_idx = distance_transform_edt(
            edt_input, return_distances=True, return_indices=True,
        )

        # Map each pixel to the label of its nearest seed in this component
        nearest_sub = sub_cores[sub_idx[0], sub_idx[1]]

        # Assign labels only for non-seed pixels within this component
        result[rmin:rmax + 1, cmin:cmax + 1][sub_grow] = nearest_sub[sub_grow]

    return result


# ---------------------------------------------------------------------------
# Tertiary expansion
# ---------------------------------------------------------------------------


def _expand_tertiary(
    final_corecoldnumber, final_ncorecoldpix, final_ncorecold,
    field, tertiary_thresh, core_operator, ny, nx,
):
    """
    Expand features to tertiary threshold using iterative binary dilation.

    This replicates the original warm anvil expansion logic but generalized
    for any field direction.

    Args:
        final_corecoldnumber: np.ndarray
            2D labeled array (core + secondary).
        final_ncorecoldpix: np.ndarray
            Pixel count per feature (core + secondary).
        final_ncorecold: int
            Number of features.
        field: np.ndarray
            2D input field.
        tertiary_thresh: float
            Threshold for tertiary expansion.
        core_operator: str
            'lt' or 'gt'.
        ny, nx: int
            Dimensions.

    Returns:
        np.ndarray: 2D labeled array with tertiary expansion.
    """
    labeled_expanded = np.copy(final_corecoldnumber)
    nexpandedpix = np.copy(final_ncorecoldpix)

    keepspreading = 1
    while keepspreading > 0:
        keepspreading = 0

        # Loop through each feature
        for ifeature in range(1, final_ncorecold + 1):
            # Create map of single feature
            featuremap = np.copy(labeled_expanded)
            featuremap[labeled_expanded != ifeature] = 0
            featuremap[labeled_expanded == ifeature] = 1

            # Find maximum extent of the feature
            extenty = np.nansum(featuremap, axis=1)
            extenty = np.array(np.where(extenty > 0))[0, :]
            miny = extenty[0]
            maxy = extenty[-1]

            extentx = np.nansum(featuremap, axis=0)
            extentx = np.array(np.where(extentx > 0))[0, :]
            minx = extentx[0]
            maxx = extentx[-1]

            # Subset data to smaller region around feature with 10-pixel buffer
            if minx <= 10:
                minx = 0
            else:
                minx = minx - 10

            if maxx >= nx - 10:
                maxx = nx
            else:
                maxx = maxx + 11

            if miny <= 10:
                miny = 0
            else:
                miny = miny - 10

            if maxy >= ny - 10:
                maxy = ny
            else:
                maxy = maxy + 11

            fieldsubset = field[miny:maxy, minx:maxx]
            fullsubset = labeled_expanded[miny:maxy, minx:maxx]
            featuresubset = featuremap[miny:maxy, minx:maxx]

            # Dilate cloud region (cross-shaped, 1 pixel)
            dilationstructure = generate_binary_structure(2, 1)
            dilatedsubset = binary_dilation(
                featuresubset, structure=dilationstructure, iterations=1,
            ).astype(featuremap.dtype)

            # Isolate dilated region
            expansionzone = dilatedsubset - featuresubset

            # Only keep pixels in dilated regions that satisfy tertiary threshold
            # and are not associated with another feature
            expansionzone[
                np.where((expansionzone == 1) & (fullsubset != 0))
            ] = 0
            if core_operator == "lt":
                expansionzone[
                    np.where((expansionzone == 1) & (fieldsubset >= tertiary_thresh))
                ] = 0
            else:
                expansionzone[
                    np.where((expansionzone == 1) & (fieldsubset <= tertiary_thresh))
                ] = 0

            # Find indices of accepted dilated regions
            expansionindices = np.column_stack(np.where(expansionzone == 1))

            # Add accepted dilated region to the feature map
            labeled_expanded[
                expansionindices[:, 0] + miny, expansionindices[:, 1] + minx
            ] = ifeature

            # Add expanded pixel count
            nexpandedpix[ifeature - 1] = (
                len(expansionindices[:, 0]) + nexpandedpix[ifeature - 1]
            )

            # Track whether any feature is still growing
            keepspreading = keepspreading + len(
                np.extract(expansionzone == 1, expansionzone)
            )

    return labeled_expanded


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def find_and_label_cores(smoothed_field, core_thresh, core_operator):
    """
    Label cores using ndimage.label.

    Args:
        smoothed_field: np.ndarray
            Array containing smoothed field data.
        core_thresh: float
            Threshold to define cores.
        core_operator: str
            'lt': cores where field < threshold.
            'gt': cores where field > threshold.

    Returns:
        labeled_cores: np.ndarray
            Array containing labeled core numbers.
        nlabelcores: int
            Number of labeled cores.
    """
    core_mask = np.zeros(smoothed_field.shape, dtype=int)
    if core_operator == "lt":
        core_indices = np.where(smoothed_field < core_thresh)
    else:
        core_indices = np.where(smoothed_field > core_thresh)
    ncorepix = np.shape(core_indices)[1]
    if ncorepix > 0:
        core_mask[core_indices] = 1
    labeled_cores, nlabelcores = label(core_mask)
    return labeled_cores, nlabelcores


def smooth_field(field, smooth_size):
    """
    Smooth a 2D field with a box convolve filter.

    Args:
        field: np.ndarray
            2D input data array.
        smooth_size: int
            Width of the Box2DKernel filter.

    Returns:
        np.ndarray: Smoothed field.
    """
    kernel = Box2DKernel(smooth_size)
    smoothed = convolve(
        field, kernel, boundary="extend",
        nan_treatment="interpolate", preserve_nan=True,
    )
    return smoothed


def classify_pixels_by_thresholds(
    field, nx, ny, edge_thresh, secondary_thresh, core_thresh,
    tertiary_thresh, core_operator,
):
    """
    Classify pixels into categories based on thresholds.

    Categories:
        1 = Core (most intense)
        2 = Secondary
        3 = Tertiary
        4 = Other cloud (between tertiary and edge)
        5 = Clear (beyond edge)

    For core_operator='lt': lower values are more intense.
    For core_operator='gt': higher values are more intense.

    Args:
        field: np.ndarray
            2D input data array.
        nx, ny: int
            Dimensions.
        edge_thresh: float
            Outermost boundary threshold.
        secondary_thresh: float
            Secondary region threshold.
        core_thresh: float
            Core threshold.
        tertiary_thresh: float
            Tertiary region threshold.
        core_operator: str
            'lt' or 'gt'.

    Returns:
        secondary_flag: np.ndarray
            Binary flag for secondary region pixels.
        core_flag: np.ndarray
            Binary flag for core pixels.
        cloud_type_map: np.ndarray
            Pixel classification (1-5).
    """
    cloud_type_map = np.zeros((ny, nx), dtype=int)
    core_flag = np.zeros((ny, nx), dtype=int)
    secondary_flag = np.zeros((ny, nx), dtype=int)

    if core_operator == "lt":
        # Lower values are more intense
        # Core: field < core_thresh
        core_indices = np.where(field < core_thresh)
        ncorepix = np.shape(core_indices)[1]
        if ncorepix > 0:
            core_flag[core_indices] = 1
            cloud_type_map[core_indices] = 1

        # Secondary: core_thresh <= field < secondary_thresh
        secondary_indices = np.where(
            (field >= core_thresh) & (field < secondary_thresh)
        )
        nsecondarypix = np.shape(secondary_indices)[1]
        if nsecondarypix > 0:
            secondary_flag[secondary_indices] = 1
            cloud_type_map[secondary_indices] = 2

        # Tertiary: secondary_thresh <= field < tertiary_thresh
        tertiary_indices = np.where(
            (field >= secondary_thresh) & (field < tertiary_thresh)
        )
        ntertiarypix = np.shape(tertiary_indices)[1]
        if ntertiarypix > 0:
            cloud_type_map[tertiary_indices] = 3

        # Other: tertiary_thresh <= field < edge_thresh
        other_indices = np.where(
            (field >= tertiary_thresh) & (field < edge_thresh)
        )
        notherpix = np.shape(other_indices)[1]
        if notherpix > 0:
            cloud_type_map[other_indices] = 4

        # Clear: field >= edge_thresh
        clear_indices = np.where(field >= edge_thresh)
        nclearpix = np.shape(clear_indices)[1]
        if nclearpix > 0:
            cloud_type_map[clear_indices] = 5

    else:
        # Higher values are more intense (e.g., reflectivity)
        # Core: field > core_thresh
        core_indices = np.where(field > core_thresh)
        ncorepix = np.shape(core_indices)[1]
        if ncorepix > 0:
            core_flag[core_indices] = 1
            cloud_type_map[core_indices] = 1

        # Secondary: secondary_thresh < field <= core_thresh
        secondary_indices = np.where(
            (field <= core_thresh) & (field > secondary_thresh)
        )
        nsecondarypix = np.shape(secondary_indices)[1]
        if nsecondarypix > 0:
            secondary_flag[secondary_indices] = 1
            cloud_type_map[secondary_indices] = 2

        # Tertiary: tertiary_thresh < field <= secondary_thresh
        tertiary_indices = np.where(
            (field <= secondary_thresh) & (field > tertiary_thresh)
        )
        ntertiarypix = np.shape(tertiary_indices)[1]
        if ntertiarypix > 0:
            cloud_type_map[tertiary_indices] = 3

        # Other: edge_thresh < field <= tertiary_thresh
        other_indices = np.where(
            (field <= tertiary_thresh) & (field > edge_thresh)
        )
        notherpix = np.shape(other_indices)[1]
        if notherpix > 0:
            cloud_type_map[other_indices] = 4

        # Clear: field <= edge_thresh
        clear_indices = np.where(field <= edge_thresh)
        nclearpix = np.shape(clear_indices)[1]
        if nclearpix > 0:
            cloud_type_map[clear_indices] = 5

    return secondary_flag, core_flag, cloud_type_map
