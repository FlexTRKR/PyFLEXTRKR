"""
Trajectory smoothing module for handling geographic coordinate data.

This module provides functions to smooth and clean trajectory data by removing outliers
based on physical speed constraints and interpolating missing values.

Original code by Wojciech Szkółka <wojtek25495@gmail.com>
Modified by Zhe Feng <zhe.feng@pnnl.gov>
"""
import numpy as np

# Function to calculate the great-circle distance between two geographic points (Haversine formula)
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth using the Haversine formula.

    Args:
        lat1, lon1 : float
            Latitude and longitude of point 1 in decimal degrees.
        lat2, lon2 : float
            Latitude and longitude of point 2 in decimal degrees.
    
    Returns:
        float
            Distance between the two points in kilometers.
    """
    R = 6371.0  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def remove_outliers_stepwise(longitudes, latitudes, max_speed_kmh=100, time_step_h=1):
    """
    Iteratively remove outliers from the trajectory based on speed threshold.

    The function identifies points that would require unrealistic movement speeds
    to reach from the previous valid point and marks them as NaN.

    Args:
        longitudes : array-like
            Array of longitudes.
        latitudes : array-like
            Array of latitudes.
        max_speed_kmh : float
            Maximum allowed speed in km/h.
        time_step_h : float
            Time step in hours between consecutive points.
    
    Returns:
        tuple
            Tuple of arrays (longitudes, latitudes) with outliers removed.
    """
    longitudes = np.array(longitudes, dtype=np.float64)
    latitudes = np.array(latitudes, dtype=np.float64)

    while True:
        found_outlier = False
        prev_valid_idx = None

        for i in range(len(longitudes)):
            # Skip already marked NaN points
            if np.isnan(longitudes[i]) or np.isnan(latitudes[i]):
                continue
            # First valid point becomes reference point
            if prev_valid_idx is None:
                prev_valid_idx = i
                continue

            # Calculate distance and speed between current point and last valid point
            distance_km = haversine(latitudes[prev_valid_idx], longitudes[prev_valid_idx], latitudes[i], longitudes[i])
            time_diff = (i - prev_valid_idx) * time_step_h
            speed_kmh = distance_km / time_diff if time_diff > 0 else float('inf')

            # Mark as outlier if speed exceeds threshold
            if speed_kmh > max_speed_kmh:
                longitudes[i] = np.nan
                latitudes[i] = np.nan
                found_outlier = True
                break  # Restart iteration after removing an outlier

            prev_valid_idx = i

        # Exit loop when no more outliers are found
        if not found_outlier:
            break

    return longitudes, latitudes

def interpolate_missing_values(longitudes, latitudes):
    """
    Interpolate missing values in the trajectory using linear interpolation.

    This fills in NaN values by creating straight-line paths between valid points.
    Only interpolates between the first and last valid points in the series.

    Args:
        longitudes : array-like
            Array of longitudes.
        latitudes : array-like
            Array of latitudes.

    Returns:
        tuple
            Tuple of arrays (longitudes, latitudes) with NaN values interpolated.
    """
    indices = np.arange(len(longitudes))
    valid_indices = ~np.isnan(longitudes) & ~np.isnan(latitudes)

    if np.any(valid_indices):
        # Find the first and last valid points
        first_valid, last_valid = np.where(valid_indices)[0][[0, -1]]
        # Interpolate between first and last valid points
        longitudes[first_valid:last_valid+1] = np.interp(indices[first_valid:last_valid+1], indices[valid_indices], longitudes[valid_indices])
        latitudes[first_valid:last_valid+1] = np.interp(indices[first_valid:last_valid+1], indices[valid_indices], latitudes[valid_indices])

    return longitudes, latitudes

def smooth_trajectory(longitudes, latitudes, max_speed_kmh=100, time_step_h=1):
    """
    Smooth the trajectory by removing outliers and interpolating missing values.

    This function uses a multi-variant approach to find the optimal smoothing:
    1. Process the entire trajectory
    2. Process from the second point (preserving first point)
    3. Process from the third point (preserving first two points)
    
    It then selects the best variant that minimizes data loss while prioritizing
    the integrity of the starting points.

    Args:
        longitudes : array-like
            Array of longitudes.
        latitudes : array-like
            Array of latitudes.
        max_speed_kmh : float
            Maximum allowed speed in km/h.
        time_step_h : float
            Time step in hours between consecutive points.
    
    Returns:
        tuple
            Tuple of arrays (longitudes, latitudes) with outliers removed and NaN values interpolated.
    """
    def apply_variant(lons, lats):
        # Remove points that exceed maximum speed threshold
        lons, lats = remove_outliers_stepwise(lons, lats, max_speed_kmh, time_step_h)
        # Check if the first point was removed (became NaN)
        first_removed = np.isnan(lons[0])
        # Count total number of NaN values to assess how much data was removed
        nan_count = np.sum(np.isnan(lons)) + np.sum(np.isnan(lats))
        return lons, lats, first_removed, nan_count

    # Store original values for first two points
    orig_lon = np.array(longitudes[:2]) if len(longitudes) >= 2 else np.array(longitudes)
    orig_lat = np.array(latitudes[:2]) if len(latitudes) >= 2 else np.array(latitudes)

    # Variant 1: Process the entire trajectory
    full_longitudes, full_latitudes, full_first_removed, full_nan_count = apply_variant(longitudes, latitudes)

    # Variant 2: Process trajectory starting from second point (preserve first point integrity)
    if len(longitudes) > 1:
        partial_longitudes, partial_latitudes, partial_first_removed, partial_nan_count = apply_variant(longitudes[1:], latitudes[1:])
        partial_longitudes = np.insert(partial_longitudes, 0, np.nan)
        partial_latitudes = np.insert(partial_latitudes, 0, np.nan)
    else:
        partial_longitudes, partial_latitudes, partial_first_removed, partial_nan_count = full_longitudes, full_latitudes, False, full_nan_count

    # Variant 3: Process trajectory starting from third point (preserve first two points integrity)
    if len(longitudes) > 2:
        partial2_longitudes, partial2_latitudes, partial2_first_removed, partial2_nan_count = apply_variant(longitudes[2:], latitudes[2:])
        partial2_longitudes = np.insert(partial2_longitudes, 0, [np.nan, np.nan])
        partial2_latitudes = np.insert(partial2_latitudes, 0, [np.nan, np.nan])
    else:
        partial2_longitudes, partial2_latitudes, partial2_first_removed, partial2_nan_count = full_longitudes, full_latitudes, False, full_nan_count

    # Store all processing variants as candidates for selection
    candidates = [
        (full_longitudes, full_latitudes, full_first_removed, full_nan_count),
        (partial_longitudes, partial_latitudes, partial_first_removed, partial_nan_count),
        (partial2_longitudes, partial2_latitudes, partial2_first_removed, partial2_nan_count),
    ]

    # Select best variant: prioritize keeping the first point, then minimize total NaNs
    best_variant = min(candidates, key=lambda x: (not x[2], x[3]))
    # Interpolate missing values in the selected variant
    best_longitudes, best_latitudes = interpolate_missing_values(best_variant[0], best_variant[1])

    # Preserve original values for first two points if they're NaN after smoothing
    if len(best_longitudes) >= 1 and np.isnan(best_longitudes[0]) and not np.isnan(orig_lon[0]):
        best_longitudes[0] = orig_lon[0]
        best_latitudes[0] = orig_lat[0]
    
    if len(best_longitudes) >= 2 and np.isnan(best_longitudes[1]) and not np.isnan(orig_lon[1]):
        best_longitudes[1] = orig_lon[1]
        best_latitudes[1] = orig_lat[1]

    return best_longitudes, best_latitudes

def interpolate_same_indices(a, A, b, B, c):
    """
    Interpolate variable values (like intensities) to fit new smoothed coordinates.
    
    This function adjusts supplementary data arrays to match the modified trajectory
    by applying interpolation at the same indices where coordinates were changed.
    
    Args:
        a : array
            Original longitudes.
        A : array
            Smoothed longitudes.
        b : array
            Original latitudes.
        B : array
            Smoothed latitudes.
        c : array
            Supplementary data to interpolate (e.g., intensity values).
            
    Returns:
        array
            Interpolated supplementary data that aligns with the smoothed coordinates.
    """
    if c is None or np.isscalar(c) or len(c) == 0:
        return c

    c = np.array(c, dtype=np.float64)
    # Find indices where coordinates were modified
    modified_indices = np.where((A != a) | (B != b) | np.isnan(a) | np.isnan(b))[0]
    # Find indices where all values are valid in both original and smoothed data
    valid_indices = np.where((A == a) & (B == b) & ~np.isnan(a) & ~np.isnan(b) & ~np.isnan(c))[0]

    if len(valid_indices) > 1 and len(modified_indices) > 0:
        # Only interpolate between the first and last valid points
        first_valid, last_valid = valid_indices[0], valid_indices[-1]
        interp_indices = modified_indices[(modified_indices >= first_valid) & (modified_indices <= last_valid)]
        if len(interp_indices) > 0:
            c[interp_indices] = np.interp(interp_indices, valid_indices, c[valid_indices])

    return c