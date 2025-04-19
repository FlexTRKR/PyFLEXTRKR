import numpy as np

# Function to calculate the great-circle distance between two geographic points (Haversine formula)
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth.

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

# Function to iteratively remove outliers if movement speed exceeds defined threshold
def remove_outliers_stepwise(longitudes, latitudes, max_speed_kmh=100, time_step_h=1):
    """
    Iteratively remove outliers from the trajectory based on speed threshold.

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
            if np.isnan(longitudes[i]) or np.isnan(latitudes[i]):
                continue
            if prev_valid_idx is None:
                prev_valid_idx = i
                continue

            distance_km = haversine(latitudes[prev_valid_idx], longitudes[prev_valid_idx], latitudes[i], longitudes[i])
            time_diff = (i - prev_valid_idx) * time_step_h
            speed_kmh = distance_km / time_diff if time_diff > 0 else float('inf')

            if speed_kmh > max_speed_kmh:
                longitudes[i] = np.nan
                latitudes[i] = np.nan
                found_outlier = True
                break  # Restart iteration after removing an outlier

            prev_valid_idx = i

        if not found_outlier:
            break

    return longitudes, latitudes

# Function to interpolate missing (NaN) values linearly between valid points
def interpolate_missing_values(longitudes, latitudes):
    """
    Interpolate missing values in the trajectory using linear interpolation.

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
        first_valid, last_valid = np.where(valid_indices)[0][[0, -1]]
        longitudes[first_valid:last_valid+1] = np.interp(indices[first_valid:last_valid+1], indices[valid_indices], longitudes[valid_indices])
        latitudes[first_valid:last_valid+1] = np.interp(indices[first_valid:last_valid+1], indices[valid_indices], latitudes[valid_indices])

    return longitudes, latitudes

# Main function to smooth the trajectory and select the optimal variant
def smooth_trajectory(longitudes, latitudes, max_speed_kmh=100, time_step_h=1):
    """
    Smooth the trajectory by removing outliers and interpolating missing values.

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
        lons, lats = remove_outliers_stepwise(lons, lats, max_speed_kmh, time_step_h)
        first_removed = np.isnan(lons[0])
        nan_count = np.sum(np.isnan(lons)) + np.sum(np.isnan(lats))
        return lons, lats, first_removed, nan_count

    full_longitudes, full_latitudes, full_first_removed, full_nan_count = apply_variant(longitudes, latitudes)

    if len(longitudes) > 1:
        partial_longitudes, partial_latitudes, partial_first_removed, partial_nan_count = apply_variant(longitudes[1:], latitudes[1:])
        partial_longitudes = np.insert(partial_longitudes, 0, np.nan)
        partial_latitudes = np.insert(partial_latitudes, 0, np.nan)
    else:
        partial_longitudes, partial_latitudes, partial_first_removed, partial_nan_count = full_longitudes, full_latitudes, False, full_nan_count

    if len(longitudes) > 2:
        partial2_longitudes, partial2_latitudes, partial2_first_removed, partial2_nan_count = apply_variant(longitudes[2:], latitudes[2:])
        partial2_longitudes = np.insert(partial2_longitudes, 0, [np.nan, np.nan])
        partial2_latitudes = np.insert(partial2_latitudes, 0, [np.nan, np.nan])
    else:
        partial2_longitudes, partial2_latitudes, partial2_first_removed, partial2_nan_count = full_longitudes, full_latitudes, False, full_nan_count

    candidates = [
        (full_longitudes, full_latitudes, full_first_removed, full_nan_count),
        (partial_longitudes, partial_latitudes, partial_first_removed, partial_nan_count),
        (partial2_longitudes, partial2_latitudes, partial2_first_removed, partial2_nan_count),
    ]

    best_variant = min(candidates, key=lambda x: (not x[2], x[3]))
    best_longitudes, best_latitudes = interpolate_missing_values(best_variant[0], best_variant[1])

    return best_longitudes, best_latitudes

# Interpolates variable values (like intensities) to fit new smoothed coordinates
def interpolate_same_indices(a, A, b, B, c):
    if c is None or np.isscalar(c) or len(c) == 0:
        return c

    c = np.array(c, dtype=np.float64)
    modified_indices = np.where((A != a) | (B != b) | np.isnan(a) | np.isnan(b))[0]
    valid_indices = np.where((A == a) & (B == b) & ~np.isnan(a) & ~np.isnan(b) & ~np.isnan(c))[0]

    if len(valid_indices) > 1 and len(modified_indices) > 0:
        first_valid, last_valid = valid_indices[0], valid_indices[-1]
        interp_indices = modified_indices[(modified_indices >= first_valid) & (modified_indices <= last_valid)]
        if len(interp_indices) > 0:
            c[interp_indices] = np.interp(interp_indices, valid_indices, c[valid_indices])

    return c