import numpy as np

def calc_cloud_boundary(height, idxcld, gap, min_thick):
    """
    Calculates cloud base, cloud top height
    ----------
    height: np.array(float)
        height array (1D)
    idxcld: np.array(int)
        Index of height containing cloud.
    gap: int
        If a gap larger than this exists, clouds are separated into different layers
    min_thick: float
        Minimum thickness of a cloud layer.

    Returns
    ----------
    cloud_base: np.ndarray(nLayers)
        Cloud-base height for each cloud layer.
    cloud_top: np.ndarray(nLayers)
        Cloud-top height for each cloud layer.
    """

    # Handle empty idxcld
    if len(idxcld) == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    # Split idxcld into layers
    Layers = np.split(idxcld, np.where(np.diff(idxcld) > gap)[0]+1)
    # Filter out empty sublists that np.split can produce
    Layers = [L for L in Layers if len(L) > 0]
    nLayers = len(Layers)

    # Create cloud_base, cloud_top arrays
    cloud_base = np.zeros(nLayers, dtype=np.float32)
    cloud_top = np.zeros(nLayers, dtype=np.float32)

    if nLayers > 0:
        # Loop over each layer
        for iLayer in range(0, nLayers):

            # Calculate layer thickness
            zb = height[Layers[iLayer][0]]
            zt = height[Layers[iLayer][-1]]
            dz = zt - zb
            # Alternatively, a layer could be required to be thicker than min_thick here:
            # if (dz > min_thick):
            cloud_base[iLayer] = zb
            cloud_top[iLayer] = zt

    return cloud_base, cloud_top

def echotop_height(dbz3d, height, z_dimname, shape_2d, dbz_thresh, gap, min_thick):
    """
    Calculates first layer echo-top height from bottom up.
    ----------
    dbz3d: np.DataArray(float)
        3D reflectivity array (Xarray DataArray), assumes in [z, y, x] order.
    height: np.array(float)
        height array (1D or 3D). For 1D, assumes vertical coordinate only.
        For 3D, assumes in the same order as dbz3d [z, y, x] order.
    shape_2d: tuple 
        (Number of points on x-direction, Number of points on y-direction)
    nz: float
        Number of points on z-direction.
    dbz_thresh: float
        Reflectivity threshold to calculate echo-top height.
    gap: int
        If a gap larger than this exists, echoes are separated into different layers
    min_thick: float
        Minimum thickness of an echo layer.

    Returns
    ----------
    echotop: np.ndarray(float)
        Echo-top height 2D array.
    """

    # Create echo-top height array
    echotop = np.full(shape_2d, np.nan, dtype=np.float32)

    # Define binary cloud mask using reflectivity threshold
    cloudmask = dbz3d > dbz_thresh
    # Get numpy array for speed
    cmask = cloudmask.squeeze().values

    # Find 2D locations with clouds anywhere in the column
    # Then get y, x indices of those cloudy points
    yidx, xidx = np.where(cloudmask.max(dim=z_dimname).squeeze().values == 1)
    # Total number of points with clouds
    npix_cloud = len(xidx)

    # Check if height is 1D or 3D
    height_is_1d = height.ndim == 1

    # Loop over each point with clouds
    for il in range(0, npix_cloud):
        # Get the profile at location y, x
        idxcld = np.array(np.where(np.squeeze(cmask[:, yidx[il], xidx[il]]) == 1)[0])
        # Check if there is any clouds defined
        if len(idxcld) > 0:
            # Get height profile: use 1D height directly or extract column from 3D height
            if height_is_1d:
                iheight = height
            else:
                iheight = height[:, yidx[il], xidx[il]]
            # Call calc_cloud_boundary to get cloud_base, cloud_top height
            cb, ct = calc_cloud_boundary(iheight, idxcld, gap, min_thick)
            # The first layer top (from bottom up) is the lowest layer echo-top height
            echotop[yidx[il], xidx[il]] = ct[0]

    return echotop


def echotop_height_fast(dbz3d, height, z_dimname, shape_2d, dbz_thresh, gap, min_thick):
    """
    Vectorized version of echotop_height.  Produces results numerically identical
    to echotop_height (within float32 precision).

    Instead of looping over every cloudy pixel in Python (~npix_cloud iterations),
    this function scans the nz vertical levels once with fully vectorized NumPy
    operations.  For a 2775×2145 domain at 100 m resolution the loop count falls
    from ~6 million to ~25, giving 100–1000× less Python overhead.

    Algorithm equivalence (matches calc_cloud_boundary/echotop_height exactly):
      For each column, the original splits the sorted list of cloud-level indices
      wherever np.diff(idxcld) > gap, then returns height[Layers[0][-1]].
      Here we replicate that logic level-by-level:
        - track last_cloud_z = most recent level with cloud (while in first layer)
        - at the next cloud level, if (k - last_cloud_z) > gap → first layer ended
        - first_layer_top_idx records the highest k within the first layer

    Parameters  (identical to echotop_height)
    ----------
    dbz3d: xarray.DataArray
        3-D reflectivity [z, y, x].
    height: np.ndarray
        1-D or 3-D height array.
    z_dimname: str
        Name of the z dimension in dbz3d.
    shape_2d: tuple
        (ny, nx) of the 2-D output domain.
    dbz_thresh: float
        Reflectivity threshold (dBZ).
    gap: int
        Maximum gap (in vertical levels) allowed within one echo layer.
        A gap of N means N or fewer consecutive non-cloud levels are bridged.
        Equivalent to np.diff(idxcld) > gap used in calc_cloud_boundary.
    min_thick: float
        Unused (kept for API compatibility with echotop_height).

    Returns
    -------
    echotop: np.ndarray float32 shape (ny, nx)
        First-layer echo-top height from bottom up.  NaN where no echo found.
    """
    # Binary cloud mask [nz, ny, nx]
    cmask = (dbz3d > dbz_thresh).squeeze().values  # bool

    nz = cmask.shape[0]
    ny, nx = shape_2d
    height_is_1d = height.ndim == 1

    # Output array
    echotop = np.full(shape_2d, np.nan, dtype=np.float32)

    # Per-column trackers (2-D arrays over the horizontal domain)
    # last_cloud_z: most recent level with cloud while still in the first layer.
    #   Initialized to -1 (no cloud seen yet).  Frozen once passed_gap=True so
    #   that subsequent layers do not corrupt the gap-detection arithmetic.
    last_cloud_z = np.full((ny, nx), -1, dtype=np.intp)
    # passed_gap: True once the first inter-layer gap has been detected
    passed_gap = np.zeros((ny, nx), dtype=bool)
    # first_layer_top_idx: highest level index in the first layer (-1 = no cloud)
    first_layer_top_idx = np.full((ny, nx), -1, dtype=np.intp)

    for k in range(nz):
        cloud_at_k = cmask[k]  # bool [ny, nx]

        # Detect a layer-breaking gap:
        #   - we've seen cloud in this column before (last_cloud_z >= 0), AND
        #   - there is cloud at level k, AND
        #   - the jump from the last cloud level exceeds `gap`
        # (mirrors: np.diff(idxcld) > gap in calc_cloud_boundary)
        cloud_seen = last_cloud_z >= 0
        gap_too_large = cloud_seen & cloud_at_k & ((k - last_cloud_z) > gap)
        passed_gap |= gap_too_large

        # This level belongs to the first layer only if it has cloud AND
        # we have not yet crossed a layer-breaking gap.
        in_first_layer = cloud_at_k & ~passed_gap

        # Update the first-layer top index
        first_layer_top_idx = np.where(in_first_layer, k, first_layer_top_idx)

        # Advance last_cloud_z ONLY while still in the first layer.
        # Freezing it after passed_gap prevents later cloud layers from
        # changing the gap-detection arithmetic for this column.
        last_cloud_z = np.where(in_first_layer, k, last_cloud_z)

    # Assign echo-top heights for columns that have a valid first layer
    valid = first_layer_top_idx >= 0
    if valid.any():
        if height_is_1d:
            echotop[valid] = height[first_layer_top_idx[valid]]
        else:
            y_idx, x_idx = np.where(valid)
            echotop[y_idx, x_idx] = height[
                first_layer_top_idx[y_idx, x_idx], y_idx, x_idx
            ]

    return echotop


def echotop_height_wrf(dbz3d, height, z_dimname, shape_2d, dbz_thresh, gap, min_thick):
    """
    Calculates first layer echo-top height from bottom up for WRF.
    
    DEPRECATED: This function is kept for backward compatibility.
    Use echotop_height() instead, which now handles both 1D and 3D height arrays.
    
    ----------
    dbz3d: np.DataArray(float)
        3D reflectivity array (Xarray DataArray), assumes in [z, y, x] order.
    height: np.array(float)
        height array (3D), assumes in the same order as dbz3d [z, y, x] order.
    shape_2d: tuple 
        (Number of points on x-direction, Number of points on y-direction)
    nz: float
        Number of points on z-direction.
    dbz_thresh: float
        Reflectivity threshold to calculate echo-top height.
    gap: int
        If a gap larger than this exists, echoes are separated into different layers
    min_thick: float
        Minimum thickness of an echo layer.

    Returns
    ----------
    echotop: np.ndarray(float)
        Echo-top height 2D array.
    """
    # Call the unified echotop_height function
    return echotop_height(dbz3d, height, z_dimname, shape_2d, dbz_thresh, gap, min_thick)