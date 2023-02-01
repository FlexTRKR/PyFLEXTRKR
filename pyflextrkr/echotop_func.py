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

    # Split idxcld into layers
    Layers = np.split(idxcld, np.where(np.diff(idxcld) > gap)[0]+1)
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
        height array (1D)
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

    # Loop over each point with clouds
    for il in range(0, npix_cloud):
        # Get the profile at location y, x
        idxcld = np.array(np.where(np.squeeze(cmask[:, yidx[il], xidx[il]]) == 1)[0])
        # Check if there is any clouds defined
        if len(idxcld) > 0:
            # Call calc_cloud_boundary to get cloud_base, cloud_top height
            cb, ct = calc_cloud_boundary(height, idxcld, gap, min_thick)
            # The first layer top (from bottom up) is the lowest layer echo-top height
            echotop[yidx[il], xidx[il]] = ct[0]

    return echotop


def echotop_height_wrf(dbz3d, height, z_dimname, shape_2d, dbz_thresh, gap, min_thick):
    """
    Calculates first layer echo-top height from bottom up for WRF.
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

    # Loop over each point with clouds
    for il in range(0, npix_cloud):
        # Get the profile at location y, x
        idxcld = np.array(np.where(np.squeeze(cmask[:, yidx[il], xidx[il]]) == 1)[0])
        iheight = height[:, yidx[il], xidx[il]]
        # Check if there is any clouds defined
        if len(idxcld) > 0:
            # Call calc_cloud_boundary to get cloud_base, cloud_top height
            cb, ct = calc_cloud_boundary(iheight, idxcld, gap, min_thick)
            # The first layer top (from bottom up) is the lowest layer echo-top height
            echotop[yidx[il], xidx[il]] = ct[0]

    return echotop