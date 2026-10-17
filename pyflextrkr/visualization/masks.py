"""
Pixel masks of tracked objects: the structure that sets the perimeter width, the perimeter of every object (labeled with
its track number) and the center of every object.

The functions are the ones that were copied into every plot_subset_*.py script. label_perimeter and the centers work on
the bounding box of each object instead of the whole grid (a global 10-km frame has 4.3 million pixels and a hundred
objects); the result is the same as the loop over the whole grid (checked in tests/test_viz_masks.py).

Called by: mcs_map, cell_map, generic_map (work_for_time_loop). Calls: numpy, scipy.ndimage.
"""
import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure, find_objects

#--------------------------------------------------------------------------
def make_dilation_structure(dilate_radius, dx, dy):
    """
    Make a circular dilation structure

    Args:
        dilate_radius: float
            Dilation radius [kilometer].
        dx: float
            Grid spacing in x-direction [kilometer].
        dy: float
            Grid spacing in y-direction [kilometer].

    Returns:
        struc: np.array
            Dilation structure array.
    """
    # Convert radius to number grids
    rad_gridx = int(dilate_radius / dx)
    rad_gridy = int(dilate_radius / dy)
    xgrd, ygrd = np.ogrid[-rad_gridx:rad_gridx+1, -rad_gridy:rad_gridy+1]
    # Make dilation structure
    strc = xgrd*xgrd + ygrd*ygrd <= (dilate_radius / dx) * (dilate_radius / dy)
    return strc

#--------------------------------------------------------------------------
def make_perimeter_structure(perim_width):
    """
    Dilation structure of a perimeter that is perim_width pixels wide (larger values make thicker outlines); 1 is the
    4-connected cross (the thinnest outline).
    """
    if perim_width > 1:
        dilationstructure = np.zeros((perim_width+1, perim_width+1), dtype=int)
        dilationstructure[1:perim_width, 1:perim_width] = 1
    else:
        dilationstructure = generate_binary_structure(2, 1)
    return dilationstructure

#--------------------------------------------------------------------------
def _object_boxes(tracknumber):
    """
    The distinct track numbers of a 2D mask (NaN = no object), the mask relabeled 1 ... n in the order of the sorted
    track numbers (0 = no object) and the bounding box (slices) of every object.
    """
    valid = ~np.isnan(tracknumber)
    tracknumber_unique, inverse = np.unique(tracknumber[valid], return_inverse=True)
    compact = np.zeros(tracknumber.shape, dtype=np.int32)
    compact[valid] = inverse.astype(np.int32) + 1
    slices = find_objects(compact) if tracknumber_unique.size > 0 else []
    return tracknumber_unique, compact, slices

#-----------------------------------------------------------------------
def label_perimeter(tracknumber, dilationstructure, periodic=(False, False)):
    """
    Labels the perimeter on a 2D map from object tracknumber masks.

    Args:
        tracknumber: 2D array
            Track number of each pixel, NaN where there is no object.
        dilationstructure: 2D array
            Structure that sets the perimeter width (make_dilation_structure, make_perimeter_structure).
        periodic: tuple of 2 bool, optional
            (x, y) The grid is periodic in x (columns) and/or y (rows): an object that crosses the edge continues on
            the other side, so there is no perimeter along the edge of the array (global grids, periodic boundary
            conditions). Default: not periodic, the edge of the array is an object boundary.

    Returns:
        tracknumber_perim: 2D int32 array
            The track number on the perimeter pixels of each object, 0 elsewhere.
    """
    if periodic[0] or periodic[1]:
        # Wrap the edges, label, and cut the wrapped margins again
        pad_y = (dilationstructure.shape[0] // 2 + 1) if periodic[1] else 0
        pad_x = (dilationstructure.shape[1] // 2 + 1) if periodic[0] else 0
        padded = np.pad(tracknumber, ((pad_y, pad_y), (pad_x, pad_x)), mode='wrap')
        perim = label_perimeter(padded, dilationstructure)
        return np.ascontiguousarray(perim[pad_y:perim.shape[0]-pad_y, pad_x:perim.shape[1]-pad_x])

    # Make an array to store the perimeter
    tracknumber_perim = np.zeros(tracknumber.shape, dtype=np.int32)

    # Get unique tracknumbers that is no nan, and the bounding box of each
    tracknumber_unique, compact, slices = _object_boxes(tracknumber)
    ny, nx = tracknumber.shape

    # Loop over each tracknumbers
    for kk, ii in enumerate(tracknumber_unique.astype(np.int32)):
        sl = slices[kk]
        ys = slice(max(sl[0].start - 1, 0), min(sl[0].stop + 1, ny))
        xs = slice(max(sl[1].start - 1, 0), min(sl[1].stop + 1, nx))
        # Isolate the track mask
        itn = compact[ys, xs] == (kk + 1)
        # Erode the mask by 1 pixel
        itn_erode = binary_erosion(itn, structure=dilationstructure).astype(itn.dtype)
        # Subtract the eroded area to get the perimeter
        iperim = np.logical_xor(itn, itn_erode)
        # Label the perimeter pixels with the track number
        sub = tracknumber_perim[ys, xs]
        sub[iperim == 1] = ii

    return tracknumber_perim

#-----------------------------------------------------------------------
def calc_cell_center(tracknumber, longitude, latitude, xx, yy):
    """
    Calculates the center location from labeled cells.

    Args:
        tracknumber: 2D array, track number of each pixel (NaN = none)
        longitude, latitude: 2D arrays, lon/lat of each pixel
        xx, yy: 1D arrays, coordinates of the columns (xx) and the rows (yy)

    Returns:
        lon_c, lat_c, xx_c, yy_c: center of each object, tracknumber_uniqe: its track numbers
    """

    # Find unique tracknumbers
    tracknumber_uniqe, compact, slices = _object_boxes(tracknumber)
    num_tracknumber = len(tracknumber_uniqe)
    # Make arrays for cell center locations
    lon_c = np.full(num_tracknumber, np.nan, dtype=float)
    lat_c = np.full(num_tracknumber, np.nan, dtype=float)
    xx_c = np.full(num_tracknumber, np.nan, dtype=float)
    yy_c = np.full(num_tracknumber, np.nan, dtype=float)

    # Loop over each tracknumbers to calculate the mean lat/lon & x/y for their center locations
    for kk in range(num_tracknumber):
        sl = slices[kk]
        mask = compact[sl] == (kk + 1)
        iyy, ixx = np.where(mask)
        lon_c[kk] = np.mean(longitude[sl][mask])
        lat_c[kk] = np.mean(latitude[sl][mask])
        xx_c[kk] = np.mean(xx[ixx + sl[1].start])
        yy_c[kk] = np.mean(yy[iyy + sl[0].start])

    return lon_c, lat_c, xx_c, yy_c, tracknumber_uniqe

#-----------------------------------------------------------------------
def calc_track_center(tracknumber, longitude, latitude):
    """
    Calculates the center location from labeled tracks.
    """

    # Find unique tracknumbers
    tracknumber_uniqe, compact, slices = _object_boxes(tracknumber)
    num_tracknumber = len(tracknumber_uniqe)
    # Make arrays for track center locations
    lon_c = np.full(num_tracknumber, np.nan, dtype=float)
    lat_c = np.full(num_tracknumber, np.nan, dtype=float)

    # Loop over each tracknumbers to calculate the mean lat/lon for their center locations
    for kk in range(num_tracknumber):
        sl = slices[kk]
        mask = compact[sl] == (kk + 1)
        lon_c[kk] = np.mean(longitude[sl][mask])
        lat_c[kk] = np.mean(latitude[sl][mask])

    return lon_c, lat_c, tracknumber_uniqe
