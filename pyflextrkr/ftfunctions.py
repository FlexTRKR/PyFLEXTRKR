import logging
import numpy as np
from collections import deque
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy.ndimage import label

def sort_renumber(
    labelcell_number2d,
    min_size,
    grid_area=None,
):
    """
    Sorts 2D labeled cells by size, and removes cells smaller than min_size.

    Args:
        labelcell_number2d: np.ndarray()
            Labeled cell number array in 2D.
        min_size: float
            Minimum size to count as a cell.
            If grid_area is None, this should be the minimum number of pixels.
            If grid_area is supplied, this should be the minimum area.
        grid_area: np.ndarray(), optional, default=None
            Area of each grid. Dimensions must match labelcell_number2d.

    Returns:
        sortedlabelcell_number2d: np.ndarray(int)
            Sorted labeled cell number array in 2D.
        sortedcell_npix: np.ndarray(int)
            Number of pixels for each labeled cell in 1D.
    """

    # Create output arrays
    sortedlabelcell_number2d = np.zeros(np.shape(labelcell_number2d), dtype=int)

    # Get number of labeled cells
    nlabelcells = np.nanmax(labelcell_number2d)

    # Check if there is any cells identified
    if nlabelcells > 0:

        labelcell_npix = np.full(nlabelcells, -999, dtype=int)
        # Loop over each labeled cell
        for ilabelcell in range(1, nlabelcells + 1):
            # Count number of pixels for the cell
            ilabelcell_npix = np.count_nonzero(labelcell_number2d == ilabelcell)
            # Check if grid_area is supplied
            if grid_area is None:
                # If cell npix > min size threshold
                if ilabelcell_npix > min_size:
                    labelcell_npix[ilabelcell - 1] = ilabelcell_npix
            else:
                # If grid_area is supplied, sum grid area for the cell
                ilabelcell_area = np.sum(grid_area[labelcell_number2d == ilabelcell])
                # If cell area > min size threshold
                if ilabelcell_area > min_size:
                    labelcell_npix[ilabelcell - 1] = ilabelcell_npix


        # # This faster approach does not work
        # # Because when labelcell_number2d is not sequentially numbered (e.g., when some cells are removed)
        # # This approach does not get the same sequence with the above one
        # # Count number of pixels for each unique cells
        # cellnum, labelcell_npix = np.unique(labelcell_number2d, return_counts=True)
        # # Remove background and cells below size threshold
        # labelcell_npix = labelcell_npix[(cellnum > 0)]
        # labelcell_npix[(labelcell_npix <= min_size)] = -999

        # Check if any of the cells passes the size threshold test
        ivalidcells = np.where(labelcell_npix > 0)[0]
        # ivalidcells = np.array(np.where(labelcell_npix > 0))[0, :]
        ncells = len(ivalidcells)

        if ncells > 0:
            # Isolate cells that satisfy size threshold
            # Add one since label numbers start at 1 and indices, which validcells reports starts at 0
            labelcell_number1d = np.copy(ivalidcells) + 1
            labelcell_npix = labelcell_npix[ivalidcells]

            # Sort cells from largest to smallest and get the sorted index
            order = np.argsort(labelcell_npix)[::-1]
            # order = order[::-1]  # Reverses the order

            # Sort the cells by size
            sortedcell_npix = np.copy(labelcell_npix[order])
            sortedcell_number1d = np.copy(labelcell_number1d[order])

            # Loop over the 2D cell number to re-number them by size
            cellstep = 0
            for icell in range(0, ncells):
                # Find 2D indices that match the cell number
                sortedcell_indices = np.where(
                    labelcell_number2d == sortedcell_number1d[icell]
                )
                # Get one of the dimensions from the 2D indices to count the size
                nsortedcellindices = len(sortedcell_indices[1])
                # Check if the size matches the sorted cell size
                if nsortedcellindices == sortedcell_npix[icell]:
                    # Renumber the cell in 2D
                    cellstep += 1
                    sortedlabelcell_number2d[sortedcell_indices] = np.copy(cellstep)

        else:
            # Return an empty array
            sortedcell_npix = np.zeros(0)
    else:
        # Return an empty array
        sortedcell_npix = np.zeros(0)

    return (
        sortedlabelcell_number2d,
        sortedcell_npix,
    )


def sort_renumber2vars(
    labelcell_number2d,
    labelcell2_number2d,
    min_cellpix,
):
    """
    Sorts 2D labeled cells by size, and removes cells smaller than min_cellpix.
    This version renumbers two variables using the same size sorting from labelcell_number2d.

    Args:
        labelcell_number2d: np.ndarray()
            Labeled cell number array in 2D.
        labelcell2_number2d: np.ndarray()
            Labeled cell number array2 in 2D.
        min_cellpix: float
            Minimum number of pixel to count as a cell.

    Returns:
        sortedlabelcell_number2d: np.ndarray(int)
            Sorted labeled cell number array in 2D.
        sortedlabelcell2_number2d: np.ndarray(int)
            Sorted labeled cell number array2 in 2D.
        sortedcell_npix: np.ndarray(int)
            Number of pixels for each labeled cell in 1D.
    """

    # Create output arrays
    sortedlabelcell_number2d = np.zeros(np.shape(labelcell_number2d), dtype=int)
    sortedlabelcell2_number2d = np.zeros(np.shape(labelcell_number2d), dtype=int)

    # Get number of labeled cells
    nlabelcells = np.nanmax(labelcell_number2d)

    # Check if there is any cells identified
    if nlabelcells > 0:

        labelcell_npix = np.full(nlabelcells, -999, dtype=int)
        # Loop over each labeled cell
        for ilabelcell in range(1, nlabelcells + 1):
            # Count number of pixels for the cell
            ilabelcell_npix = np.count_nonzero(labelcell_number2d == ilabelcell)
            # Check if cell satisfies size threshold
            if ilabelcell_npix > min_cellpix:
                labelcell_npix[ilabelcell - 1] = ilabelcell_npix

        # # This faster approach does not work
        # # Because when labelcell_number2d is not sequentially numbered (e.g., when some cells are removed)
        # # This approach does not get the same sequence with the above one
        # # Count number of pixels for each unique cells
        # cellnum, labelcell_npix2 = np.unique(labelcell_number2d, return_counts=True)
        # # Remove background and cells below size threshold
        # labelcell_npix2 = labelcell_npix2[(cellnum > 0)]
        # labelcell_npix2[(labelcell_npix2 <= min_cellpix)] = -999

        # Check if any of the cells passes the size threshold test
        ivalidcells = np.array(np.where(labelcell_npix > 0))[0, :]
        ncells = len(ivalidcells)

        if ncells > 0:
            # Isolate cells that satisfy size threshold
            # Add one since label numbers start at 1 and indices, which validcells reports starts at 0
            labelcell_number1d = np.copy(ivalidcells) + 1
            labelcell_npix = labelcell_npix[ivalidcells]

            # Sort cells from largest to smallest and get the sorted index
            order = np.argsort(labelcell_npix)
            order = order[::-1]  # Reverses the order

            # Sort the cells by size
            sortedcell_npix = np.copy(labelcell_npix[order])
            sortedcell_number1d = np.copy(labelcell_number1d[order])

            # Loop over the 2D cell number to re-number them by size
            cellstep = 0
            for icell in range(0, ncells):
                # Find 2D indices that match the cell number
                # Use the same sorted index to label labelcell2_number2d
                sortedcell_indices = np.where(
                    labelcell_number2d == sortedcell_number1d[icell]
                )
                sortedcell2_indices = np.where(
                    labelcell2_number2d == sortedcell_number1d[icell]
                )
                # Get one of the dimensions from the 2D indices to count the size
                nsortedcellindices = len(sortedcell_indices[1])
                # Check if the size matches the sorted cell size
                if nsortedcellindices == sortedcell_npix[icell]:
                    # Renumber the cell in 2D
                    cellstep += 1
                    sortedlabelcell_number2d[sortedcell_indices] = np.copy(cellstep)
                    sortedlabelcell2_number2d[sortedcell2_indices] = np.copy(cellstep)

        else:
            # Return an empty array
            sortedcell_npix = np.zeros(0)
    else:
        # Return an empty array
        sortedcell_npix = np.zeros(0)

    return (
        sortedlabelcell_number2d,
        sortedlabelcell2_number2d,
        sortedcell_npix,
    )


def link_pf_tb(
    convcold_cloudnumber,
    cloudnumber,
    pf_number,
    tb,
    tb_thresh,
):
    """
    Renumbers separated clouds over the same PF to one cloud, using the largest cloud number.

    Args:
        convcold_cloudnumber: np.ndarray(int)
            Convective-coldanvil cloud number
        cloudnumber: np.ndarray(int)
            Cloud number
        pf_number: np.ndarray(int)
            PF number
        tb: np.ndarray(float)
            Brightness temperature
        tb_thresh: float
            Temperature threshold to label PFs that have not been labeled in pf_number.
            Currently this threshold is NOT used.

    Returns:
        pf_convcold_cloudnumber: np.ndarray(int)
            Renumbered convective-coldanvil cloud number
        pf_cloudnumber: np.ndarray(int)
            Renumbered cloud number
    """

    # Get number of PFs
    npf = np.nanmax(pf_number)

    # Make a copy of the input arrays
    pf_convcold_cloudnumber = np.copy(convcold_cloudnumber)
    pf_cloudnumber = np.copy(cloudnumber)

    # Create a 2D index array with the same shape as the full image
    # This index array is used to map the indices of indices back to the full image
    arrayindex2d = np.reshape(np.arange(tb.size), tb.shape)

    # If number of PF > 0, proceed
    if npf > 0:

        # Initiallize masks to keep track of which clouds have been renumbered
        pf_convcold_mask = np.zeros(tb.shape, dtype=int)
        pf_cloud_mask = np.zeros(tb.shape, dtype=int)

        # Loop over each PF
        for ipf in range(1, npf):

            # Find pixel index for this PF
            pfidx = np.where(pf_number == ipf)
            npix_pf = len(pfidx[0])

            if npix_pf > 0:
                # Get unique cloud number defined within this PF
                # cn_uniq = np.unique(convcold_cloudnumber[pfidx])
                cn_uniq = np.unique(pf_convcold_cloudnumber[pfidx])

                # Find actual clouds (cloudnumber > 0)
                cn_uniq = cn_uniq[np.where(cn_uniq > 0)]
                nclouds_uniq = len(cn_uniq)
                # If there is at least 1 cloud, proceed
                if nclouds_uniq >= 1:

                    # Loop over each cloudnumber and get the size
                    npix_uniq = np.zeros(nclouds_uniq, dtype=np.int64)
                    for ic in range(0, nclouds_uniq):
                        # Find pixels for each cloud, save the size
                        # npix_uniq[ic] = len(np.where(convcold_cloudnumber == cn_uniq[ic])[0])
                        npix_uniq[ic] = len(
                            np.where(pf_convcold_cloudnumber == cn_uniq[ic])[0]
                        )

                    # Find cloud number that has maximum size
                    cn_max = cn_uniq[np.argmax(npix_uniq)]

                    # Loop over each cloudnumber again
                    for ic in range(0, nclouds_uniq):

                        # Find pixel locations within each cloud, and mask = 0 (cloud has not been renumbered)
                        # idx_convcold = np.where((convcold_cloudnumber == cn_uniq[ic]) & (pf_convcold_mask == 0))
                        idx_convcold = np.where(
                            (pf_convcold_cloudnumber == cn_uniq[ic])
                            & (pf_convcold_mask == 0)
                        )
                        # idx_cloud = np.where((cloudnumber == cn_uniq[ic]) & (pf_cloud_mask == 0))
                        idx_cloud = np.where(
                            (pf_cloudnumber == cn_uniq[ic]) & (pf_cloud_mask == 0)
                        )
                        if len(idx_convcold[0]) > 0:
                            # Renumber the cloud to the largest cloud number (that overlaps with this PF)
                            pf_convcold_cloudnumber[idx_convcold] = cn_max
                            pf_convcold_mask[idx_convcold] = 1
                        if len(idx_cloud[0]) > 0:
                            # Renumber the cloud to the largest cloud number (that overlaps with this PF)
                            pf_cloudnumber[idx_cloud] = cn_max
                            pf_cloud_mask[idx_cloud] = 1

                    # Find area within the PF that has no cloudnumber, Tb < warm threshold, and has not been labeled yet
                    #                     idx_nocloud = np.asarray((pf_convcold_cloudnumber[pfidx] == 0) & (tb[pfidx] < tb_thresh) & (pf_convcold_mask[pfidx] == 0)).nonzero()
                    idx_nocloud = np.asarray(
                        (pf_convcold_cloudnumber[pfidx] == 0)
                        & (pf_convcold_mask[pfidx] == 0)
                    ).nonzero()
                    if np.count_nonzero(idx_nocloud) > 0:
                        # At this point, idx_nocloud is a 1D index referring to the subset within pfidx
                        # Applying idx_nocloud of pfidx to the 2D full image index array gets the 1D indices referring to the full image,
                        # then unravel_index converts those 1D indices back to 2D, which can then be applied to the 2D full image
                        idx_loc = np.unravel_index(
                            arrayindex2d[pfidx][idx_nocloud], tb.shape
                        )
                        # Label the no cloud area using the largest cloud number
                        pf_convcold_cloudnumber[idx_loc] = cn_max
                        pf_convcold_mask[idx_loc] = 1

                    # Find area within the PF that has no cloudnumber, Tb < warm threshold, and has not been labeled yet
                    #                     idx_nocloud = np.asarray((pf_cloudnumber[pfidx] == 0) & (tb[pfidx] < tb_thresh) & (pf_cloud_mask[pfidx] == 0)).nonzero()
                    idx_nocloud = np.asarray(
                        (pf_cloudnumber[pfidx] == 0) & (pf_cloud_mask[pfidx] == 0)
                    ).nonzero()
                    if np.count_nonzero(idx_nocloud) > 0:
                        idx_loc = np.unravel_index(
                            arrayindex2d[pfidx][idx_nocloud], tb.shape
                        )
                        # Label the no cloud area using the largest cloud number
                        pf_cloudnumber[idx_loc] = cn_max
                        pf_cloud_mask[idx_loc] = 1

    else:
        # Pass input variables to output if no PFs are defined
        pf_convcold_cloudnumber = np.copy(convcold_cloudnumber)
        pf_cloudnumber = np.copy(cloudnumber)

    return (
        pf_convcold_cloudnumber,
        pf_cloudnumber,
    )

def olr_to_tb(OLR):
    """
    Convert OLR to IR brightness temperature.

    Args:
        OLR: np.array
            Outgoing longwave radiation
    
    Returns:
        tb: np.array
            Brightness temperature
    """
    # Calculate brightness temperature
    # (1984) as given in Yang and Slingo (2001)
    # Tf = tb(a+b*Tb) where a = 1.228 and b = -1.106e-3 K^-1
    # OLR = sigma*Tf^4 
    # where sigma = Stefan-Boltzmann constant = 5.67x10^-8 W m^-2 K^-4
    a = 1.228
    b = -1.106e-3
    sigma = 5.67e-8 # W m^-2 K^-4
    tf = (OLR/sigma)**0.25
    tb = (-a + np.sqrt(a**2 + 4*b*tf))/(2*b)
    return tb

def get_neighborhood(point, grid):
    """
    Given a grid of labeled points with 0=unlabeled, -1 to be processed, other # to be proccesed.

    Args:
        point: np.array
            Array containing seed points for growing
        grid: np.array
            Array containing labels.
    
    Returns:
        next_points: np.array
            Neighboring points.
    """
    shape = grid.shape
    point_grid = [
        [x + point[0], y + point[1]] for x in range(-1, 2) for y in range(-1, 2)
    ]
    next_points = []

    for idx, i_point in enumerate(point_grid):
        if i_point[0] < 0 or i_point[0] >= shape[0]:
            continue
        if i_point[1] < 0 or i_point[1] >= shape[1]:
            continue
        if i_point[0] == point[0] and i_point[1] == point[1]:
            continue
        else:  # We're good
            if grid[i_point[0], i_point[1]] == 0:
                next_points.append([i_point[0], i_point[1]])
    return next_points  # Would probably be faster to pass in deque and directly add rather than a sublist.


def grow_cells(grid):
    """
    Fast algorithm to grow and label areas based on nearest distance to the seeded regions.

    Args:
        grid: np.array
            Array containing labeled seeded regions (values > 0).
            Areas for growing = 0, areas excluded = -1.

    Returns:
        grid: np.array
            Array containing labels after growth.
    """
    seed_points = np.where(grid > 0)
    point_que = deque(
        [
            [seed_points[0][i], seed_points[1][i]]
            for i in range(np.count_nonzero(seed_points[0]))
        ]
    )
    while len(point_que) > 0:
        current_pt = point_que.popleft()
        neighbor_values = grid[
            max(current_pt[0] - 1, 0) : current_pt[0] + 2,
            max(0, current_pt[1] - 1) : current_pt[1] + 2,
        ]
        neighbors = get_neighborhood(current_pt, grid)

        for point in neighbors:
            grid[point[0], point[1]] = -1
            point_que.append(point)
        if (
            grid[current_pt[0], current_pt[1]] < 1
        ):  # Lets not reclassify currently classified points grabbed in beginning selection
            counts_v, counts_i = np.unique(
                neighbor_values[neighbor_values > 0], return_counts=True
            )
            mode_val = counts_v[np.argmax(counts_i)]
            grid[current_pt[0], current_pt[1]] = mode_val
    return grid


def skimage_watershed(fvar, config):
    """
    Label objects with skimage.watershed function

    Args:
        fvar: np.array
            2D array containing data for segmentation.
        config: dictionary
            Dictionary containing config parameters

    Returns:
        var_number: np.array
            Array containing labeled objects.
        param_dict: dictionary
            Dictionary containing parameters used from config.
    """

    # Get thresholds from config
    plm_min_distance = config['plm_min_distance']
    plm_exclude_border = config['plm_exclude_border']
    plm_threshold_abs = config['plm_threshold_abs']
    cont_thresh = config['cont_thresh']
    compa = config['compa']

    # Put parameters in a dictionary
    param_dict = {
        'plm_min_distance': plm_min_distance,
        'plm_exclude_border': plm_exclude_border,
        'plm_threshold_abs': plm_threshold_abs,
        'cont_thresh': cont_thresh,
        'compa': compa,
    }

    # Get grid indices of local maxima
    local_maxes = peak_local_max(fvar, min_distance=plm_min_distance, exclude_border=plm_exclude_border, threshold_abs=plm_threshold_abs)
    cc, dd = local_maxes.shape
    
    # Generate 2D field with shape of favr, local maxima locations marked by the maxima number
    # Field is zero where maxima not present
    markers = np.zeros(fvar.shape, dtype=int)
    for p in range(cc):
        markers[local_maxes[p,0], local_maxes[p,1]] = p + 1    # plus 1 because dont want a marker = 0.

    # Define a binary mask used in watershed algorithm
    Pmask = np.zeros(fvar.shape, dtype=int)
    Pmask[fvar > cont_thresh] = 1

    # Use watershed to define objects:
    var_number = watershed(-fvar, markers, mask=Pmask, watershed_line=True, compactness=compa)

    return var_number, param_dict


def pad_and_extend(fvar, config):
    """
    Pad and extend data based on specified periodic boundary conditions.

    Args:
        fvar: np.array
            2D array containing data for padding.
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        extended_data: np.array
            extended 2D-array fvar.
        padded_x: bool
            True if the data was padded in the x-direction, False otherwise.
        padded_y: bool
            True if the data was padded in the y-direction, False otherwise.
    """
    ext_frac = config.get('pbc_extended_fraction', 1.0)
    pbc_direction = config.get('pbc_direction', 'both')
    pad_x, pad_y = (0, 0), (0, 0)
    padded_x = padded_y = False

    if pbc_direction in ['x', 'both']:
        pad_x = (calc_extension(fvar.shape[1], ext_frac),) * 2
        padded_x = True
    if pbc_direction in ['y', 'both']:
        pad_y = (calc_extension(fvar.shape[0], ext_frac),) * 2
        padded_y = True

    extended_data = np.pad(fvar, pad_width=(pad_y, pad_x), mode='wrap')
    
    return extended_data, padded_x, padded_y

def calc_extension(size, ext_frac):
    """
    This function computes the number of elements to extend an array dimension by 
    multiplying the dimension size by the specified extension fraction.

    Args:
        size: int
            The original size of the dimension.
        ext_frac: float
            The fraction of the size used to determine the extension length.
    Returns:
        The computed extension size (int).
    """
    return int(size * ext_frac)

def cache_label_positions(segments):
    """
    Cache the positions of labels in segments (features).
    
    Args:
        segments: np.array
            A 2D array where each element represents a label assigned to a segment 
            (feature). The background is assumed to be represented by 0.

    Returns:
        label_positions_cache: dict
            A dictionary mapping each label (non-zero) to a tuple of arrays containing 
            the indices where the label occurs.
    """
    label_positions_cache = {}
    unique_labels = np.unique(segments)
    for label in unique_labels:
        if label != 0:  # Ignore background
            label_positions = np.where(segments == label)
            label_positions_cache[label] = label_positions
    return label_positions_cache

def adjust_axis(segments, axis, original_shape, ext_frac, config):
    """
    Adjust the segmented features along a specified axis based on periodic boundaries.

    Args:
        segments: np.array
            A 2D array of segmented labels where features have been identified.
        axis: int
            The axis along which to adjust the segmentation. Use 0 for y-axis and 1 
            for x-axis.
        original_shape: tuple of ints
            The shape of the original (unpadded) data array.
        ext_frac: float
            The extension fraction used to compute the extension size for padding.
        config: dictionary
            Dictionary containing config parameters

    Returns:
        segments: np.array
            The adjusted segmentation array after cropping and rolling.
        adjusted: bool
            True if adjustments were made based on detected shared labels; 
            False otherwise.

    """
    logger = logging.getLogger(__name__)
    pixel_radius = config.get('pixel_radius')
    area_thresh = config.get('area_thresh')
    # Calculate feature width threshold (in pixels) proportional to minimum area of objects defined in config
    # If there are objects with width > width_thresh, the algorithm will keep searching to refine the position to crop
    # until it reaches the edge of the extended domain.
    size_factor = 3     # adjustable multiplier factor
    width_thresh = size_factor * int(2 * np.sqrt(area_thresh / np.pi) / pixel_radius)
    ext_size = calc_extension(original_shape[axis], ext_frac)
    adjusted = False
    label_positions_cache = cache_label_positions(segments)
    
    if axis == 1:  # X-axis
        left_slice = segments[:, :ext_size]
        middle_slice = segments[:, ext_size:ext_size + original_shape[1]]
        shared_labels = np.intersect1d(left_slice[:, -1], middle_slice[:, 0])     
    elif axis == 0:  # Y-axis
        top_slice = segments[:ext_size, :]
        middle_slice = segments[ext_size:ext_size + original_shape[0], :]
        shared_labels = np.intersect1d(top_slice[-1, :], middle_slice[0, :])
    # Remove 0 (background) from shared labels
    shared_labels = shared_labels[shared_labels != 0]

    if shared_labels.size > 0 and not np.all(shared_labels == 0):
        for label in shared_labels:
            # Verify if the label spans the middle slice in Y direction
            if np.all(middle_slice == label):
                logger.warning(f"Full-domain spanning feature detected in axis {axis} with label {label}.")
                continue
            adjusted = True
            # Start with initial min_pos for the label
            min_pos = np.min(label_positions_cache[label][axis])
            
            # Iteratively refine min_pos using cache
            while True:
                # Find all labels at the current min_pos slice
                current_labels = segments[min_pos, :] if axis == 0 else segments[:, min_pos]
                non_zero_labels = current_labels[current_labels != 0]
                unique_labels, unique_npix = np.unique(non_zero_labels, return_counts=True)
                max_unique_npix = np.max(unique_npix)
                # If position includes multiple labels and the largest width > width_thresh, 
                # keep searching to refine min_pos
                if (unique_labels.size > 1) and (max_unique_npix > width_thresh):
                    min_positions = [np.min(label_positions_cache[ul][axis]) for ul in unique_labels]
                    new_min_pos = min(min_positions)
                    if new_min_pos == min_pos:
                        break
                    min_pos = new_min_pos
                else:
                    break
        # Calculate cropping and rolling adjustments
        if axis == 1:
            dx = ext_size - min_pos
            segments = segments[:, min_pos:ext_size + original_shape[1] - dx]
            segments = np.roll(segments, shift=-dx, axis=1)
        elif axis == 0:
            dy = ext_size - min_pos
            segments = segments[min_pos:ext_size + original_shape[0] - dy, :]
            segments = np.roll(segments, shift=-dy, axis=0)
        
    else:
        logger.debug(f"No shared labels found in axis {axis}.")
    return segments, adjusted

def adjust_pbc_watershed(fvar, config):
    """
    Process data to handle PBC when identifying features using watershed segmentation.
    Steps:
    1. Reads parameters related to boundary conditions from config file. 
    2. Extends and pads data.
    3. Apply watershed segmentation .
    4. Adjust axis based on PBC direction. 

    Args:
        fvar: np.array
            The original 2D data array to be segmented.
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        adjusted_segments: np.array
            The segmented features after applying watershed segmentation and adjustments 
            based on PBC.
        param_dict: dict
            A dictionary of parameters returned by the watershed segmentation 
            function.
    """
    ext_frac = config.get('pbc_extended_fraction', 1.0)
    pbc_direction = config.get('pbc_direction', 'both')
    # Step 2: Extend and pad data
    extended_data, padded_x, padded_y = pad_and_extend(fvar, config)
    # Step 3: Apply watershed segmentation
    extended_segments, param_dict = skimage_watershed(extended_data, config)
    # Step 4: Adjust axis based on PBC direction
    adjusted_segments = call_adjust_axis(extended_segments,fvar,config, padded_x, padded_y)

    return adjusted_segments, param_dict

def call_adjust_axis(extended_segments, fvar, config, padded_x, padded_y):
    """
    Process data to adjust axis based on periodic boundary directions. 

    Args:
        extended_segments: np.ndarray()
            2D numpy array of the extended and padded feature data.
        fvar: np.ndarray()
            2D numpy array of the original feature data.
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        extended_segments: np.ndarray()
            Adjusted segments according to periodic boundary considerations.
    """
    ext_frac = config.get('pbc_extended_fraction', 1.0)
    pbc_direction = config.get('pbc_direction', 'both')
    #   Initialize adjustment flags
    x_adjusted, y_adjusted = False, False
    original_shape = fvar.shape
    
    # Adjust each specified axis (X and/or Y)
    if pbc_direction in ['x', 'both']:
        extended_segments, x_adjusted = adjust_axis(extended_segments, 1, original_shape, ext_frac, config)
    if pbc_direction in ['y', 'both']:
        extended_segments, y_adjusted = adjust_axis(extended_segments, 0, original_shape, ext_frac, config)
        
    # Restore to the original structure in non-adjusted dimensions
    if padded_x and not x_adjusted:
        crop_start_x = calc_extension(original_shape[1], ext_frac)
        extended_segments = extended_segments[:, crop_start_x:crop_start_x + original_shape[1]]

    if padded_y and not y_adjusted:
        crop_start_y = calc_extension(original_shape[0], ext_frac)
        extended_segments = extended_segments[crop_start_y:crop_start_y + original_shape[0], :]

    return extended_segments

def circular_mean(values, domain_min, domain_max):
    """
    Compute the circular mean of values in a periodic domain.
    
    Args:
        values: np.ndarray()
            Array of positions
        domain_min: float
            Minimum value of the domain (e.g., 0 or -180)
        domain_max: float
            Maximum value of the domain (e.g., 100 or 180)

    Returns:
        Mean position in the original domain range
    """

    # Convert to [0, 1] range within the domain
    domain_range = domain_max - domain_min
    normalized_values = (values - domain_min) / domain_range * 2 * np.pi  # Convert to radians

    # Compute the circular mean using trigonometry
    mean_angle = np.arctan2(np.nanmean(np.sin(normalized_values)), np.nanmean(np.cos(normalized_values)))

    # Convert back to original domain
    mean_value = (mean_angle / (2 * np.pi) * domain_range) + domain_min

    # Ensure the mean value stays within the original domain
    return (mean_value - domain_min) % domain_range + domain_min

#-----------------------------------------------------------------------
def get_cloud_boundary(icloudlocationx, icloudlocationy, xdim, ydim):
    """
    Get the boundary indices of a cloud feature.

    Args:
        icloudlocationx: numpy array
            Cloud location indices in x-direction.
        icloudlocationy: numpy array
            Cloud location indices in y-direction.
        xdim: int
            Full pixel image dimension in x-direction.
        ydim: int
            Full pixel image dimension in y-direction.

    Returns:
        maxx: int
        maxy: int
        minx: int
        miny: int
    """
    # buffer = 10
    buffer = 0
    miny = np.nanmin(icloudlocationy)
    if miny <= 10:
        miny = 0
    else:
        miny = miny - buffer
    maxy = np.nanmax(icloudlocationy)
    if maxy >= ydim - 10:
        maxy = ydim
    else:
        maxy = maxy + buffer + 1
    minx = np.nanmin(icloudlocationx)
    if minx <= 10:
        minx = 0
    else:
        minx = minx - buffer
    maxx = np.nanmax(icloudlocationx)
    if maxx >= xdim - 10:
        maxx = xdim
    else:
        maxx = maxx + buffer + 1
    return maxx, maxy, minx, miny

#-----------------------------------------------------------------------
def find_max_indices_to_roll(mask_map, xdim, ydim):
    """
    Find the indices to roll the data to avoid periodic boundary condition.

    Args:
        mask_map: numpy array
            Cloudnumber 2D mask.
        xdim: int
            x dimension of domain.
        ydim: int
            y dimension of domain.

    Returns:
        shift_x_right: int
            Number of indices to shift right.
        shift_y_top: int
            Number of indices to shift top.
    """
    # Label the connected pixels
    label_mask, num_cld = label(mask_map)
    # Count number of pixels for each labeled cloud
    uniq_labels, npix = np.unique(label_mask, return_counts=True)
    # Exclude 0 (background)
    _idx = uniq_labels > 0
    uniq_labels = uniq_labels[_idx]
    npix = npix[_idx]
    # Loop over each labeled cloud to find their min/max positions
    xmax_left = []                       
    xmin_right = []
    ymax_bottom = []
    ymin_top = []
    for cc in uniq_labels:
        _y, _x = np.array(np.where(label_mask == cc))
        _maxx, _maxy, _minx, _miny = get_cloud_boundary(_x, _y, xdim, ydim)
        if _minx == 0:  # Feature touching left boundary
            xmax_left.append(_maxx)
        if _maxx == xdim:  # Feature touching right boundary
            xmin_right.append(_minx)
        if _miny == 0:  # Feature touching bottom boundary
            ymax_bottom.append(_maxy)
        if _maxy == ydim:  # Feature touching top boundary
            ymin_top.append(_miny)

    # Find max indices to roll
    # shift_x_left = np.max(xmax_left) if (len(xmax_left) > 0) else 0
    shift_x_right = (xdim - np.min(xmin_right)) if (len(xmin_right) > 0) else 0
    # shift_y_bottom = np.max(ymax_bottom) if (len(ymax_bottom) > 0) else 0
    shift_y_top = (ydim - np.min(ymin_top)) if (len(ymin_top) > 0) else 0

    return (shift_x_right, shift_y_top)

#-----------------------------------------------------------------------
def subset_roll_map(data_array, shift_x_right, shift_y_top, xdim, ydim, fillval=0):
    """
    Subset and roll the data array to avoid periodic boundary condition.

    Args:
        data_array: numpy array
            2D data array (dimensions: [y,x]).
        shift_x_right: int
            Number of indices to shift right.
        shift_y_top: int
            Number of indices to shift top.
        xdim: int
            x dimension of domain.
        ydim: int
            y dimension of domain.
        fillval: float, default=0
            Fill value in data_array to exclude.

    Returns:
        out_data_array: numpy array
            2D subsetted and rolled data array.
    """
    # Roll array to avoid periodic boundary condition
    # In X direction (axis 1): roll to the right (positive shift) by shift_x_right
    # In Y direction (axis 0): roll to the top (positive shift) by shift_y_top
    data_array_rolled = np.roll(
        data_array, shift=(np.abs(shift_y_top), np.abs(shift_x_right)), axis=(0,1),
    )
    # Find valid values
    _y, _x = np.where((~np.isnan(data_array_rolled)) & (data_array_rolled != fillval))
    # Get data boundary
    _maxx, _maxy, _minx, _miny = get_cloud_boundary(_x, _y, xdim, ydim)
    # Subset data array
    out_data_array = data_array_rolled[_miny:_maxy, _minx:_maxx]
    return out_data_array