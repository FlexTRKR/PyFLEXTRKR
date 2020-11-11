import numpy as np


def sort_renumber(labelcell_number2d, min_cellpix):
    """
    Sorts 2D labeled cells by size, and removes cells smaller than min_cellpix.
    
    Args:
        labelcell_number2d: np.ndarray()
            Labeled cell number array in 2D.
        min_cellpix: float
            Minimum number of pixel to count as a cell.

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
            # Check if cell satisfies size threshold
            if ilabelcell_npix > min_cellpix:
                labelcell_npix[ilabelcell - 1] = ilabelcell_npix

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
                sortedcell_indices = np.where(
                    labelcell_number2d == sortedcell_number1d[icell]
                )
                # Get one of the dimensions from the 2D indices to count the size
                #             nsortedcellindices = np.shape(sortedcell_indices)[1]
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

    return sortedlabelcell_number2d, sortedcell_npix


def sort_renumber2vars(labelcell_number2d, labelcell2_number2d, min_cellpix):
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

    return sortedlabelcell_number2d, sortedlabelcell2_number2d, sortedcell_npix


def link_pf_tb(convcold_cloudnumber, cloudnumber, pf_number, tb, tb_thresh):
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

    return pf_convcold_cloudnumber, pf_cloudnumber
