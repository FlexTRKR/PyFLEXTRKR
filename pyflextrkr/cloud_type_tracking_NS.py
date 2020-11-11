def cloud_type_tracking_NS(
    ct, pixel_radius, area_thresh, smoothsize, mincorecoldpix, warmanvilexpansion
):
    ######################################################################
    # Import modules
    import numpy as np
    from scipy.ndimage import label, binary_dilation, generate_binary_structure
    from skimage.morphology import (
        octagon,
    )

    ######################################################################
    # Determine dimensions
    ny, nx = np.shape(ct)

    # Calculate area of one pixel. Assumed to be a circle.
    pixel_area = pixel_radius ** 2

    # Calculate minimum number of pixels based on area threshold
    nthresh = area_thresh / pixel_area

    # Threshold for deep ('core')
    thresh_core = 4
    thresh_concu = 3
    thresh_conculow = 2

    ######################################################################
    # Use thresholds to make a map of all brightnes temperatures that fit within the criteria for convective, cold anvil, and warm anvil points. Cores = 1. Cold anvils = 2. Warm anvils = 3. Other = 4. Clear = 5. Areas do not overlap
    final_cloudtype = np.ones((ny, nx), dtype=int) * -1
    final_cloudtype[np.where(ct == thresh_core)] = 1
    final_cloudtype[np.where((ct == thresh_concu))] = 2
    final_cloudtype[np.where((ct == thresh_conculow))] = 3
    final_cloudtype[np.where((ct < thresh_conculow))] = 5

    ######################################################################
    # Create map of potential features to track. These features encompass the cores and cold anvils
    convective_flag = np.zeros((ny, nx), dtype=int)
    convective_flag[ct >= thresh_concu] = 1

    #####################################################################
    # Label features
    for i in range(0, 2):
        s = generate_binary_structure(2, 2)  # allows diagonal to be included
        convective_label, convective_number = label(convective_flag, s)

        #####################################################################
        # Loop through each feature and determine if it statstifies the area requirement. Do this by finding the number of pixels covered by the feature, multiple by pixel area, and compare the area threshold requirement.
        if convective_number > 0:
            # Initialize vectors of conveective number, number of pixels, and area to record features that statisfy area requirement
            approved_convnumber = np.empty(convective_number, dtype=int) * np.nan
            approved_convpixels = np.empty(convective_number, dtype=int) * np.nan
            approved_convarea = np.empty(convective_number, dtype=int) * np.nan

            for featurestep, ifeature in enumerate(range(1, convective_number + 1)):
                # Identify pixels from each feature and multiple by pixel area to get feature
                feature_pixels = len(
                    np.extract(convective_label == ifeature, convective_label)
                )
                feature_area = feature_pixels * pixel_area

                # If satisfies store the feature number and its area
                if feature_area > nthresh:
                    approved_convnumber[featurestep] = ifeature
                    approved_convpixels[featurestep] = feature_pixels
                    approved_convarea[featurestep] = feature_area

            # Remove blank rows in approved matrices. Itialized so has same length as if all cells passed, but that is not necessary true
            extrarows = np.array(np.where(np.isnan(approved_convnumber)))[0, :]
            if len(extrarows) > 0:
                approved_convnumber = np.delete(approved_convnumber, extrarows)
                approved_convpixels = np.delete(approved_convpixels, extrarows)
                approved_convarea = np.delete(approved_convarea, extrarows)

            ####################################################################
            # Reorder number final features based on descending area (i.e. largest to smallest)
            order = np.argsort(approved_convarea)
            order = order[::-1]
            sortedcore_npix = np.copy(approved_convpixels[order])

            ncores = len(sortedcore_npix)

            # Re-number cores
            sortedcore_number1d = np.copy(approved_convnumber[order])

            sortedcore_number2d = np.zeros((ny, nx), dtype=int)
            corestep = 0
            for isortedcore in range(0, ncores):
                sortedcore_indices = np.where(
                    convective_label == sortedcore_number1d[isortedcore]
                )
                nsortedcoreindices = np.shape(sortedcore_indices)[1]
                if nsortedcoreindices == sortedcore_npix[isortedcore]:
                    corestep = corestep + 1
                    sortedcore_number2d[sortedcore_indices] = np.copy(corestep)

            # Count the number of features in the data
            final_nclouds = approved_convnumber

            ##################################################################
            # Spread cold cores outward
            labelcorecold_number2d = np.copy(sortedcore_number2d)
            labelcorecold_npix = np.copy(sortedcore_npix)

            final_convcold_cloudnumber = np.copy(labelcorecold_number2d)

            # print('Entered dilating loop')
            keepspreading = 1
            # Keep looping through dilating code as long as at least one feature is growing. At this point limit it to 20 dilations (KB-IT IS NOT LIMITED TO 20 ANYMORE). Remove this once use real data.
            while keepspreading > 0 and keepspreading < 2000:
                keepspreading = 0

                # Loop through each feature
                for ifeature in range(1, len(final_nclouds) + 1):
                    # Create map of single feature
                    featuremap = np.copy(labelcorecold_number2d)
                    featuremap[labelcorecold_number2d != ifeature] = 0
                    featuremap[labelcorecold_number2d == ifeature] = 1

                    # Find maximum extent of the of the feature
                    extenty = np.nansum(featuremap, axis=1)
                    extenty = np.array(np.where(extenty > 0))[0, :]
                    miny = extenty[0]
                    maxy = extenty[-1]

                    extentx = np.nansum(featuremap, axis=0)
                    extentx = np.array(np.where(extentx > 0))[0, :]
                    minx = extentx[0]
                    maxx = extentx[-1]

                    # Subset ir and map data to smaller region around feature. This reduces computation time. Add a 100 pixel buffer around the edges of the feature.
                    if minx <= 100:
                        minx = 0
                    else:
                        minx = minx - 100

                    if maxx >= nx - 100:
                        maxx = nx
                    else:
                        maxx = maxx + 101

                    if miny <= 100:
                        miny = 0
                    else:
                        miny = miny - 100

                    if maxy >= ny - 100:
                        maxy = ny
                    else:
                        maxy = maxy + 101

                    ctsubset = np.copy(ct[miny:maxy, minx:maxx])
                    fullsubset = np.copy(labelcorecold_number2d[miny:maxy, minx:maxx])

                    featuresubset = np.copy(featuremap[miny:maxy, minx:maxx])

                    # Dilate cloud region
                    # dilationstructure = generate_binary_structure(2,1)  # Defines shape of growth. This grows one pixel as a cross
                    # dilationstructure = generate_binary_structure(2,2)  # Defines shape of growth. This grows one pixel as rectangle
                    dilationstructure = octagon(3, 2)
                    # dilationstructure = octagon(4,2)

                    dilatedsubset = binary_dilation(
                        featuresubset, structure=dilationstructure, iterations=1
                    ).astype(featuremap.dtype)

                    # Isolate region that was dilated.
                    expansionzone = dilatedsubset - featuresubset

                    # Only keep pixels in dilated regions that are below the warm anvil threshold and are not associated with another feature
                    expansionzone[
                        np.where((expansionzone == 1) & (fullsubset != 0))
                    ] = 0
                    # expansionzone[np.where((expansionzone == 1) & (ctsubset < 1))] = 0
                    expansionzone[np.where((expansionzone == 1) & (ctsubset < 2))] = 0

                    # Find indices of accepted dilated regions
                    expansionindices = np.column_stack(np.where(expansionzone == 1))

                    # Add the accepted dilated region to the map of the cloud numbers
                    labelcorecold_number2d[
                        expansionindices[:, 0] + miny, expansionindices[:, 1] + minx
                    ] = ifeature

                    # Add the number of expanded pixels to pixel count
                    labelcorecold_npix[ifeature - 1] = (
                        len(expansionindices[:, 0]) + labelcorecold_npix[ifeature - 1]
                    )

                    # Count the number of dilated pixels. Add to the keepspreading variable. As long as this variables is > 0 the code continues to run the dilating portion
                    keepspreading = keepspreading + len(
                        np.extract(expansionzone == 1, expansionzone)
                    )

                    convective_flag = labelcorecold_number2d

            ##############################################

        # Initialize vectors
        final_ncorepix = np.ones(len(final_nclouds), dtype=int) * -9999
        final_ncoldpix = np.ones(len(final_nclouds), dtype=int) * -9999
        final_ncorecoldpix = np.ones(len(final_nclouds), dtype=int) * -9999
        final_nwarmpix = np.ones(len(final_nclouds), dtype=int) * -9999

        # Loop through each feature
        for indexstep, ifeature in enumerate(np.arange(1, len(final_nclouds) + 1)):
            final_ncorepix[indexstep] = len(
                np.extract(
                    (labelcorecold_number2d == ifeature) & (final_cloudtype == 1),
                    labelcorecold_number2d,
                )
            )
            final_ncoldpix[indexstep] = len(
                np.extract(
                    (labelcorecold_number2d == ifeature) & (final_cloudtype == 2),
                    labelcorecold_number2d,
                )
            )
            final_ncorecoldpix[indexstep] = len(
                np.extract((labelcorecold_number2d == ifeature), labelcorecold_number2d)
            )
            final_nwarmpix[indexstep] = len(
                np.extract((labelcorecold_npix == ifeature), labelcorecold_npix)
            )

    ##################################################################
    # Output data
    return {
        "final_nclouds": final_nclouds,
        "final_ncorepix": final_ncorepix,
        "final_ncoldpix": final_ncoldpix,
        "final_ncorecoldpix": final_ncorecoldpix,
        "final_nwarmpix": final_nwarmpix,
        "final_cloudnumber": final_convcold_cloudnumber,
        "final_cloudtype": final_cloudtype,
        "final_convcold_cloudnumber": final_convcold_cloudnumber,
    }