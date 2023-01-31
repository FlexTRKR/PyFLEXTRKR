import logging
from collections import deque
import numpy as np
from scipy.ndimage import label, binary_dilation, generate_binary_structure, filters
from astropy.convolution import Box2DKernel, convolve
from pyflextrkr.ftfunctions import grow_cells


def label_and_grow_cold_clouds(
    ir,
    pixel_radius,
    tb_threshs,
    area_thresh,
    mincoldcorepix,
    smoothsize,
    warmanvilexpansion,
):
    logger = logging.getLogger(__name__)

    # Separate array threshold
    thresh_core = tb_threshs[0]  # Convective core threshold [K]
    thresh_cold = tb_threshs[1]  # Cold anvil threshold [K]
    thresh_warm = tb_threshs[2]  # Warm anvil threshold [K]
    thresh_cloud = tb_threshs[3]  # Warmest cloud area threshold [K]

    # Determine dimensions
    ny, nx = np.shape(ir)

    # Calculate area of one pixel. Assumed to be a circle.
    pixel_area = pixel_radius ** 2

    # Calculate minimum number of pixels based on area threshold
    nthresh = area_thresh / pixel_area

    ######################################################################
    # Use thresholds identify pixels containing cold core, cold anvil, and warm anvil.
    # Also create arrays with a flag for each type and fill in cloudid array.
    # Cores = 1. Cold anvils = 2. Warm anvils = 3. Other = 4. Clear = 5. Areas do not overlap
    (
        coldanvil_flag,
        core_flag,
        final_cloudid,
    ) = generate_pixel_identification_from_threshold(
        ir, nx, ny, thresh_cloud, thresh_cold, thresh_core, thresh_warm
    )

    #################################################################
    ncorepix, smoothir = smooth_and_identify_indices(
        ir, nx, ny, smoothsize, thresh_cloud, thresh_cold, thresh_core, thresh_warm
    )

    labelcore_number2d, nlabelcores = find_and_label_cold_cores(
        ncorepix, nx, ny, smoothir, thresh_core
    )

    # Check if any cores have been identified
    if nlabelcores > 0:

        # Check if cores satisfy size threshold
        labelcore_npix = np.ones(nlabelcores, dtype=int) * -9999
        temp_labelcore_idx, temp_labelcore_counts = np.unique(
            labelcore_number2d, return_counts=True
        )

        for ilabelcore in temp_labelcore_idx:
            if ilabelcore < 1 or ilabelcore > nlabelcores:
                continue  # THis is just to match previous logic
            if temp_labelcore_counts[ilabelcore] > mincoldcorepix:
                labelcore_npix[ilabelcore - 1] = temp_labelcore_counts[ilabelcore]

        # Check if any of the cores passed the size threshold test
        ivalidcores = np.array(np.where(labelcore_npix > 0))[0, :]
        ncores = len(ivalidcores)
        if ncores > 0:
            # Isolate cores that satisfy size threshold
            labelcore_number1d = (
                np.copy(ivalidcores) + 1
            )  # Add one since label numbers start at 1 and indices, which validcores reports starts at 0
            labelcore_npix = labelcore_npix[ivalidcores]

            #########################################################3
            # Sort sizes largest to smallest
            order = np.argsort(labelcore_npix)
            order = order[::-1]
            sortedcore_npix = np.copy(labelcore_npix[order])

            # Re-number cores
            sortedcore_number1d = np.copy(labelcore_number1d[order])

            sortedcore_number2d = np.zeros((ny, nx), dtype=int)
            corestep = 0
            for isortedcore in range(0, ncores):
                sortedcore_indices = np.where(
                    labelcore_number2d == sortedcore_number1d[isortedcore]
                )
                nsortedcoreindices = np.shape(sortedcore_indices)[1]
                if nsortedcoreindices == sortedcore_npix[isortedcore]:
                    corestep = corestep + 1
                    sortedcore_number2d[sortedcore_indices] = np.copy(corestep)

            #####################################################
            # Spread cold cores outward until reach cold anvil threshold. Generates cold anvil.
            labelcorecold_number2d = np.copy(sortedcore_number2d)
            labelcorecold_npix = np.copy(sortedcore_npix)
            keepspreading = 1
            # Keep looping through dilating code as long as at least one feature is growing.
            # At this point limit it to 20 dilations. Remove this once use real data.

            # We set everything we don't want to process to -1
            cold_threshold_map = np.logical_or(ir > thresh_cold, np.isnan(ir))
            temp_storage = labelcorecold_number2d[cold_threshold_map]
            labelcorecold_number2d[cold_threshold_map] = -1

            # Then we grow out seed points
            labelcorecold_number2d = grow_cells(labelcorecold_number2d)

            # Then just to match before we put back old labels.
            labelcorecold_number2d[
                cold_threshold_map
            ] = temp_storage  # We put these back how we found them
            # This is probably not necessary though.

            # Update the cloud sizes
            cloud_indices, cloud_sizes = np.unique(
                labelcorecold_number2d, return_counts=True
            )
            for index in cloud_indices:
                if index == 0:
                    continue
                labelcorecold_npix[index - 1] = cloud_sizes[index]

        #############################################################
        # Create blank core and cold anvil arrays if no cores present
        elif ncores == 0:
            labelcorecold_number2d = np.zeros((ny, nx), dtype=int)
            labelcorecold_npix = []
            sortedcore_npix = []
            sortedcorecold_number2d = []  # KB TESTING
            sortedcore_npix = []
            sortedcold_npix = []
            sortedwarm_npix = []
            final_corecoldwarmnumber = np.zeros((ny, nx), dtype=int)

        ############################################################
        # Label cold anvils that do not have a cold core

        # Find indices that satisfy cold anvil threshold or convective core threshold and is not labeled
        isolated_flag = np.zeros((ny, nx), dtype=int)
        isolated_indices = np.where(
            (labelcorecold_number2d == 0) & ((coldanvil_flag > 0) | (core_flag > 0))
        )
        nisolated = np.shape(isolated_indices)[1]
        if nisolated > 0:
            isolated_flag[isolated_indices] = 1

        # Label isolated cold cores or cold anvils
        labelisolated_number2d, nlabelisolated = label(isolated_flag)

        # Check if any features have been identified
        if nlabelisolated > 0:

            #############################################################
            # Check if features satisfy size threshold
            labelisolated_npix = np.ones(nlabelisolated, dtype=int) * -9999
            idxs, idx_counts = np.unique(labelisolated_number2d, return_counts=True)
            for idx in idxs:
                if idx < 1 or idx > nlabelisolated + 1:
                    continue
                if idx_counts[idx] > nthresh:
                    labelisolated_npix[idx - 1] = idx_counts[idx]

            ###############################################################
            # Check if any of the features are retained
            ivalidisolated = np.array(np.where(labelisolated_npix > 0))[0, :]
            nlabelisolated = len(ivalidisolated)
            if nlabelisolated > 0:
                # Isolate cores that satisfy size threshold
                labelisolated_number1d = (
                    np.copy(ivalidisolated) + 1
                )  # Add one since label numbers start at 1 and indices, which valid cores reports starts at 0
                labelisolated_npix = labelisolated_npix[ivalidisolated]

                ###########################################################
                # Sort sizes largest to smallest
                order = np.argsort(labelisolated_npix)
                order = order[::-1]
                sortedisolated_npix = np.copy(labelisolated_npix[order])

                # Re-number cores
                sortedisolated_number1d = np.copy(labelisolated_number1d[order])

                sortedisolated_number2d = np.zeros((ny, nx), dtype=int)
                isolatedstep = 0
                for isortedisolated in range(0, nlabelisolated):
                    sortedisolated_indices = np.where(
                        labelisolated_number2d
                        == sortedisolated_number1d[isortedisolated]
                    )
                    nsortedisolatedindices = np.shape(sortedisolated_indices)[1]
                    if nsortedisolatedindices == sortedisolated_npix[isortedisolated]:
                        isolatedstep = isolatedstep + 1
                        sortedisolated_number2d[sortedisolated_indices] = np.copy(
                            isolatedstep
                        )
            else:
                sortedisolated_number2d = np.zeros((ny, nx), dtype=int)
                sortedisolated_npix = []
        else:
            sortedisolated_number2d = np.zeros((ny, nx), dtype=int)
            sortedisolated_npix = []

        ##############################################################
        # Combine cases with cores and cold anvils with those that those only have cold anvils

        # Add feature to core - cold anvil map giving it a number one greater that the number of valid cores.
        # These cores are after those that have a cold anvil.
        labelcorecoldisolated_number2d = np.copy(labelcorecold_number2d)

        sortedisolated_indices = np.where(sortedisolated_number2d > 0)
        nsortedisolatedindices = np.shape(sortedisolated_indices)[1]
        if nsortedisolatedindices > 0:
            labelcorecoldisolated_number2d[sortedisolated_indices] = np.copy(
                sortedisolated_number2d[sortedisolated_indices]
            ) + np.copy(ncores)

        # Combine the npix data for cases with cores and cold anvils with those that only have cold anvils
        labelcorecoldisolated_npix = np.hstack(
            (labelcorecold_npix, sortedisolated_npix)
        )
        ncorecoldisolated = len(labelcorecoldisolated_npix)

        # Initialize cloud numbers
        labelcorecoldisolated_number1d = np.arange(1, ncorecoldisolated + 1)

        # Sort clouds by size
        order = np.argsort(labelcorecoldisolated_npix)
        order = order[::-1]
        sortedcorecoldisolated_npix = np.copy(labelcorecoldisolated_npix[order])
        sortedcorecoldisolated_number1d = np.copy(labelcorecoldisolated_number1d[order])

        # Re-number cores
        sortedcorecoldisolated_number2d = np.zeros((ny, nx), dtype=int)
        final_ncorepix = np.ones(ncorecoldisolated, dtype=int) * -9999
        final_ncoldpix = np.ones(ncorecoldisolated, dtype=int) * -9999
        final_nwarmpix = np.ones(ncorecoldisolated, dtype=int) * -9999
        featurecount = 0
        for ifeature in range(0, ncorecoldisolated):
            #             feature_indices = np.where( # Find pixels that have matching #
            #                 labelcorecoldisolated_number2d
            #                 == sortedcorecoldisolated_number1d[ifeature]
            #             )
            #             nfeatureindices = np.shape(feature_indices)[1]
            feature_indices = (
                labelcorecoldisolated_number2d
                == sortedcorecoldisolated_number1d[ifeature]
            )
            nfeatureindices = np.count_nonzero(feature_indices)

            if nfeatureindices == sortedcorecoldisolated_npix[ifeature]:
                featurecount = featurecount + 1
                sortedcorecoldisolated_number2d[feature_indices] = featurecount

                final_ncorepix[featurecount - 1] = np.nansum(core_flag[feature_indices])
                final_ncoldpix[featurecount - 1] = np.nansum(
                    coldanvil_flag[feature_indices]
                )

        ##############################################
        # Save final matrices
        final_corecoldnumber = np.copy(sortedcorecoldisolated_number2d)
        final_ncorecold = np.copy(ncorecoldisolated)

        final_ncorepix = final_ncorepix[0:featurecount]
        final_ncoldpix = final_ncoldpix[0:featurecount]

        final_ncorecoldpix = final_ncorepix + final_ncoldpix

    ######################################################################
    # If no core is found, use cold anvil threshold to identify features
    else:
        #################################################
        # Label regions with cold anvils and cores
        corecold_flag = core_flag + coldanvil_flag
        corecold_number2d, ncorecold = label(coldanvil_flag)

        ##########################################################
        # Loop through clouds and only keep those where core + cold anvil exceed threshold
        if ncorecold > 0:
            labelcorecold_number2d = np.zeros((ny, nx), dtype=int)
            labelcore_npix = np.ones(ncorecold, dtype=int) * -9999
            labelcold_npix = np.ones(ncorecold, dtype=int) * -9999
            labelwarm_npix = np.ones(ncorecold, dtype=int) * -9999
            featurecount = 0

            for ifeature in range(1, ncorecold + 1):
                feature_indices = np.where(corecold_number2d == ifeature)
                nfeatureindices = np.shape(feature_indices)[1]

                if nfeatureindices > 0:
                    temp_core = np.copy(core_flag[feature_indices])
                    temp_corenpix = np.nansum(temp_core)

                    temp_cold = np.copy(coldanvil_flag[feature_indices])
                    temp_coldnpix = np.nansum(temp_cold)

                    if temp_corenpix + temp_coldnpix >= nthresh:
                        featurecount = featurecount + 1

                        labelcorecold_number2d[feature_indices] = np.copy(featurecount)
                        labelcore_npix[featurecount - 1] = np.copy(temp_corenpix)
                        labelcold_npix[featurecount - 1] = np.copy(temp_coldnpix)

            ###############################
            # Update feature count
            ncorecold = np.copy(featurecount)
            labelcorecold_number1d = (
                np.array(np.where(labelcore_npix + labelcold_npix > 0))[0, :] + 1
            )

            ###########################################################
            # Reduce size of final arrays so only as long as number of valid features
            if ncorecold > 0:
                labelcore_npix = labelcore_npix[0:ncorecold]
                labelcold_npix = labelcold_npix[0:ncorecold]
                labelwarm_npix = labelwarm_npix[0:ncorecold]

                ##########################################################
                # Reorder base on size, largest to smallest
                labelcorecold_npix = labelcore_npix + labelcold_npix + labelwarm_npix
                order = np.argsort(labelcorecold_npix)
                order = order[::-1]
                sortedcore_npix = np.copy(labelcore_npix[order])
                sortedcold_npix = np.copy(labelcold_npix[order])
                sortedwarm_npix = np.copy(labelwarm_npix[order])

                sortedcorecold_npix = np.add(sortedcore_npix, sortedcold_npix)

                # Re-number cores
                sortedcorecold_number1d = np.copy(labelcorecold_number1d[order])

                sortedcorecold_number2d = np.zeros((ny, nx), dtype=int)
                corecoldstep = 0
                for isortedcorecold in range(0, ncorecold):
                    sortedcorecold_indices = np.where(
                        labelcorecold_number2d
                        == sortedcorecold_number1d[isortedcorecold]
                    )
                    nsortedcorecoldindices = np.shape(sortedcorecold_indices)[1]
                    if nsortedcorecoldindices == sortedcorecold_npix[isortedcorecold]:
                        corecoldstep = corecoldstep + 1
                        sortedcorecold_number2d[sortedcorecold_indices] = np.copy(
                            corecoldstep
                        )

            ##############################################
            # Save final matrices
            final_corecoldnumber = np.copy(sortedcorecold_number2d)
            final_ncorecold = np.copy(ncorecold)
            final_ncorepix = np.copy(sortedcore_npix)
            final_ncoldpix = np.copy(sortedcold_npix)
            final_nwarmpix = np.copy(sortedwarm_npix)

            final_ncorecoldpix = final_ncorepix + final_ncoldpix
        else:
            final_corecoldnumber = np.zeros((ny, nx), dtype=int)
            final_corecoldwarmnumber = np.zeros((ny, nx), dtype=int)
            final_ncorecold = 0
            final_ncorepix = np.zeros((1,), dtype=int)
            final_ncoldpix = np.zeros((1,), dtype=int)
            final_nwarmpix = np.zeros((1,), dtype=int)
            final_ncorecoldpix = np.zeros((1,), dtype=int)

    ###################################################
    # Get warm anvils, if applicable
    if final_ncorecold > 0:
        if warmanvilexpansion == 1:
            labelcorecoldwarm_number2d = np.copy(final_corecoldnumber)
            ncorecoldwarmpix = np.copy(final_ncorecoldpix)

            keepspreading = 1
            # Keep looping through dilating code as long as at least one feature is growing.
            # At this point limit it to 20 dilations. Remove this once use real data.
            while keepspreading > 0:
                keepspreading = 0

                # Loop through each feature
                for ifeature in range(1, final_ncorecold + 1):
                    # Create map of single feature
                    featuremap = np.copy(labelcorecoldwarm_number2d)
                    featuremap[labelcorecoldwarm_number2d != ifeature] = 0
                    featuremap[labelcorecoldwarm_number2d == ifeature] = 1

                    # Find maximum extent of the of the feature
                    extenty = np.nansum(featuremap, axis=1)
                    extenty = np.array(np.where(extenty > 0))[0, :]
                    miny = extenty[0]
                    maxy = extenty[-1]

                    extentx = np.nansum(featuremap, axis=0)
                    extentx = np.array(np.where(extentx > 0))[0, :]
                    minx = extentx[0]
                    maxx = extentx[-1]

                    # Subset ir and map data to smaller region around feature.
                    # This reduces computation time. Add a 10 pixel buffer around the edges of the feature.
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

                    irsubset = ir[miny:maxy, minx:maxx]
                    fullsubset = labelcorecoldwarm_number2d[miny:maxy, minx:maxx]
                    featuresubset = featuremap[miny:maxy, minx:maxx]

                    # Dilate cloud region
                    dilationstructure = generate_binary_structure(
                        2, 1
                    )  # Defines shape of growth. This grows one pixel as a cross

                    dilatedsubset = binary_dilation(
                        featuresubset, structure=dilationstructure, iterations=1
                    ).astype(featuremap.dtype)

                    # Isolate region that was dilated.
                    expansionzone = dilatedsubset - featuresubset

                    # Only keep pixels in dilated regions that are below the warm anvil threshold
                    # and are not associated with another feature
                    expansionzone[
                        np.where((expansionzone == 1) & (fullsubset != 0))
                    ] = 0
                    expansionzone[
                        np.where((expansionzone == 1) & (irsubset >= thresh_warm))
                    ] = 0

                    # Find indices of accepted dilated regions
                    expansionindices = np.column_stack(np.where(expansionzone == 1))

                    # Add the accepted dilated region to the map of the cloud numbers
                    labelcorecoldwarm_number2d[
                        expansionindices[:, 0] + miny, expansionindices[:, 1] + minx
                    ] = ifeature

                    # Add the number of expanded pixels to pixel count
                    ncorecoldwarmpix[ifeature - 1] = (
                        len(expansionindices[:, 0]) + ncorecoldwarmpix[ifeature - 1]
                    )

                    # Count the number of dilated pixels. Add to the keepspreading variable.
                    # As long as this variables is > 0 the code continues to run the dilating portion.
                    # Also at this point have a requirement that can't dilate more than 20 times.
                    # This shoudl be removed when have actual data.
                    keepspreading = keepspreading + len(
                        np.extract(expansionzone == 1, expansionzone)
                    )

            ##############################################################################
            # Save final matrices
            final_corecoldwarmnumber = np.copy(labelcorecoldwarm_number2d)
            final_ncorecoldwarmpix = np.copy(ncorecoldwarmpix)
            final_nwarmpix = ncorecoldwarmpix - final_ncorecoldpix

        #######################################################################
        # If not expanding to warm anvil just copy core-cold data
        else:
            final_corecoldwarmnumber = np.copy(final_corecoldnumber)
            final_ncorecoldwarmpix = np.copy(final_ncorecoldpix)

    ##################################################################
    # Output data. Only done if core-cold exist in this file
    return {
        "final_nclouds": final_ncorecold,
        "final_ncorepix": final_ncorepix,
        "final_ncoldpix": final_ncoldpix,
        "final_ncorecoldpix": final_ncorecoldpix,
        "final_nwarmpix": final_nwarmpix,
        "final_cloudnumber": final_corecoldwarmnumber,
        "final_cloudtype": final_cloudid,
        "final_convcold_cloudnumber": final_corecoldnumber,
    }


def find_and_label_cold_cores(ncorepix, nx, ny, smoothir, thresh_core):
    #################################################################
    # Find cold cores in smoothed data
    smoothcore_flag = np.zeros((ny, nx), dtype=int)
    smoothcore_indices = np.where(smoothir < thresh_core)
    nsmoothcorepix = np.shape(smoothcore_indices)[1]
    if ncorepix > 0:
        smoothcore_flag[smoothcore_indices] = 1
    ##############################################################
    # Label cold cores in smoothed data
    labelcore_number2d, nlabelcores = label(smoothcore_flag)
    return labelcore_number2d, nlabelcores


def smooth_and_identify_indices(
    ir, nx, ny, smoothsize, thresh_cloud, thresh_cold, thresh_core, thresh_warm
):
    # Smooth IR data prior to identifying cores using a boxcar filter.
    # Along the edges the boundary elements come from the nearest edge pixel
    # smoothir = filters.uniform_filter(ir, size=smoothsize, mode='nearest')
    kernel = Box2DKernel(smoothsize)
    smoothir = convolve(
        ir, kernel, boundary="extend", nan_treatment="interpolate", preserve_nan=True
    )
    smooth_cloudid = np.zeros((ny, nx), dtype=int)
    core_indices = np.where(smoothir < thresh_core)
    ncorepix = np.shape(core_indices)[1]
    if ncorepix > 0:
        smooth_cloudid[core_indices] = 1
    coldanvil_indices = np.where((smoothir >= thresh_core) & (smoothir < thresh_cold))
    ncoldanvilpix = np.shape(coldanvil_indices)[1]
    if ncoldanvilpix > 0:
        smooth_cloudid[coldanvil_indices] = 2
    warmanvil_indices = np.where((smoothir >= thresh_cold) & (smoothir < thresh_warm))
    nwarmanvilpix = np.shape(warmanvil_indices)[1]
    if nwarmanvilpix > 0:
        smooth_cloudid[warmanvil_indices] = 3
    othercloud_indices = np.where((smoothir >= thresh_warm) & (smoothir < thresh_cloud))
    nothercloudpix = np.shape(othercloud_indices)[1]
    if nothercloudpix > 0:
        smooth_cloudid[othercloud_indices] = 4
    clear_indices = np.where(smoothir >= thresh_cloud)
    nclearpix = np.shape(clear_indices)[1]
    if nclearpix > 0:
        smooth_cloudid[clear_indices] = 5
    return ncorepix, smoothir


def generate_pixel_identification_from_threshold(
    ir, nx, ny, thresh_cloud, thresh_cold, thresh_core, thresh_warm
):
    """Use thresholds identify pixels containing cold core, cold anvil, and warm anvil.
    Also create arrays with a flag for each type and fill in cloudid array.
    Cores = 1. Cold anvils = 2. Warm anvils = 3. Other = 4. Clear = 5. Areas do not overlap

    """
    final_cloudid = np.zeros((ny, nx), dtype=int)
    core_flag = np.zeros((ny, nx), dtype=int)
    core_indices = np.where(ir < thresh_core)
    ncorepix = np.shape(core_indices)[1]
    if ncorepix > 0:
        core_flag[core_indices] = 1
        final_cloudid[core_indices] = 1
    coldanvil_flag = np.zeros((ny, nx), dtype=int)
    coldanvil_indices = np.where((ir >= thresh_core) & (ir < thresh_cold))
    ncoldanvilpix = np.shape(coldanvil_indices)[1]
    if ncoldanvilpix > 0:
        coldanvil_flag[coldanvil_indices] = 1
        final_cloudid[coldanvil_indices] = 2
    warmanvil_flag = np.zeros((ny, nx), dtype=int)
    warmanvil_indices = np.where((ir >= thresh_cold) & (ir < thresh_warm))
    nwarmanvilpix = np.shape(warmanvil_indices)[1]
    if nwarmanvilpix > 0:
        warmanvil_flag[coldanvil_indices] = 1
        final_cloudid[warmanvil_indices] = 3
    othercloud_flag = np.zeros((ny, nx), dtype=int)
    othercloud_indices = np.where((ir >= thresh_warm) & (ir < thresh_cloud))
    nothercloudpix = np.shape(othercloud_indices)[1]
    if nothercloudpix > 0:
        othercloud_flag[othercloud_indices] = 1
        final_cloudid[othercloud_indices] = 4
    clear_flag = np.zeros((ny, nx), dtype=int)
    clear_indices = np.where(ir >= thresh_cloud)
    nclearpix = np.shape(clear_indices)[1]
    if nclearpix > 0:
        clear_flag[clear_indices] = 1
        final_cloudid[clear_indices] = 5
    return coldanvil_flag, core_flag, final_cloudid
