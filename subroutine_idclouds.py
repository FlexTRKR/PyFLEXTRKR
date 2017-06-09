# Purpose: Identifies features and labels based on brightness temperature thresholds

# Comments:
# Based on pixel spreading method and follows Futyan and DelGenio [2007]

# Inputs:
# ir - brightness temperature array for region of interest
# pixel_radius - radius of pixel in km
# tb_threshs - brightness temperature thresholds datasource, datadescription,
# area_thresh - minimum area threshold of a feature [km^2]
# warmanvilexpansion - 1 = expand cold clouds out to warm anvils. 0 = Don't expand.


# Output: (Concatenated into one dictionary)
# final_nclouds - number of features identified
# final_cloudtype - 2D map of where the ir temperatures match the core, cold anvil, warm anvil, and other cloud requirements
# final_cloudnumber - 2D map that labels each feature with a number. With the largest core labeled as 1.
# final_ncorepix - number of core pixels in each feature
# final_ncoldpix - number of cold anvil pixels in each feature
# final_nwarmpix - number of warm anvil pixels in each feature 


# Authors: Orginal IDL version written by Sally A. McFarlane (sally.mcfarlane@pnnl.gov) and modified by Zhe Feng (zhe.feng@pnnl.gov). Python version written by Hannah Barnes (hannah.barnes@pnnl.gov)


# Define function for futyan method
def futyan(ir, pixel_radius, tb_threshs, area_thresh, warmanvilexpansion):
    ######################################################################
    # Import modules
    import numpy as np
    from scipy.ndimage import label, binary_dilation, generate_binary_structure

    ######################################################################
    # Define constants:
    # Separate array threshold
    thresh_core = tb_threshs[0]     # Convective core threshold [K]
    thresh_cold = tb_threshs[1]     # Cold anvil threshold [K]
    thresh_warm = tb_threshs[2]     # Warm anvil threshold [K]
    thresh_cloud = tb_threshs[3]    # Warmest cloud area threshold [K]

    # Determine dimensions
    ny, nx = np.shape(ir)

    # Calculate area of one pixel. Assumed to be a circle.
    pixel_area = pixel_radius**2

    ######################################################################
    # Use thresholds to make a map of all brightnes temperatures that fit within the criteria for convective, cold anvil, and warm anvil points. Cores = 1. Cold anvils = 2. Warm anvils = 3. Other = 4. Clear = 5. Areas do not overlap
    final_cloudtype = np.ones((ny,nx), dtype=int)*-1
    final_cloudtype[np.where(ir < thresh_core)] = 1
    final_cloudtype[np.where((ir >= thresh_core) & (ir < thresh_cold))] = 2
    final_cloudtype[np.where((ir >= thresh_cold) & (ir < thresh_warm))] = 3
    final_cloudtype[np.where((ir >= thresh_warm) & (ir < thresh_cloud))] = 4
    final_cloudtype[np.where(ir >= thresh_cloud)] = 5

    ######################################################################
    # Create map of potential features to track. These features encompass the cores and cold anvils
    convective_flag = np.zeros((ny,nx), dtype=int)
    convective_flag[ir < thresh_cold] = 1

    #####################################################################
    # Label features 
    convective_label, convective_number = label(convective_flag)

    #####################################################################
    # Loop through each feature and determine if it statstifies the area requirement. Do this by finding the number of pixels covered by the feature, multiple by pixel area, and compare the area threshold requirement. 
    if convective_number > 0:
        # Initialize vectors of conveective number, number of pixels, and area to record features that statisfy area requirement
        approved_convnumber = np.empty(convective_number, dtype=int)*np.nan
        approved_convpixels= np.empty(convective_number, dtype=int)*np.nan
        approved_convarea = np.empty(convective_number, dtype=int)*np.nan

        for featurestep, ifeature in enumerate(range(1, convective_number+1)):
            # Identify pixels from each feature and multiple by pixel area to get feature
            feature_pixels = len(np.extract(convective_label == ifeature, convective_label))
            feature_area = feature_pixels*pixel_area

            # If statisfies store the feature number and its area 
            if feature_area > area_thresh:
                approved_convnumber[featurestep] = ifeature
                approved_convpixels[featurestep] = feature_pixels
                approved_convarea[featurestep] = feature_area

        # Remove blank rows in approved matrices. Itialized so has same length as if all cells passed, but that is not neceesary true 
        extrarows = np.array(np.where(np.isnan(approved_convnumber)))[0,:]
        if len(extrarows) > 0:
            approved_convnumber = np.delete(approved_convnumber, extrarows)
            approved_convpixels = np.delete(approved_convpixels, extrarows)
            approved_convarea = np.delete(approved_convarea, extrarows)

        ####################################################################
        # Reorder number final features based on descending area (i.e. largest to smallest)
        approved_number = len(approved_convnumber)

        if approved_number > 0:
            ordered = np.argsort(approved_convarea)
            ordered = ordered[::-1] # flips order so largest listed first

            approved_convnumber = approved_convnumber[ordered]
            final_convpixels = approved_convpixels[ordered]
            final_convarea = approved_convarea[ordered]

            # Create a map of the new labels. Needed for get warm anvil portion portion
            final_cloudnumber = np.zeros((ny,nx), dtype=int)
            for corrected, ifeature in enumerate(approved_convnumber):
                final_cloudnumber[np.where(convective_label == ifeature)] = corrected+1

            # Create map of cloudnumber labeling only core and cold anvil regions. This is done since if the expansion into the warm anvil occurrs, final_cloudnumber is changed to include those regions. It is important to have this final_convcold_cloudnumber since only the core and cold anvil are tracked. 
            final_convcold_cloudnumber = np.copy(final_cloudnumber)

            # Count the number of features in the data
            final_nclouds = approved_number

            ##################################################################
            # Add the warm anvil to features by dilating the cold anvil + core region outward.
            if warmanvilexpansion == 1:
                keepspreading = 1

                # Keep looping through dilating code as long as at least one feature is growing. At this point limit it to 20 dilations. Remove this once use real data.
                while keepspreading > 0:
                    keepspreading = 0

                    # Loop through each feature
                    for ifeature in range(1,final_nclouds+1):
                        # Create map of single feature
                        featuremap = np.copy(final_cloudnumber)
                        featuremap[final_cloudnumber != ifeature] = 0
                        featuremap[final_cloudnumber == ifeature] = 1

                        # Find maximum extent of the of the feature
                        extenty = np.nansum(featuremap, axis=1)
                        extenty = np.array(np.where(extenty > 0))[0,:]
                        miny = extenty[0]
                        maxy = extenty[-1]

                        extentx = np.nansum(featuremap, axis=0)
                        extentx = np.array(np.where(extentx > 0))[0,:]
                        minx = extentx[0]
                        maxx = extentx[-1]

                        # Subset ir and map data to smaller region around feature. This reduces computation time. Add a 10 pixel buffer around the edges of the feature.  
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
                        fullsubset = final_cloudnumber[miny:maxy, minx:maxx]
                        featuresubset = featuremap[miny:maxy, minx:maxx]

                        # Dilate cloud region
                        dilationstructure = generate_binary_structure(2,1)  # Defines shape of growth. This grows one pixel as a cross

                        dilatedsubset = binary_dilation(featuresubset, structure=dilationstructure, iterations=1).astype(featuremap.dtype)
                        
                        # Isolate region that was dilated.
                        expansionzone = dilatedsubset - featuresubset

                        # Only keep pixels in dilated regions that are below the warm anvil threshold and are not associated with another feature
                        expansionzone[np.where((expansionzone == 1) & (fullsubset != 0))] = 0
                        expansionzone[np.where((expansionzone == 1) & (irsubset >= thresh_warm))] = 0

                        # Find indices of accepted dilated regions
                        expansionindices = np.column_stack(np.where(expansionzone == 1))

                        # Add the accepted dilated region to the map of the cloud numbers
                        final_cloudnumber[expansionindices[:,0]+miny, expansionindices[:,1]+minx] = ifeature

                        # Count the number of dilated pixels. Add to the keepspreading variable. As long as this variables is > 0 the code continues to run the dilating portion. Also at this point have a requirement that can't dilate more than 20 times. This shoudl be removed when have actual data.
                        keepspreading = keepspreading + len(np.extract(expansionzone == 1, expansionzone))

        ################################################################
        # Once dilation complete calculate the number of core, cold, and warm pixels in each feature. Also create a map of cloud number for only core and cold region

        # Initialize vectors
        final_ncorepix = np.ones(final_nclouds, dtype=int)*-9999
        final_ncoldpix = np.ones(final_nclouds, dtype=int)*-9999
        final_ncorecoldpix = np.ones(final_nclouds, dtype=int)*-9999
        final_nwarmpix = np.ones(final_nclouds, dtype=int)*-9999

        # Loop through each feature
        for indexstep, ifeature in enumerate(np.arange(1,final_nclouds+1)):
            final_ncorepix[indexstep] = len(np.extract((final_convcold_cloudnumber == ifeature) & (final_cloudtype == 1), final_convcold_cloudnumber))
            final_ncoldpix[indexstep] = len(np.extract((final_convcold_cloudnumber == ifeature) & (final_cloudtype == 2), final_convcold_cloudnumber))
            final_ncorecoldpix[indexstep] = len(np.extract((final_convcold_cloudnumber == ifeature), final_convcold_cloudnumber))
            final_nwarmpix[indexstep] = len(np.extract((final_cloudnumber == ifeature) & (final_cloudtype == 3), final_cloudnumber))

        ##################################################################
        # Output data
        return {'final_nclouds':final_nclouds, 'final_ncorepix':final_ncorepix, 'final_ncoldpix':final_ncoldpix, 'final_ncorecoldpix':final_ncorecoldpix, 'final_nwarmpix':final_nwarmpix, 'final_cloudnumber':final_cloudnumber, 'final_cloudtype':final_cloudtype, 'final_convcold_cloudnumber':final_convcold_cloudnumber}
        



