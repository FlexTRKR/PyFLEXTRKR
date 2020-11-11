# Purpose: Identifies features and labels based on brightness temperature thresholds

# Comments:
# Based on pixel spreading method and follows Futyan and DelGenio [2007]

# Inputs:
# ir - brightness temperature array for region of interest
# pixel_radius - radius of pixel in km
# tb_threshs - brightness temperature thresholds datasource, datadescription,
# area_thresh - minimum area threshold of a feature [km^2]
# warmanvilexpncoldanvilpixansion - 1 = expand cold clouds out to warm anvils. 0 = Don't expand.


# Output: (Concatenated into one dictionary)
# final_nclouds - number of features identified
# final_cloudtype - 2D map of where the ir temperatures match the core, cold anvil, warm anvil, and other cloud requirements
# final_cloudnumber - 2D map that labels each feature with a number. With the largest core labeled as 1.
# final_ncorepix - number of core pixels in each feature
# final_ncoldpix - number of cold anvil pixels in each feature
# final_nwarmpix - number of warm anvil pixels in each feature


# Authors: Orginal IDL version written by Sally A. McFarlane (sally.mcfarlane@pnnl.gov) and modified by Zhe Feng (zhe.feng@pnnl.gov). Python version written by Hannah Barnes (hannah.barnes@pnnl.gov)

# Define function for futyan version 4 method. Isolates cold core then dilates to get cold and warm anvil regions.


# Define function for futyan version 3 method. Isolates cold core and cold anvil region, then dilates to get warm anvil region


# Define function for futyan version 4 method. Isolates cold core then dilates to get cold and warm anvil regions.


# Define function for futyan version 3 method. Isolates cold core and cold anvil region, then dilates to get warm anvil region NO SMOOTHING
