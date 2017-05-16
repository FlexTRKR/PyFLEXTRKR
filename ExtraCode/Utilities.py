import numpy as np
from math import pi
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

####################################################################
# Define fucntion to calculate aspect ratio and associated statistics of a cloud
def shapecharacteristics(cloudmap):

    # Purpose: Measures aspect ration and a set of properities for each of the connected components in the cloud map (which is a 2D binary representation of the location of the cloud)

    # Input:
    # cloudmap - 2D array with 0s and 1s indicating the location the cloud, must be a connected region

    # Notes:
    # Adapted from Matlab function called regionprops.m and chapter 3.2 "Object Dection" from www.cs.ucf.edu/~subh/docs/pubs/FLIR_Aerial-Tracking_chapter.pdf

    # Author:
    # Orginal IDL version written by Matus Martini. Ported into Python by Hannah C. Barnes (hannah.barnes@pnnl.gov)

    ############################################################
    # Compute centroid
    cloudindices = np.array(np.where(cloudmap == 1))
    xcenter = np.nanmean(cloudindices[1,:])
    ycenter = np.nanmean(cloudindices[0,:])

    ###########################################################
    # Define x and y so that orientation is measured counterclockwise from the horizontal axis
    x = cloudindices[1,:] - xcenter
    y = -1*(cloudindices[0,:] - ycenter) # y is made negative so the orientation calculation is measured counterclockwise from the horizontal.

    ##########################################################
    # Determine number of pixels
    n = len(x)

    ############################################################
    # Calculate normalized second central moments of the region. 1/12 is the normalized second central moment of a pixel with unit length
    uxx = np.divide(np.nansum(np.square(x)), n) + 1/12
    uyy = np.divide(np.nansum(np.square(y)), n) + 1/12
    uxy = np.divide(np.nansum(np.multiply(x, y)), n)

    ###########################################################
    # Calculate major axis, minor axis, aspect ratio, and eccentricity
    commonxy = np.sqrt(np.square(uxx-uyy) + 4*np.square(uxy))

    majoraxislength = 2*np.sqrt(2)*np.sqrt(uxx + uyy + commonxy)
    minoraxislength = 2*np.sqrt(2)*np.sqrt(uxx + uyy - commonxy)

    aspectratio = np.divide(majoraxislength, minoraxislength)

    eccentricity = np.divide(2*np.sqrt(np.square(np.divide(majoraxislength, 2)) - np.square(np.divide(minoraxislength, 2))), majoraxislength)

    ##########################################################
    # Calculate orientation
    if uyy > uxx:
        numerator = uyy - uxx + np.sqrt(np.square(uyy - uxx) + 4*np.square(uxy))
        denominator = 2*uxy
    else:
        numerator = 2*uxy
        denominator = uxx - uyy + np.sqrt(np.square(uxx -uyy) + 4*np.square(uxy))

    if numerator == 0 and denominator == 0:
        orientation = 0
    else:
        orientation = np.multiply(np.divide(180,pi), np.arctan(np.divide(numerator, denominator)))

    print('')
    print(xcenter)
    print(ycenter)
    print(majoraxislength)
    print(minoraxislength)
    print(eccentricity)
    print(orientation)

    
    ellipse = Ellipse(xy=(xcenter,ycenter), width=minoraxislength, height=majoraxislength, angle=90-orientation, facecolor='None', linewidth=3, edgecolor='y')
    fig, ax = plt.subplots(1,2)
    ax[0].pcolor(cloudmap)
    ax[0].scatter(xcenter,ycenter, s=30, color='y')
    ax[0].add_patch(ellipse)
    plt.show()

    ##########################################################
    # Return aspect ratio and statisics
    return {'aspectratio':aspectratio, 'majoraxislength':majoraxislength, 'minoraxislength':minoraxislength, 'eccentricity':eccentricity, 'orientation':orientation, 'xcentroid':xcenter, 'ycentroid':ycenter}





