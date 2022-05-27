import numpy as np
from scipy import ndimage

def background_intensity(refl, mask_goodvalues, dx, dy, bkg_rad):
    """
    Calculate background reflectivity intensity
    ----------
    refl: np.ndarray(float)
        Radar reflectivity PPI (2D)
    dx: float
        Resolution on x-direction (meters)
    dy: float
        Resolution on y-direction (meters)
    bkg_rad: float
        Background radius value to calculate reflectivity intensity (meters)

    Returns
    ----------
    refl_bkg: np.ndarray(2D)
        Background reflectivity intensity.
    """

    # Convert background bkg_radius to number of grid points
    bkg_rad_x = int(bkg_rad / dx)
    bkg_rad_y = int(bkg_rad / dy)

    # Get a background radius mask
    ygrd, xgrd = np.ogrid[-bkg_rad_y:bkg_rad_y+1, -bkg_rad_x:bkg_rad_x+1]
    mask = xgrd*xgrd + ygrd*ygrd <= (bkg_rad/dx)*(bkg_rad/dy)

    ## another way to mask
    # mask = np.zeros((bkg_rad_x*2+1, bkg_rad_y*2+1))
    # mask[bkg_rad_x,bkg_rad_y]=1
    # mask = ndimage.binary_dilation(mask,iterations=bkg_rad_x)

    # Convert to linear unit
    linrefl = np.zeros(refl.shape)
    linrefl[mask_goodvalues==1] = 10. ** (refl[mask_goodvalues==1] / 10.)
    # Apply convolution filter 
    bkg_linrefl = ndimage.convolve(linrefl, mask, mode='constant', cval=0.0)
    numPixs = ndimage.convolve(mask_goodvalues, mask, mode='constant', cval=0.0)
    bkg_linrefl[mask_goodvalues==0]=0
    numPixs[mask_goodvalues==0]=0
    
    # Calculate average linear reflectivity and convert to log values
    refl_bkg = np.zeros(refl.shape)
    refl_bkg[numPixs>0] = 10.0 * np.log10(bkg_linrefl[numPixs>0] / numPixs[numPixs>0])
    
    # Remove pixels with 0 number of pixels
    refl_bkg[mask_goodvalues==0] = np.nan

    return refl_bkg


def peakedness(refl_bkg, mask_goodvalues, minZdiff, absConvThres):
    """
    Given a background reflectivity value, we determine what the necessary
    peakedness (or difference) has to be between a grid point's reflectivity
    and the background reflectivity in order for that grid point to be labeled
    convective
    """
    
    peak = np.zeros(refl_bkg.shape)
    
    peak[refl_bkg<0] = minZdiff
    peak[refl_bkg>=0] = minZdiff * np.cos(np.pi*refl_bkg/(2*absConvThres))[refl_bkg>=0]
    peak[refl_bkg>=absConvThres] = 0
    peak[mask_goodvalues==0] = np.nan

    return peak


def dilate_conv_rad(
        types_steiner,
        refl_bkg,
        sclass,
        score,
        dx,
        dy,
        mask_goodvalues,
        mindBZuse,
        maxConvRadius,
        dBZforMaxConvRadius
):
    """
    Given a mean background reflectivity value, we determine via a step
    function what the corresponding convective radius would be
    Higher background reflectivitives are expected to have larger convective
    influence on surrounding areas, so a larger convective radius would be
    prescribed
    """
    
    # define the step function
    conv_rad_bin = np.arange(maxConvRadius+1)
    nbin = conv_rad_bin.shape[0]
    bkg_bin = np.array([0,mindBZuse] + np.arange(dBZforMaxConvRadius-15,dBZforMaxConvRadius+5,5).tolist() + [100])
    
    # conv_rad
    conv_rad = np.ones_like(refl_bkg) - 0.99999
    for ii in range(nbin):
        
        if ii==0:
            ind = refl_bkg<=bkg_bin[ii+1]
        elif ii==nbin-1:
            ind = refl_bkg>=bkg_bin[ii]
        else:
            ind = np.logical_and(refl_bkg<bkg_bin[ii+1], refl_bkg>=bkg_bin[ii])
            
        conv_rad[ind] = conv_rad_bin[ii]
    
    # dilate
    mask_goodvalues = mask_goodvalues==1
    nr = np.unique(np.floor(np.unique(conv_rad))).astype(int)
    sclass_new = np.copy(sclass)
    for iradius in nr:
        
        if iradius==0:
            continue
    
        ind = np.logical_and(np.abs(conv_rad-iradius)<0.5, score==1)

        conv_rad_gridx = int(iradius * 1000 / dx)
        conv_rad_gridy = int(iradius * 1000 / dy)
        
        xgrd, ygrd = np.ogrid[-conv_rad_gridx:conv_rad_gridx+1, -conv_rad_gridy:conv_rad_gridy+1]
        # strc = xgrd*xgrd + ygrd*ygrd <= conv_rad_gridx*conv_rad_gridy
        strc = xgrd*xgrd + ygrd*ygrd <= (iradius*1000/dx)*(iradius*1000/dy)
        
        ind_final = ndimage.binary_dilation(ind,strc,mask=mask_goodvalues)
        sclass_new[ind_final] = types_steiner['CONVECTIVE'] 
    
    return sclass_new


def make_dilation_step_func(
        mindBZuse=25,
        dBZforMaxConvRadius=40,
        bkg_refl_increment=5,
        conv_rad_increment=1,
        conv_rad_start=1,
        maxConvRadius=5
):
    """
    Makes convective radius dilation step function (Fig. 6 in Steiner et al. 1995 JAM)

    Parameters:
    ===========
    mindBZuse: float
        Minimum background reflectivity bin value (default 25 dBZ)
    dBZforMaxConvRadius: float
        Maximum background reflectivity bin value (default 40 dbZ)
    bkg_refl_increment: float
        Background reflectivity step function bin increment (default 5 dB)
    conv_rad_increment: float
        Convective radius step function increment (default 1 km)
    conv_rad_start: float
        Convective radius step function start value (default 1 km)
    maxConvRadius: float
        Maximum convective radius to cap the step function (default 5 km)

    Returns:
    ===========
    bkg_bin: ndarray
        Background reflectivity bin values
    conv_rad_bin: ndarray
        Convective radius bin values
    """

    # Define background mean reflectivity step function bins
    bkg_stepfunc_bin = np.arange(mindBZuse, dBZforMaxConvRadius+0.1, bkg_refl_increment)
    # Extend step function bins to min 0 dBZ and max 100 dBZ
    bkg_bin = np.array([0] + bkg_stepfunc_bin.tolist() + [100])

    # Create convective radii values matching step function bins
    nbin_bkg = len(bkg_stepfunc_bin)
    # Calculate convective radii bin values. Start value is defined by conv_rad_start
    conv_rad_bin = np.arange(0, nbin_bkg+1, 1) * conv_rad_increment + conv_rad_start
    # Cap the convective radius to maxConvRadius
    conv_rad_bin[np.where(conv_rad_bin > maxConvRadius)] = maxConvRadius

    return bkg_bin, conv_rad_bin


def mod_dilate_conv_rad(
        types_steiner,
        refl_bkg,
        sclass,
        score,
        mask_goodvalues,
        dx,
        dy,
        bkg_bin,
        conv_rad_bin
):
    """
    Given a mean background reflectivity value, we determine via a step function
    what the corresponding convective radius would be to dilate the convective cores.
    Higher background reflectivitives are expected to have larger convective
    influence on surrounding areas, so a larger convective radius would be
    prescribed

    Parameters:
    ===========
    types_steiner: dict
        Steiner classification type value dictionary
    refl_bkg: ndarray
        Background mean reflectivity array
    sclass: ndarray
        Steiner convective/stratiform classification array
    score: ndarray
        Steiner convective core array, same size as scalss
    mask_goodvalues: ndarray
        Mask array to indicate good reflectivity values, same size as sclass
    dx: float
        Resolution on x-direction (meters)
    dy: float
        Resolution on y-direction (meters)
    bkg_bin: ndarray (1D)
        Background reflectivity bins for dilation step function
    conv_rad_bin: ndarray (1D)
        Convective radius bins for dilation step function

    Returns:
    ===========
    sclass_new: ndarray <int>
        Updated convective/Stratiform classification, same size as sclass.
    score_dilate: ndarray <int>
        Dilated convetive core, same size as sclass_new.
    """

    # For pixel given a background reflectivity value, assign a convective radius expansion value
    conv_rad = np.ones_like(refl_bkg) - 0.99999

    nbin = len(conv_rad_bin)
    for ii in range(nbin):

        if ii==0:
            ind = refl_bkg<=bkg_bin[ii+1]
        elif ii==nbin-1:
            ind = refl_bkg>=bkg_bin[ii]
        else:
            ind = np.logical_and(refl_bkg<bkg_bin[ii+1], refl_bkg>=bkg_bin[ii])

        conv_rad[ind] = conv_rad_bin[ii]

    # Create new arrays to store the output
    score_dilate = np.copy(score)
    sclass_new = np.copy(sclass)

    mask_goodvalues = mask_goodvalues==1
    nr = np.unique(conv_rad)

    # Loop over each convective radius bin
    for iradius in nr:

        # No expansion for radius == 0
        if iradius == 0:
            continue

        # Find indices of cores closest to a certain convective radius bin value
        ind = np.logical_and(np.abs(conv_rad-iradius)<0.01, score==1)

        # Convert radius from [m] to number of grid points
        conv_rad_gridx = int(iradius * 1000 / dx)
        conv_rad_gridy = int(iradius * 1000 / dy)

        # Create a circular structure
        xgrd, ygrd = np.ogrid[-conv_rad_gridx:conv_rad_gridx+1, -conv_rad_gridy:conv_rad_gridy+1]
        # strc = xgrd*xgrd + ygrd*ygrd <= conv_rad_gridx*conv_rad_gridy
        strc = xgrd*xgrd + ygrd*ygrd <= (iradius*1000/dx) * (iradius*1000/dy)

        # Expand the convective cores
        ind_final = ndimage.binary_dilation(ind,strc,mask=mask_goodvalues)
        score_dilate[ind_final] = 1

        # Update Steiner classification for convective
        sclass_new[ind_final] = types_steiner['CONVECTIVE']

    return sclass_new, score_dilate


def label_cells(convmask, min_cellpix):
    """
    Labels convective cells, and returns sorted cell number arrays by size.
    ----------
    convmask: np.ndarray()
        Binary convective mask array.
    min_cellpix: float
        Minimum number of pixel to count as a cell.

    Returns
    ----------
    sortedlabelcell_number2d: np.ndarray(int)
        Labeled cell number array in 2D.
    sortedcell_npix: np.ndarray(int)
        Number of pixels for each labeled cell in 1D.
    """

    # Create output arrays
    sortedlabelcell_number2d = np.zeros(convmask.shape, dtype=int)

    # Label convective cells
    labelcell_number2d, nlabelcells = ndimage.label(convmask)

    # Check if there is any cells identified
    if (nlabelcells > 0):

        labelcell_npix = np.full(nlabelcells, -999, dtype=int)

        # Loop over each labeled cell
        for ilabelcell in range(1, nlabelcells + 1):
            # Count number of pixels for the cell
            ilabelcell_npix = np.count_nonzero(labelcell_number2d == ilabelcell)
            # Check if cell satisfies size threshold
            if (ilabelcell_npix > min_cellpix):
                labelcell_npix[ilabelcell - 1] = ilabelcell_npix

        # Check if any of the cells passes the size threshold test
        ivalidcells = np.array(np.where(labelcell_npix > 0))[0, :]
        ncells = len(ivalidcells)

        if (ncells > 0):
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
                sortedcell_indices = np.where(labelcell_number2d == sortedcell_number1d[icell])
                # Get one of the dimensions from the 2D indices to count the size
                #             nsortedcellindices = np.shape(sortedcell_indices)[1]
                nsortedcellindices = len(sortedcell_indices[1])
                # Check if the size matches the sorted cell size
                if (nsortedcellindices == sortedcell_npix[icell]):
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


def expand_conv_core(score, radii_expand, dx, dy, min_corenpix=1):
    """
    Expand convective cores outward to a set of specified radii sequentially.
    
    Parameters:
    ===========
    score: ndarray
        Convetive core array (2D array)
    radii_expand: ndarray
        Radii values to expand
    min_corenpix: int, optional
        Minimum number of pixels to label a core (default 1)

    Returns:
    ===========
    score_expand: ndarray <int>
        Expanded convetive core, numbered and sorted by size, same size as score
    score_sorted: ndarray <int>
        Convective core array, numbered and sorted by size, same size as score
    """

    # Sort and renumber the cores by size
    score_sorted, sortedcell_npix = label_cells(score, min_corenpix)
    ncores = len(sortedcell_npix)

    # Initialize expanded core array
    score_expand = np.copy(score_sorted)

    # Check if a convective core exists
    if (ncores > 0):

        # Loop over each radius value
        for iradius in radii_expand:

            # Loop over each core
            for ic in range(1, ncores+1):

                # Create a binary mask for the current cell
                coremap = np.zeros(score_sorted.shape, dtype=int)
                coremap[score_sorted == ic] = 1

                # Make a mask for dilatable region (this gets updated every iteration)
                mask_dilatable = (score_expand == 0)

                # Convert radius from [m] to number of grid points
                conv_rad_gridx = int(iradius * 1000 / dx)
                conv_rad_gridy = int(iradius * 1000 / dy)

                # Create a structure for dilation
                xgrd, ygrd = np.ogrid[-conv_rad_gridx:conv_rad_gridx+1, -conv_rad_gridy:conv_rad_gridy+1]
                # strc = xgrd*xgrd + ygrd*ygrd <= conv_rad_gridx*conv_rad_gridy
                strc = xgrd*xgrd + ygrd*ygrd <= (iradius*1000/dx) * (iradius*1000/dy)

                # Dilate the core, mask with dilatable area, and assign with the core number
                coremap_dilate = ndimage.binary_dilation(coremap, strc, mask=mask_dilatable)
                score_expand[coremap_dilate == 1] = ic

    return score_expand, score_sorted


def steiner_classification(
        types_steiner,
        refl,
        dx,
        dy,
        bkg_rad,
        minZdiff,
        absConvThres,
        mindBZuse,
        maxConvRadius,
        dBZforMaxConvRadius,
        truncZconvThres,
        weakEchoThres,
):
    """
    We perform the Steiner et al. (1995) algorithm for echo classification
    using only the reflectivity field in order to classify each grid point
    as either convective, stratiform or undefined. Grid points are classified
    as follows,
    0 = NO_SFC_ECHO
    1 = WEAK_ECHO
    2 = Stratiform
    3 = Convective

    Parameters:
    ===========
    refl: ndarray
        Reflectivity slice (2D Cartesion grid).
    x: ndarray
        x-coordinates
    y: ndarray
        y-coordinates
    dx: float
        Resolution on x-direction (meters)
    dy: float
        Resolution on y-direction (meters)
    bkg_rad: float
        Radius to calculate background reflectivity intensity (in meters).

    Returns:
    ========
    sclass: ndarray <int>
        Convective/Stratiform classification, same size as refl.
    score: ndarray <int>
        Convetive Core, same size as refl. 
    """

    score = np.zeros(refl.shape, dtype=int)
    sclass = np.zeros(refl.shape, dtype=int) 
    
    mask_goodvalues = np.ones(refl.shape, dtype=int)
    mask_goodvalues[np.isnan(refl)] = 0
    ny, nx = refl.shape
    
    # Calculate background reflectivity
    refl_bkg = background_intensity(refl, mask_goodvalues, dx, dy, bkg_rad)

    # If refl below truncZconvThres, use peakedness criteria
    peak = peakedness(refl_bkg, mask_goodvalues, minZdiff, absConvThres)
    
    # classification
    sclass = np.ones_like(refl) - 1 # initialized with 0
    
    sclass[mask_goodvalues==1] = types_steiner['STRATIFORM'] 
    
    ind_core = np.logical_or(refl >= truncZconvThres, (refl - refl_bkg) >= peak) # define the convective core
    score[ind_core] = 1
    sclass[ind_core] = types_steiner['CONVECTIVE'] 
    
    ind_weak = np.logical_and(refl>minZdiff, refl<weakEchoThres) # define the weak echo region
    sclass[ind_weak] = types_steiner['WEAK_ECHO'] 
    
    ind_nosfc = np.logical_and(mask_goodvalues==1, refl<minZdiff) # define the no surface echo region
    sclass[ind_nosfc] = types_steiner['NO_SURF_ECHO']  
        
    # Dilate convective radius 
    sclass_new = dilate_conv_rad(types_steiner, refl_bkg, sclass, score, dx, dy, mask_goodvalues,mindBZuse, maxConvRadius, dBZforMaxConvRadius)
    
    return (sclass_new,score)


def mod_steiner_classification(
        types_steiner,
        refl,
        mask_goodvalues,
        dx,
        dy,
        bkg_rad,
        minZdiff,
        absConvThres,
        truncZconvThres,
        weakEchoThres,
        bkg_bin,
        conv_rad_bin,
        min_corearea=1,
        remove_smallcores=True,
        return_diag=False,
):
    """
    Modified Steiner et al. (1995) algorithm for echo classification using the reflectivity field
    to classify each grid point as either convective, stratiform, weak echo or undefined.
    The main difference is added an optional filter to remove cores smaller than a specified size.
    The convective radius dilation function is also modified to directly take in the background reflectivity 
    and convective radius step function for more flexibility.

    Parameters:
    ===========
    types_steiner: dict
        Steiner classification type value dictionary
    refl: ndarray
        Reflectivity array (2D Cartesion grid)
    mask_goodvalues: ndarray
        Mask array to indicate good reflectivity values, same size as sclass
    dx: float
        Resolution on x-direction (meters)
    dy: float
        Resolution on y-direction (meters)
    bkg_rad: float
        Radius to calculate background reflectivity intensity (in meters).
    minZdiff: float
        Minimum reflectivity difference between grid point and background for the convecitve core cosine function
    absConvThres: float
        Reflectivity threshold to define convective core in the core cosine function (the falloff point for the cosine curve)
    truncZconvThres: float
        Reflectivity threshold to define convective core (Ze > truncZconvThres automatically convective)
    weakEchoThres: float
        Reflectivity threshold to define weak echo (Ze < weakEchoThres is weak echo)
    bkg_bin: ndarray (1D)
        Background reflectivity bins for dilation step function
    conv_rad_bin: ndarray (1D)
        Convective radius bins for dilation step function
    min_corearea: float, optional
        Minimum area to keep a convective core (km^2) (default 1)
    remove_smallcores: bool, optional
        A flag to remove convective cores smaller than min_corearea (default True)
    return_diag: bool, optional
        A flag to return more fields for diagnostic purpose (default False)

    Returns:
    ===========
    sclass_new: ndarray <int>
        Convective/Stratiform classification, same size as refl
    score: ndarray <int>
        Convective core array, same size as refl
    score_dilate: ndarray <int>
        Dilated convetive core, same size as refl
    """

    # Calculate background reflectivity
    refl_bkg = background_intensity(refl, mask_goodvalues, dx, dy, bkg_rad)

    # If refl below truncZconvThres, use peakedness criteria
    peak = peakedness(refl_bkg, mask_goodvalues, minZdiff, absConvThres)

    score = np.zeros(refl.shape, dtype=int)
    sclass = np.zeros(refl.shape, dtype=int)

    # Default is stratiform
    sclass[mask_goodvalues==1] = types_steiner['STRATIFORM']

    # Assign convective core (Ze > truncZconvThres, or Ze - Ze_bkg >= peak)
    # define the convective core
    ind_core = np.logical_or(refl >= truncZconvThres, (refl - refl_bkg) >= peak)
    score[ind_core] = 1
    sclass[ind_core] = types_steiner['CONVECTIVE']

    # define the weak echo region
    ind_weak = np.logical_and(refl>minZdiff, refl<weakEchoThres)
    sclass[ind_weak] = types_steiner['WEAK_ECHO']

    # define the no surface echo region
    ind_nosfc = np.logical_and(mask_goodvalues==1, refl<minZdiff)
    sclass[ind_nosfc] = types_steiner['NO_SURF_ECHO']

    # If remove_smallcores is set, then remove cores with pixels < min_corearea
    if (remove_smallcores == True):
        # Make a copy of the core array
        score_keep = np.copy(score)

        # Convert min_corearea to number of pixels
        # dx, dy are in [meter], min_corearea is in [km^2], convert all to [meter]
        min_corenpix = int(min_corearea * (1000**2) / (dx * dy))

        # Remove small cores
        # Label connected core pixels as regions
        tmpregions, num_regions = ndimage.label(score_keep)
        for rr in range(1, num_regions+1):
            rid = np.where(tmpregions == rr)
            if (len(rid[0]) < min_corenpix):
                score_keep[rid] = 0

    # Dilate convective radius
    sclass_new, score_dilate = mod_dilate_conv_rad(
        types_steiner,
        refl_bkg,
        sclass,
        score_keep,
        mask_goodvalues,
        dx,
        dy,
        bkg_bin,
        conv_rad_bin
    )

    if (return_diag == False):
        return sclass_new, score_keep, score_dilate
    if (return_diag == True):
        return sclass_new, score_keep, score_dilate, refl_bkg, peak, score
