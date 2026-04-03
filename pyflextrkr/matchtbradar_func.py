import numpy as np
import os.path
import sys
import logging
import xarray as xr
from scipy.ndimage import label
from skimage.measure import regionprops
from math import pi
from scipy.stats import skew
import warnings
from pyflextrkr.ftfunctions import sort_renumber, get_cloud_boundary, circular_mean
from pyflextrkr.ft_utilities import subset_ds_geolimit, get_pixel_area

def matchtbpf_singlefile(
    cloudid_filename,
    ir_cloudnumber,
    ir_mergecloudnumber,
    ir_splitcloudnumber,
    config,
):
    """
    Calculate PF statistics within Tb tracked MCS from a single pixel file.

    Args:
        cloudid_filename: string
            Cloudid file name.
        ir_cloudnumber: numpy array
            Cloudnumbers within this cloudid file.
        ir_mergecloudnumber: numpy array
            Cloudnumbers for merging clouds in this cloudid file.
        ir_splitcloudnumber: numpy array
            Cloudnumbers for splitting clouds in this cloudid file.
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        out_dict: dictionary
            Dictionary containing the track statistics data.
        out_dict_attrs: dictionary
            Dictionary containing the attributes of track statistics data.
    """
    logger = logging.getLogger(__name__)

    feature_varname = config.get("feature_varname", "feature_number")
    pf_rr_thres = config["pf_rr_thres"]
    pf_link_area_thresh = config["pf_link_area_thresh"]
    heavy_rainrate_thresh = config["heavy_rainrate_thresh"]
    pixel_radius = config["pixel_radius"]
    area_method = config.get("area_method", "fixed")
    pbc_direction = config.get("pbc_direction", "none")
    max_feature_frac_x = config.get("max_feature_frac_x", 0.95)
    max_feature_frac_y = config.get("max_feature_frac_y", 0.95)
    nmaxpf = config["nmaxpf"]
    mcs_core_min_area = config.get("mcs_core_min_area", 0)
    # ZF: nmaxcore cannot be different from nmaxpf without changing matchtbpf_driver.py
    # For now, make them the same
    nmaxcore = nmaxpf
    # nmaxcore = config.get("nmaxcore", 10)
    landmask_filename = config.get("landmask_filename", "")
    landmask_varname = config.get("landmask_varname", "")
    landfrac_thresh = config.get("landfrac_thresh", 0)
    landmask_x_dimname = config.get("landmask_x_dimname", None)
    landmask_y_dimname = config.get("landmask_y_dimname", None)
    landmask_x_coordname = config.get("landmask_x_coordname", None)
    landmask_y_coordname = config.get("landmask_y_coordname", None)

    fillval = config["fillval"]
    fillval_f = np.nan

    # Read landmask file
    if os.path.isfile(landmask_filename):
        dslm = xr.open_dataset(landmask_filename).squeeze()
        # Subset landmask to match geolimit
        dslm = subset_ds_geolimit(
            dslm, config,
            x_coordname=landmask_x_coordname,
            y_coordname=landmask_y_coordname,
            x_dimname=landmask_x_dimname,
            y_dimname=landmask_y_dimname,
        )
        landmask = dslm[landmask_varname].squeeze().data
    else:
        landmask = None


    # Read cloudid file
    if os.path.isfile(cloudid_filename):
        logger.info(cloudid_filename)

        # Load cloudid data
        logger.debug("Loading cloudid data")
        logger.debug(cloudid_filename)
        ds = xr.open_dataset(
            cloudid_filename,
            mask_and_scale=False,
            decode_times=False,
        )
        cloudnumbermap = ds[feature_varname].data.squeeze()
        rawrainratemap = ds["precipitation"].data.squeeze()
        cloudid_basetime = ds["base_time"].data.squeeze()
        lon = ds["longitude"].data.squeeze()
        lat = ds["latitude"].data.squeeze()
        reflectivity = ds["reflectivity_comp"].data.squeeze()
        sl3d = ds["sl3d"].data.squeeze()
        echotop10 = ds["echotop10"].data.squeeze()
        echotop20 = ds["echotop20"].data.squeeze()
        echotop30 = ds["echotop30"].data.squeeze()
        echotop40 = ds["echotop40"].data.squeeze()
        echotop45 = ds["echotop45"].data.squeeze()
        echotop50 = ds["echotop50"].data.squeeze()
        ds.close()

        # Get dimensions of data
        ydim, xdim = np.shape(lat)

        # Number of clouds
        nmatchcloud = len(ir_cloudnumber)

        if nmatchcloud > 0:
            # Load pixel area once for latlon grids
            if area_method == "latlon":
                _pixel_area = get_pixel_area(config)
            else:
                _pixel_area = None

            # Define a list of 2D variables [tracks, times]
            var_names_2d = [
                "pf_npf",
                "pf_landfrac",
                "total_rain",
                "total_heavyrain",
                "total_volrain",
                "total_heavyvolrain",
                "rainrate_heavyrain",
                "conv_rain",
                "strat_rain",
                "pf_ncore",
            ]
            pf_npf = np.full(nmatchcloud, fillval, dtype=np.int16)
            pf_landfrac = np.full(nmatchcloud, fillval_f, dtype=float)
            total_rain = np.full(nmatchcloud, fillval_f, dtype=float)
            total_heavyrain = np.full(nmatchcloud, fillval_f, dtype=float)
            total_volrain = np.full(nmatchcloud, fillval_f, dtype=float)
            total_heavyvolrain = np.full(nmatchcloud, fillval_f, dtype=float)
            rainrate_heavyrain = np.full(nmatchcloud, fillval_f, dtype=float)
            pf_lon = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_lat = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_area = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_rainrate = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_skewness = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_maxrainrate = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_majoraxis = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_minoraxis = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_aspectratio = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_orientation = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_eccentricity = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_perimeter = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_lon_centroid = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_lat_centroid = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_lon_weightedcentroid = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_lat_weightedcentroid = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_accumrain = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_accumrainheavy = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_lon_maxrainrate = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_lat_maxrainrate = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            basetime = np.full(nmatchcloud, fillval_f, dtype=float)
            # Radar PF variables
            conv_rain = np.full(nmatchcloud, fillval_f, dtype=float)
            strat_rain = np.full(nmatchcloud, fillval_f, dtype=float)
            pf_ccarea = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_sfarea = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_ccrainrate = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_sfrainrate = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_ccrainamount = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_sfrainamount = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_ccmaxechotop10 = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_ccmaxechotop20 = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_ccmaxechotop30 = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_ccmaxechotop40 = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_ccmaxechotop45 = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_ccmaxechotop50 = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_ccechotop40area = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_ccechotop45area = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_ccechotop50area = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            # Radar convective core variables
            pf_ncore = np.full(nmatchcloud, fillval, dtype=np.int16)
            pf_corearea = np.full((nmatchcloud, nmaxcore), fillval_f, dtype=float)
            pf_corelon = np.full((nmatchcloud, nmaxcore), fillval_f, dtype=float)
            pf_corelat = np.full((nmatchcloud, nmaxcore), fillval_f, dtype=float)
            pf_corelon_centroid = np.full((nmatchcloud, nmaxcore), fillval_f, dtype=float)
            pf_corelat_centroid = np.full((nmatchcloud, nmaxcore), fillval_f, dtype=float)
            pf_corelon_weightedcentroid = np.full((nmatchcloud, nmaxcore), fillval_f, dtype=float)
            pf_corelat_weightedcentroid = np.full((nmatchcloud, nmaxcore), fillval_f, dtype=float)
            pf_coremajoraxis = np.full((nmatchcloud, nmaxcore), fillval_f, dtype=float)
            pf_coreminoraxis = np.full((nmatchcloud, nmaxcore), fillval_f, dtype=float)
            pf_coreaspectratio = np.full((nmatchcloud, nmaxcore), fillval_f, dtype=float)
            pf_coreorientation = np.full((nmatchcloud, nmaxcore), fillval_f, dtype=float)
            pf_coreperimeter = np.full((nmatchcloud, nmaxcore), fillval_f, dtype=float)
            pf_coreeccentricity = np.full((nmatchcloud, nmaxcore), fillval_f, dtype=float)
            pf_coremaxechotop10 = np.full((nmatchcloud, nmaxcore), fillval_f, dtype=float)
            pf_coremaxechotop20 = np.full((nmatchcloud, nmaxcore), fillval_f, dtype=float)
            pf_coremaxechotop30 = np.full((nmatchcloud, nmaxcore), fillval_f, dtype=float)
            pf_coremaxechotop40 = np.full((nmatchcloud, nmaxcore), fillval_f, dtype=float)
            pf_coremaxechotop45 = np.full((nmatchcloud, nmaxcore), fillval_f, dtype=float)
            pf_coremaxechotop50 = np.full((nmatchcloud, nmaxcore), fillval_f, dtype=float)

            # Loop over each matched cloud number
            for imatchcloud in range(nmatchcloud):

                ittcloudnumber = ir_cloudnumber[imatchcloud]
                ittmergecloudnumber = ir_mergecloudnumber[imatchcloud]
                ittsplitcloudnumber = ir_splitcloudnumber[imatchcloud]
                basetime[imatchcloud] = cloudid_basetime

                #########################################################################
                # Intialize matrices for only MCS data
                rainrate_map = np.full((ydim, xdim), np.nan, dtype=float)
                reflectivity_map = np.full((ydim, xdim), np.nan, dtype=float)
                sl3d_map = np.full((ydim, xdim), np.nan, dtype=float)
                echotop10_map = np.full((ydim, xdim), np.nan, dtype=float)
                echotop20_map = np.full((ydim, xdim), np.nan, dtype=float)
                echotop30_map = np.full((ydim, xdim), np.nan, dtype=float)
                echotop40_map = np.full((ydim, xdim), np.nan, dtype=float)
                echotop45_map = np.full((ydim, xdim), np.nan, dtype=float)
                echotop50_map = np.full((ydim, xdim), np.nan, dtype=float)
                logger.debug(
                    ("rainrate_map allocation size: ", rainrate_map.shape)
                )

                ############################################################################
                # Find matching cloud number
                icloudlocationy, icloudlocationx = np.array(
                    np.where(cloudnumbermap == ittcloudnumber)
                )
                ncloudpix = len(icloudlocationy)

                if ncloudpix > 0:
                    logger.debug("IR Clouds Present")
                    # Add merge/split cloud pixel locations
                    icloudlocationx, \
                    icloudlocationy = add_merge_split_cloud_locations(cloudnumbermap,
                                                                      icloudlocationx,
                                                                      icloudlocationy,
                                                                      ittmergecloudnumber,
                                                                      ittsplitcloudnumber,
                                                                      logger)

                    ########################################################################
                    # Fill matrices with MCS data
                    logger.debug("Fill map with data")
                    rainrate_map[icloudlocationy, icloudlocationx] = np.copy(
                        rawrainratemap[icloudlocationy, icloudlocationx]
                    )
                    reflectivity_map[icloudlocationy, icloudlocationx] = np.copy(
                        reflectivity[icloudlocationy, icloudlocationx]
                    )
                    sl3d_map[icloudlocationy, icloudlocationx] = np.copy(
                        sl3d[icloudlocationy, icloudlocationx]
                    )
                    echotop10_map[icloudlocationy, icloudlocationx] = np.copy(
                        echotop10[icloudlocationy, icloudlocationx]
                    )
                    echotop20_map[icloudlocationy, icloudlocationx] = np.copy(
                        echotop20[icloudlocationy, icloudlocationx]
                    )
                    echotop30_map[icloudlocationy, icloudlocationx] = np.copy(
                        echotop30[icloudlocationy, icloudlocationx]
                    )
                    echotop40_map[icloudlocationy, icloudlocationx] = np.copy(
                        echotop40[icloudlocationy, icloudlocationx]
                    )
                    echotop45_map[icloudlocationy, icloudlocationx] = np.copy(
                        echotop45[icloudlocationy, icloudlocationx]
                    )
                    echotop50_map[icloudlocationy, icloudlocationx] = np.copy(
                        echotop50[icloudlocationy, icloudlocationx]
                    )

                    ########################################################################
                    ## Isolate small region of cloud data around mcs at this time
                    logger.debug("Calculate new shape statistics")

                    # Get cloud boundary
                    maxx, maxy, minx, miny = get_cloud_boundary(icloudlocationx,
                                                                icloudlocationy,
                                                                xdim,
                                                                ydim)

                    # Check cloud boundary span: if > X fraction of domain and PBC is set,
                    # roll the data so the cloud does not span the domain boundary
                    roll_flag = False
                    if (((maxx - minx) >= xdim * max_feature_frac_x) or \
                        ((maxy - miny) >= ydim * max_feature_frac_y)) and \
                        (pbc_direction != 'none'):
                        # Compute roll shifts from the largest gap in sorted pixel indices.
                        # A gap > 50% of domain length indicates a boundary crossing.
                        sorted_x = np.sort(np.unique(icloudlocationx))
                        gap_x    = np.diff(sorted_x)
                        max_gap_x_idx = np.argmax(gap_x)
                        if gap_x[max_gap_x_idx] > xdim * 0.5:
                            shift_x_right = xdim - int(sorted_x[max_gap_x_idx + 1])
                        else:
                            shift_x_right = 0
                        sorted_y = np.sort(np.unique(icloudlocationy))
                        gap_y    = np.diff(sorted_y)
                        max_gap_y_idx = np.argmax(gap_y)
                        if gap_y[max_gap_y_idx] > ydim * 0.5:
                            shift_y_top = ydim - int(sorted_y[max_gap_y_idx + 1])
                        else:
                            shift_y_top = 0
                        rolled_x = (icloudlocationx + shift_x_right) % xdim
                        rolled_y = (icloudlocationy + shift_y_top)   % ydim
                        r_maxx, r_maxy, r_minx, r_miny = get_cloud_boundary(rolled_x, rolled_y, xdim, ydim)
                        _rr_rolled   = np.roll(rainrate_map,    (shift_y_top, shift_x_right), axis=(0, 1))
                        _ref_rolled  = np.roll(reflectivity_map,(shift_y_top, shift_x_right), axis=(0, 1))
                        _sl3d_rolled = np.roll(sl3d_map,        (shift_y_top, shift_x_right), axis=(0, 1))
                        _e10_rolled  = np.roll(echotop10_map,   (shift_y_top, shift_x_right), axis=(0, 1))
                        _e20_rolled  = np.roll(echotop20_map,   (shift_y_top, shift_x_right), axis=(0, 1))
                        _e30_rolled  = np.roll(echotop30_map,   (shift_y_top, shift_x_right), axis=(0, 1))
                        _e40_rolled  = np.roll(echotop40_map,   (shift_y_top, shift_x_right), axis=(0, 1))
                        _e45_rolled  = np.roll(echotop45_map,   (shift_y_top, shift_x_right), axis=(0, 1))
                        _e50_rolled  = np.roll(echotop50_map,   (shift_y_top, shift_x_right), axis=(0, 1))
                        sub_rainrate     = _rr_rolled  [r_miny:r_maxy, r_minx:r_maxx]
                        sub_reflectivity = _ref_rolled [r_miny:r_maxy, r_minx:r_maxx]
                        sub_sl3d         = _sl3d_rolled[r_miny:r_maxy, r_minx:r_maxx]
                        sub_echotop10    = _e10_rolled [r_miny:r_maxy, r_minx:r_maxx]
                        sub_echotop20    = _e20_rolled [r_miny:r_maxy, r_minx:r_maxx]
                        sub_echotop30    = _e30_rolled [r_miny:r_maxy, r_minx:r_maxx]
                        sub_echotop40    = _e40_rolled [r_miny:r_maxy, r_minx:r_maxx]
                        sub_echotop45    = _e45_rolled [r_miny:r_maxy, r_minx:r_maxx]
                        sub_echotop50    = _e50_rolled [r_miny:r_maxy, r_minx:r_maxx]
                        # Update bounding box to compact rolled box
                        minx, miny = r_minx, r_miny
                        maxx, maxy = r_maxx, r_maxy
                        if (sub_rainrate.size > 0) and (sub_rainrate.shape[0] > 0) and \
                            (sub_rainrate.shape[1] > 0) and (np.any(sub_rainrate > pf_rr_thres)):
                            sub_rainrate_map     = sub_rainrate
                            sub_reflectivity_map = sub_reflectivity
                            sub_sl3d_map         = sub_sl3d
                            sub_echotop10_map    = sub_echotop10
                            sub_echotop20_map    = sub_echotop20
                            sub_echotop30_map    = sub_echotop30
                            sub_echotop40_map    = sub_echotop40
                            sub_echotop45_map    = sub_echotop45
                            sub_echotop50_map    = sub_echotop50
                            roll_flag = True
                        else:
                            sub_rainrate_map     = sub_rainrate
                            sub_reflectivity_map = sub_reflectivity
                            sub_sl3d_map         = sub_sl3d
                            sub_echotop10_map    = sub_echotop10
                            sub_echotop20_map    = sub_echotop20
                            sub_echotop30_map    = sub_echotop30
                            sub_echotop40_map    = sub_echotop40
                            sub_echotop45_map    = sub_echotop45
                            sub_echotop50_map    = sub_echotop50
                            shift_x_right = 0
                            shift_y_top = 0
                    else:
                        sub_rainrate_map     = np.copy(rainrate_map[miny:maxy, minx:maxx])
                        sub_reflectivity_map = np.copy(reflectivity_map[miny:maxy, minx:maxx])
                        sub_sl3d_map         = np.copy(sl3d_map[miny:maxy, minx:maxx])
                        sub_echotop10_map    = np.copy(echotop10_map[miny:maxy, minx:maxx])
                        sub_echotop20_map    = np.copy(echotop20_map[miny:maxy, minx:maxx])
                        sub_echotop30_map    = np.copy(echotop30_map[miny:maxy, minx:maxx])
                        sub_echotop40_map    = np.copy(echotop40_map[miny:maxy, minx:maxx])
                        sub_echotop45_map    = np.copy(echotop45_map[miny:maxy, minx:maxx])
                        sub_echotop50_map    = np.copy(echotop50_map[miny:maxy, minx:maxx])
                        shift_x_right = 0
                        shift_y_top = 0

                    # minx/miny reflect compact bounding box; slice always matches sub_rainrate_map.shape
                    if area_method == "latlon":
                        sub_pixel_area = _pixel_area[miny:maxy, minx:maxx]
                        mean_pixelength = np.sqrt(np.nanmean(_pixel_area))
                    else:
                        sub_pixel_area = None
                        mean_pixelength = pixel_radius

                    # Calculate total rainfall within the cold cloud shield
                    total_rain[imatchcloud] = np.nansum(sub_rainrate_map)
                    # Calculate volumetric rain (rain rate * pixel area)
                    if area_method == "latlon":
                        sub_pa = sub_pixel_area  # already computed with correct shape above
                        total_volrain[imatchcloud] = np.nansum(
                            sub_rainrate_map * sub_pa
                        )
                    else:
                        total_volrain[imatchcloud] = (
                            total_rain[imatchcloud] * pixel_radius**2
                        )
                    idx_heavyrain = np.where(sub_rainrate_map > heavy_rainrate_thresh)
                    if len(idx_heavyrain[0]) > 0:
                        total_heavyrain[imatchcloud] = np.nansum(
                            sub_rainrate_map[idx_heavyrain]
                        )
                        if area_method == "latlon":
                            total_heavyvolrain[imatchcloud] = np.nansum(
                                sub_rainrate_map[idx_heavyrain] * sub_pa[idx_heavyrain]
                            )
                        else:
                            total_heavyvolrain[imatchcloud] = (
                                total_heavyrain[imatchcloud] * pixel_radius**2
                            )
                        rainrate_heavyrain[imatchcloud] = np.nanmean(
                            sub_rainrate_map[idx_heavyrain]
                        )
                    # Calculate convective/stratiform rain
                    conv_mask = (sub_sl3d_map == 1) | (sub_sl3d_map == 2)
                    strat_mask = (sub_sl3d_map == 3)
                    conv_rain[imatchcloud] = np.nansum(sub_rainrate_map[conv_mask])
                    strat_rain[imatchcloud] = np.nansum(sub_rainrate_map[strat_mask])

                    ######################################################
                    # Derive individual convective core statistics
                    logger.debug("Calculating convective core statistics")

                    # Get convective grid indices
                    iccy, iccx = np.array(np.where(
                        (sub_sl3d_map >= 1) & (sub_sl3d_map <= 2)
                    ))
                    ncorepix = len(iccy)
                    if ncorepix > 0:
                        ####################################################
                        # Get dimensions of subset region
                        subdimy, subdimx = np.shape(sub_sl3d_map)
                        # Create binary map
                        binaryccmap = np.zeros((subdimy, subdimx), dtype=int)
                        binaryccmap[iccy, iccx] = 1
                        # Label convective cores
                        ccnumberlabelmap, numcc = label(binaryccmap)
                        # Convert min area to number of pixels
                        if area_method == "latlon":
                            min_npix = np.ceil(mcs_core_min_area / np.nanmean(sub_pixel_area)).astype(int)
                        else:
                            min_npix = np.ceil(mcs_core_min_area / (pixel_radius ** 2)).astype(int)
                        # Sort and renumber convective cores
                        cc_number, cc_npix = sort_renumber(ccnumberlabelmap, min_npix)
                        # Update convective core arrays after sorting and renumbering
                        numcc = np.nanmax(cc_number)
                        ccnumberlabelmap = cc_number

                        if numcc > 0:
                            ###################################################
                            logger.debug("Convective core present, calculating statistics")

                            # Call function to calculate individual PF statistics
                            cc_stats_dict = calc_cc_stats(
                                fillval, fillval_f,
                                lat, lon, minx, miny, nmaxcore, numcc,
                                cc_npix, ccnumberlabelmap, pixel_radius,
                                subdimx, subdimy,
                                sub_reflectivity_map,
                                sub_echotop10_map,
                                sub_echotop20_map,
                                sub_echotop30_map,
                                sub_echotop40_map,
                                sub_echotop45_map,
                                sub_echotop50_map,
                                sub_pixel_area=sub_pixel_area,
                                mean_pixelength=mean_pixelength,
                                roll_flag=roll_flag, shift_x_right=shift_x_right,
                                shift_y_top=shift_y_top, xdim=xdim, ydim=ydim,
                            )

                            # Save core feature statisitcs
                            ncc_save = cc_stats_dict["ncc_save"]
                            pf_ncore[imatchcloud] = np.copy(numcc)
                            pf_corelon[imatchcloud, 0:ncc_save] = \
                                cc_stats_dict["cclon"][0:ncc_save]
                            pf_corelat[imatchcloud, 0:ncc_save] = \
                                cc_stats_dict["cclat"][0:ncc_save]
                            pf_corearea[imatchcloud, 0:ncc_save] = \
                                cc_stats_dict["ccarea"][0:ncc_save]
                            pf_corelon_centroid[imatchcloud, 0:ncc_save] = \
                                cc_stats_dict["cclon_centroid"][0:ncc_save]
                            pf_corelat_centroid[imatchcloud, 0:ncc_save] = \
                                cc_stats_dict["cclat_centroid"][0:ncc_save]
                            pf_corelon_weightedcentroid[imatchcloud, 0:ncc_save] = \
                                cc_stats_dict["cclon_weightedcentroid"][0:ncc_save]
                            pf_corelat_weightedcentroid[imatchcloud, 0:ncc_save] = \
                                cc_stats_dict["cclat_weightedcentroid"][0:ncc_save]
                            pf_coremajoraxis[imatchcloud, 0:ncc_save] = \
                                cc_stats_dict["ccmajoraxis"][0:ncc_save]
                            pf_coreminoraxis[imatchcloud, 0:ncc_save] = \
                                cc_stats_dict["ccminoraxis"][0:ncc_save]
                            pf_coreaspectratio[imatchcloud, 0:ncc_save] = \
                                cc_stats_dict["ccaspectratio"][0:ncc_save]
                            pf_coreorientation[imatchcloud, 0:ncc_save] = \
                                cc_stats_dict["ccorientation"][0:ncc_save]
                            pf_coreperimeter[imatchcloud, 0:ncc_save] = \
                                cc_stats_dict["ccperimeter"][0:ncc_save]
                            pf_coreeccentricity[imatchcloud, 0:ncc_save] = \
                                cc_stats_dict["cceccentricity"][0:ncc_save]
                            pf_coremaxechotop10[imatchcloud, 0:ncc_save] = \
                                cc_stats_dict["ccmaxechotop10"][0:ncc_save]
                            pf_coremaxechotop20[imatchcloud, 0:ncc_save] = \
                                cc_stats_dict["ccmaxechotop20"][0:ncc_save]
                            pf_coremaxechotop30[imatchcloud, 0:ncc_save] = \
                                cc_stats_dict["ccmaxechotop30"][0:ncc_save]
                            pf_coremaxechotop40[imatchcloud, 0:ncc_save] = \
                                cc_stats_dict["ccmaxechotop40"][0:ncc_save]
                            pf_coremaxechotop45[imatchcloud, 0:ncc_save] = \
                                cc_stats_dict["ccmaxechotop45"][0:ncc_save]
                            pf_coremaxechotop50[imatchcloud, 0:ncc_save] = \
                                cc_stats_dict["ccmaxechotop50"][0:ncc_save]


                    ######################################################
                    # Derive individual PF statistics
                    logger.debug("Calculating precipitation statistics")

                    # Define a PF with rain rate & SL3D
                    ipfy, ipfx = np.array(np.where(
                        (sub_rainrate_map > pf_rr_thres) &
                        (sub_sl3d_map >= 1) & (sub_sl3d_map <= 3)
                    ))
                    nrainpix = len(ipfy)

                    if nrainpix > 0:

                        # Calculate fraction of PF over land
                        if landmask is not None:
                            # Subset landmask to current cloud area
                            sublandmask = landmask[miny:maxy, minx:maxx]
                            # Count the number of grids within the specified land fraction range
                            npix_land = np.count_nonzero(
                                (sublandmask[ipfy, ipfx] >= np.min(landfrac_thresh)) & \
                                (sublandmask[ipfy, ipfx] <= np.max(landfrac_thresh))
                            )
                            if npix_land > 0:
                                pf_landfrac[imatchcloud] = \
                                    float(npix_land) / float(nrainpix)
                            else:
                                pf_landfrac[imatchcloud] = 0
                        pass

                        ####################################################
                        ## Get dimensions of subsetted region
                        subdimy, subdimx = np.shape(sub_rainrate_map)

                        # Create binary map
                        binarypfmap = np.zeros((subdimy, subdimx), dtype=int)
                        binarypfmap[ipfy, ipfx] = 1

                        # Label precipitation features
                        pfnumberlabelmap, numpf = label(binarypfmap)

                        # Sort numpf then calculate stats
                        if area_method == "latlon":
                            min_npix = np.ceil(pf_link_area_thresh / np.nanmean(sub_pixel_area)).astype(int)
                        else:
                            min_npix = np.ceil(pf_link_area_thresh / (pixel_radius ** 2)).astype(int)

                        # Sort and renumber PFs, and remove small PFs
                        pf_number, pf_npix = sort_renumber(pfnumberlabelmap, min_npix)
                        # Update number of PFs after sorting and renumbering
                        npf_new = np.nanmax(pf_number)
                        numpf = npf_new
                        pfnumberlabelmap = pf_number

                        if numpf > 0:
                            ###################################################
                            logger.debug("PFs present, calculating statistics")

                            # Call function to calculate individual PF statistics
                            pf_stats_dict = calc_pf_stats(
                                fillval, fillval_f, heavy_rainrate_thresh,
                                lat, lon, minx, miny, nmaxpf, numpf,
                                pf_npix, pfnumberlabelmap, pixel_radius,
                                subdimx, subdimy, sub_rainrate_map,
                                sub_sl3d_map,
                                sub_echotop10_map,
                                sub_echotop20_map,
                                sub_echotop30_map,
                                sub_echotop40_map,
                                sub_echotop45_map,
                                sub_echotop50_map,
                                sub_pixel_area=sub_pixel_area,
                                mean_pixelength=mean_pixelength,
                                roll_flag=roll_flag, shift_x_right=shift_x_right,
                                shift_y_top=shift_y_top, xdim=xdim, ydim=ydim,
                            )

                            # Save precipitation feature statisitcs
                            npf_save = pf_stats_dict["npf_save"]
                            pf_npf[imatchcloud] = np.copy(numpf)
                            pf_lon[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pflon"][0:npf_save]
                            pf_lat[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pflat"][0:npf_save]
                            pf_area[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pfarea"][0:npf_save]
                            pf_rainrate[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pfrainrate"][0:npf_save]
                            pf_maxrainrate[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pfmaxrainrate"][0:npf_save]
                            pf_skewness[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pfskewness"][0:npf_save]
                            pf_majoraxis[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pfmajoraxis"][0:npf_save]
                            pf_minoraxis[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pfminoraxis"][0:npf_save]
                            pf_aspectratio[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pfaspectratio"][0:npf_save]
                            pf_orientation[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pforientation"][0:npf_save]
                            pf_eccentricity[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pfeccentricity"][0:npf_save]
                            pf_lat_centroid[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pflat_centroid"][0:npf_save]
                            pf_lon_centroid[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pflon_centroid"][0:npf_save]
                            pf_lat_weightedcentroid[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pflat_weightedcentroid"][0:npf_save]
                            pf_lon_weightedcentroid[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pflon_weightedcentroid"][0:npf_save]
                            pf_accumrain[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pfaccumrain"][0:npf_save]
                            pf_accumrainheavy[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pfaccumrainheavy"][0:npf_save]
                            pf_perimeter[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pfperimeter"][0:npf_save]
                            pf_lon_maxrainrate[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pflon_maxrainrate"][0:npf_save]
                            pf_lat_maxrainrate[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pflat_maxrainrate"][0:npf_save]
                            pf_ccarea[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pfccarea"][0:npf_save]
                            pf_sfarea[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pfsfarea"][0:npf_save]
                            pf_ccrainrate[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pfccrainrate"][0:npf_save]
                            pf_sfrainrate[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pfsfrainrate"][0:npf_save]
                            pf_ccrainamount[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pfccrainamount"][0:npf_save]
                            pf_sfrainamount[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pfsfrainamount"][0:npf_save]
                            pf_ccmaxechotop10[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pfccmaxechotop10"][0:npf_save]
                            pf_ccmaxechotop20[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pfccmaxechotop20"][0:npf_save]
                            pf_ccmaxechotop30[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pfccmaxechotop30"][0:npf_save]
                            pf_ccmaxechotop40[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pfccmaxechotop40"][0:npf_save]
                            pf_ccmaxechotop45[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pfccmaxechotop45"][0:npf_save]
                            pf_ccmaxechotop50[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pfccmaxechotop50"][0:npf_save]
                            pf_ccechotop40area[imatchcloud, 0: npf_save] = \
                                pf_stats_dict["pfccechotop40area"][0:npf_save]
                            pf_ccechotop45area[imatchcloud, 0: npf_save] = \
                                pf_stats_dict["pfccechotop45area"][0:npf_save]
                            pf_ccechotop50area[imatchcloud, 0: npf_save] = \
                                pf_stats_dict["pfccechotop50area"][0:npf_save]
                            # import matplotlib.pyplot as plt
                            # import pdb;
                            # pdb.set_trace()

            # Group outputs in dictionaries
            out_dict = {
                "pf_npf": pf_npf,
                "pf_lon": pf_lon,
                "pf_lat": pf_lat,
                "pf_area": pf_area,
                "pf_rainrate": pf_rainrate,
                "pf_skewness": pf_skewness,
                "pf_majoraxis": pf_majoraxis,
                "pf_minoraxis": pf_minoraxis,
                "pf_aspectratio": pf_aspectratio,
                "pf_orientation": pf_orientation,
                "pf_perimeter": pf_perimeter,
                "pf_eccentricity": pf_eccentricity,
                "pf_lon_centroid": pf_lon_centroid,
                "pf_lat_centroid": pf_lat_centroid,
                "pf_lon_weightedcentroid": pf_lon_weightedcentroid,
                "pf_lat_weightedcentroid": pf_lat_weightedcentroid,
                "pf_lon_maxrainrate": pf_lon_maxrainrate,
                "pf_lat_maxrainrate": pf_lat_maxrainrate,
                "pf_maxrainrate": pf_maxrainrate,
                "pf_accumrain": pf_accumrain,
                "pf_accumrainheavy": pf_accumrainheavy,
                "pf_landfrac": pf_landfrac,
                "total_rain": total_rain,
                "total_heavyrain": total_heavyrain,
                "total_volrain": total_volrain,
                "total_heavyvolrain": total_heavyvolrain,
                "rainrate_heavyrain": rainrate_heavyrain,
                # Radar PF variables
                "conv_rain": conv_rain,
                "strat_rain": strat_rain,
                "pf_ccarea": pf_ccarea,
                "pf_sfarea": pf_sfarea,
                "pf_ccrainrate": pf_ccrainrate,
                "pf_sfrainrate": pf_sfrainrate,
                "pf_ccrainamount": pf_ccrainamount,
                "pf_sfrainamount": pf_sfrainamount,
                "pf_ccmaxechotop10": pf_ccmaxechotop10,
                "pf_ccmaxechotop20": pf_ccmaxechotop20,
                "pf_ccmaxechotop30": pf_ccmaxechotop30,
                "pf_ccmaxechotop40": pf_ccmaxechotop40,
                "pf_ccmaxechotop45": pf_ccmaxechotop45,
                "pf_ccmaxechotop50": pf_ccmaxechotop50,
                "pf_cc40area": pf_ccechotop40area,
                "pf_cc45area": pf_ccechotop45area,
                "pf_cc50area": pf_ccechotop50area,
                # Radar convective core variables
                "pf_ncore": pf_ncore,
                "pf_corearea": pf_corearea,
                "pf_corelon": pf_corelon,
                "pf_corelat": pf_corelon,
                "pf_corelon_centroid": pf_corelon_centroid,
                "pf_corelat_centroid": pf_corelat_centroid,
                "pf_corelon_weightedcentroid": pf_corelon_weightedcentroid,
                "pf_corelat_weightedcentroid": pf_corelat_weightedcentroid,
                "pf_coremajoraxis": pf_coremajoraxis,
                "pf_coreminoraxis": pf_coreminoraxis,
                "pf_coreaspectratio": pf_coreaspectratio,
                "pf_coreorientation": pf_coreorientation,
                "pf_coreperimeter": pf_coreperimeter,
                "pf_coreeccentricity": pf_coreeccentricity,
                "pf_coremaxechotop10": pf_coremaxechotop10,
                "pf_coremaxechotop20": pf_coremaxechotop20,
                "pf_coremaxechotop30": pf_coremaxechotop30,
                "pf_coremaxechotop40": pf_coremaxechotop40,
                "pf_coremaxechotop45": pf_coremaxechotop45,
                "pf_coremaxechotop50": pf_coremaxechotop50,
            }
            out_dict_attrs = {
                "pf_npf": {
                    "long_name": "Number of PF in the cloud",
                    "units": "unitless",
                    "_FillValue": fillval,
                },
                "pf_lon": {
                    "long_name": "Mean longitude of PF",
                    "units": "degrees",
                    "_FillValue": fillval_f,
                },
                "pf_lat": {
                    "long_name": "Mean latitude of PF",
                    "units": "degrees",
                    "_FillValue": fillval_f,
                },
                "pf_area": {
                    "long_name": "Area of PF",
                    "units": "km^2",
                    "_FillValue": fillval_f,
                },
                "pf_rainrate": {
                    "long_name": "Mean rain rate of PF",
                    "units": "mm/h",
                    "_FillValue": fillval_f,
                },
                "pf_skewness": {
                    "long_name": "Rain rate skewness of PF",
                    "units": "unitless",
                    "_FillValue": fillval_f,
                },
                "pf_majoraxis": {
                    "long_name": "Major axis length of PF",
                    "units": "km",
                    "_FillValue": fillval_f,
                },
                "pf_minoraxis": {
                    "long_name": "Minor axis length of PF",
                    "units": "km",
                    "_FillValue": fillval_f,
                },
                "pf_aspectratio": {
                    "long_name": "Aspect ratio (major axis / minor axis) of PF",
                    "units": "unitless",
                    "_FillValue": fillval_f,
                },
                "pf_orientation": {
                    "long_name": "Orientation of major axis of PF",
                    "units": "degrees",
                    "_FillValue": fillval_f,
                },
                "pf_eccentricity": {
                    "long_name": "Eccentricity of PF",
                    "units": "unitless",
                    "_FillValue": fillval_f,
                },
                "pf_perimeter": {
                    "long_name": "Perimeter of PF",
                    "units": "km",
                    "_FillValue": fillval_f,
                },
                "pf_lon_centroid": {
                    "long_name": "Centroid longitude of PF",
                    "units": "degrees",
                    "_FillValue": fillval_f,
                },
                "pf_lat_centroid": {
                    "long_name": "Centroid latitude of PF",
                    "units": "degrees",
                    "_FillValue": fillval_f,
                },
                "pf_lon_weightedcentroid": {
                    "long_name": "Weighted centroid longitude of PF",
                    "units": "degrees",
                    "_FillValue": fillval_f,
                },
                "pf_lat_weightedcentroid": {
                    "long_name": "Weighted centroid latitude of PF",
                    "units": "degrees",
                    "_FillValue": fillval_f,
                },
                "pf_maxrainrate": {
                    "long_name": "Max rain rate of PF",
                    "units": "mm/h",
                    "_FillValue": fillval_f,
                },
                "pf_accumrain": {
                    "long_name": "Accumulate precipitation of PF",
                    "units": "mm/h",
                    "_FillValue": fillval_f,
                },
                "pf_accumrainheavy": {
                    "long_name": "Accumulated heavy precipitation of PF",
                    "units": "mm/h",
                    "_FillValue": fillval_f,
                    "heavy_rainrate_threshold": heavy_rainrate_thresh,
                },
                "pf_landfrac": {
                    "long_name": "Fraction of PF over land",
                    "units": "fraction",
                    "_FillValue": fillval_f,
                },
                "total_rain": {
                    "long_name": "Total precipitation under cold cloud shield (all rainfall included)",
                    "units": "mm/h",
                    "_FillValue": fillval_f,
                },
                "total_heavyrain": {
                    "long_name": "Total heavy precipitation under cold cloud shield",
                    "units": "mm/h",
                    "_FillValue": fillval_f,
                    "heavy_rainrate_threshold": heavy_rainrate_thresh,
                },
                "total_volrain": {
                    "long_name": "Total volumetric precipitation under cold cloud shield",
                    "units": "mm/h km^2",
                    "_FillValue": fillval_f,
                },
                "total_heavyvolrain": {
                    "long_name": "Total heavy volumetric precipitation under cold cloud shield",
                    "units": "mm/h km^2",
                    "_FillValue": fillval_f,
                    "heavy_rainrate_threshold": heavy_rainrate_thresh,
                },
                "rainrate_heavyrain": {
                    "long_name": "Mean heavy rain rate under cold cloud shield",
                    "units": "mm/h",
                    "_FillValue": fillval_f,
                    "heavy_rainrate_threshold": heavy_rainrate_thresh,
                },
                "pf_lon_maxrainrate": {
                    "long_name": "Longitude with max rain rate",
                    "units": "degree",
                    "_FillValue": fillval_f,
                },
                "pf_lat_maxrainrate": {
                    "long_name": "Latitude with max rain rate",
                    "units": "degree",
                    "_FillValue": fillval_f,
                },

                # Radar variables
                "conv_rain": {
                    "long_name": "Convective rain amount",
                    "units": "mm/h",
                    "_FillValue": fillval_f,
                },
                "strat_rain": {
                    "long_name": "Stratiform rain amount",
                    "units": "mm/h",
                    "_FillValue": fillval_f,
                },
                "pf_ccarea": {
                    "long_name": "PF convective area",
                    "units": "km^2",
                    "_FillValue": fillval_f,
                },
                "pf_sfarea": {
                    "long_name": "PF stratiform area",
                    "units": "km^2",
                    "_FillValue": fillval_f,
                },
                "pf_ccrainrate": {
                    "long_name": "PF mean convective rain rate",
                    "units": "mm/h",
                    "_FillValue": fillval_f,
                },
                "pf_sfrainrate": {
                    "long_name": "PF mean stratiform rain rate",
                    "units": "mm/h",
                    "_FillValue": fillval_f,
                },
                "pf_ccrainamount": {
                    "long_name": "PF convective rain amount",
                    "units": "mm/h",
                    "_FillValue": fillval_f,
                },
                "pf_sfrainamount": {
                    "long_name": "PF stratiform rain amount",
                    "units": "mm/h",
                    "_FillValue": fillval_f,
                },
                "pf_ccmaxechotop10": {
                    "long_name": "PF convective 10 dBZ max echo top height",
                    "units": "km",
                    "_FillValue": fillval_f,
                },
                "pf_ccmaxechotop20": {
                    "long_name": "PF convective 20 dBZ max echo top height",
                    "units": "km",
                    "_FillValue": fillval_f,
                },
                "pf_ccmaxechotop30": {
                    "long_name": "PF convective 30 dBZ max echo top height",
                    "units": "km",
                    "_FillValue": fillval_f,
                },
                "pf_ccmaxechotop40": {
                    "long_name": "PF convective 40 dBZ max echo top height",
                    "units": "km",
                    "_FillValue": fillval_f,
                },
                "pf_ccmaxechotop45": {
                    "long_name": "PF convective 45 dBZ max echo top height",
                    "units": "km",
                    "_FillValue": fillval_f,
                },
                "pf_ccmaxechotop50": {
                    "long_name": "PF convective 50 dBZ max echo top height",
                    "units": "km",
                    "_FillValue": fillval_f,
                },
                "pf_cc40area": {
                    "long_name": "PF convective area exceeding 40 dBZ",
                    "units": "km^2",
                    "_FillValue": fillval_f,
                },
                "pf_cc45area": {
                    "long_name": "PF convective area exceeding 45 dBZ",
                    "units": "km^2",
                    "_FillValue": fillval_f,
                },
                "pf_cc50area": {
                    "long_name": "PF convective area exceeding 50 dBZ",
                    "units": "km^2",
                    "_FillValue": fillval_f,
                },

                # Radar convective core variables
                "pf_ncore": {
                    "long_name": "Number of convective cores in the cloud",
                    "units": "unitless",
                    "_FillValue": fillval,
                },
                "pf_corearea": {
                    "long_name": "Convective area",
                    "units": "km^2",
                    "_FillValue": fillval_f,
                },
                "pf_corelon": {
                    "long_name": "Mean longitude of convective core",
                    "units": "degrees",
                    "_FillValue": fillval_f,
                },
                "pf_corelat": {
                    "long_name": "Mean latitude of convective core",
                    "units": "degrees",
                    "_FillValue": fillval_f,
                },
                "pf_corelon_centroid": {
                    "long_name": "Centroid longitude of convective core",
                    "units": "degrees",
                    "_FillValue": fillval_f,
                },
                "pf_corelat_centroid": {
                    "long_name": "Centroid latitude of convective core",
                    "units": "degrees",
                    "_FillValue": fillval_f,
                },
                "pf_corelon_weightedcentroid": {
                    "long_name": "Weighted centroid longitude of convective core",
                    "units": "degrees",
                    "_FillValue": fillval_f,
                },
                "pf_corelat_weightedcentroid": {
                    "long_name": "Weighted centroid latitude of convective core",
                    "units": "degrees",
                    "_FillValue": fillval_f,
                },
                "pf_coremajoraxis": {
                    "long_name": "Major axis length of convective core",
                    "units": "km",
                    "_FillValue": fillval_f,
                },
                "pf_coreminoraxis": {
                    "long_name": "Minor axis length of convective core",
                    "units": "km",
                    "_FillValue": fillval_f,
                },
                "pf_coreaspectratio": {
                    "long_name": "Aspect ratio of convective core",
                    "units": "unitless",
                    "_FillValue": fillval_f,
                },
                "pf_coreorientation": {
                    "long_name": "Orientation of major axis of convective core",
                    "units": "degree",
                    "_FillValue": fillval_f,
                },
                "pf_coreperimeter": {
                    "long_name": "Perimeter of convective core",
                    "units": "km",
                    "_FillValue": fillval_f,
                },
                "pf_coreeccentricity": {
                    "long_name": "Eccentricity of convective core",
                    "units": "unitless",
                    "_FillValue": fillval_f,
                },
                "pf_coremaxechotop10": {
                    "long_name": "Convective core 10 dBZ max echo top height",
                    "units": "km",
                    "_FillValue": fillval_f,
                },
                "pf_coremaxechotop20": {
                    "long_name": "Convective core 20 dBZ max echo top height",
                    "units": "km",
                    "_FillValue": fillval_f,
                },
                "pf_coremaxechotop30": {
                    "long_name": "Convective core 30 dBZ max echo top height",
                    "units": "km",
                    "_FillValue": fillval_f,
                },
                "pf_coremaxechotop40": {
                    "long_name": "Convective core 40 dBZ max echo top height",
                    "units": "km",
                    "_FillValue": fillval_f,
                },
                "pf_coremaxechotop45": {
                    "long_name": "Convective core 45 dBZ max echo top height",
                    "units": "km",
                    "_FillValue": fillval_f,
                },
                "pf_coremaxechotop50": {
                    "long_name": "Convective core 50 dBZ max echo top height",
                    "units": "km",
                    "_FillValue": fillval_f,
                },
            }

            return out_dict, out_dict_attrs, var_names_2d

        else:
            logger.info("No matching cloud found in cloudid: " + cloudid_filename)

    else:
        logger.info("cloudid file does not exist: " + cloudid_filename)



##########################################################################
# Custom functions

def calc_cc_stats(
        fillval, fillval_f,
        lat, lon, minx, miny, nmaxcore, numcc,
        cc_npix, ccnumberlabelmap, pixel_radius,
        subdimx, subdimy,
        sub_reflectivity_map,
        sub_echotop10_map,
        sub_echotop20_map,
        sub_echotop30_map,
        sub_echotop40_map,
        sub_echotop45_map,
        sub_echotop50_map,
        sub_pixel_area=None,
        mean_pixelength=None,
        roll_flag=False, shift_x_right=0, shift_y_top=0, xdim=None, ydim=None,
):
    """
    Calculate individual convective core statistics.

    Args:
        fillval:
        fillval_f:
        lat:
        lon:
        minx:
        miny:
        nmaxcore:
        numcc:
        cc_npix:
        ccnumberlabelmap:
        pixel_radius:
        subdimx:
        subdimy:
        sub_reflectivity_map:
        sub_echotop10_map:
        sub_echotop20_map:
        sub_echotop30_map:
        sub_echotop40_map:
        sub_echotop45_map:
        sub_echotop50_map:

    Returns:
        cc_stats_dict: dictionary
            Dictionary containing core statistics variables.
    """
    logger = logging.getLogger(__name__)
    # Initialize arrays
    ncc_save = np.nanmin([nmaxcore, numcc])
    ccnpix = np.zeros(ncc_save, dtype=float)
    ccarea = np.full(ncc_save, fillval_f, dtype=float)
    ccid = np.full(ncc_save, fillval, dtype=int)
    cclon = np.full(ncc_save, fillval_f, dtype=float)
    cclat = np.full(ncc_save, fillval_f, dtype=float)
    cclon_centroid = np.full(ncc_save, fillval_f, dtype=float)
    cclat_centroid = np.full(ncc_save, fillval_f, dtype=float)
    cclon_weightedcentroid = np.full(ncc_save, fillval_f, dtype=float)
    cclat_weightedcentroid = np.full(ncc_save, fillval_f, dtype=float)
    ccmajoraxis = np.full(ncc_save, fillval_f, dtype=float)
    ccminoraxis = np.full(ncc_save, fillval_f, dtype=float)
    ccaspectratio = np.full(ncc_save, fillval_f, dtype=float)
    ccorientation = np.full(ncc_save, fillval_f, dtype=float)
    ccperimeter = np.full(ncc_save, fillval_f, dtype=float)
    cceccentricity = np.full(ncc_save, fillval_f, dtype=float)
    ccmaxechotop10 = np.full(ncc_save, fillval_f, dtype=float)
    ccmaxechotop20 = np.full(ncc_save, fillval_f, dtype=float)
    ccmaxechotop30 = np.full(ncc_save, fillval_f, dtype=float)
    ccmaxechotop40 = np.full(ncc_save, fillval_f, dtype=float)
    ccmaxechotop45 = np.full(ncc_save, fillval_f, dtype=float)
    ccmaxechotop50 = np.full(ncc_save, fillval_f, dtype=float)

    logger.debug(
        "Looping through each core to calculate statistics"
    )
    logger.debug(("Number of cores " + str(numcc)))

    # Get the shape of the full data array
    ny, nx = lon.shape
    # Domain bounds for circular_mean (works for both global and small idealized domains)
    lon_min = np.nanmin(lon)
    lon_max = np.nanmax(lon)
    lat_min = np.nanmin(lat)
    lat_max = np.nanmax(lat)

    ###############################################
    # Loop through each PF
    for icc in range(1, ncc_save + 1):
        #######################################
        # Find indices of the core
        iiccy, iiccx = np.where(ccnumberlabelmap == icc)
        iiccnpix = len(iiccy)

        # Double check to make sure PF pixel count is the same
        if iiccnpix == cc_npix[icc - 1]:
            ##########################################
            # Compute core statistics

            # Basic statistics
            ccnpix[icc - 1] = np.copy(iiccnpix)
            # Compute core area
            if sub_pixel_area is not None:
                ccarea[icc - 1] = np.nansum(sub_pixel_area[iiccy, iiccx])
            else:
                ccarea[icc - 1] = iiccnpix * pixel_radius ** 2
            ccid[icc - 1] = np.copy(int(icc))
            if roll_flag and xdim is not None:
                _yy = (iiccy[:] + miny - shift_y_top) % ydim
                _xx = (iiccx[:] + minx - shift_x_right) % xdim
                # circular_mean uses domain bounds and works correctly for both
                # global (360°) and small idealized doubly-periodic domains.
                # np.unwrap assumed 2π (360°) wrapping and failed for non-global domains.
                cclon[icc - 1] = circular_mean(lon[_yy, _xx], lon_min, lon_max)
                cclat[icc - 1] = circular_mean(lat[_yy, _xx], lat_min, lat_max)
            else:
                cclon[icc - 1] = np.nanmean(lon[iiccy[:] + miny, iiccx[:] + minx])
                cclat[icc - 1] = np.nanmean(lat[iiccy[:] + miny, iiccx[:] + minx])

            # Convective echotop height statistics
            if iiccnpix > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    ccmaxechotop10[icc - 1] = np.nanmax(sub_echotop10_map[iiccy, iiccx])
                    ccmaxechotop20[icc - 1] = np.nanmax(sub_echotop20_map[iiccy, iiccx])
                    ccmaxechotop30[icc - 1] = np.nanmax(sub_echotop30_map[iiccy, iiccx])
                    ccmaxechotop40[icc - 1] = np.nanmax(sub_echotop40_map[iiccy, iiccx])
                    ccmaxechotop45[icc - 1] = np.nanmax(sub_echotop45_map[iiccy, iiccx])
                    ccmaxechotop50[icc - 1] = np.nanmax(sub_echotop50_map[iiccy, iiccx])

            # Generate a binary map of core
            iiccflagmap = np.zeros((subdimy, subdimx), dtype=int)
            iiccflagmap[iiccy, iiccx] = 1

            # Geometric statistics
            _sub_reflectivity_map = np.copy(sub_reflectivity_map)
            _sub_reflectivity_map[np.isnan(_sub_reflectivity_map)] = -9999
            # _sub_reflectivity_map = np.full((subdimy, subdimx), -9999, dtype=float)
            # _sub_reflectivity_map[iiccy, iiccx] = sub_reflectivity_map[iiccy, iiccx]
            ccproperties = regionprops(
                iiccflagmap,
                intensity_image=_sub_reflectivity_map,
            )

            cceccentricity[icc - 1] = ccproperties[0].eccentricity
            _pixel_length = mean_pixelength if mean_pixelength is not None else pixel_radius
            ccmajoraxis[icc - 1] = ccproperties[0].axis_major_length * _pixel_length

            # Need to treat minor axis length with an error except
            # since the python algorithm occasionally throws an error.
            try:
                ccminoraxis[icc - 1] = (ccproperties[0].axis_minor_length * _pixel_length)
            except ValueError:
                pass
            if ~np.isnan(ccminoraxis[icc - 1]) or ~np.isnan(ccmajoraxis[icc - 1]):
                ccaspectratio[icc - 1] = np.divide(ccmajoraxis[icc - 1], ccminoraxis[icc - 1])
            ccorientation[icc - 1] = (ccproperties[0].orientation) * (180 / float(pi))
            ccperimeter[icc - 1] = (ccproperties[0].perimeter * _pixel_length)
            # Guard against NaN before converting to int: regionprops centroid/
            # centroid_weighted can return NaN when the intensity image has no
            # valid positive weights (e.g. all-NaN or all-zero intensity_image).
            _c = ccproperties[0].centroid
            _cw = ccproperties[0].centroid_weighted
            _c_valid  = not (np.isnan(_c[0])  or np.isnan(_c[1]))
            _cw_valid = not (np.isnan(_cw[0]) or np.isnan(_cw[1]))
            yc_local  = int(np.round(_c[0]))  if _c_valid  else -1
            xc_local  = int(np.round(_c[1]))  if _c_valid  else -1
            ywc_local = int(np.round(_cw[0])) if _cw_valid else -1
            xwc_local = int(np.round(_cw[1])) if _cw_valid else -1
            # Map local sub-box centroid to full-domain coordinates.
            # When roll_flag is set, miny/minx are in the rolled domain;
            # undo the roll with inverse-shift modulo to recover the original index.
            if roll_flag and xdim is not None:
                ycentroid = (yc_local + miny - shift_y_top) % ydim
                xcentroid = (xc_local + minx - shift_x_right) % xdim
                yweightedcentroid = (ywc_local + miny - shift_y_top) % ydim
                xweightedcentroid = (xwc_local + minx - shift_x_right) % xdim
            else:
                ycentroid = yc_local + miny
                xcentroid = xc_local + minx
                yweightedcentroid = ywc_local + miny
                xweightedcentroid = xwc_local + minx

            # Apply the indices to get centroid lat/lon
            if (0 < ycentroid < ny) & (0 < xcentroid < nx):
                cclon_centroid[icc - 1] = lon[ycentroid, xcentroid]
                cclat_centroid[icc - 1] = lat[ycentroid, xcentroid]
            if (0 < yweightedcentroid < ny) & (0 < xweightedcentroid < nx):
                cclon_weightedcentroid[icc - 1] = lon[yweightedcentroid, xweightedcentroid]
                cclat_weightedcentroid[icc - 1] = lat[yweightedcentroid, xweightedcentroid]

            # When roll_flag is active (PBC domain), unwrap centroid lon/lat relative
            # to cclon/cclat. Use domain width as correction size (not hardcoded 360°)
            # for robust support of both global and small idealized doubly-periodic domains.
            if roll_flag and xdim is not None:
                _lon_domain = lon_max - lon_min
                _lat_domain = lat_max - lat_min
                for _arr, _ref, _domain in [
                    (cclon_centroid, cclon, _lon_domain),
                    (cclon_weightedcentroid, cclon, _lon_domain),
                    (cclat_centroid, cclat, _lat_domain),
                    (cclat_weightedcentroid, cclat, _lat_domain),
                ]:
                    _diff = _arr[icc - 1] - _ref[icc - 1]
                    if _diff > _domain / 2:
                        _arr[icc - 1] -= _domain
                    elif _diff < -_domain / 2:
                        _arr[icc - 1] += _domain

    # Put all variables in dictionary for output
    cc_stats_dict = {
        "ncc_save": ncc_save,
        "ccnpix": ccnpix,
        "ccarea": ccarea,
        "ccid": ccid,
        "cclon": cclon,
        "cclat": cclat,
        "cclon_centroid": cclon_centroid,
        "cclat_centroid": cclat_centroid,
        "cclon_weightedcentroid": cclon_weightedcentroid,
        "cclat_weightedcentroid": cclat_weightedcentroid,
        "ccmajoraxis": ccmajoraxis,
        "ccminoraxis": ccminoraxis,
        "ccaspectratio": ccaspectratio,
        "ccorientation": ccorientation,
        "ccperimeter": ccperimeter,
        "cceccentricity": cceccentricity,
        "ccmaxechotop10": ccmaxechotop10,
        "ccmaxechotop20": ccmaxechotop20,
        "ccmaxechotop30": ccmaxechotop30,
        "ccmaxechotop40": ccmaxechotop40,
        "ccmaxechotop45": ccmaxechotop45,
        "ccmaxechotop50": ccmaxechotop50,
    }
    return cc_stats_dict

def calc_pf_stats(
        fillval, fillval_f, heavy_rainrate_thresh, lat, lon, minx, miny, nmaxpf, numpf, pf_npix,
        pfnumberlabelmap, pixel_radius, subdimx, subdimy, sub_rainrate_map,
        sub_sl3d_map,
        sub_echotop10_map,
        sub_echotop20_map,
        sub_echotop30_map,
        sub_echotop40_map,
        sub_echotop45_map,
        sub_echotop50_map,
        sub_pixel_area=None,
        mean_pixelength=None,
        roll_flag=False, shift_x_right=0, shift_y_top=0, xdim=None, ydim=None,
):
    """
    Calculate individual PF statistics.

    Args:
        fillval:
        fillval_f:
        heavy_rainrate_thresh:
        lat:
        lon:
        minx:
        miny:
        nmaxpf:
        numpf:
        pf_npix:
        pfnumberlabelmap:
        pixel_radius:
        subdimx:
        subdimy:
        sub_rainrate_map:
        sub_sl3d_map:
        sub_echotop10_map,
        sub_echotop20_map,
        sub_echotop30_map,
        sub_echotop40_map,
        sub_echotop45_map,
        sub_echotop50_map,

    Returns:
        pf_stats_dict: dictionary
            Dictionary containing PF statistics variables.
    """
    logger = logging.getLogger(__name__)
    # Initialize arrays
    npf_save = np.nanmin([nmaxpf, numpf])
    pfnpix = np.zeros(npf_save, dtype=float)
    pfarea = np.full(npf_save, fillval_f, dtype=float)
    pfid = np.full(npf_save, fillval, dtype=int)
    pflon = np.full(npf_save, fillval_f, dtype=float)
    pflat = np.full(npf_save, fillval_f, dtype=float)
    pfrainrate = np.full(npf_save, fillval_f, dtype=float)
    pfmaxrainrate = np.full(npf_save, fillval_f, dtype=float)
    pfskewness = np.full(npf_save, fillval_f, dtype=float)
    pfmajoraxis = np.full(npf_save, fillval_f, dtype=float)
    pfminoraxis = np.full(npf_save, fillval_f, dtype=float)
    pfaspectratio = np.full(npf_save, fillval_f, dtype=float)
    pflon_centroid = np.full(npf_save, fillval_f, dtype=float)
    pflat_centroid = np.full(npf_save, fillval_f, dtype=float)
    pflon_weightedcentroid = np.full(npf_save, fillval_f, dtype=float)
    pflat_weightedcentroid = np.full(npf_save, fillval_f, dtype=float)
    pflon_maxrainrate = np.full(npf_save, fillval_f, dtype=float)
    pflat_maxrainrate = np.full(npf_save, fillval_f, dtype=float)
    pfeccentricity = np.full(npf_save, fillval_f, dtype=float)
    pfperimeter = np.full(npf_save, fillval_f, dtype=float)
    pforientation = np.full(npf_save, fillval_f, dtype=float)
    pfaccumrain = np.full(npf_save, fillval_f, dtype=float)
    pfaccumrainheavy = np.full(npf_save, fillval_f, dtype=float)
    # Radar features
    pfccnpix = np.full(npf_save, fillval_f, dtype=float)
    pfsfnpix = np.full(npf_save, fillval_f, dtype=float)
    pfccarea = np.full(npf_save, fillval_f, dtype=float)
    pfsfarea = np.full(npf_save, fillval_f, dtype=float)
    pfccrainrate = np.full(npf_save, fillval_f, dtype=float)
    pfsfrainrate = np.full(npf_save, fillval_f, dtype=float)
    pfccrainamount = np.full(npf_save, fillval_f, dtype=float)
    pfsfrainamount = np.full(npf_save, fillval_f, dtype=float)
    pfccmaxechotop10 = np.full(npf_save, fillval_f, dtype=float)
    pfccmaxechotop20 = np.full(npf_save, fillval_f, dtype=float)
    pfccmaxechotop30 = np.full(npf_save, fillval_f, dtype=float)
    pfccmaxechotop40 = np.full(npf_save, fillval_f, dtype=float)
    pfccmaxechotop45 = np.full(npf_save, fillval_f, dtype=float)
    pfccmaxechotop50 = np.full(npf_save, fillval_f, dtype=float)
    pfccechotop40npix = np.full(npf_save, fillval_f, dtype=float)
    pfccechotop45npix = np.full(npf_save, fillval_f, dtype=float)
    pfccechotop50npix = np.full(npf_save, fillval_f, dtype=float)
    pfccechotop40area = np.full(npf_save, fillval_f, dtype=float)
    pfccechotop45area = np.full(npf_save, fillval_f, dtype=float)
    pfccechotop50area = np.full(npf_save, fillval_f, dtype=float)

    logger.debug(
        "Looping through each feature to calculate statistics"
    )
    logger.debug(("Number of PFs " + str(numpf)))

    # Get the shape of the full data array
    ny, nx = lon.shape
    # Domain bounds for circular_mean (works for both global and small idealized domains)
    lon_min = np.nanmin(lon)
    lon_max = np.nanmax(lon)
    lat_min = np.nanmin(lat)
    lat_max = np.nanmax(lat)

    ###############################################
    # Loop through each PF
    for ipf in range(1, npf_save + 1):

        #######################################
        # Find indices of the PF
        iipfy, iipfx = np.array(
            np.where(pfnumberlabelmap == ipf)
        )
        iipfnpix = len(iipfy)

        # Find indices of the PF with heavy rain
        iipfy_heavy, iipfx_heavy = np.array(
            np.where(
                (pfnumberlabelmap == ipf)
                & (sub_rainrate_map > heavy_rainrate_thresh)
            )
        )
        iipfnpix_heavy = len(iipfy_heavy)

        # Find indices of the PF with max rain rate
        iipfy_max, iipfx_max = np.unravel_index(
            np.nanargmax(sub_rainrate_map), sub_rainrate_map.shape
        )

        # Find convective indices
        iipfy_cc, iipfx_cc = np.where(
            (pfnumberlabelmap == ipf) & (sub_sl3d_map >= 1) & (sub_sl3d_map <= 2)
        )
        iipfnpix_cc = len(iipfy_cc)

        # Find stratiform indices
        iipfy_sf, iipfx_sf = np.where(
            (pfnumberlabelmap == ipf) & (sub_sl3d_map == 3)
        )
        iipfnpix_sf = len(iipfy_sf)

        # Double check to make sure PF pixel count is the same
        if iipfnpix == pf_npix[ipf - 1]:
            ##########################################
            # Compute PF statistics

            # Basic statistics
            pfnpix[ipf - 1] = np.copy(iipfnpix)
            # Compute PF area
            if sub_pixel_area is not None:
                pfarea[ipf - 1] = np.nansum(sub_pixel_area[iipfy, iipfx])
            else:
                pfarea[ipf - 1] = iipfnpix * pixel_radius ** 2
            pfid[ipf - 1] = np.copy(int(ipf))
            if roll_flag and xdim is not None:
                _yy = (iipfy[:] + miny - shift_y_top) % ydim
                _xx = (iipfx[:] + minx - shift_x_right) % xdim
                # circular_mean uses domain bounds and works correctly for both
                # global (360°) and small idealized doubly-periodic domains.
                # np.unwrap assumed 2π (360°) wrapping and failed for non-global domains.
                pflon[ipf - 1] = circular_mean(lon[_yy, _xx], lon_min, lon_max)
                pflat[ipf - 1] = circular_mean(lat[_yy, _xx], lat_min, lat_max)
            else:
                pflon[ipf - 1] = np.nanmean(lon[iipfy[:] + miny, iipfx[:] + minx])
                pflat[ipf - 1] = np.nanmean(lat[iipfy[:] + miny, iipfx[:] + minx])

            pfrainrate[ipf - 1] = np.nanmean(sub_rainrate_map[iipfy[:], iipfx[:]])
            pfmaxrainrate[ipf - 1] = np.nanmax(sub_rainrate_map[iipfy[:], iipfx[:]])
            pfskewness[ipf - 1] = skew(sub_rainrate_map[iipfy[:], iipfx[:]])
            pfaccumrain[ipf - 1] = np.nansum(sub_rainrate_map[iipfy[:], iipfx[:]])
            if iipfnpix_heavy > 0:
                pfaccumrainheavy[ipf - 1] = np.nansum(
                    sub_rainrate_map[iipfy_heavy[:], iipfx_heavy[:]]
                )

            # Convective/stratiform rain statistics
            pfccnpix[ipf - 1] = np.copy(len(iipfy_cc))
            pfsfnpix[ipf - 1] = np.copy(len(iipfy_sf))
            # Compute convective/stratiform areas
            if sub_pixel_area is not None:
                if iipfnpix_cc > 0:
                    pfccarea[ipf - 1] = np.nansum(sub_pixel_area[iipfy_cc, iipfx_cc])
                if iipfnpix_sf > 0:
                    pfsfarea[ipf - 1] = np.nansum(sub_pixel_area[iipfy_sf, iipfx_sf])
            else:
                pfccarea[ipf - 1] = len(iipfy_cc) * pixel_radius ** 2
                pfsfarea[ipf - 1] = len(iipfy_sf) * pixel_radius ** 2
            if iipfnpix_cc > 0:
                pfccrainrate[ipf - 1] = np.nanmean(sub_rainrate_map[iipfy_cc, iipfx_cc])
                pfccrainamount[ipf - 1] = np.nansum(sub_rainrate_map[iipfy_cc, iipfx_cc])
            if iipfnpix_sf > 0:
                pfsfrainrate[ipf - 1] = np.nanmean(sub_rainrate_map[iipfy_sf, iipfx_sf])
                pfsfrainamount[ipf - 1] = np.nansum(sub_rainrate_map[iipfy_sf, iipfx_sf])

            # Convective echotop height statistics
            if iipfnpix_cc > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    pfccmaxechotop10[ipf - 1] = np.nanmax(sub_echotop10_map[iipfy_cc, iipfx_cc])
                    pfccmaxechotop20[ipf - 1] = np.nanmax(sub_echotop20_map[iipfy_cc, iipfx_cc])
                    pfccmaxechotop30[ipf - 1] = np.nanmax(sub_echotop30_map[iipfy_cc, iipfx_cc])
                    pfccmaxechotop40[ipf - 1] = np.nanmax(sub_echotop40_map[iipfy_cc, iipfx_cc])
                    pfccmaxechotop45[ipf - 1] = np.nanmax(sub_echotop45_map[iipfy_cc, iipfx_cc])
                    pfccmaxechotop50[ipf - 1] = np.nanmax(sub_echotop50_map[iipfy_cc, iipfx_cc])
                    # Area containing convective echo top > X dBZ
                    pfccechotop40npix[ipf - 1] = np.count_nonzero(
                        sub_echotop40_map[iipfy_cc, iipfx_cc] > 0
                    )
                    pfccechotop45npix[ipf - 1] = np.count_nonzero(
                        sub_echotop45_map[iipfy_cc, iipfx_cc] > 0
                    )
                    pfccechotop50npix[ipf - 1] = np.count_nonzero(
                        sub_echotop50_map[iipfy_cc, iipfx_cc] > 0
                    )

                    # Compute echotop area
                    if sub_pixel_area is not None:
                        _et40_mask = sub_echotop40_map[iipfy_cc, iipfx_cc] > 0
                        _et45_mask = sub_echotop45_map[iipfy_cc, iipfx_cc] > 0
                        _et50_mask = sub_echotop50_map[iipfy_cc, iipfx_cc] > 0
                        _cc_pixel_area = sub_pixel_area[iipfy_cc, iipfx_cc]
                        pfccechotop40area[ipf - 1] = np.nansum(_cc_pixel_area[_et40_mask]) if np.any(_et40_mask) else 0
                        pfccechotop45area[ipf - 1] = np.nansum(_cc_pixel_area[_et45_mask]) if np.any(_et45_mask) else 0
                        pfccechotop50area[ipf - 1] = np.nansum(_cc_pixel_area[_et50_mask]) if np.any(_et50_mask) else 0
                    else:
                        pfccechotop40area[ipf - 1] = pfccechotop40npix[ipf - 1] * pixel_radius ** 2
                        pfccechotop45area[ipf - 1] = pfccechotop45npix[ipf - 1] * pixel_radius ** 2
                        pfccechotop50area[ipf - 1] = pfccechotop50npix[ipf - 1] * pixel_radius ** 2

            # Generate a binary map of PF
            iipfflagmap = np.zeros((subdimy, subdimx), dtype=int)
            iipfflagmap[iipfy, iipfx] = 1

            # Geometric statistics
            _sub_rainrate_map = np.copy(sub_rainrate_map)
            _sub_rainrate_map[np.isnan(_sub_rainrate_map)] = -9999
            pfproperties = regionprops(
                iipfflagmap,
                intensity_image=_sub_rainrate_map,
            )
            pfeccentricity[ipf - 1] = pfproperties[0].eccentricity
            _pixel_length = mean_pixelength if mean_pixelength is not None else pixel_radius
            pfmajoraxis[ipf - 1] = (
                    pfproperties[0].axis_major_length * _pixel_length
            )

            # Need to treat minor axis length with an error except
            # since the python algorithm occasionally throws an error.
            try:
                pfminoraxis[ipf - 1] = (
                        pfproperties[0].axis_minor_length
                        * _pixel_length
                )
            except ValueError:
                pass
            if ~np.isnan(pfminoraxis[ipf - 1]) or \
                    ~np.isnan(pfmajoraxis[ipf - 1]):
                pfaspectratio[ipf - 1] = np.divide(
                    pfmajoraxis[ipf - 1], pfminoraxis[ipf - 1]
                )
            pforientation[ipf - 1] = (
                                         pfproperties[0].orientation
                                     ) * (180 / float(pi))
            pfperimeter[ipf - 1] = (
                    pfproperties[0].perimeter * _pixel_length
            )
            # Guard against NaN before converting to int: regionprops centroid/
            # centroid_weighted can return NaN when the intensity image has no
            # valid positive weights (e.g. all-NaN or all-zero intensity_image).
            _c = pfproperties[0].centroid
            _cw = pfproperties[0].centroid_weighted
            _c_valid  = not (np.isnan(_c[0])  or np.isnan(_c[1]))
            _cw_valid = not (np.isnan(_cw[0]) or np.isnan(_cw[1]))
            yc_local  = int(np.round(_c[0]))  if _c_valid  else -1
            xc_local  = int(np.round(_c[1]))  if _c_valid  else -1
            ywc_local = int(np.round(_cw[0])) if _cw_valid else -1
            xwc_local = int(np.round(_cw[1])) if _cw_valid else -1
            # Map local sub-box centroid to full-domain coordinates.
            # When roll_flag is set, miny/minx are in the rolled domain;
            # undo the roll with inverse-shift modulo to recover the original index.
            if roll_flag and xdim is not None:
                ycentroid = (yc_local + miny - shift_y_top) % ydim
                xcentroid = (xc_local + minx - shift_x_right) % xdim
                yweightedcentroid = (ywc_local + miny - shift_y_top) % ydim
                xweightedcentroid = (xwc_local + minx - shift_x_right) % xdim
                iipfy_max_orig = (iipfy_max + miny - shift_y_top) % ydim
                iipfx_max_orig = (iipfx_max + minx - shift_x_right) % xdim
            else:
                ycentroid = yc_local + miny
                xcentroid = xc_local + minx
                yweightedcentroid = ywc_local + miny
                xweightedcentroid = xwc_local + minx
                iipfy_max_orig = iipfy_max + miny
                iipfx_max_orig = iipfx_max + minx

            # Apply the indices to get centroid lat/lon
            if (0 < ycentroid < ny) & (0 < xcentroid < nx):
                pflon_centroid[ipf-1] = lon[ycentroid, xcentroid]
                pflat_centroid[ipf-1] = lat[ycentroid, xcentroid]
            if (0 < yweightedcentroid < ny) & (0 < xweightedcentroid < nx):
                pflon_weightedcentroid[ipf-1] = lon[yweightedcentroid, xweightedcentroid]
                pflat_weightedcentroid[ipf-1] = lat[yweightedcentroid, xweightedcentroid]

            # Max rain rate location
            pflon_maxrainrate[ipf - 1] = lon[iipfy_max_orig, iipfx_max_orig]
            pflat_maxrainrate[ipf - 1] = lat[iipfy_max_orig, iipfx_max_orig]

            # When roll_flag is active (PBC domain), unwrap centroid/max-rainrate lon/lat
            # relative to pflon/pflat. Use domain width as correction size (not hardcoded
            # 360°) for robust support of both global and small idealized doubly-periodic domains.
            if roll_flag and xdim is not None:
                _lon_domain = lon_max - lon_min
                _lat_domain = lat_max - lat_min
                for _arr, _ref, _domain in [
                    (pflon_centroid, pflon, _lon_domain),
                    (pflon_weightedcentroid, pflon, _lon_domain),
                    (pflon_maxrainrate, pflon, _lon_domain),
                    (pflat_centroid, pflat, _lat_domain),
                    (pflat_weightedcentroid, pflat, _lat_domain),
                    (pflat_maxrainrate, pflat, _lat_domain),
                ]:
                    _diff = _arr[ipf - 1] - _ref[ipf - 1]
                    if _diff > _domain / 2:
                        _arr[ipf - 1] -= _domain
                    elif _diff < -_domain / 2:
                        _arr[ipf - 1] += _domain

        else:
            sys.exit("Error: PF pixel count not matching!")
    logger.debug("Loop done")

    # Put all variables in dictionary for output
    pf_stats_dict = {
        "npf_save": npf_save,
        "pfaccumrain": pfaccumrain,
        "pfaccumrainheavy": pfaccumrainheavy,
        "pfaspectratio": pfaspectratio,
        "pfarea": pfarea,
        "pfeccentricity": pfeccentricity,
        "pflat": pflat,
        "pflon": pflon,
        "pfmajoraxis": pfmajoraxis,
        "pfmaxrainrate": pfmaxrainrate,
        "pfminoraxis": pfminoraxis,
        "pforientation": pforientation,
        "pfperimeter": pfperimeter,
        "pfnpix": pfnpix,
        "pfrainrate": pfrainrate,
        "pfskewness": pfskewness,
        "pflon_centroid": pflon_centroid,
        "pflat_centroid": pflat_centroid,
        "pflon_weightedcentroid": pflon_weightedcentroid,
        "pflat_weightedcentroid": pflat_weightedcentroid,
        "pflon_maxrainrate": pflon_maxrainrate,
        "pflat_maxrainrate": pflat_maxrainrate,
        "pfccnpix": pfccnpix,
        "pfsfnpix": pfsfnpix,
        "pfccarea": pfccarea,
        "pfsfarea": pfsfarea,
        "pfccrainrate": pfccrainrate,
        "pfsfrainrate": pfsfrainrate,
        "pfccrainamount": pfccrainamount,
        "pfsfrainamount": pfsfrainamount,
        "pfccmaxechotop10": pfccmaxechotop10,
        "pfccmaxechotop20": pfccmaxechotop20,
        "pfccmaxechotop30": pfccmaxechotop30,
        "pfccmaxechotop40": pfccmaxechotop40,
        "pfccmaxechotop45": pfccmaxechotop45,
        "pfccmaxechotop50": pfccmaxechotop50,
        "pfccechotop40npix": pfccechotop40npix,
        "pfccechotop45npix": pfccechotop45npix,
        "pfccechotop50npix": pfccechotop50npix,
        "pfccechotop40area": pfccechotop40area,
        "pfccechotop45area": pfccechotop45area,
        "pfccechotop50area": pfccechotop50area,
    }
    return pf_stats_dict


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


def add_merge_split_cloud_locations(
        cloudnumbermap,
        icloudlocationx,
        icloudlocationy,
        ittmergecloudnumber,
        ittsplitcloudnumber,
        logger,
):
    """
    Add pixel location indices of merge and split clouds to the current cloud indices.

    Args:
        cloudnumbermap:
        icloudlocationt:
        icloudlocationx:
        icloudlocationy:
        ittmergecloudnumber:
        ittsplitcloudnumber:
        logger:

    Returns:
        icloudlocationt: t-indices
        icloudlocationx: x-indices
        icloudlocationy: y-indices
    """
    ######################################################################
    # Check if any small clouds merge
    logger.debug("Finding mergers")
    idmergecloudnumber = np.array(np.where(ittmergecloudnumber > 0))[0,:]
    nmergecloud = len(idmergecloudnumber)
    if nmergecloud > 0:
        # Loop over each merging cloud
        for imc in idmergecloudnumber:
            # Find location of the merging cloud
            (
                imergelocationy,
                imergelocationx,
            ) = np.array(
                np.where(cloudnumbermap == ittmergecloudnumber[imc])
            )
            nmergepix = len(imergelocationy)

            # Add merge pixes to mcs pixels
            if nmergepix > 0:
                icloudlocationy = np.hstack(
                    (icloudlocationy, imergelocationy)
                )
                icloudlocationx = np.hstack(
                    (icloudlocationx, imergelocationx)
                )
    ######################################################################
    # Check if any small clouds split
    logger.debug("Finding splits")
    idsplitcloudnumber = np.array(np.where(ittsplitcloudnumber > 0))[0,:]
    nsplitcloud = len(idsplitcloudnumber)
    if nsplitcloud > 0:
        # Loop over each merging cloud
        for imc in idsplitcloudnumber:
            # Find location of the merging cloud
            (
                isplitlocationy,
                isplitlocationx,
            ) = np.array(
                np.where(cloudnumbermap == ittsplitcloudnumber[imc])
            )
            nsplitpix = len(isplitlocationy)

            # Add split pixels to mcs pixels
            if nsplitpix > 0:
                icloudlocationy = np.hstack(
                    (icloudlocationy, isplitlocationy)
                )
                icloudlocationx = np.hstack(
                    (icloudlocationx, isplitlocationx)
                )
    return icloudlocationx, icloudlocationy