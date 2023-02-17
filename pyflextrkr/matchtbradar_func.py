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
from pyflextrkr.ftfunctions import sort_renumber
from pyflextrkr.ft_utilities import subset_ds_geolimit

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
    nmaxpf = config["nmaxpf"]
    mcs_core_min_area = config.get("mcs_core_min_area", 0)
    # ZF: nmaxcore cannot be different from nmaxpf without changing matchtbpf_driver.py
    # For now, make them the same
    nmaxcore = nmaxpf
    # nmaxcore = config.get("nmaxcore", 10)
    pfdatasource = config["pfdatasource"]
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
        dslm = xr.open_dataset(landmask_filename)
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
            # Define a list of 2D variables [tracks, times]
            var_names_2d = [
                "pf_npf",
                "pf_landfrac",
                "total_rain",
                "total_heavyrain",
                "rainrate_heavyrain",
                "conv_rain",
                "strat_rain",
                "pf_ncore",
            ]
            pf_npf = np.full(nmatchcloud, fillval, dtype=np.int16)
            pf_landfrac = np.full(nmatchcloud, fillval_f, dtype=float)
            total_rain = np.full(nmatchcloud, fillval_f, dtype=float)
            total_heavyrain = np.full(nmatchcloud, fillval_f, dtype=float)
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

                    # Isolate region over the cloud shield
                    sub_rainrate_map = np.copy(rainrate_map[miny:maxy, minx:maxx])
                    sub_reflectivity_map = np.copy(reflectivity_map[miny:maxy, minx:maxx])
                    sub_sl3d_map = np.copy(sl3d_map[miny:maxy, minx:maxx])
                    sub_echotop10_map = np.copy(echotop10_map[miny:maxy, minx:maxx])
                    sub_echotop20_map = np.copy(echotop20_map[miny:maxy, minx:maxx])
                    sub_echotop30_map = np.copy(echotop30_map[miny:maxy, minx:maxx])
                    sub_echotop40_map = np.copy(echotop40_map[miny:maxy, minx:maxx])
                    sub_echotop45_map = np.copy(echotop45_map[miny:maxy, minx:maxx])
                    sub_echotop50_map = np.copy(echotop50_map[miny:maxy, minx:maxx])

                    # Calculate total rainfall within the cold cloud shield
                    total_rain[imatchcloud] = np.nansum(sub_rainrate_map)
                    idx_heavyrain = np.where(sub_rainrate_map > heavy_rainrate_thresh)
                    if len(idx_heavyrain[0]) > 0:
                        total_heavyrain[imatchcloud] = np.nansum(
                            sub_rainrate_map[idx_heavyrain]
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
                            )

                            # Save core feature statisitcs
                            ncc_save = cc_stats_dict["ncc_save"]
                            pf_ncore[imatchcloud] = np.copy(numcc)
                            pf_corelon[imatchcloud, 0:ncc_save] = \
                                cc_stats_dict["cclon"][0:ncc_save]
                            pf_corelat[imatchcloud, 0:ncc_save] = \
                                cc_stats_dict["cclat"][0:ncc_save]
                            pf_corearea[imatchcloud, 0:ncc_save] = \
                                cc_stats_dict["ccnpix"][0:ncc_save] * pixel_radius ** 2
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
                            #  If source is GPM, the landmask is 100% for pure water, 0% for pure land
                            if pfdatasource == "imerg":
                                npix_land = np.count_nonzero(
                                    sublandmask[ipfy, ipfx] <= landfrac_thresh
                                )
                            elif pfdatasource == "wrf":
                                # WRF: landmask is 1 for land, 0 for water
                                npix_land = np.count_nonzero(sublandmask[ipfy, ipfx] == 1)
                            else:
                                logger.warning(f"WARNING: unknown pfdatasource: {pfdatasource}")
                                logger.warning("Must define how to calculate landfrac.")
                                logger.warning("pf_landfrac will be set to 0.")
                                npix_land = 0

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
                            )

                            # Save precipitation feature statisitcs
                            npf_save = pf_stats_dict["npf_save"]
                            pf_npf[imatchcloud] = np.copy(numpf)
                            pf_lon[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pflon"][0:npf_save]
                            pf_lat[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pflat"][0:npf_save]
                            pf_area[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pfnpix"][0:npf_save] * pixel_radius**2
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
                                pf_stats_dict["pfccnpix"][0:npf_save] * pixel_radius**2
                            pf_sfarea[imatchcloud, 0:npf_save] = \
                                pf_stats_dict["pfsfnpix"][0:npf_save] * pixel_radius**2
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
                                pf_stats_dict["pfccechotop40npix"][0:npf_save] * pixel_radius**2
                            pf_ccechotop45area[imatchcloud, 0: npf_save] = \
                                pf_stats_dict["pfccechotop45npix"][0:npf_save] * pixel_radius ** 2
                            pf_ccechotop50area[imatchcloud, 0: npf_save] = \
                                pf_stats_dict["pfccechotop50npix"][0:npf_save] * pixel_radius ** 2
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
            ccid[icc - 1] = np.copy(int(icc))
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
            ccmajoraxis[icc - 1] = ccproperties[0].major_axis_length * pixel_radius

            # Need to treat minor axis length with an error except
            # since the python algorithm occasionally throws an error.
            try:
                ccminoraxis[icc - 1] = (ccproperties[0].minor_axis_length * pixel_radius)
            except ValueError:
                pass
            if ~np.isnan(ccminoraxis[icc - 1]) or ~np.isnan(ccmajoraxis[icc - 1]):
                ccaspectratio[icc - 1] = np.divide(ccmajoraxis[icc - 1], ccminoraxis[icc - 1])
            ccorientation[icc - 1] = (ccproperties[0].orientation) * (180 / float(pi))
            ccperimeter[icc - 1] = (ccproperties[0].perimeter * pixel_radius)
            [ycentroid, xcentroid] = ccproperties[0].centroid
            [yweightedcentroid, xweightedcentroid] = ccproperties[0].weighted_centroid

            # Shift the centroids by minx/miny
            # since the core is a subset from the full image
            # Round the centroid values as indices
            if (~np.isnan(ycentroid)) & (~np.isnan(xcentroid)):
                ycentroid = int(np.round(ycentroid + miny))
                xcentroid = int(np.round(xcentroid + minx))
            if (~np.isnan(yweightedcentroid)) & (~np.isnan(xweightedcentroid)):
                yweightedcentroid = int(np.round(yweightedcentroid + miny))
                xweightedcentroid = int(np.round(xweightedcentroid + minx))

            # Apply the indices to get centroid lat/lon
            if (0 < (ycentroid) < ny) & (0 < (xcentroid) < nx):
                cclon_centroid[icc - 1] = lon[ycentroid, xcentroid]
                cclat_centroid[icc - 1] = lat[ycentroid, xcentroid]
            if (0 < (yweightedcentroid) < ny) & (0 < (xweightedcentroid) < nx):
                cclon_weightedcentroid[icc - 1] = lon[yweightedcentroid, xweightedcentroid]
                cclat_weightedcentroid[icc - 1] = lat[yweightedcentroid, xweightedcentroid]

    # Put all variables in dictionary for output
    cc_stats_dict = {
        "ncc_save": ncc_save,
        "ccnpix": ccnpix,
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

    logger.debug(
        "Looping through each feature to calculate statistics"
    )
    logger.debug(("Number of PFs " + str(numpf)))

    # Get the shape of the full data array
    ny, nx = lon.shape

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
            pfid[ipf - 1] = np.copy(int(ipf))
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
            pfmajoraxis[ipf - 1] = (
                    pfproperties[0].major_axis_length * pixel_radius
            )

            # Need to treat minor axis length with an error except
            # since the python algorithm occasionally throws an error.
            try:
                pfminoraxis[ipf - 1] = (
                        pfproperties[0].minor_axis_length
                        * pixel_radius
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
                    pfproperties[0].perimeter * pixel_radius
            )
            [
                ycentroid,
                xcentroid,
            ] = pfproperties[0].centroid
            [
                yweightedcentroid,
                xweightedcentroid,
            ] = pfproperties[0].weighted_centroid

            # Shift the centroids by minx/miny
            # since the PF is a subset from the full image
            # Round the centroid values as indices
            ycentroid = int(np.round(ycentroid + miny))
            xcentroid = int(np.round(xcentroid + minx))
            yweightedcentroid = int(np.round(yweightedcentroid + miny))
            xweightedcentroid = int(np.round(xweightedcentroid + minx))

            # Apply the indices to get centroid lat/lon
            if (0 < (ycentroid) < ny) & (0 < (xcentroid) < nx):
                pflon_centroid[ipf-1] = lon[ycentroid, xcentroid]
                pflat_centroid[ipf-1] = lat[ycentroid, xcentroid]
            if (0 < (yweightedcentroid) < ny) & (0 < (xweightedcentroid) < nx):
                pflon_weightedcentroid[ipf-1] = lon[yweightedcentroid, xweightedcentroid]
                pflat_weightedcentroid[ipf-1] = lat[yweightedcentroid, xweightedcentroid]

            # Shift the x, y indices by minx/miny
            pflon_maxrainrate[ipf - 1] = lon[iipfy_max + miny, iipfx_max + minx]
            pflat_maxrainrate[ipf - 1] = lat[iipfy_max + miny, iipfx_max + minx]

        else:
            sys.exit("Error: PF pixel count not matching!")
    logger.debug("Loop done")

    # Put all variables in dictionary for output
    pf_stats_dict = {
        "npf_save": npf_save,
        "pfaccumrain": pfaccumrain,
        "pfaccumrainheavy": pfaccumrainheavy,
        "pfaspectratio": pfaspectratio,
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