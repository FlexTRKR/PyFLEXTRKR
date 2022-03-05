import numpy as np
import os.path
import sys
import logging
import xarray as xr
from scipy.ndimage import label
from skimage.measure import regionprops
from math import pi
from scipy.stats import skew
from pyflextrkr.ftfunctions import sort_renumber

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
    pfdatasource = config["pfdatasource"]
    landmask_filename = config.get("landmask_filename", "")
    landmask_varname = config.get("landmask_varname", "")
    landfrac_thresh = config.get("landfrac_thresh", 0)
    fillval = config["fillval"]
    fillval_f = np.nan

    # Read landmask file
    if os.path.isfile(landmask_filename):
        dslm = xr.open_dataset(landmask_filename)
        landmask = dslm[landmask_varname].squeeze().data
    else:
        landmask = None

    # Read cloudid file
    if os.path.isfile(cloudid_filename):
        # Load cloudid data
        logger.debug("Loading cloudid data")
        logger.debug(cloudid_filename)
        ds = xr.open_dataset(
            cloudid_filename,
            mask_and_scale=False,
            decode_times=False,
        )
        cloudnumbermap = ds[feature_varname].values
        rawrainratemap = ds["precipitation"].values
        cloudid_basetime = ds["base_time"].values
        lon = ds["longitude"].values
        lat = ds["latitude"].values
        ds.close()

        # Get dimensions of data
        ydim, xdim = np.shape(lat)

        # Number of clouds
        nmatchcloud = len(ir_cloudnumber)

        if nmatchcloud > 0:
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
            pf_lon_centroid = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_lat_centroid = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_lon_weightedcentroid = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_lat_weightedcentroid = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_accumrain = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            pf_accumrainheavy = np.full((nmatchcloud, nmaxpf), fillval_f, dtype=float)
            basetime = np.full(nmatchcloud, fillval_f, dtype=float)

            for imatchcloud in range(nmatchcloud):

                ittcloudnumber = ir_cloudnumber[imatchcloud]
                ittmergecloudnumber = ir_mergecloudnumber[imatchcloud]
                ittsplitcloudnumber = ir_splitcloudnumber[imatchcloud]
                basetime[imatchcloud] = cloudid_basetime
                # precip_basetime[imatchcloud] = cloudid_basetime

                #########################################################################
                # Intialize matrices for only MCS data
                filteredrainratemap = np.ones((ydim, xdim), dtype=float) * np.nan
                logger.debug(
                    ("filteredrainratemap allocation size: ", filteredrainratemap.shape)
                )

                ############################################################################
                # Find matching cloud number
                icloudlocationt, icloudlocationy, icloudlocationx = np.array(
                    np.where(cloudnumbermap == ittcloudnumber)
                )
                ncloudpix = len(icloudlocationy)

                if ncloudpix > 0:
                    logger.debug("IR Clouds Present")
                    # Add merge/split cloud pixel locations
                    icloudlocationt, \
                    icloudlocationx, \
                    icloudlocationy = add_merge_split_cloud_locations(cloudnumbermap,
                                                                      icloudlocationt,
                                                                      icloudlocationx,
                                                                      icloudlocationy,
                                                                      ittmergecloudnumber,
                                                                      ittsplitcloudnumber,
                                                                      logger)

                    ########################################################################
                    # Fill matrices with MCS data
                    logger.debug("Fill map with data")
                    filteredrainratemap[icloudlocationy, icloudlocationx] = np.copy(
                        rawrainratemap[
                            icloudlocationt, icloudlocationy, icloudlocationx
                        ]
                    )

                    ########################################################################
                    ## Isolate small region of cloud data around mcs at this time
                    logger.debug("Calculate new shape statistics")

                    # Get cloud boundary
                    maxx, maxy, minx, miny = get_cloud_boundary(icloudlocationx,
                                                                icloudlocationy,
                                                                xdim,
                                                                ydim)

                    ## Isolate smaller region around cloud shield
                    subrainratemap = np.copy(filteredrainratemap[miny:maxy, minx:maxx])
                    # sublon = lon[miny:maxy, minx:maxx]
                    # sublat = lat[miny:maxy, minx:maxx]

                    # Calculate total rainfall within the cold cloud shield
                    total_rain[imatchcloud] = np.nansum(subrainratemap)
                    idx_heavyrain = np.where(subrainratemap > heavy_rainrate_thresh)
                    if len(idx_heavyrain[0]) > 0:
                        total_heavyrain[imatchcloud] = np.nansum(
                            subrainratemap[idx_heavyrain]
                        )
                        rainrate_heavyrain[imatchcloud] = np.nanmean(
                            subrainratemap[idx_heavyrain]
                        )

                    ######################################################
                    # !!!!!!!!!!!!!!! Slow Step !!!!!!!!!!!!!!!
                    # Derive individual PF statistics
                    logger.debug("Calculating precipitation statistics")

                    ipfy, ipfx = np.array(np.where(subrainratemap > pf_rr_thres))
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
                        subdimy, subdimx = np.shape(subrainratemap)
                        # Dilate precipitation feature by one pixel.
                        # This slightly smooths the data so that very close precipitation features are connected
                        # Create binary map
                        # binarypfmap = np.zeros((ydim, xdim), dtype=int)
                        binarypfmap = np.zeros((subdimy, subdimx), dtype=int)
                        binarypfmap[ipfy, ipfx] = 1

                        # Dilate (aka smooth)
                        # Defines shape of growth. This grows one pixel as a cross
                        # dilationstructure = generate_binary_structure(2,1)
                        # dilatedbinarypfmap = binary_dilation(
                        #     binarypfmap,
                        #     structure=dilationstructure,
                        #     iterations=1
                        # ).astype(filteredrainratemap.dtype)

                        # Label precipitation features
                        # pfnumberlabelmap, numpf = label(dilatedbinarypfmap)
                        pfnumberlabelmap, numpf = label(binarypfmap)

                        # Sort numpf then calculate stats
                        min_npix = np.ceil(pf_link_area_thresh / (pixel_radius ** 2)).astype(int)

                        # Sort and renumber PFs, and remove small PFs
                        pf_number, pf_npix = sort_renumber(pfnumberlabelmap, min_npix)
                        # Update number of PFs after sorting and renumbering
                        npf_new = np.nanmax(pf_number)
                        numpf = npf_new
                        pfnumberlabelmap = pf_number
                        del pf_number, npf_new

                        # if npf_new > 0:
                        if numpf > 0:
                            ###################################################
                            logger.debug("PFs present, calculating statistics")

                            # Call function to calculate individual PF statistics
                            pf_stats_dict = calc_pf_stats(
                                fillval, fillval_f, heavy_rainrate_thresh,
                                lat, logger, lon, minx, miny, nmaxpf, numpf,
                                pf_npix, pfnumberlabelmap, pixel_radius,
                                subdimx, subdimy, subrainratemap,
                                # sublon, sublat,
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

            # Group outputs in dictionaries
            out_dict = {
                # "nmatchcloud": nmatchcloud,
                # "matchindices": matchindices,
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
                "pf_eccentricity": pf_eccentricity,
                "pf_lon_centroid": pf_lon_centroid,
                "pf_lat_centroid": pf_lat_centroid,
                "pf_lon_weightedcentroid": pf_lon_weightedcentroid,
                "pf_lat_weightedcentroid": pf_lat_weightedcentroid,
                "pf_maxrainrate": pf_maxrainrate,
                "pf_accumrain": pf_accumrain,
                "pf_accumrainheavy": pf_accumrainheavy,
                "pf_landfrac": pf_landfrac,
                "total_rain": total_rain,
                "total_heavyrain": total_heavyrain,
                "rainrate_heavyrain": rainrate_heavyrain,
                # basetime,
                # precip_basetime,
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
            }

            return out_dict, out_dict_attrs,

        else:
            logger.debug("No matching cloud found in cloudid: " + cloudid_filename)

    else:
        logger.debug("cloudid file does not exist: " + cloudid_filename)



##########################################################################
# Custom functions
def calc_pf_stats(
        fillval, fillval_f, heavy_rainrate_thresh, lat, logger, lon, minx, miny, nmaxpf, numpf, pf_npix,
        pfnumberlabelmap, pixel_radius, subdimx, subdimy, subrainratemap,
        # sublon, sublat,
):
    """
    Calculate individual PF statistics.

    Args:
        fillval:
        fillval_f:
        heavy_rainrate_thresh:
        lat:
        logger:
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
        subrainratemap:

    Returns:
        pf_stats_dict: dictionary
            Dictionary containing PF statistics variables.
    """
    ##############################################
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
    pfeccentricity = np.full(npf_save, fillval_f, dtype=float)
    pfperimeter = np.full(npf_save, fillval_f, dtype=float)
    pforientation = np.full(npf_save, fillval_f, dtype=float)
    pfaccumrain = np.full(npf_save, fillval_f, dtype=float)
    pfaccumrainheavy = np.full(npf_save, fillval_f, dtype=float)
    logger.debug(
        "Looping through each feature to calculate statistics"
    )
    logger.debug(("Number of PFs " + str(numpf)))
    ###############################################
    # Loop through each PF
    for ipf in range(1, npf_save + 1):

        #######################################
        # Find indices of the PF
        iipfy, iipfx = np.array(
            np.where(pfnumberlabelmap == ipf)
        )
        niipfpix = len(iipfy)

        # Find indices of the PF with heavy rain
        iipfy_heavy, iipfx_heavy = np.array(
            np.where(
                (pfnumberlabelmap == ipf)
                & (subrainratemap > heavy_rainrate_thresh)
            )
        )
        niipfpix_heavy = len(iipfy_heavy)

        # Double check to make sure PF pixel count is the same
        if niipfpix == pf_npix[ipf - 1]:
            ##########################################
            # Compute PF statistics

            # Basic statistics
            pfnpix[ipf - 1] = np.copy(niipfpix)
            pfid[ipf - 1] = np.copy(int(ipf))
            pflon[ipf - 1] = np.nanmean(
                lon[iipfy[:] + miny, iipfx[:] + minx]
            )
            pflat[ipf - 1] = np.nanmean(
                lat[iipfy[:] + miny, iipfx[:] + minx]
            )
            # pflon[ipf - 1] = np.nanmean(sublon[iipfy[:], iipfx[:]])
            # pflat[ipf - 1] = np.nanmean(sublat[iipfy[:], iipfx[:]])

            pfrainrate[ipf - 1] = np.nanmean(
                subrainratemap[iipfy[:], iipfx[:]]
            )
            pfmaxrainrate[ipf - 1] = np.nanmax(
                subrainratemap[iipfy[:], iipfx[:]]
            )
            pfskewness[ipf - 1] = skew(
                subrainratemap[iipfy[:], iipfx[:]]
            )
            pfaccumrain[ipf - 1] = np.nansum(
                subrainratemap[iipfy[:], iipfx[:]]
            )
            if niipfpix_heavy > 0:
                pfaccumrainheavy[ipf - 1] = np.nansum(
                    subrainratemap[
                        iipfy_heavy[:], iipfx_heavy[:]
                    ]
                )

            # Generate a binary map of PF
            iipfflagmap = np.zeros((subdimy, subdimx), dtype=int)
            iipfflagmap[iipfy, iipfx] = 1

            # Geometric statistics
            tfilteredrainratemap = np.copy(subrainratemap)
            tfilteredrainratemap[
                np.isnan(tfilteredrainratemap)
            ] = -9999
            pfproperties = regionprops(
                iipfflagmap,
                intensity_image=tfilteredrainratemap,
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
            # since the the PF is a subset from the full image
            # Round the centroid values as indices
            ycentroid = int(np.round(ycentroid + miny))
            xcentroid = int(np.round(xcentroid + minx))
            yweightedcentroid = int(np.round(yweightedcentroid + miny))
            xweightedcentroid = int(np.round(xweightedcentroid + minx))

            # Apply the indices to get centroid lat/lon
            pflon_centroid[ipf-1] = lon[ycentroid, xcentroid]
            pflat_centroid[ipf-1] = lat[ycentroid, xcentroid]
            pflon_weightedcentroid[ipf-1] = lon[yweightedcentroid, xweightedcentroid]
            pflat_weightedcentroid[ipf-1] = lat[yweightedcentroid, xweightedcentroid]

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
        "pfnpix": pfnpix,
        "pforientation": pforientation,
        "pfrainrate": pfrainrate,
        "pfskewness": pfskewness,
        "pflon_centroid": pflon_centroid,
        "pflon_weightedcentroid": pflon_weightedcentroid,
        "pflat_centroid": pflat_centroid,
        "pflat_weightedcentroid": pflat_weightedcentroid,
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
    miny = np.nanmin(icloudlocationy)
    if miny <= 10:
        miny = 0
    else:
        miny = miny - 10
    maxy = np.nanmax(icloudlocationy)
    if maxy >= ydim - 10:
        maxy = ydim
    else:
        maxy = maxy + 11
    minx = np.nanmin(icloudlocationx)
    if minx <= 10:
        minx = 0
    else:
        minx = minx - 10
    maxx = np.nanmax(icloudlocationx)
    if maxx >= xdim - 10:
        maxx = xdim
    else:
        maxx = maxx + 11
    return maxx, maxy, minx, miny


def add_merge_split_cloud_locations(cloudnumbermap, icloudlocationt, icloudlocationx, icloudlocationy,
                                    ittmergecloudnumber, ittsplitcloudnumber, logger):
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
                imergelocationt,
                imergelocationy,
                imergelocationx,
            ) = np.array(
                np.where(cloudnumbermap == ittmergecloudnumber[imc])
            )
            nmergepix = len(imergelocationy)

            # Add merge pixes to mcs pixels
            if nmergepix > 0:
                icloudlocationt = np.hstack(
                    (icloudlocationt, imergelocationt)
                )
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
                isplitlocationt,
                isplitlocationy,
                isplitlocationx,
            ) = np.array(
                np.where(cloudnumbermap == ittsplitcloudnumber[imc])
            )
            nsplitpix = len(isplitlocationy)

            # Add split pixels to mcs pixels
            if nsplitpix > 0:
                icloudlocationt = np.hstack(
                    (icloudlocationt, isplitlocationt)
                )
                icloudlocationy = np.hstack(
                    (icloudlocationy, isplitlocationy)
                )
                icloudlocationx = np.hstack(
                    (icloudlocationx, isplitlocationx)
                )
    return icloudlocationt, icloudlocationx, icloudlocationy