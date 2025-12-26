import numpy as np
import os.path
import sys
import logging
import warnings
import xarray as xr
from scipy.ndimage import label
from skimage.measure import regionprops
from math import pi
from scipy.stats import skew
from pyflextrkr.ftfunctions import sort_renumber
from pyflextrkr.ft_utilities import subset_ds_geolimit
from pyflextrkr.ftfunctions import circular_mean, get_cloud_boundary, find_max_indices_to_roll, subset_roll_map

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
    # pfdatasource = config["pfdatasource"]
    landmask_filename = config.get("landmask_filename", "")
    landmask_varname = config.get("landmask_varname", "")
    landfrac_thresh = config.get("landfrac_thresh", 0)
    landmask_x_dimname = config.get("landmask_x_dimname", None)
    landmask_y_dimname = config.get("landmask_y_dimname", None)
    landmask_x_coordname = config.get("landmask_x_coordname", None)
    landmask_y_coordname = config.get("landmask_y_coordname", None)
    # Parameters for handling perdiodic boundary condition
    pbc_direction = config.get("pbc_direction", "none")
    max_feature_frac_x = 0.95   # Max fraction of domain size for a feature in x-direction
    max_feature_frac_y = 0.95   # Max fraction of domain size for a feature in y-direction

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
        lon_min = np.nanmin(lon)
        lon_max = np.nanmax(lon)
        lat_min = np.nanmin(lat)
        lat_max = np.nanmax(lat)
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
                "rainrate_p95",
            ]
            
            pf_npf = np.full(nmatchcloud, fillval, dtype=np.int16)
            pf_landfrac = np.full(nmatchcloud, fillval_f, dtype=float)
            total_rain = np.full(nmatchcloud, fillval_f, dtype=float)
            total_heavyrain = np.full(nmatchcloud, fillval_f, dtype=float)
            rainrate_heavyrain = np.full(nmatchcloud, fillval_f, dtype=float)
            rainrate_p95 = np.full(nmatchcloud, fillval_f, dtype=float)
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

            # Loop over each matched cloud number
            for imatchcloud in range(nmatchcloud):

                ittcloudnumber = ir_cloudnumber[imatchcloud]
                ittmergecloudnumber = ir_mergecloudnumber[imatchcloud]
                ittsplitcloudnumber = ir_splitcloudnumber[imatchcloud]
                basetime[imatchcloud] = cloudid_basetime

                #########################################################################
                # Intialize matrices for only MCS data
                rainrate_map = np.full((ydim, xdim), np.nan, dtype=float)
                lon_map = np.full((ydim, xdim), np.nan, dtype=float)
                lat_map = np.full((ydim, xdim), np.nan, dtype=float)
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
                    lon_map[icloudlocationy, icloudlocationx] = np.copy(lon[icloudlocationy, icloudlocationx])
                    lat_map[icloudlocationy, icloudlocationx] = np.copy(lat[icloudlocationy, icloudlocationx])


                    ########################################################################
                    ## Isolate small region of cloud data around mcs at this time
                    logger.debug("Calculate new shape statistics")

                    # Get cloud boundary
                    maxx, maxy, minx, miny = get_cloud_boundary(icloudlocationx,
                                                                icloudlocationy,
                                                                xdim,
                                                                ydim)
                    
                    # Check cloud boundary span
                    # If boundary span > X fraction of domain, and periodic boundary condition is set,
                    # roll the data such that the cloud does not span across the domain boundary
                    roll_flag = False
                    if (((maxx - minx) >= xdim * max_feature_frac_x) or \
                        ((maxy - miny) >= ydim * max_feature_frac_y)) and \
                        (pbc_direction != 'none'):
                        # Subset cloudnumber mask that contains the current cloud
                        sub_mask = cloudnumbermap[miny:maxy, minx:maxx] == ittcloudnumber
                        # Find the indices to roll the array to avoid periodic boundary condition
                        shift_x_right, shift_y_top = find_max_indices_to_roll(
                            sub_mask, xdim, ydim,
                        )
                        # Subset rainrate over the cloud shield
                        sub_rainrate = np.copy(rainrate_map[miny:maxy, minx:maxx])
                        sub_lon = np.copy(lon_map[miny:maxy, minx:maxx])
                        sub_lat = np.copy(lat_map[miny:maxy, minx:maxx])
                        # Check sub_rainrate array size and 
                        # make sure there are pixels above the rainrate threshold
                        if (sub_rainrate.size > 0) and \
                            (sub_rainrate.shape[0] > 0) and \
                            (sub_rainrate.shape[1] > 0) and \
                            (np.any(sub_rainrate > pf_rr_thres)):
                            # Roll rainrate to avoid periodic boundary condition
                            sub_rainrate_map = subset_roll_map(
                                sub_rainrate, shift_x_right, shift_y_top, xdim, ydim,
                            )
                            lon_roll = subset_roll_map(sub_lon, shift_x_right, shift_y_top, xdim, ydim)
                            lat_roll = subset_roll_map(sub_lat, shift_x_right, shift_y_top, xdim, ydim)
                            roll_flag = True
                        else:
                            sub_rainrate_map = sub_rainrate
                            shift_x_right = 0
                            shift_y_top = 0
                            lon_roll = sub_lon
                            lat_roll = sub_lat
                    else:
                        # Isolate region over the cloud shield
                        sub_rainrate_map = np.copy(rainrate_map[miny:maxy, minx:maxx])
                        shift_x_right = 0
                        shift_y_top = 0
                        lon_roll = None
                        lat_roll = None

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

                    ######################################################
                    # Derive individual PF statistics
                    logger.debug("Calculating precipitation statistics")

                    ipfy, ipfx = np.array(np.where(sub_rainrate_map > pf_rr_thres))
                    nrainpix = len(ipfy)

                    # Calculate 95th percentile of rain rate
                    if nrainpix == 0:
                        rainrate_p95[imatchcloud] = 0 
                    else:
                    #  print (nrainpix)
                        rainrate_p95[imatchcloud] = np.quantile(sub_rainrate_map[ipfy, ipfx], 0.95)
                    #  print (np.quantile(sub_rainrate_map[ipfy, ipfx], 0.95))
                    
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
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", category=RuntimeWarning)
                                
                                pf_stats_dict = calc_pf_stats(
                                    fillval, fillval_f, heavy_rainrate_thresh,
                                    lat, lon, minx, miny, nmaxpf, numpf,
                                    pf_npix, pfnumberlabelmap, pixel_radius,
                                    subdimx, subdimy, sub_rainrate_map,
                                    roll_flag=roll_flag,
                                    lon_roll=lon_roll, lat_roll=lat_roll,
                                    lon_min=lon_min, lon_max=lon_max,
                                    lat_min=lat_min, lat_max=lat_max,
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
                "rainrate_p95":rainrate_p95,
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
                "rainrate_p95":{
                    "long_name": "95th percentile of rain rate under cold cloud shield",
                    "units":'mm/h',
                    "_FillValue": fillval_f,
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
            }

            return out_dict, out_dict_attrs, var_names_2d

        else:
            logger.info("No matching cloud found in cloudid: " + cloudid_filename)

    else:
        logger.info("cloudid file does not exist: " + cloudid_filename)



##########################################################################
# Custom functions
#-----------------------------------------------------------------------
def calc_pf_stats(
        fillval, fillval_f, heavy_rainrate_thresh, lat, lon, minx, miny, nmaxpf, numpf, pf_npix,
        pfnumberlabelmap, pixel_radius, subdimx, subdimy, sub_rainrate_map,
        roll_flag=False, lon_roll=None, lat_roll=None,
        lon_min=None, lon_max=None, lat_min=None, lat_max=None,
):
    """
    Calculate individual PF statistics.

    Args:
        fillval:
            Fill value from config file.
        fillval_f: float
            Fill value for float data.
        heavy_rainrate_thresh: float
            Heavy rain rate threshold.
        lat: np.array()
            2D latitude array of the full domain.
        lon: np.array()
            2D longitude array of the full domain.
        minx: int
            Lower-left corner X index of the box containing the cloud mask.
        miny: int
            Lower-left corner Y index of the box containing the cloud mask.
        nmaxpf: int
            Number of PFs to save.
        numpf: int
            Number of PFs within the cloud.
        pf_npix: np.array()
            Number of pixels in each PF.
        pfnumberlabelmap: np.array()
            Subsetted 2D labeled map of PFs.
        pixel_radius: float
            Pixel radius.
        subdimx: int
            X dimension of the subsetted region.
        subdimy: int
            Y dimension of the subsetted region.
        sub_rainrate_map: np.array()
            Subsetted 2D rain rate array.
        roll_flag: bool, default=False
            Flag to indicate if the data is rolled.
        lon_roll: np.array(), default=None
            Subsetted 2D longitude array after rolling.
        lat_roll: np.array(), default=None
            Subsetted 2D latitude array after rolling.
        lon_min: float, default=None
            Minimum longitude value of the full domain.
        lon_max: float, default=None
            Maximum longitude value of the full domain.
        lat_min: float, default=None
            Minimum latitude value of the full domain.
        lat_max: float, default=None
            Maximum latitude value of the full domain.

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

        # Double check to make sure PF pixel count is the same
        if iipfnpix == pf_npix[ipf - 1]:
            ##########################################
            # Compute PF statistics

            # Basic statistics
            pfnpix[ipf - 1] = np.copy(iipfnpix)
            pfid[ipf - 1] = np.copy(int(ipf))
            if roll_flag:
                # Remove NaN values from the rolled lon/lat arrays
                lon_roll_v = lon_roll[~np.isnan(lon_roll)]
                lat_roll_v = lat_roll[~np.isnan(lat_roll)]
                # Calculate circular mean of lon/lat
                pflon[ipf - 1] = circular_mean(lon_roll_v, lon_min, lon_max)
                pflat[ipf - 1] = circular_mean(lat_roll_v, lat_min, lat_max)
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
            # TODO: the lat/lon values below are not correct for PFs at the boundary of the domain
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
    }
    return pf_stats_dict

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
