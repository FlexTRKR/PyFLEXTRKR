import os
import sys
import logging
import numpy as np
import xarray as xr
from datetime import datetime
from scipy.signal import medfilt2d
from scipy.ndimage import label, filters
from astropy.convolution import Box2DKernel, convolve
from pyflextrkr import netcdf_io as net
from pyflextrkr.ftfunctions import olr_to_tb
from pyflextrkr.futyan3 import futyan3
from pyflextrkr.label_and_grow_cold_clouds import label_and_grow_cold_clouds
from pyflextrkr.ftfunctions import sort_renumber, sort_renumber2vars, link_pf_tb
from pyflextrkr.sl3d_func import run_sl3d

def idclouds_tbpf(
    filename,
    config,
):
    """
    Identifies convective cloud objects from infrared brightness temperature and precipitation data.

    Args:
        filename: string
            Input data filename
        config: dictionary
            Dictionary containing config parameters

    Returns:
        cloudid_outfile: string
            Cloudid file name.
    """
    np.set_printoptions(threshold=np.inf)
    logger = logging.getLogger(__name__)
    logger.debug(f"Processing {filename}.")

    # Flag to handle a special case for 'gpmirimerg'
    clouddatasource = config['clouddatasource']
    # Set medfilt2d kernel_size, this determines the filter window dimension
    medfiltsize = config.get('medfiltsize', 5)
    idclouds_hourly = config.get('idclouds_hourly', 0)
    idclouds_minute = config.get('idclouds_minute', 0)
    # Default idclouds minute difference allowed
    idclouds_dt_thresh = config.get('idclouds_dt_thresh', 5)
    # minimum and maximum brightness temperature thresholds. data outside of this range is filtered
    mintb_thresh = config['absolutetb_threshs'][0]
    maxtb_thresh = config['absolutetb_threshs'][1]
    # Get Tb thresholds
    thresh_core = config['cloudtb_core']
    thresh_cold = config['cloudtb_cold']
    thresh_warm = config['cloudtb_warm']
    thresh_cloud = config['cloudtb_cloud']
    cloudtb_threshs = [thresh_core, thresh_cold, thresh_warm, thresh_cloud]
    miss_thresh = config['miss_thresh']
    tb_varname = config.get("tb_varname", 'tb')
    geolimits = config['geolimits']
    cloudidmethod = config['cloudidmethod']
    pixel_radius = config['pixel_radius']
    area_thresh = config['area_thresh']
    mincoldcorepix = config['mincoldcorepix']
    smoothwindowdimensions = config['smoothwindowdimensions']
    warmanvilexpansion = config['warmanvilexpansion']
    olr2tb = config.get('olr2tb', False)
    olr_varname = config.get('olr_varname', None)
    # PF parameters
    linkpf = config.get('linkpf', 0)
    pcp_varname = config['pcp_varname']
    linkpf_varname = config.get('linkpf_varname', pcp_varname)
    pcp_convert_factor = config.get('pcp_convert_factor', 1)
    pf_smooth_window = config.get('pf_smooth_window', 0)
    pf_dbz_thresh = config.get('pf_dbz_thresh', 0)
    pf_link_area_thresh = config.get('pf_link_area_thresh', 0)
    feature_type = config['feature_type']
    # Output file name parameters
    tracking_outpath = config['tracking_outpath']
    cloudid_filebase = config['cloudid_filebase']

    tcoord_name = config.get('tcoord_name', 'time')
    x_coordname = config['x_coordname']
    y_coordname = config['y_coordname']
    time_dimname = config.get('time_dimname', 'time')
    x_dimname = config.get('x_dimname', 'lon')
    y_dimname = config.get('y_dimname', 'lat')
    z_dimname = config.get('z_dimname', None)

    cloudid_outfile = None
    logger.debug(filename)

    # Initialize optional variables
    sl3d_dict = None
    sl3d_attrs = None

    # Read in Tb data using xarray
    rawdata = xr.open_dataset(filename)

    # Get dimension names from the file
    dims_file = []
    for key in rawdata.dims: dims_file.append(key)
    # Find extra dimensions beyond [time, z, y, x]
    dims_keep = [time_dimname, z_dimname, y_dimname, x_dimname]
    dims_drop = list(set(dims_file) - set(dims_keep))
    # Reorder Dataset dimensions
    if z_dimname is not None:
        # Drop extra dimensions, reorder to [time, z, y, x]
        rawdata = rawdata.drop_dims(dims_drop).transpose(
            time_dimname, z_dimname, y_dimname, x_dimname, missing_dims='ignore'
        )
    else:
        # Drop extra dimensions, reorder to [time, y, x]
        rawdata = rawdata.drop_dims(dims_drop).transpose(
            time_dimname, y_dimname, x_dimname, missing_dims='ignore'
        )

    # Convert OLR to Tb if olr2tb flag is set
    if olr2tb is True:
        olr = rawdata[olr_varname].data
        original_ir = olr_to_tb(olr)
    else:
        # Read Tb from data
        original_ir = rawdata[tb_varname].data
    rawdata.close()

    lat = rawdata[y_coordname].data
    lon = rawdata[x_coordname].data
    time_decode = rawdata[tcoord_name]

    # Check coordinate dimensions
    if (lat.ndim == 1) | (lon.ndim == 1):
        # Mesh 1D coordinate into 2D
        in_lon, in_lat = np.meshgrid(lon, lat)
    elif (lat.ndim == 2) | (lon.ndim == 2):
        in_lon = lon
        in_lat = lat
    else:
        logger.critical("ERROR: Unexpected input data x, y coordinate dimensions.")
        logger.critical(f"{x_coordname} dimension: {lon.ndim}")
        logger.critical(f"{y_coordname} dimension: {lat.ndim}")
        logger.critical("Tracking will now exit.")
        sys.exit()

    ##############################################################################
    # Subset input dataset within geolimits
    # Find indices within lat/lon range set by geolimits
    indicesy, indicesx = np.array(
        np.where(
            (in_lat >= geolimits[0])
            & (in_lat <= geolimits[2])
            & (in_lon >= geolimits[1])
            & (in_lon <= geolimits[3])
        )
    )
    ymin, ymax = np.nanmin(indicesy), np.nanmax(indicesy) + 1
    xmin, xmax = np.nanmin(indicesx), np.nanmax(indicesx) + 1
    # Create a dictionary for dataset subset
    subset_dict = {
        y_dimname: slice(ymin, ymax),
        x_dimname: slice(xmin, xmax),
    }
    # Subset dataset
    rawdata = rawdata[subset_dict]
    ##############################################################################

    # Loop over each time
    for tt in range(0, len(time_decode)):
        # Process time variable
        iTime = rawdata.indexes['time'][tt]
        # Convert to basetime (i.e., Epoch time)
        file_basetime = np.array([(np.datetime64(iTime).item() - datetime(1970,1,1,0,0,0)).total_seconds()])
        # Convert to strings
        file_datestring = iTime.strftime("%Y%m%d")
        file_timestring = iTime.strftime("%H%M")
        iminute = iTime.minute

        # If idclouds_hourly is set to 1, then check if iminutes is
        # within the allowed difference from idclouds_dt_thresh
        # If so proceed, otherwise, skip this time
        if idclouds_hourly == 1:
            if np.absolute(iminute - idclouds_minute) < idclouds_dt_thresh:
                idclouds_proceed = 1
            else:
                idclouds_proceed = 0
        else:
            idclouds_proceed = 1

        # Proceed to idclodus if flag is 1
        if idclouds_proceed == 1:
            in_ir = original_ir[tt, :, :]

            # Use median filter to fill in missing values
            ir_filt = medfilt2d(in_ir, kernel_size=medfiltsize)
            # Copy the original IR data
            out_ir = np.copy(in_ir)
            # Create a mask for the missing pixels
            missmask = np.isnan(in_ir)
            # Fill in the missing pixels with the filtered values, retain the rest
            out_ir[missmask] = ir_filt[missmask]

            #####################################################
            # Mask brightness temperatures outside of normal range
            out_ir[out_ir < mintb_thresh] = np.nan
            out_ir[out_ir > maxtb_thresh] = np.nan

            # proceed if file covers the geographic region in interest
            if (len(indicesx) > 0) and (len(indicesy) > 0):

                # Determine number of missing data
                missingcount = np.count_nonzero(np.isnan(out_ir))
                ny, nx = np.shape(out_ir)
                # Proceed if fraction of missing data does not exceed threshold
                if np.divide(missingcount, (ny * nx)) < miss_thresh:
                    ######################################################
                    # Call idclouds subroutine
                    if cloudidmethod == "label_grow":
                        clouddata = label_and_grow_cold_clouds(
                            out_ir,
                            pixel_radius,
                            cloudtb_threshs,
                            area_thresh,
                            mincoldcorepix,
                            smoothwindowdimensions,
                            warmanvilexpansion,
                        )
                    elif cloudidmethod == "futyan3":
                        clouddata = futyan3(
                            out_ir,
                            pixel_radius,
                            cloudtb_threshs,
                            area_thresh,
                            warmanvilexpansion,
                        )
                    else:
                        logger.critical(f"ERROR: Unknown cloudidmethod: {cloudidmethod}")
                        logger.critical("Tracking will now exit.")
                        sys.exit()

                    ######################################################
                    # Separate output into the separate variables
                    final_nclouds = np.array([clouddata["final_nclouds"]])
                    final_ncorepix = clouddata["final_ncorepix"]
                    final_ncoldpix = clouddata["final_ncoldpix"]
                    final_ncorecoldpix = clouddata["final_ncorecoldpix"]
                    final_nwarmpix = clouddata["final_nwarmpix"]
                    final_cloudtype = np.array([clouddata["final_cloudtype"]])
                    final_cloudnumber = np.array([clouddata["final_cloudnumber"]])
                    final_convcold_cloudnumber = np.array(
                        [clouddata["final_convcold_cloudnumber"]]
                    )

                    # Option to linkpf
                    if linkpf == 1:

                        # Proceed if there is at least 1 cloud
                        if final_nclouds > 0:

                            # Convert precipitation factor to unit [mm/hour]
                            pcp = rawdata[pcp_varname].data * pcp_convert_factor

                            # For 'gpmirimerg', precipitation is averaged to 1-hourly
                            # and put in first time dimension
                            if clouddatasource == "gpmirimerg":
                                pcp = pcp[0, :, :]
                            else:
                                # For other data source take the same time as tb
                                pcp = pcp[tt, :, :]

                            # Run SL3D algorithm for 3D reflectivity data
                            if "radar3d" in feature_type:
                                sl3d_dict, sl3d_attrs = run_sl3d(rawdata, config)
                                # Set linkpf variable
                                pcp_linkpf = sl3d_dict[linkpf_varname]
                            else:
                                # Use precipitation as linkpf variable
                                pcp_linkpf = pcp                          

                            # Replace values <=0 with 0 before smoothing
                            pcp_linkpf[pcp_linkpf <= 0] = 0
                            # Smooth pcp_linkpf using convolve filter (handles NaN)
                            kernel = Box2DKernel(pf_smooth_window)
                            pcp_s = convolve(
                                np.squeeze(pcp_linkpf), kernel, 
                                boundary="extend", nan_treatment="interpolate", preserve_nan=True,
                            )
                            # Smooth PF variable, then label PF exceeding threshold
                            # pcp_s = filters.uniform_filter(
                            #     np.squeeze(pcp_linkpf),
                            #     size=pf_smooth_window,
                            #     mode="nearest",
                            # )
                            pf_number, npf = label(pcp_s >= pf_dbz_thresh)

                            # Convert PF area threshold to number of pixels
                            min_npix = np.ceil(
                                pf_link_area_thresh / (pixel_radius ** 2)
                            ).astype(int)

                            # Sort and renumber PFs, and remove small PFs
                            pf_number, pf_npix = sort_renumber(pf_number, min_npix)

                            # Call function to link clouds with PFs
                            pf_convcold_cloudnumber, pf_cloudnumber = link_pf_tb(
                                np.squeeze(final_convcold_cloudnumber),
                                np.squeeze(final_cloudnumber),
                                pf_number,
                                out_ir,
                                thresh_cloud,
                            )

                            # Sort and renumber the linkpf clouds
                            # (removes small clouds after renumbering)
                            (
                                pf_convcold_cloudnumber_sorted,
                                pf_cloudnumber_sorted,
                                npix_convcold_linkpf,
                            ) = sort_renumber2vars(
                                pf_convcold_cloudnumber,
                                pf_cloudnumber,
                                area_thresh / pixel_radius ** 2,
                            )
                            # Get number of clouds from the sorted linkpf clouds
                            nclouds_linkpf = np.nanmax(
                                pf_convcold_cloudnumber_sorted
                            )

                            # Make a copy of the original arrays
                            final_cloudnumber_orig = final_cloudnumber
                            final_convcold_cloudnumber_orig = (
                                final_convcold_cloudnumber
                            )

                            # Update output arrays
                            final_cloudnumber = np.expand_dims(
                                pf_cloudnumber_sorted, axis=0
                            )
                            final_convcold_cloudnumber = np.expand_dims(
                                pf_convcold_cloudnumber_sorted, axis=0
                            )
                            final_nclouds = np.array([nclouds_linkpf], dtype=int)
                            final_pf_number = np.expand_dims(pf_number, axis=0)
                            final_ncorecoldpix = npix_convcold_linkpf
                            if pcp.ndim == 2:
                                final_pcp = np.expand_dims(pcp, axis=0)
                            else:
                                final_pcp = pcp

                        else:
                            # Create default arrays
                            final_pcp = np.full(
                                final_convcold_cloudnumber.shape,
                                np.nan,
                                dtype=float,
                            )
                            final_pf_number = np.full(
                                final_convcold_cloudnumber.shape, 0, dtype=int
                            )
                            # Make a copy of the original arrays
                            final_cloudnumber_orig = final_cloudnumber
                            final_convcold_cloudnumber_orig = (
                                final_convcold_cloudnumber
                            )
                    else:
                        # Create default arrays
                        final_pcp = np.full(
                            final_convcold_cloudnumber.shape, np.nan, dtype=float
                        )
                        final_pf_number = np.full(
                            final_convcold_cloudnumber.shape, 0, dtype=int
                        )
                        # Make a copy of the original arrays
                        final_cloudnumber_orig = final_cloudnumber
                        final_convcold_cloudnumber_orig = final_convcold_cloudnumber

                    #######################################################
                    # output data to netcdf file, only if clouds present
                    if final_nclouds > 0:
                        # Output filename
                        cloudid_outfile = (
                            tracking_outpath +
                            cloudid_filebase +
                            file_datestring +
                            "_" +
                            file_timestring +
                            ".nc"
                        )

                        # Delete file if it already exists
                        if os.path.isfile(cloudid_outfile):
                            os.remove(cloudid_outfile)

                        # Write output to netCDF file
                        net.write_cloudid_tb(
                            cloudid_outfile,
                            file_basetime,
                            file_datestring,
                            file_timestring,
                            in_lat,
                            in_lon,
                            out_ir,
                            final_cloudtype,
                            final_convcold_cloudnumber,
                            final_cloudnumber,
                            final_nclouds,
                            final_ncorecoldpix,
                            cloudtb_threshs,
                            config,
                            precipitation=final_pcp,
                            pf_number=final_pf_number,
                            convcold_cloudnumber_orig=final_convcold_cloudnumber_orig,
                            cloudnumber_orig=final_cloudnumber_orig,
                            linkpf=linkpf,
                            pf_smooth_window=pf_smooth_window,
                            pf_dbz_thresh=pf_dbz_thresh,
                            pf_link_area_thresh=pf_link_area_thresh,
                            sl3d_dict=sl3d_dict,
                            sl3d_attrs=sl3d_attrs,
                        )
                        logger.info(f"{cloudid_outfile}")

                    else:
                        logger.info(filename)
                        logger.info("No clouds")

                else:
                    logger.info(filename)
                    logger.info("Too much missing data")
            else:
                logger.info(filename)
                logger.info(
                    "No data within specified geolimit range."
                )
    return cloudid_outfile