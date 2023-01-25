import numpy as np
import time
import os
import logging
import xarray as xr

def map_feature(
    cloudid_filename,
    filebasetime,
    file_trackindex,
    file_cloudnumber,
    file_trackstatus,
    file_mergecloudnumber,
    file_splitcloudnumber,
    trackstats_comments,
    config,
):
    """
    Map track numbers to pixel level files for generic feature tracking.

    Args:
        cloudid_filename: string
            Cloudid file name.
        filebasetime: int
            Cloudid file base time.
        file_trackindex: np.array
            Track indices for the features in the cloudid file.
        file_cloudnumber: np.array
            Matched feature numbers in the cloudid file.
        file_trackstatus: np.array
            Track status for the features in the cloudid file.
        file_mergecloudnumber: np.array
            Merge feature track number in the cloudid file.
        file_splitcloudnumber: np.array
            Split feature track number in the cloudid file.
        trackstats_comments: string
            Track status explanation.
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        tracksmap_outfile: string
            Track number pixel-level file name.
    """
    feature_varname = config.get("feature_varname", "feature_number")
    # nfeature_varname = config.get("nfeature_varname", "nfeatures")
    time_dimname = "time"
    y_dimname = "lat"
    x_dimname = "lon"
    fillval = config.get("fillval", -9999)

    np.set_printoptions(threshold=np.inf)
    logger = logging.getLogger(__name__)

    #########################################################################
    # Get cloudid file associated with this time
    file_datetime = time.strftime("%Y%m%d_%H%M", time.gmtime(np.copy(filebasetime)))
    # Load cloudid data
    ds_in = xr.open_dataset(
        cloudid_filename,
        decode_times=False,
        mask_and_scale=False
    )
    # Get dimension names from the file
    dims_file = []
    for key in ds_in.dims: dims_file.append(key)
    # Find extra dimensions beyond [time, y, x]
    dims_keep = [time_dimname, y_dimname, x_dimname]
    dims_drop = list(set(dims_file) - set(dims_keep))
    # Drop extra dimensions
    ds_in = ds_in.drop_dims(dims_drop)
    # Get necessary variables
    feature_number = ds_in[feature_varname].data
    # nfeatures = ds_in[nfeature_varname].data
    # cloudid_basetime = ds_in["base_time"].data
    # basetime_units = ds_in["base_time"].units
    # longitude = ds_in["longitude"].data
    # latitude = ds_in["latitude"].data
    # fvar = ds_in[field_varname].data
    ds_in.close()

    # Get data dimensions
    ny = ds_in.dims[y_dimname]
    nx = ds_in.dims[x_dimname]
    # [timeindex, ny, nx] = np.shape(feature_number)

    ################################################################
    # Create map of status and track number for every feature in this file
    statusmap = np.full((1, ny, nx), fillval, dtype=int)
    trackmap = np.zeros((1, ny, nx), dtype=int)
    allmergemap = np.zeros((1, ny, nx), dtype=int)
    allsplitmap = np.zeros((1, ny, nx), dtype=int)

    # Check number of matched features
    nmatchcloud = len(file_cloudnumber)
    if nmatchcloud > 0:
        ##############################################################
        # Loop over each instance matching the trackstats time
        for jj in range(0, nmatchcloud):
            # Get cloud number
            jjcloudnumber = file_cloudnumber[jj]
            jjstatus = file_trackstatus[jj]

            # Get the mask matching this feature number
            cmask = feature_number.squeeze() == jjcloudnumber

            # Label this feature with the track number.
            # Need to add one to the cloud number since have the index number and we want the track number
            if np.count_nonzero(cmask) > 0:
                trackmap[0, cmask] = file_trackindex[jj] + 1
                statusmap[0, cmask] = jjstatus
            else:
                logger.warning(f"Warning: No matching cloud pixel found: {jjcloudnumber}")

            # Get split cloudnumber within this time
            jjallsplit = file_splitcloudnumber[jj]
            # Count valid split cloudnumbers (> 0)
            splitpresent = np.count_nonzero(jjallsplit > 0)
            if splitpresent > 0:
                # Find valid split cloudnumbers (> 0)
                splittracks = jjallsplit[jjallsplit > 0]
                splitcloudid = jjcloudnumber[jjallsplit > 0]
                if len(splittracks) > 0:
                    for isplit in range(0, len(splittracks)):
                        s_cmask = feature_number.squeeze() == splitcloudid[isplit]
                        allsplitmap[0, s_cmask] = splittracks[isplit]

            # Get merge cloudnumber within this time
            jjallmerge = file_mergecloudnumber[jj]
            # Count valid split cloudnumbers (> 0)
            mergepresent = np.count_nonzero(jjallmerge > 0)
            if mergepresent > 0:
                # Find valid merge cloudnumbers (> 0)
                mergetracks = jjallmerge[jjallmerge > 0]
                mergecloudid = jjcloudnumber[jjallmerge > 0]
                if len(mergetracks) > 0:
                    for imerge in range(0, len(mergetracks)):
                        m_cmask = feature_number.squeeze() == mergecloudid[imerge]
                        allmergemap[0, m_cmask] = mergetracks[imerge]


    #####################################################################
    # Add track number variables to output dataset

    # Make a dictionary new variable coordinates
    coords = {
        time_dimname: ([time_dimname], ds_in[time_dimname].data, ds_in[time_dimname].attrs),
        y_dimname: ([y_dimname], ds_in[y_dimname].data, ds_in[y_dimname].attrs),
        x_dimname: ([x_dimname], ds_in[x_dimname].data, ds_in[x_dimname].attrs),
    }
    # Make attributes for new variables
    trackmap_attrs = {
        "long_name": "Track number in this file at a given pixel",
        "units": "unitless",
        "_FillValue": 0,
    }
    allmergemap_attrs = {
        "long_name": "Tracknumber where this track merges with",
        "units": "unitless",
        "_FillValue": 0,
    }
    allsplitmap_attrs = {
        "long_name": "Tracknumber where this track splits from",
        "units": "unitless",
        "_FillValue": 0,
    }
    statusmap_attrs = {
        "long_name": "Flag indicating history of track",
        "units": "unitless",
        "_FillValue": fillval,
        "comments": trackstats_comments,
    }
    # Convert numpy arrays to Xarray DataArrays
    trackmap_xr = xr.DataArray(trackmap, coords=coords, dims=dims_keep, attrs=trackmap_attrs)
    allmergemap_xr = xr.DataArray(allmergemap, coords=coords, dims=dims_keep, attrs=allmergemap_attrs)
    allsplitmap_xr = xr.DataArray(allsplitmap, coords=coords, dims=dims_keep, attrs=allsplitmap_attrs)
    # statusmap_xr = xr.DataArray(statusmap, coords=coords, dims=dims_keep, attrs=statusmap_attrs)

    # Create a copy of the input dataset
    ds_out = ds_in.copy(deep=True)

    # Assign new variables to output dataset
    ds_out = ds_out.assign(tracknumber=trackmap_xr)
    ds_out = ds_out.assign(merge_tracknumber=allmergemap_xr)
    ds_out = ds_out.assign(split_tracknumber=allsplitmap_xr)
    # ds_out = ds_out.assign(track_status=statusmap_xr)

    # Update global attributes
    ds_out.attrs["Title"] = "Pixel-level feature tracking data"
    ds_out.attrs["Created_on"] = time.ctime(time.time())

    #####################################################################
    # Output to netcdf file

    # Define output filename
    tracksmap_outfile = (
        config["pixeltracking_outpath"] +
        config["pixeltracking_filebase"] +
        file_datetime + ".nc"
    )

    # Delete file if it already exists
    if os.path.isfile(tracksmap_outfile):
        os.remove(tracksmap_outfile)

    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in ds_out.data_vars}
    # Write to netCDF file
    ds_out.to_netcdf(
        path=tracksmap_outfile,
        mode="w",
        format="NETCDF4",
        unlimited_dims="time",
        encoding=encoding,
    )
    logger.info(f"{tracksmap_outfile}")

    return tracksmap_outfile