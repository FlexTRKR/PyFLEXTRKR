import numpy as np
import time
import os
import logging
import xarray as xr

def mapcell_radar(
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
    Map track numbers to pixel level files.

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
    nfeature_varname = config.get("nfeature_varname", "nfeatures")
    ref_varname = config["ref_varname"]
    fillval = config.get("fillval", -9999)

    np.set_printoptions(threshold=np.inf)
    logger = logging.getLogger(__name__)

    #########################################################################
    # Get cloudid file associated with this time
    file_datetime = time.strftime("%Y%m%d_%H%M", time.gmtime(np.copy(filebasetime)))
    # Load cloudid data
    cloudiddata = xr.open_dataset(
        cloudid_filename,
        decode_times=False,
        mask_and_scale=False
    )
    feature_number = cloudiddata[feature_varname].values
    nclouds = cloudiddata[nfeature_varname].values
    cloudid_basetime = cloudiddata["base_time"].values
    basetime_units = cloudiddata["base_time"].units
    longitude = cloudiddata["longitude"].values
    latitude = cloudiddata["latitude"].values
    comp_ref = cloudiddata[ref_varname].values
    dbz_lowlevel = cloudiddata["dbz_lowlevel"].values
    conv_core = cloudiddata["conv_core"].values
    conv_mask = cloudiddata["conv_mask"].values
    # Convert ETH units from [m] to [km]
    echotop10 = cloudiddata["echotop10"].values / 1000.0
    echotop20 = cloudiddata["echotop20"].values / 1000.0
    echotop30 = cloudiddata["echotop30"].values / 1000.0
    echotop40 = cloudiddata["echotop40"].values / 1000.0
    echotop50 = cloudiddata["echotop50"].values / 1000.0
    # convcold_cloudnumber = cloudiddata['convcold_cloudnumber'].values
    cloudiddata.close()

    # Create a binary conv_mask (remove the cell number)
    conv_mask_binary = conv_mask > 0

    # Get data dimensions
    [timeindex, ny, nx] = np.shape(feature_number)

    ################################################################
    # Create map of status and track number for every feature in this file
    logger.info("Create maps of all tracks")
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

        trackmap = trackmap.astype(np.int32)
        allmergemap = allmergemap.astype(np.int32)
        allsplitmap = allsplitmap.astype(np.int32)

        # Multiply the tracknumber map with conv_mask to get the actual cell size without inflation
        trackmap_cmask = (trackmap * conv_mask_binary).astype(np.int32)

    else:
        trackmap_cmask = trackmap

    #####################################################################
    # Output maps to netcdf file

    # Define output fileame
    tracksmap_outfile = (
        config["pixeltracking_outpath"] +
        config["pixeltracking_filebase"] +
        file_datetime + ".nc"
    )

    # Delete file if it already exists
    if os.path.isfile(tracksmap_outfile):
        os.remove(tracksmap_outfile)

    # Define variable list
    varlist = {
        "base_time": (["time"], cloudid_basetime),
        "longitude": (["lat", "lon"], longitude),
        "latitude": (["lat", "lon"], latitude),
        "nclouds": (["time"], nclouds),
        "comp_ref": (["time", "lat", "lon"], comp_ref),
        "dbz_lowlevel": (["time", "lat", "lon"], dbz_lowlevel),
        "conv_core": (["time", "lat", "lon"], conv_core),
        "conv_mask": (["time", "lat", "lon"], conv_mask),
        "tracknumber": (["time", "lat", "lon"], trackmap),
        "tracknumber_cmask": (["time", "lat", "lon"], trackmap_cmask),
        "track_status": (["time", "lat", "lon"], statusmap),
        "cloudnumber": (["time", "lat", "lon"], feature_number),
        "merge_tracknumber": (["time", "lat", "lon"], allmergemap),
        "split_tracknumber": (["time", "lat", "lon"], allsplitmap),
        "echotop10": (["time", "lat", "lon"], echotop10),
        "echotop20": (["time", "lat", "lon"], echotop20),
        "echotop30": (["time", "lat", "lon"], echotop30),
        "echotop40": (["time", "lat", "lon"], echotop40),
        "echotop50": (["time", "lat", "lon"], echotop50),
    }

    # Define coordinate list
    coordlist = {
        "time": (["time"], cloudid_basetime),
        "lat": (["lat"], np.arange(0, ny)),
        "lon": (["lon"], np.arange(0, nx)),
    }

    # Define global attributes
    gattrlist = {
        "title": "Pixel-level cell tracking data",
        "contact": "Zhe Feng, zhe.feng@pnnl.gov",
        "created_on": time.ctime(time.time()),
    }

    # Define xarray dataset
    ds_out = xr.Dataset(varlist, coords=coordlist, attrs=gattrlist)

    # Specify variable attributes
    ds_out.time.attrs[
        "long_name"
    ] = "epoch time (seconds since 01/01/1970 00:00) in epoch of file"
    ds_out.time.attrs["units"] = basetime_units

    ds_out.base_time.attrs[
        "long_name"
    ] = "Epoch time (seconds since 01/01/1970 00:00) of this file"
    ds_out.base_time.attrs["units"] = basetime_units

    ds_out.longitude.attrs["long_name"] = "Grid of longitude"
    ds_out.longitude.attrs["units"] = "degrees"

    ds_out.latitude.attrs["long_name"] = "Grid of latitude"
    ds_out.latitude.attrs["units"] = "degrees"

    ds_out.nclouds.attrs["long_name"] = "Number of cells identified in this file"
    ds_out.nclouds.attrs["units"] = "unitless"

    ds_out.comp_ref.attrs["long_name"] = "Composite reflectivity"
    ds_out.comp_ref.attrs["units"] = "dBZ"

    ds_out.dbz_lowlevel.attrs["long_name"] = "Composite Low-level Reflectivity"
    ds_out.dbz_lowlevel.attrs["units"] = "dBZ"

    ds_out.conv_core.attrs[
        "long_name"
    ] = "Convective Core Mask After Reflectivity Threshold and Peakedness Steps"
    ds_out.conv_core.attrs["units"] = "unitless"
    ds_out.conv_core.attrs["_FillValue"] = 0

    ds_out.conv_mask.attrs[
        "long_name"
    ] = "Convective Region Mask After Reflectivity Threshold, Peakedness, and Expansion Steps"
    ds_out.conv_mask.attrs["units"] = "unitless"
    ds_out.conv_mask.attrs["_FillValue"] = 0

    ds_out.tracknumber.attrs["long_name"] = "Track number in this file at a given pixel"
    ds_out.tracknumber.attrs["units"] = "unitless"
    ds_out.tracknumber.attrs["_FillValue"] = 0

    ds_out.tracknumber_cmask.attrs[
        "long_name"
    ] = "Track number (conv_mask) in this file at a given pixel"
    ds_out.tracknumber_cmask.attrs["units"] = "unitless"
    ds_out.tracknumber_cmask.attrs["_FillValue"] = 0

    ds_out.track_status.attrs["long_name"] = "Flag indicating history of cloud"
    ds_out.track_status.attrs["units"] = "unitless"
    ds_out.track_status.attrs["valid_min"] = 0
    ds_out.track_status.attrs["valid_max"] = 65
    ds_out.track_status.attrs["_FillValue"] = fillval
    ds_out.track_status.attrs["comments"] = trackstats_comments

    ds_out.cloudnumber.attrs[
        "long_name"
    ] = "Number associated with the cloud at a given pixel"
    ds_out.cloudnumber.attrs["units"] = "unitless"
    ds_out.cloudnumber.attrs["_FillValue"] = 0
    # ds_out.cloudnumber.attrs['valid_min'] = 0
    # ds_out.cloudnumber.attrs['valid_max'] = np.nanmax(convcold_cloudnumber)

    # ds_out.cloudnumber_noinflate.attrs['long_name'] = 'Number associated with the cloud (no inflation) at a given pixel'
    # ds_out.cloudnumber_noinflate.attrs['units'] = 'unitless'
    # ds_out.cloudnumber_noinflate.attrs['_FillValue'] = 0

    ds_out.merge_tracknumber.attrs[
        "long_name"
    ] = "Tracknumber where this track merges with"
    ds_out.merge_tracknumber.attrs["units"] = "unitless"
    ds_out.merge_tracknumber.attrs["_FillValue"] = 0
    # ds_out.merge_tracknumber.attrs['valid_min'] = 0
    # ds_out.merge_tracknumber.attrs['valid_max'] = np.nanmax(celltrackmap_mergesplit)

    ds_out.split_tracknumber.attrs[
        "long_name"
    ] = "Tracknumber where this track splits from"
    ds_out.split_tracknumber.attrs["units"] = "unitless"
    ds_out.split_tracknumber.attrs["_FillValue"] = 0
    # ds_out.split_tracknumber.attrs['valid_min'] = 0
    # ds_out.split_tracknumber.attrs['valid_max'] = np.nanmax(celltrackmap_mergesplit)

    ds_out.echotop10.attrs["long_name"] = "10dBZ echo-top height"
    ds_out.echotop10.attrs["units"] = "km"

    ds_out.echotop20.attrs["long_name"] = "20dBZ echo-top height"
    ds_out.echotop20.attrs["units"] = "km"

    ds_out.echotop30.attrs["long_name"] = "30dBZ echo-top height"
    ds_out.echotop30.attrs["units"] = "km"

    ds_out.echotop40.attrs["long_name"] = "40dBZ echo-top height"
    ds_out.echotop40.attrs["units"] = "km"

    ds_out.echotop50.attrs["long_name"] = "50dBZ echo-top height"
    ds_out.echotop50.attrs["units"] = "km"

    # Write netcdf file
    logger.info(f"Output celltracking file: {tracksmap_outfile}")

    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encodelist = {var: comp for var in ds_out.data_vars}
    encodelist["longitude"] = dict(zlib=True, dtype="float32")
    encodelist["latitude"] = dict(zlib=True, dtype="float32")

    # Write to netCDF file
    ds_out.to_netcdf(
        path=tracksmap_outfile,
        mode="w",
        format="NETCDF4",
        unlimited_dims="time",
        encoding=encodelist,
    )
    return tracksmap_outfile