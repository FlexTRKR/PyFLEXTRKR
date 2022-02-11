import numpy as np
import time
import os
import logging
import xarray as xr

def map_generic(
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
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        tracksmap_outfile: string
            Track number pixel-level file name.
    """
    feature_varname = config.get("feature_varname", "feature_number")
    nfeature_varname = config.get("nfeature_varname", "nfeatures")
    # field_varname = config.get("field_varname")
    fillval = config.get("fillval", -9999)

    np.set_printoptions(threshold=np.inf)
    logger = logging.getLogger(__name__)

    # ###################################################################
    # # Load track stats file
    # statistics_file = (
    #     config["stats_outpath"] +
    #     config["trackstats_filebase"] +
    #     config["startdate"] + "_" +
    #     config["enddate"] + ".nc"
    # )
    # allstatdata = xr.open_dataset(statistics_file,
    #                               decode_times=False,
    #                               mask_and_scale=False)
    # trackstat_basetime = allstatdata["base_time"].values
    # trackstat_cloudnumber = allstatdata["cloudnumber"].values
    # trackstat_status = allstatdata["track_status"].values
    # track_status_explanation = allstatdata['track_status'].comments
    # trackstat_mergenumbers = allstatdata["merge_tracknumbers"].values
    # trackstat_splitnumbers = allstatdata["split_tracknumbers"].values
    # allstatdata.close()

    #########################################################################
    # Get cloudid file associated with this time
    file_datetime = time.strftime("%Y%m%d_%H%M", time.gmtime(np.copy(filebasetime)))
    filedate = np.copy(file_datetime[0:8])
    filetime = np.copy(file_datetime[9:14])

    # Load cloudid data
    cloudiddata = xr.open_dataset(cloudid_filename,
                                  decode_times=False,
                                  mask_and_scale=False)
    feature_number = cloudiddata[feature_varname].values
    nfeatures = cloudiddata[nfeature_varname].values
    cloudid_basetime = cloudiddata["base_time"].values
    basetime_units = cloudiddata["base_time"].units
    longitude = cloudiddata["longitude"].values
    latitude = cloudiddata["latitude"].values
    # fvar = cloudiddata[field_varname].values
    cloudiddata.close()

    # Get data dimensions
    [timeindex, ny, nx] = np.shape(feature_number)

    ################################################################
    # Create map of status and track number for every feature in this file
    statusmap = np.full((1, ny, nx), fillval, dtype=int)
    trackmap = np.zeros((1, ny, nx), dtype=int)
    allmergemap = np.zeros((1, ny, nx), dtype=int)
    allsplitmap = np.zeros((1, ny, nx), dtype=int)

    # Find matching time from the trackstats_basetime
    # itrack, itime = np.array(np.where(trackstat_basetime == cloudid_basetime))
    # If a match is found, that means there are tracked cells at this time
    # Proceed and label them

    itrack = file_trackindex
    nmatchcloud = len(file_cloudnumber)
    if nmatchcloud > 0:
        ##############################################################
        # Loop over each instance matching the trackstats time
        for jj in range(0, nmatchcloud):
            # Get cloud number
            # jjcloudnumber = trackstat_cloudnumber[itrack[jj], itime[jj]]
            # jjstatus = trackstat_status[itrack[jj], itime[jj]]
            jjcloudnumber = file_cloudnumber[jj]
            jjstatus = file_trackstatus[jj]

            # Find pixels matching this cloud number
            jjcloudypixels, jjcloudxpixels = np.array(
                np.where(feature_number[0, :, :] == jjcloudnumber)
            )
            # Label this cloud with the track number.
            # Need to add one to the cloud number since have the index number and we want the track number
            if len(jjcloudypixels) > 0:
                trackmap[0, jjcloudypixels, jjcloudxpixels] = itrack[jj] + 1
                statusmap[0, jjcloudypixels, jjcloudxpixels] = jjstatus
            else:
                logger.warning(f"Warning: No matching cloud pixel found: {jjcloudnumber}")

            # Get cloudnumbers and split cloudnumbers within this time
            # jjcloudnumber = trackstat_cloudnumber[itrack, itime]
            # jjallsplit = trackstat_splitnumbers[itrack, itime]
            jjcloudnumber = file_cloudnumber[jj]
            jjallsplit = file_splitcloudnumber[jj]
            # Count valid split cloudnumbers (> 0)
            splitpresent = np.count_nonzero(jjallsplit > 0)
            if splitpresent > 0:
                # Find valid split cloudnumbers (> 0)
                splittracks = jjallsplit[jjallsplit > 0]
                splitcloudid = jjcloudnumber[jjallsplit > 0]
                if len(splittracks) > 0:
                    for isplit in range(0, len(splittracks)):
                        splitypixels, splitxpixels = np.array(
                            np.where(feature_number[0, :, :] == splitcloudid[isplit])
                        )
                        allsplitmap[0, splitypixels, splitxpixels] = splittracks[isplit]

            # Get cloudnumbers and merg cloudnumbers within this time
            # jjallmerge = trackstat_mergenumbers[itrack, itime]
            jjallmerge = file_mergecloudnumber[jj]
            # Count valid split cloudnumbers (> 0)
            mergepresent = np.count_nonzero(jjallmerge > 0)
            if mergepresent > 0:
                # Find valid merge cloudnumbers (> 0)
                mergetracks = jjallmerge[jjallmerge > 0]
                mergecloudid = jjcloudnumber[jjallmerge > 0]
                if len(mergetracks) > 0:
                    for imerge in range(0, len(mergetracks)):
                        mergeypixels, mergexpixels = np.array(
                            np.where(feature_number[0, :, :] == mergecloudid[imerge])
                        )
                        allmergemap[0, mergeypixels, mergexpixels] = mergetracks[imerge]

        # trackmap = trackmap.astype(np.int32)
        # allmergemap = allmergemap.astype(np.int32)
        # allsplitmap = allsplitmap.astype(np.int32)


    #####################################################################
    # Output to netcdf file

    # Define output fileame
    tracksmap_outfile = (
        config["pixeltracking_outpath"] +
        config["pixeltracking_filebase"] +
        str(filedate) + "_" +
        str(filetime) + ".nc"
    )

    # Delete file if it already exists
    if os.path.isfile(tracksmap_outfile):
        os.remove(tracksmap_outfile)

    # Define variable dictionary
    var_dict = {
        "base_time": (["time"], cloudid_basetime),
        "longitude": (["lat", "lon"], longitude),
        "latitude": (["lat", "lon"], latitude),
        "nfeatures": (["time"], nfeatures),
        "tracknumber": (["time", "lat", "lon"], trackmap),
        "track_status": (["time", "lat", "lon"], statusmap),
        feature_varname: (["time", "lat", "lon"], feature_number),
        "merge_tracknumber": (["time", "lat", "lon"], allmergemap),
        "split_tracknumber": (["time", "lat", "lon"], allsplitmap),
    }
    # Define coordinate dictionary
    coord_dict = {
        "time": (["time"], cloudid_basetime),
        "lat": (["lat"], np.arange(0, ny)),
        "lon": (["lon"], np.arange(0, nx)),
    }
    # Define global attributes
    gattr_dict = {
        "title": "Pixel-level feature tracking data",
        "contact": "Zhe Feng, zhe.feng@pnnl.gov",
        "created_on": time.ctime(time.time()),
    }
    # Define xarray dataset
    ds_out = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

    # Specify variable attributes
    ds_out["time"].attrs["long_name"] = "Base time in Epoch"
    ds_out["time"].attrs["units"] = basetime_units

    ds_out["base_time"].attrs["long_name"] = "Base time in Epoch"
    ds_out["base_time"].attrs["units"] = basetime_units

    ds_out["longitude"].attrs["long_name"] = "Grid of longitude"
    ds_out["longitude"].attrs["units"] = "degrees"

    ds_out["latitude"].attrs["long_name"] = "Grid of latitude"
    ds_out["latitude"].attrs["units"] = "degrees"

    ds_out["nfeatures"].attrs["long_name"] = "Number of features labeled"
    ds_out["nfeatures"].attrs["units"] = "unitless"

    ds_out["tracknumber"].attrs["long_name"] = "Track number in this file at a given pixel"
    ds_out["tracknumber"].attrs["units"] = "unitless"
    ds_out["tracknumber"].attrs["_FillValue"] = 0

    ds_out["track_status"].attrs["long_name"] = "Flag indicating history of track"
    ds_out["track_status"].attrs["units"] = "unitless"
    ds_out["track_status"].attrs["_FillValue"] = fillval
    ds_out["track_status"].attrs["comments"] = trackstats_comments

    ds_out[feature_varname].attrs["long_name"] = "Labeled feature number"
    ds_out[feature_varname].attrs["units"] = "unitless"
    ds_out[feature_varname].attrs["_FillValue"] = 0

    ds_out["merge_tracknumber"].attrs[
        "long_name"
    ] = "Tracknumber where this track merges with"
    ds_out["merge_tracknumber"].attrs["units"] = "unitless"
    ds_out["merge_tracknumber"].attrs["_FillValue"] = 0

    ds_out["split_tracknumber"].attrs[
        "long_name"
    ] = "Tracknumber where this track splits from"
    ds_out["split_tracknumber"].attrs["units"] = "unitless"
    ds_out["split_tracknumber"].attrs["_FillValue"] = 0

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