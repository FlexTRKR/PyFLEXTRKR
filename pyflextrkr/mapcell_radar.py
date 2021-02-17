# Purpose: Take the cell tracks identified in the previous steps and create pixel level maps of these cells. One netcdf file is create for each time step.

# Author: Zhe Feng (zhe.feng@pnnl.gov)

# def mapcell_radar(zipped_inputs):
def mapcell_radar(
    cloudid_filename,
    filebasetime,
    stats_path,
    statistics_filebase,
    startdate,
    enddate,
    out_path,
    out_filebase,
):
    # Inputs:
    # cloudid_filebase - file header of the cloudid file create in the first step
    # filebasetime - seconds since 1970-01-01 of the file being processed
    # statistics_filebase - file header for the all track statistics file generated in the trackstats step
    # out_path - directory where cell tracks maps generated in this step will be placed
    # stats_path - directory that contains the statistics files
    # startdate - starting date and time of the full dataset
    # enddate - ending date and time of the full dataset

    #######################################################################
    # Import modules
    import numpy as np
    import time
    import os
    import xarray as xr
    from netCDF4 import Dataset

    np.set_printoptions(threshold=np.inf)

    ######################################################################
    # define constants

    ###################################################################
    # Load track stats file
    statistics_file = (
        stats_path + statistics_filebase + startdate + "_" + enddate + ".nc"
    )

    allstatdata = Dataset(statistics_file, "r")
    # Time of cloud in seconds since 01/01/1970 00:00
    trackstat_basetime = allstatdata["basetime"][:]
    # Number of the corresponding cloudid file
    trackstat_cloudnumber = allstatdata["cloudnumber"][:]
    # Flag indicating the status of the cloud
    trackstat_status = allstatdata["status"][:]
    # Track number that it merges into
    trackstat_mergenumbers = allstatdata["merge_tracknumbers"][:]  
    trackstat_splitnumbers = allstatdata["split_tracknumbers"][:]
    datasource = allstatdata.getncattr("source")
    datadescription = allstatdata.getncattr("description")
    allstatdata.close()

    #########################################################################
    # Get cloudid file associated with this time
    file_datetime = time.strftime("%Y%m%d_%H%M", time.gmtime(np.copy(filebasetime)))
    filedate = np.copy(file_datetime[0:8])
    filetime = np.copy(file_datetime[9:14])
    # print(('cloudid file: ' + cloudid_filename))

    # Load cloudid data
    cloudiddata = Dataset(cloudid_filename, "r")
    cloudid_cloudnumber = cloudiddata["cloudnumber"][:]
    # cloudid_cloudnumber_noinflate = cloudiddata['cloudnumber_noinflate'][:]
    cloudid_basetime = cloudiddata["basetime"][:]
    basetime_units = cloudiddata["basetime"].units
    # basetime_calendar = cloudiddata['basetime'].calendar
    longitude = cloudiddata["longitude"][:]
    latitude = cloudiddata["latitude"][:]
    nclouds = cloudiddata["nclouds"][:]
    comp_ref = cloudiddata["comp_ref"][:]
    dbz_lowlevel = cloudiddata["dbz_lowlevel"][:]
    conv_core = cloudiddata["conv_core"][:]
    conv_mask = cloudiddata["conv_mask"][:]
    # Convert ETH units from [m] to [km]
    echotop10 = cloudiddata["echotop10"][:] / 1000.0
    echotop20 = cloudiddata["echotop20"][:] / 1000.0
    echotop30 = cloudiddata["echotop30"][:] / 1000.0
    echotop40 = cloudiddata["echotop40"][:] / 1000.0
    echotop50 = cloudiddata["echotop50"][:] / 1000.0
    # convcold_cloudnumber = cloudiddata['convcold_cloudnumber'][:]
    cloudiddata.close()

    # cloudid_cloudnumber = cloudid_cloudnumber.astype(np.int32)
    # cloudid_cloudnumber_noinflate = cloudid_cloudnumber_noinflate.astype(np.int32)
    comp_ref = comp_ref.data
    conv_core = conv_core.data
    conv_mask = conv_mask.data
    cloudid_cloudnumber = cloudid_cloudnumber.data
    # cloudid_cloudnumber_noinflate = cloudid_cloudnumber_noinflate.data

    # Create a binary conv_mask (remove the cell number)
    conv_mask_binary = conv_mask > 0

    # Get data dimensions
    [timeindex, ny, nx] = np.shape(cloudid_cloudnumber)

    ##############################################################
    # Intiailize track maps
    # celltrackmap = np.zeros((1, ny, nx), dtype=int)
    # celltrackmap_mergesplit = np.zeros((1, ny, nx), dtype=int)

    # cellmergemap = np.zeros((1, ny, nx), dtype=int)
    # cellsplitmap = np.zeros((1, ny, nx), dtype=int)

    ################################################################
    # Create map of status and track number for every feature in this file
    print("Create maps of all tracks")
    fillval = -9999
    statusmap = np.ones((1, ny, nx), dtype=int) * fillval
    trackmap = np.zeros((1, ny, nx), dtype=int)
    allmergemap = np.zeros((1, ny, nx), dtype=int)
    allsplitmap = np.zeros((1, ny, nx), dtype=int)

    # Find matching time from the trackstats_basetime
    itrack, itime = np.array(np.where(trackstat_basetime == cloudid_basetime))
    # If a match is found, that means there are tracked cells at this time
    # Proceed and lebel them
    ntimes = len(itime)
    if ntimes > 0:

        # Loop over each instance matching the trackstats time
        for jj in range(0, ntimes):
            # Get cloud number
            jjcloudnumber = trackstat_cloudnumber[itrack[jj], itime[jj]]
            jjstatus = trackstat_status[itrack[jj], itime[jj]]

            # Find pixels matching this cloud number
            jjcloudypixels, jjcloudxpixels = np.array(
                np.where(cloudid_cloudnumber[0, :, :] == jjcloudnumber)
            )
            # Label this cloud with the track number.
            # Need to add one to the cloud number since have the index number and we want the track number
            if len(jjcloudypixels) > 0:
                trackmap[0, jjcloudypixels, jjcloudxpixels] = itrack[jj] + 1
                statusmap[0, jjcloudypixels, jjcloudxpixels] = jjstatus
                # import pdb; pdb.set_trace()
            else:
                print(
                    "Error: No matching cloud pixel found?! itrack: ",
                    itrack[jj],
                    ", itime: ",
                    itime[jj],
                )
                # sys.exit('Error: No matching cloud pixel found?!')

        # Get cloudnumbers and split cloudnumbers within this time
        jjcloudnumber = trackstat_cloudnumber[itrack, itime]
        jjallsplit = trackstat_splitnumbers[itrack, itime]
        # Count valid split cloudnumbers (> 0)
        splitpresent = np.count_nonzero(jjallsplit > 0)
        # splitpresent = len(np.array(np.where(np.isfinite(jjallsplit)))[0, :])
        if splitpresent > 0:
            # splittracks = np.copy(jjallsplit[np.where(np.isfinite(jjallsplit))])
            # splitcloudid = np.copy(jjcloudnumber[np.where(np.isfinite(jjallsplit))])
            # Find valid split cloudnumbers (> 0)
            splittracks = jjallsplit[np.where(jjallsplit > 0)]
            splitcloudid = jjcloudnumber[np.where(jjallsplit > 0)]
            if len(splittracks) > 0:
                for isplit in range(0, len(splittracks)):
                    splitypixels, splitxpixels = np.array(
                        np.where(cloudid_cloudnumber[0, :, :] == splitcloudid[isplit])
                    )
                    allsplitmap[0, splitypixels, splitxpixels] = splittracks[isplit]

        # Get cloudnumbers and merg cloudnumbers within this time
        jjallmerge = trackstat_mergenumbers[itrack, itime]
        # Count valid split cloudnumbers (> 0)
        mergepresent = np.count_nonzero(jjallmerge > 0)
        # mergepresent = len(np.array(np.where(np.isfinite(jjallmerge)))[0, :])
        if mergepresent > 0:
            # mergetracks = np.copy(jjallmerge[np.where(np.isfinite(jjallmerge))])
            # mergecloudid = np.copy(jjcloudnumber[np.where(np.isfinite(jjallmerge))])
            # Find valid merge cloudnumbers (> 0)
            mergetracks = jjallmerge[np.where(jjallmerge > 0)]
            mergecloudid = jjcloudnumber[np.where(jjallmerge > 0)]
            if len(mergetracks) > 0:
                for imerge in range(0, len(mergetracks)):
                    mergeypixels, mergexpixels = np.array(
                        np.where(cloudid_cloudnumber[0, :, :] == mergecloudid[imerge])
                    )
                    allmergemap[0, mergeypixels, mergexpixels] = mergetracks[imerge]

        trackmap = trackmap.astype(np.int32)
        allmergemap = allmergemap.astype(np.int32)
        allsplitmap = allsplitmap.astype(np.int32)

        # Multiply the tracknumber map with conv_mask to get the actual cell size without inflation
        # trackmap_cmask2 = (trackmap * conv_mask2).astype(np.int32)
        trackmap_cmask2 = (trackmap * conv_mask_binary).astype(np.int32)

    else:
        trackmap_cmask2 = trackmap

    #####################################################################
    # Output maps to netcdf file

    # Define output fileame
    celltrackmaps_outfile = (
        out_path + out_filebase + str(filedate) + "_" + str(filetime) + ".nc"
    )

    # Check if file already exists. If exists, delete
    if os.path.isfile(celltrackmaps_outfile):
        os.remove(celltrackmaps_outfile)

    # Define variable list
    varlist = {
        "basetime": (
            ["time"],
            cloudid_basetime,
        ),
        "longitude": (["lat", "lon"], longitude),
        "latitude": (["lat", "lon"], latitude),
        "nclouds": (["time"], nclouds),
        "comp_ref": (["time", "lat", "lon"], comp_ref),
        "dbz_lowlevel": (["time", "lat", "lon"], dbz_lowlevel),
        "conv_core": (["time", "lat", "lon"], conv_core),
        "conv_mask": (["time", "lat", "lon"], conv_mask),
        "tracknumber": (["time", "lat", "lon"], trackmap),
        "tracknumber_cmask2": (["time", "lat", "lon"], trackmap_cmask2),
        "track_status": (["time", "lat", "lon"], statusmap),
        "cloudnumber": (
            ["time", "lat", "lon"],
            cloudid_cloudnumber,
        ),
        "merge_tracknumber": (["time", "lat", "lon"], allmergemap),
        "split_tracknumber": (["time", "lat", "lon"], allsplitmap),
        "echotop10": (["time", "lat", "lon"], echotop10),
        "echotop20": (["time", "lat", "lon"], echotop20),
        "echotop30": (["time", "lat", "lon"], echotop30),
        "echotop40": (["time", "lat", "lon"], echotop40),
        "echotop50": (
            ["time", "lat", "lon"],
            echotop50,
        ),  # 'tracknumber': (['time', 'lat', 'lon'], trackmap_mergesplit), \
        # 'cellsplittracknumbers': (['time', 'lat', 'lon'], cellsplitmap), \
        # 'cellmergetracknumbers': (['time', 'lat', 'lon'], cellmergemap), \
    }

    # Define coordinate list
    coordlist = {
        "time": (["time"], cloudid_basetime),
        "lat": (["lat"], np.arange(0, ny)),
        "lon": (["lon"], np.arange(0, nx)),
    }

    # Define global attributes
    gattrlist = {
        "title": "Pixel level of tracked cells",
        "source": datasource,
        "description": datadescription,  # 'Main_cell_duration_hr': durationthresh, \
        # 'Merger_duration_hr': mergethresh, \
        # 'Split_duration_hr': splitthresh, \
        "contact": "Zhe Feng, zhe.feng@pnnl.gov",
        "created_on": time.ctime(time.time()),
    }

    # Define xarray dataset
    ds_out = xr.Dataset(varlist, coords=coordlist, attrs=gattrlist)

    # Track status explanation
    track_status_explanation = (
        "0: Track stops;  "
        + "1: Simple track continuation;  "
        + "2: This is the bigger cloud in simple merger;  "
        + "3: This is the bigger cloud from a simple split that stops at this time;  "
        + "4: This is the bigger cloud from a split and this cloud continues to the next time;  "
        + "5: This is the bigger cloud from a split that subsequently is the big cloud in a merger;  "
        + "13: This cloud splits at the next time step;  "
        + "15: This cloud is the bigger cloud in a merge that then splits at the next time step;  "
        + "16: This is the bigger cloud in a split that then splits at the next time step;  "
        + "18: Merge-split at same time (big merge, splitter, and big split);  "
        + "21: This is the smaller cloud in a simple merger;  "
        + "24: This is the bigger cloud of a split that is then the small cloud in a merger;  "
        + "31: This is the smaller cloud in a simple split that stops;  "
        + "32: This is a small split that continues onto the next time step;  "
        + "33: This is a small split that then is the bigger cloud in a merger;  "
        + "34: This is the small cloud in a merger that then splits at the next time step;  "
        + "37: Merge-split at same time (small merge, splitter, big split);  "
        + "44: This is the smaller cloud in a split that is smaller cloud in a merger at the next time step;  "
        + "46: Merge-split at same time (big merge, splitter, small split);  "
        + "52: This is the smaller cloud in a split that is smaller cloud in a merger at the next time step;  "
        + "65: Merge-split at same time (smaller merge, splitter, small split)"
    )

    # Specify variable attributes
    ds_out.time.attrs[
        "long_name"
    ] = "epoch time (seconds since 01/01/1970 00:00) in epoch of file"
    ds_out.time.attrs["units"] = basetime_units

    ds_out.basetime.attrs[
        "long_name"
    ] = "Epoch time (seconds since 01/01/1970 00:00) of this file"
    ds_out.basetime.attrs["units"] = basetime_units

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

    ds_out.tracknumber_cmask2.attrs[
        "long_name"
    ] = "Track number (conv_mask) in this file at a given pixel"
    ds_out.tracknumber_cmask2.attrs["units"] = "unitless"
    ds_out.tracknumber_cmask2.attrs["_FillValue"] = 0

    ds_out.track_status.attrs["long_name"] = "Flag indicating history of cloud"
    ds_out.track_status.attrs["units"] = "unitless"
    ds_out.track_status.attrs["valid_min"] = 0
    ds_out.track_status.attrs["valid_max"] = 65
    ds_out.track_status.attrs["_FillValue"] = fillval
    ds_out.track_status.attrs["comments"] = track_status_explanation

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
    print("Output celltracking file: ", celltrackmaps_outfile)

    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encodelist = {var: comp for var in ds_out.data_vars}
    encodelist["longitude"] = dict(zlib=True, dtype='float32')
    encodelist["latitude"] = dict(zlib=True, dtype='float32')

    # Write to netCDF file
    ds_out.to_netcdf(
        path=celltrackmaps_outfile,
        mode="w",
        format="NETCDF4_CLASSIC",
        unlimited_dims="time",
        encoding=encodelist,
    )
