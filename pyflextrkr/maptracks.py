#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:56:04 2020

@author: barb672
"""

# Purpose: Take the MCS identified in the previous steps and create pixel level maps of these storms. One netcdf file is create for each time step.

# Author: Original IDL code written by Zhe Feng (zhe.feng@pnnl.gov), Python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov)


def maptracks_wrf_pf(zipped_inputs):
    # Inputs:
    # cloudid_filebase - file header of the cloudid file create in the first step
    # filebasetime - seconds since 1970-01-01 of the file being processed
    # mcsstats_filebase - file header of the robust MCS statistics file generated in the robustmcs step
    # statistics_filebase - file header for the all track statistics file generated in the trackstats step
    # rainaccumulation_filebase - file header of the rain accumulation data
    # mcstracking_path - directory where mcs maps generated in this step will be placed
    # stats_path - directory that contains the statistics files
    # rainaccumulation_path - directory containing the rain accumulation data
    # pcp_thresh - pixels with hourly precipitation larger than this will be labeled with track number
    # nmaxpf - maximum number of precipitation features that can be contained within one satellite defined MCS at a given time
    # absolutetb_thresh - range of valid brightness temperatures
    # startdate - starting date and time of the full dataset
    # enddate - ending date and time of the full dataset
    # showalltracks - flag indicating whether the output should include maps of all tracks (MCS and nonMCS). this greatly slows the code.

    # Output (One netcdf file of maps for each time step):
    # basetime - seconds since 1970-01-01 of the file being processed
    # lon - grid of analyzed longitudes
    # lat - grid of analyzed latitutdes
    # nclouds - total number of identified clouds, from the cloudid file
    # tb - map of brightness temperature
    # precipitation - map of the rain accumulations
    # mask - flag showing where valid radar data present
    # cloudtype - map of the cloud types identified in the idclouds step
    # mcssplittracknumbers - map of the clouds splitting from MCSs
    # mcsmergetracknumbers - map of the clouds merging with MCSs
    # cloudnumber - map of cloud numbers associated with each cloud that were determined in the idclouds step
    # cloudtracknumbers_nomergesplit - map of MCS track numbers, excluding clouds that merge and split from the MCSs
    # cloudtracknumber -  map of MCS track numbers, includes clouds that merge and split from the MCSs
    # pftracknumber - map of the track number associated with each precipitation feature, includes precipitation features that merge and split from the MCS
    # pcptracknumber - map of the track number associated with rain accumulation data, includes rain accumulations in merging and splitting clouds
    # alltracknumber - map of all the identified tracks (MCS and nonMCS), optional
    # allsplittracknumbers - map of the clouds splitting from all tracks (MCS and nonMCS), optional
    # allmergetracknumbers - map of the clouds merging with all tracks (MCS and nonMCS), optional
    # cloudstatus - map the status that describes the evolution of the cloud, determine in the gettracks step (optional)

    ######################################################################
    # Import modules
    import numpy as np
    import time
    import os
    import sys
    import xarray as xr
    import pandas as pd
    import time, datetime, calendar
    from netCDF4 import Dataset, num2date

    np.set_printoptions(threshold=np.inf)

    # Separate inputs
    cloudid_filename = zipped_inputs[0]
    filebasetime = zipped_inputs[1]
    statistics_filebase = zipped_inputs[2]
    rainaccumulation_filebase = zipped_inputs[3]
    tracking_path = zipped_inputs[4]
    stats_path = zipped_inputs[5]
    rainaccumulation_path = zipped_inputs[6]
    pcp_thresh = zipped_inputs[7]
    nmaxpf = zipped_inputs[8]
    absolutetb_threshs = zipped_inputs[9]
    startdate = zipped_inputs[10]
    enddate = zipped_inputs[11]
    showalltracks = zipped_inputs[12]

    ######################################################################
    # define constants
    # minimum and maximum brightness temperature thresholds. data outside of this range is filtered
    mintb_thresh = absolutetb_threshs[0]  # k
    maxtb_thresh = absolutetb_threshs[1]  # k

    ##################################################################
    # Load all track stat file
    if showalltracks == 1:
        print("Loading track data")
        statistics_file = (
            stats_path + statistics_filebase + "_" + startdate + "_" + enddate + ".nc"
        )
        print(statistics_file)

        allstatdata = Dataset(statistics_file, "r")
        trackstat_basetime = allstatdata["basetime"][
            :
        ]  # Time of cloud in seconds since 01/01/1970 00:00
        trackstat_cloudnumber = allstatdata["cloudnumber"][
            :
        ]  # Number of the corresponding cloudid file
        trackstat_status = allstatdata["status"][
            :
        ]  # Flag indicating the status of the cloud
        trackstat_mergenumbers = allstatdata["mergenumbers"][
            :
        ]  # Track number that it merges into
        trackstat_splitnumbers = allstatdata["splitnumbers"][:]
        allstatdata.close()

    #########################################################################
    # Get cloudid file associated with this time
    print("Determine corresponding cloudid file and rain accumlation file")
    file_datetime = time.strftime("%Y%m%d_%H%M", time.gmtime(np.copy(filebasetime)))
    filedate = np.copy(file_datetime[0:8])
    filetime = np.copy(file_datetime[9:14])
    irainaccumulationfile = (
        rainaccumulation_path
        + rainaccumulation_filebase
        + str(filedate)
        + "."
        + str(filetime)
        + "00.nc"
    )
    print(("cloudid file: " + cloudid_filename))
    print(("rain accumulation file: " + irainaccumulationfile))

    # Load cloudid data
    print("Load cloudid data")
    cloudiddata = Dataset(cloudid_filename, "r")
    cloudid_cloudnumber = cloudiddata["convcold_cloudnumber"][:]
    cloudid_cloudtype = cloudiddata["cloudtype"][:]
    cloudid_basetime = cloudiddata["basetime"][:]
    basetime_units = cloudiddata["basetime"].units
    basetime_calendar = cloudiddata["basetime"].calendar
    longitude = cloudiddata["lon"][:]
    latitude = cloudiddata["lat"][:]
    nclouds = cloudiddata["nclouds"][:]
    tb = cloudiddata["tb"][:]
    longitude2 = cloudiddata["longitude"][:]
    latitude2 = cloudiddata["latitude"][:]
    cloudiddata.close()

    cloudid_cloudnumber = cloudid_cloudnumber.astype(np.int32)
    cloudid_cloudtype = cloudid_cloudtype.astype(np.int32)

    # Get data dimensions
    [timeindex, nlat, nlon] = np.shape(cloudid_cloudnumber)
    print("np.shape(cloudid_cloudnumber:", np.shape(cloudid_cloudnumber))

    print("Load rain data")
    if os.path.isfile(irainaccumulationfile):
        # Load WRF precip data
        rainaccumulationdata = Dataset(irainaccumulationfile, "r")
        ra_precipitation = rainaccumulationdata["rainrate"][:]  # rainrate (mm/hr)
        # ra_pf_number = rainaccumulatationdata['pf_number'][:]
        rainaccumulationdata.close()

        rapresent = "Yes"
    else:
        print("No radar data")
        ra_precipitation = np.ones((1, nlat, nlon), dtype=float) * np.nan
        rapresent = "No"

    ##############################################################
    # Intiailize track maps
    print("Initialize maps")
    trackmap = np.zeros((1, nlat, nlon), dtype=int)
    trackmap_mergesplit = np.zeros((1, nlat, nlon), dtype=int)

    pfnumbermap_mergesplit = np.zeros((1, nlat, nlon), dtype=int)
    ramap_mergesplit = np.zeros((1, nlat, nlon), dtype=int)

    mergemap = np.zeros((1, nlat, nlon), dtype=int)
    splitmap = np.zeros((1, nlat, nlon), dtype=int)

    ###############################################################
    # Create map of status and track number for every feature in this file
    fillval = -9999
    statusmap = np.ones((1, nlat, nlon), dtype=int) * fillval
    trackmap = np.zeros((1, nlat, nlon), dtype=int)
    allmergemap = np.zeros((1, nlat, nlon), dtype=int)
    allsplitmap = np.zeros((1, nlat, nlon), dtype=int)

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
                sys.exit("Error: No matching cloud pixel found?!")

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

    #    if showalltracks == 1:
    #        print('Create maps of all tracks')
    #        statusmap = np.ones((1, nlat, nlon), dtype=int)*-9999
    #        alltrackmap = np.zeros((1, nlat, nlon), dtype=int)
    #        allmergemap = np.zeros((1, nlat, nlon), dtype=int)
    #        allsplitmap = np.zeros((1, nlat, nlon), dtype=int)
    #
    #        fulltrack, fulltime = np.array(np.where(trackstat_basetime == cloudid_basetime))
    #        for ifull in range(0, len(fulltime)):
    #            ffcloudnumber = trackstat_cloudnumber[fulltrack[ifull], fulltime[ifull]]
    #            ffstatus = trackstat_status[fulltrack[ifull], fulltime[ifull]]
    #
    #            fullypixels, fullxpixels = np.array(np.where(cloudid_cloudnumber[0, :, :] == ffcloudnumber))
    #            statusmap[0, fullypixels, fullxpixels] = ffstatus
    #            alltrackmap[0, fullypixels, fullxpixels] = fulltrack[ifull] + 1
    #
    #            allmergeindices = np.array(np.where(trackstat_mergenumbers == ffcloudnumber))
    #            allmergecloudid = trackstat_cloudnumber[allmergeindices[0, :], allmergeindices[1, :]]
    #            if len(allmergecloudid) > 0:
    #                for iallmergers in range(0, np.shape(allmergeindices)[1]):
    #                    allmergeypixels, allmergexpixels =  np.array(np.where(cloudid_cloudnumber[0, :, :] == allmergecloudid[iallmergers]))
    #
    #                    allmergemap[0, allmergeypixels, allmergexpixels] = allmergeindices[0, iallmergers] + 1
    #
    #            allsplitindices = np.array(np.where(trackstat_splitnumbers == ffcloudnumber))
    #            allsplitcloudid = trackstat_cloudnumber[allsplitindices[0, :], allsplitindices[1, :]]
    #            if len(allsplitcloudid) > 0:
    #                for iallspliters in range(0, np.shape(allsplitindices)[1]):
    #                    allsplitypixels, allsplitxpixels =  np.array(np.where(cloudid_cloudnumber[0, :, :] == allsplitcloudid[iallspliters]))
    #
    #                    allsplitmap[0, allsplitypixels, allsplitxpixels] = allsplitindices[0, iallspliters] + 1
    #
    #        alltrackmap = alltrackmap.astype(np.int32)
    #        allmergemap = allmergemap.astype(np.int32)
    #        allsplitmap = allsplitmap.astype(np.int32)

    #####################################################################
    # Output maps to netcdf file
    print("Writing data")

    # Create output directories
    if not os.path.exists(tracking_path):
        os.makedirs(tracking_path)

    # Define output fileame
    trackmaps_outfile = (
        tracking_path + "tracks_" + str(filedate) + "_" + str(filetime) + ".nc"
    )
    print("trackmaps_outfile: ", trackmaps_outfile)

    # Check if file already exists. If exists, delete
    if os.path.isfile(trackmaps_outfile):
        os.remove(trackmaps_outfile)

    # Define xarray dataset
    if showalltracks == 1:
        output_data = xr.Dataset(
            {
                "basetime": (
                    ["time"],
                    np.array(
                        [
                            pd.to_datetime(
                                num2date(
                                    cloudid_basetime,
                                    units=basetime_units,
                                    calendar=basetime_calendar,
                                )
                            )
                        ],
                        dtype="datetime64[s]",
                    )[0, :],
                ),
                "lon": (["lat7", "lon7"], longitude2),
                "lat": (["lat7", "lon7"], latitude2),
                "nclouds": (["time"], nclouds),
                "tb": (["time", "nlat", "nlon"], tb),
                "precipitation": (
                    ["time", "nlat", "nlon"],
                    ra_precipitation,
                ),  #'mask': (['time', 'nlat', 'nlon'], pf_mask), \
                "cloudtype": (["time", "nlat", "nlon"], cloudid_cloudtype),
                "cloudstatus": (
                    ["time", "nlat", "nlon"],
                    statusmap,
                ),  #'alltracknumbers': (['time', 'nlat', 'nlon'], alltrackmap), \
                "alltracknumbers": (["time", "nlat", "nlon"], trackmap),
                "allsplittracknumbers": (["time", "nlat", "nlon"], allsplitmap),
                "allmergetracknumbers": (["time", "nlat", "nlon"], allmergemap),
                "cloudnumber": (["time", "nlat", "nlon"], cloudid_cloudnumber),
            },  #'pftracknumber': (['time', 'nlat', 'nlon'], mcspfnumbermap_mergesplit), \
            #'pcptracknumber': (['time', 'nlat', 'nlon'], mcsramap_mergesplit)}, \
            coords={
                "time": (["time"], cloudid_basetime)
            },  #'nlat': (['nlat'], np.arange(0, nlat)), \
            #'nlon': (['nlon'], np.arange(0, nlon))}, \
            attrs={
                "title": "Pixel level of tracked clouds and MCSs",
                "Rain_Acccumulation_Data_Present": rapresent,
                "contact": "Katelyn Barber: katelyn.barber@pnnl.gov",
                "created_on": time.ctime(time.time()),
            },
        )

        # Specify variable attributes
        output_data.basetime.attrs[
            "long_name"
        ] = "Epoch time (seconds since 01/01/1970 00:00) of this file"

        output_data.lon.attrs["long_name"] = "Grid of longitude"
        output_data.lon.attrs["units"] = "degrees"

        output_data.lat.attrs["long_name"] = "Grid of latitude"
        output_data.lat.attrs["units"] = "degrees"

        output_data.nclouds.attrs[
            "long_name"
        ] = "Number of MCSs identified in this file"
        output_data.nclouds.attrs["units"] = "unitless"

        output_data.tb.attrs["long_name"] = "brightness temperature"
        output_data.tb.attrs["min_value"] = mintb_thresh
        output_data.tb.attrs["max_value"] = maxtb_thresh
        output_data.tb.attrs["units"] = "K"

        output_data.precipitation.attrs["long_name"] = "WRF rainfall rate"
        output_data.precipitation.attrs["units"] = "mm/hr"

        output_data.cloudtype.attrs["long_name"] = "flag indicating type of ir data"
        output_data.cloudtype.attrs["units"] = "unitless"

        output_data.cloudstatus.attrs["long_name"] = "flag indicating history of cloud"
        output_data.cloudstatus.attrs["units"] = "unitless"

        output_data.alltracknumbers.attrs[
            "long_name"
        ] = "Number of the cloud track associated with the cloud at a given pixel"
        output_data.alltracknumbers.attrs["units"] = "unitless"

        output_data.allmergetracknumbers.attrs[
            "long_name"
        ] = "Number of the cloud track that this cloud merges into"
        output_data.allmergetracknumbers.attrs["units"] = "unitless"

        output_data.allsplittracknumbers.attrs[
            "long_name"
        ] = "Number of the cloud track that this cloud splits from"
        output_data.allsplittracknumbers.attrs["units"] = "unitless"

        output_data.cloudnumber.attrs[
            "long_name"
        ] = "Number associated with the cloud at a given pixel"
        output_data.cloudnumber.attrs[
            "comment"
        ] = "Extent of cloud system is defined using the warm anvil threshold"
        output_data.cloudnumber.attrs["units"] = "unitless"

        # output_data.pftracknumber.attrs['long_name'] = 'Number of the tracked mcs associated with the precipitation feature at a given pixel'
        # output_data.pftracknumber.attrs['comments'] = 'mcs includes smaller merges and splits'
        # output_data.pftracknumber.attrs['units'] = 'unitless'

        # output_data.pcptracknumber.attrs['long_name'] = 'Number of the tracked mcs associated with the accumulated precipitation at a given pixel'
        # output_data.pcptracknumber.attrs['comments'] = 'mcs includes smaller merges and splits'
        # output_data.pcptracknumber.attrs['units'] = 'unitless'

        # Write netcdf file
        print(trackmaps_outfile)
        print("")

        output_data.to_netcdf(
            path=trackmaps_outfile,
            mode="w",
            format="NETCDF4_CLASSIC",
            unlimited_dims="time",
            encoding={
                "basetime": {
                    "zlib": True,
                    "units": basetime_units,
                    "calendar": basetime_calendar,
                },
                "time": {"dtype": "int"},
                "lon": {"zlib": True, "_FillValue": np.nan},
                "lat": {"zlib": True, "_FillValue": np.nan},
                "nclouds": {"dtype": "int", "zlib": True, "_FillValue": -9999},
                "tb": {
                    "zlib": True,
                    "_FillValue": np.nan,
                },  #'mask': {'dtype': 'int','zlib':True, '_FillValue': -9999}, \
                "precipitation": {"zlib": True, "_FillValue": np.nan},
                "cloudtype": {"dtype": "int", "zlib": True, "_FillValue": -9999},
                "cloudstatus": {"dtype": "int", "zlib": True, "_FillValue": -9999},
                "allsplittracknumbers": {
                    "dtype": "int",
                    "zlib": True,
                    "_FillValue": -9999,
                },
                "allmergetracknumbers": {
                    "dtype": "int",
                    "zlib": True,
                    "_FillValue": -9999,
                },
                "alltracknumbers": {"dtype": "int", "zlib": True, "_FillValue": -9999},
                "cloudnumber": {
                    "dtype": "int",
                    "dtype": "int",
                    "zlib": True,
                    "_FillValue": -9999,
                },
            },
        )
        #'pftracknumber': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
        #'pcptracknumber': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}})
