import numpy as np
import os
import sys
import xarray as xr
import time
import logging

def mapmcs_tb_pf(
    cloudid_filename,
    filebasetime,
    config,
):
    """
    Map MCS track numbers to pixel level file.

    Args:
        cloudid_filename: string
            Cloudid file name.
        filebasetime: int
            Cloudid file base time.
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        mcstrackmaps_outfile: string
            Track number pixel-level file name.
    """

    mcsstats_filebase = "robust_mcs_tracks_"
    stats_path = config["stats_outpath"]
    startdate = config["startdate"]
    enddate = config["enddate"]
    fillval = config["fillval"]
    fillval_f = np.nan
    pcp_thresh = config["pcp_thresh"]
    showalltracks = 0

    np.set_printoptions(threshold=np.inf)
    logger = logging.getLogger(__name__)

    ##################################################################
    # Load all track stat file
    if showalltracks == 1:
        logger.info("Loading track data")
        statistics_file = (
            stats_path +
            config["trackstats_filebase"] +
            startdate + "_" +
            enddate + ".nc"
        )
        logger.info(statistics_file)

        ds_all = xr.open_dataset(statistics_file,
                                      mask_and_scale=False,
                                      decode_times=False)
        trackstat_basetime = ds_all["basetime"].values
        trackstat_cloudnumber = ds_all["cloudnumber"].values
        trackstat_status = ds_all["status"].values
        trackstat_mergenumbers = ds_all["mergenumbers"].values
        trackstat_splitnumbers = ds_all["splitnumbers"].values
        ds_all.close()

    #######################################################################
    # Load MCS track stat file
    # logger.info('Loading MCS data')
    mcsstatistics_file = (
        stats_path +
        mcsstats_filebase +
        startdate + "_" +
        enddate + ".nc"
    )

    ds_mcs = xr.open_dataset(mcsstatistics_file,
                             mask_and_scale=False,
                             decode_times=False)
    mcstrackstat_basetime = ds_mcs["base_time"].values
    mcstrackstat_cloudnumber = ds_mcs["cloudnumber"].values
    mcstrackstat_mergecloudnumber = ds_mcs["mergecloudnumber"].values
    mcstrackstat_splitcloudnumber = ds_mcs["splitcloudnumber"].values


    #########################################################################
    # Get cloudid file associated with this time
    # logger.info('Determine corresponding cloudid file and rain accumlation file')
    file_datetime = time.strftime("%Y%m%d_%H%M", time.gmtime(np.copy(filebasetime)))
    filedate = np.copy(file_datetime[0:8])
    filetime = np.copy(file_datetime[9:14])

    ds_cid = xr.open_dataset(cloudid_filename,
                             mask_and_scale=False,
                             decode_times=False)
    cloudid_cloudnumber = ds_cid[config["numbername"]].values
    cloudid_basetime = ds_cid["basetime"].values
    precipitation = ds_cid["precipitation"].values
    ds_cid.close()

    # Get data dimensions
    [timeindex, nlat, nlon] = np.shape(cloudid_cloudnumber)

    ##############################################################
    # Intiailize track maps
    # logger.info('Initialize maps')
    mcstrackmap = np.zeros((1, nlat, nlon), dtype=int)
    mcstrackmap_mergesplit = np.zeros((1, nlat, nlon), dtype=int)
    mcspcpmap_mergesplit = np.zeros((1, nlat, nlon), dtype=int)
    mcsmergemap = np.zeros((1, nlat, nlon), dtype=int)
    mcssplitmap = np.zeros((1, nlat, nlon), dtype=int)

    ###############################################################
    # Create map of status and track number for every feature in this file
    if showalltracks == 1:
        # logger.info('Create maps of all tracks')
        statusmap = np.full((1, nlat, nlon), fillval, dtype=int)
        alltrackmap = np.zeros((1, nlat, nlon), dtype=int)
        allmergemap = np.zeros((1, nlat, nlon), dtype=int)
        allsplitmap = np.zeros((1, nlat, nlon), dtype=int)

        fulltrack, fulltime = np.array(np.where(trackstat_basetime == cloudid_basetime))
        for ifull in range(0, len(fulltime)):
            ffcloudnumber = trackstat_cloudnumber[fulltrack[ifull], fulltime[ifull]]
            ffstatus = trackstat_status[fulltrack[ifull], fulltime[ifull]]

            fullypixels, fullxpixels = np.array(
                np.where(cloudid_cloudnumber[0, :, :] == ffcloudnumber)
            )
            statusmap[0, fullypixels, fullxpixels] = ffstatus
            alltrackmap[0, fullypixels, fullxpixels] = fulltrack[ifull] + 1

            allmergeindices = np.array(
                np.where(trackstat_mergenumbers == ffcloudnumber)
            )
            allmergecloudid = trackstat_cloudnumber[
                allmergeindices[0, :], allmergeindices[1, :]
            ]
            if len(allmergecloudid) > 0:
                for iallmergers in range(0, np.shape(allmergeindices)[1]):
                    allmergeypixels, allmergexpixels = np.array(
                        np.where(
                            cloudid_cloudnumber[0, :, :] == allmergecloudid[iallmergers]
                        )
                    )

                    allmergemap[0, allmergeypixels, allmergexpixels] = (
                        allmergeindices[0, iallmergers] + 1
                    )

            allsplitindices = np.array(
                np.where(trackstat_splitnumbers == ffcloudnumber)
            )
            allsplitcloudid = trackstat_cloudnumber[
                allsplitindices[0, :], allsplitindices[1, :]
            ]
            if len(allsplitcloudid) > 0:
                for iallspliters in range(0, np.shape(allsplitindices)[1]):
                    allsplitypixels, allsplitxpixels = np.array(
                        np.where(
                            cloudid_cloudnumber[0, :, :]
                            == allsplitcloudid[iallspliters]
                        )
                    )

                    allsplitmap[0, allsplitypixels, allsplitxpixels] = (
                        allsplitindices[0, iallspliters] + 1
                    )

        alltrackmap = alltrackmap.astype(np.int32)
        allmergemap = allmergemap.astype(np.int32)
        allsplitmap = allsplitmap.astype(np.int32)

    ###############################################################
    # logger.info('Generate MCS maps')
    # Get tracks
    itrack, itime = np.array(np.where(mcstrackstat_basetime == cloudid_basetime))
    ntimes = len(itime)
    if ntimes > 0:
        ##############################################################
        # Loop over each cloud in this unique file
        # logger.info('Loop over each cloud in the file')
        for jj in range(0, ntimes):
            logger.debug(('MCS #: ' + str(int(itrack[jj] + 1))))
            # Get cloud nummber
            jjcloudnumber = mcstrackstat_cloudnumber[itrack[jj], itime[jj]]

            # Find pixels assigned to this cloud number
            jjcloudypixels, jjcloudxpixels = np.array(
                np.where(cloudid_cloudnumber[0, :, :] == jjcloudnumber)
            )

            # Label this cloud with the track number.
            # In the IDL version, track numbers on pixel files are added 1
            # We don't have to do this, making the track number the same with the track stats
            # may simplify the usage of the data, but codes in gpm_mcs_post must be changed
            # TODO: Look into how much change in gpm_mcs_post is needed
            #  and decide if this change should be made
            if len(jjcloudypixels) > 0:
                mcstrackmap[0, jjcloudypixels, jjcloudxpixels] = itrack[jj] + 1
                mcstrackmap_mergesplit[0, jjcloudypixels, jjcloudxpixels] = (
                    itrack[jj] + 1
                )

                # statusmap[0, jjcloudypixels, jjcloudxpixels] = timestatus[jj]
            else:
                sys.exit("Error: No matching cloud pixel found?!")

            ###########################################################
            # Find merging clouds
            # logger.info('Find mergers')
            jjmerge = np.array(
                np.where(mcstrackstat_mergecloudnumber[itrack[jj], itime[jj], :] > 0)
            )[0, :]

            # Loop through merging clouds if present
            if len(jjmerge) > 0:
                for imerge in jjmerge:
                    # Find cloud number asosicated with the merging cloud
                    jjmergeypixels, jjmergexpixels = np.array(
                        np.where(
                            cloudid_cloudnumber[0, :, :]
                            == mcstrackstat_mergecloudnumber[
                                itrack[jj], itime[jj], imerge
                            ]
                        )
                    )

                    # Label this cloud with the track number.
                    # TODO: consider mapping with same track number
                    if len(jjmergeypixels) > 0:
                        mcstrackmap_mergesplit[0, jjmergeypixels, jjmergexpixels] = (
                            itrack[jj] + 1
                        )
                        # statusmap[0, jjmergeypixels, jjmergexpixels] = mcsmergestatus[itrack[jj], itime[jj], imerge]
                        mcsmergemap[0, jjmergeypixels, jjmergexpixels] = itrack[jj] + 1
                    else:
                        sys.exit("Error: No matching merging cloud pixel found?!")

            ###########################################################
            # Find splitting clouds
            # logger.info('Find splits')
            jjsplit = np.array(
                np.where(mcstrackstat_splitcloudnumber[itrack[jj], itime[jj], :] > 0)
            )[0, :]

            # Loop through splitting clouds if present
            if len(jjsplit) > 0:
                for isplit in jjsplit:
                    # Find cloud number asosicated with the splitting cloud
                    jjsplitypixels, jjsplitxpixels = np.array(
                        np.where(
                            cloudid_cloudnumber[0, :, :]
                            == mcstrackstat_splitcloudnumber[
                                itrack[jj], itime[jj], isplit
                            ]
                        )
                    )

                    # Label this cloud with the track number.
                    # TODO: consider mapping with same track number
                    if len(jjsplitypixels) > 0:
                        mcstrackmap_mergesplit[0, jjsplitypixels, jjsplitxpixels] = (
                            itrack[jj] + 1
                        )
                        # statusmap[0, jjsplitypixels, jjsplitxpixels] = mcssplitstatus[itrack[jj], itime[jj], isplit]
                        mcssplitmap[0, jjsplitypixels, jjsplitxpixels] = itrack[jj] + 1
                    else:
                        sys.exit("Error: No matching splitting cloud pixel found?!")

        ####################################################################
        # Create PF track number map
        mcspcpmap_mergesplit = (precipitation > pcp_thresh) * mcstrackmap_mergesplit

    # mcssplitmap = mcssplitmap.astype(np.int32)
    # mcsmergemap = mcsmergemap.astype(np.int32)
    # mcspcpmap_mergesplit = mcspcpmap_mergesplit.astype(np.int32)
    # # mcspfnumbermap_mergesplit = mcspfnumbermap_mergesplit.astype(np.int32)
    # mcstrackmap_mergesplit = mcstrackmap_mergesplit.astype(np.int32)
    # mcstrackmap = mcstrackmap.astype(np.int32)

    if showalltracks == 1:
        alltrackmap = alltrackmap.astype(np.int32)
        allsplitmap = allsplitmap.astype(np.int32)
        allmergemap = allmergemap.astype(np.int32)
        statusmap = statusmap.astype(np.int32)

    #####################################################################
    # Output maps to netcdf file
    logger.debug('Writing MCS pixel-level data')

    # # Create output directories
    # if not os.path.exists(mcstracking_path):
    #     os.makedirs(mcstracking_path)

    # Define output fileame
    mcstrackmaps_outfile = (
        config["pixeltracking_outpath"] +
        config["pixeltracking_filebase"] +
        str(filedate) + "_" +
        str(filetime) + ".nc"
    )
    # logger.info('mcstrackmaps_outfile: ', mcstrackmaps_outfile)

    # Check if file already exists. If exists, delete
    if os.path.isfile(mcstrackmaps_outfile):
        os.remove(mcstrackmaps_outfile)


    # Define variable list
    varlist = {
        "basetime": (["time"], ds_cid["basetime"], ds_cid["basetime"].attrs),
        "longitude": (["lat", "lon"], ds_cid["longitude"], ds_cid["longitude"].attrs),
        "latitude": (["lat", "lon"], ds_cid["latitude"], ds_cid["latitude"].attrs),
        "tb": (["time", "lat", "lon"], ds_cid["tb"], ds_cid["tb"].attrs),
        "precipitation": (["time", "lat", "lon"], precipitation, ds_cid["precipitation"].attrs),
        "cloudtype": (["time", "lat", "lon"], ds_cid["cloudtype"], ds_cid["cloudtype"].attrs),
        "cloudnumber": (["time", "lat", "lon"], cloudid_cloudnumber, ds_cid[config["numbername"]].attrs),
        "split_tracknumbers": (["time", "lat", "lon"], mcssplitmap),
        "merge_tracknumbers": (["time", "lat", "lon"], mcsmergemap),
        "cloudtracknumber_nomergesplit": (["time", "lat", "lon"], mcstrackmap),
        "cloudtracknumber": (["time", "lat", "lon"], mcstrackmap_mergesplit,),
        "pcptracknumber": (["time", "lat", "lon"], mcspcpmap_mergesplit),
    }
    if showalltracks == 1:
        varlist_extra = {
            "alltracknumbers": (["time", "lat", "lon"], alltrackmap),
            "allsplittracknumbers": (["time", "lat", "lon"], allsplitmap),
            "allmergetracknumbers": (["time", "lat", "lon"], allmergemap),
        }
        varlist = {**varlist, **varlist_extra}

    # Define coordinate list
    coordlist = {
        "time": (["time"], ds_cid["time"], ds_cid["time"].attrs),
        "lat": (["lat"], ds_cid["lat"], ds_cid["lat"].attrs),
        "lon": (["lon"], ds_cid["lon"], ds_cid["lon"].attrs),
    }

    # Define global attributes
    gattrlist = {
        "Title": "Robust MCS pixel-level tracking data",
        "Contact": "Zhe Feng: zhe.feng@pnnl.gov",
        "Created_on": time.ctime(time.time()),
        "startdate": startdate,
        "enddate": enddate,
        "datasource": config["datasource"],
        "precipitation_datasource": config["pfdatasource"],
        "mcs_tb_area_thresh": config["mcs_tb_area_thresh"],
        "mcs_tb_duration_thresh": config["mcs_tb_duration_thresh"],
        "mcs_pf_majoraxis_thresh": config["mcs_pf_majoraxis_thresh"],
        "mcs_pf_durationthresh": config["mcs_pf_durationthresh"],
    }

    # Define Xarray Dataset
    ds_out = xr.Dataset(varlist, coordlist, gattrlist)

    # Specify variable attributes
    ds_out["merge_tracknumbers"].attrs[
        "long_name"
    ] = "Number of the MCS track that this cloud merges into"
    ds_out["merge_tracknumbers"].attrs["units"] = "unitless"
    ds_out["merge_tracknumbers"].attrs["_FillValue"] = 0

    ds_out["split_tracknumbers"].attrs[
        "long_name"
    ] = "Number of the MCS track that this cloud splits from"
    ds_out["split_tracknumbers"].attrs["units"] = "unitless"
    ds_out["split_tracknumbers"].attrs["_FillValue"] = 0

    ds_out["cloudtracknumber_nomergesplit"].attrs[
        "long_name"
    ] = "MCS cloud track number (exclude merge/split clouds)"
    ds_out["cloudtracknumber_nomergesplit"].attrs["units"] = "unitless"
    ds_out["cloudtracknumber_nomergesplit"].attrs["_FillValue"] = 0

    ds_out["cloudtracknumber"].attrs["long_name"] = "MCS cloud track number"
    ds_out["cloudtracknumber"].attrs["units"] = "unitless"
    ds_out["cloudtracknumber"].attrs["_FillValue"] = 0

    ds_out["pcptracknumber"].attrs["long_name"] = "MCS PF track number"
    ds_out["pcptracknumber"].attrs["units"] = "unitless"
    ds_out["pcptracknumber"].attrs["_FillValue"] = 0

    if showalltracks == 1:
        ds_out["alltracknumbers"].attrs["long_name"] = "All cloud number"
        ds_out["alltracknumbers"].attrs["units"] = "unitless"
        ds_out["alltracknumbers"].attrs["_FillValue"] = 0

        ds_out["allmergetracknumbers"].attrs[
            "long_name"
        ] = "Number of the cloud track that this cloud merges into"
        ds_out["allmergetracknumbers"].attrs["units"] = "unitless"
        ds_out["allmergetracknumbers"].attrs["_FillValue"] = 0

        ds_out["allsplittracknumbers"].attrs[
            "long_name"
        ] = "Number of the cloud track that this cloud splits from"
        ds_out["allsplittracknumbers"].attrs["units"] = "unitless"
        ds_out["allsplittracknumbers"].attrs["_FillValue"] = 0

    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encodelist = {var: comp for var in ds_out.data_vars}
    encodelist["longitude"] = dict(zlib=True, dtype="float32")
    encodelist["latitude"] = dict(zlib=True, dtype="float32")
    # Write netcdf file
    ds_out.to_netcdf(
        path=mcstrackmaps_outfile,
        mode="w",
        format="NETCDF4",
        unlimited_dims="time",
        encoding=encodelist,
    )
    logger.info(mcstrackmaps_outfile)

    return mcstrackmaps_outfile