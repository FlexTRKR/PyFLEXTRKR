import numpy as np
import os
import sys
import xarray as xr
import time
import logging

def mapmcs_tb_pf(
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

    # mcsrobust_filebase = config["mcsrobust_filebase"]
    # stats_path = config["stats_outpath"]
    startdate = config["startdate"]
    enddate = config["enddate"]
    fillval = config["fillval"]
    fillval_f = np.nan
    pcp_thresh = config["pcp_thresh"]
    feature_varname = config.get("feature_varname", "feature_number")

    np.set_printoptions(threshold=np.inf)
    logger = logging.getLogger(__name__)

    #######################################################################
    # Load Zarr format MCS track stats file
    # This allows multi-thread read to increase performance
    # mcsstatistics_file_zarr = f"{stats_path}robust.zarr_{startdate}_{enddate}/"
    # ds_mcs = xr.open_zarr(
    #     mcsstatistics_file_zarr,
    #     mask_and_scale=False,
    #     decode_times=False,
    # ).compute()
    # mcsstatistics_file = f"{stats_path}{mcsrobust_filebase}{startdate}_{enddate}.nc"
    # ds_mcs = xr.open_dataset(
    #     mcsstatistics_file,
    #     mask_and_scale=False,
    #     decode_times=False,
    # ).compute()
    # mcstrackstat_basetime = ds_mcs["base_time"].values
    # mcstrackstat_cloudnumber = ds_mcs["cloudnumber"].values
    # mcstrackstat_mergecloudnumber = ds_mcs["merge_cloudnumber"].values
    # mcstrackstat_splitcloudnumber = ds_mcs["split_cloudnumber"].values
    # ds_mcs.close()

    #########################################################################
    # Get cloudid file associated with this time
    file_datetime = time.strftime("%Y%m%d_%H%M", time.gmtime(np.copy(filebasetime)))

    ds_cid = xr.open_dataset(
        cloudid_filename,
        mask_and_scale=False,
        decode_times=False
    ).compute()
    # Required variables
    cloudid_cloudnumber = ds_cid[feature_varname].data
    cloudid_basetime = ds_cid["base_time"].data
    precipitation = ds_cid["precipitation"]
    # # Pass-out variables
    # base_time = ds_cid["base_time"]
    # longitude = ds_cid["longitude"]
    # latitude = ds_cid["latitude"]
    # tb = ds_cid["tb"]
    # cloudtype = ds_cid["cloudtype"]
    # cid_time = ds_cid["time"]
    # lat = ds_cid["lat"]
    # lon = ds_cid["lon"]
    ds_cid.close()

    # file_datetime = time.strftime("%Y%m%d_%H%M", time.gmtime(np.int(cloudid_basetime.item())))

    # Get data dimensions
    [timeindex, nlat, nlon] = np.shape(cloudid_cloudnumber)

    ##############################################################
    # Intiailize track maps
    mcstrackmap = np.zeros((1, nlat, nlon), dtype=int)
    mcstrackmap_mergesplit = np.zeros((1, nlat, nlon), dtype=int)
    mcspcpmap_mergesplit = np.zeros((1, nlat, nlon), dtype=int)
    mcsmergemap = np.zeros((1, nlat, nlon), dtype=int)
    mcssplitmap = np.zeros((1, nlat, nlon), dtype=int)

    ###############################################################
    # logger.info('Generate MCS maps')
    # Get tracks
    # itrack, itime = np.array(np.where(mcstrackstat_basetime == cloudid_basetime))
    # nmatchcloud = len(itime)
    # import pdb; pdb.set_trace()

    itrack = file_trackindex
    nmatchcloud = len(file_cloudnumber)
    if nmatchcloud > 0:
        ##############################################################
        # Loop over each cloud in this unique file
        for jj in range(0, nmatchcloud):
            # logger.debug(('MCS #: ' + str(int(itrack[jj] + 1))))
            # Get cloud number
            # jjcloudnumber = mcstrackstat_cloudnumber[itrack[jj], itime[jj]]
            jjcloudnumber = file_cloudnumber[jj]

            # Get the mask matching this cloud number
            cmask = cloudid_cloudnumber.squeeze() == jjcloudnumber

            # Label this cloud with the track number.
            # In the IDL version, track numbers on pixel files are added 1
            # We don't have to do this, making the track number the same with the track stats
            # may simplify the usage of the data, but codes in gpm_mcs_post must be changed
            # TODO: Look into how much change in gpm_mcs_post is needed
            #  and decide if this change should be made
            if np.count_nonzero(cmask) > 0:
                mcstrackmap[0, cmask] = itrack[jj] + 1
                mcstrackmap_mergesplit[0, cmask] = itrack[jj] + 1
            else:
                logger.warning(f"Warning: No matching cloud pixel found: {jjcloudnumber}")

            ###########################################################
            # Find merging clouds
            jjmerge = np.where(file_mergecloudnumber[jj, :] > 0)[0]

            # Loop through merging clouds if present
            if len(jjmerge) > 0:
                for imerge in jjmerge:
                    # Get mask matching the merging cloud
                    im_number = file_mergecloudnumber[jj, imerge]
                    m_cmask = cloudid_cloudnumber.squeeze() == im_number

                    # Label this cloud with the track number.
                    # TODO: consider mapping with same track number
                    if np.count_nonzero(m_cmask) > 0:
                        mcstrackmap_mergesplit[0, m_cmask] = itrack[jj] + 1
                        mcsmergemap[0, m_cmask] = itrack[jj] + 1
                    else:
                        logger.warning(f"Warning: No matching merging cloud found: {im_number}")

            ###########################################################
            # Find splitting clouds
            jjsplit = np.where(file_splitcloudnumber[jj, :] > 0)[0]

            # Loop through splitting clouds if present
            if len(jjsplit) > 0:
                for isplit in jjsplit:
                    # Get mask matching the splitting cloud
                    is_number = file_splitcloudnumber[jj, isplit]
                    s_cmask = cloudid_cloudnumber.squeeze() == is_number

                    # Label this cloud with the track number.
                    # TODO: consider mapping with same track number
                    if np.count_nonzero(s_cmask) > 0:
                        mcstrackmap_mergesplit[0, s_cmask] = itrack[jj] + 1
                        mcssplitmap[0, s_cmask] = itrack[jj] + 1
                    else:
                        logger.warning(f"Warning: No matching splitting cloud found: {is_number}")

        ####################################################################
        # Create PF track number map
        mcspcpmap_mergesplit = (precipitation.data > pcp_thresh) * mcstrackmap_mergesplit


    #####################################################################
    # Output maps to netcdf file
    logger.debug('Writing MCS pixel-level data')

    # Define output fileame
    mcstrackmaps_outfile = (
        config["pixeltracking_outpath"] +
        config["pixeltracking_filebase"] +
        file_datetime + ".nc"
    )

    # Delete file if it already exists
    if os.path.isfile(mcstrackmaps_outfile):
        os.remove(mcstrackmaps_outfile)

    # Define variable list
    varlist = {
        "base_time": (["time"], ds_cid["base_time"].data, ds_cid["base_time"].attrs),
        "longitude": (["lat", "lon"], ds_cid["longitude"].data, ds_cid["longitude"].attrs),
        "latitude": (["lat", "lon"], ds_cid["latitude"].data, ds_cid["latitude"].attrs),
        "tb": (["time", "lat", "lon"], ds_cid["tb"].data, ds_cid["tb"].attrs),
        "precipitation": (["time", "lat", "lon"], ds_cid["precipitation"].data, ds_cid["precipitation"].attrs),
        "cloudtype": (["time", "lat", "lon"], ds_cid["cloudtype"].data, ds_cid["cloudtype"].attrs),

        # "base_time": (["time"], base_time.values, base_time.attrs),
        # "longitude": (["lat", "lon"], longitude.values, longitude.attrs),
        # "latitude": (["lat", "lon"], latitude.values, latitude.attrs),
        # "tb": (["time", "lat", "lon"], tb.values, tb.attrs),
        # "precipitation": (["time", "lat", "lon"], precipitation.values, precipitation.attrs),
        # "cloudtype": (["time", "lat", "lon"], cloudtype.values, cloudtype.attrs),
        "cloudnumber": (["time", "lat", "lon"], cloudid_cloudnumber, ds_cid[feature_varname].attrs),
        "split_tracknumbers": (["time", "lat", "lon"], mcssplitmap),
        "merge_tracknumbers": (["time", "lat", "lon"], mcsmergemap),
        "cloudtracknumber_nomergesplit": (["time", "lat", "lon"], mcstrackmap),
        "cloudtracknumber": (["time", "lat", "lon"], mcstrackmap_mergesplit,),
        "pcptracknumber": (["time", "lat", "lon"], mcspcpmap_mergesplit),
    }

    # Define coordinate list
    coordlist = {
        "time": (["time"], ds_cid["time"].data, ds_cid["time"].attrs),
        "lat": (["lat"], ds_cid["lat"].data, ds_cid["lat"].attrs),
        "lon": (["lon"], ds_cid["lon"].data, ds_cid["lon"].attrs),
        # "time": (["time"], cid_time.values, cid_time.attrs),
        # "lat": (["lat"], lat.values, lat.attrs),
        # "lon": (["lon"], lon.values, lon.attrs),
    }

    # Define global attributes
    gattrlist = {
        "Title": "Robust MCS pixel-level tracking data",
        "Contact": "Zhe Feng: zhe.feng@pnnl.gov",
        "Created_on": time.ctime(time.time()),
        "startdate": startdate,
        "enddate": enddate,
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
    # import pdb; pdb.set_trace()

    return mcstrackmaps_outfile