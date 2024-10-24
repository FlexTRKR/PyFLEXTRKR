import numpy as np
import xarray as xr
import os
import sys
import time
import warnings
import logging

def define_robust_mcs_pf(config):
    """
    Identify robust MCS based on PF statistics.
    This version is simplified for SAAG MCS tracking intercomparison.

    Args:
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        statistics_outfile: string
            Robust MCS track statistics file name.
    """

    mcspfstats_filebase = config["mcspfstats_filebase"]
    mcsrobust_filebase = config["mcsrobust_filebase"]
    stats_path = config["stats_outpath"]
    startdate = config["startdate"]
    enddate = config["enddate"]
    mcs_pf_majoraxis_thresh = config["mcs_pf_majoraxis_thresh"]
    mcs_pf_durationthresh = config["mcs_pf_durationthresh"]
    mcs_pf_majoraxis_for_lifetime = config["mcs_pf_majoraxis_for_lifetime"]
    mcs_pf_gap = config["mcs_pf_gap"]
    max_pf_majoraxis_thresh = config["max_pf_majoraxis_thresh"]
    tracks_dimname = config["tracks_dimname"]
    times_dimname = config["times_dimname"]
    pf_dimname = config["pf_dimname"]
    pixel_radius = config["pixel_radius"]
    mcs_min_rainvol_thresh = config["mcs_min_rainvol_thresh"]
    heavy_rainrate_thresh = config["heavy_rainrate_thresh"]
    mcs_volrain_duration_thresh = config["mcs_volrain_duration_thresh"]

    np.set_printoptions(threshold=np.inf)
    logger = logging.getLogger(__name__)
    logger.info("Identifying robust MCS based on PF statistics")

    # Output stats file name
    statistics_outfile = f"{stats_path}{mcsrobust_filebase}{startdate}_{enddate}.nc"

    ######################################################
    # Load MCS PF track stats
    mcspfstats_file = f"{stats_path}{mcspfstats_filebase}{startdate}_{enddate}.nc"
    logger.debug(("mcspfstats_file: ", mcspfstats_file))

    ds_pf = xr.open_dataset(mcspfstats_file,
                            mask_and_scale=False,
                            decode_times=False,)
    ntracks = ds_pf.sizes[tracks_dimname]
    ntimes = ds_pf.sizes[times_dimname]

    ir_trackduration = ds_pf["track_duration"].data
    pf_area = ds_pf["pf_area"].data
    pf_majoraxis = ds_pf["pf_majoraxis"].data
    pf_maxrainrate = ds_pf["pf_maxrainrate"].max(dim=pf_dimname).data
    time_res = float(ds_pf.attrs["time_resolution_hour"])
    fillval = ds_pf["mcs_status"].attrs["_FillValue"]

    # SAAG: total rain volume (sum of rain amount [mm/h] * pixel area [km^2])
    pf_volrain_all = ds_pf["total_rain"].data * pixel_radius**2


    ##################################################
    # Initialize matrices
    trackid_mcs = []
    trackid_nonmcs = []

    pf_mcsstatus = np.full((ntracks, ntimes), fillval, dtype=int)

    ###################################################
    # Loop through each track
    for nt in range(0, ntracks):
        logger.debug(("Track # " + str(nt)))

        ############################################
        # Isolate data from this track
        ilength = np.copy(ir_trackduration[nt]).astype(int)

        # Get the largest precipitation (1st entry in 3rd dimension)
        ipf_majoraxis = np.copy(pf_majoraxis[nt, 0:ilength, 0])
        # SAAG simplified variables
        ipf_maxrainrate = np.copy(pf_maxrainrate[nt, 0:ilength])
        ipf_volrainall = np.copy(pf_volrain_all[nt, 0:ilength])

        ######################################################
        # Apply PF major axis length criteria
        ipfmcs = np.array(
            np.where(
                (ipf_majoraxis >= mcs_pf_majoraxis_thresh)
                & (ipf_majoraxis <= max_pf_majoraxis_thresh)
            )[0]
        )
        nipfmcs = len(ipfmcs)

        if nipfmcs > 0:
            # Apply duration threshold to entire time period
            if nipfmcs * time_res > mcs_pf_durationthresh:

                # Find continuous duration indices
                groups = np.split(
                    ipfmcs, np.where(np.diff(ipfmcs) > mcs_pf_gap)[0] + 1
                )
                nbreaks = len(groups)

                # Loop over each sub-period "group"
                for igroup in range(0, nbreaks):

                    ############################################################
                    # Determine if each group satisfies duration threshold
                    igroup_indices = np.array(np.copy(groups[igroup][:]))
                    nigroup = len(igroup_indices)

                    # Duration length should be group's last index - first index + 1
                    igroup_duration = np.multiply(
                        (groups[igroup][-1] - groups[igroup][0] + 1), time_res
                    )

                    # Group satisfies duration threshold
                    if igroup_duration >= mcs_pf_durationthresh:

                        # Get PF variables for this group                        
                        # SAAG:
                        # Peak rain rate (10 mm/h) > 4 hours
                        # Minimum rainfall volume 
                        igroup_maxrainrate = np.copy(ipf_maxrainrate[igroup_indices])
                        igroup_volrainall = np.copy(ipf_volrainall[igroup_indices])

                        # Count number of times max rain rate > threshold
                        ct_maxrrtimes = np.count_nonzero(igroup_maxrainrate > heavy_rainrate_thresh)
                        # Count number of times volume rain > threshold
                        ct_volrain = np.count_nonzero(igroup_volrainall > mcs_min_rainvol_thresh)
                        # Convert counts to duration [hour]
                        dur_maxrr = float(ct_maxrrtimes) * time_res
                        dur_volrain = float(ct_volrain) * time_res

                        # Duration of max rain rate >= pf_mcs_dur [hour] and
                        # Duration of volume rain >= mcs_volrain_duration_thresh
                        if (dur_maxrr >= mcs_pf_durationthresh) & (
                            dur_volrain >= mcs_volrain_duration_thresh
                        ):
                            # Label this period as an mcs
                            pf_mcsstatus[nt, igroup_indices] = 1
                            logger.debug("MCS")
                        else:
                            trackid_nonmcs = np.append(trackid_nonmcs, int(nt))

                    else:
                        logger.debug("Not MCS")

            # Group does not satistfy duration threshold
            else:
                trackid_nonmcs = np.append(trackid_nonmcs, int(nt))
                logger.debug("Not MCS")
        else:
            logger.debug("Not MCS")

    # Find track indices that are robust MCS
    TEMP_mcsstatus = np.copy(pf_mcsstatus).astype(float)
    TEMP_mcsstatus[TEMP_mcsstatus == fillval] = np.nan
    # trackid_mcs = np.array(np.where(np.nansum(TEMP_mcsstatus, axis=1) > 0))[0, :]
    trackid_mcs = np.where(np.nansum(TEMP_mcsstatus, axis=1) > 0)[0]
    nmcs = len(trackid_mcs)

    # Stop code if not robust MCS present
    if nmcs == 0:
        sys.exit("No robust MCS found!")
    else:
        logger.info(("Number of robust MCS: " + str(int(nmcs))))

    # Isolate data associated with robust MCS
    ir_trackduration = ir_trackduration[trackid_mcs]
    # mcs_basetime = basetime[trackid_mcs]
    pf_mcsstatus = pf_mcsstatus[trackid_mcs, :]
    pf_majoraxis = pf_majoraxis[trackid_mcs, :, :]
    pf_area = pf_area[trackid_mcs, :, :]

    # Determine how long MCS track criteria is satisfied
    # TEMP_mcsstatus = np.copy(pf_mcsstatus).astype(float)
    # TEMP_mcsstatus[TEMP_mcsstatus == fillval] = np.nan
    # mcs_length = np.nansum(TEMP_mcsstatus, axis=1)

    # Get lifetime when a significant PF is present
    # warnings.filterwarnings("ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pf_maxmajoraxis = np.nanmax(pf_majoraxis, axis=2)
        pf_maxmajoraxis[pf_maxmajoraxis < mcs_pf_majoraxis_for_lifetime] = 0
        pf_maxmajoraxis[pf_maxmajoraxis > mcs_pf_majoraxis_for_lifetime] = 1
        pf_lifetime = np.multiply(np.nansum(pf_maxmajoraxis, axis=1), time_res)


    ########################################################
    # Definite life cycle stages. This part is incomplete.
    # TODO: Should implement Zhixiao Zhang's Tb-based lifecycle definition code here down the road.
    ########################################################

    # Subset robust MCS tracks from PF dataset
    # Note: the tracks_dimname cannot be used here as Xarray does not seem to have
    # a method to select data with a string variable
    dsout = ds_pf.sel(tracks=trackid_mcs)
    # Replace tracks index
    tracks_coord = np.arange(0, nmcs)
    times_coord = ds_pf[times_dimname]
    dsout[tracks_dimname] = tracks_coord

    # Convert new variables to DataArrays
    pf_lifetime = xr.DataArray(
        pf_lifetime,
        coords={tracks_dimname:tracks_coord},
        dims=(tracks_dimname),
        attrs={
            "long_name": "MCS lifetime when a significant PF is present",
            "units": "hour",
       }
    )
    pf_mcsstatus = xr.DataArray(
        pf_mcsstatus,
        coords={tracks_dimname:tracks_coord, times_dimname:times_coord},
        dims=(tracks_dimname, times_dimname),
        attrs={
            "long_name": "Flag indicating the status of MCS based on PF. 1 = Yes, 0 = No",
            "units": "unitless",
            "_FillValue": fillval,
        }
    )

    # Add new variables to dataset
    dsout["pf_lifetime"] = pf_lifetime
    dsout["pf_mcsstatus"] = pf_mcsstatus

    # Update global attributes
    dsout.attrs["MCS_PF_majoraxis_thresh"] = mcs_pf_majoraxis_thresh
    dsout.attrs["MCS_PF_duration_thresh"] = mcs_pf_durationthresh
    dsout.attrs["PF_PF_min_majoraxis_thresh"] = mcs_pf_majoraxis_for_lifetime
    dsout.attrs["heavy_rainrate_thresh"] = heavy_rainrate_thresh
    dsout.attrs["mcs_min_rainvol_thresh"] = mcs_min_rainvol_thresh
    dsout.attrs["mcs_volrain_duration_thresh"] = mcs_volrain_duration_thresh
    dsout.attrs["Created_on"] = time.ctime(time.time())


    #########################################################################################
    # Save output to netCDF file
    logger.debug("Saving data")
    logger.debug((time.ctime()))

    # Delete file if it already exists
    if os.path.isfile(statistics_outfile):
        os.remove(statistics_outfile)

    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in dsout.data_vars}

    # Write to netcdf file
    dsout.to_netcdf(path=statistics_outfile, mode="w",
                    format="NETCDF4", unlimited_dims=tracks_dimname, encoding=encoding)
    logger.info(f"{statistics_outfile}")

    # # Write to Zarr format
    # zarr_outpath = f"{stats_path}robust.zarr_{startdate}_{enddate}/"
    # # Delete directory if it already exists
    # if os.path.isdir(zarr_outpath):
    #     shutil.rmtree(zarr_outpath)
    # os.makedirs(zarr_outpath, exist_ok=True)
    # dsout.to_zarr(store=zarr_outpath, consolidated=True)
    # logger.info(f"Robust MCS Zarr: {zarr_outpath}")

    return statistics_outfile