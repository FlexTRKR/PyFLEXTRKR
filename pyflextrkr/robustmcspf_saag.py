import numpy as np
import xarray as xr
import os, shutil
import sys
import time
import warnings
import logging

def define_robust_mcs_pf(config):
    """
    Identify robust MCS based on PF statistics.

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
    coefs_pf_area = config["coefs_pf_area"]
    coefs_pf_rr = config["coefs_pf_rr"]
    coefs_pf_skew = config["coefs_pf_skew"]
    coefs_pf_heavyratio = config["coefs_pf_heavyratio"]
    max_pf_majoraxis_thresh = config["max_pf_majoraxis_thresh"]
    tracks_dimname = config["tracks_dimname"]
    times_dimname = config["times_dimname"]
    pf_dimname = config["pf_dimname"]
    pixel_radius = config["pixel_radius"]

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
    ntracks = ds_pf.dims[tracks_dimname]
    ntimes = ds_pf.dims[times_dimname]

    ir_trackduration = ds_pf["track_duration"].data
    pf_area = ds_pf["pf_area"].data
    pf_majoraxis = ds_pf["pf_majoraxis"].data
    pf_rainrate = ds_pf["pf_rainrate"].data
    pf_skewness = ds_pf["pf_skewness"].data
    pf_maxrainrate = ds_pf["pf_maxrainrate"].max(dim=pf_dimname).data
    # pf_accumrain = ds_pf['pf_accumrain'].data
    # pf_accumrainheavy = ds_pf['pf_accumrainheavy'].data
    time_res = float(ds_pf.attrs["time_resolution_hour"])
    # mcs_ir_areathresh = float(data.attrs["MCS_IR_area_thresh_km2"])
    # mcs_ir_durationthresh = float(data.attrs["MCS_IR_duration_thresh_hr"])
    # mcs_ir_eccentricitythresh = float(data.attrs["MCS_IR_eccentricity_thres"])
    # missing_val = data.attrs["missing_value"]
    fillval = ds_pf["mcs_status"].attrs["_FillValue"]
    fillval_f = ds_pf["pf_area"].attrs["_FillValue"]
    # basetime = ds_pf["base_time"].data

    # Calculate accumulate rain by summing over all PFs
    # This is the same approach as in the IDL version of the code
    # pf_volrain_all = ds_pf["pf_accumrain"].sum(dim=pf_dimname).data
    # pf_volrain_heavy = ds_pf["pf_accumrainheavy"].sum(dim=pf_dimname).data
    # TODO: Technically should use "total_rain", "total_heavyrain" variables in the file
    # !!Test the impact of this later!!
    # pf_volrain_all = ds_pf["total_rain"]
    # pf_volrain_heavy = ds_pf["total_heavyrain"].data

    # SAAG: total rain volume (sum of rain amount [mm/h] * pixel area [km^2])
    pf_volrain_all = ds_pf["total_rain"].data * pixel_radius**2


    ##################################################
    # Initialize matrices
    trackid_mcs = []
    trackid_nonmcs = []

    # pf_mcstype = np.full(ntracks, fillval, dtype=int)
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
        ipf_area = np.copy(pf_area[nt, 0:ilength, 0])
        ipf_rainrate = np.copy(pf_rainrate[nt, 0:ilength, 0])
        ipf_skewness = np.copy(pf_skewness[nt, 0:ilength, 0])
        # ipf_volrainall = np.copy(pf_volrain_all[nt, 0:ilength])
        # ifp_volrainheavy = np.copy(pf_volrain_heavy[nt, 0:ilength])
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

                    # # Compute PF fit values using the coefficients
                    # mcs_pfarea = coefs_pf_area[0] + coefs_pf_area[1] * igroup_duration
                    # mcs_rrskew = coefs_pf_skew[0] + coefs_pf_skew[1] * igroup_duration
                    # mcs_rravg = coefs_pf_rr[0] + coefs_pf_rr[1] * igroup_duration
                    # mcs_heavyratio = (
                    #     coefs_pf_heavyratio[0] + coefs_pf_heavyratio[1] * igroup_duration
                    # )

                    # Group satisfies duration threshold
                    if igroup_duration >= mcs_pf_durationthresh:

                        # Get PF variables for this group
                        
                        # SAAG:
                        # Peak rain rate (10 mm/h) > 4 hours
                        # Minimum rainfall volume 
                        igroup_maxrainrate = np.copy(ipf_maxrainrate[igroup_indices])
                        igroup_volrainall = np.copy(ipf_volrainall[igroup_indices])


                        import pdb; pdb.set_trace()
                        igroup_pfmajoraxis = np.copy(ipf_majoraxis[igroup_indices])

                        igroup_pfarea = np.copy(ipf_area[igroup_indices])
                        igroup_pfrate = np.copy(ipf_rainrate[igroup_indices])
                        igroup_pfskew = np.copy(ipf_skewness[igroup_indices])
                        igroup_volrainall = np.copy(ipf_volrainall[igroup_indices])
                        igroup_volrainheavy = np.copy(ifp_volrainheavy[igroup_indices])

                        # Count number of times when PF exceeds MCS criteria
                        ct_pftimes = np.count_nonzero(
                            (igroup_pfarea > mcs_pfarea)
                            & (igroup_pfrate > mcs_rravg)
                            & (igroup_pfskew > mcs_rrskew)
                        )
                        dur_pf = float(ct_pftimes) * time_res

                        # Calculate volumetric heavy rain ratio during this sub-period
                        heavyrain_ratio = (
                            100
                            * np.nansum(igroup_volrainheavy)
                            / np.nansum(igroup_volrainall)
                        )

                        # Duration of PF satisfying MCS criteria >= pf_mcs_dur [hour] and
                        # heavy rain ratio during the sub-period >= mcs_heavyratio
                        if (dur_pf >= mcs_pf_durationthresh) & (
                            heavyrain_ratio > mcs_heavyratio
                        ):
                            # Label this period as an mcs
                            pf_mcsstatus[nt, igroup_indices] = 1
                            logger.debug("MCS")
                        else:
                            trackid_nonmcs = np.append(trackid_nonmcs, int(nt))
                        # import pdb; pdb.set_trace()

                        ## Determine type of mcs (squall or non-squall)
                        # isquall = np.array(np.where(igroup_ccaspectratio > aspectratiothresh))[0, :]
                        # nisquall = len(isquall)

                        # if nisquall > 0:
                        #    # Label as squall
                        #    pf_mcstype[nt] = 1
                        #    pf_cctype[nt, igroup_indices[isquall]] = 1
                        # else:
                        #    # Label as non-squall
                        #    pf_mcstype[nt] = 2
                        #    pf_cctype[nt, igroup_indices[isquall]] = 2
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
    trackid_mcs = np.array(np.where(np.nansum(TEMP_mcsstatus, axis=1) > 0))[0, :]
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

    # # Process only MCSs that last >= mcs_pf_lifecyclethresh
    # lifetime = np.multiply(ir_trackduration, time_res)
    # ilongmcs = np.array(np.where(lifetime >= mcs_pf_lifecyclethresh))[0, :]
    # nlongmcs = len(ilongmcs)
    #
    # if nlongmcs > 0:
    #     # logger.info('ENTERED NLONGMCS IF STATEMENT LINES 214')
    #     # Initialize arrays
    #     cycle_complete = np.full(nmcs, fillval, dtype=int)
    #     cycle_stage = np.full((nmcs, ntimes), fillval, dtype=int)
    #     cycle_index = np.full((nmcs, 5), fillval, dtype=int)
    #
    #     # mcs_basetime = np.empty((nmcs, ntimes), dtype='datetime64[s]')
    #     # logger.info(mcs_basetime)
    #
    #     # Loop through each mcs
    #     for ilm in range(0, nlongmcs):
    #         # Initialize arrays
    #         ilm_index = np.full(5, fillval, dtype=int)
    #         ilm_cycle = np.full(ntimes, fillval, dtype=int)
    #
    #         # Isolate data from this track
    #         ilm_irtracklength = np.copy(ir_trackduration[ilongmcs[ilm]]).astype(int)
    #         ilm_pfarea = np.copy(pf_area[ilongmcs[ilm], 0:ilm_irtracklength, 0])
    #         ilm_pfmajoraxis = np.copy(
    #             pf_majoraxis[ilongmcs[ilm], 0:ilm_irtracklength, 0]
    #         )
    #         ilm_maxpfmajoraxis = np.nanmax(ilm_pfmajoraxis, axis=0)
    #
    #         # Get basetime
    #         # TEMP_basetime = np.array([pd.to_datetime(ds_pf['basetime'][trackid_mcs[ilongmcs[ilm]], 0:ilm_irtracklength].data, unit='s')])
    #         # mcs_basetime[ilm, 0:ilm_irtracklength] = TEMP_basetime
    #
    #         ##################################################################
    #         # Find indices of when convective line present and absent and when stratiform present
    #
    # #            # Find times with convective core area > 0
    # #            iccarea = np.array(np.where(ilm_maxpfccarea > 0))[0, :]
    # #            iccarea_groups = np.split(iccarea, np.where(np.diff(iccarea) > 2)[0]+1)
    # #            if len(iccarea) > 0 and len(iccarea_groups) > 1:
    # #                grouplength = np.empty(len(iccarea_groups))
    # #                for igroup in range(0, len(iccarea_groups)):
    # #                    grouplength[igroup] = len(iccarea_groups[igroup][:])
    # #                maxgroup = np.nanargmax(grouplength)
    # #                iccarea = iccarea_groups[maxgroup][:]
    # #            elif len(iccarea) > 0 :
    # #                iccarea = np.arange(iccarea[0], iccarea[-1]+1)
    # #            nccarea = len(iccarea)
    #
    # #             # Find times with major axis length > mcs_pf_majoraxis_thresh
    # #             # iccline = np.array(np.where(ilm_maxpfmajoraxis > mcs_pf_majoraxis_thresh))[0, :]
    # #             # iccline_groups = np.split(iccline, np.where(np.diff(iccline) > 2)[0]+1)
    # #             iccline = np.array(np.where(ilm_pfmajoraxis > mcs_pf_majoraxis_thresh))[0, :]
    # #             iccline_groups = np.split(iccline, np.where(np.diff(iccline) > mcs_pf_gap)[0] + 1)
    # #             if len(iccline) > 0 and len(iccline_groups) > 1:
    # #                 grouplength = np.empty(len(iccline_groups))
    # #                 for igroup in range(0, len(iccline_groups)):
    # #                     grouplength[igroup] = len(iccline_groups[igroup][:])
    # #                 maxgroup = np.nanargmax(grouplength)
    # #                 iccline = iccline_groups[maxgroup][:]
    # #             elif len(iccline) > 0:
    # #                 iccline = np.arange(iccline[0], iccline[-1]+1)
    # #             nccline = len(iccline)
    #
    # #             ###############################################################################
    # #             # Classify cloud only stage
    #
    # #             # Cloud only stage
    # # #            if nccarea > 0:
    # # #                # If first convective time is after the first cloud time, label all hours before the convective core appearance time as preconvective
    # # #                if iccarea[0] > 0 and iccarea[0] < ilm_irtracklength-1:
    # # #                    ilm_index[0] = 0 # Start of cloud only
    # # #                    ilm_cycle[0:iccarea[0]] = 1 # Time period of cloud only
    #
    # # #                ilm_index[1] = iccarea[0] # Start of unorganized convective cells
    #
    # #             # If convective line exists
    # #             if nccline > 1:
    # #                 # If the convective line occurs after the first storm time (use second index since convective line must be around for one hour prior to classifying as genesis)
    # #                 # Label when convective cores first appear, but are not organized into a line
    # #                 if iccline[1] > 0:
    # #                     ilm_index[2] = iccline[1] # Start of organized convection
    # #                     ilm_cycle[iccline[1]] = 2 # Time period of unorganzied convective cells
    # #                 else:
    # #                     sys.exit('Check convective line in track ' + str(int(ilongmcs[ilm])))
    #
    # #             ############################################################
    # #             # Final life cycle processing
    # #             istage = np.array(np.where(ilm_cycle >= 0))[0, :]
    # #             nstage = len(istage)
    #
    # #             if nstage > 0:
    # #                 cyclepresent = np.copy(ilm_cycle[istage])
    # #                 uniquecycle = np.unique(cyclepresent)
    #
    # #                 # Label as complete cycle if 1-4 present
    # #                 if len(uniquecycle) >= 4:
    # #                     cycle_complete[ilongmcs[ilm]] = 1
    #
    # #                 # Save data
    # #                 cycle_stage[ilongmcs[ilm], :] = np.copy(ilm_cycle)
    # #                 cycle_index[ilongmcs[ilm], :] = np.copy(ilm_index)

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
    dsout.attrs["coefs_pf_area"] = coefs_pf_area
    dsout.attrs["coefs_pf_rr"] = coefs_pf_rr
    dsout.attrs["coefs_pf_skew"] = coefs_pf_skew
    dsout.attrs["coefs_pf_heavyratio"] = coefs_pf_heavyratio
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