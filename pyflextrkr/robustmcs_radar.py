import numpy as np
import xarray as xr
import os, shutil
import sys
import time
import warnings
import logging
import pandas as pd

def define_robust_mcs_radar(config):
    """
    Identify robust MCS based on radar statistics.

    Args:
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        statistics_outfile: string
            Robust MCS track statistics file name.
    """
    np.set_printoptions(threshold=np.inf)
    logger = logging.getLogger(__name__)
    logger.info("Identifying robust MCS based on PF statistics")

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
    mcs_lifecycle_thresh = config["mcs_lifecycle_thresh"]
    tracks_dimname = config["tracks_dimname"]
    times_dimname = config["times_dimname"]
    pf_dimname = config["pf_dimname"]
    pixel_radius = config["pixel_radius"]

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
    pf_cc45area = ds_pf["pf_cc45area"].data
    pf_sfarea = ds_pf["pf_sfarea"].data
    pf_corearea = ds_pf["pf_corearea"].data
    pf_coremajoraxis = ds_pf["pf_coremajoraxis"].data
    pf_ccaspectratio = ds_pf["pf_coreaspectratio"].data
    fillval = ds_pf["mcs_status"].attrs["_FillValue"]
    fillval_f = ds_pf["pf_area"].attrs["_FillValue"]
    time_res = float(ds_pf.attrs["time_resolution_hour"])

    ##################################################
    # Initialize matrices
    trackid_mcs = []
    trackid_nonmcs = []

    # pf_mcstype = np.full(ntracks, fillval, dtype=int)
    pf_mcsstatus = np.full((ntracks, ntimes), fillval, dtype=int)
    #pf_cctype = np.full((ntracks, ntimes), fillval, dtype=int)

    ###################################################
    # Loop through each track
    for nt in range(0, ntracks):
        logger.debug(f"Track #: {nt}")

        ############################################
        # Isolate data from this track
        ilength = np.copy(ir_trackduration[nt]).astype(int)

        # Get the largest precipitation (1st entry in 3rd dimension)
        ipf_majoraxis = np.copy(pf_majoraxis[nt, 0:ilength, 0])
        ipf_area = np.copy(pf_area[nt, 0:ilength, 0])
        ipf_cc45area = np.copy(pf_cc45area[nt, 0:ilength, 0])
        ipf_ccmajoraxis = np.copy(pf_coremajoraxis[nt, 0:ilength, 0])
        ipf_ccaspectratio = np.copy(pf_ccaspectratio[nt, 0:ilength, 0])

        ######################################################
        # Apply radar defined MCS criteria
        # PF major axis length > thresh and contains convective echo >= 45 dbZ
        ipfmcs = np.where(
                (ipf_majoraxis >= mcs_pf_majoraxis_thresh)
                # & (ipf_majoraxis <= max_pf_majoraxis_thresh)
                & (ipf_cc45area > 0)
            )[0]
        nipfmcs = len(ipfmcs)

        if nipfmcs > 0:
            # Apply duration threshold to entire time period
            if (nipfmcs * time_res) > mcs_pf_durationthresh:

                # Find continuous duration indices
                groups = np.split(ipfmcs, np.where(np.diff(ipfmcs) > mcs_pf_gap)[0] + 1)
                nbreaks = len(groups)

                # Loop over each sub-period "group"
                for igroup in range(0, nbreaks):

                    ############################################################
                    # Determine if each group satisfies duration threshold
                    igroup_indices = np.array(np.copy(groups[igroup][:]))
                    nigroup = len(igroup_indices)

                    # Duration length: group's last index - first index + 1
                    igroup_duration = np.multiply(
                        (groups[igroup][-1] - groups[igroup][0] + 1), time_res
                    )

                    # Group satisfies duration threshold
                    if igroup_duration >= mcs_pf_durationthresh:

                        # Get radar variables for this group
                        igroup_pfmajoraxis = np.copy(ipf_majoraxis[igroup_indices])
                        igroup_ccaspectratio = np.copy(ipf_ccaspectratio[igroup_indices])

                        # Label this period as MCS
                        pf_mcsstatus[nt, igroup_indices] = 1
                        logger.debug("MCS")

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

            # Group does not satisfy duration threshold
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
        logger.info(f"Number of robust MCS: {nmcs}")

    # Isolate data associated with robust MCS
    ir_trackduration = ir_trackduration[trackid_mcs]

    # mcs_basetime = basetime[trackid_mcs]
    pf_mcsstatus = pf_mcsstatus[trackid_mcs, :]
    pf_majoraxis = pf_majoraxis[trackid_mcs, :, :]
    pf_area = pf_area[trackid_mcs, :, :]
    pf_coremajoraxis = pf_coremajoraxis[trackid_mcs, :, :]
    pf_corearea = pf_corearea[trackid_mcs, :, :]
    pf_sfarea = pf_sfarea[trackid_mcs, :, :]

    # Get lifetime when a significant PF is present
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pf_maxmajoraxis = np.nanmax(pf_majoraxis, axis=2)
        pf_maxmajoraxis[pf_maxmajoraxis < mcs_pf_majoraxis_for_lifetime] = 0
        pf_maxmajoraxis[pf_maxmajoraxis > mcs_pf_majoraxis_for_lifetime] = 1
        pf_lifetime = np.multiply(np.nansum(pf_maxmajoraxis, axis=1), time_res)

    ########################################################
    # Definite life cycle stages. Based on Coniglio et al. (2010) MWR.
    # Preconvective (1): first hour after convective core occurs
    # Genesis (2): First hour after convective line exceeds 100 km
    # Mature (3): Near continuous line with well-defined stratiform precipitation
    # 2 hours after genesis state and 2 hours before decay stage
    # Dissipiation (4): First hour after convective line is no longer observed

    # Process only MCSs with duration > mcs_lifecycle_thresh
    lifetime = np.multiply(ir_trackduration, time_res)
    ilongmcs = np.array(np.where(lifetime >= mcs_lifecycle_thresh))[0, :]
    nlongmcs = len(ilongmcs)

    if nlongmcs > 0:
        # Initialize arrays
        cycle_complete = np.full(nmcs, fillval, dtype=int)
        cycle_stage = np.full((nmcs, ntimes), fillval, dtype=int)
        cycle_index = np.full((nmcs, 5), fillval, dtype=int)

        # TODO: port MCS lifecycle stage definition codes
        # import matplotlib.pyplot as plt
        # import pdb;
        # pdb.set_trace()

        #mcs_basetime = np.empty((nmcs, ntimes), dtype="datetime64[s]")

        # Loop through each mcs
        for ilm in range(0, nlongmcs):
            # Initialize arrays
            ilm_index = np.ones(5, dtype=int) * fillval
            ilm_cycle = np.ones(ntimes, dtype=int) * fillval

            # Isolate data from this track
            ilm_irtracklength = np.copy(ir_trackduration[ilongmcs[ilm]]).astype(int)
            ilm_pfccmajoraxis = np.copy(pf_coremajoraxis[ilongmcs[ilm], 0:ilm_irtracklength, :])
            ilm_pfccarea = np.copy(pf_corearea[ilongmcs[ilm], 0:ilm_irtracklength, :])
            ilm_meansfarea = np.sum(np.copy(pf_sfarea[ilongmcs[ilm], 0:ilm_irtracklength, :]), axis=1)
            ilm_pfarea = np.copy(pf_area[ilongmcs[ilm], 0:ilm_irtracklength, 0])

            ilm_maxpfccmajoraxis = np.nanmax(ilm_pfccmajoraxis, axis=1)
            ilm_maxpfccarea = np.nanmax(ilm_pfccarea, axis=1)

            #TEMP_basetime = np.array(
            #        [
            #            pd.to_datetime(
            #                ds_pf["basetime"][
            #                    trackid_mcs[ilongmcs[ilm]], 0:ilm_irtracklength
            #                ].data,
            #                unit="s",
            #            )
            #        ]
            #)
            #mcs_basetime[ilm, 0:ilm_irtracklength] = TEMP_basetime

            ##################################################################
            # Find indices of when convective line present and absent and when stratiform present

            # Find times with convective core area > 0
            iccarea = np.array(np.where(ilm_maxpfccarea > 0))[0, :]
            iccarea_groups = np.split(iccarea, np.where(np.diff(iccarea) > 2)[0] + 1)
            if len(iccarea) > 0 and len(iccarea_groups) > 1:
                grouplength = np.empty(len(iccarea_groups))
                for igroup in range(0, len(iccarea_groups)):
                    grouplength[igroup] = len(iccarea_groups[igroup][:])
                maxgroup = np.nanargmax(grouplength)
                iccarea = iccarea_groups[maxgroup][:]
            elif len(iccarea) > 0:
                iccarea = np.arange(iccarea[0], iccarea[-1] + 1)
            nccarea = len(iccarea)

            # Find times with convective major axis length greater than 100 km
            iccline = np.array(np.where(ilm_maxpfccmajoraxis > 100))[0, :]
            iccline_groups = np.split(iccline, np.where(np.diff(iccline) > 2)[0] + 1)
            if len(iccline) > 0 and len(iccline_groups) > 1:
                grouplength = np.empty(len(iccline_groups))
                for igroup in range(0, len(iccline_groups)):
                    grouplength[igroup] = len(iccline_groups[igroup][:])
                maxgroup = np.nanargmax(grouplength)
                iccline = iccline_groups[maxgroup][:]
            elif len(iccline) > 0:
                iccline = np.arange(iccline[0], iccline[-1] + 1)
            nccline = len(iccline)

            # Find times with convective major axis length greater than 100 km
            # and stratiform area greater than the median amount of stratiform
            # ilm_meansfarea[ilm_meansfarea == fillvalue] = np.nan
            # print(ilm_maxpfccmajoraxis.shape)
            # print(ilm_meansfarea.shape)
            isfarea = np.array(
                np.where((ilm_maxpfccmajoraxis > 100) &
                         (ilm_meansfarea > np.nanmean(ilm_meansfarea)))
            )[0, :]
            isfarea_groups = np.split(isfarea, np.where(np.diff(isfarea) > 2)[0] + 1)
            if len(isfarea) > 0 and len(isfarea_groups) > 1:
                grouplength = np.empty(len(isfarea_groups))
                for igroup in range(0, len(isfarea_groups)):
                    grouplength[igroup] = len(isfarea_groups[igroup][:])
                maxgroup = np.nanargmax(grouplength)
                isfarea = isfarea_groups[maxgroup][:]
            elif len(isfarea) > 0:
                isfarea = np.arange(isfarea[0], isfarea[-1] + 1)
            nsfarea = len(isfarea)

            # Find times with convective major axis length less than 100 km
            #if nsfarea > 0:
            #    inoccline = np.array(np.where(ilm_maxpfccmajoraxis < 100))[0, :]
            #    inoccline = inoccline[
            #            np.where((inoccline > isfarea[-1]) & (inoccline > iccline[-1]))
            #    ]
            #    inoccline_groups = np.split(
            #            inoccline, np.where(np.diff(inoccline) > 2)[0] + 1
            #    )
            #    if len(inoccline) > 0 and len(inoccline_groups) > 1:
            #        grouplength = np.empty(len(inoccline_groups))
            #        for igroup in range(0, len(inoccline_groups)):
            #            grouplength[igroup] = len(inoccline_groups[igroup][:])
            #        maxgroup = np.nanargmax(grouplength)
            #        inoccline = inoccline_groups[maxgroup][:]
            #    elif len(inoccline) > 0:
            #        inoccline = np.arange(inoccline[0], inoccline[-1] + 1)
            #    nnoccline = len(inoccline)

            ###############################################################################
            # Classify cloud only stage

            # Cloud only stage
            if nccarea > 0:
                # If first convective time is after the first cloud time,
                # label all hours before the convective core appearance time as pre-convective
                if iccarea[0] > 0 and iccarea[0] < ilm_irtracklength - 1:
                    # Start of cloud only
                    ilm_index[0] = 0
                    # Time period of cloud only
                    ilm_cycle[0 : iccarea[0]] = 1
                # Start of unorganized convective cells
                ilm_index[1] = iccarea[0]

            # If convective line exists
            if nccline > 1:
                # If the convective line occurs after the first storm time
                # (use second index since convective line must be around for one hour prior to classifying as genesis)
                # Label when convective cores first appear, but are not organized into a line
                if iccline[1] > iccarea[0]:
                    # Start of organized convection
                    ilm_index[2] = iccline[1]
                    # Time period of unorganzied convective cells
                    ilm_cycle[iccarea[0] : iccline[1]] = 2
                else:
                    logger.warning(f"Lifecycle cannot be properly defined for track: {int(ilongmcs[ilm])}")

                if nsfarea > 0:
                    # Label MCS genesis.
                    # Test if stratiform area time is two time steps after the convective line
                    # and two time steps before the last time of the cloud track
                    if isfarea[0] > iccline[1] + 2:
                        # Start of mature MCS
                        ilm_index[3] = isfarea[0]
                        # Time period of organized cells before mature
                        ilm_cycle[iccline[1] : isfarea[0]] = 3
                        # Time period of mature MCS
                        ilm_cycle[isfarea[0] : isfarea[-1] + 1] = 4
                    else:
                        matureindex = isfarea[np.array(np.where(isfarea == iccline[1] + 2))[0, :]]
                        if len(matureindex) > 0:
                            ilm_index[3] = np.copy(matureindex[0])
                            # Time period of organized cells before mature
                            ilm_cycle[iccline[1] : matureindex[0]] = 3
                            # Time period of mature MCS
                            ilm_cycle[matureindex[0] : isfarea[-1] + 1] = 4

                            # if isfarea[0] > iccline[1] + 2
                            #    ilm_index[3] = isfarea[0] # Start of mature MCS
                            #    ilm_cycle[iccline[1]:isfarea[0]] = 3 # Time period of organized cells before maturation
                            #    ilm_cycle[isfarea[0]:isfarea[-1]+1] = 4 # Time period of mature mcs
                            # else:
                            #    if nsfarea > 3:
                            #        ilm_index[3] = isfarea[3] # Start of mature MCS
                            #        ilm_cycle[iccline[1]:isfarea[3]] = 3 # Time period of organized cells before maturation
                            #        ilm_cycle[isfarea[3]:isfarea[-1]+1] = 4 # Time period of mature MCS
                            # Label dissipating times. Buy default this is all times after the mature stage
                    if isfarea[-1] < ilm_irtracklength - 1:
                        # If nsfarea > 0, the dissipation stage can be defined
                        ilm_index[4] = isfarea[-1] + 1
                        # Time period of dissipation
                        ilm_cycle[isfarea[-1] + 1 : ilm_irtracklength] = 5

            ############################################################
            # Final life cycle processing
            istage = np.array(np.where(ilm_cycle >= 0))[0, :]
            nstage = len(istage)

            if nstage > 0:
                cyclepresent = np.copy(ilm_cycle[istage])
                uniquecycle = np.unique(cyclepresent)

                # Label as complete cycle if 1-4 present
                if len(uniquecycle) >= 4:
                    cycle_complete[ilongmcs[ilm]] = 1

                # Save data
                cycle_stage[ilongmcs[ilm], :] = np.copy(ilm_cycle)
                cycle_index[ilongmcs[ilm], :] = np.copy(ilm_index)

    # Subset robust MCS tracks from PF dataset
    # Note: the tracks_dimname cannot be used here as Xarray does not seem to have
    # a method to select data with a string variable
    dsout = ds_pf.sel(tracks=trackid_mcs)
    # Replace tracks index
    tracks_coord = np.arange(0, nmcs)
    times_coord = ds_pf[times_dimname]
    dsout[tracks_dimname] = tracks_coord
    lifestage_coord = np.arange(0, 5)
    dsout["lifestages"] = lifestage_coord

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

    cycle_complete = xr.DataArray(
            cycle_complete,
            coords={tracks_dimname:tracks_coord},
            dims=(tracks_dimname),
            attrs={
                "long_name": "Flag indicating if this MCS has each element in the MCS life cycle",
                "units": "unitless",
                "_FillValue": fillval,
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

    cycle_stage = xr.DataArray(
            cycle_stage,
            coords={tracks_dimname:tracks_coord, times_dimname:times_coord},
            dims=(tracks_dimname, times_dimname),
            attrs={
                "long_name": "Each time in the MCS is labeled with a flag indicating its phase in the MCS lifecycle. 1 = Cloud only, 2 = Isolated convective cores, 3 = MCS genesis, 4 = MCS maturation, 5 = MCS decay",
                "units": "unitless",
                "_FillValue": fillval,
            }
    )

    cycle_index = xr.DataArray(
            cycle_index,
            coords={tracks_dimname:tracks_coord, "lifestages":lifestage_coord},
            dims=(tracks_dimname, "lifestages"),
            attrs={
                "long_name": "Time index when each phase of the MCS life cycle starts",
                "units": "unitless",
                "_FillValue": fillval,
            }
    )

    # print("OK ljf")

    # Add new variables to dataset
    dsout["pf_lifetime"] = pf_lifetime
    dsout["pf_mcsstatus"] = pf_mcsstatus
    dsout["lifecycle_stage"] = cycle_stage
    dsout["lifecycle_complete_flag"] = cycle_complete
    dsout["lifecycle_index"] = cycle_index

    # Update global attributes
    dsout.attrs["MCS_PF_majoraxis_thresh"] = mcs_pf_majoraxis_thresh
    dsout.attrs["MCS_PF_duration_thresh"] = mcs_pf_durationthresh
    dsout.attrs["PF_PF_min_majoraxis_thresh"] = mcs_pf_majoraxis_for_lifetime
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

    return statistics_outfile
