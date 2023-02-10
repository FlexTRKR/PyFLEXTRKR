import numpy as np
import xarray as xr
import os, shutil
import sys
import time
import warnings
import logging

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
    pf_cctype = np.full((ntracks, ntimes), fillval, dtype=int)

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