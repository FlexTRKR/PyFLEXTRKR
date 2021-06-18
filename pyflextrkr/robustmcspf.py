# Purpose: Filter MCS using NMQ radar variables so that only MCSs statisfying radar thresholds are retained. The lifecycle of these robust MCS is also identified. Method similar to Coniglio et al (2010) MWR.

# Author: Original IDL code written by Zhe Feng (zhe.feng@pnnl.gov),
# Python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov),
# altered by Katelyn Barber (katelyn.barber@pnnl.gov)


def define_robust_mcs_pf(
    stats_path,
    pfstats_filebase,
    startdate,
    enddate,
    timeresolution,
    geolimits,
    mcs_pf_majoraxisthresh,
    mcs_pf_durationthresh,
    aspectratiothresh,
    lifecyclethresh,
    lengththresh,
    gapthresh,
    coefs_area,
    coefs_rr,
    coefs_skew,
    coefs_heavyratio,
    max_pf_majoraxis_thresh=5000,
):
    # Inputs:
    # stats_path - directory which stores this statistics data. this is where the output from this code will be placed
    # pfstats_filebase - file header of the precipitation feature statistics file generated in the previous code.
    # startdate - starting date and time of the data
    # enddate - ending date and time of the data
    # time_resolution - time resolution of the satellite and radar data
    # geolimits - 4-element array with plotting boundaries [lat_min, lon_min, lat_max, lon_max]
    # mcs_pf_majoraxisthresh - minimum major axis length of the largest precipitation feature in a robust MCSs
    # mcs_pf_durationthresh - minimum length of precipitation feature in a robust MCS
    # aspectratiothresh - minimum aspect ratio the largest precipitation feature must have to called a squall line
    # lifecyclethresh - minimum duration required for the lifecycles of an MCS to be classified
    # lengththresh - minimum size required for the lifecycles of an MCS to be classified
    # gapthresh - minimum allowable time gap between precipitation features that allow the storm to still be classified as a robust MCS

    # Outputs: (One netcdf file with satellite, radar, and rain accumulation statistics about MCSs that satisfy both satellite and radar requirements)
    # MCS_length - duration of MCS
    # mcs_type - flag indicating whether this is squall line, based on satellite definition
    # pf_lifetime - length of time in which precipitation is observed during each track
    # status - flag indicating the evolution of each cloud in a MCS
    # startstatus - flag indicating how a MCS starts
    # endstatus - flag indicating how a MCS ends
    # interruptions - flag indicating if the satellite data used to indentify this MCS is incomplete
    # boundary - flag indicating if a MCS touches the edge of the domain
    # base_time - seconds since 1970-01-01 for each cloud in a MCS
    # datetimestring - string of date and time of each cloud in a MCS
    # meanlat - mean latitude of the MCS
    # meanlon - mean longitude of the MCS
    # core_area - area of the core of MCS
    # ccs_area - area of the core and cold anvil of the MCS
    # cloudnumber - numbers indicating which clouds in the cloudid files are associated with a MCS
    # mergecloudnumber - numbers indicating which clouds in the cloudid files merge into this track
    # splitcloudnumber - numbers indicating which clouds in the cloudid files split from this track
    # pf_mcsstatus - flag indicating which part of the tracks are part of the robust MCS
    # lifecycle_complete_flag - flag indicating if this MCS has each element in the MCS life cycle
    # lifecycle_index - time index when each phase of the MCS life cycle starts
    # lifecycle_stage - flag indicating the lifestage of each cloud in a robust MCS
    # nmq_frac - fraction of the cloud that exists within the radar domain
    # npf - number of precipitation features at each time
    # pf_area - area of the precipitation features in a MCS
    # pf_lon - mean longitudes of the precipitaiton features in a MCS
    # pf_lat - mean latitudes of the precipitation features in a MCS
    # pf_rainrate - mean rainrates of the precipition features in a MCS
    # pf_skewness - skewness of the rainrains in precipitation features in a MCS
    # pf_majoraxislength - major axis lengths of the precipitation features in a MCS
    # pf_minoraxislength - minor axis lengths of the precipitation features in a MCS
    # pf_aspectratio - aspect ratios of the precipitation features in a MCS
    # pf_eccentricity - eccentricity of the precipitation features in a MCS
    # pf_orientation - angular position of the precipitation deatures in a MCS

    ######################################################
    # Import modules
    import numpy as np
    import xarray as xr
    import sys
    import time
    import warnings
    import logging
    import pandas as pd

    np.set_printoptions(threshold=np.inf)
    logger = logging.getLogger(__name__)


    ######################################################
    # Load mergedir mcs and pf data
    mergedirpf_statistics_file = (
        stats_path + pfstats_filebase + startdate + "_" + enddate + ".nc"
    )
    logger.info(("mergedirpf_statistics_file: ", mergedirpf_statistics_file))

    data = xr.open_dataset(mergedirpf_statistics_file, decode_times=False)
    ntracks = np.nanmax(data.coords["tracks"])
    ntimes = len(data.coords["times"])
    ncores = len(data.coords["cores"])

    ir_tracklength = data["length"].data
    pf_area = data["pf_area"].data
    pf_majoraxis = data["pf_majoraxislength"].data
    pf_rainrate = data["pf_rainrate"].data
    pf_skewness = data["pf_skewness"].data
    # pf_accumrain = data['pf_accumrain'].data
    # pf_accumrainheavy = data['pf_accumrainheavy'].data
    time_res = float(data.attrs["time_resolution_hour"])
    # if time_res > 5:
    #     time_res = (time_res)/60 # puts time res into hr
    # logger.info(time_res)
    mcs_ir_areathresh = float(data.attrs["MCS_IR_area_thresh_km2"])
    mcs_ir_durationthresh = float(data.attrs["MCS_IR_duration_thresh_hr"])
    mcs_ir_eccentricitythresh = float(data.attrs["MCS_IR_eccentricity_thres"])
    missing_val = data.attrs["missing_value"]
    basetime = data["base_time"].data

    # Calculate accumulate rain by summing over all PFs
    pf_volrain_all = data["pf_accumrain"].sum(dim="pfs").data
    pf_volrain_heavy = data["pf_accumrainheavy"].sum(dim="pfs").data

    ##################################################
    # Initialize matrices
    trackid_mcs = []
    trackid_nonmcs = []

    pf_mcstype = np.ones(ntracks, dtype=int) * missing_val
    pf_mcsstatus = np.ones((ntracks, ntimes), dtype=int) * missing_val

    ###################################################
    # Loop through each track
    for nt in range(0, ntracks):
        logger.info(("Track # " + str(nt)))

        ############################################
        # Isolate data from this track
        ilength = np.copy(ir_tracklength[nt]).astype(int)

        # Get the largest precipitation (1st entry in 3rd dimension)
        ipf_majoraxis = np.copy(pf_majoraxis[nt, 0:ilength, 0])
        ipf_area = np.copy(pf_area[nt, 0:ilength, 0])
        ipf_rainrate = np.copy(pf_rainrate[nt, 0:ilength, 0])
        ipf_skewness = np.copy(pf_skewness[nt, 0:ilength, 0])
        ipf_volrainall = np.copy(pf_volrain_all[nt, 0:ilength])
        ifp_volrainheavy = np.copy(pf_volrain_heavy[nt, 0:ilength])
        # import pdb; pdb.set_trace()

        # logger.info(ipf_majoraxis)
        # logger.info(ipf_rainrate)

        ######################################################
        # Apply precip defined MCS criteria

        # Apply PF major axis length > thresh and contains rainrates >= 1 mm/hr criteria
        # ipfmcs = np.array(np.where((ipf_majoraxis > mcs_pf_majoraxisthresh) & (ipf_rainrate > 1)))[0, :]
        ipfmcs = np.array(
            np.where(
                (ipf_majoraxis > mcs_pf_majoraxisthresh)
                & (ipf_majoraxis < max_pf_majoraxis_thresh)
            )
        )[0, :]
        nipfmcs = len(ipfmcs)
        # logger.info(nipfmcs)
        # logger.info(nipfmcs*time_res)
        # logger.info(mcs_pf_durationthresh)

        if nipfmcs > 0:
            # Apply duration threshold to entire time period
            if nipfmcs * time_res > mcs_pf_durationthresh:

                # Find continuous duration indices
                groups = np.split(
                    ipfmcs, np.where(np.diff(ipfmcs) > gapthresh)[0] + 1
                )  # KB CHANGED != to >
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

                    # Compute PF fit values using the coefficients
                    mcs_pfarea = coefs_area[0] + coefs_area[1] * igroup_duration
                    mcs_rrskew = coefs_skew[0] + coefs_skew[1] * igroup_duration
                    mcs_rravg = coefs_rr[0] + coefs_rr[1] * igroup_duration
                    mcs_heavyratio = (
                        coefs_heavyratio[0] + coefs_heavyratio[1] * igroup_duration
                    )

                    # Group satisfies duration threshold
                    # if np.multiply(len(groups[igroup][:]), time_res) > mcs_pf_durationthresh:
                    if igroup_duration >= mcs_pf_durationthresh:  # KB CHANGED

                        # Get PF variables for this group
                        # igroup_duration = len(groups[igroup])*time_res
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
                            logger.info("MCS")
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
                        logger.info("Not MCS")

            # Group does not satistfy duration threshold
            else:
                trackid_nonmcs = np.append(trackid_nonmcs, int(nt))
                logger.info("Not NCS")
        else:
            logger.info("Not MCS")

    # Isolate tracks that are robust MCS
    TEMP_mcsstatus = np.copy(pf_mcsstatus).astype(float)
    TEMP_mcsstatus[TEMP_mcsstatus == missing_val] = np.nan
    trackid_mcs = np.array(np.where(np.nansum(TEMP_mcsstatus, axis=1)))[0, :]
    nmcs = len(trackid_mcs)

    # Stop code if not robust MCS present
    if nmcs == 0:
        sys.exit("No MCS found!")
    else:
        logger.info(("Number of robust MCS: " + str(int(nmcs))))

    # Isolate data associated with robust MCS
    ir_tracklength = ir_tracklength[trackid_mcs]
    mcs_basetime = basetime[trackid_mcs]
    pf_mcsstatus = pf_mcsstatus[trackid_mcs, :]
    pf_majoraxis = pf_majoraxis[trackid_mcs, :, :]
    pf_area = pf_area[trackid_mcs, :, :]

    # Determine how long MCS track criteria is satisfied
    TEMP_mcsstatus = np.copy(pf_mcsstatus).astype(float)
    TEMP_mcsstatus[TEMP_mcsstatus == missing_val] = np.nan
    mcs_length = np.nansum(TEMP_mcsstatus, axis=1)

    # Get lifetime when a significant precip feature is present
    warnings.filterwarnings("ignore")
    pf_maxmajoraxis = np.nanmax(pf_majoraxis, axis=2)  # Creates run time warning
    pf_maxmajoraxis[pf_maxmajoraxis < lengththresh] = 0
    pf_maxmajoraxis[pf_maxmajoraxis > lengththresh] = 1
    pf_lifetime = np.multiply(np.nansum(pf_maxmajoraxis, axis=1), timeresolution)

    ########################################################
    # Definite life cycle stages. This part is incomplete.
    # Should implement Zhixiao Zhang's Tb-based lifecycle definition code here down the road.

    # Process only MCSs that last >= lifecyclethresh
    lifetime = np.multiply(ir_tracklength, time_res)
    ilongmcs = np.array(np.where(lifetime >= lifecyclethresh))[0, :]
    nlongmcs = len(ilongmcs)

    if nlongmcs > 0:
        # logger.info('ENTERED NLONGMCS IF STATEMENT LINES 214')
        # Initialize arrays
        cycle_complete = np.ones(nmcs, dtype=int) * missing_val
        cycle_stage = np.ones((nmcs, ntimes), dtype=int) * missing_val
        cycle_index = np.ones((nmcs, 5), dtype=int) * missing_val

        # mcs_basetime = np.empty((nmcs, ntimes), dtype='datetime64[s]')
        # logger.info(mcs_basetime)

        # Loop through each mcs
        for ilm in range(0, nlongmcs):
            # Initialize arrays
            ilm_index = np.ones(5, dtype=int) * missing_val
            ilm_cycle = np.ones(ntimes, dtype=int) * missing_val

            # Isolate data from this track
            ilm_irtracklength = np.copy(ir_tracklength[ilongmcs[ilm]]).astype(int)
            ilm_pfarea = np.copy(pf_area[ilongmcs[ilm], 0:ilm_irtracklength, 0])
            ilm_pfmajoraxis = np.copy(
                pf_majoraxis[ilongmcs[ilm], 0:ilm_irtracklength, 0]
            )
            ilm_maxpfmajoraxis = np.nanmax(ilm_pfmajoraxis, axis=0)

            # Get basetime
            # TEMP_basetime = np.array([pd.to_datetime(data['basetime'][trackid_mcs[ilongmcs[ilm]], 0:ilm_irtracklength].data, unit='s')])
            # mcs_basetime[ilm, 0:ilm_irtracklength] = TEMP_basetime

            ##################################################################
            # Find indices of when convective line present and absent and when stratiform present

    #            # Find times with convective core area > 0
    #            iccarea = np.array(np.where(ilm_maxpfccarea > 0))[0, :]
    #            iccarea_groups = np.split(iccarea, np.where(np.diff(iccarea) > 2)[0]+1)
    #            if len(iccarea) > 0 and len(iccarea_groups) > 1:
    #                grouplength = np.empty(len(iccarea_groups))
    #                for igroup in range(0, len(iccarea_groups)):
    #                    grouplength[igroup] = len(iccarea_groups[igroup][:])
    #                maxgroup = np.nanargmax(grouplength)
    #                iccarea = iccarea_groups[maxgroup][:]
    #            elif len(iccarea) > 0 :
    #                iccarea = np.arange(iccarea[0], iccarea[-1]+1)
    #            nccarea = len(iccarea)

    #             # Find times with major axis length > mcs_pf_majoraxisthresh
    #             # iccline = np.array(np.where(ilm_maxpfmajoraxis > mcs_pf_majoraxisthresh))[0, :]
    #             # iccline_groups = np.split(iccline, np.where(np.diff(iccline) > 2)[0]+1)
    #             iccline = np.array(np.where(ilm_pfmajoraxis > mcs_pf_majoraxisthresh))[0, :]
    #             iccline_groups = np.split(iccline, np.where(np.diff(iccline) > gapthresh)[0] + 1)
    #             if len(iccline) > 0 and len(iccline_groups) > 1:
    #                 grouplength = np.empty(len(iccline_groups))
    #                 for igroup in range(0, len(iccline_groups)):
    #                     grouplength[igroup] = len(iccline_groups[igroup][:])
    #                 maxgroup = np.nanargmax(grouplength)
    #                 iccline = iccline_groups[maxgroup][:]
    #             elif len(iccline) > 0:
    #                 iccline = np.arange(iccline[0], iccline[-1]+1)
    #             nccline = len(iccline)

    #             ###############################################################################
    #             # Classify cloud only stage

    #             # Cloud only stage
    # #            if nccarea > 0:
    # #                # If first convective time is after the first cloud time, label all hours before the convective core appearance time as preconvective
    # #                if iccarea[0] > 0 and iccarea[0] < ilm_irtracklength-1:
    # #                    ilm_index[0] = 0 # Start of cloud only
    # #                    ilm_cycle[0:iccarea[0]] = 1 # Time period of cloud only

    # #                ilm_index[1] = iccarea[0] # Start of unorganized convective cells

    #             # If convective line exists
    #             if nccline > 1:
    #                 # If the convective line occurs after the first storm time (use second index since convective line must be around for one hour prior to classifying as genesis)
    #                 # Label when convective cores first appear, but are not organized into a line
    #                 if iccline[1] > 0:
    #                     ilm_index[2] = iccline[1] # Start of organized convection
    #                     ilm_cycle[iccline[1]] = 2 # Time period of unorganzied convective cells
    #                 else:
    #                     sys.exit('Check convective line in track ' + str(int(ilongmcs[ilm])))

    #             ############################################################
    #             # Final life cycle processing
    #             istage = np.array(np.where(ilm_cycle >= 0))[0, :]
    #             nstage = len(istage)

    #             if nstage > 0:
    #                 cyclepresent = np.copy(ilm_cycle[istage])
    #                 uniquecycle = np.unique(cyclepresent)

    #                 # Label as complete cycle if 1-4 present
    #                 if len(uniquecycle) >= 4:
    #                     cycle_complete[ilongmcs[ilm]] = 1

    #                 # Save data
    #                 cycle_stage[ilongmcs[ilm], :] = np.copy(ilm_cycle)
    #                 cycle_index[ilongmcs[ilm], :] = np.copy(ilm_index)

    #################################################################################
    # Save data to netcdf file
    statistics_outfile = (
        stats_path
        + "robust_mcs_tracks_"
        + data.attrs["startdate"]
        + "_"
        + data.attrs["enddate"]
        + ".nc"
    )

    # Define xarrray dataset
    output_data = xr.Dataset(
        {
            "mcs_length": (["tracks"], data["length"][trackid_mcs]),
            "mcs_type": (["tracks"], data["mcs_type"][trackid_mcs]),
            "pf_lifetime": (["tracks"], pf_lifetime),
            "status": (["tracks", "times"], data["status"][trackid_mcs, :]),
            "startstatus": (["tracks"], data["startstatus"][trackid_mcs]),
            "endstatus": (["tracks"], data["endstatus"][trackid_mcs]),
            "interruptions": (
                ["tracks"],
                data["interruptions"][trackid_mcs],
            ),  #   'boundary': (['tracks'], data['boundary'][trackid_mcs]), \
            "base_time": (["tracks", "times"], mcs_basetime),
            "datetimestring": (
                ["tracks", "times", "characters"],
                data["datetimestring"][trackid_mcs, :, :],
            ),
            "meanlat": (["tracks", "times"], data["meanlat"][trackid_mcs, :]),
            "meanlon": (["tracks", "times"], data["meanlon"][trackid_mcs, :]),
            "core_area": (["tracks", "times"], data["core_area"][trackid_mcs, :]),
            "cloudnumber": (["tracks", "times"], data["cloudnumber"][trackid_mcs, :]),
            "mergecloudnumber": (
                ["tracks", "times", "mergesplit"],
                data["mergecloudnumber"][trackid_mcs, :, :],
            ),
            "splitcloudnumber": (
                ["tracks", "times", "mergesplit"],
                data["splitcloudnumber"][trackid_mcs, :, :],
            ),  #'pf_mcsstatus': (['tracks', 'times'], pf_mcsstatus), \
            #'lifecycle_complete_flag': (['tracks'], cycle_complete), \
            #'lifecycle_index': (['tracks', 'lifestages'], cycle_index), \
            #'lifecycle_stage': (['tracks', 'times'], cycle_stage), \
            #   'pf_frac': (['tracks', 'times'], data['pf_frac'][trackid_mcs]), \
            "npf": (["tracks", "times"], data["pf_npf"][trackid_mcs]),
            "pf_area": (["tracks", "times", "pfs"], data["pf_area"][trackid_mcs, :, :]),
            "pf_lon": (["tracks", "times", "pfs"], data["pf_lon"][trackid_mcs, :, :]),
            "pf_lat": (["tracks", "times", "pfs"], data["pf_lat"][trackid_mcs, :, :]),
            "pf_rainrate": (
                ["tracks", "times", "pfs"],
                data["pf_rainrate"][trackid_mcs, :, :],
            ),
            "pf_skewness": (
                ["tracks", "times", "pfs"],
                data["pf_skewness"][trackid_mcs, :, :],
            ),
            "pf_majoraxislength": (
                ["tracks", "times", "pfs"],
                data["pf_majoraxislength"][trackid_mcs, :, :],
            ),
            "pf_minoraxislength": (
                ["tracks", "times", "pfs"],
                data["pf_minoraxislength"][trackid_mcs, :, :],
            ),
            "pf_aspectratio": (
                ["tracks", "times", "pfs"],
                data["pf_aspectratio"][trackid_mcs, :, :],
            ),
            "pf_eccentricity": (
                ["tracks", "times", "pfs"],
                data["pf_eccentricity"][trackid_mcs, :, :],
            ),
            "pf_orientation": (
                ["tracks", "times", "pfs"],
                data["pf_orientation"][trackid_mcs, :, :],
            ),
        },
        coords={
            "tracks": (["tracks"], np.arange(1, len(trackid_mcs) + 1)),
            "times": (["times"], data.coords["times"]),
            "pfs": (["pfs"], data.coords["pfs"]),
            "cores": (["cores"], data.coords["cores"]),
            "mergesplit": (["mergesplit"], data.coords["mergesplit"]),
            "characters": (["characters"], data.coords["characters"]),
            "lifestages": (["lifestages"], np.arange(0, 5)),
        },
        attrs={
            "title": "Statistics of MCS defined using WRF precipitation features",
            "source1": data.attrs["source1"],
            "source2": data.attrs["source2"],
            "description": data.attrs["description"],
            "startdate": data.attrs["startdate"],
            "enddate": data.attrs["enddate"],
            "time_resolution_hour": data.attrs["time_resolution_hour"],
            "mergedir_pixel_radius": data.attrs["mergdir_pixel_radius"],
            "MCS_IR_area_km2": data.attrs["MCS_IR_area_thresh_km2"],
            "MCS_IR_duration_hr": data.attrs["MCS_IR_duration_thresh_hr"],
            "MCS_IR_eccentricity": data.attrs["MCS_IR_eccentricity_thres"],
            "max_number_pfs": data.attrs["max_number_pfs"],
            "MCS_PF_majoraxis_km": mcs_pf_majoraxisthresh,
            "MCS_PF_duration_hr": mcs_pf_durationthresh,
            "MCS_core_aspectratio": aspectratiothresh,
            "contact": "Katelyn Barber: katelyn.barber@pnnl.gov",
            "created_on": time.ctime(time.time()),
        },
    )

    # Specify variable attributes
    output_data["tracks"].attrs["description"] = "Total number of tracked features"
    output_data["tracks"].attrs["units"] = "unitless"

    output_data["times"].attrs[
        "description"
    ] = "Maximum number of features in a given track"
    output_data["times"].attrs["units"] = "unitless"

    output_data["pfs"].attrs[
        "long_name"
    ] = "Maximum number of precipitation features in one cloud feature"
    output_data["pfs"].attrs["units"] = "unitless"

    output_data.cores.attrs[
        "long_name"
    ] = "Maximum number of convective cores in a precipitation feature at one time"
    output_data.cores.attrs["units"] = "unitless"

    output_data.mergesplit.attrs[
        "long_name"
    ] = "Maximum number of mergers / splits at one time"
    output_data.mergesplit.attrs["units"] = "unitless"

    output_data.characters.attrs[
        "description"
    ] = "Number of characters in the date-time string"
    output_data.characters.attrs["units"] = "unitless"

    # output_data.lifestages.attrs['description'] = 'Number of MCS life stages'
    # output_data.lifestages.attrs['units'] = 'unitless'

    output_data.mcs_length.attrs["long_name"] = "Length of each MCS in each track"
    output_data.mcs_length.attrs["units"] = "Temporal resolution of orginal data"

    output_data.mcs_type.attrs["long_name"] = "Type of MCS"
    output_data.mcs_type.attrs["values"] = "1 = MCS, 2 = Squall line"
    output_data.mcs_type.attrs["units"] = "unitless"

    output_data.pf_lifetime.attrs[
        "long_name"
    ] = "Length of time in which precipitation is observed during each track"
    output_data.pf_lifetime.attrs["units"] = "hr"

    output_data.status.attrs[
        "long_name"
    ] = "Flag indicating the status of each feature in MCS"
    output_data.status.attrs["values"] = (
        f"{missing_val}=missing cloud or cloud removed due to short track, "
        + "0=track ends here, 1=cloud continues as one cloud in next file, "
        + "2=Biggest cloud in merger, 21=Smaller cloud(s) in merger, 13=Cloud that splits, "
        + "3=Biggest cloud from a split that stops after the split, "
        + "31=Smaller cloud(s) from a split that stop after the split. "
        + "The last seven classifications are added together in different combinations to describe situations."
    )
    output_data.status.attrs["min_value"] = 0
    output_data.status.attrs["max_value"] = 52
    output_data.status.attrs["units"] = "unitless"

    output_data.startstatus.attrs[
        "long_name"
    ] = "Flag indicating the status of first feature in MCS track"
    output_data.startstatus.attrs["values"] = (
        f"{missing_val}=missing cloud or cloud removed due to short track, "
        + "0=track ends here, 1=cloud continues as one cloud in next file, "
        + "2=Biggest cloud in merger, 21=Smaller cloud(s) in merger, 13=Cloud that splits, "
        + "3=Biggest cloud from a split that stops after the split, "
        + "31=Smaller cloud(s) from a split that stop after the split. "
        + "The last seven classifications are added together in different combinations to describe situations."
    )
    output_data.startstatus.attrs["min_value"] = 0
    output_data.startstatus.attrs["max_value"] = 52
    output_data.startstatus.attrs["units"] = "unitless"

    output_data.endstatus.attrs[
        "long_name"
    ] = "Flag indicating the status of last feature in MCS track"
    output_data.endstatus.attrs["values"] = (
        f"{missing_val}=missing cloud or cloud removed due to short track,  "
        + "0=track ends here, 1=cloud continues as one cloud in next file,  "
        + "2=Biggest cloud in merger, 21=Smaller cloud(s) in merger, 13=Cloud that splits,  "
        + "3=Biggest cloud from a split that stops after the split,  "
        + "31=Smaller cloud(s) from a split that stop after the split.  "
        + "The last seven classifications are added together in different combinations to describe situations."
    )
    output_data.endstatus.attrs["min_value"] = 0
    output_data.endstatus.attrs["max_value"] = 52
    output_data.endstatus.attrs["units"] = "unitless"

    output_data.interruptions.attrs["long_name"] = "flag indicating if track incomplete"
    output_data.interruptions.attrs["values"] = (
        "0 = full track available, good data.  "
        + "1 = track starts at first file, track cut short by data availability.  "
        + "2 = track ends at last file, track cut short by data availability"
    )
    output_data.interruptions.attrs["min_value"] = 0
    output_data.interruptions.attrs["max_value"] = 2
    output_data.interruptions.attrs["units"] = "unitless"

    # output_data.boundary.attrs['long_name'] = 'Flag indicating whether the core + cold anvil touches one of the domain edges.'
    # output_data.boundary.attrs['values'] = '0 = away from edge. 1= touches edge.'
    # output_data.boundary.attrs['min_value'] = 0
    # output_data.boundary.attrs['max_value'] = 1
    # output_data.boundary.attrs['units'] = 'unitless'

    output_data.base_time.attrs["standard_name"] = "time"
    output_data.base_time.attrs[
        "long_name"
    ] = "seconds since 01/01/1970 00:00 for each cloud in the mcs"

    output_data.datetimestring.attrs[
        "long_name"
    ] = "date_time for each cloud in the mcs"
    output_data.datetimestring.attrs["units"] = "unitless"

    output_data.meanlon.attrs["standard_name"] = "longitude"
    output_data.meanlon.attrs[
        "long_name"
    ] = "mean longitude of the core + cold anvil for each feature at the given time"
    output_data.meanlon.attrs["min_value"] = geolimits[1]
    output_data.meanlon.attrs["max_value"] = geolimits[3]
    output_data.meanlon.attrs["units"] = "degrees"

    output_data.meanlat.attrs["standard_name"] = "latitude"
    output_data.meanlat.attrs[
        "long_name"
    ] = "mean latitude of the core + cold anvil for each feature at the given time"
    output_data.meanlat.attrs["min_value"] = geolimits[0]
    output_data.meanlat.attrs["max_value"] = geolimits[2]
    output_data.meanlat.attrs["units"] = "degrees"

    output_data.core_area.attrs["long_name"] = "area of the cold core at the given time"
    output_data.core_area.attrs["units"] = "km^2"

    output_data.cloudnumber.attrs[
        "long_name"
    ] = "cloud number in the corresponding cloudid file of clouds in the mcs"
    output_data.cloudnumber.attrs["usage"] = (
        "to link this tracking statistics file with pixel-level cloudid files,  "
        + "use the cloudidfile and cloudnumber together to identify which cloud this current track and time is associated with"
    )
    output_data.cloudnumber.attrs["units"] = "unitless"

    output_data.mergecloudnumber.attrs[
        "long_name"
    ] = "cloud number of small, short-lived clouds merging into the MCS"
    output_data.mergecloudnumber.attrs["usage"] = (
        "to link this tracking statistics file with pixel-level cloudid files,  "
        + "use the cloudidfile and cloudnumber together to identify which cloud this current track and time is associated with"
    )
    output_data.mergecloudnumber.attrs["units"] = "unitless"

    output_data.splitcloudnumber.attrs[
        "long_name"
    ] = "cloud number of small, short-lived clouds splitting from the MCS"
    output_data.splitcloudnumber.attrs["usage"] = (
        "to link this tracking statistics file with pixel-level cloudid files, "
        + "use the cloudidfile and cloudnumber together to identify which cloud this current track and time is associated with"
    )
    output_data.splitcloudnumber.attrs["units"] = "unitless"

    # output_data.pf_frac.attrs['long_name'] = 'fraction of cold cloud shielf covered by NMQ mask'
    # output_data.pf_frac.attrs['min_value'] = 0
    # output_data.pf_frac.attrs['max_value'] = 1
    # output_data.pf_frac.attrs['units'] = 'unitless'

    output_data.npf.attrs[
        "long_name"
    ] = "number of precipitation features at a given time"
    output_data.npf.attrs["units"] = "unitless"

    output_data.pf_area.attrs[
        "long_name"
    ] = "area of each precipitation feature at a given time"
    output_data.pf_area.attrs["units"] = "km^2"

    output_data.pf_lon.attrs["standard_name"] = "longitude"
    output_data.pf_lon.attrs[
        "long_name"
    ] = "mean longitude of each precipitaiton feature at a given time"
    output_data.pf_lon.attrs["units"] = "degrees"

    output_data.pf_lat.attrs["standard_name"] = "latitude"
    output_data.pf_lat.attrs[
        "long_name"
    ] = "mean latitude of each precipitaiton feature at a given time"
    output_data.pf_lat.attrs["units"] = "degrees"

    output_data.pf_rainrate.attrs[
        "long_name"
    ] = "mean precipitation rate (from rad_hsr_1h) pf each precipitation feature at a given time"
    output_data.pf_rainrate.attrs["units"] = "mm/hr"

    output_data.pf_skewness.attrs[
        "long_name"
    ] = "skewness of each precipitation feature at a given time"
    output_data.pf_skewness.attrs["units"] = "unitless"

    output_data.pf_majoraxislength.attrs[
        "long_name"
    ] = "major axis length of each precipitation feature at a given time"
    output_data.pf_majoraxislength.attrs["units"] = "km"

    output_data.pf_minoraxislength.attrs[
        "long_name"
    ] = "minor axis length of each precipitation feature at a given time"
    output_data.pf_minoraxislength.attrs["units"] = "km"

    output_data.pf_aspectratio.attrs[
        "long_name"
    ] = "aspect ratio (major axis / minor axis) of each precipitation feature at a given time"
    output_data.pf_aspectratio.attrs["units"] = "unitless"

    output_data.pf_eccentricity.attrs[
        "long_name"
    ] = "eccentricity of each precipitation feature at a given time"
    output_data.pf_eccentricity.attrs["min_value"] = 0
    output_data.pf_eccentricity.attrs["max_value"] = 1
    output_data.pf_eccentricity.attrs["units"] = "unitless"

    output_data.pf_orientation.attrs[
        "long_name"
    ] = "orientation of the major axis of each precipitation feature at a given time"
    output_data.pf_orientation.attrs["units"] = "degrees clockwise from vertical"
    output_data.pf_orientation.attrs["min_value"] = 0
    output_data.pf_orientation.attrs["max_value"] = 360

    # output_data.pf_mcsstatus.attrs['description'] = 'Flag indicating if this time part of the MCS 1 = Yes, 0 = No'
    # output_data.pf_mcsstatus.attrs['units'] = 'unitless'

    # output_data.lifecycle_complete_flag.attrs['description'] = 'Flag indicating if this MCS has each element in the MCS life cycle'
    # output_data.lifecycle_complete_flag.attrs['units'] = 'unitless'

    # output_data.lifecycle_index.attrs['description'] = 'Time index when each phase of the MCS life cycle starts'
    # output_data.lifecycle_index.attrs['units'] = 'unitless'

    # output_data.lifecycle_stage.attrs['description'] = 'Each time in the MCS is labeled with a flag indicating its phase in the MCS lifecycle. 1 = Cloud only, 2 = Isolated convective cores, 3 = MCS genesis, 4 = MCS maturation, 5 = MCS decay'
    # output_data.lifecycle_stage.attrs['units'] = 'unitless'

    # Write netcdf file
    logger.info("")
    logger.info(statistics_outfile)

    output_data.to_netcdf(
        path=statistics_outfile,
        mode="w",
        format="NETCDF4_CLASSIC",
        unlimited_dims="tracks",
        encoding={
            "mcs_length": {"dtype": "int", "zlib": True, "_FillValue": missing_val},
            "mcs_type": {"dtype": "int", "zlib": True, "_FillValue": missing_val},
            "pf_lifetime": {"dtype": "int", "zlib": True, "_FillValue": missing_val},
            "status": {"dtype": "int", "zlib": True, "_FillValue": missing_val},
            "startstatus": {"dtype": "int", "zlib": True, "_FillValue": missing_val},
            "endstatus": {"dtype": "int", "zlib": True, "_FillValue": missing_val},
            "base_time": {
                "zlib": True
            },  # 'boundary': {'dtype': 'int', 'zlib':True, '_FillValue': missing_val}, \
            "interruptions": {"dtype": "int", "zlib": True, "_FillValue": missing_val},
            "datetimestring": {"zlib": True},
            "meanlat": {"dtype": "float32", "zlib": True, "_FillValue": np.nan},
            "meanlon": {"dtype": "float32", "zlib": True, "_FillValue": np.nan},
            "core_area": {"dtype": "float32", "zlib": True, "_FillValue": np.nan},
            "cloudnumber": {"dtype": "int", "zlib": True, "_FillValue": missing_val},
            "mergecloudnumber": {
                "dtype": "int",
                "zlib": True,
                "_FillValue": missing_val,
            },
            "splitcloudnumber": {
                "dtype": "int",
                "zlib": True,
                "_FillValue": missing_val,
            },  # 'pf_frac': {'dtype': 'float32', 'zlib':True, '_FillValue': np.nan}, \
            "pf_area": {"dtype": "float32", "zlib": True, "_FillValue": np.nan},
            "pf_lon": {"dtype": "float32", "zlib": True, "_FillValue": np.nan},
            "pf_lat": {"dtype": "float32", "zlib": True, "_FillValue": np.nan},
            "pf_rainrate": {"dtype": "float32", "zlib": True, "_FillValue": np.nan},
            "pf_skewness": {"dtype": "float32", "zlib": True, "_FillValue": np.nan},
            "pf_majoraxislength": {
                "dtype": "float32",
                "zlib": True,
                "_FillValue": np.nan,
            },
            "pf_minoraxislength": {
                "dtype": "float32",
                "zlib": True,
                "_FillValue": np.nan,
            },
            "pf_aspectratio": {"dtype": "float32", "zlib": True, "_FillValue": np.nan},
            "pf_orientation": {"dtype": "float32", "zlib": True, "_FillValue": np.nan},
            "pf_eccentricity": {"dtype": "float32", "zlib": True, "_FillValue": np.nan},
        },
    )
    #'pf_eccentricity': {'zlib':True, '_FillValue': np.nan}, \
    #'pf_mcsstatus': {'dtype': 'int', 'zlib':True, '_FillValue': missing_val}, \
    #'lifecycle_complete_flag': {'dtype': 'int', 'zlib':True, '_FillValue': missing_val}, \
    #'lifecycle_index': {'dtype': 'int', 'zlib':True, '_FillValue': missing_val}, \
    #'lifecycle_stage': {'dtype': 'int', 'zlib':True, '_FillValue': missing_val}})
