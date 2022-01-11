import numpy as np
from netCDF4 import Dataset, num2date
import time
import os
import xarray as xr
import logging

def identifymcs_tb(config):
    """
    Identify MCS using track Tb features.

    Args:
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        mcstrackstatistics_outfile: string
            MCS track statistics file name.
    """

    #######################################################################
    trackstats_filebase = config["trackstats_filebase"]
    stats_path = config["stats_outpath"]
    startdate = config["startdate"]
    enddate = config["enddate"]
    # geolimits,
    time_resolution = config["datatimeresolution"]
    mcs_tb_area_thresh = config["mcs_tb_area_thresh"]
    duration_thresh = config["mcs_tb_duration_thresh"]
    # eccentricity_thresh,
    split_duration = config["mcs_tb_split_duration"]
    merge_duration = config["mcs_tb_merge_duration"]
    nmaxmerge = config["nmaxlinks"]
    timegap = config["mcs_tb_gap"]
    tracks_dimname = config["tracks_dimname"]
    times_dimname = config["times_dimname"]
    fillval = config["fillval"]

    np.set_printoptions(threshold=np.inf)
    # import pdb; pdb.set_trace()

    ##########################################################################

    logger = logging.getLogger(__name__)

    # Load statistics file
    statistics_file = f"{stats_path}{trackstats_filebase}{startdate}_{enddate}.nc"
    logger.debug(statistics_file)


    allstatdata = Dataset(statistics_file, "r")
    # Total number of tracked features
    ntracks_all = np.nanmax(allstatdata[tracks_dimname]) + 1
    # Maximum number of features in a given track
    nmaxlength = np.nanmax(allstatdata[times_dimname]) + 1
    logger.debug(f"nmaxlength:{nmaxlength}")

    trackstat_corearea = allstatdata["core_area"][:]
    trackstat_coldarea = allstatdata["cold_area"][:]
    trackstat_ccsarea = trackstat_corearea + trackstat_coldarea
    track_duration = allstatdata["track_duration"][:]
    basetime = allstatdata["base_time"][:]
    basetime_units = allstatdata["base_time"].units
    mergecloudnumbers = allstatdata["merge_tracknumbers"][:]
    splitcloudnumbers = allstatdata["split_tracknumbers"][:]
    cloudnumbers = allstatdata["cloudnumber"][:]
    status = allstatdata["track_status"][:]
    endstatus = allstatdata["end_status"][:]
    startstatus = allstatdata["start_status"][:]
    track_interruptions = allstatdata["track_interruptions"][:]
    meanlat = allstatdata["meanlat"][:]
    meanlon = allstatdata["meanlon"][:]

    # nconv = allstatdata["nconv"][:]
    # ncoldanvil = allstatdata["ncoldanvil"][:]
    # lifetime = allstatdata["lifetime"][:]
    # eccentricity = allstatdata["eccentricity"][:]
    # basetime = allstatdata["basetime"][:]
    # basetime_units = allstatdata["basetime"].units
    # basetime_calendar = allstatdata['basetime'].calendar
    # mergecloudnumbers = allstatdata["mergenumbers"][:]
    # splitcloudnumbers = allstatdata["splitnumbers"][:]
    # cloudnumbers = allstatdata["cloudnumber"][:]
    # status = allstatdata["status"][:]
    # endstatus = allstatdata["endstatus"][:]
    # startstatus = allstatdata["startstatus"][:]
    # datetimestrings = allstatdata["datetimestrings"][:]
    # boundary = allstatdata["boundary"][:]
    # trackinterruptions = allstatdata["trackinterruptions"][:]
    # meanlat = np.array(allstatdata["meanlat"][:])
    # meanlon = np.array(allstatdata["meanlon"][:])
    # pixelradius = allstatdata.getncattr("pixel_radius_km")
    # datasource = allstatdata.getncattr("source")
    # datadescription = allstatdata.getncattr("description")
    allstatdata.close()

    logger.info(f"MCS duration threshold: {duration_thresh}")
    logger.info(f"MCS CCS area threshold: {mcs_tb_area_thresh}")


    ####################################################################
    # Set up thresholds

    # # Cold Cloud Shield (CCS) area
    # trackstat_corearea = np.multiply(nconv, pixelradius ** 2)
    # trackstat_ccsarea = np.multiply(ncoldanvil + nconv, pixelradius ** 2)

    # Convert track duration to physical time unit
    trackstat_duration = np.multiply(track_duration, time_resolution)
    # trackstat_duration = trackstat_duration.astype(np.int32)

    ##################################################################
    # Initialize matrices
    trackid_mcs = []
    # trackid_sql = []
    trackid_nonmcs = []

    mcstype = np.zeros(ntracks_all, dtype=np.int16)
    mcsstatus = np.full((ntracks_all, nmaxlength), fillval, dtype=np.int16)

    ###################################################################
    # Identify MCSs
    logger.info(f"Total number of tracks to check: {ntracks_all}")
    for nt in range(0, ntracks_all):
        # Get data for a given track
        track_corearea = np.copy(trackstat_corearea[nt, :])
        track_ccsarea = np.copy(trackstat_ccsarea[nt, :])
        # track_eccentricity = np.copy(eccentricity[nt, :])

        # Remove fill values
        track_corearea = track_corearea[
            (~np.isnan(track_corearea)) & (track_corearea != 0)
        ]
        track_ccsarea = track_ccsarea[~np.isnan(track_ccsarea)]
        # track_eccentricity = track_eccentricity[~np.isnan(track_eccentricity)]

        # Must have a cold core
        if np.shape(track_corearea)[0] != 0 and np.nanmax(track_corearea > 0):

            # Cold cloud shield area requirement
            iccs = np.array(np.where(track_ccsarea > mcs_tb_area_thresh))[0, :]
            nccs = len(iccs)

            # Find continuous times
            # groups = np.split(iccs, np.where(np.diff(iccs) != 1)[0]+1)  # ORIGINAL
            # KB TESTING TIME GAP (!=1)
            groups = np.split(iccs, np.where(np.diff(iccs) > timegap)[0] + 1)
            nbreaks = len(groups)

            # System may have multiple periods satisfying area and duration requirements. Loop over each period
            if iccs != []:
                for t in range(0, nbreaks):
                    # Duration requirement
                    # Duration length should be group's last index - first index + 1
                    duration_group = np.multiply(
                        (groups[t][-1] - groups[t][0] + 1), time_resolution
                    )
                    if duration_group >= duration_thresh:

                        # Isolate area and eccentricity for the subperiod
                        # subtrack_ccsarea = track_ccsarea[groups[t][:]]
                        # subtrack_eccentricity = track_eccentricity[groups[t][:]]

                        # Get eccentricity when the feature is the largest
                        # subtrack_imax_ccsarea = np.nanargmax(subtrack_ccsarea)
                        # subtrack_maxccsarea_eccentricity = subtrack_eccentricity[
                        #     subtrack_imax_ccsarea
                        # ]

                        mcstype[nt] = 1
                        mcsstatus[nt, groups[t][:]] = 1

                        # # Apply eccentricity requirement
                        # if subtrack_maxccsarea_eccentricity > eccentricity_thresh:
                        #     # Label as MCS
                        #     mcstype[nt] = 1
                        #     mcsstatus[nt, groups[t][:]] = 1
                        # else:
                        #     # Label as squall line
                        #     mcstype[nt] = 2
                        #     mcsstatus[nt, groups[t][:]] = 2
                        #     trackid_sql = np.append(trackid_sql, nt)

                        trackid_mcs = np.append(trackid_mcs, nt)
                    else:
                        # Size requirement met but too short of a period
                        trackid_nonmcs = np.append(trackid_nonmcs, nt)

            else:
                # Size requirement not met
                trackid_nonmcs = np.append(trackid_nonmcs, nt)

    ################################################################
    # Check that there are MCSs to continue processing
    if trackid_mcs == []:
        logger.info("There are no MCSs in the domain, the code will crash")

    ################################################################
    # Subset MCS / Squall track index
    trackid = np.array(np.where(mcstype > 0))[0, :]
    nmcs = len(trackid)
    logger.info(f"Number of Tb defined MCS: {nmcs}")

    if nmcs > 0:
        mcsstatus = mcsstatus[trackid, :]
        mcstype = mcstype[trackid]

        # mcs_duration = np.full(len(mcstype), fillval, dtype=np.int32)
        # for imcs in range(0, nmcs):
        #     mcs_duration[imcs] = len(np.array(np.where(mcsstatus[imcs, :] != fillval))[0, :])
        # Get duration when MCS status is met
        mcs_duration = np.nansum(mcsstatus > 0, axis=1)

    # trackid_mcs is the index number, want the track number so add one
    mcstracknumbers = np.copy(trackid) + 1


    ###############################################################
    # Find small merging and spliting clouds and add to MCS
    mcsmergecloudnumber = np.full((nmcs, nmaxlength, nmaxmerge), fillval, dtype=np.int32)
    mcsmergestatus = np.full((nmcs, nmaxlength, nmaxmerge), fillval, dtype=np.int32)
    mcssplitcloudnumber = np.full((nmcs, nmaxlength, nmaxmerge), fillval, dtype=np.int32)
    mcssplitstatus = np.full((nmcs, nmaxlength, nmaxmerge), fillval, dtype=np.int32)

    # Let's convert 2D to 1D arrays for performance
    split_col = np.nanmax(splitcloudnumbers, axis=1)
    merger_col = np.nanmax(mergecloudnumbers, axis=1)

    # Loop through each MCS and link small clouds merging in
    for imcs in np.arange(0, nmcs):
        ###################################################################################
        # Isolate basetime data
        if imcs == 0:
            mcsbasetime = basetime[trackid[imcs], :]
            # mcsbasetime = np.array([pd.to_datetime(num2date(basetime[trackid[imcs], :], units=basetime_units, calendar=basetime_calendar))], dtype='datetime64[s]')
        else:
            mcsbasetime = np.row_stack((mcsbasetime, basetime[trackid[imcs], :]))
            # mcsbasetime = np.concatenate((mcsbasetime, np.array([pd.to_datetime(num2date(basetime[trackid[imcs], :], units=basetime_units, calendar=basetime_calendar))], dtype='datetime64[s]')), axis=0)
        # if imcs >= 2:
        #     import pdb; pdb.set_trace()

        ###################################################################################
        # Find mergers
        mergefile = np.where(merger_col == mcstracknumbers[imcs])[0]

        for imerger in range(0, len(mergefile)):
            additionalmergefile = np.where(merger_col == mergefile[imerger] + 1)[0]

            if len(additionalmergefile) > 0:
                mergefile = np.concatenate((mergefile, additionalmergefile))

        # Loop through all merging tracks, if present
        if len(mergefile) > 0:
            # Isolate merging cases that have short duration
            mergefile = mergefile[trackstat_duration[mergefile] < merge_duration]

            # Make sure the merger itself is not an MCS
            mergingmcs = np.intersect1d(mergefile, mcstracknumbers)
            if len(mergingmcs) > 0:
                for iremove in np.arange(0, len(mergingmcs)):
                    removemerges = np.array(np.where(mergefile == mergingmcs[iremove]))[0, :]
                    mergefile[removemerges] = fillval
                mergefile = mergefile[mergefile != fillval].astype(int)

            # Continue if mergers satisfy duration and MCS restriction
            if len(mergefile) > 0:

                # Get data about merging tracks
                mergingcloudnumber = np.copy(cloudnumbers[mergefile, :])
                mergingbasetime = np.copy(basetime[mergefile, :])
                mergingstatus = np.copy(status[mergefile, :])
                # mergingdatetime = np.copy(datetimestrings[mergefile, :, :])

                # Get data about MCS track
                imcsbasetime = np.copy(
                    basetime[int(mcstracknumbers[imcs]) - 1, 0 : int(mcs_duration[imcs])]
                )

                # Loop through each timestep in the MCS track
                for t in np.arange(0, mcs_duration[imcs]):

                    # Find merging cloud times that match current mcs track time
                    timematch = np.where(mergingbasetime == imcsbasetime[int(t)])

                    if np.shape(timematch)[1] > 0:

                        # save cloud number of small mergers
                        nmergers = np.shape(timematch)[1]
                        mcsmergecloudnumber[
                            imcs, int(t), 0:nmergers
                        ] = mergingcloudnumber[timematch]
                        mcsmergestatus[imcs, int(t), 0:nmergers] = mergingstatus[
                            timematch
                        ]

        ############################################################
        # Find splits
        splitfile = np.where(split_col == mcstracknumbers[imcs])[
            0
        ]  # Need to verify these work

        # Loop through all splitting tracks, if present
        if len(splitfile) > 0:
            # Isolate splitting cases that have short duration
            splitfile = splitfile[trackstat_duration[splitfile] < split_duration]

            # Make sure the spliter itself is not an MCS
            splittingmcs = np.intersect1d(splitfile, mcstracknumbers)
            if len(splittingmcs) > 0:
                for iremove in np.arange(0, len(splittingmcs)):
                    removesplits = np.array(
                        np.where(splitfile == splittingmcs[iremove])
                    )[0, :]
                    splitfile[removesplits] = fillval
                splitfile = splitfile[splitfile != fillval].astype(int)

            # Continue if spliters satisfy duration and MCS restriction
            if len(splitfile) > 0:

                # Get data about splitting tracks
                splittingcloudnumber = np.copy(cloudnumbers[splitfile, :])
                splittingbasetime = np.copy(basetime[splitfile, :])
                splittingstatus = np.copy(status[splitfile, :])
                # splittingdatetime = np.copy(datetimestrings[splitfile, :, :])

                # Get data about MCS track
                imcsbasetime = np.copy(
                    basetime[int(mcstracknumbers[imcs]) - 1, 0 : int(mcs_duration[imcs])]
                )

                # Loop through each timestep in the MCS track
                for t in np.arange(0, mcs_duration[imcs]):

                    # Find splitting cloud times that match current mcs track time
                    timematch = np.where(splittingbasetime == imcsbasetime[int(t)])
                    if np.shape(timematch)[1] > 0:

                        # save cloud number of small splitrs
                        nspliters = np.shape(timematch)[1]
                        mcssplitcloudnumber[
                            imcs, int(t), 0:nspliters
                        ] = splittingcloudnumber[timematch]
                        mcssplitstatus[imcs, int(t), 0:nspliters] = splittingstatus[
                            timematch
                        ]

    mcsmergecloudnumber = mcsmergecloudnumber.astype(np.int32)
    mcssplitcloudnumber = mcssplitcloudnumber.astype(np.int32)


    ###########################################################################
    # Write statistics to netcdf file

    # Create file
    mcstrackstatistics_outfile = (
        stats_path + "mcs_tracks_" + startdate + "_" + enddate + ".nc"
    )

    # Check if file already exists. If exists, delete
    if os.path.isfile(mcstrackstatistics_outfile):
        os.remove(mcstrackstatistics_outfile)

    # Defie xarray dataset
    output_data = xr.Dataset(
        {
            "base_time": ([tracks_dimname, times_dimname], mcsbasetime),
            # "mcs_datetimestring": (
            #     [tracks_dimname, times_dimname, "ndatetimechars"],
            #     datetimestrings[trackid, :, :],
            # ),
            "track_duration": ([tracks_dimname], trackstat_duration[trackid]),
            "mcs_duration": ([tracks_dimname], mcs_duration),
            # "mcs_type": ([tracks_dimname], mcstype),
            "mcs_status": ([tracks_dimname, times_dimname], status[trackid, :]),
            "start_status": ([tracks_dimname], startstatus[trackid]),
            "end_status": ([tracks_dimname], endstatus[trackid]),
            # # "mcs_boundary": ([tracks_dimname, times_dimname], boundary[trackid]),
            "track_interruptions": ([tracks_dimname, times_dimname], track_interruptions[trackid, :]),
            "meanlat": ([tracks_dimname, times_dimname], meanlat[trackid, :]),
            "meanlon": ([tracks_dimname, times_dimname], meanlon[trackid, :]),
            "core_area": ([tracks_dimname, times_dimname], trackstat_corearea[trackid, :]),
            "ccs_area": ([tracks_dimname, times_dimname], trackstat_ccsarea[trackid, :]),
            "cloudnumber": ([tracks_dimname, times_dimname], cloudnumbers[trackid, :]),
            "mergecloudnumber": (
                [tracks_dimname, times_dimname, "nmergers"],
                mcsmergecloudnumber,
            ),
            "splitcloudnumber": (
                [tracks_dimname, times_dimname, "nmergers"],
                mcssplitcloudnumber,
            ),
        },
        coords={
            tracks_dimname: ([tracks_dimname], np.arange(0, nmcs)),
            times_dimname: ([times_dimname], np.arange(0, nmaxlength)),
            "nmergers": (["nmergers"], np.arange(0, nmaxmerge)),
            # "ndatetimechars": (["ndatetimechars"], np.arange(0, 13)),
        },
        attrs={
            "title": "Statistics of each MCS track",
            # "Conventions": "CF-1.6",
            "Institution": "Pacific Northwest National Laboratory",
            "Contact": "Zhe Feng: zhe.feng@pnnl.gov",
            "Created_on": time.ctime(time.time()),
            "source": config["datasource"],
            "description": config["datadescription"],
            "startdate": startdate,
            "enddate": enddate,
            "time_resolution_hour": time_resolution,
            "pixel_radius_km": config["pixel_radius"],
            "MCS_duration_hr": duration_thresh,
            "MCS_area_km^2": mcs_tb_area_thresh,
            # "MCS_eccentricity": eccentricity_thresh,
        },
    )


    # Specify variable attributes
    output_data[tracks_dimname].attrs["long_name"] = "Number of MCS tracked"
    output_data[tracks_dimname].attrs["units"] = "unitless"

    output_data[times_dimname].attrs["long_name"] = "Maximum number of times in a MCS track"
    output_data[times_dimname].attrs["units"] = "unitless"

    output_data["nmergers"].attrs[
        "long_name"
    ] = "Maximum number of allowed mergers/splits into one cloud"
    output_data["nmergers"].attrs["units"] = "unitless"

    output_data["base_time"].attrs[
        "long_name"
    ] = "epoch time (seconds since 01/01/1970 00:00) of each cloud in a mcs track"
    # output_data["base_time"].attrs["standard_name"] = "time"
    output_data["base_time"].attrs["units"] = basetime_units

    # output_data.mcs_datetimestring.attrs[
    #     "long_name"
    # ] = "date_time for each cloud in a mcs track"
    # output_data.mcs_datetimestring.attrs["units"] = "unitless"

    output_data["track_duration"].attrs[
        "long_name"
    ] = "Complete duration of each track"
    output_data["track_duration"].attrs["units"] = "unitless"
    output_data["track_duration"].attrs["comments"] = "Multiply by time_resolution_hour to convert to physical units"

    output_data["mcs_duration"].attrs["long_name"] = "Duration of MCS stage"
    output_data["mcs_duration"].attrs["units"] = "unitless"
    output_data["mcs_duration"].attrs["comments"] = "Multiply by time_resolution_hour to convert to physical units"

    # output_data.mcs_type.attrs["long_name"] = "Type of MCS"
    # output_data.mcs_type.attrs["usage"] = "1=MCS, 2=Squall Line"
    # output_data.mcs_type.attrs["units"] = "unitless"

    output_data["mcs_status"].attrs[
        "long_name"
    ] = "flag indicating the status of each cloud in mcs track"
    output_data["mcs_status"].attrs["units"] = "unitless"
    # output_data["mcs_status"].attrs["valid_min"] = 0
    # output_data["mcs_status"].attrs["valid_max"] = 65

    output_data["start_status"].attrs[
        "long_name"
    ] = "flag indicating the status of the first cloud in each mcs track"
    output_data["start_status"].attrs["units"] = "unitless"
    # output_data["start_status"].attrs["valid_min"] = 0
    # output_data["start_status"].attrs["valid_max"] = 65

    output_data["end_status"].attrs[
        "long_name"
    ] = "flag indicating the status of the last cloud in each mcs track"
    output_data["end_status"].attrs["units"] = "unitless"
    # output_data["end_status"].attrs["valid_min"] = 0
    # output_data["end_status"].attrs["valid_max"] = 65

    # output_data.mcs_boundary.attrs[
    #     "long_name"
    # ] = "Flag indicating whether the core + cold anvil touches one of the domain edges."
    # output_data.mcs_boundary.attrs["values"] = "0 = away from edge. 1= touches edge."
    # output_data.mcs_boundary.attrs["units"] = "unitless"
    # output_data.mcs_boundary.attrs["valid_min"] = 0
    # output_data.mcs_boundary.attrs["valid_min"] = 1

    output_data["track_interruptions"].attrs[
        "long_name"
    ] = "Flag indiciate if the track started and ended naturally or if the start or end of the track was artifically cut short by data availability"
    output_data["track_interruptions"].attrs[
        "values"
    ] = "0 = full track available, good data. 1 = track starts at first file, track cut short by data availability. 2 = track ends at last file, track cut short by data availability"
    output_data["track_interruptions"].attrs["units"] = "unitless"
    output_data["track_interruptions"].attrs["valid_min"] = 0
    output_data["track_interruptions"].attrs["valid_min"] = 2

    output_data["meanlat"].attrs[
        "long_name"
    ] = "Mean latitude of the core + cold anvil for each feature in a mcs track"
    output_data["meanlat"].attrs["standard_name"] = "latitude"
    output_data["meanlat"].attrs["units"] = "degrees_north"
    # output_data["meanlat"].attrs["valid_min"] = geolimits[0]
    # output_data["meanlat"].attrs["valid_max"] = geolimits[2]

    output_data["meanlon"].attrs[
        "long_name"
    ] = "Mean longitude of the core + cold anvil for each feature in a mcs track"
    output_data["meanlon"].attrs["standard_name"] = "latitude"
    output_data["meanlon"].attrs["units"] = "degrees_north"
    # output_data["meanlon"].attrs["valid_min"] = geolimits[1]
    # output_data["meanlon"].attrs["valid_max"] = geolimits[3]

    output_data["core_area"].attrs[
        "long_name"
    ] = "Area of the cold core for each feature in a mcs track"
    output_data["core_area"].attrs["units"] = "km^2"

    output_data["ccs_area"].attrs[
        "long_name"
    ] = "Area of the cold core and cold anvil for each feature in a mcs track"
    output_data["ccs_area"].attrs["units"] = "km^2"

    output_data["cloudnumber"].attrs[
        "long_name"
    ] = "Number of each cloud in a track that cooresponds to the cloudid map"
    output_data["cloudnumber"].attrs["units"] = "unitless"
    output_data["cloudnumber"].attrs[
        "usuage"
    ] = "To link this tracking statistics file with pixel-level cloudid files, use the cloudidfile and cloudnumber together to identify which cloud this current track and time is associated with"

    output_data["mergecloudnumber"].attrs[
        "long_name"
    ] = "cloud number of small, short-lived feature merging into a mcs track"
    output_data["mergecloudnumber"].attrs["units"] = "unitless"

    output_data["splitcloudnumber"].attrs[
        "long_name"
    ] = "cloud number of small, short-lived feature splitting into a mcs track"
    output_data["splitcloudnumber"].attrs["units"] = "unitless"

    # Write netcdf file
    logger.info(mcstrackstatistics_outfile)

    output_data.to_netcdf(
        path=mcstrackstatistics_outfile,
        mode="w",
        format="NETCDF4",
        unlimited_dims=tracks_dimname,
        encoding={
            "base_time": {"zlib": True},
            # "mcs_datetimestring": {"zlib": True},
            "track_duration": {"zlib": True, "_FillValue": fillval},
            "mcs_duration": {"dtype": "int", "zlib": True, "_FillValue": fillval},
            # "mcs_type": {"dtype": "int", "zlib": True, "_FillValue": fillval},
            "mcs_status": {"dtype": "int", "zlib": True, "_FillValue": fillval},
            "start_status": {"dtype": "int", "zlib": True, "_FillValue": fillval},
            "end_status": {"dtype": "int", "zlib": True, "_FillValue": fillval},
            # "mcs_boundary": {"dtype": "int", "zlib": True, "_FillValue": fillval},
            "track_interruptions": {
                "dtype": "int",
                "zlib": True,
                "_FillValue": fillval,
            },
            "meanlat": {"zlib": True, "_FillValue": np.nan},
            "meanlon": {"zlib": True, "_FillValue": np.nan},
            "core_area": {"zlib": True, "_FillValue": np.nan},
            "ccs_area": {"zlib": True, "_FillValue": np.nan},
            "cloudnumber": {"dtype": "int", "zlib": True, "_FillValue": fillval},
            "mergecloudnumber": {"dtype": "int", "zlib": True, "_FillValue": fillval},
            "splitcloudnumber": {"dtype": "int", "zlib": True, "_FillValue": fillval},
        },
    )
    return mcstrackstatistics_outfile