import numpy as np
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
        statistics_outfile: string
            MCS track statistics file name.
    """

    mcstbstats_filebase = config["mcstbstats_filebase"]
    trackstats_filebase = config["trackstats_filebase"]
    stats_path = config["stats_outpath"]
    startdate = config["startdate"]
    enddate = config["enddate"]
    time_resolution = config["datatimeresolution"]
    mcs_tb_area_thresh = config["mcs_tb_area_thresh"]
    duration_thresh = config["mcs_tb_duration_thresh"]
    split_duration = config["mcs_tb_split_duration"]
    merge_duration = config["mcs_tb_merge_duration"]
    nmaxmerge = config["nmaxlinks"]
    timegap = config["mcs_tb_gap"]
    tracks_dimname = config["tracks_dimname"]
    times_dimname = config["times_dimname"]
    fillval = config["fillval"]

    np.set_printoptions(threshold=np.inf)
    logger = logging.getLogger(__name__)
    logger.info("Identifying MCS based on Tb statistics")

    # Output stats file name
    statistics_outfile = f"{stats_path}{mcstbstats_filebase}{startdate}_{enddate}.nc"

    ##########################################################################
    # Load statistics file
    statistics_file = f"{stats_path}{trackstats_filebase}{startdate}_{enddate}.nc"
    logger.debug(statistics_file)

    ds_all = xr.open_dataset(statistics_file,
                             mask_and_scale=False,
                             decode_times=False)
    # Total number of tracked features
    ntracks_all = np.nanmax(ds_all[tracks_dimname]) + 1
    # Maximum number of times in a given track
    nmaxtimes = np.nanmax(ds_all[times_dimname]) + 1
    logger.debug(f"nmaxtimes:{nmaxtimes}")
    # Load necessary variables
    track_duration = ds_all["track_duration"].values
    trackstat_corearea = ds_all["core_area"].values
    trackstat_coldarea = ds_all["cold_area"].values
    basetime = ds_all["base_time"].values
    mergecloudnumbers = ds_all["merge_tracknumbers"].values
    splitcloudnumbers = ds_all["split_tracknumbers"].values
    cloudnumbers = ds_all["cloudnumber"].values
    status = ds_all["track_status"].values
    fillval_f = ds_all["cold_area"].attrs["_FillValue"]
    ds_all.close()

    logger.debug(f"MCS CCS area threshold: {mcs_tb_area_thresh}")
    logger.debug(f"MCS duration threshold: {duration_thresh}")

    # Convert track duration to physical time unit
    trackstat_duration = np.multiply(track_duration, time_resolution)
    # Get CCS area
    trackstat_ccsarea = trackstat_corearea + trackstat_coldarea

    ###################################################################
    # Identify MCSs
    trackid_mcs = []
    # trackid_sql = []
    trackid_nonmcs = []

    mcstype = np.zeros(ntracks_all, dtype=np.int16)
    mcsstatus = np.full((ntracks_all, nmaxtimes), fillval, dtype=np.int16)

    logger.debug(f"Total number of tracks to check: {ntracks_all}")
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
            groups = np.split(iccs, np.where(np.diff(iccs) > timegap)[0] + 1)
            nbreaks = len(groups)

            # System may have multiple periods satisfying area and duration requirements
            # Loop over each period
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
    # Get unique track indices
    trackid_mcs = np.unique(trackid_mcs.astype(int))
    # Print a critical warning message if there is no MCS
    if trackid_mcs == []:
        logger.critical("Warning: There is no MCS in the domain, the code will crash.")

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
    mcsmergecloudnumber = np.full((nmcs, nmaxtimes, nmaxmerge), fillval, dtype=np.int32)
    mcsmergestatus = np.full((nmcs, nmaxtimes, nmaxmerge), fillval, dtype=np.int32)
    mcssplitcloudnumber = np.full((nmcs, nmaxtimes, nmaxmerge), fillval, dtype=np.int32)
    mcssplitstatus = np.full((nmcs, nmaxtimes, nmaxmerge), fillval, dtype=np.int32)

    # Let's convert 2D to 1D arrays for performance
    split_col = np.nanmax(splitcloudnumbers, axis=1)
    merger_col = np.nanmax(mergecloudnumbers, axis=1)

    # Loop through each MCS and link small clouds merging in
    for imcs in np.arange(0, nmcs):
        ###################################################################################
        # Isolate basetime data
        if imcs == 0:
            mcsbasetime = basetime[trackid[imcs], :]
            # mcsbasetime = np.array([pd.to_datetime(num2date(basetime[trackid[imcs], :],
            # units=basetime_units, calendar=basetime_calendar))], dtype='datetime64[s]')
        else:
            mcsbasetime = np.row_stack((mcsbasetime, basetime[trackid[imcs], :]))
            # mcsbasetime = np.concatenate((mcsbasetime,
            # np.array([pd.to_datetime(num2date(basetime[trackid[imcs], :],
            # units=basetime_units, calendar=basetime_calendar))], dtype='datetime64[s]')), axis=0)

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

                # Get data for merging tracks
                mergingcloudnumber = np.copy(cloudnumbers[mergefile, :])
                mergingbasetime = np.copy(basetime[mergefile, :])
                mergingstatus = np.copy(status[mergefile, :])
                # mergingdatetime = np.copy(datetimestrings[mergefile, :, :])

                # Get MCS basetime
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
        splitfile = np.where(split_col == mcstracknumbers[imcs])[0]
        # Need to verify these work

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

                # Get data for splitting tracks
                splittingcloudnumber = np.copy(cloudnumbers[splitfile, :])
                splittingbasetime = np.copy(basetime[splitfile, :])
                splittingstatus = np.copy(status[splitfile, :])
                # splittingdatetime = np.copy(datetimestrings[splitfile, :, :])

                # Get MCS basetime
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
    # Prepare output dataset

    # Subset MCS tracks from all tracks dataset
    # Note: the tracks_dimname cannot be used here as Xarray does not seem to have
    # a method to select data with a string variable
    dsout = ds_all.sel(tracks=trackid)
    # Remove no use variables
    drop_vars_list = [
        "merge_tracknumbers", "split_tracknumbers",
        "start_split_tracknumber", "start_split_timeindex",
        "end_merge_tracknumber", "end_merge_timeindex",
    ]
    dsout = dsout.drop_vars(drop_vars_list)
    # Replace tracks coordinate
    tracks_coord = np.arange(0, nmcs)
    times_coord = ds_all[times_dimname]
    dsout[tracks_dimname] = tracks_coord

    # Create a flag for MCS status
    ccs_area = trackstat_ccsarea[trackid, :]
    mcs_status = np.full(ccs_area.shape, fillval, dtype=np.int16)
    mcs_status[ccs_area > mcs_tb_area_thresh] = 1
    mcs_status[(ccs_area <= mcs_tb_area_thresh) & (ccs_area > 0)] = 0

    # Define new variables dictionary
    var_dict = {
        "mcs_duration": mcs_duration,
        "mcs_status": mcs_status,
        "ccs_area": ccs_area,
        "mergecloudnumber": mcsmergecloudnumber,
        "splitcloudnumber": mcssplitcloudnumber,
    }
    var_attrs = {
        "mcs_duration": {
            "long_name": "Duration of MCS stage",
            "units": "unitless",
            "comments": "Multiply by time_resolution_hour to convert to physical units",
        },
        "mcs_status": {
            "long_name": "Flag indicating the status of MCS based on Tb. 1 = Yes, 0 = No",
            "units": "unitless",
            "_FillValue": fillval,
        },
        "ccs_area": {
            "long_name": "Area of cold cloud shield",
            "units": "km^2",
            "_FillValue": fillval_f,
        },
        "mergecloudnumber": {
            "long_name": "Cloud numbers that merge into MCS",
            "units": "unitless",
            "_FillValue": fillval,
        },
        "splitcloudnumber": {
            "long_name": "Cloud numbers that split from MCS",
            "units": "unitless",
            "_FillValue": fillval,
        },
    }

    # Create mergers/splits coordinates
    mergers_dimname = "mergers"
    mergers_coord = np.arange(0, nmaxmerge)

    # Define output variable dictionary
    varlist = {}
    for key, value in var_dict.items():
        if value.ndim == 1:
            varlist[key] = ([tracks_dimname], value, var_attrs[key])
        if value.ndim == 2:
            varlist[key] = ([tracks_dimname, times_dimname], value, var_attrs[key])
        if value.ndim == 3:
            varlist[key] = ([tracks_dimname, times_dimname, mergers_dimname], value, var_attrs[key])
    # Define coordinate list
    coordlist = {
        tracks_dimname: ([tracks_dimname], tracks_coord),
        times_dimname: ([times_dimname], times_coord),
        mergers_dimname: ([mergers_dimname], mergers_coord),
    }
    # Define extra variable Dataset
    ds_vars = xr.Dataset(varlist, coords=coordlist)

    # Merge Datasets
    dsout = xr.merge([dsout, ds_vars], compat="override", combine_attrs="no_conflicts")
    # Update global attributes
    dsout.attrs["Title"] = "Statistics of each MCS track"
    dsout.attrs["MCS_duration_hr"] = duration_thresh
    dsout.attrs["MCS_area_km^2"] = mcs_tb_area_thresh
    dsout.attrs["Created_on"] = time.ctime(time.time())


    ###########################################################################
    # Write statistics to netcdf file

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