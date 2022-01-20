import numpy as np
import time
import os
import sys
import xarray as xr
import logging
from scipy.sparse import csr_matrix

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
    trackstats_sparse_filebase = config["trackstats_sparse_filebase"]
    stats_path = config["stats_outpath"]
    startdate = config["startdate"]
    enddate = config["enddate"]
    time_resolution = config["datatimeresolution"]
    duration_range = config["duration_range"]
    mcs_tb_area_thresh = config["mcs_tb_area_thresh"]
    duration_thresh = config["mcs_tb_duration_thresh"]
    split_duration = config["mcs_tb_split_duration"]
    merge_duration = config["mcs_tb_merge_duration"]
    nmaxmerge = config["nmaxlinks"]
    timegap = config["mcs_tb_gap"]
    tracks_dimname = config["tracks_dimname"]
    times_dimname = config["times_dimname"]
    tracks_idx_varname = f"{tracks_dimname}_indices"
    times_idx_varname = f"{times_dimname}_indices"
    fillval = config["fillval"]
    fillval_f = np.nan
    max_trackduration = int(max(duration_range))

    np.set_printoptions(threshold=np.inf)
    logger = logging.getLogger(__name__)
    logger.info("Identifying MCS based on Tb statistics")

    # Output stats file name
    statistics_outfile = f"{stats_path}{mcstbstats_filebase}{startdate}_{enddate}.nc"

    ##########################################################################
    # Load statistics file
    statistics_file = f"{stats_path}{trackstats_sparse_filebase}{startdate}_{enddate}.nc"
    # statistics_file = f"{stats_path}{trackstats_filebase}{startdate}_{enddate}.nc.copy"
    logger.debug(statistics_file)

    # xr.set_options(keep_attrs=True)
    ds_all = xr.open_dataset(statistics_file,
                             mask_and_scale=False,
                             decode_times=False)
    #
    sparse_dimname = 'sparse_index'
    nsparse_data = ds_all.dims[sparse_dimname]
    ntracks = ds_all.dims[tracks_dimname]
    # Sparse array indices
    tracks_idx = ds_all[tracks_idx_varname].values
    times_idx = ds_all[times_idx_varname].values
    row_col_ind = (tracks_idx, times_idx)
    # Sparse array shapes
    shape_2d = (ntracks, max_trackduration)

    # Convert sparse arrays and put in a dictionary
    sparse_dict = {}
    sparse_attrs_dict = {}
    for ivar in ds_all.data_vars.keys():
        # Check dimension name for sparse arrays
        if ds_all[ivar].dims[0] == sparse_dimname:
            # Convert to sparse array
            sparse_dict[ivar] = csr_matrix(
                (ds_all[ivar].values, row_col_ind), shape=shape_2d, dtype=ds_all[ivar].dtype,
            )
            # Collect variable attributes
            sparse_attrs_dict[ivar] = ds_all[ivar].attrs

    # Drop all sparse variables and dimension
    ds_1d = ds_all.drop_dims(sparse_dimname)
    ds_all.close()

    # for ivar in ds_dict['data_vars'].keys(): print(ivar, ds_dict['data_vars'][ivar].sizes)

    ntracks_all = ds_1d.dims[tracks_dimname]
    # max_trackduration = int(max(duration_range))
    logger.debug(f"max_trackduration:{max_trackduration}")
    # Load necessary variables
    track_duration = ds_1d["track_duration"].values
    end_merge_tracknumber = ds_1d["end_merge_tracknumber"].values
    start_split_tracknumber = ds_1d["start_split_tracknumber"].values
    trackstat_corearea = sparse_dict["core_area"]
    trackstat_coldarea = sparse_dict["cold_area"]
    basetime = sparse_dict["base_time"]
    # mergecloudnumbers = sparse_dict["merge_tracknumbers"]
    # splitcloudnumbers = sparse_dict["split_tracknumbers"]
    cloudnumbers = sparse_dict["cloudnumber"]
    track_status = sparse_dict["track_status"]


    # Convert a few key variables to dense arrays
    # and replace fill values with NaN
    # This is a temporary solution to make the codes for
    # adding small merge/split clouds to MCS works
    # TODO: ideally the add merge/split clouds code is modified to work with sparse arrays
    # basetime_s = copy.deepcopy(basetime)

    # basetime = basetime.toarray()
    # cloudnumbers = cloudnumbers.toarray().astype(np.float32)
    # mergecloudnumbers = mergecloudnumbers.toarray().astype(np.float32)
    # splitcloudnumbers = splitcloudnumbers.toarray().astype(np.float32)
    # track_status = track_status.toarray()
    # basetime[basetime == 0] = fillval_f
    # mask = np.logical_or((cloudnumbers == 0), (cloudnumbers == fillval))
    # cloudnumbers[mask] = fillval_f
    # mask = np.logical_or((mergecloudnumbers == 0), (mergecloudnumbers == fillval))
    # mergecloudnumbers[mask] = fillval_f
    # mask = np.logical_or((splitcloudnumbers == 0), (splitcloudnumbers == fillval))
    # splitcloudnumbers[mask] = fillval_f
    # track_status[np.isnan(basetime)] = fillval


    # # Total number of tracked features
    # ntracks_all = ds_all.dims[tracks_dimname]
    # # Maximum number of times in a given track
    # max_trackduration = ds_all.dims[times_dimname]
    # logger.debug(f"max_trackduration:{max_trackduration}")
    # # Load necessary variables
    # track_duration = ds_all["track_duration"].values
    # trackstat_corearea = ds_all["core_area"].values
    # trackstat_coldarea = ds_all["cold_area"].values
    # basetime = ds_all["base_time"].values
    # mergecloudnumbers = ds_all["merge_tracknumbers"].values
    # splitcloudnumbers = ds_all["split_tracknumbers"].values
    # cloudnumbers = ds_all["cloudnumber"].values
    # track_status = ds_all["track_status"].values
    # fillval_f = ds_all["cold_area"].attrs["_FillValue"]
    # ds_all.close()

    logger.debug(f"MCS CCS area threshold: {mcs_tb_area_thresh}")
    logger.debug(f"MCS duration threshold: {duration_thresh}")

    # Convert track duration to physical time unit
    trackstat_lifetime = np.multiply(track_duration, time_resolution)
    # Get CCS area
    trackstat_ccsarea = trackstat_corearea + trackstat_coldarea

    ###################################################################
    # Identify MCSs
    trackidx_mcs = []
    trackid_nonmcs = []

    mcstype = np.zeros(ntracks_all, dtype=np.int16)
    mcsstatus = np.full((ntracks_all, max_trackduration), fillval, dtype=np.int16)

    logger.debug(f"Total number of tracks to check: {ntracks_all}")
    for nt in range(0, ntracks_all):
        # Get data for a given track
        track_corearea = trackstat_corearea[nt, :].data
        track_ccsarea = trackstat_ccsarea[nt, :].data

        # Remove fill values
        track_corearea = track_corearea[
            (~np.isnan(track_corearea)) & (track_corearea != 0)
        ]
        track_ccsarea = track_ccsarea[~np.isnan(track_ccsarea)]

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
                        mcstype[nt] = 1
                        mcsstatus[nt, groups[t][:]] = 1
                        trackidx_mcs = np.append(trackidx_mcs, nt)
                    else:
                        # Size requirement met but too short of a period
                        trackid_nonmcs = np.append(trackid_nonmcs, nt)

            else:
                # Size requirement not met
                trackid_nonmcs = np.append(trackid_nonmcs, nt)

    ################################################################
    # Get unique track indices
    trackidx_mcs = np.unique(trackidx_mcs.astype(int))
    # Provide warning message and exit if no MCS identified
    if trackidx_mcs == []:
        logger.critical("WARNING: No MCS identified.")
        logger.critical(f"Tracking will now exit.")
        sys.exit()

    ################################################################
    # Subset MCS track index
    # trackid = np.array(np.where(mcstype > 0))[0, :]
    nmcs = len(trackidx_mcs)
    logger.info(f"Number of Tb defined MCS: {nmcs}")

    if nmcs > 0:
        mcsstatus = mcsstatus[trackidx_mcs, :]
        mcstype = mcstype[trackidx_mcs]

        # Get duration when MCS status is met
        mcs_duration = np.nansum(mcsstatus > 0, axis=1)

    # trackidx_mcs is the index number, want the track number so add one
    mcstracknumbers = np.copy(trackidx_mcs) + 1


    ###############################################################
    # Find small merging and spliting clouds and add to MCS
    mcs_merge_cloudnumber = np.full((nmcs, max_trackduration, nmaxmerge), fillval, dtype=np.int32)
    mcs_merge_status = np.full((nmcs, max_trackduration, nmaxmerge), fillval, dtype=np.int32)
    mcs_merge_ccsarea = np.full((nmcs, max_trackduration, nmaxmerge), fillval_f, dtype=np.float32)
    mcs_split_cloudnumber = np.full((nmcs, max_trackduration, nmaxmerge), fillval, dtype=np.int32)
    mcs_split_status = np.full((nmcs, max_trackduration, nmaxmerge), fillval, dtype=np.int32)
    mcs_split_ccsarea = np.full((nmcs, max_trackduration, nmaxmerge), fillval_f, dtype=np.float32)

    # Loop through each MCS and link small clouds merging in
    for imcs in np.arange(0, nmcs):
        ###################################################################################
        # Find tracks that end as merging with the MCS
        mergetrack_idx = np.where(end_merge_tracknumber == mcstracknumbers[imcs])[0]
        if len(mergetrack_idx) > 0:
            # Isolate merge tracks that have short duration
            mergetrack_idx = mergetrack_idx[trackstat_lifetime[mergetrack_idx] < merge_duration]

            # Make sure the merge tracks are not MCS
            mergetrack_idx = mergetrack_idx[np.isin(mergetrack_idx, mcstracknumbers, invert=True)]
            if len(mergetrack_idx) > 0:
                # Get data for merging tracks
                mergingcloudnumber = cloudnumbers[mergetrack_idx, :].data
                mergingbasetime = basetime[mergetrack_idx, :].data
                mergingstatus = track_status[mergetrack_idx, :].data
                mergingccsarea = trackstat_ccsarea[mergetrack_idx, :].data

                # Get MCS basetime
                imcsbasetime = basetime[int(mcstracknumbers[imcs])-1, :].data

                # import pdb; pdb.set_trace()

                # Loop through each timestep in the MCS track
                for t in np.arange(0, len(imcsbasetime)):
                    # Find merging cloud times that match current MCS track time
                    timematch = np.where(mergingbasetime == imcsbasetime[int(t)])[0]
                    nmergers = len(timematch)
                    if nmergers > 0:
                        mcs_merge_cloudnumber[imcs, int(t), 0:nmergers] = mergingcloudnumber[timematch]
                        mcs_merge_status[imcs, int(t), 0:nmergers] = mergingstatus[timematch]
                        mcs_merge_ccsarea[imcs, int(t), 0:nmergers] = mergingccsarea[timematch]

        ###################################################################################
        # Find tracks that split from the MCS
        splittrack_idx = np.where(start_split_tracknumber == mcstracknumbers[imcs])[0]
        if len(splittrack_idx) > 0:
            # Isolate split tracks that have short duration
            splittrack_idx = splittrack_idx[trackstat_lifetime[splittrack_idx] < split_duration]

            # Make sure the split tracks are not MCS
            splittrack_idx = splittrack_idx[np.isin(splittrack_idx, mcstracknumbers, invert=True)]
            if len(splittrack_idx) > 0:
                # Get data for split tracks
                splittingcloudnumber = cloudnumbers[splittrack_idx, :].data
                splittingbasetime = basetime[splittrack_idx, :].data
                splittingstatus = track_status[splittrack_idx, :].data
                splittingccsarea = trackstat_ccsarea[splittrack_idx, :].data

                # Get MCS basetime
                imcsbasetime = basetime[int(mcstracknumbers[imcs]) - 1, :].data

                # Loop through each timestep in the MCS track
                for t in np.arange(0, len(imcsbasetime)):
                    # Find splitting cloud times that match current MCS track time
                    timematch = np.where(splittingbasetime == imcsbasetime[int(t)])[0]
                    nspliters = len(timematch)
                    if nspliters > 0:
                        mcs_split_cloudnumber[imcs, int(t), 0:nspliters] = splittingcloudnumber[timematch]
                        mcs_split_status[imcs, int(t), 0:nspliters] = splittingstatus[timematch]
                        mcs_split_ccsarea[imcs, int(t), 0:nspliters] = splittingccsarea[timematch]


    # import pdb;
    # pdb.set_trace()

    # # Let's convert 2D to 1D arrays for performance
    # split_col = np.nanmax(splitcloudnumbers, axis=1)
    # merger_col = np.nanmax(mergecloudnumbers, axis=1)
    #
    # # Loop through each MCS and link small clouds merging in
    # for imcs in np.arange(0, nmcs):
    #     ###################################################################################
    #     # Isolate basetime data
    #     if imcs == 0:
    #         mcsbasetime = basetime[trackidx_mcs[imcs], :]
    #         # mcsbasetime = np.array([pd.to_datetime(num2date(basetime[trackidx_mcs[imcs], :],
    #         # units=basetime_units, calendar=basetime_calendar))], dtype='datetime64[s]')
    #     else:
    #         mcsbasetime = np.row_stack((mcsbasetime, basetime[trackidx_mcs[imcs], :]))
    #         # mcsbasetime = np.concatenate((mcsbasetime,
    #         # np.array([pd.to_datetime(num2date(basetime[trackidx_mcs[imcs], :],
    #         # units=basetime_units, calendar=basetime_calendar))], dtype='datetime64[s]')), axis=0)
    #
    #     ###################################################################################
    #     # Find mergers
    #     mergefile = np.where(merger_col == mcstracknumbers[imcs])[0]
    #
    #     for imerger in range(0, len(mergefile)):
    #         additionalmergefile = np.where(merger_col == mergefile[imerger] + 1)[0]
    #
    #         if len(additionalmergefile) > 0:
    #             mergefile = np.concatenate((mergefile, additionalmergefile))
    #
    #     # Loop through all merging tracks, if present
    #     if len(mergefile) > 0:
    #         # Isolate merging cases that have short duration
    #         mergefile = mergefile[trackstat_lifetime[mergefile] < merge_duration]
    #
    #         # Make sure the merger itself is not an MCS
    #         mergingmcs = np.intersect1d(mergefile, mcstracknumbers)
    #         if len(mergingmcs) > 0:
    #             for iremove in np.arange(0, len(mergingmcs)):
    #                 removemerges = np.array(np.where(mergefile == mergingmcs[iremove]))[0, :]
    #                 mergefile[removemerges] = fillval
    #             mergefile = mergefile[mergefile != fillval].astype(int)
    #
    #         # Continue if mergers satisfy duration and MCS restriction
    #         if len(mergefile) > 0:
    #
    #             # Get data for merging tracks
    #             mergingcloudnumber = cloudnumbers[mergefile, :]
    #             mergingbasetime = basetime[mergefile, :]
    #             mergingstatus = track_status[mergefile, :]
    #
    #             # Get MCS basetime
    #             imcsbasetime = basetime[
    #                            int(mcstracknumbers[imcs])-1, 0:int(mcs_duration[imcs])
    #                            ]
    #
    #             # Loop through each timestep in the MCS track
    #             for t in np.arange(0, mcs_duration[imcs]):
    #
    #                 # Find merging cloud times that match current MCS track time
    #                 timematch = np.where(mergingbasetime == imcsbasetime[int(t)])
    #
    #                 if np.shape(timematch)[1] > 0:
    #                     # save cloud number of small mergers
    #                     nmergers = np.shape(timematch)[1]
    #                     mcs_merge_cloudnumber[
    #                         imcs, int(t), 0:nmergers
    #                     ] = mergingcloudnumber[timematch].astype(int)
    #                     mcs_merge_status[imcs, int(t), 0:nmergers] = mergingstatus[
    #                         timematch
    #                     ]
    #
    #     ############################################################
    #     # Find splits
    #     splitfile = np.where(split_col == mcstracknumbers[imcs])[0]
    #     # Need to verify these work
    #
    #     # Loop through all splitting tracks, if present
    #     if len(splitfile) > 0:
    #         # Isolate splitting cases that have short duration
    #         splitfile = splitfile[trackstat_lifetime[splitfile] < split_duration]
    #
    #         # Make sure the spliter itself is not an MCS
    #         splittingmcs = np.intersect1d(splitfile, mcstracknumbers)
    #         if len(splittingmcs) > 0:
    #             for iremove in np.arange(0, len(splittingmcs)):
    #                 removesplits = np.array(
    #                     np.where(splitfile == splittingmcs[iremove])
    #                 )[0, :]
    #                 splitfile[removesplits] = fillval
    #             splitfile = splitfile[splitfile != fillval].astype(int)
    #
    #         # Continue if spliters satisfy duration and MCS restriction
    #         if len(splitfile) > 0:
    #
    #             # Get data for splitting tracks
    #             splittingcloudnumber = cloudnumbers[splitfile, :]
    #             splittingbasetime = basetime[splitfile, :]
    #             splittingstatus = track_status[splitfile, :]
    #
    #             # Get MCS basetime
    #             imcsbasetime = basetime[
    #                            int(mcstracknumbers[imcs])-1, 0:int(mcs_duration[imcs])
    #                            ]
    #
    #             # Loop through each timestep in the MCS track
    #             for t in np.arange(0, mcs_duration[imcs]):
    #
    #                 # Find splitting cloud times that match current MCS track time
    #                 timematch = np.where(splittingbasetime == imcsbasetime[int(t)])
    #                 if np.shape(timematch)[1] > 0:
    #
    #                     # save cloud number of small splitrs
    #                     nspliters = np.shape(timematch)[1]
    #                     mcs_split_cloudnumber[
    #                         imcs, int(t), 0:nspliters
    #                     ] = splittingcloudnumber[timematch].astype(int)
    #                     mcs_split_status[imcs, int(t), 0:nspliters] = splittingstatus[
    #                         timematch
    #                     ]
    #
    # mcs_merge_cloudnumber = mcs_merge_cloudnumber.astype(np.int32)
    # mcs_split_cloudnumber = mcs_split_cloudnumber.astype(np.int32)
    # import pdb; pdb.set_trace()

    ###########################################################################
    # Prepare output dataset

    # Subset tracks in sparse array dictionary
    for ivar in sparse_dict.keys():
        sparse_dict[ivar] = sparse_dict[ivar][trackidx_mcs, :]
    # Remove the tracks/times indices variables
    sparse_dict.pop(tracks_idx_varname, None)
    sparse_dict.pop(times_idx_varname, None)

    # Create a dense mask for no clouds
    mask = sparse_dict['base_time'].toarray() == 0
    # Convert sparse dictionary to dataset
    varlist = {}
    # Define output variable dictionary
    for key, value in sparse_dict.items():
        dense_array = value.toarray()
        # Replace missing values based on variable type
        if isinstance(dense_array[0, 0], np.floating):
            dense_array[mask] = fillval_f
        else:
            dense_array[mask] = fillval
        varlist[key] = ([tracks_dimname, times_dimname], dense_array, sparse_attrs_dict[key])
    # Define coordinate
    tracks_coord = np.arange(0, nmcs)
    times_coord = np.arange(0, max_trackduration)
    coordlist = {
        tracks_dimname: ([tracks_dimname], tracks_coord),
        times_dimname: ([times_dimname], times_coord),
    }
    # Define 2D Xarray dataset
    ds_2d = xr.Dataset(varlist, coords=coordlist)
    # Subset MCS tracks from 1D dataset
    ds_1d = ds_1d.sel(tracks=trackidx_mcs)
    # Replace tracks coordinate
    ds_1d[tracks_dimname] = tracks_coord
    # Merge 1D & 2D datasets
    dsout = xr.merge([ds_1d, ds_2d], compat="override", combine_attrs="no_conflicts")


    # Subset MCS tracks from all tracks dataset
    # Note: the tracks_dimname cannot be used here as Xarray does not seem to have
    # a method to select data with a string variable
    # dsout = ds_all.sel(tracks=trackidx_mcs)
    # Remove no use variables
    drop_vars_list = [
        "merge_tracknumbers", "split_tracknumbers",
        "start_split_tracknumber", "start_split_timeindex",
        "end_merge_tracknumber", "end_merge_timeindex",
    ]
    dsout = dsout.drop_vars(drop_vars_list)
    # Replace tracks coordinate
    # tracks_coord = np.arange(0, nmcs)
    # times_coord = ds_all[times_dimname]
    # dsout[tracks_dimname] = tracks_coord

    # import pdb;
    # pdb.set_trace()

    # Create a flag for MCS status
    # ccs_area = trackstat_ccsarea[trackidx_mcs, :]
    ccs_area = dsout['core_area'] + dsout['cold_area']
    mcs_status = np.full(ccs_area.shape, fillval, dtype=np.int16)
    mcs_status[ccs_area > mcs_tb_area_thresh] = 1
    mcs_status[(ccs_area <= mcs_tb_area_thresh) & (ccs_area > 0)] = 0

    # Define new variables dictionary
    var_dict = {
        "mcs_duration": mcs_duration,
        "mcs_status": mcs_status,
        "ccs_area": ccs_area,
        "mergecloudnumber": mcs_merge_cloudnumber,
        "splitcloudnumber": mcs_split_cloudnumber,
        "merge_ccs_area": mcs_merge_ccsarea,
        "split_ccs_area": mcs_split_ccsarea,
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
        "merge_ccs_area": {
            "long_name": "Cold cloud shield area that merge into MCS",
            "units": "km^2",
            "_FillValue": fillval_f,
        },
        "split_ccs_area": {
            "long_name": "Cold cloud shield area that split from MCS",
            "units": "km^2",
            "_FillValue": fillval_f,
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