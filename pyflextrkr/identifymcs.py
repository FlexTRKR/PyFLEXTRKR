import numpy as np
import time
import os
import sys
import xarray as xr
import logging
from pyflextrkr.ft_utilities import load_sparse_trackstats

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
    logger.debug(statistics_file)

    # Load sparse tracks statistics file and convert to sparse arrays
    ds_1d, \
    sparse_attrs_dict, \
    sparse_dict = load_sparse_trackstats(max_trackduration, statistics_file,
                                         times_idx_varname, tracks_dimname,
                                         tracks_idx_varname)

    # Get necessary variables
    ntracks_all = ds_1d.dims[tracks_dimname]
    logger.debug(f"max_trackduration: {max_trackduration}")
    track_duration = ds_1d["track_duration"].values
    end_merge_tracknumber = ds_1d["end_merge_tracknumber"].values
    start_split_tracknumber = ds_1d["start_split_tracknumber"].values
    trackstat_corearea = sparse_dict["core_area"]
    trackstat_coldarea = sparse_dict["cold_area"]
    basetime = sparse_dict["base_time"]
    cloudnumbers = sparse_dict["cloudnumber"]
    track_status = sparse_dict["track_status"]

    # import pdb; pdb.set_trace()

    logger.info(f"Number of tracks to process: {ntracks_all}")
    logger.debug(f"MCS CCS area threshold: {mcs_tb_area_thresh}")
    logger.debug(f"MCS duration threshold: {duration_thresh}")

    # Convert track duration to physical time unit
    trackstat_lifetime = np.multiply(track_duration, time_resolution)

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
        # Get CCS area
        track_ccsarea = trackstat_corearea[nt, :].data + trackstat_coldarea[nt, :].data

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
                mergingccsarea = trackstat_corearea[mergetrack_idx, :].data +\
                                 trackstat_coldarea[mergetrack_idx, :].data

                # Get MCS basetime
                imcsbasetime = basetime[int(mcstracknumbers[imcs])-1, :].data

                # Loop through each timestep in the MCS track
                for t in np.arange(0, len(imcsbasetime)):
                    # Find merging cloud times that match current MCS track time
                    timematch = np.where(mergingbasetime == imcsbasetime[int(t)])[0]
                    nmergers = len(timematch)
                    if nmergers > 0:
                        # Find the smaller value between the two
                        # to make sure it fits into the array
                        nmergers_sav = np.min([nmergers, nmaxmerge])
                        mcs_merge_cloudnumber[imcs, int(t), 0:nmergers_sav] = mergingcloudnumber[timematch[0:nmergers_sav]]
                        mcs_merge_status[imcs, int(t), 0:nmergers_sav] = mergingstatus[timematch[0:nmergers_sav]]
                        mcs_merge_ccsarea[imcs, int(t), 0:nmergers_sav] = mergingccsarea[timematch[0:nmergers_sav]]
                        if (nmergers > nmaxmerge):
                            logger.warning(f'WARNING: nmergers ({nmergers}) > nmaxmerge ({nmaxmerge}), ' + \
                                'only partial merge clouds are saved.')
                            logger.warning(f'MCS track index: {imcs}')
                            logger.warning(f'Increase nmaxmerge to avoid this WARNING.')

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
                splittingccsarea = trackstat_corearea[splittrack_idx, :].data +\
                                   trackstat_coldarea[splittrack_idx, :].data

                # Get MCS basetime
                imcsbasetime = basetime[int(mcstracknumbers[imcs]) - 1, :].data

                # Loop through each timestep in the MCS track
                for t in np.arange(0, len(imcsbasetime)):
                    # Find splitting cloud times that match current MCS track time
                    timematch = np.where(splittingbasetime == imcsbasetime[int(t)])[0]
                    nspliters = len(timematch)
                    if nspliters > 0:
                        # Find the smaller value between the two
                        # to make sure it fits into the array
                        nspliters_sav = np.min([nspliters, nmaxmerge])
                        mcs_split_cloudnumber[imcs, int(t), 0:nspliters_sav] = splittingcloudnumber[timematch[0:nspliters_sav]]
                        mcs_split_status[imcs, int(t), 0:nspliters_sav] = splittingstatus[timematch[0:nspliters_sav]]
                        mcs_split_ccsarea[imcs, int(t), 0:nspliters_sav] = splittingccsarea[timematch[0:nspliters_sav]]
                        if (nspliters > nmaxmerge):
                            logger.warning(f'WARNING: nspliters ({nspliters}) > nmaxmerge ({nmaxmerge}), ' + \
                                'only partial split clouds are saved.')
                            logger.warning(f'MCS track index: {imcs}')
                            logger.warning(f'Increase nmaxmerge to avoid this WARNING.')


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
    # Note: the tracks_dimname cannot be used here as Xarray does not seem to have
    # a method to select data with a string variable
    ds_1d = ds_1d.sel(tracks=trackidx_mcs)
    # Replace tracks coordinate
    ds_1d[tracks_dimname] = tracks_coord
    # Merge 1D & 2D datasets
    dsout = xr.merge([ds_1d, ds_2d], compat="override", combine_attrs="no_conflicts")
    # Remove no use variables
    drop_vars_list = [
        "merge_tracknumbers", "split_tracknumbers",
        "start_split_tracknumber", "start_split_timeindex",
        "end_merge_tracknumber", "end_merge_timeindex",
    ]
    dsout = dsout.drop_vars(drop_vars_list)

    # Create a flag for MCS status
    ccs_area = dsout['core_area'].data + dsout['cold_area'].data
    mcs_status = np.full(ccs_area.shape, fillval, dtype=np.int16)
    mcs_status[ccs_area > mcs_tb_area_thresh] = 1
    mcs_status[(ccs_area <= mcs_tb_area_thresh) & (ccs_area > 0)] = 0

    # Define new variables dictionary
    var_dict = {
        "mcs_duration": mcs_duration,
        "mcs_status": mcs_status,
        "ccs_area": ccs_area,
        "merge_cloudnumber": mcs_merge_cloudnumber,
        "split_cloudnumber": mcs_split_cloudnumber,
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
        "merge_cloudnumber": {
            "long_name": "Cloud numbers that merge into MCS",
            "units": "unitless",
            "_FillValue": fillval,
        },
        "split_cloudnumber": {
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