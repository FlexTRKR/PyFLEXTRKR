import numpy as np
import time
import os
import sys
import xarray as xr
import logging
from pyflextrkr.ft_utilities import load_sparse_trackstats

def link_mergesplit_tracks(config):
    """
    Link small merge or split tracks to main tracks.

    Args:
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        statistics_outfile: string
            Output track statistics file name.
    """

    finalstats_filebase = config["finalstats_filebase"]
    trackstats_sparse_filebase = config["trackstats_sparse_filebase"]
    stats_path = config["stats_outpath"]
    startdate = config["startdate"]
    enddate = config["enddate"]
    time_resolution = config["datatimeresolution"]
    duration_range = config["duration_range"]
    maintrack_area_thresh = config["maintrack_area_thresh"]
    maintrack_lifetime_thresh = config["maintrack_lifetime_thresh"]
    split_duration = config["split_duration"]
    merge_duration = config["merge_duration"]
    nmaxmerge = config["nmaxlinks"]
    tracks_dimname = config["tracks_dimname"]
    times_dimname = config["times_dimname"]
    tracks_idx_varname = f"{tracks_dimname}_indices"
    times_idx_varname = f"{times_dimname}_indices"
    fillval = config["fillval"]
    fillval_f = np.nan
    max_trackduration = int(max(duration_range))

    np.set_printoptions(threshold=np.inf)
    logger = logging.getLogger(__name__)
    logger.info("Identifying main tracks based on track statistics")

    # Output stats file name
    statistics_outfile = f"{stats_path}{finalstats_filebase}{startdate}_{enddate}.nc"

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
    trackstat_area = sparse_dict["area"]
    basetime = sparse_dict["base_time"]
    cloudnumbers = sparse_dict["cloudnumber"]
    track_status = sparse_dict["track_status"]

    logger.info(f"Number of tracks to process: {ntracks_all}")

    # Convert track duration to physical time unit
    trackstat_lifetime = np.multiply(track_duration, time_resolution)

    ###################################################################################
    # Identify main tracks (using simple lifetime & max area thresholds)

    # Get track lifetime maximum area
    trackstat_maxarea = trackstat_area.max(axis=1).toarray().squeeze()

    maintrack_idx = np.array(np.where(
        (trackstat_lifetime >= maintrack_lifetime_thresh) &
        (trackstat_maxarea >= maintrack_area_thresh)
    ))[0]
    # Provide warning message and exit if no main track identified
    if maintrack_idx == []:
        logger.critical("WARNING: No main track identified.")
        logger.critical(f"Tracking will now exit.")
        sys.exit()

    # Subset main track index
    ntracks_main = len(maintrack_idx)
    logger.info(f"Number of main track defined: {ntracks_main}")

    # maintrack_idx is the index number, want the track number so add one
    maintracknumbers = np.copy(maintrack_idx) + 1

    ###################################################################################
    # Find small merge or split tracks and link to main tracks
    merge_cloudnumber = np.full((ntracks_main, max_trackduration, nmaxmerge), fillval, dtype=np.int32)
    merge_status = np.full((ntracks_main, max_trackduration, nmaxmerge), fillval, dtype=np.int32)
    merge_area = np.full((ntracks_main, max_trackduration, nmaxmerge), fillval_f, dtype=np.float32)
    split_cloudnumber = np.full((ntracks_main, max_trackduration, nmaxmerge), fillval, dtype=np.int32)
    split_status = np.full((ntracks_main, max_trackduration, nmaxmerge), fillval, dtype=np.int32)
    split_area = np.full((ntracks_main, max_trackduration, nmaxmerge), fillval_f, dtype=np.float32)

    # Loop through each main track and link small clouds merging in
    for itrack in np.arange(0, ntracks_main):
        ###################################################################################
        # Find tracks that end as merging with the main track
        mergetrack_idx = np.where(end_merge_tracknumber == maintracknumbers[itrack])[0]
        if len(mergetrack_idx) > 0:
            # Isolate merge tracks that have short duration
            mergetrack_idx = mergetrack_idx[trackstat_lifetime[mergetrack_idx] < merge_duration]

            # Make sure the merge tracks are not main track
            mergetrack_idx = mergetrack_idx[np.isin(mergetrack_idx, maintracknumbers, invert=True)]
            if len(mergetrack_idx) > 0:
                # Get data for merging tracks
                mergingcloudnumber = cloudnumbers[mergetrack_idx, :].data
                mergingbasetime = basetime[mergetrack_idx, :].data
                mergingstatus = track_status[mergetrack_idx, :].data
                mergingarea = trackstat_area[mergetrack_idx, :].data

                # Get main track basetime
                itrack_bt = basetime[int(maintracknumbers[itrack]) - 1, :].data

                # Loop through each timestep in the main track
                for t in np.arange(0, len(itrack_bt)):
                    # Find merging cloud times that match current main track time
                    timematch = np.where(mergingbasetime == itrack_bt[int(t)])[0]
                    nmergers = len(timematch)
                    if nmergers > 0:
                        # Find the smaller value between the two
                        # to make sure it fits into the array
                        nmergers_sav = np.min([nmergers, nmaxmerge])
                        merge_cloudnumber[itrack, int(t), 0:nmergers_sav] = mergingcloudnumber[
                            timematch[0:nmergers_sav]]
                        merge_status[itrack, int(t), 0:nmergers_sav] = mergingstatus[timematch[0:nmergers_sav]]
                        merge_area[itrack, int(t), 0:nmergers_sav] = mergingarea[timematch[0:nmergers_sav]]
                        if (nmergers > nmaxmerge):
                            logger.warning(f'WARNING: nmergers ({nmergers}) > nmaxmerge ({nmaxmerge}), ' + \
                                           'only partial merge clouds are saved.')
                            logger.warning(f'Main track index: {itrack}')
                            logger.warning(f'Increase nmaxmerge to avoid this WARNING.')

        ###################################################################################
        # Find tracks that split from the main track
        splittrack_idx = np.where(start_split_tracknumber == maintracknumbers[itrack])[0]
        if len(splittrack_idx) > 0:
            # Isolate split tracks that have short duration
            splittrack_idx = splittrack_idx[trackstat_lifetime[splittrack_idx] < split_duration]

            # Make sure the split tracks are not main track
            splittrack_idx = splittrack_idx[np.isin(splittrack_idx, maintracknumbers, invert=True)]
            if len(splittrack_idx) > 0:
                # Get data for split tracks
                splittingcloudnumber = cloudnumbers[splittrack_idx, :].data
                splittingbasetime = basetime[splittrack_idx, :].data
                splittingstatus = track_status[splittrack_idx, :].data
                splittingarea = trackstat_area[splittrack_idx, :].data

                # Get main track basetime
                itrack_bt = basetime[int(maintracknumbers[itrack]) - 1, :].data

                # Loop through each timestep in the main track
                for t in np.arange(0, len(itrack_bt)):
                    # Find splitting cloud times that match current main track time
                    timematch = np.where(splittingbasetime == itrack_bt[int(t)])[0]
                    nspliters = len(timematch)
                    if nspliters > 0:
                        # Find the smaller value between the two
                        # to make sure it fits into the array
                        nspliters_sav = np.min([nspliters, nmaxmerge])
                        split_cloudnumber[itrack, int(t), 0:nspliters_sav] = splittingcloudnumber[
                            timematch[0:nspliters_sav]]
                        split_status[itrack, int(t), 0:nspliters_sav] = splittingstatus[timematch[0:nspliters_sav]]
                        split_area[itrack, int(t), 0:nspliters_sav] = splittingarea[timematch[0:nspliters_sav]]
                        if (nspliters > nmaxmerge):
                            logger.warning(f'WARNING: nspliters ({nspliters}) > nmaxmerge ({nmaxmerge}), ' + \
                                           'only partial split clouds are saved.')
                            logger.warning(f'Main track index: {itrack}')
                            logger.warning(f'Increase nmaxmerge to avoid this WARNING.')


    ###########################################################################
    # Prepare output dataset

    # Subset tracks in sparse array dictionary
    for ivar in sparse_dict.keys():
        sparse_dict[ivar] = sparse_dict[ivar][maintrack_idx, :]
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
    tracks_coord = np.arange(0, ntracks_main)
    times_coord = np.arange(0, max_trackduration)
    coordlist = {
        tracks_dimname: ([tracks_dimname], tracks_coord),
        times_dimname: ([times_dimname], times_coord),
    }
    # Define 2D Xarray dataset
    ds_2d = xr.Dataset(varlist, coords=coordlist)
    # Subset main tracks from 1D dataset
    # Note: the tracks_dimname cannot be used here as Xarray does not seem to have
    # a method to select data with a string variable
    ds_1d = ds_1d.sel(tracks=maintrack_idx)
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
    dsout = dsout.drop_vars(drop_vars_list, errors='ignore')

    # Define new variables dictionary
    var_dict = {
        "merge_cloudnumber": merge_cloudnumber,
        "split_cloudnumber": split_cloudnumber,
        "merge_area": merge_area,
        "split_area": split_area,
    }
    var_attrs = {
        "merge_cloudnumber": {
            "long_name": "Cloud numbers that merge into main tracks",
            "units": "unitless",
            "_FillValue": fillval,
        },
        "split_cloudnumber": {
            "long_name": "Cloud numbers that split from main tracks",
            "units": "unitless",
            "_FillValue": fillval,
        },
        "merge_area": {
            "long_name": "Feature area that merge into main tracks",
            "units": "km^2",
            "_FillValue": fillval_f,
        },
        "split_area": {
            "long_name": "Feature area that split from main tracks",
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
    dsout.attrs["Title"] = "Statistics of each track with merge/split linked"
    dsout.attrs["Created_on"] = time.ctime(time.time())
    dsout.attrs["maintrack_area_thresh"] = maintrack_area_thresh
    dsout.attrs["maintrack_lifetime_thresh"] = maintrack_lifetime_thresh
    dsout.attrs["split_duration"] = split_duration
    dsout.attrs["merge_duration"] = merge_duration

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