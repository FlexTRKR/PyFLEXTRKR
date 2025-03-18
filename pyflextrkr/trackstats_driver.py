import numpy as np
import xarray as xr
from scipy.sparse import csr_matrix
import os
import sys
import time
import copy
import gc
import logging
import dask
from dask.distributed import wait
from pyflextrkr.trackstats_func import calc_stats_singlefile, adjust_mergesplit_numbers, get_track_startend_status

def trackstats_driver(config):
    """
    Calculate statistics of track features.

    Args:
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        trackstats_outfile: string
            Track statistics file name.
    """

    np.set_printoptions(threshold=np.inf)
    logger = logging.getLogger(__name__)

    logger.info('Calculating track statistics')

    # Get values from config dictionary
    tracknumbers_filebase = config["tracknumbers_filebase"]
    trackstats_filebase = config["trackstats_filebase"]
    trackstats_sparse_filebase = config["trackstats_sparse_filebase"]
    startdate = config["startdate"]
    enddate = config["enddate"]
    stats_path = config["stats_outpath"]
    duration_range = config["duration_range"]
    run_parallel = config["run_parallel"]
    fillval = config["fillval"]
    tracks_dimname = config["tracks_dimname"]
    times_dimname = config["times_dimname"]
    remove_shorttracks = config["remove_shorttracks"]
    trackstats_dense_netcdf = config["trackstats_dense_netcdf"]
    feature_type = config.get("feature_type", None)
    fillval_f = np.nan

    #Get PBC condition
    pbc_direction = config.get('pbc_direction', "none")
    domain_max = config.get('domain_max', "none")

    # Set output filename
    trackstats_outfile = f"{stats_path}{trackstats_filebase}{startdate}_{enddate}.nc"
    trackstats_sparse_outfile = f"{stats_path}{trackstats_sparse_filebase}{startdate}_{enddate}.nc"

    # Load track data
    logger.debug("Loading tracknumbers data")
    cloudtrack_file = f"{stats_path}{tracknumbers_filebase}{startdate}_{enddate}.nc"
    ds = xr.open_dataset(cloudtrack_file,
                         mask_and_scale=False,
                         decode_times=False,
                         concat_characters=True)
    numtracks = ds["ntracks"]
    cloudidfiles = ds["cloudid_files"].values
    nfiles = ds.sizes["nfiles"]
    tracknumbers = ds["track_numbers"].squeeze()
    trackreset = ds["track_reset"].squeeze()
    tracksplit = ds["track_splitnumbers"].squeeze()
    trackmerge = ds["track_mergenumbers"].squeeze()
    trackstatus = ds["track_status"].squeeze()
    ds.close()

    # Read one cloudid file to get domain & coordinates information
    dspix = xr.open_dataset(cloudidfiles[0])
    # Get 2D lat/lon coordinates
    latitude = dspix["latitude"].values
    longitude = dspix["longitude"].values
    nx = dspix.sizes["lon"]
    ny = dspix.sizes["lat"]
    if feature_type == "radar_cells":
        # Get x, y coordinates, and convert units from [m] to [km]
        # Positions of radar cells are calculated in [km] in calc_stats_singlefile
        x_coords = dspix["x"] / 1000.
        y_coords = dspix["y"] / 1000.
    dspix.close()
    # TODO: @laurapaccini: here is where we can convert domain fraction thresholds to absolute distance thresholds


    #########################################################################################
    # loop over files. Calculate statistics and organize matrices by tracknumber and cloud
    logger.info(f"Total number of files to process: {nfiles}")
    logger.debug("Looping over pixel files and calculating feature statistics")
    t0_files = time.time()

    results = []

    # Serial
    if run_parallel == 0:
        for nf in range(0, nfiles):
            result = calc_stats_singlefile(
                tracknumbers[nf, :],
                cloudidfiles[nf],
                trackstatus[nf, :],
                trackmerge[nf, :],
                tracksplit[nf, :],
                trackreset[nf, :],
                config,
            )
            results.append(result)

        # Serial
        final_result = results

    # Parallel
    elif run_parallel >= 1:
        for nf in range(0, nfiles):
            result = dask.delayed(calc_stats_singlefile)(
                tracknumbers[nf, :],
                cloudidfiles[nf],
                trackstatus[nf, :],
                trackmerge[nf, :],
                tracksplit[nf, :],
                trackreset[nf, :],
                config,
            )
            results.append(result)

        # Trigger dask computation
        final_result = dask.compute(*results)
        wait(final_result)

    else:
        sys.exit('Valid parallelization flag not provided.')


    #########################################################################################
    # Create arrays to store output
    logger.debug("Collecting track statistics")

    max_trackduration = int(max(duration_range))
    numtracks = int(numtracks)

    # Make a variable list and get attributes from one of the returned dictionaries
    # Loop over each return results till one that is not None
    counter = 0
    while counter < nfiles:
        if final_result[counter] is not None:
            var_names = list(final_result[counter][0].keys())
            # Get variable attributes
            var_attrs = final_result[counter][1]
            break
        counter += 1
    # Drop variables from the list
    var_names.remove("uniquetracknumbers")
    var_names.remove("numtracks")

    # Sparse array indices
    tracks_idx_varname = f"{tracks_dimname}_indices"
    times_idx_varname = f"{times_dimname}_indices"
    # Create a dictionary with variable name as key, and output arrays as values
    out_dict = {
        "track_duration": np.zeros(numtracks, dtype=np.int32),
    }
    # Create a matching dictionary for variable attributes
    out_dict_attrs = {
        "track_duration": {
            "long_name": "Duration of each track",
            "units": "unitless",
            "comments": "Multiply by time_resolution_hour to convert to physical units",
        },
    }
    # Variables that should be int type
    var_names_int = ["cloudnumber",
                     "track_status",
                     "track_interruptions",
                     "merge_tracknumbers",
                     "split_tracknumbers"]
    # Loop over variable list to create the dictionary entry
    for ivar in var_names:
        out_dict[ivar] = np.array([])
        out_dict_attrs[ivar] = var_attrs[ivar]

    # Initialize row/col indices arrays
    row_idx = np.array([])
    col_idx = np.array([])

    # Collect results
    for nf in range(0, nfiles):
        # Get the return results for this pixel file
        # The result is a tuple: (out_dict, out_dict_attrs)
        # The first entry is the dictionary containing the variables
        iResult = final_result[nf][0]
        if iResult is not None:
            # unique tracknumbers in the current file
            tracknumbertmp = iResult["uniquetracknumbers"] - 1
            # number of tracks in the current file
            numtrackstmp = iResult["numtracks"]

            # Record the current length of the track by adding 1
            out_dict["track_duration"][tracknumbertmp] = (
                    out_dict["track_duration"][tracknumbertmp] + 1
            )

            # Find track lengths that are within max_trackduration
            # Only record these to avoid array index out of bounds
            # itracklength = out_tracklength[tracknumbertmp]
            itracklength = out_dict["track_duration"][tracknumbertmp]
            ridx = itracklength <= max_trackduration
            # Loop over each variable and assign values to output dictionary
            for ivar in var_names:
                # Concatenate arrays for 2D variables
                out_dict[ivar] = np.concatenate(
                    (out_dict[ivar], iResult[ivar])
                )
            # row, column indices for sparse matrix
            # row:tracks, col:times
            row_idx = np.concatenate((row_idx, tracknumbertmp[ridx])).astype(int)
            col_idx = np.concatenate((col_idx, itracklength[ridx] - 1)).astype(int)

    #########################################################################################
    # Check data max duration against config set up
    # Provide warning message and exit if 'duration_range' is too short
    data_max_trackduration = np.nanmax(out_dict["track_duration"])
    if data_max_trackduration > max_trackduration:
        logger.critical(f"WARNING: Max track duration in data ({data_max_trackduration}) " +
                        f"exceeds 'duration_range' ({duration_range}) in the config file!")
        logger.critical(f"This would cause missing statistics in long-lived tracks!")
        logger.critical(f"Increase 'duration_range' in the config file.")
        logger.critical(f"Tracking will now exit.")
        sys.exit()

    # Convert 2D variables to sparse arrays
    row_col_ind = (row_idx, col_idx)
    shape_2d = (numtracks, max_trackduration)
    for ivar in var_names:
        if ivar not in var_names_int:
            if ivar == "base_time":
                out_dict[ivar] = csr_matrix(
                    (out_dict[ivar], row_col_ind), shape=shape_2d, dtype=np.float64,
                )
            else:
                out_dict[ivar] = csr_matrix(
                    (out_dict[ivar], row_col_ind), shape=shape_2d, dtype=np.float32,
                )
        else:
            out_dict[ivar] = csr_matrix(
                (out_dict[ivar], row_col_ind), shape=shape_2d, dtype=np.int32,
            )

    t1_files = (time.time() - t0_files) / 60.0
    logger.debug(("Files processing time (min): ", t1_files))
    logger.debug("Finish collecting track statistics")

    trackidx_all = np.arange(0, len(out_dict["track_duration"]))


    #########################################################################################
    # Record starting and ending status
    logger.debug("Getting starting and ending status")
    t0_status = time.time()

    out_dict, out_dict_attrs = get_track_startend_status(
        out_dict, out_dict_attrs, fillval, max_trackduration)

    t1_status = (time.time() - t0_status) / 60.0
    logger.debug(("Start/end status processing time (min): ", t1_status))
    logger.debug("Starting and ending status done")

    #########################################################################################
    # Remove short duration tracks
    if remove_shorttracks == 1:
        logger.debug("Removing short duration tracks")
        gc.collect()

        # Find tracks that have positive duration
        trackidx_hascloud = np.where(out_dict["track_duration"] > 0)[0]
        # Tracks that are not starting from a split, or ending in a merge
        mask_notmergesplit = np.logical_and(
            (out_dict['start_split_cloudnumber'] == fillval),
            (out_dict['end_merge_cloudnumber'] == fillval),
        )
        # Tracks that are shorter than the minimum duration
        mask_shortduration = out_dict["track_duration"] < min(duration_range)
        # Track indices that are short and not a merge or split
        # These tracks have no meaningful use
        trackidx_remove = np.where(mask_notmergesplit & mask_shortduration)[0]
        # Get track indices that exclude these short tracks
        trackidx_notshort = trackidx_all[~np.isin(trackidx_all, trackidx_remove)]

        # Find the track indices in both arrays
        trackidx_keep = np.intersect1d(trackidx_hascloud, trackidx_notshort)
        # Keep these tracks for all variables in the output dictionary
        numtracks = len(trackidx_keep)
        for ivar in out_dict.keys():
            if out_dict[ivar].ndim == 1:
                out_dict[ivar] = out_dict[ivar][trackidx_keep]
            elif out_dict[ivar].ndim == 2:
                out_dict[ivar] = out_dict[ivar][trackidx_keep, :]

    else:
        numtracks = len(out_dict["track_duration"])
        trackidx_keep = trackidx_all

    #########################################################################################
    # Correct merger and split cloud numbers
    logger.debug("Correcting mergers and splits")
    t0_ms = time.time()

    adjusted_out_mergenumber, \
    adjusted_out_splitnumber = adjust_mergesplit_numbers(
        out_dict["merge_tracknumbers"].data,
        out_dict["split_tracknumbers"].data,
        trackidx_keep,
        fillval,
    )

    # Get the row/col indices by converting 'base_time' to COO format
    row_out = out_dict['base_time'].tocoo().row
    col_out = out_dict['base_time'].tocoo().col

    # Convert adjusted merge/split number to sparse arrays
    # and update the dictionary
    row_col_ind = (row_out, col_out)
    shape_2d = (numtracks, max_trackduration)
    out_dict["merge_tracknumbers"] = csr_matrix(
        (adjusted_out_mergenumber, row_col_ind), shape=shape_2d, dtype=np.int32,
    )
    out_dict["split_tracknumbers"] = csr_matrix(
        (adjusted_out_splitnumber, row_col_ind), shape=shape_2d, dtype=np.int32,
    )
    t1_ms = (time.time() - t0_ms) / 60.0
    logger.debug(("Correct merge/split processing time (min): ", t1_ms))
    logger.debug("Merge and split adjustments done")

    #########################################################################################
    # Record starting and ending status again
    # because the tracknumbers have been adjusted
    out_dict, out_dict_attrs = get_track_startend_status(
        out_dict, out_dict_attrs, fillval, max_trackduration)

    #########################################################################################
    # Prepare to write output file
    # Add tracks/times indices to output dictionary
    out_dict[tracks_idx_varname] = row_out
    out_dict[times_idx_varname] = col_out
    out_dict_attrs[tracks_idx_varname] = {
        "long_name": "Tracks indices for constructing sparse array",
    }
    out_dict_attrs[times_idx_varname] = {
        "long_name": "Times indices for constructing sparse array",
    }

    #########################################################################################
    # Adjust lat,lon positions for PBC
    if pbc_direction != 'none':
        # Example invocation
        # TODO: @laurapaccini: add a list of variable names to adjust positions
        # Note: depending on feature_type (e.g., 'radar_cells'), there could be more variables to adjust
        # 	feature_type: 'radar_cells': 
		#       - core_meanlat, core_meanlon, core_mean_y, core_mean_x, 
        #       - cell_meanlat, cell_meanlon, cell_mean_y, cell_mean_x
        adjust_positions_for_tracks(out_dict, domain_max=domain_max)
        # Add attributes for the new variable
        out_dict_attrs['pbc_flag'] = {
            "long_name": "Flag indicating if position adjustment was applied due to PBC",
            "units": "unitless",
            "comments": "1: adjustment applied, 0: no adjustment",
        }

    # Write dense arrays output file
    if trackstats_dense_netcdf == 1:
        write_trackstats_dense(config, fillval, fillval_f,
                               max_trackduration, numtracks, out_dict, out_dict_attrs, times_dimname,
                               times_idx_varname, tracks_dimname, tracks_idx_varname, trackstats_outfile)

    # Write sparse arrays output file
    write_trackstats_sparse(config, numtracks, out_dict_attrs, out_dict, row_out, tracks_dimname,
                            trackstats_sparse_outfile)
    # import pdb; pdb.set_trace()
    return trackstats_outfile

def adjust_positions_for_tracks(out_dict, domain_max):
    """
    Adjust mean latitude and longitude for tracks to ensure continuity
    across periodic boundaries, while preserving sparse matrix structure.

    Args:
        out_dict: dictionary 
            Dictionary containing the output variables with tracks and times.
        domain_max: float
            The maximum value of the domain.

    Returns:
        None. Modifies out_dict in-place, adding a 'pbc_flag' variable.
    """
    # TODO: @laurapaccini: modify this function to take a list of variable names 
    # and adjust them one by one and replace them in the out_dict

    # Get the sparse matrices
    meanlat_sparse = out_dict['meanlat']
    meanlon_sparse = out_dict['meanlon']

    # Get the non-zero data and indices for meanlat
    meanlat_data = meanlat_sparse.data.copy()
    meanlat_indices = meanlat_sparse.indices
    meanlat_indptr = meanlat_sparse.indptr

    # Get the non-zero data and indices for meanlon
    meanlon_data = meanlon_sparse.data.copy()
    meanlon_indices = meanlon_sparse.indices
    meanlon_indptr = meanlon_sparse.indptr

    # Adjust positions for meanlat, collect flags
    pbc_flag_lat = adjust_sparse_positions(meanlat_data, meanlat_indptr, domain_max)

    # Adjust positions for meanlon, collect flags
    pbc_flag_lon = adjust_sparse_positions(meanlon_data, meanlon_indptr, domain_max)

    # Combine flags (if adjustments were made in either latitude or longitude)
    pbc_flag = np.logical_or(pbc_flag_lat, pbc_flag_lon).astype(np.int32)

    # Reconstruct sparse matrices with adjusted data
    out_dict['meanlat'] = csr_matrix((meanlat_data, meanlat_indices, meanlat_indptr),
                                     shape=meanlat_sparse.shape)
    out_dict['meanlon'] = csr_matrix((meanlon_data, meanlon_indices, meanlon_indptr),
                                     shape=meanlon_sparse.shape)
    
    # Create pbc_flag sparse matrix
    # pbc_flag_data = adjustment_flags_combined.astype(np.int32)
    # pbc_flag = csr_matrix((pbc_flag_data, meanlat_indices, meanlat_indptr),
    #                       shape=meanlat_sparse.shape)
    # Add the pbc_flag to out_dict
    out_dict['pbc_flag'] = pbc_flag

def adjust_sparse_positions(data, indptr, domain_max):
    """
    Adjust the positions in the sparse data array in-place to ensure continuity
    across periodic boundaries.

    Args:
        data: np.ndarray()
            Non-zero data values of the sparse matrix.
        indptr: np.ndarray()
            Index pointer array for the sparse matrix.
        domain_max: float
            The maximum value of the domain.

    Returns:
        adjustment_flag: np.ndarray()
            1D array of flags per track indicating if adjustment was made.
    """
    num_tracks = len(indptr) - 1
    adjustment_flag = np.zeros(num_tracks, dtype=bool)  # Initialize adjustment flags

    for i in range(num_tracks):
        start = indptr[i]
        end = indptr[i + 1]
        if start >= end:
            continue  # No data for this track
        track_values = data[start:end]
        # Replace zeros with NaNs if necessary
        track_values = np.where(track_values != 0, track_values, np.nan)
        # Adjust positions and collect adjustment flag
        adjusted_values, adjustment_made = adjust_position_continuous_simple(track_values, domain_max)
        # Update the data array in-place
        data[start:end] = adjusted_values
        # Record the adjustment flag for this track
        adjustment_flag[i] = adjustment_made
    return adjustment_flag

def adjust_position_continuous_simple(values, domain_max):
    """
    Adjust positions in a 1D array to ensure continuity across periodic boundaries.

    Args:
        values: np.ndarray()
            Array of position values for a track.
        domain_max: float
            The maximum value of the domain.

    Returns:
        adjusted_values: np.array()
            Adjusted position values (may exceed domain_max).
        adjustment_made: bool
            True if any adjustments were made, False otherwise.
    """
    # TODO: @laurapaccini: modify this function to take in two values for X & Y separately

    adjusted_values = np.copy(values)
    adjustment_made = False  # Initialize adjustment flag

    for i in range(1, len(values)):
        if np.isnan(values[i]) or np.isnan(values[i - 1]):
            continue
        diff = adjusted_values[i] - adjusted_values[i - 1]
        if diff > domain_max / 2:
            adjusted_values[i:] -= domain_max
            adjustment_made = True  # Adjustment was made
        elif diff < -domain_max / 2:
            adjusted_values[i:] += domain_max
            adjustment_made = True  # Adjustment was made
            
    return adjusted_values, adjustment_made


def write_trackstats_sparse(config, numtracks, out_dict_attrs, out_dict, row_out, tracks_dimname,
                            trackstats_sparse_outfile):
    """
    Write sparse array format track statistics to netCDF file.

    Args:
        config: dictionary
            Dictionary containing config parameters.
        numtracks: int
            Number of tracks.
        out_dict_attrs: dictionary
            Output variable attributes dictionary.
        out_dict: dictionary
            Output variables dictionary.
        row_out: np.array
            Sparse array row indices.
        tracks_dimname: string
            Tracks dimension name.
        trackstats_sparse_outfile: string
            Output trackstats netCDF filename.

    Returns:
        None.
    """
    logger = logging.getLogger(__name__)
    logger.debug("Writing trackstats netcdf (sparse) ... ")
    # Delete file if it already exists
    if os.path.isfile(trackstats_sparse_outfile):
        os.remove(trackstats_sparse_outfile)

    # Flatten the sparse arrays in the output dictionary
    for ivar in out_dict.keys():
        if out_dict[ivar].ndim == 2:
            out_dict[ivar] = out_dict[ivar].data

    sparse_dimname = 'sparse_index'
    varlist = {}
    # Define output variable dictionary
    for key, value in out_dict.items():
        # For 1D variables
        if value.size == numtracks:
            varlist[key] = ([tracks_dimname], value, out_dict_attrs[key])
        # For sparse 2D variables
        if value.size == len(row_out):
            varlist[key] = ([sparse_dimname], value, out_dict_attrs[key])
    # Define coordinate list
    coordlist = {
        tracks_dimname: ([tracks_dimname], np.arange(0, numtracks)),
        sparse_dimname: ([sparse_dimname], np.arange(0, len(row_out))),
    }
    # Define global attributes
    gattrlist = {
        "Title": 'Statistics of each track',
        "Institution": 'Pacific Northwest National Laboratory',
        "Contact": 'Zhe Feng, zhe.feng@pnnl.gov',
        "Created_on": time.ctime(time.time()),
        # "source": config["datasource"],
        # "description": config["datadescription"],
        "startdate": config["startdate"],
        "enddate": config["enddate"],
        "timegap_hour": config["timegap"],
        "time_resolution_hour": config["datatimeresolution"],
        "pixel_radius_km": config["pixel_radius"],
    }
    # Define output Xarray dataset
    dsout = xr.Dataset(varlist, coords=coordlist, attrs=gattrlist)
    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in dsout.data_vars}
    # Write to netcdf file
    dsout.to_netcdf(path=trackstats_sparse_outfile,
                    mode='w',
                    format='NETCDF4',
                    unlimited_dims=tracks_dimname,
                    encoding=encoding)
    logger.info(trackstats_sparse_outfile)
    return


def write_trackstats_dense(config, fillval, fillval_f,
                           max_trackduration, numtracks, out_dict, out_dict_attrs, times_dimname,
                           times_idx_varname, tracks_dimname, tracks_idx_varname, trackstats_outfile):
    """
    Write dense array format track statistics to netCDF file.

    Args:
        config: dictionary
            Dictionary containing config parameters.
        fillval: int
            Fill value for int type variables.
        fillval_f: float
            Fill value for float type variables.
        max_trackduration: int
            Maximum track duration.
        numtracks: int
            Number of tracks.
        out_dict: dictionary
            Output variables dictionary.
        out_dict_attrs: dictionary
            Output variable attributes dictionary.
        times_dimname: string
            Times dimension name.
        times_idx_varname: string
            Times indices variable name.
        tracks_dimname: string
            Tracks dimension name.
        tracks_idx_varname: string
            Tracks indices variable name.
        trackstats_outfile: string
            Output trackstats netCDF filename.

    Returns:
        None.
    """
    logger = logging.getLogger(__name__)
    logger.debug("Writing trackstats netcdf (dense) ... ")
    # Delete file if it already exists
    if os.path.isfile(trackstats_outfile):
        os.remove(trackstats_outfile)

    # Create a deep copy of the sparse matrix dictionary
    out_dict_dense = copy.deepcopy(out_dict)

    # Convert the sparse arrays to dense arrays
    for ivar in out_dict_dense.keys():
        if out_dict_dense[ivar].ndim == 2:
            out_dict_dense[ivar] = out_dict_dense[ivar].toarray()
    # Remove the tracks/times indices variables
    out_dict_dense.pop(tracks_idx_varname, None)
    out_dict_dense.pop(times_idx_varname, None)

    # Create a dense mask for no feature
    mask = out_dict_dense['base_time'] == 0

    varlist = {}
    # Define output variable dictionary
    for key, value in out_dict_dense.items():
        if value.ndim == 1:
            varlist[key] = ([tracks_dimname], value, out_dict_attrs[key])
        if value.ndim == 2:
            # Replace missing values based on variable type
            if isinstance(value[0, 0], np.floating):
                value[mask] = fillval_f
            else:
                value[mask] = fillval
            varlist[key] = ([tracks_dimname, times_dimname], value, out_dict_attrs[key])
    # Define coordinate list
    coordlist = {
        tracks_dimname: ([tracks_dimname], np.arange(0, numtracks)),
        times_dimname: ([times_dimname], np.arange(0, max_trackduration)),
    }
    # Define global attributes
    gattrlist = {
        "Title": 'Statistics of each track',
        "Institution": 'Pacific Northwest National Laboratory',
        "Contact": 'Zhe Feng, zhe.feng@pnnl.gov',
        "Created_on": time.ctime(time.time()),
        # "source": config["datasource"],
        # "description": config["datadescription"],
        "startdate": config["startdate"],
        "enddate": config["enddate"],
        "timegap_hour": config["timegap"],
        "time_resolution_hour": config["datatimeresolution"],
        "pixel_radius_km": config["pixel_radius"],
    }
    # Define output Xarray dataset
    dsout = xr.Dataset(varlist, coords=coordlist, attrs=gattrlist)
    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in dsout.data_vars}
    # Write to netcdf file
    dsout.to_netcdf(path=trackstats_outfile,
                    mode='w',
                    format='NETCDF4',
                    unlimited_dims=tracks_dimname,
                    encoding=encoding)
    logger.info(trackstats_outfile)
    return
