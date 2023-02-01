import numpy as np
import os
import sys
import xarray as xr
import time
import logging
import dask
from dask.distributed import wait
from pyflextrkr.ft_utilities import subset_files_timerange
from pyflextrkr.matchtbpf_func import matchtbpf_singlefile

def match_tbpf_tracks(config):
    """
    Match Tb tracked MCS with precipitation to calculate PF statistics.

    Args:
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        statistics_outfile: string
            MCS PF track statistics file name.
    """

    mcstbstats_filebase = config["mcstbstats_filebase"]
    mcspfstats_filebase = config["mcspfstats_filebase"]
    stats_path = config["stats_outpath"]
    tracking_outpath = config["tracking_outpath"]
    cloudid_filebase = config["cloudid_filebase"]
    startdate = config["startdate"]
    enddate = config["enddate"]
    nmaxpf = config["nmaxpf"]
    tracks_dimname = config["tracks_dimname"]
    times_dimname = config["times_dimname"]
    pf_dimname = config["pf_dimname"]
    run_parallel = config["run_parallel"]
    fillval = config["fillval"]
    # Minimum time difference threshold [second] to match track stats and cloudid pixel files
    match_pixel_dt_thresh = config["match_pixel_dt_thresh"]

    np.set_printoptions(threshold=np.inf)
    logger = logging.getLogger(__name__)
    logger.info("Matching Tb tracked MCS with precipitation to calculate PF statistics")

    # Output stats file name
    statistics_outfile = f"{stats_path}{mcspfstats_filebase}{startdate}_{enddate}.nc"

    #########################################################################################
    # Load MCS track stats
    logger.debug("Loading IR data")
    # logger.debug((time.ctime()))

    mcsirstats_file = f"{stats_path}{mcstbstats_filebase}{startdate}_{enddate}.nc"
    ds = xr.open_dataset(mcsirstats_file,
                         mask_and_scale=False,
                         decode_times=False)
    ir_ntracks = ds.dims[tracks_dimname]
    ir_nmaxlength = ds.dims[times_dimname]
    ir_basetime = ds["base_time"].values
    ir_cloudnumber = ds["cloudnumber"].values
    ir_mergecloudnumber = ds["merge_cloudnumber"].values
    ir_splitcloudnumber = ds["split_cloudnumber"].values

    #########################################################################################
    # Find cloudid files and get their basetime
    infiles_info = subset_files_timerange(
        tracking_outpath,
        cloudid_filebase,
        config["start_basetime"],
        config["end_basetime"],
        time_format="yyyymodd_hhmm",
    )
    cloudidfile_list = infiles_info[0]
    cloudidfile_basetime = infiles_info[1]
    nfiles = len(cloudidfile_list)

    #########################################################################################
    # Find precipitation feature in each mcs
    logger.debug(("Total Number of Tracks:" + str(ir_ntracks)))
    logger.debug("Looping over each pixel file")
    logger.debug((time.ctime()))

    # Create a list to store matchindices for each pixel file
    trackindices_all = []
    timeindices_all = []
    results = []

    # Loop over each pixel file to calculate PF statistics
    for ifile in range(nfiles):
        filename = cloudidfile_list[ifile]

        # Find all matching time indices from MCS stats file to the current cloudid file
        matchindices = np.array(np.where(np.abs(ir_basetime - cloudidfile_basetime[ifile]) < match_pixel_dt_thresh))
        # The returned match indices are for [tracks, times] dimensions respectively
        idx_track = matchindices[0]
        idx_time = matchindices[1]

        # Get cloudnumbers for this time (file)
        file_cloudnumber = ir_cloudnumber[idx_track, idx_time]
        file_mergecloudnumber = ir_mergecloudnumber[idx_track, idx_time, :]
        file_splitcloudnumber = ir_splitcloudnumber[idx_track, idx_time, :]

        # Save matchindices for the current pixel file to the overall list
        trackindices_all.append(idx_track)
        timeindices_all.append(idx_time)

        # Call function to calculate PF stats
        # Serial
        if run_parallel == 0:
            result = matchtbpf_singlefile(
                filename,
                file_cloudnumber,
                file_mergecloudnumber,
                file_splitcloudnumber,
                config,
            )
            results.append(result)
        # Parallel
        elif run_parallel >= 1:
            result = dask.delayed(matchtbpf_singlefile)(
                filename,
                file_cloudnumber,
                file_mergecloudnumber,
                file_splitcloudnumber,
                config,
            )
            results.append(result)
        else:
            sys.exit('Valid parallelization flag not provided.')

    if run_parallel == 0:
        final_result = results
    elif run_parallel >= 1:
        # Trigger dask computation
        final_result = dask.compute(*results)
        wait(final_result)
    else:
        sys.exit('Valid parallelization flag not provided.')


    #########################################################################################
    # Create arrays to store output
    logger.debug("Collecting track PF statistics.")

    maxtracklength = ir_nmaxlength
    numtracks = ir_ntracks

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

    # Loop over variable list to create the dictionary entry
    pf_dict = {}
    pf_dict_attrs = {}
    for ivar in var_names:
        pf_dict[ivar] = np.full((numtracks, maxtracklength, nmaxpf), np.nan, dtype=np.float32)
        pf_dict_attrs[ivar] = var_attrs[ivar]
    # Update 2D variables
    var_names_2d = ["pf_npf",
                    "pf_landfrac",
                    "total_rain",
                    "total_heavyrain",
                    "rainrate_heavyrain"]
    for ivar in var_names_2d:
        pf_dict[ivar] = np.full((numtracks, maxtracklength), np.nan, dtype=np.float32)

    # Collect results
    for ifile in range(0, nfiles):
        if final_result[ifile] is not None:
            # Get the return results for this pixel file
            # The result is a tuple: (out_dict, out_dict_attrs)
            # The first entry is the dictionary containing the variables
            iResult = final_result[ifile][0]

            # Get trackindices and timeindices for this file
            trackindices = trackindices_all[ifile]
            timeindices = timeindices_all[ifile]

            # Loop over each variable and assign values to output dictionary
            for ivar in var_names:
                if iResult[ivar].ndim == 1:
                    pf_dict[ivar][trackindices,timeindices] = iResult[ivar]
                if iResult[ivar].ndim == 2:
                    pf_dict[ivar][trackindices,timeindices,:] = iResult[ivar]

    # Define a dataset containing all PF variables
    varlist = {}
    # Define output variable dictionary
    for key, value in pf_dict.items():
        if value.ndim == 1:
            varlist[key] = ([tracks_dimname], value, pf_dict_attrs[key])
        if value.ndim == 2:
            varlist[key] = ([tracks_dimname, times_dimname], value, pf_dict_attrs[key])
        if value.ndim == 3:
            varlist[key] = ([tracks_dimname, times_dimname, pf_dimname], value, pf_dict_attrs[key])

    # Define coordinate list
    coordlist = {
        tracks_dimname: ([tracks_dimname], np.arange(0, numtracks)),
        times_dimname: ([times_dimname], np.arange(0, maxtracklength)),
        pf_dimname: ([pf_dimname], np.arange(0, nmaxpf)),
    }

    # Define global attributes
    gattrlist = {
        "nmaxpf": nmaxpf,
        "PF_rainrate_thresh": config["pf_rr_thres"],
        "heavy_rainrate_thresh": config["heavy_rainrate_thresh"],
        "landfrac_thresh": config["landfrac_thresh"],
    }

    # Define output Xarray dataset
    ds_pf = xr.Dataset(varlist, coords=coordlist, attrs=gattrlist)

    # Merge IR and PF datasets
    dsout = xr.merge([ds, ds_pf], compat="override", combine_attrs="no_conflicts")
    # Update time stamp
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
