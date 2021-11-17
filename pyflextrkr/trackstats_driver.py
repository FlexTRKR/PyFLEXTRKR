import numpy as np
import xarray as xr
import os
import time
import gc
import logging
import dask
from dask.distributed import Client, LocalCluster
from pyflextrkr.trackstats_func import calc_stats_singlefile, adjust_mergesplit_numbers, get_track_startend_status
from pyflextrkr import netcdf_io_trackstats as net

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

    t0_step4 = time.time()
    logger.info('Calculating cell statistics')

    # Get values from config dictionary
    tracknumbers_filebase = config["tracknumbers_filebase"]
    trackstats_filebase = config["trackstats_filebase"]
    startdate = config["startdate"]
    enddate = config["enddate"]
    stats_path = config["stats_outpath"]
    lengthrange = config["lengthrange"]
    run_parallel = config["run_parallel"]
    nprocesses = config["nprocesses"]

    # Set output filename
    trackstats_outfile = f"{stats_path}{trackstats_filebase}{startdate}_{enddate}.nc"

    # Load track data
    logger.info("Loading gettracks data")
    cloudtrack_file = f"{stats_path}{tracknumbers_filebase}{startdate}_{enddate}.nc"
    ds = xr.open_dataset(cloudtrack_file,
                         mask_and_scale=False,
                         decode_times=False,
                         concat_characters=True)
    numtracks = ds["ntracks"]
    cloudidfiles = ds["cloudid_files"].values
    nfiles = ds.sizes["nfiles"]
    numcharfilename = ds.sizes['ncharacters']
    tracknumbers = ds["track_numbers"].squeeze()
    trackreset = ds["track_reset"].squeeze()
    tracksplit = ds["track_splitnumbers"].squeeze()
    trackmerge = ds["track_mergenumbers"].squeeze()
    trackstatus = ds["track_status"].squeeze()
    ds.close()

    #########################################################################################
    # loop over files. Calculate statistics and organize matrices by tracknumber and cloud
    logger.info("Looping over files and calculating statistics for each file")
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
    if run_parallel == 1:
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


    #########################################################################################
    # Create arrays to store output
    logger.info("Creating arrays for track statistics.")

    fillval = -9999
    maxtracklength = int(max(lengthrange))
    numtracks = int(numtracks)

    # Make a variable list from one of the returned dictionaries
    var_names = list(final_result[0][0].keys())
    # Get variable attributes from one of the returned dictionaries
    var_attrs = final_result[0][1]
    # Drop variables from the list
    var_names.remove("uniquetracknumbers")
    var_names.remove("numtracks")

    # Create a dictionary with variable name as key, and output arrays as values
    # Define the first dictionary entry as 'track_duration'
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
        if ivar not in var_names_int:
            out_dict[ivar] = np.full((numtracks, maxtracklength), np.nan, dtype=np.float32)
        else:
            out_dict[ivar] = np.full((numtracks, maxtracklength), fillval, dtype=np.int32)
        out_dict_attrs[ivar] = var_attrs[ivar]
    # Update base_time to be float64, this is important to not lose time precision!
    out_dict["base_time"] = np.full((numtracks, maxtracklength), np.nan, dtype=np.float64)

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

            # Find track lengths that are within maxtracklength
            # Only record these to avoid array index out of bounds
            # itracklength = out_tracklength[tracknumbertmp]
            itracklength = out_dict["track_duration"][tracknumbertmp]
            ridx = itracklength <= maxtracklength
            # Loop over each variable and assign values to output dictionary
            for ivar in var_names:
                out_dict[ivar][
                    tracknumbertmp[ridx],
                    itracklength[ridx]-1] = iResult[ivar][ridx]

    t1_files = (time.time() - t0_files) / 60.0
    logger.info(("Files processing time (min): ", t1_files))
    logger.info("Finish collecting track statistics")

    # # Create a variable list from one of the returned dictionaries
    # var_names = list(final_result[0].keys())
    # # Drop variables from the list
    # var_names.remove('uniquetracknumbers')
    # var_names.remove('numtracks')

    #########################################################################################
    ## Remove tracks that have no cells (tracklength == 0).
    logger.info("Removing tracks with no cells")
    gc.collect()

    cloudindexpresent = np.where(out_dict["track_duration"] > 0)[0]
    # TODO: should filtering with lengthrange be applied?
    # This would filter short merge/split tracks too
    # cloudindexpresent = np.where(
    #     (out_dict["track_duration"] >= min(lengthrange)) &
    #     (out_dict["track_duration"] <= max(lengthrange))
    # )[0]
    numtracks = len(cloudindexpresent)

    out_dict["track_duration"] = out_dict["track_duration"][cloudindexpresent]
    for ivar in var_names:
        out_dict[ivar] = out_dict[ivar][cloudindexpresent, :]

    #########################################################################################
    # Correct merger and split cloud numbers
    logger.info("Correcting mergers and splits")
    t0_ms = time.time()

    adjusted_out_mergenumber, \
    adjusted_out_splitnumber = adjust_mergesplit_numbers(
        out_dict["merge_tracknumbers"],
        out_dict["split_tracknumbers"],
        cloudindexpresent,
        fillval,
    )

    # Replace merge/split number with the adjusted ones
    out_dict["merge_tracknumbers"] = adjusted_out_mergenumber
    out_dict["split_tracknumbers"] = adjusted_out_splitnumber

    t1_ms = (time.time() - t0_ms) / 60.0
    logger.info(("Correct merge/split processing time (min): ", t1_ms))
    logger.info("Merge and split adjustments done")

    #########################################################################################
    # Record starting and ending status
    logger.info("Getting starting and ending status")
    t0_status = time.time()

    out_dict, out_dict_attrs = get_track_startend_status(
        out_dict, out_dict_attrs, fillval, maxtracklength)

    t1_status = (time.time() - t0_status) / 60.0
    logger.info(("Start/end status processing time (min): ", t1_status))
    logger.info("Starting and ending status done")

    #########################################################################################
    # Write output
    logger.info("Writing trackstats netcdf ... ")
    t0_write = time.time()

    # Check if file already exists. If exists, delete
    if os.path.isfile(trackstats_outfile):
        os.remove(trackstats_outfile)

    varlist = {}
    # Define output variable dictionary
    for key, value in out_dict.items():
        if value.ndim == 1:
            varlist[key] = ([config["tracks_dimname"]], value, out_dict_attrs[key])
        if value.ndim == 2:
            varlist[key] = ([config["tracks_dimname"], config["times_dimname"]], value, out_dict_attrs[key])

    # Define coordinate list
    coordlist = {
        config["tracks_dimname"]: ([config["tracks_dimname"]], np.arange(0, numtracks)),
        config["times_dimname"]: ([config["times_dimname"]], np.arange(0, maxtracklength)),
    }

    # Define global attributes
    gattrlist = {
        'Title': 'Statistics of each track',
        'Institution': 'Pacific Northwest National Laboratory',
        'Contact': 'Zhe Feng, zhe.feng@pnnl.gov',
        "source": config["datasource"],
        "description": config["datadescription"],
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
                    unlimited_dims=config["tracks_dimname"],
                    encoding=encoding)

    t1_write = (time.time() - t0_write) / 60.0
    logger.info(("Writing file time (min): ", t1_write))
    logger.info((time.ctime()))
    logger.info(("Output saved as: ", trackstats_outfile))
    t1_step4 = (time.time() - t0_step4) / 60.0
    logger.info(("Step 4 processing time (min): ", t1_step4))

    return trackstats_outfile
