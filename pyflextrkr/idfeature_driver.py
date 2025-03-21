import sys
import logging
import xarray as xr
import pandas as pd
import dask
from dask.distributed import wait
from pyflextrkr.ft_utilities import subset_files_timerange, convert_to_cftime

def idfeature_driver(config):
    """
    Driver for feature identification.

    Args:
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        Feature identification are written to netCDF files.
    """

    logger = logging.getLogger(__name__)
    logger.info('Identifying features from raw data')

    clouddata_path = config["clouddata_path"]
    databasename = config["databasename"]
    start_basetime = config.get("start_basetime", None)
    end_basetime = config.get("end_basetime", None)
    # time_format = config["time_format"]
    run_parallel = config["run_parallel"]
    feature_type = config["feature_type"]
    input_format = config.get("input_format", "netcdf")

    # Load function depending on feature_type
    if feature_type == "generic":
        from pyflextrkr.idfeature_generic import idfeature_generic as id_feature
    elif feature_type == "radar_cells":
        from pyflextrkr.idcells_reflectivity import idcells_reflectivity as id_feature
    elif "tb_pf" in feature_type:
        from pyflextrkr.idclouds_tbpf import idclouds_tbpf as id_feature
    elif "coldpool" in feature_type:
        from pyflextrkr.idcoldpool import idcoldpool as id_feature
    else:
        logger.critical(f"ERROR: Unknown feature_type: {feature_type}")
        logger.critical("Tracking will now exit.")
        sys.exit()

    if input_format.lower() == "zarr":

        # Get precipitation data info from config
        precipdata_path = config["precipdata_path"]
        precipdata_basename = config["precipdata_basename"]
        start_date = config["startdate"]
        end_date = config["enddate"]

        # OLR Zarr filename
        fn_olr = f"{clouddata_path}{databasename}.zarr"
        fn_pr = f"{precipdata_path}{precipdata_basename}.zarr"
        # Read HEALPix zarr file
        ds_olr = xr.open_dataset(fn_olr)
        ds_pr = xr.open_dataset(fn_pr)
        # Check the calendar type of the time coordinate
        calendar = ds_olr['time'].dt.calendar
        # Add coordinates (lat and lon)
        # ds_olr = ds_olr.pipe(egh.attach_coords)
        # ds_pr = ds_pr.pipe(egh.attach_coords)
        # Convert start_date and end_date to pandas.Timestamp
        start_datetime = pd.to_datetime(start_date, format='%Y%m%d.%H%M')
        end_datetime = pd.to_datetime(end_date, format='%Y%m%d.%H%M')
        # Convert pandas.Timestamp to cftime objects based on the calendar type
        start_datetime_cftime = convert_to_cftime(start_datetime, calendar)
        end_datetime_cftime = convert_to_cftime(end_datetime, calendar)
        # Subset the Dataset using the cftime objects
        ds_olr = ds_olr.sel(time=slice(start_datetime_cftime, end_datetime_cftime))
        ds_pr = ds_pr.sel(time=slice(start_datetime_cftime, end_datetime_cftime))

        # Get the number of time steps
        nfiles = ds_olr.sizes['time']
        logger.info(f"Total number of time steps to process: {nfiles}")

        # Serial
        if run_parallel == 0:
            for ifile in range(0, nfiles):
                # Subset one time from the DataSets and combine them
                ds = xr.merge([ds_olr.isel(time=ifile), ds_pr.isel(time=ifile)])
                id_feature(ds, config)
        # Parallel
        elif run_parallel >= 1:
            results = []
            for ifile in range(0, nfiles):
                # Subset one time from the DataSets and combine them
                ds = xr.merge([ds_olr.isel(time=ifile), ds_pr.isel(time=ifile)])
                result = dask.delayed(id_feature)(ds, config)
                results.append(result)
            final_result = dask.compute(*results)
            wait(final_result)
        else:
            sys.exit('Valid parallelization flag not provided')

    elif input_format.lower() == "netcdf":

        time_format = config["time_format"]

        # Identify files to process
        infiles_info = subset_files_timerange(
            clouddata_path,
            databasename,
            start_basetime=start_basetime,
            end_basetime=end_basetime,
            time_format=time_format,
        )
        # Get file list
        rawdatafiles = infiles_info[0]
        nfiles = len(rawdatafiles)
        logger.info(f"Total number of files to process: {nfiles}")

        # Serial
        if run_parallel == 0:
            for ifile in range(0, nfiles):
                id_feature(rawdatafiles[ifile], config)
        # Parallel
        elif run_parallel >= 1:
            results = []
            for ifile in range(0, nfiles):
                result = dask.delayed(id_feature)(rawdatafiles[ifile], config)
                results.append(result)
            final_result = dask.compute(*results)
            wait(final_result)
        else:
            sys.exit('Valid parallelization flag not provided')

    logger.info('Done with features from raw data.')
    return

