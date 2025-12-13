import sys
import glob
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
    databasename = config.get("databasename", "")
    start_basetime = config.get("start_basetime", None)
    end_basetime = config.get("end_basetime", None)
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

        if "catalog_file" in config:
            import intake     # For catalogs

            # Get catalog info from config
            catalog_file = config["catalog_file"]
            catalog_location = config.get("catalog_location", None)
            catalog_source = config["catalog_source"]
            catalog_params = config.get("catalog_params", {})
            olr_varname = config['olr_varname']
            pcp_varname = config['pcp_varname']
            start_date = config["startdate"]
            end_date = config["enddate"]

            # Load the catalog
            logger.info(f"Loading HEALPix catalog: {catalog_file}")
            in_catalog = intake.open_catalog(catalog_file)
            if catalog_location is not None:
                in_catalog = in_catalog[catalog_location]

            # Get the DataSet from the catalog
            ds = in_catalog[catalog_source](**catalog_params).to_dask()

        else:
            # Get HEALPix path and basename from config
            healpix_path = config["healpix_path"]
            healpix_basename = config["healpix_basename"]
            start_date = config["startdate"]
            end_date = config["enddate"]
            pass_varname = config.get("pass_varname", None)

            # TODO: make file search more robust
            in_file = sorted(glob.glob(f"{healpix_path}{healpix_basename}*.zarr"))[0]
            # Read in Zarr data directly
            ds = xr.open_zarr(in_file)

        # Make a list of required variables based on feature_type
        if "tb_pf" in feature_type:
            keep_vars = [olr_varname, pcp_varname]
        elif "radar_cells" in feature_type:
            reflectivity_varname = config["reflectivity_varname"]
            if pass_varname is not None:
                keep_vars = [reflectivity_varname] + list(pass_varname)
            else:
                keep_vars = [reflectivity_varname]
        else:
            # Keep all variables for other feature types
            keep_vars = list(ds.data_vars)

        # Subset to keep only the required variables
        all_vars = list(ds.data_vars)
        drop_vars = [var for var in all_vars if var not in keep_vars]
        ds = ds.drop_vars(drop_vars)
        
        # Check the calendar type of the time coordinate
        calendar = ds['time'].dt.calendar
        # Convert start_date and end_date to pandas.Timestamp
        start_datetime = pd.to_datetime(start_date, format='%Y%m%d.%H%M')
        end_datetime = pd.to_datetime(end_date, format='%Y%m%d.%H%M')
        # Convert pandas.Timestamp to cftime objects based on the calendar type
        start_datetime_cftime = convert_to_cftime(start_datetime, calendar)
        end_datetime_cftime = convert_to_cftime(end_datetime, calendar)
        # Subset the Dataset using the cftime objects
        ds = ds.sel(time=slice(start_datetime_cftime, end_datetime_cftime))

        # Get the number of time steps
        nfiles = ds.sizes['time']
        logger.info(f"Total number of time steps to process: {nfiles}")

        # Serial
        if run_parallel == 0:
            for ifile in range(0, nfiles):
                # Subset one time from the DataSets and combine them
                id_feature(ds.isel(time=ifile), config)
        # Parallel
        elif run_parallel >= 1:
            results = []
            for ifile in range(0, nfiles):
                # Subset one time from the DataSets and combine them
                result = dask.delayed(id_feature)(ds.isel(time=ifile), config)
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

