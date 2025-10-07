import xarray as xr
import numpy as np
import os
import glob
import time
import logging
import intake
import requests
import easygems.healpix as egh
from functools import partial
from pyflextrkr.ft_utilities import setup_logging

def remap_to_healpix_zarr(config):
    """
    Convert pixel-level tracking mask NetCDF files to HEALPix grid in Zarr format.
    This function combines the functionality of convert_mask_to_zarr and remap_mask_to_healpix.

    Args:
        config: dictionary
            Dictionary containing config parameters

    Returns:
        out_zarr: string
            Zarr store path for the HEALPix output.
    """
    # Set the logging message level
    setup_logging()
    logger = logging.getLogger(__name__)

    start_time = time.time()
    logger.info(f"Starting NetCDF to HEALPix Zarr conversion: ...")

    # Get client if available
    try:
        from dask.distributed import get_client
        client = get_client()
        parallel = True
        logger.info(f"Using existing Dask client with {len(client.scheduler_info()['workers'])} workers")
    except ValueError:
        logger.warning("No Dask client found, continuing without explicit client")
        client = None
        parallel = False

    # ---------- INPUT CONFIGURATION ----------
    pixeltracking_outpath = config.get("pixeltracking_outpath")
    pixeltracking_filebase = config.get("pixeltracking_filebase", "mcstrack_")
    startdate = config.get("startdate")
    enddate = config.get("enddate")
    outpath = os.path.dirname(os.path.normpath(pixeltracking_outpath)) + "/"
    
    # Get preset-specific configuration
    presets = config.get("zarr_output_presets", {})
    preset_mask = presets.get("mask", {})
    preset_healpix = presets.get("healpix", {})
    write_mask = preset_mask.get("write", False)
    
    # ---------- HEALPIX CONFIGURATION ----------
    catalog_file = config.get("catalog_file")
    catalog_location = config.get("catalog_location", None)
    catalog_source = config.get("catalog_source", "")
    catalog_params = config.get("catalog_params", {})
    catalog_zoom = catalog_params.get("zoom")
    hp_zoom = preset_healpix.get("zoom", catalog_zoom)
    hp_version = preset_healpix.get("version", "v1")
    
    # Update catalog_params to match desired hp zoom level from preset
    if hp_zoom != catalog_zoom:
        catalog_params["zoom"] = hp_zoom
    
    # ---------- OUTPUT FILE PATHS ----------
    # Intermediate lat/lon Zarr (if needed)
    latlon_filebase = preset_mask.get("out_filebase", "mcs_mask_latlon_")
    latlon_zarr = f"{outpath}{latlon_filebase}{startdate}_{enddate}.zarr"
    
    # Final HEALPix Zarr output
    hp_filebase = preset_healpix.get("out_filebase", "mcs_mask_")
    out_zarr = f"{outpath}{hp_filebase}hp{hp_zoom}_{hp_version}.zarr"
    
    # Check if output exists and should be overwritten
    overwrite = preset_mask.get("overwrite", config.get("overwrite_zarr", True))
    if os.path.exists(out_zarr) and not overwrite:
        logger.warning(f"HEALPix Zarr store already exists at {out_zarr} and overwrite=False. Skipping.")
        return out_zarr
    
    # Check catalog file availability
    if catalog_file:
        if catalog_file.startswith(('http://', 'https://')):
            # Handle URL case
            try:
                response = requests.head(catalog_file, timeout=10)
                if response.status_code >= 400:
                    logger.error(f"Catalog URL {catalog_file} returned status code {response.status_code}. Skipping remap.")
                    return out_zarr
            except requests.exceptions.RequestException as e:
                logger.error(f"Error accessing catalog URL {catalog_file}: {str(e)}. Skipping remap.")
                return out_zarr
        elif os.path.isfile(catalog_file) is False:
            # Handle local file case
            logger.error(f"Catalog file {catalog_file} does not exist. Skipping remap.")
            return out_zarr
    else:
        logger.error("Catalog file not specified in config. HEALPix remapping requires a catalog.")
        return out_zarr

    # ---------- CHUNKING CONFIGURATION ----------
    chunksize_time = config.get("chunksize_time", "auto")
    chunksize_lat = config.get("chunksize_lat", "auto")
    chunksize_lon = config.get("chunksize_lon", "auto")
    chunksize_cell = config.get("chunksize_cell", "auto")
    
    # ---------- VARIABLE SELECTION ----------
    # Required coordinate variables
    required_vars = ["time", "lat", "lon"]
    
    # # Special case for "full" preset - include all variables
    # if output_preset == 'full':
    #     # Will be populated with all dataset variables later
    #     config_vars = []
    #     logger.info("Using 'full' preset: all variables will be included")        
    # Normal case - get variables from preset config or main config
    if "var_list" in preset_mask:
        config_vars = preset_mask["var_list"]
    else:
        config_vars = ["tracknumber"]
    
    if isinstance(config_vars, str):
        config_vars = [config_vars]  # Convert string to list if needed
    
    # ---------- READ INPUT FILES ----------
    logger.info("Reading input NetCDF files...")
    file_list = sorted(glob.glob(f"{pixeltracking_outpath}{pixeltracking_filebase}*.nc"))
    logger.info(f"Number of input files: {len(file_list)}")
    
    # Open as a lazy dataset
    ds = xr.open_mfdataset(
        file_list,
        combine="by_coords",
        parallel=parallel,
        chunks={},  # Defer chunking to later
        mask_and_scale=False,
    )
    logger.info(f"Finished reading input files.")
    
    # ---------- PROCESS VARIABLES ----------
    # # For 'full' preset, include all variables from the dataset
    # if output_preset == 'full':
    #     # Get all variable names
    #     all_vars = list(ds.data_vars)
    #     logger.info(f"Full preset: found {len(all_vars)} variables in dataset")
    #     config_vars = all_vars

    # Combine required vars with config vars, ensuring no duplicates
    keep_var_list = list(set(required_vars + config_vars))
    logger.info(f"Variables to include in output: {keep_var_list}")

    # Get variable rename mapping
    if "var_rename" in preset_mask:
        rename_dict = preset_mask["var_rename"]
    else:
        rename_dict = config.get("zarr_var_rename", {})
    
    if rename_dict:
        logger.info(f"Variable renaming map: {rename_dict}")
    else:
        logger.info("No variable renaming will be performed")

    # Filter to keep only existing variables
    variables_to_keep = [v for v in keep_var_list if v in ds.variables]
    
    # Create a rename dictionary containing only the variables that exist in dataset
    active_renames = {k: v for k, v in rename_dict.items() if k in variables_to_keep}
    
    # Select and rename variables in one step
    if active_renames:
        logger.info(f"Renaming variables: {active_renames}")
        ds = ds[variables_to_keep].rename(active_renames)
    else:
        # Simple subset without renaming
        ds = ds[variables_to_keep]
    
    # Apply initial chunking for processing
    ds = ds.chunk({
        "time": min(100, chunksize_time) if isinstance(chunksize_time, int) else "auto", 
        "lat": chunksize_lat, 
        "lon": chunksize_lon
    })
    
    # ---------- INTERMEDIATE ZARR (OPTIONAL) ----------
    if write_mask:
        logger.info(f"Writing intermediate lat/lon Zarr to: {latlon_zarr}")
        
        # Set proper chunking for lat/lon grid
        chunked_ds = ds.chunk({
            "time": chunksize_time, 
            "lat": chunksize_lat, 
            "lon": chunksize_lon
        })
        
        # Report dataset size
        logger.info(f"Dataset dimensions: {dict(chunked_ds.sizes)}")
        logger.info(f"Chunking scheme: time={chunksize_time}, lat={chunksize_lat}, lon={chunksize_lon}")
        
        # Write lat/lon zarr with progress tracking
        if client:
            from dask.distributed import progress
            
            # Write intermediate Zarr
            write_task = chunked_ds.to_zarr(
                latlon_zarr,
                mode="w",
                consolidated=True,
                compute=False
            )
            
            try:
                # Compute with progress tracking
                future = client.compute(write_task)
                logger.info("Writing intermediate Zarr (this may take a while)...")
                progress(future)
                
                result = future.result()
                logger.info("Intermediate Zarr write completed successfully")
                
                # Reload from the Zarr store to ensure everything is consistent
                ds = xr.open_dataset(latlon_zarr, engine='zarr')
                
            except Exception as e:
                logger.error(f"Intermediate Zarr write failed: {str(e)}")
                raise
        else:
            # Compute locally
            chunked_ds.to_zarr(latlon_zarr, mode="w", consolidated=True)
            ds = xr.open_dataset(latlon_zarr, engine='zarr')
    
    # ---------- HEALPIX REMAPPING ----------
    logger.info("Beginning HEALPix remapping...")
    
    # Save longitude sign info
    signed_lon = True if np.min(ds["lon"]) < 0 else False
    logger.info(f"Input data lon coordinate has negative values: {signed_lon}")
    
    # Load the HEALPix catalog
    logger.info(f"Loading HEALPix catalog: {catalog_file}")
    in_catalog = intake.open_catalog(catalog_file)
    if catalog_location is not None:
        in_catalog = in_catalog[catalog_location]
    
    # Get the DataSet from the catalog
    ds_hp = in_catalog[catalog_source](**catalog_params).to_dask()
    # Add lat/lon coordinates to the HEALPix DataSet
    ds_hp = ds_hp.pipe(partial(egh.attach_coords, signed_lon=signed_lon))
    
    # Assign extra coordinates (lon_hp, lat_hp) to the HEALPix coordinates
    # This is needed for limiting the extrapolation during remapping
    lon_hp = ds_hp.lon.assign_coords(cell=ds_hp.cell, lon_hp=lambda da: da)
    lat_hp = ds_hp.lat.assign_coords(cell=ds_hp.cell, lat_hp=lambda da: da)
    
    # Make sure coordinates are fixed before remapping
    ds = fix_coords(ds)

    # Calculate appropriate tolerance based on zoom level
    tolerance = calculate_healpix_tolerance(hp_zoom)
    logger.info(f"Using HEALPix tolerance of {tolerance:.4f}° at zoom level {hp_zoom}")
    
    # Remap DataSet to HEALPix
    logger.info("Applying nearest neighbor remapping to HEALPix grid...")
    fill_value = 0
    dsout_hp = ds.sel(
        lon=lon_hp, lat=lat_hp, method="nearest",
    ).where(partial(is_valid, tolerance=tolerance), fill_value)
    
    # Drop lat/lon coordinates (not needed in HEALPix)
    dsout_hp = dsout_hp.drop_vars(["lat_hp", "lon_hp", "lat", "lon"])
    # Update global attributes
    dsout_hp.attrs['Title'] = f"HEALPix remapped tracking mask data (zoom={hp_zoom})"
    dsout_hp.attrs['zoom'] = hp_zoom
    dsout_hp.attrs["Created_on"] = time.ctime(time.time())
    
    # Optimize cell chunking for HEALPix grid
    chunksize_cell = optimize_healpix_chunks(ds_hp, chunksize_cell, logger)
    
    # Make time chunks more even if needed
    if isinstance(chunksize_time, (int, float)) and chunksize_time != 'auto':
        total_times = dsout_hp.sizes['time']
        chunks = total_times // chunksize_time
        if chunks * chunksize_time < total_times:
            # We have a remainder - try to make chunks more even
            if total_times % chunks == 0:
                chunksize_time = total_times // chunks
            elif total_times % (chunks + 1) == 0:
                chunksize_time = total_times // (chunks + 1)
    
    # Set proper chunking for HEALPix output
    chunked_hp = dsout_hp.chunk({
        "time": chunksize_time, 
        "cell": chunksize_cell, 
    })
    
    # Report dataset size and chunking info
    logger.info(f"HEALPix dataset dimensions: {dict(chunked_hp.sizes)}")
    logger.info(f"HEALPix chunking scheme: time={chunksize_time}, cell={chunksize_cell}")
    
    # ---------- WRITE HEALPIX ZARR OUTPUT ----------
    logger.info(f"Starting HEALPix Zarr write to: {out_zarr}")
    
    # Create a delayed task for Zarr writing
    write_task = chunked_hp.to_zarr(
        out_zarr,
        mode="w",        
        consolidated=True,  # Enable for better performance when reading
        compute=False      # Create a delayed task
    )
    
    # Compute the task, with progress reporting
    if client:
        from dask.distributed import progress
        import psutil

        # Temporarily suppress distributed.shuffle logs during progress display
        shuffle_logger = logging.getLogger('distributed.shuffle')
        original_level = shuffle_logger.level
        shuffle_logger.setLevel(logging.ERROR)  # Only show errors, not warnings

        # Get cluster state information before processing
        memory_usage = client.run(lambda: psutil.Process().memory_info().rss / 1e9)
        logger.info(f"Current memory usage across workers (GB): {memory_usage}")
               
        try:
            # Compute with progress tracking
            future = client.compute(write_task)
            logger.info("Writing HEALPix Zarr (this may take a while)...")
            progress(future)  # Shows a progress bar in notebooks or detailed progress in terminals

            result = future.result()
            logger.info("HEALPix Zarr write completed successfully")
        except Exception as e:
            logger.error(f"HEALPix Zarr write failed: {str(e)}")
            raise
        finally:
            # Restore original log level
            shuffle_logger.setLevel(original_level)
    else:
        # Compute locally if no client
        write_task.compute()
    
    # Cleanup intermediate file if created
    # if not skip_intermediate and os.path.exists(latlon_zarr) and out_zarr != latlon_zarr:
    # if (write_mask == True) and (os.path.exists(latlon_zarr)) and (out_zarr != latlon_zarr):
    #     if config.get("remove_intermediate", False):
    #         logger.info(f"Removing intermediate lat/lon Zarr: {latlon_zarr}")
    #         import shutil
    #         try:
    #             shutil.rmtree(latlon_zarr)
    #         except Exception as e:
    #             logger.warning(f"Failed to remove intermediate file: {str(e)}")
    
    logger.info(f"HEALPix conversion complete: {out_zarr}")
    
    # Log completion time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info(f"Conversion completed in {int(hours):02}:{int(minutes):02}:{int(seconds):02} (hh:mm:ss).")
    
    return out_zarr


def fix_coords(ds, lat_dim="lat", lon_dim="lon", roll=False):
    """
    Fix coordinates in a dataset:
    1. Convert longitude from -180/+180 to 0-360 range (optional)
    2. Roll dataset to start at longitude 0 (optional)
    3. Ensure coordinates are in ascending order
    
    Parameters:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        Dataset with lat/lon coordinates
    lat_dim : str, optional
        Name of latitude dimension, default "lat"
    lon_dim : str, optional
        Name of longitude dimension, default "lon"
    roll : bool, optional, default=False
        If True, convert longitude from -180/+180 to 0-360, and roll the dataset to start at longitude 0
        
    Returns:
    --------
    xarray.Dataset or xarray.DataArray
        Dataset with fixed coordinates
    """
    if roll:
        # Find where longitude crosses from negative to positive (approx. where lon=0)
        lon_0_index = (ds[lon_dim] < 0).sum().item()
        
        # Create indexers for the roll
        lon_indices = np.roll(np.arange(ds.sizes[lon_dim]), -lon_0_index)
        
        # Roll dataset and convert longitudes to 0-360 range
        ds = ds.isel({lon_dim: lon_indices})
        lon360 = xr.where(ds[lon_dim] < 0, ds[lon_dim] + 360, ds[lon_dim])
        ds = ds.assign_coords({lon_dim: lon360})
    
    # Ensure latitude and longitude are in ascending order if needed
    if np.all(np.diff(ds[lat_dim].values) < 0):
        ds = ds.isel({lat_dim: slice(None, None, -1)})
    if np.all(np.diff(ds[lon_dim].values) < 0):
        ds = ds.isel({lon_dim: slice(None, None, -1)})
    
    return ds


def is_valid(ds, tolerance=0.1):
    """
    Limit extrapolation distance to a certain tolerance.
    This is useful for preventing extrapolation of regional data to global HEALPix grid.

    Args:
        ds (xarray.Dataset):
            The dataset containing latitude and longitude coordinates.
        tolerance (float): default=0.1
            The maximum allowed distance in [degrees] for extrapolation.

    Returns:
        xarray.DataSet.
    """
    return (np.abs(ds.lat - ds.lat_hp) < tolerance) & (np.abs(ds.lon - ds.lon_hp) < tolerance)


def calculate_healpix_tolerance(zoom_level):
    """
    Calculate appropriate tolerance for is_valid function based on HEALPix zoom level.
    Returns approximately one grid cell size in degrees.
    
    Args:
        zoom_level (int): HEALPix zoom level
        
    Returns:
        float: Tolerance in degrees
    """
    # Calculate nside from zoom level (nside = 2^zoom)
    # nside determines HEALPix resolution - each increase in zoom doubles the resolution
    nside = 2 ** zoom_level
    
    # Calculate approximate pixel size in degrees
    # Mathematical derivation:
    # - Sphere has total area of 4π steradians (= 4π × (180/π)² sq. degrees)
    # - HEALPix divides sphere into 12 × nside² equal-area pixels
    # - Each pixel has area = 4π × (180/π)² / (12 × nside²) sq. degrees
    # - Linear size = √(pixel area) ≈ 58.6 / nside degrees
    # This gives approximately the angular width of one HEALPix cell
    pixel_size_degrees = 58.6 / nside
    
    return pixel_size_degrees


def optimize_healpix_chunks(ds_hp, chunksize_cell='auto', logger=None):
    """
    Optimize chunking for HEALPix grids based on their hierarchical structure.
    
    Parameters:
    -----------
    ds_hp : xarray.Dataset
        HEALPix dataset containing cell dimension and crs attributes
    chunksize_cell : int or 'auto', optional
        Requested chunk size or 'auto' for automatic optimization
    logger : logging.Logger, optional
        Logger for debug information
        
    Returns:
    --------
    int
        Optimized chunk size for the cell dimension
    """
    
    # Get the total cell count
    total_cells = ds_hp.sizes['cell']
    
    # If a specific chunk size is requested (not 'auto'), return it directly
    if chunksize_cell != 'auto' and isinstance(chunksize_cell, (int, float)):
        return int(chunksize_cell)
    
    # Get the HEALPix nside parameter if available
    nside = getattr(ds_hp.crs, 'healpix_nside', None)
    if nside is None:
        # Estimate nside based on cell count (12 * nside^2 = total cells)
        nside = int(np.sqrt(total_cells / 12))
    
    if logger:
        logger.info(f"Total HEALPix cells: {total_cells}, estimated nside: {nside}")
    
    # For very small grids: divide into 12 chunks (one per base face)
    if total_cells <= 12288:
        chunksize_cell = max(1, total_cells // 12)
    else:
        # For larger grids, create chunks based on nested HEALPix structure
        base_face_size = total_cells // 12
        
        # Try to keep chunks around 262144 cells (good balance for most systems)
        target_chunk_size = 262144
        
        if base_face_size <= target_chunk_size:
            chunksize_cell = base_face_size
        else:
            # Find a divisor of base_face_size that creates chunks close to target size
            # Focus on powers of 4 since HEALPix has quad-tree structure
            for divisor in [2, 4, 8, 16, 32, 64, 128, 256]:
                if base_face_size // divisor <= target_chunk_size:
                    chunksize_cell = base_face_size // divisor
                    break
            else:
                # If no good divisor found, use a power of 4 near target size
                power = max(1, int(np.log(target_chunk_size) / np.log(4)))
                chunksize_cell = 4**power
    
    # Ensure chunk size isn't too small
    chunksize_cell = max(chunksize_cell, 4096)
    
    # Ensure chunk size divides evenly or leaves a smaller final chunk
    if total_cells % chunksize_cell > 0:
        # Try to find a divisor close to our calculated chunk size
        for factor in range(chunksize_cell, chunksize_cell // 2, -1):
            if total_cells % factor == 0:
                chunksize_cell = factor
                break
    
    if logger:
        logger.info(f"Optimized HEALPix chunk size: {chunksize_cell}")
        
    return int(chunksize_cell)