import xarray as xr
import os
import glob
import time
import logging
from pyflextrkr.ft_utilities import setup_logging

def convert_mask_to_zarr(config, output_preset='mask'):
    """
    Convert pixel-level tracking mask NetCDF files to Zarr format.

    Args:
        config: dictionary
            Dictionary containing config parameters
        output_preset: string
            The preset name to determine output configuration
            Options: 'mask', 'tbpr', 'full', or any custom preset defined in config

    Returns:
        out_zarr: string
            Zarr store path.
    """
    # Set the logging message level
    setup_logging()
    logger = logging.getLogger(__name__)

    start_time = time.time()
    logger.info(f"Conversion started for preset: {output_preset}...")

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

    pixeltracking_outpath = config.get("pixeltracking_outpath")
    pixeltracking_filebase = config.get("pixeltracking_filebase", "mcstrack_")
    startdate = config.get("startdate")
    enddate = config.get("enddate")
    outpath = os.path.dirname(os.path.normpath(pixeltracking_outpath)) + "/"
    
    # Get preset-specific configuration
    # Get preset dictionary if it exists in config, otherwise use empty dict
    presets = config.get("zarr_output_presets", {})
    preset_config = presets.get(output_preset, {})
    
    # Get out_filebase - use preset-specific if available, otherwise use default
    out_filebase = preset_config.get("out_filebase", "mcs_mask_latlon_")
    
    # Build output filename
    out_zarr = f"{outpath}{out_filebase}{startdate}_{enddate}.zarr"
    
    # Check if output exists and should be overwritten
    overwrite = preset_config.get("overwrite", config.get("overwrite_zarr", True))
    if os.path.exists(out_zarr) and not overwrite:
        logger.warning(f"Zarr store already exists at {out_zarr} and overwrite=False. Skipping.")
        return out_zarr
    
    # Get chunk size from config
    # chunksize_time = preset_config.get("chunksize_time", config.get("chunksize_time", "auto"))
    # chunksize_lat = preset_config.get("chunksize_lat", config.get("chunksize_lat", "auto"))
    # chunksize_lon = preset_config.get("chunksize_lon", config.get("chunksize_lon", "auto"))
    chunksize_time = config.get("chunksize_time", "auto")
    chunksize_lat = config.get("chunksize_lat", "auto")
    chunksize_lon = config.get("chunksize_lon", "auto")
    
    # Output variable list
    # Required coordinate variables
    required_vars = ["time", "lat", "lon"]
    
    # Special case for "full" preset - include all variables
    if output_preset == 'full':
        # Will be populated with all dataset variables later
        config_vars = []
        logger.info("Using 'full' preset: all variables will be included")        
    # Normal case - get variables from preset config or main config
    elif "var_list" in preset_config:
        config_vars = preset_config["var_list"]
    else:
        config_vars = ["tracknumber"]
    
    if isinstance(config_vars, str):
        config_vars = [config_vars]  # Convert string to list if needed
    
    # Find input files
    file_list = sorted(glob.glob(f"{pixeltracking_outpath}{pixeltracking_filebase}*.nc"))
    logger.info(f"Number of input files: {len(file_list)}")

    # Open as a lazy dataset
    ds = xr.open_mfdataset(
        file_list,
        combine="by_coords",
        parallel=parallel,
        chunks={},  # Defer chunking to to_zarr()
    )
    logger.info(f"Finished reading input files.")
    
    # For 'full' preset, include all variables from the dataset
    if output_preset == 'full':
        # Get all variable names, excluding any internal/hidden variables
        all_vars = list(ds.data_vars)
        logger.info(f"Full preset: found {len(all_vars)} variables in dataset")
        config_vars = all_vars

    # Combine required vars with config vars, ensuring no duplicates
    keep_var_list = list(set(required_vars + config_vars))
    logger.info(f"Variables to include in Zarr output: {keep_var_list}")

    # Get variable rename mapping from preset config if available, otherwise from main config
    if "var_rename" in preset_config:
        rename_dict = preset_config["var_rename"]
    else:
        rename_dict = config.get("zarr_var_rename", {})
    
    if rename_dict:
        logger.info(f"Variable renaming map: {rename_dict}")
    else:
        logger.info("No variable renaming will be performed")

    # Filter and rename variables
    # For 'full' preset, we already have all variables from the dataset
    if output_preset == 'full':
        variables_to_keep = keep_var_list
    else:
        # Otherwise, identify which variables to keep from those requested
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

    # Set proper chunking
    chunked_ds = ds.chunk({
        "time": chunksize_time, 
        "lat": chunksize_lat, 
        "lon": chunksize_lon
    })
    
    # Report dataset size and chunking info
    logger.info(f"Dataset dimensions: {dict(chunked_ds.sizes)}")
    logger.info(f"Chunking scheme: time={chunksize_time}, lat={chunksize_lat}, lon={chunksize_lon}")
    
    # Create a delayed task for Zarr writing
    logger.info(f"Starting Zarr write to: {out_zarr}")
    write_task = chunked_ds.to_zarr(
        out_zarr,
        mode="w",        
        consolidated=True,  # Enable for better performance when reading
        compute=False      # Create a delayed task
    )
    
    # Compute the task, with progress reporting
    if client:
        from dask.distributed import progress

        # Temporarily increase log level for the distributed shuffle module to suppress warnings
        shuffle_logger = logging.getLogger("distributed.shuffle._scheduler_plugin")
        original_level = shuffle_logger.level
        shuffle_logger.setLevel(logging.ERROR)  # Only show errors, not warnings
        
        try:
            # Compute with progress tracking
            future = client.compute(write_task)
            logger.info("Writing to Zarr (this may take a while)...")
            progress(future)  # Shows a progress bar in notebooks or detailed progress in terminals
            
            result = future.result()
            logger.info("Zarr write completed successfully")
        except Exception as e:
            logger.error(f"Zarr write failed: {str(e)}")
            raise
        finally:
            # Restore original logging level when done
            shuffle_logger.setLevel(original_level)
    else:
        # Compute locally if no client
        write_task.compute()
    
    logger.info(f"Conversion to Zarr complete: {out_zarr}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info(f"Conversion completed in {int(hours):02}:{int(minutes):02}:{int(seconds):02} (hh:mm:ss).")

    return out_zarr

