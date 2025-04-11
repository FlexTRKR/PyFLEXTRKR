import numpy as np
import xarray as xr
import os
import time
import logging
import dask.array as da
import healpix as hp
from pyflextrkr.ft_utilities import setup_logging

def prepare_grid_for_analysis(ds, var_name='mcs_mask', lon_dim='lon', lat_dim='lat',
                             target_lat_range=None, lat_spacing=None, fill_value=0):
    """
    Prepares a lat-lon grid dataset for analysis by:
    1. Converting longitude from -180/+180 to 0-360 range and rolling to start at 0
    2. Ensuring latitude is in descending order (+90 to -90)
    3. Expanding latitude range to cover as much of the globe as possible without exceeding ±90°
    
    All operations maintain lazy evaluation for Dask compatibility.
    
    Args:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        Input dataset with latitude and longitude dimensions
    var_name : str, optional
        Name of the variable to process (only used if ds is a Dataset), by default 'mcs_mask'
    lon_dim : str, optional
        Name of longitude dimension, by default 'lon'
    lat_dim : str, optional
        Name of latitude dimension, by default 'lat'
    target_lat_range : tuple, optional
        Target (min, max) latitude range. If None, will be inferred from data.
    lat_spacing : float, optional
        Latitude spacing in degrees. If None, will be inferred from data.
    fill_value : int or float, optional
        Value to fill expanded regions with, by default 0
        
    Returns:
    --------
    xarray.Dataset or xarray.DataArray
        Dataset with longitude in 0-360 range, rolled to start at 0,
        latitude in descending order (+90 to -90), and expanded to target range
    """   
    # Determine if input is DataArray or Dataset
    is_dataarray = isinstance(ds, xr.DataArray)
    
    # Step 1: Handle longitude conversion and rolling
    # Find where longitude crosses from negative to positive
    lon_0_index = (ds[lon_dim] < 0).sum().item()
    
    # Create indexers for the roll
    lon_indices = np.roll(np.arange(ds.sizes[lon_dim]), -lon_0_index)
    
    # Roll dataset and convert longitudes to 0-360 range
    rolled_ds = ds.isel({lon_dim: lon_indices})
    new_lons = xr.where(rolled_ds[lon_dim] < 0, rolled_ds[lon_dim] + 360, rolled_ds[lon_dim])
    rolled_ds = rolled_ds.assign_coords({lon_dim: new_lons})
    
    # Step 2: Handle latitude order and determine expansion needs
    lat_values = rolled_ds[lat_dim].values
    is_descending = np.all(np.diff(lat_values) < 0)
    
    # Flip latitudes if not already in descending order
    if not is_descending:
        lat_indices = np.arange(rolled_ds.sizes[lat_dim] - 1, -1, -1)
        rolled_ds = rolled_ds.isel({lat_dim: lat_indices})
        lat_values = rolled_ds[lat_dim].values
    
    # Infer lat_spacing if not provided
    if lat_spacing is None:
        # Calculate spacing from the first two points
        # Use absolute value to handle both ascending and descending orders
        lat_spacing = abs(lat_values[1] - lat_values[0])
    
    # Infer target_lat_range if not provided
    if target_lat_range is None:
        # Find the highest possible latitude value without exceeding ±90°
        # For a grid with 0.1° spacing, this would be ±89.95°
        max_possible_lat = 90.0 - (lat_spacing / 2)
        min_possible_lat = -max_possible_lat
        target_lat_range = (min_possible_lat, max_possible_lat)
    
    # Get data array and current latitude range
    if is_dataarray:
        data_array = rolled_ds
    else:
        data_array = rolled_ds[var_name]
    
    current_min_lat, current_max_lat = lat_values.min(), lat_values.max()
    target_min, target_max = sorted(target_lat_range)
    
    # Calculate expansion points needed
    south_points = int(round((current_min_lat - target_min) / lat_spacing))
    north_points = int(round((target_max - current_max_lat) / lat_spacing))

    # Step 3: If no expansion needed, return the rolled dataset
    if south_points <= 0 and north_points <= 0:
        return rolled_ds
    
    # Step 4: Create expansion arrays
    dims = data_array.dims
    lat_idx = data_array.dims.index(lat_dim)
    
    # Create new latitude coordinates
    if north_points > 0:
        # Create new northern latitudes from target max to highest current
        # We need to maintain descending order (higher to lower latitude)
        new_north_lats = np.linspace(target_max, current_max_lat + lat_spacing, north_points)
        north_shape = list(data_array.shape)
        north_shape[lat_idx] = north_points
        north_data = da.zeros(north_shape, chunks="auto", dtype=data_array.dtype)
        north_coords = {dim: data_array[dim] for dim in dims if dim != lat_dim}
        north_coords[lat_dim] = new_north_lats
        north_da = xr.DataArray(north_data, dims=dims, coords=north_coords)
    else:
        north_da = None

    if south_points > 0:
        # Create new southern latitudes from lowest current to target min
        # Also in descending order
        new_south_lats = np.linspace(current_min_lat - lat_spacing, target_min, south_points)
        south_shape = list(data_array.shape)
        south_shape[lat_idx] = south_points
        south_data = da.zeros(south_shape, chunks="auto", dtype=data_array.dtype)
        south_coords = {dim: data_array[dim] for dim in dims if dim != lat_dim}
        south_coords[lat_dim] = new_south_lats
        south_da = xr.DataArray(south_data, dims=dims, coords=south_coords)
    else:
        south_da = None

    # Step 5: Concatenate along latitude dimension
    arrays_to_concat = []
    if north_da is not None:
        arrays_to_concat.append(north_da)
    arrays_to_concat.append(data_array)
    if south_da is not None:
        arrays_to_concat.append(south_da)
    
    expanded_da = xr.concat(arrays_to_concat, dim=lat_dim)

    # Step 6: Return result based on input type
    if not is_dataarray:
        # Create a new dataset with the expanded variable and coordinates
        # Start with an empty dataset with the correct coordinates
        coords = {dim: data_array[dim] for dim in dims if dim != lat_dim}
        coords[lat_dim] = expanded_da[lat_dim]  # Use the new expanded latitude
        
        # Create new dataset with the proper dimensions
        expanded_ds = xr.Dataset(coords=coords)
        
        # Copy all variables from the original dataset except the one we're expanding
        for var in rolled_ds.data_vars:
            if var != var_name:
                # For variables that have lat dimension, we need to extend them
                if lat_dim in rolled_ds[var].dims:
                    # Create an expanded array filled with zeros/NaN for this variable
                    var_shape = list(rolled_ds[var].shape)
                    var_lat_idx = rolled_ds[var].dims.index(lat_dim)
                    var_shape[var_lat_idx] = len(expanded_da[lat_dim])
                    fill = np.nan if np.issubdtype(rolled_ds[var].dtype, np.floating) else 0
                    expanded_var = da.zeros(var_shape, chunks="auto", dtype=rolled_ds[var].dtype)
                    if fill != 0:
                        expanded_var = expanded_var * np.nan
                    
                    # Insert the original data in the proper position
                    expanded_var_da = xr.DataArray(expanded_var, dims=rolled_ds[var].dims, coords=coords)
                    
                    # We'll add this variable but handle it separately
                    expanded_ds[var] = expanded_var_da
                else:
                    # Variables without lat dimension can be copied directly
                    expanded_ds[var] = rolled_ds[var]
        
        # Finally add our main expanded variable
        expanded_ds[var_name] = expanded_da
        
        return expanded_ds
    else:
        return expanded_da
    

def remap_mask_to_healpix(config):
    """
    Remap mask data from lat/lon grid to HEALPix grid.

    Args:
        config: dictionary
                Dictionary containing config parameters

    Returns:
        out_zarr: string
            Zarr store path.
    """
    # Set the logging message level
    setup_logging()
    logger = logging.getLogger(__name__)

    start_time = time.time()
    logger.info(f"Remap to HEALPix started ...")

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

    # Get config parameters
    pixeltracking_outpath = config.get("pixeltracking_outpath")
    clouddata_path = config.get("clouddata_path")
    databasename = config.get("databasename")
    hp_zoomlev = config.get("hp_zoomlev")
    startdate = config.get("startdate")
    enddate = config.get("enddate")
    outpath = os.path.dirname(os.path.normpath(pixeltracking_outpath)) + "/"
    # Get preset-specific configuration
    presets = config.get("zarr_output_presets", {})
    in_mask_filebase = presets.get("mask", {}).get("out_filebase", "mcs_mask_latlon_")
    # Input mask Zarr store
    in_mask_dir = f"{outpath}{in_mask_filebase}{startdate}_{enddate}.zarr"
    # Input HEALPix Zarr store
    in_hp_dir = f"{clouddata_path}{databasename}.zarr"

    # Build output filename
    out_mask_filebase = presets.get("healpix", {}).get("out_filebase", "mcs_mask_hp")
    out_zarr = f"{outpath}{out_mask_filebase}{hp_zoomlev}_{startdate}_{enddate}.zarr"

    # Check if output exists and should be overwritten
    overwrite = config.get("overwrite_zarr", True)
    if os.path.exists(out_zarr) and not overwrite:
        logger.warning(f"Zarr store already exists at {out_zarr} and overwrite=False. Skipping.")
        return out_zarr

    # Check intput mask file
    if os.path.exists(in_mask_dir) is False:
        logger.error(f"Input mask file {in_mask_dir} does not exist. Skipping remap.")
        return out_zarr
    if os.path.exists(in_hp_dir) is False:
        logger.error(f"Input HEALPix file {in_hp_dir} does not exist. Skipping remap.")
        return out_zarr
    
    # Get chunk size from config
    chunksize_time = config.get("chunksize_time", "auto")
    chunksize_cell = config.get("chunksize_cell", "auto")
    
    # Read mask data
    mask_chunks = {"time": min(100, chunksize_time), "lat": "auto", "lon": "auto"}
    ds_mask = xr.open_dataset(in_mask_dir, engine='zarr', chunks=mask_chunks)

    # Modify mask grid for remapping
    ds_mask = prepare_grid_for_analysis(ds_mask)

    # Read HEALPix zarr file
    ds_hp = xr.open_dataset(in_hp_dir)

    # Make remap lat/lon for HEALPix
    remap_lons, remap_lats = hp.pix2ang(
        ds_hp.crs.healpix_nside, ds_hp.cell, nest=ds_hp.crs.healpix_order, lonlat=True,
    )
    remap_lons = remap_lons % 360
    # Convert to DataArray
    lon_hp = xr.DataArray(remap_lons, dims="cell", coords={"cell":ds_hp.cell})
    lat_hp = xr.DataArray(remap_lats, dims="cell", coords={"cell":ds_hp.cell})

    # Remap DataSet to HEALPix
    dsout_hp = ds_mask.sel(
        lon=lon_hp, lat=lat_hp, method="nearest",
    )

    # Set proper chunking
    chunked_ds = dsout_hp.chunk({
        "time": chunksize_time, 
        "cell": chunksize_cell, 
    })
    # Report dataset size and chunking info
    logger.info(f"Dataset dimensions: {dict(chunked_ds.sizes)}")
    logger.info(f"Chunking scheme: time={chunksize_time}, cell={chunksize_cell}")

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
        import psutil
        
        # Get cluster state information before processing
        memory_usage = client.run(lambda: psutil.Process().memory_info().rss / 1e9)
        logger.info(f"Current memory usage across workers (GB): {memory_usage}")
        
        # Compute with progress tracking
        future = client.compute(write_task)
        logger.info("Writing to Zarr (this may take a while)...")
        progress(future)  # Shows a progress bar in notebooks or detailed progress in terminals
        
        try:
            result = future.result()
            logger.info("Zarr write completed successfully")
        except Exception as e:
            logger.error(f"Zarr write failed: {str(e)}")
            raise
    else:
        # Compute locally if no client
        write_task.compute()
    
    logger.info(f"Remap complete: {out_zarr}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info(f"Remap completed in {int(hours):02}:{int(minutes):02}:{int(seconds):02} (hh:mm:ss).")

    return out_zarr