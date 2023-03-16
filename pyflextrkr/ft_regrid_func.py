import os
import logging
import numpy as np
import xarray as xr
import xesmf as xe

#-------------------------------------------------------------------------------------
def make_grid4regridder(gridfile_src, config):
    """
    Make source and destination grid data for xESMF regridder.

    Args:
        gridfile_src: string
            Filename for the source grid.
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        grid_src: dictionary
            Dictionary containing source grid data.
        grid_dst: dictionary
            Dictionary containing destination grid data.
    """
    # Get regrid option variables
    gridfile_dst = config.get('gridfile_dst')
    x_coordname_dst = config.get('x_coordname_dst')
    y_coordname_dst = config.get('y_coordname_dst')
    x_coordname_src = config.get('x_coordname_src')
    y_coordname_src = config.get('y_coordname_src')

    # Read source grid file
    ds_src = xr.open_dataset(gridfile_src, engine='netcdf4')
    lon_src = ds_src[x_coordname_src].squeeze().data
    lat_src = ds_src[y_coordname_src].squeeze().data

    # Read destination grid file
    ds_dst = xr.open_dataset(gridfile_dst, engine='netcdf4')
    lon_dst = ds_dst[x_coordname_dst].squeeze().data
    lat_dst = ds_dst[y_coordname_dst].squeeze().data

    # Make lat/lon bounds for source grid
    if (lat_src.ndim == 1) & (lon_src.ndim == 1):
        lat_b_src, lon_b_src = get_latlon_bounds_1d(lat_src, lon_src)
    elif (lat_src.ndim == 2) & (lon_src.ndim == 2):
        lat_b_src, lon_b_src = get_latlon_bounds_2d(lat_src, lon_src)

    # Make lat/lon bounds for destination grid
    if (lat_dst.ndim == 1) & (lon_dst.ndim == 1):
        lat_b_dst, lon_b_dst = get_latlon_bounds_1d(lat_dst, lon_dst)
    elif (lat_dst.ndim == 2) & (lon_dst.ndim == 2):
        lat_b_dst, lon_b_dst = get_latlon_bounds_2d(lat_dst, lon_dst)

    # Put grid variables in dictionaries
    grid_src = {
        'lat': lat_src, 
        'lon': lon_src,
        'lat_b': lat_b_src,
        'lon_b': lon_b_src,
    }
    grid_dst = {
        'lat': lat_dst, 
        'lon': lon_dst,
        'lat_b': lat_b_dst,
        'lon_b': lon_b_dst,
    }
    return grid_src, grid_dst

#-------------------------------------------------------------------------------------
def make_weight_file(gridfile_src, config):
    """
    Make xESMF Regridder weight file.

    Args:
        gridfile_src: string
            Filename for the source grid.
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        weight_filename: string
            Filename for the regrid weights.
    """
    logger = logging.getLogger(__name__)

    gridfile_dst = config.get('gridfile_dst')
    weight_filename = config.get('weight_filename')
    regrid_method = config.get('regrid_method', 'conservative')

    # Make source & destination grid data for regridder
    grid_src, grid_dst = make_grid4regridder(gridfile_src, config)

    # Check weight file
    weightfile_exist = os.path.isfile(weight_filename)
    if weightfile_exist == False:
        logger.info(f'Regridding weight file does not exist, building Regridder ...')
        logger.info(f'Source grid file: {gridfile_src}')
        logger.info(f'Destination grid file: {gridfile_dst}')
        # Build Regridder
        regridder = xe.Regridder(grid_src, grid_dst, method=regrid_method)
        # Write Regridder to a netCDF file
        regridder.to_netcdf(weight_filename)
        logger.info(f'Weight file saved: {weight_filename}')
    else:
        logger.info(f'Weight file exists: {weight_filename}')

    return weight_filename

#-------------------------------------------------------------------------------------
def get_latlon_bounds_2d(latin, lonin):
    """
    Calculate bounds of 2D lat/lon grid

    Args:
        latin: np.array
            Latitude center values, shape: (ny, nx)
        lonin: np.array
            Longitude center values, shape: (ny, nx)
    Returns:
        lat_b: np.array
            Latitude bound values, shape: (ny+1, nx+1)
        lon_b:np.array
            Longitude bound values, shape: (ny+1, nx+1)
    """
    # Calculate half grid size
    dlat = (latin[1:,:] - latin[:-1,:]) / 2
    # Take the last row
    dlat_row = dlat[-1,:]
    # Append last row to dlat, so it is the same shape as lat
    dlat_match = np.vstack([dlat, dlat_row])
    # Subtract dlat from lat
    lat_b = latin - dlat_match
    # Add dlat from the last row to lat to get the ny+1 bound
    lat_row = latin[-1,:] + dlat_row
    # Append to last row
    lat_b = np.vstack([lat_b, lat_row])
    # Append last column of lat_b to get the nx+1
    lat_col = lat_b[:,-1]
    lat_b = np.column_stack([lat_b, lat_col])

    # Calculate half grid size
    dlon = (lonin[:,1:] - lonin[:,:-1]) / 2
    # Take the last column
    dlon_col = dlon[:,-1]
    # Append last column to dlon, so it is the same shape as lon
    dlon_match = np.column_stack([dlon , dlon_col])
    # Subtract dlon from lon
    lon_b = lonin - dlon_match
    # Add dlon from the last column to lon to get the nx+1 bound
    lon_col = lonin[:,-1] + dlon_col
    # Append to last column
    lon_b = np.column_stack([lon_b, lon_col])
    # Append last row of lon_b to get the ny+1
    lon_row = lon_b[-1,:]
    lon_b = np.vstack([lon_b, lon_row])

    return (lat_b, lon_b)

#-------------------------------------------------------------------------------------
def get_latlon_bounds_1d(latin, lonin):
    """
    Calculate bounds of 1D lat/lon grid

    Args:
        latin: np.array
            Latitude center values, shape: (ny)
        lonin: np.array
            Longitude center values, shape: (nx)
    Returns:
        lat_b: np.array
            Latitude bound values, shape: (ny+1)
        lon_b:np.array
            Longitude bound values, shape: (nx+1)
    """
    # Calculate half grid size
    dlat = (latin[1:] - latin[:-1]) / 2
    # Take the last row
    dlat_row = dlat[-1]
    # Append last row to dlat, so it is the same shape as lat
    dlat_match = np.append(dlat, dlat_row)
    # Subtract dlat from lat
    lat_b = latin - dlat_match
    # Add dlat from the last row to lat to get the ny+1 bound
    lat_row = latin[-1] + dlat_row
    # Append to last row
    lat_b = np.append(lat_b, lat_row)

    # Calculate half grid size
    dlon = (lonin[1:] - lonin[:-1]) / 2
    # Take the last column
    dlon_col = dlon[-1]
    # Append last column to dlon, so it is the same shape as lon
    dlon_match = np.append(dlon , dlon_col)
    # Subtract dlon from lon
    lon_b = lonin - dlon_match
    # Add dlon from the last column to lon to get the nx+1 bound
    lon_col = lonin[-1] + dlon_col
    # Append to last column
    lon_b = np.append(lon_b, lon_col)

    return (lat_b, lon_b)