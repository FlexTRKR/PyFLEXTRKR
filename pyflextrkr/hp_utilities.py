
import xarray as xr
import pandas as pd
import healpix as hp

def remap_healpix_to_latlon_grid(ds_hp, latlon_filename, config):
    """
    Remap HEALPix data to a specified lat-lon grid.
    
    Parameters
    ----------
    ds_hp : xarray.Dataset
        Input Dataset in HEALPix format.
    latlon_filename : str
        Path to the netCDF file containing the target lat-lon grid.
    config : dict
        Configuration dictionary containing grid variable names.
    
    Returns
    -------
    ds_out : xarray.Dataset
        Remapped Dataset on the lat-lon grid.
    """

    x_dimname = config.get("latlon_x_dimname", "x")
    y_dimname = config.get("latlon_y_dimname", "y")
    x_coordname = config.get("latlon_x_coordname", "longitude")
    y_coordname = config.get("latlon_y_coordname", "latitude")
    time_dimname = config.get("time_dimname", "time")
    time_coordname = config.get("time_coordname", "time")

    time_coord = ds_hp[time_coordname]
    time_pd = pd.to_datetime(time_coord.dt.strftime("%Y-%m-%dT%H:%M:%S").item())

    # Read lat-lon grid file
    ds_grid = xr.open_dataset(latlon_filename, decode_timedelta=False)

    # Define the list of coordinates you want to keep
    coords_to_keep = [y_coordname, x_coordname]
    # Identify coordinates to drop (all current coords minus the ones to keep)
    # We convert to sets for efficient difference operation
    all_coords = set(ds_grid.coords.keys())
    coords_to_drop = list(all_coords - set(coords_to_keep))

    # Drop the unwanted coordinates
    ds_grid = ds_grid.drop_vars(coords_to_drop)
    # Get lat & lon grids
    lon_grid = ds_grid[x_coordname]
    lat_grid = ds_grid[y_coordname]

    # Find the HEALPix pixels that are closest to the target grid points
    # Since lat_grid and lon_grid are already 2D, pass them directly as separate arguments
    pix = xr.DataArray(
        hp.ang2pix(ds_hp.crs.healpix_nside, lon_grid, lat_grid, nest=True, lonlat=True),
        coords={y_dimname: lat_grid[y_dimname], x_dimname: lon_grid[x_dimname]},
        dims=(y_dimname, x_dimname),
    )
    
    # Remap DataSet to lat/lon grid
    ds_out = ds_hp.isel(cell=pix).expand_dims({time_dimname:[time_pd]}).compute()

    # Add lat-lon grid as data variables to the output Dataset
    # Create plain DataArrays without coordinate metadata to avoid merge conflicts
    ds_out[x_coordname] = xr.DataArray(lon_grid.values, dims=(y_dimname, x_dimname))
    ds_out[y_coordname] = xr.DataArray(lat_grid.values, dims=(y_dimname, x_dimname))

    return ds_out