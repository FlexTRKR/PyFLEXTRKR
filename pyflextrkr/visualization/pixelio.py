"""
Reading and subsetting a pixel-level tracking file for plotting: the map extent of the data, the index window around a
map extent (dateline-safe, works for any 2D lon/lat grid without creating NaNs), the time strings and the figure name.

Called by: mcs_map, cell_map, generic_map, cartesian_map (work_for_time_loop). Calls: numpy, pandas, xarray.
"""
import numpy as np
import pandas as pd

#-----------------------------------------------------------------------
def get_data_extent(ds, lon_name='longitude', lat_name='latitude'):
    """Map extent [lonmin, lonmax, latmin, latmax] of the data of an open pixel-level file."""
    lonmin = ds[lon_name].min().item()
    lonmax = ds[lon_name].max().item()
    latmin = ds[lat_name].min().item()
    latmax = ds[lat_name].max().item()
    return [lonmin, lonmax, latmin, latmax]

#-----------------------------------------------------------------------
def find_subset_window(ds, ref_varname, map_extent, lon_name='longitude', lat_name='latitude', cartesian=False):
    """
    Index window of the pixels inside a map extent, with a margin of one pixel to capture the edge points.

    Use index slicing to subset - works for any 2D lat/lon grid without creating NaNs. Longitudes of the data and of the
    extent are normalized to 0-360, so the data in either 0-360 or -180~180 and extents that cross the dateline
    (e.g. lonmin=160, lonmax=-160) work.

    Args:
        ds: xarray Dataset, an open pixel-level file
        ref_varname: string, a variable of the file with the (time,) y, x dimensions
        map_extent: list [lonmin, lonmax, latmin, latmax]
        cartesian: bool, the coordinates are x/y of a domain without a map (no longitude normalization)

    Returns:
        window: dict {dim_y: slice, dim_x: slice} for DataArray.isel, or None if the dimensions cannot be determined,
        no pixel is inside the extent, or the window is the whole grid (the caller then uses the whole domain).
    """
    lonmin, lonmax = map_extent[0], map_extent[1]
    latmin, latmax = map_extent[2], map_extent[3]

    lon_vals = ds[lon_name].values
    lat_vals = ds[lat_name].values

    # Normalize lon_vals and extent to 0-360 for consistent comparison.
    # Handles data in either 0-360 or -180~180, and extents crossing
    # the dateline (e.g. lonmin=160, lonmax=-160 -> lonmax_norm=200).
    if cartesian:
        lon_vals_norm, lonmin_norm, lonmax_norm = lon_vals, lonmin, lonmax
    else:
        lon_vals_norm = lon_vals % 360
        lonmin_norm = lonmin % 360
        lonmax_norm = lonmax % 360
    lat_mask = (lat_vals >= latmin) & (lat_vals <= latmax)
    if (not cartesian) and (lonmax - lonmin) >= 360. - 1e-6:
        # all longitudes (0 360, -180 180): the normalized ends are the same meridian, so the box would be one column
        lon_mask = np.ones_like(lat_mask)
    elif lonmin_norm <= lonmax_norm:
        lon_mask = (lon_vals_norm >= lonmin_norm) & (lon_vals_norm <= lonmax_norm)
    else:  # extent wraps around 360 deg (e.g. 350 to 30)
        lon_mask = (lon_vals_norm >= lonmin_norm) | (lon_vals_norm <= lonmax_norm)
    # Find all points within the domain
    mask = lon_mask & lat_mask

    # Find bounding box in index space
    rows, cols = np.where(mask)

    if len(rows) == 0:
        print("Warning: No data found in specified domain. Using full dataset.")
        return None

    # Get the min/max indices, with small buffer to capture edge points
    row_min = max(0, rows.min() - 1)
    row_max = min(lon_vals.shape[0], rows.max() + 2)
    col_min = max(0, cols.min() - 1)
    col_max = min(lon_vals.shape[1], cols.max() + 2)

    # The window is the whole grid (e.g. --subset 1 with the extent of a global dataset): no subset, so the whole domain is used
    if row_min == 0 and col_min == 0 and row_max >= lon_vals.shape[0] and col_max >= lon_vals.shape[1]:
        return None

    # Get dimension names (e.g., y/x, south_north/west_east, lat/lon, etc.)
    spatial_dims = [d for d in ds[ref_varname].dims if d not in ['time', 'times']]
    if len(spatial_dims) < 2:
        print("Warning: Could not determine spatial dimensions. Using full dataset.")
        return None
    dim_y, dim_x = spatial_dims[-2], spatial_dims[-1]

    print(f"Subset domain: rows [{row_min}:{row_max}], cols [{col_min}:{col_max}]")
    return {dim_y: slice(row_min, row_max), dim_x: slice(col_min, col_max)}

#-----------------------------------------------------------------------
def make_figure_name(ds, figdir, figbasename, figname_type='date_time', ifile=0, title_prefix=""):
    """
    Figure name, time string and figure title of the time of an open pixel-level file.

    Args:
        ds: xarray Dataset, an open pixel-level file (one time)
        figdir: string, output directory (with a trailing '/')
        figbasename: string, figure base name
        figname_type: string, 'date_time' (figbasenameYYYYmmdd_HHMMSS.png) or 'sequence' (figbasename00001.png)
        ifile: int, index of the file in the series (for 'sequence')
        title_prefix: string, prefix of the title, '' for none

    Returns:
        figname, timestr, suptitle
    """
    fdatetime = pd.to_datetime(ds['time'].data.item()).strftime('%Y%m%d_%H%M%S')
    timestr = pd.to_datetime(ds['time'].data.item()).strftime('%Y-%m-%d %H:%M:%S UTC')
    if title_prefix != "":
        suptitle = f"{title_prefix} | {timestr}"
    else:
        suptitle = f"{timestr}"
    if figname_type == 'date_time':
        figname = f'{figdir}{figbasename}{fdatetime}.png'
    elif figname_type == 'sequence':
        figname = f'{figdir}{figbasename}{"{:05d}".format(ifile+1)}.png'
    return figname, timestr, suptitle
