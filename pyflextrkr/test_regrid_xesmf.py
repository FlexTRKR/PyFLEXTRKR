import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr 
import xesmf as xe

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


if __name__ == '__main__':

    ds_dst = xr.open_dataset("/pscratch/sd/f/feng045/iclass/goamazon/HVMIXING/gpm/merg_2014050100_4km-pixel.nc")
    lat_dst = ds_dst['lat'].squeeze().data
    lon_dst = ds_dst['lon'].squeeze().data

    ds_src = xr.open_dataset('/global/cfs/projectdirs/encon/smhagos/HVMIXING/CTL4KMRUN01/RAW/wrfout_d01_2014-04-04_08:00:00', engine='netcdf4')
    lat_src = ds_src['XLAT'].squeeze().data
    lon_src = ds_src['XLONG'].squeeze().data

    weight_filename = f'/pscratch/sd/f/feng045/iclass/goamazon/HVMIXING/map_data/weight_conservative_wrf2gpm.nc'
    weightfile_exist = os.path.isfile(weight_filename)

    # Make 2D lat/lon bounds for source grid
    lat_b_src, lon_b_src = get_latlon_bounds_2d(lat_src, lon_src)
    # Make 1D lat/lon bounds for destination grid
    lat_b_dst, lon_b_dst = get_latlon_bounds_1d(lat_dst, lon_dst)
    # import pdb; pdb.set_trace()

    # Put grid variables in dictionaries
    grid_in = {
        'lat': lat_src, 
        'lon': lon_src,
        'lat_b': lat_b_src,
        'lon_b': lon_b_src,
    }
    grid_out = {
        'lat': lat_dst, 
        'lon': lon_dst,
        'lat_b': lat_b_dst,
        'lon_b': lon_b_dst,
    }

    # ds_src = ds_src.rename_dims({'west_east':'lon', 'south_north':'lat'})

    # Check weight file 
    if weightfile_exist == False:
        print(f'Weight file does not exist, building Regridder ...')
        # Build Regridder
        regridder_conserve = xe.Regridder(grid_in, grid_out, method='conservative')
        # Write Regridder to a netCDF file
        regridder_conserve.to_netcdf(weight_filename)
        print(f'Weight file saved: {weight_filename}')
    else:
        print(f'Weight file exists: {weight_filename}')
        # Retrieve Regridder
        regridder_conserve = xe.Regridder(grid_in, grid_out, method='conservative', weights=weight_filename)

    # OLR = ds_src['OLR']
    # lon_reg = regridder_conserve(ds_src['XLONG'])
    # lat_reg = regridder_conserve(ds_src['XLAT'])
    # OLR_reg = regridder_conserve(ds_src['OLR'])
    # RAINNC_reg = regridder_conserve(ds_src['RAINNC'])

    # import pdb; pdb.set_trace()