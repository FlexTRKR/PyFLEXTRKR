"""
Create terrain and range masks for a radar grid.

Original code provided by Ye Liu (ye.liu@pnnl.gov)
"""
import xesmf as xe
import xarray as xr
import numpy as np

def create_mask(ds, grid):
    """
    Make range masks and add to dataset

    Args:
        ds: Xarray Dataset
            Target Dataset
        grid: Xarray Dataset
            Dataset containing the lat/lon grid

    Return:
        ds: Xarray Dataset
            Target Dataset with added range mask variables
    """
    # Specify range radii values
    range_radii = [110, 100, 50, 30, 20, 10, 5]

    # Calculate distance of all grids from radar
    dist = np.sqrt(grid['x']**2 + grid['y']**2)

    # Loop over each range radii to make a mask
    for rad in range_radii:
        xmask = xr.where(dist<=rad, True, False)
        xmask = xmask.assign_attrs({
            'long_name': f'Range mask for radius {rad} km',
            'units': 'unitless',
        })
        xmask.astype(np.int8)
        ds[f'mask{rad}'] = xmask
    return ds

def mainfunc():

    # Source terrain file (e.g., Global Land One-kilometer Base Elevation (GLOBE))
    # https://catalog.data.gov/dataset/global-land-one-kilometer-base-elevation-globe-v-1
    source = '/global/cfs/cdirs/m1867/zfeng/globe_topography/ETOPO1_Ice_g_gmt4.nc'
    # Input radar data containing lat/lon grid
    radar_file = '/global/cscratch1/sd/feng045/pyflextrkr_test/cell_radar/nexrad/input/KHGX20140807.120057.nc'
    # Output terrain rangemask filename
    out_filename = '/global/cscratch1/sd/feng045/pyflextrkr_test/cell_radar/nexrad/Terrain_RangeMask.nc'
    
    # Read sample radar data (gridded by PyART)
    # Drop dimensions ['time', 'nradar'], select lowest z level
    grid = xr.open_dataset(radar_file).drop_dims(['time','nradar']).isel(z=0)
    grid = grid[['point_latitude','point_longitude']].rename(
        {'point_latitude':'latitude', 'point_longitude':'longitude'}
    )
    grid['x'] = grid['x']/1000.
    grid['y'] = grid['y']/1000.
    grid['latitude']  = grid['latitude'].astype(np.float32)
    grid['longitude'] = grid['longitude'].astype(np.float32)
    grid['latitude'] = grid['latitude'].assign_attrs({
        'long_name': 'Cartesian grid of latitude', 
    })
    grid['longitude'] = grid['longitude'].assign_attrs({
        'long_name': 'Cartesian grid of longitude', 
    })
    # Calculate mean grid spacing
    dx = grid['x'].diff(dim='x').mean().item()
    dy = grid['y'].diff(dim='y').mean().item()

    # Get radar data lat/lon boundary
    lat_min, lat_max = grid['latitude'].min().values, grid['latitude'].max().values
    lon_min, lon_max = grid['longitude'].min().values, grid['longitude'].max().values

    # Read input terrain file
    hgt = xr.open_dataset(source)['z'].squeeze()
    # Subset terrain 
    buffer = 1.0
    hgt = hgt.sel(y=slice(lat_min-buffer, lat_max+buffer), x=slice(lon_min-buffer, lon_max+buffer))
    
    # Regrid terrain to radar grid
    regridder = xe.Regridder(hgt, grid, "bilinear")
    hgt_out = regridder(hgt)
    hgt_out.name = 'hgt'
    hgt_out = hgt_out.astype(np.float32)
    hgt_out = hgt_out.assign_attrs({
        'long_name': 'Surface Elevation',
        'units': 'm',
    })

    # Make output dataset
    ds_out = grid.copy()
    # Replace hgt_out coordinates with the input grid coordinates
    ds_out['hgt'] = hgt_out.drop_vars(['longitude','latitude']).assign_coords({'y':grid['y'], 'x':grid['x']})

    # Create range masks
    ds_out = create_mask(ds_out, grid)

    # Assign global attributes
    ds_out.attrs = []
    ds_out = ds_out.assign_attrs({
        'dx': dx,
        'dy': dy,
        'input_file': source,
        'created_by': 'make_terrain_rangemask.py',
    })
    ds_out['x'] = ds_out['x'].assign_attrs({
        'long_name': 'X distance on the projection plane from the origin',
        'units': 'km'
    })
    ds_out['y'] = ds_out['y'].assign_attrs({
        'long_name': 'Y distance on the projection plane from the origin',
        'units': 'km'
    })
    # Drop z coordinate
    ds_out = ds_out.drop_vars('z')

    # Write to output
    ds_out.to_netcdf(out_filename)
    print(f'Output saved: {out_filename}')
    

if __name__ == '__main__':
    mainfunc()
