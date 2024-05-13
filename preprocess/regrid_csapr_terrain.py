"""
Regrid (coarsen) CSAPR2 terrain range mask file for tracking cells at coarser resolution.
"""
import time
import numpy as np
from scipy import ndimage
import xarray as xr

def convolve_var(in_var, kernel):
    """
    Apply convolution to a variable within a moving kernel.

    Args:
        in_var: np.array
            Input variable array, can be either 2D or 3D.
        kernel: np.array
            Kernel for weights.
    
    Returns:
        out_var: np.array
            Output variable array.
    """
    # Make an array for counting number of grids for convolution
    mask_goodvalues = (~np.isnan(in_var)).astype(float)

    # Apply convolution filter
    bkg_var = ndimage.convolve(in_var, kernel, mode='constant', cval=0.0)
    numPixs = ndimage.convolve(mask_goodvalues, kernel, mode='constant', cval=0.0)
    # Mask missing data area
    bkg_var[mask_goodvalues==0] = 0
    numPixs[mask_goodvalues==0] = 0

    # Calculate average linear reflectivity and convert to log values
    out_var = np.full(in_var.shape, np.NaN, dtype=np.float32)
    out_var[numPixs>0] = bkg_var[numPixs>0] / numPixs[numPixs>0]

    # Remove pixels with 0 number of pixels
    out_var[mask_goodvalues==0] = np.NaN

    return out_var

if __name__ == '__main__':

    # data_dir = '/gpfs/wolf/atm131/proj-shared/zfeng/cacti/csapr/corgridded_terrain.c0/'
    data_dir = '/global/cfs/cdirs/m1657/zfeng/cacti/arm/csapr/corgridded_terrain.c0/'
    in_file = f'{data_dir}CSAPR2_Taranis_Gridded_500m.Terrain_RangeMask.nc'
    out_file = f'{data_dir}CSAPR2_Taranis_Gridded_4000m.Terrain_RangeMask.nc'

    # Grid spacing in [m]
    dx = 4000
    dy = 4000
    # Regrid ratio (integer)
    regrid_ratio = 8

    # Read input terrain file
    ds = xr.open_dataset(in_file)
    longitude = ds['longitude']
    latitude = ds['latitude']
    hgt = ds['hgt']
    
    # Make a kernel for weights
    start_idx = int((regrid_ratio-1) / 2)
    kernel = np.zeros((regrid_ratio+1,regrid_ratio+1), dtype=int)
    kernel[1:regrid_ratio, 1:regrid_ratio] = 1

    # Subsample range masks
    mask_list = ['mask110', 'mask100', 'mask50', 'mask30', 'mask20', 'mask10']
    var_dict = {}
    for rr in mask_list:
        mask_data = ds[rr].data[start_idx::regrid_ratio,start_idx::regrid_ratio]
        var_dict[rr] = (['y', 'x'], mask_data, ds[rr].attrs)

    # Call convlution function
    hgt_conv = convolve_var(hgt.data, kernel)
    # Subsample every X grid points
    hgt_reg = hgt_conv[start_idx::regrid_ratio,start_idx::regrid_ratio]

    # Make output coordinate
    ny, nx = hgt_reg.shape
    xcoord = np.arange(-nx/2, nx/2, 1) * dx
    ycoord = np.arange(-ny/2, ny/2, 1) * dy
    xcoord_attrs = ds['x'].attrs
    ycoord_attrs = ds['y'].attrs

    # Add hgt to output variable dictionary
    var_dict['hgt'] = (['y', 'x'], hgt_reg, hgt.attrs)
    # Output coordinates
    coord_dict = {
        'y': (['y'], ycoord, ycoord_attrs),
        'x': (['x'], xcoord, xcoord_attrs),
    }
    # Output global attributes
    gattr_dict = {
        'dx': dx,
        'dy': dy,
        'source_file': in_file,
        'created_by': __file__,
        'Contact': 'Zhe Feng, zhe.feng@pnnl.gov',
        'Institution': 'Pacific Northwest National Laboratory',
        'Created_on': time.ctime(time.time()),
    }
    # Define xarray dataset
    ds_out = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in ds_out.data_vars}

    # Write to netcdf file
    ds_out.to_netcdf(
        path=out_file, mode='w', format='NETCDF4', encoding=encoding,
    )
    print(f'{out_file}')
