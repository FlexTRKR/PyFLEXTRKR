import glob
import os
import sys
import time
import logging
import numpy as np
from scipy import ndimage
import xarray as xr
import pandas as pd
import dask
from dask.distributed import wait

def create_semi_symmetric_array(size):
    """
    Make a semi-symmetric array around 0 increment by 1

    If the size is even, the array starts from -(size // 2) + 1 and goes up to start + size - 1 
    (e.g., for size = 10, the array would be [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5]). 
    If the size is odd, the array starts from -(size // 2) and goes up to start + size - 1 
    (e.g., for size = 9, the array would be [-4, -3, -2, -1, 0, 1, 2, 3, 4]).

    Args:
        size: int
            Size of the array.

    Returns:
        np.array
    """
    if size % 2 == 0:
        start = -(size // 2) + 1
    else:
        start = -(size // 2)
    array = np.arange(start, start + size)
    return array

def regrid_lasso_reflectivity(config):
    """
    Driver to coarsen LASSO 3D reflectivity for tracking.

    Args:
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        None.
    """

    logger = logging.getLogger(__name__)
    logger.info('Regridding LASSO reflectivity')

    in_dir = config['rawdata_path']
    in_basename = config['rawdatabasename']
    out_dir = config["clouddata_path"]
    out_basename = config['databasename']
    run_parallel = config['run_parallel']
    start_basetime = config["start_basetime"]
    end_basetime = config["end_basetime"]
    sample_time_freq = config['sample_time_freq']

    os.makedirs(out_dir, exist_ok=True)

    # Generate time marks within the start/end datetime
    start_datetime = pd.to_datetime(start_basetime, unit='s')
    end_datetime = pd.to_datetime(end_basetime, unit='s')
    datetimes = pd.date_range(start=start_datetime, end=end_datetime, freq=sample_time_freq)
    file_datetimes = datetimes.strftime('%Y%m%d.%H%M%S')

    #########################################################################################
    # Identify files to process
    in_files = []
    for tt in range(0, len(file_datetimes)):
        in_files.extend(sorted(glob.glob(f'{in_dir}{in_basename}{file_datetimes[tt]}.nc')))
    logger.info(f'Number of files to process: {len(in_files)}')

    results = []
    for ifile in in_files:
        # Serial
        if run_parallel == 0:
            result = regrid_file(ifile, in_basename, out_dir, out_basename, config)
        # Parallel
        elif run_parallel >= 1:
            result = dask.delayed(regrid_file)(ifile, in_basename, out_dir, out_basename, config)
            results.append(result)
        else:
            sys.exit('Valid parallelization flag not provided')

    if run_parallel >= 1:
        # Trigger dask computation
        final_result = dask.compute(*results)
        wait(final_result)

    logger.info('Done with regridding reflectivity files')
    return


def convolve_reflectivity(in_reflectivity, kernel):
    """
    Apply convolution to reflectivity within a moving kernel.
    This is equivalent to averaging reflectivity within a moving window.

    Args:
        in_reflectivity: np.array
            Input reflectivity array, can be either 2D or 3D.
        kernel: np.array
            Kernel for weights.
    
    Returns:
        out_reflectivity: np.array
            Output reflectivity array.
    """
    # Convert reflectivity to linear unit
    linrefl = 10. ** (in_reflectivity / 10.)
    # Make an array for counting number of grids for convolution
    mask_goodvalues = (~np.isnan(in_reflectivity)).astype(float)

    # Apply convolution filter
    bkg_linrefl = ndimage.convolve(linrefl, kernel, mode='constant', cval=0.0)
    numPixs = ndimage.convolve(mask_goodvalues, kernel, mode='constant', cval=0.0)
    # Mask missing data area
    bkg_linrefl[mask_goodvalues==0] = 0
    numPixs[mask_goodvalues==0] = 0

    # Calculate average linear reflectivity and convert to log values
    out_reflectivity = np.full(in_reflectivity.shape, np.nan, dtype=np.float32)
    out_reflectivity[numPixs>0] = 10.0 * np.log10(bkg_linrefl[numPixs>0] / numPixs[numPixs>0])

    # Remove pixels with 0 number of pixels
    out_reflectivity[mask_goodvalues==0] = np.nan
    
    return out_reflectivity


def regrid_file(in_filename, in_basename, out_dir, out_basename, config):
    """
    Regrid a file containing reflectivity data.

    Args:
        in_filename: string
            Input file name.
        out_dir: string
            Output directory name.
        out_basename: string
            Output file basename.
        config: dictionary
            Dictionary containing config parameters.
    
    Returns:
        out_filename: string
            Output file name.
    """
    logger = logging.getLogger(__name__)
    time_dimname = config.get('time_dimname', 'Time')
    reflectivity_varname = config.get('reflectivity_varname', 'REFL_10CM')
    x_varname = config.get('x_varname', 'XLONG')
    y_varname = config.get('y_varname', 'XLAT')
    z_varname = config.get('z_varname', 'HAMSL')
    x_dimname = config.get('x_dimname', 'west_east')
    y_dimname = config.get('y_dimname', 'south_north')
    z_dimname = config.get('z_dimname', 'HAMSL')
    dx = config['dx']
    dy = config['dy']
    regrid_ratio = config.get('regrid_ratio')

    # Read input data
    ds = xr.open_dataset(in_filename)
    in_time = ds[time_dimname]
    # Get variables
    REFL = ds[reflectivity_varname].squeeze()
    XLONG = ds[x_varname]
    XLAT = ds[y_varname]

    # Make a kernel for weights
    start_idx = int((regrid_ratio-1) / 2)
    kernel = np.zeros((regrid_ratio+1,regrid_ratio+1), dtype=int)
    kernel[1:regrid_ratio, 1:regrid_ratio] = 1

    # Make a 3D kernel
    kernel3d = kernel[None,:,:]
    # Call convlution function
    REFL_conv = convolve_reflectivity(REFL.data, kernel3d)

    # Subsample every X grid points
    REFL_reg = REFL_conv[:,start_idx::regrid_ratio,start_idx::regrid_ratio]
    XLONG_reg = XLONG.data[start_idx::regrid_ratio,start_idx::regrid_ratio]
    XLAT_reg = XLAT.data[start_idx::regrid_ratio,start_idx::regrid_ratio]

    # Make output filename
    nleadingchar = len(f'{in_basename}')
    fname = os.path.basename(in_filename)
    ftimestr = fname[nleadingchar:]
    out_filename = f'{out_dir}{out_basename}{ftimestr}'
    
    # Make output coordinate
    nz, ny, nx = REFL_reg.shape
    # Create a coordinate to mimic subsampling ratio of the full coordinate
    xcoord = create_semi_symmetric_array(nx) * dx
    ycoord = create_semi_symmetric_array(ny) * dy
    xcoord_attrs = {
        'long_name': 'X-coordinate grid index',
    }
    ycoord_attrs = {
        'long_name': 'Y-coordinate grid index',
    }

    # Define output variables
    # Array dimensions
    dim4d = [time_dimname, z_dimname, y_dimname, x_dimname]
    dim3d = [z_dimname, y_dimname, x_dimname]
    dim2d = [y_dimname, x_dimname]
    var_dict = {
        x_varname: (dim2d, XLONG_reg, XLONG.attrs),
        y_varname: (dim2d, XLAT_reg, XLAT.attrs),
        reflectivity_varname: (dim4d, np.expand_dims(REFL_reg, axis=0), REFL.attrs),
    }
    # Output coordinates
    coord_dict = {
        time_dimname: ([time_dimname], in_time.data),
        z_varname: ([z_dimname], ds[z_varname].data, ds[z_varname].attrs),
        y_dimname: ([y_dimname], ycoord, ycoord_attrs),
        x_dimname: ([x_dimname], xcoord, xcoord_attrs),
    }
    # Output global attributes
    gattr_dict = {
        'Title': 'Regridded reflectivity from LASSO',
        'SIMULATION_START_DATE': ds.attrs['SIMULATION_START_DATE'],
        'run_name': ds.attrs['run_name'],
        'DX': ds.attrs['DX']*regrid_ratio,
        'DY': ds.attrs['DY']*regrid_ratio,
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
        path=out_filename, mode='w', format='NETCDF4', unlimited_dims=time_dimname, encoding=encoding,
    )
    logger.info(f'{out_filename}')
    return out_filename 

