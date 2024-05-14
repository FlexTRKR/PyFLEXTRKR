import glob
import os
import sys
import time
import logging
import numpy as np
from scipy import ndimage
import xarray as xr
import dask
from dask.distributed import wait
from pyflextrkr.ft_utilities import subset_files_timerange

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


def regrid_radar_reflectivity(config):
    """
    Driver to coarsen 3D radar reflectivity for tracking.

    Args:
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        None.
    """

    logger = logging.getLogger(__name__)
    logger.info('Regridding radar reflectivity')

    in_dir = config['rawdata_path']
    in_basename = config['rawdatabasename']
    out_dir = config["clouddata_path"]
    out_basename = config['databasename']
    run_parallel = config['run_parallel']
    start_basetime = config["start_basetime"]
    end_basetime = config["end_basetime"]
    time_format = config["time_format"]

    os.makedirs(out_dir, exist_ok=True)

    #########################################################################################
    # Identify files to process
    infiles_info = subset_files_timerange(
        in_dir,
        in_basename,
        start_basetime,
        end_basetime,
        time_format=time_format,
    )
    # Get file list
    in_files = infiles_info[0]
    nfiles = len(in_files)
    logger.info(f"Total number of files to process: {nfiles}")

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
    out_reflectivity = np.full(in_reflectivity.shape, np.NaN, dtype=np.float32)
    out_reflectivity[numPixs>0] = 10.0 * np.log10(bkg_linrefl[numPixs>0] / numPixs[numPixs>0])

    # Remove pixels with 0 number of pixels
    out_reflectivity[mask_goodvalues==0] = np.NaN
    
    return out_reflectivity

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

    time_dimname = config.get('time_dimname', 'time')
    reflectivity_varname = config.get('reflectivity_varname', 'reflectivity')
    lon_varname = config.get('lon_varname', 'lon')
    lat_varname = config.get('lat_varname', 'lat')
    x_varname = config.get('x_varname', 'x')
    y_varname = config.get('y_varname', 'y')
    z_varname = config.get('z_varname', 'z')
    x_dimname = config.get('x_dimname', 'x')
    y_dimname = config.get('y_dimname', 'y')
    z_dimname = config.get('z_dimname', 'z')
    radar_lon_varname = config.get('radar_lon_varname', 'origin_longitude')
    radar_lat_varname = config.get('radar_lat_varname', 'origin_latitude')
    radar_alt_varname = config.get('radar_alt_varname', 'alt')
    dx = config['dx']
    dy = config['dy']
    regrid_ratio = config.get('regrid_ratio')

    # Read input data
    ds = xr.open_dataset(in_filename)
    in_time = ds[time_dimname]
    # Get radar location variables
    radar_lon_attrs = ds[radar_lon_varname].attrs if radar_lon_varname in ds else ""
    radar_lat_attrs = ds[radar_lat_varname].attrs if radar_lat_varname in ds else ""
    radar_alt = ds[radar_alt_varname].data if radar_alt_varname in ds else 0.0
    # Get variables
    REFL = ds[reflectivity_varname].squeeze()
    longitude = ds[lon_varname]
    latitude = ds[lat_varname]

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
    # Check lat/lon array dimension
    if longitude.ndim == 1:
        longitude_reg = longitude.data[start_idx::regrid_ratio]
        latitude_reg = latitude.data[start_idx::regrid_ratio]
    elif longitude.ndim == 2:
        longitude_reg = longitude.data[start_idx::regrid_ratio,start_idx::regrid_ratio]
        latitude_reg = latitude.data[start_idx::regrid_ratio,start_idx::regrid_ratio]
    elif longitude.ndim == 3:
        longitude_reg = longitude.data[:,start_idx::regrid_ratio,start_idx::regrid_ratio]
        latitude_reg = latitude.data[:,start_idx::regrid_ratio,start_idx::regrid_ratio]

    # Make output filename
    nleadingchar = len(f'{in_basename}')
    fname = os.path.basename(in_filename)
    ftimestr = fname[nleadingchar:]
    out_filename = f'{out_dir}{out_basename}{ftimestr}'
    
    # Make output coordinate
    nz, ny, nx = REFL_reg.shape
    xcoord = create_semi_symmetric_array(nx) * dx
    ycoord = create_semi_symmetric_array(ny) * dy
    xcoord_attrs = ds[x_varname].attrs
    ycoord_attrs = ds[y_varname].attrs
    # Get radar lat/lon from the regridded lat/lon
    xid0 = np.nanargmin(np.absolute(xcoord - 0))
    yid0 = np.nanargmin(np.absolute(ycoord - 0))
    if longitude.ndim == 1:
        radar_lon = longitude_reg[xid0].item()
        radar_lat = latitude_reg[yid0].item()
    elif longitude.ndim == 2:
        radar_lon = longitude_reg[yid0, xid0].item()
        radar_lat = latitude_reg[yid0, xid0].item()
    elif longitude.ndim == 3:
        radar_lon = longitude_reg[0, yid0, xid0].item()
        radar_lat = latitude_reg[0, yid0, xid0].item()
    # Expand dimension if needed
    # if np.isscalar(radar_lon): radar_lon = np.expand_dims(radar_lon, axis=0)
    # if np.isscalar(radar_lat): radar_lat = np.expand_dims(radar_lat, axis=0)

    # Define output variables
    # radar_lon = xr.DataArray(radar_lon, attrs=radar_lon_attrs)
    # radar_lat = xr.DataArray(radar_lat, attrs=radar_lat_attrs)
    # radar_alt = xr.DataArray(radar_alt, attrs=radar_alt_attrs)
    # Array dimensions
    dim4d = [time_dimname, z_dimname, y_dimname, x_dimname]
    dim3d = [z_dimname, y_dimname, x_dimname]
    dim2d = [y_dimname, x_dimname]
    var_dict = {
        reflectivity_varname: (dim4d, np.expand_dims(REFL_reg, axis=0), REFL.attrs),
        'origin_longitude': radar_lon, 
        'origin_latitude': radar_lat,
        'alt': radar_alt,
        # 'origin_longitude': ([time_dimname], radar_lon, radar_lon_attrs),
        # 'origin_latitude': ([time_dimname], radar_lat, radar_lat_attrs),
        # 'alt': ([time_dimname], np.expand_dims(radar_alt.data, axis=0), radar_alt.attrs),
    }
    # Add lat/lon to dictionary
    if longitude_reg.ndim == 2:
        var_dict[lon_varname] = (dim2d, longitude_reg, longitude.attrs)
        var_dict[lat_varname] = (dim2d, latitude_reg, latitude.attrs)
    elif longitude_reg.ndim == 3:
        var_dict[lon_varname] = (dim3d, longitude_reg, longitude.attrs)
        var_dict[lat_varname] = (dim3d, latitude_reg, latitude.attrs)
    # Output coordinates
    coord_dict = {
        time_dimname: ([time_dimname], in_time.data),
        z_varname: ([z_dimname], ds[z_varname].data, ds[z_varname].attrs),
        y_dimname: ([y_dimname], ycoord, ycoord_attrs),
        x_dimname: ([x_dimname], xcoord, xcoord_attrs),
    }
    # Output global attributes
    gattr_dict = {
        'Title': 'Regridded radar reflectivity',
        'DX': dx,
        'DY': dy,
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

