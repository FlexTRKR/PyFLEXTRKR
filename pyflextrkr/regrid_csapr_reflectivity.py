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

def regrid_csapr_reflectivity(config):
    """
    Driver to coarsen CSPAR 3D reflectivity for tracking.

    Args:
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        None.
    """

    logger = logging.getLogger(__name__)
    logger.info('Regridding CSPAR reflectivity')

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
    dx = config['dx']
    dy = config['dy']

    # Read input data
    ds = xr.open_dataset(in_filename)
    in_time = ds[time_dimname]
    radar_lon_attrs = ds['origin_longitude'].attrs
    radar_lat_attrs = ds['origin_latitude'].attrs
    radar_alt = ds['alt']
    REFL = ds[reflectivity_varname].squeeze()
    ncp = ds['normalized_coherent_power'].squeeze()
    # REFL_MAX = ds['REFL_MAX'].squeeze()
    longitude = ds[lon_varname]
    latitude = ds[lat_varname]

    # Make a kernel for weights
    ratio = 5   # Must be odd number
    start_idx = int((ratio-1) / 2)
    kernel = np.zeros((ratio+1,ratio+1), dtype=int)
    kernel[1:ratio, 1:ratio] = 1

    # Make a 3D kernel
    kernel3d = kernel[None,:,:]
    # Call convlution function
    REFL_conv = convolve_reflectivity(REFL.data, kernel3d)
    ncp_conv = convolve_var(ncp.data, kernel3d)
    # REFL_MAX_conv = convolve_reflectivity(REFL_MAX.data, kernel)

    # Subsample every X grid points
    REFL_reg = REFL_conv[:,start_idx::ratio,start_idx::ratio]
    ncp_reg = ncp_conv[:,start_idx::ratio,start_idx::ratio]
    # REFL_MAX_reg = REFL_MAX_conv[start_idx::ratio,start_idx::ratio]
    longitude_reg = longitude.data[:,start_idx::ratio,start_idx::ratio]
    latitude_reg = latitude.data[:,start_idx::ratio,start_idx::ratio]

    # Make output filename
    nleadingchar = len(f'{in_basename}')
    fname = os.path.basename(in_filename)
    ftimestr = fname[nleadingchar:]
    out_filename = f'{out_dir}{out_basename}{ftimestr}'
    
    # Make output coordinate
    nz, ny, nx = REFL_reg.shape
    xcoord = np.arange(-nx/2, nx/2, 1) * dx
    ycoord = np.arange(-ny/2, ny/2, 1) * dy
    xcoord_attrs = ds[x_varname].attrs
    ycoord_attrs = ds[y_varname].attrs
    # Get radar lat/lon from the regridded lat/lon
    xid0 = np.where(xcoord == 0)[0]
    yid0 = np.where(ycoord == 0)[0]
    radar_lon = longitude_reg[0, yid0, xid0]
    radar_lat = latitude_reg[0, yid0, xid0]

    # Define output variablesf
    var_dict = {
        lon_varname: ([z_dimname, y_dimname, x_dimname], longitude_reg, longitude.attrs),
        lat_varname: ([z_dimname, y_dimname, x_dimname], latitude_reg, latitude.attrs),
        reflectivity_varname: ([time_dimname, z_dimname, y_dimname, x_dimname], np.expand_dims(REFL_reg, axis=0), REFL.attrs),
        'normalized_coherent_power': ([time_dimname, z_dimname, y_dimname, x_dimname], np.expand_dims(ncp_reg, axis=0), ncp.attrs),
        'origin_longitude': ([time_dimname], radar_lon, radar_lon_attrs),
        'origin_latitude': ([time_dimname], radar_lat, radar_lat_attrs),
        'alt': ([time_dimname], np.expand_dims(radar_alt.data, axis=0), radar_alt.attrs),
        # 'REFL_MAX': (['Time', y_dimname, x_dimname], np.expand_dims(REFL_MAX_reg, axis=0), REFL_MAX.attrs),
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
        'Title': 'Regrid CSAPR reflectivity',
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

