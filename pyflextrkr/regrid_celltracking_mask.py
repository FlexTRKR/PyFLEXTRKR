import os
import sys
import time
import glob
import logging
import numpy as np
import xarray as xr
import dask
from dask.distributed import wait
from pyflextrkr.ft_utilities import subset_files_timerange

def regrid_celltracking_mask(config):
    """
    Driver to regrid pixel level files to a highger resolution.

    Args:
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        None.
    """

    logger = logging.getLogger(__name__)
    logger.info('Mapping tracked features to pixel-level files')

    in_dir = config['pixeltracking_outpath']
    in_basename = config['pixeltracking_filebase']
    out_basename = f'regrid_{in_basename}'
    out_dir = in_dir
    run_parallel = config['run_parallel']
    start_basetime = config["start_basetime"]
    end_basetime = config["end_basetime"]

    #########################################################################################
    # Identify files to process
    # Create pixel tracking file output directory
    os.makedirs(out_dir, exist_ok=True)
    in_files, infiles_basetime, \
        infiles_datestring, infiles_timestring = subset_files_timerange(
            in_dir, in_basename, start_basetime, end_basetime,
        )
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

    logger.info('Done with regridding pixel-level files')
    return


def regrid_file(in_filename, in_basename, out_dir, out_basename, config):
    """
    Regrid pixel level masks for a given input file.

    Args:
        in_filename: string
            Input file name.
        in_basename: string
            Input file basename.
        out_dir: string
            Output file directory.
        out_basename: string
            Output file basename.
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        out_filename: string
            Output pixel level file name.
    """

    logger = logging.getLogger(__name__)

    raw_dir = config['rawdata_path']
    raw_basename = config['rawdatabasename']
    x_varname = config['x_varname']
    y_varname = config['y_varname']
    regrid_ratio = config.get('regrid_ratio')
    # 
    raw_files = sorted(glob.glob(f'{raw_dir}{raw_basename}*'))
    if (len(raw_files) == 0):
        logger.critical(f"ERROR: No raw files found: {raw_dir}{raw_basename}*")
        logger.critical(f"Require raw files for native data coordinates to regrid tracking masks.")
        logger.critical("Tracking will now exit.")
        sys.exit()
    else:
        # Read raw input file
        dso = xr.open_dataset(raw_files[0])
        # Get coordinates
        xcoord_orig = dso[x_varname].squeeze()
        ycoord_orig = dso[y_varname].squeeze()
        xcoord_attrs = xcoord_orig.attrs
        ycoord_attrs = ycoord_orig.attrs
        # Check coordinate dimensions
        if (xcoord_orig.ndim == 1) | (ycoord_orig.ndim == 1):
            # Mesh 1D coordinate into 2D
            xcoord_2d, ycoord_2d = np.meshgrid(xcoord_orig, ycoord_orig)
        elif (xcoord_orig.ndim == 2) | (ycoord_orig.ndim == 2):
            xcoord_2d = xcoord_orig
            ycoord_2d = ycoord_orig
        else:
            logger.critical("ERROR: Unexpected input data x, y coordinate dimensions.")
            logger.critical(f"{x_varname} dimension: {xcoord_orig.ndim}")
            logger.critical(f"{y_varname} dimension: {ycoord_orig.ndim}")
            logger.critical("Tracking will now exit.")
            sys.exit()

    # Read input data
    ds = xr.open_dataset(in_filename, decode_times=False, mask_and_scale=False)
    time_coord = ds['time']
    ny, nx = ds.sizes['lat'], ds.sizes['lon']
    # Create a coordinate to mimic subsampling regrid_ratio:1 ratio of the full coordinate
    # regrid_ratio = 5
    xcoord = (np.linspace(2, nx*regrid_ratio+2, nx, endpoint=False, dtype=int))
    ycoord = (np.linspace(2, ny*regrid_ratio+2, ny, endpoint=False, dtype=int))
    # Replace the input data coordinate
    ds = ds.assign_coords({'lat':ycoord, 'lon':xcoord})

    # Create a full coordinate
    xcoord_out = np.arange(0, nx*regrid_ratio, 1)
    ycoord_out = np.arange(0, ny*regrid_ratio, 1)

    # Get variables for regridding
    tracknumber = ds['tracknumber']
    # tracknumber_cmask = ds['tracknumber_cmask']
    track_status = ds['track_status']
    feature_number = ds['feature_number']
    merge_tracknumber = ds['merge_tracknumber']
    split_tracknumber = ds['split_tracknumber']
    conv_core = ds['conv_core']
    conv_mask = ds['conv_mask']
    # comp_ref = ds['comp_ref']
    dbz_comp = ds['dbz_comp']
    dbz_lowlevel = ds['dbz_lowlevel']
    echotop10 = ds['echotop10']

    # Remap to the full coordinate
    # Mask variables
    tracknumber_out = tracknumber.interp(
        lon=xcoord_out, lat=ycoord_out, method='nearest', 
        assume_sorted=True, kwargs={"fill_value": "extrapolate"},
    ).data.astype(int)
    # tracknumber_cmask_out = tracknumber_cmask.interp(
    #     lon=xcoord_out, lat=ycoord_out, method='nearest', 
    #     assume_sorted=True, kwargs={"fill_value": "extrapolate"},
    # ).data.astype(int)
    track_status_out = track_status.interp(
        lon=xcoord_out, lat=ycoord_out, method='nearest', 
        assume_sorted=True, kwargs={"fill_value": "extrapolate"},
    ).data.astype(int)
    feature_number_out = feature_number.interp(
        lon=xcoord_out, lat=ycoord_out, method='nearest', 
        assume_sorted=True, kwargs={"fill_value": "extrapolate"},
    ).data.astype(int)
    merge_tracknumber_out = merge_tracknumber.interp(
        lon=xcoord_out, lat=ycoord_out, method='nearest', 
        assume_sorted=True, kwargs={"fill_value": "extrapolate"},
    ).data.astype(int)
    split_tracknumber_out = split_tracknumber.interp(
        lon=xcoord_out, lat=ycoord_out, method='nearest', 
        assume_sorted=True, kwargs={"fill_value": "extrapolate"},
    ).data.astype(int)
    conv_core_out = conv_core.interp(
        lon=xcoord_out, lat=ycoord_out, method='nearest', 
        assume_sorted=True, kwargs={"fill_value": "extrapolate"},
    ).data.astype(int)
    conv_mask_out = conv_mask.interp(
        lon=xcoord_out, lat=ycoord_out, method='nearest', 
        assume_sorted=True, kwargs={"fill_value": "extrapolate"},
    ).data.astype(int)
    # Radar variables
    dbz_comp_out = dbz_comp.interp(
        lon=xcoord_out, lat=ycoord_out, method='nearest', 
        assume_sorted=True, kwargs={"fill_value": "extrapolate"},
    ).data
    dbz_lowlevel_out = dbz_lowlevel.interp(
        lon=xcoord_out, lat=ycoord_out, method='nearest', 
        assume_sorted=True, kwargs={"fill_value": "extrapolate"},
    ).data
    echotop10_out = echotop10.interp(
        lon=xcoord_out, lat=ycoord_out, method='nearest', 
        assume_sorted=True, kwargs={"fill_value": "extrapolate"},
    ).data

    # Make output filename
    nleadingchar = len(f'{in_basename}')
    fname = os.path.basename(in_filename)
    ftimestr = fname[nleadingchar:]
    out_filename = f'{out_dir}{out_basename}{ftimestr}'

    # Define variable list
    var_dict = {
        "base_time": (["time"], time_coord.data, time_coord.attrs),
        "longitude": (["lat", "lon"], xcoord_2d.data, xcoord_attrs),
        "latitude": (["lat", "lon"], ycoord_2d.data, ycoord_attrs),
        # "nclouds": (["time"], ds['nclouds'].data, ds['nclouds'].attrs),
        "dbz_comp": (["time", "lat", "lon"], dbz_comp_out, dbz_comp.attrs),
        "dbz_lowlevel": (["time", "lat", "lon"], dbz_lowlevel_out, dbz_lowlevel.attrs),
        "conv_core": (["time", "lat", "lon"], conv_core_out, conv_core.attrs),
        "conv_mask": (["time", "lat", "lon"], conv_mask_out, conv_mask.attrs),
        "tracknumber": (["time", "lat", "lon"], tracknumber_out, tracknumber.attrs),
        # "tracknumber_cmask": (["time", "lat", "lon"], tracknumber_cmask_out, tracknumber_cmask.attrs),
        "track_status": (["time", "lat", "lon"], track_status_out, track_status.attrs),
        "feature_number": (["time", "lat", "lon"], feature_number_out, feature_number.attrs),
        "merge_tracknumber": (["time", "lat", "lon"], merge_tracknumber_out, merge_tracknumber.attrs),
        "split_tracknumber": (["time", "lat", "lon"], split_tracknumber_out, split_tracknumber.attrs),
        "echotop10": (["time", "lat", "lon"], echotop10_out, echotop10.attrs),
        # "echotop20": (["time", "lat", "lon"], echotop20),
        # "echotop30": (["time", "lat", "lon"], echotop30),
        # "echotop40": (["time", "lat", "lon"], echotop40),
        # "echotop50": (["time", "lat", "lon"], echotop50),
    }
    # Define coordinate list
    coord_dict = {
        "time": (["time"], time_coord.data, time_coord.attrs),
        "lat": (["lat"], ycoord_out),
        "lon": (["lon"], xcoord_out),
    }
    # Define global attributes
    gattr_dict = {
        "title": "Pixel-level cell tracking data regridded back to d4 grid",
        "contact": "Zhe Feng, zhe.feng@pnnl.gov",
        "created_on": time.ctime(time.time()),
    }
    # Define xarray dataset
    ds_out = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in ds_out.data_vars}

    # Write to netcdf file
    ds_out.to_netcdf(
        path=out_filename, mode='w', format='NETCDF4', unlimited_dims='time', encoding=encoding,
    )
    logger.info(f'Output saved: {out_filename}')
    return out_filename 


