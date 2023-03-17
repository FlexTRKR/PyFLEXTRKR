import numpy as np
import time
import os, sys, glob
import logging
import xarray as xr
import xesmf as xe
import dask
from dask.distributed import wait
from pyflextrkr.ft_utilities import subset_files_timerange
from pyflextrkr.ft_regrid_func import make_weight_file, make_grid4regridder

#-------------------------------------------------------------------------------------
def regrid_tracking_mask(
        config,
        pixeltracking_inpath=None,
        pixeltracking_basename=None,
        outpath_basename=None,
        outfile_basename=None,
):
    """
    Regrid pixel-level tracking mask to native grid.

    Args:
        config: dictionary
            Dictionary containing config parameters.
        pixeltracking_inpath: string
            Input pixel-level tracking file directory.
        pixeltracking_basename: string
            Input pixel-level tracking file basename.
        outpath_basename: string
            Output directory basename.
        outfile_basename: string
            Output file basename.

    Returns:
        None.
    """
     # Set the logging message level
    logger = logging.getLogger(__name__)

    # Get inputs from config
    run_parallel = config['run_parallel']
    n_workers = config['nprocesses']
    startdate = config["startdate"]
    enddate = config["enddate"]
    start_basetime = config["start_basetime"]
    end_basetime = config["end_basetime"]
    wrfout_path = config['wrfout_path']
    wrfout_basename = config['wrfout_basename']
    regrid_pixel_path_name = config.get('regrid_pixel_path_name')
    # Default pixel-level mask input directory, file basename
    if pixeltracking_inpath is None:
        pixeltracking_inpath = config["pixeltracking_outpath"]
    if pixeltracking_basename is None:
        pixeltracking_filebase = config["pixeltracking_filebase"]
    
    if outpath_basename is None:
        pixeltracking_outpath = f'{config["root_path"]}/{regrid_pixel_path_name}/{startdate}_{enddate}/'
    else:
        pixeltracking_outpath = f'{config["root_path"]}/{outpath_basename}/{startdate}_{enddate}/'
        
    if outfile_basename is None:
        outfile_basename = config["pixeltracking_filebase"]

    # Regrid parameters
    x_coordname_src = config.get('x_coordname_src')
    y_coordname_src = config.get('y_coordname_src')
    x_coordname_dst = config.get('x_coordname')
    y_coordname_dst = config.get('y_coordname')
    weight_filename_rev = config.get('weight_filename_rev')
    regrid_method_rev = config.get('regrid_method_rev')

    # Make output directory
    os.makedirs(pixeltracking_outpath, exist_ok=True)

    # Identify files to process
    # Create pixel tracking file output directory
    os.makedirs(regrid_pixel_path_name, exist_ok=True)
    inputfiles, \
    inputfiles_basetime, \
    inputfiles_datestring, \
    inputfiles_timestring = subset_files_timerange(pixeltracking_inpath,
                                                     pixeltracking_filebase,
                                                     start_basetime,
                                                     end_basetime)
    nfiles = len(inputfiles)
    logger.info(f"Total number of files to process: {nfiles}")

    # Find native input files
    wrfout_files = sorted(glob.glob(f'{wrfout_path}/{wrfout_basename}*'))

    # Make a config dictionary for reverse regridding
    config_regrid = {
        'gridfile_dst': wrfout_files[0],
        'x_coordname_dst': x_coordname_src,
        'y_coordname_dst': y_coordname_src,
        'x_coordname_src': x_coordname_dst,
        'y_coordname_src': y_coordname_dst,
        'weight_filename': weight_filename_rev,
        'regrid_method': regrid_method_rev,
    }

    # Build Regridder
    weight_filename_rev = make_weight_file(inputfiles[0], config_regrid)

    if run_parallel == 0:
        # Serial
        for ifile in range(0, nfiles):
            status = regrid_mask(inputfiles[ifile], pixeltracking_outpath, config_regrid)

    elif run_parallel >= 1:
        # Parallel
        results = []
        for ifile in range(0, nfiles):
            result = dask.delayed(regrid_mask)(inputfiles[ifile], pixeltracking_outpath, config_regrid)
            results.append(result)
        final_result = dask.compute(*results)
        wait(final_result)
    # import pdb; pdb.set_trace()
    return


#-------------------------------------------------------------------------------------
def regrid_mask(inputfile, outdir, config_regrid):
    """
    Regrid pixel-level tracking masks.

    Args:
        inputfile: string
            Input pixel-level filename.
        outdir: string
            Output directory name.
        config_regrid: dictionary
            Dictionary containing regrid parameters.

    Return:
        outfilename: string
            Output regrid filename.
    """

    logger = logging.getLogger(__name__)

    weight_filename = config_regrid.get('weight_filename')
    regrid_method = config_regrid.get('regrid_method')

    # Make output filename
    fname = os.path.basename(inputfile)
    outfilename = f'{outdir}/{fname}'

    # Make source & destination grid data for regridder
    grid_src, grid_dst = make_grid4regridder(inputfile, config_regrid)
    # Retrieve Regridder
    regridder = xe.Regridder(grid_src, grid_dst, method=regrid_method, weights=weight_filename)

    # Read input data
    ds_in = xr.open_dataset(inputfile, mask_and_scale=False)
    # Regrid variables
    tracknumber = regridder(ds_in['tracknumber'], keep_attrs=True)
    merge_tracknumber = regridder(ds_in['merge_tracknumber'], keep_attrs=True)
    split_tracknumber = regridder(ds_in['split_tracknumber'], keep_attrs=True)
    track_status = regridder(ds_in['track_status'], keep_attrs=True)
    cloudtracknumber = regridder(ds_in['cloudtracknumber'], keep_attrs=True)
    cloudmerge_tracknumber = regridder(ds_in['cloudmerge_tracknumber'], keep_attrs=True)
    cloudsplit_tracknumber = regridder(ds_in['cloudsplit_tracknumber'], keep_attrs=True)
    pcptracknumber = regridder(ds_in['pcptracknumber'], keep_attrs=True)
    # import pdb; pdb.set_trace()
 
    # Output coordinates
    x_coord = grid_dst['lon']
    y_coord = grid_dst['lat']
    x_coord = xr.DataArray(x_coord, dims=('y', 'x'))
    y_coord = xr.DataArray(y_coord, dims=('y', 'x'))
    # Make output DataSet
    var_dict = {
        # 'longitude': x_coord,
        # 'latitude': y_coord,
        'tracknumber': tracknumber,
        'merge_tracknumber': merge_tracknumber,
        'split_tracknumber': split_tracknumber,
        'track_status': track_status,
        'cloudtracknumber': cloudtracknumber,
        'cloudmerge_tracknumber': cloudmerge_tracknumber,
        'cloudsplit_tracknumber': cloudsplit_tracknumber,
        'pcptracknumber': pcptracknumber,
    }
    ds_out = xr.Dataset(var_dict)

    # Delete file if it already exists
    if os.path.isfile(outfilename):
        os.remove(outfilename)

    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in ds_out.data_vars}
    # Write to netCDF file
    ds_out.to_netcdf(
        path=outfilename,
        mode="w",
        format="NETCDF4",
        unlimited_dims="time",
        encoding=encoding,
    )
    logger.info(f"{outfilename}")
    return outfilename