import sys
import numpy as np
import xarray as xr
import pandas as pd
import logging
from pyflextrkr.steiner_func import make_dilation_step_func
from pyflextrkr.steiner_func import mod_steiner_classification
from pyflextrkr.steiner_func import expand_conv_core
from pyflextrkr.echotop_func import echotop_height
from pyflextrkr.netcdf_io import write_radar_cellid
from pyflextrkr.hp_utilities import remap_healpix_to_latlon_grid

def idcells_reflectivity(
    input_filename,
    config,
):
    """
    Identifies convective cells using composite radar reflectivity.

    Args:
        input_filename: string
            Input data filename
        config: dictionary
            Dictionary containing config parameters

    Returns:
        cloudid_outfile: string
            Cloudid file name.
    """
    np.set_printoptions(threshold=np.inf)
    logger = logging.getLogger(__name__)

    absConvThres = config['absConvThres']
    minZdiff = config['minZdiff']
    truncZconvThres = config['truncZconvThres']
    mindBZuse = config['mindBZuse']
    dBZforMaxConvRadius = config['dBZforMaxConvRadius']
    conv_rad_increment = config['conv_rad_increment']
    conv_rad_start = config['conv_rad_start']
    bkg_refl_increment = config['bkg_refl_increment']
    maxConvRadius = config['maxConvRadius']
    radii_expand = config['radii_expand']
    weakEchoThres = config['weakEchoThres']
    bkgrndRadius = config['bkgrndRadius']
    min_corearea = config.get('min_corearea', 0)
    min_cellarea = config.get('min_cellarea', 0)
    echotop_gap = config['echotop_gap']
    sfc_dz_min = config['sfc_dz_min']
    sfc_dz_max = config['sfc_dz_max']
    return_diag = config['return_diag']
    dx = config['dx']
    dy = config['dy']
    z_dimname = config.get('z_dimname', 'z')
    fillval = config['fillval']
    geolimits = config.get('geolimits', None)
    convolve_method = config.get('convolve_method', 'ndimage')
    remove_smallcores = config.get('remove_smallcores', True)
    remove_smallcells = config.get('remove_smallcells', False)
    
    # Check if input_source is specified (deprecated parameter)
    if 'input_source' in config:
        logger.info("Note: 'input_source' parameter is no longer needed. "
                   "The reflectivity reader function has been generalized and "
                   "the echo-top height function is now unified to handle all input types.")

    # Set echo classification type values
    types_steiner = {
        'NO_SURF_ECHO': 1,
        'WEAK_ECHO': 2,
        'STRATIFORM': 3,
        'CONVECTIVE': 4,
    }

    # Get composite reflectivity using generic function
    comp_dict = get_composite_reflectivity_generic(input_filename, config)
    
    # Get is_3d flag from comp_dict
    is_3d = comp_dict.get('is_3d', True)

    # Subset domain
    if (geolimits is not None):
        comp_dict = subset_domain(comp_dict, geolimits, dx, dy)
    
    # Get time coordinate to check if we need to loop over multiple times
    time_coords_all = comp_dict['time_coords']
    ntimes = len(time_coords_all)
    
    # Loop over each time step
    cloudid_outfiles = []
    for itime in range(ntimes):
        # Extract variables for this time step
        if ntimes > 1:
            # Select time slice for variables with time dimension
            time_coords = time_coords_all.isel(time=itime)
            dbz3d_filt = comp_dict['dbz3d_filt'].isel(time=itime)
            dbz_comp = comp_dict['dbz_comp'].isel(time=itime)
            dbz_lowlevel = comp_dict['dbz_lowlevel'].isel(time=itime)
            refl = comp_dict['refl'][itime]
        else:
            # Single time: use as-is (backward compatible)
            time_coords = time_coords_all
            dbz3d_filt = comp_dict['dbz3d_filt']
            dbz_comp = comp_dict['dbz_comp']
            dbz_lowlevel = comp_dict['dbz_lowlevel']
            refl = comp_dict['refl']
        
        # Get variables from dictionary (these don't have time dimension)
        x_coords = comp_dict['x_coords']
        y_coords = comp_dict['y_coords']
        grid_lat = comp_dict['grid_lat']
        grid_lon = comp_dict['grid_lon']
        mask_goodvalues = comp_dict['mask_goodvalues']
        radar_lat = comp_dict['radar_lat']
        radar_lon = comp_dict['radar_lon']
        
        # Handle height coordinate (can be 1D, 3D, or 4D)
        height = comp_dict['height']
        if ntimes > 1 and height.ndim == 4:
            # For 4D height [time, z, y, x], extract time slice
            height = height[itime, :, :, :]

        # Convert radii_expand from a list to a numpy array
        radii_expand = np.array(radii_expand)

        # Make step function for convective radius dilation
        bkg_bin, conv_rad_bin = make_dilation_step_func(
            mindBZuse,
            dBZforMaxConvRadius,
            bkg_refl_increment,
            conv_rad_increment,
            conv_rad_start,
            maxConvRadius,
        )

        # Run Steiner classification
        steiner_result = mod_steiner_classification(
            types_steiner, refl, mask_goodvalues, dx, dy,
            bkg_rad=bkgrndRadius * 1000,
            minZdiff=minZdiff,
            absConvThres=absConvThres,
            truncZconvThres=truncZconvThres,
            weakEchoThres=weakEchoThres,
            bkg_bin=bkg_bin,
            conv_rad_bin=conv_rad_bin,
            min_corearea=min_corearea,
            min_cellarea=min_cellarea,
            remove_smallcores=remove_smallcores,
            remove_smallcells=remove_smallcells,
            return_diag=return_diag,
            convolve_method=convolve_method,
        )
        
        # Extract main outputs
        convsf_steiner = steiner_result['sclass']
        core_steiner = steiner_result['score']
        core_dilate = steiner_result['score_dilate']
        
        # Extract diagnostic outputs if requested
        if return_diag:
            refl_bkg = steiner_result['refl_bkg']
            peakedness = steiner_result['peakedness']
            core_steiner_orig = steiner_result['score_orig']

        # Expand convective cell masks outward to a set of radii to
        # increase the convective cell footprint for better tracking convective cells
        core_expand, core_sorted = expand_conv_core(
            core_dilate, radii_expand, dx, dy, min_corenpix=0)

        # Calculate echo-top heights for various reflectivity thresholds
        shape_2d = refl.shape
        if not is_3d:
            # For 2D composite reflectivity input, skip echo-top height calculations
            logger.info("Input data is 2D, skipping echo-top height calculations")
            echotop10 = np.full(shape_2d, np.nan, dtype=np.float32)
            echotop20 = np.full(shape_2d, np.nan, dtype=np.float32)
            echotop30 = np.full(shape_2d, np.nan, dtype=np.float32)
            echotop40 = np.full(shape_2d, np.nan, dtype=np.float32)
            echotop50 = np.full(shape_2d, np.nan, dtype=np.float32)
        else:
            # Use unified echotop_height function (handles both 1D and 3D height arrays)
            echotop10 = echotop_height(dbz3d_filt, height, z_dimname, shape_2d,
                                       dbz_thresh=10, gap=echotop_gap, min_thick=0)
            echotop20 = echotop_height(dbz3d_filt, height, z_dimname, shape_2d,
                                       dbz_thresh=20, gap=echotop_gap, min_thick=0)
            echotop30 = echotop_height(dbz3d_filt, height, z_dimname, shape_2d,
                                       dbz_thresh=30, gap=echotop_gap, min_thick=0)
            echotop40 = echotop_height(dbz3d_filt, height, z_dimname, shape_2d,
                                       dbz_thresh=40, gap=echotop_gap, min_thick=0)
            echotop50 = echotop_height(dbz3d_filt, height, z_dimname, shape_2d,
                                       dbz_thresh=50, gap=echotop_gap, min_thick=0)

        del dbz3d_filt
        # Put all Steiner parameters in a dictionary
        steiner_params = {
            'dx': dx,
            'dy': dy,
            'absConvThres': absConvThres,
            'minZdiff': minZdiff,
            'truncZconvThres': truncZconvThres,
            'mindBZuse': mindBZuse,
            'dBZforMaxConvRadius': dBZforMaxConvRadius,
            'conv_rad_increment': conv_rad_increment,
            'conv_rad_start': conv_rad_start,
            'bkg_refl_increment': bkg_refl_increment,
            'maxConvRadius': maxConvRadius,
            'weakEchoThres': weakEchoThres,
            'bkgrndRadius': bkgrndRadius,
            'min_corearea': min_corearea,
            'echotop_gap': echotop_gap,
            'sfc_dz_min': sfc_dz_min,
            'sfc_dz_max': sfc_dz_max,
        }

        # Create variables needed for tracking
        feature_mask = core_expand
        # Count number of pixels for each feature
        unique_num, npix_feature = np.unique(feature_mask, return_counts=True)
        # Remove background (unique_num = 0)
        npix_feature = npix_feature[(unique_num > 0)]
        # Get number of features
        nfeatures = np.nanmax(feature_mask)

        # Get date/time and make output filename
        if ntimes > 1:
            # For multiple times, access scalar timestamp
            timestamp = time_coords
        else:
            # For single time, access as array
            timestamp = time_coords[0]
        # Convert to basetime (i.e., Epoch time)
        # This is a more flexible way that can handle non-standard 360 day calendar
        iTimestamp = pd.to_datetime(timestamp.dt.strftime("%Y-%m-%dT%H:%M:%S").item())
        file_basetime = np.array([(iTimestamp - pd.Timestamp('1970-01-01T00:00:00')).total_seconds()])
        # Convert to strings
        file_datestring = timestamp.dt.strftime("%Y%m%d").item()
        file_timestring = timestamp.dt.strftime("%H%M%S").item()
        cloudid_outfile = (
            config["tracking_outpath"] +
            config["cloudid_filebase"] +
            file_datestring +
            "_" +
            file_timestring +
            ".nc"
        )

        # Put time and nfeatures in a numpy array so that they can be set with a time dimension
        out_basetime = np.zeros(1, dtype=float)
        out_basetime[0] = file_basetime

        out_nfeatures = np.zeros(1, dtype=int)
        out_nfeatures[0] = nfeatures

        # Get passing variable DataSet from comp_dict (select time if multiple times)
        ds_pass = comp_dict.get('ds_pass', None)
        if ds_pass is not None and ntimes > 1:
            ds_pass = ds_pass.isel(time=itime)

        # Prepare optional diagnostic parameters
        diag_kwargs = {}
        if return_diag:
            diag_kwargs = {
                'refl_bkg': refl_bkg,
                'peakedness': peakedness,
                'core_steiner_orig': core_steiner_orig,
            }

        # Write output into netcdf file
        write_radar_cellid(
            cloudid_outfile,
            out_basetime,
            file_datestring,
            file_timestring,
            radar_lon,
            radar_lat,
            grid_lon,
            grid_lat,
            x_coords,
            y_coords,
            dbz_comp,
            dbz_lowlevel,
            input_filename,
            sfc_dz_min,
            convsf_steiner,
            core_steiner,
            core_sorted,
            core_expand,
            echotop10,
            echotop20,
            echotop30,
            echotop40,
            echotop50,
            feature_mask,
            npix_feature,
            out_nfeatures,
            config,
            steiner_params=steiner_params,
            ds_pass=ds_pass,
            **diag_kwargs,
        )
        logger.info(f"{cloudid_outfile}")
        cloudid_outfiles.append(cloudid_outfile)

    # Return list of output files (or single file for backward compatibility)
    if ntimes == 1:
        return cloudid_outfiles[0]
    else:
        return cloudid_outfiles

#--------------------------------------------------------------------------------
def subset_domain(comp_dict, geolimits, dx, dy):
    """
    Subset variables within a domain.

    Args:
        comp_dict: dictionary
            Dictionary containing input variables
        geolimits: list
            Subset domain lat/lon limits [lon_min, lat_min, lon_max, lat_max]
        dx: float
            Grid spacing in x-direction
        dy: float
            Grid spacing in y-direction

    Returns:
        comp_dict: dictionary
            Dictionary containing output variables
    """
    # Get variables from dictionary
    x_coords = comp_dict['x_coords']
    y_coords = comp_dict['y_coords']
    dbz3d_filt = comp_dict['dbz3d_filt']
    dbz_comp = comp_dict['dbz_comp']
    dbz_lowlevel = comp_dict['dbz_lowlevel']
    grid_lat = comp_dict['grid_lat']
    grid_lon = comp_dict['grid_lon']
    height = comp_dict['height']
    mask_goodvalues = comp_dict['mask_goodvalues']
    radar_lat = comp_dict['radar_lat']
    radar_lon = comp_dict['radar_lon']
    refl = comp_dict['refl']
    time_coords = comp_dict['time_coords']

    # Subset domain
    if geolimits is not None:
        # Get lat/lon limits
        buffer = 0
        latmin, latmax = geolimits[0]-buffer, geolimits[2]+buffer
        lonmin, lonmax = geolimits[1]-buffer, geolimits[3]+buffer
        
        # Check if grid_lon and grid_lat are 1D or 2D
        if grid_lon.ndim == 1 and grid_lat.ndim == 1:
            # For 1D coordinates, find indices directly
            lon_mask = (grid_lon >= lonmin) & (grid_lon <= lonmax)
            lat_mask = (grid_lat >= latmin) & (grid_lat <= latmax)
            x_idx = np.where(lon_mask)[0]
            y_idx = np.where(lat_mask)[0]
            if len(x_idx) == 0 or len(y_idx) == 0:
                raise ValueError("No data points found within specified geolimits")
            xmin, xmax = x_idx[0], x_idx[-1]
            ymin, ymax = y_idx[0], y_idx[-1]
            # Subset 1D coordinates
            grid_lon = grid_lon[xmin:xmax+1]
            grid_lat = grid_lat[ymin:ymax+1]
        else:
            # For 2D coordinates, create mask and find bounding box
            mask = ((grid_lon >= lonmin) & (grid_lon <= lonmax) & \
                    (grid_lat >= latmin) & (grid_lat <= latmax)).squeeze()
            # Get y/x indices limits from the mask
            y_idx, x_idx = np.where(mask == True)
            if len(x_idx) == 0 or len(y_idx) == 0:
                raise ValueError("No data points found within specified geolimits")
            xmin, xmax = np.min(x_idx), np.max(x_idx)
            ymin, ymax = np.min(y_idx), np.max(y_idx)
            # Subset 2D coordinates
            grid_lon = grid_lon[ymin:ymax+1, xmin:xmax+1]
            grid_lat = grid_lat[ymin:ymax+1, xmin:xmax+1]
        
        # Subset variables (same for both 1D and 2D coordinates)
        # Handle time dimension if present
        if 'time' in dbz3d_filt.dims:
            dbz3d_filt = dbz3d_filt[:, :, ymin:ymax+1, xmin:xmax+1]
            dbz_comp = dbz_comp[:, ymin:ymax+1, xmin:xmax+1]
            dbz_lowlevel = dbz_lowlevel[:, ymin:ymax+1, xmin:xmax+1]
            refl = refl[:, ymin:ymax+1, xmin:xmax+1]
        else:
            dbz3d_filt = dbz3d_filt[:, ymin:ymax+1, xmin:xmax+1]
            dbz_comp = dbz_comp[ymin:ymax+1, xmin:xmax+1]
            dbz_lowlevel = dbz_lowlevel[ymin:ymax+1, xmin:xmax+1]
            refl = refl[ymin:ymax+1, xmin:xmax+1]
        mask_goodvalues = mask_goodvalues[ymin:ymax+1, xmin:xmax+1]
        
        # Vertical coordinate (handle 3D and 4D)
        if height.ndim == 3:  # 3D: [z, y, x]
            height = height[:, ymin:ymax+1, xmin:xmax+1]
        elif height.ndim == 4:  # 4D: [time, z, y, x]
            height = height[:, :, ymin:ymax+1, xmin:xmax+1]
        # Horizontal coordinates
        nx = xmax - xmin + 1
        ny = ymax - ymin + 1
        x_coords = np.arange(0, nx) * dx
        y_coords = np.arange(0, ny) * dy
        
        # Subset ds_pass if it exists
        ds_pass = comp_dict.get('ds_pass', None)
        if ds_pass is not None:
            # Subset all variables in ds_pass that have y and x dimensions
            ds_pass_subset = {}
            for var_name in ds_pass.data_vars:
                var = ds_pass[var_name]
                # Check if variable has spatial dimensions
                if 'y' in var.dims and 'x' in var.dims:
                    # Subset spatial dimensions
                    if 'time' in var.dims:
                        ds_pass_subset[var_name] = var[:, ymin:ymax+1, xmin:xmax+1]
                    else:
                        ds_pass_subset[var_name] = var[ymin:ymax+1, xmin:xmax+1]
                else:
                    # Keep variable as-is if no spatial dimensions
                    ds_pass_subset[var_name] = var
            # Recreate Dataset with subsetted variables
            ds_pass = xr.Dataset(ds_pass_subset)

    # Update variables in the dictionary
    comp_dict['x_coords'] = x_coords
    comp_dict['y_coords'] = y_coords
    comp_dict['dbz3d_filt'] = dbz3d_filt
    comp_dict['dbz_comp'] = dbz_comp
    comp_dict['dbz_lowlevel'] = dbz_lowlevel
    comp_dict['grid_lat'] = grid_lat
    comp_dict['grid_lon'] = grid_lon
    comp_dict['height'] = height
    comp_dict['mask_goodvalues'] = mask_goodvalues
    comp_dict['radar_lat'] = radar_lat
    comp_dict['radar_lon'] = radar_lon
    comp_dict['refl'] = refl
    comp_dict['time_coords'] = time_coords
    comp_dict['ds_pass'] = ds_pass

    return comp_dict

#--------------------------------------------------------------------------------
def get_composite_reflectivity_generic(input_filename, config):
    """
    Generic function to get composite reflectivity from various input sources.
    
    This function is agnostic to different input datasets, requiring only:
    - time coordinate
    - x, y coordinates (1D or 2D)
    - z coordinate (optional, for 3D data)
    - 2D or 3D radar reflectivity variable
    
    Args:
        input_filename: string
            Input data filename
        config: dictionary
            Dictionary containing config parameters
            
    Required config parameters:
        - reflectivity_varname: name of reflectivity variable
        - x_coordname, y_coordname: coordinate variable names (backward compatible with x_varname, y_varname)
        - x_dimname, y_dimname: dimension names
        - time_dimname (optional): time dimension name, default 'time'
        - z_dimname (optional): vertical dimension name, default 'z'
        - z_coordname (optional): vertical coordinate name (backward compatible with z_varname)
        - lon_coordname, lat_coordname (optional): lat/lon coordinate names (backward compatible with lon_varname, lat_varname)
        - dx, dy: grid spacing in meters
        - radar_sensitivity: minimum reflectivity threshold
        - sfc_dz_min, sfc_dz_max: height range for low-level reflectivity
        - z_coord_type (optional): 'height' or 'pressure', default auto-detect
          For pressure coordinates, sfc_dz_min/max are in hPa (surface is high pressure)
        - default_sfc_pressure (optional): default surface pressure in hPa when terrain_file
          is not provided and z_coord_type='pressure', default 1013.25 hPa
        - composite_reflectivity_varname (optional): pre-computed composite reflectivity
        - pass_varname (optional): list of additional variables to pass through
        - is_3d (optional): flag to indicate if data is 3D, default auto-detect
        
    Returns:
        comp_dict: dictionary
            Dictionary containing standardized output variables
    """
    logger = logging.getLogger(__name__)
    
    # Get configuration parameters
    input_format = config.get("input_format", "netcdf")
    radar_sensitivity = config['radar_sensitivity']
    sfc_dz_min = config['sfc_dz_min']
    sfc_dz_max = config['sfc_dz_max']
    
    # Check required grid spacing parameters
    dx = config.get('dx', None)
    dy = config.get('dy', None)
    if dx is None:
        raise ValueError("'dx' (grid spacing in x-direction) must be specified in config")
    if dy is None:
        raise ValueError("'dy' (grid spacing in y-direction) must be specified in config")
    
    time_dimname = config.get('time_dimname', 'time')
    time_coordname = config.get('time_coordname', time_dimname)
    x_dimname = config.get('x_dimname', 'x')
    y_dimname = config.get('y_dimname', 'y')
    z_dimname = config.get('z_dimname', 'z')
    # Use new coordname parameters with backward compatibility to varname parameters
    x_coordname = config.get('x_coordname', config.get('x_varname'))
    y_coordname = config.get('y_coordname', config.get('y_varname'))
    z_coordname = config.get('z_coordname', config.get('z_varname', z_dimname))
    lon_coordname = config.get('lon_coordname', config.get('lon_varname', x_coordname))
    lat_coordname = config.get('lat_coordname', config.get('lat_varname', y_coordname))
    reflectivity_varname = config['reflectivity_varname']
    composite_reflectivity_varname = config.get('composite_reflectivity_varname', None)
    pass_varname = config.get('pass_varname', None)
    radar_lon_varname = config.get('radar_lon_varname', None)
    radar_lat_varname = config.get('radar_lat_varname', None)
    radar_alt_varname = config.get('radar_alt_varname', None)
    terrain_file = config.get('terrain_file', None)
    elev_varname = config.get('elev_varname', None)
    rangemask_varname = config.get('rangemask_varname', None)
    is_3d = config.get('is_3d', None)  # None = auto-detect
    z_coord_type = config.get('z_coord_type', None)  # None = auto-detect, 'height' or 'pressure'
    default_sfc_pressure = config.get('default_sfc_pressure', 1013.25)  # hPa, used when terrain_file=None and z_coord_type='pressure'
    default_sfc_height = config.get('default_sfc_height', 0)  # meters, used when terrain_file=None and z_coord_type='height'
    round_time_to_second = config.get('round_time_to_second', False)  # Flag to round time coordinate to nearest second
    
    # Read input file
    if input_format == 'netcdf':
        ds = xr.open_dataset(input_filename)
    elif input_format == 'zarr':
        ds_hp = input_filename
    else:
        logger.error(f'Unknown input_format: {input_format}')
        sys.exit()

    # Handle HEALPix format: remap to regular grid before proceeding
    if input_format == 'zarr':
        latlon_filename = config.get('latlon_filename', None)
        if latlon_filename is None:
            logger.error("For HEALPix zarr input, 'latlon_filename' must be specified in config")
            sys.exit()
        # Remap HEALPix to lat/lon grid
        ds = remap_healpix_to_latlon_grid(
            ds_hp,
            latlon_filename,
            config,
        )
    
    # Handle special WRF case: rename Time -> time, handle XTIME
    if 'Time' in ds.dims and time_dimname == 'time':
        if 'XTIME' in ds.variables:
            ds = ds.reset_coords(names='XTIME', drop=False)
        ds = ds.rename({'Time': time_dimname})
    
    # Get dimension names from the file
    dims_file = []
    for key in ds.dims: dims_file.append(key)
    # Find extra dimensions beyond [time, z, y, x] or [time, y, x]
    if z_dimname is not None:
        dims_keep = [time_dimname, z_dimname, y_dimname, x_dimname]
    else:
        dims_keep = [time_dimname, y_dimname, x_dimname]
    dims_drop = list(set(dims_file) - set(dims_keep))
    # Reorder Dataset dimensions
    if z_dimname is not None:
        # Drop extra dimensions, reorder to [time, z, y, x]
        ds = ds.drop_dims(dims_drop).transpose(
            time_dimname, z_dimname, y_dimname, x_dimname, missing_dims='ignore'
        )
    else:
        # Drop extra dimensions, reorder to [time, y, x]
        ds = ds.drop_dims(dims_drop).transpose(
            time_dimname, y_dimname, x_dimname, missing_dims='ignore'
        )
    
    # Get time coordinate
    if time_coordname in ds.variables:
        time_coords = ds[time_coordname]
        # Round to nearest second if requested
        if round_time_to_second and hasattr(time_coords.dt, 'round'):
            time_coords = time_coords.dt.round('s')
    else:
        time_coords = ds[time_dimname]
        if round_time_to_second and hasattr(time_coords.dt, 'round'):
            time_coords = time_coords.dt.round('s')
    
    # Auto-detect if data is 3D
    if is_3d is None:
        is_3d = z_dimname in ds[reflectivity_varname].dims

    # Get spatial dimensions
    nx = ds.sizes[x_dimname]
    ny = ds.sizes[y_dimname]
    
    # Get x, y coordinates
    if x_coordname in ds.variables:
        x_coords = ds[x_coordname].data
    else:
        # Create x coordinate from dimension using dx from config
        x_coords = np.arange(0, nx) * dx
    
    if y_coordname in ds.variables:
        y_coords = ds[y_coordname].data
    else:
        # Create y coordinate from dimension using dy from config
        y_coords = np.arange(0, ny) * dy
    
    # Get grid lat/lon (can be 1D or 2D)
    if lon_coordname in ds.variables:
        grid_lon = ds[lon_coordname].squeeze()
        # For 3D lat/lon, take first vertical level
        if z_dimname in grid_lon.dims:
            grid_lon = grid_lon.isel({z_dimname: 0})
    else:
        # Create dummy grid_lon
        grid_lon = xr.DataArray(
            np.zeros((ny, nx)), 
            coords={y_dimname: y_coords, x_dimname: x_coords},
            dims=(y_dimname, x_dimname)
        )
    
    if lat_coordname in ds.variables:
        grid_lat = ds[lat_coordname].squeeze()
        # For 3D lat/lon, take first vertical level
        if z_dimname in grid_lat.dims:
            grid_lat = grid_lat.isel({z_dimname: 0})
    else:
        # Create dummy grid_lat
        grid_lat = xr.DataArray(
            np.zeros((ny, nx)),
            coords={y_dimname: y_coords, x_dimname: x_coords},
            dims=(y_dimname, x_dimname)
        )
    
    # Convert 1D grid_lon/grid_lat to 2D for compatibility with write_radar_cellid
    if grid_lon.ndim == 1 and grid_lat.ndim == 1:
        # Create 2D meshgrid from 1D coordinates
        grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lon.data, grid_lat.data)
        grid_lon = xr.DataArray(
            grid_lon_2d,
            coords={y_dimname: y_coords, x_dimname: x_coords},
            dims=(y_dimname, x_dimname)
        )
        grid_lat = xr.DataArray(
            grid_lat_2d,
            coords={y_dimname: y_coords, x_dimname: x_coords},
            dims=(y_dimname, x_dimname)
        )

    # Get radar location (if available)
    if radar_lon_varname and radar_lon_varname in ds.variables:
        radar_lon = ds[radar_lon_varname]
        radar_lat = ds[radar_lat_varname]
        radar_alt = ds[radar_alt_varname].squeeze() if radar_alt_varname in ds.variables else 0
    else:
        # Create fake radar lat/lon for model data
        radar_lon = xr.DataArray(0, attrs={'long_name': 'Radar longitude'})
        radar_lat = xr.DataArray(0, attrs={'long_name': 'Radar latitude'})
        radar_alt = 0
    
    # Process 3D reflectivity data
    if is_3d:
        # Get vertical coordinate
        if z_coordname in ds.variables:
            height = ds[z_coordname].data
        else:
            # Special case for WRF geopotential height
            if 'PH' in ds.variables and 'PHB' in ds.variables:
                height = ((ds['PH'] + ds['PHB']).squeeze().data / 9.80665)
            else:
                logger.warning(f"Vertical coordinate {z_coordname} not found, using index")
                height = np.arange(ds.sizes[z_dimname])
        
        # Auto-detect vertical coordinate type if not specified
        if z_coord_type is None:
            # Check if coordinate is monotonically increasing or decreasing
            if z_coordname in ds.variables:
                # Get a representative vertical profile for checking
                if height.ndim == 1:
                    z_profile = height
                elif height.ndim == 3:  # 3D: [z, y, x]
                    z_profile = height[:, 0, 0]
                else:  # 4D: [time, z, y, x]
                    z_profile = height[0, :, 0, 0]
                
                # For height: typically increases from surface (low) to top (high)
                # For pressure: typically decreases from surface (high ~1000 hPa) to top (low ~0.1 hPa)
                if z_profile[0] > z_profile[-1]:
                    z_coord_type = 'pressure'
                    logger.info(f"Auto-detected vertical coordinate type: pressure (decreasing from {z_profile[0]:.2f} to {z_profile[-1]:.2f})")
                else:
                    z_coord_type = 'height'
                    logger.info(f"Auto-detected vertical coordinate type: height (increasing from {z_profile[0]:.2f} to {z_profile[-1]:.2f})")
            else:
                # Default to height if cannot determine
                z_coord_type = 'height'
                logger.warning("Cannot auto-detect vertical coordinate type, defaulting to 'height'")
        else:
            logger.info(f"Using specified vertical coordinate type: {z_coord_type}")
        
        # Standardize vertical coordinate direction for echo-top height calculations
        # Height coordinates should increase from surface to top (e.g., 0, 500, 1000, ..., 20000 m)
        # Pressure coordinates should decrease from surface to top (e.g., 1000, 950, 900, ..., 100 hPa)
        
        # Get a representative vertical profile for checking direction
        if height.ndim == 1:
            z_profile = height
        elif height.ndim == 3:  # 3D: [z, y, x]
            z_profile = height[:, 0, 0]
        else:  # 4D: [time, z, y, x]
            z_profile = height[0, :, 0, 0]
        
        # Check if vertical coordinate needs to be reversed
        need_reverse = False
        if z_coord_type == 'height':
            # Height should increase with index (surface=low, top=high)
            if z_profile[0] > z_profile[-1]:
                logger.info(f"Reversing height coordinate to go from surface ({z_profile[-1]:.2f} m) to top ({z_profile[0]:.2f} m)")
                need_reverse = True
        elif z_coord_type == 'pressure':
            # Pressure should decrease with index (surface=high, top=low)
            if z_profile[0] < z_profile[-1]:
                logger.info(f"Reversing pressure coordinate to go from surface ({z_profile[-1]:.2f} hPa) to top ({z_profile[0]:.2f} hPa)")
                need_reverse = True
        
        # Reverse the entire dataset along z dimension if needed
        if need_reverse:
            # Flip height array (handle 1D, 3D, and 4D)
            if height.ndim == 1:
                height = height[::-1]
            elif height.ndim == 3:  # 3D: [z, y, x]
                height = height[::-1, :, :]
            else:  # 4D: [time, z, y, x]
                height = height[:, ::-1, :, :]
            
            # Flip the entire dataset along z dimension to maintain consistency across all variables
            ds = ds.isel({z_dimname: slice(None, None, -1)})
        
        # Get or create surface elevation
        if terrain_file is not None:
            logger.info(f"Loading terrain file: {terrain_file}")
            dster = xr.open_dataset(terrain_file)
            dster = dster.assign_coords({
                y_dimname: ds[y_coordname], 
                x_dimname: ds[x_coordname]
            })
            sfc_elev = dster[elev_varname]
            mask_goodvalues = dster[rangemask_varname].data.astype(int)
        else:
            # Create default surface elevation/pressure
            if z_coord_type == 'pressure':
                # For pressure coordinates, use configured or default sea level pressure
                if isinstance(x_coords, np.ndarray) and x_coords.ndim == 2:
                    # For 2D coordinates, create with index dimensions only
                    sfc_elev = xr.DataArray(
                        np.full((ny, nx), default_sfc_pressure),
                        dims=(y_dimname, x_dimname)
                    )
                else:
                    sfc_elev = xr.DataArray(
                        np.full((ny, nx), default_sfc_pressure),
                        coords={y_dimname: y_coords, x_dimname: x_coords},
                        dims=(y_dimname, x_dimname)
                    )
                logger.info(f"No terrain file specified. Using default surface pressure: {default_sfc_pressure} hPa")
            else:
                # For height coordinates, use configured or default surface height
                if isinstance(x_coords, np.ndarray) and x_coords.ndim == 2:
                    # For 2D coordinates, create with index dimensions only
                    sfc_elev = xr.DataArray(
                        np.full((ny, nx), default_sfc_height),
                        dims=(y_dimname, x_dimname)
                    )
                else:
                    sfc_elev = xr.DataArray(
                        np.full((ny, nx), default_sfc_height),
                        coords={y_dimname: y_coords, x_dimname: x_coords},
                        dims=(y_dimname, x_dimname)
                    )
                logger.info(f"No terrain file specified. Using default surface height: {default_sfc_height} m")
            mask_goodvalues = np.full((ny, nx), 1, dtype=int)
        
        # Handle height coordinate transformation (AGL to MSL if needed)
        if radar_alt_varname in ds.variables:
            logger.info(f"Converting vertical coordinate {z_coordname} from AGL to MSL using radar altitude: {radar_alt}")
            z_agl = ds[z_dimname] + radar_alt
            ds[z_dimname] = z_agl
        
        # Get reflectivity data
        dbz = ds[reflectivity_varname].squeeze()

        # Apply quality control filters (optional)
        # Check for normalized coherent power filtering
        if 'normalized_coherent_power' in ds.variables:
            ncp = ds['normalized_coherent_power'].squeeze()
            dbz = dbz.where(ncp >= 0.5)
        
        # Filter reflectivity based on vertical coordinate type
        if z_coord_type == 'pressure':
            # For pressure coordinates: surface is high pressure, top is low pressure
            # Filter out levels below surface pressure (sfc_elev represents surface pressure)
            # Keep levels where pressure < surface pressure - sfc_dz_min
            dbz3d_filt = dbz.where(ds[z_coordname] < (sfc_elev - sfc_dz_min))
            
            # Calculate low-level composite reflectivity
            # Low-level is defined as pressure range from (sfc - sfc_dz_max) to (sfc - sfc_dz_min)
            # This captures levels close to but above the surface
            dbz3d_lowlevel = dbz.where(
                (ds[z_coordname] >= (sfc_elev - sfc_dz_max)) & 
                (ds[z_coordname] <= (sfc_elev - sfc_dz_min))
            )
        else:
            # For height coordinates: surface is low, top is high (traditional case)
            # Filter reflectivity below surface + minimum height
            dbz3d_filt = dbz.where(ds[z_coordname] > (sfc_elev + sfc_dz_min))
            
            # Calculate low-level composite reflectivity
            dbz3d_lowlevel = dbz.where(
                (ds[z_coordname] >= sfc_dz_min) & (ds[z_coordname] <= sfc_dz_max)
            )
        
        # Calculate composite reflectivity
        if composite_reflectivity_varname and composite_reflectivity_varname in ds.variables:
            dbz_comp = ds[composite_reflectivity_varname].squeeze()
        else:
            dbz_comp = dbz3d_filt.max(dim=z_dimname)
        
        # Calculate low-level maximum reflectivity
        dbz_lowlevel = dbz3d_lowlevel.max(dim=z_dimname)

    else:
        # 2D composite reflectivity case
        dbz_comp = ds[reflectivity_varname].squeeze()
        dbz_lowlevel = ds[reflectivity_varname].squeeze().copy()
        # Create fake 3D array for compatibility
        dbz3d_filt = dbz_comp.expand_dims(z_dimname, axis=0)
        height = np.zeros(1)
        mask_goodvalues = np.full((ny, nx), 1, dtype=int)
    
    # Extract pass-through variables if requested
    if pass_varname is not None:
        pass_varname_set = set(ds.data_vars) & set(pass_varname)
        ds_pass = ds[pass_varname_set] if pass_varname_set else None
    else:
        ds_pass = None

    # Prepare reflectivity array for Steiner classification
    refl = np.copy(dbz_comp.data)
    # Replace values below radar sensitivity and NaN with sensitivity value
    refl[(refl < radar_sensitivity) | np.isnan(refl)] = radar_sensitivity

    # TODO: Add an option to overwrite composite reflectivity array (refl) with CAPPI

    # Return standardized dictionary
    comp_dict = {
        'x_coords': x_coords,
        'y_coords': y_coords,
        'dbz3d_filt': dbz3d_filt,
        'dbz_comp': dbz_comp,
        'dbz_lowlevel': dbz_lowlevel,
        'grid_lat': grid_lat,
        'grid_lon': grid_lon,
        'height': height,
        'mask_goodvalues': mask_goodvalues,
        'radar_lat': radar_lat,
        'radar_lon': radar_lon,
        'refl': refl,
        'time_coords': time_coords,
        'ds_pass': ds_pass,
        'is_3d': is_3d,
    }
    
    return comp_dict
