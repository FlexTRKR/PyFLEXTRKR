import sys
import numpy as np
import xarray as xr
import pandas as pd
import logging
from pyflextrkr.steiner_func import make_dilation_step_func
from pyflextrkr.steiner_func import mod_steiner_classification
from pyflextrkr.steiner_func import expand_conv_core
from pyflextrkr.echotop_func import echotop_height
from pyflextrkr.echotop_func import echotop_height_wrf
from pyflextrkr.netcdf_io import write_radar_cellid

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
    input_source = config['input_source']
    geolimits = config.get('geolimits', None)
    convolve_method = config.get('convolve_method', 'ndimage')
    remove_smallcores = config.get('remove_smallcores', True)
    remove_smallcells = config.get('remove_smallcells', False)

    # Set echo classification type values
    types_powell = {
        'NO_ECHO': 1,
        'WEAK_ECHO': 2,
        'STRATIFORM': 3,
        'CONVECTIVE': 4,
        'ISO_CONV_CORE': 5,
        'ISO_CONV_FRINGE': 6,
        'UNCERTAIN': 7,
        'CS_CORE': 8,
        'ISO_CS_CORE': 9,
    }
    types_steiner = {
        'NO_SURF_ECHO': 1,
        'WEAK_ECHO': 2,
        'STRATIFORM': 3,
        'CONVECTIVE': 4,
    }

    # # Get composite reflectivity
    # if input_source == 'radar':
    #     comp_dict = get_composite_reflectivity_radar(
    #         input_filename, config)
    # elif input_source == 'csapr_cacti':
    #     comp_dict = get_composite_reflectivity_csapr_cacti(
    #         input_filename, config)
    # elif input_source == 'wrf_composite':
    #     comp_dict = get_composite_reflectivity_wrf_composite(
    #         input_filename, config
    #     )
    # elif input_source == 'wrf':
    #     comp_dict = get_composite_reflectivity_wrf(
    #         input_filename, config)
    # elif input_source == 'wrf_regrid':
    #     comp_dict = get_composite_reflectivity_wrf_regrid(
    #         input_filename, config)
    # else:
    #     logger.error(f'Unknown input_source: {input_source}')
    #     sys.exit()
    # Get composite reflectivity using generic function
    # Note: The generic function replaces all input_source specific functions
    # (radar, csapr_cacti, wrf, wrf_regrid, wrf_composite)
    comp_dict = get_composite_reflectivity_generic(input_filename, config)

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
        height = comp_dict['height']
        mask_goodvalues = comp_dict['mask_goodvalues']
        radar_lat = comp_dict['radar_lat']
        radar_lon = comp_dict['radar_lon']

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
        if return_diag == False:
            convsf_steiner, \
            core_steiner, \
            core_dilate = mod_steiner_classification(
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

        if return_diag == True:
            convsf_steiner, \
            core_steiner, \
            core_dilate, \
            refl_bkg, \
            peakedness, \
            core_steiner_orig = mod_steiner_classification(
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

        # Expand convective cell masks outward to a set of radii to
        # increase the convective cell footprint for better tracking convective cells
        core_expand, core_sorted = expand_conv_core(
            core_dilate, radii_expand, dx, dy, min_corenpix=0)

        # Calculate echo-top heights for various reflectivity thresholds
        shape_2d = refl.shape
        if (input_source == 'radar') or \
            (input_source == 'csapr_cacti') or \
            (input_source == 'wrf_regrid'):
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
        elif (input_source == 'wrf'):
            echotop10 = echotop_height_wrf(dbz3d_filt, height, z_dimname, shape_2d,
                                           dbz_thresh=10, gap=echotop_gap, min_thick=0)
            echotop20 = echotop_height_wrf(dbz3d_filt, height, z_dimname, shape_2d,
                                           dbz_thresh=20, gap=echotop_gap, min_thick=0)
            echotop30 = echotop_height_wrf(dbz3d_filt, height, z_dimname, shape_2d,
                                           dbz_thresh=30, gap=echotop_gap, min_thick=0)
            echotop40 = echotop_height_wrf(dbz3d_filt, height, z_dimname, shape_2d,
                                           dbz_thresh=40, gap=echotop_gap, min_thick=0)
            echotop50 = echotop_height_wrf(dbz3d_filt, height, z_dimname, shape_2d,
                                           dbz_thresh=50, gap=echotop_gap, min_thick=0)
        elif ('composite' in input_source):
            # For 'composite' reflectivity (2D) input source, skip echo-top height calculations
            echotop10 = np.full(shape_2d, np.nan, dtype=np.float32)
            echotop20 = np.full(shape_2d, np.nan, dtype=np.float32)
            echotop30 = np.full(shape_2d, np.nan, dtype=np.float32)
            echotop40 = np.full(shape_2d, np.nan, dtype=np.float32)
            echotop50 = np.full(shape_2d, np.nan, dtype=np.float32)

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

        # Write output into netcdf file
        if return_diag == False:
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
            )
        if return_diag == True:
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
                refl_bkg=refl_bkg,
                peakedness=peakedness,
                core_steiner_orig=core_steiner_orig,
                ds_pass=ds_pass,
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
        
        # Vertical coordinate
        if height.ndim > 1:
            height = height[:, ymin:ymax+1, xmin:xmax+1]
        # Horizontal coordinates
        nx = xmax - xmin + 1
        ny = ymax - ymin + 1
        x_coords = np.arange(0, nx) * dx
        y_coords = np.arange(0, ny) * dy

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
        - x_varname, y_varname: coordinate variable names
        - x_dimname, y_dimname: dimension names
        - time_dimname (optional): time dimension name, default 'time'
        - z_dimname (optional): vertical dimension name, default 'z'
        - z_varname (optional): vertical coordinate name
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
    x_varname = config['x_varname']
    y_varname = config['y_varname']
    z_varname = config.get('z_varname', z_dimname)
    lon_varname = config.get('lon_varname', x_varname)
    lat_varname = config.get('lat_varname', y_varname)
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
    
    # Read input file
    ds = xr.open_dataset(input_filename)
    
    # Handle special WRF case: rename Time -> time, handle XTIME
    if 'Time' in ds.dims and time_dimname == 'time':
        if 'XTIME' in ds.variables:
            ds = ds.reset_coords(names='XTIME', drop=False)
        ds = ds.rename({'Time': time_dimname})
    
    # Get time coordinate
    if time_coordname in ds.variables:
        time_coords = ds[time_coordname]
        # Round to nearest second for consistency
        if hasattr(time_coords.dt, 'round'):
            time_coords = time_coords.dt.round('s')
    else:
        time_coords = ds[time_dimname]
        if hasattr(time_coords.dt, 'round'):
            time_coords = time_coords.dt.round('s')
    
    # Get reflectivity data
    dbz = ds[reflectivity_varname].squeeze()
    
    # Auto-detect if data is 3D
    if is_3d is None:
        is_3d = z_dimname in dbz.dims
    
    # Get spatial dimensions
    nx = ds.sizes[x_dimname]
    ny = ds.sizes[y_dimname]
    
    # Get x, y coordinates
    if x_varname in ds.variables:
        x_coords = ds[x_varname].data
    else:
        # Create x coordinate from dimension using dx from config
        x_coords = np.arange(0, nx) * dx
    
    if y_varname in ds.variables:
        y_coords = ds[y_varname].data
    else:
        # Create y coordinate from dimension using dy from config
        y_coords = np.arange(0, ny) * dy
    
    # Get grid lat/lon (can be 1D or 2D)
    if lon_varname in ds.variables:
        grid_lon = ds[lon_varname].squeeze()
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
    
    if lat_varname in ds.variables:
        grid_lat = ds[lat_varname].squeeze()
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
        if z_varname in ds.variables:
            height = ds[z_varname].data
        else:
            # Special case for WRF geopotential height
            if 'PH' in ds.variables and 'PHB' in ds.variables:
                height = ((ds['PH'] + ds['PHB']).squeeze().data / 9.80665)
            else:
                logger.warning(f"Vertical coordinate {z_varname} not found, using index")
                height = np.arange(ds.sizes[z_dimname])
        
        # Auto-detect vertical coordinate type if not specified
        if z_coord_type is None:
            # Check if coordinate is monotonically increasing or decreasing
            if z_varname in ds.variables:
                z_vals = ds[z_varname].values
                # For height: typically increases from surface (low) to top (high)
                # For pressure: typically decreases from surface (high ~1000 hPa) to top (low ~0.1 hPa)
                if z_vals[0] > z_vals[-1]:
                    z_coord_type = 'pressure'
                    logger.info(f"Auto-detected vertical coordinate type: pressure (decreasing from {z_vals[0]:.2f} to {z_vals[-1]:.2f})")
                else:
                    z_coord_type = 'height'
                    logger.info(f"Auto-detected vertical coordinate type: height (increasing from {z_vals[0]:.2f} to {z_vals[-1]:.2f})")
            else:
                # Default to height if cannot determine
                z_coord_type = 'height'
                logger.warning("Cannot auto-detect vertical coordinate type, defaulting to 'height'")
        else:
            logger.info(f"Using specified vertical coordinate type: {z_coord_type}")
        
        # Handle height coordinate transformation (AGL to MSL if needed)
        if radar_alt_varname in ds.variables:
            z_agl = ds[z_dimname] + radar_alt
            ds[z_dimname] = z_agl
        
        # Get or create surface elevation
        if terrain_file is not None:
            dster = xr.open_dataset(terrain_file)
            dster = dster.assign_coords({
                y_dimname: ds[y_varname], 
                x_dimname: ds[x_varname]
            })
            sfc_elev = dster[elev_varname]
            mask_goodvalues = dster[rangemask_varname].data.astype(int)
        else:
            # Create default surface elevation/pressure
            if z_coord_type == 'pressure':
                # For pressure coordinates, use configured or default sea level pressure
                sfc_elev = xr.DataArray(
                    np.full((ny, nx), default_sfc_pressure),
                    coords={y_dimname: y_coords, x_dimname: x_coords},
                    dims=(y_dimname, x_dimname)
                )
                logger.info(f"No terrain file specified. Using default surface pressure: {default_sfc_pressure} hPa")
            else:
                # For height coordinates, use zero elevation
                sfc_elev = xr.DataArray(
                    np.zeros((ny, nx)),
                    coords={y_dimname: y_coords, x_dimname: x_coords},
                    dims=(y_dimname, x_dimname)
                )
            mask_goodvalues = np.full((ny, nx), 1, dtype=int)
        
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
            dbz3d_filt = dbz.where(ds[z_varname] < (sfc_elev - sfc_dz_min))
            
            # Calculate low-level composite reflectivity
            # Low-level is defined as pressure range from (sfc - sfc_dz_max) to (sfc - sfc_dz_min)
            # This captures levels close to but above the surface
            dbz3d_lowlevel = dbz.where(
                (ds[z_varname] >= (sfc_elev - sfc_dz_max)) & 
                (ds[z_varname] <= (sfc_elev - sfc_dz_min))
            )
        else:
            # For height coordinates: surface is low, top is high (traditional case)
            # Filter reflectivity below surface + minimum height
            dbz3d_filt = dbz.where(ds[z_varname] > (sfc_elev + sfc_dz_min))
            
            # Calculate low-level composite reflectivity
            dbz3d_lowlevel = dbz.where(
                (ds[z_varname] >= sfc_dz_min) & (ds[z_varname] <= sfc_dz_max)
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
        dbz_comp = dbz
        dbz_lowlevel = dbz.copy()
        # Create fake 3D array for compatibility
        dbz3d_filt = dbz.expand_dims(z_dimname, axis=0)
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
    }
    
    return comp_dict

#--------------------------------------------------------------------------------
def get_composite_reflectivity_wrf(input_filename, config):
    """
    Get composite reflectivity from WRF.

    Args:
        input_filename: string
            Input data filename
        config: dictionary
            Dictionary containing config parameters

    Returns:
        comp_dict: dictionary
            Dictionary containing output variables
    """
    sfc_dz_min = config['sfc_dz_min']
    sfc_dz_max = config['sfc_dz_max']
    radar_sensitivity = config['radar_sensitivity']
    time_dimname = config.get('time', 'time')
    x_dimname = config.get('x_dimname', 'x')
    y_dimname = config.get('y_dimname', 'y')
    z_dimname = config.get('z_dimname', 'z')
    x_varname = config['x_varname']
    y_varname = config['y_varname']
    # z_varname = config['z_varname']
    reflectivity_varname = config['reflectivity_varname']
    fillval = config['fillval']

    # Read WRF file
    ds = xr.open_dataset(input_filename)
    # Drop XTIME dimension, and rename 'Time' dimension to 'time'
    ds = ds.reset_coords(names='XTIME', drop=False).rename({'Time': time_dimname})
    # Rounds up to second, some model converted datetimes do not contain round second
    time_coords = ds.XTIME.dt.round('S')
    # Get data coordinates and dimensions
    # Get WRF height values
    height = (ds['PH'] + ds['PHB']).squeeze().data / 9.80665
    nx = ds.sizes[x_dimname]
    ny = ds.sizes[y_dimname]
    # nz = ds.sizes[z_dimname]
    # Create x, y coordinates to mimic radar
    dx = ds.attrs['DX']
    dy = ds.attrs['DY']
    x_coords = np.arange(0, nx) * dx
    y_coords = np.arange(0, ny) * dy
    # Create a fake radar lat/lon
    radar_lon, radar_lat = 0, 0
    # Convert to DataArray
    radar_lon = xr.DataArray(radar_lon, attrs={'long_name': 'Radar longitude'})
    radar_lat = xr.DataArray(radar_lat, attrs={'long_name': 'Radar latitude'})
    grid_lon = ds[x_varname].squeeze()
    grid_lat = ds[y_varname].squeeze()
    # Get radar variables
    dbz3d = ds[reflectivity_varname].squeeze()
    dbz3d_filt = dbz3d
    # Get composite reflectivity
    dbz_comp = dbz3d_filt.max(dim=z_dimname)
    # Get low-level reflectivity
    dbz_lowlevel = dbz3d_filt.isel(bottom_top=1)
    # Make a copy of the composite reflectivity (must do this or the dbz_comp will be altered)
    refl = np.copy(dbz_comp.values)
    # Replace all values less than min radar sensitivity, including NAN, to be equal to the sensitivity value
    # The purpose is to include areas surrounding isolated cells below radar sensitivity
    # in the background intensity calculation.
    # This differs from Steiner.
    refl[(refl < radar_sensitivity) | np.isnan(refl)] = radar_sensitivity
    # Create a good value mask (everywhere is good for WRF)
    # dster = xr.open_dataset(terrain_file)
    # mask_goodvalues = dster.mask110.values.astype(int)
    mask_goodvalues = np.full(refl.shape, 1, dtype=int)

    # Put output variables in a dictionary
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
    }
    return comp_dict

#--------------------------------------------------------------------------------
def get_composite_reflectivity_wrf_regrid(input_filename, config):
    """
    Get composite reflectivity from regridded WRF.

    Args:
        input_filename: string
            Input data filename
        config: dictionary
            Dictionary containing config parameters

    Returns:
        comp_dict: dictionary
            Dictionary containing output variables
    """
    sfc_dz_min = config['sfc_dz_min']
    sfc_dz_max = config['sfc_dz_max']
    radar_sensitivity = config['radar_sensitivity']
    time_dimname = config.get('time_dimname', 'time')
    x_dimname = config.get('x_dimname', 'x')
    y_dimname = config.get('y_dimname', 'y')
    z_dimname = config.get('z_dimname', 'z')
    x_varname = config['x_varname']
    y_varname = config['y_varname']
    z_varname = config['z_varname']
    reflectivity_varname = config['reflectivity_varname']
    composite_reflectivity_varname = config.get('composite_reflectivity_varname', '')
    fillval = config['fillval']

    # Read WRF file
    ds = xr.open_dataset(input_filename)
    # Drop XTIME dimension, and rename 'Time' dimension to 'time'
    # ds = ds.reset_coords(names='XTIME', drop=False).rename({'Time': time_dimname})
    # Rounds up to second, some model converted datetimes do not contain round second
    time_coords = ds[time_dimname].dt.round('S')
    # Get data coordinates and dimensions
    # Get WRF height values
    height = ds[z_dimname].data
    nx = ds.sizes[x_dimname]
    ny = ds.sizes[y_dimname]
    # Create x, y coordinates to mimic radar
    dx = ds.attrs['DX']
    dy = ds.attrs['DY']
    x_coords = np.arange(0, nx) * dx
    y_coords = np.arange(0, ny) * dy
    # Create a fake radar lat/lon
    radar_lon, radar_lat = 0, 0
    # Convert to DataArray
    radar_lon = xr.DataArray(radar_lon, attrs={'long_name': 'Radar longitude'})
    radar_lat = xr.DataArray(radar_lat, attrs={'long_name': 'Radar latitude'})
    grid_lon = ds[x_varname].squeeze()
    grid_lat = ds[y_varname].squeeze()

    # Get radar variables
    dbz3d = ds[reflectivity_varname].squeeze()
    dbz3d_filt = dbz3d
    # Get composite reflectivity
    if composite_reflectivity_varname != '':
        dbz_comp = ds[composite_reflectivity_varname].squeeze()
    else:
        dbz_comp = dbz3d_filt.max(dim=z_dimname)

    # Filter reflectivity outside the low-level
    dbz3d_lowlevel = dbz3d.where((ds[z_varname] >= sfc_dz_min) & (ds[z_varname] <= sfc_dz_max))
    # Get composite reflectivity
    dbz_comp = dbz3d_filt.max(dim=z_dimname)
    # Get low-level composite reflectivity
    dbz_lowlevel = dbz3d_lowlevel.max(dim=z_dimname)

    # Make a copy of the composite reflectivity (must do this or the dbz_comp will be altered)
    refl = np.copy(dbz_comp.data)
    # Replace all values less than min radar sensitivity, including NAN, to be equal to the sensitivity value
    # The purpose is to include areas surrounding isolated cells below radar sensitivity
    # in the background intensity calculation.
    # This differs from Steiner.
    refl[(refl < radar_sensitivity) | np.isnan(refl)] = radar_sensitivity
    # Create a good value mask (everywhere is good for WRF)
    # dster = xr.open_dataset(terrain_file)
    # mask_goodvalues = dster.mask110.values.astype(int)
    mask_goodvalues = np.full(refl.shape, 1, dtype=int)

    # Put output variables in a dictionary
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
    }
    return comp_dict

#--------------------------------------------------------------------------------
def get_composite_reflectivity_wrf_composite(input_filename, config):
    """
    Get composite reflectivity from WRF composite reflectivity.

    Args:
        input_filename: string
            Input data filename
        config: dictionary
            Dictionary containing config parameters

    Returns:
        comp_dict: dictionary
            Dictionary containing output variables
    """
    radar_sensitivity = config['radar_sensitivity']
    time_dimname = config.get('time_dimname', 'time')
    x_dimname = config.get('x_dimname', 'x')
    y_dimname = config.get('y_dimname', 'y')
    z_dimname = config.get('z_dimname', 'z')
    x_varname = config['x_varname']
    y_varname = config['y_varname']
    # z_varname = config['z_varname']
    reflectivity_varname = config['reflectivity_varname']
    pass_varname = config.get("pass_varname", None)

    # Read WRF file
    ds = xr.open_dataset(input_filename)
    # Get dimension names from the file
    dims_file = []
    for key in ds.dims: dims_file.append(key)
    # Find extra dimensions beyond [time, y, x]
    dims_keep = [time_dimname, y_dimname, x_dimname]
    dims_drop = list(set(dims_file) - set(dims_keep))
    # Drop extra dimensions, reorder to [time, y, x]
    ds = ds.drop_dims(dims_drop).transpose(time_dimname, y_dimname, x_dimname)
    # Rounds up to second, some model converted datetimes do not contain round second
    time_coords = ds[time_dimname].dt.round('S')
    # Get data coordinates and dimensions
    nx = ds.sizes[x_dimname]
    ny = ds.sizes[y_dimname]
    # Create x, y coordinates to mimic radar
    dx = ds.attrs['DX']
    dy = ds.attrs['DY']
    x_coords = np.arange(0, nx) * dx
    y_coords = np.arange(0, ny) * dy
    # Create a fake radar lat/lon
    radar_lon, radar_lat = 0, 0
    # Convert to DataArray
    radar_lon = xr.DataArray(radar_lon, attrs={'long_name': 'Radar longitude'})
    radar_lat = xr.DataArray(radar_lat, attrs={'long_name': 'Radar latitude'})
    grid_lon = ds[x_varname].squeeze()
    grid_lat = ds[y_varname].squeeze()

    # Get composite reflectivity
    dbz_comp = ds[reflectivity_varname].squeeze()
    # Get low-level reflectivity
    dbz_lowlevel = ds[reflectivity_varname].squeeze()
    # Replace all values with NaN
    dbz_lowlevel = dbz_lowlevel.where(dbz_lowlevel != dbz_lowlevel, np.nan)
    # Add a vertical dimension to make a fake 3D DataArray
    dbz3d = dbz_lowlevel.expand_dims(z_dimname, axis=0)
    # Make a fake height array
    height = np.zeros(1)

    # Make a copy of the composite reflectivity (must do this or the dbz_comp will be altered)
    refl = np.copy(dbz_comp.values)
    # Replace all values less than min radar sensitivity, including NAN, to be equal to the sensitivity value
    # The purpose is to include areas surrounding isolated cells below radar sensitivity
    # in the background intensity calculation.
    # This differs from Steiner.
    refl[(refl < radar_sensitivity) | np.isnan(refl)] = radar_sensitivity
    # Create a good value mask (everywhere is good for WRF)
    mask_goodvalues = np.full(refl.shape, 1, dtype=int)

    # Subset pass variable from input DataSet
    if pass_varname is not None:
        # Find the common variable names between the dataset and the list
        pass_varname = set(ds.data_vars) & set(pass_varname)
        # Subset the input dataset
        ds_pass = ds[pass_varname]
    else:
        ds_pass = None

    # Put output variables in a dictionary
    comp_dict = {
        'x_coords': x_coords,
        'y_coords': y_coords,
        'dbz3d_filt': dbz3d,
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
    }
    return comp_dict

#--------------------------------------------------------------------------------
def get_composite_reflectivity_radar(input_filename, config):
    """
    Get composite reflectivity from generic radar data.

    Args:
        input_filename: string
            Input data filename
        config: dictionary
            Dictionary containing config parameters

    Returns:
        comp_dict: dictionary
            Dictionary containing output variables
    """
    sfc_dz_min = config['sfc_dz_min']
    sfc_dz_max = config['sfc_dz_max']
    radar_sensitivity = config['radar_sensitivity']
    time_coordname = config.get('time_coordname', 'time')
    time_dimname = config.get('time', 'time')
    x_dimname = config.get('x_dimname', 'x')
    y_dimname = config.get('y_dimname', 'y')
    z_dimname = config.get('z_dimname', 'z')
    x_varname = config['x_varname']
    y_varname = config['y_varname']
    z_varname = config['z_varname']
    lon_varname = config['lon_varname']
    lat_varname = config['lat_varname']
    reflectivity_varname = config['reflectivity_varname']
    fillval = config['fillval']
    terrain_file = config.get('terrain_file', None)
    elev_varname = config.get('elev_varname', None)
    rangemask_varname = config.get('rangemask_varname', None)

    # Read radar file
    ds = xr.open_dataset(input_filename)
    # Get dimension names from the file
    dims_file = []
    for key in ds.dims: dims_file.append(key)
    # Find extra dimensions beyond [time, z, y, x]
    dims_keep = [time_dimname, z_dimname, y_dimname, x_dimname]
    dims_drop = list(set(dims_file) - set(dims_keep))
    # Drop extra dimensions, reorder to [time, z, y, x]
    ds = ds.drop_dims(dims_drop).transpose(time_dimname, z_dimname, y_dimname, x_dimname) 
    # Create time_coords
    time_coords = ds[time_coordname]
    # Get data coordinates and dimensions
    height = ds[z_dimname].squeeze().data
    y_coords = ds[y_varname].data
    x_coords = ds[x_varname].data
    # Below are variables produced by PyART gridding
    radar_lon = ds['origin_longitude']
    radar_lat = ds['origin_latitude']
    radar_alt = ds['alt']
    # Take the first vertical level for 3D lat/lon
    if z_dimname in ds[lon_varname].dims:
        grid_lon = ds[lon_varname].isel({z_dimname:0})        
        grid_lat = ds[lat_varname].isel({z_dimname:0})
    else:
        grid_lon = ds[lon_varname]
        grid_lat = ds[lat_varname]

    # Change radar height coordinate from AGL to MSL
    z_agl = ds[z_dimname] + radar_alt
    ds[z_dimname] = z_agl
    if terrain_file is not None:
        # Read terrain file
        dster = xr.open_dataset(terrain_file)
        # Assign coordinate from radar file to the terrain file so they have the same coordinates
        dster = dster.assign_coords({y_dimname: (ds[y_varname]), x_dimname: (ds[x_varname])})
        sfc_elev = dster[elev_varname]
        mask_goodvalues = dster[rangemask_varname].data.astype(int)
    else:
        # Create elevation array filled with 0
        nx = ds.sizes[x_dimname]
        ny = ds.sizes[y_dimname]
        sfc_elev = np.zeros((ny, nx), dtype=float)
        # Convert to DataArray, needed to use for masking 3D variables
        sfc_elev = xr.DataArray(sfc_elev, coords={y_dimname:y_coords, x_dimname:x_coords}, dims=(y_dimname, x_dimname))
        # Create a good value mask
        mask_goodvalues = np.full((ny, nx), 1, dtype=int)

    # Get radar variables
    dbz3d = ds[reflectivity_varname].squeeze()
    # ncp = ds['normalized_coherent_power'].squeeze()

    # Some combination of masks may be better to apply here to filter out bad signals
    # including clutter, second trip, low signal side lobes
    # but when this program was written, that optimal combination was not yet determined
    # and one needs to be careful not to remove good echoes
    # This NCP filter works well as a substitute
    # dbz3d = dbz3d.where(ncp >= 0.5)

    # Filter reflectivity below certain elevation height
    dbz3d_filt = dbz3d.where(ds[z_varname] > (sfc_elev + sfc_dz_min))
    # Filter reflectivity outside the low-level
    dbz3d_lowlevel = dbz3d.where((ds[z_varname] >= sfc_dz_min) & (ds[z_varname] <= sfc_dz_max))
    # Get composite reflectivity
    dbz_comp = dbz3d_filt.max(dim=z_dimname)
    # Get low-level composite reflectivity
    dbz_lowlevel = dbz3d_lowlevel.max(dim=z_dimname)
    # Make a copy of the composite reflectivity (must do this or the dbz_comp will be altered)
    refl = np.copy(dbz_comp.data)
    # Replace all values less than min radar sensitivity, including NAN, to be equal to the sensitivity value
    # The purpose is to include areas surrounding isolated cells below radar sensitivity in the background intensity calculation
    # This differs from Steiner.
    refl[(refl < radar_sensitivity) | np.isnan(refl)] = radar_sensitivity
    # Put output variables in a dictionary
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
        'time_coords':time_coords,
    }
    return comp_dict

#--------------------------------------------------------------------------------
def get_composite_reflectivity_csapr_cacti(input_filename, config):
    """
    Get composite reflectivity from CACTI CSAPR data.

    Args:
        input_filename: string
            Input data filename
        config: dictionary
            Dictionary containing config parameters

    Returns:
        comp_dict: dictionary
            Dictionary containing output variables
    """
    sfc_dz_min = config['sfc_dz_min']
    sfc_dz_max = config['sfc_dz_max']
    radar_sensitivity = config['radar_sensitivity']
    time_coordname = config.get('time_coordname', 'time')
    time_dimname = config.get('time', 'time')
    x_dimname = config.get('x_dimname', 'x')
    y_dimname = config.get('y_dimname', 'y')
    z_dimname = config.get('z_dimname', 'z')
    x_varname = config['x_varname']
    y_varname = config['y_varname']
    z_varname = config['z_varname']
    lon_varname = config['lon_varname']
    lat_varname = config['lat_varname']
    reflectivity_varname = config['reflectivity_varname']
    fillval = config['fillval']
    terrain_file = config.get('terrain_file', None)
    elev_varname = config.get('elev_varname', None)
    rangemask_varname = config.get('rangemask_varname', None)

    # Read radar file
    ds = xr.open_dataset(input_filename)
    # Get dimension names from the file
    dims_file = []
    for key in ds.dims: dims_file.append(key)
    # Find extra dimensions beyond [time, z, y, x]
    dims_keep = [time_dimname, z_dimname, y_dimname, x_dimname]
    dims_drop = list(set(dims_file) - set(dims_keep))
    # Drop extra dimensions, reorder to [time, z, y, x]
    ds = ds.drop_dims(dims_drop).transpose(time_dimname, z_dimname, y_dimname, x_dimname)
    # Create time_coords
    time_coords = ds[time_coordname]
    # Get data coordinates and dimensions
    height = ds[z_dimname].squeeze().data
    y_coords = ds[y_varname].data
    x_coords = ds[x_varname].data
    # Below are variables produced by PyART gridding
    radar_lon = ds['origin_longitude']
    radar_lat = ds['origin_latitude']
    radar_alt = ds['alt'].squeeze()
    # Take the first vertical level from 3D lat/lon
    grid_lon = ds[lon_varname].isel(z=0)
    grid_lat = ds[lat_varname].isel(z=0)

    # Change radar height coordinate from AGL to MSL
    z_agl = ds[z_dimname] + radar_alt
    ds[z_dimname] = z_agl

    # Read terrain file
    dster = xr.open_dataset(terrain_file)
    # Change terrain file dimension name to be consistent with radar file
    # dster = dster.rename({'latdim':y_dimname, 'londim':x_dimname})
    # Assign coordinate from radar file to the terrain file so they have the same coordinates
    dster = dster.assign_coords({y_dimname: (ds[y_varname]), x_dimname: (ds[x_varname])})
    sfc_elev = dster[elev_varname]
    # Create a good value mask
    # Use 110 km radius range mask as good value mask, make sure to convert boolean array to integer type
    # The mask_goodvalues is used in calculating background reflectivity using ndimage.convolve
    # The background_intensity function needs to calculate the number of points within a circular radius,
    # then divide the convolved (averarged) reflectivity with by the number of points to get the background reflectivity
    # The mask_goodvalues must be 0 or 1 for that to work
    mask_goodvalues = dster[rangemask_varname].data.astype(int)

    # Get radar variables
    dbz3d = ds[reflectivity_varname].squeeze()
    ncp = ds['normalized_coherent_power'].squeeze()

    # Some combination of masks may be better to apply here to filter out bad signals
    # including clutter, second trip, low signal side lobes
    # but when this program was written, that optimal combination was not yet determined
    # and one needs to be careful not to remove good echoes
    # This NCP filter works well as a substitute
    dbz3d = dbz3d.where(ncp >= 0.5)
    # Filter reflectivity below certain elevation height
    dbz3d_filt = dbz3d.where(ds[z_varname] > (sfc_elev + sfc_dz_min))
    # Filter reflectivity outside the low-level
    dbz3d_lowlevel = dbz3d.where((ds[z_varname] >= sfc_dz_min) & (ds[z_varname] <= sfc_dz_max))
    # Get composite reflectivity
    dbz_comp = dbz3d_filt.max(dim=z_dimname)
    # Get low-level composite reflectivity
    dbz_lowlevel = dbz3d_lowlevel.max(dim=z_dimname)
    # Make a copy of the composite reflectivity (must do this or the dbz_comp will be altered)
    refl = np.copy(dbz_comp.data)
    # Replace all values less than min radar sensitivity, including NAN, to be equal to the sensitivity value
    # The purpose is to include areas surrounding isolated cells below radar sensitivity in the background intensity calculation
    # This differs from Steiner.
    refl[(refl < radar_sensitivity) | np.isnan(refl)] = radar_sensitivity

    # Put output variables in a dictionary
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
        'time_coords':time_coords,
    }
    return comp_dict