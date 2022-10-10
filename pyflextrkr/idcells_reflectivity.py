import sys
import numpy as np
import xarray as xr
import logging
from datetime import datetime
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
    min_corearea = config['min_corearea']
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

    # Get composite reflectivity
    if input_source == 'radar':
        comp_dict = get_composite_reflectivity_radar(
            input_filename, config)
    elif input_source == 'csapr_cacti':
        comp_dict = get_composite_reflectivity_csapr_cacti(
            input_filename, config)
    elif input_source == 'wrf':
        comp_dict = get_composite_reflectivity_wrf(
            input_filename, config)
    elif input_source == 'wrf_regrid':
        comp_dict = get_composite_reflectivity_wrf_regrid(
            input_filename, config)
    else:
        logger.error(f'Unknown input_source: {input_source}')
        sys.exit()

    # Subset domain
    if (geolimits is not None):
        comp_dict = subset_domain(comp_dict, geolimits, dx, dy)

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
            remove_smallcores=True,
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
            remove_smallcores=True,
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
    timestamp = time_coords[0]
    # Convert to basetime (i.e., Epoch time)
    # This is a more flexible way that can handle non-standard 360 day calendar
    # file_basetime = np.array([(np.datetime64(timestamp).item() - datetime(1970,1,1,0,0,0)).total_seconds()])
    file_basetime = time_coords[0].values.tolist() / 1e9
    # Convert to strings
    file_datestring = timestamp.dt.strftime("%Y%m%d").item()
    file_timestring = timestamp.dt.strftime("%H%M").item()
    # file_datestring = timestamp.strftime("%Y%m%d")
    # file_timestring = timestamp.strftime("%H%M")
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
        )
    logger.info(f"{cloudid_outfile}")

    return cloudid_outfile

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
        # Make a 2D mask
        mask = ((grid_lon >= lonmin) & (grid_lon <= lonmax) & \
                (grid_lat >= latmin) & (grid_lat <= latmax)).squeeze()
        # Get y/x indices limits from the mask
        y_idx, x_idx = np.where(mask == True)
        xmin, xmax = np.min(x_idx), np.max(x_idx)
        ymin, ymax = np.min(y_idx), np.max(y_idx)
        # Subset variables
        dbz3d_filt = dbz3d_filt[:, ymin:ymax+1, xmin:xmax+1]
        dbz_comp = dbz_comp[ymin:ymax+1, xmin:xmax+1]
        dbz_lowlevel = dbz_lowlevel[ymin:ymax+1, xmin:xmax+1]
        mask_goodvalues = mask_goodvalues[ymin:ymax+1, xmin:xmax+1]
        refl = refl[ymin:ymax+1, xmin:xmax+1]
        grid_lon = grid_lon[ymin:ymax+1, xmin:xmax+1]
        grid_lat = grid_lat[ymin:ymax+1, xmin:xmax+1]
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
    # time_coords = ds.indexes['time']
    time_coords = ds[time_dimname]
    # Get data coordinates and dimensions
    height = ds[z_dimname].squeeze().data
    y_coords = ds[y_varname].data
    x_coords = ds[x_varname].data
    # Below are variables produced by PyART gridding
    radar_lon = ds['origin_longitude']
    radar_lat = ds['origin_latitude']
    radar_alt = ds['alt']
    # Take the first vertical level from 3D lat/lon
    grid_lon = ds[lon_varname].isel(z=0)
    grid_lat = ds[lat_varname].isel(z=0)

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
    # Reorder the dimensions using dimension names to [time, z, y, x]
    ds = ds.transpose(time_dimname, z_dimname, y_dimname, x_dimname)
    # Create time_coords
    time_coords = ds[time_dimname]
    # Get data coordinates and dimensions
    height = ds[z_dimname].squeeze().data
    y_coords = ds[y_varname].data
    x_coords = ds[x_varname].data
    # Below are variables produced by PyART gridding
    radar_lon = ds['origin_longitude']
    radar_lat = ds['origin_latitude']
    radar_alt = ds['alt']
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