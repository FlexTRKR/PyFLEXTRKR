import os
import numpy as np
import time
import xarray as xr
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
    Identifies convective cells using reflectivity.

    Args:
        input_filename:
        config:

    Returns:

    """
    feature_varname = config.get("feature_varname", "feature_number")
    nfeature_varname = config.get("nfeature_varname", "nfeatures")
    featuresize_varname = config.get("featuresize_varname", "npix_feature")

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
    radar_sensitivity = config['radar_sensitivity']
    # types_steiner = config['types_steiner']
    return_diag = config['return_diag']
    dx = config['dx']
    dy = config['dy']
    time_dimname = config.get("time", "time")
    x_dimname = config.get("x_dimname", "x")
    y_dimname = config.get("y_dimname", "y")
    z_dimname = config.get("z_dimname", "z")
    x_varname = config['x_varname']
    y_varname = config['y_varname']
    # z_varname = config['z_varname']
    reflectivity_varname = config['reflectivity_varname']
    iradar = config['iradar']
    iwrf = config['iwrf']

    types_powell = {'NO_ECHO': 1, 'WEAK_ECHO': 2, 'STRATIFORM': 3, 'CONVECTIVE': 4, 'ISO_CONV_CORE': 5,
                    'ISO_CONV_FRINGE': 6, 'UNCERTAIN': 7, 'CS_CORE': 8, 'ISO_CS_CORE': 9}
    types_steiner = {'NO_SURF_ECHO': 1, 'WEAK_ECHO': 2, 'STRATIFORM': 3, 'CONVECTIVE': 4}

    np.set_printoptions(threshold=np.inf)
    logger = logging.getLogger(__name__)

    fillval = config["fillval"]

    # From radar data
    if iradar == 1:

        # Read radar file
        ds = xr.open_dataset(input_filename)

    # From WRF data
    elif iwrf == 1:

        # Read WRF file
        ds = xr.open_dataset(input_filename)
        # Drop XTIME dimension, and rename 'Time' dimension to 'time'
        ds = ds.reset_coords(names='XTIME', drop=False).rename({'Time': 'time'})
        # Rounds up to second, some model converted datetimes do not contain round second
        time_coords = ds.XTIME.dt.round('S')
        # out_ftime = time_coords.dt.strftime("%Y%m%d.%H%M%S").item()

        # Get data coordinates and dimensions
        # height = ds[z_varname].squeeze().values
        height = (ds['PH'] + ds['PHB']).squeeze().data / 9.80665
        nx = ds.sizes[x_dimname]
        ny = ds.sizes[y_dimname]
        # nz = ds.sizes[z_dimname]
        # Create x, y coordinates
        x_coords = np.arange(0, nx) * dx
        y_coords = np.arange(0, ny) * dy
        # Create a fake radar lat/lon
        radar_lon, radar_lat = 0, 0
        # Convert to DataArray
        # x_coords = xr.DataArray(x_coords, dims=['x'], attrs={'long_name': 'x distance', 'units': 'm'})
        # y_coords = xr.DataArray(y_coords, dims=['y'], attrs={'long_name': 'y distance', 'units': 'm'})
        radar_lon = xr.DataArray(radar_lon, attrs={'long_name': 'Radar longitude'})
        radar_lat = xr.DataArray(radar_lat, attrs={'long_name': 'Radar latitude'})
        grid_lon = ds[x_varname].squeeze()
        grid_lat = ds[y_varname].squeeze()
        out_x_dimname = 'x'
        out_y_dimname = 'y'

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
        # The purpose is to include areas surrounding isolated cells below radar sensitivity in the background intensity calculation
        # This differs from Steiner.
        refl[(refl < radar_sensitivity) | np.isnan(refl)] = radar_sensitivity

        # Create a good value mask (everywhere is good for WRF)
        # dster = xr.open_dataset(terrain_file)
        # mask_goodvalues = dster.mask110.values.astype(int)
        mask_goodvalues = np.full(refl.shape, 1, dtype=np.int8)

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
        )

    # Expand convective cores outward to a set of radii to
    # make the convective region larger for better tracking convective cells
    core_expand, core_sorted = expand_conv_core(core_dilate, radii_expand, dx, dy, min_corenpix=0)

    # Calculate echo-top heights for various reflectivity thresholds
    shape_2d = refl.shape
    if (iradar == 1):
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
    elif (iwrf == 1):
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
    del dbz3d, dbz3d_filt

    # Put all Steiner parameters in a dictionary
    steiner_params = {
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

    # import matplotlib.pyplot as plt
    # plt.pcolormesh(core_sorted, cmap='jet')
    # plt.colorbar()
    # plt.show()

    # Create variables for tracking
    feature_mask = core_expand
    # Count number of pixels for each feature
    unique_num, npix_feature = np.unique(feature_mask, return_counts=True)
    # Remove background (unique_num = 0)
    npix_feature = npix_feature[(unique_num > 0)]
    # Get number of features
    nfeatures = np.nanmax(feature_mask)

    # Get date/time and make output filename
    file_basetime = time_coords[0].values.tolist() / 1e9
    file_datestring = time_coords.dt.strftime("%Y%m%d").item()
    file_timestring = time_coords.dt.strftime("%H%M").item()
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
            dx,
            dy,
            radar_lon,
            radar_lat,
            grid_lon,
            grid_lat,
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
            dx,
            dy,
            radar_lon,
            radar_lat,
            grid_lon,
            grid_lat,
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

    import pdb; pdb.set_trace()
    return