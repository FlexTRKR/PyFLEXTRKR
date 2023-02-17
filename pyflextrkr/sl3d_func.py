import numpy as np
import math
from scipy import ndimage
import warnings
from pyflextrkr.echotop_func import echotop_height

def run_sl3d(ds, config):
    """
    Process 3D radar dataset to get SL3D variables

    Args:
        ds: Xarray Dataset
            Dataset object
        config: dictionary
            Dictionary containing config parameters

    Returns:
        data_dict: dictionary
            Dictionary containing output variables
    """

    x_dimname = config['x_dimname']
    y_dimname = config['y_dimname']
    z_dimname = config['z_dimname']
    x_coordname = config['x_coordname']
    y_coordname = config['y_coordname']
    z_coordname = config['z_coordname']
    reflectivity_varname = config['reflectivity_varname']
    meltlevel_varname = config['meltlevel_varname']
    echotop_gap = config.get('echotop_gap', 0)
    dbz_lowlevel_asl = config.get('dbz_lowlevel_asl', 2.0)
    fillval = config.get('fillval', -9999)

    # Get data dimensions
    nx = ds.sizes[x_dimname]
    ny = ds.sizes[y_dimname]
    nz = ds.sizes[z_dimname]
    # Get data coordinates
    lon2d = ds[x_coordname].data
    lat2d = ds[y_coordname].data
    height = ds[z_coordname].data
    # Get data time
    Analysis_time = ds['time'].dt.strftime('%Y-%m-%dT%H:%M:%S').item()
    Analysis_month = ds['time'].dt.strftime('%m').item()
    # Get data variables
    refl3d = ds[reflectivity_varname].squeeze()
    reflArray = refl3d.data
    meltinglevelheight = ds[meltlevel_varname].squeeze().data
    # Make variables to mimic GridRad data
    # Nradobs = np.full(reflArray.shape, 10, dtype=int)
    # Nradecho = np.full(reflArray.shape, 10, dtype=int)

    # Replace NaN in melting level height with -2. (consistent with IDL version)
    # TODO: check if this is necessary
    meltinglevelheight[np.isnan(meltinglevelheight)] = -2.

    # Get low-level reflectivity
    idx_low = np.argmin(np.abs(height - dbz_lowlevel_asl))
    dbz_lowlevel = reflArray[idx_low,:,:].data
    # Get column-maximum reflectivity (composite)
    dbz_comp = refl3d.max(dim=z_dimname).data

    # Replace fillval with NaN
    reflArray[reflArray == fillval] = np.NaN

    x = {
        'values': lon2d,
        'n': nx,
    }
    y = {
        'values': lat2d,
        'n': ny,
    }
    z = {
        'values': height,
        'n': nz,
    }
    Z_H = {
        'values': reflArray,
        'missing': np.NaN,
    }
    data = {
        'x': x,
        'y': y,
        'z': z,
        # 'nobs': Nradobs,
        # 'necho': Nradecho,
        'Z_H': Z_H,
        'Analysis_month': Analysis_month,
    }

    # Perform SL3D classification
    sl3d = gridrad_sl3d(data, config, zmelt=meltinglevelheight)

    # Calculate echo-top heights for various reflectivity thresholds
    shape_2d = sl3d.shape
    echotop10 = echotop_height(refl3d, height, z_dimname, shape_2d,
                                dbz_thresh=10, gap=echotop_gap, min_thick=0)
    echotop20 = echotop_height(refl3d, height, z_dimname, shape_2d,
                                dbz_thresh=20, gap=echotop_gap, min_thick=0)
    echotop30 = echotop_height(refl3d, height, z_dimname, shape_2d,
                                dbz_thresh=30, gap=echotop_gap, min_thick=0)
    echotop40 = echotop_height(refl3d, height, z_dimname, shape_2d,
                                dbz_thresh=40, gap=echotop_gap, min_thick=0)
    echotop45 = echotop_height(refl3d, height, z_dimname, shape_2d,
                                dbz_thresh=45, gap=echotop_gap, min_thick=0)
    echotop50 = echotop_height(refl3d, height, z_dimname, shape_2d,
                                dbz_thresh=50, gap=echotop_gap, min_thick=0)

    # Put variables in dictionary
    data_dict= {
        'reflectivity_lowlevel': dbz_lowlevel,
        'reflectivity_comp': dbz_comp,
        'sl3d': sl3d,
        'echotop10': echotop10,
        'echotop20': echotop20,
        'echotop30': echotop30,
        'echotop40': echotop40,
        'echotop45': echotop45,
        'echotop50': echotop50,
    }
    # Variable attributes
    attrs_dict = {
        'reflectivity_lowlevel': {
            'long_name': f'Low-level reflectivity ({dbz_lowlevel_asl:.1f} km)',
            'units': 'dBZ',
            '_FillValue': np.NaN,
        },
        'reflectivity_comp': {
            'long_name': 'Composite (column maximum) reflectivity',
            'units': 'dBZ',
            '_FillValue': np.NaN,
        },
        'sl3d': {
            'long_name': 'SL3D classification category',
            'units': 'unitless',
            'comments': '0:NoEcho, 1:ConvectiveUpdraft, 2:Convection, ' + \
                        '3:PrecipitatingStratiform, 4:Non-PrecipitatingStratiform, 5:Anvil',
            '_FillValue': 0,
        },
        'echotop10': {
            'long_name': '10 dBZ echo top height',
            'units': 'km',
            '_FillValue': np.NaN,
        },
        'echotop20': {
            'long_name': '20 dBZ echo top height',
            'units': 'km',
            '_FillValue': np.NaN,
        },
        'echotop30': {
            'long_name': '30 dBZ echo top height',
            'units': 'km',
            '_FillValue': np.NaN,
        },
        'echotop40': {
            'long_name': '40 dBZ echo top height',
            'units': 'km',
            '_FillValue': np.NaN,
        },
        'echotop45': {
            'long_name': '45 dBZ echo top height',
            'units': 'km',
            '_FillValue': np.NaN,
        },
        'echotop50': {
            'long_name': '50 dBZ echo top height',
            'units': 'km',
            '_FillValue': np.NaN,
        },
    }
    return data_dict, attrs_dict

#-----------------------------------------------------------------------------------------
def gridrad_sl3d(data, config, **kwargs):
    """
    Classify radar echoes using the Storm Labeling in 3-D (SL3D) algorithm

    SL3D classification types:
    0 - No echo,
    1 - Convective Updraft, 2 - Convection, 3 - Precipitating Stratiform, 
    4 - Non-Precipitating Stratiform, 5 - Anvil

    Args:
        data: dictionary
            Dictionary containing input variables
        config: dictionary
            Dictionary containing config parameters
        zmelt: float, or np.array (1D or 2D), optional
            Melting level height

    Returns:
        sl3dclass: np.array
            SL3D classification types
    """

    # Get thresholds from config
    # Default values (if not supplied in config) are from original SL3D code 
    # Origin SL3D codes are provided by Cameron R. Homeyer (chomeyer@ou.edu)
    # Adaptation to Python was done by Jianfeng Li (jianfeng.li@pnnl.gov) and Zhe Feng (zhe.feng@pnnl.gov)

    # Radar data source
    radardatasource = config.get('radardatasource', None)
    # Background box size to calculate peakedness [km]
    background_Box = config.get('background_Box', 12.)
    # Reflectivity threshold to fill low-level coverage gap [dBZ]
    ReflThresh_lowlevel_gap = config.get('ReflThresh_lowlevel_gap', 20.)
    # Stratiform rain reflectivity threshold at 3 km ASL [dBZ]
    strat_EchoThresh_3km = config.get('strat_EchoThresh_3km', 20.)
    # Stratiform rain reflectivity threshold below 3 km ASL [dBZ]
    strat_EchoThresh_lt3km = config.get('strat_EchoThresh_lt3km', 10.0)
    # Column-mean reflectivitiy peakedness fraction threshold to be convective
    col_peakedness_frac = config.get('col_peakedness_frac', 0.3)
    # Above melting level reflectivity threshold [dBZ] to be convective
    abs_ConvThres_aml = config.get('abs_ConvThres_aml', 45.)
    # 25 dBZ echo-top height threshold [km] to be convective
    etop25dBZ_Thresh = config.get('etop25dBZ_Thresh', 10.0)
    # Composite reflectivity threshold [dBZ] for neighbor points to be convective
    neighbor_CompReflThresh = config.get('neighbor_CompReflThresh', 25.)
    # Reflectivity vertical gradient (low - up) threshold [dB] to be updraft
    updraft_ReflGradiant_Thresh = config.get('updraft_ReflGradiant_Thresh', 8.0)
    # Max height [km] to include reflectivity vertical gradient to be updraft
    updraft_ReflGradiant_MaxHeight = config.get('updraft_ReflGradiant_MaxHeight', 7.0)
    # Composite reflectivity threshold [dBZ] to be updraft
    updraft_CompRefl_Thresh = config.get('updraft_CompRefl_Thresh', 40.0)

    # Extract dimension sizes for ease
    nx = data['x']['n']
    ny = data['y']['n']
    nz = data['z']['n']

    # Estimate the number of grid points based on radar data source
    if (radardatasource == 'wrf'):
        # WRF has fixed grid spacing
        # Simply divide background_Box [km] by pixel_radius to get number of grid points
        pixel_radius = config.get('pixel_radius')
        ngrids = int(background_Box / pixel_radius)

    if (radardatasource == 'gridrad'):
        # Get composite grid spacing (in degrees)
        # Assumes lat, lon coordinates (data['y'], data['x']) are 2D [y, x]
        # For x dimension, take [0,:] to get 1D lon values
        # For y dimension, take [:,0] to get 1D lat values
        dx = (data['x']['values'][0,:])[1] - (data['x']['values'][0,:])[0]
        # Compute latitude mid-point of grid
        ymid = 0.5 * (data['y']['values'][:,0])[ny-1] + 0.5 * (data['y']['values'][:,0])[0]
        # Get approximate number of grid points equivalent to 12 km grid spacing
        ngrids = int(0.108 / (dx * math.cos(math.radians(ymid))))

    # Compute number of points in the box
    nsearch = 2 * ngrids + 1

    # Convert y, z to 3D arrays [z, y, x]
    # yyy is only used if no melting level height is provided
    if data['y']['values'].ndim == 1:
        yyy = data['y']['values'].reshape(1, ny, 1).repeat(nz,axis=0).repeat(nx, axis=2)
    if data['y']['values'].ndim == 2:
        yyy = np.repeat(data['y']['values'][np.newaxis, :, :], nz, axis=0)
    zzz = data['z']['values'].reshape(nz, 1, 1).repeat(ny,axis=1).repeat(nx, axis=2)

    # Find index of 3, 4, 5, and 9 km altitude
    if (data['z']['values'][0] > 3.0) : k3km = -1
    if (data['z']['values'][0] > 4.0) : k4km = -1
    if (data['z']['values'][0] > 5.0) : k5km = -1
    if (data['z']['values'][0] > 9.0) : k9km = -1

    if (data['z']['values'][nz-1] <= 3.0) : k3km = nz-1
    if (data['z']['values'][nz-1] <= 4.0) : k4km = nz-1
    if (data['z']['values'][nz-1] <= 5.0) : k5km = nz-1
    if (data['z']['values'][nz-1] <= 9.0) : k9km = nz-1

    for zindex in range (0, nz-1):
        if (data['z']['values'][zindex] <= 3.0 and data['z']['values'][zindex+1] > 3.0): k3km = zindex
        if (data['z']['values'][zindex] <= 4.0 and data['z']['values'][zindex+1] > 4.0): k4km = zindex
        if (data['z']['values'][zindex] <= 5.0 and data['z']['values'][zindex+1] > 5.0): k5km = zindex
        if (data['z']['values'][zindex] <= 9.0 and data['z']['values'][zindex+1] > 9.0): k9km = zindex

    # Set coefficients for 50th percentile melting level climatology in the U.S.
    a = [7.072, 7.896, 8.558, 7.988, 7.464, 6.728, 6.080, 6.270, 6.786, 8.670, 8.892, 7.936]
    b = [-0.124,-0.152,-0.160,-0.128,-0.100,-0.065, -0.039,-0.044,-0.067,-0.137,-0.160,-0.147]

    # Extract file month from GridRad data structure
    month = int(data['Analysis_month'])

    if ('zmelt' not in kwargs):
        # If no melting level provided, compute expected melting level for domain based on climatology
        zml = a[month-1] + b[month-1]*yyy
    else:
        zmelt = kwargs['zmelt']
        # For single value melting level
        if (zmelt.size == 1):
            if (zmelt > 0.0):
                # Copy constant melting level to three dimensions
                zml = np.full((nz, ny, nx), zmelt)
            else:
                # Else, revert to melting level climatology
                zml = a[month-1] + b[month-1]*yyy
        # For 2D melting level
        else:
            if (zmelt.ndim == 2):
                # Copy 2-D melting level to 3-D
                zml = zmelt.reshape(1, ny, nx).repeat(nz, axis=0)
            else:
                # Else, revert to melting level climatology
                zml = a[month-1] + b[month-1]*yyy


    # Find points with high reflectivity at 4 km and no echo at 3 km (potential gaps in coverage)
    ifix = (~np.isfinite(data['Z_H']['values'][k3km,:,:])) & \
        (np.isfinite(data['Z_H']['values'][k4km,:,:])) & \
        (data['Z_H']['values'][k4km,:,:] >= ReflThresh_lowlevel_gap)
    nfix = np.count_nonzero(ifix)
    if (nfix > 0):
        # Copy reflectivity at 3 km and 4 km
        tmp = data['Z_H']['values'][k3km,:,:]
        tmp2 = data['Z_H']['values'][k4km,:,:]
        tmp[ifix] = tmp2[ifix]
        # Use high reflectivity values at 4 km to replace missing obs at 3 km
        data['Z_H']['values'][k3km,:,:] = tmp
        del tmp, tmp2

    # Create array to store SL3D classification
    sl3dclass = np.zeros((ny, nx), dtype=np.short)

    # Sum depths of echo at and above 3 km (assuming 1-km GridRad v3.1 data)
    tmp = data['Z_H']['values'][k3km:,:,:]
    dzgt00dBZ = np.sum((~np.isnan(tmp)) & (tmp >=0.0), axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Get 25.0 dBZ echo top
        tmp = (data['Z_H'])['values']
        etop25dBZ = np.nanmax(((~np.isnan(tmp)) & (tmp >=25.0)) * zzz, axis=0)

        # Get column-maximum reflectivity
        dbz_comp = np.nanmax(tmp, axis=0)

        # Get column-maximum reflectivity for above melting level altitudes
        dbz_aml = np.nanmax(tmp * (zzz > (zml + 1.0)), axis=0)

    # Create array to compute peakedness in lowest 9 km altitude layer
    peak = np.full((k9km+1,ny,nx), np.NaN, dtype=data['Z_H']['values'].dtype)

    # Loop over the lowest 9 km levels
    for k in range(0, k9km+1):
        tmp = data['Z_H']['values'][k,:,:]

        # According to this thread: 
        # https://forum.image.sc/t/skimage-filters-median-using-mask-for-floating-point-image-with-nans/57289
        # scipy.ndimage.median_filter v1.7 (same as skimage.filters.median v0.17) above ignores NaN
        # But it produces incorrect values at the edge of the domain
        # These values will be removed at the end of the code
        peak[k,:,:] = tmp - ndimage.median_filter(tmp, size=nsearch)

    # Compute peakedness threshold for reflectivity value
    tmp = 10.0 - ((data['Z_H']['values'][0:k9km+1,:,:])**2) / 337.5
    peak_thresh = np.full(peak.shape, np.NaN, dtype=peak.dtype)
    largeindex = (~np.isnan(tmp)) & (tmp > 4.0)
    smallindex = (~np.isnan(tmp)) & (tmp <= 4.0)
    peak_thresh[largeindex] = tmp[largeindex]
    peak_thresh[smallindex] = 4.0

    # Compute column-mean peakedness fraction > peak_thresh
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        tmp = data['Z_H']['values'][0:k9km+1,:,:]
        mean_peak = np.sum((~np.isnan(peak_thresh)) &
                           (peak > peak_thresh), axis=0) / \
                    np.sum(np.isfinite(tmp), axis=0)

    # Find convective points 
    # those with at least x% of column exceeding peakedness or 
    # > x dBZ above melting level or echo top > x km
    iconv = (np.isfinite(mean_peak) & (mean_peak >= col_peakedness_frac)) | \
        (np.isfinite(dbz_aml) & (dbz_aml >= abs_ConvThres_aml)) | \
        (np.isfinite(etop25dBZ) & (etop25dBZ >= etop25dBZ_Thresh))
    nconv = np.count_nonzero(iconv)
    # Flag as convective
    if (nconv > 0): sl3dclass[iconv] = 2

    # Compute fraction of neighborhood with convection
    tmp = 1.0 * (sl3dclass == 2)
    conv_test = ndimage.uniform_filter(tmp, size=3, mode='nearest')

    # Find single grid point convective classifications to remove
    iremove = (sl3dclass == 2) & (conv_test <= 0.15)
    sl3dclass[iremove] = 0

    # Compute fraction of neighborhood with convection (again)
    tmp = 1.0 * (sl3dclass == 2)
    conv_test = ndimage.uniform_filter(tmp, size=3, mode='nearest')

    # Find points neighboring convective classification that have similarly intense reflectivity
    iconv2 = (conv_test > 0.001) & (dbz_comp >= neighbor_CompReflThresh)
    nconv2 = np.count_nonzero(iconv2)
    # Set classification for similarly intense regions as convection 
    # (i.e., after convective radius of Steiner et al, but strictly for adjacent grid points)
    if (nconv2 > 0): sl3dclass[iconv2] = 2

    # Find precipitating stratiform points
    istrat = ((data['Z_H']['values'][k3km,:,:] >= strat_EchoThresh_3km) | \
        (np.sum(data['Z_H']['values'][0:k3km,:,:] >= strat_EchoThresh_lt3km, axis=0) > 0)) & \
        (sl3dclass == 0)
    nstrat = np.count_nonzero(istrat)
    # Flag as precipitating stratiform
    if (nstrat > 0): sl3dclass[istrat] = 3

    # Find non-precipitating stratiform points
    itranv = ((~np.isfinite(data['Z_H']['values'][k3km,:,:])) | \
        (data['Z_H']['values'][k3km,:,:] < strat_EchoThresh_3km)) & \
        (dzgt00dBZ > 0.0) & (sl3dclass == 0)
    ntranv = np.count_nonzero(itranv)
    # Flag as non-precipitating stratiform
    if (ntranv > 0): sl3dclass[itranv] = 4

    # Find anvil
    ianvil = (np.sum(np.isfinite(data['Z_H']['values'][k3km:k5km+1,:,:]), axis=0) == 0) & \
        (dzgt00dBZ > 0.0) & \
        ((sl3dclass == 0) | (sl3dclass == 4))
    nanvil = np.count_nonzero(ianvil)
    # Flag anvil
    if (nanvil > 0): sl3dclass[ianvil] = 5

    # Compute reflectivity altitude gradient
    ddbzdz = np.roll(data['Z_H']['values'], -1, axis=0) - (data['Z_H'])['values']
    # Compute fraction of neighborhood with echo
    tmp = 1.0 * (np.isfinite(data['Z_H']['values']))
    fin_test = ndimage.uniform_filter(tmp, size=[1,3,3])

    # Search for weak echo regions
    iupdraft = (np.sum((ddbzdz >= updraft_ReflGradiant_Thresh) * \
        (fin_test >= 0.7) * (zzz <= updraft_ReflGradiant_MaxHeight), axis=0) > 0.0) & \
        (dbz_comp >= updraft_CompRefl_Thresh) & (sl3dclass == 2)
    nupdraft = np.count_nonzero(iupdraft)
    # Flag updrafts
    if (nupdraft > 0): sl3dclass[iupdraft] = 1

    # Compute fraction of neighborhood with convective updraft
    tmp = 1.0 * (sl3dclass == 1)
    updft_test = ndimage.uniform_filter(tmp, size=3)

    # Find single grid point updraft classifications to remove
    iremove = (sl3dclass == 1) & (updft_test <= 0.15)
    nremove = np.count_nonzero(iremove)
    # Replace single grid point updraft classifications with median local neighborhood classification
    if (nremove > 0): sl3dclass[iremove] = (ndimage.median_filter(sl3dclass, size=3))[iremove]

    # Set boundary grids to 0
    # The classification results near the boundary are problematic because 
    # median_filter and uniform_filter near the edge are not well defined
    sl3dclass[0:ngrids, :] = 0
    sl3dclass[-ngrids:, :] = 0
    sl3dclass[:, 0:ngrids] = 0
    sl3dclass[:, -ngrids:] = 0

    return sl3dclass
