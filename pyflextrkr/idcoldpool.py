import os
import sys
import numpy as np
import time
import xarray as xr
import pandas as pd
import logging
from scipy import integrate
from scipy.ndimage import gaussian_filter
from pyflextrkr.ftfunctions import sort_renumber, skimage_watershed, adjust_pbc_coldpools
from pyflextrkr.ft_utilities import get_timestamp_from_filename_single

#---------------------------------------------------------------------------------
def calculate_buoyancy_dmean(ds, var='thetav'):
    """
    Calculates buoyancy from potential virtual temperature.

    Arguments:
        ds: Xarray Dataset
            Input DataSet.
        var: string
            Name of potential virtual temperature variable.

    Returns:
        Buoyancy: Xarray DataArray
            Buoyancy field.
    """
    # Calculate spatial mean over X and Y
    spatial_mean = ds[var].mean(dim=("X", "Y"), keep_attrs=True)

    # Calculate buoyancy
    buoyancy = 9.81 * (ds[var] - spatial_mean) / spatial_mean

    # Create the buoyancy dataset
    ds_buoyancy = buoyancy.to_dataset(name='buoyancy')
    
    return ds_buoyancy

#---------------------------------------------------------------------------------
def calc_coldpool_intensity(buoy, zz, threshold=-0.005, min_cp_depth=0):
    """
    Calculates cold-pool intensity, essentially the integrated buoyancy from the surface up to where the threshold value is crossed.
    Original author: William.Gustafson@pnnl.gov
    edited by: Laura.Paccini@pnnl.gov
    Date: 30-Oct-2024

    Arguments:
        buoy: numpy array [z, y, x]
            Buoyancy 3D array.
        zz: numpy array [z, y, x]
            Height profile on center points [m]
        threshold: float, default=-0.005
            Buoyancy threshold for determining top of cold pool [m/s^2]
        min_cp_depth: float, default=0
            Minimum cold pool depth threshold [m].

    Returns:
        cp_dict: dictionary
            Dictionary containing cold pool variables.
    """
    # Find column-min buoyancy (buoy dimensions: [Z, Y, X])
    # jpool, ipool = np.where(np.nanmin(buoy, axis=0) < threshold)

    # Find lowest level buoyancy < threshold (i.e., 'surface' cold pools)
    jpool, ipool = np.where(np.squeeze(buoy[0,:,:]) < threshold)
    npool = len(jpool)
    
    # Initialize arrays
    nk, nj, ni = buoy.shape
    cp_intensity = np.zeros([nj, ni], dtype=np.float32)
    depthpool = np.zeros_like(cp_intensity)
    depth_base = np.zeros_like(cp_intensity)
    depth_top = np.zeros_like(cp_intensity)

    # Loop over each column that contains cold pool
    for m in range(npool):
        buoy_profile = buoy[:, jpool[m], ipool[m]]
        kpool = np.where(np.squeeze(buoy_profile < threshold))[0]

        gap = 1  # num. of levels of separation allowed before declaring a new layer
        layers = np.split(kpool, np.where(np.diff(kpool) > gap)[0]+1)
        # Ensure layers is non-empty and the first layer is non-empty
        if len(layers) > 0 and len(layers[0]) > 0:
            # Get layer top & bottom indices
            ktop = layers[0][-1]
            kbot = layers[0][0]
            ktopp1 = ktop + 1
            # Get height for this grid
            zz_profile = zz[:, jpool[m], ipool[m]]
            # Get cold pool depth
            z_depth = zz_profile[ktop] - zz_profile[kbot]
            # Check if it exceeds min coldpool depth
            if z_depth > min_cp_depth:
                # Vertically integrate buoyancy
                integrated_buoyancy = integrate.simpson(buoy_profile[kbot:ktopp1], x=zz_profile[kbot:ktopp1])
                # integrated_buoyancy = integrate.simps(buoy_profile[kbot:ktopp1], zz_profile[kbot:ktopp1])
                # depthpool[jpool[m], ipool[m]] = zz[ktop-kbot, jpool[m], ipool[m]]
                # depth_base[jpool[m], ipool[m]] = zz[kbot, jpool[m], ipool[m]]
                depthpool[jpool[m], ipool[m]] = z_depth
                depth_base[jpool[m], ipool[m]] = zz_profile[kbot]
                depth_top[jpool[m], ipool[m]] = zz_profile[ktop]
        
                # Compute cold pool intensity (Bryan & Parker 2010; Rotunno et al. 1988)
                with np.errstate(invalid='ignore'):
                    cp_intensity[jpool[m], ipool[m]] = np.sqrt(-2.0 * integrated_buoyancy)

                # import matplotlib.pyplot as plt
                # import pdb; pdb.set_trace()

    # Put variables to dictionary
    cp_dict = {
        'cp_depth': depthpool,
        'cp_intensity': cp_intensity,
        'cp_base': depth_base,
        'cp_top': depth_top,
    }
    return cp_dict

#---------------------------------------------------------------------------------
def filter_nonsfc_coldpool(cp_dict, zz_bot):
    """
    Filter non-surface cold pools.

    Arguments:
        cp_dict: dictionary
            Dictionary containing cold pool variables.
        zz_bot: numpy array [y, x]
            Height of the bottom level.

    Returns:
        out_dict: dictionary
            Dictionary containing cold pool variables.
    """
    # Mask of grids where cold pool base != lowest level
    mask = cp_dict['cp_base'] != zz_bot
    # Copy arrays
    cp_intensity_f = np.copy(cp_dict['cp_intensity'])
    cp_depth_f = np.copy(cp_dict['cp_depth'])
    cp_base_f = np.copy(cp_dict['cp_base'])
    cp_top_f = np.copy(cp_dict['cp_top'])
    # Filter
    cp_intensity_f[mask] = 0
    cp_depth_f[mask] = np.nan
    cp_base_f[mask] = np.nan
    cp_top_f[mask] = np.nan
    # Output dictionary
    out_dict = {
        'cp_intensity': cp_intensity_f,
        'cp_depth': cp_depth_f,
        'cp_base': cp_base_f,
        'cp_top': cp_top_f,
    }
    return out_dict

#---------------------------------------------------------------------------------
def idcoldpool(
    input_filename,
    config,
):
    """
    Identify cold pool.

    Arguments:
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

    databasename = config.get("databasename")
    time_format = config.get("time_format")
    feature_varname = config.get("feature_varname", "feature_number")
    nfeature_varname = config.get("nfeature_varname", "nfeatures")
    featuresize_varname = config.get("featuresize_varname", "npix_feature")
    x_dimname = config.get("x_dimname")
    y_dimname = config.get("y_dimname")
    z_dimname = config.get("z_dimname", None)
    time_dimname = config.get("time_dimname")
    time_coordname = config.get("time_coordname")
    x_coordname = config.get("x_coordname")
    y_coordname = config.get("y_coordname")
    z_coordname = config.get("z_coordname")
    field_varname = config.get("field_varname")
    # field_thresh = config.get("field_thresh")
    min_size = config.get("min_size")
    # label_method = config.get("label_method", "ndimage.label")
    buoy_thresh = config.get("buoy_thresh")
    min_cp_depth = config.get("min_cp_depth")
    buoy_smooth_sigma = config.get("buoy_smooth_sigma", 1)
    pixel_radius = config.get("pixel_radius")
    # R_earth = config.get("R_earth")
    pass_varname = config.get("pass_varname", None)
    fillval = config["fillval"]

    # Read input data
    ds = xr.open_dataset(input_filename, mask_and_scale=False)
    # Get dimension names from the file
    dims_file = []
    for key in ds.sizes: dims_file.append(key)
    # Find extra dimensions beyond [time, z, y, x]
    dims_keep = [time_dimname, z_dimname, y_dimname, x_dimname]
    dims_drop = list(set(dims_file) - set(dims_keep))
    # Reorder Dataset dimensions
    if z_dimname is not None:
        # Drop extra dimensions, reorder to [time, z, y, x]
        ds = ds.drop_dims(dims_drop).transpose(
            time_dimname, z_dimname, y_dimname, x_dimname, missing_dims='ignore',
        )
    else:
        # Drop extra dimensions, reorder to [time, y, x]
        ds = ds.drop_dims(dims_drop).transpose(
            time_dimname, y_dimname, x_dimname, missing_dims='ignore',
        )

    # Check if time dimension exists in the DataSet
    if time_dimname not in ds.sizes:
        # Add a 'time' dimension with size 1 to all variables
        ds = ds.expand_dims(time_dimname, axis=0)

    # Check if time coordinate exists in the DataSet
    if time_coordname not in ds:
        # Handle no time coordinate in Dataset
        logger.warning(f'No time coordinate: {time_coordname} found in input data')
        logger.warning(f'Will estimate time from filename based on time_format in config: {time_format}')
        # Get Timestamp from filename
        file_timestamp = get_timestamp_from_filename_single(
            input_filename, databasename, time_format=time_format,
        )
        # Add Timestamp coordinate to the Dataset
        ds = ds.assign_coords({time_coordname:file_timestamp})
        # Add time dimension to all variables in the Dataset
        ds = xr.concat([ds], dim=time_dimname)
        logger.debug(f'Added Timestamp: {file_timestamp} calculated from filename to the input data')

    # Read data variables
    ntimes = ds.sizes[time_dimname]
    x_coord = ds.coords[x_coordname]
    y_coord = ds.coords[y_coordname]
    z_coord = ds.coords[z_coordname]
    time_decode = ds[time_coordname]
    field_var = ds[field_varname]
    ds.close()

    # Check x, y coordinate dimensions
    if (y_coord.ndim == 1) | (x_coord.ndim == 1):
        # Mesh 1D coordinate into 2D
        lon2d, lat2d = np.meshgrid(x_coord, y_coord)
        lon2d = lon2d.astype(np.float32)
        lat2d = lat2d.astype(np.float32)
    elif (y_coord.ndim == 2) | (x_coord.ndim == 2):
        lon2d = x_coord.data
        lat2d = y_coord.data

    # Check z coordinate dimensions
    if (z_coord.ndim == 1):
        # Expand z_coord to shape (Z, Y, X)
        _z_coord = z_coord.data[:, np.newaxis, np.newaxis]
        # Broadcast z_coord_expanded to match the shape of field_var
        var_shape = field_var.isel({time_dimname:0}).squeeze().shape
        zz = np.broadcast_to(_z_coord, var_shape)
    elif (z_coord.ndim == 3):
        zz = z_coord.data

    # Calculate mean lat/lon grid distance (assuming fix grid size)
    # dlon = np.mean(np.abs(np.diff(lon2d, axis=1)))
    # dlat = np.mean(np.abs(np.diff(lat2d, axis=0)))

    # Calculate grid cell area (simple cosine adjustment)
    # grid_area = (R_earth**2) * np.cos(np.deg2rad(lat2d)) * np.deg2rad(dlat) * np.deg2rad(dlon)
    grid_area = pixel_radius**2

    if pass_varname is not None:
        # Find the common variable names between the dataset and the list
        pass_varname = set(ds.data_vars) & set(pass_varname)
        # Subset the input dataset
        ds_pass = ds[pass_varname]


    # Loop over each time
    for tt in range(0, ntimes):
        # Get data at this time
        iTime = time_decode[tt]
        fvar = field_var.data[tt,:,:,:].squeeze()

        # Calculate cold pool intensity
        cp_dict = calc_coldpool_intensity(fvar, zz, threshold=buoy_thresh, min_cp_depth=min_cp_depth)

        # Filter points where cold pool bottom is not at the lowest height
        zz_bot = zz[0,:,:].squeeze()
        cp_dict = filter_nonsfc_coldpool(cp_dict, zz_bot)

        # Smooth buoyancy intensity
        cp_intensity_s = gaussian_filter(cp_dict['cp_intensity'], sigma=buoy_smooth_sigma)
        # import matplotlib.pyplot as plt
        # import pdb; pdb.set_trace()

       
        ### Label feauture considering boundary conditions
        if config["pbc_direction"] != "none":

            var_number, param_dict = adjust_pbc_coldpools(cp_intensity_s, config)
        else:
            # Label feature
            var_number, param_dict = skimage_watershed(cp_intensity_s, config)

        # Sort and renumber features, filter features < min_size or grid_area
        feature_mask, npix_feature = sort_renumber(var_number, min_size)

        # Get number of features
        nfeatures = np.nanmax(feature_mask)

        # Convert to basetime (i.e., Epoch time)
        iTimestamp = pd.to_datetime(iTime.dt.strftime("%Y-%m-%dT%H:%M:%S").item())
        file_basetime = np.array([(iTimestamp - pd.Timestamp('1970-01-01T00:00:00')).total_seconds()])
        # Convert to strings
        file_datestring = iTime.dt.strftime("%Y%m%d").item()
        file_timestring = iTime.dt.strftime("%H%M%S").item()
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

        #######################################################
        # Output netcdf file
        # Define 3 variables required for tracking
        bt_attrs = {
            "long_name": "Base time in Epoch",
            "units": "Seconds since 1970-1-1 0:00:00 0:00",
        }
        featuremask_attrs = {
            "long_name": "Labeled feature number for tracking",
            "units": "unitless",
        }
        nfeatures_attrs = {
            "long_name": "Number of features labeled",
            "units": "unitless",
        }
        npix_feature_attrs = {
            "long_name": "Number of pixels for each feature",
            "units": "unitless",
        }
        # Additional variables
        cp_intensity_attrs = {
            "long_name": "Cold pool intensity (vertically integrated buoyancy)",
            "units": "m/s",
        }
        cp_depth_attrs = {
            "long_name": "Cold pool depth",
            "units": "m",
        }
        cp_base_attrs = {
            "long_name": "Cold pool base height",
            "units": "m",
        }
        cp_top_attrs = {
            "long_name": "Cold pool top height",
            "units": "m",
        }

        # Define variable dictionary
        var2d_dims = ["time", "lat", "lon"]
        var_dict = {
            "base_time": (["time"], out_basetime, bt_attrs),
            "longitude": (["lat", "lon"], lon2d, x_coord.attrs),
            "latitude": (["lat", "lon"], lat2d, y_coord.attrs),
            "cp_intensity": (var2d_dims, np.expand_dims(cp_dict['cp_intensity'], 0), cp_intensity_attrs),
            "cp_depth": (var2d_dims, np.expand_dims(cp_dict['cp_depth'], 0), cp_depth_attrs),
            "cp_base": (var2d_dims, np.expand_dims(cp_dict['cp_base'], 0), cp_base_attrs),
            "cp_top": (var2d_dims, np.expand_dims(cp_dict['cp_top'], 0), cp_top_attrs),
            feature_varname: (var2d_dims, np.expand_dims(feature_mask, 0), featuremask_attrs),
            nfeature_varname: (["time"], out_nfeatures, nfeatures_attrs),
            featuresize_varname: (["features"], npix_feature, npix_feature_attrs),
        }
        coord_dict = {
            "time": (["time"], out_basetime, bt_attrs),
            "lat": (["lat"], y_coord.data, y_coord.attrs),
            "lon": (["lon"], x_coord.data, x_coord.attrs),
            "features": (["features"], np.arange(1, nfeatures + 1)),
        }
        gattr_dict = {
            "Title": f"FeatureID file from {file_datestring}.{file_timestring}",
            "Institution": "Pacific Northwest National Laboratory",
            "Contact": "Zhe Feng: zhe.feng@pnnl.gov",
            "Created_on": time.ctime(time.time()),
            "min_size": min_size,
        }
        # Add each parameter to global attribute dictionary
        for key in param_dict:
            gattr_dict[key] = param_dict[key]

        # Add pass out variables to the output variable dictionary
        if pass_varname is not None:
            # Subset the time from the pass out Dataset
            dsp = ds_pass.isel({time_coordname:tt})
            # Loop over each pass out variable list
            for ivar in pass_varname:
                var_dict[ivar] = (var2d_dims, np.expand_dims(dsp[ivar].data, 0), dsp[ivar].attrs)

        # Define xarray dataset
        dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

        # Delete file if it already exists
        if os.path.isfile(cloudid_outfile):
            os.remove(cloudid_outfile)

        # Set encoding/compression for all variables
        comp = dict(zlib=True)
        encoding = {var: comp for var in dsout.data_vars}
        # Write to netcdf file
        dsout.to_netcdf(
            path=cloudid_outfile,
            mode='w',
            format='NETCDF4',
            encoding=encoding
        )
        logger.info(f"{cloudid_outfile}")
        # import matplotlib.pyplot as plt
        # import pdb; pdb.set_trace()

    return cloudid_outfile