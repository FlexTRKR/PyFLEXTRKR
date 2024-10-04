import os
import sys
import numpy as np
import time
import xarray as xr
import pandas as pd
import logging
from scipy.ndimage import label
from pyflextrkr.ftfunctions import sort_renumber, skimage_watershed
from pyflextrkr.ft_utilities import get_timestamp_from_filename_single

def idfeature_generic(
    input_filename,
    config,
):
    """
    Identify generic features.

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
    field_varname = config.get("field_varname")
    field_thresh = config.get("field_thresh")
    min_size = config.get("min_size")
    label_method = config.get("label_method", "ndimage.label")
    R_earth = config.get("R_earth")
    pass_varname = config.get("pass_varname", None)
    fillval = config["fillval"]

    # Get min/max field thresholds
    field_thresh_min = np.min(field_thresh)
    field_thresh_max = np.max(field_thresh)

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
    time_decode = ds[time_coordname]
    field_var = ds[field_varname]
    ds.close()

    # Check coordinate dimensions
    if (y_coord.ndim == 1) | (x_coord.ndim == 1):
        # Mesh 1D coordinate into 2D
        lon2d, lat2d = np.meshgrid(x_coord, y_coord)
        lon2d = lon2d.astype(np.float32)
        lat2d = lat2d.astype(np.float32)
    elif (y_coord.ndim == 2) | (x_coord.ndim == 2):
        lon2d = x_coord.data
        lat2d = y_coord.data

    # Calculate mean lat/lon grid distance (assuming fix grid size)
    dlon = np.mean(np.abs(np.diff(lon2d, axis=1)))
    dlat = np.mean(np.abs(np.diff(lat2d, axis=0)))

    # Calculate grid cell area (simple cosine adjustment)
    grid_area = (R_earth**2) * np.cos(np.deg2rad(lat2d)) * np.deg2rad(dlat) * np.deg2rad(dlon)

    # Alternatively, use Metpy to calculate grid_area. Need to add Metpy to library requirement.
    # Calculate dx, dy using Metpy function
    # import metpy.calc as mpcalc
    # dx, dy = mpcalc.lat_lon_grid_deltas(lon2d, lat2d)
    # # Pad dx, dy to match lat2d, lon2d grid
    # dx_ = np.zeros(lon2d.shape)
    # dx_[:,:-1] = dx
    # dx_[:,-1] = dx[:,-1]
    # dy_ = np.zeros(lat2d.shape)
    # dy_[:-1,:] = dy
    # dy_[-1,:] = dy[-1,:]
    # Compute grid_area
    # grid_area = dx_ * dy_

    if pass_varname is not None:
        # Find the common variable names between the dataset and the list
        pass_varname = set(ds.data_vars) & set(pass_varname)
        # Subset the input dataset
        ds_pass = ds[pass_varname]

    # Loop over each time
    for tt in range(0, ntimes):
        # Get data at this time
        iTime = time_decode[tt]
        fvar = field_var.data[tt,:,:]

        # Label feature
        # Simple threshold & connectivity method
        if label_method == 'ndimage.label':
            var_number, nblobs = label((field_thresh_min < fvar) & (fvar < field_thresh_max))
            param_dict = {
                'field_thresh': field_thresh,
            }

        # Watershed
        if label_method == 'skimage.watershed':
            var_number, param_dict = skimage_watershed(fvar, config)

        # Sort and renumber features, filter features < min_size or grid_area
        feature_mask, npix_feature = sort_renumber(var_number, min_size, grid_area=grid_area)

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
        # Define variable dictionary
        var_dict = {
            "base_time": (["time"], out_basetime, bt_attrs),
            "longitude": (["lat", "lon"], lon2d, x_coord.attrs),
            "latitude": (["lat", "lon"], lat2d, y_coord.attrs),
            field_varname: (["time", "lat", "lon"], np.expand_dims(fvar, 0), field_var.attrs),
            feature_varname: (["time", "lat", "lon"], np.expand_dims(feature_mask, 0), featuremask_attrs),
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
                var_dict[ivar] = (["time", "lat", "lon"], np.expand_dims(dsp[ivar].data, 0), dsp[ivar].attrs)

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

    return cloudid_outfile