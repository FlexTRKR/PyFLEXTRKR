import os
import sys
import numpy as np
import time
import xarray as xr
import logging
from datetime import datetime
from scipy.ndimage import label
from pyflextrkr.ftfunctions import sort_renumber, skimage_watershed

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

    feature_varname = config.get("feature_varname", "feature_number")
    nfeature_varname = config.get("nfeature_varname", "nfeatures")
    featuresize_varname = config.get("featuresize_varname", "npix_feature")
    x_dimname = config.get("x_dimname", "longitude")
    y_dimname = config.get("y_dimname", "latitude")
    time_dimname = config.get("time", "time")
    field_varname = config.get("field_varname")
    field_thresh = config.get("field_thresh")
    min_size = config.get("min_size")
    label_method = config.get("label_method", "ndimage.label")
    R_earth = config.get("R_earth")
    fillval = config["fillval"]

    # Get min/max field thresholds
    field_thresh_min = np.min(field_thresh)
    field_thresh_max = np.max(field_thresh)

    # Read input data
    ds = xr.open_dataset(input_filename, mask_and_scale=False)
    # Get number of dimensions
    ndims = len(ds.dims)
    # Reorder data dimensions & drop extra dimensions
    if ndims == 2:
        # Reorder dimensions: [y, x]
        ds = ds.transpose(y_dimname, x_dimname)
    elif ndims == 3:
        # Reorder dimensions: [time, y, x]
        ds = ds.transpose(time_dimname, y_dimname, x_dimname)
    elif ndims >= 4:
        # Get dimension names from the file
        dims_file = []
        for key in ds.dims: dims_file.append(key)
        # Find extra dimensions beyond [time, y, x]
        dims_keep = [time_dimname, y_dimname, x_dimname]
        dims_drop = list(set(dims_file) - set(dims_keep))
        # Drop extra dimensions, reorder to [time, y, x]
        ds = ds.drop_dims(dims_drop).transpose(time_dimname, y_dimname, x_dimname)
    else:
        logger.error(f"ERROR: Unexpected input data dimensions: {ds.dims}")
        logger.error("Must add codes to handle reading.")
        logger.error("Tracking will now exit.")
        sys.exit()

    # Read data variables
    ntimes = ds.dims[time_dimname]
    x_coord = ds.coords[x_dimname]
    y_coord = ds.coords[y_dimname]
    time_decode = ds[time_dimname]
    field_var = ds[field_varname]
    ds.close()

    # Create 2D lat/lon grid
    lon2d, lat2d = np.meshgrid(x_coord, y_coord)
    lon2d = lon2d.astype(np.float32)
    lat2d = lat2d.astype(np.float32)
    # Calculate mean lat/lon grid distance (assuming fix grid size)
    # dlon = np.array(x_coord[2] - x_coord[1])
    # dlat = np.array(y_coord[2] - y_coord[1])
    dlon = np.mean(np.abs(np.diff(x_coord)))
    dlat = np.mean(np.abs(np.diff(y_coord)))

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

    # Loop over each time
    for tt in range(0, ntimes):
        # Get data at this time
        iTime = ds.indexes['time'][tt]
        fvar = field_var.data[tt,:,:]

        # Label feature with simple threshold & connectivity method
        if label_method == 'ndimage.label':
            var_number, nblobs = label((field_thresh_min < fvar) & (fvar < field_thresh_max))
            param_dict = {
                'field_thresh': field_thresh,
            }

        if label_method == 'skimage.watershed':
            var_number, param_dict = skimage_watershed(fvar, config)

        # Sort and renumber features, filter features < min_size or grid_area
        feature_mask, npix_feature = sort_renumber(var_number, min_size, grid_area=grid_area)

        # Get number of features
        nfeatures = np.nanmax(feature_mask)

        # Get date/time and make output filename
        file_basetime = np.array([(np.datetime64(iTime).item() - datetime(1970, 1, 1, 0, 0, 0)).total_seconds()])
        file_datestring = time_decode[tt].dt.strftime("%Y%m%d").item()
        file_timestring = time_decode[tt].dt.strftime("%H%M").item()
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