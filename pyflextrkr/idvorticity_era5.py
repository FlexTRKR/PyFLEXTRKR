import os
import numpy as np
import time
import xarray as xr
import logging
from scipy.ndimage import label
from pyflextrkr.ftfunctions import sort_renumber

def idvorticity_era5(
    input_filename,
    config,
):
    """
    Identifies vorticity features from ERA5 data.

    Arguments:
        input_filename: string
            Input data filename
        config: dictionary
            Dictionary containing config parameters

    Returns:
        cloudid_outfile: string
            Cloudid file name.
    """
    feature_varname = config.get("feature_varname", "feature_number")
    nfeature_varname = config.get("nfeature_varname", "nfeatures")
    featuresize_varname = config.get("featuresize_varname", "npix_feature")
    x_dimname = config.get("x_dimname", "longitude")
    y_dimname = config.get("y_dimname", "latitude")
    time_dimname = config.get("time", "time")
    field_varname = config.get("field_varname")
    vor_thresh = config.get("vor_thresh")
    min_npix = config.get("min_npix")

    np.set_printoptions(threshold=np.inf)
    logger = logging.getLogger(__name__)

    fillval = config["fillval"]

    # Read input data
    ds = xr.open_dataset(input_filename, mask_and_scale=False)
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

    # Loop over each time
    for tt in range(0, ntimes):
        fvar = field_var.data[:,:,tt]

        # Label vorticity feature > vor_thresh
        vor_number, nvor = label(fvar > vor_thresh)

        # Sort and renumber features, filter features < min_npix
        feature_mask, npix_feature = sort_renumber(vor_number, min_npix)

        # Count number of pixels for each feature
        unique_num, npix_feature = np.unique(feature_mask, return_counts=True)
        # Remove background (unique_num = 0)
        npix_feature = npix_feature[(unique_num > 0)]

        # Get number of features
        nfeatures = np.nanmax(feature_mask)

        # Get date/time and make output filename
        file_basetime = time_decode[tt].values.tolist() / 1e9
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
            "title": f"Cloudid file from {file_datestring}.{file_timestring}",
            "institution": "Pacific Northwest National Laboratory",
            "contact": "Zhe Feng: zhe.feng@pnnl.gov",
            "created_on": time.ctime(time.time()),
            "vor_thresh": vor_thresh,
            "min_npix": min_npix,
        }
        # Define xarray dataset
        dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

        # Delete file if it already exists
        if os.path.isfile(cloudid_outfile):
            os.remove(cloudid_outfile)

        # Set encoding/compression for all variables
        comp = dict(zlib=True)
        encoding = {var: comp for var in dsout.data_vars}
        # Write to netcdf file
        dsout.to_netcdf(path=cloudid_outfile,
                        mode='w',
                        format='NETCDF4',
                        encoding=encoding)
        logger.info(f"{cloudid_outfile}")
        # import pdb; pdb.set_trace()

    return cloudid_outfile
