import os
import numpy as np
import time
import xarray as xr
import logging

def idcell_csapr(
    input_filename,
    config,
):
    """
    Identifies convective cells from CSAPR data.

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
    x_dimname = config.get("x_dimname", "x")
    y_dimname = config.get("y_dimname", "y")
    time_dimname = config.get("time", "time")

    np.set_printoptions(threshold=np.inf)
    logger = logging.getLogger(__name__)

    fillval = config["fillval"]

    # Read input data
    ds = xr.open_dataset(input_filename, engine='h5netcdf',mask_and_scale=False)
    time_decode = ds["time"]
    feature_mask = ds["conv_mask_inflated"]
    ds.close()

    # Count number of pixels for each feature
    unique_num, npix_feature = np.unique(feature_mask, return_counts=True)
    # Remove background (unique_num = 0)
    npix_feature = npix_feature[(unique_num > 0)]

    # Get number of features
    nfeatures = np.nanmax(feature_mask)

    # Get date/time and make output filename
    file_basetime = time_decode[0].values.tolist() / 1e9
    file_datestring = time_decode.dt.strftime("%Y%m%d").item()
    file_timestring = time_decode.dt.strftime("%H%M").item()
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
    # Convert to DataArray
    # These 3 variables are required for tracking
    bt_attrs = {
        "long_name": "Base time in Epoch",
        "units": "Seconds since 1970-1-1 0:00:00 0:00",
    }
    nfeatures_attrs = {
        "long_name": "Number of features labeled",
        "units": "unitless",
    }
    npix_feature_attrs = {
        "long_name": "Number of pixels for each feature",
        "units": "unitless",
    }
    out_nfeatures = xr.DataArray(
        out_nfeatures,
        coords={"time": time_decode},
        dims=("time"),
        attrs=nfeatures_attrs,
    )
    out_npix_feature = xr.DataArray(
        npix_feature,
        coords={"features": np.arange(1, nfeatures + 1)},
        dims=("features"),
        attrs=npix_feature_attrs,
    )
    out_basetime = xr.DataArray(
        out_basetime, coords={"time": time_decode}, dims=("time"), attrs=bt_attrs
    )
    feature_mask.attrs["long_name"] = "Labeled feature number for tracking"

    # Copy input dataset for output
    dsout = ds.copy(deep=True)
    # Add new variables to dataset
    dsout["base_time"] = out_basetime
    dsout[feature_varname] = feature_mask
    dsout[nfeature_varname] = out_nfeatures
    dsout[featuresize_varname] = out_npix_feature
    # Rename dimensions
    if x_dimname != "lon":
        dsout = dsout.rename_dims({x_dimname: "lon"})
    if y_dimname != "lat":
        dsout = dsout.rename_dims({y_dimname: "lat"})
    if time_dimname != "time":
        dsout = dsout.rename_dims({time_dimname: "time"})

    # Update global attributes
    dsout.attrs["created_on"] = time.ctime(time.time())

    #######################################################
    # Write output to netCDF file

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

    return cloudid_outfile
