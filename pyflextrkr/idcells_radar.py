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

    np.set_printoptions(threshold=np.inf)
    logger = logging.getLogger(__name__)

    fillval = config["fillval"]

    # Read input data
    ds = xr.open_dataset(input_filename, mask_and_scale=False)
    time_decode = ds["time"]
    out_x = ds["x"]
    out_y = ds["y"]
    out_lat = ds["latitude"]
    out_lon = ds["longitude"]
    comp_ref = ds["dbz_comp"]
    dbz_lowlevel = ds["dbz_lowlevel"]
    conv_mask_inflated = ds["conv_mask_inflated"]
    conv_core = ds["conv_core"]
    conv_mask = ds["conv_mask"]
    echotop10 = ds["echotop10"]
    echotop20 = ds["echotop20"]
    echotop30 = ds["echotop30"]
    echotop40 = ds["echotop40"]
    echotop50 = ds["echotop50"]
    dx = ds.attrs["dx"]
    dy = ds.attrs["dy"]
    ds.close()

    # Replace very small reflectivity values with nan
    comp_ref[np.where(comp_ref < -30)] = np.nan

    # Get the number of pixels for each cell.
    # conv_mask is already sorted so the returned sorted array is not needed, only the pixel count (cell size).
    # tmp, conv_npix = sort_renumber(conv_mask, 0)

    # Count number of pixels for each cell
    cellnum, conv_npix = np.unique(conv_mask, return_counts=True)
    # Remove background
    conv_npix = conv_npix[(cellnum > 0)]

    nclouds = np.nanmax(conv_mask_inflated)

    #######################################################
    # output data to netcdf file, only if clouds present
    # if nclouds > 0:
    file_basetime = time_decode[0].values.tolist() / 1e9
    file_datestring = time_decode.dt.strftime("%Y%m%d").item()
    file_timestring = time_decode.dt.strftime("%H%M").item()
    cloudid_outfile = (
        config['tracking_outpath']
        + config['datasource']
        + "_"
        + config['datadescription']
        + "_cloudid"
        + "_"
        + file_datestring
        + "_"
        + file_timestring
        + ".nc"
    )
    logger.info(f"outcloudidfile: {cloudid_outfile}")

    # Check if file exists, if it does delete it
    if os.path.isfile(cloudid_outfile):
        os.remove(cloudid_outfile)

    # Write output to netCDF file

    # Put time and nclouds in a numpy array so that they can be set with a time dimension
    out_basetime = np.zeros(1, dtype=float)
    out_basetime[0] = file_basetime

    out_nclouds = np.zeros(1, dtype=int)
    out_nclouds[0] = nclouds

    # Define variable list
    varlist = {
        "base_time": (["time"], out_basetime),
        "x": (["lon"], out_x, out_x.attrs),
        "y": (["lat"], out_y, out_y.attrs),
        "latitude": (["lat", "lon"], out_lat, out_lat.attrs),
        "longitude": (["lat", "lon"], out_lon, out_lon.attrs),
        "comp_ref": (["time", "lat", "lon"], comp_ref, comp_ref.attrs),
        "dbz_lowlevel": (["time", "lat", "lon"], dbz_lowlevel, dbz_lowlevel.attrs),
        "conv_core": (["time", "lat", "lon"], conv_core, conv_core.attrs),
        "conv_mask": (["time", "lat", "lon"], conv_mask, conv_mask.attrs),
        feature_varname: (["time", "lat", "lon"], conv_mask_inflated),
        # "cloudnumber": (["time", "lat", "lon"], conv_mask_inflated),
        "echotop10": (["time", "lat", "lon"], echotop10, echotop10.attrs),
        "echotop20": (["time", "lat", "lon"], echotop20, echotop20.attrs),
        "echotop30": (["time", "lat", "lon"], echotop30, echotop30.attrs),
        "echotop40": (["time", "lat", "lon"], echotop40, echotop40.attrs),
        "echotop50": (["time", "lat", "lon"], echotop50, echotop50.attrs),
        nfeature_varname: (["time"], out_nclouds),
        featuresize_varname: (["features"], conv_npix),
    }
    # Define coordinate list
    coordlist = {
        "time": (["time"], out_basetime),
        "lat": (["lat"], np.squeeze(out_lat[:, 0])),
        "lon": (["lon"], np.squeeze(out_lon[0, :])),
        "features": (["features"], np.arange(1, nclouds + 1)),
    }

    # Define global attributes
    gattrlist = {
        "title": "Convective cells identified in the data from "
        + file_datestring[0:4]
        + "/"
        + file_datestring[4:6]
        + "/"
        + file_datestring[6:8]
        + " "
        + file_timestring[0:2]
        + ":"
        + file_timestring[2:4]
        + " UTC",
        "institution": "Pacific Northwest National Laboratory",
        "contact": "Zhe Feng, zhe.feng@pnnl.gov",
        "created_on": time.ctime(time.time()),
        "dx": dx,
        "dy": dy,
        "minimum_cloud_area": config['area_thresh'],
    }

    # Define xarray dataset
    ds_out = xr.Dataset(varlist, coords=coordlist, attrs=gattrlist)

    # Specify variable attributes
    ds_out["time"].attrs["long_name"] = "Base time in Epoch"
    ds_out["time"].attrs["units"] = "Seconds since 1970-1-1 0:00:00 0:00"

    ds_out["base_time"].attrs["long_name"] = "Base time in Epoch"
    ds_out["base_time"].attrs["units"] = "Seconds since 1970-1-1 0:00:00 0:00"

    ds_out[nfeature_varname].attrs["long_name"] = "Number of convective cells identified"
    ds_out[nfeature_varname].attrs["units"] = "unitless"

    ds_out[featuresize_varname].attrs["long_name"] = "Number of pixels in each convective cells"
    ds_out[featuresize_varname].attrs["units"] = "unitless"

    ds_out[feature_varname].attrs[
        "long_name"
    ] = "Labeled feature number for tracking"
    ds_out[feature_varname].attrs["units"] = "unitless"
    ds_out[feature_varname].attrs["_FillValue"] = 0

    # ds_out["cloudnumber"].attrs[
    #     "long_name"
    # ] = "Grid with each classified cell given a number"
    # ds_out["cloudnumber"].attrs["units"] = "unitless"
    # ds_out["cloudnumber"].attrs["_FillValue"] = 0

    # Specify encoding list
    encodelist = {
        "time": {"zlib": True},
        "base_time": {"zlib": True, "dtype": "float"},
        "x": {"zlib": True, "dtype":"float32"},
        "y": {"zlib": True, "dtype":"float32"},
        "lon": {"zlib": True, "dtype":"float32"},
        "lat": {"zlib": True, "dtype":"float32"},
        "longitude": {"zlib": True, "dtype":"float32"},
        "latitude": {"zlib": True, "dtype":"float32"},
        "comp_ref": {"zlib": True},
        "dbz_lowlevel": {"zlib": True},
        "conv_core": {"zlib": True},
        "conv_mask": {"zlib": True, "_FillValue":0},
        feature_varname: {"zlib": True, "dtype": "int"},
        # "cloudnumber": {"zlib": True, "dtype": "int"},
        "echotop10": {"zlib": True},
        "echotop20": {"zlib": True},
        "echotop30": {"zlib": True},
        "echotop40": {"zlib": True},
        "echotop50": {"zlib": True},
        nfeature_varname: {"dtype": "int", "zlib": True},
        featuresize_varname: {"dtype": "int", "zlib": True, "_FillValue":fillval},
    }

    # Write netCDF file
    ds_out.to_netcdf(
        path=cloudid_outfile,
        mode="w",
        format="NETCDF4",
        encoding=encodelist,
        # unlimited_dims="time",
    )

    # else:
    #     logger.info(input_filename)
    #     logger.info('No clouds')

    # import pdb; pdb.set_trace()

    return cloudid_outfile
