import time
import numpy as np
import xarray as xr
from netCDF4 import Dataset, stringtochar, num2date


def write_cloudid_tb(
    cloudid_outfile,
    file_basetime,
    file_datestring,
    file_timestring,
    out_lat,
    out_lon,
    out_ir,
    cloudtype,
    convcold_cloudnumber,
    cloudnumber,
    nclouds,
    ncorecoldpix,
    cloudtb_threshs,
    config,
    **kwargs
):

    """
    Writes cloudid variables to netCDF file.

    **kwargs:
    Expects these optional arguments:
            precipitation: np.ndarray(float)
            reflectivity: np.ndarray(float)
            pf_number: np.ndarray(int)
            convcold_cloudnumber_orig: np.ndarray(int)
            cloudnumber_orig: np.ndarray(int)
            linkpf: int
            pf_smooth_window: int
            pf_dbz_thresh: float
            pf_link_area_thresh: float

    """
    feature_varname = config.get("feature_varname", "feature_number")
    nfeature_varname = config.get("nfeature_varname", "nfeatures")
    featuresize_varname = config.get("featuresize_varname", "npix_feature")

    # Define variable list
    varlist = {
        "base_time": (["time"], file_basetime),
        "latitude": (["lat", "lon"], out_lat),
        "longitude": (["lat", "lon"], out_lon),
        feature_varname: (["time", "lat", "lon"], convcold_cloudnumber),
        "tb": (["time", "lat", "lon"], np.expand_dims(out_ir, axis=0)),
        "cloudtype": (["time", "lat", "lon"], cloudtype),
        "cloudnumber": (["time", "lat", "lon"], cloudnumber),
        nfeature_varname: (["time"], nclouds),
        featuresize_varname: (["features"], ncorecoldpix),
    }
    # Now check for optional arguments, add them to varlist if provided
    if "precipitation" in kwargs:
        varlist["precipitation"] = (["time", "lat", "lon"], kwargs["precipitation"])
    if "reflectivity" in kwargs:
        varlist["reflectivity"] = (["time", "lat", "lon"], kwargs["reflectivity"])
    if "pf_number" in kwargs:
        varlist["pf_number"] = (["time", "lat", "lon"], kwargs["pf_number"])
    if "convcold_cloudnumber_orig" in kwargs:
        varlist["convcold_cloudnumber_orig"] = (
            ["time", "lat", "lon"],
            kwargs["convcold_cloudnumber_orig"],
        )
    if "cloudnumber_orig" in kwargs:
        varlist["cloudnumber_orig"] = (
            ["time", "lat", "lon"],
            kwargs["cloudnumber_orig"],
        )

    # Define coordinate list
    coordlist = {
        "time": (["time"], file_basetime),
        "lat": (["lat"], np.squeeze(out_lat[:, 0])),
        "lon": (["lon"], np.squeeze(out_lon[0, :])),
        "features": (["features"], np.arange(1, nclouds + 1),),
    }

    # Define global attributes
    gattrlist = {
        "title": "Cloudid file from "
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
        'institution': 'Pacific Northwest National Laboratory',
        'contact': 'Zhe Feng: zhe.feng@pnnl.gov',
        "created_on": time.ctime(time.time()),
        "tb_threshold_core": cloudtb_threshs[0],
        "tb_threshold_coldanvil": cloudtb_threshs[1],
        "tb_threshold_warmanvil": cloudtb_threshs[2],
        "tb_threshold_environment": cloudtb_threshs[3],
        "minimum_cloud_area": config['area_thresh'],
    }
    # Now check for optional arguments, add them to gattrlist if provided
    if "linkpf" in kwargs:
        gattrlist["linkpf"] = kwargs["linkpf"]
    if "pf_smooth_window" in kwargs:
        gattrlist["pf_smooth_window"] = kwargs["pf_smooth_window"]
    if "pf_dbz_thresh" in kwargs:
        gattrlist["pf_dbz_thresh"] = kwargs["pf_dbz_thresh"]
    if "pf_link_area_thresh" in kwargs:
        gattrlist["pf_link_area_thresh"] = kwargs["pf_link_area_thresh"]
    # Define xarray dataset
    ds_out = xr.Dataset(varlist, coords=coordlist, attrs=gattrlist)

    # Specify variable attributes
    # ds_out.time.attrs['long_name'] = 'epoch time (seconds since 01/01/1970 00:00) in epoch of file'

    ds_out["time"].attrs["long_name"] = "Base time in Epoch"
    ds_out["time"].attrs["units"] = "Seconds since 1970-1-1"
    # ds_out["time"].attrs['units'] = 'Seconds since 1970-1-1 0:00:00 0:00'

    ds_out["base_time"].attrs["long_name"] = "Base time in Epoch"
    ds_out["base_time"].attrs["units"] = "Seconds since 1970-1-1"
    # ds_out["base_time"].attrs['units'] = 'Seconds since 1970-1-1 0:00:00 0:00'

    ds_out["lat"].attrs[
        "long_name"
    ] = "Vector of latitudes, y-coordinate in Cartesian system"
    ds_out["lat"].attrs["standard_name"] = "latitude"
    ds_out["lat"].attrs["units"] = "degrees_north"

    ds_out["lon"].attrs[
        "long_name"
    ] = "Vector of longitudes, x-coordinate in Cartesian system"
    ds_out["lon"].attrs["standard_name"] = "longitude"
    ds_out["lon"].attrs["units"] = "degrees_east"

    # ds_out["features"].attrs["long_name"] = "Number of features identified"
    # ds_out["features"].attrs["units"] = "counts"

    ds_out["latitude"].attrs["long_name"] = "cartesian grid of latitude"
    ds_out["latitude"].attrs["units"] = "degrees_north"

    ds_out["longitude"].attrs["long_name"] = "cartesian grid of longitude"
    ds_out["longitude"].attrs["units"] = "degrees_east"

    ds_out["tb"].attrs["long_name"] = "brightness temperature"
    ds_out["tb"].attrs["units"] = "K"

    ds_out["cloudtype"].attrs["long_name"] = "grid of cloud classifications"
    ds_out["cloudtype"].attrs[
        "values"
    ] = "1 = core, 2 = cold anvil, 3 = warm anvil, 4 = other"
    ds_out["cloudtype"].attrs["units"] = "unitless"
    ds_out["cloudtype"].attrs["_FillValue"] = 5

    ds_out[feature_varname].attrs[
        "long_name"
    ] = "Labeled feature number for tracking"
    ds_out[feature_varname].attrs["units"] = "unitless"
    # ds_out[feature_varname].attrs["valid_min"] = 0
    # ds_out[feature_varname].attrs["valid_max"] = nclouds + 1
    # ds_out[feature_varname].attrs[
    #     "comment"
    # ] = "extend of each cloud defined using cold anvil threshold"
    ds_out[feature_varname].attrs["_FillValue"] = 0

    ds_out["cloudnumber"].attrs[
        "long_name"
    ] = "grid with each classified cloud given a number"
    ds_out["cloudnumber"].attrs["units"] = "unitless"
    # ds_out.cloudnumber.attrs["valid_min"] = 0
    # ds_out.cloudnumber.attrs["valid_max"] = nclouds + 1
    ds_out["cloudnumber"].attrs[
        "comment"
    ] = "extend of each cloud defined using warm anvil threshold"
    ds_out["cloudnumber"].attrs["_FillValue"] = 0

    ds_out[nfeature_varname].attrs[
        "long_name"
    ] = "Number of features labeled"
    ds_out[nfeature_varname].attrs["units"] = "unitless"

    ds_out[featuresize_varname].attrs[
        "long_name"
    ] = "Number of pixels for each feature"
    ds_out[featuresize_varname].attrs["units"] = "unitless"

    # Now check for optional arguments, define attributes if provided
    if "precipitation" in kwargs:
        ds_out["precipitation"].attrs["long_name"] = "Precipitation"
        ds_out["precipitation"].attrs["units"] = "mm/h"
        ds_out["precipitation"].attrs["_FillValue"] = np.nan
    if "reflectivity" in kwargs:
        ds_out["reflectivity"].attrs["long_name"] = "Radar reflectivity"
        ds_out["reflectivity"].attrs["units"] = "dBZ"
        ds_out["reflectivity"].attrs["_FillValue"] = np.nan
    if "pf_number" in kwargs:
        ds_out["pf_number"].attrs["long_name"] = "Precipitation Feature number"
        ds_out["pf_number"].attrs["units"] = "unitless"
        ds_out["pf_number"].attrs["_FillValue"] = 0
    if "convcold_cloudnumber_orig" in kwargs:
        ds_out["convcold_cloudnumber_orig"].attrs[
            "long_name"
        ] = "Number of cloud system in this file that given pixel belongs to (before linked by pf_number)"
        ds_out["convcold_cloudnumber_orig"].attrs["units"] = "unitless"
        ds_out["convcold_cloudnumber_orig"].attrs["_FillValue"] = 0
    if "cloudnumber_orig" in kwargs:
        ds_out["cloudnumber_orig"].attrs[
            "long_name"
        ] = "Number of cloud system in this file that given pixel belongs to (before linked by pf_number)"
        ds_out["cloudnumber_orig"].attrs["units"] = "unitless"
        ds_out["cloudnumber_orig"].attrs["_FillValue"] = 0

    # Specify encoding list
    zlib = True
    encodelist = {
        "lon": {"zlib": zlib, "dtype":"float32"},
        "lat": {"zlib": zlib, "dtype":"float32"},
        "features": {"zlib": zlib},
        "longitude": {"zlib": zlib, "_FillValue": np.nan, "dtype":"float32"},
        "latitude": {"zlib": zlib, "_FillValue": np.nan, "dtype":"float32"},
        "tb": {"zlib": zlib, "_FillValue": np.nan},
        "cloudtype": {"zlib": zlib},
        feature_varname: {"dtype": "int", "zlib": zlib},
        "cloudnumber": {"dtype": "int", "zlib": zlib},
        nfeature_varname: {"dtype": "int", "zlib": zlib},
        featuresize_varname: {"dtype":"int", "zlib":zlib, "_FillValue":-9999},
    }
    # Now check for optional arguments, add them to encodelist if provided
    if "precipitation" in kwargs:
        encodelist["precipitation"] = {"zlib": zlib}
    if "reflectivity" in kwargs:
        encodelist["reflectivity"] = {"zlib": zlib}
    if "pf_number" in kwargs:
        encodelist["pf_number"] = {"zlib": zlib}
    if "convcold_cloudnumber_orig" in kwargs:
        encodelist["convcold_cloudnumber_orig"] = {"zlib": zlib}
    if "cloudnumber_orig" in kwargs:
        encodelist["cloudnumber_orig"] = {"zlib": zlib}

    # Write netCDF file
    ds_out.to_netcdf(
        path=cloudid_outfile, mode="w", format="NETCDF4", encoding=encodelist
    )
    return cloudid_outfile

#----------------------------------------------------------------------------------
def write_radar_cellid(
        cloudid_outfile,
        file_basetime,
        file_datestring,
        file_timestring,
        dx,
        dy,
        radar_lon,
        radar_lat,
        out_lon,
        out_lat,
        dbz_comp,
        dbz_lowlevel,
        reflectivity_file,
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
        nfeatures,
        config,
        **kwargs,
):
    feature_varname = config.get("feature_varname", "feature_number")
    nfeature_varname = config.get("nfeature_varname", "nfeatures")
    featuresize_varname = config.get("featuresize_varname", "npix_feature")

    # Define output variables
    varlist = {
        'radar_longitude': (radar_lon.data),
        'radar_latitude': (radar_lat.data),
        'base_time': (["time"], file_basetime),
        'longitude': (['lat', 'lon'], out_lon.data),
        'latitude': (['lat', 'lon'], out_lat.data),
        # 'dbz_comp': (['time', 'lat', 'lon'], dbz_comp.expand_dims('time', axis=0)),
        # 'dbz_lowlevel': (['time', 'lat', 'lon'], dbz_lowlevel.expand_dims('time', axis=0)),
        'dbz_comp': (['time', 'lat', 'lon'], np.expand_dims(dbz_comp.data, axis=0)),
        'dbz_lowlevel': (['time', 'lat', 'lon'], np.expand_dims(dbz_lowlevel.data, axis=0)),
        'convsf': (['time', 'lat', 'lon'], np.expand_dims(convsf_steiner, axis=0)),
        'conv_core': (['time', 'lat', 'lon'], np.expand_dims(core_steiner, axis=0)),
        'conv_mask': (['time', 'lat', 'lon'], np.expand_dims(core_sorted, axis=0)),
        'conv_mask_inflated': (['time', 'lat', 'lon'], np.expand_dims(core_expand, axis=0)),
        'echotop10': (['time', 'lat', 'lon'], np.expand_dims(echotop10, axis=0)),
        'echotop20': (['time', 'lat', 'lon'], np.expand_dims(echotop20, axis=0)),
        'echotop30': (['time', 'lat', 'lon'], np.expand_dims(echotop30, axis=0)),
        'echotop40': (['time', 'lat', 'lon'], np.expand_dims(echotop40, axis=0)),
        'echotop50': (['time', 'lat', 'lon'], np.expand_dims(echotop50, axis=0)),
        feature_varname: (["time", "lat", "lon"], np.expand_dims(feature_mask, axis=0)),
        nfeature_varname: (["time"], nfeatures),
        featuresize_varname: (["features"], npix_feature),
    }
    # Output coordinates
    coordlist = {
        'time': (['time'], file_basetime),
        'lon': (['lon'], np.squeeze(out_lon.data[0, :])),
        'lat': (['lat'], np.squeeze(out_lat.data[:, 0])),
        'features': (['features'], np.arange(1, nfeatures + 1),),
    }
    # Output global attributes
    attrlist = {
        "title": "Cloudid file from "
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
        'contact': 'Zhe Feng, zhe.feng@pnnl.gov',
        'Institution': 'Pacific Northwest National Laboratory',
        'created on': time.ctime(time.time()),
        'Input_File': reflectivity_file,
        'dx': dx,
        'dy': dy,
    }
    # Check for optional keyword steiner_params
    if 'steiner_params' in kwargs:
        # Add all parameters from the dictionary to the global attribute list
        for key, value in kwargs['steiner_params'].items():
            attrlist[key] = value

    # Define xarray dataset
    ds_out = xr.Dataset(varlist, coords=coordlist, attrs=attrlist)

    ds_out['lat'].attrs['long_name'] = 'Latitudes, y-coordinate in Cartesian system'
    ds_out['lat'].attrs['standard_name'] = 'latitude'
    ds_out['lat'].attrs['units'] = 'degrees_north'
    ds_out['lon'].attrs['long_name'] = 'Longitudes, x-coordinate in Cartesian system'
    ds_out['lon'].attrs['standard_name'] = 'longitude'
    ds_out['lon'].attrs['units'] = 'degrees_east'
    ds_out['time'].attrs['long_name'] = 'Base time in Epoch'
    ds_out['time'].attrs['units'] = 'Seconds since 1970-1-1'
    ds_out['base_time'].attrs['long_name'] = 'Base time in Epoch'
    ds_out['base_time'].attrs['units'] = 'Seconds since 1970-1-1'
    ds_out['radar_longitude'].attrs = radar_lon.attrs
    ds_out['radar_latitude'].attrs = radar_lat.attrs
    ds_out['longitude'].attrs = out_lon.attrs
    ds_out['latitude'].attrs = out_lat.attrs
    ds_out['dbz_comp'].attrs['long_name'] = f'Composite Reflectivity ({sfc_dz_min}+ m above surface in each column)'
    ds_out['dbz_comp'].attrs['units'] = 'dBZ'
    ds_out['dbz_lowlevel'].attrs['long_name'] = \
        f'Composite Low-level Reflectivity ({sfc_dz_min}+ m above surface in each column)'
    ds_out['dbz_lowlevel'].attrs['units'] = 'dBZ'
    ds_out['conv_core'].attrs['long_name'] = \
        'Convective Core Mask After Reflectivity Threshold and Peakedness Steps '+\
        '(1 = convective, 0 = not convective)'
    ds_out['conv_core'].attrs['units'] = 'unitless'
    ds_out['conv_mask'].attrs['long_name'] = \
        'Convective Region Mask After Reflectivity Threshold, Peakedness, and Expansion Steps '+\
        '(1 = convective, 0 = not convective)'
    ds_out['conv_mask'].attrs['units'] = 'unitless'
    ds_out['conv_mask_inflated'].attrs['long_name'] = \
        f'Convective Region Mask Inflated by 5 km (each region is a separate number > 0)'
    ds_out['conv_mask_inflated'].attrs['units'] = 'unitless'
    ds_out['convsf'].attrs['long_name'] = f'Steiner Convective/Stratiform classification'
    ds_out['convsf'].attrs['units'] = 'unitless'
    ds_out['convsf'].attrs['comment'] = 'NAN:0, NO_SURF_ECHO:1, WEAK_ECHO:2, STRATIFORM:3, CONVECTIVE:4'
    ds_out['echotop10'].attrs['long_name'] = '10dBZ echo-top height'
    ds_out['echotop10'].attrs['units'] = 'm'
    ds_out['echotop10'].attrs['_FillValue'] = np.nan
    ds_out['echotop20'].attrs['long_name'] = '20dBZ echo-top height'
    ds_out['echotop20'].attrs['units'] = 'm'
    ds_out['echotop20'].attrs['_FillValue'] = np.nan
    ds_out['echotop30'].attrs['long_name'] = '30dBZ echo-top height'
    ds_out['echotop30'].attrs['units'] = 'm'
    ds_out['echotop30'].attrs['_FillValue'] = np.nan
    ds_out['echotop40'].attrs['long_name'] = '40dBZ echo-top height'
    ds_out['echotop40'].attrs['units'] = 'm'
    ds_out['echotop40'].attrs['_FillValue'] = np.nan
    ds_out['echotop50'].attrs['long_name'] = '50dBZ echo-top height'
    ds_out['echotop50'].attrs['units'] = 'm'
    ds_out['echotop50'].attrs['_FillValue'] = np.nan
    ds_out[feature_varname].attrs['long_name'] = 'Labeled feature number for tracking'
    ds_out[feature_varname].attrs['units'] = 'unitless'
    ds_out[feature_varname].attrs['_FillValue'] = 0
    ds_out[nfeature_varname].attrs['long_name'] = 'Number of features labeled'
    ds_out[nfeature_varname].attrs['units'] = 'unitless'
    ds_out[featuresize_varname].attrs['long_name'] = 'Number of pixels for each feature'
    ds_out[featuresize_varname].attrs['units'] = 'unitless'

    # Now check for optional arguments, add them to output dataset if provided
    if 'refl_bkg' in kwargs:
        ds_out['refl_bkg'] = (['time', 'lat', 'lon'], np.expand_dims(kwargs['refl_bkg'], axis=0))
        ds_out['refl_bkg'].attrs['long_name'] = 'Steiner background reflectivity'
        ds_out['refl_bkg'].attrs['unit'] = 'dBZ'
    if 'peakedness' in kwargs:
        ds_out['peakedness'] = (['time', 'lat', 'lon'], np.expand_dims(kwargs['peakedness'], axis=0))
        ds_out['peakedness'].attrs['long_name'] = 'Peakedness above background reflectivity'
        ds_out['peakedness'].attrs['unit'] = 'dB'
    if 'core_steiner_orig' in kwargs:
        ds_out['core_steiner_orig'] = (['time', 'lat', 'lon'], np.expand_dims(kwargs['core_steiner_orig'], axis=0))
        ds_out['core_steiner_orig'].attrs['long_name'] = 'Steiner convective core before core area filter'
        ds_out['core_steiner_orig'].attrs['unit'] = 'unitless'

    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in ds_out.data_vars}

    # Write to netcdf file
    ds_out.to_netcdf(
        path=cloudid_outfile, mode='w', format='NETCDF4', unlimited_dims='time', encoding=encoding
    )
    return cloudid_outfile

def write_cloudtype_wrf(
    cloudid_outfile,
    file_basetime,
    file_datestring,
    file_timestring,
    out_lat,
    out_lon,
    out_ct,
    original_cloudtype,
    convcold_cloudnumber,
    nclouds,
    ncorepix,
    ncoldpix,
    ncorecoldpix,
    final_cloudtype,
    cloudtb_threshs,
    geolimits,
    mintb_thresh,
    maxtb_thresh,
    area_thresh,
    **kwargs
):
    """
    Writes cloudid variables to netCDF file.

    **kwargs:
    Expects these optional arguments:
            precipitation: np.ndarray(float)
            reflectivity: np.ndarray(float)
            pf_number: np.ndarray(int)
            convcold_cloudnumber_orig: np.ndarray(int)
            cloudnumber_orig: np.ndarray(int)
            linkpf: int
            pf_smooth_window: int
            pf_dbz_thresh: float
            pf_link_area_thresh: float

    """
    #         missing_value_int = -9999

    # Define variable list
    varlist = {
        "basetime": (["time"], file_basetime),
        "filedate": (
            ["time", "ndatechar"],
            np.array([stringtochar(np.array(file_datestring))]),
        ),
        "filetime": (
            ["time", "ntimechar"],
            np.array([stringtochar(np.array(file_timestring))]),
        ),
        "latitude": (["lat", "lon"], out_lat),
        "longitude": (["lat", "lon"], out_lon),
        "ct": (["time", "lat", "lon"], np.expand_dims(out_ct, axis=0)),
        "original_cloudtype": (
            ["time", "lat", "lon"],
            np.expand_dims(original_cloudtype, axis=0),
        ),
        "convcold_cloudnumber": (
            ["time", "lat", "lon"],
            np.expand_dims(convcold_cloudnumber, axis=0),
        ),
        "nclouds": (["clouds"], nclouds),
        "ncorepix": (["clouds"], ncorepix),
        "ncoldpix": (["clouds"], ncoldpix),
        "ncorecoldpix": (
            ["clouds"],
            ncorecoldpix,
        ),  # 'nwarmpix': (['time', 'clouds'], nwarmpix), \
        "final_cloudtype": (
            ["time", "lat", "lon"],
            np.expand_dims(final_cloudtype, axis=0),
        ),
    }
    # Now check for optional arguments, add them to varlist if provided
    if "precipitation" in kwargs:
        varlist["precipitation"] = (["lat", "lon"], kwargs["precipitation"])
    if "reflectivity" in kwargs:
        varlist["reflectivity"] = (["lat", "lon"], kwargs["reflectivity"])
    if "pf_number" in kwargs:
        varlist["pf_number"] = (["lat", "lon"], kwargs["pf_number"])
    if "convcold_cloudnumber_orig" in kwargs:
        varlist["convcold_cloudnumber_orig"] = (
            ["lat", "lon"],
            kwargs["convcold_cloudnumber_orig"],
        )
        # print('final_convcold_cloudnumber_orig shape: ',convcold_cloudnumber_orig.shape)
    if "cloudnumber_orig" in kwargs:
        varlist["cloudnumber_orig"] = (["lat", "lon"], kwargs["cloudnumber_orig"])
        # print('cloudnumber_orig shape: ',cloudnumber_orig.shape)

    # Define coordinate list
    coordlist = {
        "time": (["time"], file_basetime),
        "lat": (["lat"], np.squeeze(out_lat[:, 0])),
        "lon": (["lon"], np.squeeze(out_lon[0, :])),
        "clouds": (["clouds"], np.arange(1, len(nclouds) + 1)),
        "ndatechar": (["ndatechar"], np.arange(0, 32)),
        "ntimechar": (["ntimechar"], np.arange(0, 16)),
    }

    # Define global attributes
    gattrlist = {
        "title": "Statistics about convective features identified in the data from "
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
        "convections": "CF-1.6",
        "contact": "Zhe Feng <zhe.feng@pnnl.gov>",
        "created_on": time.ctime(time.time()),
        "minimum_cloud_area": area_thresh,
    }
    # Now check for optional arguments, add them to gattrlist if provided
    if "linkpf" in kwargs:
        gattrlist["linkpf"] = kwargs["linkpf"]
    if "pf_smooth_window" in kwargs:
        gattrlist["pf_smooth_window"] = kwargs["pf_smooth_window"]
    if "pf_dbz_thresh" in kwargs:
        gattrlist["pf_dbz_thresh"] = kwargs["pf_dbz_thresh"]
    if "pf_link_area_thresh" in kwargs:
        gattrlist["pf_link_area_thresh"] = kwargs["pf_link_area_thresh"]

    # Define xarray dataset
    ds_out = xr.Dataset(varlist, coords=coordlist, attrs=gattrlist)

    # Specify variable attributes
    ds_out.time.attrs[
        "long_name"
    ] = "epoch time (seconds since 01/01/1970 00:00) in epoch of file"

    ds_out.basetime.attrs[
        "long_name"
    ] = "epoch time (seconds since 01/01/1970 00:00) in epoch of file"

    ds_out.lat.attrs[
        "long_name"
    ] = "Vector of latitudes, y-coordinate in Cartesian system"
    ds_out.lat.attrs["standard_name"] = "latitude"
    ds_out.lat.attrs["units"] = "degrees_north"
    ds_out.lat.attrs["valid_min"] = geolimits[0]
    ds_out.lat.attrs["valid_max"] = geolimits[2]

    ds_out.lon.attrs[
        "long_name"
    ] = "Vector of longitudes, x-coordinate in Cartesian system"
    ds_out.lon.attrs["standard_name"] = "longitude"
    ds_out.lon.attrs["units"] = "degrees_east"
    ds_out.lon.attrs["valid_min"] = geolimits[1]
    ds_out.lon.attrs["valid_max"] = geolimits[3]

    ds_out.clouds.attrs["long_name"] = "number of distict convective cores identified"
    ds_out.clouds.attrs["units"] = "unitless"

    ds_out.ndatechar.attrs["long_name"] = "number of characters in date string"
    ds_out.ndatechar.attrs["units"] = "unitless"

    ds_out.ntimechar.attrs["long_name"] = "number of characters in time string"
    ds_out.ntimechar.attrs["units"] = "unitless"

    ds_out.basetime.attrs[
        "long_name"
    ] = "epoch time (seconds since 01/01/1970 00:00) of file"
    ds_out.basetime.attrs["standard_name"] = "time"

    ds_out.filedate.attrs["long_name"] = "date string of file (yyyymmdd)"
    ds_out.filedate.attrs["units"] = "unitless"

    ds_out.filetime.attrs["long_name"] = "time string of file (hhmm)"
    ds_out.filetime.attrs["units"] = "unitless"

    ds_out.latitude.attrs["long_name"] = "cartesian grid of latitude"
    ds_out.latitude.attrs["units"] = "degrees_north"
    ds_out.latitude.attrs["valid_min"] = geolimits[0]
    ds_out.latitude.attrs["valid_max"] = geolimits[2]

    ds_out.longitude.attrs["long_name"] = "cartesian grid of longitude"
    ds_out.longitude.attrs["units"] = "degrees_east"
    ds_out.longitude.attrs["valid_min"] = geolimits[1]
    ds_out.longitude.attrs["valid_max"] = geolimits[3]

    # ds_out.ct.attrs['long_name'] = 'ct'
    # ds_out.ct.attrs['units'] = '1-deep, 2-high congestus'

    ds_out.original_cloudtype.attrs["long_name"] = "original cloudtype"
    ds_out.original_cloudtype.attrs[
        "units"
    ] = "4-deep, 3-high congestus, 2-low congestus, 1-shallow"

    ds_out.final_cloudtype.attrs["long_name"] = "final cloudtype"
    ds_out.final_cloudtype.attrs[
        "units"
    ] = "1-deep, 2-high congestus, 3-low congestus, 4-shallow, 5-no-type"

    ds_out.convcold_cloudnumber.attrs[
        "long_name"
    ] = "grid with each classified cloud given a number"
    ds_out.convcold_cloudnumber.attrs["units"] = "unitless"
    ds_out.convcold_cloudnumber.attrs["valid_min"] = 0
    # ds_out.convcold_cloudnumber.attrs['valid_max'] = np.int(nclouds)+1
    ds_out.convcold_cloudnumber.attrs[
        "comment"
    ] = "extend of each cloud defined using cold anvil threshold"
    ds_out.convcold_cloudnumber.attrs["_FillValue"] = 0

    ds_out.nclouds.attrs[
        "long_name"
    ] = "number of distict convective cores identified in file"
    ds_out.nclouds.attrs["units"] = "unitless"

    ds_out.ncorepix.attrs[
        "long_name"
    ] = "number of convective core pixels in each cloud feature"
    ds_out.ncorepix.attrs["units"] = "unitless"

    ds_out.ncoldpix.attrs[
        "long_name"
    ] = "number of cold anvil pixels in each cloud feature"
    ds_out.ncoldpix.attrs["units"] = "unitless"

    ds_out.ncorecoldpix.attrs[
        "long_name"
    ] = "number of convective core and cold anvil pixels in each cloud feature"
    ds_out.ncorecoldpix.attrs["units"] = "unitless"

    # ds_out.nwarmpix.attrs['long_name'] = 'number of warm anvil pixels in each cloud feature'
    # ds_out.nwarmpix.attrs['units'] = 'unitless'

    # Now check for optional arguments, define attributes if provided
    if "precipitation" in kwargs:
        ds_out.precipitation.attrs["long_name"] = "Precipitation"
        ds_out.precipitation.attrs["units"] = "mm/h"
        ds_out.precipitation.attrs["_FillValue"] = np.nan
    if "reflectivity" in kwargs:
        ds_out.reflectivity.attrs["long_name"] = "Radar reflectivity"
        ds_out.reflectivity.attrs["units"] = "dBZ"
        ds_out.reflectivity.attrs["_FillValue"] = np.nan
    if "pf_number" in kwargs:
        ds_out.pf_number.attrs["long_name"] = "Precipitation Feature number"
        ds_out.pf_number.attrs["units"] = "unitless"
        ds_out.pf_number.attrs["_FillValue"] = 0
    if "convcold_cloudnumber_orig" in kwargs:
        ds_out.convcold_cloudnumber_orig.attrs[
            "long_name"
        ] = "Number of cloud system in this file that given pixel belongs to (before linked by pf_number)"
        ds_out.convcold_cloudnumber_orig.attrs["units"] = "unitless"
        ds_out.convcold_cloudnumber_orig.attrs["_FillValue"] = 0
    if "cloudnumber_orig" in kwargs:
        ds_out.cloudnumber_orig.attrs[
            "long_name"
        ] = "Number of cloud system in this file that given pixel belongs to (before linked by pf_number)"
        ds_out.cloudnumber_orig.attrs["units"] = "unitless"
        ds_out.cloudnumber_orig.attrs["_FillValue"] = 0

    # Specify encoding list
    encodelist = {
        "time": {"zlib": True, "units": "seconds since 1970-01-01"},
        "basetime": {
            "dtype": "float",
            "zlib": True,
            "units": "seconds since 1970-01-01",
        },
        "lon": {"zlib": True},
        "lon": {"zlib": True},
        "clouds": {"zlib": True},
        "filedate": {"dtype": "str", "zlib": True},
        "filetime": {"dtype": "str", "zlib": True},
        "longitude": {"zlib": True, "_FillValue": np.nan},
        "latitude": {"zlib": True, "_FillValue": np.nan},
        "ct": {"zlib": True, "_FillValue": np.nan},
        "original_cloudtype": {"zlib": True, "_FillValue": np.nan},
        "final_cloudtype": {"zlib": True, "_FillValue": np.nan},
        "convcold_cloudnumber": {"dtype": "int", "zlib": True},
        "nclouds": {"dtype": "int", "zlib": True},
        "ncorepix": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        "ncoldpix": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        "ncorecoldpix": {"dtype": "int", "zlib": True},
        # 'nwarmpix': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
    }
    # Now check for optional arguments, add them to encodelist if provided
    if "precipitation" in kwargs:
        encodelist["precipitation"] = {"zlib": True}
    if "reflectivity" in kwargs:
        encodelist["reflectivity"] = {"zlib": True}
    if "pf_number" in kwargs:
        encodelist["pf_number"] = {"zlib": True}
    if "convcold_cloudnumber_orig" in kwargs:
        encodelist["convcold_cloudnumber_orig"] = {"zlib": True}
    if "cloudnumber_orig" in kwargs:
        encodelist["cloudnumber_orig"] = {"zlib": True}

    # Write netCDF file
    print("Output cloudid file: " + cloudid_outfile)
    # ds_out.to_netcdf(path=cloudid_outfile, mode='w', format='NETCDF4_CLASSIC', unlimited_dims='time', encoding=encodelist)
    ds_out.to_netcdf(
        path=cloudid_outfile, mode="w", format="NETCDF4_CLASSIC", encoding=encodelist
    )  # KB removed unlimited_dims



