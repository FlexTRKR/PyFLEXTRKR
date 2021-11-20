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
    ncorepix,
    ncoldpix,
    ncorecoldpix,
    nwarmpix,
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
        "latitude": (["lat", "lon"], out_lat),
        "longitude": (["lat", "lon"], out_lon),
        "tb": (["time", "lat", "lon"], np.expand_dims(out_ir, axis=0)),
        "cloudtype": (["time", "lat", "lon"], cloudtype),
        "convcold_cloudnumber": (["time", "lat", "lon"], convcold_cloudnumber),
        "cloudnumber": (["time", "lat", "lon"], cloudnumber),
        "nclouds": (["time"], nclouds),
        #'ncorepix': (['clouds'], ncorepix), \
        #'ncoldpix': (['clouds'], ncoldpix), \
        #'nwarmpix': (['clouds'], nwarmpix), \
        "ncorecoldpix": (["clouds"], ncorecoldpix),
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
        "clouds": (["clouds"], np.arange(1, nclouds + 1),),
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
        'institution': 'Pacific Northwest National Laboratory', \
        'contact': 'Zhe Feng: zhe.feng@pnnl.gov', \
        "created_on": time.ctime(time.time()),
        # "cloudid_cloud_version": cloudid_version,
        "tb_threshold_core": cloudtb_threshs[0],
        "tb_threshold_coldanvil": cloudtb_threshs[1],
        "tb_threshold_warmanvil": cloudtb_threshs[2],
        "tb_threshold_environment": cloudtb_threshs[3],
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
    # ds_out.time.attrs['long_name'] = 'epoch time (seconds since 01/01/1970 00:00) in epoch of file'

    ds_out.time.attrs["long_name"] = "Base time in Epoch"
    ds_out.time.attrs["units"] = "Seconds since 1970-1-1"
    # ds_out.time.attrs['units'] = 'Seconds since 1970-1-1 0:00:00 0:00'

    ds_out.basetime.attrs["long_name"] = "Base time in Epoch"
    ds_out.basetime.attrs["units"] = "Seconds since 1970-1-1"
    # ds_out.basetime.attrs['units'] = 'Seconds since 1970-1-1 0:00:00 0:00'

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

    ds_out.clouds.attrs["long_name"] = "number of distinct convective cores identified"
    ds_out.clouds.attrs["units"] = "1"

    ds_out.latitude.attrs["long_name"] = "cartesian grid of latitude"
    ds_out.latitude.attrs["units"] = "degrees_north"
    ds_out.latitude.attrs["valid_min"] = geolimits[0]
    ds_out.latitude.attrs["valid_max"] = geolimits[2]

    ds_out.longitude.attrs["long_name"] = "cartesian grid of longitude"
    ds_out.longitude.attrs["units"] = "degrees_east"
    ds_out.longitude.attrs["valid_min"] = geolimits[1]
    ds_out.longitude.attrs["valid_max"] = geolimits[3]

    ds_out.tb.attrs["long_name"] = "brightness temperature"
    ds_out.tb.attrs["units"] = "K"
    ds_out.tb.attrs["valid_min"] = mintb_thresh
    ds_out.tb.attrs["valid_max"] = maxtb_thresh

    ds_out.cloudtype.attrs["long_name"] = "grid of cloud classifications"
    ds_out.cloudtype.attrs[
        "values"
    ] = "1 = core, 2 = cold anvil, 3 = warm anvil, 4 = other"
    ds_out.cloudtype.attrs["units"] = "unitless"
    ds_out.cloudtype.attrs["valid_min"] = 1
    ds_out.cloudtype.attrs["valid_max"] = 5
    ds_out.cloudtype.attrs["_FillValue"] = 5

    ds_out.convcold_cloudnumber.attrs[
        "long_name"
    ] = "grid with each classified cloud given a number"
    ds_out.convcold_cloudnumber.attrs["units"] = "unitless"
    ds_out.convcold_cloudnumber.attrs["valid_min"] = 0
    ds_out.convcold_cloudnumber.attrs["valid_max"] = nclouds + 1
    ds_out.convcold_cloudnumber.attrs[
        "comment"
    ] = "extend of each cloud defined using cold anvil threshold"
    ds_out.convcold_cloudnumber.attrs["_FillValue"] = 0

    ds_out.cloudnumber.attrs[
        "long_name"
    ] = "grid with each classified cloud given a number"
    ds_out.cloudnumber.attrs["units"] = "unitless"
    ds_out.cloudnumber.attrs["valid_min"] = 0
    ds_out.cloudnumber.attrs["valid_max"] = nclouds + 1
    ds_out.cloudnumber.attrs[
        "comment"
    ] = "extend of each cloud defined using warm anvil threshold"
    ds_out.cloudnumber.attrs["_FillValue"] = 0

    ds_out.nclouds.attrs[
        "long_name"
    ] = "number of distict convective cores identified in file"
    ds_out.nclouds.attrs["units"] = "unitless"

    # ds_out.ncorepix.attrs['long_name'] = 'number of convective core pixels in each cloud feature'
    # ds_out.ncorepix.attrs['units'] = 'unitless'

    # ds_out.ncoldpix.attrs['long_name'] = 'number of cold anvil pixels in each cloud feature'
    # ds_out.ncoldpix.attrs['units'] = 'unitless'

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
        "lon": {"zlib": True, "dtype":"float32"},
        "lon": {"zlib": True, "dtype":"float32"},
        "clouds": {"zlib": True},
        "longitude": {"zlib": True, "_FillValue": np.nan, "dtype":"float32"},
        "latitude": {"zlib": True, "_FillValue": np.nan, "dtype":"float32"},
        "tb": {"zlib": True, "_FillValue": np.nan},
        "cloudtype": {"zlib": True},
        "convcold_cloudnumber": {"dtype": "int", "zlib": True},
        "cloudnumber": {"dtype": "int", "zlib": True},
        "nclouds": {"dtype": "int", "zlib": True},
        #'ncorepix': {'dtype': 'int', 'zlib':True, '_FillValue': -9999},  \
        #'ncoldpix': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
        #'nwarmpix': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
        "ncorecoldpix": {"dtype":"int", "zlib":True, "_FillValue":-9999},
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
    ds_out.to_netcdf(
        path=cloudid_outfile, mode="w", format="NETCDF4", encoding=encodelist
    )


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
        "ncorecoldpix": {
            "dtype": "int",
            "zlib": True,
        },  #                         'nwarmpix': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
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


def write_trackstats_tb(
    trackstats_outfile,
    numtracks,
    maxtracklength,
    nbintb,
    numcharfilename,
    datasource,
    datadescription,
    startdate,
    enddate,
    track_version,
    tracknumbers_version,
    timegap,
    thresh_core,
    thresh_cold,
    pixel_radius,
    geolimits,
    areathresh,
    mintb_thresh,
    maxtb_thresh,
    basetime_units,
    basetime_calendar,
    finaltrack_tracklength,
    finaltrack_basetime,
    finaltrack_cloudidfile,
    finaltrack_datetimestring,
    finaltrack_corecold_meanlat,
    finaltrack_corecold_meanlon,
    finaltrack_corecold_minlat,
    finaltrack_corecold_minlon,
    finaltrack_corecold_maxlat,
    finaltrack_corecold_maxlon,
    finaltrack_corecold_radius,
    finaltrack_corecoldwarm_radius,
    finaltrack_ncorecoldpix,
    finaltrack_ncorepix,
    finaltrack_ncoldpix,
    finaltrack_nwarmpix,
    finaltrack_corecold_cloudnumber,
    finaltrack_corecold_status,
    finaltrack_corecold_startstatus,
    finaltrack_corecold_endstatus,
    adjusted_finaltrack_corecold_mergenumber,
    adjusted_finaltrack_corecold_splitnumber,
    finaltrack_corecold_trackinterruptions,
    finaltrack_corecold_boundary,
    finaltrack_corecold_mintb,
    finaltrack_corecold_meantb,
    finaltrack_core_meantb,
    finaltrack_corecold_histtb,
    finaltrack_corecold_majoraxis,
    finaltrack_corecold_orientation,
    finaltrack_corecold_eccentricity,
    finaltrack_corecold_perimeter,
    finaltrack_corecold_xcenter,
    finaltrack_corecold_ycenter,
    finaltrack_corecold_xweightedcenter,
    finaltrack_corecold_yweightedcenter,
):
    """
    Writes Tb trackstats variables to netCDF file.
    """

    # Define variable list
    varlist = {
        "lifetime": (["ntracks"], finaltrack_tracklength),
        "basetime": (["ntracks", "nmaxlength"], finaltrack_basetime),
        "cloudidfiles": (
            ["ntracks", "nmaxlength", "nfilenamechars"],
            finaltrack_cloudidfile,
        ),
        "datetimestrings": (
            ["ntracks", "nmaxlength", "ndatetimechars"],
            finaltrack_datetimestring,
        ),
        "meanlat": (["ntracks", "nmaxlength"], finaltrack_corecold_meanlat),
        "meanlon": (["ntracks", "nmaxlength"], finaltrack_corecold_meanlon),
        "minlat": (["ntracks", "nmaxlength"], finaltrack_corecold_minlat),
        "minlon": (["ntracks", "nmaxlength"], finaltrack_corecold_minlon),
        "maxlat": (["ntracks", "nmaxlength"], finaltrack_corecold_maxlat),
        "maxlon": (["ntracks", "nmaxlength"], finaltrack_corecold_maxlon),
        "radius": (["ntracks", "nmaxlength"], finaltrack_corecold_radius),
        "radius_warmanvil": (["ntracks", "nmaxlength"], finaltrack_corecoldwarm_radius),
        "npix": (["ntracks", "nmaxlength"], finaltrack_ncorecoldpix),
        "nconv": (["ntracks", "nmaxlength"], finaltrack_ncorepix),
        "ncoldanvil": (["ntracks", "nmaxlength"], finaltrack_ncoldpix),
        "nwarmanvil": (["ntracks", "nmaxlength"], finaltrack_nwarmpix),
        "cloudnumber": (["ntracks", "nmaxlength"], finaltrack_corecold_cloudnumber),
        "status": (["ntracks", "nmaxlength"], finaltrack_corecold_status),
        "startstatus": (["ntracks"], finaltrack_corecold_startstatus),
        "endstatus": (["ntracks"], finaltrack_corecold_endstatus),
        "mergenumbers": (
            ["ntracks", "nmaxlength"],
            adjusted_finaltrack_corecold_mergenumber,
        ),
        "splitnumbers": (
            ["ntracks", "nmaxlength"],
            adjusted_finaltrack_corecold_splitnumber,
        ),
        "trackinterruptions": (["ntracks"], finaltrack_corecold_trackinterruptions),
        "boundary": (["ntracks", "nmaxlength"], finaltrack_corecold_boundary),
        "mintb": (["ntracks", "nmaxlength"], finaltrack_corecold_mintb),
        "meantb": (["ntracks", "nmaxlength"], finaltrack_corecold_meantb),
        "meantb_conv": (["ntracks", "nmaxlength"], finaltrack_core_meantb),
        "histtb": (["ntracks", "nmaxlength", "nbins"], finaltrack_corecold_histtb),
        "majoraxis": (["ntracks", "nmaxlength"], finaltrack_corecold_majoraxis),
        "orientation": (["ntracks", "nmaxlength"], finaltrack_corecold_orientation),
        "eccentricity": (["ntracks", "nmaxlength"], finaltrack_corecold_eccentricity),
        "perimeter": (["ntracks", "nmaxlength"], finaltrack_corecold_perimeter),
        "xcenter": (["ntracks", "nmaxlength"], finaltrack_corecold_xcenter),
        "ycenter": (["ntracks", "nmaxlength"], finaltrack_corecold_ycenter),
        "xcenter_weighted": (
            ["ntracks", "nmaxlength"],
            finaltrack_corecold_xweightedcenter,
        ),
        "ycenter_weighted": (
            ["ntracks", "nmaxlength"],
            finaltrack_corecold_yweightedcenter,
        ),
    }

    # Define coordinate list
    coordlist = {
        "ntracks": (["ntracks"], np.arange(0, numtracks)),
        "nmaxlength": (["nmaxlength"], np.arange(0, maxtracklength)),
        "nbins": (["nbins"], np.arange(0, nbintb - 1)),
        "nfilenamechars": (["nfilenamechars"], np.arange(0, numcharfilename)),
        "ndatetimechars": (["ndatetimechars"], np.arange(0, 13)),
    }

    # Define global attributes
    gattrlist = {
        "title": "File containing statistics for each track",
        "Conventions": "CF-1.6",
        "Institution": "Pacific Northwest National Laboratoy",
        "Contact": "Katelyn Barber: katelyn.barber@pnnl.gov",
        "Created_on": time.ctime(time.time()),
        "source": datasource,
        "description": datadescription,
        "startdate": startdate,
        "enddate": enddate,
        "track_version": track_version,
        "tracknumbers_version": tracknumbers_version,
        "timegap": str(timegap) + "-hr",
        "tb_core": thresh_core,
        "tb_coldanvil": thresh_cold,
        "pixel_radius_km": pixel_radius,
    }

    # Define xarray dataset
    output_data = xr.Dataset(varlist, coords=coordlist, attrs=gattrlist)

    # Specify variable attributes
    output_data.ntracks.attrs["long_name"] = "Total number of cloud tracks"
    output_data.ntracks.attrs["units"] = "unitless"

    output_data.nmaxlength.attrs["long_name"] = "Maximum length of a cloud track"
    output_data.nmaxlength.attrs["units"] = "unitless"

    output_data.lifetime.attrs["long_name"] = "duration of each track"
    output_data.lifetime.attrs["units"] = "Temporal resolution of data"

    output_data.basetime.attrs["long_name"] = "epoch time of each cloud in a track"
    output_data.basetime.attrs["standard_name"] = "time"

    output_data.cloudidfiles.attrs[
        "long_name"
    ] = "File name for each cloud in each track"

    output_data.datetimestrings.attrs[
        "long_name"
    ] = "date_time for for each cloud in each track"

    output_data.meanlat.attrs[
        "long_name"
    ] = "Mean latitude of the core + cold anvil for each cloud in a track"
    output_data.meanlat.attrs["standard_name"] = "latitude"
    output_data.meanlat.attrs["units"] = "degrees_north"
    output_data.meanlat.attrs["valid_min"] = geolimits[1]
    output_data.meanlat.attrs["valid_max"] = geolimits[3]

    output_data.meanlon.attrs[
        "long_name"
    ] = "Mean longitude of the core + cold anvil for each cloud in a track"
    output_data.meanlon.attrs["standard_name"] = "longitude"
    output_data.meanlon.attrs["units"] = "degrees_east"
    output_data.meanlon.attrs["valid_min"] = geolimits[0]
    output_data.meanlon.attrs["valid_max"] = geolimits[2]

    output_data.minlat.attrs[
        "long_name"
    ] = "Minimum latitude of the core + cold anvil for each cloud in a track"
    output_data.minlat.attrs["standard_name"] = "latitude"
    output_data.minlat.attrs["units"] = "degrees_north"
    output_data.minlat.attrs["valid_min"] = geolimits[1]
    output_data.minlat.attrs["valid_max"] = geolimits[3]

    output_data.minlon.attrs[
        "long_name"
    ] = "Minimum longitude of the core + cold anvil for each cloud in a track"
    output_data.minlon.attrs["standard_name"] = "longitude"
    output_data.minlon.attrs["units"] = "degrees_east"
    output_data.minlon.attrs["valid_min"] = geolimits[0]
    output_data.minlon.attrs["valid_max"] = geolimits[2]

    output_data.maxlat.attrs[
        "long_name"
    ] = "Maximum latitude of the core + cold anvil for each cloud in a track"
    output_data.maxlat.attrs["standard_name"] = "latitude"
    output_data.maxlat.attrs["units"] = "degrees_north"
    output_data.maxlat.attrs["valid_min"] = geolimits[1]
    output_data.maxlat.attrs["valid_max"] = geolimits[3]

    output_data.maxlon.attrs[
        "long_name"
    ] = "Maximum longitude of the core + cold anvil for each cloud in a track"
    output_data.maxlon.attrs["standard_name"] = "longitude"
    output_data.maxlon.attrs["units"] = "degrees_east"
    output_data.maxlon.attrs["valid_min"] = geolimits[0]
    output_data.maxlon.attrs["valid_max"] = geolimits[2]

    output_data.radius.attrs[
        "long_name"
    ] = "Equivalent radius of the core + cold anvil for each cloud in a track"
    output_data.radius.attrs["standard_name"] = "Equivalent radius"
    output_data.radius.attrs["units"] = "km"
    output_data.radius.attrs["valid_min"] = areathresh

    output_data.radius_warmanvil.attrs[
        "long_name"
    ] = "Equivalent radius of the core + cold anvil  + warm anvil for each cloud in a track"
    output_data.radius_warmanvil.attrs["standard_name"] = "Equivalent radius"
    output_data.radius_warmanvil.attrs["units"] = "km"
    output_data.radius_warmanvil.attrs["valid_min"] = areathresh

    output_data.npix.attrs[
        "long_name"
    ] = "Number of pixels in the core + cold anvil for each cloud in a track"
    output_data.npix.attrs["units"] = "unitless"
    output_data.npix.attrs["valid_min"] = int(
        areathresh / float(np.square(pixel_radius))
    )

    output_data.nconv.attrs[
        "long_name"
    ] = "Number of pixels in the core for each cloud in a track"
    output_data.nconv.attrs["units"] = "unitless"
    output_data.nconv.attrs["valid_min"] = int(
        areathresh / float(np.square(pixel_radius))
    )

    output_data.ncoldanvil.attrs[
        "long_name"
    ] = "Number of pixels in the cold anvil for each cloud in a track"
    output_data.ncoldanvil.attrs["units"] = "unitless"
    output_data.ncoldanvil.attrs["valid_min"] = int(
        areathresh / float(np.square(pixel_radius))
    )

    output_data.nwarmanvil.attrs[
        "long_name"
    ] = "Number of pixels in the warm anvil for each cloud in a track"
    output_data.nwarmanvil.attrs["units"] = "unitless"
    output_data.nwarmanvil.attrs["valid_min"] = int(
        areathresh / float(np.square(pixel_radius))
    )

    output_data.cloudnumber.attrs[
        "long_name"
    ] = "Ccorresponding cloud identification number in cloudid file for each cloud in a track"
    output_data.cloudnumber.attrs["units"] = "unitless"
    output_data.cloudnumber.attrs[
        "usage"
    ] = "To link this tracking statistics file with corresponding pixel-level cloudid files, use the cloudidfile and cloudnumber together to identify which file and cloud this track is associated with at this time"

    output_data.status.attrs[
        "long_name"
    ] = "Flag indicating evolution / behavior for each cloud in a track"
    output_data.status.attrs["units"] = "unitless"
    output_data.status.attrs["valid_min"] = 0
    output_data.status.attrs["valid_max"] = 65

    output_data.startstatus.attrs[
        "long_name"
    ] = "Flag indicating how the first cloud in a track starts"
    output_data.startstatus.attrs["units"] = "unitless"
    output_data.startstatus.attrs["valid_min"] = 0
    output_data.startstatus.attrs["valid_max"] = 65

    output_data.endstatus.attrs[
        "long_name"
    ] = "Flag indicating how the last cloud in a track ends"
    output_data.endstatus.attrs["units"] = "unitless"
    output_data.endstatus.attrs["valid_min"] = 0
    output_data.endstatus.attrs["valid_max"] = 65

    output_data.trackinterruptions.attrs[
        "long_name"
    ] = "Flag indicating if track started or ended naturally or artifically due to data availability"
    output_data.trackinterruptions.attrs[
        "values"
    ] = "0 = full track available, good data. 1 = track starts at first file, track cut short by data availability. 2 = track ends at last file, track cut short by data availability"
    output_data.trackinterruptions.attrs["valid_min"] = 0
    output_data.trackinterruptions.attrs["valid_max"] = 2
    output_data.trackinterruptions.attrs["units"] = "unitless"

    output_data.mergenumbers.attrs[
        "long_name"
    ] = "Number of the track that this small cloud merges into"
    output_data.mergenumbers.attrs[
        "usuage"
    ] = "Each row represents a track. Each column represets a cloud in that track. Numbers give the track number that this small cloud merged into."
    output_data.mergenumbers.attrs["units"] = "unitless"
    output_data.mergenumbers.attrs["valid_min"] = 1
    output_data.mergenumbers.attrs["valid_max"] = numtracks

    output_data.splitnumbers.attrs[
        "long_name"
    ] = "Number of the track that this small cloud splits from"
    output_data.splitnumbers.attrs[
        "usuage"
    ] = "Each row represents a track. Each column represets a cloud in that track. Numbers give the track number that his msallcloud splits from."
    output_data.splitnumbers.attrs["units"] = "unitless"
    output_data.splitnumbers.attrs["valid_min"] = 1
    output_data.splitnumbers.attrs["valid_max"] = numtracks

    output_data.boundary.attrs[
        "long_name"
    ] = "Flag indicating whether the core + cold anvil touches one of the domain edges."
    output_data.boundary.attrs["usuage"] = " 0 = away from edge. 1= touches edge."
    output_data.boundary.attrs["units"] = "unitless"
    output_data.boundary.attrs["valid_min"] = 0
    output_data.boundary.attrs["valid_max"] = 1

    output_data.mintb.attrs[
        "long_name"
    ] = "Minimum brightness temperature for each core + cold anvil in a track"
    output_data.mintb.attrs["standard_name"] = "brightness temperature"
    output_data.mintb.attrs["units"] = "K"
    output_data.mintb.attrs["valid_min"] = mintb_thresh
    output_data.mintb.attrs["valid_max"] = maxtb_thresh

    output_data.meantb.attrs[
        "long_name"
    ] = "Mean brightness temperature for each core + cold anvil in a track"
    output_data.meantb.attrs["standard_name"] = "brightness temperature"
    output_data.meantb.attrs["units"] = "K"
    output_data.meantb.attrs["valid_min"] = mintb_thresh
    output_data.meantb.attrs["valid_max"] = maxtb_thresh

    output_data.meantb_conv.attrs[
        "long_name"
    ] = "Mean brightness temperature for each core in a track"
    output_data.meantb_conv.attrs["standard_name"] = "brightness temperature"
    output_data.meantb_conv.attrs["units"] = "K"
    output_data.meantb_conv.attrs["valid_min"] = mintb_thresh
    output_data.meantb_conv.attrs["valid_max"] = maxtb_thresh

    output_data.histtb.attrs[
        "long_name"
    ] = "Histogram of brightess of the core + cold anvil for each cloud in a track."
    output_data.histtb.attrs["standard_name"] = "Brightness temperature"
    output_data.histtb.attrs["hist_value"] = mintb_thresh
    output_data.histtb.attrs["valid_max"] = maxtb_thresh
    output_data.histtb.attrs["units"] = "K"

    output_data.orientation.attrs[
        "long_name"
    ] = "Orientation of the major axis of the core + cold anvil for each cloud in a track"
    output_data.orientation.attrs["units"] = "Degrees clockwise from vertical"
    output_data.orientation.attrs["valid_min"] = 0
    output_data.orientation.attrs["valid_max"] = 360

    output_data.eccentricity.attrs[
        "long_name"
    ] = "Eccentricity of the major axis of the core + cold anvil for each cloud in a track"
    output_data.eccentricity.attrs["units"] = "unitless"
    output_data.eccentricity.attrs["valid_min"] = 0
    output_data.eccentricity.attrs["valid_max"] = 1

    output_data.majoraxis.attrs[
        "long_name"
    ] = "Length of the major axis of the core + cold anvil for each cloud in a track"
    output_data.majoraxis.attrs["units"] = "km"

    output_data.perimeter.attrs[
        "long_name"
    ] = "Approximnate circumference of the core + cold anvil for each cloud in a track"
    output_data.perimeter.attrs["units"] = "km"

    output_data.xcenter.attrs[
        "long_name"
    ] = "X index of the geometric center of the cloud feature for each cloud in a track"
    output_data.xcenter.attrs["units"] = "unitless"

    output_data.ycenter.attrs[
        "long_name"
    ] = "Y index of the geometric center of the cloud feature for each cloud in a track"
    output_data.ycenter.attrs["units"] = "unitless"

    output_data.xcenter_weighted.attrs[
        "long_name"
    ] = "X index of the brightness temperature weighted center of the cloud feature for each cloud in a track"
    output_data.xcenter_weighted.attrs["units"] = "unitless"

    output_data.ycenter_weighted.attrs[
        "long_name"
    ] = "Y index of the brightness temperature weighted center of the cloud feature for each cloud in a track"
    output_data.ycenter_weighted.attrs["units"] = "unitless"

    # Specify encoding list
    encodelist = {
        "lifetime": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        "basetime": {
            "zlib": True,
            "units": basetime_units,
            "calendar": basetime_calendar,
        },
        "ntracks": {"dtype": "int", "zlib": True},
        "nmaxlength": {"dtype": "int", "zlib": True},
        "cloudidfiles": {"zlib": True},
        "datetimestrings": {"zlib": True},
        "meanlat": {"zlib": True, "_FillValue": np.nan},
        "meanlon": {"zlib": True, "_FillValue": np.nan},
        "minlat": {"zlib": True, "_FillValue": np.nan},
        "minlon": {"zlib": True, "_FillValue": np.nan},
        "maxlat": {"zlib": True, "_FillValue": np.nan},
        "maxlon": {"zlib": True, "_FillValue": np.nan},
        "radius": {"zlib": True, "_FillValue": np.nan},
        "radius_warmanvil": {"zlib": True, "_FillValue": np.nan},
        "boundary": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        "npix": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        "nconv": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        "ncoldanvil": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        "nwarmanvil": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        "cloudnumber": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        "mergenumbers": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        "splitnumbers": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        "status": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        "startstatus": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        "endstatus": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        "trackinterruptions": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        "mintb": {"zlib": True, "_FillValue": np.nan},
        "meantb": {"zlib": True, "_FillValue": np.nan},
        "meantb_conv": {"zlib": True, "_FillValue": np.nan},
        "histtb": {"dtype": "int", "zlib": True},
        "majoraxis": {"zlib": True, "_FillValue": np.nan},
        "orientation": {"zlib": True, "_FillValue": np.nan},
        "eccentricity": {"zlib": True, "_FillValue": np.nan},
        "perimeter": {"zlib": True, "_FillValue": np.nan},
        "xcenter": {"zlib": True, "_FillValue": -9999},
        "ycenter": {"zlib": True, "_FillValue": -9999},
        "xcenter_weighted": {"zlib": True, "_FillValue": -9999},
        "ycenter_weighted": {"zlib": True, "_FillValue": -9999},
    }

    # Write netcdf file
    output_data.to_netcdf(
        path=trackstats_outfile,
        mode="w",
        format="NETCDF4_CLASSIC",
        unlimited_dims="ntracks",
        encoding=encodelist,
    )


def write_trackstats_ct(
    trackstats_outfile,
    numtracks,
    maxtracklength,
    numcharfilename,
    datasource,
    datadescription,
    startdate,
    enddate,
    track_version,
    tracknumbers_version,
    timegap,
    pixel_radius,
    geolimits,
    areathresh,
    basetime_units,
    basetime_calendar,
    finaltrack_tracklength,
    finaltrack_basetime,
    finaltrack_cloudidfile,
    finaltrack_datetimestring,
    finaltrack_corecold_meanlat,
    finaltrack_corecold_meanlon,
    finaltrack_corecold_minlat,
    finaltrack_corecold_minlon,
    finaltrack_corecold_maxlat,
    finaltrack_corecold_maxlon,
    finaltrack_corecold_radius,
    finaltrack_ncorecoldpix,
    finaltrack_ncorepix,
    finaltrack_ncoldpix,
    finaltrack_corecold_cloudnumber,
    finaltrack_corecold_status,
    finaltrack_corecold_startstatus,
    finaltrack_corecold_endstatus,
    adjusted_finaltrack_corecold_mergenumber,
    adjusted_finaltrack_corecold_splitnumber,
    finaltrack_corecold_trackinterruptions,
    finaltrack_corecold_boundary,  # finaltrack_corecold_majoraxis, finaltrack_corecold_orientation, finaltrack_corecold_eccentricity, \
    # finaltrack_corecold_perimeter, finaltrack_corecold_xcenter, finaltrack_corecold_ycenter, \
    # finaltrack_corecold_xweightedcenter, finaltrack_corecold_yweightedcenter, \
    finaltrack_cloudtype_low,
    finaltrack_cloudtype_conglow,
    finaltrack_cloudtype_conghigh,
    finaltrack_cloudtype_deep,
):

    """
    Writes Tb cloudtype trackstats variables to netCDF file.
    """

    # Define variable list
    varlist = {
        "lifetime": (["ntracks"], finaltrack_tracklength),
        "basetime": (["ntracks", "nmaxlength"], finaltrack_basetime),
        "cloudidfiles": (
            ["ntracks", "nmaxlength", "nfilenamechars"],
            finaltrack_cloudidfile,
        ),
        "datetimestrings": (
            ["ntracks", "nmaxlength", "ndatetimechars"],
            finaltrack_datetimestring,
        ),
        "meanlat": (["ntracks", "nmaxlength"], finaltrack_corecold_meanlat),
        "meanlon": (["ntracks", "nmaxlength"], finaltrack_corecold_meanlon),
        "minlat": (["ntracks", "nmaxlength"], finaltrack_corecold_minlat),
        "minlon": (["ntracks", "nmaxlength"], finaltrack_corecold_minlon),
        "maxlat": (["ntracks", "nmaxlength"], finaltrack_corecold_maxlat),
        "maxlon": (["ntracks", "nmaxlength"], finaltrack_corecold_maxlon),
        "radius": (["ntracks", "nmaxlength"], finaltrack_corecold_radius),
        "npix": (["ntracks", "nmaxlength"], finaltrack_ncorecoldpix),
        "nconv": (["ntracks", "nmaxlength"], finaltrack_ncorepix),
        "ncoldanvil": (["ntracks", "nmaxlength"], finaltrack_ncoldpix),
        "cloudnumber": (["ntracks", "nmaxlength"], finaltrack_corecold_cloudnumber),
        "status": (["ntracks", "nmaxlength"], finaltrack_corecold_status),
        "startstatus": (["ntracks"], finaltrack_corecold_startstatus),
        "endstatus": (["ntracks"], finaltrack_corecold_endstatus),
        "mergenumbers": (
            ["ntracks", "nmaxlength"],
            adjusted_finaltrack_corecold_mergenumber,
        ),
        "splitnumbers": (
            ["ntracks", "nmaxlength"],
            adjusted_finaltrack_corecold_splitnumber,
        ),
        "trackinterruptions": (["ntracks"], finaltrack_corecold_trackinterruptions),
        "boundary": (
            ["ntracks", "nmaxlength"],
            finaltrack_corecold_boundary,
        ),  #'majoraxis': (['ntracks', 'nmaxlength'], finaltrack_corecold_majoraxis), \
        #'orientation': (['ntracks', 'nmaxlength'], finaltrack_corecold_orientation), \
        #'eccentricity': (['ntracks', 'nmaxlength'], finaltrack_corecold_eccentricity), \
        #'perimeter': (['ntracks', 'nmaxlength'], finaltrack_corecold_perimeter), \
        #'xcenter': (['ntracks', 'nmaxlength'], finaltrack_corecold_xcenter), \
        #'ycenter': (['ntracks', 'nmaxlength'], finaltrack_corecold_ycenter), \
        #'xcenter_weighted': (['ntracks', 'nmaxlength'], finaltrack_corecold_xweightedcenter), \
        #'ycenter_weighted': (['ntracks', 'nmaxlength'], finaltrack_corecold_yweightedcenter), \
        "cloudtype_npix_low": (["ntracks", "nmaxlength"], finaltrack_cloudtype_low),
        "cloudtype_npix_conglow": (
            ["ntracks", "nmaxlength"],
            finaltrack_cloudtype_conglow,
        ),
        "cloudtype_npix_conghigh": (
            ["ntracks", "nmaxlength"],
            finaltrack_cloudtype_conghigh,
        ),
        "cloudtype_npix_deep": (["ntracks", "nmaxlength"], finaltrack_cloudtype_deep),
    }

    # Define coordinate list
    coordlist = {
        "ntracks": (["ntracks"], np.arange(0, numtracks)),
        "nmaxlength": (["nmaxlength"], np.arange(0, maxtracklength)),
        "nfilenamechars": (["nfilenamechars"], np.arange(0, numcharfilename)),
        "ndatetimechars": (["ndatetimechars"], np.arange(0, 13)),
    }

    # Define global attributes
    gattrlist = {
        "title": "File containing statistics for each track",
        "Conventions": "CF-1.6",
        "Institution": "Pacific Northwest National Laboratoy",
        "Contact": "Katelyn Barber: katelyn.barber@pnnl.gov",
        "Created_on": time.ctime(time.time()),
        "source": datasource,
        "description": datadescription,
        "startdate": startdate,
        "enddate": enddate,
        "track_version": track_version,
        "tracknumbers_version": tracknumbers_version,
        "timegap": str(timegap) + "-hr",
        "pixel_radius_km": pixel_radius,
    }

    # Define xarray dataset
    output_data = xr.Dataset(varlist, coords=coordlist, attrs=gattrlist)

    # Specify variable attributes
    output_data.ntracks.attrs["long_name"] = "Total number of cloud tracks"
    output_data.ntracks.attrs["units"] = "unitless"

    output_data.nmaxlength.attrs["long_name"] = "Maximum length of a cloud track"
    output_data.nmaxlength.attrs["units"] = "unitless"

    output_data.lifetime.attrs["long_name"] = "duration of each track"
    output_data.lifetime.attrs["units"] = "Temporal resolution of data"

    output_data.basetime.attrs["long_name"] = "epoch time of each cloud in a track"
    output_data.basetime.attrs["standard_name"] = "time"

    output_data.cloudidfiles.attrs[
        "long_name"
    ] = "File name for each cloud in each track"

    output_data.datetimestrings.attrs[
        "long_name"
    ] = "date_time for for each cloud in each track"

    output_data.meanlat.attrs[
        "long_name"
    ] = "Mean latitude of the core + cold anvil for each cloud in a track"
    output_data.meanlat.attrs["standard_name"] = "latitude"
    output_data.meanlat.attrs["units"] = "degrees_north"
    output_data.meanlat.attrs["valid_min"] = geolimits[1]
    output_data.meanlat.attrs["valid_max"] = geolimits[3]

    output_data.meanlon.attrs[
        "long_name"
    ] = "Mean longitude of the core + cold anvil for each cloud in a track"
    output_data.meanlon.attrs["standard_name"] = "longitude"
    output_data.meanlon.attrs["units"] = "degrees_east"
    output_data.meanlon.attrs["valid_min"] = geolimits[0]
    output_data.meanlon.attrs["valid_max"] = geolimits[2]

    output_data.minlat.attrs[
        "long_name"
    ] = "Minimum latitude of the core + cold anvil for each cloud in a track"
    output_data.minlat.attrs["standard_name"] = "latitude"
    output_data.minlat.attrs["units"] = "degrees_north"
    output_data.minlat.attrs["valid_min"] = geolimits[1]
    output_data.minlat.attrs["valid_max"] = geolimits[3]

    output_data.minlon.attrs[
        "long_name"
    ] = "Minimum longitude of the core + cold anvil for each cloud in a track"
    output_data.minlon.attrs["standard_name"] = "longitude"
    output_data.minlon.attrs["units"] = "degrees_east"
    output_data.minlon.attrs["valid_min"] = geolimits[0]
    output_data.minlon.attrs["valid_max"] = geolimits[2]

    output_data.maxlat.attrs[
        "long_name"
    ] = "Maximum latitude of the core + cold anvil for each cloud in a track"
    output_data.maxlat.attrs["standard_name"] = "latitude"
    output_data.maxlat.attrs["units"] = "degrees_north"
    output_data.maxlat.attrs["valid_min"] = geolimits[1]
    output_data.maxlat.attrs["valid_max"] = geolimits[3]

    output_data.maxlon.attrs[
        "long_name"
    ] = "Maximum longitude of the core + cold anvil for each cloud in a track"
    output_data.maxlon.attrs["standard_name"] = "longitude"
    output_data.maxlon.attrs["units"] = "degrees_east"
    output_data.maxlon.attrs["valid_min"] = geolimits[0]
    output_data.maxlon.attrs["valid_max"] = geolimits[2]

    output_data.radius.attrs[
        "long_name"
    ] = "Equivalent radius of the core + cold anvil for each cloud in a track"
    output_data.radius.attrs["standard_name"] = "Equivalent radius"
    output_data.radius.attrs["units"] = "km"
    output_data.radius.attrs["valid_min"] = areathresh

    output_data.npix.attrs[
        "long_name"
    ] = "Number of pixels in the core + cold anvil for each cloud in a track"
    output_data.npix.attrs["units"] = "unitless"
    output_data.npix.attrs["valid_min"] = int(
        areathresh / float(np.square(pixel_radius))
    )

    output_data.nconv.attrs[
        "long_name"
    ] = "Number of pixels in the core for each cloud in a track"
    output_data.nconv.attrs["units"] = "unitless"
    output_data.nconv.attrs["valid_min"] = int(
        areathresh / float(np.square(pixel_radius))
    )

    output_data.ncoldanvil.attrs[
        "long_name"
    ] = "Number of pixels in the cold anvil for each cloud in a track"
    output_data.ncoldanvil.attrs["units"] = "unitless"
    output_data.ncoldanvil.attrs["valid_min"] = int(
        areathresh / float(np.square(pixel_radius))
    )

    output_data.cloudnumber.attrs[
        "long_name"
    ] = "Ccorresponding cloud identification number in cloudid file for each cloud in a track"
    output_data.cloudnumber.attrs["units"] = "unitless"
    output_data.cloudnumber.attrs[
        "usage"
    ] = "To link this tracking statistics file with corresponding pixel-level cloudid files, use the cloudidfile and cloudnumber together to identify which file and cloud this track is associated with at this time"

    output_data.status.attrs[
        "long_name"
    ] = "Flag indicating evolution / behavior for each cloud in a track"
    output_data.status.attrs["units"] = "unitless"
    output_data.status.attrs["valid_min"] = 0
    output_data.status.attrs["valid_max"] = 65

    output_data.startstatus.attrs[
        "long_name"
    ] = "Flag indicating how the first cloud in a track starts"
    output_data.startstatus.attrs["units"] = "unitless"
    output_data.startstatus.attrs["valid_min"] = 0
    output_data.startstatus.attrs["valid_max"] = 65

    output_data.endstatus.attrs[
        "long_name"
    ] = "Flag indicating how the last cloud in a track ends"
    output_data.endstatus.attrs["units"] = "unitless"
    output_data.endstatus.attrs["valid_min"] = 0
    output_data.endstatus.attrs["valid_max"] = 65

    output_data.trackinterruptions.attrs[
        "long_name"
    ] = "Flag indicating if track started or ended naturally or artifically due to data availability"
    output_data.trackinterruptions.attrs[
        "values"
    ] = "0 = full track available, good data. 1 = track starts at first file, track cut short by data availability. 2 = track ends at last file, track cut short by data availability"
    output_data.trackinterruptions.attrs["valid_min"] = 0
    output_data.trackinterruptions.attrs["valid_max"] = 2
    output_data.trackinterruptions.attrs["units"] = "unitless"

    output_data.mergenumbers.attrs[
        "long_name"
    ] = "Number of the track that this small cloud merges into"
    output_data.mergenumbers.attrs[
        "usuage"
    ] = "Each row represents a track. Each column represets a cloud in that track. Numbers give the track number that this small cloud merged into."
    output_data.mergenumbers.attrs["units"] = "unitless"
    output_data.mergenumbers.attrs["valid_min"] = 1
    output_data.mergenumbers.attrs["valid_max"] = numtracks

    output_data.splitnumbers.attrs[
        "long_name"
    ] = "Number of the track that this small cloud splits from"
    output_data.splitnumbers.attrs[
        "usuage"
    ] = "Each row represents a track. Each column represets a cloud in that track. Numbers give the track number that his msallcloud splits from."
    output_data.splitnumbers.attrs["units"] = "unitless"
    output_data.splitnumbers.attrs["valid_min"] = 1
    output_data.splitnumbers.attrs["valid_max"] = numtracks

    output_data.boundary.attrs[
        "long_name"
    ] = "Flag indicating whether the core + cold anvil touches one of the domain edges."
    output_data.boundary.attrs["usuage"] = " 0 = away from edge. 1= touches edge."
    output_data.boundary.attrs["units"] = "unitless"
    output_data.boundary.attrs["valid_min"] = 0
    output_data.boundary.attrs["valid_max"] = 1

    # output_data.orientation.attrs['long_name'] = 'Orientation of the major axis of the core + cold anvil for each cloud in a track'
    # output_data.orientation.attrs['units'] = 'Degrees clockwise from vertical'
    # output_data.orientation.attrs['valid_min'] = 0
    # output_data.orientation.attrs['valid_max'] = 360

    # output_data.eccentricity.attrs['long_name'] = 'Eccentricity of the major axis of the core + cold anvil for each cloud in a track'
    # output_data.eccentricity.attrs['units'] = 'unitless'
    # output_data.eccentricity.attrs['valid_min'] = 0
    # output_data.eccentricity.attrs['valid_max'] = 1

    # output_data.majoraxis.attrs['long_name'] =  'Length of the major axis of the core + cold anvil for each cloud in a track'
    # output_data.majoraxis.attrs['units'] = 'km'

    # output_data.perimeter.attrs['long_name'] = 'Approximnate circumference of the core + cold anvil for each cloud in a track'
    # output_data.perimeter.attrs['units'] = 'km'

    # output_data.xcenter.attrs['long_name'] = 'X index of the geometric center of the cloud feature for each cloud in a track'
    # output_data.xcenter.attrs['units'] = 'unitless'

    # output_data.ycenter.attrs['long_name'] = 'Y index of the geometric center of the cloud feature for each cloud in a track'
    # output_data.ycenter.attrs['units'] = 'unitless'

    # output_data.xcenter_weighted.attrs['long_name'] = 'X index of the brightness temperature weighted center of the cloud feature for each cloud in a track'
    # output_data.xcenter_weighted.attrs['units'] = 'unitless'

    # output_data.ycenter_weighted.attrs['long_name'] = 'Y index of the brightness temperature weighted center of the cloud feature for each cloud in a track'
    # output_data.ycenter_weighted.attrs['units'] = 'unitless'

    output_data.cloudtype_npix_low.attrs[
        "long_name"
    ] = "Number of pixels labeled as low cloud in a track"
    output_data.cloudtype_npix_low.attrs["units"] = "unitless"

    output_data.cloudtype_npix_conglow.attrs[
        "long_name"
    ] = "Number of pixels labeled as low congestus cloud in a track"
    output_data.cloudtype_npix_conglow.attrs["units"] = "unitless"

    output_data.cloudtype_npix_conghigh.attrs[
        "long_name"
    ] = "Number of pixels labeled as high congestus cloud in a track"
    output_data.cloudtype_npix_conghigh.attrs["units"] = "unitless"

    output_data.cloudtype_npix_deep.attrs[
        "long_name"
    ] = "Number of pixels labeled as deep cloud in a track"
    output_data.cloudtype_npix_deep.attrs["units"] = "unitless"

    # Specify encoding list
    encodelist = {
        "lifetime": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        "basetime": {
            "zlib": True,
            "units": basetime_units,
            "calendar": basetime_calendar,
        },
        "ntracks": {"dtype": "int", "zlib": True},
        "nmaxlength": {"dtype": "int", "zlib": True},
        "cloudidfiles": {"zlib": True},
        "datetimestrings": {"zlib": True},
        "meanlat": {"zlib": True, "_FillValue": np.nan},
        "meanlon": {"zlib": True, "_FillValue": np.nan},
        "minlat": {"zlib": True, "_FillValue": np.nan},
        "minlon": {"zlib": True, "_FillValue": np.nan},
        "maxlat": {"zlib": True, "_FillValue": np.nan},
        "maxlon": {"zlib": True, "_FillValue": np.nan},
        "radius": {"zlib": True, "_FillValue": np.nan},
        "boundary": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        "npix": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        "nconv": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        "ncoldanvil": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        "cloudnumber": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        "mergenumbers": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        "splitnumbers": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        "status": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        "startstatus": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        "endstatus": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        "trackinterruptions": {
            "dtype": "int",
            "zlib": True,
            "_FillValue": -9999,
        },  #'majoraxis': {'zlib':True, '_FillValue': np.nan}, \
        #'orientation': {'zlib':True, '_FillValue': np.nan}, \
        #'eccentricity': {'zlib':True, '_FillValue': np.nan}, \
        #'perimeter': {'zlib':True, '_FillValue': np.nan}, \
        #'xcenter': {'zlib':True, '_FillValue': -9999}, \
        #'ycenter': {'zlib':True, '_FillValue': -9999}, \
        #'xcenter_weighted': {'zlib':True, '_FillValue': -9999}, \
        #'ycenter_weighted': {'zlib':True, '_FillValue': -9999}, \
        "cloudtype_npix_low": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        "cloudtype_npix_conglow": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        "cloudtype_npix_conghigh": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        "cloudtype_npix_deep": {"dtype": "int", "zlib": True, "_FillValue": -9999},
    }

    # Write netcdf file
    print("Here I am with file: ", trackstats_outfile)
    output_data.to_netcdf(
        path=trackstats_outfile,
        mode="w",
        format="NETCDF4_CLASSIC",
        unlimited_dims="ntracks",
        encoding=encodelist,
    )
