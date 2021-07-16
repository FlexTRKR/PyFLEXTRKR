# def idclouds_gpmmergir(zipped_inputs):
def idclouds_gpmmergir(
    datafilepath,
    datasource,
    datadescription,
    variablename,
    cloudid_version,
    dataoutpath,
    latlon_file,
    geolimits,
    startdate,
    enddate,
    pixel_radius,
    area_thresh,
    cloudtb_threshs,
    absolutetb_threshs,
    miss_thresh,
    cloudidmethod,
    mincoldcorepix,
    smoothsize,
    warmanvilexpansion,
    idclouds_hourly,
    idclouds_minute,
    linkpf,
    pf_smooth_window,
    pf_dbz_thresh,
    pf_link_area_thresh,
    pfvarname,
    **kwargs,
):
    """
    Identifies convective cloud objects from GPM MERGIR global satellite data.

    Arguments:
    datafilepath - path to raw data directory
    datafiledatestring - string with year, month, and day of data
    datafiletimestring - string with the hour and minute of thedata
    datafilebase - header for the raw data file
    datasource - source of the raw data
    datadescription - description of data source, included in all output file names
    variablename - name of tb data in raw data file
    cloudid_version - version of cloud identification being run, set at the start of the beginning of run_test.py
    dataoutpath - path to destination of the output
    latlon_file - filename of the file that contains the latitude and longitude data
    latname - name of latitude variable in raw data file
    longname - name of longitude variable in raw data file
    geolimits - 4-element array with plotting boundaries [lat_min, lon_min, lat_max, lon_max]
    startdate - data to start processing in yyyymmdd format
    enddate - data to stop processing in yyyymmdd format
    pixel_radius - radius of pixels in km
    area_thresh - minimum area thershold to define a feature in km^2
    cloudtb_threshs - brightness temperature thresholds
    miss_thresh - minimum amount of data required in order for the file to not to be considered corrupt.
    cloudidmethod - flag indiciating which method of cloud classification will be used
    mincoldcorepix - minimum size threshold for a cloud
    smoothsize - how many pixels to dilate as growing the warm (and cold) anvil. only used for futyan4.
    warmanvilexpansion - flag indicating whether to grow the warm anvil or ignore this step. The warm anvil is not used in tracking.
    processhalfhour
    idclouds_hourly,
    idclouds_minute,
    linkpf,
    pf_smooth_window,
    pf_dbz_thresh,
    pf_link_area_thresh,
    pfvarname
    """
    ##########################################################
    # Load modules

    from netCDF4 import Dataset
    import os
    import logging
    import numpy as np
    import xarray as xr
    from scipy.signal import medfilt2d
    from scipy.ndimage import label, filters
    from pyflextrkr import netcdf_io as net

    np.set_printoptions(threshold=np.inf)
    logger = logging.getLogger(__name__)

    ##########################################################
    # Separate inputs
    # datafilepath = zipped_inputs[0]
    # datasource = zipped_inputs[1]
    # datadescription = zipped_inputs[1]
    # variablename = zipped_inputs[2]
    # cloudid_version = zipped_inputs[3]
    # dataoutpath = zipped_inputs[4]
    # latlon_file = zipped_inputs[5]
    # geolimits = zipped_inputs[6]
    # startdate = zipped_inputs[7]
    # enddate = zipped_inputs[8]
    # pixel_radius = zipped_inputs[9]
    # area_thresh = zipped_inputs[10]
    # cloudtb_threshs = zipped_inputs[11]
    # absolutetb_threshs = zipped_inputs[12]
    # miss_thresh = zipped_inputs[13]
    # cloudidmethod = zipped_inputs[14]
    # mincoldcorepix = zipped_inputs[15]
    # smoothsize = zipped_inputs[16]
    # warmanvilexpansion = zipped_inputs[17]
    # idclouds_hourly = zipped_inputs[18]
    # idclouds_minute = zipped_inputs[19]
    # linkpf = zipped_inputs[20]
    # pf_smooth_window = zipped_inputs[21]
    # pf_dbz_thresh = zipped_inputs[22]
    # pf_link_area_thresh = zipped_inputs[23]
    # pfvarname = zipped_inputs[24]

    # define constants
    # Set medfilt2d kernel_size, this determines the filter window dimension
    if "medfiltsize" in kwargs:
        medfiltsize = kwargs["medfiltsize"]
    else:
        medfiltsize = 5
    # Default idclouds minute difference allowed
    if "idclouds_minute_diff" in kwargs:
        idclouds_minute_diff = kwargs["idclouds_minute_diff"]
    else:
        idclouds_minute_diff = 5

    # minimum and maximum brightness temperature thresholds. data outside of this range is filtered
    mintb_thresh = absolutetb_threshs[0]  # k
    maxtb_thresh = absolutetb_threshs[1]  # k

    # Get Tb thresholds
    thresh_core = cloudtb_threshs[0]  # Convective core threshold [K]
    thresh_cold = cloudtb_threshs[1]  # Cold anvil threshold [K]
    thresh_warm = cloudtb_threshs[2]  # Warm anvil threshold [K]
    thresh_cloud = cloudtb_threshs[3]  # Warmest cloud area threshold [K]

    ########################################################
    # load data:
    # geolocation data
    # if ((datasource == 'gpmmergir') | (datasource == 'gpmirimerg')):
    #     geolocation_data = xr.open_dataset(latlon_file)      # open file
    #     lat = geolocation_data['lat'].data                             # load latitude data
    #     lon = geolocation_data['lon'].data                            # load longitude data
    #     # Mesh 1D grid into 2D
    #     in_lon, in_lat = np.meshgrid(lon, lat)
    #     geolocation_data.close()

    # Brightness temperature data.
    # get date and time of each file. file name formate is "irdata_yyyymmdd_hhmm.nc" thus yyyymmdd=7:15, hhmm=16:20
    if (datasource == "gpmmergir") | (datasource == "gpmirimerg"):
        logger.info(datafilepath)

        # load brighttness temperature data. automatically removes missing values
        # rawdata = xr.open_dataset(datafilepath)                            # open file
        # original_time = rawdata['time']
        # original_ir = rawdata[variablename].data                                           # load brightness temperature data
        # rawdata.close()

        # rawdata = Dataset(datafilepath, 'r')
        # lat = rawdata['lat'][:]
        # lon = rawdata['lon'][:]
        # original_time = rawdata['time'][:]
        # basetime_units = rawdata['time'].units
        # original_ir = rawdata[variablename][:]
        # rawdata.close()

        # Read in data using xarray
        rawdata = xr.open_dataset(datafilepath)
        lat = rawdata["lat"].values
        lon = rawdata["lon"].values
        time_decode = rawdata["time"]
        original_ir = rawdata[variablename].values
        rawdata.close()

        # Mesh 1D grid into 2D
        in_lon, in_lat = np.meshgrid(lon, lat)

        # Loop over each time
        # for tt in range(0, len(original_time)):
        for tt in range(0, len(time_decode)):

            # Get the data time
            # iTime = original_time[tt]
            # iminute = iTime.dt.minute.data

            iTime = time_decode[tt]
            file_basetime = np.array([iTime.values.tolist() / 1e9])
            file_datestring = iTime.dt.strftime("%Y%m%d").item()
            file_timestring = iTime.dt.strftime("%H%M").item()
            iminute = iTime.dt.minute.item()

            # Convert basetime to strings
            # iTime = original_time[tt]
            # file_basetime = np.array([pd.to_datetime(num2date(iTime, units=basetime_units))], dtype='datetime64[s]')
            # file_bt = file_basetime.item()  # Get the datetime64 value from numpy array
            # file_datestring = file_bt.strftime("%Y") + file_bt.strftime("%m") + file_bt.strftime("%d")
            # file_timestring = file_bt.strftime("%H") + file_bt.strftime("%M")
            # iminute = float(file_bt.strftime("%M"))
            # import pdb; pdb.set_trace()

            # If idclouds_hourly is set to 1, then check if iminutes is within the allowed difference from idclouds_minutes
            # If so proceed, otherwise, skip this time
            if idclouds_hourly == 1:
                if np.absolute(iminute - idclouds_minute) < idclouds_minute_diff:
                    idclouds_proceed = 1
                else:
                    idclouds_proceed = 0
            else:
                idclouds_proceed = 1

            # Proceed to idclodus if flag is 1
            if idclouds_proceed == 1:
                in_ir = original_ir[tt, :, :]

                # Use median filter to fill in missing values
                ir_filt = medfilt2d(in_ir, kernel_size=medfiltsize)
                # Copy the original IR data
                out_ir = np.copy(in_ir)
                # Create a mask for the missing pixels
                missmask = np.isnan(in_ir)
                # Fill in the missing pixels with the filtered values, retain the rest
                out_ir[missmask] = ir_filt[missmask]

                # in_lat = np.transpose(in_lat)
                # in_lon = np.transpose(in_lon)
                # in_ir = np.transpose(in_ir)

                #####################################################
                # mask brightness temperatures outside of normal range
                in_ir[in_ir < mintb_thresh] = np.nan
                in_ir[in_ir > maxtb_thresh] = np.nan

                #####################################################
                # determine geographic region of interest is within the data set. if it is proceed and limit the data to that geographic region. if not exit the code.

                # isolate data within lat/lon range set by limit
                indicesy, indicesx = np.array(
                    np.where(
                        (in_lat >= geolimits[0])
                        & (in_lat <= geolimits[2])
                        & (in_lon >= geolimits[1])
                        & (in_lon <= geolimits[3])
                    )
                )

                # proceed if file covers the geographic region in interest
                if (len(indicesx) > 0) and (len(indicesy) > 0):
                    out_lat = np.copy(
                        in_lat[
                            np.nanmin(indicesy) : np.nanmax(indicesy) + 1,
                            np.nanmin(indicesx) : np.nanmax(indicesx) + 1,
                        ]
                    )
                    out_lon = np.copy(
                        in_lon[
                            np.nanmin(indicesy) : np.nanmax(indicesy) + 1,
                            np.nanmin(indicesx) : np.nanmax(indicesx) + 1,
                        ]
                    )
                    out_ir = np.copy(
                        in_ir[
                            np.nanmin(indicesy) : np.nanmax(indicesy) + 1,
                            np.nanmin(indicesx) : np.nanmax(indicesx) + 1,
                        ]
                    )

                    ######################################################
                    # proceed only if number of missing data does not exceed an accepable threshold
                    # determine number of missing data
                    missingcount = len(np.array(np.where(np.isnan(out_ir)))[0, :])
                    ny, nx = np.shape(out_ir)

                    if np.divide(missingcount, (ny * nx)) < miss_thresh:
                        ######################################################

                        # file_basetime = np.array([iTime.data], dtype='datetime64[ns]')
                        # file_datestring = iTime.dt.strftime("%Y%m%d").data.item()
                        # file_timestring = iTime.dt.strftime("%H%M").data.item()
                        # logger.info(file_basetime)

                        # call idclouds subroutine
                        if cloudidmethod == "futyan3":
                            from pyflextrkr.futyan3 import futyan3

                            clouddata = futyan3(
                                out_ir,
                                pixel_radius,
                                cloudtb_threshs,
                                area_thresh,
                                warmanvilexpansion,
                            )
                        elif cloudidmethod == "futyan4":
                            from pyflextrkr.label_and_grow_cold_clouds import (
                                label_and_grow_cold_clouds,
                            )

                            clouddata = label_and_grow_cold_clouds(
                                out_ir,
                                pixel_radius,
                                cloudtb_threshs,
                                area_thresh,
                                mincoldcorepix,
                                smoothsize,
                                warmanvilexpansion,
                            )

                        ######################################################
                        # separate output from futyan into the separate variables
                        final_nclouds = np.array([clouddata["final_nclouds"]])
                        final_ncorepix = clouddata["final_ncorepix"]
                        final_ncoldpix = clouddata["final_ncoldpix"]
                        final_ncorecoldpix = clouddata["final_ncorecoldpix"]
                        final_nwarmpix = clouddata["final_nwarmpix"]
                        # final_ncorepix = np.array([clouddata['final_ncorepix']])
                        # final_ncoldpix = np.array([clouddata['final_ncoldpix']])
                        # final_ncorecoldpix = np.array([clouddata['final_ncorecoldpix']])
                        # final_nwarmpix = np.array([clouddata['final_nwarmpix']])
                        final_cloudtype = np.array([clouddata["final_cloudtype"]])
                        final_cloudnumber = np.array([clouddata["final_cloudnumber"]])
                        final_convcold_cloudnumber = np.array(
                            [clouddata["final_convcold_cloudnumber"]]
                        )
                        # import pdb; pdb.set_trace()

                        # Option to linkpf
                        if linkpf == 1:
                            from pyflextrkr.ftfunctions import (
                                sort_renumber,
                                sort_renumber2vars,
                                link_pf_tb,
                            )

                            # Proceed if there is at least 1 cloud
                            if final_nclouds > 0:
                                # Read PF from idcloudfile
                                rawdata = Dataset(datafilepath, "r")
                                pcp = rawdata[pfvarname][:]
                                rawdata.close()

                                # For 'gpmirimerg', precipitation is averaged to 1-hourly and put in first time dimension
                                if datasource == "gpmirimerg":
                                    pcp = pcp[0, :, :]

                                # Smooth PF variable, then label PF exceeding threshold
                                pcp_s = filters.uniform_filter(
                                    np.squeeze(pcp),
                                    size=pf_smooth_window,
                                    mode="nearest",
                                )
                                pf_number, npf = label(pcp_s >= pf_dbz_thresh)

                                # Convert PF area threshold to number of pixels
                                min_npix = np.ceil(
                                    pf_link_area_thresh / (pixel_radius ** 2)
                                )

                                # Sort and renumber PFs, and remove small PFs
                                pf_number, pf_npix = sort_renumber(pf_number, min_npix)
                                # Update number of PFs after sorting and renumbering
                                npf = np.nanmax(pf_number)

                                # Call function to link clouds with PFs
                                pf_convcold_cloudnumber, pf_cloudnumber = link_pf_tb(
                                    np.squeeze(final_convcold_cloudnumber),
                                    np.squeeze(final_cloudnumber),
                                    pf_number,
                                    out_ir,
                                    thresh_cloud,
                                )

                                # Sort and renumber the linkpf clouds (removes small clouds after renumbering)
                                (
                                    pf_convcold_cloudnumber_sorted,
                                    pf_cloudnumber_sorted,
                                    npix_convcold_linkpf,
                                ) = sort_renumber2vars(
                                    pf_convcold_cloudnumber,
                                    pf_cloudnumber,
                                    area_thresh / pixel_radius ** 2,
                                )
                                # Get number of clouds from the sorted linkpf clouds
                                nclouds_linkpf = np.nanmax(
                                    pf_convcold_cloudnumber_sorted
                                )

                                # Make a copy of the original arrays
                                final_cloudnumber_orig = final_cloudnumber
                                final_convcold_cloudnumber_orig = (
                                    final_convcold_cloudnumber
                                )

                                # Update output arrays
                                final_cloudnumber = np.expand_dims(
                                    pf_cloudnumber_sorted, axis=0
                                )
                                final_convcold_cloudnumber = np.expand_dims(
                                    pf_convcold_cloudnumber_sorted, axis=0
                                )
                                final_nclouds = np.array([nclouds_linkpf], dtype=int)
                                final_pf_number = np.expand_dims(pf_number, axis=0)
                                # final_ncorecoldpix = np.array([npix_convcold_linkpf], dtype=int)
                                final_ncorecoldpix = npix_convcold_linkpf
                                if pcp.ndim == 2:
                                    final_pcp = np.expand_dims(pcp, axis=0)
                                else:
                                    final_pcp = pcp

                            else:
                                # Create default arrays
                                final_pcp = np.full(
                                    final_convcold_cloudnumber.shape,
                                    np.nan,
                                    dtype=float,
                                )
                                final_pf_number = np.full(
                                    final_convcold_cloudnumber.shape, 0, dtype=int
                                )
                                # Make a copy of the original arrays
                                final_cloudnumber_orig = final_cloudnumber
                                final_convcold_cloudnumber_orig = (
                                    final_convcold_cloudnumber
                                )
                        else:
                            # Create default arrays
                            final_pcp = np.full(
                                final_convcold_cloudnumber.shape, np.nan, dtype=float
                            )
                            final_pf_number = np.full(
                                final_convcold_cloudnumber.shape, 0, dtype=int
                            )
                            # Make a copy of the original arrays
                            final_cloudnumber_orig = final_cloudnumber
                            final_convcold_cloudnumber_orig = final_convcold_cloudnumber

                        #######################################################
                        # output data to netcdf file, only if clouds present
                        if final_nclouds > 0:
                            # create filename
                            cloudid_outfile = (
                                dataoutpath
                                + datasource
                                + "_"
                                + datadescription
                                + "_cloudid"
                                + cloudid_version
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
                            net.write_cloudid_wrf(
                                cloudid_outfile,
                                file_basetime,
                                file_datestring,
                                file_timestring,
                                out_lat,
                                out_lon,
                                out_ir,
                                final_cloudtype,
                                final_convcold_cloudnumber,
                                final_cloudnumber,
                                final_nclouds,
                                final_ncorepix,
                                final_ncoldpix,
                                final_ncorecoldpix,
                                final_nwarmpix,
                                cloudid_version,
                                cloudtb_threshs,
                                geolimits,
                                mintb_thresh,
                                maxtb_thresh,
                                area_thresh,
                                precipitation=final_pcp,
                                pf_number=final_pf_number,
                                convcold_cloudnumber_orig=final_convcold_cloudnumber_orig,
                                cloudnumber_orig=final_cloudnumber_orig,
                                linkpf=linkpf,
                                pf_smooth_window=pf_smooth_window,
                                pf_dbz_thresh=pf_dbz_thresh,
                                pf_link_area_thresh=pf_link_area_thresh,
                            )

                        else:
                            logger.info(datafilepath)
                            logger.info("No clouds")

                    else:
                        logger.info(datafilepath)
                        logger.info("To much missing data")
                else:
                    logger.info(datafilepath)
                    logger.info(
                        "data not within latitude, longitude range. check specified geographic range"
                    )
