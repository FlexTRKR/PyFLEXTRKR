# Purpose: identifies features and outputs netcdf files. Outputs one file for each raw data file with maps of cloud classifications and numbers. Also provides statistics about the number of pixels in each cloud.

# Author: orginial idl version written by Sally A. Mcfarlane (sally.mcfarlane@pnnl.gov) and by Zhe Feng (zhe.feng@pnnl.gov). python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov), altered by Katelyn Barber (katelyn.barber@pnnl.gov)

############################################################
# function used to handle test data
def idclouds_wrf(zipped_inputs):
    # inputs:
    # datafilepath - path to raw data directory
    # datasource - source of the raw data
    # datadescription - description of data source, included in all output file names
    # variablename - name of tb data in raw data file
    # cloudid_version - version of cloud identification being run, set at the start of the beginning of run_test.py
    # dataoutpath - path to destination of the output
    # latlon_file - filename of the file that contains the latitude and longitude data
    # latname - name of latitude variable in raw data file
    # longname - name of longitude variable in raw data file
    # geolimits - 4-element array with plotting boundaries [lat_min, lon_min, lat_max, lon_max]
    # startdate - data to start processing in yyyymmdd format
    # enddate - data to stop processing in yyyymmdd format
    # pixel_radius - radius of pixels in km
    # area_thresh - minimum area thershold to define a feature in km^2
    # cloudtb_threshs - brightness temperature thresholds
    # miss_thresh - minimum amount of data required in order for the file to not to be considered corrupt.
    # cloudidmethod - flag indiciating which method of cloud classification will be used
    # mincoldcorepix - minimum size threshold for a cloud
    # smoothsize - how many pixels to dilate as growing the warm (and cold) anvil. only used for futyan4.
    # warmanvilexpansion - flag indicating whether to grow the warm anvil or ignore this step. The warm anvil is not used in tracking.
    # processhalfhour - flag for whether process hourly or half-hourly data (0=hourly, 1=halfhourly)

    # output: (concatenated into netcdf file located in tracking subdirectory)
    # basetime - seconds since 19070-01-01 of the file
    # filedate - string of the year, month, and day of the file
    # filetime - string of the hour and minute of the file
    # longitude - longitudes used during identification process
    # latitude - latitudes used during indentification process
    # tb - brightness temperatures used during identification process
    # cloudtype - map of cloud type at each pixel (1 = core, 2 = cold anvil, 3 = warm anvil, 4 = other cloud)
    # cloudnumber - map of cloud number of each feature. includes core, cold anvil, and warm anvil area
    # nclouds - number of features identified
    # ncorepix - number of core pixels in each feature
    # ncoldpix - number of cold anvil pixels in each feature
    # ncorecoldpix - number of core and cold anvil pixels in each feature
    # nwarmpix - number of warm anvil pixels in each feature

    ##########################################################
    # Load modules

    from netCDF4 import Dataset, num2date
    import os
    import numpy as np
    import pandas as pd
    import logging

    # from scipy import signal
    # import scipy as sp
    # import numpy.ma as ma
    from scipy.ndimage import label, filters
    from pyflextrkr import netcdf_io as net

    np.set_printoptions(threshold=np.inf)
    logger = logging.getLogger(__name__)

    ########################################################
    # Separate inputs

    # datafilepath = zipped_inputs[0]
    # datasource = zipped_inputs[1]
    # datadescription = zipped_inputs[2]
    # variablename = zipped_inputs[3]
    # cloudid_version = zipped_inputs[4]
    # dataoutpath = zipped_inputs[5]
    # latlon_file = zipped_inputs[6]
    # latname = zipped_inputs[7]
    # longname = zipped_inputs[8]
    # geolimits = zipped_inputs[9]
    # startdate = zipped_inputs[10]
    # enddate = zipped_inputs[11]
    # pixel_radius = zipped_inputs[12]
    # area_thresh = zipped_inputs[13]
    # cloudtb_threshs = zipped_inputs[14]
    # absolutetb_threshs = zipped_inputs[15]
    # miss_thresh = zipped_inputs[16]
    # cloudidmethod = zipped_inputs[17]
    # mincoldcorepix = zipped_inputs[18]
    # smoothsize = zipped_inputs[19]
    # warmanvilexpansion = zipped_inputs[20]
    # processhalfhour = zipped_inputs[21]

    datafilepath = zipped_inputs[0]
    datasource = zipped_inputs[1]
    datadescription = zipped_inputs[2]
    variablename = zipped_inputs[3]
    cloudid_version = zipped_inputs[4]
    dataoutpath = zipped_inputs[5]
    latlon_file = zipped_inputs[6]
    geolimits = zipped_inputs[7]
    startdate = zipped_inputs[8]
    enddate = zipped_inputs[9]
    pixel_radius = zipped_inputs[10]
    area_thresh = zipped_inputs[11]
    cloudtb_threshs = zipped_inputs[12]
    absolutetb_threshs = zipped_inputs[13]
    miss_thresh = zipped_inputs[14]
    cloudidmethod = zipped_inputs[15]
    mincoldcorepix = zipped_inputs[16]
    smoothsize = zipped_inputs[17]
    warmanvilexpansion = zipped_inputs[18]
    processhalfhour = zipped_inputs[19]
    linkpf = zipped_inputs[20]
    pf_smooth_window = zipped_inputs[21]
    pf_dbz_thresh = zipped_inputs[22]
    pf_link_area_thresh = zipped_inputs[23]
    pfvarname = zipped_inputs[24]

    # Separate array threshold
    thresh_core = cloudtb_threshs[0]  # Convective core threshold [K]
    thresh_cold = cloudtb_threshs[1]  # Cold anvil threshold [K]
    thresh_warm = cloudtb_threshs[2]  # Warm anvil threshold [K]
    thresh_cloud = cloudtb_threshs[3]  # Warmest cloud area threshold [K]

    ##########################################################
    # define constants:
    # minimum and maximum brightness temperature thresholds. data outside of this range is filtered
    mintb_thresh = absolutetb_threshs[0]  # k
    maxtb_thresh = absolutetb_threshs[1]  # k

    #########################################################
    # Define processing flag
    if processhalfhour == 0:
        TimeIndices = 1  # Process hourly data
    else:
        TimeIndices = 2  # Process half-hourly data

    ########################################################
    # load data:
    if datasource == "WRF":
        logger.info(datafilepath)

        # load brighttness temperature data. automatically removes missing values
        rawdata = Dataset(datafilepath, "r")  # open file
        original_ir = rawdata[variablename][:]  # load brightness temperature data
        original_lat = rawdata["lat2d"][:]
        original_lon = rawdata["lon2d"][:]
        # original_precip = rawdata['rainrate'][:]
        # logger.info('original_precip shape: ', original_precip.shape)
        original_basetime = rawdata["time"][:]
        basetime_units = rawdata["time"].units
        rawdata.close()

        # Create latitude and longitude grids
        in_lon, in_lat = [original_lon, original_lat]

        for iTime in range(0, TimeIndices):
            # Replace missing ir data with mean
            datay, datax = np.array(np.ma.nonzero(original_ir[iTime, :, :]))
            in_ir = np.empty(np.shape(original_ir[iTime, :, :]), dtype=float) * np.nan
            in_ir[datay, datax] = original_ir[iTime, datay, datax]

            missingdatay, missingdatax = np.array(np.where(np.isnan(in_ir)))
            if len(missingdatay) > 0:
                for imiss in np.arange(0, len(missingdatay)):
                    if missingdatay[imiss] == 0:
                        if missingdatax[imiss] == 0:
                            subsetir = np.copy(
                                in_ir[
                                    0 : missingdatay[imiss] + 2,
                                    0 : missingdatax[imiss] + 2,
                                ]
                            )
                        else:
                            subsetir = np.copy(
                                in_ir[
                                    0 : missingdatay[imiss] + 2,
                                    missingdatax[imiss] - 1 : missingdatax[imiss] + 2,
                                ]
                            )
                    elif missingdatax[imiss] == 0:
                        subsetir = np.copy(
                            in_ir[
                                missingdatay[imiss] - 1 : missingdatay[imiss] + 2,
                                0 : missingdatax[imiss] + 2,
                            ]
                        )
                    elif missingdatay[imiss] == np.shape(original_ir)[0]:
                        if missingdatax[imiss] == np.shape(original_ir)[1]:
                            subsetir = np.copy(
                                in_ir[
                                    missingdatay[imiss] - 1 : :,
                                    missingdatax[imiss] - 1 : :,
                                ]
                            )
                        else:
                            subsetir = np.copy(
                                in_ir[
                                    missingdatay[imiss] - 1 : :,
                                    missingdatax[imiss] - 1 :: missingdatax[imiss] + 2,
                                ]
                            )
                    elif missingdatax[imiss] == np.shape(original_ir)[1]:
                        subsetir = np.copy(
                            in_ir[
                                missingdatay[imiss] - 1 : missingdatay[imiss] + 2,
                                missingdatax[imiss] - 1 : :,
                            ]
                        )
                    else:
                        subsetir = np.copy(
                            in_ir[
                                missingdatay[imiss] - 1 : missingdatay[imiss] + 2,
                                missingdatax[imiss] - 1 : missingdatax[imiss] + 2,
                            ]
                        )
                    subsetir = np.reshape(
                        subsetir, np.shape(subsetir)[0] * np.shape(subsetir)[1], 1
                    )
                    in_ir[missingdatay[imiss], missingdatax[imiss]] = np.nanmean(
                        subsetir
                    )

            #            # Deal with missing data
            #            in_ir = np.squeeze(original_ir,axis=0)
            #            logger.info('in_ir shape: ', in_ir.shape)
            #            df_tb = pd.DataFrame(in_ir)
            #            df_tb.fillna(method ='bfill', inplace = True)
            #            df_tb.fillna(method ='ffill', inplace = True)
            ##           df_tb.fillna(df_tb.mean())
            #            in_ir = df_tb.to_numpy()

            #            # Deal with missing data using interpolation method
            #            in_ir = np.squeeze(original_ir,axis=0)
            #            mask = np.isnan(in_ir)
            #            in_ir[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), in_ir[~mask])

            #####################################################
            # mask brightness temperatures outside of normal range
            in_ir[in_ir < mintb_thresh] = np.nan
            in_ir[in_ir > maxtb_thresh] = np.nan

            #####################################################
            # determine if geographic region of interest is within the data set. if it is proceed and limit the data to that geographic region. if not exit the code.

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
            if len(indicesx) > 0 and len(indicesy) > 0:
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

                    file_basetime = np.array(
                        [
                            pd.to_datetime(
                                num2date(original_basetime[iTime], units=basetime_units)
                            )
                        ],
                        dtype="datetime64[s]",
                    )
                    file_datestring = (
                        str(file_basetime)[2:6]
                        + str(file_basetime)[7:9]
                        + str(file_basetime)[10:12]
                    )
                    file_timestring = (
                        str(file_basetime)[13:15] + str(file_basetime)[16:18]
                    )

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
                    final_ncorepix = np.array([clouddata["final_ncorepix"]])
                    final_ncoldpix = np.array([clouddata["final_ncoldpix"]])
                    final_ncorecoldpix = np.array([clouddata["final_ncorecoldpix"]])
                    final_nwarmpix = np.array([clouddata["final_nwarmpix"]])
                    final_cloudtype = np.array([clouddata["final_cloudtype"]])
                    final_cloudnumber = np.array([clouddata["final_cloudnumber"]])
                    final_convcold_cloudnumber = np.array(
                        [clouddata["final_convcold_cloudnumber"]]
                    )

                    #                    final_nclouds = clouddata['final_nclouds']
                    #                    final_ncorepix = clouddata['final_ncorepix']
                    #                    final_ncoldpix = clouddata['final_ncoldpix']
                    #                    final_ncorecoldpix = clouddata['final_ncorecoldpix']
                    #                    final_nwarmpix = clouddata['final_nwarmpix']
                    #                    final_cloudtype = clouddata['final_cloudtype']

                    #                    logger.info('final_ncoldpix shape: ', final_ncoldpix.shape)
                    #                    logger.info('final_ncorecoldpix shape: ', final_ncorecoldpix.shape)
                    #                    logger.info('final_nwarmpix shape: ', final_nwarmpix.shape)
                    #                    logger.info('final_ncorepix shape: ', final_ncorepix.shape)

                    # if (final_nclouds == 0):

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

                            # Smooth PF variable, then label PF exceeding threshold
                            pcp_s = filters.uniform_filter(
                                np.squeeze(pcp), size=pf_smooth_window, mode="nearest"
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

                            logger.info(
                                f"npix_convcold_linkpf:{npix_convcold_linkpf.shape}"
                            )
                            # Get number of clouds from the sorted linkpf clouds
                            nclouds_linkpf = np.nanmax(pf_convcold_cloudnumber_sorted)

                            # Make a copy of the original arrays
                            final_cloudnumber_orig = final_cloudnumber
                            final_convcold_cloudnumber_orig = final_convcold_cloudnumber

                            # Update output arrays
                            final_cloudnumber = np.expand_dims(
                                pf_cloudnumber_sorted, axis=0
                            )
                            final_convcold_cloudnumber = np.expand_dims(
                                pf_convcold_cloudnumber_sorted, axis=0
                            )
                            final_nclouds = np.array([nclouds_linkpf], dtype=int)
                            final_pf_number = np.expand_dims(pf_number, axis=0)
                            # final_ncorecoldpix = np.array([npix_convcold_linkpf], dtype=int)  ### KB COMMENTED OUT 04/21/2020
                            final_ncorecoldpix = npix_convcold_linkpf

                        else:
                            # Create default arrays
                            pcp = np.full(final_convcold_cloudnumber.shape, dtype=float)
                            final_pf_number = np.full(
                                final_convcold_cloudnumber.shape, dtype=int
                            )
                            # Make a copy of the original arrays
                            final_cloudnumber_orig = final_cloudnumber
                            final_convcold_cloudnumber_orig = final_convcold_cloudnumber
                    else:
                        # Create default arrays
                        pcp = np.full(final_convcold_cloudnumber.shape, dtype=float)
                        final_pf_number = np.full(
                            final_convcold_cloudnumber.shape, dtype=int
                        )
                        # Make a copy of the original arrays
                        final_cloudnumber_orig = final_cloudnumber
                        final_convcold_cloudnumber_orig = final_convcold_cloudnumber

                    #######################################################
                    # output data to netcdf file, only if clouds present
                    if final_nclouds >= 0:  # KB CHANGED
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
                        logger.info(f"outcloudfile: {cloudid_outfile}")

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
                            precipitation=pcp,
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


def idclouds_mergedir(zipped_inputs):
    # inputs:
    # datafilepath - path to raw data directory
    # datafiledatestring - string with year, month, and day of data
    # datafiletimestring - string with the hour and minute of thedata
    # datafilebase - header for the raw data file
    # datasource - source of the raw data
    # datadescription - description of data source, included in all output file names
    # variablename - name of tb data in raw data file
    # cloudid_version - version of cloud identification being run, set at the start of the beginning of run_test.py
    # dataoutpath - path to destination of the output
    # latlon_file - filename of the file that contains the latitude and longitude data
    # latname - name of latitude variable in raw data file
    # longname - name of longitude variable in raw data file
    # geolimits - 4-element array with plotting boundaries [lat_min, lon_min, lat_max, lon_max]
    # startdate - data to start processing in yyyymmdd format
    # enddate - data to stop processing in yyyymmdd format
    # pixel_radius - radius of pixels in km
    # area_thresh - minimum area thershold to define a feature in km^2
    # cloudtb_threshs - brightness temperature thresholds
    # miss_thresh - minimum amount of data required in order for the file to not to be considered corrupt.
    # cloudidmethod - flag indiciating which method of cloud classification will be used
    # mincoldcorepix - minimum size threshold for a cloud
    # smoothsize - how many pixels to dilate as growing the warm (and cold) anvil. only used for futyan4.
    # warmanvilexpansion - flag indicating whether to grow the warm anvil or ignore this step. The warm anvil is not used in tracking.

    # output: (concatenated into netcdf file located in tracking subdirectory)
    # basetime - seconds since 19070-01-01 of the file
    # filedate - string of the year, month, and day of the file
    # filetime - string of the hour and minute of the file
    # longitude - longitudes used during identification process
    # latitude - latitudes used during indentification process
    # tb - brightness temperatures used during identification process
    # cloudtype - map of cloud type at each pixel (1 = core, 2 = cold anvil, 3 = warm anvil, 4 = other cloud)
    # cloudnumber - map of cloud number of each feature. includes core, cold anvil, and warm anvil area
    # nclouds - number of features identified
    # ncorepix - number of core pixels in each feature
    # ncoldpix - number of cold anvil pixels in each feature
    # ncorecoldpix - number of core and cold anvil pixels in each feature
    # nwarmpix - number of warm anvil pixels in each feature

    ##########################################################
    # Load modules

    from netCDF4 import stringtochar
    import os
    import numpy as np
    import calendar
    import time
    import xarray as xr
    import datetime
    import pandas as pd

    np.set_printoptions(threshold=np.inf)

    ########################################################
    # Separate inputs

    datafilepath = zipped_inputs[0]
    datafiledatestring = zipped_inputs[1]
    datafiletimestring = zipped_inputs[2]
    datafilebasetime = zipped_inputs[3]
    datasource = zipped_inputs[4]
    datadescription = zipped_inputs[5]
    variablename = zipped_inputs[6]
    cloudid_version = zipped_inputs[7]
    dataoutpath = zipped_inputs[8]
    latlon_file = zipped_inputs[9]
    latname = zipped_inputs[10]
    longname = zipped_inputs[11]
    geolimits = zipped_inputs[12]
    startdate = zipped_inputs[13]
    enddate = zipped_inputs[14]
    pixel_radius = zipped_inputs[15]
    area_thresh = zipped_inputs[16]
    cloudtb_threshs = zipped_inputs[17]
    absolutetb_threshs = zipped_inputs[18]
    miss_thresh = zipped_inputs[19]
    cloudidmethod = zipped_inputs[20]
    mincoldcorepix = zipped_inputs[21]
    smoothsize = zipped_inputs[22]
    warmanvilexpansion = zipped_inputs[23]

    ##########################################################
    # define constants:
    # minimum and maximum brightness temperature thresholds. data outside of this range is filtered
    mintb_thresh = absolutetb_threshs[0]  # k
    maxtb_thresh = absolutetb_threshs[1]  # k

    ########################################################
    # load data:
    # geolocation data
    if datasource == "mergedir":
        geolocation_data = xr.open_dataset(latlon_file, autoclose=True)  # open file
        in_lat = geolocation_data[latname].data  # load latitude data
        in_lon = geolocation_data[longname].data  # load longitude data

    # Brightness temperature data.
    # get date and time of each file. file name formate is "irdata_yyyymmdd_hhmm.nc" thus yyyymmdd=7:15, hhmm=16:20
    if datasource == "mergedir":
        logger.info(datafilepath)

        # load brighttness temperature data. automatically removes missing values
        rawdata = xr.open_dataset(datafilepath, autoclose=True)  # open file
        original_ir = rawdata[variablename].data  # load brightness temperature data

        # Replace missing ir data with mean
        datay, datax = np.array(np.ma.nonzero(original_ir))
        in_ir = np.empty(np.shape(original_ir), dtype=float) * np.nan
        in_ir[datay, datax] = original_ir[datay, datax]

        missingdatay, missingdatax = np.array(np.where(np.isnan(in_ir)))
        if len(missingdatay) > 0:
            for imiss in np.arange(0, len(missingdatay)):
                if missingdatay[imiss] == 0:
                    if missingdatax[imiss] == 0:
                        subsetir = np.copy(
                            in_ir[
                                0 : missingdatay[imiss] + 2, 0 : missingdatax[imiss] + 2
                            ]
                        )
                    else:
                        subsetir = np.copy(
                            in_ir[
                                0 : missingdatay[imiss] + 2,
                                missingdatax[imiss] - 1 : missingdatax[imiss] + 2,
                            ]
                        )
                elif missingdatax[imiss] == 0:
                    subsetir = np.copy(
                        in_ir[
                            missingdatay[imiss] - 1 : missingdatay[imiss] + 2,
                            0 : missingdatax[imiss] + 2,
                        ]
                    )
                elif missingdatay[imiss] == np.shape(original_ir)[0]:
                    if missingdatax[imiss] == np.shape(original_ir)[1]:
                        subsetir = np.copy(
                            in_ir[
                                missingdatay[imiss] - 1 : :, missingdatax[imiss] - 1 : :
                            ]
                        )
                    else:
                        subsetir = np.copy(
                            in_ir[
                                missingdatay[imiss] - 1 : :,
                                missingdatax[imiss] - 1 :: missingdatax[imiss] + 2,
                            ]
                        )
                elif missingdatax[imiss] == np.shape(original_ir)[1]:
                    subsetir = np.copy(
                        in_ir[
                            missingdatay[imiss] - 1 : missingdatay[imiss] + 2,
                            missingdatax[imiss] - 1 : :,
                        ]
                    )
                else:
                    subsetir = np.copy(
                        in_ir[
                            missingdatay[imiss] - 1 : missingdatay[imiss] + 2,
                            missingdatax[imiss] - 1 : missingdatax[imiss] + 2,
                        ]
                    )
                subsetir = np.reshape(
                    subsetir, np.shape(subsetir)[0] * np.shape(subsetir)[1], 1
                )
                in_ir[missingdatay[imiss], missingdatax[imiss]] = np.nanmean(subsetir)

        in_lat = np.transpose(in_lat)
        in_lon = np.transpose(in_lon)
        in_ir = np.transpose(in_ir)

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
        if len(indicesx) > 0 and len(indicesy) > 0:
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

                TEMP_basetime = calendar.timegm(
                    datetime.datetime(
                        int(datafiledatestring[0:4]),
                        int(datafiledatestring[4:6]),
                        int(datafiledatestring[6:8]),
                        int(datafiletimestring[0:2]),
                        int(datafiletimestring[2:4]),
                        0,
                        0,
                    ).timetuple()
                )
                file_basetime = pd.to_datetime(TEMP_basetime, unit="s")
                logger.info((np.array([file_basetime], dtype="datetime64[ns]")))

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
                    from pyflextrkr.subroutine_idclouds import futyan4_mergedir

                    clouddata = futyan4_mergedir(
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
                final_ncorepix = np.array([clouddata["final_ncorepix"]])
                final_ncoldpix = np.array([clouddata["final_ncoldpix"]])
                final_ncorecoldpix = np.array([clouddata["final_ncorecoldpix"]])
                final_nwarmpix = np.array([clouddata["final_nwarmpix"]])
                final_cloudtype = np.array([clouddata["final_cloudtype"]])
                final_cloudnumber = np.array([clouddata["final_cloudnumber"]])
                final_convcold_cloudnumber = np.array(
                    [clouddata["final_convcold_cloudnumber"]]
                )

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
                        + datafiledatestring
                        + "_"
                        + datafiletimestring
                        + ".nc"
                    )

                    # Check if file exists, if it does delete it
                    if os.path.isfile(cloudid_outfile):
                        os.remove(cloudid_outfile)

                    # Calculate date-time data
                    TEMP_basetime = calendar.timegm(
                        datetime.datetime(
                            int(datafiledatestring[0:4]),
                            int(datafiledatestring[4:6]),
                            int(datafiledatestring[6:8]),
                            int(datafiletimestring[0:2]),
                            int(datafiletimestring[2:4]),
                            0,
                            0,
                        ).timetuple()
                    )
                    file_basetime = np.array(
                        [pd.to_datetime(TEMP_basetime, unit="s")], dtype="datetime64[s]"
                    )

                    # Define xarray dataset
                    output_data = xr.Dataset(
                        {
                            "basetime": (["time"], file_basetime),
                            "filedate": (
                                ["time", "ndatechar"],
                                np.array([stringtochar(np.array(datafiledatestring))]),
                            ),
                            "filetime": (
                                ["time", "ntimechar"],
                                np.array([stringtochar(np.array(datafiletimestring))]),
                            ),
                            "latitude": (["lat", "lon"], out_lat),
                            "longitude": (["lat", "lon"], out_lon),
                            "tb": (
                                ["time", "lat", "lon"],
                                np.expand_dims(out_ir, axis=0),
                            ),
                            "cloudtype": (["time", "lat", "lon"], final_cloudtype),
                            "convcold_cloudnumber": (
                                ["time", "lat", "lon"],
                                final_convcold_cloudnumber,
                            ),
                            "cloudnumber": (["time", "lat", "lon"], final_cloudnumber),
                            "nclouds": (["time"], final_nclouds),
                            "ncorepix": (["time", "clouds"], final_ncorepix),
                            "ncoldpix": (["time", "clouds"], final_ncoldpix),
                            "ncorecoldpix": (["time", "clouds"], final_ncorecoldpix),
                            "nwarmpix": (["time", "clouds"], final_nwarmpix),
                        },
                        coords={
                            "time": (["time"], file_basetime),
                            "lat": (["lat"], np.squeeze(out_lat[:, 0])),
                            "lon": (["lon"], np.squeeze(out_lon[0, :])),
                            "clouds": (["clouds"], np.arange(1, final_nclouds + 1)),
                            "ndatechar": (["ndatechar"], np.arange(0, 8)),
                            "ntimechar": (["ntimechar"], np.arange(0, 4)),
                        },
                        attrs={
                            "title": "Statistics about convective features identified in the data from "
                            + datafiledatestring[0:4]
                            + "/"
                            + datafiledatestring[4:6]
                            + "/"
                            + datafiledatestring[6:8]
                            + " "
                            + datafiletimestring[0:2]
                            + ":"
                            + datafiletimestring[2:4]
                            + " utc",
                            "institution": "Pacific Northwest National Laboratory",
                            "convections": "CF-1.6",
                            "contact": "Hannah C Barnes: hannah.barnes@pnnl.gov",
                            "created_ok": time.ctime(time.time()),
                            "cloudid_cloud_version": cloudid_version,
                            "tb_threshold_core": str(int(cloudtb_threshs[0])) + "K",
                            "tb_threshold_coldanvil": str(int(cloudtb_threshs[1]))
                            + "K",
                            "tb_threshold_warmanvil": str(int(cloudtb_threshs[2]))
                            + "K",
                            "tb_threshold_environment": str(int(cloudtb_threshs[3]))
                            + "K",
                            "minimum_cloud_area": str(int(area_thresh)) + "km^2",
                        },
                    )

                    # Specify variable attributes
                    output_data.time.attrs[
                        "long_name"
                    ] = "epoch time (seconds since 01/01/1970 00:00) in epoch of file"

                    output_data.lat.attrs[
                        "long_name"
                    ] = "Vector of latitudes, y-coordinate in Cartesian system"
                    output_data.lat.attrs["standard_name"] = "latitude"
                    output_data.lat.attrs["units"] = "degrees_north"
                    output_data.lat.attrs["valid_min"] = geolimits[0]
                    output_data.lat.attrs["valid_max"] = geolimits[2]

                    output_data.lon.attrs[
                        "long_name"
                    ] = "Vector of longitudes, x-coordinate in Cartesian system"
                    output_data.lon.attrs["standard_name"] = "longitude"
                    output_data.lon.attrs["units"] = "degrees_east"
                    output_data.lon.attrs["valid_min"] = geolimits[1]
                    output_data.lon.attrs["valid_max"] = geolimits[2]

                    output_data.clouds.attrs[
                        "long_name"
                    ] = "number of distict convective cores identified"
                    output_data.clouds.attrs["units"] = "units"

                    output_data.ndatechar.attrs[
                        "long_name"
                    ] = "number of characters in date string"
                    output_data.ndatechar.attrs["units"] = "unitless"

                    output_data.ntimechar.attrs[
                        "long_name"
                    ] = "number of characters in time string"
                    output_data.ntimechar.attrs["units"] = "unitless"

                    output_data.basetime.attrs[
                        "long_name"
                    ] = "epoch time (seconds since 01/01/1970 00:00) of file"
                    output_data.basetime.attrs["standard_name"] = "time"

                    output_data.filedate.attrs[
                        "long_name"
                    ] = "date string of file (yyyymmdd)"
                    output_data.filedate.attrs["units"] = "unitless"

                    output_data.filetime.attrs[
                        "long_name"
                    ] = "time string of file (hhmm)"
                    output_data.filetime.attrs["units"] = "unitless"

                    output_data.latitude.attrs[
                        "long_name"
                    ] = "cartesian grid of latitude"
                    output_data.latitude.attrs["units"] = "degrees_north"
                    output_data.latitude.attrs["valid_min"] = geolimits[0]
                    output_data.latitude.attrs["valid_max"] = geolimits[2]

                    output_data.longitude.attrs[
                        "long_name"
                    ] = "cartesian grid of longitude"
                    output_data.longitude.attrs["units"] = "degrees_east"
                    output_data.longitude.attrs["valid_min"] = geolimits[1]
                    output_data.longitude.attrs["valid_max"] = geolimits[3]

                    output_data.tb.attrs["long_name"] = "brightness temperature"
                    output_data.tb.attrs["units"] = "K"
                    output_data.tb.attrs["valid_min"] = mintb_thresh
                    output_data.tb.attrs["valid_max"] = maxtb_thresh

                    output_data.cloudtype.attrs[
                        "long_name"
                    ] = "grid of cloud classifications"
                    output_data.cloudtype.attrs[
                        "values"
                    ] = "1 = core, 2 = cold anvil, 3 = warm anvil, 4 = other"
                    output_data.cloudtype.attrs["units"] = "unitless"
                    output_data.cloudtype.attrs["valid_min"] = 1
                    output_data.cloudtype.attrs["valid_max"] = 5

                    output_data.convcold_cloudnumber.attrs[
                        "long_name"
                    ] = "grid with each classified cloud given a number"
                    output_data.convcold_cloudnumber.attrs["units"] = "unitless"
                    output_data.convcold_cloudnumber.attrs["valid_min"] = 0
                    output_data.convcold_cloudnumber.attrs["valid_max"] = (
                        final_nclouds + 1
                    )
                    output_data.convcold_cloudnumber.attrs[
                        "comment"
                    ] = "extend of each cloud defined using cold anvil threshold"

                    output_data.cloudnumber.attrs[
                        "long_name"
                    ] = "grid with each classified cloud given a number"
                    output_data.cloudnumber.attrs["units"] = "unitless"
                    output_data.cloudnumber.attrs["valid_min"] = 0
                    output_data.cloudnumber.attrs["valid_max"] = final_nclouds + 1
                    output_data.cloudnumber.attrs[
                        "comment"
                    ] = "extend of each cloud defined using warm anvil threshold"

                    output_data.nclouds.attrs[
                        "long_name"
                    ] = "number of distict convective cores identified in file"
                    output_data.nclouds.attrs["units"] = "unitless"

                    output_data.ncorepix.attrs[
                        "long_name"
                    ] = "number of convective core pixels in each cloud feature"
                    output_data.ncorepix.attrs["units"] = "unitless"

                    output_data.ncoldpix.attrs[
                        "long_name"
                    ] = "number of cold anvil pixels in each cloud feature"
                    output_data.ncoldpix.attrs["units"] = "unitless"

                    output_data.ncorecoldpix.attrs[
                        "long_name"
                    ] = "number of convective core and cold anvil pixels in each cloud feature"
                    output_data.ncorecoldpix.attrs["units"] = "unitless"

                    output_data.nwarmpix.attrs[
                        "long_name"
                    ] = "number of warm anvil pixels in each cloud feature"
                    output_data.nwarmpix.attrs["units"] = "unitless"

                    # Write netCDF file
                    logger.info(cloudid_outfile)
                    logger.info("")

                    output_data.to_netcdf(
                        path=cloudid_outfile,
                        mode="w",
                        format="NETCDF4_CLASSIC",
                        unlimited_dims="time",
                        encoding={
                            "time": {"zlib": True, "units": "seconds since 1970-01-01"},
                            "lon": {"zlib": True},
                            "lon": {"zlib": True},
                            "clouds": {"zlib": True},
                            "basetime": {
                                "dtype": "int64",
                                "zlib": True,
                                "units": "seconds since 1970-01-01",
                            },
                            "filedate": {"dtype": "str", "zlib": True},
                            "filetime": {"dtype": "str", "zlib": True},
                            "longitude": {"zlib": True, "_FillValue": np.nan},
                            "latitude": {"zlib": True, "_FillValue": np.nan},
                            "tb": {"zlib": True, "_FillValue": np.nan},
                            "cloudtype": {"zlib": True, "_FillValue": -9999},
                            "convcold_cloudnumber": {
                                "dtype": "int",
                                "zlib": True,
                                "_FillValue": -9999,
                            },
                            "cloudnumber": {
                                "dtype": "int",
                                "zlib": True,
                                "_FillValue": -9999,
                            },
                            "nclouds": {
                                "dtype": "int",
                                "zlib": True,
                                "_FillValue": -9999,
                            },
                            "ncorepix": {
                                "dtype": "int",
                                "zlib": True,
                                "_FillValue": -9999,
                            },
                            "ncoldpix": {
                                "dtype": "int",
                                "zlib": True,
                                "_FillValue": -9999,
                            },
                            "ncorecoldpix": {
                                "dtype": "int",
                                "zlib": True,
                                "_FillValue": -9999,
                            },
                            "nwarmpix": {
                                "dtype": "int",
                                "zlib": True,
                                "_FillValue": -9999,
                            },
                        },
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


############################################/global/homes/h/hcbarnes/Tracking/testdata/################
# function used to handle test data
def idclouds_LES(zipped_inputs):
    # inputs:
    # datafilepath - path to raw data directory
    # datafiledatestring - string with year, month, and day of data
    # datafiletimestring - string with the hour and minute of thedata
    # datafilebase - header for the raw data file
    # datasource - source of the raw data
    # datadescription - description of data source, included in all output file names
    # cloudid_version - version of cloud identification being run, set at the start of the beginning of run_test.py
    # dataoutpath - path to destination of the output
    # latlon_file - filename of the file that contains the latitude and longitude data
    # geolimits - 4-element array with plotting boundaries [lat_min, lon_min, lat_max, lon_max]
    # nx - size of the domain in the x direction
    # ny - size of the domain in the y direction
    # startdate - data to start processing in yyyymmdd format
    # enddate - data to stop processing in yyyymmdd format
    # pixel_radius - radius of pixels in km
    # area_thresh - minimum area thershold to define a feature in km^2
    # cloudlwp_threshs - liquid water path thresholds
    # miss_thresh - minimum amount of data required in order for the file to not to be considered corrupt.
    # cloudidmethod - flag indiciating which method of cloud classification will be used
    # mincoldcorepix - minimum size threshold for a cloud
    # smoothsize - how many pixels to dilate as growing the warm (and cold) anvil. only used for futyan4.
    # warmanvilexpansion - flag indicating whether to grow the warm anvil or ignore this step. The warm anvil is not used in tracking.

    # output: (concatenated into netcdf file located in tracking subdirectory)
    # basetime - seconds since 19070-01-01 of the file
    # filedate - string of the year, month, and day of the file
    # filetime - string of the hour and minute of the file
    # longitude - longitudes used during identification process
    # latitude - latitudes used during indentification process
    # lwp - liquid water path used during identification process
    # cloudtype - map of cloud type at each pixel (1 = core, 2 = cold anvil, 3 = warm anvil, 4 = other cloud)
    # cloudnumber - map of cloud number of each feature. includes core, cold anvil, and warm anvil area
    # nclouds - number of features identified
    # ncorepix - number of core pixels in each feature
    # ncoldpix - number of cold anvil pixels in each feature
    # ncorecoldpix - number of core and cold anvil pixels in each feature
    # nwarmpix - number of warm anvil pixels in each feature

    ##########################################################
    # Load modules

    import os
    import numpy as np
    import calendar
    import time
    import xarray as xr
    import datetime
    import pandas as pd
    import logging

    logger = logging.getLogger(__name__)

    ########################################################
    # Separate inputs

    datafilepath = zipped_inputs[0]
    datafiledatestring = zipped_inputs[1]
    datafiletimestring = zipped_inputs[2]
    datafilebasetime = np.array([zipped_inputs[3]], dtype=int)
    datasource = zipped_inputs[4]
    datadescription = zipped_inputs[5]
    cloudid_version = zipped_inputs[6]
    dataoutpath = zipped_inputs[7]
    latlon_file = zipped_inputs[8]
    geolimits = zipped_inputs[9]
    nx = zipped_inputs[10]
    ny = zipped_inputs[11]
    startdate = zipped_inputs[12]
    enddate = zipped_inputs[13]
    pixel_radius = zipped_inputs[14]
    area_thresh = zipped_inputs[15]
    cloudlwp_threshs = zipped_inputs[16]
    absolutetb_threshs = zipped_inputs[17]
    miss_thresh = zipped_inputs[18]
    cloudidmethod = zipped_inputs[19]
    mincoldcorepix = zipped_inputs[20]
    smoothsize = zipped_inputs[21]
    warmanvilexpansion = zipped_inputs[22]

    ##########################################################
    # define constants:
    # minimum and maximum brightness temperature thresholds. data outside of this range is filtered
    mintb_thresh = absolutetb_threshs[0]  # k
    maxtb_thresh = absolutetb_threshs[1]  # k

    ########################################################
    # load data:
    # geolocation data
    if datasource == "LES":
        # Open file
        geolocation_data = np.loadtxt(latlon_file, dtype=float)

        # Load data
        in_lat = geolocation_data[:, 1]
        in_lon = geolocation_data[:, 2]

        # Transform into matrix
        in_lat = np.reshape(in_lat, (int(ny), int(nx)))
        in_lon = np.reshape(in_lon, (int(ny), int(nx)))

    # LWP data.
    if datasource == "LES":
        logger.info(datafilepath)

        # load brighttness temperature data. automatically removes missing values
        in_lwp = np.loadtxt(datafilepath, dtype=float)
        in_lwp = np.reshape(in_lwp[:, 0], (int(ny), int(nx)))
        #        in_lwp = np.reshape(in_lwp, (int(ny), int(nx)))
        #        import pdb; pdb.set_trace()

        #####################################################
        # mask brightness temperatures outside of normal range
        in_lwp[in_lwp < mintb_thresh] = np.nan
        in_lwp[in_lwp > maxtb_thresh] = np.nan

        #####################################################
        # determine geographic region of interest is within the data set. if it is proceed and limit the data to that geographic region. if not exit the code.

        # isolate data within lat/lon range set by limit
        indicesy, indicesx = np.array(
            np.where(
                (in_lat > geolimits[0])
                & (in_lat <= geolimits[2])
                & (in_lon > geolimits[1])
                & (in_lon <= geolimits[3])
            )
        )

        # proceed if file covers the geographic region in interest
        if len(indicesx) > 0 and len(indicesy) > 0:
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
            out_lwp = np.copy(
                in_lwp[
                    np.nanmin(indicesy) : np.nanmax(indicesy) + 1,
                    np.nanmin(indicesx) : np.nanmax(indicesx) + 1,
                ]
            )

            ######################################################
            # proceed only if number of missing data does not exceed an accepable threshold
            # determine number of missing data
            missingcount = len(np.array(np.where(np.isnan(out_lwp)))[0, :])
            ny, nx = np.shape(out_lwp)

            if np.divide(missingcount, (ny * nx)) < miss_thresh:
                ######################################################

                TEMP_basetime = calendar.timegm(
                    datetime.datetime(
                        int(datafiledatestring[0:4]),
                        int(datafiledatestring[4:6]),
                        int(datafiledatestring[6:8]),
                        int(datafiletimestring[0:2]),
                        int(datafiletimestring[2:4]),
                        0,
                        0,
                    ).timetuple()
                )
                file_basetime = pd.to_datetime(TEMP_basetime, unit="s")

                # call idclouds subroutine
                if cloudidmethod == "futyan3":
                    from pyflextrkr.futyan3 import futyan3

                    clouddata = futyan3(
                        out_lwp,
                        pixel_radius,
                        cloudlwp_threshs,
                        area_thresh,
                        warmanvilexpansion,
                    )
                elif cloudidmethod == "futyan4":
                    from pyflextrkr.futyan4_LES import futyan4_LES

                    clouddata = futyan4_LES(
                        out_lwp,
                        pixel_radius,
                        cloudlwp_threshs,
                        area_thresh,
                        mincoldcorepix,
                        smoothsize,
                        warmanvilexpansion,
                    )

                ######################################################
                # separate output from futyan into the separate variables
                final_nclouds = np.array([clouddata["final_nclouds"]])
                final_ncorepix = np.array([clouddata["final_ncorepix"]])
                final_ncoldpix = np.array([clouddata["final_ncoldpix"]])
                final_ncorecoldpix = np.array([clouddata["final_ncorecoldpix"]])
                final_nwarmpix = np.array([clouddata["final_nwarmpix"]])
                final_cloudtype = np.array([clouddata["final_cloudtype"]])
                final_cloudnumber = np.array([clouddata["final_cloudnumber"]])
                final_convcold_cloudnumber = np.array(
                    [clouddata["final_convcold_cloudnumber"]]
                )

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
                        + datafiledatestring
                        + "_"
                        + datafiletimestring
                        + ".nc"
                    )

                    # Check if file exists, if it does delete it
                    if os.path.isfile(cloudid_outfile):
                        os.remove(cloudid_outfile)

                    TEMP_basetime = calendar.timegm(
                        datetime.datetime(
                            int(datafiledatestring[0:4]),
                            int(datafiledatestring[4:6]),
                            int(datafiledatestring[6:8]),
                            int(datafiletimestring[0:2]),
                            int(datafiletimestring[2:4]),
                            0,
                            0,
                        ).timetuple()
                    )
                    file_basetime = np.array(
                        [pd.to_datetime(TEMP_basetime, unit="s")], dtype="datetime64[s]"
                    )
                    #                    import pdb; pdb.set_trace()
                    # Define xarray dataset
                    output_data = xr.Dataset(
                        {
                            "basetime": (
                                ["time"],
                                file_basetime,
                            ),  #                                              'filedate': (['time', 'ndatechar'],  np.array([stringtochar(np.array(datafiledatestring))])), \
                            #                                              'filetime': (['time', 'ntimechar'], np.array([stringtochar(np.array(datafiletimestring))])), \
                            "latitude": (["lat", "lon"], out_lat),
                            "longitude": (["lat", "lon"], out_lon),
                            "lwp": (
                                ["time", "lat", "lon"],
                                np.expand_dims(out_lwp, axis=0),
                            ),
                            "cloudtype": (["time", "lat", "lon"], final_cloudtype),
                            "convcold_cloudnumber": (
                                ["time", "lat", "lon"],
                                final_convcold_cloudnumber,
                            ),
                            "cloudnumber": (["time", "lat", "lon"], final_cloudnumber),
                            "nclouds": (["time"], final_nclouds),
                            "ncorepix": (["time", "clouds"], final_ncorepix),
                            "ncoldpix": (["time", "clouds"], final_ncoldpix),
                            "ncorecoldpix": (["time", "clouds"], final_ncorecoldpix),
                            "nwarmpix": (["time", "clouds"], final_nwarmpix),
                        },
                        coords={
                            "time": (["time"], file_basetime),
                            "lat": (["lat"], np.squeeze(out_lat[:, 0])),
                            "lon": (["lon"], np.squeeze(out_lon[0, :])),
                            "clouds": (
                                ["clouds"],
                                np.arange(1, final_nclouds + 1),
                            ),  #                                                     'ndatechar': (['ndatechar'], np.arange(0, 8)), \
                            #                                                     'ntimechar': (['ntimechar'], np.arange(0, 4)), \
                        },
                        attrs={
                            "title": "Statistics about convective features identified in the data from "
                            + datafiledatestring[0:4]
                            + "/"
                            + datafiledatestring[4:6]
                            + "/"
                            + datafiledatestring[6:8]
                            + " "
                            + datafiletimestring[0:2]
                            + ":"
                            + datafiletimestring[2:4]
                            + " utc",
                            "institution": "Pacific Northwest National Laboratory",
                            "conventions": "CF-1.6",
                            "contact": "Hannah C Barnes: hannah.barnes@pnnl.gov",
                            "created_ok": time.ctime(time.time()),
                            "cloudid_cloud_version": cloudid_version,
                            "lwp_threshold_core": str(int(cloudlwp_threshs[0])) + "K",
                            "lwp_threshold_coldanvil": str(int(cloudlwp_threshs[1]))
                            + "K",
                            "lwp_threshold_warmanvil": str(int(cloudlwp_threshs[2]))
                            + "K",
                            "lwp_threshold_environment": str(int(cloudlwp_threshs[3]))
                            + "K",
                            "minimum_cloud_area_km2": area_thresh,  #                                                    'minimum_cloud_area': str(int(area_thresh)) + 'km^2'})
                        },
                    )
                    # Specify variable attributes
                    output_data.time.attrs[
                        "long_name"
                    ] = "epoch time (seconds since 01/01/1970 00:00) in epoch of file"

                    output_data.lat.attrs[
                        "long_name"
                    ] = "Vector of latitudes, y-coordinate in Cartesian system"
                    output_data.lat.attrs["standard_name"] = "latitude"
                    output_data.lat.attrs["units"] = "degrees_north"
                    output_data.lat.attrs["valid_min"] = geolimits[0]
                    output_data.lat.attrs["valid_max"] = geolimits[2]

                    output_data.lon.attrs[
                        "long_name"
                    ] = "Vector of longitudes, x-coordinate in Cartesian system"
                    output_data.lon.attrs["standard_name"] = "longitude"
                    output_data.lon.attrs["units"] = "degrees_east"
                    output_data.lon.attrs["valid_min"] = geolimits[1]
                    output_data.lon.attrs["valid_max"] = geolimits[2]

                    output_data.clouds.attrs[
                        "long_name"
                    ] = "number of distict convective cores identified"
                    output_data.clouds.attrs["units"] = "unitless"

                    #                    output_data.ndatechar.attrs['long_name'] = 'number of characters in date string'
                    #                    output_data.ndatechar.attrs['units'] = 'unitless'

                    #                    output_data.ntimechar.attrs['long_name'] = 'number of characters in time string'
                    #                    output_data.ntimechar.attrs['units'] = 'unitless'

                    output_data.basetime.attrs[
                        "long_name"
                    ] = "epoch time (seconds since 01/01/1970 00:00) of file"
                    output_data.basetime.attrs["standard_name"] = "time"

                    #                    output_data.filedate.attrs['long_name'] = 'date string of file (yyyymmdd)'
                    #                    output_data.filedate.attrs['units'] = 'unitless'

                    #                    output_data.filetime.attrs['long_name'] = 'time string of file (hhmm)'
                    #                    output_data.filetime.attrs['units'] = 'unitless'

                    output_data.latitude.attrs[
                        "long_name"
                    ] = "cartesian grid of latitude"
                    output_data.latitude.attrs["units"] = "degrees_north"
                    output_data.latitude.attrs["valid_min"] = geolimits[0]
                    output_data.latitude.attrs["valid_max"] = geolimits[2]

                    output_data.longitude.attrs[
                        "long_name"
                    ] = "cartesian grid of longitude"
                    output_data.longitude.attrs["units"] = "degrees_east"
                    output_data.longitude.attrs["valid_min"] = geolimits[1]
                    output_data.longitude.attrs["valid_max"] = geolimits[3]

                    output_data.lwp.attrs["long_name"] = "liquid water path"
                    output_data.lwp.attrs["units"] = "kg/m^3"
                    output_data.lwp.attrs["valid_min"] = mintb_thresh
                    output_data.lwp.attrs["valid_max"] = maxtb_thresh

                    output_data.cloudtype.attrs[
                        "long_name"
                    ] = "grid of cloud classifications"
                    output_data.cloudtype.attrs[
                        "values"
                    ] = "1 = core, 2 = cold anvil, 3 = warm anvil, 4 = other"
                    output_data.cloudtype.attrs["units"] = "unitless"
                    output_data.cloudtype.attrs["valid_min"] = 1
                    output_data.cloudtype.attrs["valid_max"] = 5

                    output_data.convcold_cloudnumber.attrs[
                        "long_name"
                    ] = "grid with each classified cloud given a number"
                    output_data.convcold_cloudnumber.attrs["units"] = "unitless"
                    output_data.convcold_cloudnumber.attrs["valid_min"] = 0
                    output_data.convcold_cloudnumber.attrs["valid_max"] = (
                        final_nclouds + 1
                    )
                    output_data.convcold_cloudnumber.attrs[
                        "comment"
                    ] = "extend of each cloud defined using cold anvil threshold"

                    output_data.cloudnumber.attrs[
                        "long_name"
                    ] = "grid with each classified cloud given a number"
                    output_data.cloudnumber.attrs["units"] = "unitless"
                    output_data.cloudnumber.attrs["valid_min"] = 0
                    output_data.cloudnumber.attrs["valid_max"] = final_nclouds + 1
                    output_data.cloudnumber.attrs[
                        "comment"
                    ] = "extend of each cloud defined using warm anvil threshold"

                    output_data.nclouds.attrs[
                        "long_name"
                    ] = "number of distict convective cores identified in file"
                    output_data.nclouds.attrs["units"] = "unitless"

                    output_data.ncorepix.attrs[
                        "long_name"
                    ] = "number of convective core pixels in each cloud feature"
                    output_data.ncorepix.attrs["units"] = "unitless"

                    output_data.ncoldpix.attrs[
                        "long_name"
                    ] = "number of cold anvil pixels in each cloud feature"
                    output_data.ncoldpix.attrs["units"] = "unitless"

                    output_data.ncorecoldpix.attrs[
                        "long_name"
                    ] = "number of convective core and cold anvil pixels in each cloud feature"
                    output_data.ncorecoldpix.attrs["units"] = "unitless"

                    output_data.nwarmpix.attrs[
                        "long_name"
                    ] = "number of warm anvil pixels in each cloud feature"
                    output_data.nwarmpix.attrs["units"] = "unitless"

                    # Write netCDF file
                    logger.info(cloudid_outfile)
                    logger.info("")

                    output_data.to_netcdf(
                        path=cloudid_outfile,
                        mode="w",
                        format="NETCDF4_CLASSIC",
                        unlimited_dims="time",
                        encoding={
                            "time": {"zlib": True, "units": "seconds since 1970-01-01"},
                            "lon": {"zlib": True},
                            "lon": {"zlib": True},
                            "clouds": {"zlib": True},
                            "basetime": {
                                "dtype": "int64",
                                "zlib": True,
                                "units": "seconds since 1970-01-01",
                            },  #                                                    'filedate': {'dtype': 'str', 'zlib':True}, \
                            #                                                    'filetime': {'dtype': 'str', 'zlib':True}, \
                            "longitude": {"zlib": True, "_FillValue": np.nan},
                            "latitude": {"zlib": True, "_FillValue": np.nan},
                            "lwp": {"zlib": True, "_FillValue": np.nan},
                            "cloudtype": {"zlib": True, "_FillValue": -9999},
                            "convcold_cloudnumber": {
                                "dtype": "int",
                                "zlib": True,
                                "_FillValue": -9999,
                            },
                            "cloudnumber": {
                                "dtype": "int",
                                "zlib": True,
                                "_FillValue": -9999,
                            },
                            "nclouds": {
                                "dtype": "int",
                                "zlib": True,
                                "_FillValue": -9999,
                            },
                            "ncorepix": {
                                "dtype": "int",
                                "zlib": True,
                                "_FillValue": -9999,
                            },
                            "ncoldpix": {
                                "dtype": "int",
                                "zlib": True,
                                "_FillValue": -9999,
                            },
                            "ncorecoldpix": {
                                "dtype": "int",
                                "zlib": True,
                                "_FillValue": -9999,
                            },
                            "nwarmpix": {
                                "dtype": "int",
                                "zlib": True,
                                "_FillValue": -9999,
                            },
                        },
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


################################################################################################
def idclouds_ct(zipped_inputs):
    # inputs:
    # datafilepath - path to raw data directory
    # datasource - source of the raw data
    # datadescription - description of data source, included in all output file names
    # variablename - name of tb data in raw data file
    # cloudid_version - version of cloud identification being run, set at the start of the beginning of run_test.py
    # dataoutpath - path to destination of the output
    # latlon_file - filename of the file that contains the latitude and longitude data
    # latname - name of latitude variable in raw data file
    # longname - name of longitude variable in raw data file
    # geolimits - 4-element array with plotting boundaries [lat_min, lon_min, lat_max, lon_max]
    # startdate - data to start processing in yyyymmdd format
    # enddate - data to stop processing in yyyymmdd format
    # pixel_radius - radius of pixels in km
    # area_thresh - minimum area thershold to define a feature in km^2
    # cloudtb_threshs - brightness temperature thresholds
    # miss_thresh - minimum amount of data required in order for the file to not to be considered corrupt.
    # cloudidmethod - flag indiciating which method of cloud classification will be used
    # mincoldcorepix - minimum size threshold for a cloud
    # smoothsize - how many pixels to dilate as growing the warm (and cold) anvil. only used for futyan4.
    # warmanvilexpansion - flag indicating whether to grow the warm anvil or ignore this step. The warm anvil is not used in tracking.
    # processhalfhour - flag for whether process hourly or half-hourly data (0=hourly, 1=halfhourly)

    # output: (concatenated into netcdf file located in tracking subdirectory)
    # basetime - seconds since 19070-01-01 of the file
    # filedate - string of the year, month, and day of the file
    # filetime - string of the hour and minute of the file
    # longitude - longitudes used during identification process
    # latitude - latitudes used during indentification process
    # tb - brightness temperatures used during identification process
    # cloudtype - map of cloud type at each pixel (1 = core, 2 = cold anvil, 3 = warm anvil, 4 = other cloud)
    # cloudnumber - map of cloud number of each feature. includes core, cold anvil, and warm anvil area
    # nclouds - number of features identified
    # ncorepix - number of core pixels in each feature
    # ncoldpix - number of cold anvil pixels in each feature
    # ncorecoldpix - number of core and cold anvil pixels in each feature
    # nwarmpix - number of warm anvil pixels in each feature

    ##########################################################
    # Load modules

    from netCDF4 import Dataset, num2date
    import os
    import numpy as np
    import pandas as pd

    # from scipy import signal
    # import scipy as sp
    # import numpy.ma as ma
    from pyflextrkr import netcdf_io as net

    np.set_printoptions(threshold=np.inf)

    ########################################################
    # Separate inputs

    # datafilepath = zipped_inputs[0]
    # datasource = zipped_inputs[1]
    # datadescription = zipped_inputs[2]
    # variablename = zipped_inputs[3]
    # cloudid_version = zipped_inputs[4]
    # dataoutpath = zipped_inputs[5]
    # latlon_file = zipped_inputs[6]
    # latname = zipped_inputs[7]
    # longname = zipped_inputs[8]
    # geolimits = zipped_inputs[9]
    # startdate = zipped_inputs[10]
    # enddate = zipped_inputs[11]
    # pixel_radius = zipped_inputs[12]
    # area_thresh = zipped_inputs[13]
    # cloudtb_threshs = zipped_inputs[14]
    # absolutetb_threshs = zipped_inputs[15]
    # miss_thresh = zipped_inputs[16]
    # cloudidmethod = zipped_inputs[17]
    # mincoldcorepix = zipped_inputs[18]
    # smoothsize = zipped_inputs[19]
    # warmanvilexpansion = zipped_inputs[20]
    # processhalfhour = zipped_inputs[21]

    datafilepath = zipped_inputs[0]
    datasource = zipped_inputs[1]
    datadescription = zipped_inputs[2]
    variablename = zipped_inputs[3]
    cloudid_version = zipped_inputs[4]
    dataoutpath = zipped_inputs[5]
    latlon_file = zipped_inputs[6]
    geolimits = zipped_inputs[7]
    startdate = zipped_inputs[8]
    enddate = zipped_inputs[9]
    pixel_radius = zipped_inputs[10]
    area_thresh = zipped_inputs[11]
    cloudtb_threshs = zipped_inputs[12]
    absolutetb_threshs = zipped_inputs[13]
    miss_thresh = zipped_inputs[14]
    cloudidmethod = zipped_inputs[15]
    mincoldcorepix = zipped_inputs[16]
    smoothsize = zipped_inputs[17]
    warmanvilexpansion = zipped_inputs[18]
    processhalfhour = zipped_inputs[19]
    linkpf = zipped_inputs[20]
    pf_smooth_window = zipped_inputs[21]
    pf_dbz_thresh = zipped_inputs[22]
    pf_link_area_thresh = zipped_inputs[23]
    pfvarname = zipped_inputs[24]

    # define constants:
    # minimum and maximum brightness temperature thresholds. data outside of this range is filtered
    mintb_thresh = absolutetb_threshs[0]  # k
    maxtb_thresh = absolutetb_threshs[1]  # k

    # Define processing flag
    if processhalfhour == 0:
        TimeIndices = 1  # Process hourly data
        iTime = 0
    else:
        TimeIndices = 2  # Process half-hourly data

    ########################################################
    # load data:
    if datasource == "WRF":
        logger.info(datafilepath)

        # load brightness temperature data. automatically removes missing values
        rawdata = Dataset(datafilepath, "r")  # open file
        original_lat = rawdata["lat2d"][:]
        original_lon = rawdata["lon2d"][:]
        original_cloudtype = rawdata["cldtype"][:]
        # original_precip = rawdata['rainrate'][:]
        # logger.info('original_precip shape: ', original_precip.shape)
        original_basetime = rawdata["time"][:]
        basetime_units = rawdata["time"].units
        rawdata.close()

        # Area threshold for cloud type
        area_thresh = area_thresh

        # Create latitude and longitude grids
        in_lon, in_lat = [original_lon, original_lat]

        # in_ct = np.empty(np.shape(original_cloudtype[iTime, :, :]), dtype=float)*np.nan
        in_ct = original_cloudtype
        in_ct = np.squeeze(in_ct, axis=0)

        #####################################################
        # determine if geographic region of interest is within the data set. if it is proceed and limit the data to that geographic region. if not exit the code.

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
        if len(indicesx) > 0 and len(indicesy) > 0:
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
            out_ct = np.copy(
                in_ct[
                    np.nanmin(indicesy) : np.nanmax(indicesy) + 1,
                    np.nanmin(indicesx) : np.nanmax(indicesx) + 1,
                ]
            )

            ny, nx = np.shape(out_ct)

            ######################################################

            file_basetime = np.array(
                [
                    pd.to_datetime(
                        num2date(original_basetime[iTime], units=basetime_units)
                    )
                ],
                dtype="datetime64[s]",
            )
            file_datestring = (
                str(file_basetime)[2:6]
                + str(file_basetime)[7:9]
                + str(file_basetime)[10:12]
            )
            file_timestring = str(file_basetime)[13:15] + str(file_basetime)[16:18]

            # call idcloud type subroutine
            from pyflextrkr.cloud_type_tracking_NS import cloud_type_tracking_NS

            clouddata = cloud_type_tracking_NS(
                out_ct,
                pixel_radius,
                area_thresh,
                smoothsize,
                mincoldcorepix,
                warmanvilexpansion,
            )

            ######################################################
            # separate output from futyan into the separate variables
            final_nclouds = np.array(clouddata["final_nclouds"])
            final_ncorepix = np.array(clouddata["final_ncorepix"])
            # final_ncorepix = np.squeeze(final_ncorepix, axis = 0)
            final_ncoldpix = np.array(clouddata["final_ncoldpix"])
            # final_ncoldpix = np.squeeze(final_ncoldpix, axis = 0)
            final_ncorecoldpix = np.array(clouddata["final_ncorecoldpix"])
            # final_ncorecoldpix = np.squeeze(final_ncorecoldpix, axis = 0)
            final_convcold_cloudnumber = np.array(
                clouddata["final_convcold_cloudnumber"]
            )
            final_cloudtype = np.array(clouddata["final_cloudtype"])
            final_cloudnumber = np.array(clouddata["final_cloudnumber"])

            pcp = np.full(final_convcold_cloudnumber.shape, np.inf)
            final_pf_number = np.full(final_convcold_cloudnumber.shape, np.inf)
            final_convcold_cloudnumber_orig = final_convcold_cloudnumber
            final_cloudnumber_orig = final_convcold_cloudnumber_orig
            original_cloudtype = np.squeeze(original_cloudtype, axis=0)

            #######################################################
            # output data to netcdf file, only if clouds present
            if len(final_nclouds) >= 0:  # KB CHANGED
                # final_nclouds = np.squeeze(final_nclouds,axis=0)
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
                logger.info(f"outcloudfile: {cloudid_outfile}")

                # Check if file exists, if it does delete it
                if os.path.isfile(cloudid_outfile):
                    os.remove(cloudid_outfile)

                # Write output to netCDF file
                net.write_cloudtype_wrf(
                    cloudid_outfile,
                    file_basetime,
                    file_datestring,
                    file_timestring,
                    out_lat,
                    out_lon,
                    out_ct,
                    original_cloudtype,
                    final_convcold_cloudnumber,
                    final_nclouds,
                    final_ncorepix,
                    final_ncoldpix,
                    final_ncorecoldpix,
                    final_cloudtype,
                    cloudid_version,
                    cloudtb_threshs,
                    geolimits,
                    mintb_thresh,
                    maxtb_thresh,
                    area_thresh,
                    precipitation=pcp,
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
