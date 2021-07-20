# def idclouds_gpmmergir(zipped_inputs):
def idclouds_gpmmergir(
    filename, config
):
    """
    Identifies convective cloud objects from GPM MERGIR global satellite data.

    Arguments:
    filename - path to raw data directory
    datafiledatestring - string with year, month, and day of data
    datafiletimestring - string with the hour and minute of thedata
    datafilebase - header for the raw data file
    config['clouddatasource'] - source of the raw data
    config['datadescription'] - description of data source, included in all output file names
    config['cloud_variable_name'] - name of tb data in raw data file
    config['cloudid_version'] - version of cloud identification being run, set at the start of the beginning of run_test.py
    config['tracking_outpath'] - path to destination of the output
    latlon_file - filename of the file that contains the latitude and longitude data
    latname - name of latitude variable in raw data file
    longname - name of longitude variable in raw data file
    config['geolimits'] - 4-element array with plotting boundaries [lat_min, lon_min, lat_max, lon_max]
    config['startdate'] - data to start processing in yyyymmdd format
    config['enddate'] - data to stop processing in yyyymmdd format
    config['pixel_radius'] - radius of pixels in km
    area_thresh - minimum area thershold to define a feature in km^2
    config['cloudtb_threshs'] - brightness temperature thresholds
    config['miss_thresh'] - minimum amount of data required in order for the file to not to be considered corrupt.
    config['cloudidmethod'] - flag indiciating which method of cloud classification will be used
    config['mincoldcorepix'] - minimum size threshold for a cloud
    config['smoothwindowdimensions'] - how many pixels to dilate as growing the warm (and cold) anvil. only used for futyan4.
    config['warmanvilexpansion'] - flag indicating whether to grow the warm anvil or ignore this step. The warm anvil is not used in tracking.
    processhalfhour
    config['idclouds_hourly'],
    config['idclouds_minute'],
    config['linkpf'],
    config['pf_smooth_window'],
    config['pf_dbz_thresh'],
    pf_link_area_thresh,
    config['pfvarname']
    """

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


    # Set medfilt2d kernel_size, this determines the filter window dimension
    medfiltsize = config.get('medfiltsize', 5)

    # Default idclouds minute difference allowed
    idclouds_minute = config.get('idclouds_minute', 5)

    # minimum and maximum brightness temperature thresholds. data outside of this range is filtered
    mintb_thresh = config['absolutetb_threshs'][0]  # k
    maxtb_thresh = config['absolutetb_threshs'][1]  # k

    # Get Tb thresholds
    thresh_core = config['cloudtb_threshs'][0]  # Convective core threshold [K]
    thresh_cold = config['cloudtb_threshs'][1] # Cold anvil threshold [K]
    thresh_warm = config['cloudtb_threshs'][2] # Warm anvil threshold [K]
    thresh_cloud = config['cloudtb_threshs'][3]  # Warmest cloud area threshold [K]

    # Brightness temperature data.
    # get date and time of each file. file name formate is "irdata_yyyymmdd_hhmm.nc" thus yyyymmdd=7:15, hhmm=16:20
    if (config['clouddatasource'] == "gpmmergir") | (config['clouddatasource'] == "gpmirimerg"):
        logger.debug(filename)

        # Read in data using xarray
        rawdata = xr.open_dataset(filename)
        lat = rawdata["lat"].values
        lon = rawdata["lon"].values
        time_decode = rawdata["time"]
        original_ir = rawdata[config['cloud_variable_name']].values
        rawdata.close()

        # Mesh 1D grid into 2D
        in_lon, in_lat = np.meshgrid(lon, lat)

        # Loop over each time
        for tt in range(0, len(time_decode)):

            iTime = time_decode[tt]
            file_basetime = np.array([iTime.values.tolist() / 1e9])
            file_datestring = iTime.dt.strftime("%Y%m%d").item()
            file_timestring = iTime.dt.strftime("%H%M").item()
            iminute = iTime.dt.minute.item()

            # If config['idclouds_hourly'] is set to 1, then check if iminutes is within the allowed difference from config['idclouds_minute']s
            # If so proceed, otherwise, skip this time
            if config['idclouds_hourly'] == 1:
                if np.absolute(iminute - config['idclouds_minute']) < config['idclouds_minute']:
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


                #####################################################
                # mask brightness temperatures outside of normal range
                in_ir[in_ir < mintb_thresh] = np.nan
                in_ir[in_ir > maxtb_thresh] = np.nan

                #####################################################
                # determine geographic region of interest is within the data set. if it is proceed and limit the data to that geographic region. if not exit the code.

                # isolate data within lat/lon range set by limit
                indicesy, indicesx = np.array(
                    np.where(
                        (in_lat >= config['geolimits'][0])
                        & (in_lat <= config['geolimits'][2])
                        & (in_lon >= config['geolimits'][1])
                        & (in_lon <= config['geolimits'][3])
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

                    if np.divide(missingcount, (ny * nx)) < config['miss_thresh']:
                        ######################################################


                        # call idclouds subroutine
                        if config['cloudidmethod'] == "futyan3":
                            from pyflextrkr.futyan3 import futyan3

                            clouddata = futyan3(
                                out_ir,
                                config['pixel_radius'],
                                config['cloudtb_threshs'],
                                config['area_thresh'],
                                config['warmanvilexpansion'],
                            )
                        elif config['cloudidmethod'] == "futyan4":
                            from pyflextrkr.label_and_grow_cold_clouds import (
                                label_and_grow_cold_clouds,
                            )

                            clouddata = label_and_grow_cold_clouds(
                                out_ir,
                                config['pixel_radius'],
                                config['cloudtb_threshs'],
                                config['area_thresh'],
                                config['mincoldcorepix'],
                                config['smoothwindowdimensions'],
                                config['warmanvilexpansion'],
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

                        # Option to config['linkpf']
                        if config['linkpf'] == 1:
                            from pyflextrkr.ftfunctions import (
                                sort_renumber,
                                sort_renumber2vars,
                                link_pf_tb,
                            )

                            # Proceed if there is at least 1 cloud
                            if final_nclouds > 0:
                                # Read PF from idcloudfile
                                rawdata = Dataset(filename, "r")
                                pcp = rawdata[config['pfvarname']][:]
                                rawdata.close()

                                # For 'gpmirimerg', precipitation is averaged to 1-hourly and put in first time dimension
                                if config['clouddatasource'] == "gpmirimerg":
                                    pcp = pcp[0, :, :]

                                # Smooth PF variable, then label PF exceeding threshold
                                pcp_s = filters.uniform_filter(
                                    np.squeeze(pcp),
                                    size=config['pf_smooth_window'],
                                    mode="nearest",
                                )
                                pf_number, npf = label(pcp_s >= config['pf_dbz_thresh'])

                                # Convert PF area threshold to number of pixels
                                min_npix = np.ceil(
                                    config['pf_link_area_thresh'] / (config['pixel_radius'] ** 2)
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

                                # Sort and renumber the config['linkpf'] clouds (removes small clouds after renumbering)
                                (
                                    pf_convcold_cloudnumber_sorted,
                                    pf_cloudnumber_sorted,
                                    npix_convcold_linkpf,
                                ) = sort_renumber2vars(
                                    pf_convcold_cloudnumber,
                                    pf_cloudnumber,
                                    config['area_thresh'] / config['pixel_radius'] ** 2,
                                )
                                # Get number of clouds from the sorted config['linkpf'] clouds
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
                                # final_ncorecoldpix = np.array([npix_convcold_config['linkpf']], dtype=int)
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
                                config['tracking_outpath']
                                + config['clouddatasource']
                                + "_"
                                + config['datadescription']
                                + "_cloudid"
                                + config['cloudid_version']
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
                                config['cloudid_version'],
                                config['cloudtb_threshs'],
                                config['geolimits'],
                                mintb_thresh,
                                maxtb_thresh,
                                config['area_thresh'],
                                precipitation=final_pcp,
                                pf_number=final_pf_number,
                                convcold_cloudnumber_orig=final_convcold_cloudnumber_orig,
                                cloudnumber_orig=final_cloudnumber_orig,
                                linkpf=config['linkpf'],
                                pf_smooth_window=config['pf_smooth_window'],
                                pf_dbz_thresh=config['pf_dbz_thresh'],
                                pf_link_area_thresh=config['pf_link_area_thresh'],
                            )

                        else:
                            logger.info(filename)
                            logger.info("No clouds")

                    else:
                        logger.info(filename)
                        logger.info("To much missing data")
                else:
                    logger.info(filename)
                    logger.info(
                        "data not within latitude, longitude range. check specified geographic range"
                    )
