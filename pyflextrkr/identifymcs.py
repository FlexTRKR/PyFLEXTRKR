# Purpose: Subset satellite statistics file to keep only MCS. Uses brightness temperature statstics of cold cloud shield area, duration, and eccentricity base on Fritsch et al (1986) and Maddos (1980)

# Author: Original IDL code written by Zhe Feng (zhe.feng@pnnl.gov), Python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov), altered by Katelyn Barber (katelyn.barber@pnnl.gov)


def identifymcs_tb(
    statistics_filebase,
    stats_path,
    startdate,
    enddate,
    geolimits,
    time_resolution,
    area_thresh,
    duration_thresh,
    eccentricity_thresh,
    split_duration,
    merge_duration,
    nmaxmerge,
    timegap=1,
):
    # Inputs:
    # statistics_filebase - header of the track statistics file that was generated in the previous code
    # stats_path - location of the track data. also the location where the data from this code will be saved
    # startdate - starting date and time of the data
    # enddate - ending date and time of the data
    # geolimits - 4-element array with plotting boundaries [lat_min, lon_min, lat_max, lon_max]
    # time_resolution - time resolution of the raw data
    # area_thresh - satellite area threshold for MCS identificaiton
    # duration_thresh - satellite duration threshold for MCS identification
    # eccentricity_thresh - satellite eccentricity threshold used for classifying squall lines
    # split_duration - splitting tracks with durations less than this will be label as a splitting track. tracks longer that this will be labeled as their own MCS
    # merge_duration - merging tracks with durations less than this will be label as a merging track. tracks longer that this will be labeled as their own MCS
    # nmaxmerge - maximum number of clouds that can split or merge into one cloud

    # Output: (One netcdf with statisitics about each satellite defined MCS in each row)
    # mcs_basetime - seconds since 1970-01-01 for each cloud in a MCS
    # mcs_datetimestring - string of date and time of each cloud in a MCS
    # mcs_length - duration of each MCS
    # mcs_type - flag indicating whether this is squall line, based on satellite definition
    # mcs_status - flag indicating the evolution of each cloud in a MCS
    # mcs_startstatus - flag indicating how a MCS starts
    # mcs_endstatus - flag indicating how a MCS ends
    # mcs_boundary - flag indicating if a MCS touches the edge of the domain
    # mcs_trackinterruptions - flag indicating if the data used to identify this MCS is incomplete
    # mcs_meanlat - mean latitude of the MCS
    # mcs_meanlon - mean longitude of the MCS
    # mcs_corearea - area of the core of MCS
    # mcs_ccsarea - area of the core and cold anvil of the MCS
    # mcs_cloudnumber - numbers indicating which clouds in the cloudid files are associated with a MCS
    # mcs_mergenumber - numbers indicating which clouds in the cloudid files merge into this track
    # mcs_splitnumber - numbers indicating which clouds in the cloudid files split from this track

    #######################################################################
    # Import modules
    import numpy as np
    from netCDF4 import Dataset, num2date
    import time
    import os
    import sys
    import xarray as xr
    import pandas as pd

    np.set_printoptions(threshold=np.inf)

    ##########################################################################
    # Load statistics file
    statistics_file = (
        stats_path + statistics_filebase + "_" + startdate + "_" + enddate + ".nc"
    )
    print(statistics_file)

    allstatdata = Dataset(statistics_file, "r")
    ntracks_all = (
        np.nanmax(allstatdata["ntracks"]) + 1
    )  # Total number of tracked features
    nmaxlength = (
        np.nanmax(allstatdata["nmaxlength"]) + 1
    )  # Maximum number of features in a given track
    print(nmaxlength)
    nconv = allstatdata["nconv"][:]
    ncoldanvil = allstatdata["ncoldanvil"][:]
    lifetime = allstatdata["lifetime"][:]
    eccentricity = allstatdata["eccentricity"][:]
    basetime = allstatdata["basetime"][
        :
    ]  # epoch time of each cloud in a track (ntracks, nmaxlength)
    basetime_units = allstatdata["basetime"].units
    # basetime_calendar = allstatdata['basetime'].calendar
    mergecloudnumbers = allstatdata["mergenumbers"][:]
    splitcloudnumbers = allstatdata["splitnumbers"][:]
    cloudnumbers = allstatdata["cloudnumber"][:]
    status = allstatdata["status"][:]
    endstatus = allstatdata["endstatus"][:]
    startstatus = allstatdata["startstatus"][:]
    datetimestrings = allstatdata["datetimestrings"][:]
    boundary = allstatdata["boundary"][:]
    trackinterruptions = allstatdata["trackinterruptions"][:]
    meanlat = np.array(allstatdata["meanlat"][:])
    meanlon = np.array(allstatdata["meanlon"][:])
    pixelradius = allstatdata.getncattr("pixel_radius_km")
    datasource = allstatdata.getncattr("source")
    datadescription = allstatdata.getncattr("description")
    allstatdata.close()

    print("MCS duration threshold: ", duration_thresh)
    print("MCS CCS area threshold: ", area_thresh)

    ####################################################################
    # Set up thresholds

    # Cold Cloud Shield (CCS) area
    trackstat_corearea = np.multiply(nconv, pixelradius ** 2)
    trackstat_ccsarea = nconv + np.multiply(ncoldanvil, pixelradius ** 2)

    # Convert path duration to time
    trackstat_duration = np.multiply(lifetime, time_resolution)
    trackstat_duration = trackstat_duration.astype(np.int32)

    ##################################################################
    # Initialize matrices
    trackid_mcs = []
    trackid_sql = []
    trackid_nonmcs = []

    mcstype = np.zeros(ntracks_all, dtype=np.int32)
    mcsstatus = np.ones((ntracks_all, nmaxlength), dtype=np.int32) * -9999

    ###################################################################
    # Identify MCSs
    print("Total number of tracks to check: ", ntracks_all)
    for nt in range(0, ntracks_all):
        # Get data for a given track
        track_corearea = np.copy(trackstat_corearea[nt, :])
        track_ccsarea = np.copy(trackstat_ccsarea[nt, :])
        track_eccentricity = np.copy(eccentricity[nt, :])

        # Remove fill values
        track_corearea = track_corearea[
            (~np.isnan(track_corearea)) & (track_corearea != 0)
        ]
        track_ccsarea = track_ccsarea[~np.isnan(track_ccsarea)]
        track_eccentricity = track_eccentricity[~np.isnan(track_eccentricity)]

        # Must have a cold core
        if np.shape(track_corearea)[0] != 0 and np.nanmax(track_corearea > 0):

            # Cold cloud shield area requirement
            iccs = np.array(np.where(track_ccsarea > area_thresh))[0, :]
            # print('nt: ', nt)
            # print('len iccs: ', len(iccs))
            nccs = len(iccs)

            # Find continuous times
            # groups = np.split(iccs, np.where(np.diff(iccs) != 1)[0]+1)  # ORIGINAL
            groups = np.split(
                iccs, np.where(np.diff(iccs) > timegap)[0] + 1
            )  # KB TESTING TIME GAP (!=1)
            nbreaks = len(groups)

            # System may have multiple periods satisfying area and duration requirements. Loop over each period
            if iccs != []:
                for t in range(0, nbreaks):
                    # Duration requirement
                    # Duration length should be group's last index - first index + 1
                    duration_group = np.multiply(
                        (groups[t][-1] - groups[t][0] + 1), time_resolution
                    )
                    if duration_group >= duration_thresh:
                        # if np.multiply(len(groups[t][:]), time_resolution) >= duration_thresh:
                        # print('Number of times * TIME RESOLUTION GREATER THAN DUR_THRESH')

                        # Isolate area and eccentricity for the subperiod
                        subtrack_ccsarea = track_ccsarea[groups[t][:]]
                        subtrack_eccentricity = track_eccentricity[groups[t][:]]

                        # Get eccentricity when the feature is the largest
                        subtrack_imax_ccsarea = np.nanargmax(subtrack_ccsarea)
                        subtrack_maxccsarea_eccentricity = subtrack_eccentricity[
                            subtrack_imax_ccsarea
                        ]

                        # Apply eccentricity requirement
                        if subtrack_maxccsarea_eccentricity > eccentricity_thresh:
                            # Label as MCS
                            mcstype[nt] = 1
                            mcsstatus[nt, groups[t][:]] = 1
                            # print('nt: ', nt)
                            # print('eccentricity met-mcs')
                        else:
                            # Label as squall line
                            mcstype[nt] = 2
                            mcsstatus[nt, groups[t][:]] = 2
                            trackid_sql = np.append(trackid_sql, nt)
                            # print('nt: ', nt)
                            # print('eccentricity not met-sl')
                        trackid_mcs = np.append(trackid_mcs, nt)
                    else:
                        # Size requirement met but too short of a period
                        trackid_nonmcs = np.append(trackid_nonmcs, nt)
                        # print('nt: ', nt)
                        # print('size met but duration not')

            else:
                # Size requirement not met
                trackid_nonmcs = np.append(trackid_nonmcs, nt)
                # print('nt: ', nt)
                # print('size not met')

    ################################################################
    # Check that there are MCSs to continue processing
    if trackid_mcs == []:
        print("There are no MCSs in the domain, the code will crash")

    ################################################################
    # Subset MCS / Squall track index
    trackid = np.array(np.where(mcstype > 0))[0, :]
    nmcs = len(trackid)
    print("Number of Tb defined MCS: ", nmcs)

    if nmcs > 0:
        mcsstatus = mcsstatus[trackid, :]
        mcstype = mcstype[trackid]

        mcslength = np.ones(len(mcstype), dtype=np.int32) * -9999
        for imcs in range(0, nmcs):
            mcslength[imcs] = len(np.array(np.where(mcsstatus[imcs, :] != -9999))[0, :])

    # trackid_mcs is the index number, want the track number so add one
    mcstracknumbers = np.copy(trackid) + 1

    ###############################################################
    # Find small merging and spliting clouds and add to MCS
    mcsmergecloudnumber = np.ones((nmcs, nmaxlength, nmaxmerge), dtype=np.int32) * -9999
    mcsmergestatus = np.ones((nmcs, nmaxlength, nmaxmerge), dtype=np.int32) * -9999
    mcssplitcloudnumber = np.ones((nmcs, nmaxlength, nmaxmerge), dtype=np.int32) * -9999
    mcssplitstatus = np.ones((nmcs, nmaxlength, nmaxmerge), dtype=np.int32) * -9999

    # Loop through each MCS and link small clouds merging in
    for imcs in np.arange(0, nmcs):
        ###################################################################################
        # Isolate basetime data
        # print('')
        if imcs == 0:
            mcsbasetime = basetime[trackid[imcs], :]
            # mcsbasetime = np.array([pd.to_datetime(num2date(basetime[trackid[imcs], :], units=basetime_units, calendar=basetime_calendar))], dtype='datetime64[s]')
        else:
            mcsbasetime = np.row_stack((mcsbasetime, basetime[trackid[imcs], :]))
            # mcsbasetime = np.concatenate((mcsbasetime, np.array([pd.to_datetime(num2date(basetime[trackid[imcs], :], units=basetime_units, calendar=basetime_calendar))], dtype='datetime64[s]')), axis=0)
        # if imcs >= 2:
        #     import pdb; pdb.set_trace()

        ###################################################################################
        # Find mergers
        # print('Mergers')
        [mergefile, mergefeature] = np.array(
            np.where(mergecloudnumbers == mcstracknumbers[imcs])
        )
        # print((len(mergefile)))
        for imerger in range(0, len(mergefile)):
            additionalmergefile, additionalmergefeature = np.array(
                np.where(mergecloudnumbers == mergefile[imerger] + 1)
            )
            if len(additionalmergefile) > 0:
                mergefile = np.concatenate((mergefile, additionalmergefile))
                mergefeature = np.concatenate((mergefeature, additionalmergefeature))

        # Loop through all merging tracks, if present
        if len(mergefile) > 0:
            # Isolate merging cases that have short duration
            mergefeature = mergefeature[trackstat_duration[mergefile] < merge_duration]
            mergefile = mergefile[trackstat_duration[mergefile] < merge_duration]

            # Make sure the merger itself is not an MCS
            mergingmcs = np.intersect1d(mergefile, mcstracknumbers)
            if len(mergingmcs) > 0:
                for iremove in np.arange(0, len(mergingmcs)):
                    removemerges = np.array(np.where(mergefile == mergingmcs[iremove]))[
                        0, :
                    ]
                    mergefile[removemerges] = -9999
                    mergefeature[removemerges] = -9999
                mergefile = mergefile[mergefile != -9999].astype(int)
                mergefeature = mergefeature[mergefeature != -9999].astype(int)

            # Continue if mergers satisfy duration and MCS restriction
            if len(mergefile) > 0:

                # Get data about merging tracks
                mergingcloudnumber = np.copy(cloudnumbers[mergefile, :])
                mergingbasetime = np.copy(basetime[mergefile, :])
                mergingstatus = np.copy(status[mergefile, :])
                mergingdatetime = np.copy(datetimestrings[mergefile, :, :])

                # Get data about MCS track
                imcsbasetime = np.copy(
                    basetime[int(mcstracknumbers[imcs]) - 1, 0 : int(mcslength[imcs])]
                )

                # Loop through each timestep in the MCS track
                for t in np.arange(0, mcslength[imcs]):
                    # print('t: ', t)

                    # Find merging cloud times that match current mcs track time
                    timematch = np.where(mergingbasetime == imcsbasetime[int(t)])
                    # print('timematch: ', timematch)

                    if np.shape(timematch)[1] > 0:

                        # save cloud number of small mergers
                        nmergers = np.shape(timematch)[1]
                        # print('IMCS: ', imcs)
                        # print('INT (T): ', int(t))
                        # print('NMERGERS SHAPE: ', np.shape(timematch)[1])
                        # print('MERGING CLOUD NUMBER TIMEMATCH: ', mergingcloudnumber[timematch])
                        mcsmergecloudnumber[
                            imcs, int(t), 0:nmergers
                        ] = mergingcloudnumber[timematch]
                        mcsmergestatus[imcs, int(t), 0:nmergers] = mergingstatus[
                            timematch
                        ]

        ############################################################
        # Find splits
        [splitfile, splitfeature] = np.array(
            np.where(splitcloudnumbers == mcstracknumbers[imcs])
        )

        # Loop through all splitting tracks, if present
        if len(splitfile) > 0:
            # Isolate splitting cases that have short duration
            splitfeature = splitfeature[trackstat_duration[splitfile] < split_duration]
            splitfile = splitfile[trackstat_duration[splitfile] < split_duration]

            # Make sure the spliter itself is not an MCS
            splittingmcs = np.intersect1d(splitfile, mcstracknumbers)
            if len(splittingmcs) > 0:
                for iremove in np.arange(0, len(splittingmcs)):
                    removesplits = np.array(
                        np.where(splitfile == splittingmcs[iremove])
                    )[0, :]
                    splitfile[removesplits] = -9999
                    splitfeature[removesplits] = -9999
                splitfile = splitfile[splitfile != -9999].astype(int)
                splitfeature = splitfeature[splitfeature != -9999].astype(int)

            # Continue if spliters satisfy duration and MCS restriction
            if len(splitfile) > 0:

                # Get data about splitting tracks
                splittingcloudnumber = np.copy(cloudnumbers[splitfile, :])
                splittingbasetime = np.copy(basetime[splitfile, :])
                splittingstatus = np.copy(status[splitfile, :])
                splittingdatetime = np.copy(datetimestrings[splitfile, :, :])

                # Get data about MCS track
                imcsbasetime = np.copy(
                    basetime[int(mcstracknumbers[imcs]) - 1, 0 : int(mcslength[imcs])]
                )

                # Loop through each timestep in the MCS track
                for t in np.arange(0, mcslength[imcs]):

                    # Find splitting cloud times that match current mcs track time
                    timematch = np.where(splittingbasetime == imcsbasetime[int(t)])
                    if np.shape(timematch)[1] > 0:

                        # save cloud number of small splitrs
                        nspliters = np.shape(timematch)[1]
                        mcssplitcloudnumber[
                            imcs, int(t), 0:nspliters
                        ] = splittingcloudnumber[timematch]
                        mcssplitstatus[imcs, int(t), 0:nspliters] = splittingstatus[
                            timematch
                        ]

    mcsmergecloudnumber = mcsmergecloudnumber.astype(np.int32)
    mcssplitcloudnumber = mcssplitcloudnumber.astype(np.int32)

    ###########################################################################
    # Write statistics to netcdf file

    # Create file
    mcstrackstatistics_outfile = (
        stats_path + "mcs_tracks_" + startdate + "_" + enddate + ".nc"
    )

    # Check if file already exists. If exists, delete
    if os.path.isfile(mcstrackstatistics_outfile):
        os.remove(mcstrackstatistics_outfile)

    # Defie xarray dataset
    output_data = xr.Dataset(
        {
            "mcs_basetime": (["ntracks", "ntimes"], mcsbasetime),
            "mcs_datetimestring": (
                ["ntracks", "ntimes", "ndatetimechars"],
                datetimestrings[trackid, :, :],
            ),
            "track_length": (["ntracks"], trackstat_duration[trackid]),
            "mcs_length": (["ntracks"], mcslength),
            "mcs_type": (["ntracks"], mcstype),
            "mcs_status": (["ntracks", "ntimes"], status[trackid, :]),
            "mcs_startstatus": (["ntracks"], startstatus[trackid]),
            "mcs_endstatus": (["ntracks"], endstatus[trackid]),
            "mcs_boundary": (["ntracks", "ntimes"], boundary[trackid]),
            "mcs_trackinterruptions": (["ntracks"], trackinterruptions[trackid]),
            "mcs_meanlat": (["ntracks", "ntimes"], meanlat[trackid, :]),
            "mcs_meanlon": (["ntracks", "ntimes"], meanlon[trackid, :]),
            "mcs_corearea": (["ntracks", "ntimes"], trackstat_corearea[trackid, :]),
            "mcs_ccsarea": (["ntracks", "ntimes"], trackstat_ccsarea[trackid, :]),
            "mcs_cloudnumber": (["ntracks", "ntimes"], cloudnumbers[trackid, :]),
            "mcs_mergecloudnumber": (
                ["ntracks", "ntimes", "nmergers"],
                mcsmergecloudnumber,
            ),
            "mcs_splitcloudnumber": (
                ["ntracks", "ntimes", "nmergers"],
                mcssplitcloudnumber,
            ),
        },
        coords={
            "ntracks": (["ntracks"], np.arange(0, nmcs)),
            "ntimes": (["ntimes"], np.arange(0, nmaxlength)),
            "nmergers": (["nmergers"], np.arange(0, nmaxmerge)),
            "ndatetimechars": (["ndatetimechars"], np.arange(0, 13)),
        },
        attrs={
            "title": "File containing statistics for each mcs track",
            "Conventions": "CF-1.6",
            "Institution": "Pacific Northwest National Laboratory",
            "Contact": "Katelyn Barber: katelyn.barber@pnnl.gov",
            "Created_on": time.ctime(time.time()),
            "source": datasource,
            "description": datadescription,
            "startdate": startdate,
            "enddate": enddate,
            "time_resolution_hour": time_resolution,
            "pixel_radius_km": pixelradius,
            "MCS_duration_hr": duration_thresh,
            "MCS_area_km^2": area_thresh,
            "MCS_eccentricity": eccentricity_thresh,
        },
    )

    # Specify variable attributes
    output_data.ntracks.attrs["long_name"] = "Number of mcss tracked"
    output_data.ntracks.attrs["units"] = "unitless"

    output_data.ntimes.attrs["long_name"] = "Maximum number of clouds in a mcs track"
    output_data.ntimes.attrs["units"] = "unitless"

    output_data.nmergers.attrs[
        "long_name"
    ] = "Maximum number of allowed mergers/splits into one cloud"
    output_data.nmergers.attrs["units"] = "unitless"

    output_data.mcs_basetime.attrs[
        "long_name"
    ] = "epoch time (seconds since 01/01/1970 00:00) of each cloud in a mcs track"
    output_data.mcs_basetime.attrs["standard_name"] = "time"
    output_data.mcs_basetime.attrs["units"] = basetime_units

    output_data.mcs_datetimestring.attrs[
        "long_name"
    ] = "date_time for each cloud in a mcs track"
    output_data.mcs_datetimestring.attrs["units"] = "unitless"

    output_data.track_length.attrs[
        "long_name"
    ] = "Complete length/duration of each track containing an mcs each mcs track"
    output_data.track_length.attrs["units"] = "hr"

    output_data.mcs_length.attrs["long_name"] = "Length/duration of each mcs track"
    output_data.mcs_length.attrs["units"] = "hr"

    output_data.mcs_type.attrs["long_name"] = "Type of MCS"
    output_data.mcs_type.attrs["usage"] = "1=MCS, 2=Squall Line"
    output_data.mcs_type.attrs["units"] = "unitless"

    output_data.mcs_status.attrs[
        "long_name"
    ] = "flag indicating the status of each cloud in mcs track"
    output_data.mcs_status.attrs["units"] = "unitless"
    output_data.mcs_status.attrs["valid_min"] = 0
    output_data.mcs_status.attrs["valid_max"] = 65

    output_data.mcs_startstatus.attrs[
        "long_name"
    ] = "flag indicating the status of the first cloud in each mcs track"
    output_data.mcs_startstatus.attrs["units"] = "unitless"
    output_data.mcs_startstatus.attrs["valid_min"] = 0
    output_data.mcs_startstatus.attrs["valid_max"] = 65

    output_data.mcs_endstatus.attrs[
        "long_name"
    ] = "flag indicating the status of the last cloud in each mcs track"
    output_data.mcs_endstatus.attrs["units"] = "unitless"
    output_data.mcs_endstatus.attrs["valid_min"] = 0
    output_data.mcs_endstatus.attrs["valid_max"] = 65

    output_data.mcs_boundary.attrs[
        "long_name"
    ] = "Flag indicating whether the core + cold anvil touches one of the domain edges."
    output_data.mcs_boundary.attrs["values"] = "0 = away from edge. 1= touches edge."
    output_data.mcs_boundary.attrs["units"] = "unitless"
    output_data.mcs_boundary.attrs["valid_min"] = 0
    output_data.mcs_boundary.attrs["valid_min"] = 1

    output_data.mcs_trackinterruptions.attrs[
        "long_name"
    ] = "Flag indiciate if the track started and ended naturally or if the start or end of the track was artifically cut short by data availability"
    output_data.mcs_trackinterruptions.attrs[
        "values"
    ] = "0 = full track available, good data. 1 = track starts at first file, track cut short by data availability. 2 = track ends at last file, track cut short by data availability"
    output_data.mcs_trackinterruptions.attrs["units"] = "unitless"
    output_data.mcs_trackinterruptions.attrs["valid_min"] = 0
    output_data.mcs_trackinterruptions.attrs["valid_min"] = 2

    output_data.mcs_meanlat.attrs[
        "long_name"
    ] = "Mean latitude of the core + cold anvil for each feature in a mcs track"
    output_data.mcs_meanlat.attrs["standard_name"] = "latitude"
    output_data.mcs_meanlat.attrs["units"] = "degrees_north"
    output_data.mcs_meanlat.attrs["valid_min"] = geolimits[0]
    output_data.mcs_meanlat.attrs["valid_max"] = geolimits[2]

    output_data.mcs_meanlon.attrs[
        "long_name"
    ] = "Mean longitude of the core + cold anvil for each feature in a mcs track"
    output_data.mcs_meanlon.attrs["standard_name"] = "latitude"
    output_data.mcs_meanlon.attrs["units"] = "degrees_north"
    output_data.mcs_meanlon.attrs["valid_min"] = geolimits[1]
    output_data.mcs_meanlon.attrs["valid_max"] = geolimits[3]

    output_data.mcs_corearea.attrs[
        "long_name"
    ] = "Area of the cold core for each feature in a mcs track"
    output_data.mcs_corearea.attrs["units"] = "km^2"

    output_data.mcs_corearea.attrs[
        "long_name"
    ] = "Area of the cold core and cold anvil for each feature in a mcs track"
    output_data.mcs_corearea.attrs["units"] = "km^2"

    output_data.mcs_cloudnumber.attrs[
        "long_name"
    ] = "Number of each cloud in a track that cooresponds to the cloudid map"
    output_data.mcs_cloudnumber.attrs["units"] = "unitless"
    output_data.mcs_cloudnumber.attrs[
        "usuage"
    ] = "To link this tracking statistics file with pixel-level cloudid files, use the cloudidfile and cloudnumber together to identify which cloud this current track and time is associated with"

    output_data.mcs_mergecloudnumber.attrs[
        "long_name"
    ] = "cloud number of small, short-lived feature merging into a mcs track"
    output_data.mcs_mergecloudnumber.attrs["units"] = "unitless"

    output_data.mcs_splitcloudnumber.attrs[
        "long_name"
    ] = "cloud number of small, short-lived feature splitting into a mcs track"
    output_data.mcs_splitcloudnumber.attrs["units"] = "unitless"

    # Write netcdf file
    print(mcstrackstatistics_outfile)
    print("")

    output_data.to_netcdf(
        path=mcstrackstatistics_outfile,
        mode="w",
        format="NETCDF4_CLASSIC",
        unlimited_dims="ntracks",
        encoding={
            "mcs_basetime": {"zlib": True},
            "mcs_datetimestring": {"zlib": True},
            "track_length": {"zlib": True, "_FillValue": -9999},
            "mcs_length": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "mcs_type": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "mcs_status": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "mcs_startstatus": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "mcs_endstatus": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "mcs_boundary": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "mcs_trackinterruptions": {
                "dtype": "int",
                "zlib": True,
                "_FillValue": -9999,
            },
            "mcs_meanlat": {"zlib": True, "_FillValue": np.nan},
            "mcs_meanlon": {"zlib": True, "_FillValue": np.nan},
            "mcs_corearea": {"zlib": True, "_FillValue": np.nan},
            "mcs_ccsarea": {"zlib": True, "_FillValue": np.nan},
            "mcs_cloudnumber": {"zlib": True, "_FillValue": -9999},
            "mcs_mergecloudnumber": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "mcs_splitcloudnumber": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        },
    )
