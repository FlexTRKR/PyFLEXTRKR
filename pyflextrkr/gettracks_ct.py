# Purpose: Track clouds successively from the single track files produced in previous step.

# Author: IDL version written by Sally A. McFarlane (sally.mcfarlane@pnnl.gov) and revised by Zhe Feng (zhe.feng@pnnl.gov). Python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov)

# Define function to track clouds that were identified in merged ir data
def gettracknumbers(
    datasource,
    datadescription,
    datainpath,
    dataoutpath,
    startdate,
    enddate,
    timegap,
    maxnclouds,
    cloudid_filebase,
    npxname,
    tracknumbers_version,
    singletrack_filebase,
    keepsingletrack=0,
    removestartendtracks=0,
    tdimname="time",
    xdimname="lon",
    ydimname="lat",
):
    # Inputs:
    # datasource - source of the data
    # datadescription - description of data source, included in all output file names
    # datainpath - location of the single track files
    # dataoutpath - location where the statistics matrix will be saved
    # startdate - data to start processing in YYYYMMDD format
    # enddate - data to stop processing in YYYYMMDD format
    # timegap - maximum time gap (missing time) allowed (in hours) between two consecutive files
    # maxnclouds - maximum number of clouds allowed to be in one track
    # cloudid_filebase - header of the cloudid data files
    # npxname - variable name of the type of cloud that is being tracked
    # track_version - Version of track single cloud files that will be used
    # singletrack_filebase - header of the single track files

    # Optional keywords
    # keepsingletrack - Keep tracks that only have 1 fram but merge/split with others
    # removestartendtracks - Remove tracks that start at the first file or ends at the last file
    # tdimname - name of time dimension for the output netcdf file
    # xdimname - name of the x-dimension for the output netcdf file
    # ydimname - name of the y-dimentions for the output netcdf

    # Output: (One netcdf file with statistics about each cloud that is tracked in each file)
    # ntracks - number of tracks identified
    # basetimes - seconds since 1970-01-01 of each file
    # track_numbers - track number associate with a cloud
    # track_status - flag that indicates how a cloud evolves over time.
    # track_mergenumbers - track number which a small cloud merges into.
    # track_splitnumbers - track number which a small cloud splits from.
    # track_reset - flag indicating when a track starts or ends in a period with continuous data or whether there is a gap in the data.

    ################################################################################
    # Import modules
    import numpy as np
    import os, fnmatch
    import time, datetime, calendar
    from pytz import utc
    from netCDF4 import Dataset, num2date
    import sys
    import xarray as xr
    import pandas as pd
    import logging

    np.set_printoptions(threshold=np.inf)
    logger = logging.getLogger(__name__)
    #############################################################################
    # Set track numbers output file name
    tracknumbers_filebase = "tracknumbers" + tracknumbers_version
    tracknumbers_outfile = (
        dataoutpath + tracknumbers_filebase + "_" + startdate + "_" + enddate + ".nc"
    )

    ##################################################################################
    # Get single track files sort
    logger.info("Determining which files will be processed")
    logger.info((time.ctime()))
    logger.info(singletrack_filebase)
    singletrackfiles = fnmatch.filter(
        os.listdir(datainpath), singletrack_filebase + "*"
    )
    # Put in temporal order
    singletrackfiles = sorted(singletrackfiles)

    ################################################################################
    # Get date/time from filenames
    nfiles = len(singletrackfiles)
    logger.info("nfiles: ", nfiles)
    year = np.empty(nfiles, dtype=int)
    month = np.empty(nfiles, dtype=int)
    day = np.empty(nfiles, dtype=int)
    hour = np.empty(nfiles, dtype=int)
    minute = np.empty(nfiles, dtype=int)
    basetime = np.empty(nfiles, dtype="datetime64[s]")
    filedate = np.empty(nfiles)
    filetime = np.empty(nfiles)

    header = np.array(len(singletrack_filebase)).astype(int)
    for filestep, ifiles in enumerate(singletrackfiles):
        year[filestep] = int(ifiles[header : header + 4])
        month[filestep] = int(ifiles[header + 4 : header + 6])
        day[filestep] = int(ifiles[header + 6 : header + 8])
        hour[filestep] = int(ifiles[header + 9 : header + 11])
        minute[filestep] = int(ifiles[header + 11 : header + 13])

        TEMP_fulltime = calendar.timegm(
            datetime.datetime(
                year[filestep],
                month[filestep],
                day[filestep],
                hour[filestep],
                minute[filestep],
                0,
                0,
            ).timetuple()
        )
        basetime[filestep] = np.datetime64(
            np.array(
                [pd.to_datetime(TEMP_fulltime, unit="s")][0], dtype="datetime64[s]"
            )
        )

    #############################################################################
    # Keep only files and date/times within start - end time interval
    # Put start and end dates in base time
    TEMP_starttime = calendar.timegm(
        datetime.datetime(
            int(startdate[0:4]),
            int(startdate[4:6]),
            int(startdate[6:8]),
            int(startdate[9:11]),
            int(startdate[11:13]),
            0,
        ).timetuple()
    )
    start_basetime = np.datetime64(
        np.array([pd.to_datetime(TEMP_starttime, unit="s")][0], dtype="datetime64[s]")
    )

    TEMP_endtime = calendar.timegm(
        datetime.datetime(
            int(enddate[0:4]),
            int(enddate[4:6]),
            int(enddate[6:8]),
            int(enddate[9:11]),
            int(enddate[11:13]),
            0,
        ).timetuple()
    )
    end_basetime = np.datetime64(
        np.array([pd.to_datetime(TEMP_endtime, unit="s")][0], dtype="datetime64[s]")
    )

    # Identify files within the start-end date interval
    acceptdates = np.array(
        np.where((basetime >= start_basetime) & (basetime <= end_basetime))
    )[0, :]
    # Isolate files and times with start-end date interval
    basetime = basetime[acceptdates]

    files = [None] * len(acceptdates)

    filedate = [None] * len(acceptdates)
    filetime = [None] * len(acceptdates)
    filesyear = np.zeros(len(acceptdates), dtype=int)
    filesmonth = np.zeros(len(acceptdates), dtype=int)
    filesday = np.zeros(len(acceptdates), dtype=int)
    fileshour = np.zeros(len(acceptdates), dtype=int)
    filesminute = np.zeros(len(acceptdates), dtype=int)

    for filestep, ifiles in enumerate(acceptdates):
        files[filestep] = singletrackfiles[ifiles]
        filedate[filestep] = (
            str(year[ifiles]) + str(month[ifiles]).zfill(2) + str(day[ifiles]).zfill(2)
        )
        filetime[filestep] = str(hour[ifiles]).zfill(2) + str(minute[ifiles]).zfill(2)
        filesyear[filestep] = int(year[ifiles])
        filesmonth[filestep] = int(month[ifiles])
        filesday[filestep] = int(day[ifiles])
        fileshour[filestep] = int(hour[ifiles])
        filesminute[filestep] = int(minute[ifiles])

    #########################################################################
    # Determine number of gaps in dataset
    gap = 0
    for ifiles in range(1, len(acceptdates)):
        newtime = datetime.datetime(
            filesyear[ifiles],
            filesmonth[ifiles],
            filesday[ifiles],
            fileshour[ifiles],
            filesminute[ifiles],
            0,
            0,
            tzinfo=utc,
        )
        referencetime = datetime.datetime(
            filesyear[ifiles - 1],
            filesmonth[ifiles - 1],
            filesday[ifiles - 1],
            fileshour[ifiles - 1],
            filesminute[ifiles - 1],
            0,
            0,
            tzinfo=utc,
        )

        cutofftime = newtime - datetime.timedelta(minutes=timegap * 60)
        if cutofftime > referencetime:
            gap = gap + 1

    # import pdb; pdb.set_trace()
    # KB HARDCODED GAP
    # gap = 0
    ############################################################################
    # Initialize matrices
    nfiles = (
        int(len(files)) + 2 * gap
    )  # seems a bug, by Jianfeng Li, 2*gap may be not enough

    tracknumber = np.ones((1, nfiles, maxnclouds), dtype=int) * -9999
    referencetrackstatus = np.ones((nfiles, maxnclouds), dtype=float) * np.nan
    newtrackstatus = np.ones((nfiles, maxnclouds), dtype=float) * np.nan
    trackstatus = np.ones((1, nfiles, maxnclouds), dtype=int) * -9999
    trackmergenumber = np.ones((1, nfiles, maxnclouds), dtype=int) * -9999
    tracksplitnumber = np.ones((1, nfiles, maxnclouds), dtype=int) * -9999
    basetime = np.empty(nfiles, dtype="datetime64[s]")
    trackreset = np.ones((1, nfiles, maxnclouds), dtype=int) * -9999

    ############################################################################
    # Load first file
    logger.info("Processing first file")
    #    logger.info((time.ctime()))
    logger.info("datainpath: ", datainpath)
    logger.info("files[0]: ", files[0])
    singletracking_data = Dataset(datainpath + files[0], "r")  # Open file

    nclouds_reference = int(
        np.nanmax(singletracking_data["nclouds_ref"][:]) + 1
    )  # Number of clouds in reference file
    basetime_ref = singletracking_data["basetime_ref"][:]
    basetime_units = singletracking_data["basetime_ref"].units
    basetime_calendar = singletracking_data["basetime_ref"].calendar
    ref_file = singletracking_data.getncattr("ref_file")
    singletracking_data.close()

    # Make sure number of clouds does not exceed maximum. If does indicates over-segmenting data
    if nclouds_reference > maxnclouds:
        sys.exit(
            "# of clouds in reference file exceed allowed maximum number of clouds"
        )

    # Isolate file name and add it to the filelist
    basetime[0] = np.array(
        [
            pd.to_datetime(
                num2date(basetime_ref, units=basetime_units, calendar=basetime_calendar)
            )
        ],
        dtype="datetime64[s]",
    )[0, 0]

    temp_referencefile = os.path.basename(ref_file)
    strlength = len(temp_referencefile)
    cloudidfiles = np.chararray((nfiles, int(strlength)))
    cloudidfiles[0, :] = list(os.path.basename(ref_file))

    # Initate track numbers
    tracknumber[0, 0, 0 : int(nclouds_reference)] = (
        np.arange(0, int(nclouds_reference)) + 1
    )
    itrack = nclouds_reference + 1

    # Rocord that the tracks are being reset / initialized
    trackreset[0, 0, :] = 1

    ###########################################################################
    # Loop over files and generate tracks
    logger.info("Loop through the rest of the files")
    logger.info(("Number of files: " + str(nfiles)))
    #    logger.info((time.ctime()))
    ifill = 0
    for ifile in range(0, nfiles):  # use range(0, nfiles) by Jianfeng Li
        logger.info((files[ifile]))
        logger.info((time.ctime()))

        ######################################################################
        # Load single track file
        logger.info("Load track data")
        #        logger.info((time.ctime()))
        singletracking_data = Dataset(datainpath + files[ifile], "r")  # Open file
        nclouds_reference = int(
            np.nanmax(singletracking_data["nclouds_ref"][:]) + 1
        )  # Number of clouds in reference file
        nclouds_new = int(np.nanmax(singletracking_data["nclouds_new"][:]) + 1)
        basetime_ref = singletracking_data["basetime_ref"][:]
        basetime_new = singletracking_data["basetime_new"][:]
        basetime_units = singletracking_data["basetime_ref"].units
        basetime_calendar = singletracking_data[
            "basetime_ref"
        ].calendar  # Number of clouds in new file
        refcloud_forward_index = singletracking_data["refcloud_forward_index"][
            :
        ].astype(
            int
        )  # Each row represents a cloud in the reference file and the numbers in that row are indices of clouds in new file linked that cloud in the reference file
        newcloud_backward_index = singletracking_data["newcloud_backward_index"][
            :
        ].astype(
            int
        )  # Each row represents a cloud in the new file and the numbers in that row are indices of clouds in the reference file linked that cloud in the new file
        ref_file = singletracking_data.getncattr("ref_file")
        new_file = singletracking_data.getncattr("new_file")
        # ref_file = datainpath + ref_file[-36:] # to change the path after moving the files
        # new_file = datainpath + new_file[-36:]
        ref_date = singletracking_data.getncattr("ref_date")
        new_date = singletracking_data.getncattr("new_date")

        singletracking_data.close()

        # Make sure number of clouds does not exceed maximum. If does indicates over-segmenting data
        if nclouds_reference > maxnclouds:
            sys.exit(
                "# of clouds in reference file exceed allowed maximum number of clouds"
            )

        ########################################################################
        # Load cloudid files
        logger.info("Load cloudid files")
        #        logger.info((time.ctime()))
        # Reference cloudid file
        referencecloudid_data = Dataset(ref_file, "r")
        npix_reference = referencecloudid_data[npxname][:]
        logger.info("npix_reference.shape: ", npix_reference.shape)
        referencecloudid_data.close()

        # New cloudid file
        newcloudid_data = Dataset(new_file, "r")
        npix_new = newcloudid_data[npxname][:]
        newcloudid_data.close()

        # Remove possible extra time dimension to make sure npix is a 1D array
        # npix_reference = npix_reference.squeeze()
        # npix_new = npix_new.squeeze()

        ########################################################################
        # Check time gap between consecutive track files
        logger.info("Checking if time gap between files satisfactory")
        #        logger.info((time.ctime()))

        # Set previous and new times
        if ifile < 1:
            time_prev = np.copy(basetime_new[0])

        time_new = np.copy(basetime_new[0])

        # Check if files immediately follow each other. Missing files can exist. If missing files exist need to incrament index and track numbers
        if ifile > 0:
            hour_diff = np.array([time_new - time_prev]).astype(float)
            if hour_diff > (timegap * 3.6 * 10 ** 12):
                logger.info(("Track terminates on: " + ref_date))
                logger.info(("Time difference: " + str(hour_diff)))
                logger.info(("Maximum timegap allowed: " + str(timegap)))
                logger.info(("New track starts on: " + new_date))

                # Flag the previous file as the last file
                trackreset[0, ifill, :] = 2

                ifill = ifill + 2

                # Fill tracking matrices with reference data and record that the track ended
                cloudidfiles[ifill, :] = list(os.path.basename(ref_file))
                basetime[ifill] = np.array(
                    [
                        pd.to_datetime(
                            num2date(
                                basetime_ref,
                                units=basetime_units,
                                calendar=basetime_calendar,
                            )
                        )
                    ],
                    dtype="datetime64[s]",
                )[0, 0]

                # Record that break in data occurs
                trackreset[0, ifill, :] = 1

                # Treat all clouds in the reference file as new clouds
                for ncr in range(1, nclouds_reference + 1):
                    tracknumber[0, ifill, ncr - 1] = itrack
                    itrack = itrack + 1

        time_prev = time_new
        cloudidfiles[ifill + 1, :] = list(os.path.basename(new_file))
        basetime[ifill + 1] = np.array(
            [
                pd.to_datetime(
                    num2date(
                        basetime_new, units=basetime_units, calendar=basetime_calendar
                    )
                )
            ],
            dtype="datetime64[s]",
        )[0, 0]

        ########################################################################################
        # Compare forward and backward single track matirces to link new and reference clouds
        # Intiailize matrix for this time period
        logger.info("Generating tracks")
        logger.info((time.ctime()))
        trackfound = np.ones(nclouds_reference + 1, dtype=int) * -9999

        # Loop over all reference clouds
        logger.info("Looping over all clouds in the reference file")
        logger.info(("Number of clouds to process: " + str(nclouds_reference)))
        #        logger.info((time.ctime()))
        for ncr in np.arange(
            1, nclouds_reference + 1
        ):  # Looping over each reference cloud. Start at 1 since clouds numbered starting at 1.
            logger.info(("Reference cloud #: " + str(ncr)))
            #            logger.info((time.ctime()))
            if trackfound[ncr - 1] < 1:

                # Find all clouds (both forward and backward) associated with this reference cloud
                nreferenceclouds = 0
                ntemp_referenceclouds = 1  # Start by forcing to see if track exists
                temp_referenceclouds = [ncr]

                trackpresent = 0
                logger.info("Finding all associated clouds")
                #                logger.info((time.ctime()))
                while ntemp_referenceclouds > nreferenceclouds:
                    associated_referenceclouds = np.copy(temp_referenceclouds).astype(
                        int
                    )
                    nreferenceclouds = ntemp_referenceclouds

                    for nr in range(0, nreferenceclouds):
                        logger.info(("Processing cloud #: " + str(nr)))
                        #                        logger.info((time.ctime()))
                        tempncr = associated_referenceclouds[nr]

                        # Find indices of forward linked clouds.
                        newforwardindex = np.array(
                            np.where(refcloud_forward_index[0, tempncr - 1, :] > 0)
                        )  # Need to subtract one since looping based on core number and since python starts with indices at zero. Row of that core is one less than its number.
                        nnewforward = np.shape(newforwardindex)[1]
                        if nnewforward > 0:
                            core_newforward = refcloud_forward_index[
                                0, tempncr - 1, newforwardindex[0, :]
                            ]

                        # Find indices of backwards linked clouds
                        newbackwardindex = np.array(
                            np.where(newcloud_backward_index[0, :, :] == tempncr)
                        )
                        nnewbackward = np.shape(newbackwardindex)[1]
                        if nnewbackward > 0:
                            core_newbackward = (
                                newbackwardindex[0, :] + 1
                            )  # Need to add one since want the core index, which starts at one. But this is using that row number, which starts at zero.

                        # Put all the indices associated with new clouds linked to the reference cloud in one vector
                        if nnewforward > 0:
                            if trackpresent == 0:
                                associated_newclouds = core_newforward[:].astype(int)
                                trackpresent = trackpresent + 1
                            else:
                                associated_newclouds = np.append(
                                    associated_newclouds, core_newforward.astype(int)
                                )

                        if nnewbackward > 0:
                            if trackpresent == 0:
                                associated_newclouds = core_newbackward[:]
                                trackpresent = trackpresent + 1
                            else:
                                associated_newclouds = np.append(
                                    associated_newclouds, core_newbackward.astype(int)
                                )

                        if nnewbackward == 0 and nnewforward == 0:
                            associated_newclouds = []

                        # If the reference cloud is linked to a new cloud
                        if trackpresent > 0:
                            # Sort and find the unique new clouds associated with the reference cloud
                            if len(associated_newclouds) > 1:
                                associated_newclouds = np.unique(
                                    np.sort(associated_newclouds)
                                )
                            nnewclouds = len(associated_newclouds)

                            # Find reference clouds associated with each new cloud. Look to see if these new clouds are linked to other cells in the reference file as well.
                            for nnew in range(0, nnewclouds):
                                # Find associated reference clouds
                                referencecloudindex = np.array(
                                    np.where(
                                        refcloud_forward_index[0, :, :]
                                        == associated_newclouds[nnew]
                                    )
                                )
                                nassociatedreference = np.shape(referencecloudindex)[1]
                                if nassociatedreference > 0:
                                    temp_referenceclouds = np.append(
                                        temp_referenceclouds, referencecloudindex[0] + 1
                                    )
                                    temp_referenceclouds = np.unique(
                                        np.sort(temp_referenceclouds)
                                    )

                            ntemp_referenceclouds = len(temp_referenceclouds)
                        else:
                            nnewclouds = 0

                #################################################################
                # Now get the track status
                logger.info("Determining status of clouds in track")
                #                logger.info((time.ctime()))
                if nnewclouds > 0:
                    ############################################################
                    # Find the largest reference and new clouds
                    # Largest reference cloud
                    # allreferencepix = npix_reference[0, associated_referenceclouds-1] # Need to subtract one since associated_referenceclouds gives core index and matrix starts at zero

                    allreferencepix = npix_reference[
                        associated_referenceclouds - 1
                    ]  # Need to subtract one since associated_referenceclouds gives core index and matrix starts at zero
                    largestreferenceindex = np.argmax(allreferencepix)
                    largest_referencecloud = associated_referenceclouds[
                        largestreferenceindex
                    ]  # Cloud number of the largest reference cloud

                    # Largest new cloud
                    # allnewpix = npix_new[0, associated_newclouds-1] # Need to subtract one since associated_newclouds gives cloud number and the matrix starts at zero
                    allnewpix = npix_new[
                        associated_newclouds - 1
                    ]  # Need to subtract one since associated_newclouds gives cloud number and the matrix starts at zero
                    largestnewindex = np.argmax(allnewpix)
                    largest_newcloud = associated_newclouds[
                        largestnewindex
                    ]  # Cloud numberof the largest new cloud

                    # logger.info(associated_referenceclouds)
                    # logger.info(associated_newclouds)

                    if nnewclouds == 1 and nreferenceclouds == 1:
                        ############################################################
                        # Simple continuation
                        logger.info("Simple continuation")
                        #                        logger.info((time.ctime()))

                        # Check trackstatus already has a valid value. This will prtrack splits from a previous step being overwritten

                        # logger.info(trackstatus[ifill,ncr-1])
                        referencetrackstatus[ifill, ncr - 1] = 1
                        trackfound[ncr - 1] = 1
                        tracknumber[0, ifill + 1, associated_newclouds - 1] = np.copy(
                            tracknumber[0, ifill, ncr - 1]
                        )

                    elif nreferenceclouds > 1:
                        ##############################################################
                        # Merging only

                        # Loop through the reference clouds and assign th track to the largestst one, the rest just go away
                        if nnewclouds == 1:
                            logger.info("Merge only")
                            #                            logger.info((time.ctime()))
                            for tempreferencecloud in associated_referenceclouds:
                                trackfound[tempreferencecloud - 1] = 1

                                # logger.info(trackstatus[ifill,tempreferencecloud-1])

                                # If this reference cloud is the largest fragment of the merger, label this reference time (file) as the larger part of merger (2) and merging at the next time (ifile + 1)
                                if tempreferencecloud == largest_referencecloud:
                                    referencetrackstatus[
                                        ifill, tempreferencecloud - 1
                                    ] = 2
                                    tracknumber[
                                        0, ifill + 1, associated_newclouds - 1
                                    ] = np.copy(
                                        tracknumber[
                                            0, ifill, largest_referencecloud - 1
                                        ]
                                    )
                                # If this reference cloud is the smaller fragment of the merger, label the reference time (ifile) as the small merger (12) and merging at the next time (file + 1)
                                else:
                                    referencetrackstatus[
                                        ifill, tempreferencecloud - 1
                                    ] = 21
                                    trackmergenumber[
                                        0, ifill, tempreferencecloud - 1
                                    ] = np.copy(
                                        tracknumber[
                                            0, ifill, largest_referencecloud - 1
                                        ]
                                    )

                        #################################################################
                        # Merging and spliting
                        else:
                            logger.info("Merging and splitting")
                            #                            logger.info((time.ctime()))

                            # Loop over the reference clouds and assign the track the largest one
                            for tempreferencecloud in associated_referenceclouds:
                                trackfound[tempreferencecloud - 1] = 1

                                # If this is the larger fragment ofthe merger, label the reference time (ifill) as large merger (2) and the actual merging track at the next time [ifill+1]
                                if tempreferencecloud == largest_referencecloud:
                                    referencetrackstatus[
                                        ifill, tempreferencecloud - 1
                                    ] = (2 + 13)
                                    tracknumber[
                                        0, ifill + 1, largest_newcloud - 1
                                    ] = np.copy(
                                        tracknumber[
                                            0, ifill, largest_referencecloud - 1
                                        ]
                                    )
                                # For the smaller fragment of the merger, label the reference time (ifill) as the small merge and have the actual merging occur at the next time (ifill+1)
                                else:
                                    referencetrackstatus[
                                        ifill, tempreferencecloud - 1
                                    ] = (21 + 13)
                                    trackmergenumber[
                                        0, ifill, tempreferencecloud - 1
                                    ] = np.copy(
                                        tracknumber[
                                            0, ifill, largest_referencecloud - 1
                                        ]
                                    )

                            # Loop through the new clouds and assign the smaller ones a new track
                            for tempnewcloud in associated_newclouds:

                                # For the smaller fragment of the split, label the new time (ifill+1) as the small split because the cloud only occurs at the new time step
                                if tempnewcloud != largest_newcloud:
                                    newtrackstatus[ifill + 1, tempnewcloud - 1] = 31

                                    tracknumber[0, ifill + 1, tempnewcloud - 1] = itrack
                                    itrack = itrack + 1

                                    tracksplitnumber[
                                        0, ifill + 1, tempnewcloud - 1
                                    ] = np.copy(
                                        tracknumber[
                                            0, ifill, largest_referencecloud - 1
                                        ]
                                    )

                                    trackreset[0, ifill + 1, tempnewcloud - 1] = 0
                                # For the larger fragment of the split, label the new time (ifill+1) as the large split so that is consistent with the small fragments. The track continues to follow this cloud so the tracknumber is not incramented.
                                else:
                                    newtrackstatus[ifill + 1, tempnewcloud - 1] = 3
                                    tracknumber[
                                        0, ifill + 1, tempnewcloud - 1
                                    ] = np.copy(
                                        tracknumber[
                                            0, ifill, largest_referencecloud - 1
                                        ]
                                    )

                    #####################################################################
                    # Splitting only
                    elif nnewclouds > 1:
                        logger.info("Splitting only")
                        #                        logger.info((time.ctime()))
                        # Label reference cloud as a pure split
                        referencetrackstatus[ifill, ncr - 1] = 13
                        tracknumber[0, ifill, ncr - 1] = np.copy(
                            tracknumber[0, ifill, largest_referencecloud - 1]
                        )

                        # Loop over the clouds and assign new tracks to the smaller ones
                        for tempnewcloud in associated_newclouds:
                            # For the smaller fragment of the split, label the new time (ifill+1) as teh small split (13) because the cloud only occurs at the new time.
                            if tempnewcloud != largest_newcloud:
                                newtrackstatus[ifill + 1, tempnewcloud - 1] = 31

                                tracknumber[0, ifill + 1, tempnewcloud - 1] = itrack
                                itrack = itrack + 1

                                tracksplitnumber[
                                    0, ifill + 1, tempnewcloud - 1
                                ] = np.copy(tracknumber[0, ifill, ncr - 1])

                                trackreset[0, ifill + 1, tempnewcloud - 1] = 0
                            # For the larger fragment of the split, label new time (ifill+1) as the large split (3) so that is consistent with the small fragments
                            else:
                                newtrackstatus[ifill + 1, tempnewcloud - 1] = 3
                                tracknumber[0, ifill + 1, tempnewcloud - 1] = np.copy(
                                    tracknumber[0, ifill, ncr - 1]
                                )

                    else:
                        sys.exit(str(ncr) + " How did we get here?")

                ######################################################################################
                # No new clouds. Track dissipated
                else:
                    logger.info("Track ending")
                    #                    logger.info((time.ctime()))

                    trackfound[ncr - 1] = 1

                    referencetrackstatus[ifill, ncr - 1] = 0

        ##############################################################################
        # Find any clouds in the new track that don't have a track number. These are new clouds this file
        logger.info("Identifying new tracks")
        #        logger.info((time.ctime()))
        for ncn in range(1, int(nclouds_new) + 1):
            if tracknumber[0, ifill + 1, ncn - 1] < 0:
                tracknumber[0, ifill + 1, ncn - 1] = itrack
                itrack = itrack + 1

                trackreset[0, ifill + 1, ncn - 1] = 0

        #############################################################################
        # Flag the last file in the dataset
        if ifile == nfiles - 2:
            logger.info("WE ARE AT THE LAST FILE")
            for ncn in range(1, int(nclouds_new) + 1):
                trackreset[0, ifill + 1, :] = 2
            break

        ##############################################################################
        # Increment to next fill
        ifill = ifill + 1

    trackstatus[0, :, :] = np.nansum(
        np.dstack((referencetrackstatus, newtrackstatus)), 2
    )
    trackstatus[np.isnan(trackstatus)] = -9999

    logger.info("Tracking Done")

    #################################################################
    # Create histograms of the values in tracknumber. This effectively counts the number of times each track number appaers in tracknumber, which is equivalent to calculating the length of the track.
    tracklengths, trackbins = np.histogram(
        np.copy(tracknumber[0, :, :]),
        bins=np.arange(1, itrack + 1, 1),
        range=(1, itrack + 1),
    )

    #################################################################
    # Remove all tracks that have only one cloud.
    logger.info("Removing short tracks")
    #    logger.info((time.ctime()))

    # Identify single cloud tracks
    singletracks = np.array(np.where(tracklengths <= 1))[0, :]
    # singletracks = np.array(np.where(tracklengths <= 6))[0,:] # KB changed to decrease total number of tracks to be analyzed
    nsingletracks = len(singletracks)
    logger.info("nsingletracks: ", nsingletracks)
    # singleindices = np.logical_or(tracknumber[0, :, :] == singletracks)

    # Loop over single cloudtracks
    nsingleremove = 0
    for strack in singletracks:
        logger.info(f"single track: {str(strack + 1)}")
        logger.info("Getting track index")
        # Indentify clouds in this track
        cloudindex = np.array(
            np.where(tracknumber[0, :, :] == int(strack + 1))
        )  # Need to add one since singletracks lists the index in the matrix, which starts at zero. Track number starts at one.

        logger.info("Filtering data")
        # Only remove single track if it is not small merger or small split. This is only done if keepsingletrack == 1. This is the default.
        if keepsingletrack == 1:
            if (
                tracksplitnumber[0, cloudindex[0], cloudindex[1]] < 0
                and trackmergenumber[0, cloudindex[0], cloudindex[1]] < 0
            ):
                tracknumber[0, cloudindex[0], cloudindex[1]] = -2
                trackstatus[0, cloudindex[0], cloudindex[1]] = -9999
                nsingleremove = nsingleremove + 1
                tracklengths[strack] = -9999

        # Remove all single tracks. This corresponds to keepsingletrack == 0.
        else:
            logger.info("we are in this loop")
            logger.info(f"nsingletracks: {nsingletracks})
            tracknumber[0, cloudindex[0], cloudindex[1]] = -2
            trackstatus[0, cloudindex[0], cloudindex[1]] = -9999
            nsingleremove = nsingleremove + 1
            tracklengths[strack] = -9999

    #######################################################################
    # Save file
    logger.info("Writing all track statistics file")
    logger.info((time.ctime()))
    logger.info(tracknumbers_outfile)
    logger.info("")

    # Check if file already exists. If exists, delete
    if os.path.isfile(tracknumbers_outfile):
        os.remove(tracknumbers_outfile)

    output_data = xr.Dataset(
        {
            "ntracks": (["time"], np.array([itrack])),
            "basetimes": (["nfiles"], basetime),
            "cloudid_files": (["nfiles", "ncharacters"], cloudidfiles),
            "track_numbers": (["time", "nfiles", "nclouds"], tracknumber),
            "track_status": (["time", "nfiles", "nclouds"], trackstatus.astype(int)),
            "track_mergenumbers": (["time", "nfiles", "nclouds"], trackmergenumber),
            "track_splitnumbers": (["time", "nfiles", "nclouds"], tracksplitnumber),
            "track_reset": (["time", "nfiles", "nclouds"], trackreset),
        },
        coords={
            "time": (["time"], np.arange(0, 1)),
            "nfiles": (["nfiles"], np.arange(nfiles)),
            "nclouds": (["nclouds"], np.arange(0, maxnclouds)),
            "ncharacters": (["ncharacters"], np.arange(0, strlength)),
        },
        attrs={
            "Title": "Indicates the track each cloud is linked to. Flags indicate how the clouds transition(evolve) between files.",
            "Conventions": "CF-1.6",
            "Insitution": "Pacific Northwest National Laboratory",
            "Contact": "Katelyn Barber: katelyn.barber@pnnl.gov",
            "Created": time.ctime(time.time()),
            "source": datasource,
            "description": datadescription,
            "singletrack_filebase": singletrack_filebase,
            "startdate": startdate,
            "enddate": enddate,
            "timegap": str(timegap) + "-hours",
        },
    )

    # Set variable attributes
    output_data.ntracks.attrs["long_name"] = "number of cloud tracks"
    output_data.ntracks.attrs["units"] = "unitless"

    output_data.basetimes.attrs[
        "long_name"
    ] = "epoch time (seconds since 01/01/1970 00:00) of cloudid_files"
    output_data.basetimes.attrs["standard_name"] = "time"

    output_data.cloudid_files.attrs[
        "long_name"
    ] = "filename of each cloudid file used during tracking"
    output_data.cloudid_files.attrs["units"] = "unitless"

    output_data.track_numbers.attrs["long_name"] = "cloud track number"
    output_data.track_numbers.attrs[
        "usage"
    ] = "Each row represents a cloudid file. Each row represents a cloud in that file. The number indicates the track associate with that cloud. This follows the largest cloud in mergers and splits."
    output_data.track_numbers.attrs["units"] = "unitless"
    output_data.track_numbers.attrs["valid_min"] = 1
    output_data.track_numbers.attrs["valid_max"] = itrack - 1

    output_data.track_status.attrs[
        "long_name"
    ] = "Flag indicating evolution / behavior for each cloud in a track"
    output_data.track_status.attrs["units"] = "unitless"
    output_data.track_status.attrs["valid_min"] = 0
    output_data.track_status.attrs["valid_max"] = 65

    output_data.track_mergenumbers.attrs[
        "long_name"
    ] = "Number of the track that this small cloud merges into"
    output_data.track_mergenumbers.attrs[
        "usage"
    ] = "Each row represents a cloudid file. Each column represets a cloud in that file. Numbers give the track number associated with the small clouds in mergers."
    output_data.track_mergenumbers.attrs["units"] = "unitless"
    output_data.track_mergenumbers.attrs["valid_min"] = 1
    output_data.track_mergenumbers.attrs["valid_max"] = itrack - 1

    output_data.track_splitnumbers.attrs[
        "long_name"
    ] = "Number of the track that this small cloud splits from"
    output_data.track_splitnumbers.attrs[
        "usage"
    ] = "Each row represents a cloudid file. Each column represets a cloud in that file. Numbers give the track number associated with the small clouds in the split"
    output_data.track_splitnumbers.attrs["units"] = "unitless"
    output_data.track_splitnumbers.attrs["valid_min"] = 1
    output_data.track_splitnumbers.attrs["valid_max"] = itrack - 1

    output_data.track_reset.attrs[
        "long_name"
    ] = "flag of track starts and adrupt track stops"
    output_data.track_reset.attrs[
        "usage"
    ] = "Each row represents a cloudid file. Each column represents a cloud in that file. Numbers indicate if the track started or adruptly ended during this file."
    output_data.track_reset.attrs[
        "values"
    ] = "0=Track starts and ends within a period of continuous data. 1=Track starts as the first file in the data set or after a data gap. 2=Track ends because data ends or gap in data."
    output_data.track_reset.attrs["units"] = "unitless"
    output_data.track_reset.attrs["valid_min"] = 0
    output_data.track_reset.attrs["valid_max"] = 2

    # Write netcdf file
    output_data.to_netcdf(
        path=tracknumbers_outfile,
        mode="w",
        format="NETCDF4_CLASSIC",
        unlimited_dims="ntracks",
        encoding={
            "ntracks": {"dtype": "int", "zlib": True},
            "basetimes": {
                "dtype": "int64",
                "zlib": True,
                "units": "seconds since 1970-01-01",
            },
            "cloudid_files": {"zlib": True,},
            "track_numbers": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "track_status": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "track_mergenumbers": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "track_splitnumbers": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "track_reset": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        },
    )
