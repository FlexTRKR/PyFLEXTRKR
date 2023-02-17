import numpy as np
import time
import sys
import os
from netCDF4 import Dataset
import xarray as xr
import logging
from pyflextrkr.ft_utilities import subset_files_timerange

def gettracknumbers(config):
    """
    Track features sequentially from the single track files.

    Arguments:
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        tracknumbers_outfile: string
            Track numbers output filename.
    """

    # Get parameters from config
    singletrack_filebase = config["singletrack_filebase"]
    tracknumbers_filebase = config["tracknumbers_filebase"]
    tracking_outpath = config["tracking_outpath"]
    stats_outpath = config["stats_outpath"]
    startdate = config["startdate"]
    enddate = config["enddate"]
    timegap = config["timegap"]
    maxnclouds = config["maxnclouds"]
    featuresize_varname = config.get("featuresize_varname", "npix_feature")
    start_basetime = config["start_basetime"]
    end_basetime = config["end_basetime"]
    fillval = config["fillval"]

    logger = logging.getLogger(__name__)
    np.set_printoptions(threshold=np.inf)
    logger.info('Tracking features sequentially from single track files')

    # Set track numbers output file name
    tracknumbers_outfile = f"{stats_outpath}{tracknumbers_filebase}{startdate}_{enddate}.nc"

    # Identify files to process
    files, \
    files_basetime, \
    files_datestring, \
    files_timestring = subset_files_timerange(tracking_outpath,
                                              singletrack_filebase,
                                              start_basetime,
                                              end_basetime)

    ############################################################################
    # Initialize matrices
    nfiles = len(files)
    logger.info(f"Total number of files to process: {nfiles}")

    fillval_f = np.nan
    missingfrac = 0.3
    nfiles_m = int(nfiles*(1.+missingfrac))
    tracknumber = np.full((1, nfiles_m, maxnclouds), fillval, dtype=int)
    referencetrackstatus = np.full((nfiles_m, maxnclouds), fillval_f, dtype=float)
    newtrackstatus = np.full((nfiles_m, maxnclouds), fillval_f, dtype=float)
    trackstatus = np.full((1, nfiles_m, maxnclouds), fillval, dtype=int)
    trackmergenumber = np.full((1, nfiles_m, maxnclouds), fillval, dtype=int)
    tracksplitnumber = np.full((1, nfiles_m, maxnclouds), fillval, dtype=int)
    basetime = np.empty(nfiles_m, dtype="datetime64[s]")
    trackreset = np.full((1, nfiles_m, maxnclouds), fillval, dtype=int)

    ############################################################################
    # Load first file
    logger.debug("Processing first file")
    logger.debug(f"tracking_outpath: {tracking_outpath}")
    logger.debug(f"files[0]: {files[0]}")
    # singletracking_data = Dataset(tracking_outpath + files[0], "r")
    singletracking_data = Dataset(files[0], "r")

    # Number of clouds in reference file
    nclouds_reference = int(np.nanmax(singletracking_data["nclouds_ref"][:]) + 1)
    basetime_ref = singletracking_data["basetime_ref"][:]
    ref_file = f"{tracking_outpath}{singletracking_data.getncattr('ref_file')}"
    singletracking_data.close()

    # Make sure number of clouds does not exceed maximum.
    if nclouds_reference > maxnclouds:
        logger.critical(f"Error: Number of clouds in reference file exceed allowed maximum number of clouds")
        logger.critical(f"nclouds_reference: {nclouds_reference}, nmaxclouds: {maxnclouds}")
        logger.critical("Increase maxnclouds in the config file.")
        sys.exit("Code exits in gettracks.py")

    # Isolate file name and add it to the filelist
    basetime[0] = basetime_ref.item()

    temp_referencefile = os.path.basename(ref_file)
    strlength = len(temp_referencefile)
    cloudidfiles = np.chararray((nfiles_m, int(strlength)))
    cloudidfiles[0, :] = list(os.path.basename(ref_file))

    # Initate track numbers
    tracknumber[0, 0, 0 : int(nclouds_reference)] = (
        np.arange(0, int(nclouds_reference)) + 1
    )
    itrack = nclouds_reference + 1

    # Record that the tracks are being reset / initialized
    trackreset[0, 0, :] = 1

    ###########################################################################
    # Loop over files and generate tracks
    logger.debug("Loop through the rest of the files")
    logger.debug(f"Number of files: {str(nfiles)}")
    logger.debug((time.ctime()))
    ifill = 0

    for ifile in range(0, nfiles):
        logger.info(os.path.basename(files[ifile]))

        ######################################################################
        # Load single track file
        # logger.debug('Load track data')
        # logger.debug((time.ctime()))
        # singletracking_data = Dataset(tracking_outpath + files[ifile], "r")
        singletracking_data = Dataset(files[ifile], "r")
        # Number of clouds in reference file
        nclouds_reference = int(np.nanmax(singletracking_data["nclouds_ref"][:]) + 1)
        nclouds_new = int(np.nanmax(singletracking_data["nclouds_new"][:]) + 1)
        basetime_ref = singletracking_data["basetime_ref"][:]
        basetime_new = singletracking_data["basetime_new"][:]
        # Number of clouds in new file
        refcloud_forward_index = singletracking_data["refcloud_forward_index"][:].astype(int)
        # Each row represents a cloud in the reference file and
        # the numbers in that row are indices of clouds in new file linked that cloud in the reference file
        newcloud_backward_index = singletracking_data["newcloud_backward_index"][:].astype(int)
        # Each row represents a cloud in the new file and
        # the numbers in that row are indices of clouds in the reference file linked that cloud in the new file
        ref_file = f"{tracking_outpath}{singletracking_data.getncattr('ref_file')}"
        new_file = f"{tracking_outpath}{singletracking_data.getncattr('new_file')}"
        ref_date = f"{tracking_outpath}{singletracking_data.getncattr('ref_date')}"
        new_date = f"{tracking_outpath}{singletracking_data.getncattr('new_date')}"

        singletracking_data.close()

        # Make sure number of clouds does not exceed maximum
        if nclouds_reference > maxnclouds:
            logger.critical(f"Error: Number of clouds in reference file exceed allowed maximum number of clouds")
            logger.critical(f"nclouds_reference: {nclouds_reference}, nmaxclouds: {maxnclouds}")
            logger.critical("Increase maxnclouds in the config file.")
            sys.exit("Code exits in gettracks.py")

        ########################################################################
        # Load cloudid files
        # logger.debug('Load cloudid files')
        # logger.debug((time.ctime()))
        # Reference cloudid file
        referencecloudid_data = Dataset(ref_file, "r")
        npix_reference = referencecloudid_data[featuresize_varname][:]
        referencecloudid_data.close()

        # New cloudid file
        newcloudid_data = Dataset(new_file, "r")
        npix_new = newcloudid_data[featuresize_varname][:]
        newcloudid_data.close()

        # Remove possible extra time dimension to make sure npix is a 1D array
        # npix_reference = npix_reference.squeeze()
        # npix_new = npix_new.squeeze()

        ########################################################################
        # Check time gap between consecutive track files
        # logger.debug('Checking if time gap between files satisfactory')
        # logger.debug((time.ctime()))

        # Set previous and new times
        if ifile < 1:
            time_prev = np.copy(basetime_new[0])

        time_new = np.copy(basetime_new[0])

        # Check if files immediately follow each other. Missing files can exist.
        # If missing files exist need to increment index and track numbers
        if ifile > 0:
            hour_diff = np.array([time_new - time_prev]).astype(float)
            if hour_diff > (timegap * 3.6 * 10 ** 12):
                logger.debug(f"Track terminates on: {ref_date}")
                logger.debug(f"Time difference: {str(hour_diff)}")
                logger.debug(f"Maximum timegap allowed: {str(timegap)}")
                logger.debug(f"New track starts on: {new_date}")

                # Flag the previous file as the last file
                trackreset[0, ifill, :] = 2

                ifill = ifill + 2

                # Fill tracking matrices with reference data and record that the track ended
                cloudidfiles[ifill, :] = list(os.path.basename(ref_file))
                basetime[ifill] = basetime_ref.item()

                # Record that break in data occurs
                trackreset[0, ifill, :] = 1

                # Treat all clouds in the reference file as new clouds
                for ncr in range(1, nclouds_reference + 1):
                    tracknumber[0, ifill, ncr - 1] = itrack
                    itrack = itrack + 1

        time_prev = time_new
        cloudidfiles[ifill + 1, :] = list(os.path.basename(new_file))
        basetime[ifill + 1] = basetime_new.item()

        ########################################################################################
        # Compare forward and backward single track matirces to link new and reference clouds
        # Intiailize matrix for this time period
        # logger.debug('Generating tracks')
        # logger.debug((time.ctime()))
        trackfound = np.ones(nclouds_reference + 1, dtype=int) * -9999

        # Loop over all reference clouds
        # logger.debug('Looping over all clouds in the reference file')
        # logger.debug(('Number of clouds to process: ' + str(nclouds_reference)))
        # logger.debug((time.ctime()))
        for ncr in np.arange(
            1, nclouds_reference + 1
        ):  # Looping over each reference cloud. Start at 1 since clouds numbered starting at 1.
            # logger.debug(('Reference cloud #: ' + str(ncr)))
            # logger.debug((time.ctime()))
            if trackfound[ncr - 1] < 1:

                # Find all clouds (both forward and backward) associated with this reference cloud
                nreferenceclouds = 0
                ntemp_referenceclouds = 1  # Start by forcing to see if track exists
                temp_referenceclouds = [ncr]

                trackpresent = 0
                # logger.debug('Finding all associated clouds')
                # logger.debug((time.ctime()))
                while ntemp_referenceclouds > nreferenceclouds:
                    associated_referenceclouds = np.copy(temp_referenceclouds).astype(
                        int
                    )
                    nreferenceclouds = ntemp_referenceclouds

                    for nr in range(0, nreferenceclouds):
                        # logger.debug(('Processing cloud #: ' + str(nr)))
                        # logger.debug((time.ctime()))
                        tempncr = associated_referenceclouds[nr]

                        # Find indices of forward linked clouds.
                        # Need to subtract one since looping based on core number and
                        # since python starts with indices at zero.
                        # Row of that core is one less than its number.
                        newforwardindex = np.array(
                            np.where(refcloud_forward_index[0, tempncr - 1, :] > 0)
                        )
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
                            # Need to add one since want the core index, which starts at one.
                            # But this is using that row number, which starts at zero.
                            core_newbackward = (newbackwardindex[0, :] + 1)

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

                            # Find reference clouds associated with each new cloud.
                            # Look to see if these new clouds are linked to other cells in the reference file as well.
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

                if nnewclouds > 0:
                    ############################################################
                    # Find the largest reference and new clouds
                    # Largest reference cloud
                    # Need to subtract one since associated_referenceclouds gives core index and matrix starts at zero
                    allreferencepix = npix_reference[associated_referenceclouds - 1]
                    largestreferenceindex = np.argmax(allreferencepix)
                    # Cloud number of the largest reference cloud
                    largest_referencecloud = associated_referenceclouds[largestreferenceindex]

                    # Largest new cloud
                    # Need to subtract one since associated_newclouds gives cloud number and the matrix starts at zero
                    allnewpix = npix_new[associated_newclouds - 1]
                    largestnewindex = np.argmax(allnewpix)
                    # Cloud number of the largest new cloud
                    largest_newcloud = associated_newclouds[largestnewindex]

                    if nnewclouds == 1 and nreferenceclouds == 1:
                        ############################################################
                        # Simple continuation

                        # Check trackstatus already has a valid value.
                        # This will prtrack splits from a previous step being overwritten

                        # logger.debug(trackstatus[ifill,ncr-1])
                        referencetrackstatus[ifill, ncr - 1] = 1
                        trackfound[ncr - 1] = 1
                        tracknumber[0, ifill + 1, associated_newclouds - 1] = np.copy(
                            tracknumber[0, ifill, ncr - 1]
                        )

                    elif nreferenceclouds > 1:
                        ##############################################################
                        # Merging only

                        # Loop through the reference clouds and assign the track to the largest one,
                        # the rest just go away
                        if nnewclouds == 1:
                            for tempreferencecloud in associated_referenceclouds:
                                trackfound[tempreferencecloud - 1] = 1

                                # If this reference cloud is the largest fragment of the merger,
                                # label this reference time (file) as the larger part of merger (2)
                                # and merging at the next time (ifile + 1)
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
                                # If this reference cloud is the smaller fragment of the merger,
                                # label the reference time (ifile) as the small merger (12)
                                # and merging at the next time (file + 1)
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

                            # Loop over the reference clouds and assign the track the largest one
                            for tempreferencecloud in associated_referenceclouds:
                                trackfound[tempreferencecloud - 1] = 1

                                # If this is the larger fragment ofthe merger,
                                # label the reference time (ifill) as large merger (2)
                                # and the actual merging track at the next time [ifill+1]
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
                                # For the smaller fragment of the merger,
                                # label the reference time (ifill) as the small merge and
                                # have the actual merging occur at the next time (ifill+1)
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

                                # For the smaller fragment of the split,
                                # label the new time (ifill+1) as the small split
                                # because the cloud only occurs at the new time step
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
                                # For the larger fragment of the split,
                                # label the new time (ifill+1) as the large split
                                # so that is consistent with the small fragments.
                                # The track continues to follow this cloud so the tracknumber is not incramented.
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
                        # logger.debug('Splitting only')
                        # logger.debug((time.ctime()))
                        # Label reference cloud as a pure split
                        referencetrackstatus[ifill, ncr - 1] = 13
                        tracknumber[0, ifill, ncr - 1] = np.copy(
                            tracknumber[0, ifill, largest_referencecloud - 1]
                        )

                        # Loop over the clouds and assign new tracks to the smaller ones
                        for tempnewcloud in associated_newclouds:
                            # For the smaller fragment of the split,
                            # label the new time (ifill+1) as teh small split (13)
                            # because the cloud only occurs at the new time.
                            if tempnewcloud != largest_newcloud:
                                newtrackstatus[ifill + 1, tempnewcloud - 1] = 31

                                tracknumber[0, ifill + 1, tempnewcloud - 1] = itrack
                                itrack = itrack + 1

                                tracksplitnumber[
                                    0, ifill + 1, tempnewcloud - 1
                                ] = np.copy(tracknumber[0, ifill, ncr - 1])

                                trackreset[0, ifill + 1, tempnewcloud - 1] = 0
                            # For the larger fragment of the split,
                            # label new time (ifill+1) as the large split (3)
                            # so that is consistent with the small fragments
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

                    trackfound[ncr - 1] = 1

                    referencetrackstatus[ifill, ncr - 1] = 0

        ##############################################################################
        # Find any clouds in the new track that don't have a track number.
        # These are new clouds this file

        for ncn in range(1, int(nclouds_new) + 1):
            if tracknumber[0, ifill + 1, ncn - 1] < 0:
                tracknumber[0, ifill + 1, ncn - 1] = itrack
                itrack = itrack + 1

                trackreset[0, ifill + 1, ncn - 1] = 0

        #############################################################################
        # Flag the last file in the dataset
        if ifile == nfiles - 1:
            logger.debug("WE ARE AT THE LAST FILE")
            for ncn in range(1, int(nclouds_new) + 1):
                trackreset[0, ifill + 1, :] = 2
            ifill = ifill + 1
            break

        ##############################################################################
        # Increment to next fill
        ifill = ifill + 1

    trackstatus[0, :, :] = np.nansum(
        np.dstack((referencetrackstatus, newtrackstatus)), 2
    )
    trackstatus[np.isnan(trackstatus)] = -9999

    logger.debug("Tracking Done")

    nfiles = ifill + 1

    # #################################################################
    # # Create histograms of the values in tracknumber.
    # # This effectively counts the number of times each track number appaers in tracknumber,
    # # which is equivalent to calculating the length of the track.
    # tracklengths, trackbins = np.histogram(
    #     np.copy(tracknumber[0, :, :]),
    #     bins=np.arange(1, itrack + 1, 1),
    #     range=(1, itrack + 1),
    # )

    # # #################################################################
    # # Remove all tracks that have only one cloud.
    # logger.debug("Removing short tracks")
    # # logger.debug((time.ctime()))
    #
    # # Identify single cloud tracks
    # singletracks = np.array(np.where(tracklengths <= 1))[0, :]
    # nsingletracks = len(singletracks)
    # # singleindices = np.logical_or(tracknumber[0, :, :] == singletracks)
    #
    # # Loop over single cloudtracks
    # nsingleremove = 0
    # for strack in singletracks:
    #
    #     # Indentify clouds in this track
    #     # Need to add one since singletracks lists the index in the matrix, which starts at zero.
    #     # Track number starts at one.
    #     cloudindex = np.array(
    #         np.where(tracknumber[0, :, :] == int(strack + 1))
    #     )
    #
    #     # Only remove single track if it is not small merger or small split.
    #     # This is only done if keepsingletrack == 1. This is the default.
    #     if keepsingletrack == 1:
    #         if (
    #             tracksplitnumber[0, cloudindex[0], cloudindex[1]] < 0
    #             and trackmergenumber[0, cloudindex[0], cloudindex[1]] < 0
    #         ):
    #             tracknumber[0, cloudindex[0], cloudindex[1]] = -2
    #             trackstatus[0, cloudindex[0], cloudindex[1]] = -9999
    #             nsingleremove = nsingleremove + 1
    #             tracklengths[strack] = -9999
    #
    #     # Remove all single tracks. This corresponds to keepsingletrack == 0.
    #     else:
    #         tracknumber[0, cloudindex[0], cloudindex[1]] = -2
    #         trackstatus[0, cloudindex[0], cloudindex[1]] = -9999
    #         nsingleremove = nsingleremove + 1
    #         tracklengths[strack] = -9999

    #######################################################################
    # Save file
    logger.debug("Writing all track statistics file")
    logger.debug((time.ctime()))

    # Check if file already exists. If exists, delete
    if os.path.isfile(tracknumbers_outfile):
        os.remove(tracknumbers_outfile)

    # Define output variables dictionary
    var_dict = {
        "ntracks": (["time"], np.array([itrack])),
        "basetimes": (["nfiles"], basetime[:nfiles]),
        "cloudid_files": (["nfiles", "ncharacters"], cloudidfiles[:nfiles,:]),
        "track_numbers": (["time", "nfiles", "nclouds"], tracknumber[:,:nfiles,:]),
        "track_status": (["time", "nfiles", "nclouds"], trackstatus[:,:nfiles,:].astype(int)),
        "track_mergenumbers": (["time", "nfiles", "nclouds"], trackmergenumber[:,:nfiles,:]),
        "track_splitnumbers": (["time", "nfiles", "nclouds"], tracksplitnumber[:,:nfiles,:]),
        "track_reset": (["time", "nfiles", "nclouds"], trackreset[:,:nfiles,:]),
        }
    coord_dict = {
        "time": (["time"], np.arange(0, 1)),
        "nfiles": (["nfiles"], np.arange(nfiles)),
        "nclouds": (["nclouds"], np.arange(0, maxnclouds)),
        "ncharacters": (["ncharacters"], np.arange(0, strlength)),
    }
    gattr_dict = {
        "Title": "Indicates the track each cloud is linked to. " + \
                 "Flags indicate how the clouds transition(evolve) between files.",
        # "Conventions": "CF-1.6",
        "Insitution": "Pacific Northwest National Laboratory",
        "Contact": "Zhe Feng: zhe.feng@pnnl.gov",
        "Created": time.ctime(time.time()),
        # "source": datasource,
        # "description": datadescription,
        "singletrack_filebase": singletrack_filebase,
        "startdate": startdate,
        "enddate": enddate,
        "timegap": str(timegap) + "-hours",
    }
    # Define Xarray dataset
    ds_out = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict,)

    # Set variable attributes
    ds_out.ntracks.attrs["long_name"] = "number of cloud tracks"
    ds_out.ntracks.attrs["units"] = "unitless"

    ds_out.basetimes.attrs["long_name"] = "epoch time (seconds since 01/01/1970 00:00) of cloudid_files"
    ds_out.basetimes.attrs["standard_name"] = "time"

    ds_out.cloudid_files.attrs["long_name"] = "filename of each cloudid file used during tracking"
    ds_out.cloudid_files.attrs["units"] = "unitless"

    ds_out.track_numbers.attrs["long_name"] = "cloud track number"
    ds_out.track_numbers.attrs["usage"] = "size: 1 by time by number of clouds. " + \
    "Each column represents a cloudid file (time dimension). " + \
    "Each row represents a cloud in that file (ex. row 0=cloud 1, row 1000=cloud 1001) through time. " + \
    "The values indicate the track that cloud is in. This follows the largest cloud in mergers and splits."

    ds_out.track_numbers.attrs["units"] = "unitless"
    ds_out.track_numbers.attrs["valid_min"] = 1
    ds_out.track_numbers.attrs["valid_max"] = itrack - 1

    ds_out.track_status.attrs[
        "long_name"
    ] = "Flag indicating evolution / behavior for each cloud in a track"
    ds_out.track_status.attrs["units"] = "unitless"
    ds_out.track_status.attrs["valid_min"] = 0
    ds_out.track_status.attrs["valid_max"] = 65

    ds_out.track_mergenumbers.attrs[
        "long_name"
    ] = "Number of the track that this small cloud merges into"
    ds_out.track_mergenumbers.attrs[
        "usage"
    ] = "size: 1 by time by number of clouds. Each column represents a cloudid file (time dimension). " + \
        "Each row represets a cloud in that file through time. " + \
        "Values give the track number associated with the small clouds in mergers."

    ds_out.track_mergenumbers.attrs["units"] = "unitless"
    ds_out.track_mergenumbers.attrs["valid_min"] = 1
    ds_out.track_mergenumbers.attrs["valid_max"] = itrack - 1

    ds_out.track_splitnumbers.attrs[
        "long_name"
    ] = "Number of the track that this small cloud splits from"
    ds_out.track_splitnumbers.attrs[
        "usage"
    ] = "size: 1 by time by number of clouds. Each column represents a cloudid file (time). " + \
        "Each row represets a cloud in that file through time. " + \
        "Values give the track number associated with the small clouds in the split"
    ds_out.track_splitnumbers.attrs["units"] = "unitless"
    ds_out.track_splitnumbers.attrs["valid_min"] = 1
    ds_out.track_splitnumbers.attrs["valid_max"] = itrack - 1

    ds_out.track_reset.attrs[
        "long_name"
    ] = "flag of track starts and abrupt track stops"
    ds_out.track_reset.attrs[
        "usage"
    ] = "Each row represents a cloudid file. Each column represents a cloud in that file. " + \
        "Numbers indicate if the track started or adruptly ended during this file."
    ds_out.track_reset.attrs[
        "values"
    ] = "0=Track starts and ends within a period of continuous data. " + \
        "1=Track starts as the first file in the data set or after a data gap. " + \
        "2=Track ends because data ends or gap in data."
    ds_out.track_reset.attrs["units"] = "unitless"
    ds_out.track_reset.attrs["valid_min"] = 0
    ds_out.track_reset.attrs["valid_max"] = 2

    # Write netcdf file
    ds_out.to_netcdf(
        path=tracknumbers_outfile,
        mode="w",
        format="NETCDF4_CLASSIC",
        # unlimited_dims="ntracks",
        encoding={
            "ntracks": {"dtype": "int", "zlib": True},
            "basetimes": {
                "dtype": "int64",
                "zlib": True,
                "units": "seconds since 1970-01-01",
            },
            "cloudid_files": {
                "zlib": True,
            },
            "track_numbers": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "track_status": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "track_mergenumbers": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "track_splitnumbers": {"dtype": "int", "zlib": True, "_FillValue": -9999},
            "track_reset": {"dtype": "int", "zlib": True, "_FillValue": -9999},
        },
    )
    logger.info(tracknumbers_outfile)
    logger.info('Get track numbers done.')
    return tracknumbers_outfile
