# Define function to track clouds that were identified in merged ir data
def gettracknumbers_mergedir(datasource, datadescription, datainpath, dataoutpath, startdate, enddate, timegap, nmaxclouds, cloudid_filebase, npxname, tracknumbers_version, singletrack_filebase, keepsingletrack=0, removestartendtracks=0, tdimname='time', xdimname='lon', ydimname='lat'):
    # Purpose: Track clouds successively from teh singel cloud files produced in trackclouds_mergedir.py.

    # Author: IDL version written by Sally A. McFarlane (sally.mcfarlane@pnnl.gov) and re, linewidth=2vised by Zhe Feng (, zlib=True, complevel=5, fill_value=fillvaluezhe.feng@pnnl.gov). Python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov)

    # Currently must be run on fill dataset one at a time.

    # Inputs:
    # datasource - source of the data
    # datadescription - description of data source, included in all output file names
    # startdate - data to start processing in YYYYMMDD format
    # enddate - data to stop processing in YYYYMMDD format
    # timegap - M High values indicates refmean horizontal distribution of reflectivity more peaked. Its value has been replicated for each profile so this data has profiler time resolution.aximum time gap (missing time) allowed (in hours) between two consecutive files
    # cloudid_filepath - Directory of cloudid files
    # npxname - variable name fo reht number of cloud object pixels
    # track_version - Version of track single cloud files
    # track_filepath - Directory of the track single cloud files
    # track_filebase - File base name of the track single cloud files

    # Optional keywords
    # keepsingletrack - Keep tracks that only have 1 fram but merge/split with others
    # removestartendtracks - Remove tracks that start at the first file or ends at the last file
    # tdimname - name of time dimension for the output netcdf file
    # xdimname - name of the x-dimension for the output netcdf file
    # ydimname - name of the y-dimentions for the output netcdf

    # Output
    # tracknumber - track number of the given cloud
    # trackmergesplitnumber - for any cloud in which there is a smaller fragment of a merger or split, this gives the track number of the larger fragment. This is used so you can connect merged/split clouds.
    # trackstatuss - -9999=default/no cloud in current file; 0=track stops, cloud not found in next file; 1=cloud track continues to next file; 2=cloud continues to the next file and it is the larger fragment of a merger; 3=cloud is the larger fragment of a split from the previous file; 10=start of a new track; 11=track stops, cloud is the smaller fragment of a merger, 12=new track, cloud is the smaller fragment of a split from the previous file


    ################################################################################
    # Import modules
    import numpy as np
    import os, fnmatch
    import time, datetime, calendar
    from pytz import timezone, utc
    from netCDF4 import Dataset, stringtochar, chartostring
    import sys

    #############################################################################
    # Set track numbers output file name
    tracknumbers_filebase = 'tracknumbers' + tracknumbers_version
    tracknumbers_outfile = dataoutpath + tracknumbers_filebase + '_' + startdate + '_' + enddate + '.nc'

    ##################################################################################
    # Get single track files sort
    singletrackfiles = fnmatch.filter(os.listdir(datainpath), singletrack_filebase +'*')

    # Put in temporal order
    singletrackfiles = sorted(singletrackfiles)

    ################################################################################
    # Get date/time from filenames
    nfiles = len(singletrackfiles)
    year = np.empty(nfiles, dtype=int)
    month = np.empty(nfiles, dtype=int)
    day = np.empty(nfiles, dtype=int)
    hour = np.empty(nfiles, dtype=int)
    minute = np.empty(nfiles, dtype=int)
    basetime = np.empty(nfiles, dtype=int)
    filedate = np.empty(nfiles)
    filetime = np.empty(nfiles)

    header = np.array(len(singletrack_filebase)).astype(int)
    for filestep, ifiles in enumerate(singletrackfiles):
        year[filestep] = int(ifiles[header:header+4])
        month[filestep] = int(ifiles[header+4:header+6])
        day[filestep] = int(ifiles[header+6:header+8])
        hour[filestep] = int(ifiles[header+9:header+11])
        minute[filestep] = int(ifiles[header+11:header+13])

        TEMP_fulltime = datetime.datetime(year[filestep], month[filestep], day[filestep], hour[filestep], minute[filestep], 0, tzinfo=utc)
        basetime[filestep] = calendar.timegm(TEMP_fulltime.timetuple())

    #############################################################################
    # Keep only files and date/times within start - end time interval
    # Put start and end dates in base time
    TEMP_starttime = datetime.datetime(int(startdate[0:4]), int(startdate[4:6]), int(startdate[6:8]), 0, 0, 0, tzinfo=utc)
    start_basetime = calendar.timegm(TEMP_starttime.timetuple())

    TEMP_endtime = datetime.datetime(int(enddate[0:4]), int(enddate[4:6]), int(enddate[6:8]), 23, 0, 0, tzinfo=utc)
    end_basetime = calendar.timegm(TEMP_endtime.timetuple())

    # Identify files within the start-end date interval
    acceptdates = np.array(np.where((basetime >= start_basetime) & (basetime <= end_basetime)))[0,:]

    # Isolate files and times with start-end date interval
    basetime = basetime[acceptdates]

    files = [None]*len(acceptdates)
    filedate = [None]*len(acceptdates)
    filetime = [None]*len(acceptdates)
    filesyear = np.zeros(len(acceptdates), dtype=int)
    filesmonth = np.zeros(len(acceptdates), dtype=int) 
    filesday = np.zeros(len(acceptdates), dtype=int)
    fileshour = np.zeros(len(acceptdates), dtype=int)
    filesminute = np.zeros(len(acceptdates), dtype=int)

    for filestep, ifiles in enumerate(acceptdates):
        files[filestep] = singletrackfiles[ifiles]
        filedate[filestep] = str(year[ifiles]) + str(month[ifiles]).zfill(2) + str(day[ifiles]).zfill(2)
        filetime[filestep] = str(hour[ifiles]).zfill(2) + str(minute[ifiles]).zfill(2)
        filesyear[filestep] = int(year[ifiles])
        filesmonth[filestep] = int(month[ifiles])
        filesday[filestep] = int(day[ifiles])
        fileshour[filestep] = int(hour[ifiles])
        filesminute[filestep] = int(minute[ifiles])

    #########################################################################
    # Determine number of gaps in dataset
    gap = 0
    for ifiles in range(1,len(acceptdates)):
        newtime = datetime.datetime(filesyear[ifiles], filesmonth[ifiles], filesday[ifiles], fileshour[ifiles], filesminute[ifiles], 0, tzinfo=utc)
        referencetime = datetime.datetime(filesyear[ifiles-1], filesmonth[ifiles-1], filesday[ifiles-1], fileshour[ifiles-1], filesminute[ifiles-1], 0, tzinfo=utc)

        cutofftime = newtime - datetime.timedelta(minutes=timegap*60)

        if cutofftime > referencetime:
            gap = gap + 1

    ############################################################################
    # Initialize matrices
    fillvalue = -9999

    nfiles = int(len(files))+2*gap
    tracknumber = np.ones((nfiles,nmaxclouds), dtype=int)*fillvalue
    trackstatus = np.ones((nfiles,nmaxclouds), dtype=int)*fillvalue
    trackmergenumber = np.ones((nfiles,nmaxclouds), dtype=int)*fillvalue
    tracksplitnumber = np.ones((nfiles,nmaxclouds), dtype=int)*fillvalue
    basetime = np.ones(nfiles, dtype=int)*fillvalue
    trackreset = np.ones((nfiles,nmaxclouds), dtype=int)*fillvalue

    ############################################################################
    # Load first file
    singletracking_data = Dataset(datainpath + files[0], 'r')                                        # Open file
    nclouds_reference = singletracking_data.variables['nclouds_ref'][:]                               # Number of clouds in reference file
    nclouds_new = singletracking_data.variables['nclouds_new'][:]                                     # Number of clouds in new file
    basetime_reference = singletracking_data.variables['basetime_ref'][:]                             # Base time of reference file
    basetime_new = singletracking_data.variables['basetime_new'][:]                                   # Base time of new file
    reference_file = str(singletracking_data.ref_file)                                                # File name of ref file
    new_file = str(singletracking_data.new_file)                                                      # File name of new file
    reference_date = str(singletracking_data.ref_date)                                                # Date of ref file
    new_date = str(singletracking_data.new_date)                                                      # Date of new file
    singletracking_data.close()                                                                       # close file

    # Make sure number of clouds does not exceed maximum. If does indicates over-segmenting data
    if nclouds_reference > nmaxclouds:
        sys.exit('# of clouds in reference file exceed allowed maximum number of clouds')

    # Isolate file name and add it to the filelist
    basetime[0] = np.copy(basetime_reference)

    temp_referencefile = os.path.basename(reference_file)
    strlength = len(temp_referencefile)
    cloudidfiles = np.empty((nfiles,strlength), dtype=str)
    cloudidfiles[0,:] = stringtochar(np.array(os.path.basename(reference_file)))

    # Initate track numbers
    tracknumber[0,0:int(nclouds_reference)] = np.arange(0,int(nclouds_reference))+1
    itrack = nclouds_reference + 1

    # Rocord that the tracks are being reset / initialized
    trackreset[0,:] = 1

    ###########################################################################
    # Loop over files and generate tracks
    ifill = 0
    for ifile in range(0,nfiles-1):
        print(files[ifile])

        ######################################################################
        # Load single track file
        singletracking_data = Dataset(datainpath + files[ifile], 'r')                                    # Open file
        basetime_reference = singletracking_data.variables['basetime_ref'][:]                             # Base time of reference file
        basetime_new = singletracking_data.variables['basetime_new'][:]                                   # Base time of new file
        nclouds_reference = singletracking_data.variables['nclouds_ref'][:]                               # Number of clouds in reference file
        nclouds_new = singletracking_data.variables['nclouds_new'][:]                                     # Number of clouds in new file
        refcloud_forward_index = singletracking_data.variables['refcloud_forward_index'][:]               # Each row represents a cloud in the reference file and the numbers in that row are indices of clouds in new file linked that cloud in the reference file
        newcloud_backward_index = singletracking_data.variables['newcloud_backward_index'][:]             # Each row represents a cloud in the new file and the numbers in that row are indices of clouds in the reference file linked that cloud in the new file
        reference_file = str(singletracking_data.ref_file)                                                # cloudid filename of ref file
        new_file = str(singletracking_data.new_file)                                                      # cloudid filename of new file
        reference_date = str(singletracking_data.ref_date)                                                # Date of ref file
        new_date = str(singletracking_data.new_date)                                                      # Date of new file
        singletracking_data.close()

        # Make sure number of clouds does not exceed maximum. If does indicates over-segmenting data
        if nclouds_reference > nmaxclouds:
            sys.exit('# of clouds in reference file exceed allowed maximum number of clouds')

        ########################################################################
        # Load cloudid files
        # Reference cloudid file
        referencecloudid_data = Dataset(reference_file, 'r')
        npix_reference = referencecloudid_data.variables[npxname][:]
        referencecloudid_data.close()

        # New cloudid file
        newcloudid_data = Dataset(new_file, 'r')
        npix_new = newcloudid_data.variables[npxname][:]
        newcloudid_data.close()
         
        ########################################################################
        # Check time gap between consecutive track files

        # Set previous and new times
        if ifile < 1:
            time_prev = np.copy(basetime_new)

        time_new = np.copy(basetime_new)

        # Check if files immediately follow each other. Missing files can exist. If missing files exist need to incrament index and track numbers
        if ifile > 0:
            hour_diff = (time_new - time_prev)/float(3600)
            if hour_diff > timegap:
                print('Track terminates on: ' + reference_date)
                print('Time difference: ' + str(hour_diff))
                print('Maximum timegap allowed: ' + str(timegap))
                print('New track starts on: ' + new_date)

                # Flag the previous file as the last file
                trackreset[ifill,:] = 2

                ifill = ifill + 2

                # Fill tracking matrices with reference data and record that the track ended
                cloudidfiles[ifill,:] = stringtochar(np.array(os.path.basename(temp_referencefile)))
                basetime[ifill] = np.copy(basetime_reference)

                # Record that break in data occurs
                trackreset[ifill,:] = 1

                # Treat all clouds in the reference file as new clouds
                for ncr in range(1,nclouds_reference+1):
                    tracknumber[ifill,ncr-1] = itrack
                    itrack = itrack + 1

        time_prev = time_new
        cloudidfiles[ifill + 1,:] = stringtochar(np.array(os.path.basename(new_file)))
        basetime[ifill + 1] = basetime_new

        ########################################################################################
        # Compare forward and backward single track matirces to link new and reference clouds
        # Intiailize matrix for this time period
        trackfound = np.ones(nclouds_reference+1, dtype=int)*fillvalue

        # Loop over all reference clouds
        for ncr in np.arange(1,nclouds_reference+1): # Looping over each reference cloud. Start at 1 since clouds numbered starting at 1. 
            if trackfound[ncr-1] < 1:

                # Find all clouds (both forward and backward) associated with this reference cloud
                nreferenceclouds = 0
                ntemp_referenceclouds = 1 # Start by forcing to see if track exists
                temp_referenceclouds = [ncr]

                trackpresent = 0
                while ntemp_referenceclouds > nreferenceclouds:
                    associated_referenceclouds = np.copy(temp_referenceclouds)
                    nreferenceclouds = ntemp_referenceclouds

                    for nr in range(0,nreferenceclouds):
                        tempncr = associated_referenceclouds[nr]

                        # Find indices of forward linked clouds.
                        newforwardindex = np.array(np.where(refcloud_forward_index[0,tempncr-1,:] > 0 )) # Need to subtract one since looping based on core number and since python starts with indices at zero. Row of that core is one less than its number. 
                        nnewforward = np.shape(newforwardindex)[1]
                        if nnewforward > 0 :
                            core_newforward = refcloud_forward_index[0,tempncr-1,newforwardindex[0,:]]

                        # Find indices of backwards linked clouds
                        newbackwardindex = np.array(np.where(newcloud_backward_index[0,:,:] == tempncr))
                        nnewbackward = np.shape(newbackwardindex)[1]
                        if nnewbackward > 0:
                            core_newbackward = newbackwardindex[0,:]+1 # Need to add one since want the core index, which starts at one. But this is using that row number, which starts at zero.

                        # Put all the indices associated with new clouds linked to the reference cloud in one vector
                        if nnewforward > 0:
                            if trackpresent == 0:
                                associated_newclouds = core_newforward[:]
                                trackpresent = trackpresent + 1
                            else:
                                associated_newclouds = np.append(associated_newclouds, core_newforward)

                        if nnewbackward > 0:
                            if trackpresent == 0:
                                associated_newclouds = core_newbackward[:]
                                trackpresent = trackpresent + 1
                            else:
                                associated_newclouds = np.append(associated_newclouds, core_newbackward)
                        if nnewbackward == 0 and nnewforward == 0:
                            associated_newclouds = []

                        # If the reference cloud is linked to a new cloud
                        if trackpresent > 0:
                            # Sort and find the unique new clouds associated with the reference cloud
                            if len(associated_newclouds) > 1:
                                associated_newclouds = np.unique(np.sort(associated_newclouds))
                            nnewclouds = len(associated_newclouds)

                            # Find reference clouds associated with each new cloud. Look to see if these new clouds are linked to other cells in the reference file as well. 
                            for nnew in range(0,nnewclouds):
                                # Find associated reference clouds
                                referencecloudindex = np.array(np.where(refcloud_forward_index[0,:,:] == associated_newclouds[nnew])) 
                                nassociatedreference = np.shape(referencecloudindex)[1]

                                if nassociatedreference > 0:
                                    temp_referenceclouds = np.append(temp_referenceclouds,referencecloudindex[0]+1)
                                    temp_referenceclouds = np.unique(np.sort(temp_referenceclouds))

                            ntemp_referenceclouds = len(temp_referenceclouds)
                        else:
                            nnewclouds = 0

                #################################################################
                # Now get the track status
                if nnewclouds > 0:

                    ############################################################
                    # Find the largest reference and new clouds

                    # Largest reference cloud
                    allreferencepix = npix_reference[0,associated_referenceclouds-1] # Need to subtract one since associated_referenceclouds gives core index and matrix starts at zero
                    largestreferenceindex = np.argmax(allreferencepix)
                    largest_referencecloud = associated_referenceclouds[largestreferenceindex] # Cloud number of the largest reference cloud

                    # Largest new cloud
                    allnewpix = npix_new[0,associated_newclouds-1] # Need to subtract one since associated_newclouds gives cloud number and the matrix starts at zero
                    largestnewindex = np.argmax(allnewpix)
                    largest_newcloud = associated_newclouds[largestnewindex] # Cloud numberof the largest new cloud

                    #print(associated_referenceclouds)
                    #print(associated_newclouds)

                    if nnewclouds == 1 and nreferenceclouds == 1:
                        ############################################################
                        # Simple continuation

                        # Check trackstatus already has a valid value. This will prtrack splits from a previous step being overwritten

                        #print(trackstatus[ifill,ncr-1])
                        if trackstatus[ifill,ncr-1] == fillvalue:
                            trackstatus[ifill,ncr-1] = 1
                        else:
                            trackstatus[ifill,ncr-1]= trackstatus[ifill,ncr-1] + 1

                        trackfound[ncr-1] = 1
                        tracknumber[ifill+1,associated_newclouds-1] = tracknumber[ifill,ncr-1]

                        #print('Continuation')
                        #print(ncr-1)
                        #print(associated_newclouds-1)
                        #print(trackstatus[ifill,ncr-1])
                        #raw_input('Waiting for User')

                    elif nreferenceclouds > 1:
                        ##############################################################
                        # Merging only

                        # Loop through the reference clouds and assign th track to the largestst one, the rest just go away
                        if nnewclouds == 1:
                            for tempreferencecloud in associated_referenceclouds:
                                trackfound[tempreferencecloud-1] = 1

                                #print(trackstatus[ifill,tempreferencecloud-1])

                                # If this reference cloud is the largest fragment of the merger, label this reference time (file) as the larger part of merger (2) and merging at the next time (ifile + 1)
                                if tempreferencecloud == largest_referencecloud:
                                    if trackstatus[ifill,tempreferencecloud-1] == fillvalue:
                                        trackstatus[ifill,tempreferencecloud-1] = 2
                                    else:
                                        trackstatus[ifill,tempreferencecloud-1] = trackstatus[ifill,tempreferencecloud-1] + 2
                                    tracknumber[ifill+1,associated_newclouds-1] = tracknumber[ifill,largest_referencecloud-1]
                                # If this reference cloud is the smaller fragment of the merger, label the reference time (ifile) as the small merger (12) and merging at the next time (file + 1)
                                else:
                                    if trackstatus[ifill,tempreferencecloud-1] == fillvalue:
                                        trackstatus[ifill,tempreferencecloud-1] = 21
                                    else:
                                        trackstatus[ifill,tempreferencecloud-1] = trackstatus[ifill,tempreferencecloud-1] + 21
                                    trackmergenumber[ifill,tempreferencecloud-1] = tracknumber[ifill,largest_referencecloud-1]

                                #print('Merge Only')
                                #print(largest_referencecloud-1)
                                #print(tempreferencecloud-1)
                                #print(associated_newclouds-1)
                                #print(trackstatus[ifill,tempreferencecloud-1])
                                #print(trackmergenumber[ifill,tempreferencecloud-1])
                                #raw_input('Waiting for User')

                        #################################################################
                        # Merging and spliting
                        else:
                            #print(trackstatus[ifill,tempreferencecloud-1])
                            #print('Merger and Split')
                           
                            # Loop over the reference clouds and assign the track the largest one
                            for tempreferencecloud in associated_referenceclouds:
                                trackfound[tempreferencecloud-1] = 1

                                #print(trackstatus[ifill,tempreferencecloud-1])

                                # If this is the larger fragment ofthe merger, label the reference time (ifill) as large merger (2) and the actual merging track at the next time [ifill+1]
                                if tempreferencecloud == largest_referencecloud:
                                    if trackstatus[ifill,tempreferencecloud-1] == fillvalue:
                                        trackstatus[ifill,tempreferencecloud-1] = 2 + 13
                                    else:
                                        trackstatus[ifill,tempreferencecloud-1] = trackstatus[ifill,tempreferencecloud-1] + 2 + 13
                                    tracknumber[ifill+1,largest_newcloud-1] = tracknumber[ifill,largest_referencecloud-1]
                                # For the smaller fragment of the merger, label the reference time (ifill) as the small merge and have the actual merging occur at the next time (ifill+1)
                                else:
                                    if trackstatus[ifill,tempreferencecloud-1] == fillvalue:
                                        trackstatus[ifill,tempreferencecloud-1] = 21
                                    else:
                                        trackstatus[ifill,tempreferencecloud-1] = trackstatus[ifill,tempreferencecloud-1] + 21
                                    trackmergenumber[ifill,tempreferencecloud-1] = tracknumber[ifill,largest_referencecloud-1]

                                #print(tempreferencecloud-1)
                                #print(largest_referencecloud-1)
                                #print(trackstatus[ifill,tempreferencecloud-1])
                                #print(trackmergenumber[ifill,tempreferencecloud-1])
                                #raw_input('Waiting for User')

                            # Loop through the new clouds and assign the smaller ones a new track
                            for tempnewcloud in associated_newclouds:

                                # For the smaller fragment of the split, label the new time (ifill+1) as the small split because the cloud only occurs at the new time step
                                if tempnewcloud != largest_newcloud:
                                    trackstatus[ifill+1,tempnewcloud-1] = 31

                                    tracknumber[ifill+1,tempnewcloud-1] = itrack
                                    itrack = itrack + 1

                                    tracksplitnumber[ifill+1,tempnewcloud-1] = tracknumber[ifill,largest_referencecloud-1]

                                    trackreset[ifill+1,tempnewcloud-1] = 0
                                # For the larger fragment of the split, label the new time (ifill+1) as the large split so that is consistent with the small fragments. The track continues to follow this cloud so the tracknumber is not incramented. 
                                else:
                                    trackstatus[ifill+1,tempnewcloud-1] = 3
                                    tracknumber[ifill+1,tempnewcloud-1] = tracknumber[ifill,largest_referencecloud-1]

                                #print(tempnewcloud-1)
                                #print(largest_newcloud-1)
                                #print(trackstatus[ifill+1,tempnewcloud-1])
                                #print(tracksplitnumber[ifill+1,tempnewcloud-1])
                                #raw_input('Waiting for User')

                    #####################################################################
                    # Splitting only

                    elif nnewclouds > 1:
                        # Label reference cloud as a pure split
                        #print(trackstatus[ifill,ncr-1])
                        if trackstatus[ifill,ncr-1] == fillvalue:
                            trackstatus[ifill,ncr-1] = 13
                        else:
                            trackstatus[ifill,ncr-1] = trackstatus[ifill,ncr-1] + 13

                        # Loop over the clouds and assign new tracks to the smaller ones
                        for tempnewcloud in associated_newclouds:
                            # For the smaller fragment of the split, label the new time (ifill+1) as teh small split (13) because the cloud only occurs at the new time. 
                            if tempnewcloud != largest_newcloud:
                                trackstatus[ifill+1,tempnewcloud-1] = 31

                                tracknumber[ifill+1,tempnewcloud-1] = itrack
                                itrack = itrack + 1

                                tracksplitnumber[ifill+1,tempnewcloud-1] = tracknumber[ifill,ncr-1]

                                trackreset[ifill+1,tempnewcloud-1] = 0
                            # For the larger fragment of the split, label new time (ifill+1) as the large split (3) so that is consistent with the small fragments
                            else:
                                trackstatus[ifill+1,tempnewcloud-1] = 3
                                tracknumber[ifill+1,tempnewcloud-1] = tracknumber[ifill,ncr-1]

                            #print('Split only')
                            #print(ncr-1)
                            #print(tempnewcloud-1)
                            #print(largest_newcloud-1)
                            #print(trackstatus[ifill+1,tempnewcloud-1])
                            #print(trackstatus[ifill,ncr-1])
                            #raw_input('Waiting for User')

                    else:
                        sys.exit(str(ncr) + ' How did we get here?')

                ######################################################################################
                # No new clouds. Track dissipated
                else:
                    trackfound[ncr-1] = 1

                    #print(trackstatus[ifill,ncr-1])

                    if trackstatus[ifill,ncr-1] < 0:
                        trackstatus[ifill,ncr-1] = 0

                    #print('Track ends')
                    #print(ncr-1)
                    #print(trackstatus[ifill, ncr-1])
                    #raw_input('Waiting for User')

        ##############################################################################
        # Find any clouds in the new track that don't have a track number. These are new clouds this file
        for ncn in range(1,nclouds_new+1):
            if tracknumber[ifill+1,ncn-1] < 0:
                tracknumber[ifill+1,ncn-1] = itrack
                itrack = itrack + 1

                trackreset[ifill+1,ncn-1] = 0

        #############################################################################
        # Flag the last file in the dataset
        if ifile == nfiles-2:
            for ncn in range(1,nclouds_new+1):
                trackreset[ifill+1,:] = 2

        ##############################################################################
        # Incrament to next fill
        ifill = ifill + 1

    print('Tracking Done')

    #################################################################
    # Remove all tracks that have only one cloud.
    # Create histograms of the values in tracknumber. This effectively counts the number of times each track number appaers in tracknumber, which is equivalent to calculating the length of the track. 
    tracklengths, trackbins = np.histogram(tracknumber, bins=np.arange(1,itrack+1,1), range=(1,itrack+1))

    # Identify single cloud tracks
    singletracks = np.array(np.where(tracklengths <= 1))[0,:]
    nsingletracks = len(singletracks)

    # Loop over single cloudtracks
    nsingleremove = 0
    for strack in singletracks:
        # Indentify clouds in this track
        cloudindex = np.array(np.where(tracknumber == strack+1)) # Need to add one since singletracks lists the index in the matrix, which starts at zero. Track number starts at one.

        # Only remove single track if it is not small merger or small split. This is only done if keepsingletrack == 1.
        if keepsingletrack == 1:
            if trackstatus[cloudindex[0], cloudindex[1]] not in [21, 24, 52, 31, 32, 18, 44, 46]:
                tracknumber[cloudindex[0], cloudindex[1]] = -2
                trackstatus[cloudindex[0], cloudindex[1]] = fillvalue
                nsingleremove = nsingleremove + 1
                tracklengths[strack] = fillvalue

        # Remove all single tracks. This corresponds to keepsingletrack == 0, which is the default
        else:
            tracknumber[cloudindex[0], cloudindex[1]] = -2
            trackstatus[cloudindex[0], cloudindex[1]] = fillvalue
            nsingleremove = nsingleremove + 1
            tracklengths[strack] = fillvalue

    #######################################################################
    # Save file
    filesave = Dataset(tracknumbers_outfile, 'w', format='NETCDF4_CLASSIC')

    # set global attributes
    filesave.Convenctions = 'CF-1.6'
    filesave.title = 'Indicates the track each cloud is linked to. Flags indicate how the clouds transition(evolve) between files.'
    filesave.setncattr('source', datasource)
    filesave.setncattr('description', datadescription)
    filesave.setncattr('singletrack_filebase', singletrack_filebase)
    filesave.setncattr('startdate', startdate)
    filesave.setncattr('enddate', enddate)
    filesave.setncattr('timegap', str(timegap) + '-hours')
    filesave.setncattr('Contact', 'Hannah C Barnes: hannah.barnes@pnnl.gov')
    filesave.institution = 'Pacific Northwest National Laboratory'
    filesave.history = 'Created ' + time.ctime(time.time())

    # create netcdf dimensions
    filesave.createDimension('time', None)
    filesave.createDimension('nfiles', nfiles)
    filesave.createDimension('nclouds', nmaxclouds)
    filesave.createDimension('ncharacters', strlength)

    # define variables
    ntracks = filesave.createVariable('ntracks', 'i4', 'time', zlib=True)
    ntracks.long_name = 'number of cloud system tracks'
    ntracks.units = 'unitless'

    basetimes = filesave.createVariable('basetimes', 'i4', 'nfiles', zlib=True, complevel=5, fill_value=fillvalue)
    basetimes.long_name = 'base time of cloudid_files'
    basetimes.units = 'seconds since 01/01/1970 00:00'
    basetimes.fill_value = fillvalue
    basetimes.standard_name = 'time'

    cloudid_files = filesave.createVariable('cloudid_files', 'S1', ('nfiles', 'ncharacters'), zlib=True, complevel=5, fill_value=fillvalue)
    cloudid_files.longname = 'filenames of each cloudid file used for tracking'
    cloudid_files.fill_value = fillvalue

    track_numbers = filesave.createVariable('track_numbers', 'i4', ('time', 'nfiles', 'nclouds'), zlib=True, complevel=5, fill_value=fillvalue)
    track_numbers.longname = 'cloud track number'
    track_numbers.description = 'Each row represents a cloudid file. Each row represents a cloud in that file. The number indicates the track associate with that cloud. This follows the largest cloud in mergers and splits.'
    track_numbers.fill_value = fillvalue
    track_numbers.min_value = 1
    track_numbers.max_value = itrack - 1
    track_numbers.units = 'unitless'

    track_status = filesave.createVariable('track_status', 'i4', ('time', 'nfiles', 'nclouds'), zlib=True, complevel=5, fill_value=fillvalue)
    track_status.longname = 'flag indicating transition of cloud'
    track_status.description = 'Each row represents a cloudid file. Each row represents a cloud in that file. The numbers indicate how that cloud transitions over time'
    track_status.values = '-9999=missing cloud or cloud removed due to short track, 0=track ends after this file, 1=cloud continues as one cloud in next file, 2=Bigger cloud that merges into one cloud in the next file, 21=Smaller cloud that merges into one cloud in the next file, 3=Bigger cloud that split from one cloud in the previous timestep, 31=Smaller cloud that split from one cloud in the previous file, 4=Bigger cloud that split from one cloud in the previous step. This cloud continues as one cloud in the next file. 32=smaller cloud that plot from one cloud in the previous file. This cloud continues as one cloud in the next file, 5=Bigger cloud that split from one cloud in the previous file. In the next file this cloud is the big cloud in a merger, 24=Bigger cloud that split from one cloud in the previous file. This cloud is then the smaller cloud in a merger that occurs in the next file, 33=Smaller cloud that split from one cloud in the previous file. In the next file this cloud is the big cloud in a merger, 52=Smaller cloud that split from one cloud in the previous file. This cloud is then the smaller cloud in a merger that occurs in the next file'
    track_status.min_value = 0
    track_status.max_value = 52
    track_status.fill_value = fillvalue
    track_status.units = 'unitless'

    track_mergenumbers = filesave.createVariable('track_mergenumbers', 'i4', ('time', 'nfiles', 'nclouds'), zlib=True, complevel=5, fill_value=fillvalue)
    track_mergenumbers.longname = 'cloud track number'
    track_mergenumbers.description = 'Each row represents a cloudid file. Each column represets a cloud in that file. Numbers give the track number associated with the small clouds in mergers.'
    track_mergenumbers.fill_value = fillvalue
    track_mergenumbers.min_value = 1
    track_mergenumbers.max_value = itrack-1
    track_mergenumbers.units = 'unitless'

    track_splitnumbers = filesave.createVariable('track_splitnumbers', 'i4', ('time', 'nfiles', 'nclouds'), zlib=True, complevel=5, fill_value=fillvalue)
    track_splitnumbers.longname = 'cloud track number'
    track_splitnumbers.description = 'Each row represents a cloudid file. Each column represets a cloud in that file. Numbers give the track number associated with the small clouds in splits.'
    track_splitnumbers.fill_value = fillvalue
    track_splitnumbers.min_value = 1
    track_splitnumbers.max_value = itrack-1
    track_splitnumbers.units = 'unitless'

    track_reset = filesave.createVariable('track_reset', 'i4', ('time', 'nfiles', 'nclouds'), zlib=True, complevel=5, fill_value=fillvalue)
    track_reset.longname = 'flag of track starts and adrupt track stops'
    track_reset.description = 'Each row represents a cloudid file. Each column represents a cloud in that file. Numbers indicate if the track started or adruptly ended during this file.'
    track_reset.values = '0=Track starts and ends within a period of continuous data. 1=Track starts as the first file in the data set or after a data gap. 2=Track ends because data ends or gap in data.'
    track_reset.fill_value = fillvalue
    track_reset.min_value = 10
    track_reset.max_value = 20
    track_reset.units = 'unitless'

    # fill variables
    ntracks[:] = itrack
    basetimes[:] = basetime[:]
    cloudid_files[:,:] = cloudidfiles[:,:]
    track_numbers[0,:,:] = tracknumber[:,:]
    track_status[0,:,:] = trackstatus[:,:]
    track_mergenumbers[0,:,:] = trackmergenumber[:,:]
    track_splitnumbers[0,:,:] = tracksplitnumber[:,:]
    track_reset[0,:,:] = trackreset[:,:]    

    # write and close file
    filesave.close()






