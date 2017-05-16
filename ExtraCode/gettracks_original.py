import numpy as np
import os, fnmatch
import datetime, calendar
from pytz import timezone, utc
from netCDF4 import Dataset
import sys

# Define function to track clouds that were identified in merged ir data
def gettracknumbers_mergedir(datasource, datadescription, dataoutpath, startdate, enddate, timegap, cloudid_filebase, npxname, tracknumbers_version, singletrack_filebase, keepsingletrack=0, removestartendtracks=0, tdimname='time', xdimname='lon', ydimname='lat'):
    # Purpose: Track clouds successively from teh singel cloud files produced in trackclouds_mergedir.py.

    # Author: IDL version written by Sally A. McFarlane (sally.mcfarlane@pnnl.gov) and revised by Zhe Feng (zhe.feng@pnnl.gov). Python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov)

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
    # trackresults - -9999=default/no cloud in current file; 0=track stops, cloud not found in next file; 1=cloud track continues to next file; 2=cloud continues to the next file and it is the larger fragment of a merger; 3=cloud is the larger fragment of a split from the previous file; 10=start of a new track; 11=track stops, cloud is the smaller fragment of a merger, 12=new track, cloud is the smaller fragment of a split from the previous file


    ################################################################################
    # Set constants
    # Number of maximum clouds in a frame
    nmaxclouds = 3000

    # Set track numbers output file name
    tracknumbers_filebase = 'tracknumbers' + tracknumbers_version
    tracknumbers_outfile = dataoutpath + tracknumbers_filebase + '_' + startdate + '_' + enddate + '.nc'

    ##################################################################################
    # Get single track files
    singletrackfiles = fnmatch.filter(os.listdir(dataoutpath), singletrack_filebase +'*')

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
    for filestep, ifiles in enumerate(range(0,len(acceptdates))):
        files[filestep] = singletrackfiles[ifiles]
        filedate[filestep] = str(year[ifiles]) + str(month[ifiles]).zfill(2) + str(day[ifiles]).zfill(2)
        filetime[filestep] = str(hour[ifiles]).zfill(2) + str(minute[ifiles]).zfill(2)

    ############################################################################
    # Initialize matrices
    fillvalue = -9999

    nfiles = int(len(files))
    tracknumber = np.ones((nfiles,nmaxclouds), dtype=int)*fillvalue
    trackresult = np.ones((nfiles,nmaxclouds), dtype=int)*fillvalue
    trackmergesplitnumber = np.ones((nfiles,nmaxclouds), dtype=int)*fillvalue
    basetimes = np.ones(nfiles, dtype=int)*fillvalue

    cloudidfiles = [None]*nfiles

    ############################################################################
    # Load first file
    singletracking_data = Dataset(dataoutpath + files[0], 'r')                                        # Open file
    nclouds_ref = singletracking_data.variables['nclouds_ref'][:]                                     # Number of clouds in reference file
    nclouds_new = singletracking_data.variables['nclouds_new'][:]                                     # Number of clouds in new file
    ref_file = str(singletracking_data.ref_file)                                                      # File name of ref file
    new_file = str(singletracking_data.new_file)                                                      # File name of new file
    ref_date = str(singletracking_data.ref_date)                                                      # Date of ref file
    new_date = str(singletracking_data.new_date)                                                      # Date of new file
    singletracking_data.close()                                                                       # close file

    # Make sure number of clouds does not exceed maximum. If does indicates over-segmenting data
    if nclouds_ref > nmaxclouds:
        sys.exit('# of clouds in reference file exceed allowed maximum number of clouds')

    # Isolate file name and add it to the filelist
    temp_ref_file = os.path.basename(ref_file)
    cloudidfiles[0] = os.path.basename(ref_file)

    # Initate track numbers
    tracknumber[0,1:nclouds_ref+1] = np.arange(0,nclouds_ref)+1
    itrack = nclouds_ref + 1
    print(itrack)

    ###########################################################################
    # Loop over files and generate tracks
    for ifile in range(0,nfiles-1):
        print(files[ifile])

        ######################################################################
        # Load single track file
        singletracking_data = Dataset(dataoutpath + files[ifile], 'r')                                    # Open file
        basetime_ref = singletracking_data.variables['basetime_ref'][:]                                   # Base time of reference file
        basetime_new = singletracking_data.variables['basetime_new'][:]                                   # Base time of new file
        nclouds_ref = singletracking_data.variables['nclouds_ref'][:]                                     # Number of clouds in reference file
        nclouds_new = singletracking_data.variables['nclouds_new'][:]                                     # Number of clouds in new file
        refcloud_forward_index = singletracking_data.variables['refcloud_forward_index'][:]               # Each row represents a cloud in the reference file and the numbers in that row are indices of clouds in new file linked that cloud in the reference file
        newcloud_backward_index = singletracking_data.variables['newcloud_backward_index'][:]             # Each row represents a cloud in the new file and the numbers in that row are indices of clouds in the reference file linked that cloud in the new file
        ref_file = str(singletracking_data.ref_file)                                                      # cloudid filename of ref file
        new_file = str(singletracking_data.new_file)                                                      # cloudid filename of new file
        ref_date = str(singletracking_data.ref_date)                                                      # Date of ref file
        new_date = str(singletracking_data.new_date)                                                      # Date of new file
        singletracking_data.close()

        # Make sure number of clouds does not exceed maximum. If does indicates over-segmenting data
        if nclouds_ref > nmaxclouds:
            sys.exit('# of clouds in reference file exceed allowed maximum number of clouds')

        ########################################################################
        # Load cloudid files
        # Reference cloudid file
        referencecloudid_data = Dataset(ref_file, 'r')
        npix_reference = referencecloudid_data.variables[npxname][:]
        referencecloudid_data.close()

        # New cloudid file
        newcloudid_data = Dataset(new_file, 'r')
        npix_new = newcloudid_data.variables[npxname][:]
        newcloudid_data.close()
         
        ########################################################################
        # Start looping
        # Set previous and new times
        if ifile < 1:
            time_prev = np.copy(basetime_new)

        time_new = np.copy(basetime_new)

        # Check if files immediately follow each other. Missing files can exist. If missing files exist need to incrament index and track numbers
        if ifile > 0:
            hour_diff = (time_new - time_prev)/float(3600)
            if hour_diff > timegap:
                print('Track terminates on: ' + ref_date)
                print('Time difference: ' + str(hour_diff))
                print('Maximum timegap allowed: ' + str(timegap))
                print('New track starts on: ' + new_date)

                # Fill tracking matrices with reference data and record that the track ended
                cloudidfiles[ifile] = temp_ref_file
                basetimes[ifile] = no.copy(basetime_new)

                # Treat all clouds in the reference file as new clodus
                for nreference in range(1,nclouds_ref+1):
                    if tracknumber[ifile,nreference] < 0:
                        tracknumber[ifile,nreference] = itrack
                        trackresult[ifile,nreference] = 10
                        itrack = itrack + 1

        time_prev = time_new
        cloudidfiles[ifile + 1] = new_file
        basetimes[ifile + 1] = basetime_new

        ########################################################################################
        # Compare two tracking files and get all the new and reference clouds 

        # Intiailize matrix for this time period
        trackfound = np.empty(nclouds_ref+1, dtype=int)*fillvalue

        # Loop over all reference clouds
        for ncr in np.arange(1,nclouds_ref+1): # Looping over each reference cloud. Start at 1 since clouds numbered starting at 1. 
            if trackfound[ncr] < 1:

                # Find all clouds (both forward and backward) associated with this reference cloud
                nreferenceclouds = 0
                ntemp_referenceclouds = 1 # Start by forcing to see if track exists
                temp_referenceclouds = [ncr]

                print(ncr)
                print(ntemp_referenceclouds)
                print(nreferenceclouds)
                print(refcloud_forward_index)
                print(newcloud_backward_index)
                raw_input('Waiting for User')

                trackpresent = 0
                while ntemp_referenceclouds > nreferenceclouds:
                    associated_referenceclouds = np.copy(temp_referenceclouds)
                    nreferenceclouds = ntemp_referenceclouds

                    print(temp_referenceclouds)
                    print(associated_referenceclouds)
                    print(nreferenceclouds)
                    raw_input('Waiting for User')

                    for nr in range(0,nreferenceclouds):
                        tempncr = associated_referenceclouds[nr]
                        print(tempncr)

                        # Find indices of forward linked clouds.
                        newforwardindex = np.array(np.where(refcloud_forward_index[0,tempncr-1,:] > 0 )) # Need to subtract one since looping based on core number and since python starts with indices at zero. Row of that core is one less than its number. 
                        nnewforward = np.shape(newforwardindex)[1]
                        print(refcloud_forward_index[0,tempncr-1,:])
                        print(newforwardindex)
                        print(nnewforward) 
                        if nnewforward > 0 :
                            core_newforward = refcloud_forward_index[0,tempncr-1,newforwardindex[0,:]]
                            #core_newforward = np.squeeze(core_newforward)
                            print(core_newforward)
                        raw_input('Waiting for User')

                        # Find indices of backwards linked clouds
                        newbackwardindex = np.array(np.where(newcloud_backward_index[0,:,:] == tempncr))
                        nnewbackward = np.shape(newbackwardindex)[1]
                        print(newcloud_backward_index[0,:,:])
                        print(tempncr)
                        print(newbackwardindex)
                        print(nnewbackward)
                        if nnewbackward > 0:
                            core_newbackward = newbackwardindex[0,:]+1 # Need to add one since want the core index, which starts at one. But this is using that row number, which starts at zero.
                            #core_newbackward = np.squeeze(core_newbackward)
                            print(core_newbackward)
                        raw_input('Waiting for User')

                        # Put all the indices associated with new clouds linked to the reference cloud in one vector
                        if nnewforward > 0:
                            if trackpresent == 0:
                                associated_newclouds = core_newforward[:]
                                trackpresent = trackpresent + 1
                            else:
                                associated_newclouds = np.append(associated_newclouds, core_newforward)

                            #if nnewbackward > 0:
                            #    if trackpresent == 0:
                            #        associated_newclouds = core_newforward
                            #        trackpresent = trackpresent + 1
                            #else:
                            #    associated_newclouds = np.append(associated_newclouds, core_newbackward)
                        #elif nnewbackward > 0:
                        if nnewbackward > 0:
                            if trackpresent == 0:
                                associated_newclouds = core_newbackward[:]
                                trackpresent = trackpresent + 1
                            else:
                                associated_newclouds = np.append(associated_newclouds, core_newbackward)
                        if nnewbackward == 0 and nnewforward == 0:
                            associated_newclouds = []

                        print(associated_newclouds)
                        print(trackpresent)
                        raw_input('Waiting for User')

                        # If the reference cloud is linked to a new cloud
                        if trackpresent > 0:
                            # Sort and find the unique new clouds associated with the reference cloud
                            if len(associated_newclouds) > 1:
                                associated_newclouds = np.unique(np.sort(associated_newclouds))
                            nnewclouds = len(associated_newclouds)
                            print(associated_newclouds)
                            print(nnewclouds)
                            raw_input('Waiting for User')

                            # Find reference clouds associated with each new cloud. Look to see if these new clouds are linked to other cells in the reference file as well. 
                            for nnew in range(0,nnewclouds):
                                # Find associated reference clouds
                                referencecloudindex = np.array(np.where(refcloud_forward_index[0,:,:] == associated_newclouds[nnew])) 
                                nassociatedreference = np.shape(referencecloudindex)[1]
                                print(refcloud_forward_index[0,:,:])
                                print(associated_newclouds[nnew])
                                print(referencecloudindex)
                                print(nassociatedreference)

                                print(ncr)
                                print(temp_referenceclouds)
                                if nassociatedreference > 0:
                                    temp_referenceclouds = np.append(temp_referenceclouds,referencecloudindex[0]+1)
                                    print(temp_referenceclouds)
                                    temp_referenceclouds = np.unique(np.sort(temp_referenceclouds))
                                    print(temp_referenceclouds)
                                raw_input('Waiting for User')

                            print(temp_referenceclouds)
                            ntemp_referenceclouds = len(temp_referenceclouds)
                            print(ntemp_referenceclouds)
                            raw_input('Waiting for User')
                        else:
                            nnewclouds = 0

                        print(nnewclouds)
                        print(associated_newclouds)
                        print(associated_referenceclouds)
                        raw_input('Waiting for User')

                #################################################################
                # Now get the track status
                if nnewclouds > 0:

                    if nnnewclouds == 1 and nreferenceclouds == 1:
                        ############################################################
                        # Simple continuation

                        # Check trackresult already has a valid value. This will prevent splits from a previous step being overwritten
                        if trackresult[ifile,ncr] == fillvalue:
                            trackresult[ifile,ncr] = 1

                        trackfound[ncr] = 1
                        tracknumber[ifile+1,associated_newclouds[0]] = tracknumber[ifile,ncr]

                    elif nreferenceclouds > 1:
                        ##############################################################
                        # Merging only

                        # Find the largest reference cloud
                        allreferencepix = npix_reference[associated_referenceclouds-1] # Need to subtract one since associated_referenceclouds gives core index and matrix starts at zero
                        largereferenceindex = np.argmax(allreferencepix)[0]
                        large_ncr = associated_referenceclouds[largereferenceindex]

                        # Loop through the reference clouds and assign th track to the largest one, the rest just go away
                        if nenewclouds == 1:
                            for nr in range(0,nreferenceclouds):
                                tempncr = associated_referenceclouds[nr]
                                trackfound[tempncr] = 1

                                # If this reference cloud is the larger fragment of the merger, label this reference time (file) as the large merger (2) and merging at the next time (ifile + 1)
                                if nr == large_ncr:
                                    trackresult[ifle,tempncr] = 2
                                    trackresult[ifle+1,tempncr] = tracknumber[ifile,large_ncr]
                                # If this reference cloud is the smaller fragment of the merger, label the reference time (ifile) as the smal merger (12) and merging at the next time (file + 1)
                                else:
                                    trackresult[ifile,tempncr] = 12
                                    trackmergesplitnumber[ifile,tempncr] = tracknumber[ifile,large_ncr]

    raw_input('Waiting for User')


