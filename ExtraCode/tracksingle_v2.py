import numpy as np
import glob
import os
import re
import fnmatch
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import time, datetime, calendar
from pytz import timezone, utc
import sys

# Name: trackclouds_singlefile

# Purpose: Track clouds in successive pairs of cloudid files. Output netCDF file for each pair of cloudid files.

# Comment: Futyan and DelGenio (2007) - tracking procedure

# Authors: IDL version written by Sally A. McFarlane (sally.mcfarlane@pnnl.gov) and revised by Zhe Feng (zhe.feng@pnnl.gov). Python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov)

# Inputs:
# datasource - source of the data
# datapath - path to data
# datadescription - description of data source, included in all output file names
# timegap - M High values indicates refmean horizontal distribution of reflectivity more peaked. Its value has been replicated for each profile so this data has profiler time resolution.aximum time gap (missing time) allowed (in hours) between two consecutive files
# startdate - data to start processing in YYYYMMDD format
# enddate - data to stop processing in YYYYMMDD format
# id_filepath - path to input cloudid files
# id_filebase - basename of cloudid files

def trackclouds_mergedir(dataoutpath, datasource, datadescription, track_version, timegap, othresh, startdate, enddate, cloudid_filepath, cloudid_filebase):

    ##############################################################
    # Set constants
    # Version information
    outfilebase = 'track' + track_version + '_'

    # Number of different clouds a single cloud can link to
    maxlinks = 10

    ###############################################################
    # Loop through all the files in the directory. Only keep file name of files within start and end time span.
    files = []

    leadingfill = len(cloudid_filebase)
    for file in os.listdir(cloudid_filepath):
       if file.endswith(".nc") and file[0:leadingfill] == cloudid_filebase and file[leadingfill+1:leadingfill+9]>=startdate and file[leadingfill+10:-3]<=enddate:    # Isolate cases within time period set by startdate and enddate
           files = np.append(files, file)

    # Put files in sequential order
    files = sorted(files)

    ##############################################################
    # Loop over files, comparing two files at a time
    nfiles = len(files)
    print(nfiles)

    for ifile in np.arange(1,nfiles):
        ########################################################
        # Determine the reference time, new time, and time based on timegap
        newfile = files[ifile]
        newyear = int(newfile[leadingfill+1:leadingfill+5])
        newmonth = int(newfile[leadingfill+5:leadingfill+7])
        newday = int(newfile[leadingfill+7:leadingfill+9])
        newhour = int(newfile[leadingfill+10:-5])
        newminute = int(newfile[leadingfill+12:-3])

        referencefile = files[ifile-1]
        referenceyear = int(referencefile[leadingfill+1:leadingfill+5])
        referencemonth = int(referencefile[leadingfill+5:leadingfill+7])
        referenceday = int(referencefile[leadingfill+7:leadingfill+9])
        referencehour = int(referencefile[leadingfill+10:-5])
        referenceminute = int(referencefile[leadingfill+12:-3])

        newtime = datetime.datetime(newyear, newmonth, newday, newhour, newminute, 0, tzinfo=utc)
        referencetime = datetime.datetime(referenceyear, referencemonth, referenceday, referencehour, referenceminute, 0, tzinfo=utc)

        cutofftime = newtime - datetime.timedelta(minutes=timegap*60)

        ##########################################################
        # Compare cut-off time with time of preceeding file. If reference time larger than cutoff time and smaller than new time preceed. Otherwise move onto next file
        if cutofftime < referencetime and newtime > referencetime:
            print('Linking:')
            print(referencetime)
            print(newtime)

            ##############################################################
            # Load cloudid file from before, called reference file
            reference_data = Dataset(cloudid_filepath + files[ifile-1], 'r')                                        # Open file
            reference_convcold_cloudnumber = reference_data.variables['convcold_cloudnumber'][:]                    # Load cloud id map
            nreference = reference_data.variables['nclouds'][:]                                                     # Load number of clouds / features
            reference_data.close()                                                                                  # close file

            reference_datetime = datetime.datetime(referenceyear, referencemonth, referenceday, referencehour, referenceminute, 0)
            reference_basetime = calendar.timegm(reference_datetime.timetuple())

            reference_filedatetime = str(int(referenceyear)) + str(int(referencemonth)).zfill(2) + str(int(referenceday)).zfill(2) + '_' + str(int(referencehour)).zfill(2) + str(int(referenceminute)).zfill(2)

            ##########################################################
            # Load cloudid file from before, called new file
            new_data = Dataset(cloudid_filepath + files[ifile], 'r')                                # Open file
            new_convcold_cloudnumber = new_data.variables['convcold_cloudnumber'][:]                # Load cloud id map
            nnew = new_data.variables['nclouds'][:]                                                 # Load number of clouds / features
            new_data.close()                                                                        # close file
            
            new_datetime = datetime.datetime(newyear, newmonth, newday, newhour, newminute, 0)
            new_basetime = calendar.timegm(new_datetime.timetuple())

            new_filedatetime = str(int(newyear)) + str(int(newmonth)).zfill(2) + str(int(newday)).zfill(2) + '_' + str(int(newhour)).zfill(2) + str(int(newminute)).zfill(2)

            ############################################################
            # Get size of data
            times, ny, nx = np.shape(new_convcold_cloudnumber)

            #######################################################
            # Initialize matrices
            reference_forward_index = np.ones((nreference, maxlinks), dtype=int)*-9999
            reference_forward_size = np.ones((nreference, maxlinks), dtype=int)*-9999
            new_backward_index = np.ones((nnew, maxlinks), dtype=int)*-9999
            new_backward_size =  np.ones((nnew, maxlinks), dtype=int)*-9999

            ######################################################
            # Loop through each cloud / feature in reference time and look for overlaping clouds / features in the new file
            for refindex in np.arange(1,nreference+1):
                # Locate where the cloud in the reference file overlaps with any cloud in the new file
                forward_matchindices = np.where((reference_convcold_cloudnumber == refindex) & (new_convcold_cloudnumber != 0))

                # Get the convcold_cloudnumber of the clouds in the new file that overlap the cloud in the reference file
                forward_newindex = new_convcold_cloudnumber[forward_matchindices]
                unique_forwardnewindex = np.unique(forward_newindex)

                # Calculate size of reference cloud in terms of number of pixels
                sizeref = len(np.extract(reference_convcold_cloudnumber == refindex, reference_convcold_cloudnumber))

                # Loop through the overlapping clouds in the new file, determining if they statisfy the overlap requirement
                forward_nmatch = 0 # Initialize overlap counter
                for matchindex in unique_forwardnewindex: 
                    sizematch = len(np.extract(forward_newindex == matchindex, forward_newindex))

                    if sizematch/float(sizeref) > othresh:
                        if forward_nmatch > maxlinks:
                            print('reference: ' + number_filepath + files[ifile-1])
                            print('new: ' + number_filepath + files[ifile])
                            sys.exit('More than ' + str(int(maxlinks)) + ' clouds in new file match with reference cloud?!')
                        else:
                            reference_forward_index[refindex-1, forward_nmatch] = matchindex
                            reference_forward_size[refindex-1, forward_nmatch] = len(np.extract(new_convcold_cloudnumber == matchindex, new_convcold_cloudnumber))

                            forward_nmatch = forward_nmatch + 1

            ######################################################
            # Loop through each cloud / feature at new time and look for overlaping clouds / features in the reference file
            for newindex in np.arange(1,nnew+1):
                # Locate where the cloud in the new file overlaps with any cloud in the reference file
                backward_matchindices = np.where((new_convcold_cloudnumber == newindex) & (reference_convcold_cloudnumber != 0))

                # Get the convcold_cloudnumber of the clouds in the reference file that overlap the cloud in the new file
                backward_refindex = reference_convcold_cloudnumber[backward_matchindices]
                unique_backwardrefindex = np.unique(backward_refindex)

                # Calculate size of reference cloud in terms of number of pixels
                sizenew = len(np.extract(new_convcold_cloudnumber == newindex, new_convcold_cloudnumber))

                # Loop through the overlapping clouds in the new file, determining if they statisfy the overlap requirement
                backward_nmatch = 0 # Initialize overlap counter
                for matchindex in unique_backwardrefindex: 
                    sizematch = len(np.extract(backward_refindex == matchindex, backward_refindex))

                    if sizematch/float(sizenew) > othresh:
                        if backward_nmatch > maxlinks:
                            print('reference: ' + number_filepath + files[ifile-1])
                            print('new: ' + number_filepath + files[ifile])
                            sys.exit('More than ' + str(int(maxlinks)) + ' clouds in reference file match with new cloud?!')
                        else:
                            new_backward_index[newindex-1, backward_nmatch] = matchindex
                            new_backward_size[newindex-1, backward_nmatch] = len(np.extract(reference_convcold_cloudnumber == matchindex, reference_convcold_cloudnumber))

                            backward_nmatch = backward_nmatch + 1

            #########################################################
            # Save forward and backward indices and linked sizes in netcdf file

            # create file
            filesave = Dataset(dataoutpath + outfilebase + new_filedatetime + '.nc', 'w', format='NETCDF4_CLASSIC')

            # set global attributes
            filesave.Convenctions = 'CF-1.6'
            filesave.title = 'Indices linking clouds in two consecutive files forward and backward in time and the size of the linked cloud'
            filesave.institution = 'Pacific Northwest National Laboratory'
            filesave.setncattr('Contact', 'Hannah C Barnes: hannah.barnes@pnnl.gov')
            filesave.history = 'Created ' + time.ctime(time.time())
            filesave.setncattr('new_date', new_filedatetime)
            filesave.setncattr('ref_date', reference_filedatetime)
            filesave.setncattr('new_file', cloudid_filepath + files[ifile])
            filesave.setncattr('ref_file', cloudid_filepath + files[ifile-1])
            filesave.setncattr('tracking_version_number', track_version)
            filesave.setncattr('overlap_threshold', str(othresh) +'%')
            filesave.setncattr('maximum_gap_allowed', str(timegap)+ ' hr')

            # create netcdf dimensions
            filesave.createDimension('time', None)
            filesave.createDimension('nclouds_new', nnew)
            filesave.createDimension('nclouds_ref', nreference)
            filesave.createDimension('nlinks', maxlinks)

            # define variables
            basetime_new = filesave.createVariable('basetime_new', 'i4', 'time', zlib=True)
            basetime_new.long_name = 'base time of new file in epoch'
            basetime_new.units = 'seconds since 01/01/1970 00:00'
            basetime_new.standard_name = 'time'

            basetime_ref = filesave.createVariable('basetime_ref', 'i4', 'time', zlib=True)
            basetime_ref.long_name = 'base time of reference in epoch'
            basetime_ref.units = 'seconds since 01/01/1970 00:00'
            basetime_ref.standard_name = 'time'

            nclouds_new = filesave.createVariable('nclouds_new', 'i4', 'time', zlib=True)
            nclouds_new.long_name = 'number of cloud systems in new file'
            nclouds_new.units = 'unitless'

            nclouds_ref = filesave.createVariable('nclouds_ref', 'i4', 'time', zlib=True)
            nclouds_ref.long_name = 'number of cloud systems in reference file'
            nclouds_ref.units = 'unitless'

            nlinks = filesave.createVariable('nlinks', 'i4', 'time', zlib=True)
            nlinks.long_name = 'maximum number of clouds that can be linked to a given cloud'
            nlinks.units = 'unitless'

            newcloud_backward_index = filesave.createVariable('newcloud_backward_index', 'i4', ('time', 'nclouds_new', 'nlinks'), zlib=True, fill_value=-9999)
            newcloud_backward_index.long_name = 'new cloud index'
            newcloud_backward_index.description = 'each row provides all reference cloud indices linked to that cloud in the new file'
            newcloud_backward_index.units = 'unitless'
            newcloud_backward_index.min_value = 1
            newcloud_backward_index.max_value = nreference

            refcloud_forward_index = filesave.createVariable('refcloud_forward_index', 'i4', ('time', 'nclouds_ref', 'nlinks'), zlib=True, fill_value=-9999)
            refcloud_forward_index.long_name = 'new cloud index'
            refcloud_forward_index.description = 'each row provides all new cloud indices linked to that cloud in the reference file'
            refcloud_forward_index.units = 'unitless'
            refcloud_forward_index.min_value = 1
            refcloud_forward_index.max_value = nnew

            newcloud_backward_size = filesave.createVariable('newcloud_backward_size', 'f4', ('time', 'nclouds_new', 'nlinks'), zlib=True, fill_value=-9999)
            newcloud_backward_size.long_name = 'new cloud area'
            newcloud_backward_size.description = 'each row provides the area of all reference clouds linked to that cloud in the new file'
            newcloud_backward_size.units = 'km^2'

            refcloud_forward_size = filesave.createVariable('refcloud_forward_size', 'f4', ('time', 'nclouds_ref', 'nlinks'), zlib=True, fill_value=-9999)
            refcloud_forward_size.long_name = 'ref cloud area'
            refcloud_forward_size.description = 'each row provides the area of all new clouds linked to that cloud in the reference file'
            refcloud_forward_size.units = 'km^2'

            # fill variables
            basetime_new[:] = reference_basetime
            basetime_ref[:] = new_basetime
            nclouds_new[:] = nnew
            nclouds_ref[:] = nreference
            nlinks[:] = maxlinks
            newcloud_backward_index[0,:,:] = new_backward_index[:,:]
            refcloud_forward_index[0,:,:] = reference_forward_index[:,:]
            newcloud_backward_size[0,:,:] = new_backward_size[:,:]
            refcloud_forward_size[0,:,:] = reference_forward_size[:,:]

            # write and close file
            filesave.close()


            











