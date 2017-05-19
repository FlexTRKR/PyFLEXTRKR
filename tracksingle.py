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

def trackclouds_mergedir(cloudidfilenames, datestrings, timestrings, basetimes, dataoutpath, track_version, timegap, nmaxlinks, othresh, startdate, enddate):
    ##############################################################
    # Set constants
    # Version information
    outfilebase = 'track' + track_version + '_'

    ########################################################
    # Isolate new and reference file and base times
    new_file = cloudidfilenames[1]
    new_datestring = datestrings[1]
    new_timestring = timestrings[1]
    new_basetime = basetimes[1]
    new_filedatetime = new_datestring + '_' + new_timestring
    
    reference_file = cloudidfilenames[0]
    reference_datestring = datestrings[0]
    reference_timestring = timestrings[0]
    reference_basetime = basetimes[0]
    reference_filedatetime = reference_datestring + '_' + reference_timestring

    # Check that new and reference files differ by less than timegap in hours. Use base time (which is the seconds since 01-Jan-1970 00:00:00). Divide base time difference between the files by 3600 to get difference in hours
    hour_diff = (new_basetime - reference_basetime)/float(3600)
    if hour_diff < timegap and hour_diff > 0:
        print('Linking:')

        ##############################################################
        # Load cloudid file from before, called reference file
        print(reference_filedatetime)

        reference_data = Dataset(reference_file, 'r')                                        # Open file
        reference_convcold_cloudnumber = reference_data.variables['convcold_cloudnumber'][:]                    # Load cloud id map
        nreference = reference_data.variables['nclouds'][:]                                                     # Load number of clouds / features
        reference_data.close()                                                                                  # close file
        
        ##########################################################
        # Load cloudid file from before, called new file
        print(new_filedatetime)

        new_data = Dataset(new_file, 'r')                                # Open file
        new_convcold_cloudnumber = new_data.variables['convcold_cloudnumber'][:]                # Load cloud id map
        nnew = new_data.variables['nclouds'][:]                                                 # Load number of clouds / features
        new_data.close()                                                                        # close file

        ############################################################
        # Get size of data
        times, ny, nx = np.shape(new_convcold_cloudnumber)

        #######################################################
        # Initialize matrices
        reference_forward_index = np.ones((nreference, nmaxlinks), dtype=int)*-9999
        reference_forward_size = np.ones((nreference, nmaxlinks), dtype=int)*-9999
        new_backward_index = np.ones((nnew, nmaxlinks), dtype=int)*-9999
        new_backward_size =  np.ones((nnew, nmaxlinks), dtype=int)*-9999

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
                    if forward_nmatch > nmaxlinks:
                        print('reference: ' + number_filepath + files[ifile-1])
                        print('new: ' + number_filepath + files[ifile])
                        sys.exit('More than ' + str(int(nmaxlinks)) + ' clouds in new file match with reference cloud?!')
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
                    if backward_nmatch > nmaxlinks:
                        print('reference: ' + number_filepath + files[ifile-1])
                        print('new: ' + number_filepath + files[ifile])
                        sys.exit('More than ' + str(int(nmaxlinks)) + ' clouds in reference file match with new cloud?!')
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
        filesave.setncattr('new_file', new_file)
        filesave.setncattr('ref_file', reference_file)
        filesave.setncattr('tracking_version_number', track_version)
        filesave.setncattr('overlap_threshold', str(othresh) +'%')
        filesave.setncattr('maximum_gap_allowed', str(timegap)+ ' hr')
        
        # create netcdf dimensions
        filesave.createDimension('time', None)
        filesave.createDimension('nclouds_new', nnew)
        filesave.createDimension('nclouds_ref', nreference)
        filesave.createDimension('nlinks', nmaxlinks)
        
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
        newcloud_backward_index.long_name = 'reference cloud index'
        newcloud_backward_index.description = 'each row represents a cloud in the new file and the numbers in that row provide all reference cloud indices linked to that new cloud'
        newcloud_backward_index.units = 'unitless'
        newcloud_backward_index.min_value = 1
        newcloud_backward_index.max_value = nreference
        
        refcloud_forward_index = filesave.createVariable('refcloud_forward_index', 'i4', ('time', 'nclouds_ref', 'nlinks'), zlib=True, fill_value=-9999)
        refcloud_forward_index.long_name = 'new cloud index'
        refcloud_forward_index.description = 'each row represents a cloud in the reference file and the numbers provide all new cloud indices linked to that reference cloud'
        refcloud_forward_index.units = 'unitless'
        refcloud_forward_index.min_value = 1
        refcloud_forward_index.max_value = nnew
        
        newcloud_backward_size = filesave.createVariable('newcloud_backward_size', 'f4', ('time', 'nclouds_new', 'nlinks'), zlib=True, fill_value=-9999)
        newcloud_backward_size.long_name = 'reference cloud area'
        newcloud_backward_size.description = 'each row represents a cloud in the new file and the numbers provide the area of all reference clouds linked to that new cloud'
        newcloud_backward_size.units = 'km^2'
        
        refcloud_forward_size = filesave.createVariable('refcloud_forward_size', 'f4', ('time', 'nclouds_ref', 'nlinks'), zlib=True, fill_value=-9999)
        refcloud_forward_size.long_name = 'new cloud area'
        refcloud_forward_size.description = 'each row represents a cloud in the reference file and the numbers provide the area of all new clouds linked to that reference cloud'
        refcloud_forward_size.units = 'km^2'
        
        # fill variables
        basetime_new[:] = reference_basetime
        basetime_ref[:] = new_basetime
        nclouds_new[:] = nnew
        nclouds_ref[:] = nreference
        nlinks[:] = nmaxlinks
        newcloud_backward_index[0,:,:] = new_backward_index[:,:]
        refcloud_forward_index[0,:,:] = reference_forward_index[:,:]
        newcloud_backward_size[0,:,:] = new_backward_size[:,:]
        refcloud_forward_size[0,:,:] = reference_forward_size[:,:]

        # write and close file
        filesave.close()


            











