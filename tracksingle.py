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

def trackclouds_mergedir(zipped_inputs): #firstcloudidfilename, secondcloudidfilename, firstdatestring, seconddatestring, firsttimestring, secondtimestring, firstbasetime, secondbasetime, dataoutpath, track_version, timegap, nmaxlinks, othresh, startdate, enddate):
    ########################################################
    import numpy as np
    import glob
    import os
    import re
    import fnmatch
    from netCDF4 import Dataset
    import time, datetime, calendar
    from pytz import timezone, utc
    import sys
    import xarray as xr

    # Separate inputs
    firstcloudidfilename = zipped_inputs[0]
    secondcloudidfilename = zipped_inputs[1]
    firstdatestring = zipped_inputs[2]
    seconddatestring = zipped_inputs[3]
    firsttimestring = zipped_inputs[4]
    secondtimestring = zipped_inputs[5]
    firstbasetime = zipped_inputs[6]
    secondbasetime = zipped_inputs[7]
    dataoutpath = zipped_inputs[8]
    track_version = zipped_inputs[9]
    timegap = zipped_inputs[10]
    nmaxlinks = zipped_inputs[11]
    othresh = zipped_inputs[12]
    startdate = zipped_inputs[13]
    enddate = zipped_inputs[14]

    ########################################################
    # Set constants
    # Version information
    outfilebase = 'track' + track_version + '_'

    fillvalue = -9999

    ########################################################
    # Isolate new and reference file and base times
    new_file = secondcloudidfilename
    new_datestring = seconddatestring
    new_timestring = secondtimestring
    new_basetime = secondbasetime
    new_filedatetime = str(new_datestring) + '_' + str(new_timestring)
    
    reference_file = firstcloudidfilename
    reference_datestring = firstdatestring
    reference_timestring = firsttimestring
    reference_basetime = firstbasetime
    reference_filedatetime = str(reference_datestring) + '_' + str(reference_timestring)

    # Check that new and reference files differ by less than timegap in hours. Use base time (which is the seconds since 01-Jan-1970 00:00:00). Divide base time difference between the files by 3600 to get difference in hours
    hour_diff = (np.subtract(new_basetime, reference_basetime))/float(3600)
    if hour_diff < timegap and hour_diff > 0:
        print('Linking:')

        ##############################################################
        # Load cloudid file from before, called reference file
        print(reference_filedatetime)

        reference_data = xr.open_dataset(reference_file, autoclose=True)                                # Open file
        reference_convcold_cloudnumber = reference_data['convcold_cloudnumber'].data                    # Load cloud id map
        nreference = reference_data['nclouds'].data                                                     # Load number of clouds / features

        # Replace nans with -9999
        reference_convcold_cloudnumber[np.where(~np.isfinite(reference_convcold_cloudnumber))] = fillvalue

        ##########################################################
        # Load cloudid file from before, called new file
        print(new_filedatetime)

        new_data = xr.open_dataset(new_file, autoclose=True)                            # Open file
        new_convcold_cloudnumber = new_data['convcold_cloudnumber'].data                # Load cloud id map
        nnew = new_data['nclouds'].data                                                 # Load number of clouds / features

        # Replace nans with -9999
        new_convcold_cloudnumber[np.where(~np.isfinite(new_convcold_cloudnumber))] = fillvalue

        ############################################################
        # Get size of data
        times, ny, nx = np.shape(new_convcold_cloudnumber)

        #######################################################
        # Initialize matrices
        reference_forward_index = np.ones((1, int(nreference), int(nmaxlinks)), dtype=int)*fillvalue
        reference_forward_size = np.ones((1, int(nreference), int(nmaxlinks)), dtype=int)*fillvalue
        new_backward_index = np.ones((1, int(nnew), int(nmaxlinks)), dtype=int)*fillvalue
        new_backward_size =  np.ones((1, int(nnew), int(nmaxlinks)), dtype=int)*fillvalue

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
                        reference_forward_index[0, int(refindex)-1, forward_nmatch] = matchindex
                        reference_forward_size[0, int(refindex)-1, forward_nmatch] = len(np.extract(new_convcold_cloudnumber == matchindex, new_convcold_cloudnumber))
                        
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
                        new_backward_index[0, int(newindex)-1, backward_nmatch] = matchindex
                        new_backward_size[0, int(newindex)-1, backward_nmatch] = len(np.extract(reference_convcold_cloudnumber == matchindex, reference_convcold_cloudnumber))
                        
                        backward_nmatch = backward_nmatch + 1

        #########################################################
        # Save forward and backward indices and linked sizes in netcdf file

        # create filename
        track_outfile = dataoutpath + outfilebase + new_filedatetime + '.nc'

        # Check if file already exists. If exists, delete
        if os.path.isfile(track_outfile):
            os.remove(track_outfile)

        print('Writing all tracks')
        print(track_outfile)
        print('')

        # Define xarracy dataset
        output_data = xr.Dataset({'basetime_new': (['time'], new_data['basetime'].data), \
                                  'basetime_ref': (['time'], reference_data['basetime'].data), \
                                  'newcloud_backward_index': (['time', 'nclouds_new', 'nlinks'], new_backward_index), \
                                  'newcloud_backward_size': (['time', 'nclouds_new', 'nlinks'], new_backward_size), \
                                  'refcloud_forward_index': (['time', 'nclouds_ref', 'nlinks'], reference_forward_index), \
                                  'refcloud_forward_size': (['time', 'nclouds_ref', 'nlinks'], reference_forward_size)}, \
                                 coords={'time': (['time'], np.arange(0, 1)), \
                                         'nclouds_new': (['nclouds_new'], np.arange(0, nnew)), \
                                         'nclouds_ref': (['nclouds_ref'], np.arange(0, nreference)), \
                                         'nlinks': (['nlinks'], np.arange(0, nmaxlinks))}, \
                                 attrs={'title': 'Indices linking clouds in two consecutive files forward and backward in time and the size of the linked cloud', \
                                        'Conventions':'CF-1.6', \
                                        'Institution': 'Pacific Northwest National Laboratoy', \
                                        'Contact': 'Hannah C Barnes: hannah.barnes@pnnl.gov', \
                                        'Created_on':  time.ctime(time.time()), \
                                        'new_date': new_filedatetime, \
                                        'ref_date': reference_filedatetime, \
                                        'new_file': new_file, \
                                        'ref_file': reference_file, \
                                        'tracking_version_number': track_version, \
                                        'overlap_threshold': str(othresh) +'%', \
                                        'maximum_gap_allowed': str(timegap)+ ' hr'})

        # Specify variable attributes
        output_data.nclouds_new.attrs['long_name'] = 'number of cloud in new file'
        output_data.nclouds_new.attrs['units'] = 'unitless'

        output_data.nclouds_ref.attrs['long_name'] = 'number of cloud in reference file'
        output_data.nclouds_ref.attrs['units'] = 'unitless'

        output_data.nlinks.attrs['long_name'] = 'maximum number of clouds that can be linked to a given cloud'
        output_data.nlinks.attrs['units'] = 'unitless'

        output_data.basetime_new.attrs['long_name'] = 'epoch time (seconds since 01/01/1970 00:00) of new file'
        output_data.basetime_new.attrs['standard_name'] = 'time'

        output_data.basetime_ref.attrs['long_name'] = 'epoch time (seconds since 01/01/1970 00:00) of reference file'
        output_data.basetime_ref.attrs['standard_name'] = 'time'

        output_data.newcloud_backward_index.attrs['long_name'] = 'reference cloud index'
        output_data.newcloud_backward_index.attrs['usage'] = 'each row represents a cloud in the new file and the numbers in that row provide all reference cloud indices linked to that new cloud'
        output_data.newcloud_backward_index.attrs['units'] = 'unitless'
        output_data.newcloud_backward_index.attrs['valid_min'] = 1
        output_data.newcloud_backward_index.attrs['valid_max'] = nreference

        output_data.refcloud_forward_index.attrs['long_name'] = 'new cloud index'
        output_data.refcloud_forward_index.attrs['usage'] =  'each row represents a cloud in the reference file and the numbers provide all new cloud indices linked to that reference cloud'
        output_data.refcloud_forward_index.attrs['units'] = 'unitless'
        output_data.refcloud_forward_index.attrs['valid_min'] = 1
        output_data.refcloud_forward_index.attrs['valid_max'] = nnew

        output_data.newcloud_backward_size.attrs['long_name'] = 'reference cloud area'
        output_data.newcloud_backward_size.attrs['usage'] = 'each row represents a cloud in the new file and the numbers provide the area of all reference clouds linked to that new cloud'
        output_data.newcloud_backward_size.attrs['units'] = 'km^2'

        output_data.refcloud_forward_size.attrs['long_name'] = 'new cloud area'
        output_data.refcloud_forward_size.attrs['usage'] = 'each row represents a cloud in the reference file and the numbers provide the area of all new clouds linked to that reference cloud'
        output_data.refcloud_forward_size.attrs['units'] = 'km^2'

        # Write netcdf files
        output_data.to_netcdf(path=track_outfile, mode='w', format='NETCDF4_CLASSIC', unlimited_dims='times', \
                              encoding={'basetime_new': {'dtype':'int64', 'zlib':True}, \
                                        'basetime_ref': {'dtype':'int64', 'zlib':True}, \
                                        'newcloud_backward_index': {'dtype': 'int', 'zlib':True, '_FillValue': fillvalue}, \
                                        'newcloud_backward_size': {'dtype': 'int', 'zlib':True, '_FillValue': fillvalue}, \
                                        'refcloud_forward_index': {'dtype': 'int', 'zlib':True, '_FillValue': fillvalue}, \
                                        'refcloud_forward_size': {'dtype': 'int', 'zlib':True, '_FillValue': fillvalue}})












