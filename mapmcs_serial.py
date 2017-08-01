# Purpose: Subset statistics file to keep only MCS. Uses brightness temperature statstics of cold cloud shield area, duration, and eccentricity base on Fritsch et al (1986) and Maddos (1980)

# Author: Original IDL code written by Zhe Feng (zhe.feng@pnnl.gov), Python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov)

def mapmcs_mergedir(filebasetime, mcsstats_filebase, statistics_filebase, mcstracking_path, stats_path, tracking_path, cloudid_filebase, absolutetb_threshs, startdate, enddate):
    #######################################################################
    # Import modules
    import numpy as np
    from netCDF4 import Dataset
    import time
    import os
    import sys

    ######################################################################
    # define constants:
    # minimum and maximum brightness temperature thresholds. data outside of this range is filtered
    mintb_thresh = absolutetb_threshs[0]    # k
    maxtb_thresh = absolutetb_threshs[1]    # k

    fillvalue = -9999

    ##################################################################
    # Load all track stat file
    statistics_file = stats_path + statistics_filebase + '_' + startdate + '_' + enddate + '.nc'
    print(statistics_file)

    allstatdata = Dataset(statistics_file, 'r')
    trackstat_basetime = allstatdata.variables['basetime'][:] # Time of cloud in seconds since 01/01/1970 00:00
    trackstat_cloudnumber = allstatdata.variables['cloudnumber'][:] # Number of the corresponding cloudid file
    trackstat_status = allstatdata.variables['status'][:] # Flag indicating the status of the cloud
    allstatdata.close()

    #######################################################################
    # Load MCS track stat file
    mcsstatistics_file = stats_path + mcsstats_filebase + startdate + '_' + enddate + '.nc'
    print(mcsstatistics_file)

    allmcsdata = Dataset(mcsstatistics_file, 'r')
    mcstrackstat_basetime = allmcsdata.variables['mcs_basetime'][:] # basetime of each cloud in the tracked mcs
    mcstrackstat_status = allmcsdata.variables['mcs_status'][:] # flag indicating the status of each cloud in the tracked mcs
    mcstrackstat_cloudnumber = allmcsdata.variables['mcs_cloudnumber'][:] # number of cloud in the corresponding cloudid file for each cloud in the tracked mcs
    mcstrackstat_mergecloudnumber = allmcsdata.variables['mcs_mergecloudnumber'][:] # number of cloud in the corresponding cloud file that merges into the tracked mcs
    mcstrackstat_splitcloudnumber = allmcsdata.variables['mcs_splitcloudnumber'][:] # number of cloud in the corresponding cloud file that splits into the tracked mcs
    source = str(Dataset.getncattr(allstatdata, 'source'))
    description = str(Dataset.getncattr(allstatdata, 'description'))
    pixel_radius = str(Dataset.getncattr(allstatdata, 'pixel_radius_km'))
    area_thresh = str(Dataset.getncattr(allstatdata, 'MCS_area_km**2'))
    duration_thresh = str(Dataset.getncattr(allstatdata, 'MCS_duration_hour'))
    eccentricity_thresh = str(Dataset.getncattr(allstatdata, 'MCS_eccentricity'))
    allmcsdata.close()

    for itime in range(0,30):
        print(mcstrackstat_mergecloudnumber[itime,:,0:10])
        raw_input('waiting')
    #########################################################################
    # Get tracks and times associated with this time
    itrack, itime = np.array(np.where(mcstrackstat_basetime == filebasetime))
    timestatus = np.copy(mcstrackstat_status[itrack,itime])
    ntimes = len(itime)

    print(itrack)
    print(itime)

    if ntimes > 0:
        # Get cloudid file associated with this time
        file_datetime = time.strftime("%Y%m%d_%H%M", time.gmtime(np.copy(filebasetime)))
        filedate = np.copy(file_datetime[0:8])
        filetime = np.copy(file_datetime[9:14])
        ifile = tracking_path + cloudid_filebase + file_datetime + '.nc'
        print(ifile)

        if os.path.isfile(ifile):
            # Load cloudid data
            cloudiddata = Dataset(ifile, 'r')
            cloudid_basetime = cloudiddata.variables['basetime'][:]
            cloudid_latitude = cloudiddata.variables['latitude'][:]
            cloudid_longitude = cloudiddata.variables['longitude'][:]
            cloudid_tb = cloudiddata.variables['tb'][:]
            cloudid_cloudnumber = cloudiddata.variables['cloudnumber'][:]
            cloudid_cloudtype = cloudiddata.variables['cloudtype'][:]
            cloudid_nclouds = cloudiddata.variables['nclouds'][:]
            cloudiddata.close()

            # Get data dimensions
            [timeindex, nlat, nlon] = np.shape(cloudid_cloudnumber)
                    
            # Intiailize track maps
            mcstrackmap = np.ones((nlat,nlon), dtype=int)*fillvalue
            mcstrackmap_mergesplit = np.ones((nlat,nlon), dtype=int)*fillvalue
            statusmap = np.ones((nlat,nlon), dtype=int)*fillvalue
            trackmap = np.ones((nlat,nlon), dtype=int)*fillvalue
            #mcstrackmap = np.zeros((nlat,nlon), dtype=int)
            #mcstrackmap_mergesplit = np.zeros((nlat,nlon), dtype=int)

            ###############################################################
            # Create map of status and track number for every feature in this file
            fulltrack, fulltime = np.array(np.where(trackstat_basetime == filebasetime))
            for ifull in range(0,len(fulltime)):
                ffcloudnumber = trackstat_cloudnumber[fulltrack[ifull], fulltime[ifull]]
                ffstatus = trackstat_status[fulltrack[ifull], fulltime[ifull]]
                
                fullypixels, fullxpixels = np.array(np.where(cloudid_cloudnumber[0,:,:] == ffcloudnumber))

                statusmap[fullypixels, fullxpixels] = ffstatus
                trackmap[fullypixels, fullxpixels] = fulltrack[ifull] + 1

            ##############################################################
            # Loop over each cloud in this unique file
            for jj in range(0,ntimes):
                print('JJ:' + str(jj))
                # Get cloud nummber
                jjcloudnumber = mcstrackstat_cloudnumber[itrack[jj],itime[jj]]

                # Find pixels assigned to this cloud number
                jjcloudypixels, jjcloudxpixels = np.array(np.where(cloudid_cloudnumber[0,:,:] == jjcloudnumber))

                # Label this cloud with the track number. Need to add one to the cloud number since have the index number and we want the track number
                if len(jjcloudypixels) > 0:
                    mcstrackmap[jjcloudypixels, jjcloudxpixels] = itrack[jj] + 1
                    mcstrackmap_mergesplit[jjcloudypixels, jjcloudxpixels] = itrack[jj] + 1

                    #statusmap[jjcloudypixels, jjcloudxpixels] = timestatus[jj] 
                else:
                    sys.exit('Error: No matching cloud pixel found?!')

                ###########################################################
                # Find merging clouds
                jjmerge = np.array(np.where(mcstrackstat_mergecloudnumber[itrack[jj], itime[jj],:] > 0))[0,:]

                # Loop through merging clouds if present
                if len(jjmerge) > 0:
                    for imerge in jjmerge:
                        # Find cloud number asosicated with the merging cloud
                        jjmergeypixels, jjmergexpixels = np.array(np.where(cloudid_cloudnumber[0,:,:] == mcstrackstat_mergecloudnumber[itrack[jj], itime[jj], imerge]))
                        
                        # Label this cloud with the track number. Need to add one to the cloud number since have the index number and we want the track number
                        if len(jjmergeypixels) > 0:
                            mcstrackmap_mergesplit[jjmergeypixels, jjmergexpixels] = itrack[jj] + 1
                            #statusmap[jjmergeypixels, jjmergexpixels] = mcsmergestatus[itrack[jj], itime[jj], imerge]
                        else:
                            sys.exit('Error: No matching merging cloud pixel found?!')

                ###########################################################
                # Find splitting clouds
                jjsplit = np.array(np.where(mcstrackstat_splitcloudnumber[itrack[jj], itime[jj],:] > 0))[0,:]
                np.set_printoptions(threshold=np.inf)
                print(mcstrackstat_splitcloudnumber[itrack[jj], itime[jj],0:50])
                print(jjsplit)
                raw_input('Check')

                # Loop through splitting clouds if present
                if len(jjsplit) > 0:
                    for isplit in jjsplit:
                        print(isplit)
                        print(mcstrackstat_splitcloudnumber[itrack[jj], itime[jj], 0:50])
                        # Find cloud number asosicated with the splitting cloud
                        jjsplitypixels, jjsplitxpixels = np.array(np.where(cloudid_cloudnumber[0,:,:] == mcstrackstat_splitcloudnumber[itrack[jj], itime[jj], isplit]))
                        print(mcstrackstat_splitcloudnumber[itrack[jj], itime[jj], isplit])
                        raw_input('Waiting')
                                
                        # Label this cloud with the track number. Need to add one to the cloud number since have the index number and we want the track number
                        if len(jjsplitypixels) > 0:
                            mcstrackmap_mergesplit[jjsplitypixels, jjsplitxpixels] = itrack[jj] + 1
                            print('Split')
                            print(itrack)
                            print(itrack[jj])
                            #statusmap[jjsplitypixels, jjsplitxpixels] = mcssplitstatus[itrack[jj], itime[jj], isplit]
                        else:
                            sys.exit('Error: No matching splitting cloud pixel found?!')
                    raw_input('check 2')
            print('Stop')

            #####################################################################
            # Output maps to netcdf file

            # Create output directories
            if not os.path.exists(mcstracking_path):
                os.makedirs(mcstracking_path)

            # Create file
            mcsmcstrackmaps_outfile = mcstracking_path + 'mcstracks_' + str(filedate) + '_' + str(filetime) + '.nc'
            filesave = Dataset(mcsmcstrackmaps_outfile, 'w', format='NETCDF4_CLASSIC')

            # Set global attributes
            filesave.Convenctions = 'CF-1.6'
            filesave.title = 'Pixel level of tracked clouds and MCSs'
            filesave.institution = 'Pacific Northwest National Laboratory'
            filesave.setncattr('Contact', 'Hannah C Barnes: hannah.barnes@pnnl.gov')
            filesave.history = 'Created ' + time.ctime(time.time())
            filesave.setncattr('source', source)
            filesave.setncattr('description', description)
            filesave.setncattr('pixel_radius_km', pixel_radius)
            filesave.setncattr('MCS_area_km^2', area_thresh)
            filesave.setncattr('MCS_duration_hour', duration_thresh)
            filesave.setncattr('MCS_eccentricity', eccentricity_thresh)
                
            # Create dimensions
            filesave.createDimension('time', None)
            filesave.createDimension('lat', nlat)
            filesave.createDimension('lon', nlon)
            filesave.createDimension('ndatetimechars', 13)
            
            # Define variables
            basetime = filesave.createVariable('mcs_basetime', 'i4', ('time'), zlib=True, complevel=5, fill_value=fillvalue)
            basetime.standard_name = 'time'
            basetime.long_name = 'epoch time'
            basetime.description = 'basetime of clouds in this file'
            basetime.units = 'seconds since 01/01/1970 00:00'
            basetime.fill_value = fillvalue

            latitude = filesave.createVariable('latitude', 'f4', ('lat', 'lon'), zlib=True, complevel=5, fill_value=fillvalue)
            latitude.long_name = 'y-coordinate in Cartesian system'
            latitude.valid_min = np.nanmin(np.nanmin(cloudid_latitude))
            latitude.valid_max = np.nanmax(np.nanmax(cloudid_latitude))
            latitude.axis = 'Y'
            latitude.units = 'degrees_north'
            latitude.standard_name = 'latitude'
                    
            longitude = filesave.createVariable('longitude', 'f4', ('lat', 'lon'), zlib=True, complevel=5, fill_value=fillvalue)
            longitude.valid_min = np.nanmin(np.nanmin(cloudid_longitude))
            longitude.valid_max = np.nanmax(np.nanmax(cloudid_longitude))
            longitude.axis = 'X'
            longitude.long_name = 'x-coordinate in Cartesian system'
            longitude.units = 'degrees_east'
            longitude.standard_name = 'longitude'
                    
            nclouds = filesave.createVariable('nclouds', 'i4', 'time', zlib=True, complevel=5, fill_value=fillvalue)
            nclouds.long_name = 'number of distict convective cores identified in file'
            nclouds.units = 'unitless'
            
            tb = filesave.createVariable('tb', 'f4', ('time', 'lat', 'lon'), zlib=True, complevel=5, fill_value=fillvalue)
            tb.long_name = 'brightness temperature'
            tb.units = 'K'
            tb.valid_min = mintb_thresh
            tb.valid_max = maxtb_thresh
            tb.standard_name = 'brightness_temperature'
            tb.fill_value = fillvalue
            
            cloudnumber = filesave.createVariable('cloudnumber', 'i4', ('time', 'lat', 'lon'), zlib=True, complevel=5, fill_value=0)
            cloudnumber.long_name = 'number of cloud system that a given pixel belongs to'
            cloudnumber.units = 'unitless'
            cloudnumber.comment = 'the extend of the cloud system is defined using the warm anvil threshold'
            cloudnumber.fillvalue = 0
                    
            cloudstatus = filesave.createVariable('cloudstatus', 'i4', ('time', 'lat', 'lon'), zlib=True, complevel=5, fill_value=fillvalue)
            cloudstatus.long_name = 'flag indicating status of the flag'
            cloudstatus.values = "-9999=missing cloud or cloud removed due to short track, 0=track ends here, 1=cloud continues as one cloud in next file, 2=Biggest cloud in merger, 21=Smaller cloud(s) in merger, 13=Cloud that splits, 3=Biggest cloud from a split that stops after the split, 31=Smaller cloud(s) from a split that stop after the split. The last seven classifications are added together in different combinations to describe situations."
            cloudstatus.units = 'unitless'
            cloudstatus.comment = 'the extend of the cloud system is defined using the warm anvil threshold'
            cloudstatus.fillvalue = fillvalue 

            tracknumber = filesave.createVariable('tracknumber', 'f4', ('time', 'lat', 'lon'), zlib=True, complevel=5, fill_value=fillvalue)
            tracknumber.long_name = 'track number that a given pixel belongs to'
            tracknumber.units = 'unitless'
            tracknumber.comment = 'the extend of the cloud system is defined using the warm anvil threshold'
            tracknumber.fillvalue = fillvalue

            mcstracknumber = filesave.createVariable('mcstracknumber', 'f4', ('time', 'lat', 'lon'), zlib=True, complevel=5, fill_value=fillvalue)
            mcstracknumber.long_name = 'mcs track number that a given pixel belongs to'
            mcstracknumber.units = 'unitless'
            mcstracknumber.comment = 'the extend of the cloud system is defined using the warm anvil threshold'
                    
            mcstracknumber_mergesplit = filesave.createVariable('mcstracknumber_mergesplit', 'i4', ('time', 'lat', 'lon'), zlib=True, complevel=5, fill_value=fillvalue)
            mcstracknumber_mergesplit.long_name = 'mcs track number that a given pixel belongs to, includes clouds that merge into and split from each mcs'
            mcstracknumber_mergesplit.units = 'unitless'
            mcstracknumber_mergesplit.comment = 'the extend of the cloud system is defined using the warm anvil threshold'

            # Fill variables
            basetime[:] = cloudid_basetime
            longitude[:,:] = cloudid_longitude
            latitude[:,:] = cloudid_latitude
            nclouds[:] = cloudid_nclouds
            tb[0,:,:] = cloudid_tb
            cloudnumber[0,:,:] = cloudid_cloudnumber[:,:]
            cloudstatus[0,:,:] = statusmap[:,:]
            tracknumber[0,:,:] = trackmap[:,:]
            mcstracknumber[0,:,:] = mcstrackmap[:,:]
            mcstracknumber_mergesplit[0,:,:] = mcstrackmap_mergesplit[:,:]
                
            # Close and save file
            filesave.close()
                
        else:
            sys.exit(ifile + ' does not exist?!"')
    else:
        sys.exit('No MCSs')

