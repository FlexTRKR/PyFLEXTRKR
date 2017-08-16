# Purpose: match mergedir tracked MCS with NMQ CSA to calculate radar-based statsitics underneath the cloud features.

# Author: Original IDL code written by Zhe Feng (zhe.feng@pnnl.gov), Python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov)

def identifypf_mergedir_rainrate(mcsstats_filebase, cloudid_filebase, stats_path, cloudidtrack_path, pfdata_path, startdate, enddate, nmaxpf, nmaxcore, nmaxpix):
    import numpy as np
    from netCDF4 import Dataset
    import os.path

    np.set_printoptions(threshold=np.inf)

    #########################################################
    # Set constants
    fillvalue = -9999

    #########################################################
    # Load MCS track stats
    mcsirstatistics_file = stats_path + mcsstats_filebase + startdate + '_' + enddate + '.nc'
    print(mcsirstatistics_file)

    mcsirstatdata = Dataset(mcsirstatistics_file, 'r')
    mcs_ir_ntracks = len(mcsirstatdata.dimensions['ntracks']) # Total number of tracked features
    mcs_ir_nmaxlength = len(mcsirstatdata.dimensions['ntimes']) # Maximum number of features in a given track
    mcs_ir_nmaxmergesplits = len(mcsirstatdata.dimensions['nmergers']) # Maximum number of features in a given track
    ir_description = str(Dataset.getncattr(mcsirstatdata, 'description'))
    ir_source = str(Dataset.getncattr(mcsirstatdata, 'source'))
    ir_time_res = str(Dataset.getncattr(mcsirstatdata, 'time_resolution_hour'))
    ir_pixel_radius = str(Dataset.getncattr(mcsirstatdata, 'pixel_radius_km'))
    mcsarea_thresh = str(Dataset.getncattr(mcsirstatdata, 'MCS_area_km**2'))
    mcsduration_thresh = str(Dataset.getncattr(mcsirstatdata, 'MCS_duration_hour'))
    mcseccentricity_thresh = str(Dataset.getncattr(mcsirstatdata, 'MCS_eccentricity'))
    mcs_ir_basetime = mcsirstatdata.variables['mcs_basetime'][:] # time of each cloud in mcs track
    mcs_ir_datetimestring = mcsirstatdata.variables['mcs_datetimestring'][:] # time of each cloud in mcs track
    mcs_ir_length = mcsirstatdata.variables['mcs_length'][:] # duration of mcs track
    mcs_ir_meanlat = mcsirstatdata.variables['mcs_meanlat'][:] # mean latitude of the core and cold anvil of each cloud in mcs track
    mcs_ir_meanlon = mcsirstatdata.variables['mcs_meanlon'][:] # mean longitude of the core and cold anvil of each cloud in mcs track
    mcs_ir_corearea = mcsirstatdata.variables['mcs_corearea'][:] # area of each cold core in mcs track
    mcs_ir_ccsarea = mcsirstatdata.variables['mcs_ccsarea'][:] # area of each cold core + cold anvil in mcs track
    mcs_ir_cloudnumber = mcsirstatdata.variables['mcs_cloudnumber'][:] # number that corresponds to this cloud in the pixel level cloudid files
    mcs_ir_mergecloudnumber = mcsirstatdata.variables['mcs_mergecloudnumber'][:] # Cloud number of a small cloud that merges into an MCS
    mcs_ir_splitcloudnumber = mcsirstatdata.variables['mcs_splitcloudnumber'][:] # Cloud number of asmall cloud that splits from an MCS
    mcs_ir_type = mcsirstatdata.variables['mcs_type'][:] # type of mcs
    mcs_ir_status = mcsirstatdata.variables['mcs_status'][:] # flag describing how clouds in mcs track change over time
    mcs_ir_startstatus = mcsirstatdata.variables['mcs_startstatus'][:] # status of the first cloud in the mcs
    mcs_ir_endstatus = mcsirstatdata.variables['mcs_endstatus'][:] # status of the last cloud in the mcs
    mcs_ir_boundary = mcsirstatdata.variables['mcs_boundary'][:] # flag indicating whether the mcs track touches the edge of the data
    mcs_ir_interruptions = mcsirstatdata.variables['mcs_trackinterruptions'][:] # flag indicating if break exists in the track data
    mcsirstatdata.close()

    #########################################################
    # Reduce time resolution. Only keep data at the top of the hour. Need to do since NLDAS data is only hourly

    # Initialize matrices
    ntimes = mcs_ir_nmaxlength/2

    mcs_ir_length_hr = np.ones(mcs_ir_ntracks, dtype=int)*fillvalue
    mcs_ir_basetime_hr = np.ones((mcs_ir_ntracks, ntimes), dtype=int)*fillvalue
    mcs_ir_meanlat_hr = np.ones((mcs_ir_ntracks, ntimes), dtype=int)*fillvalue
    mcs_ir_meanlon_hr = np.ones((mcs_ir_ntracks, ntimes), dtype=int)*fillvalue
    mcs_ir_corearea_hr = np.ones((mcs_ir_ntracks, ntimes), dtype=int)*fillvalue
    mcs_ir_ccsarea_hr = np.ones((mcs_ir_ntracks, ntimes), dtype=int)*fillvalue
    mcs_ir_cloudnumber_hr = np.ones((mcs_ir_ntracks, ntimes), dtype=int)*fillvalue
    mcs_ir_mergecloudnumber_hr = np.ones((mcs_ir_ntracks, ntimes,  mcs_ir_nmaxmergesplits), dtype=int)*fillvalue
    mcs_ir_splitcloudnumber_hr = np.ones((mcs_ir_ntracks, ntimes,  mcs_ir_nmaxmergesplits), dtype=int)*fillvalue
    mcs_ir_type_hr = np.ones((mcs_ir_ntracks, ntimes), dtype=int)*fillvalue
    mcs_ir_status_hr = np.ones((mcs_ir_ntracks, ntimes), dtype=int)*fillvalue
    mcs_ir_startstatus_hr = np.ones((mcs_ir_ntracks, ntimes), dtype=int)*fillvalue
    mcs_ir_endstatus_hr = np.ones((mcs_ir_ntracks, ntimes), dtype=int)*fillvalue
    mcs_ir_boundary_hr = np.ones((mcs_ir_ntracks, ntimes), dtype=int)*fillvalue
    mcs_ir_interruptions_hr = np.ones((mcs_ir_ntracks, ntimes), dtype=int)*fillvalue
    mcs_ir_datetimestring_hr = [[['' for x in range(13)] for y in range(int( mcs_ir_nmaxlength))] for z in range(int(ntimes))]

    # Fill matrices
    for it in range(0, mcs_ir_ntracks):
        # Identify the top of the hour using date-time string
        hourtop = np.copy(mcs_ir_datetimestring[it,:, 11:12])
        hourlyindices = np.array(np.where(hourtop == '0'))[0, :]
        nhourlyindices = len(hourlyindices)

        # Subset the data
        mcs_ir_length_hr[it] = np.copy(nhourlyindices)
        mcs_ir_meanlat_hr[it, 0:nhourlyindices] = np.copy(mcs_ir_meanlat[it, hourlyindices])
        mcs_ir_meanlon_hr[it, 0:nhourlyindices] = np.copy(mcs_ir_meanlon[it, hourlyindices])
        mcs_ir_corearea_hr[it, 0:nhourlyindices] = np.copy(mcs_ir_corearea[it, hourlyindices])
        mcs_ir_ccsarea_hr[it, 0:nhourlyindices] = np.copy(mcs_ir_ccsarea[it, hourlyindices])
        mcs_ir_cloudnumber_hr[it, 0:nhourlyindices] = np.copy(mcs_ir_cloudnumber[it, hourlyindices])
        mcs_ir_mergecloudnumber_hr[it, 0:nhourlyindices, :] = np.copy(mcs_ir_mergecloudnumber[it, hourlyindices, :])
        mcs_ir_splitcloudnumber_hr[it, 0:nhourlyindices, :] = np.copy(mcs_ir_splitcloudnumber[it, hourlyindices, :])
        mcs_ir_status_hr[it, 0:nhourlyindices] = np.copy(mcs_ir_status[it, hourlyindices])
        mcs_ir_basetime_hr[it, 0:nhourlyindices] = np.copy(mcs_ir_basetime[it, hourlyindices])
        mcs_ir_datetimestring_hr[it][0:nhourlyindices] = np.copy(mcs_ir_datetimestring[it, hourlyindices, :])

    ###################################################################
    # Intialize precipitation statistic matrices

    # Variables for each precipitation feature
    pf_npf = np.ones((mcs_ir_ntracks, ntimes), dtype=int)*fillvalue
    pf_lon = np.ones((mcs_ir_ntracks, ntimes, nmaxpf), dtype=float)*fillvalue
    pf_lat = np.ones((mcs_ir_ntracks, ntimes, nmaxpf), dtype=float)*fillvalue
    pf_npix = np.ones((mcs_ir_ntracks, ntimes, nmaxpf), dtype=float)*fillvalue
    pf_rainrate = np.ones((mcs_ir_ntracks, ntimes, nmaxpf), dtype=float)*fillvalue
    pf_skew = np.ones((mcs_ir_ntracks, ntimes, nmaxpf), dtype=float)*fillvalue
    pf_majoraxis = np.ones((mcs_ir_ntracks, ntimes, nmaxpf), dtype=float)*fillvalue
    pf_aspectratio = np.ones((mcs_ir_ntracks, ntimes, nmaxpf), dtype=float)*fillvalue
    pf_dbz40npix = np.ones((mcs_ir_ntracks, ntimes, nmaxpf), dtype=float)*fillvalue
    pf_dbz45npix = np.ones((mcs_ir_ntracks, ntimes, nmaxpf), dtype=float)*fillvalue
    pf_dbz50npix = np.ones((mcs_ir_ntracks, ntimes, nmaxpf), dtype=float)*fillvalue

    # Variables average for the largest few precipitation features
    pf_ccrate = np.ones((mcs_ir_ntracks, ntimes), dtype=int)*fillvalue
    pf_ccnpix = np.ones((mcs_ir_ntracks, ntimes), dtype=int)*fillvalue
    pf_ccdbz10 = np.ones((mcs_ir_ntracks, ntimes), dtype=int)*fillvalue
    pf_ccdbz20 = np.ones((mcs_ir_ntracks, ntimes), dtype=int)*fillvalue
    pf_ccdbz30 = np.ones((mcs_ir_ntracks, ntimes), dtype=int)*fillvalue
    pf_ccdbz40 = np.ones((mcs_ir_ntracks, ntimes), dtype=int)*fillvalue
    pf_sfnpix = np.ones((mcs_ir_ntracks, ntimes), dtype=int)*fillvalue
    pf_sfrainrate = np.ones((mcs_ir_ntracks, ntimes), dtype=int)*fillvalue

    # Variables for each convetive core
    pf_ncores = np.ones((mcs_ir_ntracks, ntimes), dtype=int)*fillvalue
    pf_corelon = np.ones((mcs_ir_ntracks, ntimes, nmaxcore), dtype=float)*fillvalue
    pf_corelat = np.ones((mcs_ir_ntracks, ntimes, nmaxcore), dtype=float)*fillvalue
    pf_corenpix = np.ones((mcs_ir_ntracks, ntimes, nmaxcore), dtype=float)*fillvalue
    pf_coremajoraxis = np.ones((mcs_ir_ntracks, ntimes, nmaxcore), dtype=float)*fillvalue
    pf_coreaspectratio = np.ones((mcs_ir_ntracks, ntimes, nmaxcore), dtype=float)*fillvalue
    pf_coreorientation = np.ones((mcs_ir_ntracks, ntimes, nmaxcore), dtype=float)*fillvalue
    pf_coreeccentricity = np.ones((mcs_ir_ntracks, ntimes, nmaxcore), dtype=float)*fillvalue
    pf_coremaxdbz10 = np.ones((mcs_ir_ntracks, ntimes, nmaxcore), dtype=float)*fillvalue
    pf_coremaxdbz20 = np.ones((mcs_ir_ntracks, ntimes, nmaxcore), dtype=float)*fillvalue
    pf_coremaxdbz30 = np.ones((mcs_ir_ntracks, ntimes, nmaxcore), dtype=float)*fillvalue
    pf_coremaxdbz40 = np.ones((mcs_ir_ntracks, ntimes, nmaxcore), dtype=float)*fillvalue

    # Variables for accumulated rainfall
    pf_nuniqpix = np.ones(mcs_ir_ntracks, dtype=float)*fillvalue
    pf_locidx = np.ones((mcs_ir_ntracks, nmaxpix), dtype=float)*fillvalue
    pf_durtime = np.ones((mcs_ir_ntracks, nmaxpix), dtype=float)*fillvalue
    pf_durrainrate = np.ones((mcs_ir_ntracks, nmaxpix), dtype=float)*fillvalue

    ##############################################################
    # Find precipitation feature in each mcs

    # Loop over each track
    for it in range(0, mcs_ir_ntracks):
        print('Processing track ' + str(int(it)))

        # Isolate ir statistics about this track
        itlength = np.copy(mcs_ir_length_hr[it])
        itbasetime = np.copy(mcs_ir_basetime_hr[it, :])
        itdatetimestring = np.copy(mcs_ir_datetimestring_hr[it][:][:])
        itstatus = np.copy(mcs_ir_status_hr[it, :])
        itcloudnumber = np.copy(mcs_ir_cloudnumber_hr[it, :])
        itmergecloudnumber = np.copy(mcs_ir_mergecloudnumber[it, :, :])
        itsplitcloudnumber = np.copy(mcs_ir_splitcloudnumber[it, :, :])

        # Loop through each time in the track
        irindices = np.array(np.where(itbasetime > 0))[0, :]
        for itt in irindices:
            # Isolate the data at this time
            ittbasetime = np.copy(itbasetime[itt])
            ittstatus = np.copy(itstatus[itt])
            ittcloudnumber = np.copy(itcloudnumber[itt])
            ittmergecloudnumber = np.copy(itmergecloudnumber[itt, :])
            ittsplitcloudnumber = np.copy(itsplitcloudnumber[itt, :])
            ittdatetimestring = np.copy(itdatetimestring[itt])

            if ittdatetimestring[11:12] == 0:
                # Generate date file names
                ittdatetimestring = ''.join(ittdatetimestring)
                cloudid_filename = cloudidtracking_path + cloudid_filebase + ittdatetimestring + '.nc'
                pf_filename = pfdata_path + 'csa4km_' + ittdatetimestring + '00.nc'

                pf_outfile = stats_path + 'mcs_tracks_nmq_' + startdate + '_' + enddate + '.nc'

                ########################################################################
                # Load data
                if os.path.isfile(cloudid_filename) and os.path.isfile(pf_filename):
                    # Load cloudid data
                    print(cloudid_filename)
                    cloudiddata = Dataset(cloudid_filename, 'r')
                    irlat = cloudiddata.variables['latitude'][:]
                    irlon = cloudiddata.variables['longitude'][:]
                    tbmap = cloudiddata.variables['tb'][:]
                    cloudnumbermap = cloudid.variables['cloudnumber'][:]
                    cloudiddata.close()

                    # Read precipitation data
                    print(pf_filename)
                    pfdata = Dataset(pf_filename, 'r')
                    pflat = pfdata.variables['lat2d'][:]
                    pflon = pfdata.variables['lon2d'][:]
                    dbzmap = pfdata.variables['dbz_convsf'][:] # map of reflectivity 
                    dbz10map = pfdata.variables['dbz10_height'][:] # map of 10 dBZ ETHs
                    dbz20map = pfdata.variables['dbz20_height'][:] # map of 20 dBZ ETHs
                    dbz30map = pfdata.variables['dbz30_height'][:] # map of 30 dBZ ETHs
                    dbz40map = pfdata.variables['dbz40_height'][:] # map of 40 dBZ ETHs
                    dbz45map = pfdata.variables['dbz45_height'][:] # map of 45 dBZ ETH
                    dbz50map = pfdata.variables['dbz50_height'][:] # map of 50 dBZ ETHs
                    csamap = pfdata.variables['csa'][:] # map of convective, stratiform, anvil categories
                    rainratemap = pfdata.variables['rainrate'][:] # map of rain rate
                    pfnumbermap = pfdata.variables['pf_number'][:] # map of the precipitation feature number attributed to that pixel
                    dataqualitymap = pfdata.variables['mask'][:] # map if good (0) and bad (1) data
                    pfxspacing = pfdata.variables['x_spacing'][:] # distance between grid points in x direction
                    pfyspacing = pfdata.variables['y_spacing'][:] # distance between grid points in y direction
                    pfdata.close()

                    ############################################################################
                    # Find matches

                else:
                    print('One or both files do not exist: ' + cloudidfilename + ', ' + pfdata_filename)
                                    
            else:
                print('Half-hourly data ?!:' + ittdatetimestring)
