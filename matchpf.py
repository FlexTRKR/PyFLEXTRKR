# Purpose: match mergedir tracked MCS with NMQ CSA to calculate radar-based statsitics underneath the cloud features.

# Author: Original IDL code written by Zhe Feng (zhe.feng@pnnl.gov), Python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov)

def identifypf_mergedir_rainrate(mcsstats_filebase, cloudid_filebase, stats_path, cloudidtrack_path, pfdata_path, rainaccumulation_path, startdate, enddate, nmaxpf, nmaxcore, nmaxpix):
    import numpy as np
    from netCDF4 import Dataset
    import os.path
    from scipy.ndimage import label
    from skimage.measure import regionprops
    import matplotlib.pyplot as plt
    from math import pi

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

    ###################################################################
    # Intialize precipitation statistic matrices

    # Variables for each precipitation feature
    pf_npf = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength), dtype=int)*fillvalue
    pf_lon = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength, nmaxpf), dtype=float)*fillvalue
    pf_lat = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength, nmaxpf), dtype=float)*fillvalue
    pf_npix = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength, nmaxpf), dtype=float)*fillvalue
    pf_rainrate = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength, nmaxpf), dtype=float)*fillvalue
    pf_skew = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength, nmaxpf), dtype=float)*fillvalue
    pf_majoraxis = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength, nmaxpf), dtype=float)*fillvalue
    pf_aspectratio = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength, nmaxpf), dtype=float)*fillvalue
    pf_dbz40npix = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength, nmaxpf), dtype=float)*fillvalue
    pf_dbz45npix = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength, nmaxpf), dtype=float)*fillvalue
    pf_dbz50npix = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength, nmaxpf), dtype=float)*fillvalue

    # Variables average for the largest few precipitation features
    pf_ccrate = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength), dtype=int)*fillvalue
    pf_ccnpix = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength), dtype=int)*fillvalue
    pf_ccdbz10 = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength), dtype=int)*fillvalue
    pf_ccdbz20 = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength), dtype=int)*fillvalue
    pf_ccdbz30 = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength), dtype=int)*fillvalue
    pf_ccdbz40 = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength), dtype=int)*fillvalue
    pf_sfnpix = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength), dtype=int)*fillvalue
    pf_sfrainrate = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength), dtype=int)*fillvalue

    # Variables for each convective core
    pf_ncores = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength), dtype=int)*fillvalue
    pf_corelon = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength, nmaxcore), dtype=float)*fillvalue
    pf_corelat = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength, nmaxcore), dtype=float)*fillvalue
    pf_corenpix = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength, nmaxcore), dtype=float)*fillvalue
    pf_coremajoraxis = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength, nmaxcore), dtype=float)*fillvalue
    pf_coreaspectratio = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength, nmaxcore), dtype=float)*fillvalue
    pf_coreorientation = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength, nmaxcore), dtype=float)*fillvalue
    pf_coreeccentricity = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength, nmaxcore), dtype=float)*fillvalue
    pf_coremaxdbz10 = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength, nmaxcore), dtype=float)*fillvalue
    pf_coremaxdbz20 = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength, nmaxcore), dtype=float)*fillvalue
    pf_coremaxdbz30 = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength, nmaxcore), dtype=float)*fillvalue
    pf_coremaxdbz40 = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength, nmaxcore), dtype=float)*fillvalue
    pf_coreavgdbz10 = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength, nmaxcore), dtype=float)*fillvalue
    pf_coreavgdbz20 = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength, nmaxcore), dtype=float)*fillvalue
    pf_coreavgdbz30 = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength, nmaxcore), dtype=float)*fillvalue
    pf_coreavgdbz40 = np.ones((mcs_ir_ntracks, mcs_ir_nmaxlength, nmaxcore), dtype=float)*fillvalue

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
        itlength = np.copy(mcs_ir_length[it])
        itbasetime = np.copy(mcs_ir_basetime[it, :])
        itdatetimestring = np.copy(mcs_ir_datetimestring[it][:][:])
        itstatus = np.copy(mcs_ir_status[it, :])
        itcloudnumber = np.copy(mcs_ir_cloudnumber[it, :])
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

            if ittdatetimestring[11:12] == '0':
                # Generate date file names
                ittdatetimestring = ''.join(ittdatetimestring)
                cloudid_filename = cloudidtrack_path + cloudid_filebase + ittdatetimestring + '.nc'
                pf_filename = pfdata_path + 'csa4km_' + ittdatetimestring + '00.nc'
                rainaccumulation_filename = rainaccumulation_path + 'regrid_q2_' + ittdatetimestring[0:8] + '.' + ittdatetimestring[9::] + '00.nc'

                pf_outfile = stats_path + 'mcs_tracks_nmq_' + startdate + '_' + enddate + '.nc'

                ########################################################################
                # Load data

                # Load cloudid and precip feature data
                if os.path.isfile(cloudid_filename) and os.path.isfile(pf_filename):
                    # Load cloudid data
                    print(cloudid_filename)
                    cloudiddata = Dataset(cloudid_filename, 'r')
                    irlat = cloudiddata.variables['latitude'][:]
                    irlon = cloudiddata.variables['longitude'][:]
                    tbmap = cloudiddata.variables['tb'][:]
                    cloudnumbermap = cloudiddata.variables['cloudnumber'][:]
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

                    # Get dimensions of data. Data should be preprocesses so that their latittude and longitude dimensions are the same
                    ny, nx = np.shape(irlat)

                    # Load accumulation data is available. If not present fill array with fill value
                    if os.path.isfile(rainaccumulation_filename):
                        rainaccumulationdata = Dataset(rainaccumulation_filename, 'r')
                        rainaccumulationmap = rainaccumulationdata.variables['precipitation'][:]
                        rainaccumulationdata.close()
                    else:
                        rainaccumulationmap = np.ones((ny, nx), dtype=float)*fillvalue

                    ##########################################################################
                    # Get dimensions of data. Data should be preprocesses so that their latittude and longitude dimensions are the same
                    ydim, xdim = np.shape(irlat)

                    ############################################################################
                    # Find matching cloud number
                    icloudlocationt, icloudlocationy, icloudlocationx = np.array(np.where(cloudnumbermap == ittcloudnumber))
                    ncloudpix = len(icloudlocationy)

                    if ncloudpix > 0:
                        ######################################################################
                        # Check if any small clouds merge
                        idmergecloudnumber = np.array(np.where(ittmergecloudnumber > 0))[0, :]
                        nmergecloud = len(idmergecloudnumber)

                        if nmergecloud > 0:
                            # Loop over each merging cloud
                            for imc in idmergecloudnumber:
                                # Find location of the merging cloud
                                imergelocationt, imergelocationy, imergelocationx = np.array((np.where(cloudnumbermap == ittmergecloudnumber[imc])))
                                nmergepix = len(imergelocationy)

                                # Add merge pixes to mcs pixels
                                if nmergepix > 0:
                                    icloudlocationy = np.hstack((icloudlocationy, imergelocationy))
                                    icloudlocationx = np.hstack((icloudlocationx, imergelocationx))

                        ######################################################################
                        # Check if any small clouds split
                        idsplitcloudnumber = np.array(np.where(ittsplitcloudnumber > 0))[0, :]
                        nsplitcloud = len(idsplitcloudnumber)

                        if nsplitcloud > 0:
                            # Loop over each merging cloud
                            for imc in idsplitcloudnumber:
                                # Find location of the merging cloud
                                isplitlocationt, isplitlocationy, isplitlocationx = np.array((np.where(cloudnumbermap == ittsplitcloudnumber[imc])))
                                nsplitpix = len(isplitlocationy)

                                # Add split pixes to mcs pixels
                                if nsplitpix > 0:
                                    icloudlocationy = np.hstack((icloudlocationy, isplitlocationy))
                                    icloudlocationx = np.hstack((icloudlocationx, isplitlocationx))

                        #########################################################################
                        # isolate small region of cloud data around mcs at this time

                        # Set edges of boundary
                        miny = np.nanmin(icloudlocationy)
                        if miny <= 10:
                            miny = 0
                        else:
                            miny = miny - 10

                        maxy = np.nanmax(icloudlocationy)
                        if maxy >= ny - 10:
                            maxy = ny
                        else:
                            maxy = maxy + 11

                        minx = np.nanmin(icloudlocationx)
                        if minx <= 10:
                            minx = 0
                        else:
                            minx = minx - 10

                        maxx = np.nanmax(icloudlocationx)
                        if maxx >= nx - 10:
                            maxx = nx
                        else:
                            maxx = maxx + 11

                        # Isolate smaller region around cloud shield
                        subtbmap = np.copy(tbmap[0, miny:maxy, minx:maxx])
                        subcloudnumbermap = np.copy(cloudnumbermap[0, miny:maxy, minx:maxx])
                        subdbzmap = np.copy(dbzmap[0, miny:maxy, minx:maxx])
                        subdbz10map = np.copy(dbz10map[0, miny:maxy, minx:maxx])
                        subdbz20map = np.copy(dbz20map[0, miny:maxy, minx:maxx])
                        subdbz30map = np.copy(dbz30map[0, miny:maxy, minx:maxx])
                        subdbz40map = np.copy(dbz40map[0, miny:maxy, minx:maxx])
                        subdbz45map = np.copy(dbz45map[0, miny:maxy, minx:maxx])
                        subdbz50map = np.copy(dbz50map[0, miny:maxy, minx:maxx])
                        subcsamap = np.copy(csamap[0, miny:maxy, minx:maxx])
                        subrainratemap = np.copy(rainratemap[0, miny:maxy, minx:maxx])
                        subrainaccumulationmap = np.copy(rainaccumulationmap[0, miny:maxy, minx:maxx])
                        sublat = np.copy(pflat[miny:maxy, minx:maxx])
                        sublon = np.copy(pflon[miny:maxy, minx:maxx])

                        #######################################################
                        # Get dimensions of subsetted region
                        subdimy, subdimx = np.shape(subtbmap)

                        ######################################################
                        # Initialize convective map
                        ccflagmap = np.zeros((subdimy, subdimx), dtype=int)

                        #####################################################
                        # Get convective core statistics
                        icy, icx = np.array(np.where(subcsamap == 6))
                        nc = len(icy)

                        if nc  > 0 :
                            # Fill in convective map
                            ccflagmap[icy, icx] = 1

                            # Number convective regions
                            ccnumberlabelmap, ncc = label(ccflagmap)

                            # If convective cores exist calculate statistics
                            if ncc > 0:
                                plt.figure()
                                im = plt.pcolormesh(ccnumberlabelmap)
                                plt.colorbar(im)

                                # Initialize matrices
                                cclon = np.ones(ncc, dtype=float)*fillvalue
                                cclat = np.ones(ncc, dtype=float)*fillvalue
                                ccnpix = np.ones(ncc, dtype=float)*fillvalue
                                ccmajoraxis = np.ones(ncc, dtype=float)*fillvalue
                                ccaspectratio = np.ones(ncc, dtype=float)*fillvalue
                                ccorientation = np.ones(ncc, dtype=float)*fillvalue
                                cceccentricity = np.ones(ncc, dtype=float)*fillvalue
                                ccmaxdbz10 = np.ones(ncc, dtype=float)*fillvalue
                                ccmaxdbz20 = np.ones(ncc, dtype=float)*fillvalue
                                ccmaxdbz30 = np.ones(ncc, dtype=float)*fillvalue
                                ccmaxdbz40 = np.ones(ncc, dtype=float)*fillvalue
                                ccavgdbz10 = np.ones(ncc, dtype=float)*fillvalue
                                ccavgdbz20 = np.ones(ncc, dtype=float)*fillvalue
                                ccavgdbz30 = np.ones(ncc, dtype=float)*fillvalue
                                ccavgdbz40 = np.ones(ncc, dtype=float)*fillvalue

                                # Loop over each core
                                for cc in range(1, ncc+1):
                                    # Isolate core
                                    iiccy, iiccx = np.array(np.where(ccnumberlabelmap == cc))

                                    # Number of pixels in the core
                                    ccnpix[cc-1] = len(iiccy)

                                    # Get mean latitude and longitude
                                    cclon[cc-1] = np.nanmean(sublon[iiccy, iiccx])
                                    cclat[cc-1] = np.nanmean(sublat[iiccy, iiccx])

                                    # Get echo top height statistics
                                    ccmaxdbz10[cc-1] = np.nanmax(subdbz10map[iiccy, iiccx])
                                    ccmaxdbz10[cc-1] = np.nanmax(subdbz10map[iiccy, iiccx])
                                    ccmaxdbz20[cc-1] = np.nanmax(subdbz20map[iiccy, iiccx])
                                    ccmaxdbz30[cc-1] = np.nanmax(subdbz30map[iiccy, iiccx])
                                    ccmaxdbz40[cc-1] = np.nanmax(subdbz40map[iiccy, iiccx])
                                    ccavgdbz10[cc-1] = np.nanmean(subdbz10map[iiccy, iiccx])
                                    ccavgdbz20[cc-1] = np.nanmean(subdbz20map[iiccy, iiccx])
                                    ccavgdbz30[cc-1] = np.nanmean(subdbz30map[iiccy, iiccx])
                                    ccavgdbz40[cc-1] = np.nanmean(subdbz40map[iiccy, iiccx])

                                    # Generate map of convective core
                                    iiccflagmap = np.zeros((subdimy, subdimx), dtype=int)
                                    iiccflagmap[iiccy, iiccx] = 1

                                    # Get core geometric statistics
                                    coreproperties = regionprops(iiccflagmap)
                                    cceccentricity[cc-1] = coreproperties[0].eccentricity
                                    ccmajoraxis[cc-1] = coreproperties[0].major_axis_length
                                    ccorientation[cc-1] = (coreproperties[0].orientation)*(180/float(pi))
                                    ccminoraxis = coreproperties[0].minor_axis_length
                                    ccaspectratio[cc-1] = np.divide(ccmajoraxis[cc-1], ccminoraxis)

                                print(ccnpix)
                                print(cclon)
                                print(cclat)
                                print(ccmaxdbz10)
                                print(ccavgdbz10)
                                print(cceccentricity)

                                ####################################################
                                # Sort based on size, largest to smallest
                                order = np.argsort(ccnpix)
                                order = order[::-1]

                                scclon = np.copy(cclon[order])
                                scclat = np.copy(cclat[order])
                                sccnpix = np.copy(ccnpix[order])
                                sccmajoraxis = np.copy(ccmajoraxis[order])
                                sccaspectratio = np.copy(ccaspectratio[order])
                                sccorientation = np.copy(ccorientation[order])
                                scceccentricity = np.copy(cceccentricity[order])
                                sccmaxdbz10 = np.copy(ccmaxdbz10[order])
                                sccmaxdbz20 = np.copy(ccmaxdbz20[order])
                                sccmaxdbz30 = np.copy(ccmaxdbz30[order])
                                sccmaxdbz40 = np.copy(ccmaxdbz40[order])
                                sccavgdbz10 = np.copy(ccavgdbz10[order])
                                sccavgdbz20 = np.copy(ccavgdbz20[order])
                                sccavgdbz30 = np.copy(ccavgdbz30[order])
                                sccavgdbz40 = np.copy(ccavgdbz40[order])

                                print(sccnpix)
                                print(scclon)
                                print(scclat)
                                print(sccmaxdbz10)
                                print(sccavgdbz10)
                                print(scceccentricity)
                                plt.show()

                                ###################################################
                                # Save convective core statisitcs
                                pf_ncores[it, itt] = np.copy(ncc)

                                ncore_save = np.nanmin([nmaxcore, ncc])
                                pf_corelon[it, itt, 0:ncore_save-1] = np.copy(scclon[0:ncore_save-1])
                                pf_corelat[it, itt, 0:ncore_save-1] = np.copy(scclat[0:ncore_save-1])
                                pf_corenpix[it, itt, 0:ncore_save-1] = np.copy(sccnpix[0:ncore_save-1])
                                pf_coremajoraxis[it, itt, 0:ncore_save-1] = np.copy(sccmajoraxis[0:ncore_save-1])
                                pf_coreaspectratio[it, itt, 0:ncore_save-1] = np.copy(sccmajoraxis[0:ncore_save-1])
                                pf_coreorientation[it, itt, 0:ncore_save-1] = np.copy(sccorientation[0:ncore_save-1])
                                pf_coreeccentricity[it, itt, 0:ncore_save-1] = np.copy(scceccentricity[0:ncore_save-1])
                                pf_coremaxdbz10[it, itt, 0:ncore_save-1] = np.copy(sccmaxdbz10[0:ncore_save-1])
                                pf_coremaxdbz20[it, itt, 0:ncore_save-1] = np.copy(sccmaxdbz20[0:ncore_save-1])
                                pf_coremaxdbz30[it, itt, 0:ncore_save-1] = np.copy(sccmaxdbz30[0:ncore_save-1])
                                pf_coremaxdbz40[it, itt, 0:ncore_save-1] = np.copy(sccmaxdbz40[0:ncore_save-1])
                                pf_coreavgdbz10[it, itt, 0:ncore_save-1] = np.copy(sccavgdbz10[0:ncore_save-1])
                                pf_coreavgdbz20[it, itt, 0:ncore_save-1] = np.copy(sccavgdbz20[0:ncore_save-1])
                                pf_coreavgdbz30[it, itt, 0:ncore_save-1] = np.copy(sccavgdbz30[0:ncore_save-1])
                                pf_coreavgdbz40[it, itt, 0:ncore_save-1] = np.copy(sccavgdbz40[0:ncore_save-1])



                else:
                    print('One or both files do not exist: ' + cloudidfilename + ', ' + pfdata_filename)
                                    
            else:
                print(ittdatetimestring)
                print('Half-hourly data ?!:' + str(ittdatetimestring))
                raw_input('waiting')
