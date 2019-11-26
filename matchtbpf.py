# Purpose: match mergedir tracked MCS with WRF rainrate statistics underneath the cloud features.

# Author: Original IDL code written by Zhe Feng (zhe.feng@pnnl.gov), Python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov), Python WRF version modified from original IDL and python versions by Katelyn Barber (katelyn.barber@pnnl.gov)

def identifypf_wrf_rain(mcsstats_filebase, cloudid_filebase, rainaccumulation_filebase, stats_path,cloudidtrack_path,rainaccumulation_path, startdate, enddate, geolimits, nmaxpf, nmaxcore, nmaxclouds, rr_min, pixel_radius, irdatasource, precipdatasource, datadescription, datatimeresolution, mcs_irareathresh, mcs_irdurationthresh, mcs_ireccentricitythresh,pf_link_area_thresh):

    # Input:
    # mcsstats_filebase - file header of the mcs statistics file that has the satellite data and was produced in the previous step
    # cloudid_filebase - file header of the cloudid file created in the idclouds step
    # pfdata_filebase - file header of the radar data
    # rainaccumulation_filebase - file header of the rain accumulation data
    # stats_path - directory which stores this statistics data. this is where the output from this code will be placed.
    # cloudidtrack_path - directory that contains the cloudid data created in the idclouds step
    # pfdata_path - directory that contains the radar data
    # rainaccumulation_path - directory containing the rain accumulation data
    # startdate - starting date and time of the data
    # enddate - ending date and time of the data
    # geolimits - 4-element array with plotting boundaries [lat_min, lon_min, lat_max, lon_max]
    # namxpf - maximum number of precipitation features that can exist within one satellite defined MCS at a given time
    # nmaxcore - maximum number of convective cores that can exist within one satellite defined MCS at a given time
    # nmaxclouds - maximum number of clouds allowed to be within one track
    # rr_min - minimum rain rate used when classifying precipitation features
    # pixel_radius - radius of pixels in km
    # irdatasource - source of the raw satellite data
    # nmqdatasource - source of the radar data
    # datadescription - description of the satellite data source
    # datatimeresolution - time resolution of the satellite data
    # mcs_irareathresh - satellite area threshold for MCS identificaiton
    # mcs_irdurationthresh - satellite duration threshold for MCS identification
    # mcs_ireccentricitythresh - satellite eccentricity threshold used for classifying squall lines

    # Output: (One netcdf with statistics about the satellite, radar, and rain accumulation characteristics for each satellite defined MCS)
    # mcs_length - duration of MCS portion of each track
    # length - total duration of each track (MCS and nonMCS components)
    # mcs_type - flag indicating whether this is squall line, based on satellite definition
    # status - flag indicating the evolution of each cloud in a MCS
    # startstatus - flag indicating how a MCS starts
    # endstatus - flag indicating how a MCS ends
    # trackinterruptions - flag indicating if the data used to identify this MCS is incomplete
    # boundary - flag indicating if a MCS touches the edge of the domain
    # basetime - seconds since 1970-01-01 for each cloud in a MCS
    # datetimestring - string of date and time of each cloud in a MCS
    # meanlat - mean latitude of the MCS
    # meanlon - mean longitude of the MCS
    # core_area - area of the core of MCS
    # ccs_area - area of the core and cold anvil of the MCS
    # cloudnumber - numbers indicating which clouds in the cloudid files are associated with a MCS
    # mergecloudnumber - numbers indicating which clouds in the cloudid files merge into this track
    # splitcloudnumber - numbers indicating which clouds in the cloudid files split from this track
    # nmq_frac - fraction of the cloud that exists within the radar domain
    # npf - number of precipitation features at reach time
    # pf_area - area of the precipitation features in a MCS
    # pf_lon - mean longitudes of the precipitaiton features in a MCS
    # pf_lat - mean latitudes of the precipitation features in a MCS
    # pf_rainrate - mean rainrates of the precipition features in a MCS
    # pf_skewness - skewness of the rainrains in precipitation features in a MCS
    # pf_majoraxislength - major axis lengths of the precipitation features in a MCS
    # pf_minoraxislength - minor axis lengths of the precipitation features in a MCS
    # pf_aspectratio - aspect ratios of the precipitation features in a MCS
    # pf_eccentricity - eccentricity of the precipitation features in a MCS
    # pf_orientation - angular position of the precipitation deatures in a MCS
    
    import numpy as np
    import os.path
    from netCDF4 import Dataset, num2date, chartostring
    from scipy.ndimage import label, binary_dilation, generate_binary_structure
    from skimage.measure import regionprops
    from math import pi
    from scipy.stats import skew
    import xarray as xr
    import time
    import pandas as pd
    import time, datetime, calendar
    np.set_printoptions(threshold=np.inf)

    #########################################################
    # Load MCS track stats
    print('Loading IR data')
    print((time.ctime()))
    mcsirstatistics_file = stats_path + mcsstats_filebase + startdate + '_' + enddate + '.nc'

    mcsirstatdata = Dataset(mcsirstatistics_file, 'r')
    ir_ntracks = np.nanmax(mcsirstatdata['ntracks']) + 1
    ir_nmaxlength = np.nanmax(mcsirstatdata['ntimes']) + 1
    ir_basetime = mcsirstatdata['mcs_basetime'][:]
    #print(ir_basetime)
    basetime_units =  mcsirstatdata['mcs_basetime'].units
    #print(basetime_units)
    basetime_calendar = mcsirstatdata['mcs_basetime'].calendar
    #print(basetime_calendar)
    ir_datetimestring = mcsirstatdata['mcs_datetimestring'][:]
    ir_cloudnumber = mcsirstatdata['mcs_cloudnumber'][:]
    ir_mergecloudnumber = mcsirstatdata['mcs_mergecloudnumber'][:]
    ir_splitcloudnumber = mcsirstatdata['mcs_splitcloudnumber'][:]
    ir_mcslength = mcsirstatdata['mcs_length'][:]
    ir_tracklength = mcsirstatdata['track_length'][:]
    ir_mcstype = mcsirstatdata['mcs_type'][:]
    ir_status = mcsirstatdata['mcs_status'][:]
    ir_startstatus = mcsirstatdata['mcs_startstatus'][:]
    ir_endstatus = mcsirstatdata['mcs_endstatus'][:]
    ir_trackinterruptions = mcsirstatdata['mcs_trackinterruptions'][:]
    ir_boundary = mcsirstatdata['mcs_boundary'][:]
    ir_meanlat = np.array(mcsirstatdata['mcs_meanlat'][:])
    ir_meanlon = np.array(mcsirstatdata['mcs_meanlon'][:])
    ir_corearea = mcsirstatdata['mcs_corearea'][:]
    ir_ccsarea = mcsirstatdata['mcs_ccsarea'][:]
    mcsirstatdata.close()
    
    #ir_datetimestring = ir_datetimestring[:, :, :, 0]
    ir_datetimestring = ir_datetimestring[:, :, :]
    
    # Change time to minutes if data time resolution is less
    # than one hr or else attributes will be 0 in output
    # file
    if datatimeresolution < 1:     # less than 1 hr
        datatimeresolution = datatimeresolution*60    # time in minutes

    pfn_lmap = []
    ###################################################################
    # Intialize precipitation statistic matrices
    print('Initializing matrices')
    print((time.ctime()))

    # Variables for each precipitation feature
    pf_npf = np.ones((ir_ntracks, ir_nmaxlength), dtype=int)*-9999
    npf_avgnpix = np.ones((ir_ntracks, ir_nmaxlength), dtype=int)*-9999
    npf_avgrainrate = np.ones((ir_ntracks, ir_nmaxlength), dtype=int)*-9999
    pf_pflon = np.ones((ir_ntracks, ir_nmaxlength, nmaxpf), dtype=float)*np.nan
    pf_pflat = np.ones((ir_ntracks, ir_nmaxlength, nmaxpf), dtype=float)*np.nan
    pf_pfnpix = np.ones((ir_ntracks, ir_nmaxlength, nmaxpf), dtype=int)*-9999
    pf_pfrainrate = np.ones((ir_ntracks, ir_nmaxlength, nmaxpf), dtype=float)*np.nan
    pf_pfskewness = np.ones((ir_ntracks, ir_nmaxlength, nmaxpf), dtype=float)*np.nan
    pf_pfmajoraxis = np.ones((ir_ntracks, ir_nmaxlength, nmaxpf), dtype=float)*np.nan
    pf_pfminoraxis = np.ones((ir_ntracks, ir_nmaxlength, nmaxpf), dtype=float)*np.nan
    pf_pfaspectratio = np.ones((ir_ntracks, ir_nmaxlength, nmaxpf), dtype=float)*np.nan
    pf_pforientation = np.ones((ir_ntracks, ir_nmaxlength, nmaxpf), dtype=float)*np.nan
    pf_pfeccentricity = np.ones((ir_ntracks, ir_nmaxlength, nmaxpf), dtype=float)*np.nan
    pf_pfycentroid = np.ones((ir_ntracks, ir_nmaxlength, nmaxpf), dtype=float)*np.nan
    pf_pfxcentroid = np.ones((ir_ntracks, ir_nmaxlength, nmaxpf), dtype=float)*np.nan
    pf_pfyweightedcentroid = np.ones((ir_ntracks, ir_nmaxlength, nmaxpf), dtype=float)*np.nan
    pf_pfxweightedcentroid = np.ones((ir_ntracks, ir_nmaxlength, nmaxpf), dtype=float)*np.nan
    #pf_pfrr8mm = np.zeros((ir_ntracks, ir_nmaxlength, nmaxpf), dtype=np.long)
    #pf_pfrr10mm = np.zeros((ir_ntracks, ir_nmaxlength, nmaxpf), dtype=np.long)
    basetime = np.empty((ir_ntracks, ir_nmaxlength), dtype='datetime64[s]')
    precip_basetime = np.empty((ir_ntracks, ir_nmaxlength))

    pf_frac = np.ones((ir_ntracks, ir_nmaxlength), dtype=float)*np.nan

    ##############################################################
    # Find precipitation feature in each mcs
    print(('Total Number of Tracks:' + str(ir_ntracks)))

    # Loop over each track
    print('Looping over each track')
    print((time.ctime()))
    print(ir_ntracks)
    for it in range(0, ir_ntracks):
        print(('Processing track ' + str(int(it))))
        print((time.ctime()))

        # Isolate ir statistics about this track
        itbasetime = np.copy(ir_basetime[it, :])
        itdatetimestring = np.copy(ir_datetimestring[it][:][:])
        itcloudnumber = np.copy(ir_cloudnumber[it, :])
        itmergecloudnumber = np.copy(ir_mergecloudnumber[it, :, :])
        itsplitcloudnumber = np.copy(ir_splitcloudnumber[it, :, :])

        statistics_outfile = stats_path + 'mcs_tracks_' + precipdatasource + '_' + startdate + '_' + enddate + '.nc'
        # Loop through each time in the track
        irindices = np.array(np.where(itcloudnumber > 0))[0, :]
        print('Looping through each time step')
        print(('Number of time steps: ' + str(len(irindices))))
        for itt in irindices:
            print(('Time step #: ' + str(itt)))
            # Isolate the data at this time
            basetime = np.array([pd.to_datetime(num2date(itbasetime[itt], units=basetime_units, calendar=basetime_calendar))], dtype='datetime64[s]')[0]
            precip_basetime[it,itt] = itbasetime[itt]

            ittcloudnumber = np.copy(itcloudnumber[itt])
            ittmergecloudnumber = np.copy(itmergecloudnumber[itt, :])
            ittsplitcloudnumber = np.copy(itsplitcloudnumber[itt, :])
            ittdatetimestring = np.copy(itdatetimestring[itt])
            #ittdatetimestring = str(chartostring(ittdatetimestring[:,0]))
            ittdatetimestring = ''.join(ittdatetimestring)
            
            # Generate date file names
            cloudid_filename = cloudidtrack_path + cloudid_filebase + ittdatetimestring + '.nc'
            rainaccumulation_filename = rainaccumulation_path + rainaccumulation_filebase + '_' + ittdatetimestring[0:4] + '-' + ittdatetimestring[4:6] + '-' + ittdatetimestring[6:8] + '_' + ittdatetimestring[9:11] + ':' + ittdatetimestring[11:13] + ':00.nc'

            ########################################################################
            # Load data

            # Load cloudid and precip feature data
            if os.path.isfile(cloudid_filename) and os.path.isfile(rainaccumulation_filename):
                count = 0
                print('Data Present')
                # Load cloudid data
                print('Loading cloudid data')
                print(cloudid_filename)
                cloudiddata = Dataset(cloudid_filename, 'r')
                cloudnumbermap = cloudiddata['cloudnumber'][:]
                cloudtype = cloudiddata['cloudtype'][:]
                cloudiddata.close()

                # Read precipitation data
                print('Loading precip data')
                print(rainaccumulation_filename)
                pfdata = Dataset(rainaccumulation_filename, 'r')
                rawrainratemap = pfdata['rainrate'][:] # map of rain rate
                lon = pfdata['lon2d'][:]
                lat = pfdata['lat2d'][:]
                pfdata.close()

                ##########################################################################
                # Get dimensions of data. Data should be preprocesses so that their latittude and longitude dimensions are the same
                ydim, xdim = np.shape(lat)

                #########################################################################
                # Intialize matrices for only MCS data
                filteredrainratemap = np.ones((ydim, xdim), dtype=float)*np.nan
                print('filteredrainratemap allocation size: ', filteredrainratemap.shape)

                ############################################################################
                # Find matching cloud number
                icloudlocationt, icloudlocationy, icloudlocationx = np.array(np.where(cloudnumbermap == ittcloudnumber))
                ncloudpix = len(icloudlocationy)

                if ncloudpix > 0:
                    print('IR Clouds Present')
                    ######################################################################
                    # Check if any small clouds merge
                    print('Finding mergers')
                    idmergecloudnumber = np.array(np.where(ittmergecloudnumber > 0))[0, :]
                    nmergecloud = len(idmergecloudnumber)

                    if nmergecloud > 0:
                        # Loop over each merging cloud
                        for imc in idmergecloudnumber:
                            # Find location of the merging cloud
                            imergelocationt, imergelocationy, imergelocationx = np.array(np.where(cloudnumbermap == ittmergecloudnumber[imc]))
                            nmergepix = len(imergelocationy)

                            # Add merge pixes to mcs pixels
                            if nmergepix > 0:
                                icloudlocationt = np.hstack((icloudlocationt, imergelocationt))
                                icloudlocationy = np.hstack((icloudlocationy, imergelocationy))
                                icloudlocationx = np.hstack((icloudlocationx, imergelocationx))

                    ######################################################################
                    # Check if any small clouds split
                    print('Finding splits')
                    idsplitcloudnumber = np.array(np.where(ittsplitcloudnumber > 0))[0, :]
                    nsplitcloud = len(idsplitcloudnumber)

                    if nsplitcloud > 0:
                        # Loop over each merging cloud
                        for imc in idsplitcloudnumber:
                            # Find location of the merging cloud
                            isplitlocationt, isplitlocationy, isplitlocationx = np.array(np.where(cloudnumbermap == ittsplitcloudnumber[imc]))
                            nsplitpix = len(isplitlocationy)

                            # Add split pixels to mcs pixels
                            if nsplitpix > 0:
                                icloudlocationt = np.hstack((icloudlocationt, isplitlocationt))
                                icloudlocationy = np.hstack((icloudlocationy, isplitlocationy))
                                icloudlocationx = np.hstack((icloudlocationx, isplitlocationx))

                    ########################################################################
                    # Fill matrices with mcs data
                    print('Fill map with data')
                    filteredrainratemap[icloudlocationy, icloudlocationx] = np.copy(rawrainratemap[icloudlocationt,icloudlocationy,icloudlocationx])

                    ########################################################################
                    ## Isolate small region of cloud data around mcs at this time
                    #print('Calculate new shape statistics')

                    ## Set edges of boundary
                    #miny = np.nanmin(icloudlocationy)
                    #if miny <= 10:
                    #    miny = 0
                    #else:
                        #miny = miny - 10

                    #maxy = np.nanmax(icloudlocationy)
                    #if maxy >= ydim - 10:
                        #maxy = ydim
                    #else:
                        #maxy = maxy + 11

                    #minx = np.nanmin(icloudlocationx)
                    #if minx <= 10:
                        #minx = 0
                    #else:
                        #minx = minx - 10

                    #maxx = np.nanmax(icloudlocationx)
                    #if maxx >= xdim - 10:
                        #maxx = xdim
                    #else:
                        #maxx = maxx + 11

                    ## Isolate smaller region around cloud shield
                    #subrainratemap = np.copy(filteredrainratemap[miny:maxy, minx:maxx])
 
                    #########################################################
                    ## Get dimensions of subsetted region
                    #subdimy, subdimx = np.shape(subrainratemap)

                    ######################################################   !!!!!!!!!!!!!!! Slow Step !!!!!!!!1
                    # Derive precipitation feature statistics
                    print('Calculating precipitation statistics')
                    rawrainratemap = np.squeeze(rawrainratemap, axis = 0)
                    ipfy, ipfx = np.array(np.where(rawrainratemap > rr_min))
                    nrainpix = len(ipfy)

                    if nrainpix > 0:
                        ####################################################
                        # Dilate precipitation feature by one pixel. This slightly smooths the data so that very close precipitation features are connected
                        # Create binary map
                        binarypfmap = np.zeros((ydim, xdim), dtype=int)
                        binarypfmap[ipfy, ipfx] = 1

                        # Dilate (aka smooth)
                        dilationstructure = generate_binary_structure(2,1)  # Defines shape of growth. This grows one pixel as a cross

                        dilatedbinarypfmap = binary_dilation(binarypfmap, structure=dilationstructure, iterations=1).astype(filteredrainratemap.dtype)

                        # Label precipitation features
                        pfnumberlabelmap, numpf = label(dilatedbinarypfmap)
                                                    
                        # Sort numpf then calculate stats
                        min_npix = np.ceil(pf_link_area_thresh / (pixel_radius**2))
                            
                        # Sort and renumber PFs, and remove small PFs
                        from ftfunctions import sort_renumber
                        pf_number, pf_npix = sort_renumber(pfnumberlabelmap, min_npix)
                        # Update number of PFs after sorting and renumbering
                        npf_new = np.nanmax(pf_number)

                        if npf_new > 0:
                            print('PFs present, initializing matrices')

                           ##############################################
                           # Initialize matrices
                            pfnpix = np.zeros(numpf, dtype=float)
                            test = np.ones(numpf, dtype=float)*np.nan
                            pfid = np.ones(numpf, dtype=int)*-9999
                            pflon = np.ones(numpf, dtype=float)*np.nan
                            pflat = np.ones(numpf, dtype=float)*np.nan
                            pfrainrate = np.ones(numpf, dtype=float)*np.nan
                            pfskewness = np.ones(numpf, dtype=float)*np.nan
                            pfmajoraxis = np.ones(numpf, dtype=float)*np.nan
                            pfminoraxis = np.ones(numpf, dtype=float)*np.nan
                            pfaspectratio = np.ones(numpf, dtype=float)*np.nan
                            #pf8mm = np.zeros(numpf, dtype=np.long)
                            #pf10mm = np.zeros(numpf, dtype=np.long)
                            pfxcentroid = np.ones(numpf, dtype=float)*np.nan
                            pfycentroid = np.ones(numpf, dtype=float)*np.nan
                            pfxweightedcentroid = np.ones(numpf, dtype=float)*np.nan
                            pfyweightedcentroid = np.ones(numpf, dtype=float)*np.nan
                            pfeccentricity = np.ones(numpf, dtype=float)*np.nan
                            pfperimeter = np.ones(numpf, dtype=float)*np.nan
                            pforientation = np.ones(numpf, dtype=float)*np.nan

                            print('Looping through each feature to calculate statistics')
                            print(('Number of PFs ' + str(numpf)))
                            ###############################################
                            # Loop through each feature
                            for ipf in range(1, npf_new+1):

                                #######################################
                                # Find associated indices
                                iipfy, iipfx = np.array(np.where(pfnumberlabelmap == ipf))
                                niipfpix = len(iipfy)

                                if niipfpix > 0:
                                    ##########################################
                                    # Compute statistics

                                    # Basic statistics
                                    pfnpix[ipf-1] = np.copy(niipfpix)
                                    pfid[ipf-1] = np.copy(int(ipf))
                                    pfrainrate[ipf-1] = np.nanmean(filteredrainratemap[iipfy[:], iipfx[:]])
                                    pfskewness[ipf-1] = skew(filteredrainratemap[iipfy[:], iipfx[:]])
                                    pflon[ipf-1] = np.nanmean(lat[iipfy[:], iipfx[:]])
                                    pflat[ipf-1] = np.nanmean(lon[iipfy[:], iipfx[:]])
                                    
                                    ## Count number of pixel rainrate greater than specified threshold
                                    #print('rawrainratemap_size: ', rawrainratemap.shape)
                                    #iirry, iirrx = np.array(np.where(rawrainratemap > 8))
                                    #print('size of iirry:', iirry.shape)
                                    #if iirry > 0:
                                        #pf8mm[ipf-1] = iirry
                                    
                                    #iirry10 = np.array(np.where(rawrainratemap > 10))
                                    #if iirry10 > 0:
                                        #pf10mm[ipf-1] = iirry

                                    # Generate map of convective core
                                    iipfflagmap = np.zeros((ydim, xdim), dtype=int)
                                    iipfflagmap[iipfy, iipfx] = 1

                                    # Geometric statistics
                                    tfilteredrainratemap = np.copy(filteredrainratemap)
                                    tfilteredrainratemap[np.isnan(tfilteredrainratemap)] = -9999
                                    pfproperties = regionprops(iipfflagmap, intensity_image=tfilteredrainratemap)
                                    pfeccentricity[ipf-1] = pfproperties[0].eccentricity
                                    pfmajoraxis[ipf-1] = pfproperties[0].major_axis_length*pixel_radius
                                    # Need to treat minor axis length with an error except since the python algorthim occsionally throws an error. 
                                    try:
                                        pfminoraxis[ipf-1] = pfproperties[0].minor_axis_length*pixel_radius
                                    except ValueError:
                                        pass
                                    if ~np.isnan(pfminoraxis[ipf-1]) or ~np.isnan(pfmajoraxis[ipf-1]):
                                        pfaspectratio[ipf-1] = np.divide(pfmajoraxis[ipf-1], pfminoraxis[ipf-1])
                                    pforientation[ipf-1] = (pfproperties[0].orientation)*(180/float(pi))
                                    pfperimeter[ipf-1] = pfproperties[0].perimeter*pixel_radius
                                    [pfycentroid[ipf-1], pfxcentroid[ipf-1]] = pfproperties[0].centroid
                                    [pfyweightedcentroid[ipf-1], pfxweightedcentroid[ipf-1]] = pfproperties[0].weighted_centroid
                                    
                            print('Loop done')
                            ##############################################################
                            # Sort precipitation features by size, large to small
                            print('Sorting PFs by size')
                            pforder = np.argsort(pfnpix) # returns the indices that sorts the array
                            pforder = pforder[::-1] # flips the order to largest to smallest

                            spfnpix = pfnpix[pforder]
                            spfid = pfid[pforder]
                            spfrainrate = pfrainrate[pforder]
                            spfskewness = pfskewness[pforder]
                            spflon = pflon[pforder]
                            spflat = pflat[pforder]
                            spfeccentricity = pfeccentricity[pforder]
                            spfmajoraxis = pfmajoraxis[pforder]
                            spfminoraxis = pfminoraxis[pforder]
                            spfaspectratio = pfaspectratio[pforder]
                            spforientation = pforientation[pforder]
                            spfycentroid = pfycentroid[pforder]
                            spfxcentroid = pfxcentroid[pforder]
                            spfxweightedcentroid = pfxweightedcentroid[pforder]
                            spfyweightedcentroid = pfyweightedcentroid[pforder]
                            #spf8mm = pf8mm[pforder]
                            #spf10mm = pf10mm[pforder]

                            ###################################################
                            # Save precipitation feature statisitcs
                            pf_npf[it, itt] = np.copy(numpf)

                            npf_save = np.nanmin([nmaxpf, numpf])
                            pf_pflon[it, itt, 0:npf_save]= spflon[0:npf_save]
                            pf_pflat[it, itt, 0:npf_save] = spflat[0:npf_save]
                            pf_pfnpix[it, itt, 0:npf_save] = spfnpix[0:npf_save]
                            pf_pfrainrate[it, itt, 0:npf_save] = spfrainrate[0:npf_save]
                            pf_pfskewness[it, itt, 0:npf_save] = spfskewness[0:npf_save]
                            pf_pfmajoraxis[it, itt, 0:npf_save] = spfmajoraxis[0:npf_save]
                            pf_pfminoraxis[it, itt, 0:npf_save] = spfminoraxis[0:npf_save]
                            pf_pfaspectratio[it, itt, 0:npf_save] = spfaspectratio[0:npf_save]
                            pf_pforientation[it, itt, 0:npf_save] = spforientation[0:npf_save]
                            pf_pfeccentricity[it, itt, 0:npf_save] = spfeccentricity[0:npf_save]
                            pf_pfycentroid[it, itt, 0:npf_save] = spfycentroid[0:npf_save]
                            pf_pfxcentroid[it, itt, 0:npf_save] = spfxcentroid[0:npf_save]
                            pf_pfxweightedcentroid[it, itt, 0:npf_save] = spfxweightedcentroid[0:npf_save]
                            pf_pfyweightedcentroid[it, itt, 0:npf_save] = spfyweightedcentroid[0:npf_save]
                            #pf_pfrr8mm[it, itt, 0:npf_save] = spf8mm[0:npf_save]
                            #pf_pfrr10mm[it, itt, 0:npf_save] = spf10mm[0:npf_save]
                        
                else:
                    print(('One or both files do not exist: ' + cloudid_filename + ', ' + rainaccumulation_filename))
                            
            else:
                print(ittdatetimestring)
                print(('Half-hourly data ?!:' + str(ittdatetimestring)))

    ###################################
    # Convert number of pixels to area
    print('Converting pixels to area')
    print((time.ctime()))
    
    pf_pfarea = np.multiply(pf_pfnpix, np.square(pixel_radius))
    
    ##################################
    # Save output to netCDF file
    print('Saving data')
    print((time.ctime()))
    
    # Check if file already exists. If exists, delete
    if os.path.isfile(statistics_outfile):
        os.remove(statistics_outfile)
    
    #import pdb; pdb.set_trace()
    # Definte xarray dataset
    output_data = xr.Dataset({'mcs_length':(['track'], np.squeeze(ir_mcslength)), \
                              'length': (['track'], ir_tracklength), \
                              'mcs_type': (['track'], ir_mcstype), \
                              'status': (['track', 'time'], ir_status), \
                              'startstatus': (['track'], ir_startstatus), \
                              'endstatus': (['track'], ir_endstatus), \
                              'interruptions': (['track'], ir_trackinterruptions), \
                              'boundary': (['track'], ir_boundary), \
                              'basetime': (['track', 'time'], precip_basetime), \
                              'datetimestring': (['track', 'time', 'characters'], ir_datetimestring), \
                              'meanlat': (['track', 'time'], ir_meanlat), \
                              'meanlon': (['track', 'time'], ir_meanlon), \
                              'core_area': (['track', 'time'], ir_corearea), \
                              'ccs_area': (['track', 'time'],  ir_ccsarea), \
                              'cloudnumber': (['track', 'time'],  ir_cloudnumber), \
                              'mergecloudnumber': (['track', 'time', 'mergesplit'], ir_mergecloudnumber), \
                              'splitcloudnumber': (['track', 'time', 'mergesplit'], ir_splitcloudnumber), \
                              'pf_frac': (['track', 'time'], pf_frac), \
                              'pf_npf': (['track', 'time'], pf_npf), \
                              'pf_area': (['track', 'time', 'pfs'], pf_pfarea), \
                              'pf_lon': (['track', 'time', 'pfs'], pf_pflon), \
                              'pf_lat': (['track', 'time', 'pfs'], pf_pflat), \
                              'pf_rainrate': (['track', 'time', 'pfs'], pf_pfrainrate), \
                              'pf_skewness': (['track', 'time', 'pfs'], pf_pfskewness), \
                              'pf_majoraxislength': (['track', 'time', 'pfs'], pf_pfmajoraxis), \
                              'pf_minoraxislength': (['track', 'time', 'pfs'], pf_pfminoraxis), \
                              'pf_aspectratio': (['track', 'time', 'pfs'], pf_pfaspectratio), \
                              'pf_eccentricity': (['track', 'time', 'pfs'], pf_pfeccentricity), \
                              'pf_orientation': (['track', 'time', 'pfs'], pf_pforientation)}, \
                             coords={'track': (['track'], np.arange(1, ir_ntracks+1)), \
                                     'time': (['time'], np.arange(0, ir_nmaxlength)), \
                                     'pfs':(['pfs'], np.arange(0, nmaxpf)), \
                                     'cores': (['cores'], np.arange(0, nmaxcore)), \
                                     'mergesplit': (['mergesplit'], np.arange(0, np.shape(ir_mergecloudnumber)[2])), \
                                     'characters': (['characters'], np.ones(13)*-9999)}, \
                             attrs={'title':'File containing ir and precpitation statistics for each track', \
                                    'source1':irdatasource, \
                                    'source2':precipdatasource, \
                                    'description':datadescription, \
                                    'startdate':startdate, \
                                    'enddate':enddate, \
                                    'time_resolution_hour_or_minutes':str(int(datatimeresolution)), \
                                    'mergdir_pixel_radius':pixel_radius, \
                                    'MCS_IR_area_thresh_km2':str(int(mcs_irareathresh)), \
                                    'MCS_IR_duration_thresh_hr':str(int(mcs_irdurationthresh)), \
                                    'MCS_IR_eccentricity_thres':str(float(mcs_ireccentricitythresh)), \
                                    'max_number_pfs':str(int(nmaxpf)), \
                                    'contact':'Katelyn Barber: katelyn.barber@pnnl.gov', \
                                    'created_on':time.ctime(time.time())})
    
    # Specify variable attributes
    output_data.track.attrs['description'] =  'Total number of tracked features'
    output_data.track.attrs['units'] = 'unitless'
    
    output_data.time.attrs['dscription'] = 'Maximum number of features in a given track'
    output_data.time.attrs['units'] = 'unitless'
    
    output_data.pfs.attrs['long_name'] = 'Maximum number of precipitation features in one cloud feature'
    output_data.pfs.attrs['units'] = 'unitless'
    
    output_data.cores.attrs['long_name'] = 'Maximum number of convective cores in a precipitation feature at one time'
    output_data.cores.attrs['units'] = 'unitless'
    
    output_data.mergesplit.attrs['long_name'] = 'Maximum number of mergers / splits at one time'
    output_data.mergesplit.attrs['units'] = 'unitless'
    
    output_data.length.attrs['long_name'] = 'Length of track containing each track'
    output_data.length.attrs['units'] = 'Temporal resolution of orginal data'
    
    output_data.mcs_length.attrs['long_name'] = 'Length of each MCS in each track'
    output_data.mcs_length.attrs['units'] = 'Temporal resolution of orginal data'
    
    output_data.mcs_type.attrs['long_name'] = 'Type of MCS'
    output_data.mcs_type.attrs['values'] = '1 = MCS, 2 = Squall line'
    output_data.mcs_type.attrs['units'] = 'unitless'
    
    output_data.status.attrs['long_name'] = 'Flag indicating the status of each feature in MCS'
    output_data.status.attrs['values'] = '-9999=missing cloud or cloud removed due to short track, 0=track ends here, 1=cloud continues as one cloud in next file, 2=Biggest cloud in merger, 21=Smaller cloud(s) in merger, 13=Cloud that splits, 3=Biggest cloud from a split that stops after the split, 31=Smaller cloud(s) from a split that stop after the split. The last seven classifications are added together in different combinations to describe situations.'
    output_data.status.attrs['min_value'] = 0
    output_data.status.attrs['max_value'] = 52
    output_data.status.attrs['units'] = 'unitless'
    
    output_data.startstatus.attrs['long_name'] = 'Flag indicating the status of first feature in MCS track'
    output_data.startstatus.attrs['values'] = '-9999=missing cloud or cloud removed due to short track, 0=track ends here, 1=cloud continues as one cloud in next file, 2=Biggest cloud in merger, 21=Smaller cloud(s) in merger, 13=Cloud that splits, 3=Biggest cloud from a split that stops after the split, 31=Smaller cloud(s) from a split that stop after the split. The last seven classifications are added together in different combinations to describe situations.'
    output_data.startstatus.attrs['min_value'] = 0
    output_data.startstatus.attrs['max_value'] = 52
    output_data.startstatus.attrs['units'] = 'unitless'
    
    output_data.endstatus.attrs['long_name'] = 'Flag indicating the status of last feature in MCS track'
    output_data.endstatus.attrs['values'] = '-9999=missing cloud or cloud removed due to short track, 0=track ends here, 1=cloud continues as one cloud in next file, 2=Biggest cloud in merger, 21=Smaller cloud(s) in merger, 13=Cloud that splits, 3=Biggest cloud from a split that stops after the split, 31=Smaller cloud(s) from a split that stop after the split. The last seven classifications are added together in different combinations to describe situations.'
    output_data.endstatus.attrs['min_value'] = 0
    output_data.endstatus.attrs['max_value'] = 52
    output_data.endstatus.attrs['units'] = 'unitless'
    
    output_data.interruptions.attrs['long_name'] = 'flag indicating if track incomplete'
    output_data.interruptions.attrs['values'] = '0 = full track available, good data. 1 = track starts at first file, track cut short by data availability. 2 = track ends at last file, track cut short by data availability'
    output_data.interruptions.attrs['min_value'] = 0
    output_data.interruptions.attrs['max_value'] = 2
    output_data.interruptions.attrs['units'] = 'unitless'
    
    output_data.boundary.attrs['long_name'] = 'Flag indicating whether the core + cold anvil touches one of the domain edges.'
    output_data.boundary.attrs['values'] = '0 = away from edge. 1= touches edge.'
    output_data.boundary.attrs['min_value'] = 0
    output_data.boundary.attrs['max_value'] = 1
    output_data.boundary.attrs['units'] = 'unitless'
    
    output_data.basetime.attrs['long_name'] = 'epoch time (seconds since 01/01/1970 00:00) of file'
    output_data.basetime.attrs['standard_name'] = 'time'
    
    #output_data.basetime.attrs['standard_name'] = 'time'
    #output_data.basetime.attrs['long_name'] = 'basetime of cloud at the given time'
    
    output_data.datetimestring.attrs['long_name'] = 'date-time'
    output_data.datetimestring.attrs['long_name'] = 'date_time for each cloud in the mcs'
    
    output_data.meanlon.attrs['standard_name'] = 'longitude'
    output_data.meanlon.attrs['long_name'] = 'mean longitude of the core + cold anvil for each feature at the given time'
    output_data.meanlon.attrs['min_value'] = geolimits[1]
    output_data.meanlon.attrs['max_value'] = geolimits[3]
    output_data.meanlon.attrs['units'] = 'degrees'
    
    output_data.meanlat.attrs['standard_name'] = 'latitude'
    output_data.meanlat.attrs['long_name'] = 'mean latitude of the core + cold anvil for each feature at the given time'
    output_data.meanlat.attrs['min_value'] = geolimits[0]
    output_data.meanlat.attrs['max_value'] = geolimits[2]
    output_data.meanlat.attrs['units'] = 'degrees'
    
    output_data.core_area.attrs['long_name'] = 'area of the cold core at the given time'
    output_data.core_area.attrs['units'] = 'km^2'
    
    output_data.ccs_area.attrs['long_name'] = 'area of the cold core and cold anvil at the given time'
    output_data.ccs_area.attrs['units'] = 'km^2'
    
    output_data.cloudnumber.attrs['long_name'] = 'cloud number in the corresponding cloudid file of clouds in the mcs'
    output_data.cloudnumber.attrs['usage'] = 'to link this tracking statistics file with pixel-level cloudid files, use the cloudidfile and cloudnumber together to identify which cloud this current track and time is associated with'
    output_data.cloudnumber.attrs['units'] = 'unitless'
    
    output_data.mergecloudnumber.attrs['long_name'] = 'cloud number of small, short-lived clouds merging into the MCS'
    output_data.mergecloudnumber.attrs['usage'] = 'to link this tracking statistics file with pixel-level cloudid files, use the cloudidfile and cloudnumber together to identify which cloud this current track and time is associated with'
    output_data.mergecloudnumber.attrs['units'] = 'unitless'
    
    output_data.splitcloudnumber.attrs['long_name'] = 'cloud number of small, short-lived clouds splitting from the MCS'
    output_data.splitcloudnumber.attrs['usage'] = 'to link this tracking statistics file with pixel-level cloudid files, use the cloudidfile and cloudnumber together to identify which cloud this current track and time is associated with'
    output_data.splitcloudnumber.attrs['units'] = 'unitless'
    
    output_data.pf_frac.attrs['long_name'] = 'fraction of cold cloud shielf covered by rainrate mask'
    output_data.pf_frac.attrs['units'] = 'unitless'
    output_data.pf_frac.attrs['min_value'] = 0
    output_data.pf_frac.attrs['max_value'] = 1
    output_data.pf_frac.attrs['units'] = 'unitless'
    
    output_data.pf_npf.attrs['long_name'] = 'number of precipitation features at a given time'
    output_data.pf_npf.attrs['units'] = 'unitless'
    
    output_data.pf_area.attrs['long_name'] = 'area of each precipitation feature at a given time'
    output_data.pf_area.attrs['units'] = 'km^2'
    
    output_data.pf_lon.attrs['standard_name'] = 'longitude'
    output_data.pf_lon.attrs['long_name'] = 'mean longitude of each precipitaiton feature at a given time'
    output_data.pf_lon.attrs['units'] = 'degrees'
    
    output_data.pf_lat.attrs['standard_name'] = 'latitude'
    output_data.pf_lat.attrs['long_name'] = 'mean latitude of each precipitaiton feature at a given time'
    output_data.pf_lat.attrs['units'] = 'degrees'
    
    output_data.pf_rainrate.attrs['long_name'] = 'mean precipitation rate (from rad_hsr_1h) pf each precipitation feature at a given time'
    output_data.pf_rainrate.attrs['units'] = 'mm/hr'
    
    output_data.pf_skewness.attrs['long_name'] = 'skewness of each precipitation feature at a given time'
    output_data.pf_skewness.attrs['units'] = 'unitless'
    
    output_data.pf_majoraxislength.attrs['long_name'] = 'major axis length of each precipitation feature at a given time'
    output_data.pf_majoraxislength.attrs['units'] = 'km'
    
    output_data.pf_minoraxislength.attrs['long_name'] = 'minor axis length of each precipitation feature at a given time'
    output_data.pf_minoraxislength.attrs['units'] = 'km'
    
    output_data.pf_aspectratio.attrs['long_name'] = 'aspect ratio (major axis / minor axis) of each precipitation feature at a given time'
    output_data.pf_aspectratio.attrs['units'] = 'unitless'
    
    output_data.pf_eccentricity.attrs['long_name'] = 'eccentricity of each precipitation feature at a given time'
    output_data.pf_eccentricity.attrs['min_value'] = 0
    output_data.pf_eccentricity.attrs['max_value'] = 1
    output_data.pf_eccentricity.attrs['units'] = 'unitless'
    
    output_data.pf_orientation.attrs['long_name'] = 'orientation of the major axis of each precipitation feature at a given time'
    output_data.pf_orientation.attrs['units'] = 'degrees clockwise from vertical'
    output_data.pf_orientation.attrs['min_value'] = 0
    output_data.pf_orientation.attrs['max_value'] = 360
    
    
    # Write netcdf file
    print('')
    print(statistics_outfile)
    output_data.to_netcdf(path=statistics_outfile, mode='w', format='NETCDF4_CLASSIC', unlimited_dims='track', \
                          encoding={'mcs_length': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                    'mcs_type': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                    'status': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                    'startstatus': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                    'endstatus': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                    #'basetime': {'dtype': 'int64', 'zlib':True}, \
                                    #'basetime': {'dtype': 'int64', 'zlib':True, 'units': 'seconds since 1970-01-01'}, \
                                    #'basetime': {'zlib':True, 'units': 'seconds since 1970-01-01'}, \
                                    #'basetime': {'zlib':True, 'units': basetime_units, 'calendar': basetime_calendar}, \
                                    'basetime': {'zlib':True}, \
                                    'datetimestring': {'zlib':True}, \
                                    'boundary': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                    'interruptions': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                    'meanlat': {'zlib':True, '_FillValue': np.nan}, \
                                    'meanlon': {'zlib':True, '_FillValue': np.nan}, \
                                    'core_area': {'zlib':True, '_FillValue': np.nan}, \
                                    'ccs_area': {'zlib':True, '_FillValue': np.nan}, \
                                    'cloudnumber': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                    'mergecloudnumber': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                    'splitcloudnumber': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                    'pf_frac': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_area': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_lon': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_lat': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_rainrate': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_skewness': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_majoraxislength': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_minoraxislength': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_aspectratio': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_orientation': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_eccentricity': {'zlib':True, '_FillValue': np.nan}})


