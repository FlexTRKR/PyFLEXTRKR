def matchtbpf_singlefile(cloudid_filename, cloudidtrack_path, cloudid_filebase, rainaccumulation_path, rainaccumulation_filebase, ir_basetime, ir_cloudnumber, ir_mergecloudnumber, ir_splitcloudnumber, rr_min, pf_link_area_thresh, pixel_radius, nmaxpf):
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
    import glob
    import sys
    from multiprocessing import Pool

    prelength=len(cloudidtrack_path)+len(cloudid_filebase)
    ittdatetimestring=cloudid_filename[prelength:(prelength+13)]
    rainaccumulation_filename = rainaccumulation_path + rainaccumulation_filebase + ittdatetimestring[0:8] + ittdatetimestring[9:11] + '_4km-pixel.nc'

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
        cloudid_basetime = cloudiddata['basetime'][:]
        basetime_units = cloudiddata['basetime'].units
        basetime_calendar = cloudiddata['basetime'].calendar
        cloudiddata.close()

        # Read precipitation data
        print('Loading precip data')
        print(rainaccumulation_filename)
        pfdata = Dataset(rainaccumulation_filename, 'r')
        rawrainratemap = pfdata['precipitationCal'][:] # map of rain rate
        lon1d = pfdata['lon'][:]
        lat1d = pfdata['lat'][:]
        pfdata.close()

        lon, lat = np.meshgrid(lon1d, lat1d, indexing='xy')

        ##########################################################################
        # Get dimensions of data. Data should be preprocesses so that their latittude and longitude dimensions are the same
        ydim, xdim = np.shape(lat)

        matchindices=np.array(np.where(ir_basetime == cloudid_basetime))
        nmatchcloud=len(matchindices[0])
        if (nmatchcloud > 0):
            pf_npf = np.ones(nmatchcloud, dtype=int)*-9999
            #npf_avgnpix = np.ones(nmatchcloud, dtype=int)*-9999
            #npf_avgrainrate = np.ones(nmatchcloud, dtype=int)*-9999
            pf_pflon = np.ones((nmatchcloud, nmaxpf), dtype=float)*np.nan
            pf_pflat = np.ones((nmatchcloud, nmaxpf), dtype=float)*np.nan
            pf_pfnpix = np.ones((nmatchcloud, nmaxpf), dtype=int)*-9999
            pf_pfrainrate = np.ones((nmatchcloud, nmaxpf), dtype=float)*np.nan
            pf_pfskewness = np.ones((nmatchcloud, nmaxpf), dtype=float)*np.nan
            pf_pfmajoraxis = np.ones((nmatchcloud, nmaxpf), dtype=float)*np.nan
            pf_pfminoraxis = np.ones((nmatchcloud, nmaxpf), dtype=float)*np.nan
            pf_pfaspectratio = np.ones((nmatchcloud, nmaxpf), dtype=float)*np.nan
            pf_pforientation = np.ones((nmatchcloud, nmaxpf), dtype=float)*np.nan
            pf_pfeccentricity = np.ones((nmatchcloud, nmaxpf), dtype=float)*np.nan
            pf_pfycentroid = np.ones((nmatchcloud, nmaxpf), dtype=float)*np.nan
            pf_pfxcentroid = np.ones((nmatchcloud, nmaxpf), dtype=float)*np.nan
            pf_pfyweightedcentroid = np.ones((nmatchcloud, nmaxpf), dtype=float)*np.nan
            pf_pfxweightedcentroid = np.ones((nmatchcloud, nmaxpf), dtype=float)*np.nan
            #pf_pfrr8mm = np.zeros((nmatchloud, nmaxpf), dtype=np.long)
            #pf_pfrr10mm = np.zeros((nmatchloud, nmaxpf), dtype=np.long)
            basetime = np.empty(nmatchcloud, dtype='datetime64[s]')
            precip_basetime = np.empty(nmatchcloud)
            #pf_frac = np.ones(nmatchcloud, dtype=float)*np.nan

            for imatchcloud in range(nmatchcloud):
                ittcloudnumber=ir_cloudnumber[matchindices[0,imatchcloud],matchindices[1,imatchcloud]]
                ittmergecloudnumber=ir_mergecloudnumber[matchindices[0,imatchcloud],matchindices[1,imatchcloud],:]
                ittsplitcloudnumber=ir_splitcloudnumber[matchindices[0,imatchcloud],matchindices[1,imatchcloud],:]
                basetime[imatchcloud] = np.array(pd.to_datetime(num2date(cloudid_basetime, units=basetime_units, calendar=basetime_calendar)), dtype='datetime64[s]')[0]
                precip_basetime[imatchcloud] = cloudid_basetime

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
                    print('Calculate new shape statistics')

                    ## Set edges of boundary
                    miny = np.nanmin(icloudlocationy)
                    if miny <= 10:
                        miny = 0
                    else:
                        miny = miny - 10

                    maxy = np.nanmax(icloudlocationy)
                    if maxy >= ydim - 10:
                        maxy = ydim
                    else:
                        maxy = maxy + 11

                    minx = np.nanmin(icloudlocationx)
                    if minx <= 10:
                        minx = 0
                    else:
                        minx = minx - 10

                    maxx = np.nanmax(icloudlocationx)
                    if maxx >= xdim - 10:
                        maxx = xdim
                    else:
                        maxx = maxx + 11

                    ## Isolate smaller region around cloud shield
                    subrainratemap = np.copy(filteredrainratemap[miny:maxy, minx:maxx])
 
                    #########################################################
                    ## Get dimensions of subsetted region
                    subdimy, subdimx = np.shape(subrainratemap)

                    ######################################################   !!!!!!!!!!!!!!! Slow Step !!!!!!!!1
                    # Derive precipitation feature statistics
                    print('Calculating precipitation statistics')
                    #rawrainratemap = np.squeeze(rawrainratemap, axis = 0)
                    #ipfy, ipfx = np.array(np.where(rawrainratemap > rr_min))
                    ipfy, ipfx = np.array(np.where(subrainratemap > rr_min))
                    nrainpix = len(ipfy)

                    if nrainpix > 0:
                        ####################################################
                        # Dilate precipitation feature by one pixel. This slightly smooths the data so that very close precipitation features are connected
                        # Create binary map
                        #binarypfmap = np.zeros((ydim, xdim), dtype=int)
                        binarypfmap = np.zeros((subdimy, subdimx), dtype=int)
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
                        numpf = npf_new
                        pfnumberlabelmap = pf_number
                        del pf_number, npf_new

                        #if npf_new > 0:
                        if numpf > 0:
                            npf_save = np.nanmin([nmaxpf, numpf])
                            print('PFs present, initializing matrices')

                           ##############################################
                           # Initialize matrices
                            pfnpix = np.zeros(npf_save, dtype=float)
                            test = np.ones(npf_save, dtype=float)*np.nan
                            pfid = np.ones(npf_save, dtype=int)*-9999
                            pflon = np.ones(npf_save, dtype=float)*np.nan
                            pflat = np.ones(npf_save, dtype=float)*np.nan
                            pfrainrate = np.ones(npf_save, dtype=float)*np.nan
                            pfskewness = np.ones(npf_save, dtype=float)*np.nan
                            pfmajoraxis = np.ones(npf_save, dtype=float)*np.nan
                            pfminoraxis = np.ones(npf_save, dtype=float)*np.nan
                            pfaspectratio = np.ones(npf_save, dtype=float)*np.nan
                            #pf8mm = np.zeros(npf_save, dtype=np.long)
                            #pf10mm = np.zeros(npf_save, dtype=np.long)
                            pfxcentroid = np.ones(npf_save, dtype=float)*np.nan
                            pfycentroid = np.ones(npf_save, dtype=float)*np.nan
                            pfxweightedcentroid = np.ones(npf_save, dtype=float)*np.nan
                            pfyweightedcentroid = np.ones(npf_save, dtype=float)*np.nan
                            pfeccentricity = np.ones(npf_save, dtype=float)*np.nan
                            pfperimeter = np.ones(npf_save, dtype=float)*np.nan
                            pforientation = np.ones(npf_save, dtype=float)*np.nan

                            print('Looping through each feature to calculate statistics')
                            print(('Number of PFs ' + str(numpf)))
                            ###############################################
                            # Loop through each feature
                            for ipf in range(1, npf_save+1):

                                #######################################
                                # Find associated indices
                                iipfy, iipfx = np.array(np.where(pfnumberlabelmap == ipf))
                                niipfpix = len(iipfy)

                                if (niipfpix != pf_npix[ipf -1]):
                                    sys.exit("pixel count not match")
                                else:
                                    ##########################################
                                    # Compute statistics

                                    # Basic statistics
                                    pfnpix[ipf-1] = np.copy(niipfpix)
                                    pfid[ipf-1] = np.copy(int(ipf))
                                    #pfrainrate[ipf-1] = np.nanmean(filteredrainratemap[iipfy[:], iipfx[:]])
                                    pfrainrate[ipf-1] = np.nanmean(subrainratemap[iipfy[:], iipfx[:]])
                                    #pfskewness[ipf-1] = skew(filteredrainratemap[iipfy[:], iipfx[:]])
                                    pfskewness[ipf-1] = skew(subrainratemap[iipfy[:], iipfx[:]])
                                    pflon[ipf-1] = np.nanmean(lat[iipfy[:]+miny, iipfx[:]+minx])
                                    pflat[ipf-1] = np.nanmean(lon[iipfy[:]+miny, iipfx[:]+minx])
                                    
                                    ## Count number of pixel rainrate greater than specified threshold
                                    #print('subrainratemap_size: ', subrainratemap.shape)
                                    #iirry, iirrx = np.array(np.where(subrainratemap > 8))
                                    #print('size of iirry:', iirry.shape)
                                    #if iirry > 0:
                                        #pf8mm[ipf-1] = iirry
                                    
                                    #iirry10 = np.array(np.where(subrainratemap > 10))
                                    #if iirry10 > 0:
                                        #pf10mm[ipf-1] = iirry

                                    # Generate map of convective core
                                    iipfflagmap = np.zeros((subdimy, subdimx), dtype=int)
                                    iipfflagmap[iipfy, iipfx] = 1

                                    # Geometric statistics
                                    tfilteredrainratemap = np.copy(subrainratemap)
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
                                    pfycentroid[ipf-1]=pfycentroid[ipf-1]+miny
                                    pfxcentroid[ipf-1]=pfxcentroid[ipf-1]+minx
                                    pfyweightedcentroid[ipf-1]=pfyweightedcentroid[ipf-1]+miny
                                    pfxweightedcentroid[ipf-1]=pfxweightedcentroid[ipf-1]+minx
                                    
                            print('Loop done')

                            ###################################################
                            # Save precipitation feature statisitcs
                            pf_npf[imatchcloud] = np.copy(numpf)
                            pf_pflon[imatchcloud, 0:npf_save]= pflon[0:npf_save]
                            pf_pflat[imatchcloud, 0:npf_save] = pflat[0:npf_save]
                            pf_pfnpix[imatchcloud, 0:npf_save] = pfnpix[0:npf_save]
                            pf_pfrainrate[imatchcloud, 0:npf_save] = pfrainrate[0:npf_save]
                            pf_pfskewness[imatchcloud, 0:npf_save] = pfskewness[0:npf_save]
                            pf_pfmajoraxis[imatchcloud, 0:npf_save] = pfmajoraxis[0:npf_save]
                            pf_pfminoraxis[imatchcloud, 0:npf_save] = pfminoraxis[0:npf_save]
                            pf_pfaspectratio[imatchcloud, 0:npf_save] = pfaspectratio[0:npf_save]
                            pf_pforientation[imatchcloud, 0:npf_save] = pforientation[0:npf_save]
                            pf_pfeccentricity[imatchcloud, 0:npf_save] = pfeccentricity[0:npf_save]
                            pf_pfycentroid[imatchcloud, 0:npf_save] = pfycentroid[0:npf_save]
                            pf_pfxcentroid[imatchcloud, 0:npf_save] = pfxcentroid[0:npf_save]
                            pf_pfxweightedcentroid[imatchcloud, 0:npf_save] = pfxweightedcentroid[0:npf_save]
                            pf_pfyweightedcentroid[imatchcloud, 0:npf_save] = pfyweightedcentroid[0:npf_save]
                            #pf_pfrr8mm[imatchcloud, 0:npf_save] = pf8mm[0:npf_save]
                            #pf_pfrr10mm[imatchcloud, 0:npf_save] = pf10mm[0:npf_save]

            return nmatchcloud, matchindices, pf_npf, pf_pflon, pf_pflat, pf_pfnpix, pf_pfrainrate, pf_pfskewness, pf_pfmajoraxis, \
                pf_pfminoraxis, pf_pfaspectratio, pf_pforientation, pf_pfeccentricity, pf_pfycentroid, pf_pfxcentroid, \
                pf_pfyweightedcentroid, pf_pfxweightedcentroid, basetime, precip_basetime


