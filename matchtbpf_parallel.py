# Purpose: match mergedir tracked MCS with WRF rainrate statistics underneath the cloud features.

# Author: Original IDL code written by Zhe Feng (zhe.feng@pnnl.gov), Python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov), Python WRF version modified from original IDL and python versions by Katelyn Barber (katelyn.barber@pnnl.gov)

def identifypf_wrf_rain(mcsstats_filebase, cloudid_filebase, rainaccumulation_filebase, stats_path,cloudidtrack_path,rainaccumulation_path, startdate, enddate, geolimits, nmaxpf, nmaxcore, nmaxclouds, rr_min, pixel_radius, irdatasource, precipdatasource, datadescription, datatimeresolution, mcs_irareathresh, mcs_irdurationthresh, mcs_ireccentricitythresh,pf_link_area_thresh, nprocesses):

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
    import glob
    from multiprocessing import Pool
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
    #npf_avgnpix = np.ones((ir_ntracks, ir_nmaxlength), dtype=int)*-9999
    #npf_avgrainrate = np.ones((ir_ntracks, ir_nmaxlength), dtype=int)*-9999
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

    #pf_frac = np.ones((ir_ntracks, ir_nmaxlength), dtype=float)*np.nan

    ##############################################################
    # Find precipitation feature in each mcs
    print(('Total Number of Tracks:' + str(ir_ntracks)))

    # Loop over each track
    print('Looping over each track')
    print((time.ctime()))
    print(ir_ntracks)

    statistics_outfile = stats_path + 'mcs_tracks_' + precipdatasource + '_' + startdate + '_' + enddate + '.nc'
    cloudidfilelist=glob.glob(cloudidtrack_path + cloudid_filebase+'*.nc')
    nfiles=len(cloudidfilelist)
    from matchtbpf_single import matchtbpf_singlefile
    with Pool(nprocesses) as pool:
        results=pool.starmap(matchtbpf_singlefile,[(filename, cloudidtrack_path, cloudid_filebase, rainaccumulation_path, \
            rainaccumulation_filebase, ir_basetime, ir_cloudnumber, ir_mergecloudnumber, ir_splitcloudnumber, \
            rr_min, pf_link_area_thresh, pixel_radius, nmaxpf) for filename in cloudidfilelist])
        pool.close()

    for ifile in range(nfiles):
        tmp=results[ifile]
        if (tmp is not None):
            nmatchcloudtmp=tmp[0]
            matchindicestmp=tmp[1]
            for imatch in range(nmatchcloudtmp):
                pf_npf[matchindicestmp[0,imatch],matchindicestmp[1,imatch]]=tmp[2][imatch]
                pf_pflon[matchindicestmp[0,imatch],matchindicestmp[1,imatch],:]=tmp[3][imatch,:]
                pf_pflat[matchindicestmp[0,imatch],matchindicestmp[1,imatch],:]=tmp[4][imatch,:]
                pf_pfnpix[matchindicestmp[0,imatch],matchindicestmp[1,imatch],:]=tmp[5][imatch,:]
                pf_pfrainrate[matchindicestmp[0,imatch],matchindicestmp[1,imatch],:]=tmp[6][imatch,:]
                pf_pfskewness[matchindicestmp[0,imatch],matchindicestmp[1,imatch],:]=tmp[7][imatch,:]
                pf_pfmajoraxis[matchindicestmp[0,imatch],matchindicestmp[1,imatch],:]=tmp[8][imatch,:]
                pf_pfminoraxis[matchindicestmp[0,imatch],matchindicestmp[1,imatch],:]=tmp[9][imatch,:]
                pf_pfaspectratio[matchindicestmp[0,imatch],matchindicestmp[1,imatch],:]=tmp[10][imatch,:]
                pf_pforientation[matchindicestmp[0,imatch],matchindicestmp[1,imatch],:]=tmp[11][imatch,:]
                pf_pfeccentricity[matchindicestmp[0,imatch],matchindicestmp[1,imatch],:]=tmp[12][imatch,:]
                pf_pfycentroid[matchindicestmp[0,imatch],matchindicestmp[1,imatch],:]=tmp[13][imatch,:]
                pf_pfxcentroid[matchindicestmp[0,imatch],matchindicestmp[1,imatch],:]=tmp[14][imatch,:]
                pf_pfyweightedcentroid[matchindicestmp[0,imatch],matchindicestmp[1,imatch],:]=tmp[15][imatch,:]
                pf_pfxweightedcentroid[matchindicestmp[0,imatch],matchindicestmp[1,imatch],:]=tmp[16][imatch,:]
                basetime[matchindicestmp[0,imatch],matchindicestmp[1,imatch]]=tmp[17][imatch]
                precip_basetime[matchindicestmp[0,imatch],matchindicestmp[1,imatch]]=tmp[18][imatch]

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
                              'boundary': (['track', 'time'], ir_boundary), \
                              'basetime': (['track', 'time'], precip_basetime), \
                              'datetimestring': (['track', 'time', 'characters'], ir_datetimestring), \
                              'meanlat': (['track', 'time'], ir_meanlat), \
                              'meanlon': (['track', 'time'], ir_meanlon), \
                              'core_area': (['track', 'time'], ir_corearea), \
                              'ccs_area': (['track', 'time'],  ir_ccsarea), \
                              'cloudnumber': (['track', 'time'],  ir_cloudnumber), \
                              'mergecloudnumber': (['track', 'time', 'mergesplit'], ir_mergecloudnumber), \
                              'splitcloudnumber': (['track', 'time', 'mergesplit'], ir_splitcloudnumber), \
                              #'pf_frac': (['track', 'time'], pf_frac), \
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
    
    #output_data.pf_frac.attrs['long_name'] = 'fraction of cold cloud shielf covered by rainrate mask'
    #output_data.pf_frac.attrs['units'] = 'unitless'
    #output_data.pf_frac.attrs['min_value'] = 0
    #output_data.pf_frac.attrs['max_value'] = 1
    #output_data.pf_frac.attrs['units'] = 'unitless'
    
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
                                    #'pf_frac': {'zlib':True, '_FillValue': np.nan}, \
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


