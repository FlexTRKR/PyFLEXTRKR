# Purpose: Filter MCS using NMQ radar variables so that only MCSs statisfying radar thresholds are retained. The lifecycle of these robust MCS is also identified. Method similar to Coniglio et al (2010) MWR. 

# Author: Original IDL code written by Zhe Feng (zhe.feng@pnnl.gov), Python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov), altered by Katelyn Barber (katelyn.barber@pnnl.gov)

def filtermcs_wrf_rain(stats_path, pfstats_filebase, startdate, enddate, timeresolution, geolimits, majoraxisthresh, durationthresh, aspectratiothresh, lifecyclethresh, lengththresh, gapthresh):
    # Inputs:
    # stats_path - directory which stores this statistics data. this is where the output from this code will be placed
    # pfstats_filebase - file header of the precipitation feature statistics file generated in the previous code.
    # startdate - starting date and time of the data
    # enddate - ending date and time of the data
    # time_resolution - time resolution of the satellite and radar data
    # geolimits - 4-element array with plotting boundaries [lat_min, lon_min, lat_max, lon_max]
    # majoraxisthresh - minimum major axis length of the largest precipitation feature in a robust MCSs
    # durationthresh - minimum length of precipitation feature in a robust MCS
    # aspectratiothresh - minimum aspect ratio the largest precipitation feature must have to called a squall line
    # lifecyclethresh - minimum duration required for the lifecycles of an MCS to be classified
    # lengththresh - minimum size required for the lifecycles of an MCS to be classified
    # gapthresh - minimum allowable time gap between precipitation features that allow the storm to still be classified as a robust MCS

    # Outputs: (One netcdf file with satellite, radar, and rain accumulation statistics about MCSs that satisfy both satellite and radar requirements)
    # MCS_length - duration of MCS
    # mcs_type - flag indicating whether this is squall line, based on satellite definition
    # pf_lifetime - length of time in which precipitation is observed during each track
    # status - flag indicating the evolution of each cloud in a MCS
    # startstatus - flag indicating how a MCS starts
    # endstatus - flag indicating how a MCS ends
    # interruptions - flag indicating if the satellite data used to indentify this MCS is incomplete
    # boundary - flag indicating if a MCS touches the edge of the domain
    # base_time - seconds since 1970-01-01 for each cloud in a MCS
    # datetimestring - string of date and time of each cloud in a MCS
    # meanlat - mean latitude of the MCS
    # meanlon - mean longitude of the MCS
    # core_area - area of the core of MCS
    # ccs_area - area of the core and cold anvil of the MCS
    # cloudnumber - numbers indicating which clouds in the cloudid files are associated with a MCS
    # mergecloudnumber - numbers indicating which clouds in the cloudid files merge into this track
    # splitcloudnumber - numbers indicating which clouds in the cloudid files split from this track
    # pf_mcsstatus - flag indicating which part of the tracks are part of the robust MCS
    # lifecycle_complete_flag - flag indicating if this MCS has each element in the MCS life cycle
    # lifecycle_index - time index when each phase of the MCS life cycle starts
    # lifecycle_stage - flag indicating the lifestage of each cloud in a robust MCS
    # nmq_frac - fraction of the cloud that exists within the radar domain
    # npf - number of precipitation features at each time
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


    ######################################################
    # Import modules
    import numpy as np
    import xarray as xr
    import sys
    import time
    import warnings
    import pandas as pd
    np.set_printoptions(threshold=np.inf)

    ######################################################
    # Load mergedir mcs and pf data
    mergedirpf_statistics_file = stats_path + pfstats_filebase + startdate + '_' + enddate + '.nc'

    data = xr.open_dataset(mergedirpf_statistics_file, autoclose=True)
    ntracks = np.nanmax(data.coords['track'])
    ntimes = len(data.coords['time'])
    ncores =  len(data.coords['cores'])

    ir_tracklength = data['length'].data

    pf_area = data['pf_area'].data
    pf_majoraxis = data['pf_majoraxislength'].data
    pf_rainrate = data['pf_rainrate'].data   

    time_res = float(data.attrs['time_resolution_hour_or_minutes']) 
    if time_res > 5:
        time_res = (time_res)/60 # puts time res into hr
    print(time_res)
    mcs_ir_areathresh = float(data.attrs['MCS_IR_area_thresh_km2'])
    mcs_ir_durationthresh = float(data.attrs['MCS_IR_duration_thresh_hr'])
    mcs_ir_eccentricitythresh = float(data.attrs['MCS_IR_eccentricity_thres'])
    basetime = data['basetime'].data

    ##################################################
    # Initialize matrices
    trackid_mcs = []
    trackid_nonmcs = []

    pf_mcstype = np.ones(ntracks, dtype=int)*-9999
    pf_mcsstatus = np.ones((ntracks, ntimes), dtype=int)*-9999

    ###################################################
    # Loop through each track
    for nt in range(0, ntracks):
        print(('Track # ' + str(nt)))

        ############################################
        # Isolate data from this track 
        ilength = np.copy(ir_tracklength[nt]).astype(int)

        # Get the largest precipitation (1st entry in 3rd dimension)
        ipf_majoraxis = np.copy(pf_majoraxis[nt, 0:ilength, 0])
        print(ipf_majoraxis)
        ipf_rainrate = np.copy(pf_rainrate[nt, 0:ilength, 0])
        print(ipf_rainrate)

        ######################################################
        # Apply precip defined MCS criteria

        # Apply PF major axis length > thresh and contains rainrates >= 1 mm/hr criteria
        ipfmcs = np.array(np.where((ipf_majoraxis > majoraxisthresh) & (ipf_rainrate > 1)))[0, :]
        nipfmcs = len(ipfmcs)
        print(nipfmcs)
        print(nipfmcs*time_res)
        print(durationthresh)

        if nipfmcs > 0 :
            # Apply duration threshold to entire time period
            if nipfmcs*time_res > durationthresh:

                # Find continuous duration indices
                groups = np.split(ipfmcs, np.where(np.diff(ipfmcs) > gapthresh)[0]+1) # KB CHANGED != to >
                nbreaks = len(groups)

                for igroup in range(0, nbreaks):

                    ############################################################
                    # Determine if each group satisfies duration threshold
                    igroup_indices = np.array(np.copy(groups[igroup][:]))
                    nigroup = len(igroup_indices)

                    # Group satisfies duration threshold
                    #if np.multiply(len(groups[igroup][:]), time_res) > durationthresh:
                    if np.multiply((groups[igroup][-1]-groups[igroup][0]), time_res) >= durationthresh:   # KB CHANGED

                        # Get radar variables for this group
                        igroup_duration = len(groups[igroup])*time_res
                        igroup_pfmajoraxis = np.copy(ipf_majoraxis[igroup_indices])

                        # Label this period as an mcs
                        pf_mcsstatus[nt, igroup_indices] = 1
                        print('MCS')

                        ## Determine type of mcs (squall or non-squall)
                        #isquall = np.array(np.where(igroup_ccaspectratio > aspectratiothresh))[0, :]
                        #nisquall = len(isquall)

                        #if nisquall > 0:
                        #    # Label as squall
                        #    pf_mcstype[nt] = 1
                        #    pf_cctype[nt, igroup_indices[isquall]] = 1
                        #else:
                        #    # Label as non-squall
                        #    pf_mcstype[nt] = 2
                        #    pf_cctype[nt, igroup_indices[isquall]] = 2
                    else:
                        print('Not MCS')

            # Group does not satistfy duration threshold
            else:
                trackid_nonmcs = np.append(trackid_nonmcs, nt)
                print('Not NCS')
        else:
            print('Not MCS')

    # Isolate tracks that have robust MCS
    TEMP_mcsstatus = np.copy(pf_mcsstatus).astype(float)
    TEMP_mcsstatus[TEMP_mcsstatus == -9999] = np.nan
    trackid_mcs = np.array(np.where(np.nansum(TEMP_mcsstatus, axis=1)))[0, :]
    nmcs = len(trackid_mcs)

    # Stop code if not robust MCS present
    if nmcs == 0:
        sys.exit('No MCS found!')
    else:
        print(('Number of robust MCS: ' + str(int(nmcs))))

    # Isolate data associated with robust MCS
    ir_tracklength = ir_tracklength[trackid_mcs]
    mcs_basetime = basetime[trackid_mcs]
    #print(mcs_basetime)

    #pf_mcstype = pf_mcstype[trackid_mcs]
    pf_mcsstatus = pf_mcsstatus[trackid_mcs, :]
    pf_majoraxis = pf_majoraxis[trackid_mcs, :, :]
    pf_area = pf_area[trackid_mcs, :, :]

    # Determine how long MCS track criteria is satisfied
    TEMP_mcsstatus = np.copy(pf_mcsstatus).astype(float)
    TEMP_mcsstatus[TEMP_mcsstatus == -9999] = np.nan
    mcs_length = np.nansum(TEMP_mcsstatus, axis=1)

    # Get lifetime when a significant precip feature is present
    warnings.filterwarnings('ignore')
    pf_maxmajoraxis = np.nanmax(pf_majoraxis, axis=2) # Creates run time warning
    pf_maxmajoraxis[pf_maxmajoraxis < lengththresh] = 0
    pf_maxmajoraxis[pf_maxmajoraxis > lengththresh] = 1
    pf_lifetime = np.multiply(np.nansum(pf_maxmajoraxis, axis=1), timeresolution)

    ########################################################
    # Definite life cycle stages. Based on Coniglio et al. (2010) MWR.
    # Preconvective (1): first hour after convective core occurs
    # Genesis (2): First hour after convective line exceeds 100 km
    # Mature (3): Near continuous line with well defined stratiform precipitation. 2 hours after genesis state and 2 hours before decay stage
    # Dissipiation (4): First hour after convective line is no longer observed

    # Process only MCSs that last at least 8 hours
    lifetime = np.multiply(ir_tracklength, time_res)
    ilongmcs = np.array(np.where(lifetime >= lifecyclethresh))[0, :]
    nlongmcs = len(ilongmcs)

    if nlongmcs > 0:
        print('ENTERED NLONGMCS IF STATEMENT LINES 214')
        # Initialize arrays
        cycle_complete = np.ones(nmcs, dtype=int)*-9999
        cycle_stage = np.ones((nmcs, ntimes), dtype=int)*-9999
        cycle_index = np.ones((nmcs, 5), dtype=int)*-9999

        mcs_basetime = np.empty((nmcs, ntimes), dtype='datetime64[s]')
        print(mcs_basetime)

        # Loop through each mcs
        for ilm in range(0, nlongmcs):
            # Initialize arrays
            ilm_index = np.ones(5, dtype=int)*-9999
            ilm_cycle = np.ones(ntimes, dtype=int)*-9999

            # Isolate data from this track
            ilm_irtracklength = np.copy(ir_tracklength[ilongmcs[ilm]]).astype(int)
            ilm_pfarea = np.copy(pf_area[ilongmcs[ilm], 0:ilm_irtracklength, 0])
            ilm_pfmajoraxis = np.copy(pf_majoraxis[ilongmcs[ilm], 0:ilm_irtracklength, 0])
            ilm_maxpfmajoraxis = np.nanmax(ilm_pfmajoraxis, axis=0)

            # Get basetime
            TEMP_basetime = np.array([pd.to_datetime(data['basetime'][trackid_mcs[ilongmcs[ilm]], 0:ilm_irtracklength].data, unit='s')])
            mcs_basetime[ilm, 0:ilm_irtracklength] = TEMP_basetime

            ##################################################################
            # Find indices of when convective line present and absent and when stratiform present

#            # Find times with convective core area > 0
#            iccarea = np.array(np.where(ilm_maxpfccarea > 0))[0, :]
#            iccarea_groups = np.split(iccarea, np.where(np.diff(iccarea) > 2)[0]+1)
#            if len(iccarea) > 0 and len(iccarea_groups) > 1:
#                grouplength = np.empty(len(iccarea_groups))
#                for igroup in range(0, len(iccarea_groups)):
#                    grouplength[igroup] = len(iccarea_groups[igroup][:])
#                maxgroup = np.nanargmax(grouplength)
#                iccarea = iccarea_groups[maxgroup][:]
#            elif len(iccarea) > 0 :
#                iccarea = np.arange(iccarea[0], iccarea[-1]+1)
#            nccarea = len(iccarea)

            # Find times with major axis length greater than 100 km
            iccline = np.array(np.where(ilm_maxpfmajoraxis > 100))[0, :]
            iccline_groups = np.split(iccline, np.where(np.diff(iccline) > 2)[0]+1)
            if len(iccline) > 0 and len(iccline_groups) > 1:
                grouplength = np.empty(len(iccline_groups))
                for igroup in range(0, len(iccline_groups)):
                    grouplength[igroup] = len(iccline_groups[igroup][:])
                maxgroup = np.nanargmax(grouplength)
                iccline = iccline_groups[maxgroup][:]
            elif len(iccline) > 0:
                iccline = np.arange(iccline[0], iccline[-1]+1)
            nccline = len(iccline)

            ###############################################################################
            # Classify cloud only stage 

            # Cloud only stage
#            if nccarea > 0:
#                # If first convective time is after the first cloud time, label all hours before the convective core appearance time as preconvective
#                if iccarea[0] > 0 and iccarea[0] < ilm_irtracklength-1:
#                    ilm_index[0] = 0 # Start of cloud only
#                    ilm_cycle[0:iccarea[0]] = 1 # Time period of cloud only

#                ilm_index[1] = iccarea[0] # Start of unorganized convective cells 

            # If convective line exists
            if nccline > 1:
                # If the convective line occurs after the first storm time (use second index since convective line must be around for one hour prior to classifying as genesis)
                # Label when convective cores first appear, but are not organized into a line
                if iccline[1] > 0:
                    ilm_index[2] = iccline[1] # Start of organized convection
                    ilm_cycle[iccline[1]] = 2 # Time period of unorganzied convective cells
                else:
                    sys.exit('Check convective line in track ' + str(int(ilongmcs[ilm])))

            ############################################################
            # Final life cycle processing
            istage = np.array(np.where(ilm_cycle >= 0))[0, :]
            nstage = len(istage)

            if nstage > 0:
                cyclepresent = np.copy(ilm_cycle[istage])
                uniquecycle = np.unique(cyclepresent)
                
                # Label as complete cycle if 1-4 present
                if len(uniquecycle) >= 4:
                    cycle_complete[ilongmcs[ilm]] = 1

                # Save data
                cycle_stage[ilongmcs[ilm], :] = np.copy(ilm_cycle)
                cycle_index[ilongmcs[ilm], :] = np.copy(ilm_index)

    #################################################################################
    # Save data to netcdf file
    statistics_outfile = stats_path + 'robust_mcs_tracks_' + data.attrs['source2'] + '_' + data.attrs['startdate'] + '_' + data.attrs['enddate'] + '.nc'

    # Define xarrray dataset
    output_data = xr.Dataset({'mcs_length': (['track'], data['length'][trackid_mcs]), \
                              'mcs_type': (['track'], data['mcs_type'][trackid_mcs]), \
                              'pf_lifetime': (['track'], pf_lifetime), \
                              'status': (['track', 'time'], data['status'][trackid_mcs, :]), \
                              'startstatus': (['track'], data['startstatus'][trackid_mcs]), \
                              'endstatus': (['track'], data['endstatus'][trackid_mcs]), \
                              'interruptions': (['track'], data['interruptions'][trackid_mcs]), \
                              'boundary': (['track'], data['boundary'][trackid_mcs]), \
                              'base_time': (['track', 'time'], mcs_basetime), \
                              'datetimestring': (['track', 'time', 'characters'], data['datetimestring'][trackid_mcs, :, :]), \
                              'meanlat': (['track', 'time'], data['meanlat'][trackid_mcs, :]), \
                              'meanlon': (['track', 'time'], data['meanlon'][trackid_mcs, :]), \
                              'core_area': (['track', 'time'], data['core_area'][trackid_mcs, :]), \
                              'cloudnumber': (['track', 'time'], data['cloudnumber'][trackid_mcs, :]), \
                              'mergecloudnumber': (['track', 'time', 'mergesplit'], data['mergecloudnumber'][trackid_mcs, :, :]), \
                              'splitcloudnumber': (['track', 'time', 'mergesplit'], data['splitcloudnumber'][trackid_mcs, :, :]), \
                              #'pf_mcsstatus': (['track', 'time'], pf_mcsstatus), \
                              #'lifecycle_complete_flag': (['track'], cycle_complete), \
                              #'lifecycle_index': (['track', 'lifestages'], cycle_index), \
                              #'lifecycle_stage': (['track', 'time'], cycle_stage), \
                              'pf_frac': (['track', 'time'], data['pf_frac'][trackid_mcs]), \
                              'npf': (['track', 'time'], data['pf_npf'][trackid_mcs]), \
                              'pf_area': (['track', 'time', 'pfs'], data['pf_area'][trackid_mcs, :, :]), \
                              'pf_lon': (['track', 'time', 'pfs'], data['pf_lon'][trackid_mcs, :, :]), \
                              'pf_lat': (['track', 'time', 'pfs'], data['pf_lat'][trackid_mcs, :, :]), \
                              'pf_rainrate': (['track', 'time', 'pfs'], data['pf_rainrate'][trackid_mcs, :, :]), \
                              'pf_skewness': (['track', 'time', 'pfs'], data['pf_skewness'][trackid_mcs, :, :]), \
                              'pf_majoraxislength': (['track', 'time', 'pfs'], data['pf_majoraxislength'][trackid_mcs, :, :]), \
                              'pf_minoraxislength': (['track', 'time', 'pfs'], data['pf_minoraxislength'][trackid_mcs, :, :]), \
                              'pf_aspectratio': (['track', 'time', 'pfs'], data['pf_aspectratio'][trackid_mcs, :, :]), \
                              'pf_eccentricity': (['track', 'time', 'pfs'], data['pf_eccentricity'][trackid_mcs, :, :]), \
                              'pf_orientation': (['track', 'time', 'pfs'], data['pf_orientation'][trackid_mcs, :, :])}, \
                             coords={'track': (['track'], np.arange(1, len(trackid_mcs)+1)), \
                                    'time': (['time'], data.coords['time']), \
                                    'pfs': (['pfs'], data.coords['pfs']), \
                                    'cores': (['cores'], data.coords['cores']), \
                                    'mergesplit': (['mergesplit'], data.coords['mergesplit']), \
                                    'characters': (['characters'], data.coords['characters']), \
                                    'lifestages': (['lifestages'], np.arange(0, 5))}, \
                             attrs={'title':'Statistics of MCS defined using WRF precipitation features', \
                                    'source1': data.attrs['source1'], \
                                    'source2': data.attrs['source2'], \
                                    'description': data.attrs['description'], \
                                    'startdate': data.attrs['startdate'], \
                                    'enddate': data.attrs['enddate'], \
                                    'time_resolution_hour': data.attrs['time_resolution_hour_or_minutes'], \
                                    'mergedir_pixel_radius': data.attrs['mergdir_pixel_radius'], \
                                    'MCS_IR_area_km2': data.attrs['MCS_IR_area_thresh_km2'], \
                                    'MCS_IR_duration_hr': data.attrs['MCS_IR_duration_thresh_hr'], \
                                    'MCS_IR_eccentricity': data.attrs['MCS_IR_eccentricity_thres'], \
                                    'max_number_pfs': data.attrs['max_number_pfs'], \
                                    'MCS_PF_majoraxis_km': str(int(majoraxisthresh)), \
                                    'MCS_PF_duration_hr': str(int(durationthresh)), \
                                    'MCS_core_aspectratio': str(int(aspectratiothresh)), \
                                    'contact':'Katelyn Barber: katelyn.barber@pnnl.gov', \
                                    'created_on':time.ctime(time.time())})

    # Specify variable attributes
    output_data.track.attrs['description'] = 'Total number of tracked features'
    output_data.track.attrs['units'] = 'unitless'

    output_data.time.attrs['description'] = 'Maximum number of features in a given track'
    output_data.time.attrs['units'] = 'unitless'

    output_data.pfs.attrs['long_name'] = 'Maximum number of precipitation features in one cloud feature'
    output_data.pfs.attrs['units'] = 'unitless'

    output_data.cores.attrs['long_name'] = 'Maximum number of convective cores in a precipitation feature at one time'
    output_data.cores.attrs['units'] = 'unitless'

    output_data.mergesplit.attrs['long_name'] = 'Maximum number of mergers / splits at one time'
    output_data.mergesplit.attrs['units'] = 'unitless'

    output_data.characters.attrs['description'] = 'Number of characters in the date-time string'
    output_data.characters.attrs['units'] = 'unitless'

    #output_data.lifestages.attrs['description'] = 'Number of MCS life stages'
    #output_data.lifestages.attrs['units'] = 'unitless'

    output_data.mcs_length.attrs['long_name'] = 'Length of each MCS in each track'
    output_data.mcs_length.attrs['units'] = 'Temporal resolution of orginal data'

    output_data.mcs_type.attrs['long_name'] = 'Type of MCS'
    output_data.mcs_type.attrs['values'] = '1 = MCS, 2 = Squall line'
    output_data.mcs_type.attrs['units'] = 'unitless'

    output_data.pf_lifetime.attrs['long_name'] = 'Length of time in which precipitation is observed during each track'
    output_data.pf_lifetime.attrs['units'] = 'hr'

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

    output_data.base_time.attrs['standard_name'] = 'time'
    output_data.base_time.attrs['long_name'] = 'seconds since 01/01/1970 00:00 for each cloud in the mcs'

    output_data.datetimestring.attrs['long_name'] = 'date_time for each cloud in the mcs'
    output_data.datetimestring.attrs['units'] = 'unitless'

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

    output_data.cloudnumber.attrs['long_name'] = 'cloud number in the corresponding cloudid file of clouds in the mcs'
    output_data.cloudnumber.attrs['usage'] = 'to link this tracking statistics file with pixel-level cloudid files, use the cloudidfile and cloudnumber together to identify which cloud this current track and time is associated with'
    output_data.cloudnumber.attrs['units'] = 'unitless'

    output_data.mergecloudnumber.attrs['long_name'] = 'cloud number of small, short-lived clouds merging into the MCS'
    output_data.mergecloudnumber.attrs['usage'] = 'to link this tracking statistics file with pixel-level cloudid files, use the cloudidfile and cloudnumber together to identify which cloud this current track and time is associated with'
    output_data.mergecloudnumber.attrs['units'] = 'unitless'

    output_data.splitcloudnumber.attrs['long_name'] = 'cloud number of small, short-lived clouds splitting from the MCS'
    output_data.splitcloudnumber.attrs['usage'] = 'to link this tracking statistics file with pixel-level cloudid files, use the cloudidfile and cloudnumber together to identify which cloud this current track and time is associated with'
    output_data.splitcloudnumber.attrs['units'] = 'unitless'

    output_data.pf_frac.attrs['long_name'] = 'fraction of cold cloud shielf covered by NMQ mask'
    output_data.pf_frac.attrs['min_value'] = 0
    output_data.pf_frac.attrs['max_value'] = 1
    output_data.pf_frac.attrs['units'] = 'unitless'

    output_data.npf.attrs['long_name'] = 'number of precipitation features at a given time'
    output_data.npf.attrs['units'] = 'unitless'

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

    #output_data.pf_mcsstatus.attrs['description'] = 'Flag indicating if this time part of the MCS 1 = Yes, 0 = No'
    #output_data.pf_mcsstatus.attrs['units'] = 'unitless'

    #output_data.lifecycle_complete_flag.attrs['description'] = 'Flag indicating if this MCS has each element in the MCS life cycle'
    #output_data.lifecycle_complete_flag.attrs['units'] = 'unitless'

    #output_data.lifecycle_index.attrs['description'] = 'Time index when each phase of the MCS life cycle starts'
    #output_data.lifecycle_index.attrs['units'] = 'unitless'

    #output_data.lifecycle_stage.attrs['description'] = 'Each time in the MCS is labeled with a flag indicating its phase in the MCS lifecycle. 1 = Cloud only, 2 = Isolated convective cores, 3 = MCS genesis, 4 = MCS maturation, 5 = MCS decay'
    #output_data.lifecycle_stage.attrs['units'] = 'unitless'

    # Write netcdf file
    print('')
    print(statistics_outfile)

    output_data.to_netcdf(path=statistics_outfile, mode='w', format='NETCDF4_CLASSIC', unlimited_dims='track', \
                          encoding={'mcs_length': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                    'mcs_type': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                    'pf_lifetime': {'dtype': 'int','zlib':True, '_FillValue': -9999}, \
                                    'status': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                    'startstatus': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                    'endstatus': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                    'base_time': {'zlib':True}, \
                                    'boundary': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                    'interruptions': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                    'datetimestring': {'zlib':True}, \
                                    'meanlat': {'zlib':True, '_FillValue': np.nan}, \
                                    'meanlon': {'zlib':True, '_FillValue': np.nan}, \
                                    'core_area': {'zlib':True, '_FillValue': np.nan}, \
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
                                    #'pf_eccentricity': {'zlib':True, '_FillValue': np.nan}, \
                                    #'pf_mcsstatus': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                    #'lifecycle_complete_flag': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                    #'lifecycle_index': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                    #'lifecycle_stage': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}})






