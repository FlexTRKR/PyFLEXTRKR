# Purpose: Filter MCS using NMQ radar variables so that only MCSs statisfying radar thresholds are retained. The lifecycle of these robust MCS is also identified. Method similar to Coniglio et al (2010) MWR. 

# Author: Original IDL code written by Zhe Feng (zhe.feng@pnnl.gov), Python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov)

def filtermcs_mergedir_nmq(stats_path, pfstats_filebase, startdate, enddate, timeresolution, geolimits, majoraxisthresh, durationthresh, aspectratiothresh, lifecyclethresh, lengththresh, gapthresh):
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
    # pf_dbz40area - area covered by 40 dbZ echos in a precipitaiton feature in a MCS
    # pf_dbz50area - area covered by 50 dbZ echos in a precipitaiton feature in a MCS
    # pf_ccrainrate - mean rain rate of the largest few convective cores in a MCS
    # pf_sfrainrate - mean rain rate of the largest few stratiform regions in a MCS
    # pf_ccarea - mean area of the largest few convective cores in a MCS
    # pf_sfarea - mean area of the largest few stratiform regions in a MCS
    # pf_ccdbz10 - average height of the 10 dBZ echo top in the largest few convective cores in a MCS
    # pf_ccdbz20 - average height of the 20 dBZ echo top in the largest few convective cores in a MCS
    # pf_ccdbz30 - average height of the 30 dBZ echo top in the largest few convective cores in a MCS
    # pf_ccdbz40 - average height of the 40 dBZ echo top in the largest few convective cores in a MCS
    # pf_ncores - number of convective cores at each time during an MCS
    # pf_corelon - mean longitude of the convective cores in a MCS
    # pf_corelat - mean latitude of the convective cores in a MCS
    # pf_corearea - area of the convective cores in a MCS
    # pf_coremajoraxislength - major axis length of convective cores in a MCS
    # pf_coreminoraxislength - minor axis length of convective cores in a MCS
    # pf_coreaspectratio - aspect ratio of convective cores in a MCS
    # pf_coreccentricity - eccentricity of convective cores in a MCS
    # of_coreorientation - angular position of convective cores in a MCS
    # pf_coremaxdbz10 - maximum height of the 10 dBZ echo contour in convective cores
    # pf_coremaxdbz20 - maximum height of the 20 dBZ echo contour in convective cores
    # pf_coremaxdbz30 - maximum height of the 30 dBZ echo contour in convective cores
    # pf_coremaxdbz40 - maximum height of the 40 dBZ echo contour in convective cores
    # pf_coreavgdbz10 - average height of the 10 dBZ echo contour in convective cores
    # pf_coreavgdbz20 - average height of the 20 dBZ echo contour in convective cores
    # pf_coreavgdbz30 - average height of the 30 dBZ echo contour in convective cores
    # pf_coreavgdbz40 - average height of the 40 dBZ echo contour in convective cores


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
    pf_dbz50area = data['pf_dbz50area'].data

    pf_meansfarea = data['pf_sfarea']

    pf_ccarea = data['pf_corearea'].data
    pf_ccmajoraxis = data['pf_coremajoraxislength'].data
    pf_ccaspectratio = data['pf_coreaspectratio'].data

    time_res = float(data.attrs['time_resolution_hour'])
    mcs_ir_areathresh = float(data.attrs['MCS_IR_area_thresh_km2'])
    mcs_ir_durationthresh = float(data.attrs['MCS_IR_duration_thresh_hr'])
    mcs_ir_eccentricitythresh = float(data.attrs['MCS_IR_eccentricity_thres'])

    ##################################################
    # Initialize matrices
    trackid_mcs = []
    trackid_nonmcs = []

    pf_mcstype = np.ones(ntracks, dtype=int)*-9999
    pf_mcsstatus = np.ones((ntracks, ntimes), dtype=int)*-9999
    pf_cctype = np.ones((ntracks, ntimes), dtype=int)*-9999

    ###################################################
    # Loop through each track
    for nt in range(0, ntracks):
        print('Track # ' + str(nt))

        ############################################
        # Isolate data from this track 
        ilength = np.copy(ir_tracklength[nt]).astype(int)

        # Get the largest precipitation (1st entry in 3rd dimension)
        ipf_majoraxis = np.copy(pf_majoraxis[nt, 0:ilength, 0])
        ipf_dbz50area = np.copy(pf_dbz50area[nt, 0:ilength, 0])

        # Get the cooresponding convective core and stratiform region data (use the largest feature (1st entry in 3rd dimenstion), when applicable) 
        ipf_meansfarea = np.copy(pf_meansfarea[nt, 0:ilength])
        ipf_ccmajoraxis = np.copy(pf_ccmajoraxis[nt, 0:ilength, 0])
        ipf_ccaspectratio = np.copy(pf_ccaspectratio[nt, 0:ilength, 0])

        ######################################################
        # Apply radar defined MCS criteria

        # Apply PF major axis length > thresh and contains echo >= 50 dbZ criteria
        ipfmcs = np.array(np.where((ipf_majoraxis > majoraxisthresh) & (ipf_dbz50area > 0)))[0, :]
        nipfmcs = len(ipfmcs)

        if nipfmcs > 0 :
            # Apply duration threshold to entire time period
            if nipfmcs*time_res > durationthresh:

                # Find continuous duration indices
                groups = np.split(ipfmcs, np.where(np.diff(ipfmcs) != gapthresh)[0]+1)
                nbreaks = len(groups)

                for igroup in range(0, nbreaks):

                    ############################################################
                    # Determine if each group statisfies duration threshold
                    igroup_indices = np.array(np.copy(groups[igroup][:]))
                    nigroup = len(igroup_indices)

                    # Group satisfies duration threshold
                    if np.multiply(len(groups[igroup][:]), time_res) > durationthresh:

                        # Get radar variables for this group
                        igroup_duration = len(groups[igroup])*time_res
                        igroup_pfmajoraxis = np.copy(ipf_majoraxis[igroup_indices])
                        igroup_ccmajoraxis = np.copy(ipf_ccmajoraxis[igroup_indices])
                        igroup_ccaspectratio = np.copy(ipf_ccaspectratio[igroup_indices])

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
        print('Number of robust MCS: ' + str(int(nmcs)))

    # Isolate data associated with robust MCS
    ir_tracklength = ir_tracklength[trackid_mcs]

    #pf_mcstype = pf_mcstype[trackid_mcs]
    pf_mcsstatus = pf_mcsstatus[trackid_mcs, :]
    pf_majoraxis = pf_majoraxis[trackid_mcs, :, :]
    pf_area = pf_area[trackid_mcs, :, :]

    pf_ccmajoraxis = pf_ccmajoraxis[trackid_mcs, :, :]
    pf_ccarea = pf_ccarea[trackid_mcs, :, :]
    #pf_cctype = pf_cctype[trackid_mcs, :]

    pf_meansfarea = pf_meansfarea[trackid_mcs, :]

    # Determine how long MCS track criteria is statisfied
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
        # Initialize arrays
        cycle_complete = np.ones(nmcs, dtype=int)*-9999
        cycle_stage = np.ones((nmcs, ntimes), dtype=int)*-9999
        cycle_index = np.ones((nmcs, 5), dtype=int)*-9999

        mcs_basetime = np.empty((nmcs, ntimes), dtype='datetime64[s]')

        # Loop through each mcs
        for ilm in range(0, nlongmcs):
            # Initialize arrays
            ilm_index = np.ones(5, dtype=int)*-9999
            ilm_cycle = np.ones(ntimes, dtype=int)*-9999

            # Isolate data from this track
            ilm_irtracklength = np.copy(ir_tracklength[ilongmcs[ilm]]).astype(int)
            ilm_pfcctype = np.copy(pf_cctype[ilongmcs[ilm], 0:ilm_irtracklength])
            ilm_pfccmajoraxis = np.copy(pf_ccmajoraxis[ilongmcs[ilm], 0:ilm_irtracklength, :])
            ilm_pfccarea = np.copy(pf_ccarea[ilongmcs[ilm], 0:ilm_irtracklength, :])
            ilm_meansfarea = np.copy(pf_meansfarea[ilongmcs[ilm], 0:ilm_irtracklength])
            ilm_pfarea = np.copy(pf_area[ilongmcs[ilm], 0:ilm_irtracklength, 0])

            ilm_maxpfccmajoraxis = np.nanmax(ilm_pfccmajoraxis, axis=1)
            ilm_maxpfccarea = np.nanmax(ilm_pfccarea, axis=1)

            # Get basetime
            TEMP_basetime = np.array([pd.to_datetime(data['basetime'][trackid_mcs[ilongmcs[ilm]], 0:ilm_irtracklength].data, unit='s')])
            mcs_basetime[ilm, 0:ilm_irtracklength] = TEMP_basetime

            ##################################################################
            # Find indices of when convective line present and absent and when stratiform present

            # Find times with convective core area > 0
            iccarea = np.array(np.where(ilm_maxpfccarea > 0))[0, :]
            iccarea_groups = np.split(iccarea, np.where(np.diff(iccarea) > 2)[0]+1)
            if len(iccarea) > 0 and len(iccarea_groups) > 1:
                grouplength = np.empty(len(iccarea_groups))
                for igroup in range(0, len(iccarea_groups)):
                    grouplength[igroup] = len(iccarea_groups[igroup][:])
                maxgroup = np.nanargmax(grouplength)
                iccarea = iccarea_groups[maxgroup][:]
            elif len(iccarea) > 0 :
                iccarea = np.arange(iccarea[0], iccarea[-1]+1)
            nccarea = len(iccarea)

            # Find times with convective major axis length greater than 100 km
            iccline = np.array(np.where(ilm_maxpfccmajoraxis > 100))[0, :]
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

            # Find times with convective major axis length greater than 100 km and stratiform area greater than the median amount of stratiform
            #ilm_meansfarea[ilm_meansfarea == fillvalue] = np.nan
            isfarea = np.array(np.where((ilm_maxpfccmajoraxis > 100) & (ilm_meansfarea > np.nanmean(ilm_meansfarea))))[0, :]
            isfarea_groups = np.split(isfarea, np.where(np.diff(isfarea) > 2)[0]+1)
            if len(isfarea) > 0 and len(isfarea_groups) > 1:
                grouplength = np.empty(len(isfarea_groups))
                for igroup in range(0, len(isfarea_groups)):
                    grouplength[igroup] = len(isfarea_groups[igroup][:])
                maxgroup = np.nanargmax(grouplength)
                isfarea = isfarea_groups[maxgroup][:]
            elif len(isfarea) > 0 :
                isfarea = np.arange(isfarea[0], isfarea[-1]+1)
            nsfarea = len(isfarea)

            # Find times with convective major axis length less than 100 km
            if nsfarea > 0:
                inoccline = np.array(np.where(ilm_maxpfccmajoraxis < 100))[0, :]
                inoccline = inoccline[np.where((inoccline > isfarea[-1]) & (inoccline > iccline[-1]))]
                inoccline_groups = np.split(inoccline, np.where(np.diff(inoccline) > 2)[0]+1)
                if len(inoccline) > 0 and len(inoccline_groups) > 1:
                    grouplength = np.empty(len(inoccline_groups))
                    for igroup in range(0, len(inoccline_groups)):
                        grouplength[igroup] = len(inoccline_groups[igroup][:])
                    maxgroup = np.nanargmax(grouplength)
                    inoccline = inoccline_groups[maxgroup][:]
                elif len(inoccline) > 0:
                    inoccline = np.arange(inoccline[0], inoccline[-1]+1)
                nnoccline = len(inoccline)

            ###############################################################################
            # Classify cloud only stage 

            # Cloud only stage
            if nccarea > 0:
                # If first convective time is after the first cloud time, label all hours before the convective core appearance time as preconvective
                if iccarea[0] > 0 and iccarea[0] < ilm_irtracklength-1:
                    ilm_index[0] = 0 # Start of cloud only
                    ilm_cycle[0:iccarea[0]] = 1 # Time period of cloud only

                ilm_index[1] = iccarea[0] # Start of unorganized convective cells 

            # If convective line exists
            if nccline > 1:
                # If the convective line occurs after the first storm time (use second index since convective line must be around for one hour prior to classifying as genesis)
                # Label when convective cores first appear, but are not organized into a line
                if iccline[1] > iccarea[0]:
                    ilm_index[2] = iccline[1] # Start of organized convection
                    ilm_cycle[iccarea[0]:iccline[1]] = 2 # Time period of unorganzied convective cells
                else:
                    sys.exit('Check convective line in track ' + str(int(ilongmcs[ilm])))

                if nsfarea > 0:
                    # Label MCS genesis. Test if stratiform area time is two timesteps after the convective line and two time steps before the last time of the cloud track
                    if isfarea[0] > iccline[1] + 2:
                        ilm_index[3] = isfarea[0] # Start of mature MCS
                        ilm_cycle[iccline[1]:isfarea[0]] = 3 # Time period of organized cells before maturation

                        ilm_cycle[isfarea[0]:isfarea[-1]+1] = 4 # Time period of mature mcs
                    else:
                        matureindex = isfarea[np.array(np.where(isfarea == iccline[1] + 2))[0, :]]
                        if len(matureindex) > 0:
                            ilm_index[3] = np.copy(matureindex[0])
                            ilm_cycle[iccline[1]:matureindex[0]] = 3 # Time period of organized cells before maturation

                            ilm_cycle[matureindex[0]:isfarea[-1]+1] = 4 # Time period of mature mcs

                            #if isfarea[0] > iccline[1] + 2
                            #    ilm_index[3] = isfarea[0] # Start of mature MCS
                            #    ilm_cycle[iccline[1]:isfarea[0]] = 3 # Time period of organized cells before maturation
                        
                            #    ilm_cycle[isfarea[0]:isfarea[-1]+1] = 4 # Time period of mature mcs
                            #else:
                            #    if nsfarea > 3:
                            #        ilm_index[3] = isfarea[3] # Start of mature MCS
                            #        ilm_cycle[iccline[1]:isfarea[3]] = 3 # Time period of organized cells before maturation
                    
                            #        ilm_cycle[isfarea[3]:isfarea[-1]+1] = 4 # Time period of mature MCS

                            # Label dissipating times. Buy default this is all times after the mature stage
                            ilm_index[4] =  isfarea[-1]+1 
                            ilm_cycle[isfarea[-1]+1:ilm_irtracklength+1] = 5 # Time period of dissipation

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
                              'ccs_area': (['track', 'time'], data['ccs_area'][trackid_mcs, :]), \
                              'cloudnumber': (['track', 'time'], data['cloudnumber'][trackid_mcs, :]), \
                              'mergecloudnumber': (['track', 'time', 'mergesplit'], data['mergecloudnumber'][trackid_mcs, :, :]), \
                              'splitcloudnumber': (['track', 'time', 'mergesplit'], data['splitcloudnumber'][trackid_mcs, :, :]), \
                              #'pf_mcstype': (['tracks'], pf_mcstype), \
                              'pf_mcsstatus': (['track', 'time'], pf_mcsstatus), \
                              #'pf_cctype': (['track', 'time'], pf_cctype), \
                              'lifecycle_complete_flag': (['track'], cycle_complete), \
                              'lifecycle_index': (['track', 'lifestages'], cycle_index), \
                              'lifecycle_stage': (['track', 'time'], cycle_stage), \
                              'nmq_frac': (['track', 'time'], data['nmq_frac'][trackid_mcs]), \
                              'npf': (['track', 'time'], data['npf'][trackid_mcs]), \
                              'pf_area': (['track', 'time', 'pfs'], data['pf_area'][trackid_mcs, :, :]), \
                              'pf_lon': (['track', 'time', 'pfs'], data['pf_lon'][trackid_mcs, :, :]), \
                              'pf_lat': (['track', 'time', 'pfs'], data['pf_lat'][trackid_mcs, :, :]), \
                              'pf_rainrate': (['track', 'time', 'pfs'], data['pf_rainrate'][trackid_mcs, :, :]), \
                              'pf_skewness': (['track', 'time', 'pfs'], data['pf_skewness'][trackid_mcs, :, :]), \
                              'pf_majoraxislength': (['track', 'time', 'pfs'], data['pf_majoraxislength'][trackid_mcs, :, :]), \
                              'pf_minoraxislength': (['track', 'time', 'pfs'], data['pf_minoraxislength'][trackid_mcs, :, :]), \
                              'pf_aspectratio': (['track', 'time', 'pfs'], data['pf_aspectratio'][trackid_mcs, :, :]), \
                              'pf_eccentricity': (['track', 'time', 'pfs'], data['pf_eccentricity'][trackid_mcs, :, :]), \
                              'pf_orientation': (['track', 'time', 'pfs'], data['pf_orientation'][trackid_mcs, :, :]), \
                              'pf_dbz40area': (['track', 'time', 'pfs'], data['pf_dbz40area'][trackid_mcs, :, :]), \
                              'pf_dbz50area': (['track', 'time', 'pfs'], data['pf_dbz50area'][trackid_mcs, :, :]), \
                              'pf_ccrainrate': (['track', 'time'], data['pf_ccrainrate'][trackid_mcs, :]), \
                              'pf_sfrainrate': (['track', 'time'], data['pf_sfrainrate'][trackid_mcs, :]), \
                              'pf_ccarea': (['track', 'time'], data['pf_ccarea'][trackid_mcs, :]), \
                              'pf_sfarea': (['track', 'time'], data['pf_sfarea'][trackid_mcs, :]), \
                              'pf_ccdbz10': (['track', 'time'], data['pf_ccdbz10'][trackid_mcs, :]), \
                              'pf_ccdbz20': (['track', 'time'], data['pf_ccdbz20'][trackid_mcs, :]), \
                              'pf_ccdbz30': (['track', 'time'], data['pf_ccdbz30'][trackid_mcs, :]), \
                              'pf_ccdbz40': (['track', 'time'], data['pf_ccdbz40'][trackid_mcs, :]), \
                              'pf_ncores': (['track', 'time'], data['pf_ncores'][trackid_mcs, :]), \
                              'pf_corelon': (['track', 'time', 'cores'], data['pf_corelon'][trackid_mcs, :, :]), \
                              'pf_corelat': (['track', 'time', 'cores'], data['pf_corelat'][trackid_mcs, :, :]), \
                              'pf_corearea': (['track', 'time', 'cores'], data['pf_corearea'][trackid_mcs, :, :]), \
                              'pf_coremajoraxislength': (['track', 'time', 'cores'], data['pf_coremajoraxislength'][trackid_mcs, :, :]), \
                              'pf_coreminoraxislength': (['track', 'time', 'cores'], data['pf_coreminoraxislength'][trackid_mcs, :, :]), \
                              'pf_coreaspectratio': (['track', 'time', 'cores'], data['pf_coreaspectratio'][trackid_mcs, :, :]), \
                              'pf_coreorientation': (['track', 'time', 'cores'], data['pf_coreorientation'][trackid_mcs, :, :]), \
                              'pf_coreeccentricity': (['track', 'time', 'cores'], data['pf_coreeccentricity'][trackid_mcs, :, :]), \
                              'pf_coremaxdbz10': (['track', 'time', 'cores'], data['pf_coremaxdbz10'][trackid_mcs, :, :]), \
                              'pf_coremaxdbz20': (['track', 'time', 'cores'], data['pf_coremaxdbz20'][trackid_mcs, :, :]), \
                              'pf_coremaxdbz30': (['track', 'time', 'cores'], data['pf_coremaxdbz30'][trackid_mcs, :, :]), \
                              'pf_coremaxdbz40': (['track', 'time', 'cores'], data['pf_coremaxdbz40'][trackid_mcs, :, :]), \
                              'pf_coreavgdbz10': (['track', 'time', 'cores'], data['pf_coreavgdbz10'][trackid_mcs, :, :]), \
                              'pf_coreavgdbz20': (['track', 'time', 'cores'], data['pf_coreavgdbz20'][trackid_mcs, :, :]), \
                              'pf_coreavgdbz30': (['track', 'time', 'cores'], data['pf_coreavgdbz30'][trackid_mcs, :, :]), \
                              'pf_coreavgdbz40': (['track', 'time', 'cores'], data['pf_coreavgdbz40'][trackid_mcs, :, :])}, \
                             coords = {'track': (['track'], np.arange(1, len(trackid_mcs)+1)), \
                                       'time': (['time'], data.coords['time']), \
                                       'pfs': (['pfs'], data.coords['pfs']), \
                                       'cores': (['cores'], data.coords['cores']), \
                                       'mergesplit': (['mergesplit'], data.coords['mergesplit']), \
                                       'characters': (['characters'], data.coords['characters']), \
                                       'lifestages': (['lifestages'], np.arange(0, 5))}, \
                             attrs={'title':'Statistics of MCS definedusing NMQ precipitation features', \
                                    'source1': data.attrs['source1'], \
                                    'source2': data.attrs['source2'], \
                                    'description': data.attrs['description'], \
                                    'startdate': data.attrs['startdate'], \
                                    'enddate': data.attrs['enddate'], \
                                    'time_resolution_hour': data.attrs['time_resolution_hour'], \
                                    'mergedir_pixel_radius': data.attrs['mergdir_pixel_radius'], \
                                    'MCS_IR_area_km2': data.attrs['MCS_IR_area_thresh_km2'], \
                                    'MCS_IR_duration_hr': data.attrs['MCS_IR_duration_thresh_hr'], \
                                    'MCS_IR_eccentricity': data.attrs['MCS_IR_eccentricity_thres'], \
                                    'max_number_pfs': data.attrs['max_number_pfs'], \
                                    'MCS_PF_majoraxis_km': str(int(majoraxisthresh)), \
                                    'MCS_PF_duration_hr': str(int(durationthresh)), \
                                    'MCS_core_aspectratio': str(int(aspectratiothresh)), \
                                    'contact':'Hannah C Barnes: hannah.barnes@pnnl.gov', \
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

    output_data.lifestages.attrs['description'] = 'Number of MCS life stages'
    output_data.lifestages.attrs['units'] = 'unitless'

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

    output_data.pf_dbz40area.attrs['long_name'] = 'area of the precipitation feature with column maximum reflectivity >= 40 dBZ at a given time'
    output_data.pf_dbz40area.attrs['units'] = 'km^2'

    output_data.pf_dbz50area.attrs['long_name'] = 'area of the precipitation feature with column maximum reflectivity >= 50 dBZ at a given time'
    output_data.pf_dbz50area.attrs['units'] = 'km^2'

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

    output_data.nmq_frac.attrs['long_name'] = 'fraction of cold cloud shielf covered by NMQ mask'
    output_data.nmq_frac.attrs['min_value'] = 0
    output_data.nmq_frac.attrs['max_value'] = 1
    output_data.nmq_frac.attrs['units'] = 'unitless'

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

    output_data.pf_ccrainrate.attrs['long_name'] = 'mean rain rate of the largest several the convective cores at a given time'
    output_data.pf_ccrainrate.attrs['units'] = 'mm/hr'

    output_data.pf_sfrainrate.attrs['long_name'] = 'mean rain rate in the largest several statiform regions at a given time'
    output_data.pf_sfrainrate.attrs['units'] = 'mm/hr'

    output_data.pf_ccarea.attrs['long_name'] = 'total area of the largest several convective cores at a given time'
    output_data.pf_ccarea.attrs['units'] = 'km^2'

    output_data.pf_sfarea.attrs['long_name'] = 'total area of the largest several stratiform regions at a given time'
    output_data.pf_sfarea.attrs['units'] = 'km^2'

    output_data.pf_ccdbz10.attrs['long_name'] = 'mean 10 dBZ echo top height of the largest several convective cores at a given time'
    output_data.pf_ccdbz10.attrs['units'] = 'km'

    output_data.pf_ccdbz20.attrs['long_name'] = 'mean 20 dBZ echo top height of the largest several convective cores at a given time'
    output_data.pf_ccdbz20.attrs['units'] = 'km'

    output_data.pf_ccdbz30.attrs['long_name'] = 'mean 30 dBZ echo top height the largest several convective cores at a given time'
    output_data.pf_ccdbz30.attrs['units'] = 'km'

    output_data.pf_ccdbz40.attrs['long_name'] = 'mean 40 dBZ echo top height of the largest several convective cores at a given time'
    output_data.pf_ccdbz40.attrs['units'] = 'km'

    output_data.pf_ncores.attrs['long_name'] = 'number of convective cores (radar identified) in a precipitation feature at a given time'
    output_data.pf_ncores.attrs['units'] = 'unitless'

    output_data.pf_corelon.attrs['standard_name'] = 'longitude'
    output_data.pf_corelon.attrs['long_name'] = 'mean longitude of each convective core in a precipitation features at the given time'
    output_data.pf_corelon.attrs['units'] = 'degrees'

    output_data.pf_coreeccentricity.attrs['long_name'] = 'eccentricity of each convective core in the precipitation feature at a given time'
    output_data.pf_coreeccentricity.attrs['min_value'] = 0
    output_data.pf_coreeccentricity.attrs['max_value'] = 1
    output_data.pf_coreeccentricity.attrs['units'] = 'unitless'

    output_data.pf_orientation.attrs['long_name'] = 'orientation of the major axis of each precipitation feature at a given time'
    output_data.pf_orientation.attrs['units'] = 'degrees clockwise from vertical'
    output_data.pf_orientation.attrs['min_value'] = 0
    output_data.pf_orientation.attrs['max_value'] = 360

    output_data.pf_corelat.attrs['standard_name'] = 'latitude'
    output_data.pf_corelat.attrs['long_name'] = 'mean latitude of each convective core in a precipitation features at the given time'
    output_data.pf_corelat.attrs['units'] = 'degrees'

    output_data.pf_corearea.attrs['long_name'] = 'area of each convective core in the precipitatation feature at the given time'
    output_data.pf_corearea.attrs['units'] = 'km^2'

    output_data.pf_coremajoraxislength.attrs['long_name'] = 'major axis length of each convective core in the precipitation feature at a given time'
    output_data.pf_coremajoraxislength.attrs['units'] = 'km'

    output_data.pf_coreminoraxislength.attrs['long_name'] = 'minor axis length of each convective core in the precipitation feature at a given time'
    output_data.pf_coreminoraxislength.attrs['units'] = 'km'

    output_data.pf_coreaspectratio.attrs['long_name'] = 'aspect ratio (major / minor axis length) of each convective core in the precipitation feature at a given time'
    output_data.pf_coreaspectratio.attrs['units'] = 'unitless'

    output_data.pf_coreeccentricity.attrs['long_name'] = 'eccentricity of each convective core in the precipitation feature at a given time'
    output_data.pf_coreeccentricity.attrs['min_value'] = 0
    output_data.pf_coreeccentricity.attrs['max_value'] = 1
    output_data.pf_coreeccentricity.attrs['units'] = 'unitless'

    output_data.pf_coreorientation.attrs['long_name'] = 'orientation of the major axis of each convective core in the precipitation feature at a given time'
    output_data.pf_coreorientation.attrs['units'] = 'degrees clockwise from vertical'
    output_data.pf_coreorientation.attrs['min_value'] = 0
    output_data.pf_coreorientation.attrs['max_value'] = 360

    output_data.pf_coremaxdbz10.attrs['long_name'] = 'maximum 10-dBZ echo top height in each convective core in the precipitation features at a given time'
    output_data.pf_coremaxdbz10.attrs['units'] = 'km'

    output_data.pf_coremaxdbz20.attrs['long_name'] = 'maximum 20-dBZ echo top height in each convective core in the precipitation features at a given time'
    output_data.pf_coremaxdbz20.attrs['units'] = 'km'

    output_data.pf_coremaxdbz30.attrs['long_name'] = 'maximum 30-dBZ echo top height in each convective core in the precipitation features at a given time'
    output_data.pf_coremaxdbz30.attrs['units'] = 'km'

    output_data.pf_coremaxdbz40.attrs['long_name'] = 'maximum 40-dBZ echo top height in each convective core in the precipitation features at a given time'
    output_data.pf_coremaxdbz40.attrs['units'] = 'km'

    output_data.pf_coreavgdbz10.attrs['long_name'] = 'mean 10-dBZ echo top height in each convective core in the precipitation features at a given time'
    output_data.pf_coreavgdbz10.attrs['units'] = 'km'

    output_data.pf_coreavgdbz20.attrs['long_name'] = 'mean 20-dBZ echo top height in each convective core in the precipitation features at a given time'
    output_data.pf_coreavgdbz20.attrs['units'] = 'km'

    output_data.pf_coreavgdbz30.attrs['long_name'] = 'mean 30-dBZ echo top height in each convective core in the precipitation features at a given time'
    output_data.pf_coreavgdbz30.attrs['units'] = 'km'

    output_data.pf_coreavgdbz40.attrs['long_name'] = 'mean 40-dBZ echo top height in each convective core in the precipitation features at a given time'
    output_data.pf_coreavgdbz40.attrs['units'] = 'km'

    #output_data.pf_mcstype.attrs['description'] = 'Flag indicating type of MCS. 1 = Squall, 2 = Non-Squall'
    #output_data.pf_mcstype.attrs['units'] = 'unitless'

    #output_data.pf_cctype.attrs['description'] = 'Flag indicating type of MCS. 1 = Squall, 2 = Non-Squall'
    #output_data.pf_cctype.attrs['units'] = 'unitless'

    output_data.pf_mcsstatus.attrs['description'] = 'Flag indicating if this time part of the MCS 1 = Yes, 0 = No'
    output_data.pf_mcsstatus.attrs['units'] = 'unitless'

    output_data.lifecycle_complete_flag.attrs['description'] = 'Flag indicating if this MCS has each element in the MCS life cycle'
    output_data.lifecycle_complete_flag.attrs['units'] = 'unitless'

    output_data.lifecycle_index.attrs['description'] = 'Time index when each phase of the MCS life cycle starts'
    output_data.lifecycle_index.attrs['units'] = 'unitless'

    output_data.lifecycle_stage.attrs['description'] = 'Each time in the MCS is labeled with a flag indicating its phase in the MCS lifecycle. 1 = Cloud only, 2 = Isolated convective cores, 3 = MCS genesis, 4 = MCS maturation, 5 = MCS decay'
    output_data.lifecycle_stage.attrs['units'] = 'unitless'

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
                                    'base_time': {'zlib':True, 'units': 'seconds since 1970-01-01'}, \
                                    'boundary': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                    'interruptions': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                    'datetimestring': {'zlib':True}, \
                                    'meanlat': {'zlib':True, '_FillValue': np.nan}, \
                                    'meanlon': {'zlib':True, '_FillValue': np.nan}, \
                                    'core_area': {'zlib':True, '_FillValue': np.nan}, \
                                    'ccs_area': {'zlib':True, '_FillValue': np.nan}, \
                                    'cloudnumber': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                    'mergecloudnumber': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                    'splitcloudnumber': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                    'nmq_frac': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_area': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_lon': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_lat': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_dbz40area': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_dbz50area': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_rainrate': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_skewness': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_majoraxislength': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_minoraxislength': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_aspectratio': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_orientation': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_eccentricity': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_ccarea': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_sfarea': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_ccrainrate': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_sfrainrate': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_ccdbz40': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_ncores': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                    'pf_corelon': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_corelat': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_corearea': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_coremajoraxislength': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_minoraxislength': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_coreaspectratio': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_coreorientation': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_coreeccentricity': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_coremaxdbz10': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_coremaxdbz20': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_coremaxdbz30': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_coremaxdbz40': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_coreavgdbz10': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_coreavgdbz20': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_coreavgdbz30': {'zlib':True, '_FillValue': np.nan}, \
                                    'pf_coreavgdbz40': {'zlib':True, '_FillValue': np.nan}, \
                                    #'pf_mcstype': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                    #'pf_cctype': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                    'pf_mcsstatus': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                    'lifecycle_complete_flag': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                    'lifecycle_index': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                                    'lifecycle_stage': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}})






