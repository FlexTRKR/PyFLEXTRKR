# Purpose: Filter MCS using NMQ radar variables so that only robust MCSs are included.

# Comments: Method similar to Coniglio et al (2010) MWR. 

# Author: Original IDL code written by Zhe Feng (zhe.feng@pnnl.gov), Python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov)

def filtermcs_mergedir_nmq(stats_path, pfstats_filebase, startdate, enddate, majoraxisthresh, durationthresh, aspectratiothresh, lifecyclethresh, lengththresh, gapthresh):
    ######################################################
    # Import modules
    import numpy as np
    import xarray as xr
    import sys

    np.set_printoptions(threshold=np.inf)

    ######################################################
    # Set constants
    fillvalue = -9999

    ######################################################
    # Load mergedir mcs and pf data
    mergedirpf_statistics_file = stats_path + pfstats_filebase + startdate + '_' + enddate + '.nc'

    data = xr.open_dataset(mergedirpf_statistics_file, autoclose=True)
    ntracks = np.nanmax(data.coords['track'])
    ntimes = np.nanmax(data.coords['time'])

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

    pf_mcstype = np.ones(ntracks, dtype=int)*fillvalue
    pf_mcsstatus = np.ones((ntracks, ntimes), dtype=float)*fillvalue
    pf_cctype = np.ones((ntracks, ntimes), dtype=float)*fillvalue

    ###################################################
    # Loop through each track
    for nt in range(0, ntracks):

        ############################################
        # Isolate data from this track 
        ilength = np.copy(ir_tracklength[nt])

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
                        igroup_dbz40area = np.copy(ipf_dbz40area[igroup_indices])
                        igroup_ccmajoraxis = np.copy(ipf_ccmajoraxis[igroup_indices])
                        igroup_ccaspectratio = np.copy(ipf_ccaspectratio[igroup_indices])

                        # Label this period as an mcs
                        pf_mcsstatus[nt, igroup_indices] = 1

                        # Determine type of mcs (squall or non-squall)
                        isquall = np.array(np.where(igroup_ccaspectratio > aspectratiothresh))[0, :]
                        nisquall = len(isquall)

                        if nisquall > 0:
                            # Label as squall
                            pf_mcstype[nt] = 1
                            pf_cctype[nt, igroup_indices[isquall]] = 1
                        else:
                            # Label as non-squall
                            pf_mcstype[nt] = 2
                            pf_cctype[nt, igroup_indices[isquall]] = 2

            # Group does not satistfy duration threshold
            else:
                trackid_nonmcs = np.append(trackid_nonmcs, nt)
                
    # Isolate tracks that have robust MCS
    trackid_mcs = np.array(np.where(pf_mcstype > 0))[0, :]
    nmcs = len(trackid_mcs)

    # Stop code if not robust MCS present
    if nmcs == 0:
        sys.exit('No MCS found!')
    else:
        print('Number of robust MCS: ' + str(int(nmcs)))

    # Isolate data associated with robust MCS
    ir_tracklength = ir_tracklength[trackid_mcs]

    pf_mcstype = pf_mcstype[trackid_mcs]
    pf_mcsstatus = pf_mcsstatus[trackid_mcs, :]
    pf_majoraxis = pf_majoraxis[trackid_mcs, :, :]
    pf_area = pf_area[trackid_mcs, :, :]

    pf_ccmajoraxis = pf_ccmajoraxis[trackid_mcs, :, :]
    pf_ccarea = pf_ccarea[trackid_mcs, :, :]
    pf_cctype = pf_cctype[trackid_mcs, :]

    pf_meansfarea = pf_meansfarea[trackid_mcs, :]

    # Determine how long MCS track criteria is statisfied
    TEMP_mcsstatus = np.copy(pf_mcsstatus)
    TEMP_mcsstatus[TEMP_mcsstatus == fillvalue] = np.nan
    mcs_length = np.nansum(TEMP_mcsstatus, axis=1)

    # Get lifetime when a significant precip feature is present
    pf_maxmajoraxis = np.nanmax(pf_majoraxis, axis=2)
    pf_maxmajoraxis[pf_maxmajoraxis < lengththresh] = 0
    pf_maxmajoraxis[pf_maxmajoraxis > lengththresh] = 1
    pf_length = np.nansum(pf_maxmajoraxis, axis=1)

    ########################################################
    # Definite life cycle stages. Based on Coniglio et al. (2010) MWR.
    # Preconvective: first hour after convective core occurs
    # Genesis: First hour after convective line exceeds 100 km
    # Mature: Near continuous line with well defined stratiform precipitation. 2 hours after genesis state and 2 hours before decay stage
    # Dissipiation: First hour hafter convective line is no longer observed

    # Process only MCSs that last at least 8 hours
    lifetime = np.multiply(ir_tracklength, time_res)
    ilongmcs = np.array(np.where(lifetime >= lifecyclethresh))[0, :]
    nlongmcs = len(ilongmcs)

    if nlongmcs > 0:
        # Initialize arrays
        lifecycle_complete = np.ones(nmcs, dtype=float)*fillvalue
        lifecycle_stage = np.ones((ntimes, nmcs), dtype=float)*fillvalue
        lifecycle_index = np.ones((4, nmcs), dtype=int)*fillvalue

        # Loop through each mcs
        for ilm in range(0, nlongmcs):
            # Initialize arrays
            ilm_index = np.ones(4, dtype=float)*fillvalue
            ilm_lifecycle = np.ones(ntimes, dtype=float)*fillvalue

            # Isolate data from this track
            ilm_irtracklength = np.copy(ir_tracklength[ilongmcs[ilm]])
            ilm_pfcctype = np.copy(pf_cctype[ilongmcs[ilm], 0:ilm_irtracklength])
            ilm_pfccmajoraxis = np.copy(pf_ccmajoraxis[ilongmcs[ilm], 0:ilm_irtracklength, :])
            ilm_pfccarea = np.copy(pf_ccarea[ilongmcs[ilm], 0:ilm_irtracklength, :])
            ilm_meansfarea = np.copy(pf_meansfarea[ilongmcs[ilm], 0:ilm_irtracklength])
            ilm_pfarea = np.copy(pf_area[ilongmcs[ilm], 0:ilm_irtracklength, 0])

            ilm_maxpfccmajoraxis = np.nanmax(ilm_pfccmajoraxis, axis=1)
            ilm_maxpfccarea = np.nanmax(ilm_pfccarea, axis=1)

            ##################################################################
            # Classify preconvective time

            # Find times with convective core area > 0
            iccarea = np.array(np.where(ilm_maxpfccarea > 0))[0, :]
            nccarea = len(iccarea)

            if nccarea > 0:
                # If first convective time is after the first cloud time, label all hours before the convective core appearance time as preconvective
                if iccarea[0] > 0 and iccarea[0] < ilm_irtracklength-1:
                    ilm_index[0] = iccarea[0]
                    ilm_lifecycle[0:iccarea[0]] = 1
                else:
                    ilm_index[0] = 0
                    ilm_lifecycle[0] = 1

            ################################################################
            # Find indices of when convective line present and absent and when stratiform present

            # Find times with convective major axis length greater than 100 km
            iccline = np.array(np.where(ilm_maxpfccmajoraxis > 100))[0, :]
            nccline = len(iccline)

            # Find times with convective major axis length greater than 100 km and stratiform area greater than the median amount of stratiform
            ilm_meansfarea[ilm_meansfarea == fillvalue] = np.nan
            isfarea = np.array(np.where((ilm_maxpfccmajoraxis > 100) & (ilm_meansfarea > np.nanmean(ilm_meansfarea))))
            nsfarea = len(isfarea)

            # Find times with convective major axis length less than 100 km
            inoccline = np.array(np.where(ilm_maxpfccmajoraxis < 100))[0, :]
            nnoccline = len(inoccline)

            ##################################################################
            # If convective line exists
            if nccline > 1:
                ####################################################
                # Label genesis
                # If the convective line occurs after the first storm time (use second index since convective line must be around for one hour prior to classifying as genesis)
                if iccline[1] > iccarea[0]:
                    ilm_index[1] = iccline[1]
                    ilm_lifecyle[iccarea[0]+1:iccline[1]] = 2
                else:
                    sys.exit('Check convective line in track ' + str(int(ilongmcs[ilm])))

                if nsfarea > 0:
                    # Test if stratiform area time is two timesteps after the convective line and two time steps before the last time of the cloud track
                    if isfarea[0] > iccline[1]:
                        ilm_index[2] = isfarea[0]
                        ilm_lifecycle[iccline[1]:isfarea[0]] = 2
                    else:
                        if nsfarea > 1:
                            ilm_index[2] = isfarea[1]
                            
                ###################################################
                # Label mature
                if nsfarea > 1:
                    ilm_lifecycle[isfarea[1:-1]] = 3
                else:
                    ilm_lifecycle[isfarea] = 3

                ################################################
                # Label dissipating times. Buy default this is all times after the mature stage

                # Include weakening convective line
                if isfarea[-1] < ilm_irtracklength-1:
                    ilm_index[3] = isfarea[-1] + 1
                    ilm_lifecycle[isfarea[-1]+1:ilm_irtracklength]

                # Include convection that no longer has line characteristics at or after mature stage
                if nnoccline > 0:
                    if inoccline[0] >= isfarea[-1]:
                        ilm_index[4] = inoccline[0]
                        ilm_lifecycle[inoccline] = 5

        ############################################################
        # Final life cycle processing
        istage = np.array(np.where(ilm_lifecycle >= 0))[0, :]
        nstage = len(istage)

        if nstaga > 0:
            lifecyclepresent = np.copy(ilm_lifecycle[istage])
            uniquelifecycle = np.unique(lifecyclepresent)[0, :]

            # Label as complete lifecycle if 1-4 present
            if len(uniquelifecycle) >= 4:
                lifecycle_complete[trackid_mcs(imcs)] = 1

            # Save data
            lifecycle_stage[ilongmcs[ilm], :] = np.copy(ilm_lifecycle)
            lifecycle_index[ilongmcs[ilm], :] = np.copy(ilm_index)

    #################################################################################
    # Save data to netcdf file



