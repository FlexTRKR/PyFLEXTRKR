# Purpose: Filter MCS using NMQ radar variables so that only robust MCSs are included.

# Comments: Method similar to Coniglio et al (2010) MWR. 

# Author: Original IDL code written by Zhe Feng (zhe.feng@pnnl.gov), Python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov)

def filtermcs_mergedir_nmq(stats_path, pfstats_filebase, startdate, enddate, majoraxisthresh, durationthresh, aspectratiothresh, lifecyclethresh, lengththresh, gapthresh):
    ######################################################
    # Import modules
    import numpy as np
    import xarray as xr

    ######################################################
    # Set constants
    fillvalue = -9999

    ######################################################
    # Load mergedir mcs and pf data
    mergedirpf_statistics_file = stats_path + pfstats_filebase + startdate + '_' + enddate + '.nc'

    data = xr.open_dataset(mergedirpf_statistics_file, autoclose=True)
    ntracks = np.nanmax(data.coords['track'])
    ntimes = np.nanmax(data.coords['time'])
    nmaxmerge = np.nanmax(data.coords['mergesplit'])
    nmaxpf = np.nanmax(data.coords['pfs'])
    nmaxcore = np.nanmax(data.coords['cores'])

    basetime = data['basetime']
    datetimestring = data['datetimestring']

    ir_mcslength = data['mcs_length']
    ir_tracklength = data['length']
    ir_status = data['status']
    ir_startstatus = data['startstatus']
    ir_endstatus = data['endstatus']
    ir_mcstype = data['mcs_type']
    ir_meanlat = data['meanlat']
    ir_meanlon = data['meanlon']
    ir_corearea = data['core_area']
    ir_ccsarea = data['ccs_area']
    ir_cloudnumber = data['cloudnumber']
    ir_mergecloudnumber = data['mergecloudnumber']
    ir_splitcloudnumber = data['splitcloudnumber']

    nmq_frac = data['nmq_frac']
    numpf = data['npf']
    pf_area = data['pf_area']
    pf_lon = data['pf_lon']
    pf_lat = data['pf_lat']
    pf_rainrate = data['pf_rainrate']
    pf_skewness = data['pf_skewness']
    pf_majoraxislength = data['pf_majoraxislength']
    pf_aspect = data['pf_aspectratio']
    pf_dbz40area = data['pf_dbz40area']
    pf_dbz45area = data['pf_dbz45area']
    pf_dbz50area = data['pf_dbz50area']

    pf_meanccrainrate = data['pf_ccrainrate']
    pf_meannccarea = data['pf_ccarea']
    pf_meanccdbz10 = data['pf_ccdbz10']
    pf_meanccdbz20 = data['pf_ccdbz20']
    pf_meanccdbz30 = data['pf_ccdbz30']
    pf_meanccdbz40 = data['pf_ccdbz40']
    pf_meansfrainrate = data['pf_sfrainrate']
    pf_meansfarea = data['pf_sfarea']

    pf_ncc = data['pf_ncores']
    pf_cclon = data['pf_corelon']
    pf_cclat = data['pf_corelat']
    pf_ccarea = data['pf_corearea']
    pf_ccmajoraxislength = data['pf_coremajoraxislength']
    pf_ccaspectratio = data['pf_coreaspectratio']
    pf_ccmaxdbz10 = data['pf_coremaxdbz10']
    pf_ccmaxdbz20 = data['pf_coremaxdbz20']
    pf_ccmaxdbz30 = data['pf_coremaxdbz30']
    pf_ccmaxdbz40 = data['pf_coremaxdbz40']
    pf_ccavgdbz10 = data['pf_coreavgdbz10']
    pf_ccavgdbz20 = data['pf_coreavgdbz20']
    pf_ccavgdbz30 = data['pf_coreavgdbz30']
    pf_ccavgdbz40 = data['pf_coreavgdbz40']

    mergedir_source = data.attrs['source1']
    radar_source = data.attrs['source2']
    datadescription = data.attrs['description']
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
        istart = np.copy(ir_startstatus[nt])
        iend = np.copy(ir_endstatus[nt])

        # Get the largest precipitation (1st entry in 3rd dimension)
        ipf_majoraxislength = np.copy(pf_majoraxislength[nt, 0:ilength, 0])
        ipf_dbz40area = np.copy(pf_dbz40area[nt, 0:ilength, 0])
        ipf_dbz45area = np.copy(pf_dbz45area[nt, 0:ilength, 0])
        ipf_dbz50area = np.copy(pf_dbz50area[nt, 0:ilength, 0])

        # Get the cooresponding convective core and stratiform region data (use the largest feature (1st entry in 3rd dimenstion), when applicable) 
        ipf_meansfarea = np.copy(pf_meansfarea[nt, 0:ilength])
        ipf_ccmajoraxislength = np.copy(pf_ccmajoraxislength[nt, 0:ilength, 0])
        ipf_ccaspectratio = np.copy(pf_ccaspectratio[nt, 0:ilength, 0])
        ipf_nmqfrac = np.copy(nmq_frac[nt, 0:ilength])

        ######################################################
        # Apply radar defined MCS criteria

        # Apply PF major axis length > thresh and contains echo >= 50 dbZ criteria
        ipfmcs = np.array(np.where((ipf_majoraxislength > majoraxisthresh) & (ipf_dbz50area > 0)))[0, :]
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
                        igroup_pfmajoraxislength = np.copy(ipf_majoraxislength[igroup_indices])
                        igroup_dbz40area = np.copy(ipf_dbz40area[igroup_indices])
                        igroup_ccmajoraxislength = np.copy(ipf_ccmajoraxislength[igroup_indices])
                        igroup_ccaspectratio = np.copy(ipf_ccaspectratio[igroup_indices])

                        # Label this period as an mcs
                        pf_mcstype[nt, igroup_indices] = 1

                        # Determine type of mcs (squall or non-squall)
                        isquall = np.array(np.where(igroup_ccaspectioratio > aspectratiothresh))[0, :]
                        nisquall = len(isquall)

                        if nisquall > 0:
                            # Label as squall
                            pf_mcstype[nt] = 1
                            pf_cctype[nt] = 1
                        else:
                            # Label as non-squall
                            pf_mcstype[nt] = 2
                            pf_cctype[nt] = 2

            # Group does not satistfy duration threshold
            else:
                trackid_nonmcs = np.append(trackid_nonmcs, nt)
                


