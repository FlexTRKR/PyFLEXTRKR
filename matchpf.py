# Purpose: match mergedir tracked MCS with NMQ CSA to calculate radar-based statsitics underneath the cloud features.

# Author: Original IDL code written by Zhe Feng (zhe.feng@pnnl.gov), Python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov)

def identifypf_mergedir_rainrate(mcsstats_filebase, stats_path, startdate, enddate):
    import numpy as np
    from netCDF4 import Dataset

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
    # Reduce time resolution. Only keep data at the top of the hour

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

        raw_input('waiting')




