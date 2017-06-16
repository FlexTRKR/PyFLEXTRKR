# Purpose: Subset statistics file to keep only MCS

# Author: Original IDL code written by Zhe Feng (zhe.feng@pnnl.gov), Python version written by Hannah C. Barnes (hannah.barnes@pnnl.gov)

def mergedir(statistics_file, area_thresh, duration_thresh, eccentricity_thresh, split_duration, merge_duration, time_resolution):
    #######################################################################
    # Import modules
    import numpy as np
    from netCDF4 import Dataset

    ######################################################################
    # Load statistics file
    allstatdata = Dataset(statistics_file, 'r')
    ntracks_all = len(allstatdata.dimension['ntracks']) # Total number of tracked features
    namxlength = len(allstatdata.dimension['nmaxlength']) # Maximum number of features in a given track
    length = allstatdata.variable['lifetime'][:] # Duration of each track
    basetime = allstatdata.variable['basetime'][:] # Time of cloud in seconds since 01/01/1970 00:00
    datetime = allstatdata.variable['satetimestring'][:]
    meanlat = allstatdata.variable['meanlat'][:] # Mean latitude of the core and cold anvil
    meanlon = allstatdata.variable['meanlon'][:] # Mean longitude of the core and cold anvil
    cloudnumber = allstatdata.variable['cloudnumber'][:] # Number of the corresponding cloudid file
    status = allstatdata.variable['status'][:] # Flag indicating the status of the cloud
    startstatus = allstatdata.variable['startstatus'][:] # Flag indicating the status of the first feature in each track
    endstatus = allstatdata.variable['endstatus'][:] # Flag indicating the status of the last feature in each track 
    mergenumbers = allstatdata.variable['mergenumbers'][:] # Number of a small feature that merges onto a bigger feature
    splitnumbers = allstatdata.variable['splitnumbers'][:] # Number of a small feature that splits onto a bigger feature
    npix_corecold = allstatdata.variable['npix'][:] # Number of pixels in the core and cold anvil
    npix_core = allstatdata.variable['nconv'][:] # Number of pixels in the core
    npix_cold = allstatdata.variable['ncoldanvil'][:] # Number of pixels in the cold anvil
    majoraxis = allstatdata.variable['majoraxis'][:] # Length of the major axis of the core and cold anvil
    eccentricity = allstatdata.variable['eccentricity'][:] # Eccentricity of the core and cold anvil
    tb_coldanvil = getncattr(allstatdata, tb_coldanvil) # Brightness temperature threshold for cold anvil
    pixel_radius = getncattr(allstatdata, pixel_radisu_km) # Radius of one pixel in dataset
    source = str(getncattr(allstatdata, source))
    description = str(getncattr(allstatdata, description))
    track_version = str(getncattr(allstatdata,track_version))
    tracknumbers_version = str(getncattr(allstatdata, tracknumbers_version))
    allstatdata.close()

    fillvalue = -9999

    ####################################################################
    # Set up thresholds

    # Cold Cloud Shield (CCS) area
    corearea = npix_core * pixel_radius**2
    ccsarea = npix_corecold * pixel_radius**2

    # Convert path duration to time
    duration = length * time_resolution





