import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

################################################
# Set locations
IDL_Location = '/global/project/projectdirs/m1657/zfeng/usa/mergedir/stats/'
Python_Location = '/global/homes/h/hcbarnes/Tracking/SatelliteRadar/stats/'

Figure_Location = '/global/homes/h/hcbarnes/Tracking/SatelliteRadar/figures/'

#############################################
# Set analysis time
Processing_Start = 20110401
Processing_End = 20110831

############################################
# Set file names
IDL_FileName = IDL_Location + 'robust_mcs_tracks_' + str(Processing_Start) + '_' + str(Processing_End) + '.nc'
Python_FileName = Python_Location + 'robust_mcs_tracks_nmq_' + str(Processing_Start) + '_' + str(Processing_End) + '.nc'

############################################
# Load data
IDL_Data = xr.open_dataset(IDL_FileName, autoclose=True)
IDL_time = np.array(IDL_Data['times'].data)
IDL_lifetime = np.array(IDL_Data['pf_length'].data) # track
IDL_coldanvilarea = np.array(IDL_Data['ccs_area'].data)/float(1000) # track, time
IDL_pfarea = np.array(IDL_Data['pf_area'][:, :, 0])/float(1000) # track, time, pfs
#IDL_coremajoraxis = np.array(IDL_Data['pf_coremajoraxislength'].data) # track, time, cores
IDL_pfmajoraxis = np.array(IDL_Data['pf_majoraxislength'][:, :, 0]) # track, time, pfs
IDL_corearea = np.array(IDL_Data['pf_corearea'][:, :, 0])/float(1000) # track, time, pfs
IDL_ccarea = np.array(IDL_Data['pf_ccarea'].data)/float(1000) # track, time
IDL_sfarea = np.array(IDL_Data['pf_sfarea'].data)/float(1000) # track, time
#IDL_coreaspectratio = np.array(IDL_Data['pf_coreaspectratio'][:, :, 0])  # track, time, cores
IDL_pfaspectratio = np.array(IDL_Data['pf_aspectratio'][:, :, 0]) # track, time, pfs
IDL_40dbz = np.array(IDL_Data['pf_coremaxdbz40'][:, :, 0]) # track, time, pfs

Python_Data = xr.open_dataset(Python_FileName, autoclose=True)
Python_time = np.array(Python_Data['time'].data)
Python_lifetime = np.array(Python_Data['pf_lifetime'].data)
Python_coldanvilarea = np.array(Python_Data['ccs_area'].data)/float(1000)
Python_pfarea = np.array(Python_Data['pf_area'][:, :, 0])/float(1000) 
#Python_coremajoraxis = np.array(Python_Data['pf_coremajoraxislength'].data)
Python_pfmajoraxis = np.array(Python_Data['pf_majoraxislength'][:, :, 0])
Python_corearea = np.array(Python_Data['pf_corearea'][:, :, 0])/float(1000) # track, time, pfs
Python_ccarea = np.array(Python_Data['pf_ccarea'].data)/float(1000)
Python_sfarea = np.array(Python_Data['pf_sfarea'].data)/float(1000)
#Python_coreaspectratio = np.array(Python_Data['pf_coreaspectratio'][:, :, 0])
Python_pfaspectratio = np.array(Python_Data['pf_aspectratio'][:, :, 0])
Python_40dbz = np.array(Python_Data['pf_coremaxdbz40'][:, :, 0]) # track, time, pfs

############################################
# Plot PDFs

# Lifetime 
hIDL_lifetime, bins_lifetime = np.histogram(IDL_lifetime, bins=np.arange(5, 45, 4))

hPython_lifetime, bins_lifetime = np.histogram(Python_lifetime, bins=np.arange(5, 45, 4))

bins_lifetime = np.divide((bins_lifetime[0:-1] + bins_lifetime[1::]), 2)

plt.figure()
plt.title('Robust MCS lifetime', fontsize=14, y=1.01)
plt.plot(bins_lifetime, hIDL_lifetime, color='tomato', linewidth=3)
plt.plot(bins_lifetime, hPython_lifetime, color='dodgerblue', linewidth=3)
plt.legend(['IDL (mean: ' + str(np.round(np.nanmean(IDL_lifetime), 2)) + ' hr)', 'Python (mean: ' + str(np.round(np.nanmean(Python_lifetime), 2)) + ' hr)'], fontsize=12)
plt.xlim(5, 40)
plt.xlabel('Lifetime [hr]', fontsize=12, labelpad=5)
plt.ylim(ymin=0)
plt.ylabel('Number of MCSs', fontsize=12, labelpad=10)
plt.tick_params(labelsize=10)
plt.grid(True, linestyle=':')
plt.savefig(Figure_Location + 'IDLPythonComparison_Season_LifetimePDF.png')
plt.close()

# Maximum pf major axis length
IDL_maxpfmajoraxis = np.nanmax(IDL_pfmajoraxis, axis=1)
hIDL_maxpfmajoraxis, bins_maxpfmajoraxis = np.histogram(IDL_maxpfmajoraxis, bins=np.arange(150, 1600, 100))

Python_maxpfmajoraxis = np.nanmax(Python_pfmajoraxis, axis=1)
hPython_maxpfmajoraxis, bins_maxpfmajoraxis = np.histogram(Python_maxpfmajoraxis, bins=np.arange(150, 1600, 100))

bins_maxpfmajoraxis = np.divide((bins_maxpfmajoraxis[0:-1] + bins_maxpfmajoraxis[1::]), 2)

plt.figure()
plt.title('Maximum Precipitation Feature Major Axis Length', fontsize=14, y=1.01)
plt.plot(bins_maxpfmajoraxis, hIDL_maxpfmajoraxis, color='tomato', linewidth=3)
plt.plot(bins_maxpfmajoraxis, hPython_maxpfmajoraxis, color='dodgerblue', linewidth=3)
plt.legend(['IDL (mean: ' + str(np.round(np.nanmean(IDL_maxpfmajoraxis), 2)) + ' km)', 'Python (mean: ' + str(np.round(np.nanmean(Python_maxpfmajoraxis), 2)) + ' km)'], fontsize=12)
plt.xlim(150, 1500)
plt.xlabel('Axis Length [km]', fontsize=12, labelpad=5)
plt.ylim(ymin=0)
plt.ylabel('Number of MCSs', fontsize=12, labelpad=10)
plt.tick_params(labelsize=10)
plt.grid(True, linestyle=':')
plt.savefig(Figure_Location + 'IDLPythonComparison_Season_PrecipFeatureMajorAxisPDF.png')
plt.close()

# Maximum pf aspect ratio
IDL_pfaspectratio[np.where(~np.isfinite(IDL_pfaspectratio))] = np.nan
IDL_maxpfaspectratio = np.nanmax(IDL_pfaspectratio, axis=1)
hIDL_maxpfaspectratio, bins_maxpfaspectratio = np.histogram(IDL_maxpfaspectratio, bins=np.arange(0, 11, 0.5), range=[0, 15])

Python_pfaspectratio[np.where(~np.isfinite(Python_pfaspectratio))] = np.nan
Python_maxpfaspectratio = np.nanmax(Python_pfaspectratio, axis=1)
hPython_maxpfaspectratio, bins_maxpfaspectratio = np.histogram(Python_maxpfaspectratio, bins=np.arange(0, 11, 0.5), range=[0, 15])

bins_maxpfaspectratio = np.divide((bins_maxpfaspectratio[0:-1] + bins_maxpfaspectratio[1::]), 2)

plt.figure()
plt.title('Maximum Precipitation Feature Aspect Ratio', fontsize=14, y=1.01)
plt.plot(bins_maxpfaspectratio, hIDL_maxpfaspectratio, color='tomato', linewidth=3)
plt.plot(bins_maxpfaspectratio, hPython_maxpfaspectratio, color='dodgerblue', linewidth=3)
plt.legend(['IDL (mean: ' + str(np.round(np.nanmean(IDL_maxpfaspectratio), 2)) + ')', 'Python (mean: ' + str(np.round(np.nanmean(Python_maxpfaspectratio), 2)) + ')'], fontsize=12)
plt.xlim(1.5, 10)
plt.xlabel('Aspect Ratio', fontsize=12, labelpad=5)
plt.ylim(ymin=0)
plt.ylabel('Number of MCSs', fontsize=12, labelpad=10)
plt.tick_params(labelsize=10)
plt.grid(True, linestyle=':')
plt.savefig(Figure_Location + 'IDLPythonComparison_Season_PrecipFeatureAspectRatioPDF.png')
plt.close()

# Maximum pf area
IDL_maxpfarea = np.nanmax(IDL_pfarea, axis=1)
hIDL_maxpfarea, bins_maxpfarea = np.histogram(IDL_maxpfarea, bins=np.arange(1, 300, 20))

Python_maxpfarea = np.nanmax(Python_pfarea, axis=1)
hPython_maxpfarea, bins_maxpfarea = np.histogram(Python_maxpfarea, bins=np.arange(1, 300, 20))

bins_maxpfarea = np.divide((bins_maxpfarea[0:-1] + bins_maxpfarea[1::]), 2)

plt.figure()
plt.title('Maximum Precipitation Feature Area', fontsize=14, y=1.01)
plt.plot(bins_maxpfarea, hIDL_maxpfarea, color='tomato', linewidth=3)
plt.plot(bins_maxpfarea, hPython_maxpfarea, color='dodgerblue', linewidth=3)
plt.legend(['IDL (mean: ' + str(np.round(np.nanmean(IDL_maxpfarea), 2)) + ' km^2)', 'Python (mean: ' + str(np.round(np.nanmean(Python_maxpfarea), 2)) + ' km^2)'], fontsize=12)
plt.xlim(0, 250)
plt.xlabel('Area [10^3 km^2]', fontsize=12, labelpad=5)
plt.ylim(ymin=0)
plt.ylabel('Number of MCSs', fontsize=12, labelpad=10)
plt.tick_params(labelsize=10)
plt.grid(True, linestyle=':')
plt.savefig(Figure_Location + 'IDLPythonComparison_Season_PrecipFeaturePDF.png')
plt.close()

# Largest convective area
IDL_maxcorearea = np.nanmax(IDL_corearea, axis=1)
hIDL_maxcorearea, bins_maxcorearea = np.histogram(IDL_maxcorearea, bins=np.arange(1, 30, 2), range=[0, 30])

Python_maxcorearea = np.nanmax(Python_corearea, axis=1)
hPython_maxcorearea, bins_maxcorearea = np.histogram(Python_maxcorearea, bins=np.arange(1, 30, 2), range=[0, 30])

bins_maxcorearea = np.divide((bins_maxcorearea[0:-1] + bins_maxcorearea[1::]), 2)

plt.figure()
plt.title('Maximum Area of Largest Core', fontsize=14, y=1.01)
plt.plot(bins_maxcorearea, hIDL_maxcorearea, color='tomato', linewidth=3)
plt.plot(bins_maxcorearea, hPython_maxcorearea, color='dodgerblue', linewidth=3)
plt.legend(['IDL (mean: ' + str(np.round(np.nanmean(IDL_maxcorearea), 2)) + ' km^2)', 'Python (mean: ' + str(np.round(np.nanmean(Python_maxcorearea), 2)) + ' km^2)'], fontsize=12)
plt.xlim(0, 25)
plt.xlabel('Area [10^3 km^2]', fontsize=12, labelpad=5)
plt.ylim(ymin=0)
plt.ylabel('Number of MCSs', fontsize=12, labelpad=10)
plt.tick_params(labelsize=10)
plt.grid(True, linestyle=':')
plt.savefig(Figure_Location + 'IDLPythonComparison_Season_ConvectiveAreaPDF.png')
plt.close()

#IDL_maxccarea = np.nanmax(IDL_ccarea, axis=1)
#hIDL_maxccarea, bins_maxccarea = np.histogram(IDL_maxccarea, bins=np.arange(100, 60000, 3000), range=[0, 100000])

#Python_maxccarea = np.nanmax(Python_ccarea, axis=1)
#hPython_maxccarea, bins_maxccarea = np.histogram(Python_maxccarea, bins=np.arange(100, 60000, 3000), range=[0, 100000])

#bins_maxccarea = np.divide((bins_maxccarea[0:-1] + bins_maxccarea[1::]), 2)

#plt.figure()
#plt.title('Maximum Area of Largest Few Cores', fontsize=14, y=1.01)
#plt.plot(bins_maxccarea, hIDL_maxccarea, color='tomato', linewidth=3)
#plt.plot(bins_maxccarea, hPython_maxccarea, color='dodgerblue', linewidth=3)
#plt.legend(['IDL (mean: ' + str(np.round(np.nanmean(IDL_maxccarea), 2)) + ' km^2)', 'Python (mean: ' + str(np.round(np.nanmean(Python_maxccarea), 2)) + ' km^2)'], fontsize=12)
#plt.xlim(100, 55000)
#plt.xlabel('Area [km^2]', fontsize=12, labelpad=10)
#plt.ylim(ymin=0)
#plt.ylabel('Number of MCSs', fontsize=12, labelpad=10)
#plt.tick_params(labelsize=10)
#plt.grid(True, linestyle=':')

# Largest stratiform area
IDL_maxsfarea = np.nanmax(IDL_sfarea, axis=1)
hIDL_maxsfarea, bins_maxsfarea = np.histogram(IDL_maxsfarea, bins=np.arange(1, 300, 15), range=[0, 300])

Python_maxsfarea = np.nanmax(Python_sfarea, axis=1)
hPython_maxsfarea, bins_maxsfarea = np.histogram(Python_maxsfarea, bins=np.arange(1, 300, 15), range=[0, 300])

bins_maxsfarea = np.divide((bins_maxsfarea[0:-1] + bins_maxsfarea[1::]), 2)

plt.figure()
plt.title('Maxmim Area of Largest Few Stratiform Regions', fontsize=14, y=1.01)
plt.plot(bins_maxsfarea, hIDL_maxsfarea, color='tomato', linewidth=3)
plt.plot(bins_maxsfarea, hPython_maxsfarea, color='dodgerblue', linewidth=3)
plt.legend(['IDL (mean: ' + str(np.round(np.nanmean(IDL_maxsfarea), 2)) + ' km^2)', 'Python (mean: ' + str(np.round(np.nanmean(Python_maxsfarea), 2)) + ' km^2)'], fontsize=12)
plt.xlim(1, 250)
plt.xlabel('Area [10^3 km^2]', fontsize=12, labelpad=5)
plt.ylim(ymin=0)
plt.ylabel('Number of MCSs', fontsize=12, labelpad=10)
plt.tick_params(labelsize=10)
plt.grid(True, linestyle=':')
plt.savefig(Figure_Location + 'IDLPythonComparison_Season_StratiformAreaPDF.png')
plt.close()

#plt.show()

##############################################
# Create Composites

# Find medium-long MCSs
lengthrange = [8, 30]

IDL_Indices = np.array(np.where(((IDL_lifetime >= lengthrange[0]) & (IDL_lifetime <= lengthrange[1]))))[0, :]
nIDL = len(IDL_Indices)

Python_Indices = np.array(np.where(((Python_lifetime >= lengthrange[0]) & (Python_lifetime <= lengthrange[1]))))[0, :]
nPython = len(Python_Indices)

# Initialize matrices
nNormalized = 11
NormalizedTimeSteps = np.linspace(0, 1, nNormalized)

NormalizedIDL_coldanvilarea = np.empty((nIDL, nNormalized), dtype=float)*np.nan
NormalizedIDL_pfarea = np.empty((nIDL, nNormalized), dtype=float)*np.nan
NormalizedIDL_pfmajoraxis = np.empty((nIDL, nNormalized), dtype=float)*np.nan
NormalizedIDL_corearea = np.empty((nIDL, nNormalized), dtype=float)*np.nan
NormalizedIDL_sfarea = np.empty((nIDL, nNormalized), dtype=float)*np.nan
NormalizedIDL_pfaspectratio = np.empty((nIDL, nNormalized), dtype=float)*np.nan
NormalizedIDL_40dbz = np.empty((nIDL, nNormalized), dtype=float)*np.nan

NormalizedPython_coldanvilarea = np.empty((nPython, nNormalized), dtype=float)*np.nan
NormalizedPython_pfarea = np.empty((nPython, nNormalized), dtype=float)*np.nan
NormalizedPython_pfmajoraxis = np.empty((nPython, nNormalized), dtype=float)*np.nan
NormalizedPython_corearea = np.empty((nPython, nNormalized), dtype=float)*np.nan
NormalizedPython_sfarea = np.empty((nPython, nNormalized), dtype=float)*np.nan
NormalizedPython_pfaspectratio = np.empty((nPython, nNormalized), dtype=float)*np.nan
NormalizedPython_40dbz = np.empty((nPython, nNormalized), dtype=float)*np.nan

# Normalize IDL time series
for iIDL in range(0, nIDL):
    # Find first and last time were pf area greater than 0
    pf_Indices = np.array(np.where(IDL_pfarea[IDL_Indices[iIDL], :] > 0))[0, :]
    TimeRange = np.array([pf_Indices[0], pf_Indices[-1]]).astype(int)

    # Isolate data of interest
    Subset_time = np.copy(IDL_time[TimeRange[0]:TimeRange[1]+1])
    Subset_coldanvilarea = np.copy(IDL_coldanvilarea[IDL_Indices[iIDL], TimeRange[0]:TimeRange[1]+1])
    Subset_pfarea = np.copy(IDL_pfarea[IDL_Indices[iIDL], TimeRange[0]:TimeRange[1]+1])
    Subset_pfmajoraxis = np.copy(IDL_pfmajoraxis[IDL_Indices[iIDL], TimeRange[0]:TimeRange[1]+1])
    Subset_corearea = np.copy(IDL_corearea[IDL_Indices[iIDL], TimeRange[0]:TimeRange[1]+1])
    Subset_sfarea = np.copy(IDL_sfarea[IDL_Indices[iIDL], TimeRange[0]:TimeRange[1]+1])
    Subset_pfaspectratio = np.copy(IDL_pfaspectratio[IDL_Indices[iIDL], TimeRange[0]:TimeRange[1]+1])
    Subset_40dbz = np.copy(IDL_40dbz[IDL_Indices[iIDL], TimeRange[0]:TimeRange[1]+1])

    # Create normalized time
    Subset_time = np.subtract(Subset_time, Subset_time[0])
    NormalizedTimeSteps = np.linspace(Subset_time[0], Subset_time[-1], num=nNormalized)

    # Identify indices of finite values
    Indices_coldanvilarea = np.isfinite(Subset_coldanvilarea)
    Indices_pfarea = np.isfinite(Subset_pfarea)
    Indices_pfmajoraxis = np.isfinite(Subset_pfmajoraxis)
    Indices_corearea = np.isfinite(Subset_corearea)
    Indices_sfarea = np.isfinite(Subset_sfarea)
    Indices_pfaspectratio = np.isfinite(Subset_pfaspectratio)
    Indices_40dbz = np.isfinite(Subset_40dbz)

    # Normalize data (np.interp(Indices of the interpolated data set, indices of raw data to interpolate, values of the data being interpolated))
    NormalizedIDL_coldanvilarea[iIDL, :] = np.interp(NormalizedTimeSteps, Subset_time[Indices_coldanvilarea], Subset_coldanvilarea[Indices_coldanvilarea])
    NormalizedIDL_pfarea[iIDL, :] = np.interp(NormalizedTimeSteps, Subset_time[Indices_pfarea], Subset_pfarea[Indices_pfarea])
    NormalizedIDL_pfmajoraxis[iIDL, :] = np.interp(NormalizedTimeSteps, Subset_time[Indices_pfmajoraxis], Subset_pfmajoraxis[Indices_pfmajoraxis])
    NormalizedIDL_corearea[iIDL, :] = np.interp(NormalizedTimeSteps, Subset_time[Indices_corearea], Subset_corearea[Indices_corearea])
    NormalizedIDL_sfarea[iIDL, :] = np.interp(NormalizedTimeSteps, Subset_time[Indices_sfarea], Subset_sfarea[Indices_sfarea])
    NormalizedIDL_pfaspectratio[iIDL, :] = np.interp(NormalizedTimeSteps, Subset_time[Indices_pfaspectratio], Subset_pfaspectratio[Indices_pfaspectratio])
    NormalizedIDL_40dbz[iIDL, :] = np.interp(NormalizedTimeSteps, Subset_time[Indices_40dbz], Subset_40dbz[Indices_40dbz])

# Normalize Python time series
for iPython in range(0, nPython):
    # Find first and last time were pf area greater than 0
    pf_Indices = np.array(np.where(Python_pfarea[Python_Indices[iPython], :] > 0))[0, :]
    TimeRange = np.array([pf_Indices[0], pf_Indices[-1]]).astype(int)

    # Isolate data of interest
    Subset_time = np.copy(Python_time[TimeRange[0]:TimeRange[1]+1])
    Subset_coldanvilarea = np.copy(Python_coldanvilarea[Python_Indices[iPython], TimeRange[0]:TimeRange[1]+1])
    Subset_pfarea = np.copy(Python_pfarea[Python_Indices[iPython], TimeRange[0]:TimeRange[1]+1])
    Subset_pfmajoraxis = np.copy(Python_pfmajoraxis[Python_Indices[iPython], TimeRange[0]:TimeRange[1]+1])
    Subset_corearea = np.copy(Python_corearea[Python_Indices[iPython], TimeRange[0]:TimeRange[1]+1])
    Subset_sfarea = np.copy(Python_sfarea[Python_Indices[iPython], TimeRange[0]:TimeRange[1]+1])
    Subset_pfaspectratio = np.copy(Python_pfaspectratio[Python_Indices[iPython], TimeRange[0]:TimeRange[1]+1])
    Subset_40dbz = np.copy(Python_40dbz[Python_Indices[iPython], TimeRange[0]:TimeRange[1]+1])

    # Create normalized time
    Subset_time = np.subtract(Subset_time, Subset_time[0])
    NormalizedTimeSteps = np.linspace(Subset_time[0], Subset_time[-1], num=nNormalized)

    # Identify indices of finite values
    Indices_coldanvilarea = np.isfinite(Subset_coldanvilarea)
    Indices_pfarea = np.isfinite(Subset_pfarea)
    Indices_pfmajoraxis = np.isfinite(Subset_pfmajoraxis)
    Indices_corearea = np.isfinite(Subset_corearea)
    Indices_sfarea = np.isfinite(Subset_sfarea)
    Indices_pfaspectratio = np.isfinite(Subset_pfaspectratio)
    Indices_40dbz = np.isfinite(Subset_40dbz)

    # Normalize data (np.interp(Indices of the interpolated data set, indices of raw data to interpolate, values of the data being interpolated))
    NormalizedPython_coldanvilarea[iPython, :] = np.interp(NormalizedTimeSteps, Subset_time[Indices_coldanvilarea], Subset_coldanvilarea[Indices_coldanvilarea])
    NormalizedPython_pfarea[iPython, :] = np.interp(NormalizedTimeSteps, Subset_time[Indices_pfarea], Subset_pfarea[Indices_pfarea])
    NormalizedPython_pfmajoraxis[iPython, :] = np.interp(NormalizedTimeSteps, Subset_time[Indices_pfmajoraxis], Subset_pfmajoraxis[Indices_pfmajoraxis])
    NormalizedPython_corearea[iPython, :] = np.interp(NormalizedTimeSteps, Subset_time[Indices_corearea], Subset_corearea[Indices_corearea])
    NormalizedPython_sfarea[iPython, :] = np.interp(NormalizedTimeSteps, Subset_time[Indices_sfarea], Subset_sfarea[Indices_sfarea])
    NormalizedPython_pfaspectratio[iPython, :] = np.interp(NormalizedTimeSteps, Subset_time[Indices_pfaspectratio], Subset_pfaspectratio[Indices_pfaspectratio])
    NormalizedPython_40dbz[iPython, :] = np.interp(NormalizedTimeSteps, Subset_time[Indices_40dbz], Subset_40dbz[Indices_40dbz])

########################################
# Calculate Composite statistics
Percentiles = [25, 50 , 75]

PercentilesIDL_coldanvilarea = np.percentile(NormalizedIDL_coldanvilarea, Percentiles, axis=0)
PercentilesIDL_pfarea = np.percentile(NormalizedIDL_pfarea, Percentiles, axis=0)
PercentilesIDL_pfmajoraxis = np.percentile(NormalizedIDL_pfmajoraxis, Percentiles, axis=0)
PercentilesIDL_corearea = np.percentile(NormalizedIDL_corearea, Percentiles, axis=0)
PercentilesIDL_sfarea = np.percentile(NormalizedIDL_sfarea, Percentiles, axis=0)
PercentilesIDL_pfaspectratio = np.percentile(NormalizedIDL_pfaspectratio, Percentiles, axis=0)
PercentilesIDL_40dbz = np.percentile(NormalizedIDL_40dbz, Percentiles, axis=0)

PercentilesPython_coldanvilarea = np.percentile(NormalizedPython_coldanvilarea, Percentiles, axis=0)
PercentilesPython_pfarea = np.percentile(NormalizedPython_pfarea, Percentiles, axis=0)
PercentilesPython_pfmajoraxis = np.percentile(NormalizedPython_pfmajoraxis, Percentiles, axis=0)
PercentilesPython_corearea = np.percentile(NormalizedPython_corearea, Percentiles, axis=0)
PercentilesPython_sfarea = np.percentile(NormalizedPython_sfarea, Percentiles, axis=0)
PercentilesPython_pfaspectratio = np.percentile(NormalizedPython_pfaspectratio, Percentiles, axis=0)
PercentilesPython_40dbz = np.percentile(NormalizedPython_40dbz, Percentiles, axis=0)

#######################################
# Plot Composite statistics
ModifiedTimeSteps = np.arange(0, 1.1, 0.1)

# Cold anvil area
plt.figure()
plt.title('Cold Cloud Shield Area', fontsize=14, y=1.01)
plt.plot(ModifiedTimeSteps, PercentilesIDL_coldanvilarea[1, :], '-o', linewidth=3, markersize=10, color='tomato')
plt.plot(ModifiedTimeSteps, PercentilesPython_coldanvilarea[1, :], '-o', linewidth=3, markersize=8, color='dodgerblue')
plt.legend(['IDL', 'Python'], fontsize=12)
plt.fill_between(ModifiedTimeSteps, PercentilesIDL_coldanvilarea[0, :], PercentilesIDL_coldanvilarea[2, :], facecolor='tomato', alpha=0.2)
plt.fill_between(ModifiedTimeSteps, PercentilesPython_coldanvilarea[0, :], PercentilesPython_coldanvilarea[2, :], facecolor='dodgerblue', alpha=0.2)
plt.xlim(0, 1)
plt.xlabel('Normalized MCS Time', fontsize=12, labelpad=5)
plt.ylim(ymin=0)
plt.ylabel('Cloud Shield Area [10^3 km^2]', fontsize=12, labelpad=10)
plt.tick_params(labelsize=10)
plt.grid(True, linestyle=':', color='gray')
plt.savefig(Figure_Location + 'IDLPythonComparison_Season_NormalizedCloudShieldArea.png')
plt.close()

# Precipitation feature area
plt.figure()
plt.title('Precipitation Feature Area', fontsize=14, y=1.01)
plt.plot(ModifiedTimeSteps, PercentilesIDL_pfarea[1, :], '-o', linewidth=3, markersize=10, color='tomato')
plt.plot(ModifiedTimeSteps, PercentilesPython_pfarea[1, :], '-o', linewidth=3, markersize=8, color='dodgerblue')
plt.legend(['IDL', 'Python'], fontsize=12)
plt.fill_between(ModifiedTimeSteps, PercentilesIDL_pfarea[0, :], PercentilesIDL_pfarea[2, :], facecolor='tomato', alpha=0.2)
plt.fill_between(ModifiedTimeSteps, PercentilesPython_pfarea[0, :], PercentilesPython_pfarea[2, :], facecolor='dodgerblue', alpha=0.2)
plt.xlim(0, 1)
plt.xlabel('Normalized MCS Time', fontsize=12, labelpad=5)
plt.ylim(ymin=0)
plt.ylabel('Precipitation Feature Area [10^3 km^2]', fontsize=12, labelpad=10)
plt.tick_params(labelsize=10)
plt.grid(True, linestyle=':', color='gray')
plt.savefig(Figure_Location + 'IDLPythonComparison_Season_NormalizedPrecipFeatureArea.png')
plt.close()

# Precipitation feature major axis
plt.figure()
plt.title('Precipitation Feature Major Axis', fontsize=14, y=1.01)
plt.plot(ModifiedTimeSteps, PercentilesIDL_pfmajoraxis[1, :], '-o', linewidth=3, markersize=10, color='tomato')
plt.plot(ModifiedTimeSteps, PercentilesPython_pfmajoraxis[1, :], '-o', linewidth=3, markersize=8, color='dodgerblue')
plt.legend(['IDL', 'Python'], fontsize=12)
plt.fill_between(ModifiedTimeSteps, PercentilesIDL_pfmajoraxis[0, :], PercentilesIDL_pfmajoraxis[2, :], facecolor='tomato', alpha=0.2)
plt.fill_between(ModifiedTimeSteps, PercentilesPython_pfmajoraxis[0, :], PercentilesPython_pfmajoraxis[2, :], facecolor='dodgerblue', alpha=0.2)
plt.xlim(0, 1)
plt.xlabel('Normalized MCS Time', fontsize=12, labelpad=5)
plt.ylim(ymin=0)
plt.ylabel('Precipitation Feature Major Axis [km]', fontsize=12, labelpad=10)
plt.tick_params(labelsize=10)
plt.grid(True, linestyle=':', color='gray')
plt.savefig(Figure_Location + 'IDLPythonComparison_Season_NormalizedMajorAxis.png')
plt.close()

# Maximum convective core area
plt.figure()
plt.title('Area of Largest Convective Core', fontsize=14, y=1.01)
plt.plot(ModifiedTimeSteps, PercentilesIDL_corearea[1, :], '-o', linewidth=3, markersize=10, color='tomato')
plt.plot(ModifiedTimeSteps, PercentilesPython_corearea[1, :], '-o', linewidth=3, markersize=8, color='dodgerblue')
plt.legend(['IDL', 'Python'], fontsize=12)
plt.fill_between(ModifiedTimeSteps, PercentilesIDL_corearea[0, :], PercentilesIDL_corearea[2, :], facecolor='tomato', alpha=0.2)
plt.fill_between(ModifiedTimeSteps, PercentilesPython_corearea[0, :], PercentilesPython_corearea[2, :], facecolor='dodgerblue', alpha=0.2)
plt.xlim(0, 1)
plt.xlabel('Normalized MCS Time', fontsize=12, labelpad=5)
plt.ylim(ymin=0)
plt.ylabel('Convective Area [10^3 km^2]', fontsize=12, labelpad=10)
plt.tick_params(labelsize=10)
plt.grid(True, linestyle=':', color='gray')
plt.savefig(Figure_Location + 'IDLPythonComparison_Season_NormalizedConvectiveArea.png')
plt.close()

# Stratiform
plt.figure()
plt.title('Area of Largest Few Stratiform Regions', fontsize=14, y=1.01)
plt.plot(ModifiedTimeSteps, PercentilesIDL_sfarea[1, :], '-o', linewidth=3, markersize=10, color='tomato')
plt.plot(ModifiedTimeSteps, PercentilesPython_sfarea[1, :], '-o', linewidth=3, markersize=8, color='dodgerblue')
plt.legend(['IDL', 'Python'], fontsize=12)
plt.fill_between(ModifiedTimeSteps, PercentilesIDL_sfarea[0, :], PercentilesIDL_sfarea[2, :], facecolor='tomato', alpha=0.2)
plt.fill_between(ModifiedTimeSteps, PercentilesPython_sfarea[0, :], PercentilesPython_sfarea[2, :], facecolor='dodgerblue', alpha=0.2)
plt.xlim(0, 1)
plt.xlabel('Normalized MCS Time', fontsize=12, labelpad=5)
plt.ylim(ymin=0)
plt.ylabel('Stratiform Area [10^3 km^2]', fontsize=12, labelpad=10)
plt.tick_params(labelsize=10)
plt.grid(True, linestyle=':', color='gray')
plt.savefig(Figure_Location + 'IDLPythonComparison_Season_NormalizedStratiformArea.png')
plt.close()

# Precipitation feature aspect ratio
plt.figure()
plt.title('Precipitation Feature Aspect Ratio', fontsize=14, y=1.01)
plt.plot(ModifiedTimeSteps, PercentilesIDL_pfaspectratio[1, :], '-o', linewidth=3, markersize=10, color='tomato')
plt.plot(ModifiedTimeSteps, PercentilesPython_pfaspectratio[1, :], '-o', linewidth=3, markersize=8, color='dodgerblue')
plt.legend(['IDL', 'Python'], fontsize=12)
plt.fill_between(ModifiedTimeSteps, PercentilesIDL_pfaspectratio[0, :], PercentilesIDL_pfaspectratio[2, :], facecolor='tomato', alpha=0.2)
plt.fill_between(ModifiedTimeSteps, PercentilesPython_pfaspectratio[0, :], PercentilesPython_pfaspectratio[2, :], facecolor='dodgerblue', alpha=0.2)
plt.xlim(0, 1)
plt.xlabel('Normalized MCS Time', fontsize=12, labelpad=5)
plt.ylim(ymin=1)
plt.ylabel('Precipitation Feature Aspect Ratio', fontsize=12, labelpad=10)
plt.tick_params(labelsize=10)
plt.grid(True, linestyle=':', color='gray')
plt.savefig(Figure_Location + 'IDLPythonComparison_Season_NormalizedAspectRatio.png')
plt.close()

# 40 dBZ Height
plt.figure()
plt.title('40 dBZ Height of Largest Precipitation Feature', fontsize=14, y=1.01)
plt.plot(ModifiedTimeSteps, PercentilesIDL_40dbz[1, :], '-o', linewidth=3, markersize=10, color='tomato')
plt.plot(ModifiedTimeSteps, PercentilesPython_40dbz[1, :], '-o', linewidth=3, markersize=8, color='dodgerblue')
plt.legend(['IDL', 'Python'], fontsize=12)
plt.fill_between(ModifiedTimeSteps, PercentilesIDL_40dbz[0, :], PercentilesIDL_40dbz[2, :], facecolor='tomato', alpha=0.2)
plt.fill_between(ModifiedTimeSteps, PercentilesPython_40dbz[0, :], PercentilesPython_40dbz[2, :], facecolor='dodgerblue', alpha=0.2)
plt.xlim(0, 1)
plt.xlabel('Normalized MCS Time', fontsize=12, labelpad=5)
plt.ylim(ymin=3)
plt.ylabel('Height [km]', fontsize=12, labelpad=10)
plt.tick_params(labelsize=10)
plt.grid(True, linestyle=':', color='gray')
plt.savefig(Figure_Location + 'IDLPythonComparison_Season_Normalized40DBZ.png')
plt.close()

#plt.show()
