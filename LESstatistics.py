import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

##########################################
# Set locations
Data_Location = '/scratch2/scratchdirs/hcbarnes/LES/stats/'

########################################
# Set file names
Early_FileName = 'cell_tracks_20160830.1600_20160830.1800.nc'
Middle_FileName = 'cell_tracks_20160830.1800_20160830.2000.nc'
Late_FileName = 'cell_tracks_20160830.2000_20160830.2300.nc'

########################################
# Load data of interest
Early_Data = xr.open_dataset(Data_Location + Early_FileName, autoclose=True)
Early_time = np.array(Early_Data['ntimes'].data)
Early_lifetime = np.array(Early_Data['cell_length'].data)*60 # tracks
Early_cellarea = np.array(Early_Data['cell_ccsarea'].data)*(1000^2) # tracks, time

Middle_Data = xr.open_dataset(Data_Location + Middle_FileName, autoclose=True)
Middle_time = np.array(Middle_Data['ntimes'].data)
Middle_lifetime = np.array(Middle_Data['cell_length'].data)*60 # tracks
Middle_cellarea = np.array(Middle_Data['cell_ccsarea'].data)*(1000^2) # tracks, time

Late_Data = xr.open_dataset(Data_Location + Late_FileName, autoclose=True)
Late_time = np.array(Late_Data['ntimes'].data)
Late_lifetime = np.array(Late_Data['cell_length'].data)*60 # tracks
Late_cellarea = np.array(Late_Data['cell_ccsarea'].data)*(1000^2) # tracks, time

#######################################
# Plot PDFs

# Cell duration
hEarly_lifetime, bins_lifetime = np.histogram(Early_lifetime, bins=np.arange(0, 40, 1))
hMiddle_lifetime, bins_lifetime = np.histogram(Middle_lifetime, bins=np.arange(0, 40, 1))
hLate_lifetime, bins_lifetime = np.histogram(Late_lifetime, bins=np.arange(0, 40, 1))

bins_lifetime = np.divide((bins_lifetime[0:-1] + bins_lifetime[1::]), 2)

plt.figure()
plt.title('Track Duration', fontsize=14, y=1.01)
plt.plot(bins_lifetime, hEarly_lifetime, color='forestgreen', linewidth=3)
plt.plot(bins_lifetime, hMiddle_lifetime, color='tomato', linewidth=3)
plt.plot(bins_lifetime, hLate_lifetime, color='dodgerblue', linewidth=3)
plt.legend(['16-18 UTC (mean: ' + str(np.round(np.nanmean(Early_lifetime), 2)) + ' mins)', '18-20 UTC (mean: ' + str(np.round(np.nanmean(Middle_lifetime), 2)) + ' mins)', '20-23 UTC (mean: ' + str(np.round(np.nanmean(Late_lifetime), 2)) +' mins)'])
plt.xlim(0, 30)
plt.xlabel('Lifetime [minutes]', fontsize=12, labelpad=5)
plt.ylim(ymin=0)
plt.ylabel('Number of Cells', fontsize=12, labelpad=10)
plt.tick_params(labelsize=10)
plt.grid(True, linestyle=':')

# Maximum cell size
Early_maxcellarea = np.nanmax(Early_cellarea, axis=1)
hEarly_maxcellarea, bins_maxcellarea = np.histogram(Early_maxcellarea, bins=np.arange(0, 3000, 100))

Middle_maxcellarea = np.nanmax(Middle_cellarea, axis=1)
hMiddle_maxcellarea, bins_maxcellarea = np.histogram(Middle_maxcellarea, bins=np.arange(0, 3000, 100))

Late_maxcellarea = np.nanmax(Late_cellarea, axis=1)
hLate_maxcellarea, bins_maxcellarea = np.histogram(Late_maxcellarea, bins=np.arange(0, 3000, 100))

bins_maxcellarea = np.divide((bins_maxcellarea[0:-1] + bins_maxcellarea[1::]), 2)

print(np.nanmin(Early_maxcellarea), np.nanmin(Middle_maxcellarea), np.nanmin(Late_maxcellarea))
print(np.nanmax(Early_maxcellarea), np.nanmax(Middle_maxcellarea), np.nanmax(Late_maxcellarea))

plt.figure()
plt.title('Maximum Area', fontsize=14, y=1.01)
plt.plot(bins_maxcellarea, hEarly_maxcellarea, color='forestgreen', linewidth=3)
plt.plot(bins_maxcellarea, hMiddle_maxcellarea, color='tomato', linewidth=3)
plt.plot(bins_maxcellarea, hLate_maxcellarea, color='dodgerblue', linewidth=3)
plt.legend(['16-18 UTC (mean: ' + str(np.round(np.nanmean(Early_maxcellarea), 2)) + ' m^2)', '18-20 UTC (mean: ' + str(np.round(np.nanmean(Middle_maxcellarea), 2)) + ' m^2)', '20-23 UTC (mean: ' + str(np.round(np.nanmean(Late_maxcellarea), 2)) +' m^2)'])
plt.xlim(0, 2000)
plt.xlabel('Cell Area [m^2]', fontsize=12, labelpad=5)
plt.ylim(ymin=0)
plt.ylabel('Number of Cells', fontsize=12, labelpad=10)
plt.tick_params(labelsize=10)
plt.grid(True, linestyle=':')

########################################
# Create composites

# Find moderately long cell tracks
DurationRange = [5, 20]

Early_Indices = np.array(np.where(((Early_lifetime >= DurationRange[0]) & (Early_lifetime <= DurationRange[1]))))[0, :]
nEarly = len(Early_Indices)

Middle_Indices = np.array(np.where(((Middle_lifetime >= DurationRange[0]) & (Middle_lifetime <= DurationRange[1]))))[0, :]
nMiddle = len(Middle_Indices)

Late_Indices = np.array(np.where(((Late_lifetime >= DurationRange[0]) & (Late_lifetime <= DurationRange[1]))))[0, :]
nLate = len(Late_Indices)

# Initialize matrices
nNormalized = 11
NormalizedTimeSteps = np.linspace(0, 1, nNormalized)

NormalizedEarly_cellarea = np.empty((nEarly, nNormalized), dtype=float)*np.nan
NormalizedMiddle_cellarea = np.empty((nMiddle, nNormalized), dtype=float)*np.nan
NormalizedLate_cellarea = np.empty((nLate, nNormalized), dtype=float)*np.nan

# Normalize early time series
for iEarly in range(0,  nEarly):
    # Find find first and last time were cell area greater than 0
    cell_Indices = np.array(np.where(Early_cellarea[Early_Indices[iEarly], :]))[0, :]
    Range = np.array([cell_Indices[0], cell_Indices[-1]]).astype(int)

    # Isolate data of interest
    Subset_time = np.copy(Early_time[Range[0]:Range[1]+1])
    Subset_cellarea = np.copy(Early_cellarea[Early_Indices[iEarly], Range[0]:Range[1]+1])

    # Create time data for normalization
    Subset_time = np.subtract(Subset_time, Subset_time[0])
    ModifiedTimeSteps = np.linspace(Subset_time[0], Subset_time[-1], num=nNormalized)

    # Identify indices of finite values
    Indices_cellarea = np.isfinite(Subset_cellarea)

    # Normalized data (np.interp(Indices of the interpolated data set, indices of raw data to interpolate, values of the data being interpolated))
    NormalizedEarly_cellarea[iEarly, :] = np.interp(ModifiedTimeSteps, Subset_time[Indices_cellarea], Subset_cellarea[Indices_cellarea])

# Normalize middle time series
for iMiddle in range(0,  nMiddle):
    # Find find first and last time were cell area greater than 0
    cell_Indices = np.array(np.where(Middle_cellarea[Middle_Indices[iMiddle], :]))[0, :]
    Range = np.array([cell_Indices[0], cell_Indices[-1]]).astype(int)

    # Isolate data of interest
    Subset_time = np.copy(Middle_time[Range[0]:Range[1]+1])
    Subset_cellarea = np.copy(Middle_cellarea[Middle_Indices[iMiddle], Range[0]:Range[1]+1])

    # Create time data for normalization
    Subset_time = np.subtract(Subset_time, Subset_time[0])
    ModifiedTimeSteps = np.linspace(Subset_time[0], Subset_time[-1], num=nNormalized)

    # Identify indices of finite values
    Indices_cellarea = np.isfinite(Subset_cellarea)

    # Normalized data (np.interp(Indices of the interpolated data set, indices of raw data to interpolate, values of the data being interpolated))
    NormalizedMiddle_cellarea[iMiddle, :] = np.interp(ModifiedTimeSteps, Subset_time[Indices_cellarea], Subset_cellarea[Indices_cellarea])

# Normalize late time series
for iLate in range(0, nLate):
    # Find find first and last time were cell area greater than 0
    cell_Indices = np.array(np.where(Late_cellarea[Late_Indices[iLate], :]))[0, :]
    Range = np.array([cell_Indices[0], cell_Indices[-1]]).astype(int)

    # Isolate data of interest
    Subset_time = np.copy(Late_time[Range[0]:Range[1]+1])
    Subset_cellarea = np.copy(Late_cellarea[Late_Indices[iLate], Range[0]:Range[1]+1])

    # Create time data for normalization
    Subset_time = np.subtract(Subset_time, Subset_time[0])
    ModifiedTimeSteps = np.linspace(Subset_time[0], Subset_time[-1], num=nNormalized)

    # Identify indices of finite values
    Indices_cellarea = np.isfinite(Subset_cellarea)

    # Normalized data (np.interp(Indices of the interpolated data set, indices of raw data to interpolate, values of the data being interpolated))
    NormalizedLate_cellarea[iLate, :] = np.interp(ModifiedTimeSteps, Subset_time[Indices_cellarea], Subset_cellarea[Indices_cellarea])

########################################
# Calculate Composite statistics
Percentiles = [25, 50 , 75]

PercentilesEarly_cellarea = np.percentile(NormalizedEarly_cellarea, Percentiles, axis=0)
PercentilesMiddle_cellarea = np.percentile(NormalizedMiddle_cellarea, Percentiles, axis=0)
PercentilesLate_cellarea = np.percentile(NormalizedLate_cellarea, Percentiles, axis=0)

#######################################
# Plot Composite statistics

# cell area
plt.figure()
plt.title('Cell Area', fontsize=14, y=1.01)
plt.plot(NormalizedTimeSteps, PercentilesEarly_cellarea[1, :], '-o', linewidth=3, markersize=10, color='forestgreen')
plt.plot(NormalizedTimeSteps, PercentilesMiddle_cellarea[1, :], '-o', linewidth=3, markersize=10, color='tomato')
plt.plot(NormalizedTimeSteps, PercentilesLate_cellarea[1, :], '-o', linewidth=3, markersize=10, color='dodgerblue')
plt.legend(['16-18 UTC', '18-20 UTC', '20-23 UTC'], fontsize=12)
plt.fill_between(NormalizedTimeSteps, PercentilesEarly_cellarea[0, :], PercentilesEarly_cellarea[2, :], facecolor='forestgreen', alpha=0.2)
plt.fill_between(NormalizedTimeSteps, PercentilesMiddle_cellarea[0, :], PercentilesMiddle_cellarea[2, :], facecolor='tomato', alpha=0.2)
plt.fill_between(NormalizedTimeSteps, PercentilesLate_cellarea[0, :], PercentilesLate_cellarea[2, :], facecolor='dodgerblue', alpha=0.2)
plt.xlim(0, 1)
plt.xlabel('Normalized Time', fontsize=12, labelpad=5)
plt.ylim(ymin=0)
plt.ylabel('Cell Area [m^2]', fontsize=12, labelpad=10)
plt.tick_params(labelsize=10)
plt.grid(True, linestyle=':', color='gray')

plt.show()
