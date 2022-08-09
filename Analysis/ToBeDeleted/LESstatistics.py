# First enter source activate /global/homes/h/hcbarnes/python_parallel

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from math import pi
import pandas as pd
import datetime

##########################################
# Set locations
Data_Location = '/scratch2/scratchdirs/hcbarnes/LES/stats/'
Figure_Location = '/scratch2/scratchdirs/hcbarnes/LES/figures/'

########################################
# Set file names
Early_FileName = 'cell_tracks_20160830.1600_20160830.1800.nc'
Middle_FileName = 'cell_tracks_20160830.1800_20160830.2000.nc'
Late_FileName = 'cell_tracks_20160830.2000_20160830.2300.nc'

########################################
# Load data of interest
Early_Data = xr.open_dataset(Data_Location + Early_FileName, autoclose=True)
Early_time = np.array(Early_Data['ntimes'].data)
Early_basetime = np.array(Early_Data['cell_basetime'].data)
Early_lifetime = np.array(Early_Data['cell_length'].data)*60 # tracks
Early_cellarea = np.array(Early_Data['cell_ccsarea'].data)*(1000**2) # tracks, time
Early_cellarea[np.where(Early_cellarea == 0)] = np.nan
Early_eqdiameter = np.multiply(2, np.sqrt(np.divide(Early_cellarea, float(pi))))

Middle_Data = xr.open_dataset(Data_Location + Middle_FileName, autoclose=True)
Middle_time = np.array(Middle_Data['ntimes'].data)
Middle_basetime = np.array(Middle_Data['cell_basetime'].data)
Middle_lifetime = np.array(Middle_Data['cell_length'].data)*60 # tracks
Middle_cellarea = np.array(Middle_Data['cell_ccsarea'].data)*(1000**2) # tracks, time
Middle_cellarea[np.where(Middle_cellarea == 0)] = np.nan
Middle_eqdiameter = np.multiply(2, np.sqrt(np.divide(Middle_cellarea, float(pi))))

Late_Data = xr.open_dataset(Data_Location + Late_FileName, autoclose=True)
Late_time = np.array(Late_Data['ntimes'].data)
Late_basetime = np.array(Late_Data['cell_basetime'].data)
Late_lifetime = np.array(Late_Data['cell_length'].data)*60 # tracks
Late_cellarea = np.array(Late_Data['cell_ccsarea'].data)*(1000**2) # tracks, time
Late_cellarea[np.where(Late_cellarea == 0)] = np.nan
Late_eqdiameter = np.multiply(2, np.sqrt(np.divide(Late_cellarea, float(pi))))

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
plt.xlim(2, 30)
plt.xlabel('Lifetime [minutes]', fontsize=12, labelpad=5)
plt.ylim(ymin=0)
plt.ylabel('Number of Cells', fontsize=12, labelpad=10)
plt.tick_params(labelsize=10)
plt.grid(True, linestyle=':')
plt.savefig(Figure_Location + 'LES_DurationPDF.png')
plt.close()

# Maximum equivalent diameter
Early_maxeqdiameter = np.nanmax(Early_eqdiameter, axis=1)
hEarly_maxeqdiameter, bins_maxeqdiameter = np.histogram(Early_maxeqdiameter, bins=np.arange(200, 3500, 100))

Middle_maxeqdiameter = np.nanmax(Middle_eqdiameter, axis=1)
hMiddle_maxeqdiameter, bins_maxeqdiameter = np.histogram(Middle_maxeqdiameter, bins=np.arange(200, 3500, 100))

Late_maxeqdiameter = np.nanmax(Late_eqdiameter, axis=1)
hLate_maxeqdiameter, bins_maxeqdiameter = np.histogram(Late_maxeqdiameter, bins=np.arange(200, 3500, 100))

bins_maxeqdiameter = np.divide((bins_maxeqdiameter[0:-1] + bins_maxeqdiameter[1::]), 2)

plt.figure()
plt.title('Maximum Equivalent Diameter', fontsize=14, y=1.01)
plt.plot(bins_maxeqdiameter, hEarly_maxeqdiameter, color='forestgreen', linewidth=3)
plt.plot(bins_maxeqdiameter, hMiddle_maxeqdiameter, color='tomato', linewidth=3)
plt.plot(bins_maxeqdiameter, hLate_maxeqdiameter, color='dodgerblue', linewidth=3)
plt.legend(['16-18 UTC (mean: ' + str(np.round(np.nanmean(Early_maxeqdiameter), 2)) + ' m)', '18-20 UTC (mean: ' + str(np.round(np.nanmean(Middle_maxeqdiameter), 2)) + ' m)', '20-23 UTC (mean: ' + str(np.round(np.nanmean(Late_maxeqdiameter), 2)) +' m)'])
plt.xlim(200, 3000)
plt.xlabel('Diameter [m]', fontsize=12, labelpad=5)
plt.ylim(ymin=0)
plt.ylabel('Number of Cells', fontsize=12, labelpad=10)
plt.tick_params(labelsize=10)
plt.grid(True, linestyle=':')
plt.savefig(Figure_Location + 'LES_EqDiameterPDF.png')
plt.close()

########################################
# Create equivalent diameter groups
LifetimeRange = np.arange(5, 45, 5)

plt.figure()
plt.title('Mean Diameter over Time', fontsize=14)
for iRange in LifetimeRange:
    # Early
    EarlyIndices = np.array(np.where(Early_lifetime == iRange))[0, :]
    if len(EarlyIndices) > 10:
        Early_meaneqdiameter = np.nanmean(Early_eqdiameter[EarlyIndices, :], axis=0)
        EarlyPlot, = plt.plot(Early_meaneqdiameter, linewidth=3, color='forestgreen', label='16-18 UTC')

    # Middle
    MiddleIndices = np.array(np.where(Middle_lifetime == iRange))[0, :]
    if len(MiddleIndices) > 10:
        Middle_meaneqdiameter = np.nanmean(Middle_eqdiameter[MiddleIndices, :], axis=0)
        MiddlePlot, = plt.plot(Middle_meaneqdiameter, linewidth=3, color='tomato', label='18-20 UTC')

    # Late
    LateIndices = np.array(np.where(Late_lifetime == iRange))[0, :]
    if len(LateIndices) > 10:
        Late_meaneqdiameter = np.nanmean(Late_eqdiameter[LateIndices, :], axis=0)
        LatePlot, = plt.plot(Late_meaneqdiameter, linewidth=3, color='dodgerblue', label='20-23 UTC')

plt.legend([EarlyPlot, MiddlePlot, LatePlot], ['16-18 UTC', '18-20 UTC', '20-23 UTC'], fontsize=10)
plt.xlim(0, 40)
plt.xlabel('Time [mins]', fontsize=10)
plt.ylabel('Mean Equivalent Diameter [m]', fontsize=10)
plt.tick_params(labelsize=10)
plt.grid(True, linestyle=':')
plt.savefig(Figure_Location + 'LES_EqDiameterEvolution.png')
plt.close()

############################################
# Plot lifetime by cell start time

# Join data
AllLifetime = np.concatenate((Early_lifetime, Middle_lifetime, Late_lifetime))
AllEqDiameter = np.concatenate((np.nanmax(Early_eqdiameter, axis=1), np.nanmax(Middle_eqdiameter, axis=1), np.nanmax(Late_eqdiameter, axis=1)))

# Determine what times will be analyzed
AllBasetimes = np.concatenate((Early_basetime[:, 0], Middle_basetime[:, 0], Late_basetime[:, 0]))

ForLifetimeBoxPlot = [None]*14
ForEqdiameterBoxPlot = [None]*14
Labels = [None]*14
IntervalStart = (datetime.datetime(2016, 8, 30, 16, 0))
for iInterval in range(0, 14):
    TIntervalStart = np.array([pd.to_datetime(IntervalStart)], dtype='datetime64[ns]')
    IntervalEnd = IntervalStart + datetime.timedelta(minutes=30)
    TIntervalEnd = np.array([pd.to_datetime(IntervalEnd)], dtype='datetime64[ns]')
    Indices = np.array(np.where(((AllBasetimes >= TIntervalStart) & (AllBasetimes < TIntervalEnd))))[0, :]

    ForLifetimeBoxPlot[iInterval] = np.copy(AllLifetime[Indices])
    ForEqdiameterBoxPlot[iInterval] = np.copy(AllEqDiameter[Indices])
    Labels[iInterval] = str(IntervalStart + datetime.timedelta(minutes=15))[11:16]

    IntervalStart = IntervalEnd

# Plot
plt.figure()
plt.title('Evolution of Lifetime', fontsize=12)
plt.boxplot(ForLifetimeBoxPlot, whis=[5, 95], showmeans=True, showfliers=False, showcaps=False, labels=Labels)
plt.xlabel('Start Time of Cell [UTC]', fontsize=10)
plt.xticks(rotation=25)
plt.ylim(3, 30)
plt.ylabel('Track Lifetime [mins]', fontsize=10, labelpad=10)
plt.grid(True, linestyle=':', color='gray')
plt.tick_params(labelsize=8)
plt.savefig(Figure_Location + 'LES_DurationTimeSeries.png')
plt.close()

plt.figure()
plt.title('Evolution of Maximum Equivalent Diameter', fontsize=12)
plt.boxplot(ForEqdiameterBoxPlot, whis=[5, 95], showmeans=True, showfliers=False, showcaps=False, labels=Labels)
plt.xlabel('Start Time of Cell [UTC]', fontsize=10)
plt.xticks(rotation=25)
plt.ylim(350, 2500)
plt.ylabel('Equivalent Diameter [m]', fontsize=10, labelpad=10)
plt.grid(True, linestyle=':', color='gray')
plt.tick_params(labelsize=8)
plt.savefig(Figure_Location + 'LES_EqDiameterTimeSeries.png')
plt.close()

#plt.show()



