import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from scipy import signal
from math import pi
np.set_printoptions(threshold=np.inf)

######################################
# Set start time
Hour = 21
Minute = 0

AnalysisTime = np.array(pd.to_datetime(pd.DataFrame({'year': [2016], 'month': [8], 'day':[30], 'hour':[Hour], 'minute':[Minute]})), dtype='datetime64[ns]')

####################################
# Set LES region (Domain: latitude = 36.05-37.15, longitude = -98.12--96.79)
LatitudeRegion = [36.1, 36.6]
LongitudeRegion = [-98.0, -97.4]

###################################
# Set Area threshold
#AreaThresh = 1
EqDiameterThresh = 1000

##########################################
# Set locations
Stat_Location = '/global/cscratch1/sd/hcbarnes/LES/stats/'
Map_Location = '/global/cscratch1/sd/hcbarnes/LES/celltracking/20160830.2000_20160830.2300/'
Figure_Location = '/scratch2/scratchdirs/hcbarnes/LES/figures/'

##########################################
# Set file name
FileName = 'cell_tracks_20160830.2000_20160830.2300.nc'

#########################################
# Load statistics data

Stat_DataHandle = xr.open_dataset(Stat_Location + FileName, autoclose=True)
Stat_Basetime = Stat_DataHandle['cell_basetime'].data # track, time
Stat_Area = Stat_DataHandle['cell_ccsarea'].data # track, time
Stat_MeanLat = Stat_DataHandle['cell_meanlat'].data # track, time
MinLat = Stat_DataHandle['cell_meanlat'].valid_min
MaxLat = Stat_DataHandle['cell_meanlat'].valid_max
Stat_MeanLon = Stat_DataHandle['cell_meanlon'].data # track, time
MinLon = Stat_DataHandle['cell_meanlon'].valid_min
MaxLon = Stat_DataHandle['cell_meanlon'].valid_max

Stat_Area = np.multiply(Stat_Area, 1000**2)
Stat_EqDiameter = np.multiply(2, np.sqrt(np.divide(Stat_Area, pi)))

##############################################
# Isolate times
TimeIndices = np.array(np.where(Stat_Basetime == AnalysisTime))

print(AnalysisTime)

idMeanLon = np.copy(Stat_MeanLon[TimeIndices[0], TimeIndices[1]])
idMeanLat = np.copy(Stat_MeanLat[TimeIndices[0], TimeIndices[1]])
idArea = np.copy(Stat_Area[TimeIndices[0], TimeIndices[1]])
idEqDiameter = np.copy(Stat_EqDiameter[TimeIndices[0], TimeIndices[1]])

Stat_Basetime = Stat_Basetime[TimeIndices[0], :]
Stat_Area = Stat_Area[TimeIndices[0], :]
Stat_EqDiameter = Stat_EqDiameter[TimeIndices[0], :]
Stat_MeanLat = Stat_MeanLat[TimeIndices[0], :]
Stat_MeanLon = Stat_MeanLon[TimeIndices[0], :]

###############################################
# Restrict area of LES clouds (Domain: latitude = 36.05-37.15, longitude = -98.12--96.79)

LocationIndices = np.array(np.where(((idMeanLon >= LongitudeRegion[0]) & (idMeanLon <= LongitudeRegion[1]) & (idMeanLat >= LatitudeRegion[0]) & (idMeanLat <= LatitudeRegion[1]))))[0, :]

idArea = idArea[LocationIndices]
idEqDiameter = idEqDiameter[LocationIndices]

Stat_Basetime = Stat_Basetime[LocationIndices, :]
Stat_Area = Stat_Area[LocationIndices, :]
Stat_EqDiameter = Stat_EqDiameter[LocationIndices, :]
Stat_MeanLat = Stat_MeanLat[LocationIndices, :]
Stat_MeanLon = Stat_MeanLon[LocationIndices, :]

########################################
# Restrict based on size
#AreaIndices = np.array(np.where(idArea >= AreaThresh))[0, :]
EqDiameterIndices = np.array(np.where(idEqDiameter >= EqDiameterThresh))[0, :]

Stat_Basetime = Stat_Basetime[EqDiameterIndices, :]
Stat_Area = Stat_Area[EqDiameterIndices, :]
Stat_EqDiameter = Stat_EqDiameter[EqDiameterIndices, :]
Stat_MeanLat = Stat_MeanLat[EqDiameterIndices, :]
Stat_MeanLon = Stat_MeanLon[EqDiameterIndices, :]

#########################################
# Get map of cells

# Get file name
if Hour < 10:
    Hour = '0' + str(int(Hour))
else:
    Hour = str(int(Hour))

if Minute < 10:
    Minute = '0' + str(int(Minute))
else:
    Minute = str(int(Minute))

Map_FileName = 'celltracks_20160830_' + str(int(Hour)) + Minute + '.nc'

# Load data
Map_DataHandle = xr.open_dataset(Map_Location + Map_FileName, autoclose=True)
#TrackMap = Map_DataHandle['celltracknumber'][0, :, :].data
LWP = Map_DataHandle['lwp'][0, :, :].data
Latitude = Map_DataHandle['lat'].data
Longitude = Map_DataHandle['lon'].data

#TrackMap[np.where(TrackMap > 0)] = 1

########################################
# Plot map

# Duration
plt.figure()
plt.title('Tracks of Cells with Eq. Diameter >= ' + str(EqDiameterThresh) + ' km at \n ' + str(AnalysisTime)[2:12] + ' ' + str(AnalysisTime)[13:18] + ' UTC', fontsize=10)
im = plt.pcolormesh(Longitude, Latitude, np.ma.masked_invalid(np.atleast_2d(LWP)), cmap='gray_r', vmin=0, vmax=2)
cbar = plt.colorbar(im)
cbar.ax.set_ylabel('Liquid Water Path', fontsize=10)
cbar.ax.tick_params(labelsize=8)
#plt.contour(Longitude, Latitude, TrackMap, colors='gray', linewidths=1, linestyles=':')
for iCell in range(0, np.shape(Stat_Basetime)[0]):
    PlotLon = np.copy(Stat_MeanLon[iCell, :])
    PlotLon = PlotLon[np.where(np.isfinite(PlotLon))]

    PlotLat = np.copy(Stat_MeanLat[iCell, :])
    PlotLat = PlotLat[np.where(np.isfinite(PlotLat))]

    iDuration = len(PlotLat)
    if iDuration < 10:
        plt.plot(PlotLon, PlotLat, linewidth=2, color='darkorange')
    elif iDuration >= 10 and iDuration < 20:
        plt.plot(PlotLon, PlotLat, linewidth=2, color='red')
    elif iDuration >= 20 and iDuration < 30:
        plt.plot(PlotLon, PlotLat, linewidth=2, color='blueviolet')
    elif iDuration >= 30 and iDuration < 40:
        plt.plot(PlotLon, PlotLat, linewidth=2, color='blue')
    elif iDuration >= 40 and iDuration < 50:
        plt.plot(PlotLon, PlotLat, linewidth=2, color='c')
    else:
        plt.plot(PlotLon, PlotLat, linewidth=2, color='limegreen')

plt.xlim(LongitudeRegion[0], LongitudeRegion[1])
#plt.xlim(MinLon, MaxLon)
plt.xlabel('Longitude', fontsize=10)
plt.ylim(LatitudeRegion[0], LatitudeRegion[1])
#plt.ylim(MinLat, MaxLat)
plt.ylabel('Latitude', fontsize=10)
plt.grid(True, linestyle=':', color='gray')
plt.tick_params(labelsize=8)
plt.text(-97.2, 36.5, 'Cell Duration', fontsize=10, color='black')
plt.text(-97.2, 36.45, '<10 mins', fontsize=10, color='darkorange')
plt.text(-97.2, 36.4, '10-20 mins', fontsize=10, color='red')
plt.text(-97.2, 36.35, '20-30 mins', fontsize=10, color='blueviolet')
plt.text(-97.2, 36.3, '30-40 mins', fontsize=10, color='blue')
plt.text(-97.2, 36.25, '40-50 mins', fontsize=10, color='c')
plt.text(-97.2, 36.2, '>50 mins', fontsize=10, color='limegreen')
plt.subplots_adjust(right=0.8)
plt.savefig(Figure_Location + 'LESmap_Duration_' + str(AnalysisTime)[2:12] + '_' + str(AnalysisTime)[13:18] + '.png')
plt.close()

# Eqivalent Diameter
plt.figure()
plt.title('Tracks of Cells with Eq. Diameter >= ' + str(EqDiameterThresh) + ' km at \n ' + str(AnalysisTime)[2:12] + ' ' + str(AnalysisTime)[13:18] + ' UTC', fontsize=10)
im = plt.pcolormesh(Longitude, Latitude, np.ma.masked_invalid(np.atleast_2d(LWP)), cmap='gray_r', vmin=0, vmax=2)
cbar = plt.colorbar(im)
cbar.ax.set_ylabel('Liquid Water Path', fontsize=10)
cbar.ax.tick_params(labelsize=8)
#plt.contour(Longitude, Latitude, TrackMap, colors='gray', linewidths=1, linestyles=':')
for iCell in range(0, np.shape(Stat_Basetime)[0]):
    PlotLon = np.copy(Stat_MeanLon[iCell, :])
    PlotLon = PlotLon[np.where(np.isfinite(PlotLon))]

    PlotLat = np.copy(Stat_MeanLat[iCell, :])
    PlotLat = PlotLat[np.where(np.isfinite(PlotLat))]

    iEqDiameter = np.nanmax(Stat_EqDiameter[iCell, :])
    if iEqDiameter < 1500:
        plt.plot(PlotLon, PlotLat, linewidth=2, color='darkorange')
    elif iEqDiameter >= 1500 and iEqDiameter < 2000:
        plt.plot(PlotLon, PlotLat, linewidth=2, color='red')
    elif iEqDiameter >= 2000 and iEqDiameter < 2500:
        plt.plot(PlotLon, PlotLat, linewidth=2, color='blueviolet')
    elif iEqDiameter >= 2500 and iEqDiameter < 3000:
        plt.plot(PlotLon, PlotLat, linewidth=2, color='blue')
    elif iEqDiameter >= 3000 and iEqDiameter < 3500:
        plt.plot(PlotLon, PlotLat, linewidth=2, color='c')
    else:
        plt.plot(PlotLon, PlotLat, linewidth=2, color='limegreen')

plt.xlim(LongitudeRegion[0], LongitudeRegion[1])
#plt.xlim(MinLon, MaxLon)
plt.xlabel('Longitude', fontsize=10)
plt.ylim(LatitudeRegion[0], LatitudeRegion[1])
#plt.ylim(MinLat, MaxLat)
plt.ylabel('Latitude', fontsize=10)
plt.grid(True, linestyle=':', color='gray')
plt.tick_params(labelsize=8)
plt.text(-97.2, 36.5, 'Max. Eq. Diameter', fontsize=10, color='black')
plt.text(-97.2, 36.45, '<1500 m', fontsize=10, color='darkorange')
plt.text(-97.2, 36.4, '1500-2000 m', fontsize=10, color='red')
plt.text(-97.2, 36.35, '2000-2500 m ', fontsize=10, color='blueviolet')
plt.text(-97.2, 36.3, '2500-3000 m', fontsize=10, color='blue')
plt.text(-97.2, 36.25, '3000-3500 m', fontsize=10, color='c')
plt.text(-97.2, 36.2, '>3500 m', fontsize=10, color='limegreen')
plt.subplots_adjust(right=0.75)
plt.savefig(Figure_Location + 'LESmap_EqDiameter_' + str(AnalysisTime)[2:12] + '_' + str(AnalysisTime)[13:18] + '.png')
plt.close()

#plt.show()
