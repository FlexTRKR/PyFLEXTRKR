# Before running enter ... source activate /global/homes/h/hcbarnes/python_parallel

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from mpl_toolkits.basemap import Basemap
import time
import datetime
import glob
import dask
import os
import fnmatch

################################################
# Set locations
#IDL_Location = '/global/project/projectdirs/m1867/zfeng/usa/mergedir/mcstracking/20110517_20110527/'
IDL_Location = '/global/project/projectdirs/m1657/zfeng/usa/mergedir/mcstracking/20110401_20110831/'

#Python_Location = '/global/homes/h/hcbarnes/Tracking/SatelliteRadar/mcstracking/20110517_20110527/'
Python_Location = '/global/homes/h/hcbarnes/Tracking/SatelliteRadar/mcstracking/20110401_20110831/'

Figure_Location = '/global/homes/h/hcbarnes/Tracking/SatelliteRadar/figures/'

##############################################
# Set analysis time
AnalyzeYear = 2011
AnalyzeMonth = 4

##############################################
# Isolate files to analyze

AllIDLFiles = fnmatch.filter(os.listdir(IDL_Location), 'mcstrack*.nc')
IDL_Files = [None]*len(AllIDLFiles)
FileStep = 0
for iFile in AllIDLFiles:
    if int(iFile[9:13]) == AnalyzeYear and int(iFile[13:15]) == AnalyzeMonth:
        IDL_Files[FileStep] = IDL_Location + iFile
        FileStep = FileStep + 1
IDL_Files = IDL_Files[0:FileStep]
IDL_Files = sorted(IDL_Files)

AllPythonFiles = fnmatch.filter(os.listdir(Python_Location), 'mcstrack*.nc')
Python_Files = [None]*len(AllPythonFiles)
FileStep = 0
for iFile in AllPythonFiles:
    if int(iFile[10:14]) == AnalyzeYear and int(iFile[14:16]) == AnalyzeMonth:
        Python_Files[FileStep] = Python_Location + iFile
        FileStep = FileStep + 1
Python_Files = Python_Files[0:FileStep]
Python_Files = sorted(Python_Files)

################################################
# Load data

#IDLdata = xr.open_mfdataset(IDL_Location + '*.nc', concat_dim='time', autoclose=True)
IDLdata = xr.open_mfdataset(IDL_Files, concat_dim='time', autoclose=True)
IDLbasetime = IDLdata['base_time'].data
IDLCloudTrack = np.array(IDLdata['cloudtracknumber'].data)
IDLPrecipitation = np.array(IDLdata['precipitation'].data)
IDLBrightnessTemperature = np.array(IDLdata['tb'].data)
Longitude = np.array(IDLdata['longitude'].data)
Latitude= np.array(IDLdata['latitude'].data)
nIDL = len(IDLbasetime)

#PythonData = xr.open_mfdataset(Python_Location + '*.nc', concat_dim='time', autoclose=True)
PythonData = xr.open_mfdataset(Python_Files, concat_dim='time', autoclose=True)
PythonBasetime = np.array(PythonData['basetime'].data)
PythonCloudTrack = np.array(PythonData['cloudtracknumber'].data)
PythonPrecipitation = np.array(PythonData['precipitation'].data)
nPython = len(PythonBasetime)

PythonTimeCoordinate = PythonData.coords['time'].values

#################################################
# Make sure all the same files are here
if nIDL > nPython:
    DifferentBasetimes = np.setdiff1d(IDLbasetime, PythonBasetime)
    DifferentIndices = np.ones(len(DifferentBasetimes))
    for iDifferent in range(0, len(DifferentBasetimes)):
        DifferentIndices[iDifferent] = np.array(np.where(IDLbasetime == DifferentBasetimes[iDifferent]))
    SameIndices = np.setdiff1d(np.arange(0, nIDL), DifferentIndices)

    IDLbasetime = IDLbasetime[SameIndices]
    IDLCloudTrack = IDLCloudTrack[SameIndices, :, :]
    IDLPrecipitation = IDLPrecipitation[SameIndices, :, :]
    IDLBrightnessTemperature = IDLBrightnessTemperature[SameIndices, :, :]

else:
    DifferentBasetimes = np.setdiff1d(PythonBasetime, IDLbasetime)
    DifferentIndices = np.ones(len(DifferentBasetimes))
    for iDifferent in range(0, len(DifferentBasetimes)):
        DifferentIndices[iDifferent] = np.array(np.where(PythonBasetime == DifferentBasetimes[iDifferent]))
    SameIndices = np.setdiff1d(np.arange(0, nPython), DifferentIndices)

    PythonBasetime = PythonBasetime[SameIndices]
    PythonCloudTrack = PythonCloudTrack[SameIndices, :, :]
    PythonPrecipitation = PythonPrecipitation[SameIndices, :, :]

nFiles = len(PythonBasetime)

####################################################
# Get data dimensions
IDLNumFiles, IDLNumLat, IDLNumLong = np.shape(Longitude)
PythonNumFiles, PythonNumLat, PythonNumLong = np.shape(PythonCloudTrack)

#####################################################
# Determine number of mcs
NumIDLTracks = np.nanmax(IDLCloudTrack)
NumPythonTracks = np.nanmax(PythonCloudTrack)

####################################################
# Convert cloud tracks to flag
IDLflag = np.zeros((IDLNumFiles, IDLNumLat, IDLNumLong), dtype=float)
IDLflag[np.where(IDLCloudTrack > 0)] = 1

PythonFlag = np.zeros((PythonNumFiles, PythonNumLat, PythonNumLong), dtype=float)
PythonFlag[np.where(PythonCloudTrack > 0)] = 1

###################################################
# Calculate precipitation statistics
IDLPrecipitation[np.isnan(IDLPrecipitation)] = 0
IDLTotalPrecip = np.sum(IDLPrecipitation, axis=0)
IDL_MCSPrecip = np.sum(np.multiply(IDLPrecipitation, IDLflag), axis=0)
IDL_MCSPrecipFraction = np.divide(IDL_MCSPrecip, IDLTotalPrecip)

PythonPrecipitation[np.isnan(PythonPrecipitation)] = 0
PythonTotalPrecip = np.sum(PythonPrecipitation, axis=0)
Python_MCSPrecip = np.sum(np.multiply(PythonPrecipitation, PythonFlag), axis=0)
Python_MCSPrecipFraction = np.divide(Python_MCSPrecip, PythonTotalPrecip)

IDLTotalFiles = np.ones((IDLNumLat, IDLNumLong), dtype=float)*IDLNumFiles
IDL_MCSFrequency = np.divide(np.sum(IDLflag, axis=0), IDLTotalFiles)

PythonTotalFiles = np.ones((PythonNumLat, PythonNumLong), dtype=float)*PythonNumFiles
Python_MCSFrequency = np.divide(np.sum(PythonFlag, axis=0), PythonTotalFiles)

###############################################
# Calculate differences (I - P)
Difference_MCSPrecip = np.subtract(IDL_MCSPrecip, Python_MCSPrecip)
Difference_MCSPrecipFraction = np.subtract(IDL_MCSPrecipFraction, Python_MCSPrecipFraction)
Difference_MCSFrequency = np.subtract(IDL_MCSFrequency, Python_MCSFrequency)

#################################################
# Make areas of no precipitation be nan
IDLTotalPrecip[IDLTotalPrecip == 0] = np.nan
PythonTotalPrecip[PythonTotalPrecip == 0] = np.nan

IDL_MCSPrecip[IDL_MCSPrecip == 0] = np.nan
Python_MCSPrecip[Python_MCSPrecip == 0] = np.nan

IDL_MCSPrecipFraction[~np.isfinite(Python_MCSPrecipFraction)] = np.nan
IDL_MCSPrecipFraction[~np.isfinite(Python_MCSPrecipFraction)] = np.nan

IDL_MCSFrequency[IDL_MCSFrequency == 0] = np.nan
Python_MCSFrequency[Python_MCSFrequency == 0] = np.nan

##################################################
# Plot total accumulation map
PrecipMax = np.nanmax([np.nanmax(IDLTotalPrecip), np.nanmax(PythonTotalPrecip)])

plt.figure()
plt.title('Accumulated Precipitation \n All Features', fontsize=14, y=1.01)
map = Basemap(lon_0=-90, lat_0=37, llcrnrlon=-110, llcrnrlat=25, urcrnrlon=-70, urcrnrlat=50)
map.drawcoastlines(linewidth=1)
map.drawcountries(linewidth=1)
map.drawstates(linewidth=1)
LongMap, LatMap = map(Longitude[0, :, :], Latitude[0, :, :])
im = map.pcolormesh(LongMap, LatMap, np.ma.masked_invalid(np.atleast_2d(IDLTotalPrecip)), cmap='CMRmap_r', vmin=0, vmax=PrecipMax)
cbar = plt.colorbar(im)
cbar.ax.set_ylabel('Accumulated Precipitation [mm]', fontsize=12, labelpad=10)
cbar.ax.tick_params(labelsize=10)
plt.savefig(Figure_Location + 'TotalPrecipitation_' + str(int(AnalyzeYear)) + '0' + str(int(AnalyzeMonth)) + '.png')
#plt.savefig(Figure_Location + 'TotalPrecipitation.png')
plt.close()

##################################################
# Plot IDL total accumulation maps

plt.figure()
plt.title('Accumulated Precipitation \n IDL MCSs only', fontsize=14, y=1.01)
map = Basemap(lon_0=-90, lat_0=37, llcrnrlon=-110, llcrnrlat=25, urcrnrlon=-70, urcrnrlat=50)
map.drawcoastlines(linewidth=1)
map.drawcountries(linewidth=1)
map.drawstates(linewidth=1)
im = map.pcolormesh(LongMap, LatMap, np.ma.masked_invalid(np.atleast_2d(IDL_MCSPrecip)), cmap='CMRmap_r', vmin=0, vmax=PrecipMax)
cbar = plt.colorbar(im)
cbar.ax.set_ylabel('Accumulated MCS Precipitation [mm]', fontsize=12, labelpad=10)
cbar.ax.tick_params(labelsize=10)
plt.savefig(Figure_Location + 'IDL_MCSPrecipitation_' + str(int(AnalyzeYear)) + '0' + str(int(AnalyzeMonth)) + '.png')
#plt.savefig(Figure_Location + 'IDL_MCSPrecipitation.png')
plt.close()

#FractionMax = np.nanmax([np.nanmax(IDL_MCSPrecipFraction), np.nanmax(Python_MCSPrecipFraction)])
#plt.figure()
#plt.title('Fraction of Accumulated Precipitation \n Attributable to IDL MCSs', fontsize=14, y=1.01)
#map = Basemap(lon_0=-90, lat_0=37, llcrnrlon=-110, llcrnrlat=25, urcrnrlon=-70, urcrnrlat=50)
#map.drawcoastlines(linewidth=1)
#map.drawcountries(linewidth=1)
#map.drawstates(linewidth=1)
#im = map.pcolormesh(LongMap, LatMap, np.ma.masked_invalid(np.atleast_2d(IDL_MCSPrecipFraction)), cmap='CMRmap_r', vmin=0, vmax=FractionMax)
#cbar = plt.colorbar(im)
#cbar.ax.set_ylabel('Accumulated MCS Precipitation Fraction (MCS / Total)', fontsize=12, labelpad=10)
#cbar.ax.tick_params(labelsize=10)
#plt.savefig(Figure_Location + 'IDL_MCSPrecipFraction_' + str(int(AnalyzeYear)) + '0' + str(int(AnalyzeMonth)) + '.png')
##plt.savefig(Figure_Location + 'IDL_MCSPrecipFraction.png')
#plt.close()

FrequencyMax = np.nanmax([np.nanmax(IDL_MCSFrequency), np.nanmax(Python_MCSFrequency)])
plt.figure()
plt.title('IDL MCS Frequency', fontsize=14, y=1.01)
map = Basemap(lon_0=-90, lat_0=37, llcrnrlon=-110, llcrnrlat=25, urcrnrlon=-70, urcrnrlat=50)
map.drawcoastlines(linewidth=1)
map.drawcountries(linewidth=1)
map.drawstates(linewidth=1)
im = map.pcolormesh(LongMap, LatMap, np.ma.masked_invalid(np.atleast_2d(IDL_MCSFrequency)), cmap='CMRmap_r', vmin=0, vmax=FrequencyMax)
cbar = plt.colorbar(im)
cbar.ax.set_ylabel('MCS Frequency (# MCS / Total # Files)', fontsize=12, labelpad=10)
cbar.ax.tick_params(labelsize=10)
plt.savefig(Figure_Location + 'IDL_MCSFrequency_' + str(int(AnalyzeYear)) + '0' + str(int(AnalyzeMonth)) + '.png')
#plt.savefig(Figure_Location + 'IDL_MCSFrequency.png')
plt.close()

##################################################
# Plot Python total accumulation maps

plt.figure()
plt.title('Accumulated Precipitation \n Python MCSs only', fontsize=14, y=1.01)
map = Basemap(lon_0=-90, lat_0=37, llcrnrlon=-110, llcrnrlat=25, urcrnrlon=-70, urcrnrlat=50)
map.drawcoastlines(linewidth=1)
map.drawcountries(linewidth=1)
map.drawstates(linewidth=1)
im = map.pcolormesh(LongMap, LatMap, np.ma.masked_invalid(np.atleast_2d(Python_MCSPrecip)), cmap='CMRmap_r', vmin=0, vmax=PrecipMax)
cbar = plt.colorbar(im)
cbar.ax.set_ylabel('Accumulated MCS Precipitation [mm]', fontsize=12, labelpad=10)
cbar.ax.tick_params(labelsize=10)
plt.savefig(Figure_Location + 'Python_MCSPrecipitation_' + str(int(AnalyzeYear)) + '0' + str(int(AnalyzeMonth)) + '.png')
#plt.savefig(Figure_Location + 'Python_MCSPrecipitation.png')
plt.close()

#plt.figure()
#plt.title('Fraction of Accumulated Precipitation \n Attributable to Python MCSs', fontsize=14, y=1.01)
#map = Basemap(lon_0=-90, lat_0=37, llcrnrlon=-110, llcrnrlat=25, urcrnrlon=-70, urcrnrlat=50)
#map.drawcoastlines(linewidth=1)
#map.drawcountries(linewidth=1)
#map.drawstates(linewidth=1)
#im = map.pcolormesh(LongMap, LatMap, np.ma.masked_invalid(np.atleast_2d(Python_MCSPrecipFraction)), cmap='CMRmap_r', vmin=0, vmax=FractionMax)
#cbar = plt.colorbar(im)
#cbar.ax.set_ylabel('Accumulated MCS Precipitation Fraction (MCS / Total)', fontsize=12, labelpad=10)
#cbar.ax.tick_params(labelsize=10)
#plt.savefig(Figure_Location + 'Python_MCSPrecipFraction_' + str(int(AnalyzeYear)) + '0' + str(int(AnalyzeMonth)) + '.png')
##plt.savefig(Figure_Location + 'Python_MCSPrecipFraction.png')
#plt.close()

plt.figure()
plt.title('Python MCS Frequency', fontsize=14, y=1.01)
map = Basemap(lon_0=-90, lat_0=37, llcrnrlon=-110, llcrnrlat=25, urcrnrlon=-70, urcrnrlat=50)
map.drawcoastlines(linewidth=1)
map.drawcountries(linewidth=1)
map.drawstates(linewidth=1)
im = map.pcolormesh(LongMap, LatMap, np.ma.masked_invalid(np.atleast_2d(Python_MCSFrequency)), cmap='CMRmap_r', vmin=0, vmax=FrequencyMax)
cbar = plt.colorbar(im)
cbar.ax.set_ylabel('MCS Frequency (# MCS / Total # Files)', fontsize=12, labelpad=10)
cbar.ax.tick_params(labelsize=10)
plt.savefig(Figure_Location + 'Python_MCSFrequency_' + str(int(AnalyzeYear)) + '0' + str(int(AnalyzeMonth)) + '.png')
#plt.savefig(Figure_Location + 'Python_MCSFrequency.png')
plt.close()

#########################################################
# Plot Differences

if np.absolute(np.nanmin(Difference_MCSPrecip)) > np.nanmax(Difference_MCSPrecip):
    ColorValues =  np.absolute(np.nanmin(Difference_MCSPrecip))
else:
    ColorValues =  np.nanmax(Difference_MCSPrecip)
plt.figure()
plt.title('IDL - Python Version Difference \n MCS Precipitation', fontsize=14, y=1.01)
map = Basemap(lon_0=-90, lat_0=37, llcrnrlon=-110, llcrnrlat=25, urcrnrlon=-70, urcrnrlat=50)
map.drawcoastlines(linewidth=1)
map.drawcountries(linewidth=1)
map.drawstates(linewidth=1)
im = map.pcolormesh(Longitude[0, :, :], Latitude[0, :, :], np.ma.masked_invalid(np.atleast_2d(Difference_MCSPrecip)), cmap='coolwarm', vmin=-1*ColorValues, vmax=ColorValues)
cbar = plt.colorbar(im)
cbar.ax.set_ylabel('MCS Precipitation Difference (IDL - Python)', fontsize=12, labelpad=10)
cbar.ax.tick_params(labelsize=10)
plt.savefig(Figure_Location + 'IDLPython_MCSPrecipitationDifference_' + str(int(AnalyzeYear)) + '0' + str(int(AnalyzeMonth)) + '.png')
#plt.savefig(Figure_Location + 'IDLPython_MCSPrecipitationDifference.png')
plt.close()

#if np.absolute(np.nanmin(Difference_MCSPrecipFraction)) > np.nanmax(Difference_MCSPrecipFraction):
#    ColorValues =  np.absolute(np.nanmin(Difference_MCSPrecipFraction))
#else:
#    ColorValues =  np.nanmax(Difference_MCSPrecipFraction)
#plt.figure()
#plt.title('IDL - Python Version Difference \n MCS Precipitation Fraction', fontsize=14, y=1.01)
#map = Basemap(lon_0=-90, lat_0=37, llcrnrlon=-110, llcrnrlat=25, urcrnrlon=-70, urcrnrlat=50)
#map.drawcoastlines(linewidth=1)
#map.drawcountries(linewidth=1)
#map.drawstates(linewidth=1)
#im = map.pcolormesh(Longitude[0, :, :], Latitude[0, :, :], np.ma.masked_invalid(np.atleast_2d(Difference_MCSPrecipFraction)), cmap='coolwarm', vmin=-1*ColorValues, vmax=ColorValues)
#cbar = plt.colorbar(im)
#cbar.ax.set_ylabel('MCS Precipitation Fraction Difference (IDL - Python)', fontsize=12, labelpad=10)
#cbar.ax.tick_params(labelsize=10)
##plt.savefig(Figure_Location + 'IDLPython_MCSPrecipFractionDifference_' + str(int(AnalyzeYear)) + '0' + str(int(AnalyzeMonth)) + '.png')
#plt.savefig(Figure_Location + 'IDLPython_MCSPrecipFractionDifference.png')
#plt.close()

if np.absolute(np.nanmin(Difference_MCSFrequency)) > np.nanmax(Difference_MCSFrequency):
    ColorValues =  np.absolute(np.nanmin(Difference_MCSFrequency))
else:
    ColorValues =  np.nanmax(Difference_MCSFrequency)
plt.figure()
plt.title('IDL - Python Version Difference \n MCS Frequency', fontsize=14, y=1.01)
map = Basemap(lon_0=-90, lat_0=37, llcrnrlon=-110, llcrnrlat=25, urcrnrlon=-70, urcrnrlat=50)
map.drawcoastlines(linewidth=1)
map.drawcountries(linewidth=1)
map.drawstates(linewidth=1)
im = map.pcolormesh(Longitude[0, :, :], Latitude[0, :, :], np.ma.masked_invalid(np.atleast_2d(Difference_MCSFrequency)), cmap='coolwarm', vmin=-1*ColorValues, vmax=ColorValues)
cbar = plt.colorbar(im)
cbar.ax.set_ylabel('MCS Frequency Difference (IDL - Python)', fontsize=12, labelpad=10)
cbar.ax.tick_params(labelsize=10)
plt.savefig(Figure_Location + 'IDLPython_MCSFrequencyDifference_' + str(int(AnalyzeYear)) + '0' + str(int(AnalyzeMonth)) + '.png')
#plt.savefig(Figure_Location + 'IDLPython_MCSFrequencyDifference.png')
plt.close()

#plt.close()
#plt.show()

############################################################3
# Plot individual cases
list_Latitude = [Latitude]*nFiles
list_Longitude = [Longitude]*nFiles

IDLCloudTrack[np.isnan(IDLCloudTrack)] = 0

Input = zip(IDLBrightnessTemperature, IDLCloudTrack, PythonCloudTrack, PythonBasetime, list_Latitude, list_Longitude)

PythonIndex = np.empty(IDLNumFiles, dtype='int')
MatchingDate = [None]*IDLNumFiles
MatchingTime = [None]*IDLNumFiles
for ifile in range(0, IDLNumFiles):
    ibasetime = np.copy(IDLbasetime[ifile]).astype('datetime64[ns]')
    PythonIndex[ifile] = np.array(np.where(PythonBasetime == ibasetime))[0]
    ibasetime = ibasetime.astype('datetime64[s]')

    MatchingDate[ifile] = str(np.array(PythonBasetime[PythonIndex[ifile]]).astype('datetime64[s]'))[0:10]
    MatchingTime[ifile] = str(np.array(PythonBasetime[PythonIndex[ifile]]).astype('datetime64[s]'))[11:16]

print(PythonIndex)
print(MatchingDate)
print(MatchingTime)


    #ibasetime = ibasetime.astype('datetime64[s]')
    #fig = plt.figure()
    #fig.suptitle('IDL and Python Brightness Temperature and MCS Tracks \n ' + str(ibasetime) + ' UTC', fontsize=16, y=0.75)
    #fig.set_figheight(10)
    #fig.set_figwidth(20)

    #ax0 = fig.add_axes([0.1, 0.25, 0.25, 0.5])
    #ax1 = fig.add_axes([0.4, 0.25, 0.25, 0.5])
    #ax2 = fig.add_axes([0.7, 0.25, 0.25, 0.5])

    #ax0.set_title('Raw Brightness Temperature', fontsize=14, y=1.05)
    #map0 = Basemap(lon_0=-90, lat_0=37, llcrnrlon=-110, llcrnrlat=25, urcrnrlon=-70, urcrnrlat=50, ax=ax0)
    #map0.drawcoastlines(linewidth=1)
    #map0.drawcountries(linewidth=1)
    #map0.drawstates(linewidth=1)
    #im0 = map0.pcolormesh(LongMap, LatMap, np.ma.masked_invalid(np.atleast_2d(IDLBrightnessTemperature[ifile, :, :])), cmap='gist_stern', vmin=200, vmax=300)
    #cbar0 = plt.colorbar(im0, ax=ax0, fraction=0.03, pad=0.04)
    #cbar0.ax.set_xlabel('K', fontsize=14, labelpad=10)
    #cbar0.ax.tick_params(labelsize=12)

    #IDLcase = np.copy(IDLCloudTrack[ifile, :, :])
    #IDLcase[np.isnan(IDLcase)] = 0
    #ax1.set_title('IDL MCS Tracks', fontsize=14, y=1.05)
    #map1 = Basemap(lon_0=-90, lat_0=37, llcrnrlon=-110, llcrnrlat=25, urcrnrlon=-70, urcrnrlat=50, ax=ax1)
    #map1.drawcoastlines(linewidth=1)
    #map1.drawcountries(linewidth=1)
    #map1.drawstates(linewidth=1)
    #im1 = map1.pcolormesh(LongMap, LatMap, np.ma.masked_invalid(np.atleast_2d(IDLcase)), cmap='nipy_spectral_r', vmin=0, vmax=NumIDLTracks+1, linewidth=0)
    #cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.03, pad=0.04)
    #cbar1.ax.set_xlabel('#', fontsize=14, labelpad=10)
    #ax1.set_xlabel('Tracks Present: ' + str(np.unique(IDLcase)[1::]), fontsize=14, labelpad=20)
    #cbar1.ax.tick_params(labelsize=12)

    #PythonCase = np.copy(PythonCloudTrack[PythonIndex, :, :])
    #PythonCase[np.isnan(PythonCase)] = 0
    #ax2.set_title('Python MCS Tracks', fontsize=14, y=1.05)
    #map2 = Basemap(lon_0=-90, lat_0=37, llcrnrlon=-110, llcrnrlat=25, urcrnrlon=-70, urcrnrlat=50, ax=ax2)
    #map2.drawcoastlines(linewidth=1)
    #map2.drawcountries(linewidth=1)
    #map2.drawstates(linewidth=1)
    #im2 = map2.pcolormesh(LongMap, LatMap, np.ma.masked_invalid(np.atleast_2d(PythonCase[0, :, :])), cmap='nipy_spectral_r', vmin=0, vmax=NumPythonTracks+1, linewidth=0)
    #cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.03, pad=0.04)
    #cbar2.ax.set_xlabel('#', fontsize=14, labelpad=10)
    #ax2.set_xlabel('Tracks Present: ' + str(np.unique(PythonCase)[1::]), fontsize=14, labelpad=20)
    #cbar2.ax.tick_params(labelsize=12)

    #plt.savefig(Figure_Location + 'IDLPythonComparison_' + str(np.array(PythonBasetime[PythonIndex]))[2:12] + '_' + str(np.array(PythonBasetime[PythonIndex]))[13:18] + '.png')
    #plt.close()
