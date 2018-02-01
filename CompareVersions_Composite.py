# Before running enter ... source activate /global/homes/h/hcbarnes/python_parallel

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
import xarray as xr
from mpl_toolkits.basemap import Basemap
import time
import datetime
import glob
import dask
import os
import fnmatch
import pandas as pd
np.set_printoptions(threshold=np.inf)

################################################
# Set locations
#IDL_Location = '/global/project/projectdirs/m1867/zfeng/usa/mergedir/mcstracking/20110517_20110527/'
IDL_Location = '/global/cscratch1/sd/hcbarnes/mcs/IDL/mcstracking/20110401_20110831/'
#IDL_Location = '/global/project/projectdirs/m1657/zfeng/usa/mergedir/mcstracking/20110401_20110831/'

#Python_Location = '/global/homes/h/hcbarnes/Tracking/SatelliteRadar/mcstracking/20110517_20110527/'
Python_Location = '/global/cscratch1/sd/hcbarnes/mcs/Python/mcstracking/20110401_20110831/'
#Python_Location = '/global/homes/h/hcbarnes/Tracking/SatelliteRadar/mcstracking/20110401_20110831/'

Figure_Location = '/global/homes/h/hcbarnes/Tracking/SatelliteRadar/figures/'

##############################################
# Set analysis time
AnalyzeYear = 2011
AnalyzeMonth = 8

##############################################
# Function that plots individual maps of Python cloud tracks, IDL cloud tracks, and cloud brightness temperature
def Plot_CloudTracks(zipped_inputs):

    ########################################
    # Seperate inputs
    FileStep = zipped_inputs[0]
    BrightnessTemperature = zipped_inputs[1]
    IDLCloudTracks = zipped_inputs[2]
    MinIDLTracks = zipped_inputs[3]
    MaxIDLTracks = zipped_inputs[4]
    PythonCloudTracks = zipped_inputs[5]
    MinPythonTracks = zipped_inputs[6]
    MaxPythonTracks = zipped_inputs[7]
    BaseTime = zipped_inputs[8]
    Latitude = zipped_inputs[9]
    Longitude = zipped_inputs[10]

    #####################################
    # Set figure location
    Figure_Location = '/global/homes/h/hcbarnes/Tracking/SatelliteRadar/figures/'

    ######################################
    # Separate Basetime in day and time
    BaseTime = str(pd.to_datetime(BaseTime))
    Date = BaseTime[0:10]
    Time = BaseTime[11:16]

    ######################################
    # Create figure
    fig = plt.figure()
    fig.suptitle('IDL and Python Brightness Temperature and MCS Tracks \n ' + Date + ' ' + Time + ' UTC', fontsize=10, y=0.75)
    fig.set_figheight(10)
    fig.set_figwidth(20)

    ax0 = fig.add_axes([0.05, 0.25, 0.25, 0.5], projection=ccrs.PlateCarree())
    ax1 = fig.add_axes([0.38, 0.25, 0.25, 0.5], projection=ccrs.PlateCarree())
    ax2 = fig.add_axes([0.7, 0.25, 0.25, 0.5], projection=ccrs.PlateCarree())

    #ax0 = plt.axes(projection=ccrs.PlateCarree())
    ax0.set_title('Brightness Temperature', fontsize=10, y=1.05)
    ax0.add_feature(cfeature.COASTLINE)
    ax0.add_feature(cfeature.BORDERS)
    ax0.add_feature(cfeature.LAKES, alpha=0.5)
    ax0.add_feature(cfeature.OCEAN, alpha=0.5)
    ax0.add_feature(states_provinces, edgecolor='gray')
    ax0.set_xmargin(0)
    ax0.set_ymargin(0)
    im0 = ax0.pcolormesh(Longitude, Latitude, np.ma.masked_invalid(np.atleast_2d(BrightnessTemperature)), cmap='gist_stern', vmin=200, vmax=300, transform=ccrs.PlateCarree())
    cbar0 = plt.colorbar(im0, ax=ax0, fraction=0.03, pad=0.04)
    cbar0.ax.set_xlabel('K', fontsize=8, labelpad=8)
    cbar0.ax.tick_params(labelsize=8)

    #ax1 = plt.axes(projection=ccrs.PlateCarree())
    ax1.set_title('IDL MCS Tracks', fontsize=10, y=1.05)
    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(cfeature.BORDERS)
    ax1.add_feature(cfeature.LAKES, alpha=0.5)
    ax1.add_feature(cfeature.OCEAN, alpha=0.5)
    ax1.add_feature(states_provinces, edgecolor='gray')
    ax1.set_xmargin(0)
    ax1.set_ymargin(0)
    im1 = ax1.pcolormesh(Longitude, Latitude, np.ma.masked_invalid(np.atleast_2d(IDLCloudTracks)), cmap='nipy_spectral_r', vmin=MinIDLTracks, vmax=MaxIDLTracks, linewidth=0, transform=ccrs.PlateCarree())
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.03, pad=0.04)
    cbar1.ax.set_xlabel('#', fontsize=8, labelpad=10)
    ax1.set_xlabel('Tracks Present: ' + str(np.unique(IDLCloudTracks)[1::]), fontsize=8)
    cbar1.ax.tick_params(labelsize=8)

    #ax2 = plt.axes(projection=ccrs.PlateCarree())
    ax2.set_title('Python MCS Tracks', fontsize=10, y=1.05)
    ax2.add_feature(cfeature.COASTLINE)
    ax2.add_feature(cfeature.BORDERS)
    ax2.add_feature(cfeature.LAKES, alpha=0.5)
    ax2.add_feature(cfeature.OCEAN, alpha=0.5)
    ax2.add_feature(states_provinces, edgecolor='gray')
    ax2.set_xmargin(0)
    ax2.set_ymargin(0)
    im2 = ax2.pcolormesh(Longitude, Latitude, np.ma.masked_invalid(np.atleast_2d(PythonCloudTracks)), cmap='nipy_spectral_r', vmin=MinPythonTracks, vmax=MaxPythonTracks, linewidth=0, transform=ccrs.PlateCarree())
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.03, pad=0.04)
    cbar2.ax.set_xlabel('#', fontsize=8, labelpad=10)
    ax2.set_xlabel('Tracks Present: ' + str(np.unique(PythonCloudTracks)[1::]), fontsize=8)
    cbar2.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(Figure_Location + 'IDLPythonComparison_' + Date + '_' + Time + '.png')
    plt.close()

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
print('Loading data')

#IDLdata = xr.open_mfdataset(IDL_Location + '*.nc', concat_dim='time', autoclose=True)
IDLdata = xr.open_mfdataset(IDL_Files, concat_dim='time', autoclose=True)
IDLbasetime = IDLdata['base_time'].data
IDLCloudTrack = np.array(IDLdata['cloudtracknumber'].data)
IDLPrecipitation = np.array(IDLdata['precipitation'].data)
IDLBrightnessTemperature = np.array(IDLdata['tb'].data)
Longitude = np.array(IDLdata['longitude'].data)
Latitude= np.array(IDLdata['latitude'].data)
IDLCloudTrack[np.isnan(IDLCloudTrack)] = 0
nIDL = len(IDLbasetime)
IDLCloudTrackNumbers = np.unique(IDLCloudTrack)
MinIDL = IDLCloudTrackNumbers[1]
MaxIDL = IDLCloudTrackNumbers[-1]

#PythonData = xr.open_mfdataset(Python_Location + '*.nc', concat_dim='time', autoclose=True)
PythonData = xr.open_mfdataset(Python_Files, concat_dim='time', autoclose=True)
PythonBasetime = np.array(PythonData['basetime'].data)
PythonCloudTrack = np.array(PythonData['cloudtracknumber'].data)
PythonPrecipitation = np.array(PythonData['precipitation'].data)
nPython = len(PythonBasetime)
PythonCloudTrackNumbers = np.unique(PythonCloudTrack)
MinPython = PythonCloudTrackNumbers[1]
MaxPython = PythonCloudTrackNumbers[-1]

PythonTimeCoordinate = PythonData.coords['time'].values

print('Data Loaded')

#################################################
# Make sure all the same files are here
print('Matching Data')

if nIDL > nPython:
    DifferentBasetimes = np.setdiff1d(IDLbasetime, PythonBasetime)
    DifferentIndices = np.ones(len(DifferentBasetimes))
    for iDifferent in range(0, len(DifferentBasetimes)):
        DifferentIndices[iDifferent] = np.array(np.where(IDLbasetime == DifferentBasetimes[iDifferent]))
    SameIDLIndices = np.setdiff1d(np.arange(0, nIDL), DifferentIndices).astype(int)

    IDL_MatchedFiles = list(IDL_Files[i] for i in SameIDLIndices)
    IDLbasetime = IDLbasetime[SameIDLIndices]
    IDLCloudTrack = IDLCloudTrack[SameIDLIndices, :, :]
    IDLPrecipitation = IDLPrecipitation[SameIDLIndices, :, :]
    IDLBrightnessTemperature = IDLBrightnessTemperature[SameDILIndices, :, :]

    DifferentBasetimes = np.setdiff1d(PythonBasetime, IDLbasetime)
    DifferentIndices = np.ones(len(DifferentBasetimes))
    for iDifferent in range(0, len(DifferentBasetimes)):
        DifferentIndices[iDifferent] = np.array(np.where(PythonBasetime == DifferentBasetimes[iDifferent]))
    SamePythonIndices = np.setdiff1d(np.arange(0, nPython), DifferentIndices).astype(int)

    Python_MatchedFiles = list(Python_Files[i] for i in SamePythonIndices)
    PythonBasetime = PythonBasetime[SamePythonIndices]
    PythonCloudTrack = PythonCloudTrack[SamePythonIndices, :, :]
    PythonPrecipitation = PythonPrecipitation[SamePythonIndices, :, :]

else:
    DifferentBasetimes = np.setdiff1d(PythonBasetime, IDLbasetime)
    DifferentIndices = np.ones(len(DifferentBasetimes))
    for iDifferent in range(0, len(DifferentBasetimes)):
        DifferentIndices[iDifferent] = np.array(np.where(PythonBasetime == DifferentBasetimes[iDifferent]))
    SamePythonIndices = np.setdiff1d(np.arange(0, nPython), DifferentIndices).astype(int)

    Python_MatchedFiles = list(Python_Files[i] for i in SamePythonIndices)
    PythonBasetime = PythonBasetime[SamePythonIndices]
    PythonCloudTrack = PythonCloudTrack[SamePythonIndices, :, :]
    PythonPrecipitation = PythonPrecipitation[SamePythonIndices, :, :]

    DifferentBasetimes = np.setdiff1d(IDLbasetime, PythonBasetime)
    DifferentIndices = np.ones(len(DifferentBasetimes))
    for iDifferent in range(0, len(DifferentBasetimes)):
        DifferentIndices[iDifferent] = np.array(np.where(IDLbasetime == DifferentBasetimes[iDifferent]))
    SameIDLIndices = np.setdiff1d(np.arange(0, nIDL), DifferentIndices).astype(int)

    IDL_MatchedFiles = list(IDL_Files[i] for i in SameIDLIndices)
    IDLbasetime = IDLbasetime[SameIDLIndices]
    IDLCloudTrack = IDLCloudTrack[SameIDLIndices, :, :]
    IDLPrecipitation = IDLPrecipitation[SameIDLIndices, :, :]
    IDLBrightnessTemperature = IDLBrightnessTemperature[SameIDLIndices, :, :]

nFiles = len(PythonBasetime)

####################################################
# Save list of individual data to process

Intro = '/global/homes/h/hcbarnes/Tracking/Python/run_IDLPythonComparison_TaskFarmer.sh '
ProcessingList = np.vstack(([Intro]*nFiles, np.asarray(IDL_MatchedFiles), [' ']*nFiles, np.asarray(Python_MatchedFiles), [' ']*nFiles, [str(int(np.unique(IDLCloudTrack)[1]))]*nFiles, [' ']*nFiles, [str(int(np.unique(IDLCloudTrack)[-1]))]*nFiles, [' ']*nFiles, [str(int(np.unique(PythonCloudTrack)[1]))]*nFiles, [' ']*nFiles, [str(int(np.unique(PythonCloudTrack)[-1]))]*nFiles))
with open('IDLPythonComparisonList.txt', 'w') as f:
    for iFile in range(0, nFiles):
        f.write(str(''.join(ProcessingList[:, iFile])+'\n'))

####################################################
# Get data dimensions
IDLNumFiles, IDLNumLat, IDLNumLong = np.shape(IDLCloudTrack)
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
print('Calculating cumuluative precipitation statistics')

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
print('Plotting cumulutative statistics')

states_provinces = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='50m', facecolor='none')

PrecipMax = np.nanmax([np.nanmax(IDLTotalPrecip), np.nanmax(PythonTotalPrecip)])

plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
plt.title('Accumulated Precipitation \n All Features', fontsize=14, y=1.01)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.OCEAN, alpha=0.5)
ax.add_feature(states_provinces, edgecolor='gray')
ax.set_xmargin(0)
ax.set_ymargin(0)
im = plt.pcolormesh(Longitude[0, :, :], Latitude[0, :, :], np.ma.masked_invalid(np.atleast_2d(IDLTotalPrecip)), cmap='CMRmap_r', vmin=0, vmax=PrecipMax/float(1.5), transform=ccrs.PlateCarree())
cbar = plt.colorbar(im)
cbar.ax.set_ylabel('Accumulated Precipitation [mm]', fontsize=12, labelpad=10)
cbar.ax.tick_params(labelsize=10)
plt.savefig(Figure_Location + 'TotalPrecipitation_' + str(int(AnalyzeYear)) + '0' + str(int(AnalyzeMonth)) + '.png')
#plt.savefig(Figure_Location + 'TotalPrecipitation.png')
plt.close()

##################################################
# Plot IDL total accumulation maps

plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
plt.title('Accumulated Precipitation \n IDL MCSs only', fontsize=14, y=1.01)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.OCEAN, alpha=0.5)
ax.add_feature(states_provinces, edgecolor='gray')
ax.set_xmargin(0)
ax.set_ymargin(0)
im = plt.pcolormesh(Longitude[0, :, :], Latitude[0, :, :], np.ma.masked_invalid(np.atleast_2d(IDL_MCSPrecip)), cmap='CMRmap_r', vmin=0, vmax=PrecipMax/float(1.5), transform=ccrs.PlateCarree())
cbar = plt.colorbar(im)
cbar.ax.set_ylabel('Accumulated MCS Precipitation [mm]', fontsize=12, labelpad=10)
cbar.ax.tick_params(labelsize=10)
plt.savefig(Figure_Location + 'IDL_MCSPrecipitation_' + str(int(AnalyzeYear)) + '0' + str(int(AnalyzeMonth)) + '.png')
#plt.savefig(Figure_Location + 'IDL_MCSPrecipitation.png')
plt.close()

#FractionMax = np.nanmax([np.nanmax(IDL_MCSPrecipFraction), np.nanmax(Python_MCSPrecipFraction)])
#plt.figure()
#ax = plt.axes(projection=ccrs.PlateCarree())
#plt.title('Fraction of Accumulated Precipitation \n Attributable to IDL MCSs', fontsize=14, y=1.01)
#ax.add_feature(cfeature.COASTLINE)
#ax.add_feature(cfeature.BORDERS)
#ax.add_feature(cfeature.LAKES, alpha=0.5)
#ax.add_feature(cfeature.OCEAN, alpha=0.5)
#ax.add_feature(states_provinces, edgecolor='gray')
#ax.set_xmargin(0)
#ax.set_ymargin(0)
#im = plt.pcolormesh(Longitude[0, :, :], Latitude[0, :, :], np.ma.masked_invalid(np.atleast_2d(IDL_MCSPrecipFraction)), cmap='CMRmap_r', vmin=0, vmax=FractionMax, transform=ccrs.PlateCarree())
#cbar = plt.colorbar(im)
#cbar.ax.set_ylabel('Accumulated MCS Precipitation Fraction (MCS / Total)', fontsize=12, labelpad=10)
#cbar.ax.tick_params(labelsize=10)
#plt.savefig(Figure_Location + 'IDL_MCSPrecipFraction_' + str(int(AnalyzeYear)) + '0' + str(int(AnalyzeMonth)) + '.png')
##plt.savefig(Figure_Location + 'IDL_MCSPrecipFraction.png')
#plt.close()

FrequencyMax = np.nanmax([np.nanmax(IDL_MCSFrequency), np.nanmax(Python_MCSFrequency)])
plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
plt.title('IDL MCS Frequency', fontsize=14, y=1.01)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.OCEAN, alpha=0.5)
ax.add_feature(states_provinces, edgecolor='gray')
ax.set_xmargin(0)
ax.set_ymargin(0)
im = plt.pcolormesh(Longitude[0, :, :], Latitude[0, :, :], np.ma.masked_invalid(np.atleast_2d(IDL_MCSFrequency)), cmap='CMRmap_r', vmin=0, vmax=FrequencyMax, transform=ccrs.PlateCarree())
cbar = plt.colorbar(im)
cbar.ax.set_ylabel('MCS Frequency (# MCS / Total # Files)', fontsize=12, labelpad=10)
cbar.ax.tick_params(labelsize=10)
plt.savefig(Figure_Location + 'IDL_MCSFrequency_' + str(int(AnalyzeYear)) + '0' + str(int(AnalyzeMonth)) + '.png')
#plt.savefig(Figure_Location + 'IDL_MCSFrequency.png')
plt.close()

##################################################
# Plot Python total accumulation maps

plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
plt.title('Accumulated Precipitation \n Python MCSs only', fontsize=14, y=1.01)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.OCEAN, alpha=0.5)
ax.add_feature(states_provinces, edgecolor='gray')
ax.set_xmargin(0)
ax.set_ymargin(0)
im = plt.pcolormesh(Longitude[0, :, :], Latitude[0, :, :], np.ma.masked_invalid(np.atleast_2d(Python_MCSPrecip)), cmap='CMRmap_r', vmin=0, vmax=PrecipMax/float(1.5), transform=ccrs.PlateCarree())
cbar = plt.colorbar(im)
cbar.ax.set_ylabel('Accumulated MCS Precipitation [mm]', fontsize=12, labelpad=10)
cbar.ax.tick_params(labelsize=10)
plt.savefig(Figure_Location + 'Python_MCSPrecipitation_' + str(int(AnalyzeYear)) + '0' + str(int(AnalyzeMonth)) + '.png')
#plt.savefig(Figure_Location + 'Python_MCSPrecipitation.png')
plt.close()

#plt.figure()
#ax = plt.axes(projection=ccrs.PlateCarree())
#plt.title('Fraction of Accumulated Precipitation \n Attributable to Python MCSs', fontsize=14, y=1.01)
#ax.add_feature(cfeature.COASTLINE)
#ax.add_feature(cfeature.BORDERS)
#ax.add_feature(cfeature.LAKES, alpha=0.5)
#ax.add_feature(cfeature.OCEAN, alpha=0.5)
#ax.add_feature(states_provinces, edgecolor='gray')
#ax.set_xmargin(0)
#ax.set_ymargin(0)
#im = plt.pcolormesh(Longitude[0, :, :], Latitude[0, :, :], np.ma.masked_invalid(np.atleast_2d(Python_MCSPrecipFraction)), cmap='CMRmap_r', vmin=0, vmax=FractionMax, transform=ccrs.PlateCarree())
#cbar = plt.colorbar(im)
#cbar.ax.set_ylabel('Accumulated MCS Precipitation Fraction (MCS / Total)', fontsize=12, labelpad=10)
#cbar.ax.tick_params(labelsize=10)
#plt.savefig(Figure_Location + 'Python_MCSPrecipFraction_' + str(int(AnalyzeYear)) + '0' + str(int(AnalyzeMonth)) + '.png')
##plt.savefig(Figure_Location + 'Python_MCSPrecipFraction.png')
#plt.close()

plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
plt.title('Python MCS Frequency', fontsize=14, y=1.01)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.OCEAN, alpha=0.5)
ax.add_feature(states_provinces, edgecolor='gray')
ax.set_xmargin(0)
ax.set_ymargin(0)
im = plt.pcolormesh(Longitude[0, :, :], Latitude[0, :, :], np.ma.masked_invalid(np.atleast_2d(Python_MCSFrequency)), cmap='CMRmap_r', vmin=0, vmax=FrequencyMax, transform=ccrs.PlateCarree())
cbar = plt.colorbar(im)
cbar.ax.set_ylabel('MCS Frequency (# MCS / Total # Files)', fontsize=12, labelpad=10)
cbar.ax.tick_params(labelsize=10)
plt.savefig(Figure_Location + 'Python_MCSFrequency_' + str(int(AnalyzeYear)) + '0' + str(int(AnalyzeMonth)) + '.png')
#plt.savefig(Figure_Location + 'Python_MCSFrequency.png')
plt.close()

#########################################################
# Plot Differences

if np.absolute(np.nanmin(Difference_MCSPrecip)) > np.nanmax(Difference_MCSPrecip):
    ColorValues =  np.absolute(np.nanmin(Difference_MCSPrecip))/3
else:
    ColorValues =  np.nanmax(Difference_MCSPrecip)/3
plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
plt.title('IDL - Python Version Difference \n MCS Precipitation', fontsize=14, y=1.01)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.OCEAN, alpha=0.5)
ax.add_feature(states_provinces, edgecolor='gray')
ax.set_xmargin(0)
ax.set_ymargin(0)
im = plt.pcolormesh(Longitude[0, :, :], Latitude[0, :, :], np.ma.masked_invalid(np.atleast_2d(Difference_MCSPrecip)), cmap='bwr', vmin=-1*ColorValues, vmax=ColorValues, transform=ccrs.PlateCarree())
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
#ax = plt.axes(projection=ccrs.PlateCarree())
#plt.title('IDL - Python Version Difference \n MCS Precipitation Fraction', fontsize=14, y=1.01)
#ax.add_feature(cfeature.COASTLINE)
#ax.add_feature(cfeature.BORDERS)
#ax.add_feature(cfeature.LAKES, alpha=0.5)
#ax.add_feature(cfeature.OCEAN, alpha=0.5)
#ax.add_feature(states_provinces, edgecolor='gray')
#ax.set_xmargin(0)
#ax.set_ymargin(0)
#im = plt.pcolormesh(Longitude[0, :, :], Latitude[0, :, :], np.ma.masked_invalid(np.atleast_2d(Difference_MCSPrecipFraction)), cmap='coolwarm', vmin=-1*ColorValues, vmax=ColorValues, transform=ccrs.PlateCarree())
#cbar = plt.colorbar(im)
#cbar.ax.set_ylabel('MCS Precipitation Fraction Difference (IDL - Python)', fontsize=12, labelpad=10)
#cbar.ax.tick_params(labelsize=10)
##plt.savefig(Figure_Location + 'IDLPython_MCSPrecipFractionDifference_' + str(int(AnalyzeYear)) + '0' + str(int(AnalyzeMonth)) + '.png')
#plt.savefig(Figure_Location + 'IDLPython_MCSPrecipFractionDifference.png')
#plt.close()

if np.absolute(np.nanmin(Difference_MCSFrequency)) > np.nanmax(Difference_MCSFrequency):
    ColorValues =  np.absolute(np.nanmin(Difference_MCSFrequency))/1.5
else:
    ColorValues =  np.nanmax(Difference_MCSFrequency)/1.5
plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
plt.title('IDL - Python Version Difference \n MCS Frequency', fontsize=14, y=1.01)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.OCEAN, alpha=0.5)
ax.add_feature(states_provinces, edgecolor='gray')
ax.set_xmargin(0)
ax.set_ymargin(0)
im = plt.pcolormesh(Longitude[0, :, :], Latitude[0, :, :], np.ma.masked_invalid(np.atleast_2d(Difference_MCSFrequency)), cmap='bwr', vmin=-1*ColorValues, vmax=ColorValues, transform=ccrs.PlateCarree())
cbar = plt.colorbar(im)
cbar.ax.set_ylabel('MCS Frequency Difference (IDL - Python)', fontsize=12, labelpad=10)
cbar.ax.tick_params(labelsize=10)
plt.savefig(Figure_Location + 'IDLPython_MCSFrequencyDifference_' + str(int(AnalyzeYear)) + '0' + str(int(AnalyzeMonth)) + '.png')
#plt.savefig(Figure_Location + 'IDLPython_MCSFrequencyDifference.png')
plt.close()

#############################################################3
## Plot individual cases
#print('Plotting individual times')

## Generate zipped list of inputs
#List_MinIDLTracks = [MinIDL]*nFiles
#List_MaxIDLTracks = [MaxIDL]*nFiles
#List_MinPythonTracks = [MinPython]*nFiles
#List_MaxPythonTracks = [MaxPython]*nFiles

#IDLCloudTrack[np.isnan(IDLCloudTrack)] = 0

#Inputs_CloudTrack = zip(np.arange(0, nFiles), IDLBrightnessTemperature, IDLCloudTrack, List_MinIDLTracks, List_MaxIDLTracks, PythonCloudTrack, List_MinPythonTracks, List_MaxPythonTracks, PythonBasetime, Latitude, Longitude)

##for iFile in range(0, nFiles):
##    Plot_CloudTracks(Inputs_CloudTrack[iFile])

#if __name__ == '__main__':
#    print('Plotting Individual Comparisions')
#    pool = Pool(12)
#    pool.map(Plot_CloudTracks, Inputs_CloudTrack)
#    pool.close()
#    pool.join()

