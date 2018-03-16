import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import scipy.io as sio
from math import pi
import pandas as pd
import fnmatch
import os
import sys
import datetime
import time
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.ndimage import label
import matplotlib.ticker as mticker
np.set_printoptions(threshold=np.inf)

####################################
# Set LES region (Domain: latitude = 36.05-37.15, longitude = -98.12--96.79)
# LES domain
LES_Latitude = [36.05, 37.15]
LES_Longitude = [-98.12, -96.79]

# TSI
TSI_Latitude = 36.6
TSI_Longitude = -97.48

# I9
I9_Latitude = 36.47
I9_Longitude = -97.42

# I10
I10_Latitude = 36.66
I10_Longitude = -97.62

# I8
I8_Latitude = 36.71
I8_Longitude = -97.38

# I4
I4_Latitude = 36.57
I4_Longitude = -97.36

# I6
I6_Latitude = 36.78
I6_Longitude = -97.54

# I5
I5_Latitude = 36.49
I5_Longitude = -97.59

# I7
I7_Latitude = 36.79
I7_Longitude = -97.44

##################################
# Set region options
RegionRadius = 0.04

LatitudeLocations = [TSI_Latitude, I9_Latitude, I10_Latitude, I7_Latitude]
LongitudeLocations = [TSI_Longitude, I9_Longitude, I10_Longitude, I7_Longitude]

Figure_Location = '/scratch2/scratchdirs/hcbarnes/LES/figures/'

#################################
# Set liquid water threshold
LiquidWaterThreshold = 0.05

##################################
# Manually generate LES data
print('Load LES pixel level data')
LES_Location1 = '/scratch2/scratchdirs/hcbarnes/LES/celltracking/20160830.1600_20160830.1800/'
Files1 = fnmatch.filter(os.listdir(LES_Location1), '*.nc')
LES_Location2 = '/scratch2/scratchdirs/hcbarnes/LES/celltracking/20160830.1800_20160830.2000/'
Files2 = fnmatch.filter(os.listdir(LES_Location2), '*.nc')
LES_Location3 = '/scratch2/scratchdirs/hcbarnes/LES/celltracking/20160830.2000_20160830.2300/'
Files3 = fnmatch.filter(os.listdir(LES_Location3), '*.nc')

AllFiles = [None]*(len(Files1) + len(Files2) + len(Files3))
AllDateTimeString = [None]*(len(Files1) + len(Files2) + len(Files3))
FileStep = 0
for iFile in range(0, len(Files1)):
    AllFiles[FileStep] = LES_Location1 + Files1[iFile]
    AllDateTimeString[FileStep] = str(Files1[iFile])[-16:-3]
    FileStep = FileStep + 1
for iFile in range(0, len(Files2)):
    AllFiles[FileStep] = LES_Location2 + Files2[iFile]
    AllDateTimeString[FileStep] = str(Files2[iFile])[-16:-3]
    FileStep = FileStep + 1
for iFile in range(0, len(Files1)):
    AllFiles[FileStep] = LES_Location3 + Files3[iFile]
    AllDateTimeString[FileStep] = str(Files3[iFile])[-16:-3]
    FileStep = FileStep + 1
AllFiles = np.sort(AllFiles[0:FileStep]).tolist()
#AllFiles = AllFiles[209:229]

DataHandle = xr.open_mfdataset(AllFiles, autoclose=True)
LES_BaseTime = np.array(DataHandle['basetime'].data)
#LES_TrackNumbers = np.array(DataHandle['celltracknumber'].data)
LES_lwp = np.array(DataHandle['lwp'].data)

########################################
# Restrict LES liquid water path data
LES_lwp[np.where(LES_lwp < LiquidWaterThreshold)] = np.nan

#######################################
# Load LES latitude and longitude grid
print('Loading LES latitude and longitude')
LatLonData = xr.open_dataset('/scratch2/scratchdirs/hcbarnes/LES/celltracking/20160830.1800_20160830.2000/celltracks_20160830_1838.nc', autoclose=True)
LES_LatitudeGrid = np.array(LatLonData['lat'].data)
LES_LongitudeGrid = np.array(LatLonData['lon'].data)

#########################################
# Load LASSO data
print('Loading LASSO data')
LASSOData = xr.open_dataset('/global/project/projectdirs/m1657/zfeng/hiscale/lasso/sim0017/wrfstat_d01_2016-08-30_12:00:00.nc', autoclose=True)
LASSO_BaseTime = np.array(LASSOData['XTIME'].data)
LASSO_lwp = np.array(LASSOData['CSS_LWP'].data)

########################################
# Restrict LES liquid water path data
LASSO_lwp[np.where(LASSO_lwp < LiquidWaterThreshold)] = np.nan

########################################
# Create LASSO lat/lon domain. Same resolution of LES and centered at same point, so take center of LES domain and extend 140 in each direction
LESy, LESx = np.shape(LES_LatitudeGrid)
LESy_Center = np.round(LESy/float(2)).astype(int)
LESx_Center = np.round(LESx/float(2)).astype(int)

LASSO_LatitudeGrid = np.copy(LES_LatitudeGrid[LESy_Center-70:LESy_Center+71, LESx_Center-70:LESx_Center+71])
LASSO_LongitudeGrid = np.copy(LES_LongitudeGrid[LESy_Center-70:LESy_Center+71, LESx_Center-70:LESx_Center+71])

#############################################
# Load my TSI data
print('Loading TSI data')
TSI_Data = xr.open_dataset('/global/homes/h/hcbarnes/Tracking/LES/data/TSI.nc', autoclose=True)
TSIraw_Times = TSI_Data['basetime'].data
TSIraw_EqDiameter = TSI_Data['equivalent_diameter'].data
TSIraw_CloudArea = TSI_Data['cloud_area'].data
TSIraw_DomainArea = TSI_Data['domain_area'].data

############################################
# Calculating cloud fraction and equivalent diameter time series

TSI_CloudFraction = np.ones(14, dtype=float)*np.nan
TSI_EqDiameter = np.ones((3, 14), dtype=float)*np.nan
LES_CloudFraction = np.ones((len(LatitudeLocations), 14), dtype=float)*np.nan
LES_EqDiameter = np.ones((3, len(LatitudeLocations), 14), dtype=float)*np.nan 
LASSO_CloudFraction = np.ones((len(LatitudeLocations), 14), dtype=float)*np.nan
LASSO_EqDiameter = np.ones((3, len(LatitudeLocations), 14), dtype=float)*np.nan

Labels = [None]*14

IntervalStart = (datetime.datetime(2016, 8, 30, 16, 0))
for iInterval in range(0, 14):
    TIntervalStart = np.array([pd.to_datetime(IntervalStart)], dtype='datetime64[ns]')
    IntervalEnd = IntervalStart + datetime.timedelta(minutes=30)
    TIntervalEnd = np.array([pd.to_datetime(IntervalEnd)], dtype='datetime64[ns]')

    Labels[iInterval] = str(IntervalStart + datetime.timedelta(minutes=15))[11:16]

    print(TIntervalStart, TIntervalEnd)

    # TSI
    TSIraw_Indices = np.array(np.where(((TSIraw_Times > TIntervalStart) & (TSIraw_Times <= TIntervalEnd))))[0, :] # For my data
    if len(TSIraw_Indices) > 10:
        print('TSI Present')
        # Select My Data
        tTSI_EqDiameter = np.copy(TSIraw_EqDiameter[TSIraw_Indices, :])
        tTSI_CloudArea = np.copy(TSIraw_CloudArea[TSIraw_Indices, :])
        tTSI_DomainArea = np.copy(TSIraw_DomainArea[TSIraw_Indices])

        # Filter data
        tTSI_CloudArea[np.where(tTSI_EqDiameter <= 100)] = np.nan
        tTSI_DomainArea[np.where(np.nansum(tTSI_EqDiameter, axis=1) == 0)] = np.nan
        tTSI_EqDiameter[np.where(tTSI_EqDiameter <= 100)] = np.nan

        tTSI_CloudArea = tTSI_CloudArea[np.where(np.isfinite(tTSI_CloudArea))]

        # Calculate fractions
        TSI_CloudFraction[iInterval] = np.divide(np.nansum(tTSI_CloudArea), np.nansum(tTSI_DomainArea))

        # Equivalent diameter percentiles
        TSI_EqDiameter[:, iInterval] = np.nanpercentile(tTSI_EqDiameter, [25, 50, 75])

    # LES and LASSO 
    for iLocation in range(0, len(LatitudeLocations)):
        LatitudeRegion = [LatitudeLocations[iLocation]-RegionRadius, LatitudeLocations[iLocation]+RegionRadius]
        LongitudeRegion = [LongitudeLocations[iLocation]-RegionRadius, LongitudeLocations[iLocation]+RegionRadius]

        ###############################################
        # LES

        # Get area of region
        LESLatLonIndices = np.array(np.where(((LES_LongitudeGrid >= LongitudeRegion[0]) & (LES_LongitudeGrid <= LongitudeRegion[1]) & (LES_LatitudeGrid >= LatitudeRegion[0]) & (LES_LatitudeGrid <= LatitudeRegion[1]))))
        LESLatLonIndicesy = [np.nanmin(np.unique(LESLatLonIndices[0, :])), np.nanmax(np.unique(LESLatLonIndices[0, :]))]
        LESLatLonIndicesx = [np.nanmin(np.unique(LESLatLonIndices[1, :])), np.nanmax(np.unique(LESLatLonIndices[1, :]))]

        # Get data in time interval 
        #LES_TrackNumbers_Subset = np.copy(LES_TrackNumbers[:, LESLatLonIndicesy[0]:LESLatLonIndicesy[1]+1, LESLatLonIndicesx[0]:LESLatLonIndicesx[1]+1])
        LES_LWP_Subset = np.copy(LES_lwp[:, LESLatLonIndicesy[0]:LESLatLonIndicesy[1]+1, LESLatLonIndicesx[0]:LESLatLonIndicesx[1]+1])

        # Get times of interest
        LES_Indices = np.array(np.where(((LES_BaseTime > TIntervalStart) & (LES_BaseTime <= TIntervalEnd))))[0, :]
        print('LES Present')
        # Get data in time interval
        tLES_lwp_Subset = np.copy(LES_LWP_Subset[LES_Indices, :, :])

        # Determine equivalent diameter
        nFiles = np.shape(tLES_lwp_Subset)[0]
        tLES_EqDiameter_Subset = []
        for iTime in range(0, nFiles):
            # Label clouds
            tLESmap = np.zeros((np.shape(tLES_lwp_Subset)[1], np.shape(tLES_lwp_Subset)[2]), dtype=int)
            tLESmap[np.where(~np.isnan(tLES_lwp_Subset[iTime, :, :]))] = 1
            LabeledClouds, nClouds = label(tLESmap)

            # Get equivalent diameters of clouds
            for iCloud in range(0, nClouds):
                iCloudPixels = len(np.array(np.where(LabeledClouds == iCloud+1))[0, :])
                if iCloudPixels >= 4:
                    tLES_EqDiameter_Subset = np.append(tLES_EqDiameter_Subset, np.multiply(2*np.sqrt(np.divide(iCloudPixels, pi)), 100))

        if len(tLES_EqDiameter_Subset) > 10:
            LES_EqDiameter[:, iLocation, iInterval] = np.nanpercentile(tLES_EqDiameter_Subset, [25, 50, 75])
            
            # Calculate cloud fraction
            tTotalPixels = np.shape(tLES_lwp_Subset)[0] * np.shape(tLES_lwp_Subset)[1] * np.shape(tLES_lwp_Subset)[2]
            tCloudyPixels = len(np.array(np.where(~np.isnan(tLES_lwp_Subset)))[0, :])
        
            LES_CloudFraction[iLocation, iInterval] = tCloudyPixels/float(tTotalPixels)

        ####################################
        # LASSO data

        # Get area of region
        LASSOLatLonIndices = np.array(np.where(((LASSO_LongitudeGrid >= LongitudeRegion[0]) & (LASSO_LongitudeGrid <= LongitudeRegion[1]) & (LASSO_LatitudeGrid >= LatitudeRegion[0]) & (LASSO_LatitudeGrid <= LatitudeRegion[1]))))
        if np.shape(LASSOLatLonIndices)[1] > 0:
            LASSOLatLonIndicesy = [np.nanmin(np.unique(LASSOLatLonIndices[0, :])), np.nanmax(np.unique(LASSOLatLonIndices[0, :]))]
            LASSOLatLonIndicesx = [np.nanmin(np.unique(LASSOLatLonIndices[1, :])), np.nanmax(np.unique(LASSOLatLonIndices[1, :]))]

            # Get data in time interval 
            #LES_TrackNumbers_Subset = np.copy(LES_TrackNumbers[:, LASSOLatLonIndicesy[0]:LASSOLatLonIndicesy[1]+1, LASSOLatLonIndicesx[0]:LASSOLatLonIndicesx[1]+1])
            LASSO_LWP_Subset = np.copy(LASSO_lwp[:, LASSOLatLonIndicesy[0]:LASSOLatLonIndicesy[1]+1, LASSOLatLonIndicesx[0]:LASSOLatLonIndicesx[1]+1])

            # Get times of interest
            LASSO_Indices = np.array(np.where(((LASSO_BaseTime > TIntervalStart) & (LASSO_BaseTime <= TIntervalEnd))))[0, :]
            print('LASSO present')
            # Get data in time interval
            tLASSO_lwp_Subset = np.copy(LASSO_LWP_Subset[LASSO_Indices, :, :])

            # Determine equivalent diameter
            nFiles = np.shape(tLASSO_lwp_Subset)[0]
            tLASSO_EqDiameter_Subset = []
            CloudStep = 0
            for iTime in range(0, nFiles):
                # Label clouds
                tLASSOmap = np.zeros((np.shape(tLASSO_lwp_Subset)[1], np.shape(tLASSO_lwp_Subset)[2]), dtype=int)
                tLASSOmap[np.where(tLASSO_lwp_Subset[iTime, :, :] > 0)] = 1
                LabeledClouds, nClouds = label(tLASSOmap)

                if nClouds > 1:
                    # Get equivalent diameters of clouds
                    for iCloud in range(0, nClouds):
                        iCloudPixels = len(np.array(np.where(LabeledClouds == iCloud+1))[0, :])
                        if iCloudPixels >= 4:
                            tLASSO_EqDiameter_Subset = np.append(tLASSO_EqDiameter_Subset, np.multiply(2*np.sqrt(np.divide(iCloudPixels, pi)), 100))

            if len(tLASSO_EqDiameter_Subset) > 10:
                LASSO_EqDiameter[:, iLocation, iInterval] = np.nanpercentile(tLASSO_EqDiameter_Subset, [25, 50, 75])

                # Calculate cloud fraction
                tTotalPixels = np.shape(tLASSO_lwp_Subset)[0] * np.shape(tLASSO_lwp_Subset)[1] * np.shape(tLASSO_lwp_Subset)[2]
                tCloudyPixels = len(np.array(np.where(~np.isnan(tLASSO_lwp_Subset)))[0, :])
            
                LASSO_CloudFraction[iLocation, iInterval] = tCloudyPixels/float(tTotalPixels)

    IntervalStart = IntervalEnd

#############################################
# Plot cloud fraction

plt.figure()
plt.title('Cloud Fraction \n  LES clouds LWP > ' + str(LiquidWaterThreshold), fontsize=14, y=1.01)
plt.plot(np.arange(0, 14), LES_CloudFraction[0, :], color='firebrick', linewidth=2)
plt.plot(np.arange(0, 14), LES_CloudFraction[1, :], color='dodgerblue', linewidth=2)
plt.plot(np.arange(0, 14), LES_CloudFraction[2, :], color='forestgreen', linewidth=2)
plt.plot(np.arange(0, 14), LES_CloudFraction[3, :], color='chocolate', linewidth=2)
plt.plot(np.arange(0, 14), TSI_CloudFraction, color='black', linewidth=3)
plt.legend(['LES-TSI', 'LES-I9', 'LES-I10', 'LES_I6', 'OBS-TSI'], fontsize=10)
plt.xticks(np.arange(0, 14), Labels, rotation=25)
plt.xlim(1, 13)
plt.xlabel('Time [UTC]', fontsize=8)
plt.ylabel('Cloud Fraction (Cloudy Area / Total Area)', fontsize=8)
#plt.ylim(100, 1400)
plt.grid(True, linestyle=':', color='gray')
plt.tick_params(labelsize=8)
plt.savefig(Figure_Location + 'ComparisonCloudFraction_LES-TSI.png')
plt.close()

plt.figure()
plt.title('Cloud Fraction \n  LES and LASSO clouds LWP > ' + str(LiquidWaterThreshold), fontsize=14, y=1.01)
plt.plot(np.arange(0, 14), LES_CloudFraction[0, :], color='firebrick', linewidth=3)
plt.plot(np.arange(0, 14), LASSO_CloudFraction[0, :], color='blue', linewidth=3)
plt.plot(np.arange(0, 14), TSI_CloudFraction[:], color='black', linewidth=5)
plt.legend(['LES-TSI', 'LASSO-TSI', 'OBS-TSI'], fontsize=10)
plt.xticks(np.arange(0, 14), Labels, rotation=25)
plt.xlim(1, 13)
plt.xlabel('Time [UTC]', fontsize=8)
plt.ylabel('Cloud Fraction (Cloudy Area / Total Area)', fontsize=8)
#plt.ylim(100, 1400)
plt.grid(True, linestyle=':', color='gray')
plt.tick_params(labelsize=8)
plt.savefig(Figure_Location + 'ComparisonCloudFraction_LES-LASSO-TSI.png')
plt.close()

#################################################
# Plot equivalent diameter
plt.figure()
plt.title('Cloud Equivalent Diameter \n  LES and LASSO clouds LWP > ' + str(LiquidWaterThreshold), fontsize=14, y=1.01)
plt.fill_between(np.arange(0, 14), LES_EqDiameter[0, 0, :], LES_EqDiameter[2, 0, :], color='firebrick', alpha=0.2, linewidth=0)
plt.fill_between(np.arange(0, 14), LASSO_EqDiameter[0, 0, :], LASSO_EqDiameter[2, 0, :], color='blue', alpha=0.2, linewidth=0)
plt.fill_between(np.arange(0, 14), TSI_EqDiameter[0, :], TSI_EqDiameter[2, :], color='black', alpha=0.2, linewidth=0)
plt.legend(['LES-TSI', 'LASSO-TSI', 'OBS-TSI'], fontsize=10, loc='upper left')
plt.plot(np.arange(0, 14), LES_EqDiameter[1, 0, :], color='firebrick', linewidth=3)
plt.plot(np.arange(0, 14), LASSO_EqDiameter[1, 0, :], color='blue', linewidth=3)
plt.plot(np.arange(0, 14), TSI_EqDiameter[1, :], color='black', linewidth=3)
plt.xticks(np.arange(0, 14), Labels, rotation=25)
plt.xlim(1, 13)
plt.xlabel('Time [UTC]', fontsize=8)
plt.ylabel('Equivalent Diameter [m]', fontsize=8)
#plt.ylim(100, 1400)
plt.grid(True, linestyle=':', color='gray')
plt.tick_params(labelsize=8)
plt.savefig(Figure_Location + 'ComparisonCloudEqDiameter_LES-LASSO-TSI.png')
plt.close()
#plt.show()

raw_input('wait')

########################################
# Plot examples
ExpansionRegion = 200

ExpandedLongitude = np.copy(LES_LongitudeGrid[LatLonIndicesy[0]-ExpansionRegion:LatLonIndicesy[1]+ExpansionRegion+1, LatLonIndicesx[0]-ExpansionRegion:LatLonIndicesx[1]+ExpansionRegion+1])
SmallLongitude = np.copy(LES_LongitudeGrid[LatLonIndicesy[0]:LatLonIndicesy[1], LatLonIndicesx[0]:LatLonIndicesx[1]])
LongitudeRange = [np.nanmin(SmallLongitude), np.nanmax(SmallLongitude)]

ExpandedLatitude = np.copy(LES_LatitudeGrid[LatLonIndicesy[0]-ExpansionRegion:LatLonIndicesy[1]+ExpansionRegion+1, LatLonIndicesx[0]-ExpansionRegion:LatLonIndicesx[1]+ExpansionRegion+1])
SmallLatitude = np.copy(LES_LatitudeGrid[LatLonIndicesy[0]:LatLonIndicesy[1], LatLonIndicesx[0]:LatLonIndicesx[1]])
LatitudeRange = [np.nanmin(SmallLatitude), np.nanmax(SmallLatitude)]

for iMap in range(0, len(AllFiles), 30):
    # liquid water path
    iPixels = np.shape(LWP[iMap, :, :])[0] * np.shape(LWP[iMap, :, :])[1]
    iTrackCounts = len(np.array(np.where(~np.isnan(LWP[iMap, :, :])))[0, :])

    plt.figure()
    plt.title(str(LES_BaseTime[iMap]))
    im = plt.pcolormesh(LES_LongitudeGrid[LatLonIndicesy[0]-ExpansionRegion:LatLonIndicesy[1]+ExpansionRegion+1, LatLonIndicesx[0]-ExpansionRegion:LatLonIndicesx[1]+ExpansionRegion+1], LES_LatitudeGrid[LatLonIndicesy[0]-ExpansionRegion:LatLonIndicesy[1]+ExpansionRegion+1, LatLonIndicesx[0]-ExpansionRegion:LatLonIndicesx[1]+ExpansionRegion+1], np.ma.masked_invalid(np.atleast_2d(LES_lwp[iMap, LatLonIndicesy[0]-ExpansionRegion:LatLonIndicesy[1]+ExpansionRegion+1, LatLonIndicesx[0]-ExpansionRegion:LatLonIndicesx[1]+ExpansionRegion+1])), vmin=0, vmax=1.5)
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('Liquid Water Path', labelpad=10)
    plt.plot([LongitudeRange[0], LongitudeRange[1], LongitudeRange[1], LongitudeRange[1], LongitudeRange[0], LongitudeRange[0]], [LatitudeRange[0], LatitudeRange[0], LatitudeRange[1], LatitudeRange[1], LatitudeRange[1], LatitudeRange[0]], linewidth=2, color='black')
    plt.grid(True)
    plt.xlabel('Cloud Fraction: ' + str(iTrackCounts/float(iPixels)))
    plt.show()

    ## Track numbers
    #iTracksPresent, iTrackCounts = np.unique(LES_TrackNumbers[iMap, LatLonIndicesy[0]:LatLonIndicesx[1]+1, LatLonIndicesx[0]:LatLonIndicesx[1]+1], return_counts=True)
    #iPixels = np.nansum(iTrackCounts)
    #iTrackCounts = iTrackCounts[np.where(iTracksPresent > 0)]

    #plt.figure()
    #plt.title(str(LES_BaseTime[iMap]))
    #im = plt.pcolormesh(LES_LongitudeGrid[LatLonIndicesy[0]-ExpansionRegion:LatLonIndicesy[1]+ExpansionRegion+1, LatLonIndicesx[0]-ExpansionRegion:LatLonIndicesx[1]+ExpansionRegion+1], LES_LatitudeGrid[LatLonIndicesy[0]-ExpansionRegion:LatLonIndicesy[1]+ExpansionRegion+1, LatLonIndicesx[0]-ExpansionRegion:LatLonIndicesx[1]+ExpansionRegion+1], LES_TrackNumbers[iMap, LatLonIndicesy[0]-ExpansionRegion:LatLonIndicesy[1]+ExpansionRegion+1, LatLonIndicesx[0]-ExpansionRegion:LatLonIndicesx[1]+ExpansionRegion+1], cmap='nipy_spectral_r')
    #cbar = plt.colorbar(im)
    #cbar.ax.set_ylabel('Track Number', labelpad=10)
    #plt.plot([LongitudeRange[0], LongitudeRange[1], LongitudeRange[1], LongitudeRange[1], LongitudeRange[0], LongitudeRange[0]], [LatitudeRange[0], LatitudeRange[0], LatitudeRange[1], LatitudeRange[1], LatitudeRange[1], LatitudeRange[0]], linewidth=2, color='black')
    #plt.grid(True)
    #plt.xlabel('Cloud Fraction: ' + str(np.nansum(iTrackCounts)/float(iPixels)))
    #plt.show()

