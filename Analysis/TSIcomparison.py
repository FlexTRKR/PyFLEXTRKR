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

######################################
# Data locations
TSI_Location = '/global/homes/h/hcbarnes/Tracking/LES/data/'
LES_Location = '/scratch2/scratchdirs/hcbarnes/LES/stats/'

TSI_BaseName = 'tsiproj_'

Figure_Location = '/scratch2/scratchdirs/hcbarnes/LES/figures/'

###################################
# Plot map

# Create state map
states_provinces = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='50m', facecolor='none')

ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(states_provinces, edgecolor='gray')
plt.xlim(-98.2, -96.7)
plt.ylim(36, 37.2)
grid = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, color='gray', linestyle=':')
grid.xlabels_top = False
grid.ylabels_right = False
grid.xlocator = mticker.FixedLocator(np.arange(-98.6, -96.2, 0.4))
grid.ylocator = mticker.FixedLocator(np.arange(35.4, 37.6, 0.2))
grid.xformatter = LONGITUDE_FORMATTER
grid.yformatter = LATITUDE_FORMATTER
grid.xlabel_style = {'size': 10}
grid.xlabel_style = {'size': 10}
ax.set_title('LES Domain, Facility Locations, and Analysis Domains', fontsize=12)
plt.scatter(TSI_Longitude, TSI_Latitude, marker='*', s=30, color='tomato')
plt.annotate('TSI', (TSI_Longitude-0.03, TSI_Latitude-0.08))
plt.scatter(I9_Longitude, I9_Latitude, marker='o', s=10, color='dodgerblue')
plt.annotate('I9', (I9_Longitude-0.02, I9_Latitude-0.08))
plt.scatter(I9_Longitude, I9_Latitude, marker='o', s=10, color='dodgerblue')
plt.annotate('I10', (I10_Longitude-0.11, I10_Latitude-0.02))
plt.scatter(I10_Longitude, I10_Latitude, marker='o', s=10, color='dodgerblue')
plt.annotate('I8', (I8_Longitude+0.02, I8_Latitude-0.02))
plt.scatter(I8_Longitude, I8_Latitude, marker='o', s=10, color='dodgerblue')
plt.annotate('I4', (I4_Longitude+0.02, I4_Latitude-0.02))
plt.scatter(I4_Longitude, I4_Latitude, marker='o', s=10, color='dodgerblue')
plt.annotate('I6', (I6_Longitude-0.06, I6_Latitude-0.02))
plt.scatter(I6_Longitude, I6_Latitude, marker='o', s=10, color='dodgerblue')
plt.annotate('I5', (I5_Longitude-0.06, I5_Latitude-0.02))
plt.scatter(I5_Longitude, I5_Latitude, marker='o', s=10, color='dodgerblue')
plt.annotate('I7', (I7_Longitude+0.04, I7_Latitude-0.01))
plt.scatter(I7_Longitude, I7_Latitude, marker='o', s=10, color='dodgerblue')
plt.plot([LES_Longitude[0], LES_Longitude[0], LES_Longitude[1], LES_Longitude[1], LES_Longitude[0]], [LES_Latitude[0], LES_Latitude[1], LES_Latitude[1], LES_Latitude[0], LES_Latitude[0]], linewidth=2, color='forestgreen')
for iLocation in range(0, len(LatitudeLocations)):
    plt.plot([LongitudeLocations[iLocation]+RegionRadius, LongitudeLocations[iLocation]+RegionRadius, LongitudeLocations[iLocation]-RegionRadius, LongitudeLocations[iLocation]-RegionRadius, LongitudeLocations[iLocation]+RegionRadius], [LatitudeLocations[iLocation]+RegionRadius, LatitudeLocations[iLocation]-RegionRadius, LatitudeLocations[iLocation]-RegionRadius, LatitudeLocations[iLocation]+RegionRadius, LatitudeLocations[iLocation]+RegionRadius], color='black', linewidth=1)
plt.savefig(Figure_Location + 'SGPmap.png')
plt.close()

####################################
## Generate TSI data
#TSI_Location = '/global/project/projectdirs/m1657/zfeng/hiscale/tsiproj/'

## Isolate TSI data to process
#TSI_AllFiles = fnmatch.filter(os.listdir(TSI_Location), TSI_BaseName+'*.mat')
#TSI_AllFiles = np.sort(TSI_AllFiles)
#nTSIFiles = len(TSI_AllFiles)

## Initialize TSI matrices
#MaxTSI = 100

#TSI_Times = np.empty(nTSIFiles, dtype='datetime64[s]')
#TSI_DomainArea = np.empty(nTSIFiles, dtype=float)
#TSI_GridArea = np.empty(nTSIFiles, dtype=float)
#TSI_EqDiameter = np.empty((nTSIFiles, MaxTSI), dtype='float')*np.nan
#TSI_PixelCounts = np.empty((nTSIFiles, MaxTSI), dtype='float')*np.nan
#TSI_CloudArea = np.empty((nTSIFiles, MaxTSI), dtype='float')*np.nan

## Set data edge
#Domain = np.zeros((960, 960), dtype=int)
#r = 479
#d = 2*r
#rx, ry = d/2, d/2
#x, y = np.indices((d, d))
#Domain[1:-1, 1:-1] = (np.abs(np.hypot(rx - x, ry - y)-r) < 0.5).astype(int)

## Conduct TSI analysis
#print('Analyzing TSI data')
#MaxObs = 0
#for iTSI in range(0, nTSIFiles):
#    # Load data
#    TSI_FileName = str(np.copy(TSI_AllFiles[iTSI]))
#    TSI_Data = np.array(sio.loadmat(TSI_Location + TSI_FileName, variable_names='tsi_lab')['tsi_lab'])
#    TSI_GridVector = np.array(sio.loadmat(TSI_Location + TSI_FileName, variable_names='xg')['xg'])
#    XGrid, YGrid = np.meshgrid(TSI_GridVector[0, :], TSI_GridVector[0, :])
#    TSI_GridSpacing = np.unique(np.subtract(TSI_GridVector[0, 1::], TSI_GridVector[0, 0:-1]))[0]
#    TSI_GridArea[iTSI] = np.square(TSI_GridSpacing) # m

#    print(TSI_FileName)

#    # Calculate domain area
#    DomainRadius = int(np.divide(np.shape(TSI_GridVector)[1], 2))
#    TSI_DomainArea[iTSI] = pi * ((DomainRadius*TSI_GridSpacing)**2)

#    # Get times
#    TSI_Times[iTSI] = np.array([pd.to_datetime(pd.DataFrame({'year': np.array([TSI_FileName[8:12]]).astype(int), \
#                                                            'month': np.array([TSI_FileName[12:14]]).astype(int), \
#                                                            'day': np.array([TSI_FileName[14:16]]).astype(int), \
#                                                            'hour': np.array([TSI_FileName[17:19]]).astype(int), \
#                                                            'minute': np.array([TSI_FileName[19:21]]).astype(int), \
#                                                            'second': np.array([TSI_FileName[21:23]]).astype(int)}), unit='ns')[0]], dtype='datetime64[ns]')[0]

##    # Remove clouds that touch the edge of the TSI domain
##    EdgeClouds = np.array(np.where((TSI_Data > 0) & (Domain == 1)))
##    EdgeClouds = TSI_Data[EdgeClouds[0, :], EdgeClouds[1, :]]
##    for iEdge in EdgeClouds:
##        TSI_Data[np.where(TSI_Data == iEdge)] = 0

##    #plt.figure()
##    #im = plt.pcolormesh(TSI_Data, cmap='nipy_spectral_r')
##    #plt.contour(Domain, color='w', linewidth=2)
##    #plt.colorbar(im)

#    # Calculate number of pixels
#    UniqueClouds = np.unique(TSI_Data)[1::]
#    nTSIClouds = len(UniqueClouds)

#    CloudStep = 0
#    for iCloud in UniqueClouds:
#        tPixels = len(np.array(np.where(TSI_Data == iCloud))[0, :])
#        if tPixels > 3:
#            TSI_PixelCounts[iTSI, CloudStep] = tPixels
#            CloudStep = CloudStep + 1

#    # Calculate area and equivalent diameter
#    if nTSIClouds <= MaxTSI:
#        TSI_CloudArea[iTSI, 0:nTSIClouds] = np.multiply(TSI_PixelCounts[iTSI, 0:nTSIClouds], TSI_GridArea[iTSI])

#        TSI_EqDiameter[iTSI, 0:nTSIClouds] = np.multiply(2, np.sqrt(np.divide(TSI_CloudArea[iTSI, 0:nTSIClouds], float(pi))))
#    else:
#        sys.exit('Too Many Clouds: ' + str(int(nClouds)))

#    if nTSIClouds > MaxObs:
#        MaxObs = nTSIClouds

#    #plt.show()

#TSI_PixelCounts = TSI_PixelCounts[:, 0:MaxObs+1]
#TSI_CloudArea = TSI_CloudArea[:, 0:MaxObs+1]
#TSI_EqDiameter = TSI_EqDiameter[:, 0:MaxObs+1]

## Create TSI netcdf filename
#TSI_OutFile = '/global/homes/h/hcbarnes/Tracking/LES/data/TSI.nc'
#print(TSI_OutFile)

## Check if file already exists. If exists, delete
#if os.path.isfile(TSI_OutFile):
#    os.remove(TSI_OutFile)

### Define xarray dataset
#output_data = xr.Dataset({'basetime': (['time'], TSI_Times), \
#                          'grid_size': (['time'], TSI_GridArea), \
#                          'domain_area': (['time'], TSI_DomainArea), \
#                          'npixels': (['time', 'clouds'], TSI_PixelCounts), \
#                          'cloud_area': (['time', 'clouds'], TSI_CloudArea), \
#                          'equivalent_diameter': (['time', 'clouds'], TSI_EqDiameter)}, \
#                         coords={'time':(['time'], np.arange(0, nTSIFiles)), \
#                                'clouds':(['clouds'], np.arange(0, MaxObs+1))}, \
#                         attrs={'title':'Cloud Statistics from TSI', \
#                                'source':'Jessica Kleiss, jkleiss@lclark.edu', \
#                                'grid_resolution': str(TSI_GridSpacing), \
#                                'created_by':'Hannah C Barnes: hannah.barnes@pnnl.gov', \
#                                'created_on':time.ctime(time.time())})

## Set variable attributes
#output_data.basetime.attrs['long_name'] = 'Epoch time of these clouds'

#output_data.grid_size.attrs['long_name'] = 'Area of one grid space'
#output_data.grid_size.attrs['units'] = 'm'

#output_data.domain_area.attrs['long_name'] = 'Entire area measured by the TSI'
#output_data.domain_area.attrs['units'] = 'm'

#output_data.npixels.attrs['long_name'] = 'Number of Pixels of each cloud at this time'
#output_data.npixels.attrs['units'] = 'unitless'

#output_data.cloud_area.attrs['long_name'] = 'Area of each cloud at this time'
#output_data.cloud_area.attrs['units'] = 'm^2'

#output_data.equivalent_diameter.attrs['long_name'] = 'Equivalent diameter of each cloud at this time'
#output_data.equivalent_diameter.attrs['units'] = 'm'

## Write data
#output_data.to_netcdf(path=TSI_OutFile, mode='w', format='NETCDF4_CLASSIC', unlimited_dims='time', \
#                      encoding={'basetime': {'zlib':True, 'units': 'seconds since 1970-01-01'}, \
#                                'grid_size': {'zlib':True}, \
#                                'domain_area': {'zlib':True}, \
#                                'npixels': {'zlib':True}, \
#                                'cloud_area': {'zlib':True}, \
#                                'equivalent_diameter': {'zlib':True}})

#raw_input('Restart')

#############################################
# Load my TSI data
print('Loading TSI data')
TSI_FileName = TSI_Location + 'TSI.nc'
print(TSI_FileName)

TSI_Data = xr.open_dataset(TSI_FileName, autoclose=True)
TSIraw_Times = TSI_Data['basetime'].data
TSIraw_CloudArea = TSI_Data['cloud_area'].data
TSIraw_EqDiameter = TSI_Data['equivalent_diameter'].data
TSIraw_DomainArea = TSI_Data['domain_area'].data

############################################
## Load Jessica's TSI analysis

## Load data
#TSIanalysis_Data = np.array(sio.loadmat(TSI_Location + 'TSIanalysis_Jessica.mat', variable_names=['cloud_statistics'])['cloud_statistics'])
#TSIanalysis_MatlabTimes = TSIanalysis_Data[:, 0]
#TSIanalysis_MaxZenith = TSIanalysis_Data[:, 7]
#TSIanalysis_Area = TSIanalysis_Data[:, 10]
#TSIanalysis_Data = np.array(sio.loadmat(TSI_Location + 'TSIanalysis_Jessica.mat', variable_names=['day_statistics'])['day_statistics'])
#TSIanalysis_TotalArea = TSIanalysis_Data[:, 2]
#TSIanalysis_MatlabDays = TSIanalysis_Data[:, 0]

## Convert matlab time to python time
#TSIanalysis_PythonTimes = np.empty(len(TSIanalysis_MatlabTimes), dtype='datetime64[s]')
#for iTime in range(0, len(TSIanalysis_MatlabTimes)):
#    TSIanalysis_PythonTimes[iTime] = datetime.datetime.fromordinal(int(TSIanalysis_MatlabTimes[iTime])) + datetime.timedelta(days=TSIanalysis_MatlabTimes[iTime]%1) - datetime.timedelta(days = 366)

#TSIanalysis_PythonDays = np.empty(len(TSIanalysis_MatlabDays), dtype='datetime64[s]')
#for iDay in range(0, len(TSIanalysis_MatlabDays)):
#    TSIanalysis_PythonDays[iDay] = datetime.datetime.fromordinal(int(TSIanalysis_MatlabDays[iDay])) + datetime.timedelta(days=TSIanalysis_MatlabDays[iDay]%1) - datetime.timedelta(days = 366)
#    if str(TSIanalysis_PythonDays[iDay])[-1] == '9':
#         TSIanalysis_PythonDays[iDay] = np.datetime64(TSIanalysis_PythonDays[iDay]) + np.timedelta64(1, 's')

## Calculate equivalent diameter
#TSIanalysis_EqDiameter = np.multiply(2, np.sqrt(np.divide(TSIanalysis_Area, pi)))

###############################################
## Compare TSI data versions
#TSIanalysis_EqDiameterGroups = np.empty((len(TSIanalysis_TotalArea), 100), dtype=float)*np.nan
#TSIanalysis_TimesGroups = np.empty(len(TSIanalysis_TotalArea), dtype='datetime64[s]')
#for iTime in range(0, len(TSIraw_Times)):
#    # Select data
#    AnalysisIndices = np.array(np.where(((TSIanalysis_PythonTimes == TSIraw_Times[iTime]) | (TSIanalysis_PythonTimes == np.datetime64(TSIraw_Times[iTime]) - np.timedelta64(1, 's')))))
#    tTSI_EqDiameter = np.copy(TSIanalysis_EqDiameter[AnalysisIndices])
#    tTSI_MaxZenith = np.copy(TSIanalysis_MaxZenith[AnalysisIndices])

#    # Filter data
#    tTSI_EqDiameter[np.where(tTSI_EqDiameter < 100)] = np.nan
#    #tTSI_EqDiameter[np.where(tTSI_MaxZenith >= 65)] = np.nan

#    # Assign data
#    nTSI = len(tTSI_EqDiameter[0, :])
#    TSIanalysis_EqDiameterGroups[iTime, 0:nTSI] = tTSI_EqDiameter[0, :]
#    TSIanalysis_TimesGroups[iTime] = np.copy(TSIraw_Times[iTime])

#    tAnalysis = np.copy(TSIanalysis_EqDiameterGroups[iTime, 0:nTSI])
#    tAnalysis = tAnalysis[np.where(~np.isnan(tAnalysis))]
#    tRaw = np.copy(TSIraw_EqDiameter[iTime, :])
#    tRaw[tRaw < 100] = np.nan
#    tRaw = tRaw[np.where(~np.isnan(tRaw))]
#    print(tAnalysis)
#    print(tRaw)
#    print('')
#    raw_input('check')

#############################################3
# Load LES data
print('Loading LES data')

# Set file names
Early_FileName = 'cell_tracks_20160830.1600_20160830.1800.nc'
Middle_FileName = 'cell_tracks_20160830.1800_20160830.2000.nc'
Late_FileName = 'cell_tracks_20160830.2000_20160830.2300.nc'

# Load data of interest
Early_Data = xr.open_dataset(LES_Location + Early_FileName, autoclose=True)
Early_basetime = np.array(Early_Data['cell_basetime'].data)
Early_meanlon = np.array(Early_Data['cell_meanlon'].data)
Early_meanlat = np.array(Early_Data['cell_meanlat'].data)
Early_cellarea = np.array(Early_Data['cell_ccsarea'].data)*(1000**2) # tracks, time
Early_eqdiameter = np.multiply(2, np.sqrt(np.divide(Early_cellarea, float(pi))))

Middle_Data = xr.open_dataset(LES_Location + Middle_FileName, autoclose=True)
Middle_basetime = np.array(Middle_Data['cell_basetime'].data)
Middle_meanlon = np.array(Middle_Data['cell_meanlon'].data)
Middle_meanlat = np.array(Middle_Data['cell_meanlat'].data)
Middle_cellarea = np.array(Middle_Data['cell_ccsarea'].data)*(1000**2) # tracks, time
Middle_eqdiameter = np.multiply(2, np.sqrt(np.divide(Middle_cellarea, float(pi))))

Late_Data = xr.open_dataset(LES_Location + Late_FileName, autoclose=True)
Late_basetime = np.array(Late_Data['cell_basetime'].data)
Late_meanlon = np.array(Late_Data['cell_meanlon'].data)
Late_meanlat = np.array(Late_Data['cell_meanlat'].data)
Late_cellarea = np.array(Late_Data['cell_ccsarea'].data)*(1000**2) # tracks, time
Late_eqdiameter = np.multiply(2, np.sqrt(np.divide(Late_cellarea, float(pi))))

# Reshape data
yEarly, xEarly = np.shape(Early_eqdiameter)
Early_basetime = np.reshape(Early_basetime, yEarly*xEarly)
Early_meanlon = np.reshape(Early_meanlon, yEarly*xEarly)
Early_meanlat = np.reshape(Early_meanlat, yEarly*xEarly)
Early_cellarea = np.reshape(Early_cellarea, yEarly*xEarly)
Early_eqdiameter = np.reshape(Early_eqdiameter, yEarly*xEarly)

yMiddle, xMiddle = np.shape(Middle_eqdiameter)
Middle_basetime = np.reshape(Middle_basetime, yMiddle*xMiddle)
Middle_meanlon = np.reshape(Middle_meanlon, yMiddle*xMiddle)
Middle_meanlat = np.reshape(Middle_meanlat, yMiddle*xMiddle)
Middle_cellarea = np.reshape(Middle_cellarea, yMiddle*xMiddle)
Middle_eqdiameter = np.reshape(Middle_eqdiameter, yMiddle*xMiddle)

yLate, xLate = np.shape(Late_eqdiameter)
Late_basetime = np.reshape(Late_basetime, yLate*xLate)
Late_meanlon = np.reshape(Late_meanlon, yLate*xLate)
Late_meanlat = np.reshape(Late_meanlat, yLate*xLate)
Late_cellarea = np.reshape(Late_cellarea, yLate*xLate)
Late_eqdiameter = np.reshape(Late_eqdiameter, yLate*xLate)

# Join and vectorize data
LES_Area = np.concatenate((Early_cellarea, Middle_cellarea, Late_cellarea))
LES_EqDiameter = np.concatenate((Early_eqdiameter, Middle_eqdiameter, Late_eqdiameter))
LES_Times = np.concatenate((Early_basetime, Middle_basetime, Late_basetime))
LES_MeanLon = np.concatenate((Early_meanlon, Middle_meanlon, Late_meanlon))
LES_MeanLat = np.concatenate((Early_meanlat, Middle_meanlat, Late_meanlat))

# Load latitude and longitude grid
LatLonData = xr.open_dataset('/scratch2/scratchdirs/hcbarnes/LES/celltracking/20160830.1800_20160830.2000/celltracks_20160830_1838.nc', autoclose=True)
latitude = np.array(LatLonData['lat'].data)
longitude = np.array(LatLonData['lon'].data)

##################################################
# Generate data for box plots
print('Creating box data')

TSI_EqdiaFractionBoxPlot = np.ones((5, 14), dtype=float)*np.nan
TSI_EqdiameterBoxPlot = np.ones((5, 14), dtype=float)*np.nan
TSI_EqdiaFractionMean = np.ones(14, dtype=float)*np.nan
TSI_EqdiameterMean = np.ones(14, dtype=float)*np.nan
TSI_CloudFraction = np.ones(14, dtype=float)*np.nan

LES_EqdiaFractionBoxPlot = np.ones((len(LatitudeLocations), 5, 14), dtype=float)*np.nan
LES_EqdiameterBoxPlot = np.ones((len(LatitudeLocations), 5, 14), dtype=float)*np.nan
LES_EqdiaFractionMean = np.ones((len(LatitudeLocations), 14), dtype=float)*np.nan
LES_EqdiameterMean = np.ones((len(LatitudeLocations), 14), dtype=float)*np.nan
LES_CloudFraction = np.ones((len(LatitudeLocations), 14), dtype=float)*np.nan

Labels = [None]*14

IntervalStart = (datetime.datetime(2016, 8, 30, 16, 0))
for iInterval in range(0, 14):
    TIntervalStart = np.array([pd.to_datetime(IntervalStart)], dtype='datetime64[ns]')
    IntervalEnd = IntervalStart + datetime.timedelta(minutes=30)
    TIntervalEnd = np.array([pd.to_datetime(IntervalEnd)], dtype='datetime64[ns]')

    Labels[iInterval] = str(IntervalStart + datetime.timedelta(minutes=15))[11:16]

    # TSI
    #TSIanalysis_Indices1 = np.array(np.where(((TSIanalysis_PythonTimes > TIntervalStart) & (TSIanalysis_PythonTimes <= TIntervalEnd))))[0, :] # For Jessica's data
    #TSIanalysis_Indices2 = np.array(np.where(((TSIanalysis_PythonDays > TIntervalStart) & (TSIanalysis_PythonDays <= TIntervalEnd))))[0, :] # For Jessica's data
    #if len(TSIanalysis_Indices1) > 0 and len(TSIanalysis_Indices2) > 0:
    TSIraw_Indices = np.array(np.where(((TSIraw_Times > TIntervalStart) & (TSIraw_Times <= TIntervalEnd))))[0, :] # For my data
    if len(TSIraw_Indices) > 0:

        ## Select Jessica's data
        #tTSI_EqDiameter = np.copy(TSIanalysis_EqDiameter[TSIanalysis_Indices1])
        #tTSI_CloudArea = np.copy(TSIanalysis_Area[TSIanalysis_Indices1])
        #tTSI_TotalArea = np.copy(TSIanalysis_TotalArea[TSIanalysis_Indices2])
        #tTSI_Time1 = np.copy(TSIanalysis_PythonTimes[TSIanalysis_Indices1])
        #tTS_Time2 = np.copy(TSIanalysis_PythonDays[TSIanalysis_Indices2])
        ##tTSI_MaxZenith = np.copy(TSIanalysis_MaxZenith[TSIanalysis_Indices])
        ##tTSI_EqDiameter[np.where(tTSI_MaxZenith >= 65)] = np.nan

        # Select My Data
        tTSI_EqDiameter = np.copy(TSIraw_EqDiameter[TSIraw_Indices, :])
        tTSI_CloudArea = np.copy(TSIraw_CloudArea[TSIraw_Indices, :])
        tTSI_DomainArea = np.copy(TSIraw_DomainArea[TSIraw_Indices])

        # Filter data
        tTSI_CloudArea[np.where(tTSI_EqDiameter <= 100)] = np.nan
        tTSI_EqDiameter[np.where(tTSI_EqDiameter <= 100)] = np.nan
        tTSI_DomainArea[np.where(np.nansum(tTSI_EqDiameter, axis=1) == 0)] = np.nan

        tTSI_CloudArea = tTSI_CloudArea[np.where(np.isfinite(tTSI_CloudArea))]
        tTSI_EqDiameter = tTSI_EqDiameter[np.where(np.isfinite(tTSI_EqDiameter))]

        # Calculate fractions
        TSI_CloudFraction[iInterval] = np.divide(np.nansum(tTSI_CloudArea), np.nansum(tTSI_DomainArea))

        tTSI_EqDiameter = np.sort(tTSI_EqDiameter)
        tTSI_Sum = np.nansum(tTSI_EqDiameter)

        tTSI_Fraction = np.empty(len(tTSI_EqDiameter), dtype=float)*np.nan
        for iTSI in range(0, len(tTSI_EqDiameter)):
            tTSI_Fraction[iTSI] = np.divide(np.nansum(tTSI_EqDiameter[0:iTSI+1]), tTSI_Sum)

        # Calculate box plots
        TSI_EqdiameterBoxPlot[:, iInterval] = np.nanpercentile(tTSI_EqDiameter, [5, 25, 50, 75, 95])

        for PercentileStep, iPercentile in enumerate([0.05, 0.25, 0.50, 0.75, 0.95]):
            tIndice = np.array(np.where(tTSI_Fraction >= iPercentile))[0]
            if len(tIndice) > 0:
                TSI_EqdiaFractionBoxPlot[PercentileStep, iInterval] = np.copy(tTSI_EqDiameter[tIndice[0]])
            else:
                TSI_EqdiaFractionBoxPlot[PercentileStep, iInterval] = np.nan

        # Calculate means
        TSI_EqdiameterMean[iInterval] = np.nanmean(tTSI_EqDiameter)

        tMean = np.nanmean(tTSI_Fraction)
        MeanIndice = np.array(np.where(tTSI_Fraction >= tMean))[0]
        if len(MeanIndice) > 0:
            TSI_EqdiaFractionMean[iInterval] = np.copy(tTSI_EqDiameter[MeanIndice[0]])
        else:
            TSI_EqdiaFractionMean[iInterval] = np.nan

    # LES
    for iLocation in range(0, len(LatitudeLocations)):
        LatitudeRegion = [LatitudeLocations[iLocation]-RegionRadius, LatitudeLocations[iLocation]+RegionRadius]
        LongitudeRegion = [LongitudeLocations[iLocation]-RegionRadius, LongitudeLocations[iLocation]+RegionRadius]

        # Get area of region
        LatLonIndices = np.array(np.where(((longitude >= LongitudeRegion[0]) & (longitude <= LongitudeRegion[1]) & (latitude >= LatitudeRegion[0]) & (latitude <= LatitudeRegion[1]))))
        LatLonIndicesy = [np.unique(LatLonIndices[0, :])[0], np.unique(LatLonIndices[0, :])[-1]]
        LatLonIndicesx = [np.unique(LatLonIndices[1, :])[0], np.unique(LatLonIndices[1, :])[-1]]
        LES_TotalArea = (LatLonIndicesy[1]-LatLonIndicesy[0]+1) * (LatLonIndicesy[1]-LatLonIndicesy[0]+1) * 100**2

        ###############################################
        # Restrict area of LES clouds 
        Indices = np.array(np.where(((LES_MeanLon >= LongitudeRegion[0]) & (LES_MeanLon <= LongitudeRegion[1]) & (LES_MeanLat >= LatitudeRegion[0]) & (LES_MeanLat <= LatitudeRegion[1]))))[0, :]

        LES_CloudArea_Subset = np.copy(LES_Area[Indices])
        LES_EqDiameter_Subset = np.copy(LES_EqDiameter[Indices])
        LES_Times_Subset = np.copy(LES_Times[Indices])

        LES_Indices = np.array(np.where(((LES_Times_Subset > TIntervalStart) & (LES_Times_Subset <= TIntervalEnd))))[0, :]
        if len(LES_Indices) > 0:

            # Get data in time interval
            tLES_CloudArea = np.copy(LES_CloudArea_Subset[LES_Indices])

            tLES_EqDiameter = np.copy(LES_EqDiameter_Subset[LES_Indices])
            tLES_EqDiameter = tLES_EqDiameter[np.where(np.isfinite(tLES_EqDiameter))]
            tLES_EqDiameter = np.sort(tLES_EqDiameter)

            tLES_Times = np.copy(LES_Times_Subset[LES_Indices])
            tNumTimes = len(np.unique(tLES_Times))

            # Calculate fraction
            tLES_Sum = np.nansum(tLES_EqDiameter)

            tLES_Fraction = np.empty(len(tLES_EqDiameter), dtype=float)*np.nan
            for iLES in range(0, len(tLES_EqDiameter)):
                tLES_Fraction[iLES] = np.divide(np.nansum(tLES_EqDiameter[0:iLES+1]), tLES_Sum)

            # Calculate percentiles
            LES_CloudFraction[iLocation, iInterval] = np.divide(np.nansum(tLES_CloudArea), LES_TotalArea*tNumTimes)

            LES_EqdiameterBoxPlot[iLocation, :, iInterval] = np.nanpercentile(tLES_EqDiameter, [5, 25, 50, 75, 95])

            for PercentileStep, iPercentile in enumerate([0.05, 0.25, 0.50, 0.75, 0.95]):
                tIndice = np.array(np.where(tLES_Fraction >= iPercentile))[0]
                if len(tIndice) > 0:
                    LES_EqdiaFractionBoxPlot[iLocation, PercentileStep, iInterval] = np.copy(tLES_EqDiameter[tIndice[0]])
                else:
                    LES_EqdiaFractionBoxPlot[iLocation, PercentileStep, iInterval] = np.nan

            # Calculate means
            LES_EqdiameterMean[iLocation, iInterval] = np.nanmean(tLES_EqDiameter)

            tMean = np.nanmean(tLES_Fraction)
            MeanIndice = np.array(np.where(tLES_Fraction >= tMean))[0]
            if len(MeanIndice) > 0:
                LES_EqdiaFractionMean[iLocation, iInterval] = np.copy(tLES_EqDiameter[MeanIndice[0]])
            else:
                LES_EqdiaFractionMean[iLocation, iInterval] = np.nan

    IntervalStart = IntervalEnd

#############################################
# Plot cloud fraction
plt.figure()
plt.title('Cloud Fraction', fontsize=14, y=1.01)
plt.plot(np.arange(0, 14), LES_CloudFraction[0, :], color='firebrick', linewidth=2)
plt.plot(np.arange(0, 14), LES_CloudFraction[1, :], color='dodgerblue', linewidth=2)
plt.plot(np.arange(0, 14), LES_CloudFraction[2, :], color='forestgreen', linewidth=2)
plt.plot(np.arange(0, 14), LES_CloudFraction[3, :], color='chocolate', linewidth=2)
plt.plot(np.arange(0, 14), TSI_CloudFraction, color='dimgrey', linewidth=3)
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
#plt.show()

############################################
# Plot box plots

fig, ax = plt.subplots(2, 1)
fig.suptitle('Observed and Modeled Cloud Equivalent Diameter', fontsize=14, y=0.95)

ax[0].set_title('LES Data', fontsize=12, y=0.8, x=0.1)
ax[0].fill_between(np.arange(0, 14), LES_EqdiameterBoxPlot[0, 1, :], LES_EqdiameterBoxPlot[0, 3, :], color='firebrick', alpha=0.2, linewidth=0)
ax[0].fill_between(np.arange(0, 14), LES_EqdiameterBoxPlot[1, 1, :], LES_EqdiameterBoxPlot[1, 3, :], color='dodgerblue', alpha=0.2, linewidth=0)
ax[0].fill_between(np.arange(0, 14), LES_EqdiameterBoxPlot[2, 1, :], LES_EqdiameterBoxPlot[2, 3, :], color='forestgreen', alpha=0.2, linewidth=0)
ax[0].fill_between(np.arange(0, 14), LES_EqdiameterBoxPlot[3, 1, :], LES_EqdiameterBoxPlot[3, 3, :], color='chocolate', alpha=0.2, linewidth=0)
ax[0].legend(['TSI', 'I9', 'I10', 'I6'], fontsize=10, loc='lower right', ncol=4)
ax[0].plot(np.arange(0, 14), LES_EqdiameterBoxPlot[0, 2, :], linewidth=3, color='firebrick')
ax[0].plot(np.arange(0, 14), LES_EqdiameterBoxPlot[1, 2, :], linewidth=3, color='dodgerblue')
ax[0].plot(np.arange(0, 14), LES_EqdiameterBoxPlot[2, 2, :], linewidth=3, color='forestgreen')
ax[0].plot(np.arange(0, 14), LES_EqdiameterBoxPlot[3, 2, :], linewidth=3, color='chocolate')
ax[0].set_xticks(np.arange(0, 16))
ax[0].set_xticklabels(Labels, rotation=25)
ax[0].set_xlim(2, 11)
ax[0].set_ylabel('Equivalent Diameter [m]', fontsize=8)
ax[0].set_ylim(100, 1600)
ax[0].grid(True, linestyle=':', color='gray')
ax[0].tick_params(labelsize=8)

ax[1].set_title('TSI Data', fontsize=12, y=0.8, x=0.1)
ax[1].fill_between(np.arange(0, 14), LES_EqdiameterBoxPlot[0, 1, :], LES_EqdiameterBoxPlot[0, 3, :], color='firebrick', alpha=0.25, linewidth=0)
ax[1].fill_between(np.arange(0, 14), TSI_EqdiameterBoxPlot[1, :], TSI_EqdiameterBoxPlot[3, :], color='dimgrey', alpha=0.25, linewidth=0)
ax[1].legend(['LES-TSI', 'OBS-TSI'], fontsize=10, loc='lower right')
ax[1].plot(np.arange(0, 14), LES_EqdiameterBoxPlot[0, 2, :], linewidth=3, color='firebrick')
ax[1].plot(np.arange(0, 14), TSI_EqdiameterBoxPlot[2, :], linewidth=3, color='dimgrey')
ax[1].set_xticks(np.arange(0, 16))
ax[1].set_xticklabels(Labels, rotation=25)
ax[1].set_xlim(2, 11)
ax[1].set_ylim(100, 1400)
ax[1].set_ylabel('Equivalent Diameter [m]', fontsize=8)
ax[1].set_xlabel('Time of Cell [UTC]', fontsize=8)
ax[1].grid(True, linestyle=':', color='gray')
ax[1].tick_params(labelsize=8)

plt.savefig(Figure_Location + 'ComparisonEqDia_LES-TSI.png')
plt.close()

fig, ax = plt.subplots(2, 1)
fig.suptitle('Observed and Modeled Cloud Equivalent Diameter Fraction', fontsize=12, y=0.95)

ax[0].set_title('LES Data', fontsize=12, y=0.8, x=0.1)
ax[0].fill_between(np.arange(0, 14), LES_EqdiaFractionBoxPlot[0, 1, :], LES_EqdiaFractionBoxPlot[0, 3, :], color='firebrick', alpha=0.2, linewidth=0)
ax[0].fill_between(np.arange(0, 14), LES_EqdiaFractionBoxPlot[1, 1, :], LES_EqdiaFractionBoxPlot[1, 3, :], color='dodgerblue', alpha=0.2, linewidth=0)
ax[0].fill_between(np.arange(0, 14), LES_EqdiaFractionBoxPlot[2, 1, :], LES_EqdiaFractionBoxPlot[2, 3, :], color='forestgreen', alpha=0.2, linewidth=0)
ax[0].fill_between(np.arange(0, 14), LES_EqdiaFractionBoxPlot[3, 1, :], LES_EqdiaFractionBoxPlot[3, 3, :], color='chocolate', alpha=0.2, linewidth=0)
ax[0].legend(['TSI', 'I9', 'I10', 'I6'], fontsize=10, loc='lower right', ncol=4)
ax[0].plot(np.arange(0, 14), LES_EqdiaFractionBoxPlot[0, 2, :], linewidth=3, color='firebrick')
ax[0].plot(np.arange(0, 14), LES_EqdiaFractionBoxPlot[1, 2, :], linewidth=3, color='dodgerblue')
ax[0].plot(np.arange(0, 14), LES_EqdiaFractionBoxPlot[2, 2, :], linewidth=3, color='forestgreen')
ax[0].plot(np.arange(0, 14), LES_EqdiaFractionBoxPlot[3, 2, :], linewidth=3, color='chocolate')
ax[0].set_xticks(np.arange(0, 16))
ax[0].set_xticklabels(Labels, rotation=20)
ax[0].set_xlim(2, 11)
ax[0].set_ylabel('Equivalent Diameter [m]', fontsize=8)
ax[0].set_ylim(200, 2000)
ax[0].grid(True, linestyle=':', color='gray')
ax[0].tick_params(labelsize=8)

ax[1].set_title('TSI Data', fontsize=12, y=0.8, x=0.1)
ax[1].fill_between(np.arange(0, 14), LES_EqdiaFractionBoxPlot[0, 1, :], LES_EqdiaFractionBoxPlot[0, 3, :], color='firebrick', alpha=0.25, linewidth=0)
ax[1].fill_between(np.arange(0, 14), TSI_EqdiaFractionBoxPlot[1, :], TSI_EqdiaFractionBoxPlot[3, :], color='dimgrey', alpha=0.25, linewidth=0)
ax[1].legend(['LES_TSI', 'OBS-TSI'], fontsize=10, loc='lower right')
ax[1].plot(np.arange(0, 14), LES_EqdiaFractionBoxPlot[0, 2, :], linewidth=3, color='firebrick')
ax[1].plot(np.arange(0, 14), TSI_EqdiaFractionBoxPlot[2, :], linewidth=3, color='dimgrey')
ax[1].set_xticks(np.arange(0, 16))
ax[1].set_xticklabels(Labels, rotation=20)
ax[1].set_xlim(2, 11)
ax[1].set_ylim(200, 2000)
ax[1].set_ylabel('Equivalent Diameter [m]', fontsize=8)
ax[1].set_xlabel('Time of Cell [UTC]', fontsize=8)
ax[1].grid(True, linestyle=':', color='gray')
ax[1].tick_params(labelsize=8)

plt.savefig(Figure_Location + 'ComparisonEqDiaFraction_LES-TSI.png')
plt.close()
#plt.show()
