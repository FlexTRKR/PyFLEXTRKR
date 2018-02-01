import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
np.set_printoptions(threshold=np.inf)

#############################
# Set figure location
Figure_Location = '/global/homes/h/hcbarnes/Tracking/SatelliteRadar/figures/'

##############################
# Load data
print(str(sys.argv[1]))
print(str(sys.argv[2]))
print('')

IDL_Data = xr.open_dataset(str(sys.argv[1]))
Latitude = np.array(IDL_Data['latitude'].data)
Longitude = np.array(IDL_Data['longitude'].data)
BrightnessTemperature = np.array(IDL_Data['tb'][0, :, :])
IDLCloudTracks = np.array(IDL_Data['cloudtracknumber'][0, :, :])

Python_Data = xr.open_dataset(str(sys.argv[2]))
PythonCloudTracks = np.array(Python_Data['cloudtracknumber'][0, :, :])

######################################
# Separate Basetime in day and time
BaseTime = Python_Data['basetime'].data
Date = str(BaseTime)[2:12]
Time = str(BaseTime)[13:18]

#####################################
# Create state map
states_provinces = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='50m', facecolor='none')

######################################
# Create figure
fig = plt.figure()
fig.suptitle('IDL and Python Brightness Temperature and MCS Tracks \n ' + Date + ' ' + Time + ' UTC', fontsize=12, y=0.8)
fig.set_figheight(10)
fig.set_figwidth(20)

ax0 = fig.add_axes([0.07, 0.25, 0.25, 0.5], projection=ccrs.PlateCarree())
ax1 = fig.add_axes([0.37, 0.25, 0.25, 0.5], projection=ccrs.PlateCarree())
ax2 = fig.add_axes([0.67, 0.25, 0.25, 0.5], projection=ccrs.PlateCarree())

ax0.set_title('Brightness Temperature', fontsize=10, y=1.1)
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

IDLCloudTracks[np.where(np.isnan(IDLCloudTracks))] = 0
ax1.set_title('IDL MCS Tracks \n Tracks Present: ' + str(np.unique(IDLCloudTracks)[1::]), fontsize=10, y=1.1)
ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cfeature.BORDERS)
ax1.add_feature(cfeature.LAKES, alpha=0.5)
ax1.add_feature(cfeature.OCEAN, alpha=0.5)
ax1.add_feature(states_provinces, edgecolor='gray')
ax1.set_xmargin(0)
ax1.set_ymargin(0)
im1 = ax1.pcolormesh(Longitude, Latitude, np.ma.masked_invalid(np.atleast_2d(IDLCloudTracks)), cmap='nipy_spectral_r', vmin=int(sys.argv[3]), vmax=int(sys.argv[4]), linewidth=0, transform=ccrs.PlateCarree())
cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.03, pad=0.04)
cbar1.ax.set_xlabel('#', fontsize=8, labelpad=10)
ax1.set_xlabel('Tracks Present: ' + str(np.unique(IDLCloudTracks)[1::]), fontsize=8)
cbar1.ax.tick_params(labelsize=8)

PythonCloudTracks[np.where(np.isnan(PythonCloudTracks))] = 0
ax2.set_title('Python MCS Tracks \n Tracks Present: ' + str(np.unique(PythonCloudTracks)[1::]), fontsize=10, y=1.1)
ax2.add_feature(cfeature.COASTLINE)
ax2.add_feature(cfeature.BORDERS)
ax2.add_feature(cfeature.LAKES, alpha=0.5)
ax2.add_feature(cfeature.OCEAN, alpha=0.5)
ax2.add_feature(states_provinces, edgecolor='gray')
ax2.set_xmargin(0)
ax2.set_ymargin(0)
im2 = ax2.pcolormesh(Longitude, Latitude, np.ma.masked_invalid(np.atleast_2d(PythonCloudTracks)), cmap='nipy_spectral_r', vmin=int(sys.argv[5]), vmax=int(sys.argv[6]), linewidth=0, transform=ccrs.PlateCarree())
cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.03, pad=0.04)
cbar2.ax.set_xlabel('#', fontsize=8, labelpad=10)
ax2.set_xlabel('Tracks Present: ' + str(np.unique(PythonCloudTracks)[1::]), fontsize=8)
cbar2.ax.tick_params(labelsize=8)

plt.savefig(Figure_Location + 'IDLPythonComparison_' + Date + '_' + Time + '.png')
plt.close()
