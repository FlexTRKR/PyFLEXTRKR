import numpy as np
import matplotlib.pyplot as plt
import gzip
import os, fnmatch
import math
import datetime, calendar
from pytz import timezone, utc

###########################################
# Set data location
LES_location = '/global/homes/h/hcbarnes/LES/data/'

##########################################
# Load latitude, longitude, and elevation data
GeographicData = np.loadtxt(LES_location + 'coordinates_d02_big.dat', dtype=float)
ElevationVector = GeographicData[:, 0]
LatitudeVector = GeographicData[:, 1]
LongitudeVector = GeographicData[:, 2]

print(np.nanmin(LatitudeVector))
print(np.nanmin(LongitudeVector))
print(np.nanmax(LatitudeVector))
print(np.nanmax(LongitudeVector))

#########################################
# Reshape geographic vectors into matrices
NumXGrids = 1200
NumYGrids = 1200

ElevationMatrix = np.reshape(ElevationVector, (NumYGrids, NumXGrids))
LatitudeMatrix = np.reshape(LatitudeVector, (NumYGrids, NumXGrids))
LongitudeMatrix = np.reshape(LongitudeVector, (NumYGrids, NumXGrids))

##########################################
# Load single LES file
LESfilename = 'outmet_d02_2016-08-30_20:44:00'

iYear = int(LESfilename[11:15])
iMonth = int(LESfilename[16:18])
iDay = int(LESfilename[19:21])
iHour = int(LESfilename[22:24])
iMinute = int(LESfilename[25:27])
TEMP_basetime = datetime.datetime(iYear, iMonth, iDay, iHour, iMinute, 0, tzinfo=utc)
BaseTime = calendar.timegm(TEMP_basetime.timetuple())

LWPvector = np.loadtxt(LES_location + LESfilename, dtype=float)
LWPvector[LWPvector < 0.05] = np.nan
LWPmatrix = np.reshape(LWPvector, (NumYGrids, NumXGrids))

plt.figure()
plt.title(str(iYear) + '/' + str(iMonth) + '/' + str(iDay) + ' ' + str(iHour) + ':' + str(iMinute) + ' UTC', fontsize=14, y=1.01)
im = plt.pcolormesh(LongitudeMatrix, LatitudeMatrix, np.ma.masked_invalid(np.atleast_2d(LWPmatrix)), cmap='nipy_spectral_r', vmin=0, vmax=2.5)
plt.colorbar(im)
plt.xlabel('Longitude', fontsize=12, labelpad=10)
plt.xlim(np.nanmin(LongitudeVector), np.nanmax(LongitudeVector))
plt.ylabel('Latitude', fontsize=12, labelpad=10)
plt.ylim(np.nanmin(LatitudeVector), np.nanmax(LatitudeVector))
plt.tick_params(labelsize=10)
plt.show()

raw_input('wait')

##########################################
# Load all LES output data

# Get all file names LES directory
allLWPfilenames = fnmatch.filter(os.listdir(LES_location), 'outmet_d02*')
allLWPfilenames = sorted(allLWPfilenames)
NumLWPfilenames = len(allLWPfilenames)

# Intialize matrix
LWPmatrix = np.empty((NumLWPfilenames, NumYGrids, NumXGrids), dtype=float)*np.nan
BaseTime = np.zeros(NumLWPfilenames, dtype=int)

# Load data
for Fstep, iF in enumerate(allLWPfilenames):

    iYear = int(iF[11:15])
    iMonth = int(iF[16:18])
    iDay = int(iF[19:21])
    iHour = int(iF[22:24])
    iMinute = int(iF[25:27])

    TEMP_basetime = datetime.datetime(iYear, iMonth, iDay, iHour, iMinute, 0, tzinfo=utc)
    BaseTime[Fstep] = calendar.timegm(TEMP_basetime.timetuple())

    DateTimeString = iF[11:-3]

    LWPvector = np.loadtxt(LES_location + str(iF), dtype=float)
    LWPvector[LWPvector == 0] = np.nan
    LWPmatrix[Fstep, :, :] = np.reshape(LWPvector, (NumYGrids, NumXGrids))

#########################################################
# Select one time to plot

cYear = 2016
cMonth = 8
cDay = 30
cHour = 18
cMinute = 38

TEMP_basetime = datetime.datetime(cYear, cMonth, cDay, cHour, cMinute, 0, tzinfo=utc)
cBaseTime[Fstep] = calendar.timegm(TEMP_basetime.timetuple())

iC = np.array(np.where(BaseTime == cBaseTime))[0]

plt.figure()
plt.title(str(cYear) + '/' + str(cMonth) + '/' + str(cDay) + ' ' + str(cHour) + ':' + str(cMinute) + ' UTC', fontsize=14, y=1.01)
im = plt.pcolormesh(LongitudeMatrix, LatitudeMatrix, np.ma.masked_invalid(np.atleast_2d(LWPmatrix[iC, :, :])))
plt.colorbar(im)
plt.xlabel('Longitude', fontsize=12, labelpad=10)
plt.ylabel('Latitude', fontsize=12, labelpad=10)
plt.tick_params(labelsize=10)
plt.show()

