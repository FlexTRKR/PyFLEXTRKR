import numpy as np
from netCDF4 import Dataset
import datetime
import calendar
from pytz import timezone, utc

###########################################################
# Set date and cloud of interest
InterestTime = datetime.datetime(int(2011), int(05), int(24), 15, 00, 0, tzinfo=utc)
InterestBaseTime = calendar.timegm(InterestTime.timetuple())

InterestCloud = 3

AreaThreshold = 6e4

##################################################
# Load data
Data = Dataset('/Users/barn327/Tracking/scripts/python/stats_tracknumbersv4.4_20110517_20110527.nc', 'r')
BaseTime = Data.variables['basetime'][:]
DateTime = Data.variables['datetimestring'][:]
CloudNumber = Data.variables['cloudnumber'][:]
MergeSplitNumber = Data.variables['mergesplitnumber'][:]
CloudStatus = Data.variables['result'][:]
CorePixels = Data.variables['nconv'][:]
ColdPixels = Data.variables['ncoldanvil'][:]
CoreColdPixels = Data.variables['npix'][:]
MeanLatitude = Data.variables['meanlat'][:]
MeanLongitude = Data.variables['meanlon'][:]
PixelRadius = Dataset.getncattr(Data, 'pixel_radius_km')
Data.close()

for itrack in range(0, len(CloudNumber)):
    testy, testx = np.array(np.where(MergeSplitNumber == itrack + 1))
    print(itrack)
    print(testy, testx)
    for imergesplit in range(0, len(testy)):
        print(np.array(np.where(BaseTime[itrack, :] == BaseTime[testy[imergesplit], testx[imergesplit]])))
    print(BaseTime[itrack, 0:50])
    print(BaseTime[testy, testx])
    raw_input('Waiting')
print(np.shape(CloudNumber))
print(np.shape(MergeSplitNumber))
print(np.shape(BaseTime))
raw_input('Waiting')

#####################################################
# Calculate area
CoreColdArea = (CoreColdPixels)*PixelRadius**2

####################################################
# Find track of interest
TrackLocation, CloudLocation = np.where((BaseTime == InterestBaseTime) & (CloudNumber == InterestCloud))

###################################################
# Print track data
TrackArea = CoreColdArea[TrackLocation, :]

# Cold cloud shield area requirement
AboveThreshold = np.array(np.where(TrackArea[0,:] > AreaThreshold))[0,:]
nClouds = len(AboveThreshold)

# Find continuous times
Groups = np.split(AboveThreshold, np.where(np.diff(AboveThreshold) != 1)[0]+1)
nBreaks = len(Groups)

print(TrackLocation)
print(TrackArea)
print(AboveThreshold)
print(Groups)
