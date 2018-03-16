import numpy as np
import xarray as xr
import datetime
np.set_printoptions(threshold=np.inf)

###########################
# Set time of interest
Month = '05'
Day = '08'
Hour = '09'

##############################
# Set start and end date
StartTime = 20110401
EndTime = 20110831

################################
# Set data locations
IDLStatsLocation = '/global/project/projectdirs/m1657/zfeng/usa/mergedir/stats/'
PythonStatsLocation = '/global/homes/h/hcbarnes/Tracking/SatelliteRadar/stats/'

#############################
IDL_AllFileName = 'mcs_tracks_nmq_' + str(StartTime) + '_' + str(EndTime) + '.nc'
IDL_RobustFileName = 'robust_mcs_tracks_' + str(StartTime) + '_' + str(EndTime) + '.nc'

Python_AllFileName = 'mcs_tracks_nmq_' + str(StartTime) + '_' + str(EndTime) + '.nc'
Python_RobustFileName = 'robust_mcs_tracks_nmq_' + str(StartTime) + '_' + str(EndTime) + '.nc'

###########################
# Load data
IDL_AllDataHandle = xr.open_dataset(IDLStatsLocation + IDL_AllFileName, autoclose=True)
IDL_AllBasetime = IDL_AllDataHandle['base_time'].data
IDL_AllLatitude = IDL_AllDataHandle['meanlat'].data
IDL_AllLongitude = IDL_AllDataHandle['meanlon'].data
IDL_All50area = IDL_AllDataHandle['pf_dbz50area'].data
IDL_AllMajoraxis = IDL_AllDataHandle['pf_majoraxislength'].data
IDL_AllEnd = IDL_AllDataHandle['endtrackresult'].data

IDL_RobustDataHandle = xr.open_dataset(IDLStatsLocation + IDL_RobustFileName, autoclose=True)
IDL_RobustBasetime = IDL_RobustDataHandle['base_time'].data
IDL_RobustLatitude = IDL_RobustDataHandle['meanlat'].data
IDL_RobustLongitude = IDL_RobustDataHandle['meanlon'].data
IDL_Robust50area = IDL_RobustDataHandle['pf_dbz50area'].data
IDL_RobustMajoraxis = IDL_RobustDataHandle['pf_majoraxislength'].data

Python_AllDataHandle = xr.open_dataset(PythonStatsLocation + Python_AllFileName, autoclose=True)
Python_AllBasetime = Python_AllDataHandle['basetime'].data
Python_AllLatitude = Python_AllDataHandle['meanlat'].data
Python_AllLongitude = Python_AllDataHandle['meanlon'].data
Python_All50area = Python_AllDataHandle['pf_dbz50area'].data
Python_AllMajoraxis = Python_AllDataHandle['pf_majoraxislength'].data
Python_AllEnd = Python_AllDataHandle['endstatus'].data

Python_RobustDataHandle = xr.open_dataset(PythonStatsLocation + Python_RobustFileName, autoclose=True)
Python_RobustBasetime = Python_RobustDataHandle['base_time'].data
Python_RobustLatitude = Python_RobustDataHandle['meanlat'].data
Python_RobustLongitude = Python_RobustDataHandle['meanlon'].data
Python_Robust50area = Python_RobustDataHandle['pf_dbz50area'].data
Python_RobustMajoraxis = Python_RobustDataHandle['pf_majoraxislength'].data

##########################
# Set IDL Index
print(IDL_AllBasetime[:, 0])
InvestigationTime = np.array('2011-' + Month + '-' + Day + 'T' + Hour + ':00', dtype='datetime64[ns]')
PossibleIndices = np.array(np.where(IDL_AllBasetime[:, 0] == InvestigationTime))[0, :]
print('Possible Indices (All): ' + str(PossibleIndices))
print('All Start Times: ' + str(IDL_AllBasetime[PossibleIndices, 0]))
print('All Mean Latitude: ' + str(IDL_AllLatitude[PossibleIndices, 0]))
print('All Mean Longitude: ' + str(IDL_AllLongitude[PossibleIndices, 0]))
print('All End Status: ' + str(IDL_AllEnd[PossibleIndices]))
IDL_AllIndex = np.array(raw_input('Select IDL Index: ')).astype(int)

print(IDL_RobustBasetime[:, 0])
PossibleIndices = np.array(np.where(IDL_RobustBasetime[:, 0] == InvestigationTime))[0, :]
print('Possible Indices (Robust): ' + str(PossibleIndices))
print('Robust Start Times: ' + str(IDL_RobustBasetime[PossibleIndices, 0]))
print('Robust Mean Latitude: ' + str(IDL_RobustLatitude[PossibleIndices, 0]))
print('Robust Mean Longitude: ' + str(IDL_RobustLongitude[PossibleIndices, 0]))
IDL_RobustIndex = np.array(raw_input('Select IDL Index: ')).astype(int)

##########################
# Set Python Index
print(Python_AllBasetime[:, 0])
PossibleIndices = np.array(np.where(Python_AllBasetime[:, 0] == InvestigationTime))[0, :]
print('Possible Indices: ' + str(PossibleIndices))
print('Start Times: ' + str(Python_AllBasetime[PossibleIndices, 0]))
print('Mean Latitude: ' + str(Python_AllLatitude[PossibleIndices, 0]))
print('Mean Longitude: ' + str(Python_AllLongitude[PossibleIndices, 0]))
print('End Status: ' + str(Python_AllEnd[PossibleIndices]))
PythonIndex = np.array(raw_input('Select Python Index: ')).astype(int)

print(Python_RobustBasetime[:, 0])
PossibleIndices = np.array(np.where(Python_RobustBasetime[:, 0] == InvestigationTime))[0, :]
print('Possible Indices (Robust): ' + str(PossibleIndices))
print('Robust Start Times: ' + str(Python_RobustBasetime[PossibleIndices, 0]))
print('Robust Mean Latitude: ' + str(Python_RobustLatitude[PossibleIndices, 0]))
print('Robust Mean Longitude: ' + str(Python_RobustLongitude[PossibleIndices, 0]))
Python_RobustIndex = np.array(raw_input('Select Python Index: ')).astype(int)

##########################
# Print track data
nIDLTimes_All = len(np.array(np.where(~np.isnan(IDL_AllLatitude[IDL_AllIndex, :])))[0, :])
nIDLTimes_Robust = len(np.array(np.where(~np.isnan(IDL_RobustLatitude[IDL_RobustIndex, :])))[0, :])

nPythonTimes_All = len(np.array(np.where(~np.isnan(Python_AllLatitude[Python_AllIndex, :])))[0, :])
nPythonTimes_Robust = len(np.array(np.where(~np.isnan(Python_AllLatitude[Python_AllIndex, :])))[0, :])

print('All - IDL Times: ' + str(IDL_AllBasetime[IDL_AllIndex, 0:nIDLTimes_All]))
print('All - Python Times: ' + str(Python_AllBasetime[Python_AllIndex, 0:nPythonTimes_All]))
print('Robust - IDL Times: ' + str(IDL_RobustBasetime[IDL_RobustIndex, 0:nIDLTimes_Robust]))
print('Robust - Python Times: ' + str(Python_RobustBasetime[Python_RobustIndex, 0:nPythonTimes_Robust]))
raw_input('check')

print('All - Largest IDL 50 dBZ area: ' + str(IDL_All50area[IDL_AllIndex, 0:nIDLTimes_All,  0]))
print('All - Largest Python 50 dBZ area: ' + str(Python_All50area[Python_AllIndex, 0:nPythonTimes_All,  0]))
print('Robust - Largest IDL 50 dBZ area: ' + str(IDL_Robust50area[IDL_RobustIndex, 0:nIDLTimes_Robust,  0]))
print('Robust - Largest Python 50 dBZ area: ' + str(Python_Robust50area[Python_RobustIndex, 0:nPythonTimes_Robust,  0]))
raw_input('check')

print('All - Largest IDL major axis: ' + str(IDL_AllMajoraxis[IDL_AllIndex, 0:nIDLTimes_All, 0]))
print('All - Largest Python major axis: ' + str(Python_AllMajoraxis[Python_AllIndex, 0:nPythonTimes_All, 0]))
print('Robust - Largest IDL major axis: ' + str(IDL_RobustMajoraxis[IDL_RobustIndex, 0:nIDLTimes_Robust, 0]))
print('Robust - Largest Python major axis: ' + str(Python_RobustMajoraxis[Python_RobustIndex, 0:nPythonTimes_Robust, 0]))
raw_input('check')

print('All - IDL Mean Latitude: ' + str(IDL_AllLatitude[IDL_AllIndex, 0:nIDLTimes_All]))
print('All - Python Mean Latitude: ' + str(Python_AllLatitude[Python_AllIndex, 0:nPythonTimes_All]))
print('Robust - IDL Mean Latitude: ' + str(IDL_RobustLatitude[IDL_RobustlIndex, 0:nIDLTimes_Robust]))
print('Robust - Python Mean Latitude: ' + str(Python_RobustLatitude[Python_RobustIndex, 0:nPythonTimes_Robust]))
raw_input('check')

print('All - IDL Mean Longitude: ' + str(IDL_AllLongitude[IDL_AllIndex, 0:nIDLTimes_All]))
print('All - Python Mean Longitude: ' + str(Python_AllLongitude[Python_AllIndex, 0:nPythonTimes_All]))
print('Robust - IDL Mean Longitude: ' + str(IDL_RobustLongitude[IDL_RobustIndex, 0:nIDLTimes_Robust]))
print('Robust - Python Mean Longitude: ' + str(Python_RobustLongitude[Python_RobustIndex, 0:nPythonTimes_Robust]))
raw_input('check')
