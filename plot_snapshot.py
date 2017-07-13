import numpy as np
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import os
import fnmatch
import datetime
import calendar
from pytz import timezone, utc

# Specify days to run
startdate = '20110517'
enddate = '20110527'

# Set path to data
data_path = '/global/homes/h/hcbarnes/Tracking/Satellite/mcstracking/'
basename = 'mcstracks_'

# Set figure location
figure_location = '/global/homes/h/hcbarnes/Tracking/Satellite/maps/'

######################################################################
# Calculate start and end basetime
starttime = datetime.datetime(int(startdate[0:4]), int(startdate[4:6]), int(startdate[6:8]), 0, 0, 0, tzinfo=utc)
start_basetime = calendar.timegm(starttime.timetuple())

endtime = datetime.datetime(int(enddate[0:4]), int(enddate[4:6]), int(enddate[6:8]), 23, 0, 0, tzinfo=utc)
end_basetime = calendar.timegm(endtime.timetuple())

######################################################################
# Isolate all possible files
datafiles = fnmatch.filter(os.listdir(data_path), basename+'*')

# Loop through files, identifying files within the startdate - enddate interval
nleadingchar = np.array(len(basename)).astype(int)

for ifile in datafiles:
    filetime = datetime.datetime(int(ifile[nleadingchar:nleadingchar+4]), int(ifile[nleadingchar+4:nleadingchar+6]), int(ifile[nleadingchar+6:nleadingchar+8]), int(ifile[nleadingchar+9:nleadingchar+11]), int(ifile[nleadingchar+11:nleadingchar+13]), 0, tzinfo=utc)
    filebasetime = calendar.timegm(filetime.timetuple())
    
    if filebasetime >= start_basetime and filebasetime <= end_basetime:
        datafile_path = data_path + ifile
        timestring = ifile[nleadingchar+9:nleadingchar+11] + ifile[nleadingchar+11:nleadingchar+13]
        datestring = ifile[nleadingchar:nleadingchar+4] + ifile[nleadingchar+4:nleadingchar+6] + ifile[nleadingchar+6:nleadingchar+8]

        #############################################################
        # Load data
        data = Dataset(datafile_path, 'r')
        tb = data.variables['tb'][:]
        cloudtracks = data.variables['tracknumber'][:]
        mcstracks = data.variables['mcstracknumber'][:]
        latitude = data.variables['latitude'][:]
        longitude = data.variables['longitude'][:]
        data.close()

        cloudtracks = np.ma.masked_invalid(np.atleast_2d(cloudtracks[0, :, :]))
        mcstracks =  np.ma.masked_invalid(np.atleast_2d(mcstracks[0, :, :]))

        fig = plt.figure()
        fig.suptitle(datestring + ' ' + timestring + 'UTC', fontsize=16)

        ax0 = fig.add_axes([0.23, 0.65, 0.5, 0.25])
        ax1 = fig.add_axes([0.23, 0.35, 0.5, 0.25])
        ax2 = fig.add_axes([0.23, 0.05, 0.5, 0.25])

        ax0.set_title('Brightness Temperature [K]', fontsize=12)
        map0 = Basemap(lon_0=-89, lat_0=38, llcrnrlon=-110, llcrnrlat=25, urcrnrlon=-70, urcrnrlat=50, ax=ax0, resolution='l')
        map0.drawcoastlines(linewidth=1)
        map0.drawcountries(linewidth=1)
        map0.drawstates(linewidth=1)
        im0 = map0.pcolormesh(longitude, latitude, tb[0, :, :], vmin=210, vmax=290)
        cbar0 = plt.colorbar(im0, ax=ax0)
        cbar0.ax.tick_params(labelsize=10)
        #cbar0.set_ticks(np.arange(200, 300, 0))

        ax1.set_title('Cloud Tracks', fontsize=12)
        map1 = Basemap(lon_0=-89, lat_0=38, llcrnrlon=-110, llcrnrlat=25, urcrnrlon=-70, urcrnrlat=50, ax=ax1, resolution='l')
        map1.drawcoastlines(linewidth=1)
        map1.drawcountries(linewidth=1)
        map1.drawstates(linewidth=1)
        im1 = map1.pcolormesh(longitude, latitude, cloudtracks, vmin=np.nanmin(np.nanmin(cloudtracks)), vmax=np.nanmax(np.nanmax(cloudtracks)))
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.ax.tick_params(labelsize=10)
        cbar1.set_ticks(np.arange(np.nanmin(np.nanmin(cloudtracks)), np.nanmax(np.nanmax(cloudtracks))+10, (np.nanmax(np.nanmax(cloudtracks)) - (np.nanmin(np.nanmin(cloudtracks))))/6))

        ax2.set_title('Mesoscale Convective System Tracks', fontsize=12)
        map2 = Basemap(lon_0=-89, lat_0=38, llcrnrlon=-110, llcrnrlat=25, urcrnrlon=-70, urcrnrlat=50, ax=ax2, resolution='l')
        map2.drawcoastlines(linewidth=1)
        map2.drawcountries(linewidth=1)
        map2.drawstates(linewidth=1)
        im2 = map2.pcolormesh(longitude, latitude, mcstracks, vmin=0, vmax=30)
        #im2 = map2.pcolormesh(longitude, latitude, mcstracks, vmin=np.nanmin(np.nanmin(mcstracks)), vmax=np.nanmax(np.nanmax(mcstracks)))
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_ticks(np.arange(0, 30 , 5))
        cbar2.ax.tick_params(labelsize=10)

        plt.savefig(figure_location + 'TB-CloudTrack-MCSTrack_' + datestring + '-' + timestring + '.png', dpi=300)
        plt.close()
        #plt.show()


        
