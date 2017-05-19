from netCDF4 import Dataset, stringtochar
import os
import numpy as np
import sys
import datetime
import calendar
import time
import matplotlib.pyplot as plt

# purpose: identifies features and outputs netcdf files

# author: orginial idl version written by sally a. mcfarlane (sally.mcfarlane@pnnl.gov) and then modified by zhe feng (zhe.feng@pnnl.gov). python version written by hannah barnes (hannah.barnes@pnnl.gov)

############################################/global/homes/h/hcbarnes/Tracking/testdata/################
# function used to handle test data
def idclouds_mergedir(datafilepath, datafiledatestring, datafiletimestring, datafilebasetime, datasource, datadescription, cloudid_version, dataoutpath, latlon_file, geolimits, startdate, enddate, pixel_radius, area_thresh, cloudtb_threshs, absolutetb_threshs, miss_thresh, warmanvilexpansion):
    # inputs:
    # datasource - source of the data
    # datadescription - description of data source, included in all output file names
    # datapath - path of the data directory
    # databasename - base name of data files
    # cloudid_version - version of cloud identification being run, set at the start of the beginning of run_test.py
    # dataoutpath - path to destination of the output
    # latlon_file - filename of the file that contains the latitude and longitude data
    # geolimits - 4-element array with plotting boundaries [lat_min, lon_min, lat_max, lon_max]
    # startdate - data to start processing in yyyymmdd format
    # enddate - data to stop processing in yyyymmdd format
    # pixel_radius - radius of pixels in km
    # area_thresh - minimum area thershold to define a feature in km^2
    # tb_threshs - brightness temperature thresholds 
    # miss_thresh - minimum amount of data required in order for the file to not to be considered corrupt. 

    # output: (concatenated into netcdf file located in tracking subdirectory)
    # lon - longitudes used during identification process
    # lat - latitudes used during indentification process
    # tb - brightness temperatures used during identification process
    # cloudtype - map of cloud type at each pixel (1 = core, 2 = cold anvil, 3 = warm anvil, 4 = other cloud)
    # cloudnumber - map of cloud number of each feature. includes core, cold anvil, and warm anvil area
    # nclouds - number of features identified
    # ncorepix - number of core pixels in each feature
    # ncoldpix - number of cold anvil pixels in each feature
    # nwarmpix - number of warm anvil pixels in each feature

    ##########################################################
    # define constants:
    # minimum and maximum brightness temperature thresholds. data outside of this range is filtered
    mintb_thresh = absolutetb_threshs[0]    # k
    maxtb_thresh = absolutetb_threshs[1]    # k

    ########################################################
    # load data:
    # geolocation data
    if datasource == 'mergedir':
        geolocation_data = Dataset(latlon_file, 'r')      # open file
        in_lat = geolocation_data.variables['lat'][:]     # load latitude data
        in_lon = geolocation_data.variables['lon'][:]     # load longitude data
        geolocation_data.close()                          # close file

    # Brightness temperature data. 
    # get date and time of each file. file name formate is "irdata_yyyymmdd_hhmm.nc" thus yyyymmdd=7:15, hhmm=16:20
    if datasource == 'mergedir':
        print(datafilepath)

        # set fill value
        fillval= -9999

        # load brighttness temperature data. automatically removes missing values
        rawdata = Dataset(datafilepath, 'r')           # open file
        in_ir = rawdata.variables['tb'][:]                # load brightness temperature data
        rawdata.close()                                   # close file

        #####################################################
        # mask brightness temperatures outside of normal range
        in_ir[in_ir < mintb_thresh] = fillval
        in_ir[in_ir > maxtb_thresh] = fillval
        
        #####################################################
        # determine geographic region of interest is within the data set. if it is proceed and limit the data to that geographic region. if not exit the code.

        # isolate data within lat/lon range set by limit
        lat_indices = np.array(np.where((in_lat[:,0] > geolimits[0]) & (in_lat[:,0] < geolimits[2])))[0,:]
        lon_indices = np.array(np.where((in_lon[0,:] > geolimits[1]) & (in_lon[0,:] < geolimits[3])))[0,:]
        
        # proceed if file covers the geographic region in interest
        if len(lat_indices) > 0 and len(lon_indices) > 0:
            out_lat = in_lat[lat_indices[0]:lat_indices[-1], lon_indices[0]:lon_indices[-1]]
            out_lon = in_lon[lat_indices[0]:lat_indices[-1], lon_indices[0]:lon_indices[-1]]
            out_ir = in_ir[0, lat_indices[0]:lat_indices[-1], lon_indices[0]:lon_indices[-1]]

            ######################################################
            # proceed only if number of missing data does not exceed an accepable threshold
            # determine number of missing data
            missingcount = len(np.array(np.where(out_ir == fillval))[0,:])
            ny, nx = np.shape(out_ir)

            if np.divide(missingcount, (ny*nx)) < miss_thresh:
                ######################################################
                # call idclouds subroutine
                from subroutine_idclouds import futyan

                clouddata = futyan(out_ir, pixel_radius, cloudtb_threshs, area_thresh, warmanvilexpansion)

                ######################################################
                # separate output from futyan into the separate variables
                final_nclouds = clouddata['final_nclouds']
                final_ncorepix = clouddata['final_ncorepix']
                final_ncoldpix = clouddata['final_ncoldpix']
                final_ncorecoldpix = clouddata['final_ncorecoldpix']
                final_nwarmpix = clouddata['final_nwarmpix']
                final_cloudtype = clouddata['final_cloudtype']
                final_cloudnumber = clouddata['final_cloudnumber']
                final_convcold_cloudnumber = clouddata['final_convcold_cloudnumber']
                
                #######################################################
                # output data to netcdf file
                
                # create file
                filesave = Dataset(dataoutpath + datasource + '_' + datadescription + '_cloudid' + cloudid_version + '_' + datafiledatestring + '_' + datafiletimestring + '.nc', 'w', format='NETCDF4_CLASSIC')
                
                # set global attributes
                filesave.Convenctions = 'CF-1.6'
                filesave.title = 'Statistics about convective features identified in the data from ' + datafiledatestring[0:4] + '/' + datafiledatestring[4:6] + '/' + datafiledatestring[6:8] + ' ' + datafiletimestring[0:2] + ':' + datafiletimestring[2:4] + ' utc'
                filesave.institution = 'Pacific Northwest National Laboratory'
                filesave.setncattr('Contact', 'Hannah C Barnes: hannah.barnes@pnnl.gov')
                filesave.history = 'Created ' + time.ctime(time.time())
                filesave.setncattr('cloudid_cloud_version', cloudid_version)
                filesave.setncattr('tir_threshold_cold', str(int(cloudtb_threshs[0])) + 'K')
                filesave.setncattr('tir_threshold_coldanvil', str(int(cloudtb_threshs[1])) + 'K')
                filesave.setncattr('tir_threshold_warmanvil', str(int(cloudtb_threshs[2])) + 'K')
                filesave.setncattr('tir_threshold_environment', str(int(cloudtb_threshs[3])) + 'K')
                filesave.setncattr('minimum_cloud_area', str(int(area_thresh)) + 'km^2')
                
                # create netcdf dimensions
                filesave.createDimension('time', None)
                filesave.createDimension('lat', ny)
                filesave.createDimension('lon', nx)
                filesave.createDimension('nclouds', final_nclouds)
                filesave.createDimension('ndatechar', 8)
                filesave.createDimension('ntimechar', 4)
                
                # define netcdf variables
                basetime = filesave.createVariable('basetime', 'i4', 'time', zlib=True)
                basetime.long_name = 'base time in epoch'
                basetime.units = 'seconds since 01/01/1970 00:00'
                basetime.standard_name = 'time'
                
                filedate = filesave.createVariable('filedate', 'S1', ('time', 'ndatechar'), zlib=True)
                filedate.long_name = 'date of file (yyyymmdd)'
                filedate.units = 'unitless'
                
                filetime = filesave.createVariable('filetime', 'S1', ('time', 'ntimechar'), zlib=True)
                filetime.long_name = 'time of file (hhmm)'
                filetime.units = 'unitless'

                lat = filesave.createVariable('lat', 'f4', 'lat', zlib=True)
                lat.long_name = 'y-coordinate in Cartesian system'
                lat.valid_min = geolimits[0]
                lat.valid_max = geolimits[2]
                lat.axis = 'Y'
                lat.units = 'degrees_north'
                lat.standard_name = 'latitude'

                lon = filesave.createVariable('lon', 'f4', 'lon', zlib=True)
                lon.valid_min = geolimits[1]
                lon.valid_max = geolimits[3]
                lon.axis = 'X'
                lon.long_name = 'x-coordinate in Cartesian system'
                lon.units = 'degrees_east'
                lon.standard_name = 'longitude'
                        
                latitude = filesave.createVariable('latitude', 'f4', ('lat', 'lon'), zlib=True, complevel=5)
                latitude.long_name = 'latitude'
                latitude.valid_min = geolimits[0]
                latitude.valid_max = geolimits[2]
                latitude.units = 'degrees_north'
                latitude.standard_name = 'latitude'
                
                longitude = filesave.createVariable('longitude', 'f4', ('lat', 'lon'), zlib=True, complevel=5)
                longitude.long_name = 'longitude'
                longitude.valid_min = geolimits[1]
                longitude.valid_max = geolimits[3]
                longitude.units = 'degrees_east'
                longitude.standard_name = 'longitude'
                
                tb = filesave.createVariable('tb', 'f4', ('time', 'lat', 'lon'), zlib=True, complevel=5, fill_value=fillval)
                tb.long_name = 'brightness temperature'
                tb.units = 'K'
                tb.valid_min = mintb_thresh
                tb.valid_max = maxtb_thresh
                tb.standard_name = 'brightness_temperature'
                
                cloudtype = filesave.createVariable('cloudtype', 'i4', ('time', 'lat', 'lon'), zlib=True, complevel=5, fill_value=5)
                cloudtype.long_name = 'cloud type of pixel: 1 = convective core, 2 = cold anvil, 3 = warm anvil, 4 = other cloud'
                cloudtype.units = 'unitless'
                cloudtype.valid_min = 1
                cloudtype.valid_max = 4
                
                convcold_cloudnumber = filesave.createVariable('convcold_cloudnumber', 'i4', ('time', 'lat', 'lon'), zlib=True, complevel=5, fill_value=0)
                convcold_cloudnumber.long_name = 'number of cloud system in this file at each pixel'
                convcold_cloudnumber.units = 'unitless'
                convcold_cloudnumber.valid_min = 1
                convcold_cloudnumber.valid_max = final_nclouds
                convcold_cloudnumber.comment = 'the extent of the cloud system is defined using the cold anvil threshold'

                cloudnumber = filesave.createVariable('cloudnumber', 'i4', ('time', 'lat', 'lon'), zlib=True, complevel=5, fill_value=0)
                cloudnumber.long_name = 'number of cloud system in this file at a pixel'
                cloudnumber.units = 'unitless'
                cloudnumber.valid_min = 1
                cloudnumber.valid_max = final_nclouds
                cloudnumber.comment = 'the extend of the cloud system is defined using the warm anvil threshold'
                        
                nclouds = filesave.createVariable('nclouds', 'i4', 'time', zlib=True)
                nclouds.long_name = 'number of distict convective cores identified in file'
                nclouds.units = 'unitless'
                nclouds.valid_min = 0
                nclouds.valid_max = final_nclouds

                ncorepix = filesave.createVariable('ncorepix', 'i4', ('time', 'nclouds'), zlib=True, fill_value=fillval)
                ncorepix.long_name = 'number of convective core pixels in each cloud feature'
                ncorepix.units = 'unitless'
                ncorepix.valid_min = 0
                
                ncoldpix = filesave.createVariable('ncoldpix', 'i4', ('time', 'nclouds'), zlib=True, fill_value=fillval)
                ncoldpix.long_name = 'number of cold anvil pixels in each cloud feature'
                ncoldpix.units = 'unitless'
                ncoldpix.valid_min = 0

                ncorecoldpix = filesave.createVariable('ncorecoldpix', 'i4', ('time', 'nclouds'), zlib=True, fill_value=fillval)
                ncorecoldpix.long_name = 'number of convective core and cold anvil pixels in each cloud feature'
                ncorecoldpix.units = 'unitless'
                ncorecoldpix.valid_min = 0

                nwarmpix = filesave.createVariable('nwarmpix', 'i4', ('time', 'nclouds'), zlib=True, fill_value=fillval)
                nwarmpix.long_name = 'number of warm anvil pixels in each cloud feature'
                nwarmpix.units = 'unitless'
                nwarmpix.valid_min = 0

                # fill netcdf variables with data
                basetime[:] = datafilebasetime
                filedate[0,:] = stringtochar(np.array(datafiledatestring))
                filetime[0,:] = stringtochar(np.array(datafiletimestring))
                lon[:] = np.squeeze(in_lon[0,lon_indices[0]:lon_indices[-1]])
                lat[:] = in_lat[lat_indices[0]:lat_indices[-1],0]
                longitude[:] = out_lon[:,:]
                latitude[:] = out_lat[:,:]
                tb[0,:,:] = out_ir[:,:]
                cloudtype[0,:,:] = final_cloudtype[:,:]
                convcold_cloudnumber[0,:,:] = final_convcold_cloudnumber[:,:]
                cloudnumber[0,:,:] = final_cloudnumber[:,:]
                nclouds[:] = final_nclouds
                ncorepix[0,:] = final_ncorepix[:]
                ncoldpix[0,:] = final_ncoldpix[:]
                ncorecoldpix[0,:] = final_ncorecoldpix[:]
                nwarmpix[0,:] = final_nwarmpix[:]
                
                # save and close file
                filesave.close()

            else:
                print(file)
                sys.exit('data not within latitude, longitude range. check specified geographic range')


                
                







