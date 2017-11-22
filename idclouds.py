# purpose: identifies features and outputs netcdf files

# author: orginial idl version written by sally a. mcfarlane (sally.mcfarlane@pnnl.gov) and then modified by zhe feng (zhe.feng@pnnl.gov). python version written by hannah barnes (hannah.barnes@pnnl.gov)

############################################/global/homes/h/hcbarnes/Tracking/testdata/################
# function used to handle test data
def idclouds_mergedir(zipped_inputs):
    # inputs:
    # datasnp.array([ource - source of the data
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
    # Load modules

    from netCDF4 import Dataset, stringtochar
    import os
    import numpy as np
    import sys
    import datetime
    import calendar
    import time

    ########################################################
    # Separate inputs

    datafilepath = zipped_inputs[0]
    datafiledatestring = zipped_inputs[1]
    datafiletimestring = zipped_inputs[2]
    datafilebasetime = zipped_inputs[3]
    datasource = zipped_inputs[4]
    datadescription = zipped_inputs[5]
    variablename = zipped_inputs[6]
    cloudid_version = zipped_inputs[7]
    dataoutpath = zipped_inputs[8]
    latlon_file = zipped_inputs[9]
    latname = zipped_inputs[10]
    longname = zipped_inputs[11]
    geolimits = zipped_inputs[12]
    startdate = zipped_inputs[13]
    enddate = zipped_inputs[14]
    pixel_radius = zipped_inputs[15]
    area_thresh = zipped_inputs[16]
    cloudtb_threshs = zipped_inputs[17]
    absolutetb_threshs = zipped_inputs[18]
    miss_thresh = zipped_inputs[19]
    cloudidmethod = zipped_inputs[20]
    mincoldcorepix = zipped_inputs[21]
    smoothsize = zipped_inputs[22]
    warmanvilexpansion = zipped_inputs[23]

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
        in_lat = geolocation_data.variables[latname][:]     # load latitude data
        in_lon = geolocation_data.variables[longname][:]     # load longitude data
        geolocation_data.close()                          # close file

    # Brightness temperature data. 
    # get date and time of each file. file name formate is "irdata_yyyymmdd_hhmm.nc" thus yyyymmdd=7:15, hhmm=16:20
    if datasource == 'mergedir':
        print(datafilepath)

        # set fill value
        fillvalue = -9999

        # load brighttness temperature data. automatically removes missing values
        rawdata = Dataset(datafilepath, 'r')                            # open file
        original_ir = rawdata.variables[variablename][:]                # load brightness temperature data
        rawdata.close()                                                 # close file

        #in_ir = np.ma.getdata(original_ir)
        #mask = np.ma.getmaskarray(original_ir)
        #in_ir[np.array(np.where(mask == True))] = fillvalue

        # Replace missing ir data with mean
        datay, datax = np.array(np.ma.nonzero(original_ir))
        in_ir = np.ones(np.shape(original_ir), dtype=float)*fillvalue
        in_ir[datay, datax] = original_ir[datay, datax]

        missingdatay, missingdatax = np.array(np.where(in_ir < -90))
        if len(missingdatay) > 0:
            for imiss in np.arange(0,len(missingdatay)):
                if missingdatay[imiss] == 0:
                    if missingdatax[imiss] == 0:
                        subsetir = np.copy(in_ir[0:missingdatay[imiss]+2, 0:missingdatax[imiss]+2])
                    else:
                        subsetir = np.copy(in_ir[0:missingdatay[imiss]+2, missingdatax[imiss]-1:missingdatax[imiss]+2])
                elif missingdatax[imiss] == 0:
                    subsetir = np.copy(in_ir[missingdatay[imiss]-1:missingdatay[imiss]+2, 0:missingdatax[imiss]+2])
                elif missingdatay[imiss] == np.shape(original_ir)[0]:
                    if missingdatax[imiss] == np.shape(original_ir)[1]:
                        subsetir = np.copy(in_ir[missingdatay[imiss]-1::, missingdatax[imiss]-1::])
                    else:
                        subsetir = np.copy(in_ir[missingdatay[imiss]-1::, missingdatax[imiss]-1::missingdatax[imiss]+2])
                elif missingdatax[imiss] == np.shape(original_ir)[1]:
                     subsetir = np.copy(in_ir[missingdatay[imiss]-1:missingdatay[imiss]+2, missingdatax[imiss]-1::])
                else:
                    subsetir = np.copy(in_ir[missingdatay[imiss]-1:missingdatay[imiss]+2, missingdatax[imiss]-1:missingdatax[imiss]+2])
                subsetir[subsetir == fillvalue] = np.nan
                subsetir = np.reshape(subsetir, np.shape(subsetir)[0]*np.shape(subsetir)[1] , 1)
                in_ir[missingdatay[imiss], missingdatax[imiss]] = np.nanmean(subsetir)


        in_lat = np.transpose(in_lat)
        in_lon = np.transpose(in_lon)
        in_ir = np.transpose(in_ir)

        #####################################################
        # mask brightness temperatures outside of normal range
        in_ir[in_ir < mintb_thresh] = fillvalue
        in_ir[in_ir > maxtb_thresh] = fillvalue

        #####################################################
        # determine geographic region of interest is within the data set. if it is proceed and limit the data to that geographic region. if not exit the code.

        # isolate data within lat/lon range set by limit
        indices = np.where((in_lat > geolimits[0]) & (in_lat <= geolimits[2]) & (in_lon > geolimits[1]) & (in_lon <= geolimits[3]))

        # proceed if file covers the geographic region in interest
        if len(indices[0]) > 0 and len(indices[1]) > 0:
            out_lat = np.reshape(np.copy(in_lat[indices]), (indices[0][-1]+1, indices[1][-1]+1))
            out_lon = np.reshape(np.copy(in_lon[indices]), (indices[0][-1]+1, indices[1][-1]+1))
            out_ir = np.reshape(np.copy(in_ir[indices]), (indices[0][-1]+1, indices[1][-1]+1))

            ######################################################
            # proceed only if number of missing data does not exceed an accepable threshold
            # determine number of missing data
            missingcount = len(np.array(np.where(np.isnan(out_ir)))[0,:])
            ny, nx = np.shape(out_ir)

            if np.divide(missingcount, (ny*nx)) < miss_thresh:
                ######################################################
                # call idclouds subroutine
                if cloudidmethod == 'futyan3':
                    from subroutine_idclouds import futyan3
                    clouddata = futyan3(out_ir, pixel_radius, cloudtb_threshs, area_thresh, warmanvilexpansion)
                elif cloudidmethod == 'futyan4':
                    from subroutine_idclouds import futyan4_mergedir
                    clouddata = futyan4_mergedir(out_ir, pixel_radius, cloudtb_threshs, area_thresh, mincoldcorepix, smoothsize, warmanvilexpansion)

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
                # output data to netcdf file, only if clouds present
                if final_nclouds > 0:
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
                
                    tb = filesave.createVariable('tb', 'f4', ('time', 'lat', 'lon'), zlib=True, complevel=5)
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

                    ncorepix = filesave.createVariable('ncorepix', 'i4', ('time', 'nclouds'), zlib=True, fill_value=fillvalue)
                    ncorepix.long_name = 'number of convective core pixels in each cloud feature'
                    ncorepix.units = 'unitless'
                    ncorepix.valid_min = 0
                
                    ncoldpix = filesave.createVariable('ncoldpix', 'i4', ('time', 'nclouds'), zlib=True, fill_value=fillvalue)
                    ncoldpix.long_name = 'number of cold anvil pixels in each cloud feature'
                    ncoldpix.units = 'unitless'
                    ncoldpix.valid_min = 0

                    ncorecoldpix = filesave.createVariable('ncorecoldpix', 'i4', ('time', 'nclouds'), zlib=True, fill_value=fillvalue)
                    ncorecoldpix.long_name = 'number of convective core and cold anvil pixels in each cloud feature'
                    ncorecoldpix.units = 'unitless'
                    ncorecoldpix.valid_min = 0
                    
                    nwarmpix = filesave.createVariable('nwarmpix', 'i4', ('time', 'nclouds'), zlib=True, fill_value=fillvalue)
                    nwarmpix.long_name = 'number of warm anvil pixels in each cloud feature'
                    nwarmpix.units = 'unitless'
                    nwarmpix.valid_min = 0

                    # fill netcdf variables with data
                    basetime[:] = datafilebasetime
                    filedate[0,:] = stringtochar(np.array(datafiledatestring))
                    filetime[0,:] = stringtochar(np.array(datafiletimestring))
                    lon[:] = np.squeeze(out_lon[0, :])
                    lat[:] = np.squeeze(out_lat[:, 0])
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
                    print(datafilepath)
                    print('No clouds')

            else:
                print(datafilepath)
                print('To much missing data')
        else:
            print(datafilepath)
            print('data not within latitude, longitude range. check specified geographic range')

############################################/global/homes/h/hcbarnes/Tracking/testdata/################
# function used to handle test data
def idclouds_LES(zipped_inputs):
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
    # Load modules

    from netCDF4 import Dataset, stringtochar
    import os
    import numpy as np
    import sys
    import datetime
    import calendar
    import time
    import xarray as xr
    import datetime
    import matplotlib.pyplot as plt

    ########################################################
    # Separate inputs

    datafilepath = zipped_inputs[0]
    datafiledatestring = zipped_inputs[1]
    datafiletimestring = zipped_inputs[2]
    datafilebasetime = np.array([zipped_inputs[3]], dtype=int)
    datasource = zipped_inputs[4]
    datadescription = zipped_inputs[5]
    cloudid_version = zipped_inputs[6]
    dataoutpath = zipped_inputs[7]
    latlon_file = zipped_inputs[8]
    geolimits = zipped_inputs[9]
    nx = zipped_inputs[10]
    ny = zipped_inputs[11]
    startdate = zipped_inputs[12]
    enddate = zipped_inputs[13]
    pixel_radius = zipped_inputs[14]
    area_thresh = zipped_inputs[15]
    cloudtb_threshs = zipped_inputs[16]
    absolutetb_threshs = zipped_inputs[17]
    miss_thresh = zipped_inputs[18]
    cloudidmethod = zipped_inputs[19]
    mincoldcorepix = zipped_inputs[20]
    smoothsize = zipped_inputs[21]
    warmanvilexpansion = zipped_inputs[22]

    ##########################################################
    # define constants:
    # minimum and maximum brightness temperature thresholds. data outside of this range is filtered
    mintb_thresh = absolutetb_threshs[0]    # k
    maxtb_thresh = absolutetb_threshs[1]    # k

    ########################################################
    # load data:
    # geolocation data
    if datasource == 'LES':
        # Open file
        geolocation_data = np.loadtxt(latlon_file, dtype=float)

        # Load data
        in_lat = geolocation_data[:, 1]                              
        in_lon = geolocation_data[:, 2]

        # Transform into matrix
        in_lat = np.reshape(in_lat, (ny, nx))
        in_lon = np.reshape(in_lon, (ny, nx))

    # LWP data. 
    if datasource == 'LES':
        print(datafilepath)

        # set fill value
        fillvalue= -9999

        # load brighttness temperature data. automatically removes missing values
        in_lwp = np.loadtxt(datafilepath, dtype=float) 
        in_lwp = np.reshape(in_lwp, (ny, nx))

        #####################################################
        # mask brightness temperatures outside of normal range
        in_lwp[in_lwp < mintb_thresh] = fillvalue
        in_lwp[in_lwp > maxtb_thresh] = fillvalue

        #####################################################
        # determine geographic region of interest is within the data set. if it is proceed and limit the data to that geographic region. if not exit the code.

        # isolate data within lat/lon range set by limit
        indicesy, indicesx = np.array(np.where((in_lat > geolimits[0]) & (in_lat <= geolimits[2]) & (in_lon > geolimits[1]) & (in_lon <= geolimits[3])))

        # proceed if file covers the geographic region in interest
        if len(indicesx) > 0 and len(indicesy) > 0:
            out_lat = np.copy(in_lat[np.nanmin(indicesy):np.nanmax(indicesy), np.nanmin(indicesx):np.nanmax(indicesx)])
            out_lon = np.copy(in_lon[np.nanmin(indicesy):np.nanmax(indicesy), np.nanmin(indicesx):np.nanmax(indicesx)])
            out_lwp = np.copy(in_lwp[np.nanmin(indicesy):np.nanmax(indicesy), np.nanmin(indicesx):np.nanmax(indicesx)])

            ######################################################
            # proceed only if number of missing data does not exceed an accepable threshold
            # determine number of missing data
            missingcount = len(np.array(np.where(np.isnan(out_lwp)))[0,:])
            ny, nx = np.shape(out_lwp)

            if np.divide(missingcount, (ny*nx)) < miss_thresh:
                ######################################################
                # call idclouds subroutine
                if cloudidmethod == 'futyan3':
                    from subroutine_idclouds import futyan3
                    clouddata = futyan3(out_lwp, pixel_radius, cloudtb_threshs, area_thresh, warmanvilexpansion)
                elif cloudidmethod == 'futyan4':
                    from subroutine_idclouds import futyan4_LES
                    clouddata = futyan4_LES(out_lwp, pixel_radius, cloudtb_threshs, area_thresh, mincoldcorepix, smoothsize, warmanvilexpansion)

                ######################################################
                # separate output from futyan into the separate variables
                final_nclouds = np.array([clouddata['final_nclouds']])
                final_ncorepix = np.array([clouddata['final_ncorepix']])
                final_ncoldpix = np.array([clouddata['final_ncoldpix']])
                final_ncorecoldpix = np.array([clouddata['final_ncorecoldpix']])
                final_nwarmpix = np.array([clouddata['final_nwarmpix']])
                final_cloudtype = np.array([clouddata['final_cloudtype']])
                final_cloudnumber = np.array([clouddata['final_cloudnumber']])
                final_convcold_cloudnumber = np.array([clouddata['final_convcold_cloudnumber']])

                #######################################################
                # output data to netcdf file, only if clouds present
                if final_nclouds > 0:
                    # create filename
                    cloudid_outfile = dataoutpath + datasource + '_' + datadescription + '_cloudid' + cloudid_version + '_' + datafiledatestring + '_' + datafiletimestring + '.nc'

                    # Check if file exists, if it does delete it
                    if os.path.isfile(cloudid_outfile):
                        os.remove(cloudid_outfile)

                    file_basetime = calendar.timegm(datetime.datetime(int(datafiledatestring[0:4]), int(datafiledatestring[4:6]), int(datafiledatestring[6:8]), int(datafiletimestring[0:2]), int(datafiletimestring[2:4])).timetuple())

                    # Define xarray dataset
                    output_data = xr.Dataset({'basetime': (['time'], np.array([file_basetime], dtype='datetime64[ns]')), \
                                              'filedate': (['time', 'ndatechar'],  np.array([stringtochar(np.array(datafiledatestring))])), \
                                              'filetime': (['time', 'ntimechar'], np.array([stringtochar(np.array(datafiletimestring))])), \
                                              'latitude': (['lat', 'lon'], out_lat), \
                                              'longitude': (['lat', 'lon'], out_lon), \
                                              'lwp': (['time', 'lat', 'lon'], np.expand_dims(out_lwp, axis=0)), \
                                              'cloudtype': (['time', 'lat', 'lon'], final_cloudtype), \
                                              'convcold_cloudnumber': (['time', 'lat', 'lon'], final_convcold_cloudnumber), \
                                              'cloudnumber': (['time', 'lat', 'lon'], final_cloudnumber), \
                                              'nclouds': (['time'], final_nclouds), \
                                              'ncorepix': (['time', 'clouds'], final_ncorepix), \
                                              'ncoldpix': (['time', 'clouds'], final_ncoldpix), \
                                              'ncorecoldpix': (['time', 'clouds'], final_ncorecoldpix), \
                                              'nwarmpix': (['time', 'clouds'], final_nwarmpix)}, \
                                             coords={'time': (['time'], np.array([file_basetime], dtype='datetime64[ns]')), \
                                                     'lat': (['lat'], np.squeeze(out_lat[:, 0])), \
                                                     'lon': (['lon'], np.squeeze(out_lon[0, :])), \
                                                     'clouds': (['clouds'],  np.arange(1, final_nclouds+1)), \
                                                     'ndatechar': (['ndatechar'], np.arange(0, 8)), \
                                                     'ntimechar': (['ntimechar'], np.arange(0, 4))}, \
                                             attrs={'title': 'Statistics about convective features identified in the data from ' + datafiledatestring[0:4] + '/' + datafiledatestring[4:6] + '/' + datafiledatestring[6:8] + ' ' + datafiletimestring[0:2] + ':' + datafiletimestring[2:4] + ' utc', \
                                                    'institution': 'Pacific Northwest National Laboratory', \
                                                    'convections': 'CF-1.6', \
                                                    'contact': 'Hannah C Barnes: hannah.barnes@pnnl.gov', \
                                                    'created_ok': time.ctime(time.time()), \
                                                    'cloudid_cloud_version': cloudid_version, \
                                                    'lwp_threshold_core':  str(int(cloudtb_threshs[0])) + 'K', \
                                                    'lwp_threshold_coldanvil': str(int(cloudtb_threshs[1])) + 'K', \
                                                    'lwp_threshold_warmanvil': str(int(cloudtb_threshs[2])) + 'K', \
                                                    'lwp_threshold_environment': str(int(cloudtb_threshs[3])) + 'K', \
                                                    'minimum_cloud_area': str(int(area_thresh)) + 'km^2'})

                    # Specify variable attributes
                    output_data.time.attrs['long_name'] = 'epoch time (seconds since 01/01/1970 00:00) in epoch of file'

                    output_data.lat.attrs['long_name'] = 'Vector of latitudes, y-coordinate in Cartesian system'
                    output_data.lat.attrs['standard_name'] = 'latitude'
                    output_data.lat.attrs['units'] = 'degrees_north'
                    output_data.lat.attrs['valid_min'] = geolimits[0]
                    output_data.lat.attrs['valid_max'] = geolimits[2]

                    output_data.lon.attrs['long_name'] = 'Vector of longitudes, x-coordinate in Cartesian system'
                    output_data.lon.attrs['standard_name'] = 'longitude'
                    output_data.lon.attrs['units'] = 'degrees_east'
                    output_data.lon.attrs['valid_min'] = geolimits[1]
                    output_data.lon.attrs['valid_max'] = geolimits[2]

                    output_data.clouds.attrs['long_name'] = 'number of distict convective cores identified'
                    output_data.clouds.attrs['units'] = 'unitless'

                    output_data.ndatechar.attrs['long_name'] = 'number of characters in date string'
                    output_data.ndatechar.attrs['units'] = 'unitless'

                    output_data.ntimechar.attrs['long_name'] = 'number of characters in time string'
                    output_data.ntimechar.attrs['units'] = 'unitless'

                    output_data.basetime.attrs['long_name'] = 'epoch time (seconds since 01/01/1970 00:00) of file'
                    output_data.basetime.attrs['standard_name'] = 'time'

                    output_data.filedate.attrs['long_name'] = 'date string of file (yyyymmdd)'
                    output_data.filedate.attrs['units'] = 'unitless'

                    output_data.filetime.attrs['long_name'] = 'time string of file (hhmm)'
                    output_data.filetime.attrs['units'] = 'unitless'

                    output_data.latitude.attrs['long_name'] = 'cartesian grid of latitude'
                    output_data.latitude.attrs['units'] = 'degrees_north'
                    output_data.latitude.attrs['valid_min'] = geolimits[0]
                    output_data.latitude.attrs['valid_max'] = geolimits[2]

                    output_data.longitude.attrs['long_name'] = 'cartesian grid of longitude'
                    output_data.longitude.attrs['units'] = 'degrees_east'
                    output_data.longitude.attrs['valid_min'] = geolimits[1]
                    output_data.longitude.attrs['valid_max'] = geolimits[3]

                    output_data.lwp.attrs['long_name'] = 'liquid water path'
                    output_data.lwp.attrs['units'] = 'g/m^3 (?)'
                    output_data.lwp.attrs['valid_min'] = mintb_thresh
                    output_data.lwp.attrs['valid_max'] = maxtb_thresh

                    output_data.cloudtype.attrs['long_name'] = 'grid of cloud classifications'
                    output_data.cloudtype.attrs['values'] = '1 = core, 2 = cold anvil, 3 = warm anvil, 4 = other'
                    output_data.cloudtype.attrs['units'] = 'unitless'
                    output_data.cloudtype.attrs['valid_min'] = 1
                    output_data.cloudtype.attrs['valid_max'] = 5

                    output_data.convcold_cloudnumber.attrs['long_name'] = 'grid with each classified cloud given a number'
                    output_data.convcold_cloudnumber.attrs['units'] = 'unitless'
                    output_data.convcold_cloudnumber.attrs['valid_min'] = 0
                    output_data.convcold_cloudnumber.attrs['valid_max'] = final_nclouds+1
                    output_data.convcold_cloudnumber.attrs['comment'] = 'extend of each cloud defined using cold anvil threshold'

                    output_data.cloudnumber.attrs['long_name'] = 'grid with each classified cloud given a number'
                    output_data.cloudnumber.attrs['units'] = 'unitless'
                    output_data.cloudnumber.attrs['valid_min'] = 0
                    output_data.cloudnumber.attrs['valid_max'] = final_nclouds+1
                    output_data.cloudnumber.attrs['comment'] = 'extend of each cloud defined using warm anvil threshold'

                    output_data.nclouds.attrs['long_name'] = 'number of distict convective cores identified in file'
                    output_data.nclouds.attrs['units'] = 'unitless'

                    output_data.ncorepix.attrs['long_name'] = 'number of convective core pixels in each cloud feature'
                    output_data.ncorepix.attrs['units'] = 'unitless'

                    output_data.ncoldpix.attrs['long_name'] = 'number of cold anvil pixels in each cloud feature'
                    output_data.ncoldpix.attrs['units'] = 'unitless'

                    output_data.ncorecoldpix.attrs['long_name'] = 'number of convective core and cold anvil pixels in each cloud feature'
                    output_data.ncorecoldpix.attrs['units'] = 'unitless'

                    output_data.nwarmpix.attrs['long_name'] = 'number of warm anvil pixels in each cloud feature'
                    output_data.nwarmpix.attrs['units'] = 'unitless'

                    # Write netCDF file
                    print(cloudid_outfile)
                    print('')

                    output_data.to_netcdf(path=cloudid_outfile, mode='w', format='NETCDF4_CLASSIC', unlimited_dims='time', \
                                          encoding={'time': {'zlib':True, '_FillValue': fillvalue}, \
                                                    'lon': {'zlib':True, '_FillValue': fillvalue}, \
                                                    'lon': {'zlib':True, '_FillValue': fillvalue}, \
                                                    'clouds': {'zlib':True, '_FillValue': fillvalue}, \
                                                    'basetime': {'zlib':True, '_FillValue': fillvalue}, \
                                                    'filedate': {'dtype': 'str', 'zlib':True, '_FillValue': fillvalue}, \
                                                    'filetime': {'dtype': 'str', 'zlib':True, '_FillValue': fillvalue}, \
                                                    'longitude': {'zlib':True, '_FillValue': fillvalue}, \
                                                    'latitude': {'zlib':True, '_FillValue': fillvalue}, \
                                                    'lwp': {'zlib':True, '_FillValue': fillvalue}, \
                                                    'cloudtype': {'zlib':True, '_FillValue': fillvalue}, \
                                                    'convcold_cloudnumber': {'dtype': 'int', 'zlib':True, '_FillValue': fillvalue}, \
                                                    'cloudnumber': {'dtype': 'int', 'zlib':True, '_FillValue': fillvalue}, \
                                                    'nclouds': {'dtype': 'int', 'zlib':True, '_FillValue': fillvalue},  \
                                                    'ncorepix': {'dtype': 'int', 'zlib':True, '_FillValue': fillvalue},  \
                                                    'ncoldpix': {'dtype': 'int', 'zlib':True, '_FillValue': fillvalue}, \
                                                    'ncorecoldpix': {'dtype': 'int', 'zlib':True, '_FillValue': fillvalue}, \
                                                    'nwarmpix': {'dtype': 'int', 'zlib':True, '_FillValue': fillvalue}})

                else:
                    print(datafilepath)
                    print('No clouds')

            else:
                print(datafilepath)
                print('To much missing data')
        else:
            print(datafilepath)
            print('data not within latitude, longitude range. check specified geographic range')

                
                







\
