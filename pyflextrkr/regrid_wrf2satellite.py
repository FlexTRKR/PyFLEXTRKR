# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import xesmf as xe
import numpy as np
from netCDF4 import Dataset as ncfile
satinput = ncfile("/global/project/projectdirs/encon/smhagos/HVMIXING/OBS/TB/merg_2014042508_4km-pixel.nc")
latout= satinput.variables["lat"][:]
lonout= satinput.variables["lon"][:]
dirin = '/global/project/projectdirs/m1657/chen696/WRF_Amazon_SH/CTL4KMRUN01/INPUT_DIR/'
dirout = '/global/cfs/projectdirs/encon/smhagos/HVMIXING/CTL1KMRUN01REG/'

latlonfile = ncfile('/global/project/projectdirs/m1657/chen696/WRF_Amazon_SH/CTL4KMRUN01/INPUT_DIR/wrfout_rainrate_tb_2014-04-07_11:00:00.nc')
latin= latlonfile.variables["lat2d"][:,0]
lonin= latlonfile.variables["lon2d"][0,:]

ilat = np.where(np.abs(latout-np.min(latin))==np.min(np.abs(latout-np.min(latin))))[0][0]
flat = np.where(np.abs(latout-np.max(latin))==np.min(np.abs(latout-np.max(latin))))[0][0]
ilon = np.where(np.abs(lonout-np.min(lonin))==np.min(np.abs(lonout-np.min(lonin))))[0][0]
flon = np.where(np.abs(lonout-np.max(lonin))==np.min(np.abs(lonout-np.max(lonin))))[0][0]

latout = latout[ilat:flat]
lonout = lonout[ilon:flon]
regridder = xe.Regridder(grid_in, grid_out, "bilinear")

def regrid(filein):
    dirin = '/global/project/projectdirs/m1657/chen696/WRF_Amazon_SH/CTL4KMRUN01/INPUT_DIR/'
    dirout = '/global/cfs/projectdirs/encon/smhagos/HVMIXING/CTL1KMRUN01REG/'  
    wrfinput = ncfile(dirin+filein)
    rainrate= wrfinput.variables["rainrate"][:,:]
    tb= wrfinput.variables["tb"][:]
    lat2d = wrfinput.variables["lat2d"][:]
    lon2d = wrfinput.variables["lon2d"][:]
    timeout = wrfinput.variables["time"][:]
    grid_in = {"lon": lonin, "lat": latin}
    grid_out = {"lon": lonout, "lat": latout}
#    regridder = xe.Regridder(grid_in, grid_out, "bilinear")
    rainrate= wrfinput.variables["rainrate"][:,:]
    tb= wrfinput.variables["tb"][:]
    tb.fill_value = 0
    rainrate.fill_value = 0
    lat2d.fill_value =0
    lon2d.fill_value=0
    rainrateout= regridder(rainrate)
    tbout = regridder(tb)
    lat2dout = regridder(lat2d)
    lon2dout = regridder(lon2d)
    fileout =filein[0:len(filein)-3]+'_reg.nc'
    fileout = ncfile(dirout+fileout, 'w', format='NETCDF3_64BIT')
    fileout.createDimension('time', 1)
    fileout.createDimension('lat', len(latout))
    fileout.createDimension('lon', len(lonout))
    tb  = fileout.createVariable('tb', 'f4', ('time','lat','lon'))
    rainrate  = fileout.createVariable('rainrate', 'f4', ('time','lat','lon'))
    lat2d  = fileout.createVariable('lat2d', 'f4', ('lat','lon'))
    lon2d  = fileout.createVariable('lon2d', 'f4', ('lat','lon'))
    lat = fileout.createVariable('lat', 'f4', ('lat'))
    lon = fileout.createVariable('lon', 'f4', ('lon'))
    time = fileout.createVariable('time', 'f4', ('time'))
    tb[:] = tbout
    rainrate[:] = rainrateout
    lat2d[:] = lat2dout
    lon2d[:] = lon2dout
    time[:] = timeout
    lat[:] = latout
    lon[:] = lonout
    fileout.close()
    print(filein," done")
    return