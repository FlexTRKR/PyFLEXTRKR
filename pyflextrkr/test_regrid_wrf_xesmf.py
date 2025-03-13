import xarray as xr
import xesmf as xe

if __name__ == '__main__':

    ingrid_file = '/global/cfs/projectdirs/encon/smhagos/HVMIXING/CTL4KMRUN01/wrfout_d01_2014-04-01_00:00:00'
    outgrid_file = '/global/project/projectdirs/encon/smhagos/HVMIXING/OBS/TB/merg_2014042508_4km-pixel.nc'

    ds_in = xr.open_dataset(ingrid_file)
    lon_src = ds_in['XLONG'].squeeze()
    lat_src = ds_in['XLAT'].squeeze()
    grid_src = xe.util.grid_from_dataset(lon_src, lat_src)

    ds_in = ds_in.rename({"XLONG": "lon", "XLAT": "lat"})

    ds_out = xr.open_dataset(outgrid_file)


    import pdb; pdb.set_trace()