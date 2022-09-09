import glob, os, sys
import numpy as np
import xarray as xr
import logging
import calendar
import datetime
import time
import pytz
#from joblib import Parallel, delayed
#import pyproj
#import ipyparallel as ipp
# import netCDF4
# from netCDF4 import Dataset
# import gridrad_sl3d as gs
# import regrid
from multiprocessing import Pool
from pyflextrkr.sl3d_func import gridrad_sl3d
from pyflextrkr.ft_utilities import load_config

n_jobs=1

# os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
# os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

# Finish adding input variables to function
def write_output_file(out_file, data_dict):
    # Define xarray dataset
    var_dict = {
        'sl3d': (['lon', 'lat'], data_dict['sl3d']),
    }
    coord_dict = {
        'longitude': (['lon'], data_dict['dst_lon']),
        'latitude': (['lat'], data_dict['dst_lat']),
    }
    gattr_dict = {
        'title': 'SL3D classification',
        'Analysis_time': data_dict['analysis_time'],
        'contact': 'Zhe Feng: zhe.feng@pnnl.gov',
        'created_on': time.ctime(time.time()),
    }
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

    # Specify attributes
    dsout.longitude.attrs['long_name'] = 'Longitude'
    dsout.longitude.attrs['units'] = 'degree East'
    dsout.latitude.attrs['long_name'] = 'Latitude'
    dsout.latitude.attrs['units'] = 'degree North'
    dsout.sl3d.attrs['long_name'] = 'sl3d category'
    dsout.sl3d.attrs['units'] = 'unitless'
    dsout.sl3d.attrs['comment'] = '0:, 1:, 2:, 3:, 4:,'

    # Write output to file
    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in dsout.data_vars}
    dsout.to_netcdf(path=out_file, mode='w', format='NETCDF4', unlimited_dims='time', encoding=encoding)
                #  encoding={'sl3d': {'zlib': True, '_FillValue': -1, 'dtype': 'short'},
                #            'longitude': {'zlib': True},
                #            'latitude': {'zlib': True}})


# def process_file(source_nexrad, source_mltlvl):
def process_file(infilename, config):

    # print(source_nexrad, source_mltlvl)

    ds = xr.open_dataset(infilename)
    Analysis_time = ds['time'].dt.strftime('%Y-%m-%dT%H:%M:%S').item()
    # Get data dimensions
    nx = ds.sizes['lon']
    ny = ds.sizes['lat']
    nz = ds.sizes['level']
    # Get data coordinates
    height = ds['level'].data
    lon2d = ds['lon2d'].data
    lat2d = ds['lat2d'].data
    # Get data variables
    reflArray = ds['reflectivity'].squeeze().data
    meltinglevelheight = ds['meltinglevelheight'].squeeze().data
    # Make variables to mimic GridRad data
    Nradobs = np.full(reflArray.shape, 10, dtype=int)
    Nradecho = np.full(reflArray.shape, 10, dtype=int)

    # Replace missing_val with NaN
    missing_val = -9999.
    reflArray[reflArray == missing_val] = np.NaN

    # id = Dataset(source_nexrad, "r", format="NETCDF4")
    # Analysis_time           = str(id.getncattr('Analysis_time'          ))
    # height  = id.variables['height' ][:]
    # reflArraytmp  = id.variables['Reflectivity' ][0,:,:,:].transpose(2,1,0)
    # Nradobs  = id.variables['Nradobs' ][0,:,:,:].transpose(2,1,0)
    # Nradecho  = id.variables['Nradecho' ][0,:,:,:].transpose(2,1,0)
    # wvalues  = id.variables['wReflectivity' ]

    #i9999=np.where((reflArray == -9999.) | (reflArray > 100000000))
    #print(reflArray)
    # reflArraytmp2 = np.ma.filled(reflArraytmp.astype(float), np.nan)
    # reflArray = np.asarray(reflArraytmp2,dtype=np.float32)
    # del reflArraytmp,reflArraytmp2

    # i9999=np.where((reflArray == -9999.))
    # n9999=len(i9999[0])
    # if (n9999 > 0): reflArray[i9999]=np.nan

    x = {
        'values' : lon2d,
        'n' : nx,
    }
    y = {
        'values' : lat2d,  
        'n' : ny,
    }
    z = {
        'values' : height,
        'n' : nz,
    }
    Z_H = {
        'values' : reflArray,
        'missing' : np.NaN,
        # 'wvalues' : wvalues[0,:,:,:].transpose(2,1,0),
        # 'wmissing' : wvalues._FillValue,
    }
    data = {
        'x': x,
        'y': y,
        'z': z,
        'nobs' : Nradobs,
        'necho': Nradecho,
        'Z_H' : Z_H,
        'Analysis_time' : Analysis_time,
    }

    # id2 = Dataset(source_mltlvl, "r", format="NETCDF4")
    # ERAI_0Ctmp  = (id2.variables['ERAI_0C' ][0,:,:]).transpose(1,0)
    # del id2

    # ERAI_0C = np.ma.filled(ERAI_0Ctmp.astype(float), -2.)
    # del ERAI_0Ctmp

    #print(ERAI_0C.size)
    # import pdb; pdb.set_trace()

    sl3d = gridrad_sl3d(data, config, zmelt=meltinglevelheight)
    import pdb; pdb.set_trace()

    data_dict= {'sl3d': sl3d, \
            'dst_lat': dst_lat, \
            'dst_lon': dst_lon, \
            'analysis_time': Analysis_time}

    output_file=output_path+os.path.basename(source_nexrad)
    write_output_file(output_file,data_dict)

if __name__=='__main__':

    # Set the logging message level
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load configuration file
    config_file = sys.argv[1]
    config = load_config(config_file)
    # Get inputs from config
    run_parallel = config['run_parallel']
    n_workers = config['nprocesses']
    indir = config['clouddata_path']
    outdir = config['clouddata_path']
    inbasename = config['regrid_basename']
    outbasename = config['databasename']

    # Find all WRF files
    filelist = sorted(glob.glob(f'{indir}/{inbasename}*'))
    nfiles = len(filelist)
    logger.info(f'Number of files: {nfiles}')

    # if run_parallel == 0:
        # Serial version
        # for ifile in range(0, nfiles):
    for ifile in range(48, 49):
        result = process_file(filelist[ifile], config)

    import pdb; pdb.set_trace()