import glob, os, sys
import numpy as np
import xarray as xr
import logging
import time
import dask
from dask.distributed import Client, LocalCluster, wait
from pyflextrkr.sl3d_func import gridrad_sl3d
from pyflextrkr.ft_utilities import load_config
from pyflextrkr.echotop_func import echotop_height

#--------------------------------------------------------------------------------------------------------
def write_output_file(out_file, data_dict, config):
    """
    Write output to netCDF file

    Args:
        out_file: string
            Output file name
        data_dict: dictionary
            Dictionary containing output variables
        config: dictionary
            Dictionary containing config parameters

    Returns:
        out_file: string
            Output file name
    """

    tb_varname = config['tb_varname']
    pcp_varname = config['pcp_varname']

    # Define xarray dataset
    var_dict = {
        'latitude': (['lon', 'lat'], data_dict['latitude']),
        'longitude': (['lon', 'lat'], data_dict['longitude']),
        'dbz_lowlevel': (['time', 'lon', 'lat'], np.expand_dims(data_dict['dbz_lowlevel'], axis=0)),
        'dbz_comp': (['time', 'lon', 'lat'], np.expand_dims(data_dict['dbz_comp'], axis=0)),
        'sl3d': (['time', 'lon', 'lat'], np.expand_dims(data_dict['sl3d'], axis=0)),
        'echotop_10dbz': (['time', 'lon', 'lat'], np.expand_dims(data_dict['echotop10'], axis=0)),
        'echotop_20dbz': (['time', 'lon', 'lat'], np.expand_dims(data_dict['echotop20'], axis=0)),
        'echotop_30dbz': (['time', 'lon', 'lat'], np.expand_dims(data_dict['echotop30'], axis=0)),
        'echotop_40dbz': (['time', 'lon', 'lat'], np.expand_dims(data_dict['echotop40'], axis=0)),
        'echotop_45dbz': (['time', 'lon', 'lat'], np.expand_dims(data_dict['echotop45'], axis=0)),
        'echotop_50dbz': (['time', 'lon', 'lat'], np.expand_dims(data_dict['echotop50'], axis=0)),
        tb_varname: (['time', 'lon', 'lat'], np.expand_dims(data_dict[tb_varname].data, axis=0)),
        pcp_varname: (['time', 'lon', 'lat'], np.expand_dims(data_dict[pcp_varname].data, axis=0)),
    }
    coord_dict = {
        'lat': (['lat'], np.squeeze(data_dict['latitude'][:, 0])),
        'lon': (['lon'], np.squeeze(data_dict['longitude'][0, :])),
    }
    gattr_dict = {
        'title': 'SL3D classification',
        'Analysis_time': data_dict['Analysis_time'],
        'contact': 'Zhe Feng: zhe.feng@pnnl.gov',
        'created_on': time.ctime(time.time()),
    }
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

    # Specify attributes
    dsout['longitude'].attrs['long_name'] = 'Longitude'
    dsout['longitude'].attrs['units'] = 'degree East'
    dsout['latitude'].attrs['long_name'] = 'Latitude'
    dsout['latitude'].attrs['units'] = 'degree North'
    dsout['dbz_lowlevel'].attrs['long_name'] = f'Low-level reflectivity ({data_dict["dbz_lowlevel_asl"]:.1f} km)'
    dsout['dbz_lowlevel'].attrs['units'] = 'dBZ'
    dsout['dbz_comp'].attrs['long_name'] = 'Composite reflectivity'
    dsout['dbz_comp'].attrs['units'] = 'dBZ'
    sl3d_label_comments = '0:NoEcho, 1:ConvectiveUpdraft, 2:Convection, ' + \
        '3:PrecipitatingStratiform, 4:Non-PrecipitatingStratiform, 5:Anvil'
    dsout['sl3d'].attrs['long_name'] = 'sl3d category'
    dsout['sl3d'].attrs['units'] = 'unitless'
    dsout['sl3d'].attrs['comment'] = sl3d_label_comments
    dsout['echotop_10dbz'].attrs['long_name'] = '10 dBZ echo top height'
    dsout['echotop_10dbz'].attrs['units'] = 'km'
    dsout['echotop_20dbz'].attrs['long_name'] = '20 dBZ echo top height'
    dsout['echotop_20dbz'].attrs['units'] = 'km'
    dsout['echotop_30dbz'].attrs['long_name'] = '30 dBZ echo top height'
    dsout['echotop_30dbz'].attrs['units'] = 'km'
    dsout['echotop_40dbz'].attrs['long_name'] = '40 dBZ echo top height'
    dsout['echotop_40dbz'].attrs['units'] = 'km'
    dsout['echotop_45dbz'].attrs['long_name'] = '45 dBZ echo top height'
    dsout['echotop_45dbz'].attrs['units'] = 'km'
    dsout['echotop_50dbz'].attrs['long_name'] = '50 dBZ echo top height'
    dsout['echotop_50dbz'].attrs['units'] = 'km'
    dsout[tb_varname].attrs = data_dict[tb_varname].attrs
    dsout[pcp_varname].attrs = data_dict[pcp_varname].attrs
    # Write output to file
    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in dsout.data_vars}
    dsout.to_netcdf(path=out_file, mode='w', format='NETCDF4', unlimited_dims='time', encoding=encoding)
    logger.info(f'Output: {out_file}')
    return out_file


#--------------------------------------------------------------------------------------------------------
def process_file(infilename, config):
    """
    Process a 3D radar file to get 2D variables

    Args:
        infilename: string
            Input file name
        config: dictionary
            Dictionary containing config parameters

    Returns:
        output_file: string
            Output file name
    """

    # print(source_nexrad, source_mltlvl)
    outdir = config['clouddata_path']
    outbasename = config['databasename']
    t_dimname = config['t_dimname']
    x_dimname = config['x_dimname']
    y_dimname = config['y_dimname']
    z_dimname = config['z_dimname']
    x_varname = config['x_varname']
    y_varname = config['y_varname']
    z_varname = config['z_varname']
    reflectivity_varname = config['reflectivity_varname']
    meltlevel_varname = config['meltlevel_varname']
    tb_varname = config['tb_varname']
    pcp_varname = config['pcp_varname']
    echotop_gap = config.get('echotop_gap', 0)
    dbz_lowlevel_asl = config.get('dbz_lowlevel_asl', 2.0)

    # Read input data
    ds = xr.open_dataset(infilename)
    # Reorder the dimensions using dimension names to [time, z, y, x]
    ds = ds.transpose(t_dimname, z_dimname, y_dimname, x_dimname)
    # Get data dimensions
    nx = ds.sizes[x_dimname]
    ny = ds.sizes[y_dimname]
    nz = ds.sizes[z_dimname]
    # Get data coordinates
    lon2d = ds[x_varname].data
    lat2d = ds[y_varname].data
    height = ds[z_varname].data
    # Get data time
    Analysis_time = ds['time'].dt.strftime('%Y-%m-%dT%H:%M:%S').item()
    Analysis_month = ds['time'].dt.strftime('%m').item()
    # Get data variables
    refl3d = ds[reflectivity_varname].squeeze()
    reflArray = refl3d.data
    meltinglevelheight = ds[meltlevel_varname].squeeze().data
    tb = ds[tb_varname].squeeze()
    pcp = ds[pcp_varname].squeeze()
    # Make variables to mimic GridRad data
    Nradobs = np.full(reflArray.shape, 10, dtype=int)
    Nradecho = np.full(reflArray.shape, 10, dtype=int)

    # Get low-level reflectivity
    idx_low = np.argmin(np.abs(height - dbz_lowlevel_asl))
    dbz_lowlevel = reflArray[idx_low,:,:]
    # Get column-maximum reflectivity (composite)
    dbz_comp = refl3d.max(dim=z_dimname)

    # Replace missing_val with NaN
    missing_val = -9999.
    reflArray[reflArray == missing_val] = np.NaN

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
        'Analysis_month' : Analysis_month,
    }

    # Perform SL3D classification
    sl3d = gridrad_sl3d(data, config, zmelt=meltinglevelheight)

    # Calculate echo-top heights for various reflectivity thresholds
    shape_2d = sl3d.shape
    echotop10 = echotop_height(refl3d, height, z_dimname, shape_2d,
                                dbz_thresh=10, gap=echotop_gap, min_thick=0)
    echotop20 = echotop_height(refl3d, height, z_dimname, shape_2d,
                                dbz_thresh=20, gap=echotop_gap, min_thick=0)
    echotop30 = echotop_height(refl3d, height, z_dimname, shape_2d,
                                dbz_thresh=30, gap=echotop_gap, min_thick=0)
    echotop40 = echotop_height(refl3d, height, z_dimname, shape_2d,
                                dbz_thresh=40, gap=echotop_gap, min_thick=0)
    echotop45 = echotop_height(refl3d, height, z_dimname, shape_2d,
                                dbz_thresh=45, gap=echotop_gap, min_thick=0)
    echotop50 = echotop_height(refl3d, height, z_dimname, shape_2d,
                                dbz_thresh=50, gap=echotop_gap, min_thick=0)

    data_dict= {
        'latitude': lat2d,
        'longitude': lon2d,
        'dbz_lowlevel': dbz_lowlevel,
        'dbz_comp': dbz_comp,
        'sl3d': sl3d,
        'echotop10': echotop10,
        'echotop20': echotop20,
        'echotop30': echotop30,
        'echotop40': echotop40,
        'echotop45': echotop45,
        'echotop50': echotop50,
        tb_varname: tb,
        pcp_varname: pcp,
        'Analysis_time': Analysis_time,
        'dbz_lowlevel_asl': dbz_lowlevel_asl,
    }

    # Write output to netCDF file
    output_datetime = ds['time'].dt.strftime('%Y-%m-%d_%H:%M:%S').item()
    output_file = f'{outdir}{outbasename}{output_datetime}.nc'
    result = write_output_file(output_file, data_dict, config)

    # import pdb; pdb.set_trace()
    return output_file


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
    inbasename = config['regrid_basename']

    # Find all WRF files
    filelist = sorted(glob.glob(f'{indir}/{inbasename}*'))
    nfiles = len(filelist)
    logger.info(f'Number of files: {nfiles}')

    if config['run_parallel'] == 1:
        # Set Dask temporary directory for workers
        dask_tmp_dir = config.get("dask_tmp_dir", "./")
        dask.config.set({'temporary-directory': dask_tmp_dir})
        # Local cluster
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
        client = Client(cluster)

    # Serial
    if run_parallel == 0:
        for ifile in range(0, nfiles):
            result = process_file(filelist[ifile], config)
    # Parallel
    elif run_parallel >= 1:
        results = []
        for ifile in range(0, nfiles):
            result = dask.delayed(process_file)(filelist[ifile], config)
            results.append(result)
        final_result = dask.compute(*results)
        wait(final_result)
    else:
        sys.exit('Valid parallelization flag not provided')
