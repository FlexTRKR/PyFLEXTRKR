"""
Example script to map NEXRAD radar data to a Cartesian grid using PyART for cell tracking

Courtesy of Ye Liu (ye.liu@pnnl.gov)
"""
import numpy as np
import pyart
from glob import glob
from joblib import Parallel, delayed

def grid_pyart(filename, config, time=None):
    """
    Map radar data to a Cartesian grid and write to a netCDF file.

    Args:
        filename: string
            Input radar data file name
        config: dictionary
            Dictionary containing Cartesian grid parameters
        time: string
            String for output file name

    Returns:
        output_filename: string
            Output gridded radar file name

    """
    print(time)
    grid_shape = config['grid_shape']
    grid_limits = config['grid_limits']
    output_dir = config['output_dir']

    # Read intput data
    radar = pyart.io.read(filename)

    radar.fields['reflectivity']['data'][:, -10:] = np.ma.masked
    gatefilter = pyart.correct.GateFilter(radar)

    # Map radar to Cartesian grid
    # https://arm-doe.github.io/pyart/API/generated/pyart.map.grid_from_radars.html
    grid = pyart.map.grid_from_radars(
        (radar,), gatefilters=(gatefilter, ),
        grid_shape=grid_shape,
        grid_limits=grid_limits,
        **config['grid_params'],
    )

    # Write output to netCDF file
    # https://arm-doe.github.io/pyart/API/generated/pyart.io.write_grid.html
    # Note: set write_point_lon_lat_alt=True to write
    # point_longitude, point_latitude and point_altitude variables in the
    # output netCDF file. These variables are required for tracking.
    output_filename = f'{output_dir}/{time}.nc'
    pyart.io.write_grid(output_filename, grid,
        arm_alt_lat_lon_variables=True,
        write_point_x_y_z=True,
        write_point_lon_lat_alt=True,
    )
    return output_filename


if __name__ == '__main__':

    # Specify Cartesian grid parameters in config
    config = dict(
        grid_shape=(45, 441, 441),
        grid_limits=((0, 22000), (-110000.0, 110000.0), (-110000.0, 110000.0)),
        grid_params=dict(fields=['reflectivity']),
        output_dir='/data_directory/gridded/',
    )
    # Input data directory
    data_path = '/data_directory/'
    # Search input data files
    files = sorted(glob(f'{data_path}/KHGX*'))

    # Loop over each file to grid
#    for f in files: grid_pyart(f, config, time=f'{f[13:21]}.{f[22:28]}')
    Parallel(n_jobs=64)(delayed(grid_pyart)(f, config, time=f'{f[13:21]}.{f[22:28]}') for f in files)
