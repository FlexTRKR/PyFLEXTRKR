#!/usr/bin/env python
import math
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
import xarray as xr
from netCDF4 import Dataset
import sys, os


def infill_file(filename):
    ''' Infill a file based on mps threshold

    This script interpolates high values of mps for storm tracking. 
    It writes the file back to a subdirectory labeled filtered.
    '''

    trackname = 'tracks'  # default
    #trackname = 'ntracks'

    timename = 'times'  #  default
    #timename = 'nmaxlength'

    dirname, fname = os.path.split(filename)
    if dirname == '':
        dirname='./'
    # Create output directory
    outdirname = dirname + '/filtered/'
    os.makedirs(outdirname, exist_ok=True)

    dset = xr.open_dataset(filename, drop_variables=['ntracks','nmaxlength']).load()
    move_r = dset['movement_r_meters_per_second']
    mask_r = move_r < 50
    mask_nan = np.logical_not(np.isnan(move_r))
    total_mask = np.logical_and(mask_r, mask_nan)

    fill_value = np.nan

    nr = dset['movement_r_meters_per_second'].values
    ntheta = dset['movement_theta'].values
    mx = dset['movement_storm_x'].values
    my = dset['movement_storm_y'].values

    for track in np.arange(0, len(dset[trackname])):                      
        if np.count_nonzero(dset['movement_r_meters_per_second'][track] < 50)< 3:
            print('Not enough values in Track %d' % track)
            nr[track]=np.nan
            ntheta[track]=np.nan
            mx[track]=np.nan
            my[track]=np.nan
        else:
            x = dset[timename][total_mask[track]]
            r = dset['movement_r_meters_per_second'][track][total_mask[track]]
            theta = dset['movement_theta'][track][total_mask[track]]
            intp_r = interp1d(x, r, kind='quadratic', fill_value=fill_value, bounds_error=False)
            intp_theta = interp1d(x, theta, kind='quadratic', fill_value=fill_value, bounds_error=False)
            mov_x = 3.6 * intp_r(dset[timename][mask_nan[track]]) * np.cos(np.pi/180.0*intp_theta(dset[timename][mask_nan[track]]))
            mov_y = 3.6 * intp_r(dset[timename][mask_nan[track]]) * np.sin(np.pi/180.0*intp_theta(dset[timename][mask_nan[track]]))

            nr[track][mask_nan[track]] = intp_r(dset[timename][mask_nan[track]])
            ntheta[track][mask_nan[track]] = intp_theta(dset[timename][mask_nan[track]])
            mx[track][mask_nan[track]] = mov_x
            my[track][mask_nan[track]] = mov_y

            
    dset['movement_r_meters_per_second'] = ((trackname, timename), nr)
    dset['movement_theta'] = ((trackname, timename), ntheta)
    dset['movement_storm_x'] = ((trackname, timename), mx)
    dset['movement_storm_y'] = ((trackname, timename), my)

#    dset.to_netcdf(dirname + '/filtered/' + fname)
    dset.to_netcdf(outdirname + fname)


if __name__ == '__main__': 
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        print(f'Processing {filename}')
        infill_file(filename)
    else:
        print('Need filename to process')

