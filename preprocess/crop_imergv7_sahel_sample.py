#!/usr/bin/env python
r"""
Make the sample data of the Sahel Tb + IMERG V07 demo (config/demo_mcs_imergv7_sahel.sh, tar file gpm_tb_imergv7_sahel.tar.gz): hourly files of
brightness temperature and precipitation over the Sahel (0-30N, 20W-60E, 0.1 degree) for 22-23 August 2025, and the land-sea mask of the same box.

Where the data come from. The combined Tb + IMERG V07B files (/global/cfs/cdirs/wcm_shr/gsharing/MCS_Global/IR_IMERG_Combined_V07B/2025/merg_YYYYMMDDHH_10km-pixel.nc)
of this folder stop on 30 April 2025, so the hours of August are taken from the pixel-level files of the global MCS tracking V3 (Feng21), mcstrack_YYYYMMDD_HH3000.nc,
which hold the input Tb (`tb`) and precipitation (`precipitation`) on the same 0.1 degree grid. The new files have the structure of the raw files: the file name
merg_YYYYMMDDHH_10km-pixel.nc, the variables Tb and precipitation (time, lat, lon) with the attributes of the raw variables, the global attributes of the raw
files (the hour of the file set), and the attributes `description`, `source`, `Tb_gap_fill` and `history`. One time step per file (HH:30, what idclouds_minute 30 of
the config reads), 48 files for 2 days.

The Tb of the pixel-level files is the Tb after the pre-processing of the cloud identification (pyflextrkr/idclouds_tbpf.py): missing pixels are filled with a 5 x 5
median filter of the neighbouring Tb (scipy.signal.medfilt2d, medfiltsize of the config), pixels that stay missing are NaN, and Tb outside 160-330 K is NaN. The
precipitation is not changed. Check on the global files of 1 January 2025, 00:30 UTC, which exist in both forms: precipitation identical; Tb identical at every
pixel where the raw Tb is valid; 5264 pixels of the raw Tb are missing, 1815 in the pixel-level file.

Usage (a compute node is not needed, 48 small files):
    python crop_imergv7_sahel_sample.py /path/to/work/sahel_sample --tar /path/to/gpm_tb_imergv7_sahel.tar.gz
    python crop_imergv7_sahel_sample.py /path/to/work/sahel_sample --start 2025-08-22T00 --end 2025-08-23T23 --lat_range 0 30 --lon_range -20 60
The output folder gets the 48 files and IMERG_landmask_sahel.nc; --tar also makes the tar file (the land-sea mask first, the files at the top level, like gpm_tb_imerg.tar.gz).
"""
import argparse
import datetime
import os
import tarfile

import numpy as np
import pandas as pd
import xarray as xr

PIXEL_DIR = '/global/cfs/cdirs/wcm_shr/gsharing/MCS_Global/V3/Feng21/mcstracking_f21/20250101.0000_20251001.0000/'
RAW_TEMPLATE = '/global/cfs/cdirs/wcm_shr/gsharing/MCS_Global/IR_IMERG_Combined_V07B/2025/merg_2025010100_10km-pixel.nc'
LANDMASK = '/global/cfs/cdirs/wcm_shr/gsharing/MCS_Global/V3/map_data/IMERG_landmask_180W-180E_60S-60N.nc'
SCRIPT = 'preprocess/crop_imergv7_sahel_sample.py (PyFLEXTRKR)'

TB_GAP_FILL = ("Tb is the field `tb` of the PyFLEXTRKR pixel-level tracking file of the global MCS tracking V3, i.e. the input Tb after the pre-processing of the cloud "
               "identification: missing pixels were filled with a 5 x 5 median filter of the neighbouring Tb (scipy.signal.medfilt2d, medfiltsize 5), pixels that stay "
               "missing are NaN, and Tb outside 160-330 K is NaN. In a check on the global files of 2025-01-01 00:30 UTC (raw V07B file and pixel-level file) the "
               "Tb is identical at every pixel where the raw Tb is valid; the raw Tb has 5264 missing pixels, the pre-filled field 1815. Precipitation is not changed.")


def encoding(ds, shape):
    """Compressed netCDF like the raw files: float32 with NaN fill, zlib level 4, time as days since 1970-01-01."""
    enc = {v: dict(dtype='float32', zlib=True, complevel=4, shuffle=True, _FillValue=np.float32('nan'), chunksizes=(1,) + shape) for v in ('Tb', 'precipitation')}
    enc['time'] = dict(dtype='float64', units='days since 1970-01-01 00:00:00')
    return enc


def hour_file(hour, template, args):
    """The cropped file of one hour (the :30 time step): xarray Dataset with the structure and attributes of the raw V07B file."""
    pixel_file = f'{args.pixel_dir}mcstrack_{hour:%Y%m%d_%H}3000.nc'
    with xr.open_dataset(pixel_file) as pix:
        box = pix[['tb', 'precipitation']].sel(lat=slice(*args.lat_range), lon=slice(*args.lon_range)).load()
    ds = xr.Dataset({'Tb': box['tb'].astype('float32'), 'precipitation': box['precipitation'].astype('float32')},
                    coords={'time': box['time'], 'lat': box['lat'], 'lon': box['lon']})
    ds['Tb'].attrs = dict(template['Tb'].attrs)
    ds['precipitation'].attrs = dict(template['precipitation'].attrs)
    ds['time'].attrs = dict(template['time'].attrs)
    ds['lat'].attrs = dict(template['lat'].attrs)
    ds['lon'].attrs = dict(template['lon'].attrs)
    # the global attributes of the raw files, the ones that name the hour set for this hour; the production time and the temporary input file of the raw
    # files are not true for these files, and the history is written again
    attrs = {k: v for k, v in template.attrs.items() if k not in ('ProductionTime', 'input_file', 'history')}
    day, hh = f'{hour:%Y-%m-%d}', f'{hour:%H}'
    attrs.update(BeginDate=day, BeginTime=f'{hh}:00:00.000Z', EndDate=day, EndTime=f'{hh}:59:59.999Z',
                 FileHeader=f'StartGranuleDateTime={day}T{hh}:00:00.000Z;\nStopGranuleDateTime={day}T{hh}:59:59.999Z',
                 InputPointer=f'merg_{hour:%Y%m%d%H}_4km-pixel')
    attrs['description'] = (f"Brightness temperature (Tb) and precipitation of the combined GPM IMERG V07B / merged IR dataset on a 0.1 degree grid, {hour:%Y-%m-%d %H}:30 UTC, "
                            f"cropped to {args.lat_range[0]:g}-{args.lat_range[1]:g}N, {-args.lon_range[0]:g}W-{args.lon_range[1]:g}E (Sahel): sample data of the PyFLEXTRKR demo "
                            f"demo_mcs_imergv7_sahel.sh. Same structure as the files of IR_IMERG_Combined_V07B/2025/merg_YYYYMMDDHH_10km-pixel.nc, with one time step.")
    attrs['source'] = f'{pixel_file} (fields tb and precipitation of the pixel-level files of the global MCS tracking V3, Feng21)'
    attrs['Tb_gap_fill'] = TB_GAP_FILL
    attrs['history'] = f'{datetime.datetime.now():%a %b %d %H:%M:%S %Y}: tb renamed Tb and cropped from {pixel_file} by {SCRIPT}'
    ds.attrs = attrs
    return ds


def landmask_file(args):
    with xr.open_dataset(LANDMASK) as lm:
        ds = lm.sel(lat=slice(*args.lat_range), lon=slice(*args.lon_range)).load()
    ds.attrs['description'] = (f"Land-sea mask of the 0.1 degree IMERG grid (0-100: land fraction in percent), cropped from {LANDMASK} to "
                               f"{args.lat_range[0]:g}-{args.lat_range[1]:g}N, {-args.lon_range[0]:g}W-{args.lon_range[1]:g}E (Sahel) by {SCRIPT}: sample data of the "
                               f"PyFLEXTRKR demo demo_mcs_imergv7_sahel.sh.")
    return ds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('output_dir', help='folder for the 48 files and the land-sea mask (made if needed)')
    ap.add_argument('--start', default='2025-08-22T00', help='first hour (file merg_YYYYMMDDHH, time step HH:30), default 2025-08-22T00')
    ap.add_argument('--end', default='2025-08-23T23', help='last hour, default 2025-08-23T23')
    ap.add_argument('--lat_range', type=float, nargs=2, default=[0., 30.], metavar=('LAT_MIN', 'LAT_MAX'))
    ap.add_argument('--lon_range', type=float, nargs=2, default=[-20., 60.], metavar=('LON_MIN', 'LON_MAX'))
    ap.add_argument('--pixel_dir', default=PIXEL_DIR, help='folder of the pixel-level mcstrack_*.nc files')
    ap.add_argument('--tar', default=None, help='also make this tar.gz file (land-sea mask first, files at the top level)')
    args = ap.parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    with xr.open_dataset(RAW_TEMPLATE) as raw:
        template = raw.load()
    names = []
    # the land-sea mask
    lm = landmask_file(args)
    lm_path = os.path.join(args.output_dir, 'IMERG_landmask_sahel.nc')
    lm.to_netcdf(lm_path, encoding={'landseamask': dict(dtype='float32', zlib=True, complevel=4, shuffle=True, _FillValue=None)})
    names.append(lm_path)
    # the hourly files
    for hour in pd.date_range(args.start, args.end, freq='h'):
        ds = hour_file(hour, template, args)
        path = os.path.join(args.output_dir, f'merg_{hour:%Y%m%d%H}_10km-pixel.nc')
        ds.to_netcdf(path, encoding=encoding(ds, (ds.sizes['lat'], ds.sizes['lon'])))
        names.append(path)
        print(path, f'{os.path.getsize(path) / 1e6:.2f} MB', flush=True)
    if args.tar:
        with tarfile.open(args.tar, 'w:gz') as tar:
            for path in names:
                tar.add(path, arcname=os.path.basename(path))
        print(f'{args.tar}: {len(names)} files, {os.path.getsize(args.tar) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
