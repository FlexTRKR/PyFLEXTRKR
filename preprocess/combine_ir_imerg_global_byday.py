"""
Script to combine IR and IMERG data, average IMERG to hourly, and outputs netCDF file.
"""
__author__ = "Zhe.Feng@pnnl.gov"
__date__ = "19-Mar-2024"

import numpy as np
import xarray as xr
import os, sys, glob

def combine_ir_imerg(imergfiles, irfile, outfile):

    # Read the IR file
    dsir = xr.open_dataset(irfile, decode_times=False)

    # Read the IMERG files
    ds = xr.open_mfdataset(imergfiles, concat_dim='time', combine='nested').load()
    # Average the precipitation in time
    precipitation = ds[pcp_varname].mean(dim='time', keep_attrs=True)
    # Reorder the DataArray dimensions
    precipitation = precipitation.transpose('time', 'lat', 'lon', missing_dims='ignore')
    # Expand time dimension
    precipitation = precipitation.expand_dims(dim={'time':dsir.time})

    # Define output var list
    var_dict = {
        tb_varname: (['time', 'lat', 'lon'], dsir[tb_varname].data, dsir[tb_varname].attrs), \
        pcp_varname: (['time', 'lat', 'lon'], precipitation.data, precipitation.attrs)
    }
    # Coordinate list
    coord_dict = {
        'lon': (['lon'], dsir.lon.data), \
        'lat': (['lat'], dsir.lat.data), \
        'time': (['time'], dsir.time.data), \
    }
    # Global attribute list (copy from IR file)
    gattr_dict = dsir.attrs

    # Define output dataset
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

    # Copy IR variable attributes
    dsout['time'].attrs = dsir['time'].attrs
    dsout['lon'].attrs = dsir['lon'].attrs
    dsout['lat'].attrs = dsir['lat'].attrs
    # Add precipitation attribute
    dsout[pcp_varname].attrs['long_name'] = pcp_varname
    # dsout[pcp_varname].attrs['units'] = 'mm/hr'

    # Subset region
    dsout = dsout.sel({'lat':slice(min(lat_bounds), max(lat_bounds))})

    fillvalue = np.nan
    # Set encoding/compression for all variables
    comp = dict(zlib=True, _FillValue=fillvalue, dtype='float32')
    encoding = {var: comp for var in dsout.data_vars}

    # Write to netCDF file
    dsout.to_netcdf(path=outfile, mode='w', format='NETCDF4', unlimited_dims='time', encoding=encoding)
    print(f'Output saved: {outfile}')


if __name__ == "__main__":
    # Get the date from input (format: '20130101')
    date = sys.argv[1]
    # Get the Phase ('Summer' or 'Winter')
    Phase = sys.argv[2]

    # Get year from the filename
    year = date[0:4]

    # irdir = f'/pscratch/sd/f/feng045/waccem/MERGIR_Global/Regrid/{year}/'
    # imergdir = f'/pscratch/sd/f/feng045/waccem/IMERG_Global_V06B/{year}/'
    # outdir = f'/pscratch/sd/f/feng045/waccem/IR_IMERG_Combined/{year}/'
    # Phase = 'Summer'
    irdir = f'/pscratch/sd/f/feng045/DYAMOND/GPM_DYAMOND/GPM_MERGIR_V1_regrid/{Phase}/'
    imergdir = f'/pscratch/sd/f/feng045/DYAMOND/GPM_DYAMOND/GPM_3IMERGHH_V07B/{Phase}/'
    outdir = f'/pscratch/sd/f/feng045/DYAMOND/GPM_DYAMOND/IR_IMERG_Combined/{Phase}/'

    # Input file basenames and suffix
    imerg_basename = '3B-HHR.MS.MRG.3IMERG.'
    ir_basename = 'merg_'
    in_suffix = '_4km-pixel.nc'
    # Output file basenames and suffix
    out_basename = ir_basename
    out_suffix = '_10km-pixel.nc'

    # Tb and precipitation variable names
    tb_varname = 'Tb'
    pcp_varname = 'precipitation'

    # Subset latitude bounds
    lat_bounds = [-60., 60.]

    # Make output directory
    os.makedirs(outdir, exist_ok=True)

    # Find all IMERG files for the date
    inputfiles = sorted(glob.glob(f'{imergdir}{imerg_basename}{date}*nc4'))
    nfiles = len(inputfiles)
    print(f'Number of IR files: {nfiles}')

    # Get all hours from the filenames
    filehours = []
    for ih in range(0, nfiles):
        # Find string position after the date + 2
        # Example filename format is: 3B-HHR.MS.MRG.3IMERG.20000601-S000000-E002959.0000.V06B.HDF5.nc4
        # Example filename format is: 3B-HHR.MS.MRG.3IMERG.20160801-S000000-E002959.0000.V07B.HDF5.nc4
        strpos = inputfiles[ih].find(date)+len(date)+2
        # Append the hours to the list
        filehours.append(inputfiles[ih][strpos:strpos+2])
        
    # Find the unique hours
    filehours_unique = np.unique(filehours)
    nhours = len(filehours_unique)


    # Loop over each unique hour
    for ih in range(0, nhours):
        
        # Find indices that match the unique hour
        idx = np.where(np.array(filehours) == filehours_unique[ih])[0]
        # Get the IMERG filenames
        imergfiles = np.array(inputfiles)[idx]
        
        # IR filename
        irfile = f'{irdir}{ir_basename}{date}{filehours_unique[ih]}{in_suffix}'

        # Output filename
        outfile = f'{outdir}{out_basename}{date}{filehours_unique[ih]}{out_suffix}'
        
        # Check if IR file exist
        if os.path.isfile(irfile) == True:

            # Call function to combine two files
            combine_ir_imerg(imergfiles, irfile, outfile)

        else:
            print(f'Warning, IR file is missing: {irfile}')
            print('No output is created for this time.')

