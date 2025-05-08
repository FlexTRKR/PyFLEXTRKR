"""
Coarsen a landseamask netCDF file to a specified resolution and outputs to a new netCDF file.
"""
__author__ = "Zhe.Feng@pnnl.gov"
__date__ = "07-May-2025"

import xarray as xr
import numpy as np
import os
import argparse

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Coarsen a landseamask netCDF file to a specified resolution")
    parser.add_argument("--input", "-i", 
                        default="/pscratch/sd/f/feng045/DYAMOND/maps/IMERG_landmask_180W-180E_60S-60N.nc",
                        help="Input netCDF file path")
    parser.add_argument("--output", "-o", 
                        help="Output netCDF file path (default: input filename with target resolution suffix)")
    parser.add_argument("--resolution", "-r", type=float, default=0.5,
                        help="Target resolution in degrees (default: 0.5)")
    
    args = parser.parse_args()
    
    # Set up input/output paths
    input_file = args.input
    target_resolution = args.resolution
    
    # Open the original netCDF file
    print(f"Opening {input_file}...")
    ds = xr.open_dataset(input_file)
    
    # Print some information about the original dataset
    print("\nOriginal dataset:")
    print(f"Dimensions: {ds.dims}")
    print(f"Longitude range: {ds.lon.values.min()} to {ds.lon.values.max()}")
    print(f"Latitude range: {ds.lat.values.min()} to {ds.lat.values.max()}")
    
    # Determine the current resolution
    lon_res = np.round(np.abs(ds.lon[1] - ds.lon[0]).item(), 4)
    lat_res = np.round(np.abs(ds.lat[1] - ds.lat[0]).item(), 4)
    print(f"Original resolution: {lon_res}° (lon) x {lat_res}° (lat)")
    
    # Calculate coarsening factors
    lon_factor = np.round(target_resolution / lon_res, 1)
    lat_factor = np.round(target_resolution / lat_res, 1)
    
    # Check if the factors are integers or nearly integers
    lon_factor_int = int(np.round(lon_factor))
    lat_factor_int = int(np.round(lat_factor))
    
    print(f"Target resolution: {target_resolution}°")
    print(f"Calculated coarsening factors: {lon_factor} (lon), {lat_factor} (lat)")
    print(f"Using integer coarsening factors: {lon_factor_int} (lon), {lat_factor_int} (lat)")
    
    # Set output filename if not specified
    if args.output:
        output_file = args.output
    else:
        input_dirname = os.path.dirname(input_file)
        input_basename = os.path.basename(input_file)
        # Include the target resolution in the filename
        output_basename = input_basename.replace('.nc', f'_{target_resolution}deg.nc')
        output_file = os.path.join(input_dirname, output_basename)

    # Coarsen the data using the calculated factors
    print(f"\nCoarsening data...")
    if lon_factor_int >= 1 and lat_factor_int >= 1:
        coarsened_ds = ds.coarsen(lon=lon_factor_int, lat=lat_factor_int, boundary='trim').mean()
        
        # For cases where the exact resolution is important
        if abs(lon_factor - lon_factor_int) > 0.01 or abs(lat_factor - lat_factor_int) > 0.01:
            print("Warning: Coarsening factors were rounded to integers, which may result in a slightly different resolution than requested.")
    else:
        print("Error: Target resolution is finer than the original resolution. Cannot coarsen.")
        return
    
    # Print some information about the coarsened dataset
    actual_lon_res = np.round(np.abs(coarsened_ds.lon[1] - coarsened_ds.lon[0]).item(), 4)
    actual_lat_res = np.round(np.abs(coarsened_ds.lat[1] - coarsened_ds.lat[0]).item(), 4)
    
    print("\nCoarsened dataset:")
    print(f"Dimensions: {coarsened_ds.dims}")
    print(f"Actual resolution: {actual_lon_res}° (lon) x {actual_lat_res}° (lat)")
    print(f"Longitude range: {coarsened_ds.lon.values.min()} to {coarsened_ds.lon.values.max()}")
    print(f"Latitude range: {coarsened_ds.lat.values.min()} to {coarsened_ds.lat.values.max()}")
    
    # Save the coarsened dataset to a new netCDF file
    print(f"\nSaving coarsened dataset to {output_file}...")
    coarsened_ds.to_netcdf(output_file)
    print("Done!")

if __name__ == "__main__":
    main()