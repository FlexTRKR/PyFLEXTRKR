#!/bin/bash

# This script regrids the MERG IR global 4 km data to match the IMERG precipitation global 10 km grid

# Creates an output filename
# $1 is the input file name with full path, $2 is the output directory
# This strips the file path from the input file name, and adds output file path
outfn="$2"/`basename ${1%.*}.nc`

# Remap weight file
mapping_file='/pscratch/sd/f/feng045/waccem/map_data/weight_ir_4km_to_10km_conserve.nc'

# Activate E3SM Unified software environment, which includes ESMF remap library "ncremap"
source /global/common/software/e3sm/anaconda_envs/load_latest_e3sm_unified_pm-cpu.sh

# Regrid using ncremap
# The "-P mpas --mss_val=-9999" option is recommended by the NCO developer Charles Zender
ncremap -i $1 -m ${mapping_file} -o $outfn -P mpas --mss_val=-9999 

# Another option is "--rnr_thr=0.0"
#ncremap -i $1 -m ${mapping_file} -o $outfn --rnr_thr=0.0

#ncremap -i $1 -m ${mapping_file} -o $outfn -P mpas --mss_val=-9999 -4
