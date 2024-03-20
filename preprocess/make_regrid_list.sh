#!/bin/bash

# This script creates a list of files for regridding using NERSC Task Farmer

# season='Summer'
season='Winter'
indir=/pscratch/sd/f/feng045/DYAMOND/GPM_DYAMOND/GPM_MERGIR_V1/${season}/
outdir=/pscratch/sd/f/feng045/DYAMOND/GPM_DYAMOND/GPM_MERGIR_V1_regrid/${season}/

# Regridding script (must have full path)
reg_script=/pscratch/sd/f/feng045/codes/dyamond/preprocess/regrid_global_ir_4km_to_10km.sh
# reg_script=/global/homes/f/feng045/program/PyFLEXTRKR-dev/preprocess/regrid_global_ir_4km_to_10km.sh

# Task list filename
task_filename=task_regrid_${season}.txt

# Create a task list file, and put "mkdir" on the first line
echo mkdir -p ${outdir} > ${task_filename}

# List all input netCDF files and pass them into the task list file
ls ${indir}/*.nc4 | awk -v outdir=${outdir} -v reg_script=${reg_script} '{print reg_script, $1, outdir}' >> ${task_filename}
echo ${task_filename}


#--------------------------------------------------------------
# The following example makes a task list for each year
#--------------------------------------------------------------

# # Specify the start/end year
# start_year=2020
# end_year=2020

# for iyear in $(seq $start_year $end_year); do
#   # Input/output directory for a specific year
#   iindir=${indir}${iyear}
#   ioutdir=${outdir}${iyear}

#   # Create a task list file, and put "mkdir" on the first line
#   task_filename=task_regrid_${iyear}
#   echo "mkdir -p ${ioutdir}" > ${task_filename}

#   # List all input netCDF files and pass them into the task list file
#   ls ${iindir}/*.nc4 | awk -v outdir=${ioutdir} -v reg_script=${reg_script} '{print reg_script, $1, outdir}' >> ${task_filename}
#   echo ${task_filename}
# done