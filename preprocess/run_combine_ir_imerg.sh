#!/bin/bash

# This wrapper script runs the Python code combine_ir_imerg_global_byday.py to 
# comnbine IR and IMERG data to a netCDF file

# Activate Python environment
source activate /global/common/software/m1867/python/py310

# $1 is a date string (e.g., 20180601), $2 is phase (e.g., 'Summer' or 'Winter')
python /pscratch/sd/f/feng045/codes/dyamond/preprocess/combine_ir_imerg_global_byday.py $1 $2
# python /global/homes/f/feng045/program/PyFLEXTRKR-dev/preprocess/combine_ir_imerg_global_byday.py $1