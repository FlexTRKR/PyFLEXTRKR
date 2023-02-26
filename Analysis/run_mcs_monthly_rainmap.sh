#!/bin/bash
# Wrapper shell script that runs the Python code for TaskFarmer
source activate /global/common/software/m1867/python/flextrkr
# conda activate flextrkr
python /global/homes/f/feng045/program/PyFLEXTRKR-dev/Analysis/calc_tbpf_mcs_monthly_rainmap.py $1 $2 $3
# python /global/homes/f/feng045/program/PyFLEXTRKR/Analysis/calc_tbpf_mcs_monthly_rainmap_gridrad.py $1 $2 $3