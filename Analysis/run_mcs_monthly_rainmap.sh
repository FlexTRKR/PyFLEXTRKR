#!/bin/bash
# Wrapper shell script that runs the Python code for TaskFarmer
# source activate /global/common/software/m1867/python/flextrkr
conda activate flextrkr
python ~/program/PyFLEXTRKR/Analysis/calc_tbpf_mcs_monthly_rainmap.py $1 $2 $3