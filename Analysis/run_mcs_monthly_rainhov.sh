#!/bin/bash
# Wrapper shell script that runs the Python code for TaskFarmer
source activate /global/common/software/m1867/python/testflex
python ~/program/PyFLEXTRKR-dev/Analysis/calc_tbpf_mcs_monthly_rainhov.py $1 $2 $3 $4 $5 $6 $7