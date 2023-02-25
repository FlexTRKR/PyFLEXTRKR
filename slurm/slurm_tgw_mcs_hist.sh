#!/bin/bash
#SBATCH -A m1867
#SBATCH -J TGWhist
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -q regular
#SBATCH -C haswell
#SBATCH --exclusive
#SBATCH --output=log_tgw_mcs_hist.log

date

# module load python
conda activate /global/common/software/m1867/python/flextrkr

# Run Python
cd /global/homes/f/feng045/program/PyFLEXTRKR-dev
python ./runscripts/run_mcs_tbpf.py ./config/config_tgw_mcs_hist.yml

date