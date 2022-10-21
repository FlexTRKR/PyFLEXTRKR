#!/bin/bash
#SBATCH -A m1867
#SBATCH -J celltracking
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -q debug
#SBATCH -C haswell
#SBATCH --exclusive
#SBATCH --output=log_celltracking.log

date

# module load python
conda activate /global/common/software/m1867/python/flextrkr

# Run Python
cd /global/homes/f/feng045/program/PyFLEXTRKR
python ./runscripts/run_celltracking.py ./config/config_csapr500m_example.yml

date