#!/bin/bash
#SBATCH -A m1642
#SBATCH --job-name=run_LESdata.py
#SBATCH -p regular
#SBATCH --ntasks=1
#SBATCH --mail-user=hannah.barnes@pnnl.gov
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --output=run_LESdata.py.out
#SBATCH --exclusive
module load python/2.7-anaconda
source activate /global/homes/h/hcbarnes/python_parallel
module load jpeg
unset PYTHONSTARTUP
date
ipython run_LESdata.py
date
