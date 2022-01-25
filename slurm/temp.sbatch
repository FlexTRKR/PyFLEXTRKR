#!/bin/bash
#SBATCH -A m1657
#SBATCH --job-name=run_mcsdata_allradarmatch.py
#SBATCH -p regular
#SBATCH --ntasks=4
#SBATCH --mail-user=hannah.barnes@pnnl.gov
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output=run_mcsdata_allradarmatch.py.out
#SBATCH --exclusive
module load python/2.7-anaconda
source activate /global/homes/h/hcbarnes/python_parallel
module load jpeg
unset PYTHONSTARTUP
date
ipcontroller --ip='*' &
sleep 10
srun ipengine &
sleep 45
ipython run_mcsdata_allradarmatch.py
date
