#!/bin/bash
#SBATCH -A m2637
#SBATCH -J CASE
#SBATCH -t 00:05:00
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --exclusive
#SBATCH --output=log_scream_mcs_CASE.log
#SBATCH --mail-type=END
#SBATCH --mail-user=zhe.feng@pnnl.gov

date

module load python
source activate /global/common/software/m1867/python/pyflex

# Run Python
cd /global/homes/f/feng045/program/PyFLEXTRKR-dev/runscripts
python run_mcs_tbpfradar3d.py /global/homes/f/feng045/program/PyFLEXTRKR-dev/config/config_scream_mcs_CASE.yml

date