#!/bin/bash
#SBATCH -A m1657
#SBATCH -J SAAGmcs
#SBATCH -t 00:05:00
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --exclusive
#SBATCH --output=log_mcs_saag_test.log
#SBATCH --mail-type=END
#SBATCH --mail-user=zhe.feng@pnnl.gov

date

module load python
source activate /global/common/software/m1867/python/pyflex

# Run Python
cd /global/homes/f/feng045/program/PyFLEXTRKR-dev/runscripts
python run_mcs_tbpf_saag.py /global/homes/f/feng045/program/PyFLEXTRKR-dev/config/config_mcs_saag_example.yml

date