#!/bin/bash
#SBATCH -A m2637
#SBATCH -J YEAR
#SBATCH -t 01:30:00
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --exclusive
#SBATCH --output=log_gridrad_mcs_YEAR.log
#SBATCH --mail-type=END
#SBATCH --mail-user=zhe.feng@pnnl.gov

date

module load python
source activate /global/common/software/m1867/python/flextrkr

# Run Python
cd /global/homes/f/feng045/program/PyFLEXTRKR-dev/runscripts
python run_mcs_tbpfradar3d_wrf.py /global/homes/f/feng045/program/pyflex_config/config/config_gridrad_mcs_YEAR.yml

date