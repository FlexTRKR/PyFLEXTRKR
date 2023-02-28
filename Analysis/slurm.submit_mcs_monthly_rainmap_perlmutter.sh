#!/bin/bash
#SBATCH -A m1867
#SBATCH --job-name=rainmap
#SBATCH -p debug
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=128
#SBATCH -C cpu
#SBATCH --time=00:15:00
#SBATCH --exclusive
#SBATCH --verbose
#SBATCH --mail-user=zhe.feng@pnnl.gov
#SBATCH --mail-type=END
#SBATCH --output=log_mcs_monthly_rainmap_gridrad.log
date
# export PATH=$PATH:/usr/common/tig/taskfarmer/1.5/bin:$(pwd)
module load taskfarmer
export THREADS=32

cd /global/homes/f/feng045/program/PyFLEXTRKR-dev/Analysis
runcommands.sh tasklist_mcs_monthly_rainmap_gridrad_rerun.txt
# runcommands.sh tasklist_mcs_monthly_rainmap_gridrad.txt
# runcommands.sh processlist_mcs_monthly_rainmap_gpm_saag.txt
# runcommands.sh processlist_mcs_monthly_rainmap_wrf_saag.txt
# runcommands.sh processlist_mcs_monthly_rainhov_wrf_saag.txt
date