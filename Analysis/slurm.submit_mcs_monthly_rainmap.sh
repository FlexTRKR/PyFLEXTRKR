#!/bin/bash
#SBATCH -A m1867
#SBATCH --job-name=rainmap
#SBATCH -p debug
#SBATCH --nodes=2
#SBATCH --cpus-per-task=64
#SBATCH -C haswell
#SBATCH --time=00:05:00
#SBATCH --exclusive
#SBATCH --verbose
#SBATCH --mail-user=zhe.feng@pnnl.gov
#SBATCH --mail-type=END
#SBATCH --output=log_mcs_monthly_rainmap_saag.log
date
export PATH=$PATH:/usr/common/tig/taskfarmer/1.5/bin:$(pwd)
export THREADS=12
runcommands.sh processlist_mcs_monthly_rainmap_gpm_saag.txt
# runcommands.sh processlist_mcs_monthly_rainmap_wrf_saag.txt
# runcommands.sh processlist_mcs_monthly_rainhov_wrf_saag.txt
date