#!/bin/bash
#SBATCH -A atm123
#SBATCH -J STARTDATEENSMEMBER
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH -p batch_all
#SBATCH --exclusive
#SBATCH --output=log_CONFIG_NAME.log
#SBATCH --mail-type=END
#SBATCH --mail-user=zhe.feng@pnnl.gov

# source /ccsopen/home/zhe1feng1/.bashrc
# module load python
# conda activate /ccsopen/home/zhe1feng1/anaconda3/envs/flextrkr

date

# Run Python
cd /ccsopen/home/zhe1feng1/program/PyFLEXTRKR
python ./runscripts/run_celltracking_lasso.py ./config/CONFIG_NAME.yml

date