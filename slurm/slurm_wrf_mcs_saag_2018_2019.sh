#!/bin/bash
#SBATCH -A atm123
#SBATCH -J 2018_2019
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=18
#SBATCH -p batch_high_memory
#SBATCH --exclusive
#SBATCH --output=log_wrf_mcs_saag_2018_2019.log
#SBATCH --mail-type=END
#SBATCH --mail-user=zhe.feng@pnnl.gov

# Set n_tasks = n_nodes x ntasks-per-node
n_tasks=18

source /ccsopen/home/zhe1feng1/.bashrc
# module load python
conda activate /ccsopen/home/zhe1feng1/anaconda3/envs/flextrkr

#export MALLOC_TRIM_THRESHOLD_=0

# Increase limit on number of open files
ulimit -n 32000
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=360s
export DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=360s

# Generate a scheduler filename with a random string
random_str=`echo $RANDOM | md5sum | head -c 10`
scheduler_file=/gpfs/wolf/atm123/proj-shared/zhefeng/scheduler_${random_str}.json

# Start Dask scheduler manually
# dask-scheduler --scheduler-file=$scheduler_file &

## Start dask cluster
#srun -u dask-mpi \
#--scheduler-file=$scheduler_file \
#--nthreads=1 \
#--memory-limit='auto' \
#--worker-class distributed.Worker \
#--local-directory=/tmp &
#
#sleep 5

# srun -N 5 --ntasks-per-node=32 dask-worker \
# --scheduler-file=$scheduler_file \
# --memory-limit='8GB' \
# --worker-class distributed.Worker \
# --local-directory=/tmp &

# sleep 10

# Run Python
cd /ccsopen/home/zhe1feng1/program/PyFLEXTRKR
# python ./runscripts/run_mcs_tbpf.py ./config/config_wrf_mcs_saag_cu2_2018_2019.yml $scheduler_file $n_tasks
python ./runscripts/run_mcs_tbpf.py ./config/config_wrf_mcs_saag_cu2_2018_2019.yml
