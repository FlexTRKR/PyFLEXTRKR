#!/bin/bash
#SBATCH -A m1867
#SBATCH -J mcstracking
#SBATCH --time=00:30:00
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=2
#SBATCH -q debug
#SBATCH -C haswell
#SBATCH --exclusive
#SBATCH --output=log_mcstracking.log

date 

# Set n_tasks = n_nodes x ntasks-per-node
# n_tasks=64

# module load python
conda activate /global/common/software/m1867/python/flextrkr

#export MALLOC_TRIM_THRESHOLD_=0

# Increase limit on number of open files
ulimit -n 32000
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=360s
export DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=360s

# Generate a scheduler filename with a random string
random_str=`echo $RANDOM | md5sum | head -c 10`
scheduler_file=$SCRATCH/scheduler_${random_str}.json

# Start Dask scheduler manually
dask-scheduler --scheduler-file=$scheduler_file &

## Start dask cluster
#srun -u dask-mpi \
#--scheduler-file=$scheduler_file \
#--nthreads=1 \
#--memory-limit='auto' \
#--worker-class distributed.Worker \
#--local-directory=/tmp &
#
#sleep 5

srun -N 5 --ntasks-per-node=16 dask-worker \
--scheduler-file=$scheduler_file \
--memory-limit='8GB' \
--worker-class distributed.Worker \
--local-directory=/tmp &

sleep 10

# Run Python
cd /global/homes/f/feng045/program/PyFLEXTRKR
python ./runscripts/run_mcs_tbpf.py ./config/config_imerg_mcs_tbpf_example.yml $scheduler_file

date