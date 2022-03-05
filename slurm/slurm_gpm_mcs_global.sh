#!/bin/bash
#SBATCH -A m1867
#SBATCH -J STARTDATE
#SBATCH -t 00:15:00
#SBATCH -N 10
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=4
#SBATCH -q regular
#SBATCH -C haswell
#SBATCH --exclusive
#SBATCH --output=log_gpm_global_STARTDATE_ENDDATE.log
#SBATCH --mail-type=END
#SBATCH --mail-user=zhe.feng@pnnl.gov

# Set n_tasks = n_nodes x ntasks-per-node
n_tasks=160

#module load python
conda activate /global/common/software/m1867/python/flextrkr-mpi

#export MALLOC_TRIM_THRESHOLD_=0

# Increase limit on number of open files
ulimit -n 32000

export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=300s
export DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=300s

# Generate a random string
random_str=`echo $RANDOM | md5sum | head -c 10`

# Start Dask scheduler manually
dask-scheduler --scheduler-file=$SCRATCH/scheduler_${random_str}.json &

## Start dask cluster
#srun -u dask-mpi \
#--scheduler-file=$SCRATCH/scheduler_${random_str}.json \
#--nthreads=1 \
#--memory-limit='auto' \
#--worker-class distributed.Worker \
#--local-directory=/tmp &
#
#sleep 5

srun -N 10 --ntasks-per-node=16 dask-worker \
--scheduler-file=$SCRATCH/scheduler_${random_str}.json \
--memory-limit='6GB' \
--worker-class distributed.Worker \
--local-directory=/tmp &

sleep 5

# Run Python
cd /global/homes/f/feng045/program/PyFLEXTRKR
python ./runscripts/run_mcs_tbpf_gpm.py ./config/config_gpm_mcs_global_STARTDATE_ENDDATE.yml scheduler_${random_str}.json $n_tasks
