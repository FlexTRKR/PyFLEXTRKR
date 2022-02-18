#!/bin/bash
#SBATCH -A m1867
#SBATCH -J mcsYEAR
#SBATCH -t 00:30:00
#SBATCH -N 10
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=2
#SBATCH -q debug
#SBATCH -C haswell
#SBATCH --exclusive
#SBATCH --output=log_gpm_global_YEAR.log
#SBATCH --mail-type=END
#SBATCH --mail-user=zhe.feng@pnnl.gov

# Set n_tasks = n_nodes x ntasks-per-node
n_tasks=310

#module load python
conda activate /global/common/software/m1867/python/flextrkr-mpi

#export MALLOC_TRIM_THRESHOLD_=0

# Generate a random string
random_str=`echo $RANDOM | md5sum | head -c 10`

# Start dask cluster
srun -u dask-mpi \
--scheduler-file=$SCRATCH/scheduler_${random_str}.json \
--nthreads=1 \
--memory-limit='auto' \
--worker-class distributed.Worker \
--local-directory=/tmp &

sleep 5

# Run Python
cd /global/homes/f/feng045/program/PyFLEXTRKR
python ./runscripts/run_mcs_tbpf_gpm.py ./config/config_gpm_mcs_global_YEAR.yml scheduler_${random_str}.json $n_tasks
