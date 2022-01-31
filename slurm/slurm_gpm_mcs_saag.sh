#!/bin/bash
#SBATCH -A m1867
#SBATCH -t 01:00:00
#SBATCH -N 10
#SBATCH -n 100
#SBATCH --cpus-per-task 3
#SBATCH -q regular
#SBATCH -C haswell
#SBATCH --exclusive
#SBATCH --output=log_dask-mpi.log
#SBATCH --mail-type=END
#SBATCH --mail-user=zhe.feng@pnnl.gov

module load python
conda activate /global/common/software/m1867/python/flextrkr-mpi

# Start dask cluster
srun -u dask-mpi \
--scheduler-file=$SCRATCH/scheduler.json \
--nthreads=1 \
--memory-limit='auto' \
--worker-class distributed.Worker \
--local-directory=/tmp &

sleep 5

# Run Python
cd /global/homes/f/feng045/program/PyFLEXTRKR
python ./runscripts/run_gpm_irpf_mcs_new.py ./config/config_gpm_mcs_saag.yml