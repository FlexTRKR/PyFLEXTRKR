#!/bin/bash
#SBATCH -A m1867
#SBATCH -J gpm_asia
#SBATCH -t 00:30:00
#SBATCH -N 20
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=2
#SBATCH -q debug
#SBATCH -C haswell
#SBATCH --exclusive
#SBATCH --output=log_gpm_asia.log
#SBATCH --mail-type=END
#SBATCH --mail-user=zhe.feng@pnnl.gov

module load python
conda activate /global/common/software/m1867/python/flextrkr-mpi

#export MALLOC_TRIM_THRESHOLD_=0

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
python ./runscripts/run_mcs_tbpf_gpm.py ./config/config_gpm_mcs_asia_2018.yml