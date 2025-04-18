#!/bin/bash
#SBATCH -A m1867
#SBATCH -J screamhp9
#SBATCH --qos=regular
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=128   # 128 workers per node (1 worker per core)
#SBATCH --cpus-per-task=1       # 1 CPU per worker
#SBATCH -C cpu
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --mail-user=zhe.feng@pnnl.gov
#SBATCH --mail-type=END
#SBATCH --output=log_mcs_scream_healpix.log
date

# Calculate total tasks
ntasks=$(( $SLURM_NNODES * $SLURM_NTASKS_PER_NODE ))

echo "Total tasks: $ntasks"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"

echo "Starting scheduler and workers..."

# Generate a scheduler filename with a random string
random_str=`echo $RANDOM | md5sum | head -c 10`
scheduler_file=$SCRATCH/scheduler_${random_str}.json

rm -f $scheduler_file

module load python
source activate /global/common/software/m1867/python/pyflex-dev

# Set environment variables for timeouts globally
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=3600s
export DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=3600s

# Start Dask Scheduler
dask scheduler \
    --interface hsn0 \
    --scheduler-file $scheduler_file &

dask_pid=$!

# Wait for the scheduler to start
sleep 5
until [ -f $scheduler_file ]
do
    sleep 5
done

echo "Starting workers"

# Start Dask Workers
srun --ntasks=$ntasks --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
     dask worker \
     --scheduler-file $scheduler_file \
     --interface hsn0 \
     --nthreads 1 \
     --memory-limit auto &

# Wait a bit to ensure workers have started
sleep 10

# Run Python
python /global/homes/f/feng045/program/PyFLEXTRKR-dev/runscripts/run_mcs_tbpf_mcsmip.py \
    /global/homes/f/feng045/program/PyFLEXTRKR-dev/config/config_mcs_tbpf_scream_healpix9.yml \
    $scheduler_file

# Clean up the scheduler
echo "Cleaning up scheduler..."
kill -9 $dask_pid

date
