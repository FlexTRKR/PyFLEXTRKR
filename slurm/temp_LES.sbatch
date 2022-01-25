#!/bin/csh
#SBATCH --account emsla50742
#SBATCH --job-name=run_LESdata.py
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=jingyi.chen@pnnl.gov
#SBATCH --mail-type END
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --output=./log/run_LESdata.py.out
#SBATCH --exclusive
cd /home/chen696/pyflextrkr-12hr/
module load hdf5/1.10.1
setenv HDF5_USE_FILE_LOCKING FALSE
date
srun -n 1 python run_LESdata.py config_sensitivity5.json
date
