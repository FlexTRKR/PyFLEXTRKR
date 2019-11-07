#!/bin/bash

ncores=1
echo 'enter number of nodes':
read nnodes

ncpus=$(($ncores*$nnodes))
#echo 'ncores:'${ncores}
#echo 'nnodes:'${nnodes}
echo 'ncpus:'${ncpus}

echo 'enter python code':
read pythoncode

echo '#!/bin/csh' > temp_LES.sbatch
echo '#SBATCH --account emsla50742' >> temp_LES.sbatch
echo '#SBATCH --job-name='${pythoncode} >> temp_LES.sbatch
echo '#SBATCH --ntasks-per-node='${ncores} >> temp_LES.sbatch
echo '#SBATCH --mail-user=jingyi.chen@pnnl.gov' >> temp_LES.sbatch
echo '#SBATCH --mail-type END' >> temp_LES.sbatch
echo '#SBATCH --nodes='$nnodes >> temp_LES.sbatch
echo '#SBATCH --time=48:00:00' >> temp_LES.sbatch
echo '#SBATCH --output=./log/'${pythoncode}'.out' >> temp_LES.sbatch
echo '#SBATCH --exclusive' >> temp_LES.sbatch
echo 'cd /home/chen696/pyflextrkr-12hr/' >> temp_LES.sbatch

echo 'module load hdf5/1.10.1' >> temp_LES.sbatch
echo 'setenv HDF5_USE_FILE_LOCKING FALSE' >> temp_LES.sbatch
echo 'date' >> temp_LES.sbatch
echo 'srun -n '${ncpus}' python' ${pythoncode} 'config_sensitivity5.json' >> temp_LES.sbatch
#echo 'python' ${pythoncode} 'config_sensitivity5.json' >> temp_LES.sbatch
echo 'date' >> temp_LES.sbatch

sbatch temp_LES.sbatch
