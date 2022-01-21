#!/bin/bash
ncores=24
echo 'enter number of nodes':
read nnodes
ncpus=$(($ncores*$nnodes))
echo 'ncores:'${ncores}
echo 'nnodes:'${nnodes}
echo 'ncpus:'${ncpus}
echo 'enter python code':
read pythoncode
echo '#!/bin/bash' > temp.sbatch
echo '#SBATCH -A m1642' >> temp.sbatch
echo '#SBATCH --job-name='${pythoncode} >> temp.sbatch
echo '#SBATCH -p debug' >> temp.sbatch
echo '#SBATCH --ntasks='${ncores} >> temp.sbatch
echo '#SBATCH --mail-user=hannah.barnes@pnnl.gov' >> temp.sbatch
echo '#SBATCH --nodes='$nnodes >> temp.sbatch
echo '#SBATCH --time=00:30:00' >> temp.sbatch
echo '#SBATCH --output='${pythoncode}'.out' >> temp.sbatch
echo '#SBATCH --exclusive' >> temp.sbatch
echo 'module load python/2.7-anaconda' >> temp.sbatch
echo 'source activate /global/homes/h/hcbarnes/python_parallel' >> temp.sbatch
echo 'module load jpeg' >> temp.sbatch
echo 'unset PYTHONSTARTUP'>> temp.sbatch
echo 'date' >> temp.sbatch

#Launching controller
echo "ipcontroller --ip='*' &" >> temp.sbatch
echo "sleep 10" >> temp.sbatch
# Launching engines
echo "srun ipengine &" >> temp.sbatch
echo "sleep 45" >> temp.sbatch
# Launch job
echo "ipython ${pythoncode}" >> temp.sbatch

echo 'date' >> temp.sbatch
