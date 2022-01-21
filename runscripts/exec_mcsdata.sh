#!/bin/bash
ncores=32
echo 'enter number of nodes':
read nnodes
ncpus=$(($ncores*$nnodes))
#echo 'ncores:'${ncores}
#echo 'nnodes:'${nnodes}
echo 'ncpus:'${ncpus}
echo 'enter python code':
read pythoncode
echo '#!/bin/bash' > temp.sbatch
echo '#BATCH -A m1867' >> temp.sbatch
echo '#SBATCH --job-name='${pythoncode} >> temp.sbatch
echo '#SBATCH -p debug' >> temp.sbatch
echo '#SBATCH --ntasks='${ncores} >> temp.sbatch
echo '#SBATCH --mail-user=hannah.barnes@pnnl.gov' >> temp.sbatch
echo '#SBATCH --nodes='$nnodes >> temp.sbatch
echo '#SBATCH --time=00:30:00' >> temp.sbatch
echo '#SBATCH --output=/log/'${pythoncode}'.out' >> temp.sbatch
echo '#SBATCH --exclusive' >> temp.sbatch
echo 'cd ' >> temp.sbatch
echo 'bash' >> temp.sbatch
echo 'module load python/2.7-anaconda' >> temp.sbatch
#echo 'source activate /global/homes/f/feng045/edison-envs/python' >> temp.sbatch
echo 'cd /global/u1/h/hcbarnes/Tracking/Python' >> temp.sbatch
echo 'date' >> temp.sbatch
echo 'srun -n '${ncpus}' python' ${pythoncode} >> temp.sbatch
echo 'date' >> temp.sbatch
sbatch temp.sbatch
