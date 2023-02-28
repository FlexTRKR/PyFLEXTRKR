#!/bin/bash
#SBATCH -J mcs_tbze_quicklook
#SBATCH -A m1867
#SBATCH -t 00:20:00
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -C cpu
#SBATCH --exclusive
#SBATCH --output=log_mcs_tbze_quicklook_%A_%a.log
#SBATCH --mail-type=END
#SBATCH --mail-user=zhe.feng@pnnl.gov
#SBATCH --array=1-13

date
source activate flextrkr
# cd /global/homes/f/feng045/program/PyFLEXTRKR/Analysis/

# Takes a specified line ($SLURM_ARRAY_TASK_ID) from the task file
LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p tasklist_mcs_tbze_quicklook.txt)
echo $LINE
# Run the line as a command
$LINE

date