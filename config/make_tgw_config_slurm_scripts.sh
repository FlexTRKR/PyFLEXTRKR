#!/bin/bash
# Create TGW MCS tracking config and slurm scripts

START_YEAR=2004
END_YEAR=2005

# Flag to submit slurm job
submit_job=0

# Directories
config_dir="/global/homes/f/feng045/program/PyFLEXTRKR-dev/config/"
slurm_dir="/global/homes/f/feng045/program/PyFLEXTRKR-dev/slurm/"

# Config template
config_template=${config_dir}"config_tgw_mcs_hist_template.yml"
# Slurm template
slurm_template=${slurm_dir}"slurm_tgw_mcs_template.sh"

# Basenames for config & slurm scripts
config_basename="config_tgw_mcs_hist_"
slurm_basename="slurm_tgw_mcs_hist_"

# Loop over year
for year in $(seq $START_YEAR $END_YEAR); do
    sdate=${year}0301.0000
    edate=${year}1101.0000

    config_file=${config_dir}${config_basename}${year}.yml
    slurm_file=${slurm_dir}${slurm_basename}${year}.sh

    sed "s/STARTDATE/"${sdate}"/g;s/ENDDATE/"${edate}"/g" $config_template > ${config_file}
    sed "s/YEAR/"${year}"/g" $slurm_template > ${slurm_file}
    echo ${config_file}
    echo ${slurm_file}

    # Submit job
    if [ $((submit_job)) -eq 1 ]; then
        sbatch ${slurm_file}
    fi
done