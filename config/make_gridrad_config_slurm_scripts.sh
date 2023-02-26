#!/bin/bash
# Create GridRad MCS tracking config and slurm scripts

START_YEAR=2004
END_YEAR=2021

# Flag to submit slurm job
submit_job=0

# Directories
config_dir="/global/homes/f/feng045/program/PyFLEXTRKR-dev/config/"
slurm_dir="/global/homes/f/feng045/program/PyFLEXTRKR-dev/slurm/"

# Config template
config_template=${config_dir}"config_gridrad_mcs_template.yml"
# Slurm template
slurm_template=${slurm_dir}"slurm_gridrad_mcs_template.sh"

# Basenames for config & slurm scripts
config_basename="config_gridrad_mcs_"
slurm_basename="slurm_gridrad_mcs_"

# Loop over year
for year in $(seq $START_YEAR $END_YEAR); do
    # 2004-2017
    if (( $year <= 2017 )); then
        year1=$(($year+1))
        sdate=${year}0101.0000
        edate=${year1}0101.0000
    else
    # 2018-2021
        sdate=${year}0401.0000
        edate=${year}0901.0000
    fi

    config_file=${config_dir}${config_basename}${year}.yml
    slurm_file=${slurm_dir}${slurm_basename}${year}.sh

    sed "s/STARTDATE/"${sdate}"/g;s/ENDDATE/"${edate}"/g;s/YEAR/"${year}"/g" $config_template > ${config_file}
    sed "s/YEAR/"${year}"/g" $slurm_template > ${slurm_file}
    echo ${config_file}
    echo ${slurm_file}

    # Submit job
    if [ $((submit_job)) -eq 1 ]; then
        sbatch ${slurm_file}
    fi
done