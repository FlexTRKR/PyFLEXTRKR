#!/bin/bash
# Create GPM MCS tracking config file and slurm script for a given year

START=2018
END=2019

config_dir="/global/homes/f/feng045/program/PyFLEXTRKR/config/"
slurm_dir="/global/homes/f/feng045/program/PyFLEXTRKR/slurm/"
config_template=${config_dir}"config_gpm_mcs_global.yml"
slurm_template=${slurm_dir}"slurm_gpm_mcs_global.sh"
config_basename="config_gpm_mcs_global_"
slurm_basename="slurm_gpm_mcs_global_"
# template=${config_dir}"config_gpm_mcs_asia.yml"
#config_basename="config_gpm_mcs_asia_"

# Loop over year
for year in $(seq $START $END); do
    year1=$((year+1))
    config_file=${config_dir}${config_basename}${year}.yml
    slurm_file=${slurm_dir}${slurm_basename}${year}.sh
    sed "s/STARTDATE/"${year}"0101.0000/g;s/ENDDATE/"${year1}"0101.0000/g" $config_template > ${config_file}
    sed "s/YEAR/"${year}"/g" $slurm_template > ${slurm_file}
    echo ${config_file}
    echo ${slurm_file}
    # sbatch ${slurm_file}
done
