#!/bin/bash
# Create GridRad MCS tracking config and slurm scripts

START_YEAR=2018
END_YEAR=2021

submit_job="no"

config_dir="/global/homes/f/feng045/program/PyFLEXTRKR/config/"
slurm_dir="/global/homes/f/feng045/program/PyFLEXTRKR/slurm/"

config_template=${config_dir}"config_gridrad_mcs_template.yml"

slurm_template=${slurm_dir}"slurm_lasso_wrf100m_template.sh"

config_basename="config_gridrad_mcs_"
slurm_basename="slurm_gridrad_mcs_"

# Loop over year
for year in $(seq $START_YEAR $END_YEAR); do
    # Start/end date for tracking period
    sdate=${year}0401.0000
    edate=${year}0831.0000
    # sdate=${year}0101.0000
    # edate=${year}1231.0000

    config_file=${config_dir}${config_basename}${year}.yml
    # slurm_file=${slurm_dir}${slurm_basename}${sdate}_${edate}.sh

    sed "s/STARTDATE/"${sdate}"/g;s/ENDDATE/"${edate}"/g" $config_template > ${config_file}
    # sed "s/STARTDATE/"${sdate}"/g;s/ENDDATE/"${edate}"/g" $slurm_template > ${slurm_file}
    echo ${config_file}
    # echo ${slurm_file}

    # # Submit job
    # if [[ "${submit_job}" == "yes" ]]; then
    #     sbatch ${slurm_file}
done