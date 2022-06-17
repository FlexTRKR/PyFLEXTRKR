#!/bin/bash
# Create LASSO cell tracking config and slurm scripts

config_dir="/ccsopen/home/zhe1feng1/program/PyFLEXTRKR/config/"
slurm_dir="/ccsopen/home/zhe1feng1/program/PyFLEXTRKR/slurm/"
# D4 (100m) 5min tracking
config_template=${config_dir}"config_lasso_wrf100m_template.yml"
# D4 (100m) 15min tracking
# config_template=${config_dir}"config_lasso_wrf100m_15min_template.yml"
slurm_template=${slurm_dir}"slurm_lasso_wrf100m_template.sh"
# D3 (500m) 15min tracking
# config_template=${config_dir}"config_lasso_wrf500m_template.yml"
# slurm_template=${slurm_dir}"slurm_lasso_wrf500m_template.sh"
config_basename="config_lasso_"
slurm_basename="slurm_lasso_"
submit_job="no"

# start_dates=(
#     "20190123"
# )
# ens_members=(
#     "eda05"
# )
# Full list of runs
start_dates=(
    "20181129" "20181129" 
    "20181204" "20181204" 
    "20181205" 
    "20181219" 
    "20190122" 
    "20190123"
    "20190125" "20190125"
    "20190129" "20190129"
    "20190208" "20190208"
)
ens_members=(
    "gefs00" "gefs03" 
    "gefs18" "gefs19" 
    "gefs01" 
    "eda09" 
    "gefs01" 
    "eda05"
    "eda07" "gefs11"
    "eda09" "gefs11"
    "eda03" "eda08"
)

# Loop over list
for ((i = 0; i < ${#start_dates[@]}; ++i)); do   
    sdate=${start_dates[$i]}
    # edate=${end_dates[$i]}
    edate="$((sdate+1))"
    ensmember=${ens_members[$i]}

    config_name=${config_basename}${sdate}_${ensmember}
    config_file=${config_dir}${config_name}.yml
    slurm_file=${slurm_dir}${slurm_basename}${sdate}_${ensmember}.sh

    sed "s/STARTDATE/"${sdate}"/g;s/ENDDATE/"${edate}/g";s/ENSMEMBER/"${ensmember}"/g" ${config_template} > ${config_file}
    sed "s/STARTDATE/"${sdate}"/g;s/ENSMEMBER/"${ensmember}"/g;s/CONFIG_NAME/"${config_name}"/g" ${slurm_template} > ${slurm_file}
    echo ${config_file}
    echo ${slurm_file}
    if [[ "${submit_job}" == "yes" ]]; then
        # echo ${slurm_file}
        sbatch ${slurm_file}
    fi
done