#!/bin/bash
# Create 

config_dir="/ccsopen/home/zhe1feng1/program/PyFLEXTRKR/config/"
slurm_dir="/ccsopen/home/zhe1feng1/program/PyFLEXTRKR/slurm/"
config_template=${config_dir}"config_lasso_wrf100m_template.yml"
slurm_template=${slurm_dir}"slurm_lasso_template.sh"
config_basename="config_lasso_"
slurm_basename="slurm_lasso_"

# start_dates=(
#     "20181204" "20181204"
#     "20181205" 
# )
# end_dates=(
#     "20181205" "20181205"
#     "20181206" 
# )
# ens_members=(
#     "gefs_en18" "gefs_en19"
#     "gefs_en01"
# )

start_dates=(
    "20181205" "20181219" "20190122" "20190125" "20190125"
)
# end_dates=(
#     "20181205"
# )
ens_members=(
    "gefs01" "eda09" "gefs01" "eda07" "gefs11"
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
    # echo ${slurm_template}
    echo ${config_file}
    echo ${slurm_file}
done