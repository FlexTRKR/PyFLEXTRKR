#!/bin/bash
# Create GPM MCS tracking config file for a given year

START=2019
END=2019

config_dir="/global/homes/f/feng045/program/PyFLEXTRKR/config/"
template=${config_dir}"config_gpm_mcs_asia.yml"

# Loop over year
for year in $(seq $START $END); do
    echo ${year}
    year1=$((year+1))
    sed "s/STARTDATE/"${year}"0101.0000/g;s/ENDDATE/"${year1}"0101.0000/g" $template > ${config_dir}config_gpm_mcs_asia_${year}.yml
done
