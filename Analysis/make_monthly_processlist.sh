#!/bin/bash

# Make a list of months to process monthly statistics for TaskFarmer

# Example:
# run_mcs_monthly_rainmap.sh config.yml 2018 6
# run_mcs_monthly_rainmap.sh config.yml 2018 7
# run_mcs_monthly_rainmap.sh config.yml 2018 8

START=2004
END=2021

smonth=01
emonth=12

config_basename='/global/homes/f/feng045/program/PyFLEXTRKR-dev/config/config_gridrad_mcs_'
# config='~/program/PyFLEXTRKR/config/config_wrf_mcs_saag.yml'
# config='~/program/PyFLEXTRKR/config/config_gpm_mcs_saag.yml'

#runscript="run_mcs_monthly_statsmap.sh"
runscript="run_mcs_monthly_rainmap.sh"
# runscript="run_mcs_monthly_rainhov.sh"

# listname="processlist_mcs_monthly_rainmap_wrf_saag.txt"
listname="tasklist_mcs_monthly_rainmap_gridrad.txt"

# Create an empty file, will overwrite if exists
> ${listname}

for iyear in $(seq $START $END); do
    for imon in $(seq ${smonth} ${emonth}); do
#   for imon in {${smonth}..${emonth}}; do
    # iyear1=$((iyear+1))
    # sdate=${iyear}0601.0000
    # edate=${iyear1}0601.0000
    config=${config_basename}${iyear}.yml
    echo $PWD/${runscript} ${config} ${iyear} ${imon} >> ${listname}
  done
done

echo ${listname}
