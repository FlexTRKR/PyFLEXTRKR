#!/bin/bash
# Create GPM MCS tracking config file and slurm script for a given year

START_YEAR=2014
END_YEAR=2014
START_MON=2
END_MON=12

# Flag to make monthly scripts (1:yes, 0:no)
make_month=1
submit_month=1
# Flag to make annual scripts (1:yes, 0:no)
make_year=0
submit_year=0

config_dir="/global/homes/f/feng045/program/PyFLEXTRKR/config/"
slurm_dir="/global/homes/f/feng045/program/PyFLEXTRKR/slurm/"
config_template=${config_dir}"config_gpm_mcs_global.yml"
slurm_template=${slurm_dir}"slurm_gpm_mcs_global.sh"
config_basename="config_gpm_mcs_global_"
slurm_basename="slurm_gpm_mcs_global_"
# template=${config_dir}"config_gpm_mcs_asia.yml"
# config_basename="config_gpm_mcs_asia_"

#######################################################################
# Generate monthly scripts (for steps 1-2)
#######################################################################
if [ $((make_month)) -eq 1 ]; then
  # Loop over year
  for year in $(seq $START_YEAR $END_YEAR); do
    year1=$((year+1))
    # Loop over month
    for mon in $(seq $START_MON $END_MON); do
      # Pad month with 0
      smon=$(printf "%02d" $mon)
      # Start date
      sdate=${year}${smon}01.0000
      # Check if next month is <= 12
      if [ $((mon+1)) -le 12 ]; then
        # Set end date to next month
        emon=$(printf "%02d" $((mon+1)))
        edate=${year}${emon}01.0100
      else
        # Set end date to next year on Jan 1
        edate=${year1}0101.0100
      fi
  #    echo ${sdate}-${edate}

      config_file=${config_dir}${config_basename}${sdate}_${edate}.yml
      slurm_file=${slurm_dir}${slurm_basename}${sdate}_${edate}.sh

      sed "s/STARTDATE/"${sdate}"/g;s/ENDDATE/"${edate}"/g" $config_template > ${config_file}
      sed "s/STARTDATE/"${sdate}"/g;s/ENDDATE/"${edate}"/g" $slurm_template > ${slurm_file}
  #    echo ${config_file}
      echo ${slurm_file}

      # Submit job
      if [ $((submit_month)) -eq 1 ]; then
        sbatch ${slurm_file}
      fi
    done
  done
fi

#######################################################################
# Generate yearly scripts (for steps 3-9)
#######################################################################
if [ $((make_year)) -eq 1 ]; then
  # Loop over year
  for year in $(seq $START_YEAR $END_YEAR); do
      year1=$((year+1))
      sdate=${year}0101.0000
      edate=${year1}0101.0000
      config_file=${config_dir}${config_basename}${sdate}_${edate}.yml
      slurm_file=${slurm_dir}${slurm_basename}${sdate}_${edate}.sh
      sed "s/STARTDATE/"${year}"0101.0000/g;s/ENDDATE/"${year1}"0101.0000/g" $config_template > ${config_file}
      sed "s/STARTDATE/"${year}"0101.0000/g;s/ENDDATE/"${year1}"0101.0000/g" $slurm_template > ${slurm_file}
      echo ${config_file}
      echo ${slurm_file}

      # Submit job
      if [ $((submit_year)) -eq 1 ]; then
        sbatch ${slurm_file}
      fi
  done
fi
