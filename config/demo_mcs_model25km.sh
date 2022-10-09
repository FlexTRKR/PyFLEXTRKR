#!/bin/bash
###############################################################################################
# This script demonstrates running MCS tracking on model OLR + precipitation data
# To run this demo script:
# 1. Modify the dir_demo to a directory on your computer to download the sample data
# 2. Run the script: bash ./demo_mcs_model25km.sh
# 
# By default the demo config uses 4 processors for parallel processing, 
#    assuming most computers have at least 4 CPU cores. 
#    If your computer has more than 4 processors, you may modify 'nprocesses' 
#    in config_model25km_mcs_tbpf_example.yml.
###############################################################################################

# Specify directory for the demo data
dir_demo='/global/cscratch1/sd/feng045/pyflextrkr_test/mcs_tbpf/e3sm/'

# Demo input data directory
dir_input=${dir_demo}'input/'

# Create the demo directory
mkdir -p ${dir_input}

# Download sample model Tb+Precipitation data:
echo 'Downloading demo input data ...'
wget https://portal.nersc.gov/project/m1867/PyFLEXTRKR/sample_data/tb_pcp/e3sm_tbpcp.tar.gz -O ${dir_input}/e3sm_tbpcp.tar.gz

# Extract intput data
echo 'Extracting demo input data ...'
tar -xvzf ${dir_input}e3sm_tbpcp.tar.gz -C ${dir_input}
# Remove downloaded tar file
rm -fv ${dir_input}e3sm_tbpcp.tar.gz

# Example config file name
config_demo='config_mcs_demo.yml'

# Add '\' to each '/' in directory names
dir_input1=$(echo ${dir_input} | sed 's_/_\\/_g')
dir_demo1=$(echo ${dir_demo} | sed 's_/_\\/_g')
# Replace input directory names in example config file
sed 's/TB_RR_DIR/'${dir_input1}'/g;s/TRACK_DIR/'${dir_demo1}'/g' config_model25km_mcs_tbpf_example.yml > ${config_demo}
echo 'Created new config file: ${config_demo}'

# Activate PyFLEXTRKR conda environment
echo 'Activating PyFLEXTRKR environment ...'
conda activate testflex

# Run tracking
echo 'Running PyFLEXTRKR ...'
python ../runscripts/run_mcs_tbpf.py ${config_demo}

echo 'Demo tracking completed!'