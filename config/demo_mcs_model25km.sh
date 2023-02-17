#!/bin/bash
###############################################################################################
# This script demonstrates running MCS tracking on model OLR + precipitation data
# To run this demo script:
# 1. Modify the dir_demo to a directory on your computer to download the sample data
# 2. Run the script: bash demo_mcs_model25km.sh
# 
# By default the demo config uses 4 processors for parallel processing, 
#    assuming most computers have at least 4 CPU cores. 
#    If your computer has more than 4 processors, you may modify 'nprocesses' 
#    in config_model25km_mcs_tbpf_example.yml to reduce the run time.
###############################################################################################

# Specify directory for the demo data
dir_demo='/Users/feng045/data/demo/mcs_tbpf/e3sm/'

# Example config file name
config_demo='config_mcs_demo.yml'

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

# Add '\' to each '/' in directory names
dir_input1=$(echo ${dir_input} | sed 's_/_\\/_g')
dir_demo1=$(echo ${dir_demo} | sed 's_/_\\/_g')
# Replace input directory names in example config file
sed 's/INPUT_DIR/'${dir_input1}'/g;s/TRACK_DIR/'${dir_demo1}'/g' config_model25km_mcs_tbpf_example.yml > ${config_demo}
echo 'Created new config file: '${config_demo}

# Activate PyFLEXTRKR conda environment
echo 'Activating PyFLEXTRKR environment ...'
conda activate flextrkr

# Run tracking
echo 'Running PyFLEXTRKR ...'
python ../runscripts/run_mcs_tbpf.py ${config_demo}
echo 'Tracking is done.'

# Make quicklook plots
echo 'Making quicklook plots ... (robust MCS)'
quicklook_dir1=${dir_demo}'/quicklooks_robust/'
python ../Analysis/plot_subset_tbpf_mcs_tracks_demo.py -s '2007-05-07T00' -e '2007-05-12T00' \
    -c ${config_demo} -o horizontal -p 1 --figsize 10 10 --output ${quicklook_dir1}
echo 'View quicklook plots here: '${quicklook_dir1}

#echo 'Making quicklook plots ... (Tb-only MCS)'
#quicklook_dir2=${dir_demo}'/quicklooks_tb/'
#python ../Analysis/plot_subset_tbpf_mcs_tracks_demo.py -s '2007-05-07T00' -e '2007-05-12T00' \
#    -c ${config_demo} -o horizontal -p 1 --figsize 10 10 --output ${quicklook_dir2} \
#    --trackstats_file '/Users/feng045/data/demo/mcs_tbpf/e3sm/stats/mcs_tracks_pf_20070507.0000_20070512.0000.nc' \
#    --pixel_path '/Users/feng045/data/demo/mcs_tbpf/e3sm/mcstracking_tb/20070507.0000_20070512.0000/'
#echo 'View quicklook plots here: '${quicklook_dir2}

# Make animation using ffmpeg
echo 'Making animations from quicklook plots ...'
ffmpeg -framerate 2 -pattern_type glob -i ${quicklook_dir1}'*.png' -c:v libx264 -r 10 -crf 20 -pix_fmt yuv420p \
    -y ${quicklook_dir1}mcs_robust_animation.mp4
echo 'View animation here: '${quicklook_dir1}mcs_robust_animation.mp4

## Make animation using ffmpeg
#echo 'Making animations from quicklook plots ...'
#ffmpeg -framerate 2 -pattern_type glob -i ${quicklook_dir2}'*.png' -c:v libx264 -r 10 -crf 20 -pix_fmt yuv420p \
#    -y ${quicklook_dir2}mcs_tb_animation.mp4
#echo 'View animation here: '${quicklook_dir2}mcs_tb_animation.mp4

echo 'Demo tracking completed!'