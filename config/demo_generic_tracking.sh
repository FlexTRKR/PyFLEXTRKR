#!/bin/bash
###############################################################################################
# This script demonstrates running generic feature tracking (Z500 anomaly)
# To run this demo script:
# 1. Modify the dir_demo to a directory on your computer to download the sample data
# 2. Run the script: bash demo_generic_tracking.sh
#
# By default the demo config uses 4 processors for parallel processing,
#    assuming most computers have at least 4 CPU cores.
###############################################################################################

# Specify directory for the demo data
dir_demo='/Users/feng045/data/demo/general_tracking/z500_blocking/'

# Example config file name
config_demo='config_era5_z500.yml'

# Demo input data directory
dir_input=${dir_demo}'input/'

# Create the demo directory
mkdir -p ${dir_input}

# Download sample data:
echo 'Downloading demo input data ...'
wget https://portal.nersc.gov/project/m1867/PyFLEXTRKR/sample_data/generic/ERA5_z500_anom.tar.gz -O ${dir_input}ERA5_z500_anom.tar.gz

# Extract intput data
echo 'Extracting demo input data ...'
tar -xvzf ${dir_input}ERA5_z500_anom.tar.gz -C ${dir_input}
# Remove downloaded tar file
rm -fv ${dir_input}ERA5_z500_anom.tar.gz

# Add '\' to each '/' in directory names
dir_input1=$(echo ${dir_input} | sed 's_/_\\/_g')
dir_demo1=$(echo ${dir_demo} | sed 's_/_\\/_g')
# Replace input directory names in example config file
sed 's/INPUT_DIR/'${dir_input1}'/g;s/TRACK_DIR/'${dir_demo1}'/g' config_era5_z500_example.yml > ${config_demo}
echo 'Created new config file: '${config_demo}

# # Activate PyFLEXTRKR conda environment
# echo 'Activating PyFLEXTRKR environment ...'
# conda activate flextrkr

# Run tracking
echo 'Running PyFLEXTRKR ...'
python ../runscripts/run_generic_tracking.py ${config_demo}
echo 'Tracking is done.'

# Make quicklook plots
echo 'Making quicklook plots ...'
quicklook_dir=${dir_demo}'/quicklooks_trackpaths/'
python ../Analysis/plot_subset_generic_tracks_demo.py -s '1979-06-01' -e '1979-08-31' \
    -c ${config_demo} --figsize 8 4 -p 1 --output ${quicklook_dir}
echo 'View quicklook plots here: '${quicklook_dir}

# Make animation using ffmpeg
echo 'Making animations from quicklook plots ...'
ffmpeg -framerate 4 -pattern_type glob -i ${quicklook_dir}'*.png' -c:v libx264 -r 10 -crf 20 -pix_fmt yuv420p \
    -y ${quicklook_dir}quicklook_animation.mp4
echo 'View animation here: '${quicklook_dir}quicklook_animation.mp4

echo 'Demo completed!'