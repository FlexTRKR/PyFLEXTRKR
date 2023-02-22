#!/bin/bash
###############################################################################################
# This script demonstrates running MCS tracking on WRF Tb + radar data
# To run this demo script:
# 1. Modify the dir_demo to a directory on your computer to download the sample data
# 2. Run the script: bash demo_mcs_wrf_tbradar.sh
# 
# By default the demo config uses 8 processors for parallel processing,
#    assuming most computers have at least 8 CPU cores.
#    You may adjust 'nprocesses' in the config_wrf_mcs_tbradar_example.yml.
#    Running with more processors will run the demo faster since the demo WRF data is 
#    a real simulation with a decent size domain 1000 x 1000 (lat x lon) and 96 time frames.
###############################################################################################

# Specify directory for the demo data
dir_demo='/Users/feng045/data/demo/mcs_tbpfradar3d/wrf/'

# Example config file name
config_demo='config_mcs_demo.yml'

# Demo input data directory
dir_input=${dir_demo}'input/'

# Create the demo directory
mkdir -p ${dir_input}

# Download sample WRF Tb+Precipitation data:
echo 'Downloading demo input data ...'
wget https://portal.nersc.gov/project/m1867/PyFLEXTRKR/sample_data/tb_radar/wrf_tbradar.tar.gz \
  -O ${dir_input}/wrf_tbradar.tar.gz

# Extract intput data
echo 'Extracting demo input data ...'
tar -xvzf ${dir_input}wrf_tbradar.tar.gz -C ${dir_input}
# Remove downloaded tar file
rm -fv ${dir_input}wrf_tbradar.tar.gz

# Add '\' to each '/' in directory names
dir_input1=$(echo ${dir_input} | sed 's_/_\\/_g')
dir_demo1=$(echo ${dir_demo} | sed 's_/_\\/_g')
# Replace input directory names in example config file
sed 's/INPUT_DIR/'${dir_input1}'/g;s/TRACK_DIR/'${dir_demo1}'/g' config_wrf_mcs_tbradar_example.yml > ${config_demo}
echo 'Created new config file: '${config_demo}

# Activate PyFLEXTRKR conda environment
echo 'Activating PyFLEXTRKR environment ...'
conda activate flextrkr

# Run tracking
echo 'Running PyFLEXTRKR ...'
python ../runscripts/run_mcs_tbpfradar3d_wrf.py ${config_demo}
echo 'Tracking is done.'

# Make quicklook plots
echo 'Making quicklook plots ...'
quicklook_dir=${dir_demo}'/quicklooks_trackpaths/'
python ../Analysis/plot_subset_tbze_mcs_tracks_demo.py -s '2015-05-06T00' -e '2015-05-10T00' \
    -c ${config_demo} -o horizontal -p 1 --figsize 8 12 --output ${quicklook_dir}
echo 'View quicklook plots here: '${quicklook_dir}

# Make animation using ffmpeg
echo 'Making animations from quicklook plots ...'
ffmpeg -framerate 2 -pattern_type glob -i ${quicklook_dir}'*.png' -c:v libx264 -r 10 -crf 20 -pix_fmt yuv420p \
    -y ${quicklook_dir}quicklook_animation.mp4
echo 'View animation here: '${quicklook_dir}quicklook_animation.mp4

echo 'Demo completed!'