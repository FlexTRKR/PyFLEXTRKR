#!/bin/bash
###############################################################################################
# This script demonstrates running MCS tracking on WRF Tb + precipitation data 
# regridded on a specified grid.
# To run this demo script:
# 1. Modify the dir_demo to a directory on your computer to download the sample data
# 2. Run the script: bash demo_mcs_wrf_tbpf_regrid.sh
# 
# By default the demo config uses 4 processors for parallel processing, 
#    assuming most computers have at least 4 CPU cores. 
#    If your computer has more than 4 processors, you may modify 'nprocesses' 
#    in config_wrfregrid_mcs_tbpf_example.yml to reduce the run time.
###############################################################################################

# Specify directory for the demo data
dir_demo='/pscratch/sd/f/feng045/demo/mcs_tbpf/wrf_regrid/'

# Example config file name
config_demo='config_mcs_demo.yml'

# Demo input data directory
dir_input=${dir_demo}'input/'

# Create the demo directory
mkdir -p ${dir_input}

# Download sample WRF Tb+Precipitation data:
echo 'Downloading demo input data ...'
wget https://portal.nersc.gov/project/m1867/PyFLEXTRKR/sample_data/tb_pcp/wrfout.tar.gz -O ${dir_input}/wrfout.tar.gz

# Extract intput data
echo 'Extracting demo input data ...'
tar -xvzf ${dir_input}wrfout.tar.gz -C ${dir_input}
# Remove downloaded tar file
rm -fv ${dir_input}wrfout.tar.gz

# Add '\' to each '/' in directory names
dir_input1=$(echo ${dir_input} | sed 's_/_\\/_g')
dir_demo1=$(echo ${dir_demo} | sed 's_/_\\/_g')
# Replace input directory names in example config file
sed 's/INPUT_DIR/'${dir_input1}'/g;s/TRACK_DIR/'${dir_demo1}'/g' config_wrfregrid_mcs_tbpf_example.yml > ${config_demo}
echo 'Created new config file: '${config_demo}

# Activate PyFLEXTRKR conda environment
echo 'Activating PyFLEXTRKR environment ...'
conda activate flextrkr

# Run tracking
echo 'Running PyFLEXTRKR ...'
python ../runscripts/run_mcs_tbpf_wrf.py ${config_demo}
echo 'Tracking is done.'

# Make quicklook plots
echo 'Making quicklook plots ...'
quicklook_dir=${dir_demo}'/quicklooks_trackpaths/'
python ../Analysis/plot_subset_tbpf_mcs_tracks_demo.py -s '2014-04-02T00' -e '2014-04-04T00' \
   -c ${config_demo} -o horizontal -p 1 --figsize 8 10 --output ${quicklook_dir}
echo 'View quicklook plots here: '${quicklook_dir}

# Make animation using ffmpeg
echo 'Making animations from quicklook plots ...'
ffmpeg -framerate 2 -pattern_type glob -i ${quicklook_dir}'*.png' -c:v libx264 -r 10 -crf 20 -pix_fmt yuv420p \
   -y ${quicklook_dir}quicklook_animation.mp4
echo 'View animation here: '${quicklook_dir}quicklook_animation.mp4

echo 'Demo completed!'