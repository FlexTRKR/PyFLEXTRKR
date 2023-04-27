#!/bin/bash
###############################################################################################
# This script demonstrates running MCS tracking on idealized Tb + precipitation data
# To run this demo script:
# 1. Modify the dir_demo to a directory on your computer to download the sample data
# 2. Modify the test input basename
# 3. Run the script: bash demo_mcs_tbpf_idealized.sh
#
# By default the demo config uses 4 processors for parallel processing,
#    assuming most computers have at least 4 CPU cores.
###############################################################################################

# Specify directory for the demo data
# There are a total of 4 tests (e.g., test1, test2, test3, test4)
dir_demo='/Users/feng045/data/demo/mcs_tbpf/idealized/test2/'

# Test input file basename
# There are a total of 4 tests (e.g., 'MCS-test-1_', 'MCS-test-2_', etc.)
data_basename='MCS-test-2_'

# Example config file name
config_demo='config_mcs_demo.yml'

# Demo input data directory
dir_input=${dir_demo}'input/'

# Create the demo directory
mkdir -p ${dir_input}

# Download sample WRF Tb+Precipitation data:
echo 'Downloading demo input data ...'
wget https://portal.nersc.gov/project/m1867/PyFLEXTRKR/sample_data/tb_pcp/idealized_tbpcp.tar.gz -O ${dir_input}/idealized_tbpcp.tar.gz

# Extract intput data
echo 'Extracting demo input data ...'
tar -xvzf ${dir_input}idealized_tbpcp.tar.gz -C ${dir_input} ${data_basename}*.nc
# Remove downloaded tar file
rm -fv ${dir_input}idealized_tbpcp.tar.gz

# Add '\' to each '/' in directory names
dir_input1=$(echo ${dir_input} | sed 's_/_\\/_g')
dir_demo1=$(echo ${dir_demo} | sed 's_/_\\/_g')
# Replace input directory names in example config file
sed 's/INPUT_DIR/'${dir_input1}'/g;s/TRACK_DIR/'${dir_demo1}'/g;s/BASENAME/'${data_basename}'/g' config_mcs_idealized.yml > ${config_demo}
echo 'Created new config file: '${config_demo}

# Activate PyFLEXTRKR conda environment
echo 'Activating PyFLEXTRKR environment ...'
conda activate flextrkr

# Run tracking
echo 'Running PyFLEXTRKR ...'
python ../runscripts/run_mcs_tbpf.py ${config_demo}
echo 'Tracking is done.'

# Make quicklook plots
echo 'Making quicklook plots ...'
quicklook_dir=${dir_demo}'/quicklooks_trackpaths/'
python ../Analysis/plot_subset_tbpf_mcs_tracks_demo.py -s '2020-01-01T00' -e '2020-01-02T00' \
    -c ${config_demo} -o vertical -p 1 --figsize 10 8 --output ${quicklook_dir}
echo 'View quicklook plots here: '${quicklook_dir}

# Make animation using ffmpeg
echo 'Making animations from quicklook plots ...'
ffmpeg -framerate 2 -pattern_type glob -i ${quicklook_dir}'*.png' -c:v libx264 -r 10 -crf 20 -pix_fmt yuv420p \
    -y ${quicklook_dir}quicklook_animation.mp4
echo 'View animation here: '${quicklook_dir}quicklook_animation.mp4

echo 'Demo completed!'