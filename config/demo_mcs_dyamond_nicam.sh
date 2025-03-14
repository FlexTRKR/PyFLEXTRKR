#!/bin/bash
###############################################################################################
# This script demonstrates running MCS tracking on DYAMOND NICAM OLR + precipitation data
# To run this demo script:
# 1. Modify the dir_demo to a directory on your computer to download the sample data
# 2. Run the script: bash demo_mcs_dyamond_nicam.sh
# 
# By default the demo config uses 24 processors for parallel processing, 
#    assuming most computers have at least 24 CPU cores. 
#    If your computer has more than 4 processors, you may modify 'nprocesses' 
#    in config_dyamond_nicam_example.yml to reduce the run time.
###############################################################################################

# Specify directory for the demo data
dir_demo='/pscratch/sd/f/feng045/demo/mcs_tbpf/nicam/'

# Example config file name
config_demo='config_mcs_demo.yml'

# Demo input data directory
dir_input=${dir_demo}'input/'

# Create the demo directory
mkdir -p ${dir_input}

# Download sample DYAMOND NICAM OLR+Precipitation data:
echo 'Downloading demo input data ...'
wget https://portal.nersc.gov/project/m1867/PyFLEXTRKR/sample_data/tb_pcp/dyamond_nicam_tbpcp.tar.gz -O ${dir_input}/dyamond_nicam_tbpcp.tar.gz

# Extract intput data
echo 'Extracting demo input data ...'
tar -xvzf ${dir_input}dyamond_nicam_tbpcp.tar.gz -C ${dir_input}
# Remove downloaded tar file
rm -fv ${dir_input}dyamond_nicam_tbpcp.tar.gz

# Add '\' to each '/' in directory names
dir_input1=$(echo ${dir_input} | sed 's_/_\\/_g')
dir_demo1=$(echo ${dir_demo} | sed 's_/_\\/_g')
# Replace input directory names in example config file
sed 's/INPUT_DIR/'${dir_input1}'/g;s/TRACK_DIR/'${dir_demo1}'/g' config_dyamond_nicam_example.yml > ${config_demo}
echo 'Created new config file: '${config_demo}

# # Activate PyFLEXTRKR conda environment
# echo 'Activating PyFLEXTRKR environment ...'
# source activate flextrkr

# Run tracking
echo 'Running PyFLEXTRKR ...'
python ../runscripts/run_mcs_tbpf.py ${config_demo}
echo 'Tracking is done.'

# Make quicklook plots
# Plotting subset domain boundary
figbasename='spcz_'
minlon=120.0
maxlon=230.0
minlat=-30.0
maxlat=15.0
# Start/end dates
startdate='2020-02-01T00'
enddate='2020-02-02T00'

echo 'Making quicklook plots ...'
quicklook_dir=${dir_demo}'/quicklooks_trackpaths/'
python ../Analysis/plot_subset_tbpf_mcs_tracks_demo.py -s ${startdate} -e ${enddate} \
    -c ${config_demo} \
    -o horizontal --figsize 10 10 \
    -p 1 \
    --extent ${minlon} ${maxlon} ${minlat} ${maxlat} \
    --figbasename ${figbasename} \
    --output ${quicklook_dir}
echo 'View quicklook plots here: '${quicklook_dir}

# Make animation using ffmpeg
echo 'Making animations from quicklook plots ...'
ffmpeg -framerate 2 -pattern_type glob -i ${quicklook_dir}'*.png' -c:v libx264 -r 10 -crf 20 -pix_fmt yuv420p \
    -vf "scale=1200:-1" \
    -y ${quicklook_dir}quicklook_animation.mp4
echo 'View animation here: '${quicklook_dir}quicklook_animation.mp4

echo 'Demo completed!'