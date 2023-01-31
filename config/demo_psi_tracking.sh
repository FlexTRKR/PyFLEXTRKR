#!/bin/bash
###############################################################################################
# This script demonstrates running PSI anomaly feature tracking
# To run this demo script:
# 1. Modify the dir_demo to a directory on your computer to download the sample data
# 2. Run the script: bash demo_psi_tracking.sh
#
# By default the demo config uses 4 processors for parallel processing,
#    assuming most computers have at least 4 CPU cores.
###############################################################################################

# Specify directory for the demo data
dir_demo='/pscratch/sd/f/feng045/waccem/ERA5_PSI_tracking/'

# Example config file name
config_demo='config_era5_psi.yml'

# Demo input data directory
dir_input=${dir_demo}'input/'

# Create the demo directory
mkdir -p ${dir_input}

# Activate PyFLEXTRKR conda environment
echo 'Activating PyFLEXTRKR environment ...'
conda activate flextrkr

# Run tracking
echo 'Running PyFLEXTRKR ...'
python ../runscripts/run_generic_tracking.py ${config_demo}
echo 'Tracking is done.'

# Make quicklook plots
echo 'Making quicklook plots ...'
quicklook_dir=${dir_demo}'/quicklooks_trackpaths/'
python ../Analysis/plot_subset_psi_tracks_demo.py -s '2012-06-01T00' -e '2012-07-01T00' \
    -c ${config_demo} --figsize 15 3 -p 1 --output ${quicklook_dir}
echo 'View quicklook plots here: '${quicklook_dir}

# Make animation using ffmpeg
echo 'Making animations from quicklook plots ...'
ffmpeg -framerate 8 -pattern_type glob -i ${quicklook_dir}'*.png' -c:v libx264 -r 10 -crf 20 -pix_fmt yuv420p \
    -y ${quicklook_dir}quicklook_animation.mp4
echo 'View animation here: '${quicklook_dir}quicklook_animation.mp4

echo 'Demo completed!'