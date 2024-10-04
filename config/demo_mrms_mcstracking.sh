#!/bin/bash
###############################################################################################
# This script demonstrates running MRMS MCS tracking (2D reflectivity)
# To run this demo script:
# 1. Modify the dir_demo to a directory on your computer to download the sample data
# 2. Run the script: bash demo_mrms_mcstracking.sh
#
# By default the demo config uses 128 processors for parallel processing.
###############################################################################################

# Specify directory for the demo data
dir_demo='/pscratch/sd/f/feng045/usa/MRMS/'

# Example config file name
config_demo='config_mrms_mcs_composite_reflectivity.yml'

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
python ../Analysis/plot_subset_generic_tracks_demo.py -s '2024-08-17T14' -e '2024-08-18T13' \
    -c ${config_demo} -p 1 --output ${quicklook_dir} --extent 230. 244. 40. 50.
echo 'View quicklook plots here: '${quicklook_dir}

# Make animation using ffmpeg
# Animation settings
vfscale='1200:-1'
framerate=2

echo 'Making animations from quicklook plots ...'
ffmpeg -framerate ${framerate} -pattern_type glob -i ${quicklook_dir}'*.png' -c:v libx264 -r 10 -crf 20 -pix_fmt yuv420p \
    -y ${quicklook_dir}quicklook_animation.mp4
echo 'View animation here: '${quicklook_dir}quicklook_animation.mp4

echo 'Demo completed!'