#!/bin/bash
###############################################################################################
# This script demonstrates running WRF MCS tracking (2D reflectivity)
# To run this demo script:
# 1. Modify the dir_demo to a directory on your computer to download the sample data
# 2. Run the script: bash demo_uw_wrf_mcstracking.sh
#
# By default the demo config uses 12 processors for parallel processing.
###############################################################################################

# Specify start/end datetime
start_date='2024-08-17T12'
end_date='2024-08-18T13'
# Plotting map domain (lonmin lonmax latmin latmax)
map_extent='-126.5 -116. 41.5 49.5'

# Specify directory for the demo data
dir_demo='/pscratch/sd/s/smheflin/uw_wrf/for_tracking_1.33km/'
quicklook_dir='/pscratch/sd/f/feng045/usa/UW_WRF/quicklooks_trackpaths/'
animation_dir='/pscratch/sd/f/feng045/usa/UW_WRF/animations/'
animation_filename=${animation_dir}mcs_tracking_${start_date}_${end_date}.mp4

# Make quicklook & animation directories
mkdir -p ${quicklook_dir}
mkdir -p ${animation_dir}

# Example config file name
config_file='config_uw_wrf_reflectivity.yml'
# Plotting code name
plot_code='/global/homes/f/feng045/program/PyFLEXTRKR-dev/Analysis/plot_subset_generic_tracks_demo.py'

# # Activate PyFLEXTRKR conda environment
# echo 'Activating PyFLEXTRKR environment ...'
# conda activate flextrkr

# # Run tracking
# echo 'Running PyFLEXTRKR ...'
# python ../runscripts/run_generic_tracking.py ${config_file}
# echo 'Tracking is done.'

# Make quicklook plots
echo 'Making quicklook plots ...'

python ${plot_code} -s ${start_date} -e ${end_date} -c ${config_file} \
    -p 1 --output ${quicklook_dir} --extent "${map_extent}" --subset 0
echo 'View quicklook plots here: '${quicklook_dir}

# Make animation using ffmpeg
# Animation settings
vfscale='1200:-1'
framerate=2
# echo 'Making animations from quicklook plots ...'
ffmpeg -framerate ${framerate} -pattern_type glob -i ${quicklook_dir}'*.png' -c:v libx264 -r 10 -crf 20 -pix_fmt yuv420p \
    -y  ${animation_filename}
echo 'View animation here: '${animation_filename}

echo 'Demo completed!'