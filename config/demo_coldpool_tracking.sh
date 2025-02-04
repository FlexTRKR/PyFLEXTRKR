#!/bin/bash
###############################################################################################
# This script demonstrates running cold pool tracking from PINACLES
# To run this demo script:
# 1. Modify the dir_demo to a directory on your computer to download the sample data
# 2. Run the script: bash demo_mrms_mcstracking.sh
#
# By default the demo config uses 128 processors for parallel processing.
###############################################################################################

# Specify start/end datetime
# start_date='2020-05-01T00'
# end_date='2020-09-01T00'
start_date='2000-01-30T00' #'2000-01-26T00' #
end_date='2000-02-01T00' #'2000-01-28T00' #
# Plotting map domain (lonmin lonmax latmin latmax)
# map_extent='0. 6. 0. 6.'
map_extent='0. 600. 0. 600.'  # (xmin xmax ymin ymax)
run_parallel=0

# Specify directory for the demo data
dir_demo='/Users/pacc275/local_documents/output_tracking//tracking_coldpool_sub_testPOS_t4/'
quicklook_dir=${dir_demo}'/quicklooks_trackpaths/'
animation_dir=${dir_demo}'/animations/'
animation_filename=${animation_dir}cp_tracking_${start_date}_${end_date}.mp4

# Make quicklook & animation directories
mkdir -p ${quicklook_dir}
mkdir -p ${animation_dir}

# Example config file name
config_file='/Users/pacc275/Library/CloudStorage/OneDrive-PNNL/Documents/repositories/PyFLEXTRKR-dev/config/config_coldpool_buoyancy.yml'

# Activate PyFLEXTRKR conda environment
# echo 'Activating PyFLEXTRKR environment ...'
# source activate pyflex

# Run tracking
echo 'Running PyFLEXTRKR ...'
python /Users/pacc275/Library/CloudStorage/OneDrive-PNNL/Documents/repositories/PyFLEXTRKR-dev/runscripts/run_generic_tracking.py ${config_file}
echo 'Tracking is done.'

# Make quicklook plots
echo 'Making quicklook plots ...'
python /Users/pacc275/Library/CloudStorage/OneDrive-PNNL/Documents/repositories/PyFLEXTRKR-dev/Analysis/plot_subset_coldpool_tracks_nomap.py -s ${start_date} -e ${end_date} -c ${config_file} \
    -p ${run_parallel} --output ${quicklook_dir} \
    --extent "${map_extent}" --subset 0
echo 'View quicklook plots here: '${quicklook_dir}

# Make animation using ffmpeg
# Animation settings
vfscale='1200:-1'
framerate=1
echo 'Making animations from quicklook plots ...'
ffmpeg -framerate ${framerate} -pattern_type glob -i ${quicklook_dir}'*.png' -c:v libx264 -r 10 -crf 20 -pix_fmt yuv420p \
    -y ${animation_filename}
echo 'View animation here: '${animation_filename}

echo 'Demo completed!'