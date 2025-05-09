#!/bin/bash
###############################################################################################
# This script plots MCS tracking animations
###############################################################################################

# Specify start/end datetime
start_date='2004-06-27T00'
end_date='2004-06-27T11'
# Define domain map extent
lon_min=-110
lon_max=95
lat_min=20
lat_max=40
# Run in parallel
run_parallel=0
# Tracking config file
config_file='/global/homes/f/feng045/program/PyFLEXTRKR-dev/config/config_gridrad_mcs_2004.yml'
# Quicklook/animation output directories
quicklook_dir='/pscratch/sd/f/feng045/usa/gridrad_v3/v3_final/quicklook_test/2004/'
animation_dir='/pscratch/sd/f/feng045/usa/gridrad_v3/v3_final/quicklook_test/'
animation_filename=${animation_dir}mcs_tbdbz_${start_date}_${end_date}.mp4
# Tracking pixel-level time format
time_format='yyyymodd_hhmmss'
# Plotting code
code_name='/global/homes/f/feng045/program/PyFLEXTRKR-dev/Analysis/plot_subset_tbze_mcs_tracks_1panel_demo.py'

# Make quicklook plots
echo 'Making quicklook plots ...'
python ${code_name} -s ${start_date} -e ${end_date} -c ${config_file} \
    --extent ${extent} ${lon_min} ${lon_max} ${lat_min} ${lat_max} \
    --subset 1 \
    --time_format ${time_format} \
    -p ${run_parallel} \
    --output ${quicklook_dir}
echo 'View quicklook plots here: '${quicklook_dir}

# # Make animation using ffmpeg
# mkdir -p ${animation_dir}
# echo 'Making animations from quicklook plots ...'
# ffmpeg -framerate 2 -pattern_type glob -i ${quicklook_dir}'*.png' -c:v libx264 -r 10 -crf 20 -pix_fmt yuv420p \
#     -y ${animation_filename}
# echo 'View animation here: '${animation_filename}