#!/bin/bash
###############################################################################################
# This script plots MCS tracking animations
###############################################################################################

# Specify start/end datetime
start_date='2020-06-21T00'
end_date='2020-06-24T00'
# Define domain map extent
lon_min=-20.0
lon_max=40.0
lat_min=-5.0
lat_max=25.0
# Run in parallel
run_parallel=1
# Tracking config file
config_file='/global/homes/f/feng045/program/PyFLEXTRKR-dev/config/config_mcs_tbpf_scream_healpix9.yml'
# Quicklook/animation output directories
quicklook_dir='/pscratch/sd/w/wcmca1/scream-cess-healpix/mcs_tracking_hp9/quicklooks/'
animation_dir='/pscratch/sd/w/wcmca1/scream-cess-healpix/mcs_tracking_hp9/animation/'
animation_filename=${animation_dir}mcs_tbpf_${start_date}_${end_date}.mp4
# Tracking pixel-level time format
time_format='yyyymodd_hhmmss'
# Plotting code
code_name='/global/homes/f/feng045/program/PyFLEXTRKR-dev/Analysis/plot_subset_tbpf_mcs_tracks_1panel_demo.py'

# Make quicklook plots
echo 'Making quicklook plots ...'
python ${code_name} -s ${start_date} -e ${end_date} -c ${config_file} \
    --extent ${extent} ${lon_min} ${lon_max} ${lat_min} ${lat_max} \
    --subset 1 \
    --time_format ${time_format} \
    -p ${run_parallel} \
    --output ${quicklook_dir}
echo 'View quicklook plots here: '${quicklook_dir}

# Make animation using ffmpeg
mkdir -p ${animation_dir}
echo 'Making animations from quicklook plots ...'
ffmpeg -framerate 2 -pattern_type glob -i ${quicklook_dir}'*.png' -c:v libx264 -r 10 -crf 20 -pix_fmt yuv420p \
    -y ${animation_filename}
echo 'View animation here: '${animation_filename}