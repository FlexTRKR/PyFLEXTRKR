#!/bin/bash
###############################################################################################
# This script plots MCS tracking animations
# Set config_file, quicklook_dir and animation_dir to your own tracking run: the /path/to/... values are placeholders.
###############################################################################################

# The PyFLEXTRKR repository (this script is in visualization/examples/)
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

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
config_file="${repo_dir}/config/config_mcs_tbpf_scream_healpix9.yml"
# Quicklook/animation output directories
quicklook_dir='/path/to/mcs_tracking_hp9/quicklooks/'
animation_dir='/path/to/mcs_tracking_hp9/animation/'
animation_filename=${animation_dir}mcs_tbpf_${start_date}_${end_date}.mp4
# Tracking pixel-level time format
time_format='yyyymodd_hhmmss'
# Plotting code: in the repository, found from the location of this script (so it runs from any folder)
code_name=${repo_dir}/visualization/plot_mcs_tracks.py

# Make quicklook plots
echo 'Making quicklook plots ...'
python ${code_name} -s ${start_date} -e ${end_date} -c ${config_file} \
    --product tbpf --panels 1 \
    --extent ${lon_min} ${lon_max} ${lat_min} ${lat_max} \
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