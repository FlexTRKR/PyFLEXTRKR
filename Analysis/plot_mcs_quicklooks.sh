#!/bin/bash
###############################################################################################
# This script plots MCS tracking animations
###############################################################################################

# Specify start/end datetime
start_date='2019-01-20T00'
end_date='2019-01-23T00'
orientation='horizontal'
run_parallel=1
# Tracking config file
config_file='/global/homes/f/feng045/program/pyflex_config/config/config_imerg_mcs_tbpf_2019.yml'
# Quicklook/animation output directories
quicklook_dir='/pscratch/sd/f/feng045/waccem/mcs_global/quicklooks/2019/'
animation_dir='/pscratch/sd/f/feng045/waccem/mcs_global/quicklooks/animation/'
animation_filename=${animation_dir}mcs_tbpf_${start_date}_${end_date}.mp4
# Tracking pixel-level time format
time_format='yyyymodd_hhmm'
# Plotting code
code_name='/global/homes/f/feng045/program/PyFLEXTRKR-dev/Analysis/plot_subset_tbpf_mcs_tracks_demo.py'

# Make quicklook plots
# --extent specifies subset region: minlon maxlon minlat maxlat (cannot have decimals yet)
echo 'Making quicklook plots ...'
python ${code_name} -s ${start_date} -e ${end_date} -c ${config_file} \
    --extent ${extent} 90 140 -10 10 \
    --subset 1 \
    --time_format ${time_format} \
    -o ${orientation} \
    -p ${run_parallel} \
    --output ${quicklook_dir}
echo 'View quicklook plots here: '${quicklook_dir}

# Make animation using ffmpeg
mkdir -p ${animation_dir}
echo 'Making animations from quicklook plots ...'
ffmpeg -framerate 2 -pattern_type glob -i ${quicklook_dir}'*.png' -c:v libx264 -r 10 -crf 20 -pix_fmt yuv420p \
    -y ${animation_filename}
echo 'View animation here: '${animation_filename}