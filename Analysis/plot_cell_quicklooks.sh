#!/bin/bash
###############################################################################################
# This script plots CACTI CSAPR2 cell tracking animations
###############################################################################################

# Specify start/end datetime
start_date='2018-11-13T12'
end_date='2018-11-14T00'
radar_lat=-32.12641
radar_lon=-64.72837
run_parallel=1
# Tracking config file
config_file='/ccsopen/home/zhe1feng1/program/pyflex_config/config/config_csapr500m_cacti_fullcampaign.yml'
# Quicklook/animation output directories
quicklook_dir='/gpfs/wolf2/arm/atm131/proj-shared/zfeng/cacti/csapr/quicklooks_trackpaths/20181113/'
animation_dir='/gpfs/wolf2/arm/atm131/proj-shared/zfeng/cacti/csapr/quicklooks_trackpaths/animation/'
animation_filename=${animation_dir}dbz_comp_20181113.mp4
# Tracking pixel-level time format
time_format='yyyymodd_hhmmss'
# Variable name to plot in the pixel-level files
# varname='comp_ref'
varname='dbz_comp'
# Plotting code
code_name='/ccsopen/home/zhe1feng1/program/PyFLEXTRKR-dev/Analysis/plot_subset_cell_tracks_demo.py'

# Make quicklook plots
echo 'Making quicklook plots ...'
python ${code_name} -s ${start_date} -e ${end_date} -c ${config_file} \
    --time_format ${time_format} --varname ${varname} \
    --radar_lat ${radar_lat} --radar_lon ${radar_lon} -p ${run_parallel} --figsize 8 7 --output ${quicklook_dir}
echo 'View quicklook plots here: '${quicklook_dir}

# Make animation using ffmpeg
echo 'Making animations from quicklook plots ...'
ffmpeg -framerate 2 -pattern_type glob -i ${quicklook_dir}'*.png' -c:v libx264 -r 10 -crf 20 -pix_fmt yuv420p \
    -y ${animation_filename}
echo 'View animation here: '${animation_filename}