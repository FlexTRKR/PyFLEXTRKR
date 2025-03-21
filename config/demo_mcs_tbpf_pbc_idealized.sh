#!/bin/bash
###############################################################################################
# This script demonstrates running MCS tracking on idealized Tb + precipitation data 
# with periodic boundary conditions (PBC)
# To run this demo script:
# 1. Modify the dir_demo to a directory on your computer to download the sample data
# 2. Modify the test input basename
# 3. Run the script: bash demo_mcs_tbpf_idealized.sh
#
# By default the demo config uses 4 processors for parallel processing,
#    assuming most computers have at least 4 CPU cores.
###############################################################################################

# Specify start/end datetime
start_date='2020-01-01T00' 
end_date='2020-01-03T00' 

# Plotting map domain (lonmin lonmax latmin latmax)
map_extent='0. 1500 0. 1500'  # (xmin xmax ymin ymax)
run_parallel=1

# Demo output directory
dir_demo='/pscratch/sd/p/paccini/temp/output_tracking/tracking_mcs_idealized_demo_v2/' 
quicklook_dir=${dir_demo}'/quicklooks_trackpaths/'
animation_dir=${dir_demo}'/animations/'
animation_filename=${animation_dir}mcs_tracking2_${start_date}_${end_date}.mp4

# Example config file name
config_demo='./config_mcs_pbc_idealized.yml'

# Make quicklook & animation directories
mkdir -p ${quicklook_dir}
mkdir -p ${animation_dir}

# Run tracking
echo 'Running PyFLEXTRKR ...'
python ../runscripts/run_mcs_tbpf_mcsmip.py ${config_demo}
echo 'Tracking is done.'

# Make quicklook plots
echo 'Making quicklook plots ...'
python ../Analysis/plot_subset_tbpf_demo_pbc.py -s ${start_date} -e ${end_date} \
    -c ${config_demo} -p ${run_parallel}  --output ${quicklook_dir} \
    --extent "${map_extent}" --subset 0
echo 'View quicklook plots here: '${quicklook_dir}

# Make animation using ffmpeg
vfscale='1200:-1'
framerate=2
echo 'Making animations from quicklook plots ...'
ffmpeg -framerate ${framerate} -pattern_type glob -i ${quicklook_dir}'*.png' -c:v libx264 -r 10 -crf 20 -pix_fmt yuv420p \
    -y ${animation_filename}
echo 'View animation here: '${animation_filename}

echo 'Demo completed!'