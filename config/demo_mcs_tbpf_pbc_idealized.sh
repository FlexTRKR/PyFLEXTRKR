#!/bin/bash
###############################################################################################
# This script demonstrates running MCS tracking on idealized Tb + precipitation data 
# with periodic boundary conditions (PBC)
# To run this demo script:
# 1. Modify the dir_demo to a directory on your computer to download the sample data
# 2. Modify the test input basename
# 3. Run the script: bash demo_mcs_tbpf_pbc_idealized.sh
#
# By default the demo runs the script in serial mode.
###############################################################################################

# Specify directory for the demo data
dir_demo='/pscratch/sd/f/feng045/demo/mcs_tbpf/idealized/periodic_boundary/'

# Example config file name
config_demo='./config_mcs_pbc_idealized_demo.yml'

# Demo input data directory
dir_input=${dir_demo}'input/'

# Create the demo directory
mkdir -p ${dir_input}

# Download idealized Tb+Precipitation data:
echo 'Downloading demo input data ...'
wget https://portal.nersc.gov/cfs/m1867/PyFLEXTRKR/sample_data/tb_pcp/MCS_idealized_periodic_2020-01-01_00-00-00.nc -O ${dir_input}/MCS_idealized_periodic_2020-01-01_00-00-00.nc

# Specify start/end datetime
start_date='2020-01-01T00' 
end_date='2020-01-03T00' 

# Plotting map domain (lonmin lonmax latmin latmax)
map_extent='0. 1500 0. 1500'  # (xmin xmax ymin ymax)
run_parallel=0

# Demo output directory
quicklook_dir=${dir_demo}'/quicklooks_trackpaths/'
animation_dir=${dir_demo}'/animations/'
animation_filename=${animation_dir}mcs_tracking_${start_date}_${end_date}.mp4

# Add '\' to each '/' in directory names
dir_input1=$(echo ${dir_input} | sed 's_/_\\/_g')
dir_demo1=$(echo ${dir_demo} | sed 's_/_\\/_g')
# Replace input directory names in example config file
sed 's/INPUT_DIR/'${dir_input1}'/g;s/TRACK_DIR/'${dir_demo1}'/g' config_mcs_pbc_idealized.yml > ${config_demo}
echo 'Created new config file: '${config_demo}

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