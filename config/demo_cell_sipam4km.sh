#!/bin/bash
###############################################################################################
# This script demonstrates running cell tracking on GoAmazon SIPAM data coarsen to 4 km
# To run this demo script:
# 1. Modify the dir_demo to a directory on your computer to download the sample data
# 2. Run the script: bash demo_cell_sipam4km.sh
# 
# By default the demo config uses 4 processors for parallel processing, 
#    assuming most computers have at least 4 CPU cores. 
#    If your computer has more than 4 processors, you may modify 'nprocesses' 
#    in config_csapr4km_example.yml to reduce the run time.
###############################################################################################

# Specify directory for the demo data
dir_demo='/pscratch/sd/f/feng045/SAAG/hist/cell_tracking/sipam/'

# Example config file name
config_demo='config_sipam4km_example.yml'

# Demo input data directory
dir_input=${dir_demo}'CAPPI_v2/'

# Create the demo directory
mkdir -p ${dir_input}

# plot_starttime='2014-03-19T06'
# plot_endtime='2014-03-20T00'
plot_starttime='2014-03-17T00'
plot_endtime='2014-03-18T00'
radar_lat=-3.1489
radar_lon=-59.9914

# # Download sample ARM CSAPR reflectivity data:
# echo 'Downloading demo input data ...'
# wget https://portal.nersc.gov/project/m1867/PyFLEXTRKR/sample_data/radar/taranis_corcsapr2.tar.gz -O ${dir_input}/taranis_corcsapr2.tar.gz

# # Extract intput data
# echo 'Extracting demo input data ...'
# tar -xvzf ${dir_input}taranis_corcsapr2.tar.gz -C ${dir_input}
# # Remove downloaded tar file
# rm -fv ${dir_input}taranis_corcsapr2.tar.gz

# # Add '\' to each '/' in directory names
# dir_input1=$(echo ${dir_input} | sed 's_/_\\/_g')
# dir_demo1=$(echo ${dir_demo} | sed 's_/_\\/_g')
# # Replace input directory names in example config file
# sed 's/INPUT_DIR/'${dir_input1}'/g;s/TRACK_DIR/'${dir_demo1}'/g' config_csapr4km_example.yml > ${config_demo}
# echo 'Created new config file: '${config_demo}

# # Activate PyFLEXTRKR conda environment
# echo 'Activating PyFLEXTRKR environment ...'
# conda activate pyflex

# Run tracking
echo 'Running PyFLEXTRKR ...'
python ../runscripts/run_celltracking_lasso.py ${config_demo}
echo 'Tracking is done.'

# Make quicklook plots
echo 'Making quicklook plots ...'
quicklook_dir=${dir_demo}'/quicklooks_trackpaths/'
python ../Analysis/plot_subset_cell_tracks_demo.py -s ${plot_starttime} -e ${plot_endtime} \
    -c ${config_demo} --radar_lat ${radar_lat} --radar_lon ${radar_lon} -p 1 --figsize 8 7 --output ${quicklook_dir}
echo 'View quicklook plots here: '${quicklook_dir}

# Make animation using ffmpeg
echo 'Making animations from quicklook plots ...'
filename_animation=${quicklook_dir}quicklook_animation_${plot_starttime}_${plot_endtime}.mp4
ffmpeg -framerate 2 -pattern_type glob -i ${quicklook_dir}'*.png' -c:v libx264 -r 10 -crf 20 -pix_fmt yuv420p \
    -y ${filename_animation}
echo 'View animation here: '${filename_animation}

echo 'Demo completed!'