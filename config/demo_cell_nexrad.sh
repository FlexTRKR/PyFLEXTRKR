#!/bin/bash
###############################################################################################
# This script demonstrates running cell tracking on NEXRAD data (KHGX)
# To run this demo script:
# 1. Modify the dir_demo to a directory on your computer to download the sample data
# 2. Run the script: bash demo_cell_nexrad.sh
# 
# By default the demo config uses 4 processors for parallel processing, 
#    assuming most computers have at least 4 CPU cores. 
#    If your computer has more than 4 processors, you may modify 'nprocesses' 
#    in config_nexrad500m_example.yml to reduce the run time.
###############################################################################################

# Specify directory for the demo data
# dir_demo='/Users/feng045/data/demo/cell_radar/nexrad/'
dir_demo='/pscratch/sd/f/feng045/demo/cell_radar/nexrad/'

# Example config file name
config_demo='config_cell_demo.yml'

# Demo input data directory
dir_input=${dir_demo}'input/'

# Create the demo directory
mkdir -p ${dir_input}

# Download sample NEXRAD reflectivity data:
echo 'Downloading demo input data ...'
wget https://portal.nersc.gov/project/m1867/PyFLEXTRKR/sample_data/radar/nexrad_reflectivity1.tar.gz -O ${dir_input}/nexrad_reflectivity1.tar.gz

# Extract intput data
echo 'Extracting demo input data ...'
tar -xvzf ${dir_input}nexrad_reflectivity1.tar.gz -C ${dir_input}
# Remove downloaded tar file
rm -fv ${dir_input}nexrad_reflectivity1.tar.gz

# Add '\' to each '/' in directory names
dir_input1=$(echo ${dir_input} | sed 's_/_\\/_g')
dir_demo1=$(echo ${dir_demo} | sed 's_/_\\/_g')
# Replace input directory names in example config file
sed 's/INPUT_DIR/'${dir_input1}'/g;s/TRACK_DIR/'${dir_demo1}'/g' config_nexrad500m_example.yml > ${config_demo}
echo 'Created new config file: '${config_demo}

# # Activate PyFLEXTRKR conda environment
# echo 'Activating PyFLEXTRKR environment ...'
# conda activate flextrkr

# Run tracking
echo 'Running PyFLEXTRKR ...'
python ../runscripts/run_celltracking.py ${config_demo}
echo 'Tracking is done.'

# Make quicklook plots
echo 'Making quicklook plots ...'
quicklook_dir=${dir_demo}'/quicklooks_trackpaths/'
# The options of ../visualization/examples/make_cell_animation_nexrad.py: a square map of 2 x 105 km around the radar on a LambertConformal projection
# centered on it (round range circles), title, coastlines, country and state borders
python ../visualization/plot_cell_tracks.py -s '2014-08-07T12' -e '2014-08-07T15' \
    -c ${config_demo} --radar_lat 29.4719 --radar_lon -95.0787 --extent -96.163 -93.994 28.528 30.416 --subset 1 \
    --projection LambertConformal --central_longitude -95.0787 --central_latitude 29.4719 --standard_parallels 29.4719 \
    --title_prefix 'NEXRAD (KHGX)' --figsize 8 7.2 --lon_ticks -96 -95.5 -95 -94.5 -94 --lat_ticks 28.5 29 29.5 30 \
    --draw_land 1 --land_linewidth 1 --draw_border 1 --draw_state 1 --map_resolution 10m --figbasename cell_tracks_ \
    -p 1 --output ${quicklook_dir}
echo 'View quicklook plots here: '${quicklook_dir}

# Make animation using ffmpeg
echo 'Making animations from quicklook plots ...'
# (the scale filter makes the width and height even, which yuv420p needs: the figure sizes of these plots are not always even)
ffmpeg -framerate 4 -pattern_type glob -i ${quicklook_dir}'*.png' -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
    -c:v libx264 -r 10 -crf 20 -pix_fmt yuv420p \
    -y ${quicklook_dir}quicklook_animation.mp4
echo 'View animation here: '${quicklook_dir}quicklook_animation.mp4

echo 'Demo completed!'
