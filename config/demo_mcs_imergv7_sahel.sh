#!/bin/bash
###############################################################################################
# This script demonstrates running MCS tracking on GPM IMERG V07B Tb + precipitation data over the Sahel (22-23 August 2025, 0-30N, 20W-60E)
# The input data are made by preprocess/crop_imergv7_sahel_sample.py (their global attributes describe where they come from)
# To run this demo script:
# 1. Modify the dir_demo to a directory on your computer to download the sample data
# 2. Run the script: bash demo_mcs_imergv7_sahel.sh
# 
# By default the demo config uses 4 processors for parallel processing, 
#    assuming most computers have at least 4 CPU cores. 
#    If your computer has more than 4 processors, you may modify 'nprocesses' 
#    in config_imergv7_mcs_tbpf_sahel.yml to reduce the run time.
###############################################################################################

# Specify directory for the demo data
dir_demo='/pscratch/sd/f/feng045/demo/mcs_tbpf/imergv7_sahel/'

# Example config file name
config_demo='config_mcs_demo.yml'

# Demo input data directory
dir_input=${dir_demo}'input/'

# Create the demo directory
mkdir -p ${dir_input}

# Download sample GPM IMERG V07B Tb+Precipitation data (48 hourly files and the land-sea mask, 12 MB):
echo 'Downloading demo input data ...'
wget https://portal.nersc.gov/project/m1867/PyFLEXTRKR/sample_data/tb_pcp/gpm_tb_imergv7_sahel.tar.gz -O ${dir_input}/gpm_tb_imergv7_sahel.tar.gz

# Extract intput data
echo 'Extracting demo input data ...'
tar -xvzf ${dir_input}gpm_tb_imergv7_sahel.tar.gz -C ${dir_input}
# Remove downloaded tar file
rm -fv ${dir_input}gpm_tb_imergv7_sahel.tar.gz

# Add '\' to each '/' in directory names
dir_input1=$(echo ${dir_input} | sed 's_/_\\/_g')
dir_demo1=$(echo ${dir_demo} | sed 's_/_\\/_g')
# Replace input directory names in example config file
sed 's/INPUT_DIR/'${dir_input1}'/g;s/TRACK_DIR/'${dir_demo1}'/g' config_imergv7_mcs_tbpf_sahel.yml > ${config_demo}
echo 'Created new config file: '${config_demo}

# Activate PyFLEXTRKR conda environment
# echo 'Activating PyFLEXTRKR environment ...'
## sourcing conda
#case $(uname -s) in
#	'Linux')     echo "LinUx" && . $HOME/miniconda3/etc/profile.d/conda.sh;;
#	'Darwin')    echo "Mac" && . $HOME/miniconda3/etc/profile.d/conda.sh;;
#        *);;
#esac
# conda activate flextrkr

# Run tracking
echo 'Running PyFLEXTRKR ...'
python ../runscripts/run_mcs_tbpf.py ${config_demo}
echo 'Tracking is done.'

# Make quicklook plots
echo 'Making quicklook plots ...'
quicklook_dir=${dir_demo}'/quicklooks_trackpaths/'
# The options of ../visualization/examples/make_mcs_animation_imergv7_sahel.py: Tb (top) and precipitation (bottom) with a white halo around the tracks;
# the map is the domain of the (cropped) data, no --extent needed
python ../visualization/plot_mcs_tracks.py -s '2025-08-22T00' -e '2025-08-24T00' \
    -c ${config_demo} --product tbpf --panels 2 --layout top_bottom --title_prefix 'GPM Tb+IMERGv07' --draw_border 1 \
    --track_halo_color white --track_halo_alpha 0.7 --track_text_alpha 0.7 --show_pfdiam 1 --map_resolution 50m \
    --figsize_x 10 --dpi 100 --fontsize 12 --figbasename mcs_tbpf_ -p 1 --output ${quicklook_dir}
echo 'View quicklook plots here: '${quicklook_dir}

# Make animation using ffmpeg
echo 'Making animations from quicklook plots ...'
# (the scale filter makes the width and height even, which yuv420p needs: the figure sizes of these plots are not always even)
ffmpeg -framerate 2 -pattern_type glob -i ${quicklook_dir}'*.png' -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
    -c:v libx264 -r 10 -crf 20 -pix_fmt yuv420p \
    -y ${quicklook_dir}quicklook_animation.mp4
echo 'View animation here: '${quicklook_dir}quicklook_animation.mp4

echo 'Demo completed!'
