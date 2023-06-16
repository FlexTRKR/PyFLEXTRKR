#!/bin/bash
#SBATCH --job-name=vol_demo_cell_nerxrad_0
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH -N 1
#SBATCH -n 9
#SBATCH --output=./R_%x.out
#SBATCH --error=./R_%x.err

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


# export TMPDIR=/scratch/$USER

## Prepare Test Directories
TEST_NAME='wrf_tbradar'

# Specify directory for the demo data
dir_demo="/qfs/projects/oddite/tang584/flextrkr_runs/${TEST_NAME}" #NFS
mkdir -p $dir_demo
# Example config file name
# config_example='config_wrf_mcs_tbradar_example.yml'
config_example='config_wrf_mcs_tbradar_short.yml'
config_demo='config_wrf_mcs_tbradar_demo.yml'
cp ./$config_demo $dir_demo
# Demo input data directory
dir_input="/qfs/projects/oddite/tang584/flextrkr_runs/input_data/${TEST_NAME}"


PREPARE_CONFIG () {

    # Add '\' to each '/' in directory names
    dir_input1=$(echo ${dir_input} | sed 's_/_\\/_g')
    dir_demo1=$(echo ${dir_demo} | sed 's_/_\\/_g')
    # Replace input directory names in example config file
    sed 's/INPUT_DIR/'${dir_input1}'/g;s/TRACK_DIR/'${dir_demo1}'/g' ${config_example} > ${config_demo}
    echo 'Created new config file: '${config_demo}
}

RUN_TRACKING () {
    # Run tracking
    echo 'Running PyFLEXTRKR w/ VOL ...'


    # LD_LIBRARY_PATH=$TRACKER_VOL_DIR:$LD_LIBRARY_PATH \
    #     HDF5_VOL_CONNECTOR="${VOL_NAME} under_vol=0;under_info={};path=${SCRIPT_DIR}/vol-${task_id}_${FUNCNAME[0]}.log;level=2;format=" \
    #     HDF5_DRIVER=hdf5_hermes_vfd \
    #     HDF5_PLUGIN_PATH=$TRACKER_VOL_DIR:${HERMES_INSTALL_DIR}/lib \

    # LD_LIBRARY_PATH=$TRACKER_VOL_DIR:$LD_LIBRARY_PATH \
    #     HDF5_VOL_CONNECTOR="${VOL_NAME} under_vol=0;under_info={};path=./vol-${FUNCNAME[0]}.log;level=2;format=" \
    #     HDF5_PLUGIN_PATH=$TRACKER_VOL_DIR \
    # HDF5_DRIVER=hdf5_hermes_vfd \
    #     HDF5_PLUGIN_PATH=${HERMES_INSTALL_DIR}/lib \
    #     HERMES_CONF=$HERMES_CONF \
    #     HERMES_CLIENT_CONF=$HERMES_CLIENT_CONF \
    
    set -x

    schema_file=data-stat-vol.yaml
    rm -rf ./*$schema_file
    # touch $schema_file

    # LD_PRELOAD=/share/apps/gcc/9.1.0/lib64/libasan.so \
    LD_LIBRARY_PATH=$TRACKER_VOL_DIR:$LD_LIBRARY_PATH \
        HDF5_VOL_CONNECTOR="${VOL_NAME} under_vol=0;under_info={};path=${schema_file};level=2;format=" \
        HDF5_PLUGIN_PATH=$TRACKER_VOL_DIR:$HDF5_PLUGIN_PATH \
        python ../runscripts/run_mcs_tbpfradar3d_wrf.py ${config_demo} &> ${FUNCNAME[0]}-vol.log
    
    set +x 

    echo 'Tracking is done.'
}

MAKE_QUICKLOOK_PLOTS () {
    # Make quicklook plots
    echo 'Making quicklook plots ...'
    quicklook_dir=${dir_demo}'/quicklooks_trackpaths/'
    python ../Analysis/plot_subset_tbze_mcs_tracks_demo.py -s '2015-05-06T00' -e '2015-05-10T00' \
        -c ${config_demo} -o horizontal -p 1 --figsize 8 12 --output ${quicklook_dir}
    echo 'View quicklook plots here: '${quicklook_dir}
}

MAKE_ANIMATION () {
    # Make animation using ffmpeg
    echo 'Making animations from quicklook plots ...'
    ffmpeg -framerate 2 -pattern_type glob -i ${quicklook_dir}'*.png' -c:v libx264 -r 10 -crf 20 -pix_fmt yuv420p \
        -y ${quicklook_dir}quicklook_animation.mp4
    echo 'View animation here: '${quicklook_dir}quicklook_animation.mp4
}


# source ./load_hermes_deps.sh
source ./env_var.sh

# # Activate PyFLEXTRKR conda environment
echo 'Activating PyFLEXTRKR environment ...'
source activate pyflextrkr_copy # flextrkr pyflextrkr


# export PYTHONLOGLEVEL=ERROR


PREPARE_CONFIG

start_time=$(($(date +%s%N)/1000000))
RUN_TRACKING
duration=$(( $(date +%s%N)/1000000 - $start_time))
echo "RUN_TRACKING done... $duration milliseconds elapsed."


echo 'Demo completed!'