#!/bin/bash
#SBATCH --job-name=dl_demo_cell_nerxrad_0
#SBATCH --partition=slurm
#SBATCH --time=00:30:00
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --output=./R_%x.out
#SBATCH --error=./R_%x.err

## --exclude=a100-[05]
## --exclude=dc[009-099,119]

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

## Prepare Slurm Host Names and IPs
NODE_NAMES=`echo $SLURM_JOB_NODELIST|scontrol show hostnames`

hostlist=$(echo "$NODE_NAMES" | tr '\n' ',')
echo "hostlist: $hostlist"

rm -rf ./host_ip
# echo "localhost" > ./host_ip
touch ./host_ip
host_arr=()
for node in $NODE_NAMES
do
    host_arr+=("$node")
    nost_ip=`getent hosts "$node.ibnet" | awk '{ print $1 }'`
    echo "$nost_ip" >> ./host_ip
done

cat ./host_ip
ib_hostlist=$(cat ./host_ip | xargs | sed -e 's/ /,/g')
echo "ib_hostlist: $ib_hostlist"


## Prepare Test Directories
TEST_NAME='wrf_tbradar'

# Specify directory for the demo data
dir_demo="/qfs/projects/oddite/tang584/flextrkr_runs/${TEST_NAME}" #NFS
mkdir -p $dir_demo
rm -rf $dir_demo/*
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
    echo 'Running PyFLEXTRKR w/ VOL+VFD ...'

    schema_file=data-stat-dl.yaml
    rm -rf ./*$schema_file
    # touch $schema_file
    rm -rf ./*vfd-$schema_file

    # HDF5_PLUGIN_PATH=${HERMES_INSTALL_DIR}/lib:$TRACKER_VOL_DIR:$HDF5_PLUGIN_PATH \

    LD_LIBRARY_PATH=$TRACKER_VOL_DIR:$LD_LIBRARY_PATH \
        HDF5_VOL_CONNECTOR="${VOL_NAME} under_vol=0;under_info={};path=${schema_file};level=2;format=" \
        HDF5_PLUGIN_PATH=${HERMES_INSTALL_DIR}/lib:$TRACKER_VOL_DIR:$HDF5_PLUGIN_PATH \
        HDF5_DRIVER=hdf5_hermes_vfd \
        HERMES_CONF=$HERMES_CONF \
        HERMES_CLIENT_CONF=$HERMES_CLIENT_CONF \
        HDF5_DRIVER_CONFIG="true ${HERMES_PAGE_SIZE}" \
        python ../runscripts/run_mcs_tbpfradar3d_wrf.py ${config_demo} &> ${FUNCNAME[0]}-dl.log
    
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


HERMES_DIS_CONFIG () {

    echo "SLURM_JOB_NODELIST = $(echo $SLURM_JOB_NODELIST|scontrol show hostnames)"
    NODE_NAMES=$(echo $SLURM_JOB_NODELIST|scontrol show hostnames)

    prefix="dc00" #dc dc00 a100-0
    sed "s/\$HOST_BASE_NAME/\"${prefix}\"/" $HERMES_DEFAULT_CONF  > $HERMES_CONF
    mapfile -t node_range < <(echo "$NODE_NAMES" | sed "s/${prefix}//g")
    rpc_host_number_range="[$(printf "%s," "${node_range[@]}" | sed 's/,$//')]"
    # rpc_host_number_range="[]"
    sed -i "s/\$HOST_NUMBER_RANGE/${rpc_host_number_range}/" $HERMES_CONF

    hostfile_path="$(pwd)/host_ip"
    sed -i "s#\$HOSTFILE_PATH#${hostfile_path}#" $HERMES_CONF

    network_device=`ucx_info -d | grep Device | cut -d' ' -f11 | grep mlx5 | head -1`
    sed -i "s/\$NETWORK_DEVICE/${network_device}/" $HERMES_CONF

    echo "hostfile_path=${hostfile_path}"
    echo "node_range=${node_range[@]}"
    echo "rpc_host_number_range=$rpc_host_number_range"

    # INTERCEPT_PATHS=$(sed "s/\$TEST_OUT_PATH/${TEST_OUT_PATH}/g" i${ITER_COUNT}_sim_files.txt)
    # echo "$INTERCEPT_PATHS" >> $HERMES_CONF

    # echo "]" >> $HERMES_CONF

}

STOP_DAEMON () {

    set -x
    HERMES_CONF=$HERMES_CONF srun -n1 -N1 --oversubscribe --mpi=pmi2 \
        ${HERMES_INSTALL_DIR}/bin/finalize_hermes &

    set +x
}


START_HERMES_DAEMON () {
    # --mca shmem_mmap_priority 80 \ \
    # -mca mca_verbose stdout 
    # -x UCX_NET_DEVICES=mlx5_0:1 \
    # -mca btl self -mca pml ucx \
    # srun -n$SLURM_JOB_NUM_NODES -w $hostlist rm -rf $DEV1_DIR
    # srun -n$SLURM_JOB_NUM_NODES -w $hostlist mkdir -p $DEV1_DIR

    echo `which ucx_info`

    rm -rf $DEV2_DIR $DEV1_DIR
    mkdir -p $DEV2_DIR $DEV1_DIR

    echo "Starting hermes_daemon..."
    set -x


    # LD_PRELOAD=${HERMES_INSTALL_DIR}/lib/libhdf5_hermes_vfd.so:$LD_PRELOAD \
    HERMES_CONF=$HERMES_CONF srun -n$SLURM_JOB_NUM_NODES -w $hostlist --oversubscribe --mpi=pmi2 \
        ${HERMES_INSTALL_DIR}/bin/hermes_daemon &> ${FUNCNAME[0]}.log &

    # echo ls -l $DEV1_DIR/hermes_slabs
    sleep 5
    echo "Show hermes slabs : "
    # srun -n$SLURM_JOB_NUM_NODES -w $hostlist --oversubscribe ls -l $DEV1_DIR/*
    ls -l $DEV1_DIR/*
    ls -l $DEV2_DIR/*
    set +x
}


source ./load_hermes_deps.sh
source ./env_var.sh

# # Activate PyFLEXTRKR conda environment
echo 'Activating PyFLEXTRKR environment ...'
source activate pyflextrkr_copy # flextrkr pyflextrkr


# export PYTHONLOGLEVEL=ERROR
srun -n1 -N1 killall hermes_daemon

PREPARE_CONFIG

set -x

HERMES_DIS_CONFIG
START_HERMES_DAEMON

start_time=$(($(date +%s%N)/1000000))
RUN_TRACKING
duration=$(( $(date +%s%N)/1000000 - $start_time))
echo "RUN_TRACKING done... $duration milliseconds elapsed."

STOP_DAEMON

echo 'Demo completed!'

sacct -j $SLURM_JOB_ID -o jobid,submit,start,end,state
rm -rf $dir_demo/core.*
rm -rf $dir_demo/device*.hermes

cat RUN_TRACKING-dl.log | grep "H5FD__hermes" > RUN_TRACKING-dl-hm.log