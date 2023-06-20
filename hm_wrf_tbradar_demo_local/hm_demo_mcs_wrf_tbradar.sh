#!/bin/bash
#SBATCH --job-name=hm_demo_cell_nerxrad_0
#SBATCH --partition=short
#SBATCH --time=01:30:00
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --output=./R_%x.out
#SBATCH --error=./R_%x.err

## --exclude=dc[009-099,119] --exclude=a100-[05] --exclude=dl[05-10]

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

# Function to clean up background processes
cleanup() {
    echo "Cleaning up background processes..."
    kill $(jobs -p) 2>/dev/null
    wait $(jobs -p) 2>/dev/null
    echo "Cleanup complete."
}

# Set trap to call cleanup function on script termination
trap cleanup EXIT


## Prepare Slurm Host Names and IPs
NODE_NAMES=`echo $SLURM_JOB_NODELIST|scontrol show hostnames`

hostlist=$(echo "$NODE_NAMES" | tr '\n' ',')
echo "hostlist: $hostlist"

rm -rf ./host_ip
touch ./host_ip
echo "localhost" > ./host_ip
# host_arr=()
# for node in $NODE_NAMES
# do
#     host_arr+=("$node")
#     nost_ip=`getent hosts "$node.ibnet" | awk '{ print $1 }'`
#     echo "$nost_ip" >> ./host_ip
# done

cat ./host_ip
ib_hostlist=$(cat ./host_ip | xargs | sed -e 's/ /,/g')
echo "ib_hostlist: $ib_hostlist"


## Prepare Test Directories
TEST_NAME='wrf_tbradar'
exp_dir="$HOME/experiments/$USER/flextrkr_runs"

# Specify directory for the demo data
dir_demo="${exp_dir}/hm_${TEST_NAME}" #local

mkdir -p $dir_demo
rm -rf $dir_demo/*
# Example config file name
config_example='config_wrf_mcs_tbradar_example.yml'
# config_example='config_wrf_mcs_tbradar_short.yml'
config_demo='config_wrf_mcs_tbradar_demo.yml'
cp ./$config_demo $dir_demo
# Demo input data directory
dir_input="${exp_dir}/input_data/${TEST_NAME}"


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
    echo 'Running PyFLEXTRKR w/ VFD ...'


    # LD_LIBRARY_PATH=$DLIFE_VOL_DIR:$LD_LIBRARY_PATH \
    #     HDF5_VOL_CONNECTOR="${VOL_NAME} under_vol=0;under_info={};path=${SCRIPT_DIR}/vol-${task_id}_${FUNCNAME[0]}.log;level=2;format=" \
    #     HDF5_DRIVER=hdf5_hermes_vfd \
    #     HDF5_PLUGIN_PATH=$DLIFE_VOL_DIR:${HERMES_INSTALL_DIR}/lib \

    # LD_LIBRARY_PATH=$DLIFE_VOL_DIR:$LD_LIBRARY_PATH \
    #     HDF5_VOL_CONNECTOR="${VOL_NAME} under_vol=0;under_info={};path=${SCRIPT_DIR}/vol-${task_id}_${FUNCNAME[0]}.log;level=2;format=" \
    #     HDF5_PLUGIN_PATH=$DLIFE_VOL_DIR \
    #         srun -n1 -N1 --oversubscribe --mpi=pmi2 \
    
    
    HDF5_DRIVER=hdf5_hermes_vfd \
        HDF5_PLUGIN_PATH=${HERMES_INSTALL_DIR}/lib:$HDF5_PLUGIN_PATH \
        HERMES_CONF=$HERMES_CONF \
        HERMES_CLIENT_CONF=$HERMES_CLIENT_CONF \
        HDF5_DRIVER_CONFIG="true ${HERMES_PAGESIZE}" \
        python ../runscripts/run_mcs_tbpfradar3d_wrf.py ${config_demo} &> ${FUNCNAME[0]}-hm.log
    
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

    # echo "SLURM_JOB_NODELIST = $(echo $SLURM_JOB_NODELIST|scontrol show hostnames)"
    # NODE_NAMES=$(echo $SLURM_JOB_NODELIST|scontrol show hostnames)
    NODE_NAMES=""

    prefix="localhost" #dc dc00 a100-0
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

    rm -rf $DEV2_DIR $DEV1_DIR
    mkdir -p $DEV2_DIR $DEV1_DIR

    echo "Starting hermes_daemon..."
    set -x
    # export LD_PRELOAD=/lib64/libucs.so:$LD_PRELOAD
    # export LD_LIBRARY_PATH=/lib64/ucx:$LD_LIBRARY_PATH
    # -x LD_PRELOAD=/usr/lib64/libucs.so$LD_PRELOAD \
    # -x UCX_NET_DEVICES=mlx5_0:1 \
    # mpirun --host $ib_hostlist --npernode 1 \
    #     -x HERMES_CONF=$HERMES_CONF ${HERMES_INSTALL_DIR}/bin/hermes_daemon &> ${FUNCNAME[0]}.log &

    HERMES_CONF=$HERMES_CONF ${HERMES_INSTALL_DIR}/bin/hermes_daemon &> ${FUNCNAME[0]}.log &

    # echo ls -l $DEV1_DIR/hermes_slabs
    sleep 5
    echo "Show hermes slabs : "
    # srun -n$SLURM_JOB_NUM_NODES -w $hostlist --oversubscribe ls -l $DEV1_DIR/*
    ls -l $DEV1_DIR/*
    ls -l $DEV2_DIR/*
    set +x
}


MON_MEM () {
log_name=mem_usage
log_file="${log_name}-hm.log"

    echo "Logging mem usage to $log_file"

    index=0  # Initialize the index variable

    free -h | awk -v idx="$index" 'BEGIN{OFS="\t"} NR==1{print "Index\t","Type\t" $0} NR==2{print idx, $0}' > "$log_file"

    while true; do
    # Run the `free` command and append the formatted output to the log file using `tee`
    free -h | awk -v idx="$index" 'BEGIN{OFS="\t"} NR==2{print idx, $0}' >> "$log_file"

    # Increment the index
    ((index++))

    # Sleep for a desired interval before running the loop again
    sleep 1
    done
}

date

MON_MEM &

source ./load_hermes_deps.sh
source ./env_var.sh

# # Activate PyFLEXTRKR conda environment
echo 'Activating PyFLEXTRKR environment ...'
source activate flextrkr # pyflextrkr flextrkr

export FLUSH_MEM=FALSE # TRUE for flush, FALSE for no flush
export CURR_TASK=""

# export PYTHONLOGLEVEL=ERROR
# export PYTHONLOGLEVEL=INFO

killall hermes_daemon
sleep 2

PREPARE_CONFIG

set -x

echo 3 | sudo tee /proc/sys/vm/drop_caches

HERMES_DIS_CONFIG
START_HERMES_DAEMON
(

# ulimit -v $(( 7 * 1024 * 1024 )) # in KB

start_time=$(($(date +%s%N)/1000000))
RUN_TRACKING
duration=$(( $(date +%s%N)/1000000 - $start_time))
echo "RUN_TRACKING done... $duration milliseconds elapsed." >> RUN_TRACKING-hm.log

)


echo 'Demo completed!'

sacct -j $SLURM_JOB_ID -o jobid,submit,start,end,state
rm -rf $dir_demo/core.*