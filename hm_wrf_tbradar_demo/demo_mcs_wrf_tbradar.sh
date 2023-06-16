#!/bin/bash
#SBATCH --job-name=demo_cell_nerxrad_0
#SBATCH --partition=slurm
#SBATCH --time=01:30:00
#SBATCH -N 1
#SBATCH -n 60
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

# # Specify directory for the demo data
# dir_demo='/qfs/people/tang584/scripts/PyFLEXTRKR/hm_nexrad_demo' 
# # Example config file name
# config_demo='config_nexrad_cell_demo.yml'
# # Demo input data directory
# dir_input='/qfs/people/tang584/scripts/PyFLEXTRKR/input_data/nexrad_reflectivity1' #data downloaded

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

# dir_script="/people/tang584/scripts/PyFLEXTRKR"

PREPARE_CONFIG () {

    # Add '\' to each '/' in directory names
    dir_raw1=$(echo ${dir_input} | sed 's_/_\\/_g')
    dir_input1=$(echo ${dir_input} | sed 's_/_\\/_g')
    dir_demo1=$(echo ${dir_demo} | sed 's_/_\\/_g')
    # Replace input directory names in example config file
    sed 's/INPUT_DIR/'${dir_input1}'/g;s/TRACK_DIR/'${dir_demo1}'/g;s/RAW_DATA/'${dir_raw1}'/g' ${config_example} > ${config_demo}
    # sed 's/INPUT_DIR/'${dir_input1}'/g;s/TRACK_DIR/'${dir_demo1}'/g' ${config_example} > ${config_demo}
    echo 'Created new config file: '${config_demo}
}

RUN_TRACKING () {
    # Run tracking

    echo 'Running PyFLEXTRKR ...'
    python ../runscripts/run_mcs_tbpfradar3d_wrf.py ${config_demo} &> ${FUNCNAME[0]}-demo.log
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

MON_MEM () {
log_name=mem_usage
log_file="${log_name}-demo.log"

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

# spack load ior
# timeout 45 srun -N1 -n10 ior -w -r -t 1m -b 30g 

# # Activate PyFLEXTRKR conda environment
echo 'Activating PyFLEXTRKR environment ...'
source activate pyflextrkr_copy # pyflextrkr flextrkr

export FLUSH_MEM=TRUE # TRUE for flush, FALSE for no flush
export INVALID_OS_CACHE=TRUE # TRUE for invalid, FALSE for no invalid
export CURR_TASK=""

PREPARE_CONFIG

start_time=$(($(date +%s%N)/1000000))
RUN_TRACKING
duration=$(( $(date +%s%N)/1000000 - $start_time))
echo "RUN_TRACKING done... $duration milliseconds elapsed."


echo 'Demo completed!'
date

sacct -j $SLURM_JOB_ID --format="JobID,JobName,Partition,CPUTime,AllocCPUS,State,ExitCode,MaxRSS,MaxVMSize"
