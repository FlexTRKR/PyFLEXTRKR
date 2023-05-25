#!/bin/bash

#CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# source /share/apps/python/miniconda3.7/etc/profile.d/conda.sh

#spack load hdf5 mochi-thallium catch2 glpk gflags glog
#spack find --loaded

USER=$(whoami)

HERMES_VERSION="vfd_hermes"
#8dec_hermes dec_hermes 1dec_hermes debug_1dec_hermes
# dec_hermes-vfd

# User directories
MNT_HOME=$HOME #/people/$USER
INSTALL_DIR=$HOME/install
DL_DIR=$HOME/download
SCRIPT_DIR=$MNT_HOME/scripts/local-co-scheduling
CONFIG_DIR=./hermes_configs

# Hermes running dirs -----------
STAGE_DIR=$MNT_HOME/hermes_stage
HERMES_REPO=$STAGE_DIR/hermes
MOCHI_REPO=$STAGE_DIR/mochi
SPACK_DIR=$MNT_HOME/spack

# Hermes config files -----------
DEFAULT_CONF_NAME=hermes_server_default.yaml
HERMES_DEFAULT_CONF=$CONFIG_DIR/$DEFAULT_CONF_NAME

CONF_NAME=hermes_server.yaml
# CONF_NAME=hermes_deception.yaml
export HERMES_CONF=$CONFIG_DIR/$CONF_NAME

CLIENT_CONF_NAME=hermes_client.yaml
export HERMES_CLIENT_CONF=$CONFIG_DIR/$CLIENT_CONF_NAME

# HERMES_INSTALL_DIR=$INSTALL_DIR/hermes
HERMES_INSTALL_DIR=$INSTALL_DIR/$HERMES_VERSION


# Debug
ASAN_LIB=""
# HERMES_INSTALL_DIR=$INSTALL_DIR/debug_hermes
# HERMES_INSTALL_DIR=$INSTALL_DIR/8_hermes


# System storage dirs -----------
# DEV0_DIR="" # this is memory
export DEV1_DIR=/scratch/$USER/hermes_slabs # this is BurstBuffer
export DEV2_DIR=/rcfs/projects/chess/$USER/hermes_slabs # this is Parallel File System
mkdir -p $DEV1_DIR/
mkdir -p $DEV2_DIR/
rm -rf $DEV1_DIR/*
rm -rf $DEV2_DIR/*

# export DEV1_DIR=/scratch/$USER # this is BurstBuffer
# export DEV2_DIR=/rcfs/projects/chess/$USER # this is Parallel File System

# export DEV1_DIR="." # current dir
# export DEV2_DIR="." # current dir
# export DEV1_DIR="/tmp" # current dir
# export DEV2_DIR="/tmp" # current dir

# Other tools dirs -----------
BENCHMARKS_DIR=$HERMES_REPO/benchmarks
HDF5_REPO=$DL_DIR/hdf5-hdf5-1_13_1
IOR_REPO=$STAGE_DIR/ior
IOR_INSTALL=$INSTALL_DIR/ior
HDF5_INSTALL=$INSTALL_DIR/hdf5
DLIFE_VOL_DIR=$SCRIPT_DIR/vol-datalife/src
# DLIFE_VOL_DIR=$SCRIPT_DIR/vol-src
VOL_NAME="datalife"

# RECORDER_REPO=$DL_DIR/Recorder-2.3.2
# RECORDER_INSTALL=$INSTALL_DIR/recorder
LOG_DIR=$SCRIPT_DIR/tmp_outputs
mkdir -p $LOG_DIR

#conda activate /files0/oddite/conda/ddmd/ # original global env
#conda activate hermes_ddmd # local

PY_VENV=$SCRIPT_DIR/venv_ddmd


export GLOG_minloglevel=2
export FLAGS_logtostderr=2
export HDF5_USE_FILE_LOCKING='FALSE' #'TRUE' 'FALSE'
# export MPICH_GNI_NDREG_ENTRIES=1024
# export I_MPI_HYDRA_TOPOLIB=ipl
# export I_MPI_PMI_LIBRARY=libpmi2.so

export HERMES_TRAIT_PATH=$HERMES_INSTALL_DIR/lib
echo "HERMES_TRAIT_PATH = $HERMES_TRAIT_PATH"

# export OFI_INTERFACE=ib0

export HERMES_PAGESIZE=8192 #for Hermes_VFD
export HERMES_PAGE_SIZE=131072 #for hermes_POSIX
# page size : 4096 8192 32768 65536 131072 262144 524288 1048576 4194304 8388608
# default : 1048576