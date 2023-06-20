#!/bin/bash

# spack load --only dependencies hermes
# spack unload mpich
# module purge
# source /share/apps/python/miniconda3.7/etc/profile.d/conda.sh


. $HOME/spack/share/spack/setup-env.sh

# spack load --only dependencies hermes
# spack install netcdf-c~mpi ^hdf5@1.14.0~mpi
spack load netcdf-c hdf5


#spack load mochi-thallium@0.10.0 catch2@3.0.1 glpk glog yaml-cpp mpich #automake
spack load boost cereal mochi-thallium@0.8.3 catch2@3.0.1 glpk glog yaml-cpp mpich 

