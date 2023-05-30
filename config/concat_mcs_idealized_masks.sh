#!/bin/bash
###############################################################################################
# This script combines individual MCS pixel mask files from idealized tracking to
# a single file, and renames variables/dimensions.
###############################################################################################

# Specify directory for the demo data
# There are a total of 4 tests (e.g., test1, test2, test3, test4)
test_name='test1'

# Test file directory
dir_demo='/Users/feng045/data/demo/mcs_tbpf/idealized/'${test_name}'/'
# Test MCS pixel mask directory
out_dirname=${dir_demo}'/mcstracking/20200101.0000_20200103.0000/'
# Output file directory
out_filename='MCS_mask_idealized_'${test_name}'_Feng.nc'

# Concat MCS mask files
ncrcat -O -v time,longitude,latitude,tb,precipitation,cloudtracknumber,feature_number \
  ${out_dirname}mcstrack*.nc ${out_dirname}${out_filename}

# Rename variables
ncrename -v cloudtracknumber,MCS_objects -v tb,Tb -v precipitation,PR -v feature_number,BT_objects ${out_dirname}${out_filename}

# Rename dimensions
ncrename -d lat,yc -d lon,xc ${out_dirname}${out_filename}

echo ${out_dirname}${out_filename}