#!/bin/bash

# Concatenantes MCS pixel-level masks from a continuous tracking period to a single netCDF file
# Follows the SAAG common data format propocal
# Output filename: feng_WY2019_GPM_SAAG-MCS-mask-file.nc

module load nco

rootdir='/gpfs/wolf/atm131/proj-shared/zfeng/SAAG/mcs_tracking/'
# data_source='GPM'
data_source='WRF'

# Make output directory
outdir=${rootdir}${data_source}'/mcs_mask/'
mkdir -p ${outdir}

# Find all periods sub-directory, exclude "."
pdir=${rootdir}${data_source}/20??_20??/
# pdir=${rootdir}${data_source}/2010_2011/
periods=$(find ${pdir} -maxdepth 1 -type d -not -name . -name 20??_20?? -printf '%f\n' | sort -n)

# Loop over each period directory
for ip in ${periods}; do
    # Find MCS mask sub-directory
    ipdir=${rootdir}${data_source}/${ip}/mcstracking_saag/20*
    mdir=$(find ${ipdir} -maxdepth 1 -type d -not -name . -printf '%f\n' | sort -n)
    # Mask sub-directory
    maskdir=${rootdir}${data_source}/${ip}/mcstracking_saag/${mdir}/
    echo ${maskdir}
    # Make output filename
    year=${ip:5:4}
    outfile=${outdir}feng_WY${year}_${data_source}_SAAG-MCS-mask-file.nc
    # echo ${outfile}
    # cmd='ncrcat -h -v time,lon,lat,cloudtracknumber '${maskdir}'mcstrack_20100601*nc '${outfile}
    # echo ${cmd}
    # Concatenate files
    ncrcat -O -h -v time,lon,lat,cloudtracknumber ${maskdir}mcstrack_*nc ${outfile}
    # Rename mask variable name
    ncrename -v cloudtracknumber,mcs_mask ${outfile}
    # Print output filename
    echo ${outfile}
done
