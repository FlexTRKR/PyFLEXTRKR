# Preprocess GPM to combine Tb and IMERG precipitation data

The example codes in this directory combine the NCEP/CPC Global Merged IR Tb data with the GPM IMERG precipitation data into hourly files that can be used for MCS tracking.

# **1. Input Datasets**

---
The IR Tb and IMERG precipitation data can be obtained from the following:

* [NCEP/CPC Level 3 Merged Infrared Brightness Temperatures](https://gpm.nasa.gov/data/directory/ncepcpc-level-3-merged-infrared-brightness-temperatures-0)
* [GPM IMERG Final Precipitation L3 Half Hourly 0.1 degree x 0.1 degree V07](https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGHH_07/summary?keywords=%22IMERG%20final%22)

# **2. Processing Steps**
---
There are two separate steps to combine the two datasets:

* Regrid the 4-km IR Tb data to match the GPM 10-km IMERG grid
* Combine the regridded IR TB data with the GPM IMERG precipitation data

For efficiency, regridding the global 4-km IR Tb data uses [ESMF Regridding software](https://earthsystemmodeling.org/regrid/). Installing ESMF on a particular machine is beyond the scope of this documentation, but the following resources may be useful:

* [Installing ESMF](https://earthsystemmodeling.org/docs/release/latest/ESMF_usrdoc/)
* [Installing ESMPy](https://earthsystemmodeling.org/esmpy_doc/release/latest/html/install.html)
* [ESMF on GitHub](https://github.com/esmf-org/esmf)

The example here also makes use of the parallel processing tool called [Task Farmer developed by NERSC](https://docs.nersc.gov/jobs/workflow/taskfarmer/), making it easy to parallel processing large amount of files.

Assuming you are working on NERSC, below are the steps for combining the two datasets:

### 1. Generate a weight file for regridding

A global IR Tb 4-km to GPM IMERG 10-km weight file using conservative method can be downloaded from [here](https://portal.nersc.gov/project/m1867/PyFLEXTRKR/sample_data/tb_pcp/weight_ir_4km_to_10km_conserve.tar.gz).

### 2. Regrid Tb data

Modify the Bash shell script `regrid_global_ir_4km_to_10km.sh` to point to your weight file, the example shell script uses the E3SM unified software environment installed on NERSC, which has the command line tool `ncremap` for regridding.

Make a list of files to regrid: `./make_regrid_list.sh`

Submit the slurm job: `sbatch slurm_regrid_Summer.sh`

### 3. Combine regridded Tb and IMERG precipitation data

Make a list of dates to combine the regridded Tb and IMERG data: 

`python make_combine_ir_imerg_list.py Summer`

Submit the slurm job: `sbatch slurm_combine_Summer.sh`

Note that the Python code `combine_ir_imerg_global_byday.py` averages the 30-min GPM IMERG precipitation data to hourly, while retaining the two 30-min IR Tb data in the combined hourly files. This choice is made following our previous study [Feng et al. (2021) JGR](https://doi.org/10.1029/2020JD034202), and explained in issue [#82](https://github.com/FlexTRKR/PyFLEXTRKR/issues/82#issuecomment-1934696781).