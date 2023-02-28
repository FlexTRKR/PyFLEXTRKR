# **PyFLEXTRKR: a Flexible Feature Tracking Python Software for Convective Cloud Analysis**

# **1. Introduction**

---
The Python FLEXible object TRacKeR (PyFLEXTRKR) is a flexible atmospheric feature tracking software package. The software can track any 2D objects and handle merging and splitting explicitly. PyFLEXTRKR has specific capabilities to track convective clouds from a variety of observations and model simulations, including: 1) individual convective cells, and 2) mesoscale convective systems (MCSs) using radar, satellite, and model data. The package has scalable parallelization options and has been optimized to work on large datasets including global high resolution data.

## For a more detailed user guide, click [this link](https://github.com/FlexTRKR/PyFLEXTRKR/blob/main/UserGuide.md).


# **2. Input Data Requirements**

---
PyFLEXTRKR works with netCDF files using Xarray's capability to handle N-dimension arrays of gridded data. Currently, PyFLEXTRKR supports: 

1. Tracking convective cells using radar reflectivity data [[Feng et al. (2022), MWR](https://doi.org/10.1175/MWR-D-21-0237.1)]; 
2. Tracking MCSs using infrared brightness temperature (Tb) data from geostationary satellites, or outgoing longwave radiation (OLR) data from model simulations, with optional collocated precipitation data [[Feng et al. (2021), JGR](https://doi.org/10.1029/2020JD034202)] or 3D radar reflectivity data [[Feng et al. (2018) JAMES](https://doi.org/10.1029/2018MS001305); [Feng et al. (2019), JCLI](https://doi.org/10.1175/JCLI-D-19-0137.1)] to identify robust MCSs;
3. Tracking generic 2D objects defined by customizable feature identification functions.

The input data must contain at least 3 dimensions: *time, y, x*, with corresponding coordinates of *time, latitude, longitude*. The *latitude* and *longitude* coordinates can  be either 1D or 2D. But the data must be on a fixed 2D grid (any projection is fine) since PyFLEXTRKR only supports tracking data on 2D arrays. Irregular grids such as those in E3SM or MPAS model must first be regridded to a regular grid before tracking. Additional variable names and coordinate names are specified in the config file. See [user guide](https://github.com/FlexTRKR/PyFLEXTRKR/blob/main/UserGuide.md) for more details in preparing the input dataset.

# **3. Installing PyFLEXTRKR**

---
Clone PyFLEXTRKR to your local computer (e.g., /PyFLEXTRKR):

```bash
git clone https://github.com/FlexTRKR/PyFLEXTRKR.git
```

Go to that directory:

```bash
cd /PyFLEXTRKR
```

Use the included environment.yml file to create a Conda virtual environment, make sure you change `conda_env_dir` to where your conda environments are installed:

```bash
conda env create -f environment.yml --prefix /conda_env_dir/flextrkr
```

**Pro Tips:** using [mamba](https://anaconda.org/conda-forge/mamba) to create the virtual environment is much faster:

```bash
mamba env create -f environment.yml --prefix /conda_env_dir/flextrkr
```

After setting up the Conda virtual environment, activate it with:

```bash
conda activate flextrkr
```

Then install the package with:

```bash
pip install -e .
```

Any changes to the source code will be reflected in the running version.  

# **4. Example Data and Runscripts**

---
Several scripts are provided to download example input data, run tracking, and produce visualizations of the tracking results below:

1. [Convective cell tracking from 500 m gridded NEXRAD radar data](https://github.com/FlexTRKR/PyFLEXTRKR/blob/main/config/demo_cell_nexrad.sh)

2. [Convective cell tracking from 500 m gridded ARM CSAPR radar data](https://github.com/FlexTRKR/PyFLEXTRKR/blob/main/config/demo_cell_csapr.sh)

![](https://portal.nersc.gov/project/m1867/PyFLEXTRKR/figures/nexrad_celltracking_animation_small.gif)

3. [MCS tracking from 10 km GPM IMERG Tb + precipitation data](https://github.com/FlexTRKR/PyFLEXTRKR/blob/main/config/demo_mcs_imerg.sh)

4. [MCS tracking from 4 km WRF Tb + precipitation data](https://github.com/FlexTRKR/PyFLEXTRKR/blob/main/config/demo_mcs_wrf_tbpf.sh)

5. [MCS tracking from 4 km WRF Tb + 3D reflectivity data](https://github.com/FlexTRKR/PyFLEXTRKR/blob/main/config/demo_mcs_wrf_tbradar.sh)

6. [MCS tracking from 25 km model OLR + precipitation data](https://github.com/FlexTRKR/PyFLEXTRKR/blob/main/config/demo_mcs_model25km.sh)

![](https://portal.nersc.gov/project/m1867/PyFLEXTRKR/figures/imerg_mcstracking_animation_small.gif)

7. [Generic feature tracking (e.g., 500 hPa geopotential height anomaly)](https://github.com/FlexTRKR/PyFLEXTRKR/blob/main/config/demo_generic_tracking.sh)

![](https://portal.nersc.gov/project/m1867/PyFLEXTRKR/figures/z500_tracking_animation_small.gif)


To run these demo scripts, download the script, modify the `dir_demo` in the script to a directory on your computer to store the sample data, and run the following command:


```bash
bash demo_cell_nexrad.sh
```

The demo script downloads and untar the sample data, runs the tracking code, and generates visualizations. Once the demo script finishes running, a sub-directory within the `dir_demo` named `quicklooks_trackpaths` will be created that contains quicklook visualization of the tracking results, as shown in the example animations above.


# **5. Running PyFLEXTRKR**

---

To run the code, type the following in the command line:

Activate PyFLEXTRKR virtual environment:

```bash
conda activate flextrkr
```

Run PyFLEXTRKR:

```bash
python ../runscripts/run_celltracking.py ./config/config_nexrad500m_example.yml
```

```bash
python ./runscripts/run_mcs_tbpf.py ./config/config_wrf4km_mcs_tbpf_example.yml
```

### **Example run scripts and config files are in the highlighted directories:**
![](https://portal.nersc.gov/project/m1867/PyFLEXTRKR/figures/run_command_explanation.png)


# **6. References**

---

Feng, Z., Hardin, J., Barnes, H. C., Leung, L. R., Varble, A., & Zhang, Z. (2022). PyFLEXTRKR: a Flexible Feature Tracking Python Software for Convective Cloud Analysis. EGUsphere [preprint]. [https://doi.org/10.5194/egusphere-2022-1136]()

Feng, Z., Varble, A., Hardin, J., Marquis, J., Hunzinger, A., Zhang, Z., & Thieman, M. (2022). Deep Convection Initiation, Growth, and Environments in the Complex Terrain of Central Argentina during CACTI. Monthly Weather Review, 150(5), 1135-1155. [https://doi.org/10.1175/MWR-D-21-0237.1]()

Feng, Z., Leung, L. R., Liu, N., Wang, J., Houze, R. A., Li, J., et al. (2021). A Global High‐Resolution Mesoscale Convective System Database Using Satellite‐Derived Cloud Tops, Surface Precipitation, and Tracking. Journal of Geophysical Research: Atmospheres, 126(8). [https://doi.org/10.1029/2020JD034202]()

---

