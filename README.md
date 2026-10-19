# **PyFLEXTRKR: a Flexible Feature Tracking Python Software for Convective Cloud Analysis**

![](https://portal.nersc.gov/project/m1867/PyFLEXTRKR/figures/cover_image.png)

# **1. Introduction**

---
The Python FLEXible object TRacKeR (PyFLEXTRKR) is a flexible atmospheric feature tracking software package. The software can track **any 2D objects** and handle merging and splitting explicitly. PyFLEXTRKR has specific capabilities to track convective clouds from a variety of observations and model simulations, including: 1) individual convective cells, and 2) mesoscale convective systems (MCSs) using radar, satellite, and model data. The package has scalable parallelization options and has been optimized to work on large datasets including global kilometer-scale data.

## For a more detailed user guide, click [this link](https://github.com/FlexTRKR/PyFLEXTRKR/blob/main/UserGuide.md).


# **2. Input Data Requirements**

---
PyFLEXTRKR works with netCDF files using Xarray's capability to handle N-dimension arrays of gridded data. Currently, PyFLEXTRKR supports tracking: 

1. Individual convective cells using radar reflectivity data [[Feng et al. (2022), MWR](https://doi.org/10.1175/MWR-D-21-0237.1)]; 
2. MCSs using infrared brightness temperature (Tb) data from geostationary satellites, or outgoing longwave radiation (OLR) data from model simulations, with optional collocated precipitation data [[Feng et al. (2021), JGR](https://doi.org/10.1029/2020JD034202)] or 3D radar reflectivity data [[Feng et al. (2018) JAMES](https://doi.org/10.1029/2018MS001305); [Feng et al. (2019), JCLI](https://doi.org/10.1175/JCLI-D-19-0137.1); [Feng et al. (2024), JGR](https://doi.org/10.1029/2023JD039605)] to identify robust MCSs;
3. Generic 2D objects defined by customizable feature identification functions.

The input data must contain at least 3 dimensions: *time, y, x*, with corresponding coordinates of *time, latitude, longitude*. The *latitude* and *longitude* coordinates can  be either 1D or 2D. But the data must be on a fixed 2D grid (any projection is fine) since PyFLEXTRKR only supports tracking data on 2D arrays. Irregular grids such as those in E3SM or MPAS model must first be regridded to a regular grid before tracking. Additional variable names and coordinate names are specified in the config file. See [user guide](https://github.com/FlexTRKR/PyFLEXTRKR/blob/main/UserGuide.md) for more details in preparing the input dataset.

# **3. Installing PyFLEXTRKR**

---
PyFLEXTRKR requires Python 3.11 or above. It is tested with Python 3.11 and 3.12 on Linux, macOS and Windows, and the `environment.yml` file below creates a Python 3.12 environment.

## **Install via Conda**

The easiest way to install the most recent version of PyFLEXTRKR is via conda through the conda-forge channel:

```
conda install pyflextrkr -c conda-forge
```

This will also install all dependencies and should be sufficient for most users. You can update the installation by:

```
conda update pyflextrkr -c conda-forge
```

## **Install from source via Github**

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

or ...

```bash
conda create -n flextrkr -c conda-forge --file requirements.txt
```

The visualization scripts (quicklook maps and animations, see [`visualization/README.md`](visualization/README.md)) also need cartopy, matplotlib, colorcet, colormaps and ffmpeg. They are in `environment.yml` but not in `requirements.txt`, so use the `environment.yml` option if you want to make the maps and animations.

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

![](docs/figures/readme/cell_nexrad_20140807.gif)

*Convective cells from the 500 m NEXRAD demo (KHGX, Houston, 7 August 2014, 12-15 UTC): composite reflectivity with the cell outlines, track paths and symbols colored by lifetime, cell numbers and radar range rings (every 20 km) on a square map around the radar. Made with [`visualization/examples/make_cell_animation_nexrad.py`](visualization/examples/make_cell_animation_nexrad.py), which calls [`visualization/plot_cell_tracks.py`](visualization/plot_cell_tracks.py) and ffmpeg (the same mp4 and GIF from PNG files with [`visualization/pngs_to_animation.py`](visualization/pngs_to_animation.py)), see [cell_maps.md](docs/visualization/cell_maps.md).*

2. [Convective cell tracking from 500 m gridded ARM CSAPR radar data](https://github.com/FlexTRKR/PyFLEXTRKR/blob/main/config/demo_cell_csapr.sh)

3. [MCS tracking from 5 km Himawari Tb data](https://github.com/FlexTRKR/PyFLEXTRKR/blob/main/config/demo_mcs_himawari.sh) (Tb only)

![](docs/figures/visualization/mcs_himawari_tb.gif)

*Himawari Tb with the tracked MCSs (outline, track path and number), made with [`visualization/examples/make_mcs_animation_himawari_tb.py`](visualization/examples/make_mcs_animation_himawari_tb.py); more animations and their settings are in [`visualization/README.md`](visualization/README.md#more-mcs-looks).*

4. [MCS tracking from 10 km GPM IMERG Tb + precipitation data](https://github.com/FlexTRKR/PyFLEXTRKR/blob/main/config/demo_mcs_imerg.sh)

![](docs/figures/readme/mcs_global_imerg_2days.gif)

*Global MCS tracking from GPM Tb + IMERG precipitation, 1-2 January 2025: Tb with the track numbers (top), precipitation with the tracked MCSs shaded (bottom). Made with [`visualization/make_mcs_animation.py`](visualization/make_mcs_animation.py) (Robinson projection centered at 225E, 60S-60N), see [global_mcs_animation.md](docs/visualization/global_mcs_animation.md).*

![](docs/figures/readme/mcs_sahel_2days.gif)

*The same tracking over the Sahel, 22-23 August 2025, with a white halo around the tracks (`--track_halo_color white --track_halo_alpha 0.7`).*

5. [MCS tracking from 4 km WRF Tb + precipitation data](https://github.com/FlexTRKR/PyFLEXTRKR/blob/main/config/demo_mcs_wrf_tbpf.sh)

6. [MCS tracking from 4 km WRF Tb + 3D reflectivity data](https://github.com/FlexTRKR/PyFLEXTRKR/blob/main/config/demo_mcs_wrf_tbradar.sh)

![](docs/figures/readme/mcs_wrf_tbze_2days.gif)

*WRF Tb (left) and composite reflectivity with the tracked MCSs shaded (right), 7-8 May 2015, on the Lambert conformal grid of the simulation. Made with [`visualization/examples/make_mcs_animation_wrf_tbze.py`](visualization/examples/make_mcs_animation_wrf_tbze.py).*

7. [MCS tracking from 25 km model OLR + precipitation data](https://github.com/FlexTRKR/PyFLEXTRKR/blob/main/config/demo_mcs_model25km.sh)

8. [Generic feature tracking (e.g., 500 hPa geopotential height anomaly)](https://github.com/FlexTRKR/PyFLEXTRKR/blob/main/config/demo_generic_tracking.sh)

![](docs/figures/readme/generic_z500_1979.gif)

*Tracked 500 hPa geopotential height anomalies (ERA5), 1 June - 30 August 1979, one frame per day. Made with [`visualization/plot_generic_tracks.py`](visualization/plot_generic_tracks.py) and [`visualization/pngs_to_animation.py`](visualization/pngs_to_animation.py), see [generic_maps.md](docs/visualization/generic_maps.md).*


To run these demo scripts, clone the repository, modify the `dir_demo` in the script to a directory on your computer to store the sample data, and run the script from the `config/` directory (the scripts call `../runscripts/` and `../visualization/`), for example:


```bash
cd config
bash demo_cell_nexrad.sh
```

The demo script downloads and untar the sample data, runs the tracking code, and generates visualizations. Once the demo script finishes running, a sub-directory within the `dir_demo` named `quicklooks_trackpaths` will be created that contains quicklook visualization of the tracking results, as shown in the example animations above.

The quicklook maps are made by the scripts in [`visualization/`](visualization) (MCS, convective cells, generic features, idealized domains; map projections, track style options, animations): see [`visualization/README.md`](visualization/README.md). The old `Analysis/plot_subset_*.py` commands still work.


# **5. Running PyFLEXTRKR**

---

To run the code, type the following in the command line:

Activate PyFLEXTRKR virtual environment:

```bash
conda activate flextrkr
```

Run PyFLEXTRKR:

```bash
python ./runscripts/run_celltracking.py ./config/config_nexrad500m_example.yml
```

```bash
python ./runscripts/run_mcs_tbpf.py ./config/config_wrf4km_mcs_tbpf_example.yml
```

### **Example run scripts and config files are in the highlighted directories:**
![](https://portal.nersc.gov/project/m1867/PyFLEXTRKR/figures/run_command_explanation.png)


# **6. Statistical Analysis**

---
### A [**Gallery of Statistical Analysis**](https://github.com/FlexTRKR/PyFLEXTRKR/blob/main/AnalysisGallery.md) that can be applied to the tracking outputs is provided. Below are some examples.

![](https://portal.nersc.gov/project/m1867/PyFLEXTRKR/sample_data/saag/figures/imerg_wrf_mcs_irnumber_map_annual_2018-2019.gif)

![](https://portal.nersc.gov/project/m1867/PyFLEXTRKR/sample_data/saag/figures/imerg_wrf_mcs_rainmap_annual_2018-2019.gif)

![](https://portal.nersc.gov/project/m1867/PyFLEXTRKR/sample_data/saag/figures/imerg_wrf_mcs_rainfrac_annual_2018-2019.gif)

![](https://portal.nersc.gov/project/m1867/PyFLEXTRKR/sample_data/saag/figures/kde1d_mcs_land_2x4_Amazon.gif)

![](https://portal.nersc.gov/project/m1867/PyFLEXTRKR/sample_data/saag/figures/mcs_composite_evolution_Amazon_land_crop.png)



# **7. References**

---

Feng, Z., Hardin, J., Barnes, H. C., Li, J., Leung, L. R., Varble, A., & Zhang, Z. (2023). PyFLEXTRKR: a flexible feature tracking Python software for convective cloud analysis. *Geosci. Model Dev.*, 16(10), 2753-2776. [https://doi.org/10.5194/gmd-16-2753-2023](https://doi.org/10.5194/gmd-16-2753-2023)

Feng, Z., Varble, A., Hardin, J., Marquis, J., Hunzinger, A., Zhang, Z., & Thieman, M. (2022). Deep Convection Initiation, Growth, and Environments in the Complex Terrain of Central Argentina during CACTI. *Monthly Weather Review*, 150(5), 1135-1155. [https://doi.org/10.1175/MWR-D-21-0237.1](https://doi.org/10.1175/MWR-D-21-0237.1)

Feng, Z., Leung, L. R., Liu, N., Wang, J., Houze, R. A., Li, J., et al. (2021). A Global High‐Resolution Mesoscale Convective System Database Using Satellite‐Derived Cloud Tops, Surface Precipitation, and Tracking. *Journal of Geophysical Research: Atmospheres*, 126(8). [https://doi.org/10.1029/2020JD034202](https://doi.org/10.1029/2020JD034202)

Feng, Z., Leung, L. R., Houze, R. A., Hagos, S., Hardin, J., Yang, Q., et al. (2018). Structure and Evolution of Mesoscale Convective Systems: Sensitivity to Cloud Microphysics in Convection-Permitting Simulations Over the United States. *Journal of Advances in Modeling Earth Systems*, 10(7), 1470-1494. [https://doi.org/10.1029/2018MS001305](https://doi.org/10.1029/2018MS001305)

Feng, Z., Houze, R. A., Leung, L. R., Song, F., Hardin, J. C., Wang, J., et al. (2019). Spatiotemporal Characteristics and Large-scale Environments of Mesoscale Convective Systems East of the Rocky Mountains. *Journal of Climate*, 32(21), 7303-7328. [https://doi.org/10.1175/JCLI-D-19-0137.1](https://doi.org/10.1175/JCLI-D-19-0137.1)

Feng, Z., Chen, X., & Leung, L. R. (2024). How Might the May 2015 Flood in the U.S. Southern Great Plains Induced by Clustered MCSs Unfold in the Future? *Journal of Geophysical Research: Atmospheres*, 129(8), e2023JD039605. [https://doi.org/10.1029/2023JD039605](https://doi.org/10.1029/2023JD039605)

---

