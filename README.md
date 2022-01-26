# PyFlexTRKR
PyFlexTRKR is a python package for both MCS and cell-tracking from Satellite and Radar data. 

---
To use PyFlexTRKR please use the attached environment.yml file and create a conda virtual environment
```bash
conda env create -f environment.yml
```
Then when running pyflextrkr you can first activate its virtual environment with
```bash
conda activate flextrkr
```
Then the package can be installed (in editable/dev mode for now) with
```bash
pip install -e .
```
Then any changes to the source code will be reflected in the running version.  

Example to run the code:
```bash
python ./runscripts/run_cacti_csapr.py ./config/config_csapr500m_nersc.yaml
```

## Note:
I'll need to update these readme notes later on if this ever gets used more widely, 
but this is meant primarily for us to use to coordinate right now. 
