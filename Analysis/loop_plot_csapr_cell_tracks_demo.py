"""
Loop over CSAPR dates to plot quicklook plots for LASSO cases.
"""
import subprocess

if __name__ == "__main__":
    # start_dates = [
    #     "20181129", 
    # ]
    # ens_members = [
    #     "gefs00", 
    # ]
    # start_dates = [
    #     "20190123", 
    # ]
    # ens_members = [
    #     "eda05", 
    # ]
    # Full set of runs
    start_dates = [
        "20181129",
        "20181204",
        "20181205", 
        "20181219", 
        "20190122", 
        "20190123",
        "20190125",
        "20190129",
        "20190208",
    ]
    code_name = 'plot_subset_cell_tracks_demo.py'
    config_filename = '../config/config_csapr500m_cu2.yml'
    out_dir_root = '/gpfs/wolf/cli120/proj-shared/zfeng/cacti/csapr/quicklooks_trackpaths/'

    for ii in range(0, len(start_dates)):
        idate = start_dates[ii]
        # imember = ens_members[ii]
        year = idate[0:4]
        month = idate[4:6]
        day = idate[6:8]
        sdate = f'{year}-{month}-{day}T12:00'
        edate = f'{year}-{month}-{day}T23:55'
        out_dir = f'{out_dir_root}/{idate}/'        
        cmd = f'python {code_name} -s {sdate} -e {edate} -c {config_filename} -p 1 -o {out_dir}'
        print(cmd)
        subprocess.run(cmd, shell=True)
        
    # import pdb; pdb.set_trace()

    