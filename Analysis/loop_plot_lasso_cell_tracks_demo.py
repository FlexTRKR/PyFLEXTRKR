"""
Loop over LASSO runs to plot quicklook plots.
"""
import subprocess

if __name__ == "__main__":
    # start_dates = [
    #     "20181204", 
    # ]
    # ens_members = [
    #     "gefs_en18", 
    # ]
    # start_dates = [
    #     "20190123", 
    #     "20190129", "20190129",
    # ]
    # ens_members = [
    #     "eda05", 
    #     "eda09", "gefs11",
    # ]
    # Full set of runs
    start_dates = [
        "20181129", "20181129",
        "20181204", "20181204",
        "20181205", 
        "20181219", 
        "20190122", 
        "20190123",
        "20190125", "20190125",
        "20190129", "20190129",
        "20190208", "20190208",
    ]
    # # LES
    # ens_members = [
    #     "gefs00", "gefs03",
    #     "gefs18", "gefs19",
    #     "gefs01", 
    #     "eda09", 
    #     "gefs01", 
    #     "eda05",
    #     "eda07", "gefs11",
    #     "eda09", "gefs11",
    #     "eda03", "eda08",
    # ]
    # MESO
    ens_members = [
        "gefs_en00", "gefs_en03",
        "gefs_en18", "gefs_en19",
        "gefs_en01", 
        "eda_en09", 
        "gefs_en01", 
        "eda_en05",
        "eda_en07", "gefs_en11",
        "eda_en09", "gefs_en11",
        "eda_en03", "eda_en08",
    ]
    code_name = 'plot_subset_cell_tracks_demo.py'
    config_basename = '../config/config_lasso_'
    # out_dir_root = '/gpfs/wolf/cli120/proj-shared/zfeng/cacti/les/quicklooks_trackpaths/'
    out_dir_root = '/gpfs/wolf/cli120/proj-shared/zfeng/cacti/meso/quicklooks_trackpaths/'
    domain = 'd2'
    # domain = 'd3'
    # domain = 'd4'
    # domain = 'd4_15min'

    for ii in range(0, len(start_dates)):
        idate = start_dates[ii]
        imember = ens_members[ii]
        year = idate[0:4]
        month = idate[4:6]
        day = idate[6:8]
        sdate = f'{year}-{month}-{day}T12:00'
        edate = f'{year}-{month}-{day}T23:55'
        config = f'{config_basename}{idate}_{imember}.yml'
        out_dir = f'{out_dir_root}/{idate}/{imember}/{domain}/'        
        cmd = f'python {code_name} -s {sdate} -e {edate} -c {config} -p 1 -o {out_dir}'
        print(cmd)
        subprocess.run(cmd, shell=True)
        
    # import pdb; pdb.set_trace()

    