"""
Loop over LASSO runs to make quicklook plots.
"""
import subprocess
import sys

if __name__ == "__main__":

    # Specify resolution: 'les' or 'meso'
    resolution = 'les'
    # Specify domain
    # domain = 'd2'
    # domain = 'd3'
    domain = 'd4'
    # domain = 'd4_15min'

    # start_dates = [
    #     "20181129", 
    # ]
    # ens_members = [
    #     "gefs00", 
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

    if resolution == 'les':
        # LES
        ens_members = [
            "gefs00", "gefs03",
            "gefs18", "gefs19",
            "gefs01", 
            "eda09", 
            "gefs01", 
            "eda05",
            "eda07", "gefs11",
            "eda09", "gefs11",
            "eda03", "eda08",
        ]
    elif resolution == 'meso':
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
    else:
        sys.exit('Unknown resolution!')

    code_name = 'plot_subset_cell_tracks_demo.py'
    config_basename = '../config/config_lasso_'
    out_dir_root = f'/gpfs/wolf/atm131/proj-shared/zfeng/cacti/{resolution}/quicklooks_trackpaths/'

    figsize = [8,7]
    extent = [-65.9, -63.6, -33.1, -31.15]
    radar_lon, radar_lat = -64.7284, -32.1264
    parallel = 1

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
        cmd = f'python {code_name} -s {sdate} -e {edate} -c {config} -p {parallel} --radar_lat {radar_lat} --radar_lon {radar_lon} ' + \
              f'--output {out_dir} --figsize {figsize[0]} {figsize[1]} --extent {extent[0]} {extent[1]} {extent[2]} {extent[3]}'
        print(cmd)
        subprocess.run(cmd, shell=True)
        
    