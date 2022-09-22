"""
Make animations from LASSO cell tracking quicklook plots.
"""
import os
import subprocess

if __name__ == "__main__":

    # Specify domain and framerate
    # domain = 'd3'
    # domain = 'd4'
    domain = 'd4_15min'

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

    # Input figures root directory
    in_dir_root = '/gpfs/wolf/atm131/proj-shared/zfeng/cacti/les/quicklooks_trackpaths/'
    # Output animation directory
    out_dir = f'{in_dir_root}animation/'
    os.makedirs(out_dir, exist_ok=True)

    # Determine framerate based on domain
    if domain == 'd4':
        framerate = 6  # for 5min
    elif (domain == 'd4_15min') | (domain == 'd3'):
        framerate = 2  # for 15min

    # Loop over dates
    for ii in range(0, len(start_dates)):
        idate = start_dates[ii]
        imember = ens_members[ii]
        in_dir = f'{in_dir_root}{idate}/{imember}/{domain}/'
        out_filename = f'{out_dir}{idate}_{imember}_{domain}.mp4'
        # Make ffmpeg command
        cmd = f"ffmpeg -framerate {framerate} -pattern_type glob -i '{in_dir}*.png' -c:v libx264 -r 10 -crf 20 -pix_fmt yuv420p {out_filename}"
        print(cmd)
        # Remove output file if it exists
        if os.path.isfile(out_filename):
            os.remove(out_filename)
        # Run command
        subprocess.run(f'{cmd}', shell=True)
        print(out_filename)