"""
Make animations from GridRad MCS tracking quicklook plots.
"""
import os
import sys
import subprocess

if __name__ == "__main__":

    # Get inputs from command line
    start_year = int(sys.argv[1])
    end_year = int(sys.argv[2])
    # Start/end months
    # start_month = 4
    # end_month = 8
    start_month = 1
    end_month = 12

    # Determine framerate
    framerate = 2

    # Scale the image (x:y), -1 preserves aspect ratio
    scale = '1200:-1'

    # Input figures root directory
    # in_dir_root = f'/global/cscratch1/sd/feng045/usa/gridrad_v2/quicklooks_maxze/'
    in_dir_root = f'/pscratch/sd/f/feng045/usa/gridrad_v3/quicklooks_maxze/'
    # Output animation directory
    out_dir = f'{in_dir_root}animation/'
    os.makedirs(out_dir, exist_ok=True)

    # Loop over years
    for iyear in range(start_year, end_year+1):
        # Loop over months
        for imon in range(start_month, end_month+1):
            idate = f'{iyear}{imon:02}'
            print(idate)
            # in_dir = f'{in_dir_root}/{iyear}0401_{iyear}0901/'
            in_dir = f'{in_dir_root}/{iyear}0101_{iyear}1231/'
            in_images = f'{in_dir}*{idate}*.png'
            out_filename = f'{out_dir}mcs_{idate}.mp4'
            # Make ffmpeg command
            cmd = f"ffmpeg -framerate {framerate} -pattern_type glob -i '{in_images}' -vf 'scale={scale}' -c:v libx264 -r 10 -crf 20 -pix_fmt yuv420p -y {out_filename}"
            print(cmd)
            # Remove output file if it exists
            if os.path.isfile(out_filename):
                os.remove(out_filename)
            # Run command
            subprocess.run(f'{cmd}', shell=True)
            print(out_filename)
