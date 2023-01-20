"""
Make task list and slurm script (Job Array) to plot GridRad MCS track path quicklooks.

Example task list:
python plot_subset_tbze_mcs_tracks_demo.py -s 2018-04-01T00 -e 2018-09-01T00 -c config.yml -o horizontal -p 1 --figsize 10 13 --output /figdir/ --figbasename mcs_
python plot_subset_tbze_mcs_tracks_demo.py -s 2019-04-01T00 -e 2019-09-01T00 -c config.yml -o horizontal -p 1 --figsize 10 13 --output /figdir/ --figbasename mcs_
...

Each line will be submitted as a slurm job using Job Array.
"""
import sys
import textwrap
import subprocess
import pandas as pd

if __name__ == "__main__":

    # Get inputs from command line
    start_year = int(sys.argv[1])
    end_year = int(sys.argv[2])
    # Examples:
    # start_year = 2018
    # end_year = 2020

    # Submit job at run time
    submit_job = True

    start_month = 4
    end_month = 9

    # period = f"{start_year[0:4]}"
    task_type = f"mcs_tbze_quicklook"

    # Python analysis code name
    code_dir = "/global/homes/f/feng045/program/PyFLEXTRKR/Analysis/"
    python_codename = f"{code_dir}plot_subset_tbze_mcs_tracks_demo.py"

    # Tracking config file
    config_dir = "/global/homes/f/feng045/program/PyFLEXTRKR/config/"
    config_basename = "config_gridrad_mcs_"

    fig_dir = "/global/cscratch1/sd/feng045/usa/gridrad_v2/quicklooks/"
    figbasename = "mcs_"
    figsize = "10 13"

    # Make task and slurm file name
    task_filename = f"tasklist_{task_type}.txt"
    slurm_filename = f"slurm.submit_{task_type}.sh"


    # Create the list of job tasks needed by SLURM...
    task_file = open(task_filename, "w")
    ntasks = 0

    # Loop over years
    for iyear in range(start_year, end_year+1):
        sdate = f"{iyear}-{start_month:02}-01T00"
        edate = f"{iyear}-{end_month:02}-01T00"
        out_dir = f"{fig_dir}{iyear}{start_month:02}01_{iyear}{end_month:02}01/"
        config = f"{config_dir}/{config_basename}{iyear}.yml"
        print(sdate, edate)
        cmd = f"python {python_codename} -s {sdate} -e {edate} -c {config} " + \
            f"-o horizontal -p 1 --figsize {figsize} --output {out_dir} --figbasename {figbasename}"
        task_file.write(f"{cmd}\n")
        ntasks += 1
    task_file.close()
    print(task_filename)

    # # Make monthly start/end dates for the tracking period
    # start_dates = pd.date_range(f'{start_date}', f'{end_date}', freq='1MS')
    # end_dates = start_dates + pd.DateOffset(months=1)

    # # Create task commands
    # for idate in start_dates: 
    #     cmd = f'python {python_codename} {config} {idate.year} {idate.month}'
    #     task_file.write(f"{cmd}\n")
    #     ntasks += 1
    # task_file.close()
    # print(task_filename)

    # import pdb; pdb.set_trace()

    # Create a SLURM submission script for the above task list...
    slurm_file = open(slurm_filename, "w")
    text = f"""\
        #!/bin/bash
        #SBATCH -J {task_type}
        #SBATCH -A m1867
        #SBATCH -t 00:30:00
        #SBATCH -q regular
        #SBATCH -N 1
        #SBATCH -n 64
        #SBATCH -C haswell
        #SBATCH --exclusive
        #SBATCH --output=log_{task_type}_%A_%a.log
        #SBATCH --mail-type=END
        #SBATCH --mail-user=zhe.feng@pnnl.gov
        #SBATCH --array=1-{ntasks}

        date
        source activate flextrkr
        # cd {code_dir}

        # Takes a specified line ($SLURM_ARRAY_TASK_ID) from the task file
        LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p {task_filename})
        echo $LINE
        # Run the line as a command
        $LINE

        date
        """
    slurm_file.writelines(textwrap.dedent(text))
    slurm_file.close()
    print(slurm_filename)

    # Run command
    if submit_job == True:
        # cmd = f'sbatch --array=1-{ntasks}%{njobs_run} {slurm_filename}'
        cmd = f'sbatch --array=1-{ntasks} {slurm_filename}'
        print(cmd)
        subprocess.run(f'{cmd}', shell=True)