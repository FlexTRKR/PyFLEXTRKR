"""
Make task list and slurm script (Job Array) to calculate monthly MCS statistics.

Example task list:
python calc_tbpf_mcs_monthly_rainmap.py config.yml 2018 6
python calc_tbpf_mcs_monthly_rainmap.py config.yml 2018 7
python calc_tbpf_mcs_monthly_rainmap.py config.yml 2018 8
...

Each line will be submitted as a slurm job using Job Array.
"""
import sys
import textwrap
import subprocess
import pandas as pd

if __name__ == "__main__":

    # Get inputs from command line
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    data_source = sys.argv[3]
    # Examples:
    # start_date = '2018-6'
    # end_date = '2019-5'
    # data_source = 'gpm'

    # Submit job at run time
    submit_job = True

    period = f"{start_date[0:4]}-{end_date[0:4]}"
    # task_type = f"mcs_monthly_rainmap_{data_source}_saag"
    task_type = f"mcs_monthly_statsmap_{data_source}_saag"

    # Python analysis code name
    code_dir = "/ccsopen/home/zhe1feng1/program/PyFLEXTRKR/Analysis/"
    # python_codename = f"{code_dir}calc_tbpf_mcs_monthly_rainmap.py"
    python_codename = f"{code_dir}calc_tbpf_mcs_monthly_statsmap.py"

    # Tracking config file
    config_dir = "/ccsopen/home/zhe1feng1/program/pyflex_config/"
    config = f"{config_dir}config/config_{data_source}_mcs_saag_cu2_{period}.yml"

    # Make task and slurm file name
    task_filename = f"tasklist_{task_type}_{period}.txt"
    slurm_filename = f"slurm.submit_{task_type}_{period}.sh"


    # Make monthly start dates for the tracking period
    start_dates = pd.date_range(f'{start_date}', f'{end_date}', freq='1MS')

    # Create the list of job tasks needed by SLURM...
    task_file = open(task_filename, "w")
    ntasks = 0

    # Create task commands
    for idate in start_dates: 
        cmd = f'python {python_codename} {config} {idate.year} {idate.month}'
        task_file.write(f"{cmd}\n")
        ntasks += 1
    task_file.close()
    print(task_filename)

    # Create a SLURM submission script for the above task list...
    slurm_file = open(slurm_filename, "w")
    text = f"""\
        #!/bin/bash
        #SBATCH --job-name={period}
        #SBATCH -A atm131
        #SBATCH --time=02:00:00
        #SBATCH -p batch_all
        #SBATCH -N 1
        #SBATCH --ntasks=128
        #SBATCH --exclusive
        #SBATCH --output=log_{task_type}_{period}_%A_%a.log
        #SBATCH --mail-type=END
        #SBATCH --mail-user=zhe.feng@pnnl.gov
        #SBATCH --array=1-{ntasks}

        date
        # conda activate flextrkr
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