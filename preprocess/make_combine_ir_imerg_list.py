"""
Creates a list of dates for combining IR and IMERG data using Task Farmer
"""

import pandas as pd
import sys

if __name__ == "__main__":
    # Get Phase from input 
    Phase = sys.argv[1]

    if Phase == 'Summer':
        start_date = f"2016-08-01"
        end_date = f"2016-09-10"
    if Phase == 'Winter':
        start_date = f"2020-01-20"
        end_date = f"2020-03-01"

    task_filename = f"task_combine_{Phase}.txt"

    # Wrapper shell script name
    script_name = f"/pscratch/sd/f/feng045/codes/dyamond/preprocess/run_combine_ir_imerg.sh"
    # script_name = f"/global/homes/f/feng045/program/PyFLEXTRKR-dev/preprocess/run_combine_ir_imerg.sh"

    # Create a date range for the given year
    # Set the year
    # year = 2020
    # start_date = f"{year}-01-01"
    # end_date = f"{year}-12-31"
    date_list = pd.date_range(start=start_date, end=end_date, freq='1D')
    # Task file name
    # task_filename = f"task_combine_{year}.txt"

    # Open a text file in write mode
    out_file = open(task_filename, "w")

    # Loop over each day in the date range
    for day in date_list:
        # Format the date string 'yyyymmdd'
        date = day.strftime("%Y%m%d")
        # Make the command
        cmd = f"{script_name} {date} {Phase}\n"
        # Write to text file
        out_file.write(cmd)

    # Close the text file
    out_file.close()
    print(f"{task_filename}")