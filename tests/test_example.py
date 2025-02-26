from pyflextrkr.ft_utilities import get_basetime_from_string

# Date-string format: 'yyyymodd.hhmm'
datestring = "20250101.1200"

# Convert a date-string to Epoch time
base_time = get_basetime_from_string(datestring)
print(f"Epoch time for date-string {datestring}: {base_time}")