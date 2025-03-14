from pyflextrkr.ft_utilities import get_basetime_from_string

# Date-string format: 'yyyymodd.hhmm'
datestring = "20250101.1200"
print(f"Date string {datestring}")

# Basic test to check the date format
def test_datestring_format():
    assert len(datestring) == 13, "Date string length should be 13"
    assert datestring[8] == '.', "Date string should have a dot at position 9"
    assert datestring[:8].isdigit(), "Date part should be digits"
    assert datestring[9:].isdigit(), "Time part should be digits"

# Test conversion of date-string to Epoch time
def test_get_basetime_from_string():
    base_time = get_basetime_from_string(datestring)
    print(f"Epoch time for date-string {datestring}: {base_time}")

    # Check if the returned base_time is a valid numeric value
    assert isinstance(base_time, (int, float)), "Epoch time should be a number"
    assert base_time > 0, "Epoch time should be positive"

# Placeholder for future numpy tests
def test_numpy_placeholder():
    assert True  # To make pytest happy
