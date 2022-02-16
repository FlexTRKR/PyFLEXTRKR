import numpy as np
import os
import sys
from pyflextrkr.ft_utilities import load_config, convert_trackstats_sparse2dense

if __name__ == '__main__':
    # Load configuration file
    config_file = sys.argv[1]
    config = load_config(config_file)

    trackstats_filebase = config["trackstats_filebase"]
    trackstats_sparse_filebase = config["trackstats_sparse_filebase"]
    startdate = config["startdate"]
    enddate = config["enddate"]
    stats_path = config["stats_outpath"]
    duration_range = config["duration_range"]
    tracks_dimname = config["tracks_dimname"]
    times_dimname = config["times_dimname"]
    fillval = config["fillval"]
    fillval_f = np.nan

    # Get dimensions and indices variable names
    max_trackduration = int(max(duration_range))
    tracks_idx_varname = f"{tracks_dimname}_indices"
    times_idx_varname = f"{times_dimname}_indices"

    # Trackstats file names
    trackstats_sparse_file = f"{stats_path}{trackstats_sparse_filebase}{startdate}_{enddate}.nc"
    trackstats_dense_file = f"{stats_path}{trackstats_filebase}{startdate}_{enddate}.nc"

    # Delete file if it already exists
    if os.path.isfile(trackstats_dense_file):
        os.remove(trackstats_dense_file)

    # Call function to convert file
    status = convert_trackstats_sparse2dense(
        trackstats_sparse_file,
        trackstats_dense_file,
        max_trackduration,
        tracks_idx_varname,
        times_idx_varname,
        tracks_dimname,
        times_dimname,
        fillval,
        fillval_f,
    )
    print(f"{trackstats_dense_file}")

