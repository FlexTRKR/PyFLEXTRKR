"""
Regression test for issue #147: `np.chararray` was removed from NumPy's main
namespace, breaking `gettracknumbers()` in pyflextrkr/gettracks.py.

This test does not need any tracking data. It reproduces the exact
write/read round trip for the `cloudid_files` character-array variable in a
`tracknumbers_*.nc` file:

  - write side mirrors pyflextrkr/gettracks.py (build the (nfiles,
    ncharacters) S1 array, write it via xr.Dataset.to_netcdf with the same
    NETCDF4_CLASSIC format and zlib encoding)
  - read side mirrors pyflextrkr/trackstats_driver.py (xr.open_dataset with
    mask_and_scale=False, decode_times=False, concat_characters=True,
    followed by netCDF4.chartostring)

No local/demo data is required, so this runs under plain `pytest tests/`.
"""
import inspect
import os

import numpy as np
import pytest
import xarray as xr
from netCDF4 import chartostring

import pyflextrkr.gettracks as gettracks


FILENAMES = [
    "cloudid_20200101_0000.nc",
    "cloudid_20200101_0030.nc",
    "cloudid_20200101_0100.nc",
]


def _build_cloudidfiles_array(filenames):
    """Build the (nfiles, ncharacters) char array the way gettracks.py does."""
    strlength = len(filenames[0])
    assert all(len(f) == strlength for f in filenames), \
        "test filenames must share a fixed length, like gettracks.py assumes"

    cloudidfiles = np.zeros((len(filenames), strlength), dtype="S1")
    for i, fname in enumerate(filenames):
        cloudidfiles[i, :] = list(fname)
    return cloudidfiles, strlength


def test_no_chararray_in_gettracks_source():
    """Guard against re-introducing the removed np.chararray API."""
    source = inspect.getsource(gettracks)
    assert "np.chararray" not in source, (
        "np.chararray was removed from NumPy's main namespace (issue #147); "
        "use np.zeros(shape, dtype='S1') instead"
    )


def test_cloudidfiles_array_is_s1_dtype():
    """np.zeros(..., dtype='S1') is the drop-in replacement for np.chararray."""
    cloudidfiles, strlength = _build_cloudidfiles_array(FILENAMES)
    assert cloudidfiles.dtype == np.dtype("S1")
    assert cloudidfiles.shape == (len(FILENAMES), strlength)


def test_cloudid_files_roundtrip_through_netcdf(tmp_path):
    """
    Full round trip: build the char array as gettracks.py does, write it out
    the same way, then read it back and decode it the same way
    trackstats_driver.py does. This is the produce/consume pair that
    issue #147 broke.
    """
    cloudidfiles, strlength = _build_cloudidfiles_array(FILENAMES)
    nfiles = len(FILENAMES)

    # --- Write side: mirrors pyflextrkr/gettracks.py ---
    var_dict = {
        "cloudid_files": (["nfiles", "ncharacters"], cloudidfiles),
    }
    coord_dict = {
        "nfiles": (["nfiles"], np.arange(nfiles)),
        "ncharacters": (["ncharacters"], np.arange(0, strlength)),
    }
    ds_out = xr.Dataset(var_dict, coords=coord_dict)

    outfile = tmp_path / "tracknumbers_test.nc"
    ds_out.to_netcdf(
        path=outfile,
        mode="w",
        format="NETCDF4_CLASSIC",
        encoding={"cloudid_files": {"zlib": True}},
    )

    # --- Read side: mirrors pyflextrkr/trackstats_driver.py ---
    ds_in = xr.open_dataset(
        outfile,
        mask_and_scale=False,
        decode_times=False,
        concat_characters=True,
    )
    values = ds_in["cloudid_files"].values
    ds_in.close()

    assert values.dtype == np.dtype("S1")
    assert ds_in["cloudid_files"].dims == ("nfiles", "ncharacters")
    assert values.shape == (nfiles, strlength)

    decoded = [chartostring(values[i]).item() for i in range(nfiles)]
    assert decoded == FILENAMES
