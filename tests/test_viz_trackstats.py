"""
Tests of pyflextrkr.visualization.trackstats (the tracks of a period as dictionaries) and pixelio (the index window
around a map extent, the figure names) on the synthetic data of tests/viz_synthetic.py. Needs xarray and netCDF4 only.
"""
import glob
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import viz_synthetic as S
from pyflextrkr.visualization import pixelio, trackstats


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("viz_trackstats")
    configs, paths = S.build(root)
    return configs, paths


DT = pd.Timedelta(hours=1).to_pytimedelta()


#-----------------------------------------------------------------------
def test_mcs_track_stats_period_selection(synthetic):
    _, paths = synthetic
    d = trackstats.get_mcs_track_stats(paths['mcs_stats'], '2025-06-01T00:00:00', '2025-06-01T04:00:00', DT)
    assert d['ntracks'] == 2
    assert d['time_res'] == 1.0
    assert list(d['lifetime'].values) == [3.0, 2.0]       # track_duration * time resolution [hour]
    assert d['track_ccs_lon'].shape == (2, S.NFRAMES)
    assert d['track_pf_diam'].shape == (2, S.NFRAMES)
    assert np.nanmax(d['track_pf_diam'].values) == pytest.approx(20.0)     # 2 sqrt(area / pi), area = pi 10^2
    # only the tracks that were initiated within the period
    d = trackstats.get_mcs_track_stats(paths['mcs_stats'], '2025-06-01T01:00:00', '2025-06-01T04:00:00', DT)
    assert d['ntracks'] == 0
    d = trackstats.get_mcs_track_stats(paths['mcs_stats'], '2025-06-01T00:00:00', '2025-06-01T00:10:00', DT)
    assert d['ntracks'] == 0


def test_mcs_track_stats_smooth_center_and_pf_fallback(synthetic, tmp_path):
    _, paths = synthetic
    ds = xr.open_dataset(paths['mcs_stats']).load()
    ds['meanlon_smooth'] = ds['meanlon'] + 1.0
    ds['meanlat_smooth'] = ds['meanlat'] + 1.0
    smooth_file = str(tmp_path / 'smooth.nc')
    ds.to_netcdf(smooth_file)
    args = (smooth_file, '2025-06-01T00:00:00', '2025-06-01T04:00:00', DT)
    smooth = trackstats.get_mcs_track_stats(*args)
    raw = trackstats.get_mcs_track_stats(*args, use_smooth=False)
    shift = smooth['track_ccs_lon'].values - raw['track_ccs_lon'].values
    np.testing.assert_allclose(shift[~np.isnan(shift)], 1.0)
    # no PF variables (Tb-only MCS tracking): the PF center is the CCS center and the PF diameter is all NaN
    nopf = ds[['base_time', 'track_duration', 'meanlon', 'meanlat']]
    nopf.attrs = ds.attrs
    nopf_file = str(tmp_path / 'nopf.nc')
    nopf.to_netcdf(nopf_file)
    d = trackstats.get_mcs_track_stats(nopf_file, '2025-06-01T00:00:00', '2025-06-01T04:00:00', DT)
    np.testing.assert_array_equal(d['track_pf_lon'].values, d['track_ccs_lon'].values)
    assert np.isnan(d['track_pf_diam'].values).all()


def test_cell_generic_and_cartesian_track_stats(synthetic):
    _, paths = synthetic
    period = ('2025-06-01T00:00:00', '2025-06-01T04:00:00', DT)
    cell = trackstats.get_cell_track_stats(paths['cell_stats'], *period)
    assert cell['ntracks'] == 2 and cell['cell_lon'].shape == (2, S.NFRAMES)
    assert set(cell) >= {'cell_bt', 'cell_lon', 'cell_lat', 'start_split_tracknumber', 'end_merge_tracknumber', 'lifetime'}
    generic = trackstats.get_generic_track_stats(paths['mcs_stats'], *period)
    assert generic['ntracks'] == 2 and generic['meanlat'].shape == (2, S.NFRAMES)
    # x / y domain: positions times the scale factors, the PF positions are optional
    raw = trackstats.get_cartesian_track_stats(paths['mcs_stats'], *period)
    scaled = trackstats.get_cartesian_track_stats(paths['mcs_stats'], *period, xscale=1e-3, yscale=2e-3)
    np.testing.assert_allclose(scaled['track_meanlon'].values, raw['track_meanlon'].values * 1e-3)
    np.testing.assert_allclose(scaled['track_meanlat'].values, raw['track_meanlat'].values * 2e-3)
    np.testing.assert_allclose(scaled['track_pflon'].values, raw['track_pflon'].values * 1e-3)
    assert 'track_pflon' not in trackstats.get_cartesian_track_stats(paths['mcs_stats'], *period, with_pf=False)


def test_cell_track_stats_min_hours(synthetic):
    _, paths = synthetic
    period = ('2025-06-01T00:00:00', '2025-06-01T04:00:00', DT)
    every = trackstats.get_cell_track_stats(paths['cell_stats'], *period)
    assert every['tracknumbers_all'] is None and every['ntracks'] == 2                # the default: all tracks
    # track 1 lives 3 hours, track 2 lives 2 hours; "at least" includes the track of exactly min_hours
    long_ = trackstats.get_cell_track_stats(paths['cell_stats'], *period, min_hours=2.5)
    assert long_['ntracks'] == 1 and list(long_['tracknumbers_all']) == [1]
    assert long_['cell_lon'].shape == (1, S.NFRAMES) and float(long_['lifetime'][0]) == 3.0
    both = trackstats.get_cell_track_stats(paths['cell_stats'], *period, min_hours=2.0)
    assert both['ntracks'] == 2 and list(both['tracknumbers_all']) == [1, 2]
    # no track long enough: no track and no track number
    none = trackstats.get_cell_track_stats(paths['cell_stats'], *period, min_hours=10)
    assert none['ntracks'] == 0 and len(none['tracknumbers_all']) == 0
    # no track in the period: still a valid answer
    empty = trackstats.get_cell_track_stats(paths['cell_stats'], '2025-07-01T00:00:00', '2025-07-01T04:00:00', DT, min_hours=2.5)
    assert empty['ntracks'] == 0 and list(empty['tracknumbers_all']) == [1]


#-----------------------------------------------------------------------
def make_grid_dataset(lon, lat, name='tb'):
    """Pixel file like dataset on a rectilinear grid given as 2D longitude / latitude."""
    lon2d, lat2d = np.meshgrid(lon, lat)
    ny, nx = lon2d.shape
    return xr.Dataset({name: (('time', 'y', 'x'), np.zeros((1, ny, nx))), 'longitude': (('y', 'x'), lon2d),
                       'latitude': (('y', 'x'), lat2d)})


def test_find_subset_window_regional_grid():
    ds = make_grid_dataset(np.arange(100., 130., 0.5), np.arange(-10., 10., 0.5))
    # extent 110-120E, 0-5N: the pixels inside plus one pixel margin on each side
    win = pixelio.find_subset_window(ds, 'tb', [110., 120., 0., 5.])
    assert set(win) == {'y', 'x'}
    sub = ds.isel(win)
    assert sub['longitude'].min() == 109.5 and sub['longitude'].max() == 120.5
    assert sub['latitude'].min() == -0.5 and sub['latitude'].max() == 5.5
    # the window is clipped at the edge of the grid
    win = pixelio.find_subset_window(ds, 'tb', [95., 105., -15., 0.])
    assert win['x'].start == 0 and win['y'].start == 0
    # no pixel inside: None (the whole grid is used)
    assert pixelio.find_subset_window(ds, 'tb', [10., 20., 0., 5.]) is None


def test_find_subset_window_of_the_whole_grid_is_no_subset():
    # --subset 1 with the extent of a global dataset: the window is the whole grid, so the whole domain is used
    ds = make_grid_dataset(np.arange(0., 360., 2.), np.arange(-10., 10.1, 1.0))
    assert pixelio.find_subset_window(ds, 'tb', [0., 360., -10., 10.]) is None
    assert pixelio.find_subset_window(ds, 'tb', [-1., 361., -11., 11.]) is None
    assert pixelio.find_subset_window(ds, 'tb', [10., 100., -10., 10.]) is not None
    # all longitudes, a band of latitudes: the whole width
    ds = make_grid_dataset(np.arange(-180., 180., 2.), np.arange(-60., 60.1, 1.0))
    win = pixelio.find_subset_window(ds, 'tb', [-180., 180., -20., 20.])
    assert win['x'] == slice(0, 180) and win['y'].stop - win['y'].start == 43      # 41 rows and one more on each side


@pytest.mark.parametrize("data_lon", ['0-360', '-180-180'])
@pytest.mark.parametrize("extent", [[170., -170., -5., 5.], [170., 190., -5., 5.]])
def test_find_subset_window_across_the_dateline(data_lon, extent):
    lon = np.arange(0., 360., 2.)
    if data_lon == '-180-180':
        lon = ((lon + 180) % 360) - 180
        lon = np.sort(lon)
        lon_all = lon
    else:
        lon_all = lon
    ds = make_grid_dataset(lon_all, np.arange(-10., 10.1, 1.0))
    win = pixelio.find_subset_window(ds, 'tb', extent)
    sub = ds.isel(win)
    lon_in = sub['longitude'].values % 360
    # the pixels 170-190E (data in 0-360) are there; for -180~180 data the window is the bounding box of the pixels
    # on both sides of the dateline: it needs the 2 columns at the edges of the array, so the box is wide (documented)
    assert ((lon_in >= 170) & (lon_in <= 190)).sum() >= 11 * sub['longitude'].shape[0]
    assert sub['latitude'].min() <= -5 and sub['latitude'].max() >= 5


def test_make_figure_name():
    ds = xr.Dataset({'x': (('time',), [1.0])}, coords={'time': [np.datetime64('2025-06-02T12:30:00')]})
    name, timestr, title = pixelio.make_figure_name(ds, '/out/', 'mcs_', 'date_time', 0, '')
    assert name == '/out/mcs_20250602_123000.png'
    assert timestr == '2025-06-02 12:30:00 UTC' and title == timestr
    name, _, title = pixelio.make_figure_name(ds, '/out/', 'img', 'sequence', 6, 'GPM Tb+IMERG V07')
    assert name == '/out/img00007.png'
    assert title == 'GPM Tb+IMERG V07 | 2025-06-02 12:30:00 UTC'


def test_get_data_extent(synthetic):
    configs, paths = synthetic
    ds = xr.open_dataset(sorted(glob.glob(paths['mcs_pixel_dir'] + '*.nc'))[0])
    assert pixelio.get_data_extent(ds) == [S.LON0, S.LON0 + S.DLON * (S.NX - 1), S.LAT0, S.LAT0 + S.DLAT * (S.NY - 1)]
