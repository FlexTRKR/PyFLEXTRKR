"""
Tests of pyflextrkr.visualization.masks: the perimeter of labeled objects (bounding-box version == the loop over the
whole grid that the old plot_subset_*.py scripts had), the periodic option, and the object centers.
Needs numpy and scipy only.
"""
import numpy as np
import pytest
from scipy.ndimage import binary_erosion, generate_binary_structure

from pyflextrkr.visualization import masks


#-----------------------------------------------------------------------
def label_perimeter_loop(tracknumber, dilationstructure):
    """The original code of the plot scripts: erosion of every object on the whole grid."""
    tracknumber_unique = np.unique(tracknumber[~np.isnan(tracknumber)]).astype(np.int32)
    tracknumber_perim = np.zeros(tracknumber.shape, dtype=np.int32)
    for ii in tracknumber_unique:
        itn = tracknumber == ii
        itn_erode = binary_erosion(itn, structure=dilationstructure).astype(itn.dtype)
        iperim = np.logical_xor(itn, itn_erode)
        tracknumber_perim[iperim == 1] = ii
    return tracknumber_perim


def random_objects(seed, shape=(60, 80), nobj=12):
    """Track number mask (NaN = none) with random rectangles and discs, some of them touching the edges."""
    rng = np.random.default_rng(seed)
    tn = np.full(shape, np.nan)
    iy, ix = np.ogrid[:shape[0], :shape[1]]
    for n in range(1, nobj + 1):
        cy, cx = rng.integers(0, shape[0]), rng.integers(0, shape[1])
        if n % 2:
            tn[(abs(iy - cy) <= rng.integers(1, 6)) & (abs(ix - cx) <= rng.integers(1, 9))] = n
        else:
            tn[(iy - cy)**2 + (ix - cx)**2 <= rng.integers(2, 30)] = n
    return tn


#-----------------------------------------------------------------------
@pytest.mark.parametrize("structure", [
    generate_binary_structure(2, 1),                  # perim_width 1
    masks.make_perimeter_structure(1),
    masks.make_perimeter_structure(3),                # thicker outline
    masks.make_dilation_structure(10, 10, 10),        # km based, the MCS default
    masks.make_dilation_structure(20, 10, 10),
])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_label_perimeter_equals_loop_over_whole_grid(seed, structure):
    tn = random_objects(seed)
    assert np.array_equal(masks.label_perimeter(tn, structure), label_perimeter_loop(tn, structure))


def test_label_perimeter_edge_cases():
    st = generate_binary_structure(2, 1)
    # no object
    empty = np.full((10, 12), np.nan)
    out = masks.label_perimeter(empty, st)
    assert out.shape == empty.shape and out.dtype == np.int32 and not out.any()
    # one pixel, one object that fills the grid (the edge of the array is an object boundary)
    one = np.full((5, 5), np.nan)
    one[2, 2] = 7
    assert masks.label_perimeter(one, st)[2, 2] == 7
    full = np.ones((6, 6))
    perim = masks.label_perimeter(full, st)
    assert np.array_equal(perim, label_perimeter_loop(full, st))
    assert perim[0, 0] == 1 and perim[3, 3] == 0
    # track numbers that are not consecutive
    tn = np.full((9, 9), np.nan)
    tn[1:4, 1:4] = 5
    tn[5:8, 5:8] = 1234
    assert set(np.unique(masks.label_perimeter(tn, st))) == {0, 5, 1234}


def test_label_perimeter_periodic_edges():
    st = generate_binary_structure(2, 1)
    # an object that crosses the x edge: 3 columns at the right edge, 2 columns at the left edge, same rows
    tn = np.full((12, 20), np.nan)
    tn[3:9, 17:20] = 1
    tn[3:9, 0:2] = 1
    plain = masks.label_perimeter(tn, st)
    wrapped = masks.label_perimeter(tn, st, periodic=(True, False))
    # not periodic: the edge of the array is a boundary; periodic: no outline along the edge columns inside the rows
    assert plain[5, 0] == 1 and plain[5, 19] == 1
    assert wrapped[5, 0] == 0 and wrapped[5, 19] == 0
    # same as rolling the object into the middle, outlining and rolling back
    rolled = np.roll(tn, 10, axis=1)
    expected = np.roll(label_perimeter_loop(rolled, st), -10, axis=1)
    assert np.array_equal(wrapped, expected)
    # the same in y
    tn_y = tn.T.copy()
    wrapped_y = masks.label_perimeter(tn_y, st, periodic=(False, True))
    assert np.array_equal(wrapped_y, wrapped.T)
    # periodic in both directions: an object at a corner is one object
    corner = np.full((10, 10), np.nan)
    corner[0:2, 0:2] = 3
    corner[8:10, 8:10] = 3
    corner[0:2, 8:10] = 3
    corner[8:10, 0:2] = 3
    both = masks.label_perimeter(corner, st, periodic=(True, True))
    rolled = np.roll(corner, (5, 5), axis=(0, 1))
    assert np.array_equal(both, np.roll(label_perimeter_loop(rolled, st), (-5, -5), axis=(0, 1)))


def test_perimeter_structures():
    assert np.array_equal(masks.make_perimeter_structure(1), generate_binary_structure(2, 1))
    s3 = masks.make_perimeter_structure(3)
    assert s3.shape == (4, 4) and s3[1:3, 1:3].all() and s3.sum() == 4
    st = masks.make_dilation_structure(10, 10, 10)          # radius 1 grid
    assert st.shape == (3, 3) and st[1, 1]
    st = masks.make_dilation_structure(30, 10, 10)          # radius 3 grids, a disc
    assert st.shape == (7, 7) and st[3, 3] and st[0, 3] and not st[0, 0]


def test_cell_and_track_centers_equal_plain_loops():
    tn = random_objects(5, shape=(40, 50), nobj=8)
    ny, nx = tn.shape
    lon = 100. + 0.25 * np.arange(nx)[None, :] + 0.01 * np.arange(ny)[:, None]
    lat = -20. + 0.25 * np.arange(ny)[:, None] + 0.02 * np.arange(nx)[None, :]
    xx = np.arange(nx) * 2.0
    yy = np.arange(ny) * 3.0
    lon_c, lat_c, xx_c, yy_c, uniq = masks.calc_cell_center(tn, lon, lat, xx, yy)
    lon_t, lat_t, uniq_t = masks.calc_track_center(tn, lon, lat)
    expected = np.unique(tn[~np.isnan(tn)])
    assert np.array_equal(uniq, expected) and np.array_equal(uniq_t, expected)
    for k, n in enumerate(expected):
        iy, ix = np.where(tn == n)
        assert lon_c[k] == pytest.approx(lon[tn == n].mean())
        assert lat_c[k] == pytest.approx(lat[tn == n].mean())
        assert xx_c[k] == pytest.approx(xx[ix].mean())
        assert yy_c[k] == pytest.approx(yy[iy].mean())
        assert lon_t[k] == pytest.approx(lon_c[k]) and lat_t[k] == pytest.approx(lat_c[k])
    # no object: empty arrays
    out = masks.calc_track_center(np.full((4, 4), np.nan), lon[:4, :4], lat[:4, :4])
    assert all(len(a) == 0 for a in out)
