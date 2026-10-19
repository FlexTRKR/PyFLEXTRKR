"""
End-to-end tests of the plot families on the synthetic data of tests/viz_synthetic.py: each command line (main(argv))
makes the PNG files of four frames, and the options of the new code change the pictures as they should.

The tests need matplotlib, cartopy, colormaps and colorcet (not in the CI environment: skipped there). Coastlines are
off (--draw_land 0), so no Natural Earth file is read and nothing is downloaded.
"""
import glob
import os

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("cartopy")
pytest.importorskip("colormaps")
pytest.importorskip("colorcet")
Image = pytest.importorskip("PIL.Image")

import cartopy.crs as ccrs
from cartopy.geodesic import Geodesic

import viz_synthetic as S
from pyflextrkr.visualization import mcs_cli, cell_cli, generic_cli, cartesian_cli, mcs_map, cell_map, generic_map

SMALL = ['--draw_land', '0', '--dpi', '50']      # small pictures, no coastline files
ONE_FRAME = ('2025-06-01T01', '2025-06-01T02')   # the 01:30 frame only


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    return S.build(tmp_path_factory.mktemp("viz_frames"))


@pytest.fixture
def run(data, tmp_path, monkeypatch):
    """run(module, config_key, argv, period=..., outname=...) -> the sorted PNG files that main(argv) made."""
    configs, _ = data
    counter = {'n': 0}

    def _run(module, key, argv, period=(S.START, S.END), outname=None, **kwargs):
        for m in (mcs_cli, cell_cli, generic_cli, cartesian_cli):
            monkeypatch.setattr(m, 'load_config', lambda f, make_dirs=True, c=configs[key]: dict(c))
        counter['n'] += 1
        out = tmp_path / (outname or f"out{counter['n']}")
        module.main(['-s', period[0], '-e', period[1], '-c', 'unused.yml', '--output', str(out), '-p', '0'] + argv, **kwargs)
        return sorted(glob.glob(f'{out}/*.png'))
    return _run


def load(path):
    return np.asarray(Image.open(path).convert('RGB')).astype(int)


def count_color(img, rgb, tol=40):
    return int((np.abs(img - np.array(rgb)).max(axis=2) <= tol).sum())


def mean_difference(a, b):
    assert a.shape == b.shape
    return np.abs(a - b).mean()


#-----------------------------------------------------------------------
def test_mcs_tbpf_two_panels_both_layouts(run, capsys):
    pictures = {}
    for layout in ('top_bottom', 'left_right'):
        files = run(mcs_cli, 'mcs', SMALL + ['--layout', layout, '--figsize', '10', '8', '--extent'] + [str(v) for v in S.EXTENT])
        assert [os.path.basename(f) for f in files] == [f"{t:%Y%m%d_%H%M%S}.png" for t in S.TIMES]
        assert Image.open(files[0]).size == (500, 400)          # figsize x dpi
        pictures[layout] = load(files[1])
    assert mean_difference(pictures['top_bottom'], pictures['left_right']) > 0
    # the old names of the option and its values: the same pictures, with a deprecation warning
    capsys.readouterr()
    for old, new in (('horizontal', 'top_bottom'), ('vertical', 'left_right')):
        files = run(mcs_cli, 'mcs', SMALL + ['-o', old, '--figsize', '10', '8', '--extent'] + [str(v) for v in S.EXTENT])
        assert np.array_equal(load(files[1]), pictures[new])
        assert 'DeprecationWarning' in capsys.readouterr().err


@pytest.mark.parametrize("options", [
    ['--product', 'tbze'],
    ['--product', 'tbpf', '--panels', '1'],
    ['--product', 'tbze', '--panels', '1'],
    ['--product', 'tb'],
    ['--product', 'tbze', '--var2_varname', 'reflectivity_comp', '--subset', '1', '--extent', '105', '125', '-8', '8'],
], ids=lambda o: ' '.join(o))
def test_mcs_products_make_a_frame_per_time(run, options):
    files = run(mcs_cli, 'mcs', SMALL + options)
    assert len(files) == S.NFRAMES
    assert all(os.path.getsize(f) > 3000 for f in files)


def test_skip_empty_leaves_out_the_frames_without_tracks(run):
    assert len(run(mcs_cli, 'mcs', SMALL + ['--skip_empty', '0'])) == 4
    assert len(run(mcs_cli, 'mcs', SMALL + ['--skip_empty', '1'])) == 3


def test_robinson_projection_with_central_longitude_and_lat_range(run):
    files = run(mcs_cli, 'mcs', SMALL + ['--projection', 'Robinson', '--central_longitude', '115', '--extent', '100', '130', '-10', '10'])
    assert len(files) == 4
    # a global map of the 60S-60N band, periodic outlines (the V3 global MCS maps)
    files = run(mcs_cli, 'mcs', SMALL + ['--projection', 'Robinson', '--central_longitude', '225', '--extent', '0', '360', '-90', '90',
                                         '--lat_range', '-60', '60', '--periodic_perimeter', 'x'], period=ONE_FRAME)
    assert len(files) == 1
    width, height = Image.open(files[0]).size
    assert height < width * 1.5       # the two panels of a 0.374 aspect map, not a square map
    # the projection changes the picture
    plain = run(mcs_cli, 'mcs', SMALL + ['--extent', '100', '130', '-10', '10'], period=ONE_FRAME)
    lcc = run(mcs_cli, 'mcs', SMALL + ['--projection', 'LambertConformal', '--extent', '100', '130', '-10', '10'], period=ONE_FRAME)
    assert load(plain[0]).shape != load(lcc[0]).shape or mean_difference(load(plain[0]), load(lcc[0])) > 0


@pytest.mark.parametrize("projection_options", [
    ['--projection', 'Robinson', '--central_longitude', '0', '--extent', '0', '360', '-90', '90'],
    ['--projection', 'Mollweide', '--central_longitude', '115', '--extent', '0', '360', '-90', '90'],
    ['--projection', 'EqualEarth', '--central_longitude', '180', '--extent', '0', '360', '-90', '90', '--lat_range', '-40', '40'],
    ['--projection', 'Mercator', '--extent', '90', '140', '-20', '20'],
    ['--projection', 'Mercator', '--extent', '0', '360', '-60', '60', '--lat_range', '-50', '50'],
    ['--projection', 'LambertConformal', '--central_longitude', '115', '--standard_parallels', '-15', '-5', '--extent', '95', '135', '-15', '15'],
    ['--projection', 'AlbersEqualArea', '--central_longitude', '115', '--central_latitude', '0', '--standard_parallels', '5', '15',
     '--extent', '95', '135', '-15', '15'],
    ['--projection', 'Orthographic', '--central_longitude', '115', '--central_latitude', '0', '--extent', '0', '360', '-90', '90'],
    ['--projection', 'NorthPolarStereo', '--extent', '-180', '180', '-10', '90'],
    ['--projection', 'SouthPolarStereo', '--extent', '-180', '180', '-90', '10'],
], ids=lambda o: o[1] + ('_lat_range' if '--lat_range' in o else ''))
def test_projections_make_a_frame(run, projection_options):
    assert len(run(mcs_cli, 'mcs', SMALL + ['--panels', '1'] + projection_options, period=ONE_FRAME)) == 1


def test_lat_range_for_a_polar_projection_is_an_error(run):
    with pytest.raises(ValueError, match='lat_range'):
        run(mcs_cli, 'mcs', SMALL + ['--projection', 'NorthPolarStereo', '--extent', '-180', '180', '20', '90', '--lat_range', '20', '90'],
            period=ONE_FRAME)


def test_track_numbers_can_be_switched_off_for_one_panel(run):
    """--show_track_numbers 1 0: numbers on panel (a) (top, 2 panels stacked), none on panel (b) (bottom)."""
    options = SMALL + ['--layout', 'top_bottom', '--figsize', '10', '12']
    both = load(run(mcs_cli, 'mcs', options + ['--show_track_numbers', '1', '1'], period=ONE_FRAME)[0])
    first = load(run(mcs_cli, 'mcs', options + ['--show_track_numbers', '1', '0'], period=ONE_FRAME)[0])
    none = load(run(mcs_cli, 'mcs', options + ['--show_track_numbers', '0'], period=ONE_FRAME)[0])
    default = load(run(mcs_cli, 'mcs', options, period=ONE_FRAME)[0])
    assert np.array_equal(both, default)
    height = both.shape[0]
    top, bottom = slice(0, int(0.42 * height)), slice(int(0.5 * height), height)
    # panel (a) is the same with and without the numbers of panel (b); panel (b) changes
    assert np.array_equal(both[top], first[top])
    assert not np.array_equal(both[bottom], first[bottom])
    # no numbers at all: both panels change; and the dark pixels (black text) are fewer
    assert not np.array_equal(first[top], none[top]) and np.array_equal(first[bottom], none[bottom])
    assert count_color(none, (0, 0, 0), tol=60) < count_color(first, (0, 0, 0), tol=60) < count_color(both, (0, 0, 0), tol=60)
    # one value per panel: a second value for a 1-panel map is an error
    with pytest.raises(ValueError, match='one per panel'):
        run(mcs_cli, 'mcs', SMALL + ['--panels', '1', '--show_track_numbers', '1', '0'], period=ONE_FRAME)


def test_map_keeps_its_shape_and_the_colorbars_follow_the_map(run, monkeypatch):
    recorded = []
    original = mcs_map.fit_colorbars_to_maps

    def spy(pairs, orientation):
        original(pairs, orientation)
        for ax, caxes in pairs:
            recorded.append((orientation, ax.get_position(), [c.get_position() for c in caxes], ax.figure.get_size_inches()))
    monkeypatch.setattr(mcs_map, 'fit_colorbars_to_maps', spy)
    # a figure much wider than the map (30E x 20N): the map is centered in its cell and the colorbars have its extent
    run(mcs_cli, 'mcs', SMALL + ['--panels', '1', '--figsize', '12', '5', '--extent', '100', '130', '-10', '10'], period=ONE_FRAME)
    orientation, pos, cpos, size = recorded[-1]
    assert orientation == 'vertical' and len(cpos) == 2
    assert pos.width * size[0] / (pos.height * size[1]) == pytest.approx(30. / 20., rel=1e-3)      # lon span / lat span
    for c in cpos:
        assert (c.y0, c.height) == pytest.approx((pos.y0, pos.height)) and c.x0 > pos.x1
    # 2 panels: a colorbar for each map; stacked panels have colorbars at the right, side by side panels below
    recorded.clear()
    run(mcs_cli, 'mcs', SMALL + ['--layout', 'left_right', '--figsize', '14', '5', '--extent', '100', '130', '-10', '10'], period=ONE_FRAME)
    assert [r[0] for r in recorded] == ['horizontal', 'horizontal'] and len(recorded) == 2
    for orientation, pos, cpos, size in recorded:
        assert pos.width * size[0] / (pos.height * size[1]) == pytest.approx(1.5, rel=1e-3)
        assert (cpos[0].x0, cpos[0].width) == pytest.approx((pos.x0, pos.width)) and cpos[0].y0 < pos.y0
    # --map_aspect auto: the old look, the map fills its cell and nothing is moved
    recorded.clear()
    run(mcs_cli, 'mcs', SMALL + ['--panels', '1', '--figsize', '12', '5', '--map_aspect', 'auto'], period=ONE_FRAME)
    assert recorded == []


def test_colorbar_text_stays_inside_the_figure_of_the_frames(run, monkeypatch):
    """A small figure with large text: the labels of the 2 colorbars of a 1-panel map end inside the figure, clear of each other."""
    seen = []
    original = mcs_map.keep_colorbar_text_inside

    def spy(fig, pairs, orientation, **kwargs):
        original(fig, pairs, orientation, **kwargs)
        renderer = fig.canvas.get_renderer()
        seen.append([fig.transFigure.inverted().transform_bbox(c.get_tightbbox(renderer)) for ax, caxes in pairs for c in caxes])
    monkeypatch.setattr(mcs_map, 'keep_colorbar_text_inside', spy)
    options = SMALL + ['--panels', '1', '--figsize', '5', '5', '--fontsize', '14', '--extent', '100', '130', '-10', '10']
    run(mcs_cli, 'mcs', options, period=ONE_FRAME)
    first, second = seen[-1]
    assert second.x1 <= 1.0 and first.x1 <= second.x0 + 1e-6
    # a gap that was asked for is not widened (--cbar_spacing), the text still ends inside the figure
    run(mcs_cli, 'mcs', options + ['--cbar_spacing', '0.01', '--cbar_thickness', '0.02'], period=ONE_FRAME)
    first, second = seen[-1]
    assert second.x1 <= 1.0 and first.x1 > second.x0
    # 2 panels: one colorbar for each map, the same shrink for both
    run(mcs_cli, 'mcs', SMALL + ['--layout', 'top_bottom', '--figsize', '5', '6', '--fontsize', '14', '--extent', '100', '130', '-10', '10'],
        period=ONE_FRAME)
    assert len(seen[-1]) == 2 and all(b.x1 <= 1.0 for b in seen[-1])
    run(mcs_cli, 'mcs', SMALL + ['--layout', 'left_right', '--figsize', '8', '5', '--fontsize', '14', '--extent', '100', '130', '-10', '10'],
        period=ONE_FRAME)
    assert len(seen[-1]) == 2 and all(b.y0 >= 0.0 for b in seen[-1])


def test_colorbar_options_change_the_frame(run):
    base = load(run(mcs_cli, 'mcs', SMALL + ['--panels', '1', '--figsize', '8', '6'], period=ONE_FRAME)[0])
    thin = load(run(mcs_cli, 'mcs', SMALL + ['--panels', '1', '--figsize', '8', '6', '--cbar_thickness', '0.006'], period=ONE_FRAME)[0])
    wide = load(run(mcs_cli, 'mcs', SMALL + ['--panels', '1', '--figsize', '8', '6', '--cbar_spacing', '0.2'], period=ONE_FRAME)[0])
    assert thin.shape == wide.shape == base.shape
    assert mean_difference(thin, base) > 0 and mean_difference(wide, base) > 0
    # the colorbars of 2 panels: thickness for both layouts
    for layout in ('top_bottom', 'left_right'):
        options = SMALL + ['--layout', layout, '--figsize', '8', '8']
        default = load(run(mcs_cli, 'mcs', options, period=ONE_FRAME)[0])
        thick = load(run(mcs_cli, 'mcs', options + ['--cbar_thickness', '0.05'], period=ONE_FRAME)[0])
        assert mean_difference(thick, default) > 0


def test_pf_circles_off_take_the_entry_out_of_the_legend(run, monkeypatch):
    """--show_pfdiam 0: no PF circles, and no 'PF Diam' entry in the legend of panel (b)."""
    legends = []
    original = mcs_map.keep_colorbar_text_inside

    def spy(fig, pairs, orientation, **kwargs):
        original(fig, pairs, orientation, **kwargs)
        legends.append([[t.get_text() for t in ax.get_legend().get_texts()] for ax, caxes in pairs])
    monkeypatch.setattr(mcs_map, 'keep_colorbar_text_inside', spy)
    options = SMALL + ['--figsize', '10', '8', '--extent'] + [str(v) for v in S.EXTENT]
    on = load(run(mcs_cli, 'mcs', options + ['--show_pfdiam', '1'], period=ONE_FRAME)[0])
    assert legends[-1] == [['MCS Tracks', 'MCS Mask'], ['MCS Tracks', 'PF Diam (x1.0)']]
    off = load(run(mcs_cli, 'mcs', options + ['--show_pfdiam', '0'], period=ONE_FRAME)[0])
    assert legends[-1] == [['MCS Tracks', 'MCS Mask'], ['MCS Tracks']]
    assert mean_difference(on, off) > 0


def test_figure_size_follows_the_shape_of_the_map_when_it_is_not_given(run, capsys):
    run(mcs_cli, 'mcs', SMALL + ['--panels', '1', '--extent', '100', '130', '-10', '10'], period=ONE_FRAME)
    out = capsys.readouterr().out
    width, height = [float(v) for v in out.split('Figure size (width, height) in inches: [')[1].split(']')[0].split(',')]
    # the figure is about as high as the map (20 degrees) is wide ... plus the margins of the layout: shorter than wide
    assert width == 10. and 3. < height < 8.
    # a map much wider than high (the 60S-60N band of the globe): the figure is not shorter than the titles need
    run(mcs_cli, 'mcs', SMALL + ['--panels', '1', '--projection', 'Robinson', '--central_longitude', '225', '--extent', '0', '360', '-90', '90',
                                 '--lat_range', '-60', '60'], period=ONE_FRAME)
    out = capsys.readouterr().out
    width, height = [float(v) for v in out.split('Figure size (width, height) in inches: [')[1].split(']')[0].split(',')]
    assert width == 10. and 3.7 < height < 4.5                       # 3.7 for the shape of the map alone, and the margins of the titles in inches
    run(mcs_cli, 'mcs', SMALL + ['--panels', '1', '--extent', '100', '130', '-10', '10', '--map_aspect', 'auto'], period=ONE_FRAME)
    out = capsys.readouterr().out
    assert 'Figure size (width, height) in inches: [10, 10]' not in out      # the old formula: width x aspect, no layout margins
    width, height = [float(v) for v in out.split('Figure size (width, height) in inches: [')[1].split(']')[0].split(',')]
    assert height == pytest.approx(10. * 20. / 30., abs=0.01)


def test_the_domain_of_the_data_in_the_projection_without_extent(run, monkeypatch):
    extents = []
    original = mcs_map.set_map_domain

    def spy(ax, proj, map_extent, ext_lon0=None, ext_lon1=None, lat_range=None, extent_xy=None):
        extents.append(extent_xy)
        return original(ax, proj, map_extent, ext_lon0, ext_lon1, lat_range, extent_xy)
    monkeypatch.setattr(mcs_map, 'set_map_domain', spy)
    lcc = ['--projection', 'LambertConformal', '--central_longitude', '115', '--standard_parallels', '5', '15']
    run(mcs_cli, 'mcs', SMALL + ['--panels', '1'] + lcc, period=ONE_FRAME)
    assert extents[-1] is not None and len(extents[-1]) == 4
    # with --extent the box asked for decides, not the data; PlateCarree has no projected domain either
    run(mcs_cli, 'mcs', SMALL + ['--panels', '1', '--extent', '100', '130', '-10', '10'] + lcc, period=ONE_FRAME)
    assert extents[-1] is None
    run(mcs_cli, 'mcs', SMALL + ['--panels', '1'], period=ONE_FRAME)
    assert extents[-1] is None


def test_track_style_options_change_the_picture(run):
    base = load(run(mcs_cli, 'mcs', SMALL, period=ONE_FRAME)[0])
    variants = {
        'halo': ['--track_halo_color', 'white', '--track_halo_width', '3'],
        'halo_alpha': ['--track_halo_color', 'white', '--track_halo_width', '3', '--track_halo_alpha', '0.3'],
        'color_lw': ['--track_color', 'red', '--track_lw', '4'],
        'marker': ['--track_marker', '*', '--track_marker_size', '14', '--track_marker_color', 'blue'],
        'text': ['--track_text_size', '20', '--track_text_color', 'green', '--track_text_alpha', '0.5'],
        'no_numbers': ['--show_track_numbers', '0'],
        'linestyle_alpha': ['--track_ls', 'dashed', '--track_alpha', '0.4'],
        'grid_alpha': ['--grid_alpha', '0.2'],
    }
    images = {}
    for name, options in variants.items():
        images[name] = load(run(mcs_cli, 'mcs', SMALL + options, period=ONE_FRAME)[0])
        assert images[name].shape == base.shape, name
        assert mean_difference(images[name], base) > 0, f"--{name} does not change the picture"
    # a halo with alpha is lighter than the opaque halo: closer to the picture without halo
    assert mean_difference(images['halo_alpha'], base) < mean_difference(images['halo'], base)
    # path color and width: red pixels on the picture
    assert count_color(images['color_lw'], (255, 0, 0)) > count_color(base, (255, 0, 0)) + 20
    assert count_color(images['marker'], (0, 0, 255)) > count_color(base, (0, 0, 255))
    # the track numbers are black text: fewer dark pixels without them
    assert count_color(images['no_numbers'], (0, 0, 0), tol=60) < count_color(base, (0, 0, 0), tol=60)


def test_track_lookback_days_decides_which_older_tracks_are_drawn(run):
    # the frame at 01:30 of tracks that were initiated at 00:30, before the start of the period
    with_tracks = load(run(mcs_cli, 'mcs', SMALL, period=ONE_FRAME)[0])
    without = load(run(mcs_cli, 'mcs', SMALL + ['--track_lookback_days', '0'], period=ONE_FRAME)[0])
    assert mean_difference(with_tracks, without) > 0
    assert count_color(without, (128, 0, 128)) < count_color(with_tracks, (128, 0, 128))     # the track color 'purple'


def test_cell_frames(run):
    options = SMALL + ['--radar_lat', '0.0', '--radar_lon', '115.0']
    files = run(cell_cli, 'cell', options)
    assert [os.path.basename(f) for f in files] == [f"{t:%Y%m%d_%H%M%S}.png" for t in S.TIMES]
    assert len(run(cell_cli, 'cell', options + ['--skip_empty', '1'])) == 3
    # the radar location defaults to the center of the domain, subset and extent
    assert len(run(cell_cli, 'cell', SMALL + ['--subset', '1', '--extent', '105', '125', '-8', '8'], period=ONE_FRAME)) == 1
    # track style with halo
    base = load(run(cell_cli, 'cell', options, period=ONE_FRAME)[0])
    halo = load(run(cell_cli, 'cell', options + ['--track_halo_color', 'white', '--track_halo_alpha', '0.5'], period=ONE_FRAME)[0])
    assert mean_difference(base, halo) > 0


def test_cell_options_of_plotmap_change_the_picture_as_intended(run, data, monkeypatch):
    # no circles and lines (they are tested above); the 01:30 frame has both cells
    options = SMALL + ['--radar_lat', '0.0', '--radar_lon', '115.0', '--show_rangecircle', '0', '--show_azimuth', '0', '--perim_alpha', '1']
    base = load(run(cell_cli, 'cell', options, period=ONE_FRAME)[0])
    dark = lambda img: count_color(img, (0, 0, 0), tol=30)             # the outlines (opaque), the numbers, the text
    # only the cells of tracks that live at least 2.5 hours: the cell of track 2 (2 hours) has no outline, number or path
    long_only = load(run(cell_cli, 'cell', options + ['--min_hours', '2.5'], period=ONE_FRAME)[0])
    assert dark(long_only) < dark(base)
    assert mean_difference(long_only, load(run(cell_cli, 'cell', options + ['--cell_min_hours', '2.5'], period=ONE_FRAME)[0])) == 0   # plotmap's name
    # one outline color per track: no longer black; the same colors with the other cell removed (the first track keeps its color)
    colored = load(run(cell_cli, 'cell', options + ['--perim_cmap', 'tab10'], period=ONE_FRAME)[0])
    assert mean_difference(colored, base) > 0 and dark(colored) < dark(base)
    # a ring just outside the outlines (red here), under the outlines
    ring = load(run(cell_cli, 'cell', options + ['--perim_halo', 'red'], period=ONE_FRAME)[0])
    assert count_color(ring, (255, 0, 0), tol=30) > count_color(base, (255, 0, 0), tol=30)
    # terrain under the reflectivity (the file of the synthetic data: 0 to 3000 m from west to east)
    terrain = load(run(cell_cli, 'cell', options + ['--show_terrain', '1'], period=ONE_FRAME)[0])
    assert mean_difference(terrain, base) > 0
    configs, _ = data
    no_terrain = {k: v for k, v in configs['cell'].items() if k != 'terrain_file'}
    monkeypatch.setattr(cell_cli, 'load_config', lambda f, make_dirs=True: dict(no_terrain))
    with pytest.raises(SystemExit):
        cell_cli.main(['-s', ONE_FRAME[0], '-e', ONE_FRAME[1], '-c', 'unused.yml', '-p', '0', '--output', '/nonexistent/', '--show_terrain', '1'] + SMALL)
    # the line style of the gridlines
    grid = options + ['--grid_linewidth', '1', '--dpi', '100']
    solid = load(run(cell_cli, 'cell', grid + ['--grid_linestyle', '-'], period=ONE_FRAME)[0])
    dotted = load(run(cell_cli, 'cell', grid + ['--grid_linestyle', ':'], period=ONE_FRAME)[0])
    assert mean_difference(solid, dotted) > 0


def test_cell_cities_and_other_radars_are_marked(run, data):
    configs, _ = data
    configs['cell_clear'] = dict(configs['cell'], radar_sensitivity=6.0)       # the 5 dBZ background is not drawn: white, the cells stay
    options = SMALL + ['--radar_lat', '0.0', '--radar_lon', '115.0', '--show_rangecircle', '0', '--show_azimuth', '0', '--dpi', '100']
    run_clear = lambda extra: load(run(cell_cli, 'cell_clear', options + extra, period=ONE_FRAME)[0])
    base = run_clear([])
    yellow, cyan = (255, 255, 0), (0, 255, 255)
    # a city: a yellow diamond and its name; the name is black
    city = run_clear(['--city', 'Town', '125', '-5'])
    assert count_color(city, yellow) > count_color(base, yellow) + 10
    assert count_color(city, cyan) == count_color(base, cyan)
    # another radar: a cyan circle with its name and a range circle (radius 0: no circle)
    marker = run_clear(['--radar_site', 'Radar', '110', '6', '--radar_site_radius', '0'])
    ring = run_clear(['--radar_site', 'Radar', '110', '6', '--radar_site_radius', '500'])
    assert count_color(marker, cyan) > count_color(base, cyan) + 10
    assert count_color(ring, cyan) > count_color(marker, cyan)
    # a place outside the map is not drawn (and has no label in the margin)
    outside = run_clear(['--city', 'Far', '40', '50'])
    assert np.array_equal(outside, base)


@pytest.mark.parametrize("projection", [[], ['--projection', 'LambertConformal', '--central_longitude', '115', '--standard_parallels', '30', '40']])
def test_cell_range_circles_and_azimuth_lines_are_centered_on_the_radar(run, monkeypatch, projection):
    """
    The range circles and the azimuth lines are geodesic lines in lon / lat, drawn with the transform of the data: the right radius
    around the radar, and registered like the data. (ax.tissot and transform=ccrs.Geodetic() were not: with PROJ 9.8 they are
    moved north or south by up to 0.19 degree, most at mid-latitudes, so the radar is at 35N. With the older PROJ of this test
    environment the display checks pass for both ways of drawing: they are the guard where PROJ 9.8 is used.)
    """
    figures = []
    plot_map = cell_map.plot_map
    monkeypatch.setattr(cell_map, 'plot_map', lambda *args, **kwargs: figures.append(plot_map(*args, **kwargs)) or figures[-1])
    radar = np.array([115.0, 35.0])
    run(cell_cli, 'cell', ['--draw_land', '0', '--dpi', '100', '--figsize', '10', '8', '--radar_lon', '115', '--radar_lat', '35'] + projection,
        period=ONE_FRAME)
    ax = figures[0].axes[0]
    thin = [line for line in ax.lines if line.get_linewidth() == pytest.approx(0.4) and line.get_color() == 'k']
    rings = [line for line in thin if len(line.get_xdata()) == 100]
    rays = [line for line in thin if len(line.get_xdata()) == 21]
    assert len(rings) == 5 and len(rays) == 5
    # where things are on the figure: the transform of the data, as the clouds and the tracks are drawn
    def placed_like_the_data(points):
        return ax.transData.transform(ax.projection.transform_points(ccrs.PlateCarree(), points[:, 0], points[:, 1])[:, :2])
    radar_xy = placed_like_the_data(radar[None, :])[0]
    for line, radius_km in zip(rings, range(20, 101, 20)):
        points = np.column_stack([line.get_xdata(), line.get_ydata()])
        np.testing.assert_allclose(Geodesic().inverse(np.tile(radar, (len(points), 1)), points)[:, 0], radius_km * 1e3, atol=1.)   # the radius
        shown = line.get_transform().transform(points)
        np.testing.assert_allclose(shown, placed_like_the_data(points), atol=1e-6)                       # drawn with the transform of the data
        radius_px = np.hypot(*(shown - radar_xy).T).mean()
        assert np.hypot(*(shown[:-1].mean(axis=0) - radar_xy)) < 0.005 * radius_px                      # centered on the radar
    for line, azimuth in zip(rays, (0, 90, 180, 270, 360)):
        points = np.column_stack([line.get_xdata(), line.get_ydata()])
        np.testing.assert_allclose(line.get_transform().transform(points), placed_like_the_data(points), atol=1e-6)
        np.testing.assert_allclose(placed_like_the_data(points[:1])[0], radar_xy, atol=1e-6)            # starts at the radar
        out = Geodesic().inverse(radar[None, :], points[-1][None, :])
        assert out[0, 0] == pytest.approx(200e3, abs=1.) and out[0, 1] % 360 == pytest.approx(azimuth % 360, abs=1e-4)


def test_generic_frames(run):
    options = SMALL + ['--field_varname', 'tb', '--cmap', 'viridis', '--levels', '200', '300', '5', '--cbticks', '200', '300', '20',
                       '--cblabel', 'Tb (K)']
    files = run(generic_cli, 'mcs', options)
    assert len(files) == S.NFRAMES
    # an extent and a projection
    files = run(generic_cli, 'mcs', options + ['--extent', '100', '130', '-10', '10', '--projection', 'Mollweide'], period=ONE_FRAME)
    assert len(files) == 1


def test_cartesian_frames(run, data):
    _, paths = data
    options = ['--dpi', '50', '--trackstats_file', paths['mcs_stats'], '--extent', '100 130 -10 10']
    files = run(cartesian_cli, 'mcs', options)
    assert len(files) == S.NFRAMES
    files = run(cartesian_cli, 'mcs', options + ['--product', 'field', '--field_varname', 'tb', '--levels', '200', '300', '5',
                                                 '--cbticks', '200', '300', '20', '--cblabel', 'Tb (K)'])
    assert len(files) == S.NFRAMES
    # the track style options apply to the x/y plots too
    base = load(run(cartesian_cli, 'mcs', options, period=ONE_FRAME)[0])
    halo = load(run(cartesian_cli, 'mcs', options + ['--track_halo_color', 'yellow', '--track_color', 'red'], period=ONE_FRAME)[0])
    assert mean_difference(base, halo) > 0


#-----------------------------------------------------------------------
def test_parallel_run_makes_the_same_pictures_as_serial(run, data, monkeypatch, tmp_path):
    """run_parallel=1: the Dask workers get the tracks once (scatter) and make the same PNG files as the serial loop."""
    configs, _ = data
    argv = SMALL + ['--product', 'tb']
    serial = run(mcs_cli, 'mcs', argv, outname='serial')
    out = tmp_path / 'parallel'
    mcs_cli.main(['-s', S.START, '-e', S.END, '-c', 'unused.yml', '--output', str(out), '-p', '1', '--workers', '2'] + argv)
    parallel = sorted(glob.glob(f'{out}/*.png'))
    assert [os.path.basename(f) for f in parallel] == [os.path.basename(f) for f in serial]
    for a, b in zip(serial, parallel):
        assert np.array_equal(load(a), load(b)), os.path.basename(a)


def test_old_command_through_its_stub_defaults(run, capsys):
    """The stub of plot_subset_tb_mcs_tracks_demo.py: the defaults of the old script (Tb only, frames without tracks skipped)."""
    stub = run(mcs_cli, 'mcs', ['--draw_land', '0', '--dpi', '50'], legacy_name='plot_subset_tb_mcs_tracks_demo')
    assert 'has moved to visualization/plot_mcs_tracks.py' in capsys.readouterr().err
    assert len(stub) == 3
    # the same as the new command with the options that the old script had built in
    new = run(mcs_cli, 'mcs', ['--draw_land', '0', '--dpi', '50', '--product', 'tb', '--fontsize', '13', '--figsize', '10', '10',
                               '--skip_empty', '1', '--track_smooth', '0', '--map_aspect', 'auto'])
    assert [os.path.basename(f) for f in new] == [os.path.basename(f) for f in stub]
    for a, b in zip(stub, new):
        assert np.array_equal(load(a), load(b))


#-----------------------------------------------------------------------
@pytest.mark.parametrize("module, key, family, argv, per_frame", [
    (mcs_cli, 'mcs', mcs_map, ['--product', 'tbpf'], 2),                      # 2 panels: the same style on both maps
    (mcs_cli, 'mcs', mcs_map, ['--product', 'tbze', '--panels', '1'], 1),
    (mcs_cli, 'mcs', mcs_map, ['--product', 'tb'], 1),
    (cell_cli, 'cell', cell_map, [], 1),
    (generic_cli, 'mcs', generic_map, ['--field_varname', 'tb', '--levels', '200', '300', '5', '--cbticks', '200', '300', '20'], 1),
], ids=['mcs_2panel', 'mcs_1panel', 'mcs_tb', 'cell', 'generic'])
def test_the_map_feature_options_reach_every_map(run, monkeypatch, module, key, family, argv, per_frame):
    """--draw_* and the style options of the map features arrive at every map of a frame (both panels of a 2-panel figure)."""
    recorded = []
    monkeypatch.setattr(family, 'add_map_features', lambda ax, draw, styles, resolution: recorded.append((dict(draw), styles, resolution)))
    options = ['--dpi', '50', '--draw_land', '1', '--draw_border', '1', '--draw_state', '0', '--map_resolution', '110m',
               '--land_color', 'red', '--land_linewidth', '3', '--border_alpha', '0.5', '--river_color', 'blue'] + argv
    run(module, key, options, period=ONE_FRAME)
    assert len(recorded) == per_frame
    for draw, styles, resolution in recorded:
        assert (draw['land'], draw['border'], draw['state']) == (True, True, False) and resolution == '110m'
        assert styles['land'] == dict(color='red', linewidth=3.0, alpha=1.0)
        default = 'k' if module is mcs_cli else 'gray'                      # the MCS maps are black, the other maps gray
        assert styles['border'] == dict(color=default, linewidth=0.8, alpha=0.5)
        assert styles['state'] == dict(color=default, linewidth=0.8, alpha=1.0)
        assert styles['river'] == dict(color='blue', linewidth=0.8, alpha=1.0)
