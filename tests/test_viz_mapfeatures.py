"""
Tests of the background map features (coastlines, country borders, state borders, rivers): the same default style for every
feature (gray, 0.8 points, opaque), a style option for each feature (--land_color, --border_linewidth, --state_alpha, ...), the drawing
order land > border > state > river above everything that is plotted, and the old look of the Analysis/plot_subset_*.py stubs.
Skipped where matplotlib / cartopy are not installed (the CI environment has both, cartopy through easygems, so they run there). Nothing is downloaded: the axes are replaced
by a recorder.
"""
import ast
import glob
import os

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("cartopy")

from pyflextrkr.visualization import mapcore, legacy, mcs_cli, cell_cli, generic_cli, cell_map, mcs_map, generic_map

REQUIRED = ['-s', '2025-06-01T00', '-e', '2025-06-01T06', '-c', 'config.yml']
FEATURES = ('land', 'border', 'state', 'river')


class RecordingAxes:
    """An axes that only records add_feature calls."""
    def __init__(self):
        self.calls = []

    def add_feature(self, feature, **kwargs):
        self.calls.append((feature, kwargs))


#-----------------------------------------------------------------------
def test_every_feature_has_the_same_default_style():
    styles = mapcore.map_feature_styles()
    assert set(styles) == set(FEATURES) == set(mapcore.MAP_FEATURES)
    for name in FEATURES:
        assert styles[name] == dict(color='gray', linewidth=0.8, alpha=1.0), name
    assert mapcore.map_feature_styles({}) == mapcore.map_feature_styles(None) == styles


def test_an_option_changes_only_its_own_feature_and_its_own_value():
    args = {'land_linewidth': 2.0, 'state_color': 'red', 'river_alpha': 0.5, 'border_color': None, 'draw_land': 1}
    styles = mapcore.map_feature_styles(args)
    assert styles['land'] == dict(color='gray', linewidth=2.0, alpha=1.0)
    assert styles['border'] == dict(color='gray', linewidth=0.8, alpha=1.0)           # None: the default
    assert styles['state'] == dict(color='red', linewidth=0.8, alpha=1.0)
    assert styles['river'] == dict(color='gray', linewidth=0.8, alpha=0.5)


def test_the_mcs_maps_default_to_black_lines_the_other_maps_to_gray():
    """Gray lines are hard to see on the Tb colormaps (contrast 1.4-3 : 1, black 7-16 : 1), so the MCS maps are black; width and alpha are the same."""
    assert mapcore.MCS_MAP_FEATURE_COLOR == 'k'
    black = mapcore.map_feature_styles(None, color=mapcore.MCS_MAP_FEATURE_COLOR)
    for name in FEATURES:
        assert black[name] == dict(color='k', linewidth=0.8, alpha=1.0), name
    # an option still wins over the color of the family
    styles = mapcore.map_feature_styles({'land_color': 'red', 'river_linewidth': 2}, color='k')
    assert styles['land']['color'] == 'red' and styles['border']['color'] == 'k' and styles['river'] == dict(color='k', linewidth=2.0, alpha=1.0)
    assert mapcore.map_feature_styles(None, color=None) == mapcore.map_feature_styles()          # no family color: gray


# the style of the new commands (no legacy name) of each family
FAMILY_DEFAULT_COLOR = {mcs_cli: 'k', cell_cli: 'gray', generic_cli: 'gray'}


@pytest.mark.parametrize("module", [mcs_cli, cell_cli, generic_cli], ids=lambda m: m.__name__.split('.')[-1])
def test_the_three_map_families_have_the_twelve_options(module):
    default = FAMILY_DEFAULT_COLOR[module]
    color = None if default == 'gray' else default
    args = module.parse_cmd_args(REQUIRED)
    for name in FEATURES:
        for key in ('color', 'linewidth', 'alpha'):
            assert args[f'{name}_{key}'] is None, f'--{name}_{key} is not an option or has a default'
    assert mapcore.map_feature_styles(args, color=color) == mapcore.map_feature_styles(None, color=color)
    args = module.parse_cmd_args(REQUIRED + ['--land_color', 'red', '--land_linewidth', '2', '--border_alpha', '0.4',
                                            '--state_linewidth', '0.3', '--river_color', 'blue'])
    styles = mapcore.map_feature_styles(args, color=color)
    assert styles['land'] == dict(color='red', linewidth=2.0, alpha=1.0)
    assert styles['border'] == dict(color=default, linewidth=0.8, alpha=0.4)
    assert styles['state'] == dict(color=default, linewidth=0.3, alpha=1.0)
    assert styles['river'] == dict(color='blue', linewidth=0.8, alpha=1.0)


def test_add_map_features_draws_the_asked_features_with_their_style_and_zorder(monkeypatch):
    made = []
    monkeypatch.setattr(mapcore.cfeature, 'NaturalEarthFeature', lambda category, name, resolution: made.append((category, name, resolution)) or (category, name))
    ax = RecordingAxes()
    styles = mapcore.map_feature_styles({'state_color': 'red', 'state_alpha': 0.5})
    mapcore.add_map_features(ax, dict(land=True, border=False, state=True), styles, '50m')
    assert made == [('physical', 'land', '50m'), ('cultural', 'admin_1_states_provinces_lakes', '50m')]      # border and river are not drawn
    (_, land), (_, state) = ax.calls
    assert land == dict(facecolor='none', edgecolor='gray', linewidth=0.8, alpha=1.0, zorder=mapcore.MAP_FEATURE_ZORDER['land'])
    assert state == dict(facecolor='none', edgecolor='red', linewidth=0.8, alpha=0.5, zorder=mapcore.MAP_FEATURE_ZORDER['state'])
    # no styles: the defaults
    ax = RecordingAxes()
    mapcore.add_map_features(ax, dict(river=True), None, '110m')
    assert ax.calls[0][1]['edgecolor'] == 'gray' and ax.calls[0][1]['linewidth'] == 0.8
    # nothing asked, nothing drawn
    ax = RecordingAxes()
    mapcore.add_map_features(ax, {}, None, '110m')
    assert ax.calls == []


def test_drawing_order_land_border_state_river_above_everything_that_is_plotted():
    """
    land on top of border on top of state on top of river, and all above the shading, outlines, track paths, symbols, numbers, gridlines
    and range circles: the zorder values of the plot functions (literals in zorder=..., and the defaults of the *_zorder
    entries of plot_info) are all below the river; only the markers of cities and radars (>= 8) are above the features.
    """
    z = mapcore.MAP_FEATURE_ZORDER
    assert z['land'] > z['border'] > z['state'] > z['river']
    folder = os.path.dirname(mapcore.__file__)
    values = []
    for name in ('mcs_map', 'cell_map', 'generic_map'):
        path = os.path.join(folder, f'{name}.py')
        for node in ast.walk(ast.parse(open(path, encoding='utf-8').read())):
            if isinstance(node, ast.keyword) and node.arg == 'zorder' and isinstance(node.value, ast.Constant):
                values.append((name, node.value.lineno, node.value.value))
            # plot_info.get('grid_zorder', None), plot_info.get('var2_zorder', 2), ...
            if isinstance(node, ast.Call) and getattr(node.func, 'attr', None) == 'get' and node.args and \
                    isinstance(node.args[0], ast.Constant) and str(node.args[0].value).endswith('zorder') and len(node.args) > 1 \
                    and isinstance(node.args[1], ast.Constant) and node.args[1].value is not None:
                values.append((name, node.lineno, node.args[1].value))
    assert values
    for name, line, value in values:
        assert value < z['river'] or value >= 8, f"{name}.py line {line}: zorder {value} would draw over / under the map features"
    # the radar-site and city markers of the cell maps are the ones above the features
    assert any(n == 'cell_map' and v >= 8 for n, _, v in values)
    # the presets of the MCS maps do not set a zorder above the features either
    presets = open(os.path.join(folder, 'mcs_cli.py'), encoding='utf-8').read()
    assert "'map_zorder'" not in presets and 'map_edgecolor' not in presets and 'land_linewidth' not in presets


def test_no_plot_function_keeps_its_own_style_of_the_features():
    """The style comes from the options only: no plot_info / map_info keys map_edgecolor, land_linewidth, land_linewidth2, map_zorder, river_color in the maps and command lines."""
    folder = os.path.dirname(mapcore.__file__)
    for path in sorted(glob.glob(os.path.join(folder, '*_map.py')) + glob.glob(os.path.join(folder, '*_cli.py'))):
        text = open(path, encoding='utf-8').read()
        for old in ('map_edgecolor', 'land_linewidth2', "'map_zorder'", "map_info.get('river_color'", "plot_info['land_linewidth']", "plot_info.get('land_linewidth'"):
            assert old not in text, f"{os.path.basename(path)} still has {old}"


#-----------------------------------------------------------------------
LEGACY_STYLE = {                       # old script -> (color, line width) of land, border, state, river
    'plot_subset_tbpf_mcs_tracks_demo': (('k', 2), ('k', 0.8), ('k', 0.8), ('gray', 0.5)),
    'plot_subset_tbpf_mcs_tracks_1panel_demo': (('k', 2), ('k', 0.8), ('k', 0.8), ('gray', 0.5)),
    'plot_subset_tbze_mcs_tracks_demo': (('k', 2), ('k', 0.8), ('k', 0.8), ('gray', 0.5)),
    'plot_subset_tbze_mcs_tracks_1panel_demo': (('k', 2), ('k', 0.8), ('k', 0.8), ('gray', 0.5)),
    'plot_subset_tb_mcs_tracks_demo': (('dimgray', 1), ('dimgray', 0.8), ('dimgray', 0.8), ('gray', 0.5)),
    'plot_subset_cell_tracks_demo': (('k', 1), ('k', 0.8), ('gray', 0.5), ('gray', 0.5)),
    'plot_subset_cell_tracks_ETH_demo': (('k', 1), ('k', 0.8), ('gray', 0.5), ('gray', 0.5)),
    'plot_subset_generic_tracks_demo': (('gray', 1), ('gray', 0.8), ('gray', 0.8), ('gray', 0.5)),
    'plot_subset_psi_tracks_demo': (('gray', 1), ('gray', 0.8), ('gray', 0.8), ('gray', 0.5)),
}
FAMILY = {'mcs': mcs_cli, 'cell': cell_cli, 'generic': generic_cli}


@pytest.mark.parametrize("name", sorted(LEGACY_STYLE))
def test_the_old_scripts_keep_the_style_of_their_map_features(name):
    module = FAMILY['mcs' if '_mcs_' in name else 'cell' if '_cell_' in name else 'generic']
    color_of_family = mapcore.MCS_MAP_FEATURE_COLOR if module is mcs_cli else None
    styles = mapcore.map_feature_styles(module.parse_cmd_args(REQUIRED, legacy_name=name), color=color_of_family)
    for feature, (color, linewidth) in zip(FEATURES, LEGACY_STYLE[name]):
        assert (styles[feature]['color'], styles[feature]['linewidth'], styles[feature]['alpha']) == (color, linewidth, 1.0), (name, feature)
    # an option on the command line still replaces the old style
    args = module.parse_cmd_args(REQUIRED + ['--land_linewidth', '0.8', '--land_color', 'gray'], legacy_name=name)
    assert mapcore.map_feature_styles(args, color=color_of_family)['land'] == dict(color='gray', linewidth=0.8, alpha=1.0)
    # the new commands (no legacy name) have the new default
    assert mapcore.map_feature_styles(module.parse_cmd_args(REQUIRED), color=color_of_family)['land']['color'] == ('k' if module is mcs_cli else 'gray')
