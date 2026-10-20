"""
Tests of the command lines of the plot families (pyflextrkr/visualization/*_cli.py): the arguments, the extent given as four
numbers or as one quoted string, the conversion of an options dictionary for the wrapper scripts, the defaults that keep
the old Analysis/plot_subset_*.py commands working, and the layout rules that keep "Go to Definition" working in the
scripts (static imports, no sys.path edits, no subprocess).
Skipped where matplotlib / cartopy are not installed (the CI environment has both, cartopy through easygems, so they run there).
"""
import argparse
import ast
import importlib
import os

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("cartopy")

from pyflextrkr.visualization import cli_common, legacy
from pyflextrkr.visualization import mcs_cli, cell_cli, generic_cli, cartesian_cli

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIRED = ['-s', '2025-06-01T00', '-e', '2025-06-01T06', '-c', 'config.yml']

# old script -> command line module that has to serve it
FAMILY_OF = {
    'plot_subset_tbpf_mcs_tracks_demo': mcs_cli,
    'plot_subset_tbpf_mcs_tracks_1panel_demo': mcs_cli,
    'plot_subset_tbze_mcs_tracks_demo': mcs_cli,
    'plot_subset_tbze_mcs_tracks_1panel_demo': mcs_cli,
    'plot_subset_tb_mcs_tracks_demo': mcs_cli,
    'plot_subset_cell_tracks_demo': cell_cli,
    'plot_subset_cell_tracks_ETH_demo': cell_cli,
    'plot_subset_generic_tracks_demo': generic_cli,
    'plot_subset_psi_tracks_demo': generic_cli,
    'plot_subset_tbpf_demo_pbc': cartesian_cli,
    'plot_subset_tbpf_tracks_pinacles_pbc': cartesian_cli,
    'plot_subset_coldpool_tracks_nomap': cartesian_cli,
}


#-----------------------------------------------------------------------
def test_parse_extent_four_numbers_or_one_string():
    four = cli_common.parse_extent(['140.0', '-140.0', '-20.0', '20.0'])
    one = cli_common.parse_extent(['140.0 -140.0 -20.0 20.0'])
    commas = cli_common.parse_extent(['140.0,-140.0,-20.0,20.0'])
    assert four == one == commas == [140.0, -140.0, -20.0, 20.0]
    assert cli_common.parse_extent(None) is None
    with pytest.raises(argparse.ArgumentTypeError):
        cli_common.parse_extent(['1', '2', '3'])


def test_extent_option_in_both_forms_reaches_the_arguments():
    for extent in (['--extent', '0', '1500', '0', '1500'], ['--extent', '0. 1500 0. 1500']):
        args = mcs_cli.parse_cmd_args(REQUIRED + extent)
        assert args['extent'] == [0., 1500., 0., 1500.]
    assert mcs_cli.parse_cmd_args(REQUIRED)['extent'] is None


def test_options_to_argv():
    argv = cli_common.options_to_argv({
        'start': '2025-06-01T00', 'extent': [140., 220., -20., 20.], 'draw_border': True, 'draw_state': False,
        'track_color': None, 'central_longitude': 225, 'lat_range': [-60, 60], 'figbasename': 'mcs_',
    })
    assert argv == ['--start', '2025-06-01T00', '--extent', '140.0', '220.0', '-20.0', '20.0', '--draw_border', '1',
                    '--draw_state', '0', '--central_longitude', '225', '--lat_range', '-60', '60', '--figbasename', 'mcs_']
    # a value that starts with '-' must not be read as an option
    assert cli_common.options_to_argv({'figbasename': '-x'}) == ['--figbasename=-x']
    # and the arguments parse
    args = mcs_cli.parse_cmd_args(cli_common.options_to_argv(
        {'start': 'a', 'end': 'b', 'config': 'c', 'extent': [1, 2, 3, 4], 'lat_range': [-60, 60], 'draw_land': True}))
    assert args['extent'] == [1., 2., 3., 4.] and args['lat_range'] == [-60., 60.] and args['draw_land'] == 1


def test_new_options_default_to_the_original_look():
    args = mcs_cli.parse_cmd_args(REQUIRED)
    assert args['product'] == 'tbpf' and args['panels'] == 2 and args['layout'] == 'top_bottom'
    assert args['projection'] == 'PlateCarree' and args['central_longitude'] is None and args['lat_range'] is None
    assert args['track_halo_color'] is None and args['track_color'] is None and args['track_marker'] is None
    assert args['periodic_perimeter'] == 'none' and args['track_smooth'] == 1 and args['skip_empty'] is None
    assert args['workers'] == 4 and args['map_aspect'] == 'equal' and args['show_track_numbers'] is None
    # the generic and x/y families take the number of workers from the config, as their old scripts did
    assert generic_cli.parse_cmd_args(REQUIRED)['workers'] is None
    assert cartesian_cli.parse_cmd_args(REQUIRED)['workers'] is None
    assert cell_cli.parse_cmd_args(REQUIRED)['workers'] == 4
    # the x/y family has no map options
    with pytest.raises(SystemExit):
        cartesian_cli.parse_cmd_args(REQUIRED + ['--projection', 'Robinson'])


def test_show_track_numbers_one_value_per_panel():
    assert mcs_cli.parse_cmd_args(REQUIRED + ['--show_track_numbers', '0'])['show_track_numbers'] == [0]
    assert mcs_cli.parse_cmd_args(REQUIRED + ['--show_track_numbers', '1', '0'])['show_track_numbers'] == [1, 0]
    info = cli_common.apply_track_style_args({}, {'show_track_numbers': [1, 0]}, npanels=2)
    assert info['show_tracknumbers'] is True and info['show_tracknumbers2'] is False
    info = cli_common.apply_track_style_args({}, {'show_track_numbers': [0]}, npanels=2)
    assert info['show_tracknumbers'] is False and 'show_tracknumbers2' not in info
    assert cli_common.apply_track_style_args({}, {'show_track_numbers': None}, npanels=2) == {}
    with pytest.raises(ValueError, match='one per panel'):
        cli_common.apply_track_style_args({}, {'show_track_numbers': [1, 0]}, npanels=1)
    # the wrappers give lists of booleans
    assert cli_common.options_to_argv({'show_track_numbers': [True, False]}) == ['--show_track_numbers', '1', '0']


def test_dash_names_of_the_old_options_still_parse():
    args = mcs_cli.parse_cmd_args(REQUIRED + ['--dbz_varname', 'reflectivity_lowlevel'])
    assert args['var2_varname'] == 'reflectivity_lowlevel'
    args = mcs_cli.parse_cmd_args(REQUIRED + ['-o', 'left_right', '-p', '1', '--workers', '8', '--figsize', '10', '8', '--subset', '1'])
    assert args['layout'] == 'left_right' and args['run_parallel'] == 1 and args['workers'] == 8
    assert args['figsize'] == [10., 8.] and args['subset'] == 1
    assert (args['start_datetime'], args['end_datetime'], args['config_file']) == ('2025-06-01T00', '2025-06-01T06', 'config.yml')


#-----------------------------------------------------------------------
def test_panel_layout_names_and_the_deprecated_orientation_names(capsys):
    """--layout top_bottom|left_right; -o and --orientation are old names of the same option, horizontal/vertical print a warning."""
    capsys.readouterr()
    for option in ('--layout', '-o', '--orientation'):
        assert mcs_cli.parse_cmd_args(REQUIRED + [option, 'left_right'])['layout'] == 'left_right'
        assert mcs_cli.parse_cmd_args(REQUIRED + [option, 'top_bottom'])['layout'] == 'top_bottom'
    assert capsys.readouterr().err == ''                          # the new names are quiet
    for option in ('--layout', '-o', '--orientation'):
        for old, new in (('horizontal', 'top_bottom'), ('vertical', 'left_right')):
            assert mcs_cli.parse_cmd_args(REQUIRED + [option, old])['layout'] == new
            err = capsys.readouterr().err
            assert 'DeprecationWarning' in err and f"'{old}'" in err and f"--layout {new}" in err
    assert mcs_cli.parse_cmd_args(REQUIRED)['layout'] == 'top_bottom' and capsys.readouterr().err == ''
    with pytest.raises(SystemExit):
        mcs_cli.parse_cmd_args(REQUIRED + ['--layout', 'diagonal'])
    # the wrappers give the option by name
    assert cli_common.options_to_argv({'layout': 'left_right'}) == ['--layout', 'left_right']


def test_colorbar_options_default_to_the_layout_of_the_old_scripts():
    args = mcs_cli.parse_cmd_args(REQUIRED)
    assert args['cbar_thickness'] is None and args['cbar_spacing'] is None
    args = mcs_cli.parse_cmd_args(REQUIRED + ['--cbar_thickness', '0.01', '--cbar_spacing', '0.1'])
    assert (args['cbar_thickness'], args['cbar_spacing']) == (0.01, 0.1)


#-----------------------------------------------------------------------
def test_every_old_script_has_defaults_and_a_family():
    assert set(legacy.LEGACY) == set(FAMILY_OF)


@pytest.mark.parametrize("name", sorted(FAMILY_OF))
def test_legacy_defaults_are_the_parsed_defaults(name):
    module = FAMILY_OF[name]
    args = module.parse_cmd_args(REQUIRED, legacy_name=name)
    options = module.parse_cmd_args(REQUIRED)
    for key, value in legacy.defaults(name).items():
        assert key in options, f"{name}: '{key}' is not an option of {module.__name__}"
        assert args[key] == value, f"{name}: {key}"
    # a command line option replaces the default of the old script
    if 'dpi' in legacy.defaults(name):
        assert module.parse_cmd_args(REQUIRED + ['--dpi', '77'], legacy_name=name)['dpi'] == 77


def test_legacy_commands_keep_the_old_shape_of_the_map():
    for name, module in FAMILY_OF.items():
        if module is cartesian_cli:
            continue
        assert module.parse_cmd_args(REQUIRED, legacy_name=name)['map_aspect'] == 'auto', name


def test_legacy_defaults_of_the_main_families():
    tbpf = mcs_cli.parse_cmd_args(REQUIRED, legacy_name='plot_subset_tbpf_mcs_tracks_1panel_demo')
    assert (tbpf['product'], tbpf['panels'], tbpf['skip_empty']) == ('tbpf', 1, 1)
    tbze = mcs_cli.parse_cmd_args(REQUIRED, legacy_name='plot_subset_tbze_mcs_tracks_demo')
    assert (tbze['product'], tbze['panels'], tbze['dpi'], tbze['draw_state']) == ('tbze', 2, 300, 1)
    tb = mcs_cli.parse_cmd_args(REQUIRED, legacy_name='plot_subset_tb_mcs_tracks_demo')
    assert (tb['product'], tb['panels'], tb['figsize'], tb['track_smooth'], tb['workers']) == ('tb', 1, [10., 10.], 0, None)
    eth = cell_cli.parse_cmd_args(REQUIRED, legacy_name='plot_subset_cell_tracks_ETH_demo')
    assert (eth['product'], eth['skip_empty']) == ('eth', 1)
    psi = generic_cli.parse_cmd_args(REQUIRED, legacy_name='plot_subset_psi_tracks_demo')
    assert psi['cmap'] == 'RdBu_r' and psi['levels'] == [-3e7, 3e7, 0.5e6] and psi['grid_linewidth'] == 0


def test_stubs_in_analysis_forward_to_their_own_name():
    """Each old script is a stub that calls main(legacy_name=<its own file name>) of the right family."""
    for name, module in FAMILY_OF.items():
        path = os.path.join(REPO, 'Analysis', f'{name}.py')
        assert os.path.isfile(path), path
        tree = ast.parse(open(path, encoding='utf-8').read())
        imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        assert [(n.module, [a.name for a in n.names]) for n in imports] == [(module.__name__, ['main'])], name
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'main']
        assert len(calls) == 1 and calls[0].keywords[0].arg == 'legacy_name', name
        assert calls[0].keywords[0].value.value == name, name


#-----------------------------------------------------------------------
def test_cell_options_of_plotmap_and_their_names():
    """terrain, outline colors per track, a ring around the outlines, tracks that live long enough, cities and other radars."""
    args = cell_cli.parse_cmd_args(REQUIRED)
    assert (args['perim_cmap'], args['perim_halo'], args['min_hours'], args['show_terrain'], args['grid_linestyle']) == (None, None, 0.0, 0, '--')
    assert args['city'] is None and args['radar_site'] is None and args['radar_site_radius'] == 150.
    args = cell_cli.parse_cmd_args(REQUIRED + ['--perim_cmap', 'glasbey_dark', '--perim_halo', 'white', '--min_hours', '1.5', '--show_terrain', '1',
                                               '--grid_linestyle', ':', '--city', 'A', '10', '20', '--city', 'Alta Gracia', '-64.4', '-31.6',
                                               '--radar_site', 'R', '12', '22', '--radar_site_radius', '40'])
    assert (args['perim_cmap'], args['perim_halo'], args['min_hours'], args['show_terrain'], args['grid_linestyle']) == ('glasbey_dark', 'white', 1.5, 1, ':')
    assert args['city'] == [['A', '10', '20'], ['Alta Gracia', '-64.4', '-31.6']] and args['radar_site'] == [['R', '12', '22']]
    assert args['radar_site_radius'] == 40.
    # the names that plotmap.py of conus_tracking uses
    args = cell_cli.parse_cmd_args(REQUIRED + ['--cell_perim_cmap', 'tab10', '--cell_halo', 'red', '--cell_min_hours', '2'])
    assert (args['perim_cmap'], args['perim_halo'], args['min_hours']) == ('tab10', 'red', 2.0)
    # the places become (name, lon, lat); a wrapper gives a list of lists, which repeats the option
    assert cell_cli.parse_sites([['A', '10', '20'], ['B', '-1.5', '2']], '--city') == [('A', 10., 20.), ('B', -1.5, 2.)]
    assert cell_cli.parse_sites(None, '--city') == []
    with pytest.raises(SystemExit):
        cell_cli.parse_sites([['A', 'east', '20']], '--city')
    argv = cli_common.options_to_argv({'city': [['A', 10, 20], ['B', -1.5, 2]], 'min_hours': 1.5, 'perim_cmap': None})
    assert argv == ['--city', 'A', '10', '20', '--city', 'B', '-1.5', '2', '--min_hours', '1.5']
    assert cell_cli.parse_cmd_args(REQUIRED + argv)['city'] == [['A', '10', '20'], ['B', '-1.5', '2']]


#-----------------------------------------------------------------------
ENTRY_SCRIPTS = ['plot_mcs_tracks.py', 'plot_cell_tracks.py', 'plot_generic_tracks.py', 'plot_cartesian_tracks.py',
                 'examples/make_mcs_animation.py', 'examples/make_mcs_animation_wrf_tbze.py', 'examples/make_mcs_animation_imerg_1panel.py',
                 'examples/make_mcs_animation_himawari_tb.py', 'examples/make_cell_animation_nexrad.py', 'pngs_to_animation.py']


def python_files_of_the_visualization_entry_points():
    files = [os.path.join(REPO, 'visualization', f) for f in ENTRY_SCRIPTS]
    files += [os.path.join(REPO, 'Analysis', f'{name}.py') for name in FAMILY_OF]
    return files


@pytest.mark.parametrize("path", python_files_of_the_visualization_entry_points(), ids=os.path.basename)
def test_entry_points_use_static_imports_only(path):
    """No sys.path edits, no subprocess, no importlib / getattr dispatch; every pyflextrkr import resolves to a name."""
    assert os.path.isfile(path), path
    tree = ast.parse(open(path, encoding='utf-8').read())
    forbidden_modules = {'subprocess', 'importlib', 'runpy'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not ({a.name.split('.')[0] for a in node.names} & forbidden_modules), path
        if isinstance(node, ast.ImportFrom):
            assert node.module.split('.')[0] not in forbidden_modules, path
            if node.module.startswith('pyflextrkr'):
                module = importlib.import_module(node.module)
                for alias in node.names:
                    assert hasattr(module, alias.name), f"{path}: {node.module} has no {alias.name}"
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
            # sys.path.insert / sys.path.append
            assert not (getattr(node.value.value, 'id', None) == 'sys' and node.value.attr == 'path'), f"{path}: sys.path edit"
        if isinstance(node, ast.Call) and getattr(node.func, 'id', None) in ('__import__', 'exec', 'eval'):
            pytest.fail(f"{path}: dynamic import / exec")


def test_library_modules_import_only_what_they_use_without_cycles():
    """Every module of the package imports on its own (no import cycle, no missing name)."""
    for name in ['mapcore', 'masks', 'colors', 'trackstats', 'trackdraw', 'pixelio', 'runner', 'animation', 'cli_common',
                 'legacy', 'mcs_map', 'mcs_cli', 'mcs_animation', 'cell_map', 'cell_cli', 'cell_animation', 'wrapper_common', 'generic_map',
                 'generic_cli', 'cartesian_map', 'cartesian_cli']:
        importlib.import_module(f'pyflextrkr.visualization.{name}')
