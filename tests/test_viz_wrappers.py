"""
Tests of the animation wrappers: visualization/make_mcs_animation.py and visualization/examples/make_mcs_animation_*.py (the parameters
are variables at the top of the file, mcs_animation.main(globals()) turns them into the plot options, the mp4 and the GIF) and
visualization/make_cell_animation.py. The parameter block becomes the arguments of the plot command (options_to_argv), which the real command
line parses; --set NAME=VALUE changes a parameter. The plot function and ffmpeg are replaced, so nothing is plotted or encoded.
Skipped where matplotlib / cartopy are not installed (the CI environment has neither).
"""
import importlib.util
import os
import sys

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("cartopy")
pytest.importorskip("yaml")

from pyflextrkr.ft_utilities import load_config
from pyflextrkr.visualization import mcs_cli, cell_cli, mcs_animation

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = ['make_mcs_animation_wrf_tbze', 'make_mcs_animation_imerg_1panel', 'make_mcs_animation_himawari_tb']


def load_wrapper(name):
    """The wrapper script as a module (its parameter block runs, main() does not)."""
    folder = 'visualization/examples' if name in EXAMPLES else 'visualization'
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, folder, f'{name}.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_mcs_wrapper(name, monkeypatch, tmp_path, overrides=()):
    """mcs_animation.main with the parameters of a wrapper and the plot function replaced: returns the argument list of the plot command."""
    module = load_wrapper(name)
    recorded = []
    monkeypatch.setattr(mcs_animation, 'plot_mcs_tracks', lambda argv: recorded.append(list(argv)))
    argv = ['--set', f'fig_root={tmp_path}/', '--set', 'run_ffmpeg=false']
    for item in overrides:
        argv += ['--set', item]
    mcs_animation.main(vars(module), argv)
    assert len(recorded) == 1
    return recorded[0]


def run_cell_wrapper(module, monkeypatch, tmp_path, overrides=()):
    """main() of the cell wrapper with the plot function replaced."""
    recorded = []
    monkeypatch.setattr(module, 'plot_cell_tracks', lambda argv: recorded.append(list(argv)))
    argv = ['wrapper', '--set', f'figdir={tmp_path}/frames/', '--set', 'run_ffmpeg=false']
    for item in overrides:
        argv += ['--set', item]
    monkeypatch.setattr(sys, 'argv', argv)
    module.main()
    assert len(recorded) == 1
    return recorded[0]


#-----------------------------------------------------------------------
def test_mcs_wrapper_gives_the_global_robinson_map(monkeypatch, tmp_path):
    args = mcs_cli.parse_cmd_args(run_mcs_wrapper('make_mcs_animation', monkeypatch, tmp_path))
    # the defaults of the wrapper: the global map of the V3 tracking, 1-2 January 2025
    assert args['start_datetime'].startswith('2025-01-01') and args['end_datetime'].startswith('2025-01-03')
    assert args['product'] == 'tbpf' and args['panels'] == 2 and args['layout'] == 'top_bottom'
    assert args['projection'] == 'Robinson' and args['central_longitude'] == 225 and args['lat_range'] == [-60., 60.]
    assert args['extent'] is None and args['subset'] == 0 and args['periodic_perimeter'] == 'x'
    assert args['fontsize'] == 14 and args['show_track_numbers'] == [1, 0]
    # a light look: the clouds matter more than the tracks (no halo, no PF circles, small faint numbers, small symbol)
    assert args['track_halo_color'] is None and args['show_pfdiam'] == 0
    assert (args['track_text_size'], args['track_text_alpha'], args['track_marker_size']) == (7, 0.6, 4)
    # the track statistics and the pixel-level files come from the config file of the repository
    assert args['trackstats_file'] is None and args['pixeltracking_path'] is None
    assert os.path.basename(args['config_file']) == 'config_imerg_global_mcs_v3_2025.yml' and os.path.isfile(args['config_file'])


def test_the_config_file_of_the_global_wrapper_points_to_the_outputs():
    import yaml
    config = yaml.safe_load(open(os.path.join(REPO, 'config', 'config_imerg_global_mcs_v3_2025.yml')))
    root = config['root_path'].rstrip('/')
    assert root.endswith('MCS_Global/V3/Feng21')
    assert (config['stats_path_name'], config['pixel_path_name']) == ('stats_f21', 'mcstracking_f21')
    assert (config['startdate'], config['enddate']) == ('20250101.0000', '20251001.0000')
    assert config['pixeltracking_filebase'] == 'mcstrack_' and config['mcsfinal_filebase'] == 'mcs_tracks_final_'


def test_mcs_wrapper_set_overrides(monkeypatch, tmp_path, capsys):
    name = 'make_mcs_animation'
    argv = run_mcs_wrapper(name, monkeypatch, tmp_path,
                           ['start_date=2025-01-05T00', 'lat_range=[-30,30]', 'track_halo_color=white', 'show_track_numbers=[true,true]',
                            'fontsize=16', 'layout=left_right', 'show_pfdiam=true', 'cbar_thickness=0.03'])
    args = mcs_cli.parse_cmd_args(argv)
    assert args['start_datetime'] == '2025-01-05T00' and args['lat_range'] == [-30., 30.] and args['track_halo_color'] == 'white'
    assert args['show_track_numbers'] == [1, 1] and args['fontsize'] == 16 and args['layout'] == 'left_right'
    assert args['show_pfdiam'] == 1 and args['cbar_thickness'] == 0.03
    # a date with minutes and seconds is read by yaml as a date: it is made a string again
    args = mcs_cli.parse_cmd_args(run_mcs_wrapper(name, monkeypatch, tmp_path, ['start_date=2025-01-05T06:30:00']))
    assert args['start_datetime'].startswith('2025-01-05T06:30')
    args = mcs_cli.parse_cmd_args(run_mcs_wrapper(name, monkeypatch, tmp_path, ['start_date=2025-01-05T06:30']))
    assert args['start_datetime'].startswith('2025-01-05T06:30')
    # one panel: the second value of show_track_numbers is not given (the command line takes one value per panel), no layout
    args = mcs_cli.parse_cmd_args(run_mcs_wrapper(name, monkeypatch, tmp_path, ['data_type=tb']))
    assert args['product'] == 'tb' and args['panels'] == 1 and args['show_track_numbers'] == [1] and args['layout'] == 'top_bottom'
    args = mcs_cli.parse_cmd_args(run_mcs_wrapper(name, monkeypatch, tmp_path, ['plot_type=1panel']))
    assert args['panels'] == 1 and args['show_track_numbers'] == [1]
    # the product: reflectivity
    args = mcs_cli.parse_cmd_args(run_mcs_wrapper(name, monkeypatch, tmp_path, ['data_type=tbze']))
    assert args['product'] == 'tbze' and args['var2_varname'] == 'reflectivity_lowlevel'
    with pytest.raises(SystemExit):
        run_mcs_wrapper(name, monkeypatch, tmp_path, ['not_a_parameter=1'])
    with pytest.raises(ValueError, match='plot_type'):
        run_mcs_wrapper(name, monkeypatch, tmp_path, ['plot_type=3panel'])


def test_the_old_orientation_parameter_is_the_layout_with_a_deprecation_warning(monkeypatch, tmp_path, capsys):
    capsys.readouterr()
    args = mcs_cli.parse_cmd_args(run_mcs_wrapper('make_mcs_animation', monkeypatch, tmp_path, ['orientation=vertical']))
    assert args['layout'] == 'left_right'
    assert 'DeprecationWarning' in capsys.readouterr().err
    # a wrapper that still has the variable `orientation` (an old copy of the script)
    module = load_wrapper('make_mcs_animation')
    values = dict(vars(module))
    values['orientation'] = 'horizontal'
    values.pop('layout')
    recorded = []
    monkeypatch.setattr(mcs_animation, 'plot_mcs_tracks', lambda a: recorded.append(list(a)))
    mcs_animation.main(values, ['--set', f'fig_root={tmp_path}/', '--set', 'run_ffmpeg=false'])
    assert mcs_cli.parse_cmd_args(recorded[0])['layout'] == 'top_bottom'
    assert 'DeprecationWarning' in capsys.readouterr().err


def test_mcs_wrappers_have_the_video_and_gif_parameters():
    for name in ['make_mcs_animation'] + EXAMPLES:
        module = load_wrapper(name)
        names = [k for k in vars(module) if not k.startswith('_') and not callable(vars(module)[k]) and k not in ('os',)]
        assert not [k for k in names if k not in mcs_animation.DEFAULTS], name         # every variable of a wrapper is a parameter
    assert mcs_animation.DEFAULTS['video_preset'] is None        # no -preset option: the x264 default (docs/visualization/animations.md)
    assert mcs_animation.DEFAULTS['gif_width'] is None and mcs_animation.DEFAULTS['gif_dither'] == 'none'
    module = load_wrapper('make_cell_animation')
    assert module.video_preset is None and 'video_preset' in module._PARAMETERS


def test_example_wrappers_give_their_maps(monkeypatch, tmp_path):
    wrf = mcs_cli.parse_cmd_args(run_mcs_wrapper('make_mcs_animation_wrf_tbze', monkeypatch, tmp_path))
    assert (wrf['product'], wrf['panels'], wrf['layout']) == ('tbze', 2, 'left_right')
    assert wrf['projection'] == 'LambertConformal' and wrf['central_longitude'] == -97 and wrf['standard_parallels'] == [31., 35.]
    assert wrf['extent'] is None and wrf['subset'] == 0 and wrf['draw_state'] == 1 and wrf['var2_varname'] == 'reflectivity_comp'
    imerg = mcs_cli.parse_cmd_args(run_mcs_wrapper('make_mcs_animation_imerg_1panel', monkeypatch, tmp_path))
    assert (imerg['product'], imerg['panels'], imerg['figsize_x'], imerg['dpi']) == ('tbpf', 1, 8, 100)
    assert (imerg['cbar_thickness'], imerg['cbar_spacing']) == (0.02, 0.1) and imerg['extent'] is None and imerg['figsize'] is None
    himawari = mcs_cli.parse_cmd_args(run_mcs_wrapper('make_mcs_animation_himawari_tb', monkeypatch, tmp_path))
    assert (himawari['product'], himawari['panels'], himawari['figsize'], himawari['dpi']) == ('tb', 1, [8., 8.], 100)
    for name in EXAMPLES:                                         # the config file that the demo script writes
        module = load_wrapper(name)
        assert os.path.normpath(module.config_file) == os.path.join(REPO, 'config', 'config_mcs_demo.yml')
        assert module.gif_width and module.start_date < module.end_date


def test_mp4_and_gif_steps(monkeypatch, tmp_path):
    """make_mp4 gets the PNG files of the period and the settings, make_gif the mp4 and the GIF settings (ffmpeg is replaced)."""
    module = load_wrapper('make_mcs_animation_imerg_1panel')
    calls = {}
    monkeypatch.setattr(mcs_animation, 'plot_mcs_tracks', lambda argv: None)
    monkeypatch.setattr(mcs_animation, 'frames_in_range', lambda figdir, base, start, end: calls.setdefault('frames', [f'{figdir}{base}1.png']))

    def fake_mp4(files, out, **kwargs):
        calls['mp4'] = (files, out, kwargs)
        return 0, ['ffmpeg']

    def fake_gif(source, out, **kwargs):
        calls['gif'] = (source, out, kwargs)
        open(out, 'wb').write(b'GIF')
        return 0, ['ffmpeg']
    monkeypatch.setattr(mcs_animation, 'make_mp4', fake_mp4)
    monkeypatch.setattr(mcs_animation, 'make_gif', fake_gif)
    mcs_animation.main(vars(module), ['--set', f'fig_root={tmp_path}/', '--set', 'run_plotting=false'])
    files, mp4, kwargs = calls['mp4']
    assert mp4.endswith('animations_1panel_tbpf/mcs_tbpf_2019-01-25_2019-01-27.mp4') and files[0].endswith('mcs_tbpf_1.png')
    assert (kwargs['input_framerate'], kwargs['output_framerate'], kwargs['crf'], kwargs['width'], kwargs['preset']) == (2, 10, 20, 800, None)
    source, gif, kwargs = calls['gif']
    assert source == mp4 and gif == mp4[:-4] + '.gif'
    assert (kwargs['fps'], kwargs['width'], kwargs['start'], kwargs['duration'], kwargs['colors'], kwargs['dither']) == (2, 480, 0.0, 12, 256, 'none')
    # no gif_width: no GIF
    calls.clear()
    mcs_animation.main(vars(module), ['--set', f'fig_root={tmp_path}/', '--set', 'run_plotting=false', '--set', 'gif_width=null'])
    assert 'mp4' in calls and 'gif' not in calls
    # gif_fps replaces the frame rate of the GIF
    calls.clear()
    mcs_animation.main(vars(module), ['--set', f'fig_root={tmp_path}/', '--set', 'run_plotting=false', '--set', 'gif_fps=5'])
    assert calls['gif'][2]['fps'] == 5


def test_cell_wrapper_options_parse(monkeypatch, tmp_path):
    module = load_wrapper('make_cell_animation')
    args = cell_cli.parse_cmd_args(run_cell_wrapper(module, monkeypatch, tmp_path))
    assert args['product'] == 'refl' and args['radar_lat'] == module.radar_lat and args['extent'] is not None
    assert args['projection'] == 'PlateCarree'
    args = cell_cli.parse_cmd_args(run_cell_wrapper(module, monkeypatch, tmp_path,
                                                    ['projection=LambertConformal', 'central_longitude=-90', 'track_halo_color=white']))
    assert args['projection'] == 'LambertConformal' and args['central_longitude'] == -90 and args['track_halo_color'] == 'white'


#-----------------------------------------------------------------------
def test_reading_a_config_makes_no_folders_with_make_dirs_false(tmp_path):
    config_file = tmp_path / 'config.yml'
    config_file.write_text(f"""startdate: '20250101.0000'
enddate: '20250102.0000'
root_path: '{tmp_path}/run/'
tracking_path_name: 'tracking'
stats_path_name: 'stats'
pixel_path_name: 'mcstracking'
""")
    config = load_config(str(config_file), make_dirs=False)
    assert not (tmp_path / 'run').exists()
    assert config['stats_outpath'] == f'{tmp_path}/run//stats/'
    assert config['pixeltracking_outpath'] == f'{tmp_path}/run//mcstracking/20250101.0000_20250102.0000/'
    load_config(str(config_file))                                  # the default: the tracking pipeline needs the folders
    assert (tmp_path / 'run' / 'tracking').is_dir() and (tmp_path / 'run' / 'stats').is_dir()
    assert (tmp_path / 'run' / 'mcstracking' / '20250101.0000_20250102.0000').is_dir()
