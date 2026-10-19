"""
Tests of the animation wrappers: visualization/make_mcs_animation.py and visualization/examples/make_mcs_animation_*.py (the parameters
are variables at the top of the file, mcs_animation.main(globals()) turns them into the plot options, the mp4 and the GIF) and
visualization/make_cell_animation.py. The parameter block becomes the arguments of the plot command (options_to_argv), which the real command
line parses; --set NAME=VALUE changes a parameter. The plot function and ffmpeg are replaced, so nothing is plotted or encoded.
Skipped where matplotlib / cartopy are not installed (the CI environment has neither).
"""
import importlib.util
import math
import os
import sys

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("cartopy")
pytest.importorskip("yaml")

from pyflextrkr.ft_utilities import load_config
from pyflextrkr.visualization import mcs_cli, cell_cli, mcs_animation, animation

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = ['make_mcs_animation_wrf_tbze', 'make_mcs_animation_imerg_1panel', 'make_mcs_animation_himawari_tb']
CELL_EXAMPLES = ['make_cell_animation_nexrad']


def load_wrapper(name):
    """The wrapper script as a module (its parameter block runs, main() does not)."""
    folder = 'visualization/examples' if name in EXAMPLES + CELL_EXAMPLES else 'visualization'
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
    # the video settings of the reference wrapper (conus_tracking): playback_fps (PNG files per second), a quality of CRF 24, 1920 pixels at most
    defaults = {k: mcs_animation.DEFAULTS[k] for k in ('playback_fps', 'frame_stride', 'video_fps', 'video_quality', 'output_width')}
    assert defaults == dict(playback_fps=2, frame_stride=1, video_fps=None, video_quality=24, output_width=1920)
    for name in ['make_mcs_animation'] + EXAMPLES:                # hourly files: 2 PNG files per second, the settings are in every wrapper
        module = load_wrapper(name)
        assert (module.playback_fps, module.frame_stride, module.video_fps, module.video_quality, module.output_width, module.video_preset) == \
            (2, 1, None, 24, 1920, None), name
        assert not [k for k in ('input_framerate', 'output_framerate') if hasattr(module, k)], name
    module = load_wrapper('make_cell_animation')
    assert module.video_preset is None and 'video_preset' in module._PARAMETERS
    assert (module.playback_fps, module.frame_stride, module.video_fps, module.video_quality, module.output_width) == (12, 1, None, 24, 1920)
    assert not [k for k in ('input_framerate', 'output_framerate') if k in module._PARAMETERS]


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
    frames = lambda figdir, base, start, end: calls.setdefault('frames', [f'{figdir}{base}{i}.png' for i in range(1, 5)])
    monkeypatch.setattr(mcs_animation, 'frames_in_range', frames)

    def fake_mp4(files, out, **kwargs):
        calls['mp4'] = (files, out, kwargs)
        open(out, 'wb').write(b'MP4')
        return 0, ['ffmpeg']

    def fake_gif(source, out, **kwargs):
        calls['gif'] = (source, out, kwargs)
        open(out, 'wb').write(b'GIF')
        return 0, ['ffmpeg']
    monkeypatch.setattr(animation, 'make_mp4', fake_mp4)           # mcs_animation.make_animation calls these two
    monkeypatch.setattr(animation, 'make_gif', fake_gif)
    set_args = ['--set', f'fig_root={tmp_path}/', '--set', 'run_plotting=false']
    mcs_animation.main(vars(module), set_args)
    files, mp4, kwargs = calls['mp4']
    assert mp4.endswith('animations_1panel_tbpf/mcs_tbpf_2019-01-25_2019-01-27.mp4') and len(files) == 4 and files[0].endswith('mcs_tbpf_1.png')
    assert (kwargs['playback_fps'], kwargs['video_fps'], kwargs['crf'], kwargs['width'], kwargs['preset']) == (2, None, 24, 1920, None)
    source, gif, kwargs = calls['gif']
    assert source == mp4 and gif == mp4[:-4] + '.gif'
    assert (kwargs['fps'], kwargs['width'], kwargs['start'], kwargs['duration'], kwargs['colors'], kwargs['dither']) == (2, 480, 0.0, 8, 256, 'none')
    # no gif_width: no GIF
    calls.clear()
    mcs_animation.main(vars(module), set_args + ['--set', 'gif_width=null'])
    assert 'mp4' in calls and 'gif' not in calls
    # gif_fps replaces the frame rate of the GIF; the video settings can be set from the command line
    calls.clear()
    mcs_animation.main(vars(module), set_args + ['--set', 'gif_fps=5', '--set', 'playback_fps=4', '--set', 'video_quality=20',
                                                 '--set', 'output_width=1000', '--set', 'video_preset=slow', '--set', 'video_fps=8'])
    assert calls['gif'][2]['fps'] == 5
    kwargs = calls['mp4'][2]
    assert (kwargs['playback_fps'], kwargs['video_fps'], kwargs['crf'], kwargs['width'], kwargs['preset']) == (4, 8, 20, 1000, 'slow')
    # frame_stride: every 2nd PNG file; the GIF follows the playback rate when gif_fps is not set
    calls.clear()
    mcs_animation.main(vars(module), set_args + ['--set', 'frame_stride=2', '--set', 'playback_fps=3'])
    assert [os.path.basename(f) for f in calls['mp4'][0]] == ['mcs_tbpf_1.png', 'mcs_tbpf_3.png'] and calls['gif'][2]['fps'] == 3


def test_the_old_framerate_parameters_are_playback_fps_and_video_fps_with_a_deprecation_warning(monkeypatch, tmp_path, capsys):
    """input_framerate (PNG files per second) is playback_fps, output_framerate (file frame rate) is video_fps: in a wrapper and with --set."""
    module = load_wrapper('make_mcs_animation_himawari_tb')
    calls = {}
    monkeypatch.setattr(mcs_animation, 'plot_mcs_tracks', lambda argv: None)
    monkeypatch.setattr(mcs_animation, 'frames_in_range', lambda figdir, base, start, end: [f'{figdir}{base}1.png'])
    monkeypatch.setattr(mcs_animation, 'make_animation', lambda files, out, **kwargs: calls.update(kwargs) or 0)
    capsys.readouterr()
    values = dict(vars(module))                                    # an old copy of the script has the old variables
    values.pop('playback_fps'); values.pop('video_fps')
    values.update(input_framerate=3, output_framerate=12)
    mcs_animation.main(values, ['--set', f'fig_root={tmp_path}/', '--set', 'run_plotting=false'])
    assert (calls['playback_fps'], calls['video_fps']) == (3, 12)
    out = capsys.readouterr().out
    assert 'input_framerate is now called playback_fps' in out and 'output_framerate is now called video_fps' in out
    mcs_animation.main(vars(module), ['--set', f'fig_root={tmp_path}/', '--set', 'run_plotting=false', '--set', 'input_framerate=4'])
    assert calls['playback_fps'] == 4 and 'input_framerate is now called playback_fps' in capsys.readouterr().out


def test_cell_wrapper_makes_the_mp4_with_the_video_settings(monkeypatch, tmp_path):
    module = load_wrapper('make_cell_animation')
    calls = {}
    monkeypatch.setattr(module, 'frames_in_range', lambda figdir, base, start, end: [f'{figdir}{base}1.png', f'{figdir}{base}2.png'])
    monkeypatch.setattr(module, 'make_animation', lambda files, out, **kwargs: calls.update(files=files, out=out, **kwargs) or 0)
    monkeypatch.setattr(sys, 'argv', ['wrapper', '--set', f'figdir={tmp_path}/frames/', '--set', f'animation_dir={tmp_path}/anim/',
                                      '--set', 'run_plotting=false', '--set', 'video_quality=22'])
    module.main()
    assert calls['out'].endswith('anim/cell_tracks_2020-06-13T00_2020-06-16T00.mp4') and len(calls['files']) == 2
    assert (calls['playback_fps'], calls['frame_stride'], calls['video_fps'], calls['video_quality'], calls['output_width'], calls['video_preset']) == \
        (12, 1, None, 22, 1920, None)
    # ffmpeg fails: the script exits with an error
    monkeypatch.setattr(module, 'make_animation', lambda files, out, **kwargs: 1)
    with pytest.raises(SystemExit):
        module.main()


def test_cell_wrapper_options_parse(monkeypatch, tmp_path):
    module = load_wrapper('make_cell_animation')
    args = cell_cli.parse_cmd_args(run_cell_wrapper(module, monkeypatch, tmp_path))
    assert args['product'] == 'refl' and args['radar_lat'] == module.radar_lat and args['extent'] is not None
    assert args['projection'] == 'PlateCarree'
    args = cell_cli.parse_cmd_args(run_cell_wrapper(module, monkeypatch, tmp_path,
                                                    ['projection=LambertConformal', 'central_longitude=-90', 'track_halo_color=white']))
    assert args['projection'] == 'LambertConformal' and args['central_longitude'] == -90 and args['track_halo_color'] == 'white'
    # the options of plotmap: off by default; a list of lists of places repeats --city / --radar_site
    assert (args['perim_cmap'], args['perim_halo'], args['min_hours'], args['show_terrain']) == (None, None, 0.0, 0)
    assert args['city'] is None and args['radar_site'] is None and args['radar_site_radius'] == 150
    args = cell_cli.parse_cmd_args(run_cell_wrapper(module, monkeypatch, tmp_path,
                                                    ['perim_cmap=glasbey_dark', 'perim_halo=white', 'min_hours=1.5', 'show_terrain=true',
                                                     'grid_linestyle=":"', 'cities=[["A",10,20],["B",11,21]]', 'radar_sites=[["R",5,6]]',
                                                     'radar_site_radius=40']))
    assert (args['perim_cmap'], args['perim_halo'], args['min_hours'], args['show_terrain'], args['grid_linestyle']) == ('glasbey_dark', 'white', 1.5, 1, ':')
    assert args['city'] == [['A', '10', '20'], ['B', '11', '21']] and args['radar_site'] == [['R', '5', '6']] and args['radar_site_radius'] == 40


def test_nexrad_cell_example_makes_a_square_map_with_round_range_circles(monkeypatch, tmp_path):
    module = load_wrapper('make_cell_animation_nexrad')
    recorded = []
    monkeypatch.setattr(module, 'plot_cell_tracks', lambda argv: recorded.append(list(argv)))
    monkeypatch.setattr(sys, 'argv', ['wrapper', '--set', f'fig_root={tmp_path}/', '--set', 'run_ffmpeg=false'])
    module.main()
    args = cell_cli.parse_cmd_args(recorded[0])
    # the title, the features and their line widths
    assert args['title_prefix'] == 'NEXRAD (KHGX)'
    assert (args['draw_land'], args['land_linewidth'], args['draw_border'], args['draw_state']) == (1, 2.0, 1, 1)
    # a LambertConformal map centered on the radar, a square of 2 * half_width_km around it: as long in km east-west as north-south
    assert args['projection'] == 'LambertConformal' and args['central_longitude'] == module.radar_lon
    assert args['central_latitude'] == module.radar_lat and args['standard_parallels'] == [module.radar_lat]
    lon0, lon1, lat0, lat1 = args['extent']
    assert (lon0 + lon1) / 2 == pytest.approx(module.radar_lon, abs=1e-3) and (lat0 + lat1) / 2 == pytest.approx(module.radar_lat, abs=1e-3)
    west_east_km = (lon1 - lon0) * 111.19 * math.cos(math.radians(module.radar_lat))
    south_north_km = (lat1 - lat0) * 111.19
    assert west_east_km == pytest.approx(2 * module.half_width_km, abs=0.5) and south_north_km == pytest.approx(2 * module.half_width_km, abs=0.5)
    assert args['figsize'] == [8.0, 7.2]
    assert args['lon_ticks'] == module.lon_ticks and args['lat_ticks'] == module.lat_ticks
    # another half width moves the extent; projection=null: PlateCarree (no projection parameters)
    recorded.clear()
    monkeypatch.setattr(sys, 'argv', ['wrapper', '--set', f'fig_root={tmp_path}/', '--set', 'run_ffmpeg=false', '--set', 'projection=null'])
    module.main()
    args = cell_cli.parse_cmd_args(recorded[0])
    assert args['projection'] == 'PlateCarree' and args['central_longitude'] is None and args['standard_parallels'] is None


def test_nexrad_cell_example_makes_the_mp4_and_the_gif(monkeypatch, tmp_path):
    module = load_wrapper('make_cell_animation_nexrad')
    calls = {}
    monkeypatch.setattr(module, 'frames_in_range', lambda figdir, base, start, end: [f'{figdir}{base}1.png', f'{figdir}{base}2.png'])
    monkeypatch.setattr(module, 'make_animation', lambda files, out, **kwargs: calls.update(files=files, out=out, **kwargs) or 0)
    monkeypatch.setattr(sys, 'argv', ['wrapper', '--set', f'fig_root={tmp_path}/', '--set', 'run_plotting=false', '--set', 'gif_width=800'])
    module.main()
    assert calls['out'] == f'{tmp_path}/animations_cell/cell_tracks_2014-08-07T12_2014-08-07T15.mp4'
    assert (calls['playback_fps'], calls['video_quality'], calls['output_width'], calls['gif_width'], calls['gif_duration']) == (4, 18, None, 800, None)
    # a parameter that does not exist is an error
    monkeypatch.setattr(sys, 'argv', ['wrapper', '--set', 'no_such_parameter=1'])
    with pytest.raises(SystemExit):
        module.main()


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
