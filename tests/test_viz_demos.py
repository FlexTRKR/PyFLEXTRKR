"""
The plot commands of the demo scripts (config/demo_*.sh) and of the docstrings of the animation examples (visualization/examples/make_*animation*.py)
make the same pictures as the wrapper itself: each demo script that has an example wrapper runs plot_*_tracks.py with the options of the wrapper
(the style, the map, the figure size, the file names), so a demo run and a wrapper run write identical PNG files into the same folder
({root_path}/quicklooks_trackpaths/). The commands are read from the files and parsed with the real command lines; the period, the
config file, the output folder and the number of workers may differ. Skipped where matplotlib / cartopy are not installed.
"""
import os
import re
import shlex

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("cartopy")

from pyflextrkr.visualization import mcs_cli, cell_cli, mcs_animation, cell_animation
from test_viz_wrappers import load_wrapper, REPO

REQUIRED = ['-s', '2025-06-01T00', '-e', '2025-06-01T06', '-c', 'config.yml']
# not part of the picture: the period, the config file, where the files go, how the work is shared
NOT_PICTURE = {'start_datetime', 'end_datetime', 'config_file', 'out_dir', 'workers', 'run_parallel'}

# (demo script, example wrapper, plot module, animation library)
PAIRS = [
    ('demo_cell_nexrad.sh', 'make_cell_animation_nexrad', cell_cli, cell_animation),
    ('demo_mcs_wrf_tbradar.sh', 'make_mcs_animation_wrf_tbze', mcs_cli, mcs_animation),
    ('demo_mcs_imerg.sh', 'make_mcs_animation_imerg_1panel', mcs_cli, mcs_animation),
    ('demo_mcs_himawari.sh', 'make_mcs_animation_himawari_tb', mcs_cli, mcs_animation),
    ('demo_mcs_imergv7_sahel.sh', 'make_mcs_animation_imergv7_sahel', mcs_cli, mcs_animation),
]


def plot_command(text, script_pattern):
    """The words of the first command that runs plot_*_tracks.py in a shell script or a docstring (continuation lines joined, comments left out)."""
    text = re.sub(r'\\\n', ' ', text)
    for line in text.splitlines():
        if re.search(script_pattern, line) and not line.lstrip().startswith('#'):
            words = shlex.split(line.split('#')[0].strip())
            i = [k for k, w in enumerate(words) if re.search(script_pattern, w)][0]
            return words[i + 1:]
    raise AssertionError(f'no command {script_pattern}')


def picture_options(module, words):
    """The parsed options of a plot command that matter for the picture, without the ones that are the defaults."""
    words = [w.replace('${config_demo}', 'config.yml').replace('${quicklook_dir}', '/out/') for w in words]
    defaults = module.parse_cmd_args(REQUIRED)
    args = module.parse_cmd_args(words)
    for a in (defaults, args):
        if a['time_format'] is None:                              # the command line reads 'yyyymodd_hhmmss' when there is none
            a['time_format'] = 'yyyymodd_hhmmss'
        for key in ('draw_border', 'draw_state', 'draw_river'):   # not given: not drawn (0)
            a[key] = a[key] or 0
    if args.get('figsize') is not None:
        defaults['figsize_x'] = args['figsize_x']                # the width is not used when the figure size is given
    return {k: v for k, v in args.items() if k not in NOT_PICTURE and v != defaults[k]}


def wrapper_options(library, example, monkeypatch, tmp_path):
    recorded = []
    plot = 'plot_cell_tracks' if library is cell_animation else 'plot_mcs_tracks'
    monkeypatch.setattr(library, plot, lambda argv: recorded.append(list(argv)))
    library.main(vars(load_wrapper(example)), ['--set', f'fig_root={tmp_path}/', '--set', 'run_ffmpeg=false'])
    return recorded[0]


@pytest.mark.parametrize("demo, example, module, library", PAIRS, ids=[p[1] for p in PAIRS])
def test_the_demo_script_plots_with_the_options_of_the_example(demo, example, module, library, monkeypatch, tmp_path):
    expected = picture_options(module, wrapper_options(library, example, monkeypatch, tmp_path))
    demo_text = open(os.path.join(REPO, 'config', demo)).read()
    from_demo = picture_options(module, plot_command(demo_text, r'visualization/plot_(mcs|cell)_tracks\.py'))
    assert from_demo == expected, f'{demo} and {example} differ'


@pytest.mark.parametrize("example, module, library", [('make_mcs_animation', mcs_cli, mcs_animation)] + [(p[1], p[2], p[3]) for p in PAIRS],
                         ids=['make_mcs_animation'] + [p[1] for p in PAIRS])
def test_the_plot_command_in_the_docstring_of_the_example_is_the_one_it_runs(example, module, library, monkeypatch, tmp_path):
    """Copy, paste and run: the command in the header is not an older version of the options."""
    expected = picture_options(module, wrapper_options(library, example, monkeypatch, tmp_path))
    from_doc = picture_options(module, plot_command(load_wrapper(example).__doc__, r'\.\./plot_(mcs|cell)_tracks\.py'))
    # the global wrapper's plain command leaves the track lookback of the period; the files go to the folder of the docstring
    assert from_doc == expected, example


def test_docstrings_have_commands_that_can_be_pasted():
    """No '>' in front of the commands and no doubled backslash: the docstrings with line continuations are raw strings and the commands are plain."""
    import glob
    paths = sorted(glob.glob(os.path.join(REPO, 'visualization', '*.py')) + glob.glob(os.path.join(REPO, 'visualization', 'examples', '*.py'))
                   + glob.glob(os.path.join(REPO, 'pyflextrkr', 'visualization', '*.py')))
    assert len(paths) > 30
    for path in paths:
        text = open(path).read()
        if '"""' not in text:
            continue
        head, doc = text.split('"""')[0], text.split('"""')[1]
        name = os.path.relpath(path, REPO)
        assert not re.search(r'^>', doc, re.M), f'{name}: a line of the docstring starts with ">"'
        assert '\\\\\n' not in doc, f'{name}: doubled backslash at the end of a line'
        if re.search(r'\\\n', doc):
            assert head.endswith('r'), f'{name}: the docstring has line continuations, so it must be a raw string (r""")'


@pytest.mark.parametrize("demo, example, module, library", PAIRS, ids=[p[1] for p in PAIRS])
def test_the_demo_registry_plots_with_the_options_of_the_demo_script(demo, example, module, library):
    """tests/run_demo_tests.py (--with-plots) runs the same plot command as config/demo_*.sh."""
    from run_demo_tests import DEMOS
    info = DEMOS[demo[:-3]]
    demo_text = open(os.path.join(REPO, 'config', demo)).read()
    from_demo = picture_options(module, plot_command(demo_text, r'visualization/plot_(mcs|cell)_tracks\.py'))
    from_registry = picture_options(module, shlex.split(info.plot_args) + ['-c', 'config.yml'])      # the registry adds -c and --output itself
    assert from_registry == from_demo, demo
