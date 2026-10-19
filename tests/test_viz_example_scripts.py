"""
Tests of the batch example scripts in visualization/examples/: make_mcs_quicklook_slurm_tasks.py (the task list and the Slurm job array
script of the quicklooks of several years: no account, environment, e-mail or path of a person in it, only what is given as an option)
and make_gridrad_mcstracks_animation.py (one mp4 per month of the quicklooks of a year). Nothing is plotted, encoded or submitted.
Skipped where matplotlib / cartopy are not installed (the CI environment has neither).
"""
import importlib.util
import os
import subprocess

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("cartopy")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_example(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, 'visualization', 'examples', f'{name}.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#-----------------------------------------------------------------------
def test_the_task_list_has_one_plot_command_per_year(tmp_path):
    tasks = load_example('make_mcs_quicklook_slurm_tasks')
    args = tasks.parse_args(['/data/config_gridrad_mcs_{year}.yml', '/figs/', '--years', '2018', '2020', '--months', '4', '8'])
    lines = tasks.task_lines(args)
    assert len(lines) == 3
    first = lines[0].split()
    assert first[0] == 'python' and first[1] == os.path.join(REPO, 'visualization', 'plot_mcs_tracks.py') and os.path.isfile(first[1])
    assert lines[0].endswith('-s 2018-04-01T00 -e 2018-09-01T00 -c /data/config_gridrad_mcs_2018.yml --product tbze --layout top_bottom -p 1 '
                             '--figsize 10 13 --output /figs/2018/ --figbasename mcs_')
    assert '-c /data/config_gridrad_mcs_2020.yml' in lines[2] and '--output /figs/2020/' in lines[2]
    # to December: until the first day of January of the next year
    assert tasks.period_of(2019, 1, 12) == ('2019-01-01T00', '2020-01-01T00')
    assert tasks.period_of(2019, 6, 6) == ('2019-06-01T00', '2019-07-01T00')


def test_the_slurm_script_has_only_the_lines_that_were_asked_for(tmp_path):
    tasks = load_example('make_mcs_quicklook_slurm_tasks')
    out = tmp_path / 'jobs'
    tasks.main(['/data/config_{year}.yml', '/figs', '--years', '2018', '2019', '--output_dir', str(out)])
    script = (out / 'slurm.submit_mcs_tbze_quicklook.sh').read_text()
    task_file = out / 'tasklist_mcs_tbze_quicklook.txt'
    assert len(task_file.read_text().splitlines()) == 2
    for absent in ('-A ', ' -q ', 'mail', 'conda', 'source activate', 'pyflex', 'm2637', '@'):
        assert absent not in script, absent
    assert '#SBATCH --array=1-2' in script and '#SBATCH -t 01:00:00' in script and '# activate the PyFLEXTRKR environment here' in script
    assert f'sed -n "$SLURM_ARRAY_TASK_ID"p {task_file}' in script
    assert subprocess.run(['bash', '-n', str(out / 'slurm.submit_mcs_tbze_quicklook.sh')]).returncode == 0
    # with the options
    tasks.main(['/data/config_{year}.yml', '/figs', '--years', '2018', '2018', '--output_dir', str(out), '--product', 'tbpf', '--account', 'm0000',
                '--qos', 'debug', '--constraint', 'cpu', '--mail_user', 'someone@example.org', '--env_command', 'conda activate /env/pyflex'])
    script = (out / 'slurm.submit_mcs_tbpf_quicklook.sh').read_text()
    for present in ('#SBATCH -A m0000', '#SBATCH -q debug', '#SBATCH -C cpu', '#SBATCH --mail-user=someone@example.org', 'conda activate /env/pyflex',
                    '#SBATCH --array=1-1', '#SBATCH -J mcs_tbpf_quicklook'):
        assert present in script, present
    with pytest.raises(SystemExit):
        tasks.main(['/data/config.yml', '/figs', '--years', '2018', '2019'])           # no {year}


#-----------------------------------------------------------------------
def test_one_mp4_per_month_of_the_frames_of_each_year(monkeypatch, tmp_path):
    animation = load_example('make_gridrad_mcstracks_animation')
    frames = tmp_path / 'frames'
    for year, names in {2018: ['20180401_000000', '20180401_010000', '20180415_120000', '20180601_000000', '20180901_000000'],
                        2019: ['20190402_000000']}.items():
        (frames / str(year)).mkdir(parents=True)
        for n in names:
            (frames / str(year) / f'mcs_{n}.png').write_bytes(b'x')
    (frames / '2018' / 'other_20180401_000000.png').write_bytes(b'x')                  # another base name: not used
    calls = []
    monkeypatch.setattr(animation, 'make_mp4', lambda files, out, **kwargs: calls.append((sorted(os.path.basename(f) for f in files), out, kwargs)) or (0, ['ffmpeg']))
    animation.main([str(frames), str(tmp_path / 'mp4'), '--years', '2018', '2019', '--months', '4', '6', '--playback_fps', '4'])
    made = {os.path.basename(out): files for files, out, _ in calls}
    assert sorted(made) == ['mcs_201804.mp4', 'mcs_201806.mp4', 'mcs_201904.mp4']          # May and the months of 2019 without frames are skipped
    assert made['mcs_201804.mp4'] == ['mcs_20180401_000000.png', 'mcs_20180401_010000.png', 'mcs_20180415_120000.png']
    assert made['mcs_201806.mp4'] == ['mcs_20180601_000000.png']
    kwargs = calls[0][2]
    assert (kwargs['playback_fps'], kwargs['crf'], kwargs['width']) == (4, 20, 1200)
    assert (tmp_path / 'mp4').is_dir()
    # ffmpeg fails: an error
    monkeypatch.setattr(animation, 'make_mp4', lambda files, out, **kwargs: (1, ['ffmpeg']))
    with pytest.raises(SystemExit):
        animation.main([str(frames), str(tmp_path / 'mp4'), '--years', '2018', '2018', '--months', '4', '4'])
