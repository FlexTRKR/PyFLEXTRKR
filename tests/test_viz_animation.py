"""
Tests of pyflextrkr.visualization.animation: the PNG files of a time range, the mp4 made with ffmpeg. Needs pandas;
the mp4 test also needs ffmpeg and Pillow and is skipped without them.
"""
import os
import re
import shutil
import subprocess

import numpy as np
import pytest

from pyflextrkr.visualization import animation


def touch(directory, names):
    for n in names:
        open(os.path.join(directory, n), 'wb').close()


#-----------------------------------------------------------------------
def test_frames_in_range_selects_the_time_range_inclusive(tmp_path):
    names = [f'mcs_20250601_{h:02d}3000.png' for h in range(6)]
    touch(tmp_path, names + ['mcs_notatime.png', 'other_20250601_013000.png', 'mcs_20250601_013000.txt', 'mcs_20250601_9999.png'])
    figdir = f'{tmp_path}/'
    out = animation.frames_in_range(figdir, 'mcs_', '2025-06-01T01:30', '2025-06-01T03:30')
    assert [os.path.basename(f) for f in out] == names[1:4]       # both ends included, sorted
    # a date only: the whole day
    assert len(animation.frames_in_range(figdir, 'mcs_', '2025-06-01', '2025-06-02')) == 6
    # nothing in the range
    assert animation.frames_in_range(figdir, 'mcs_', '2025-07-01', '2025-07-02') == []
    # another base name
    assert [os.path.basename(f) for f in animation.frames_in_range(figdir, 'other_', '2025-06-01', '2025-06-02')] == ['other_20250601_013000.png']


def test_frames_in_range_with_no_base_name(tmp_path):
    touch(tmp_path, ['20250601_003000.png', '20250602_003000.png'])
    out = animation.frames_in_range(f'{tmp_path}/', '', '2025-06-01T00', '2025-06-01T23')
    assert [os.path.basename(f) for f in out] == ['20250601_003000.png']


#-----------------------------------------------------------------------
def make_pngs(directory, n=4, size=(90, 60)):
    """n small PNG files (frame_0.png ...) with different gray levels; size = (width, height)."""
    Image = pytest.importorskip("PIL.Image")
    files = []
    for i in range(n):
        path = str(directory / f'frame_{i}.png')
        Image.fromarray(np.full((size[1], size[0], 3), 40 * i, dtype=np.uint8)).save(path)
        files.append(path)
    return files


def need_ffmpeg():
    if shutil.which(animation.find_ffmpeg()) is None:
        pytest.skip("ffmpeg is not installed")
    return animation.find_ffmpeg()


def probe(video):
    """(width, height, number of frames) of a video file, read with ffmpeg."""
    result = subprocess.run([animation.find_ffmpeg(), '-hide_banner', '-i', video, '-map', '0:v:0', '-c', 'copy', '-f', 'null', '-'],
                            capture_output=True, text=True)
    width, height = re.search(r'Video: .*?, (\d+)x(\d+)', result.stderr).groups()
    return int(width), int(height), int(re.findall(r'frame=\s*(\d+)', result.stderr)[-1])


def test_video_frame_rate_is_the_playback_fps_repeated_to_at_least_10():
    """None: a whole number of repeats of every PNG file, at least 10 frames per second; a given value is used as it is."""
    rates = {playback: animation.video_frame_rate(playback) for playback in (0.5, 1, 2, 3, 5, 6, 10, 12, 24)}
    assert rates == {0.5: 10, 1: 10, 2: 10, 3: 12, 5: 10, 6: 12, 10: 10, 12: 12, 24: 24}
    assert animation.video_frame_rate(2, 8) == 8


def test_command_text_can_be_pasted_into_a_shell():
    cmd = ['ffmpeg', '-vf', "scale='min(1920,iw)':-2", '-y', 'my movie.mp4', "it's.mp4"]
    assert animation.command_text(cmd) == '''ffmpeg -vf "scale='min(1920,iw)':-2" -y 'my movie.mp4' "it's.mp4"'''


def test_scale_filter_is_never_larger_than_the_picture_and_always_even():
    assert animation.scale_filter(1920) == "scale='min(1920,trunc(iw/2)*2)':-2"
    assert animation.scale_filter(1921) == "scale='min(1920,trunc(iw/2)*2)':-2"        # an odd width is made even too
    assert animation.scale_filter(None) == animation.scale_filter(0) == 'scale=trunc(iw/2)*2:trunc(ih/2)*2'


def test_list_png_files_from_a_directory_a_pattern_or_a_list(tmp_path):
    touch(tmp_path, ['b.png', 'a.png', 'c.png', 'notes.txt'])
    names = lambda files: [os.path.basename(f) for f in files]
    assert names(animation.list_png_files(str(tmp_path))) == ['a.png', 'b.png', 'c.png']          # a directory: all its PNG files, sorted
    assert names(animation.list_png_files(str(tmp_path / '[ab].png'))) == ['a.png', 'b.png']       # a glob pattern
    assert names(animation.list_png_files(str(tmp_path / 'c.png'))) == ['c.png']                   # one file
    given = [str(tmp_path / 'c.png'), str(tmp_path / 'a.png')]
    assert animation.list_png_files(given) == given                                                # a list keeps its order
    assert animation.list_png_files(str(tmp_path / 'none_*.png')) == []


def test_make_mp4_command_and_options(tmp_path, monkeypatch):
    """The ffmpeg command: input rate (-r before -i) = playback_fps, file rate (-r after the codec), -crf, scale, preset only when given. No ffmpeg run needed."""
    calls = []

    def fake_run(cmd, **kw):
        listfile = cmd[cmd.index('-i') + 1]
        calls.append((cmd, [line for line in open(listfile).read().split('\n') if line], listfile))
        return type('R', (), {'returncode': 0})()
    monkeypatch.setattr(animation.subprocess, 'run', fake_run)
    pngs = [str(tmp_path / f'{i}.png') for i in range(5)]
    returncode, cmd = animation.make_mp4(pngs, str(tmp_path / 'm.mp4'))
    assert returncode == 0 and calls[0][0] == cmd and not os.path.exists(calls[0][2])      # the list of files is removed again
    assert cmd[cmd.index('-r') + 1] == '2' and cmd.index('-r') < cmd.index('-i')                  # PNG files per second (playback_fps)
    assert cmd[cmd.index('-r', cmd.index('-i')) + 1] == '10'                                        # frame rate of the file, automatic
    assert cmd[cmd.index('-crf') + 1] == '24' and cmd[cmd.index('-vf') + 1] == "scale='min(1920,trunc(iw/2)*2)':-2"
    assert cmd[cmd.index('-pix_fmt') + 1] == 'yuv420p' and '-preset' not in cmd and cmd[-2:] == ['-y', str(tmp_path / 'm.mp4')]
    assert len(calls[0][1]) == 5
    # a preset, the file rate, the quality and the width given; every 2nd PNG file
    animation.make_mp4(pngs, str(tmp_path / 'm.mp4'), playback_fps=12, frame_stride=2, video_fps=24, crf=20, preset='slow', width=None)
    cmd, listed, _ = calls[1]
    assert cmd[cmd.index('-r') + 1] == '12' and cmd[cmd.index('-r', cmd.index('-i')) + 1] == '24'
    assert cmd[cmd.index('-preset') + 1] == 'slow' and cmd[cmd.index('-crf') + 1] == '20'
    assert cmd[cmd.index('-vf') + 1] == 'scale=trunc(iw/2)*2:trunc(ih/2)*2'
    assert listed == [f"file '{p}'" for p in pngs[::2]]
    with pytest.raises(ValueError, match='no PNG files'):
        animation.make_mp4(str(tmp_path / 'nothing_*.png'), str(tmp_path / 'm.mp4'))


def test_make_mp4(tmp_path):
    """ffmpeg is run: 4 PNG files at 2 per second are 2 seconds, written at 10 frames per second; sizes are made even, a width is never enlarged."""
    need_ffmpeg()
    files = make_pngs(tmp_path, 4, size=(91, 61))                               # odd sizes: yuv420p needs even ones
    out_file = str(tmp_path / 'movie.mp4')
    returncode, cmd = animation.make_mp4(files, out_file, playback_fps=2, crf=28, width=1920)
    assert returncode == 0 and probe(out_file) == (90, 60, 20)                  # never enlarged, 4 files x 5 repeats
    assert animation.make_mp4(files, out_file, width=48)[0] == 0 and probe(out_file)[:2] == (48, 32)
    assert animation.make_mp4(files, out_file, video_fps=2)[0] == 0 and probe(out_file)[2] == 4          # file rate = playback rate: no repeats
    assert animation.make_mp4(files, out_file, frame_stride=2, width=None)[0] == 0 and probe(out_file) == (90, 60, 10)
    # a directory and a glob pattern are the PNG files themselves
    assert animation.make_mp4(str(tmp_path), out_file)[0] == 0 and probe(out_file)[2] == 20
    assert animation.make_mp4(str(tmp_path / 'frame_[01].png'), out_file)[0] == 0 and probe(out_file)[2] == 10


def test_make_animation_prints_the_settings_and_makes_the_mp4_and_the_gif(tmp_path, monkeypatch, capsys):
    """make_animation: every stride-th PNG file, the settings printed, the GIF only with gif_width (the ffmpeg commands are replaced)."""
    calls = {}

    def fake_mp4(files, out, **kwargs):
        calls['mp4'] = (files, out, kwargs)
        open(out, 'wb').write(b'x' * 2000)
        return 0, ['ffmpeg', 'mp4']

    def fake_gif(source, out, **kwargs):
        calls['gif'] = (source, out, kwargs)
        open(out, 'wb').write(b'GIF')
        return 0, ['ffmpeg', 'gif']
    monkeypatch.setattr(animation, 'make_mp4', fake_mp4)
    monkeypatch.setattr(animation, 'make_gif', fake_gif)
    pngs = [str(tmp_path / f'{i}.png') for i in range(6)]
    mp4 = str(tmp_path / 'm.mp4')
    assert animation.make_animation(pngs, mp4, playback_fps=2, frame_stride=3) == 0
    out = capsys.readouterr().out
    assert 'Playback: 2 PNG files per second' in out and 'every 3th PNG: 2 files, duration 1 s' in out
    assert 'Video file: 10 frames per second (each PNG repeated 5 times' in out and 'Video quality (CRF): 24' in out
    files, _, kwargs = calls['mp4']
    assert files == pngs[::3] and kwargs['crf'] == 24 and kwargs['width'] == 1920 and kwargs['video_fps'] is None and kwargs['preset'] is None
    assert 'gif' not in calls
    # a GIF: gif_fps None is the playback rate
    assert animation.make_animation(pngs, mp4, playback_fps=4, gif_width=300, gif_duration=2, gif_colors=128) == 0
    source, gif_file, kwargs = calls['gif']
    assert source == mp4 and gif_file == str(tmp_path / 'm.gif')
    assert (kwargs['fps'], kwargs['width'], kwargs['start'], kwargs['duration'], kwargs['colors'], kwargs['dither']) == (4, 300, 0.0, 2, 128, 'none')
    # a failing ffmpeg gives its return code
    monkeypatch.setattr(animation, 'make_mp4', lambda files, out, **kw: (1, ['ffmpeg']))
    assert animation.make_animation(pngs, mp4) == 1
    with pytest.raises(ValueError, match='no PNG files'):
        animation.make_animation([], mp4)


def test_pngs_to_animation_command_line(tmp_path, capsys):
    """visualization/pngs_to_animation.py (animation.main): the plain ffmpeg commands with --print_command, the files with ffmpeg."""
    assert animation.main([str(tmp_path / 'frames_*.png'), 'movie.mp4', '--print_command', '--gif_width', '480', '--gif_duration', '8']) == 0
    lines = capsys.readouterr().out.strip().split('\n')
    assert lines[0].startswith("ffmpeg -framerate 2 -pattern_type glob -i ")
    assert """-vf "scale='min(1920,trunc(iw/2)*2)':-2" -c:v libx264 -r 10 -crf 24 -pix_fmt yuv420p -y movie.mp4""" in lines[0]
    assert lines[1].startswith("ffmpeg -t 8 -i movie.mp4 -vf") and 'palettegen=max_colors=256:stats_mode=full' in lines[1] and lines[1].endswith('-loop 0 -y movie.gif')
    # a directory becomes the pattern of its PNG files
    assert animation.main([str(tmp_path), 'movie.mp4', '--print_command', '--playback_fps', '12', '--video_quality', '20', '--output_width', '0',
                           '--video_preset', 'slow']) == 0
    line = capsys.readouterr().out.strip()
    assert f"-framerate 12 -pattern_type glob -i '{tmp_path}/*.png'" in line
    assert 'scale=trunc(iw/2)*2:trunc(ih/2)*2' in line and '-r 12 -crf 20' in line and line.endswith('-preset slow -y movie.mp4')
    with pytest.raises(SystemExit):
        animation.main(['frames/', 'movie.avi'])                                                  # the last argument is the mp4 file


def test_pngs_to_animation_makes_the_mp4_and_gif_from_a_directory(tmp_path, capsys):
    need_ffmpeg()
    frames = tmp_path / 'frames'
    frames.mkdir()
    make_pngs(frames, 4)
    out_file = str(tmp_path / 'out' / 'movie.mp4')
    assert animation.main([str(frames), out_file, '--playback_fps', '2', '--gif_width', '32']) == 0
    assert probe(out_file) == (90, 60, 20) and os.path.getsize(str(tmp_path / 'out' / 'movie.gif')) > 0
    assert 'Animation created successfully' in capsys.readouterr().out


#-----------------------------------------------------------------------
def gif_command(monkeypatch, **kwargs):
    calls = []
    monkeypatch.setattr(animation.subprocess, 'run', lambda cmd, **kw: calls.append(cmd) or type('R', (), {'returncode': 0})())
    animation.make_gif('movie.mp4', 'movie.gif', **kwargs)
    cmd = calls[0]
    return cmd, cmd[cmd.index('-vf') + 1]


def test_make_gif_palette_options(monkeypatch):
    """The defaults: 2 frames per second, one palette of 256 colors for all frames, no dither. No ffmpeg run needed."""
    cmd, vf = gif_command(monkeypatch)
    assert vf.startswith("fps=2,scale='min(800,iw)':-1:flags=lanczos")
    assert 'palettegen=max_colors=256:stats_mode=full' in vf and 'paletteuse=dither=none' in vf and 'new=1' not in vf
    assert cmd[cmd.index('-ss') + 1] == '0.0' and '-t' not in cmd and cmd[-2:] == ['-y', 'movie.gif']
    # a palette for each frame, error diffusion, 128 colors, the changed rectangle only
    cmd, vf = gif_command(monkeypatch, palette='frame', dither='sierra2_4a', colors=128, diff_mode='rectangle', fps=5, width=480)
    assert 'fps=5' in vf and "min(480,iw)" in vf
    assert 'palettegen=max_colors=128:stats_mode=single' in vf and 'paletteuse=dither=sierra2_4a:new=1:diff_mode=rectangle' in vf
    # the part of the video
    cmd, vf = gif_command(monkeypatch, start=2.5, duration=12)
    assert cmd[cmd.index('-ss') + 1] == '2.5' and cmd[cmd.index('-t') + 1] == '12'
    with pytest.raises(ValueError, match='palette'):
        gif_command(monkeypatch, palette='local')


def test_make_gif(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    ffmpeg = animation.find_ffmpeg()
    if shutil.which(ffmpeg) is None:
        pytest.skip("ffmpeg is not installed")
    files = make_pngs(tmp_path, 4)
    mp4 = str(tmp_path / 'movie.mp4')
    assert animation.make_mp4(files, mp4, playback_fps=2, crf=18, width=48)[0] == 0
    gif = str(tmp_path / 'movie.gif')
    returncode, cmd = animation.make_gif(mp4, gif, fps=2, width=32)
    assert returncode == 0 and os.path.getsize(gif) > 0
    image = Image.open(gif)
    assert image.format == 'GIF' and image.size[0] == 32 and image.n_frames >= 3
