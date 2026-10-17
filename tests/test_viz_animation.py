"""
Tests of pyflextrkr.visualization.animation: the PNG files of a time range, the mp4 made with ffmpeg. Needs pandas;
the mp4 test also needs ffmpeg and Pillow and is skipped without them.
"""
import os
import shutil

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
def test_make_mp4_preset_option(tmp_path, monkeypatch):
    """preset None: no -preset option (x264 default); a preset name: -preset NAME. No ffmpeg run needed."""
    calls = []
    monkeypatch.setattr(animation.subprocess, 'run', lambda cmd, **kw: calls.append(cmd) or type('R', (), {'returncode': 0})())
    pngs = [str(tmp_path / 'a.png')]
    animation.make_mp4(pngs, str(tmp_path / 'm.mp4'))
    animation.make_mp4(pngs, str(tmp_path / 'm.mp4'), preset='slow')
    assert '-preset' not in calls[0]
    assert calls[1][calls[1].index('-preset') + 1] == 'slow' and calls[1][-2:] == ['-y', str(tmp_path / 'm.mp4')]


def test_make_mp4(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    ffmpeg = animation.find_ffmpeg()
    if shutil.which(ffmpeg) is None:
        pytest.skip("ffmpeg is not installed")
    files = []
    for i in range(4):
        path = str(tmp_path / f'frame_{i}.png')
        Image.fromarray(np.full((60, 90, 3), 40 * i, dtype=np.uint8)).save(path)
        files.append(path)
    out_file = str(tmp_path / 'movie.mp4')
    returncode, cmd = animation.make_mp4(files, out_file, input_framerate=2, output_framerate=10, crf=28, width=48)
    assert returncode == 0 and os.path.getsize(out_file) > 0
    assert cmd[0] == ffmpeg and '-crf' in cmd and 'scale=48:-2' in ' '.join(cmd)


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
    files = []
    for i in range(4):
        path = str(tmp_path / f'frame_{i}.png')
        Image.fromarray(np.full((60, 90, 3), 40 * i, dtype=np.uint8)).save(path)
        files.append(path)
    mp4 = str(tmp_path / 'movie.mp4')
    assert animation.make_mp4(files, mp4, input_framerate=2, output_framerate=10, crf=18, width=48)[0] == 0
    gif = str(tmp_path / 'movie.gif')
    returncode, cmd = animation.make_gif(mp4, gif, fps=2, width=32)
    assert returncode == 0 and os.path.getsize(gif) > 0
    image = Image.open(gif)
    assert image.format == 'GIF' and image.size[0] == 32 and image.n_frames >= 3
