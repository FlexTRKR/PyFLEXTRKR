"""
Tests of pyflextrkr.visualization.trackdraw: the halo (a contrasting outline with its own transparency) of the track
paths, initiation symbols and track numbers, and the colour levels of the track numbers.
Skipped where matplotlib is not installed (the CI environment has it, so they run there).
"""
import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import binary_dilation

pytest.importorskip("matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyflextrkr.visualization import trackdraw


def render(halo_color=None, halo_width=4.0, halo_alpha=1.0, what='path'):
    """RGB array of a small figure: a black line, a black symbol or a black character on a white background."""
    halo_path, halo_marker, halo_text = trackdraw.halo_effects(halo_color, halo_width, halo_alpha, path_linewidth=2.0)
    fig = plt.figure(figsize=(2, 2), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    if what == 'path':
        ax.plot([0.1, 0.9], [0.5, 0.5], color='k', linewidth=2.0, path_effects=halo_path)
    elif what == 'marker':
        ax.scatter([0.5], [0.5], s=40, color='k', path_effects=halo_marker)
    else:
        ax.text(0.5, 0.5, '8', color='k', fontsize=40, ha='center', va='center', path_effects=halo_text)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(int)
    plt.close(fig)
    return img


#-----------------------------------------------------------------------
def test_no_halo_color_gives_no_path_effects():
    assert trackdraw.halo_effects(None) == (None, None, None)
    assert trackdraw.halo_effects('') == (None, None, None)


@pytest.mark.parametrize("what", ['path', 'marker', 'text'])
def test_halo_is_drawn_and_alpha_makes_it_lighter(what):
    plain = render(None, what=what)
    opaque = render('red', 4.0, 1.0, what=what)
    faint = render('red', 4.0, 0.3, what=what)
    # the halo changes the picture, and the drawn object itself stays black on top
    assert not np.array_equal(plain, opaque)
    assert not np.array_equal(opaque, faint)
    # pixels of the halo (no line / symbol / character there): red when opaque, a mixture with white when faint
    # (2 pixels away from the object: its anti-aliased edge differs a little with and without path effects)
    near_object = binary_dilation(plain.min(axis=2) < 255, iterations=2)
    halo_only = ~near_object & (opaque[..., 1] < 100)
    assert halo_only.sum() > 20
    assert (opaque[halo_only][:, 0] > 200).all() and (opaque[halo_only][:, 1] < 100).all()
    green_faint = faint[halo_only][:, 1]
    assert green_faint.min() > opaque[halo_only][:, 1].max() and green_faint.mean() > 100
    # alpha 0: no halo colour (every pixel is a gray: black on white, as without halo)
    clear = render('red', 4.0, 0.0, what=what)
    assert np.abs(clear[..., 0] - clear[..., 1]).max() <= 2
    # the object itself is drawn on top of the halo: the same amount of black "ink" (the anti-aliasing of a thin line
    # shifts by a sub-pixel with path effects, the total does not change)
    def ink(img):
        return (255 - img.max(axis=2)).sum()
    assert ink(opaque) == pytest.approx(ink(plain), rel=0.2)


def test_halo_width_is_outside_the_edge():
    narrow = render('red', 1.0, 1.0)
    wide = render('red', 6.0, 1.0)
    # red (halo) pixels of a wide halo cover more of the picture
    assert ((wide[..., 0] > 200) & (wide[..., 1] < 100)).sum() > ((narrow[..., 0] > 200) & (narrow[..., 1] < 100)).sum() * 1.5


#-----------------------------------------------------------------------
def make_lifetime(ntracks, first=0):
    return xr.DataArray(np.ones(ntracks), dims='tracks', coords={'tracks': np.arange(first, first + ntracks)})


def test_track_number_levels():
    levels = trackdraw.track_number_levels({'lifetime': make_lifetime(5)})
    assert levels[0] == 1 and levels[-1] == 5 and len(levels) == 5
    # one track: two levels, so that the colour fill has a range
    levels = trackdraw.track_number_levels({'lifetime': make_lifetime(1, first=6)})
    assert list(levels) == [7., 8.]
    # more than 256 tracks: 256 levels (the colormap size)
    levels = trackdraw.track_number_levels({'lifetime': make_lifetime(1000)})
    assert len(levels) == 256 and levels[0] == 1 and levels[-1] == 1000
    # no track
    assert len(trackdraw.track_number_levels({'lifetime': make_lifetime(0)})) == 2


#-----------------------------------------------------------------------
def perimeter_mesh(perim, **kwargs):
    """draw_perimeter on a small axes: the QuadMesh and its RGBA color of every pixel (masked pixels are transparent)."""
    fig = plt.figure(figsize=(4, 3), dpi=50)
    ax = fig.add_axes([0, 0, 1, 1])
    mesh = trackdraw.draw_perimeter(ax, np.arange(perim.shape[1], dtype=float), np.arange(perim.shape[0], dtype=float), perim, **kwargs)
    rgba = np.asarray(mesh.to_rgba(mesh.get_array(), alpha=mesh.get_alpha())).reshape(perim.shape + (4,))
    plt.close(fig)
    return rgba


def test_draw_perimeter_in_one_color_and_in_one_color_per_track():
    perim = np.zeros((6, 8), dtype=np.int32)
    perim[1:3, 1:3] = 1
    perim[3:5, 4:7] = 2
    perim[0, 7] = 7
    drawn = perim > 0
    # one color, with the transparency of the drawing
    rgba = perimeter_mesh(perim, color='red', alpha=0.5)
    np.testing.assert_allclose(rgba[drawn], [[1, 0, 0, 0.5]] * int(drawn.sum()))
    assert (rgba[~drawn][:, 3] == 0).all()                     # nothing where there is no perimeter
    assert trackdraw.draw_perimeter(None, None, None, None) is None
    # one color per track: the color index is the track number - 1, modulo the number of colors
    colors = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]], dtype=float)
    rgba = perimeter_mesh(perim, cmap_colors=colors)
    np.testing.assert_allclose(rgba[1, 1], colors[0])          # track 1
    np.testing.assert_allclose(rgba[3, 4], colors[1])          # track 2
    np.testing.assert_allclose(rgba[0, 7], colors[0])          # track 7: index 6 % 3 = 0
    # with the track numbers that are drawn: the color is their rank, so the colors do not depend on the other tracks
    rgba = perimeter_mesh(perim, cmap_colors=colors, rank_tracknumbers=np.array([2, 7]))
    np.testing.assert_allclose(rgba[3, 4], colors[0])          # track 2 is the first
    np.testing.assert_allclose(rgba[0, 7], colors[1])          # track 7 is the second
    assert (rgba[~drawn][:, 3] == 0).all()


def test_cmap_colors_are_all_the_colors_of_a_colormap():
    from pyflextrkr.visualization import colors
    tab10 = colors.cmap_colors('tab10')
    assert tab10.shape == (10, 4) and tab10[0] == pytest.approx((31 / 255, 119 / 255, 180 / 255, 1.0))
    assert colors.cmap_colors('viridis').shape == (256, 4)
