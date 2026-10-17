"""
Tests of the colorbar layout of the MCS maps: the GridSpec numbers (figure_layout) keep the layout of the old scripts by
default and follow --cbar_thickness / --cbar_spacing, and the text of the colorbars stays inside the figure and clear of the
next colorbar (mapcore.keep_colorbar_text_inside). Skipped where matplotlib / cartopy are not installed.
"""
import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("cartopy")
pytest.importorskip("colormaps")
pytest.importorskip("colorcet")

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from pyflextrkr.visualization import mcs_map
from pyflextrkr.visualization.mapcore import keep_colorbar_text_inside
from pyflextrkr.visualization.mapcore import figure_margins as mapcore_figure_margins

pytestmark = pytest.mark.filterwarnings("ignore")


def bars_of(layout, fig):
    """Position (x0, y0, w, h) of the map and of its colorbars in the figure, from the GridSpec numbers of figure_layout."""
    gs = gridspec.GridSpec(figure=fig, **layout['gridspec'])
    gs.update(**layout['update'])
    if layout['cb_gridspec'] is None:
        return [gs[i].get_position(fig).bounds for i in range(gs.nrows * gs.ncols)]
    gs_cb = gridspec.GridSpecFromSubplotSpec(subplot_spec=gs[1], **layout['cb_gridspec'])
    return [gs[0].get_position(fig).bounds, gs_cb[0].get_position(fig).bounds, gs_cb[1].get_position(fig).bounds]


#-----------------------------------------------------------------------
def test_figure_layout_default_is_the_layout_of_the_old_scripts():
    fig = plt.figure(figsize=(8, 8))
    # 1 panel with 2 colorbars: the numbers of the old code (width ratios [1, 0.1], inner [0.01, 0.01] with wspace 5)
    old = gridspec.GridSpec(1, 2, figure=fig, height_ratios=[1], width_ratios=[1, 0.1])
    old.update(wspace=0.05, left=0.05, right=0.95, top=0.9, bottom=0.08)
    inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=old[1], height_ratios=[1], width_ratios=[0.01, 0.01], wspace=5)
    expected = [old[0].get_position(fig).bounds, inner[0].get_position(fig).bounds, inner[1].get_position(fig).bounds]
    assert np.allclose(bars_of(mcs_map.figure_layout(1, True), fig), expected, atol=1e-12)
    assert mcs_map.figure_layout(1, True, grid_top=0.92)['update']['top'] == 0.92
    # Tb only
    layout = mcs_map.figure_layout(1, False)
    assert layout['gridspec']['width_ratios'] == [1, 0.03] and layout['update'] == mcs_map.GRID_1PANEL_TBONLY['update']
    # 2 panels
    layout = mcs_map.figure_layout(2, True, 'left_right')
    assert layout['gridspec'] == dict(nrows=2, ncols=2, height_ratios=[1, 0.02], width_ratios=[1, 1])
    assert layout['update'] == dict(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.15, hspace=0.15)
    layout = mcs_map.figure_layout(2, True, 'top_bottom')
    assert layout['gridspec'] == dict(nrows=2, ncols=2, height_ratios=[0.4, 0.4], width_ratios=[1, 0.02])
    assert layout['update'] == dict(left=0.05, right=0.93, bottom=0.05, top=0.9, wspace=0.03, hspace=0.2)


def test_colorbar_thickness_and_spacing_are_fractions_of_the_map():
    fig = plt.figure(figsize=(8, 8))
    # 1 panel, 2 colorbars side by side: thickness and gap as fractions of the map width
    (mx, my, mw, mh), (x1, y1, w1, h1), (x2, y2, w2, h2) = bars_of(mcs_map.figure_layout(1, True, cbar_thickness=0.03, cbar_spacing=0.2), fig)
    assert w1 / mw == pytest.approx(0.03) and w2 / mw == pytest.approx(0.03) and (x2 - (x1 + w1)) / mw == pytest.approx(0.2)
    assert (y1, h1) == pytest.approx((my, mh))
    # thickness alone keeps the default gap (5/70 of the map width)
    (mx, my, mw, mh), (x1, y1, w1, h1), (x2, y2, w2, h2) = bars_of(mcs_map.figure_layout(1, True, cbar_thickness=0.03), fig)
    assert w1 / mw == pytest.approx(0.03) and (x2 - (x1 + w1)) / mw == pytest.approx(5 / 70)
    # Tb only and the colorbars at the right of 2 stacked panels: the width of the bar / the width of the map
    (mx, my, mw, mh), (cx, cy, cw, ch) = bars_of(mcs_map.figure_layout(1, False, cbar_thickness=0.05), fig)
    assert cw / mw == pytest.approx(0.05)
    mapa, cbara, mapb, cbarb = bars_of(mcs_map.figure_layout(2, True, 'top_bottom', cbar_thickness=0.04), fig)
    assert cbara[2] / mapa[2] == pytest.approx(0.04) and cbarb[2] / mapb[2] == pytest.approx(0.04)
    # side by side panels, colorbars below: the height of the bar / the height of the map (the same ratio for any orientation)
    mapa, mapb, cbara, cbarb = bars_of(mcs_map.figure_layout(2, True, 'left_right', cbar_thickness=0.04), fig)
    assert cbara[3] / mapa[3] == pytest.approx(0.04) and cbarb[3] / mapb[3] == pytest.approx(0.04)


def test_margins_in_inches_keep_the_size_of_the_maps():
    fig = plt.figure(figsize=(10, 4))
    nominal = mcs_map.figure_layout(1, True)
    size, margins = mapcore_figure_margins([10., 3.7], nominal['update'], 10)
    layout = mcs_map.figure_layout(1, True, margins_in=margins, height=size[1])
    assert (layout['update']['top'] - layout['update']['bottom']) * size[1] == pytest.approx((0.9 - 0.08) * 3.7, abs=0.01)   # the map cell [inches]
    assert (1 - layout['update']['top']) * size[1] == pytest.approx(margins[0]) and layout['update']['bottom'] * size[1] == pytest.approx(margins[1])
    # the same for 2 panels, and the numbers of the other layout keys are the nominal ones
    for panels, has_var2, name in ((2, True, 'top_bottom'), (2, True, 'left_right'), (1, False, None)):
        nominal = mcs_map.figure_layout(panels, has_var2, name)
        layout = mcs_map.figure_layout(panels, has_var2, name, margins_in=(0.6, 0.4), height=5.0)
        assert layout['gridspec'] == nominal['gridspec'] and layout['update']['left'] == nominal['update']['left']
        assert layout['update']['top'] == pytest.approx(1 - 0.6 / 5.0) and layout['update']['bottom'] == pytest.approx(0.4 / 5.0)


#-----------------------------------------------------------------------
def make_colorbars(width, height, fontsize, vertical, n=2, step=0.1):
    """A figure with a map and n colorbars (with long labels) placed as in the figure of the MCS maps."""
    mpl.rcParams['font.size'] = fontsize
    fig = plt.figure(figsize=(width, height), dpi=50)
    FigureCanvasAgg(fig)
    labels = ['Tb (K)', 'Precipitation (mm h$^{-1}$)'][:n]
    ticks = [np.arange(180, 301, 20), [2, 3, 4, 5, 6, 8, 10, 15, 20, 30]]
    if vertical:
        ax = fig.add_axes([0.05, 0.08, 0.75, 0.8])
        boxes = [[0.82 + step * i, 0.08, 0.014, 0.8] for i in range(n)]       # step: distance between the colorbars
    else:
        ax = fig.add_axes([0.1, 0.45, 0.8, 0.45])
        boxes = [[0.1, 0.3 - 0.15 * i, 0.8, 0.03] for i in range(n)]
    caxes = []
    for i, box in enumerate(boxes):
        cax = fig.add_axes(box)
        mappable = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(ticks[i][0], ticks[i][-1]), cmap='viridis')
        fig.colorbar(mappable, cax=cax, label=labels[i], ticks=ticks[i], extend='both', orientation='vertical' if vertical else 'horizontal')
        caxes.append(cax)
    return fig, ax, caxes


def text_boxes(fig, caxes):
    renderer = fig.canvas.get_renderer()
    return [fig.transFigure.inverted().transform_bbox(c.get_tightbbox(renderer)) for c in caxes]


@pytest.mark.parametrize("spacing_auto", [True, False])
def test_text_of_colorbars_at_the_right_stays_inside_the_figure(spacing_auto):
    fig, ax, (cax1, cax2) = make_colorbars(6, 6, fontsize=18, vertical=True)
    before = text_boxes(fig, [cax1, cax2])
    assert before[1].x1 > 1.0 and before[0].x1 > cax2.get_position().x0           # cut off at the right, overlap of the bars
    shape = ax.get_position().width * 6 / (ax.get_position().height * 6)
    gap = cax2.get_position(original=True).x0 - cax1.get_position(original=True).x1
    keep_colorbar_text_inside(fig, [(ax, [cax1, cax2])], 'vertical', spacing_auto=spacing_auto)
    after = text_boxes(fig, [cax1, cax2])
    assert after[1].x1 <= 1.0 and after[1].x1 == pytest.approx(1.0 - 4 / 72 / 6, abs=2e-3)       # 4 points from the edge
    # the map keeps its shape and the colorbars have its height
    assert ax.get_position().width * 6 / (ax.get_position().height * 6) == pytest.approx(shape)
    for cax in (cax1, cax2):
        # (the position that was set; the axes of a colorbar with extensions draws its body a little shorter)
        box = cax.get_position(original=True)
        assert (box.y0, box.height) == pytest.approx((ax.get_position().y0, ax.get_position().height))
    if spacing_auto:
        assert after[0].x1 <= cax2.get_position().x0                               # clear of the next colorbar
    else:
        gap_after = cax2.get_position(original=True).x0 - cax1.get_position(original=True).x1
        assert gap_after == pytest.approx(gap)                                      # the gap that was asked for is kept


def test_nothing_moves_when_the_text_fits():
    fig, ax, (cax1, cax2) = make_colorbars(10, 6, fontsize=8, vertical=True, step=0.075)
    positions = [a.get_position(original=True).bounds for a in (ax, cax1, cax2)]
    keep_colorbar_text_inside(fig, [(ax, [cax1, cax2])], 'vertical')
    assert [a.get_position(original=True).bounds for a in (ax, cax1, cax2)] == positions


def test_text_of_colorbars_below_the_maps_stays_inside_the_figure():
    fig, ax, caxes = make_colorbars(6, 4, fontsize=18, vertical=False, n=1)
    caxes[0].set_position([0.1, 0.12, 0.8, 0.03])
    assert text_boxes(fig, caxes)[0].y0 < 0.0                                       # the label is below the figure
    p = ax.get_position()
    keep_colorbar_text_inside(fig, [(ax, caxes)], 'horizontal')
    q = ax.get_position()
    assert text_boxes(fig, caxes)[0].y0 == pytest.approx(4 / 72 / 4, abs=2e-3)
    assert q.y1 == pytest.approx(p.y1) and q.width * 6 / (q.height * 4) == pytest.approx(p.width * 6 / (p.height * 4))
    box = caxes[0].get_position(original=True)
    assert (box.x0, box.width) == pytest.approx((q.x0, q.width))
