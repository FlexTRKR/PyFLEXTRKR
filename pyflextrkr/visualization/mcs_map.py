"""
MCS track maps: brightness temperature (Tb) with the tracked MCS (outline, track path, track number), plus a second field
(precipitation 'tbpf', or radar reflectivity 'tbze') or no second field (Tb only 'tb'), on 2 panels or on 1 panel.

This is the plotting code of the former Analysis/plot_subset_tbpf_mcs_tracks_demo.py (2 panels),
plot_subset_tbpf_mcs_tracks_1panel_demo.py, plot_subset_tbze_mcs_tracks_demo.py, plot_subset_tbze_mcs_tracks_1panel_demo.py
and plot_subset_tb_mcs_tracks_demo.py. They differ only in the second variable (pixel_dict['var2'] and the 'var2_*'
entries of plot_info: variable, levels, colormap, labels, titles), so one plot function per layout serves all of them.

    plot_map_2panels   Tb (a) and the second field with the MCS shaded by track number (b), side by side or stacked
    plot_map_1panel    Tb and the second field on one axes, MCS outlined in a color per track
    work_for_time_loop reads one pixel-level file, makes the dictionaries and calls the plot function

Called by: runner.run_frames (from mcs_cli.main), serial or on Dask workers.
Calls: mapcore (projection, domain), masks (perimeters), pixelio (subset, figure name), trackdraw (halo).
Importing this module does not select a matplotlib backend; mcs_cli.main and the Dask workers select 'agg'.
"""
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import warnings
warnings.filterwarnings("ignore")
from pyflextrkr.visualization.mapcore import (
    get_projections, set_map_domain, add_map_features, projection_key, norm_lon, in_map_extent, fit_colorbars_to_maps, keep_colorbar_text_inside,
    lat_labels_on_boundary, geodesic_circle,
)
from pyflextrkr.visualization.masks import make_dilation_structure, make_perimeter_structure, label_perimeter
from pyflextrkr.visualization.pixelio import get_data_extent, find_subset_window, make_figure_name
from pyflextrkr.visualization.trackdraw import halo_effects, track_number_levels

# Grid layout of the 2-panel figure (figure_layout below gives the numbers to the figure and to the figure size of mcs_cli).
# 'left_right': left + right panel, colorbars below; 'top_bottom': top + bottom panel, colorbars to the right
GRID_2PANELS = {
    'left_right': dict(
        gridspec=dict(nrows=2, ncols=2, height_ratios=[1, 0.02], width_ratios=[1, 1]),
        update=dict(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.15, hspace=0.15),
    ),
    'top_bottom': dict(
        gridspec=dict(nrows=2, ncols=2, height_ratios=[0.4, 0.4], width_ratios=[1, 0.02]),
        update=dict(left=0.05, right=0.93, bottom=0.05, top=0.9, wspace=0.03, hspace=0.2),
    ),
}
# Grid layout of the 1-panel figure: map + 2 colorbars (with a second field), or map + 1 colorbar (Tb only)
GRID_1PANEL = dict(
    gridspec=dict(nrows=1, ncols=2, height_ratios=[1], width_ratios=[1, 0.1]),
    update=dict(wspace=0.05, left=0.05, right=0.95, top=0.9, bottom=0.08),
)
GRID_1PANEL_TBONLY = dict(
    gridspec=dict(nrows=1, ncols=2, height_ratios=[1], width_ratios=[1, 0.03]),
    update=dict(wspace=0.05, hspace=0.05, left=0.1, right=0.9, top=0.92, bottom=0.08),
)
# Thickness of a colorbar as a fraction of the map width (colorbars at the right of the map) or the map height (colorbars below it),
# and the gap between the 2 colorbars of a 1-panel map as a fraction of the map width
CBAR_THICKNESS = dict(two_panels=0.02, one_panel=1 / 70, tb_only=0.03)
CBAR_SPACING = 5 / 70

def figure_layout(panels, has_var2, layout='top_bottom', cbar_thickness=None, cbar_spacing=None, grid_top=None,
                  margins_in=None, height=None):
    """
    The GridSpec numbers of the figure with the map(s) and the colorbar(s). The figure (plot_map_2panels, plot_map_1panel)
    and the figure size that follows from the shape of the map (mcs_cli.main) both use this, so they agree.

    Args:
        panels: 1 or 2
        has_var2: 1 panel: with a second field and its own colorbar (False: Tb only)
        layout: 2 panels: 'top_bottom' (colorbars at the right of the maps) or 'left_right' (colorbars below)
        cbar_thickness: thickness of a colorbar as a fraction of the map width (bars at the right) or height (bars below);
            None: CBAR_THICKNESS (0.02 for 2 panels, 1/70 for 1 panel with 2 colorbars, 0.03 for Tb only)
        cbar_spacing: 1 panel with 2 colorbars: the gap between them as a fraction of the map width (None: CBAR_SPACING)
        grid_top: top of the 1-panel map in the figure (None: GRID_1PANEL)
        margins_in, height: margins above and below the maps in inches (mapcore.figure_margins) and the height of the figure
            in inches, for a short figure (None: the margins of the layout, fractions of the height)

    Returns:
        dict(gridspec, update, cb_gridspec): the arguments of gridspec.GridSpec, of its update, and of the
        GridSpecFromSubplotSpec that holds the 2 colorbars of a 1-panel map (None for the other figures)
    """
    cb_gridspec = None
    if panels == 2:
        base = GRID_2PANELS[layout]
        gridspec = dict(base['gridspec'])
        update = dict(base['update'])
        t = CBAR_THICKNESS['two_panels'] if cbar_thickness is None else cbar_thickness
        if layout == 'left_right':
            gridspec['height_ratios'] = [1, t]
        else:
            gridspec['width_ratios'] = [1, t]
    elif has_var2:
        t = CBAR_THICKNESS['one_panel'] if cbar_thickness is None else cbar_thickness
        s = CBAR_SPACING if cbar_spacing is None else cbar_spacing
        update = dict(GRID_1PANEL['update'])
        if grid_top is not None:
            update['top'] = grid_top
        # the 2 colorbars side by side in the second column, thickness t each, gap s (fractions of the map width)
        gridspec = {**GRID_1PANEL['gridspec'], 'width_ratios': [1, 2 * t + s]}
        cb_gridspec = dict(nrows=1, ncols=2, height_ratios=[1], width_ratios=[1, 1], wspace=s / t)
    else:
        t = CBAR_THICKNESS['tb_only'] if cbar_thickness is None else cbar_thickness
        gridspec = {**GRID_1PANEL_TBONLY['gridspec'], 'width_ratios': [1, t]}
        update = dict(GRID_1PANEL_TBONLY['update'])
    if margins_in is not None:
        update['top'] = 1.0 - margins_in[0] / height
        update['bottom'] = margins_in[1] / height
    return dict(gridspec=gridspec, update=update, cb_gridspec=cb_gridspec)

#-----------------------------------------------------------------------
def plot_map_2panels(pixel_dict, plot_info, map_info, track_dict):
    """
    Plot 2 panels with Tb (a), the second field and the MCS tracks (b) snapshot.

    Args:
        pixel_dict: dictionary
            Dictionary containing pixel data variables:
                lon: longitude array
                lat: latitude array
                tb: brightness temperature array
                var2: second field array (precipitation or radar reflectivity)
                tracknumber: track number mask array
                tracknumber_perim: track perimeter array
                pixel_bt: pixel base time
        plot_info: dictionary
            Dictionary containing plotting setup variables:
                levels: dict with 'tb_levels', 'var2_levels', 'tn_levels'
                cmaps: dict with 'tb_cmap', 'var2_cmap', 'tn_cmap'
                titles: dict with 'tb_title', 'var2_title'
                cblabels: dict with 'tb_label', 'var2_label'
                cbticks: dict with 'tb_ticks', 'var2_ticks'
                fontsize: font size for labels
                marker_size: marker size for tracks
                tracknumber_fontsize: font size for track numbers
                trackpath_linewidth: line width for track paths
                pfdiam_linewidth: line width for PF diameter circles
                trackpath_color: color for track paths
                mcsperim_color: color for MCS perimeter (legend)
                pfdiam_color: color for PF diameter circles
                pfdiam_scale: scaling factor for PF diameter
                map_resolution: resolution for map features
                suptitle: figure title
                figname: output figure filename
                figsize: figure size tuple
                dpi: figure DPI
                panel_layout: 'top_bottom' (panel (a) above panel (b)) or 'left_right'
              optional (the defaults are the look of the original scripts):
                trackpath_linestyle, trackpath_alpha: line style and transparency of the track paths
                track_marker, track_marker_color: symbol at the initiation location (default 'o', the track color)
                tracknumber_color, tracknumber_weight, tracknumber_alpha: track numbers
                show_tracknumbers, show_tracknumbers2: write the track numbers on panel (a) / (b) (default True; (b) as (a))
                map_aspect: 'equal' (the map keeps its shape, the colorbars follow it) or 'auto' (default: the map fills its panel)
                cbar_thickness: thickness of the colorbars as a fraction of the map width / height (default: CBAR_THICKNESS)
                cbar_spacing: 1-panel map: gap between the 2 colorbars as a fraction of the map width (default CBAR_SPACING, widened if the text needs room)
                margins_in: (top, bottom) margins in inches of a short figure (mapcore.figure_margins; default: the fractions of the layout)
                track_halo_color, track_halo_width, track_halo_alpha: halo around paths, symbols and numbers
                perim_color, perim_alpha: draw the MCS outline on (a) in this color instead of 'cool_r'
                tn_alpha: transparency of the track-number shading in (b)
                var2_mask_le: mask the second field where it is <= the lowest level (default: < lowest level)
                show_pfdiam: draw the PF diameter circles (default True; only where the track has a PF)
                show_legend, legend_loc: the legends
        map_info: dictionary
            Dictionary containing map boundary info:
                map_extent: [lonmin, lonmax, latmin, latmax]
                lonv: longitude tick values (optional)
                latv: latitude tick values (optional)
                draw_land: boolean to draw coastlines (optional, default=True)
                draw_border: boolean to draw country borders (optional)
                draw_state: boolean to draw state borders (optional)
                draw_river: boolean to draw rivers (optional)
                map_feature_styles: color, linewidth and alpha of each map feature, from mapcore.map_feature_styles (optional,
                    default: gray, 0.8 points, opaque)
                box_lon: longitude values for box overlay (optional)
                box_lat: latitude values for box overlay (optional)
                projection, central_longitude, central_latitude, standard_parallels: map projection (optional,
                    default PlateCarree, see mapcore.get_projections)
                lat_range: (lat_min, lat_max) to crop the latitude band of the map (optional)
                grid_linewidth: line width of the gridlines (optional, default 0.5)
                grid_alpha: transparency of the gridlines (optional, default opaque)
                map_extent_xy: [xmin, xmax, ymin, ymax] domain in the coordinates of the projection (optional, mapcore.data_extent_xy)
        track_dict: dictionary
            Dictionary containing tracking data variables:
                ntracks: number of tracks
                lifetime: track lifetime array
                track_bt: track base time array
                track_ccs_lon: track CCS longitude array
                track_ccs_lat: track CCS latitude array
                track_pf_lon: track PF longitude array
                track_pf_lat: track PF latitude array
                track_pf_diam: track PF diameter array
                dt_thres: time threshold for track plotting
                time_res: time resolution in hours

    Returns:
        fig: object
            Figure handle.
    """

    # Get pixel data from dictionary
    lon = pixel_dict['lon']
    lat = pixel_dict['lat']
    tb = pixel_dict['tb']
    var2 = pixel_dict['var2']
    tracknumber = pixel_dict['tracknumber']
    tn_perim = pixel_dict['tracknumber_perim']
    pixel_bt = pixel_dict['pixel_bt']
    # Get track data from dictionary
    ntracks = track_dict['ntracks']
    lifetime = track_dict['lifetime']
    track_bt = track_dict['track_bt']
    track_ccs_lon = track_dict['track_ccs_lon']
    track_ccs_lat = track_dict['track_ccs_lat']
    track_pf_lon = track_dict['track_pf_lon']
    track_pf_lat = track_dict['track_pf_lat']
    track_pf_diam = track_dict['track_pf_diam']
    dt_thres = track_dict['dt_thres']
    time_res = track_dict['time_res']
    # Get plot info from dictionary
    levels = plot_info['levels']
    cmaps = plot_info['cmaps']
    titles = plot_info['titles']
    cblabels = plot_info['cblabels']
    cbticks = plot_info['cbticks']
    fontsize = plot_info['fontsize']
    marker_size = plot_info['marker_size']
    tracknumber_fontsize = plot_info['tracknumber_fontsize']
    trackpath_linewidth = plot_info['trackpath_linewidth']
    pfdiam_linewidth = plot_info['pfdiam_linewidth']
    trackpath_color = plot_info['trackpath_color']
    mcsperim_color = plot_info['mcsperim_color']
    pfdiam_color = plot_info['pfdiam_color']
    pfdiam_scale = plot_info['pfdiam_scale']
    map_resolution = plot_info['map_resolution']
    # timestr = plot_info['timestr']
    suptitle = plot_info['suptitle']
    figname = plot_info['figname']
    figsize = plot_info['figsize']
    dpi = plot_info['dpi']
    panel_layout = plot_info['panel_layout']
    # Track style (the defaults are the look of the original scripts)
    trackpath_linestyle = plot_info.get('trackpath_linestyle', '-')
    trackpath_alpha = plot_info.get('trackpath_alpha', 1.0)
    track_marker = plot_info.get('track_marker', 'o')
    track_marker_color = plot_info.get('track_marker_color') or trackpath_color
    tracknumber_color = plot_info.get('tracknumber_color', 'k')
    tracknumber_weight = plot_info.get('tracknumber_weight', 'bold')
    tracknumber_alpha = plot_info.get('tracknumber_alpha', 1.0)
    show_tracknumbers = plot_info.get('show_tracknumbers', True)
    show_tracknumbers2 = plot_info.get('show_tracknumbers2', show_tracknumbers)     # the second panel (b)
    show_pfdiam = plot_info.get('show_pfdiam', True)
    show_legend = plot_info.get('show_legend', True)
    legend_loc = plot_info.get('legend_loc', 'lower right')
    perim_color = plot_info.get('perim_color', None)
    perim_alpha = plot_info.get('perim_alpha', 1.0)
    tn_alpha = plot_info.get('tn_alpha', 0.7)
    tn_norm = plot_info.get('tn_norm', 'continuous')
    var2_mask_le = plot_info.get('var2_mask_le', False)
    # Details of the look that differ between the products (see the product blocks of mcs_cli.main)
    grid_zorder = plot_info.get('grid_zorder', None)    # None: the gridlines are drawn after the data (above them)
    cb_minorticks_off = plot_info.get('cb_minorticks_off', True)
    # Halo around the track paths, symbols and numbers (path effects; None: no halo)
    halo_path, halo_marker, halo_text = halo_effects(
        plot_info.get('track_halo_color'), plot_info.get('track_halo_width', 2.0),
        plot_info.get('track_halo_alpha', 1.0), trackpath_linewidth,
    )
    # Map domain, lat/lon ticks, background map features
    map_extent = map_info['map_extent']
    lonv = map_info.get('lonv', None)
    latv = map_info.get('latv', None)
    draw_land = map_info.get('draw_land', True)
    draw_border = map_info.get('draw_border', False)
    draw_state = map_info.get('draw_state', False)
    draw_river = map_info.get('draw_river', False)
    draw_features = dict(land=draw_land, border=draw_border, state=draw_state, river=draw_river)
    feature_styles = map_info.get('map_feature_styles')    # from the command line (None: the default style)
    box_lon = map_info.get('box_lon', None)
    box_lat = map_info.get('box_lat', None)
    projection = map_info.get('projection', 'PlateCarree')
    lat_range = map_info.get('lat_range', None)
    grid_linewidth = map_info.get('grid_linewidth', 0.5)
    grid_alpha = map_info.get('grid_alpha', None)
    is_platecarree = projection_key(projection) == 'platecarree'
    extent_xy = map_info.get('map_extent_xy', None)       # the domain in projected coordinates (the data, no --extent)
    map_aspect = plot_info.get('map_aspect', 'auto')     # 'equal': the map keeps its shape, 'auto': it fills its cell

    # Time difference matching pixel-time and track time
    dt_match = 1  # [min]

    # Marker style for tracks
    marker_style = dict(edgecolor=track_marker_color, facecolor=track_marker_color, linestyle='-', marker=track_marker)

    # Set up map projection (PlateCarree with the central longitude from the extent, or the projection that was asked for)
    proj, data_proj, _ext_lon0, _ext_lon1 = get_projections(
        map_extent, projection, map_info.get('central_longitude'), map_info.get('central_latitude'),
        map_info.get('standard_parallels'),
    )

    # Set up figure
    mpl.rcParams['font.size'] = fontsize
    # mpl.rcParams['font.family'] = 'Helvetica'
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='w')
    layout = figure_layout(2, True, panel_layout, plot_info.get('cbar_thickness'), margins_in=plot_info.get('margins_in'), height=figsize[1])
    gs = gridspec.GridSpec(**layout['gridspec'])
    gs.update(**layout['update'])
    if panel_layout == 'left_right':
        # left + right panel, colorbars below
        ax2 = plt.subplot(gs[0,1], projection=proj)
        cax1 = plt.subplot(gs[1,0])
        cb_orientation = 'horizontal'
    else:
        # top + bottom panel, colorbars to the right
        ax2 = plt.subplot(gs[1,0], projection=proj)
        cax1 = plt.subplot(gs[0,1])
        cb_orientation = 'vertical'
    # Figure title: time
    fig.suptitle(suptitle, fontsize=fontsize*1.5, fontweight='bold', y=0.99)

    # Create separate rectangle patches for each axis
    if box_lon is not None and box_lat is not None:
        rect1 = mpatches.Rectangle(
            xy=(min(box_lon), min(box_lat)),
            width=max(box_lon) - min(box_lon),
            height=max(box_lat) - min(box_lat),
            linewidth=2,
            edgecolor='firebrick',
            transform=data_proj,  # Use data_proj instead of proj for consistency
            zorder=4,
            facecolor='none'  # Make it transparent
        )
        rect2 = mpatches.Rectangle(
            xy=(min(box_lon), min(box_lat)),
            width=max(box_lon) - min(box_lon),
            height=max(box_lat) - min(box_lat),
            linewidth=2,
            edgecolor='firebrick',
            transform=data_proj,  # Use data_proj instead of proj for consistency
            zorder=4,
            facecolor='none'  # Make it transparent
        )
    else:
        rect1 = None
        rect2 = None

    #################################################################
    # Tb Panel
    ax1 = plt.subplot(gs[0,0], projection=proj)
    set_map_domain(ax1, proj, map_extent, _ext_lon0, _ext_lon1, lat_range, extent_xy)
    add_map_features(ax1, draw_features, feature_styles, map_resolution)
    if rect1 is not None:
        ax1.add_patch(rect1)
    if is_platecarree and map_aspect == 'auto':
        # the map fills its cell of the figure (it is stretched if the cell has another shape: the old look)
        ax1.set_aspect('auto', adjustable=None)
    else:
        # the map keeps its shape (the axes shrink to fit its cell of the figure); the colorbars follow it (fit_colorbars_to_maps)
        ax1.set_aspect('equal', adjustable='box')
    ax1.set_title(titles['tb_title'], loc='left')

    cmap = plt.get_cmap(cmaps['tb_cmap'])
    norm = mpl.colors.BoundaryNorm(levels['tb_levels'], ncolors=cmap.N, clip=True)
    tb_masked = np.ma.masked_where((np.isnan(tb)), tb)
    cf1 = ax1.pcolormesh(lon, lat, tb_masked, norm=norm, cmap=cmap, transform=data_proj, zorder=2)
    # Overplot cloudtracknumber boundary
    # Overlay boundary of cloudtracknumber on Tb
    Tn = np.ma.masked_where(tn_perim == 0, tn_perim)
    Tn[Tn > 0] = 10
    if perim_color is None:
        tn1 = ax1.pcolormesh(lon, lat, Tn, cmap='cool_r', transform=data_proj, zorder=3)
    else:
        # Outline in a color of your choice
        tn1 = ax1.pcolormesh(lon, lat, Tn, cmap=mpl.colors.ListedColormap([perim_color]), vmin=0, vmax=1,
                             transform=data_proj, zorder=3, alpha=perim_alpha)
    # Tb Colorbar
    cb1 = plt.colorbar(cf1, cax=cax1, label=cblabels['tb_label'], ticks=cbticks['tb_ticks'],
                       extend='both', orientation=cb_orientation)
    # Turn off minor ticks on colorbar
    if cb_minorticks_off:
        cb1.ax.minorticks_off()

    # Add gridlines after plotting data for ax1
    gl = ax1.gridlines(crs=data_proj, draw_labels=True, linestyle='--', linewidth=grid_linewidth, alpha=grid_alpha, zorder=4)
    if grid_zorder is not None:
        gl.set_zorder(grid_zorder)
    gl.right_labels = False
    gl.top_labels = False
    if not is_platecarree:
        # labels outside the map, not inline and not rotated
        gl.x_inline = False
        gl.y_inline = False
        gl.rotate_labels = False
        gl.geo_labels = False
    if (lonv is not None) & (latv is not None):
        gl.xlocator = mpl.ticker.FixedLocator(lonv)
        gl.ylocator = mpl.ticker.FixedLocator(latv)
    lat_labels_on_boundary(ax1, proj, gl, map_extent, lat_range, latv if (lonv is not None) & (latv is not None) else None)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    if is_platecarree:
        ax1.xaxis.set_major_formatter(lon_formatter)
        ax1.yaxis.set_major_formatter(lat_formatter)

    #################################################################
    # Second Field Panel (precipitation, reflectivity)
    set_map_domain(ax2, proj, map_extent, _ext_lon0, _ext_lon1, lat_range, extent_xy)
    add_map_features(ax2, draw_features, feature_styles, map_resolution)
    if rect2 is not None:
        ax2.add_patch(rect2)
    if is_platecarree and map_aspect == 'auto':
        # the map fills its cell of the figure (it is stretched if the cell has another shape: the old look)
        ax2.set_aspect('auto', adjustable=None)
    else:
        # the map keeps its shape (the axes shrink to fit its cell of the figure); the colorbars follow it (fit_colorbars_to_maps)
        ax2.set_aspect('equal', adjustable='box')
    ax2.set_title(titles['var2_title'], loc='left')

    # MCS track number mask
    cmap = plt.get_cmap(cmaps['tn_cmap'])
    # If only 1 track, add a number to make at least 2 color levels
    if len(levels['tn_levels']) == 1:
        levels['tn_levels'] = np.append(levels['tn_levels'], levels['tn_levels']+1)
    norm = mpl.colors.BoundaryNorm(levels['tn_levels'], ncolors=cmap.N, clip=True)
    tracknumber_masked = np.ma.masked_invalid(tracknumber)
    if tn_norm == 'boundary':
        # one color per level (tn_norm='boundary')
        cm1 = ax2.pcolormesh(lon, lat, tracknumber_masked, norm=norm, cmap=cmap, transform=data_proj, zorder=2, alpha=tn_alpha)
    else:
        cm1 = ax2.pcolormesh(lon, lat, tracknumber_masked, vmin=min(levels['tn_levels']), vmax=max(levels['tn_levels']),
                             cmap=cmap, transform=data_proj, zorder=2, alpha=tn_alpha)

    # Second field (precipitation, reflectivity)
    cmap = plt.get_cmap(cmaps['var2_cmap'])
    norm = mpl.colors.BoundaryNorm(levels['var2_levels'], ncolors=cmap.N, clip=True)
    if var2_mask_le:
        var2_masked = np.ma.masked_where(((var2 <= min(levels['var2_levels']))), var2)
    else:
        var2_masked = np.ma.masked_where(((var2 < min(levels['var2_levels']))), var2)
    cf2 = ax2.pcolormesh(lon, lat, var2_masked, norm=norm, cmap=cmap, transform=data_proj, zorder=2)
    # Colorbar
    cax2 = plt.subplot(gs[1,1])
    cb2 = plt.colorbar(cf2, cax=cax2, label=cblabels['var2_label'], ticks=cbticks['var2_ticks'],
                       extend='both', orientation=cb_orientation)
    # Turn off minor ticks on colorbar
    if cb_minorticks_off:
        cb2.ax.minorticks_off()

    # Add gridlines after plotting data for ax2
    gl = ax2.gridlines(crs=data_proj, draw_labels=True, linestyle='--', linewidth=grid_linewidth, alpha=grid_alpha, zorder=3)
    if grid_zorder is not None:
        gl.set_zorder(grid_zorder)
    gl.right_labels = False
    gl.top_labels = False
    if not is_platecarree:
        gl.x_inline = False
        gl.y_inline = False
        gl.rotate_labels = False
        gl.geo_labels = False
    if (lonv is not None) & (latv is not None):
        gl.xlocator = mpl.ticker.FixedLocator(lonv)
        gl.ylocator = mpl.ticker.FixedLocator(latv)
    lat_labels_on_boundary(ax2, proj, gl, map_extent, lat_range, latv if (lonv is not None) & (latv is not None) else None)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    if is_platecarree:
        ax2.xaxis.set_major_formatter(lon_formatter)
        ax2.yaxis.set_major_formatter(lat_formatter)

    #################################################################
    # Plot track centroids and paths
    for itrack in range(0, ntracks):
        # Get duration of the track
        ilifetime = lifetime.data[itrack]
        itracknum = lifetime.tracks.data[itrack]+1
        idur = (ilifetime / time_res).astype(int)
        idiam = track_pf_diam.data[itrack,:idur]
        # Get basetime of the track and the track end time
        ibt = track_bt.data[itrack,:idur]
        ibt_end = np.nanmax(ibt)
        # Compute time difference between current pixel-level data time and the end time of the track
        idt_end = (pixel_bt - ibt_end).astype('timedelta64[h]')
        # Proceed if time difference is <= threshold
        # This means for tracks that end longer than the time threshold are not plotted
        if (idt_end <= dt_thres):
            # Find times in track data <= current pixel-level file time
            idx_cut = np.where(ibt <= pixel_bt)[0]
            idur_cut = len(idx_cut)
            if (idur_cut > 0):
                # CCS track path
                color_vals = np.repeat(ilifetime, idur_cut)
                size_vals = np.repeat(marker_size, idur_cut)
                size_vals[0] = marker_size * 2
                cc1 = ax1.plot(track_ccs_lon.data[itrack,idx_cut], track_ccs_lat.data[itrack,idx_cut],
                               lw=trackpath_linewidth, ls=trackpath_linestyle, color=trackpath_color,
                               alpha=trackpath_alpha, path_effects=halo_path, transform=data_proj, zorder=3)
                cc2 = ax2.plot(track_ccs_lon.data[itrack,idx_cut], track_ccs_lat.data[itrack,idx_cut],
                               lw=trackpath_linewidth, ls=trackpath_linestyle, color=trackpath_color,
                               alpha=trackpath_alpha, path_effects=halo_path, transform=data_proj, zorder=3)
                # PF track path
                # pf1 = ax1.plot(track_pf_lon.data[itrack,idx_cut], track_pf_lat.data[itrack,idx_cut],
                #                 lw=trackpath_linewidth, ls='-', color=trackpath_color, transform=data_proj, zorder=3)
                # pf2 = ax2.plot(track_pf_lon.data[itrack,idx_cut], track_pf_lat.data[itrack,idx_cut],
                #                 lw=trackpath_linewidth, ls='-', color=trackpath_color, transform=data_proj, zorder=3)
                # Initiation location
                cl1 = ax1.scatter(track_ccs_lon.data[itrack,0], track_ccs_lat.data[itrack,0], s=marker_size*2,
                                  alpha=trackpath_alpha, path_effects=halo_marker, transform=data_proj, zorder=4,
                                  **marker_style)
                cl2 = ax2.scatter(track_ccs_lon.data[itrack,0], track_ccs_lat.data[itrack,0], s=marker_size*2,
                                  alpha=trackpath_alpha, path_effects=halo_marker, transform=data_proj, zorder=4,
                                  **marker_style)

        # Find the closest time from track times
        idt = np.abs((ibt - pixel_bt).astype('timedelta64[m]'))
        idx_match = np.argmin(idt)
        idt_match = idt[idx_match]
        # Get PF radius from the matched tracks
        _irad = idiam[idx_match] / 2
        _ilon = track_pf_lon.data[itrack,idx_match]
        _ilat = track_pf_lat.data[itrack,idx_match]
        _iccslon = track_ccs_lon.data[itrack,idx_match]
        _iccslat = track_ccs_lat.data[itrack,idx_match]
        # Proceed if time difference is < dt_match
        if (idt_match < dt_match):
            # Plot PF diameter circle - check that all values are valid (not NaN)
            if show_pfdiam and ~np.isnan(_irad) and ~np.isnan(_ilon) and ~np.isnan(_ilat):
                # Positions of global tracks are not limited to -180~+180 (periodic domain)
                if not is_platecarree:
                    _ilon = norm_lon(_ilon)
                # (a geodesic circle drawn with the transform of the data, not ax.tissot: see mapcore.geodesic_circle)
                ax2.plot(*geodesic_circle(_ilon, _ilat, _irad*pfdiam_scale), color=pfdiam_color, lw=pfdiam_linewidth,
                         transform=data_proj, zorder=3)
            # Overplot tracknumbers at current frame
            # Normalize to 0-360 for dateline-safe lon check (in_map_extent)
            if show_tracknumbers and in_map_extent(_iccslon, _iccslat, map_extent):
                ax1.text(_iccslon+0.05, _iccslat+0.05, f'{itracknum:.0f}', color=tracknumber_color,
                         size=tracknumber_fontsize, ha='left', va='center', weight=tracknumber_weight,
                         alpha=tracknumber_alpha, path_effects=halo_text, clip_on=not is_platecarree, transform=data_proj, zorder=5)
            if show_tracknumbers2 and in_map_extent(_iccslon, _iccslat, map_extent):
                ax2.text(_iccslon+0.05, _iccslat+0.05, f'{itracknum:.0f}', color=tracknumber_color,
                         size=tracknumber_fontsize, ha='left', va='center', weight=tracknumber_weight,
                         alpha=tracknumber_alpha, path_effects=halo_text, clip_on=not is_platecarree, transform=data_proj, zorder=5)

    # Custom legend for track paths
    if perim_color is not None:
        mcsperim_color = perim_color
    legend_elements1 = [
        mpl.lines.Line2D([0], [0], color=trackpath_color, marker=track_marker, lw=trackpath_linewidth, label='MCS Tracks'),
        mpl.lines.Line2D([0], [0], marker='o', lw=0, markerfacecolor='None',
                         markeredgecolor=mcsperim_color, markersize=12, label='MCS Mask'),
    ]
    legend_elements2 = [
        mpl.lines.Line2D([0], [0], color=trackpath_color, marker=track_marker, lw=trackpath_linewidth, label='MCS Tracks'),
    ]
    if show_pfdiam:
        legend_elements2.append(mpl.lines.Line2D([0], [0], marker='o', lw=0, markerfacecolor='None',
                                                 markeredgecolor=pfdiam_color, markersize=12, label=f'PF Diam (x{pfdiam_scale:.1f})'))
    if show_legend:
        ax1.legend(handles=legend_elements1, loc=legend_loc)
        ax2.legend(handles=legend_elements2, loc=legend_loc)

    # Thread-safe figure output
    canvas = FigureCanvas(fig)
    # The maps keep their shape: the colorbars have the extent of their map and their text stays inside the figure
    if not (is_platecarree and map_aspect == 'auto'):
        fit_colorbars_to_maps([(ax1, [cax1]), (ax2, [cax2])], cb_orientation)
        keep_colorbar_text_inside(fig, [(ax1, [cax1]), (ax2, [cax2])], cb_orientation)
    fig.savefig(figname)

    return fig

#-----------------------------------------------------------------------
def plot_map_1panel(pixel_dict, plot_info, map_info, track_dict):
    """
    Plot single panel with Tb, the second field (precipitation or reflectivity; none for Tb only) and MCS tracks
    snapshot. The MCS are outlined with a color per track.

    Args:
        pixel_dict, plot_info, map_info, track_dict: dictionaries as in plot_map_2panels, with these differences:
            pixel_dict['var2'] is None for Tb only;
            plot_info: tb_alpha, var2_alpha: transparency of the Tb and second field layers;
                titles: dict with 'tb_title' only; no panel_layout; levels/cmaps/cblabels/cbticks have no 'var2_*' for Tb only
                tn_cmap (cmaps) is the colormap of the MCS outlines (one color per track number)
                perim_color, perim_alpha: draw all outlines in this color instead
                var2_zorder: zorder of the second field layer (default 2)

    Returns:
        fig: object
            Figure handle.
    """

    # Get pixel data from dictionary
    lon = pixel_dict['lon']
    lat = pixel_dict['lat']
    tb = pixel_dict['tb']
    var2 = pixel_dict['var2']
    tracknumber = pixel_dict['tracknumber']
    tn_perim = pixel_dict['tracknumber_perim']
    pixel_bt = pixel_dict['pixel_bt']
    # Get track data from dictionary
    ntracks = track_dict['ntracks']
    lifetime = track_dict['lifetime']
    track_bt = track_dict['track_bt']
    track_ccs_lon = track_dict['track_ccs_lon']
    track_ccs_lat = track_dict['track_ccs_lat']
    track_pf_lon = track_dict['track_pf_lon']
    track_pf_lat = track_dict['track_pf_lat']
    track_pf_diam = track_dict['track_pf_diam']
    dt_thres = track_dict['dt_thres']
    time_res = track_dict['time_res']
    # Get plot info from dictionary
    levels = plot_info['levels']
    cmaps = plot_info['cmaps']
    tb_alpha = plot_info['tb_alpha']
    var2_alpha = plot_info.get('var2_alpha', 1.0)
    titles = plot_info['titles']
    cblabels = plot_info['cblabels']
    cbticks = plot_info['cbticks']
    fontsize = plot_info['fontsize']
    marker_size = plot_info['marker_size']
    tracknumber_fontsize = plot_info['tracknumber_fontsize']
    trackpath_linewidth = plot_info['trackpath_linewidth']
    trackpath_color = plot_info['trackpath_color']
    mcsperim_color = plot_info['mcsperim_color']
    map_resolution = plot_info['map_resolution']
    # timestr = plot_info['timestr']
    suptitle = plot_info['suptitle']
    figname = plot_info['figname']
    figsize = plot_info['figsize']
    dpi = plot_info['dpi']
    # Track style (the defaults are the look of the original scripts)
    trackpath_linestyle = plot_info.get('trackpath_linestyle', '-')
    trackpath_alpha = plot_info.get('trackpath_alpha', 1.0)
    track_marker = plot_info.get('track_marker', 'o')
    track_marker_color = plot_info.get('track_marker_color') or trackpath_color
    tracknumber_color = plot_info.get('tracknumber_color', 'k')
    tracknumber_weight = plot_info.get('tracknumber_weight', 'bold')
    tracknumber_alpha = plot_info.get('tracknumber_alpha', 1.0)
    show_tracknumbers = plot_info.get('show_tracknumbers', True)
    show_legend = plot_info.get('show_legend', True)
    legend_loc = plot_info.get('legend_loc', 'lower right')
    perim_color = plot_info.get('perim_color', None)
    perim_alpha = plot_info.get('perim_alpha', 1.0)
    var2_zorder = plot_info.get('var2_zorder', 2)
    var2_mask_le = plot_info.get('var2_mask_le', False)
    # Details of the look that differ between the products (see the product blocks of mcs_cli.main)
    perim_zorder = plot_info.get('perim_zorder', 4)     # zorder of the MCS outlines
    grid_zorder = plot_info.get('grid_zorder', None)    # None: the gridlines are drawn after the data (above them)
    cb_minorticks_off = plot_info.get('cb_minorticks_off', True)
    title_style = plot_info.get('title_style', 'bold')
    # Halo around the track paths, symbols and numbers (path effects; None: no halo)
    halo_path, halo_marker, halo_text = halo_effects(
        plot_info.get('track_halo_color'), plot_info.get('track_halo_width', 2.0),
        plot_info.get('track_halo_alpha', 1.0), trackpath_linewidth,
    )
    # Map domain, lat/lon ticks, background map features
    map_extent = map_info['map_extent']
    lonv = map_info.get('lonv', None)
    latv = map_info.get('latv', None)
    draw_land = map_info.get('draw_land', True)
    draw_border = map_info.get('draw_border', False)
    draw_state = map_info.get('draw_state', False)
    draw_river = map_info.get('draw_river', False)
    draw_features = dict(land=draw_land, border=draw_border, state=draw_state, river=draw_river)
    feature_styles = map_info.get('map_feature_styles')    # from the command line (None: the default style)
    box_lon = map_info.get('box_lon', None)
    box_lat = map_info.get('box_lat', None)
    projection = map_info.get('projection', 'PlateCarree')
    lat_range = map_info.get('lat_range', None)
    grid_linewidth = map_info.get('grid_linewidth', 0.5)
    grid_alpha = map_info.get('grid_alpha', None)
    is_platecarree = projection_key(projection) == 'platecarree'
    extent_xy = map_info.get('map_extent_xy', None)       # the domain in projected coordinates (the data, no --extent)
    map_aspect = plot_info.get('map_aspect', 'auto')     # 'equal': the map keeps its shape, 'auto': it fills its cell

    # Time difference matching pixel-time and track time
    dt_match = 1  # [min]

    # Marker style for tracks
    marker_style = dict(edgecolor=track_marker_color, facecolor=track_marker_color, linestyle='-', marker=track_marker)

    # Set up map projection (PlateCarree with the central longitude from the extent, or the projection that was asked for)
    proj, data_proj, _ext_lon0, _ext_lon1 = get_projections(
        map_extent, projection, map_info.get('central_longitude'), map_info.get('central_latitude'),
        map_info.get('standard_parallels'),
    )

    # Set up figure
    mpl.rcParams['font.size'] = fontsize
    # mpl.rcParams['font.family'] = 'Helvetica'
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='w')

    layout = figure_layout(1, var2 is not None, None, plot_info.get('cbar_thickness'), plot_info.get('cbar_spacing'),
                           plot_info.get('grid_top'), plot_info.get('margins_in'), figsize[1])
    gs = gridspec.GridSpec(**layout['gridspec'])
    gs.update(**layout['update'])
    if var2 is not None:
        # Left: the map, right: the 2 colorbars side by side (GridSpecFromSubplotSpec)
        gs_cb = gridspec.GridSpecFromSubplotSpec(subplot_spec=gs[1], **layout['cb_gridspec'])
        ax1 = plt.subplot(gs[0], projection=proj)
        cax1 = plt.subplot(gs_cb[0])
        cax2 = plt.subplot(gs_cb[1])
    else:
        # Tb only: map + one colorbar
        ax1 = plt.subplot(gs[0], projection=proj)
        cax1 = plt.subplot(gs[1])
    # Figure title: time
    if title_style == 'plain':
        fig.text(0.5, 0.96, suptitle, fontsize=fontsize*1.4, ha='center')
    else:
        fig.suptitle(suptitle, fontsize=fontsize*1.5, fontweight='bold', y=0.99)

    # Create rectangle patch
    if box_lon is not None and box_lat is not None:
        rect1 = mpatches.Rectangle(
            xy=(min(box_lon), min(box_lat)),
            width=max(box_lon) - min(box_lon),
            height=max(box_lat) - min(box_lat),
            linewidth=2,
            edgecolor='firebrick',
            transform=data_proj,  # Use data_proj instead of proj for consistency
            zorder=4,
            facecolor='none'  # Make it transparent
        )
    else:
        rect1 = None

    #################################################################
    # Tb Panel
    set_map_domain(ax1, proj, map_extent, _ext_lon0, _ext_lon1, lat_range, extent_xy)
    add_map_features(ax1, draw_features, feature_styles, map_resolution)
    if rect1 is not None:
        ax1.add_patch(rect1)
    if is_platecarree and map_aspect == 'auto':
        # the map fills its cell of the figure (it is stretched if the cell has another shape: the old look)
        ax1.set_aspect('auto', adjustable=None)
    else:
        # the map keeps its shape (the axes shrink to fit its cell of the figure); the colorbars follow it (fit_colorbars_to_maps)
        ax1.set_aspect('equal', adjustable='box')
    ax1.set_title(titles['tb_title'], loc='left')

    # Tb
    cmap = plt.get_cmap(cmaps['tb_cmap'])
    norm = mpl.colors.BoundaryNorm(levels['tb_levels'], ncolors=cmap.N, clip=True)
    tb_masked = np.ma.masked_where((np.isnan(tb)), tb)
    cf1 = ax1.pcolormesh(lon, lat, tb_masked, norm=norm, cmap=cmap, transform=data_proj, zorder=2, alpha=tb_alpha)
    # Overplot cloudtracknumber boundary
    cmap = plt.get_cmap(cmaps['tn_cmap'])
    Tn = np.ma.masked_where(tn_perim == 0, tn_perim)
    if perim_color is None:
        tn1 = ax1.pcolormesh(lon, lat, Tn, vmin=min(levels['tn_levels']), vmax=max(levels['tn_levels']),
                             cmap=cmap, transform=data_proj, zorder=perim_zorder, alpha=1)
    else:
        # All outlines in a color of your choice
        tn1 = ax1.pcolormesh(lon, lat, np.ma.masked_where(tn_perim == 0, np.ones(tn_perim.shape)), vmin=0, vmax=1,
                             cmap=mpl.colors.ListedColormap([perim_color]), transform=data_proj, zorder=perim_zorder, alpha=perim_alpha)
        mcsperim_color = perim_color
    if var2 is not None:
        # Second field (precipitation, reflectivity)
        cmap = plt.get_cmap(cmaps['var2_cmap'])
        norm = mpl.colors.BoundaryNorm(levels['var2_levels'], ncolors=cmap.N, clip=True)
        if var2_mask_le:
            var2_masked = np.ma.masked_where(((var2 <= min(levels['var2_levels']))), var2)
        else:
            var2_masked = np.ma.masked_where(((var2 < min(levels['var2_levels']))), var2)
        cf2 = ax1.pcolormesh(lon, lat, var2_masked, norm=norm, cmap=cmap, transform=data_proj, zorder=var2_zorder,
                             alpha=var2_alpha)
    # Tb Colorbar
    cb1 = plt.colorbar(cf1, cax=cax1, label=cblabels['tb_label'], ticks=cbticks['tb_ticks'],
                       extend='both', orientation='vertical')
    # Turn off minor ticks on colorbar
    if cb_minorticks_off:
        cb1.ax.minorticks_off()
    if var2 is not None:
        # Second field Colorbar
        cb2 = plt.colorbar(cf2, cax=cax2, label=cblabels['var2_label'], ticks=cbticks['var2_ticks'],
                           extend='both', orientation='vertical')
        if cb_minorticks_off:
            cb2.ax.minorticks_off()

    # Add gridlines after plotting data for ax1
    gl = ax1.gridlines(crs=data_proj, draw_labels=True, linestyle='--', linewidth=grid_linewidth, alpha=grid_alpha, zorder=4)
    if grid_zorder is not None:
        gl.set_zorder(grid_zorder)
    gl.right_labels = False
    gl.top_labels = False
    if not is_platecarree:
        # labels outside the map, not inline and not rotated
        gl.x_inline = False
        gl.y_inline = False
        gl.rotate_labels = False
        gl.geo_labels = False
    if (lonv is not None) & (latv is not None):
        gl.xlocator = mpl.ticker.FixedLocator(lonv)
        gl.ylocator = mpl.ticker.FixedLocator(latv)
    lat_labels_on_boundary(ax1, proj, gl, map_extent, lat_range, latv if (lonv is not None) & (latv is not None) else None)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    if is_platecarree:
        ax1.xaxis.set_major_formatter(lon_formatter)
        ax1.yaxis.set_major_formatter(lat_formatter)

    #################################################################
    # Plot track centroids and paths
    for itrack in range(0, ntracks):
        # Get duration of the track
        ilifetime = lifetime.data[itrack]
        itracknum = lifetime.tracks.data[itrack]+1
        idur = (ilifetime / time_res).astype(int)
        idiam = track_pf_diam.data[itrack,:idur]
        # Get basetime of the track and the track end time
        ibt = track_bt.data[itrack,:idur]
        ibt_end = np.nanmax(ibt)
        # Compute time difference between current pixel-level data time and the end time of the track
        idt_end = (pixel_bt - ibt_end).astype('timedelta64[h]')
        # Proceed if time difference is <= threshold
        # This means for tracks that end longer than the time threshold are not plotted
        if (idt_end <= dt_thres):
            # Find times in track data <= current pixel-level file time
            idx_cut = np.where(ibt <= pixel_bt)[0]
            idur_cut = len(idx_cut)
            if (idur_cut > 0):
                # Track path
                color_vals = np.repeat(ilifetime, idur_cut)
                size_vals = np.repeat(marker_size, idur_cut)
                size_vals[0] = marker_size * 2
                cc1 = ax1.plot(track_ccs_lon.data[itrack,idx_cut], track_ccs_lat.data[itrack,idx_cut],
                               lw=trackpath_linewidth, ls=trackpath_linestyle, color=trackpath_color,
                               alpha=trackpath_alpha, path_effects=halo_path, transform=data_proj, zorder=3)
                # Initiation location
                cl1 = ax1.scatter(track_ccs_lon.data[itrack,0], track_ccs_lat.data[itrack,0], s=marker_size*2,
                                  alpha=trackpath_alpha, path_effects=halo_marker, transform=data_proj, zorder=4,
                                  **marker_style)

        # Find the closest time from track times
        idt = np.abs((ibt - pixel_bt).astype('timedelta64[m]'))
        idx_match = np.argmin(idt)
        idt_match = idt[idx_match]
        # Get CCS center lat/lon from the matched tracks
        _iccslon = track_ccs_lon.data[itrack,idx_match]
        _iccslat = track_ccs_lat.data[itrack,idx_match]
        # Get PF radius from the matched tracks
        _irad = idiam[idx_match] / 2
        _ilon = track_pf_lon.data[itrack,idx_match]
        _ilat = track_pf_lat.data[itrack,idx_match]
        # Proceed if time difference is < dt_match
        if (idt_match < dt_match):
            # Overplot tracknumbers at current frame
            # Normalize to 0-360 for dateline-safe lon check (in_map_extent)
            if show_tracknumbers and in_map_extent(_iccslon, _iccslat, map_extent):
                ax1.text(_iccslon+0.05, _iccslat+0.05, f'{itracknum:.0f}', color=tracknumber_color,
                         size=tracknumber_fontsize, ha='left', va='center', weight=tracknumber_weight,
                         alpha=tracknumber_alpha, path_effects=halo_text, clip_on=not is_platecarree, transform=data_proj, zorder=5)

    # Custom legend for track paths
    legend_elements1 = [
        mpl.lines.Line2D([0], [0], color=trackpath_color, marker=track_marker, lw=trackpath_linewidth, label='MCS Tracks'),
        mpl.lines.Line2D([0], [0], marker='o', lw=0, markerfacecolor='None',
                         markeredgecolor=mcsperim_color, markersize=12, label='MCS Mask'),
    ]
    if show_legend:
        ax1.legend(handles=legend_elements1, loc=legend_loc)

    # Thread-safe figure output
    canvas = FigureCanvas(fig)
    # The map keeps its shape: the colorbars have the extent of the map and their text stays inside the figure
    if not (is_platecarree and map_aspect == 'auto'):
        pairs = [(ax1, [cax1, cax2] if var2 is not None else [cax1])]
        fit_colorbars_to_maps(pairs, 'vertical')
        keep_colorbar_text_inside(fig, pairs, 'vertical', spacing_auto=plot_info.get('cbar_spacing') is None)
    fig.savefig(figname)

    return fig

#-----------------------------------------------------------------------
def work_for_time_loop(datafile, track_dict, map_info, plot_info, config, ifile):
    """
    Work with a pixel-level file.

    Args:
        datafile: string
            Pixel-level file name.
        track_dict: dictionary
            Dictionary containing tracking data variables.
        map_info: dictionary
            Dictionary containing map boundary info.
        plot_info: dictionary
            Dictionary containing plotting setup variables (figdir, figbasename, figname_type, title_prefix,
            tb_varname, var2_varname (None: Tb only), panels (1 or 2), perim_thick [km] and pixel_radius or
            perim_width [pixels], skip_empty, periodic_perimeter, ...).
        config: dictionary
            Dictionary containing config parameters.
        ifile: int
            Index of the file in the series (figure name for figname_type='sequence').

    Returns:
        1: success.
    """

    map_extent = map_info.get('map_extent', None)
    subset = map_info.get('subset', 0)
    perim_thick = plot_info.get('perim_thick')
    perim_width = plot_info.get('perim_width')
    figdir = plot_info.get('figdir')
    figbasename = plot_info.get('figbasename')
    figname_type = plot_info.get('figname_type', 'date_time')
    title_prefix = plot_info.get('title_prefix', "")
    pixel_radius = plot_info.get('pixel_radius')
    tb_varname = plot_info.get('tb_varname', 'tb')
    var2_varname = plot_info.get('var2_varname')
    panels = plot_info['panels']
    skip_empty = plot_info.get('skip_empty', False)
    periodic_perimeter = plot_info.get('periodic_perimeter', 'none')

    # Read pixel-level data
    ds = xr.open_dataset(datafile)
    pixel_bt = ds['time'].data

    # Get map extent from data
    if map_extent is None:
        map_extent = get_data_extent(ds)
        map_info['map_extent'] = map_extent

    # Make dilation structure (larger values make thicker outlines)
    if perim_width is not None:
        # perimeter width in pixels
        dilationstructure = make_perimeter_structure(perim_width)
    else:
        # Make a dilation structure (perimeter width in km)
        dilationstructure = make_dilation_structure(perim_thick, pixel_radius, pixel_radius)

    # Get tracknumbers
    tn = ds['cloudtracknumber'].squeeze()
    # Only plot if there is track in the frame
    if skip_empty and not (np.nanmax(tn) > 0):
        ds.close()
        return 1

    # Tracknumber color levels for MCS masks (limit to 256 to fit in a colormap)
    # Add to plot_info dictionary
    plot_info['levels']['tn_levels'] = track_number_levels(track_dict)

    # Subset pixel data within the map domain
    window = None
    if subset == 1:
        window = find_subset_window(ds, tb_varname, map_extent)
    if window is not None:
        # Subset using index slicing - maintains rectangular structure, no NaNs
        tb_sub = ds[tb_varname].isel(window).squeeze()
        var2_sub = ds[var2_varname].isel(window).squeeze() if var2_varname is not None else None
        tracknumber_sub = ds['cloudtracknumber'].isel(window).squeeze()
        lon_sub = ds['longitude'].isel(window)
        lat_sub = ds['latitude'].isel(window)
    else:
        tb_sub = ds[tb_varname].squeeze()
        var2_sub = ds[var2_varname].squeeze() if var2_varname is not None else None
        tracknumber_sub = ds['cloudtracknumber'].squeeze()
        lon_sub = ds['longitude']
        lat_sub = ds['latitude']

    # Longitudes of a 0-360 grid in -180~+180 for the map projections other than PlateCarree
    if projection_key(map_info.get('projection', 'PlateCarree')) != 'platecarree' and float(lon_sub.max()) > 180:
        lon_sub = norm_lon(lon_sub)

    # Get object perimeters
    # (periodic: the objects that cross the edge of a periodic domain continue on the other side; the whole domain only)
    periodic = (False, False)
    if window is None and periodic_perimeter != 'none':
        pbc_direction = str(ds.attrs.get('pbc_direction', 'none')) if periodic_perimeter == 'auto' else periodic_perimeter
        periodic = (pbc_direction in ['x', 'both'], pbc_direction in ['y', 'both'])
    tn_perim = label_perimeter(tracknumber_sub.data, dilationstructure, periodic=periodic)

    # Plotting variables
    figname, timestr, suptitle = make_figure_name(ds, figdir, figbasename, figname_type, ifile, title_prefix)

    # Put pixel data in a dictionary
    pixel_dict = {
        'lon': lon_sub,
        'lat': lat_sub,
        'tb': tb_sub,
        'var2': var2_sub,
        'tracknumber': tracknumber_sub,
        'tracknumber_perim': tn_perim,
        'pixel_bt': pixel_bt,
    }
    plot_info['timestr'] = timestr
    plot_info['figname'] = figname
    plot_info['suptitle'] = suptitle

    if panels == 2:
        fig = plot_map_2panels(pixel_dict, plot_info, map_info, track_dict)
    else:
        fig = plot_map_1panel(pixel_dict, plot_info, map_info, track_dict)
    plt.close(fig)
    print(figname)

    ds.close()
    return 1
