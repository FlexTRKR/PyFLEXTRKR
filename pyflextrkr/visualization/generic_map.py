"""
Generic feature track maps: a 2D field (Z500 anomaly, stream function, vorticity, reflectivity, ...) with the tracked
features (outline, track path, track number).

This is the plotting code of the former Analysis/plot_subset_generic_tracks_demo.py and plot_subset_psi_tracks_demo.py (the
stream-function maps are the same code with another variable, levels and colormap).

    plot_map           the map of one time
    work_for_time_loop reads one pixel-level file, makes the dictionaries and calls plot_map

Called by: runner.run_frames (from generic_cli.main), serial or on Dask workers.
Calls: mapcore (projection, domain), masks (perimeters, centers), pixelio (subset, figure name), trackdraw (halo).
"""
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import warnings
warnings.filterwarnings("ignore")
from pyflextrkr.visualization.mapcore import (
    get_projections, set_map_domain, projection_key, in_map_extent, fit_colorbars_to_maps, lat_labels_on_boundary,
)
from pyflextrkr.visualization.masks import make_perimeter_structure, label_perimeter, calc_track_center
from pyflextrkr.visualization.pixelio import get_data_extent, find_subset_window, make_figure_name
from pyflextrkr.visualization.trackdraw import halo_effects

# Grid layout of the figure: map + colorbar (also used by generic_cli to compute the figure size from the map aspect)
GRID_MAP = dict(
    gridspec=dict(nrows=1, ncols=2, height_ratios=[1], width_ratios=[1, 0.03]),
    update=dict(wspace=0.05, hspace=0.05, left=0.1, right=0.9, top=0.92, bottom=0.08),
)

#-----------------------------------------------------------------------
def plot_map(pixel_dict, plot_info, map_info, track_dict):
    """
    Plotting function.

    Args:
        pixel_dict: dictionary
            Dictionary containing pixel-level variables
        plot_info: dictionary
            Dictionary containing plotting variables; optional (the defaults are the look of the original scripts):
                trackpath_linestyle, trackpath_alpha: track path
                track_marker, track_marker_color: symbol at the initiation location
                tracknumber_color, tracknumber_weight, tracknumber_alpha, show_tracknumbers: track numbers
                track_halo_color, track_halo_width, track_halo_alpha: halo around paths, symbols and numbers
                track_number_source: 'track' (the position of the track at this time, default) or 'mask' (the center
                    of the track mask)
                show_symbols: symbols at all positions of the track path (default: only at the initiation location)
                map_aspect: 'equal' (the map keeps its shape, the colorbar follows it) or 'auto' (default: the map fills its panel)
        map_info: dictionary
            Dictionary containing mapping variables (map_extent_xy: [xmin, xmax, ymin, ymax], the domain in the coordinates of the
            projection, optional: mapcore.data_extent_xy)
        track_dict: dictionary
            Dictionary containing tracking variables

    Returns:
        fig: object
            Figure object.
    """

    # Get pixel data from dictionary
    pixel_bt = pixel_dict['pixel_bt']
    xx = pixel_dict['longitude']
    yy = pixel_dict['latitude']
    fvar = pixel_dict['fvar']
    tn_perim = pixel_dict['tn_perim']
    lon_tn = pixel_dict['lon_tn']
    lat_tn = pixel_dict['lat_tn']
    tracknumbers = pixel_dict['tracknumber_unique']
    # Get track data from dictionary
    ntracks = track_dict['ntracks']
    lifetime = track_dict['lifetime']
    base_time = track_dict['base_time']
    meanlon = track_dict['meanlon']
    meanlat = track_dict['meanlat']
    # Get plot info from dictionary
    fontsize = plot_info['fontsize']
    levels = plot_info['levels']
    cmap = plot_info['cmap']
    # titles = plot_info['titles']
    remove_oob_low = plot_info.get('remove_oob_low', False)
    remove_oob_high = plot_info.get('remove_oob_high', False)
    cblabels = plot_info['cblabels']
    cbticks = plot_info['cbticks']
    marker_size = plot_info['marker_size']
    tracknumber_fontsize = plot_info['tracknumber_fontsize']
    trackpath_linewidth = plot_info['trackpath_linewidth']
    trackpath_color = plot_info['trackpath_color']
    map_edgecolor = plot_info['map_edgecolor']
    map_resolution = plot_info['map_resolution']
    timestr = plot_info['timestr']
    figname = plot_info['figname']
    figsize = plot_info['figsize']
    dpi = plot_info.get('dpi', 200)
    dt_thres = track_dict['dt_thres']
    time_res = track_dict['time_res']
    # Track style (the defaults are the look of the original scripts)
    trackpath_linestyle = plot_info.get('trackpath_linestyle', '-')
    trackpath_alpha = plot_info.get('trackpath_alpha', 1.0)
    track_marker = plot_info.get('track_marker', 'o')
    track_marker_color = plot_info.get('track_marker_color') or trackpath_color
    tracknumber_color = plot_info.get('tracknumber_color', 'k')
    tracknumber_weight = plot_info.get('tracknumber_weight', 'bold')
    tracknumber_alpha = plot_info.get('tracknumber_alpha', 1.0)
    show_tracknumbers = plot_info.get('show_tracknumbers', True)
    track_number_source = plot_info.get('track_number_source', 'track')
    show_symbols = plot_info.get('show_symbols', False)
    # Halo around the track paths, symbols and numbers (path effects; None: no halo)
    halo_path, halo_marker, halo_text = halo_effects(
        plot_info.get('track_halo_color'), plot_info.get('track_halo_width', 2.0),
        plot_info.get('track_halo_alpha', 1.0), trackpath_linewidth,
    )
    # Map domain, lat/lon ticks, background map features
    map_extent = map_info['map_extent']
    lonv = map_info['lonv']
    latv = map_info['latv']
    draw_land = map_info.get('draw_land', True)
    draw_border = map_info.get('draw_border', False)
    draw_state = map_info.get('draw_state', False)
    draw_river = map_info.get('draw_river', False)
    projection = map_info.get('projection', 'PlateCarree')
    lat_range = map_info.get('lat_range', None)
    grid_linewidth = map_info.get('grid_linewidth', 0.5)
    grid_alpha = map_info.get('grid_alpha', None)
    land_linewidth = plot_info.get('land_linewidth', 1)
    is_platecarree = projection_key(projection) == 'platecarree'
    extent_xy = map_info.get('map_extent_xy', None)       # the domain in projected coordinates (the data, no --extent)
    map_aspect = plot_info.get('map_aspect', 'auto')     # 'equal': the map keeps its shape, 'auto': it fills its cell

    # Time difference matching pixel-time and track time
    dt_match = 1  # [min]

    # Marker style for track center
    marker_style = dict(edgecolor=track_marker_color, facecolor=track_marker_color, linestyle='-', marker=track_marker)

    # Set up map projection (PlateCarree with the central longitude from the extent, or the projection that was asked for)
    proj, data_proj, _ext_lon0, _ext_lon1 = get_projections(
        map_extent, projection, map_info.get('central_longitude'), map_info.get('central_latitude'),
        map_info.get('standard_parallels'),
    )
    land = cfeature.NaturalEarthFeature('physical', 'land', map_resolution)
    borders = cfeature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land', map_resolution)
    states = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lakes', map_resolution)
    rivers = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', map_resolution)

    # Set up figure
    mpl.rcParams['font.size'] = fontsize
    # mpl.rcParams['font.family'] = 'Helvetica'
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = gridspec.GridSpec(**GRID_MAP['gridspec'])
    gs.update(**GRID_MAP['update'])
    ax1 = plt.subplot(gs[0], projection=proj)
    cax1 = plt.subplot(gs[1])
    # Plot background map elements
    set_map_domain(ax1, proj, map_extent, _ext_lon0, _ext_lon1, lat_range, extent_xy)
    if is_platecarree and map_aspect == 'auto':
        # the map fills its cell of the figure (it is stretched if the cell has another shape: the old look)
        ax1.set_aspect('auto', adjustable=None)
    else:
        # the map keeps its shape (the axes shrink to fit its cell of the figure); the colorbar follows it (fit_colorbars_to_maps)
        ax1.set_aspect('equal', adjustable='box')
    if draw_land == True:
        ax1.add_feature(land, facecolor='none', edgecolor=map_edgecolor, linewidth=land_linewidth, zorder=3)
    if draw_border == True:
        ax1.add_feature(borders, edgecolor=map_edgecolor, facecolor='none', linewidth=0.8, zorder=3)
    if draw_state == True:
        ax1.add_feature(states, edgecolor=map_edgecolor, facecolor='none', linewidth=0.8, zorder=3)
    if draw_river == True:
        ax1.add_feature(rivers, edgecolor=map_edgecolor, facecolor='none', linewidth=0.5, zorder=3)
    # Grid lines
    gl = ax1.gridlines(crs=data_proj, draw_labels=True, linestyle='--', linewidth=grid_linewidth, alpha=grid_alpha)
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

    # Plot variable
    cmap = plt.get_cmap(cmap)
    norm_ref = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    # Remove out-of-bounds values
    if (remove_oob_low):
        fvar = np.ma.masked_where(fvar < min(levels), fvar)
    if (remove_oob_high):
        fvar = np.ma.masked_where(fvar > max(levels), fvar)
    cf1 = ax1.pcolormesh(xx, yy, fvar, norm=norm_ref, cmap=cmap, transform=data_proj, zorder=2)
    # Overplot tracknumber perimeters
    Tn = np.ma.masked_where(tn_perim == 0, tn_perim)
    Tn[Tn > 0] = 10
    tn1 = ax1.pcolormesh(xx, yy, Tn, cmap='gray', transform=data_proj, zorder=3)
    # Variable colorbar
    cb1 = plt.colorbar(cf1, cax=cax1, label=cblabels, ticks=cbticks, extend='both')
    ax1.set_title(timestr)

    # Plot track centroids and paths
    for itrack in range(0, ntracks):
        # Get duration of the track
        ilifetime = lifetime.values[itrack]
        itracknum = lifetime.tracks.data[itrack]+1
        idur = (ilifetime / time_res).astype(int)
        # Get basetime of the track and the last time
        ibt = base_time.values[itrack,:idur]
        ibt_end = np.nanmax(ibt)
        # Compute time difference between current pixel-level data time and the last time of the track
        idt = (pixel_bt - ibt_end).astype('timedelta64[m]')
        # Proceed if time difference is <= threshold
        # This means for tracks that end longer than the time threshold are not plotted
        if (idt <= dt_thres):
            # Find times in track data <= current pixel-level file time
            idx_cut = np.where(ibt <= pixel_bt)[0]
            idur_cut = len(idx_cut)
            if (idur_cut > 0):
                color_vals = np.repeat(ilifetime, idur_cut)
                # Track path
                size_vals = np.repeat(marker_size, idur_cut)
                size_vals[0] = marker_size * 2   # Make start symbol size larger
                cc = ax1.plot(meanlon.values[itrack,idx_cut], meanlat.values[itrack,idx_cut],
                              lw=trackpath_linewidth, ls=trackpath_linestyle, color=trackpath_color,
                              alpha=trackpath_alpha, path_effects=halo_path, transform=data_proj, zorder=3)
                if show_symbols:
                    cl = ax1.scatter(meanlon.values[itrack,idx_cut], meanlat.values[itrack,idx_cut],
                                     s=size_vals, alpha=trackpath_alpha, path_effects=halo_marker,
                                     transform=data_proj, zorder=4, **marker_style)
                else:
                    # Initiation location
                    cl1 = ax1.scatter(meanlon.values[itrack,0], meanlat.values[itrack,0], s=marker_size*2,
                                      alpha=trackpath_alpha, path_effects=halo_marker, transform=data_proj,
                                      zorder=4, **marker_style)

        # Find the closest time from track times
        idt = np.abs((ibt - pixel_bt).astype('timedelta64[m]'))
        idx_match = np.argmin(idt)
        idt_match = idt[idx_match]
        # Get center lat/lon from the matched tracks
        _ilon = meanlon.data[itrack,idx_match]
        _ilat = meanlat.data[itrack,idx_match]
        # Proceed if time difference is < dt_match
        if (idt_match < dt_match):
            # Overplot tracknumbers at current frame
            # Normalize to 0-360 for dateline-safe lon check (in_map_extent)
            if show_tracknumbers and track_number_source == 'track' and in_map_extent(_ilon, _ilat, map_extent):
                ax1.text(_ilon+0.02, _ilat+0.02, f'{itracknum:.0f}', color=tracknumber_color, size=tracknumber_fontsize,
                         ha='left', va='center', weight=tracknumber_weight, alpha=tracknumber_alpha,
                         path_effects=halo_text, clip_on=not is_platecarree, transform=data_proj, zorder=5)

    # Overplot tracknumbers at the center of the track masks at current frame
    if show_tracknumbers and track_number_source == 'mask':
        for ii in range(0, len(lon_tn)):
            if in_map_extent(lon_tn[ii], lat_tn[ii], map_extent):
                ax1.text(lon_tn[ii]+0.02, lat_tn[ii]+0.02, f'{tracknumbers[ii]:.0f}', color=tracknumber_color,
                         size=tracknumber_fontsize, weight=tracknumber_weight, ha='left', va='center',
                         alpha=tracknumber_alpha, path_effects=halo_text, clip_on=not is_platecarree, transform=data_proj, zorder=4)

    # The map keeps its shape: the colorbar has the extent of the map
    if not (is_platecarree and map_aspect == 'auto'):
        fit_colorbars_to_maps([(ax1, [cax1])], 'vertical')

    # Thread-safe figure output
    canvas = FigureCanvas(fig)
    fig.savefig(figname)

    return fig

#-----------------------------------------------------------------------
def work_for_time_loop(datafile, track_dict, map_info, plot_info, config, ifile):
    """
    Process data for a single frame and make the plot.

    Args:
        datafile: string
            Pixel-level data filename
        track_dict: dictionary
            Dictionary containing tracking variables
        map_info: dictionary
            Dictionary containing mapping variables
        plot_info: dictionary
            Dictionary containing plotting variables (figdir, figbasename, field_varname, perim_width, ...)
        config: dictionary
            Dictionary containing config parameters (field_varname if not in plot_info)
        ifile: int
            Index of the file in the series

    Returns:
        1.
    """

    map_extent = map_info.get('map_extent', None)
    subset = map_info.get('subset', 0)
    figdir = plot_info.get('figdir')
    figbasename = plot_info.get('figbasename')
    figname_type = plot_info.get('figname_type', 'date_time')
    perim_width = plot_info.get('perim_width', 1)
    time_label_format = plot_info.get('time_label_format', "%Y-%m-%d %H:%M:%S UTC")
    figname_time_format = plot_info.get('figname_time_format', "%Y%m%d_%H%M%S")

    # Read pixel-level data
    ds = xr.open_dataset(datafile)
    pixel_bt = ds.time.data

    # Get map extent from data
    if map_extent is None:
        map_extent = get_data_extent(ds)
        map_info['map_extent'] = map_extent

    # Make dilation structure (larger values make thicker outlines)
    dilationstructure = make_perimeter_structure(perim_width)

    # Data variable names
    field_varname = plot_info.get('field_varname') or config["field_varname"]

    # Get tracknumbers
    # tn = ds['cloudtracknumber'].squeeze()

    # Only plot if there is feature in the frame
    # if (np.nanmax(tn) > 0):
    # Subset pixel data within the map domain
    window = None
    if subset == 1:
        window = find_subset_window(ds, field_varname, map_extent)
    if window is not None:
        # Subset using index slicing
        lon_sub = ds['longitude'].isel(window).data
        lat_sub = ds['latitude'].isel(window).data
        fvar = ds[field_varname].isel(window).squeeze()
        tracknumber_sub = ds['cloudtracknumber'].isel(window).squeeze()
    else:
        fvar = ds[field_varname].squeeze()
        tracknumber_sub = ds['cloudtracknumber'].squeeze()
        lon_sub = ds['longitude'].data
        lat_sub = ds['latitude'].data

    # Get object perimeters
    tn_perim = label_perimeter(tracknumber_sub.data, dilationstructure)

    # Calculates track center locations
    lon_tn, lat_tn, tn_unique = calc_track_center(tracknumber_sub.data, lon_sub, lat_sub)

    # Plotting variables
    figname, timestr, suptitle = make_figure_name(ds, figdir, figbasename, figname_type, ifile, '')
    if time_label_format != "%Y-%m-%d %H:%M:%S UTC" or figname_time_format != "%Y%m%d_%H%M%S":
        # (the stream-function maps have the minute in the title and the figure name)
        timestr = ds['time'].squeeze().dt.strftime(time_label_format).data
        figname = f"{figdir}{figbasename}{ds['time'].squeeze().dt.strftime(figname_time_format).data.item()}.png"

    # Put variables in dictionaries
    pixel_dict = {
        'pixel_bt': pixel_bt,
        'longitude': lon_sub,
        'latitude': lat_sub,
        'fvar': fvar,
        'tn': tracknumber_sub,
        'tn_perim': tn_perim,
        'lon_tn': lon_tn,
        'lat_tn': lat_tn,
        'tracknumber_unique': tn_unique,
    }
    plot_info['timestr'] = timestr
    plot_info['figname'] = figname
    # Call plotting function
    fig = plot_map(pixel_dict, plot_info, map_info, track_dict)
    plt.close(fig)
    print(figname)

    ds.close()
    return 1
