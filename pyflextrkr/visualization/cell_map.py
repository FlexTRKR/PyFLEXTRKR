"""
Convective cell track maps: a radar field (composite reflectivity, echo-top height, ...) with the tracked cells (outline,
track path and symbols colored by lifetime, cell numbers) and optional radar range circles and azimuth lines.

This is the plotting code of the former Analysis/plot_subset_cell_tracks_demo.py and plot_subset_cell_tracks_ETH_demo.py
(the echo-top map is the same code with another variable, levels and colormap).

    plot_map           the map of one time
    work_for_time_loop reads one pixel-level file, makes the dictionaries and calls plot_map

Called by: runner.run_frames (from cell_cli.main), serial or on Dask workers.
Calls: mapcore (projection, domain), masks (perimeters, centers), pixelio (subset, figure name), trackdraw (halo).
"""
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import warnings
warnings.filterwarnings("ignore")
from pyflextrkr.visualization.mapcore import (
    get_projections, set_map_domain, projection_key, in_map_extent, calc_latlon, fit_colorbars_to_maps, lat_labels_on_boundary,
)
from pyflextrkr.visualization.masks import make_perimeter_structure, label_perimeter, calc_cell_center
from pyflextrkr.visualization.pixelio import get_data_extent, find_subset_window, make_figure_name
from pyflextrkr.visualization.trackdraw import halo_effects

# Grid layout of the figure: map + colorbar (also used by cell_cli to compute the figure size from the map aspect)
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
            Dictionary containing plotting variables; optional (the defaults are the look of the original script):
                trackpath_color (default 'k'), trackpath_linestyle, trackpath_alpha: track path
                track_marker, tracknumber_color, tracknumber_weight, tracknumber_alpha: symbols and cell numbers
                track_halo_color, track_halo_width, track_halo_alpha: halo around paths, symbols and numbers
                perim_color, perim_alpha: color and transparency of the cell outlines (default black, mask_alpha)
                show_legend: no legend in this map
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
    var_fill = pixel_dict['var_fill']
    conv_mask = pixel_dict['conv_mask']
    tn_perim = pixel_dict['tn_perim']
    lon_tn = pixel_dict['lon_tn']
    lat_tn = pixel_dict['lat_tn']
    tracknumbers = pixel_dict['tracknumber_unique']
    # Get track data from dictionary
    ntracks = track_dict['ntracks']
    lifetime = track_dict['lifetime']
    cell_bt = track_dict['cell_bt']
    cell_lon = track_dict['cell_lon']
    cell_lat = track_dict['cell_lat']
    dt_thres = track_dict['dt_thres']
    time_res = track_dict['time_res']
    # Get plot info from dictionary
    var_scale = plot_info.get('var_scale', 1)
    levels = plot_info['levels']
    cmaps = plot_info['cmaps']
    # titles = plot_info['titles']
    cblabels = plot_info['cblabels']
    cbticks = plot_info['cbticks']
    fontsize = plot_info['fontsize']
    fontsize_tracks = plot_info.get('fontsize_tracks', fontsize * 0.8)
    timestr = plot_info['timestr']
    title_prefix = plot_info.get('title_prefix', '')
    figname = plot_info['figname']
    figsize = plot_info['figsize']
    dpi = plot_info.get('dpi', 300)
    show_tracks = plot_info.get('show_tracks', True)
    show_paths = plot_info.get('show_paths', True)
    show_symbols = plot_info.get('show_symbols', True)
    shade_alpha = plot_info.get('shade_alpha', 1)
    mask_alpha = plot_info.get('mask_alpha', 1)
    map_edgecolor = plot_info['map_edgecolor']
    map_resolution = plot_info['map_resolution']
    marker_size = plot_info['marker_size']
    lw_centroid = plot_info['lw_centroid']
    cmap_tracks = plot_info['cmap_tracks']
    cblabel_tracks = plot_info['cblabel_tracks']
    cbticks_tracks = plot_info['cbticks_tracks']
    lev_lifetime = plot_info['lev_lifetime']
    radii = plot_info['radii']
    azimuths = plot_info['azimuths']
    radar_lon = plot_info['radar_lon']
    radar_lat = plot_info['radar_lat']
    show_rangecircle = plot_info.get('show_rangecircle', True)
    show_azimuth = plot_info.get('show_azimuth', True)
    radar_sensitivity = plot_info.get('radar_sensitivity', 0)
    # Track style (the defaults are the look of the original script)
    trackpath_color = plot_info.get('trackpath_color', 'k')
    trackpath_linestyle = plot_info.get('trackpath_linestyle', '-')
    trackpath_alpha = plot_info.get('trackpath_alpha', 1.0)
    track_marker = plot_info.get('track_marker', 'o')
    track_marker_edgecolor = plot_info.get('track_marker_color') or 'k'
    tracknumber_color = plot_info.get('tracknumber_color', 'k')
    tracknumber_weight = plot_info.get('tracknumber_weight', 'bold')
    tracknumber_alpha = plot_info.get('tracknumber_alpha', 1.0)
    show_tracknumbers = plot_info.get('show_tracknumbers', True)
    perim_color = plot_info.get('perim_color', None)
    # Halo around the track paths, symbols and numbers (path effects; None: no halo)
    halo_path, halo_marker, halo_text = halo_effects(
        plot_info.get('track_halo_color'), plot_info.get('track_halo_width', 2.0),
        plot_info.get('track_halo_alpha', 1.0), np.max(lw_centroid),
    )

    # Map domain, lat/lon ticks, background map features
    map_extent = map_info['map_extent']
    lonv = map_info['lonv']
    latv = map_info['latv']
    draw_land = map_info.get('draw_land', False)
    draw_border = map_info.get('draw_border', False)
    draw_state = map_info.get('draw_state', False)
    draw_river = map_info.get('draw_river', False)
    projection = map_info.get('projection', 'PlateCarree')
    lat_range = map_info.get('lat_range', None)
    grid_linewidth = map_info.get('grid_linewidth', 0.)
    grid_alpha = map_info.get('grid_alpha', None)
    land_linewidth = plot_info.get('land_linewidth', 1)
    is_platecarree = projection_key(projection) == 'platecarree'
    extent_xy = map_info.get('map_extent_xy', None)       # the domain in projected coordinates (the data, no --extent)
    map_aspect = plot_info.get('map_aspect', 'auto')     # 'equal': the map keeps its shape, 'auto': it fills its cell

    # Set up track lifetime colors
    cmap_lifetime = plt.get_cmap(cmap_tracks)
    norm_lifetime = mpl.colors.BoundaryNorm(lev_lifetime, ncolors=cmap_lifetime.N, clip=True)

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
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='w')
    gs = gridspec.GridSpec(**GRID_MAP['gridspec'])
    gs.update(**GRID_MAP['update'])
    ax1 = plt.subplot(gs[0], projection=proj)
    cax1 = plt.subplot(gs[1])

    set_map_domain(ax1, proj, map_extent, _ext_lon0, _ext_lon1, lat_range, extent_xy)
    if is_platecarree and map_aspect == 'auto':
        # the map fills its cell of the figure (it is stretched if the cell has another shape: the old look)
        ax1.set_aspect('auto', adjustable=None)
    else:
        # the map keeps its shape (the axes shrink to fit its cell of the figure); the colorbar follows it (fit_colorbars_to_maps)
        ax1.set_aspect('equal', adjustable='box')
    # Add map features
    if draw_land == True:
        ax1.add_feature(land, edgecolor=map_edgecolor, facecolor='none', linewidth=land_linewidth, zorder=4)
    if draw_border == True:
        ax1.add_feature(borders, edgecolor=map_edgecolor, facecolor='none', linewidth=0.8, zorder=4)
    if draw_state == True:
        ax1.add_feature(states, edgecolor='gray', facecolor='none', linewidth=0.5, zorder=4)
    if draw_river == True:
        ax1.add_feature(rivers, edgecolor='gray', facecolor='none', linewidth=0.5, zorder=4)
    # Gridlines and lat/lon labels
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

    # Colorfill variable
    cmap = plt.get_cmap(cmaps)
    norm_pcm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    # Mask out values below radar sensitivity threshold
    var_fill = np.ma.masked_where(var_fill < radar_sensitivity, var_fill)
    # Scale the variable
    var_fill = var_fill * var_scale
    var_fill = np.ma.masked_where(var_fill < min(levels), var_fill)
    cf1 = ax1.pcolormesh(xx, yy, var_fill, norm=norm_pcm, cmap=cmap, transform=data_proj, zorder=2, alpha=shade_alpha)
    # Overplot cell tracknumber perimeters
    Tn = np.ma.masked_where(tn_perim == 0, tn_perim)
    Tn[Tn > 0] = 10
    if perim_color is None:
        tn1 = ax1.pcolormesh(xx, yy, Tn, cmap='gray', transform=data_proj, zorder=3, alpha=mask_alpha)
    else:
        # Outline in a color of your choice
        tn1 = ax1.pcolormesh(xx, yy, Tn, cmap=mpl.colors.ListedColormap([perim_color]), vmin=0, vmax=20,
                             transform=data_proj, zorder=3, alpha=mask_alpha)

    # Plot track centroids and paths
    if show_tracks:
        marker_style_s = dict(edgecolor=track_marker_edgecolor, linestyle='-', marker=track_marker)
        marker_style_m = dict(edgecolor=track_marker_edgecolor, linestyle='-', marker=track_marker)
        marker_style_l = dict(edgecolor=track_marker_edgecolor, linestyle='-', marker=track_marker)
        for itrack in range(0, ntracks):
            # Get duration of the track
            ilifetime = lifetime.values[itrack]
            itracknum = lifetime.tracks.data[itrack]+1
            idur = (ilifetime / time_res).astype(int)
            # Get basetime of the track and the last time
            ibt = cell_bt.values[itrack,:idur]
            ibt_last = np.nanmax(ibt)
            # Compute time difference between current pixel-level data time and the last time of the track
            idt = (pixel_bt - ibt_last).astype('timedelta64[m]')
            # Proceed if time difference is <= threshold
            # This means for tracks that end longer than the time threshold are not plotted
            if (idt <= dt_thres):
                # Find times in track data <= current pixel-level file time
                idx_cut = np.where(ibt <= pixel_bt)[0]
                idur_cut = len(idx_cut)
                if (idur_cut > 0):
                    color_vals = np.repeat(ilifetime, idur_cut)
                    # Change centroid marker, linewidth based on track lifetime [hour]
                    if (ilifetime < 1):
                        lw_c = lw_centroid[0]
                        size_c = marker_size[0]
                        marker_style = marker_style_s
                    elif ((ilifetime >= 1) & (ilifetime < 2)):
                        lw_c = lw_centroid[1]
                        size_c = marker_size[1]
                        marker_style = marker_style_m
                    elif (ilifetime >= 2):
                        lw_c = lw_centroid[2]
                        size_c = marker_size[2]
                        marker_style = marker_style_l
                    else:
                        lw_c = 0
                        size_c = 0
                    size_vals = np.repeat(size_c, idur_cut)
                    size_vals[0] = size_c * 2   # Make CI symbol size larger
                    if show_paths:
                        cc = ax1.plot(cell_lon.values[itrack,idx_cut], cell_lat.values[itrack,idx_cut], lw=lw_c,
                                      ls=trackpath_linestyle, color=trackpath_color, alpha=trackpath_alpha,
                                      path_effects=halo_path, transform=data_proj, zorder=3)
                    if show_symbols:
                        cl = ax1.scatter(cell_lon.values[itrack,idx_cut], cell_lat.values[itrack,idx_cut], s=size_vals, c=color_vals,
                                         norm=norm_lifetime, cmap=cmap_lifetime, alpha=trackpath_alpha,
                                         path_effects=halo_marker, transform=data_proj, zorder=4, **marker_style)

        # Plot colorbar for tracks
        if show_symbols:
            cax = inset_axes(ax1, width="100%", height="100%", bbox_to_anchor=(.04, .97, .3, .03), bbox_transform=ax1.transAxes)
            cbinset = mpl.colorbar.ColorbarBase(cax, cmap=cmap_lifetime, norm=norm_lifetime, orientation='horizontal', label=cblabel_tracks)
            cbinset.set_ticks(cbticks_tracks)

    # Overplot cell tracknumbers at current frame
    if show_tracknumbers:
        for ii in range(0, len(lon_tn)):
            # Normalize to 0-360 for dateline-safe lon check (in_map_extent)
            if in_map_extent(lon_tn[ii], lat_tn[ii], map_extent):
                ax1.text(lon_tn[ii]+0.02, lat_tn[ii]+0.02, f'{tracknumbers[ii]:.0f}', color=tracknumber_color, size=fontsize_tracks,
                         weight=tracknumber_weight, ha='left', va='center', alpha=tracknumber_alpha,
                         path_effects=halo_text, clip_on=not is_platecarree, transform=data_proj, zorder=4)

    # Plot range circles around radar
    if show_rangecircle:
        for ii in range(0, len(radii)):
            rr = ax1.tissot(rad_km=radii[ii], lons=radar_lon, lats=radar_lat, n_samples=100, facecolor='None', edgecolor='k', lw=0.4, zorder=3)
    # Plot azimuth lines
    if show_azimuth:
        for ii in range(0, len(azimuths)):
            lon2, lat2 = calc_latlon(radar_lon, radar_lat, 200, azimuths[ii])
            ax1.plot([radar_lon,lon2], [radar_lat,lat2], color='k', lw=0.4, transform=ccrs.Geodetic(), zorder=5)
    # Reflectivity colorbar
    cb1 = plt.colorbar(cf1, cax=cax1, label=cblabels, ticks=cbticks, extend='both')
    # Set title with optional prefix
    title = f"{title_prefix} | {timestr}" if title_prefix else timestr
    ax1.set_title(title)

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
            Dictionary containing plotting variables
        config: dictionary
            Dictionary containing config parameters (not used)
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
    title_prefix = plot_info.get('title_prefix', '')
    varname_fill = plot_info.get('varname_fill')
    radar_lat = plot_info.get('radar_lat')
    radar_lon = plot_info.get('radar_lon')
    perim_width = plot_info.get('perim_width', 1)
    skip_empty = plot_info.get('skip_empty', False)

    # Read pixel-level data
    ds = xr.open_dataset(datafile)
    pixel_bt = ds.time.data

    # Only plot if there is cell in the frame
    if skip_empty and not (np.nanmax(ds['tracknumber'].squeeze()) > 0):
        ds.close()
        return 1

    # Get map extent from data
    if map_extent is None:
        map_extent = get_data_extent(ds)
        map_info['map_extent'] = map_extent

    # Calculate radar location from domain center if not provided
    if radar_lat is None or radar_lon is None:
        # Calculate center of domain
        center_lon = (map_extent[0] + map_extent[1]) / 2
        center_lat = (map_extent[2] + map_extent[3]) / 2

        # Find closest grid point to center
        lon_vals = ds['longitude'].values
        lat_vals = ds['latitude'].values
        distances = np.sqrt((lon_vals - center_lon)**2 + (lat_vals - center_lat)**2)
        min_idx = np.unravel_index(np.argmin(distances), lon_vals.shape)

        radar_lon = lon_vals[min_idx]
        radar_lat = lat_vals[min_idx]

        print(f"⚠️  Radar location not provided. Using domain center:")
        print(f"   Radar latitude: {radar_lat:.4f}°")
        print(f"   Radar longitude: {radar_lon:.4f}°")

        # Update plot_info with calculated radar location
        plot_info['radar_lon'] = radar_lon
        plot_info['radar_lat'] = radar_lat

    # Make dilation structure (larger values make thicker outlines)
    # perim_width = 1 is generate_binary_structure(2,1)
    dilationstructure = make_perimeter_structure(perim_width)

    # Get cell tracknumbers
    # tn = ds['tracknumber'].squeeze()

    # Subset pixel data within the map domain
    window = None
    if subset == 1:
        window = find_subset_window(ds, varname_fill, map_extent)
    if window is not None:
        # Subset using index slicing
        dim_y, dim_x = list(window)
        lon_sub = ds['longitude'].isel(window).squeeze().values
        lat_sub = ds['latitude'].isel(window).squeeze().values
        var_fill = ds[varname_fill].isel(window).squeeze()
        convmask_sub = ds['conv_mask'].isel(window).squeeze()
        tracknumber_sub = ds['tracknumber'].isel(window).squeeze()
        xx_sub = ds['lon'].isel({dim_x: window[dim_x]}).squeeze().values
        yy_sub = ds['lat'].isel({dim_y: window[dim_y]}).squeeze().values
    else:
        xx_sub = ds['lon'].values
        yy_sub = ds['lat'].values
        lon_sub = ds['longitude'].squeeze().values
        lat_sub = ds['latitude'].squeeze().values
        var_fill = ds[varname_fill].squeeze()
        convmask_sub = ds['conv_mask'].squeeze()
        tracknumber_sub = ds['tracknumber'].squeeze()

    # Get object perimeters
    tn_perim = label_perimeter(tracknumber_sub.data, dilationstructure)

    # Apply tracknumber to conv_mask
    tnconv = tracknumber_sub.where(convmask_sub > 0).data

    # Calculates cell center locations
    lon_tn, lat_tn, xx_tn, yy_tn, tnconv_unique = calc_cell_center(tnconv, lon_sub, lat_sub, xx_sub, yy_sub)

    # titles = [timestr]
    figname, timestr, suptitle = make_figure_name(ds, figdir, figbasename, figname_type, ifile, title_prefix)
    # (the cell maps have the title in the axes: the time string only)

    # Put variables in dictionaries
    pixel_dict = {
        'pixel_bt': pixel_bt,
        'longitude': lon_sub,
        'latitude': lat_sub,
        'var_fill': var_fill,
        'tn': tracknumber_sub,
        'conv_mask': convmask_sub,
        'tn_perim': tn_perim,
        'lon_tn': lon_tn,
        'lat_tn': lat_tn,
        'tracknumber_unique': tnconv_unique,
    }
    plot_info['timestr'] = timestr
    plot_info['title_prefix'] = title_prefix
    plot_info['figname'] = figname

    # Call plotting function
    fig = plot_map(pixel_dict, plot_info, map_info, track_dict)
    plt.close(fig)
    print(figname)

    ds.close()
    return 1
