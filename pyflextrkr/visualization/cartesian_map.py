"""
Track maps of a domain in x/y (no map): idealized and cloud-resolving simulations in a doubly periodic domain, cold pools.
A field (or two: Tb and precipitation) with the tracked features (outline, track path, track number); the tracks can
cross the edges of a periodic domain (--pbc 1): their positions are wrapped back into the domain and the track path is
split where it wraps.

This is the plotting code of the former Analysis/plot_subset_tbpf_demo_pbc.py, plot_subset_tbpf_tracks_pinacles_pbc.py
(Tb + precipitation) and plot_subset_coldpool_tracks_nomap.py (one field).

    plot_map           the map of one time
    work_for_time_loop reads one pixel-level file, makes the dictionaries and calls plot_map

Called by: runner.run_frames (from cartesian_cli.main), serial or on Dask workers.
Calls: masks (perimeters, centers), pixelio (subset, figure name), trackdraw (halo).
"""
import sys
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import warnings
warnings.filterwarnings("ignore")
from pyflextrkr.visualization.masks import make_dilation_structure, label_perimeter, calc_track_center
from pyflextrkr.visualization.pixelio import get_data_extent, find_subset_window, make_figure_name
from pyflextrkr.visualization.trackdraw import halo_effects

# Grid layout of the figure: map + 1 colorbar (one field), map + 2 colorbars (two fields)
GRID_ONE_FIELD = dict(
    gridspec=dict(nrows=1, ncols=2, height_ratios=[1], width_ratios=[1, 0.03]),
    update=dict(wspace=0.05, hspace=0.05, left=0.1, right=0.9, top=0.92, bottom=0.08),
)
GRID_TWO_FIELDS = dict(
    gridspec=dict(nrows=1, ncols=2, height_ratios=[1], width_ratios=[1, 0.1]),
    update=dict(wspace=0.05, left=0.05, right=0.95, top=0.92, bottom=0.08),
)

#-----------------------------------------------------------------------
def plot_map(pixel_dict, plot_info, map_info, track_dict):
    """
    Plotting function.

    Args:
        pixel_dict: dictionary
            Dictionary containing pixel-level variables (longitude, latitude: x, y of the pixels; var1; var2 or None;
            tn: track number mask; tn_perim: its perimeters; pixel_bt)
        plot_info: dictionary
            Dictionary containing plotting variables:
                levels, cmaps, cbticks, cblabels: dictionaries with 'var1_*' and 'var2_*'
                titles: dictionary with 'var1_title' (two fields)
                perim_plot: 'pcolormesh' or 'contour'; perim_linewidth, perim_color (contour), mask_alpha (pcolormesh)
                pbc: True to wrap the tracks at the domain edges; show_pf_markers: red triangles at the PF positions
                xlabel, ylabel, figname, figsize, dpi, fontsize, remove_oob_low, remove_oob_high
                the track style (trackpath_*, track_marker, tracknumber_*, track_halo_*), see mcs_map
        map_info: dictionary
            Dictionary containing the domain: map_extent [xmin, xmax, ymin, ymax]
        track_dict: dictionary
            Dictionary containing tracking variables (get_cartesian_track_stats)

    Returns:
        fig: object
            Figure object.
    """

    # Get pixel data from dictionary
    pixel_bt = pixel_dict['pixel_bt']
    xx = pixel_dict['longitude']
    yy = pixel_dict['latitude']
    fvar = pixel_dict['var1']
    pcp = pixel_dict['var2']
    tn_perim = pixel_dict['tn_perim']
    tn = pixel_dict['tn']
    ntracks = track_dict['ntracks']
    lifetime = track_dict['lifetime']
    track_bt = track_dict['track_bt']
    track_meanlon = track_dict['track_meanlon']
    track_meanlat = track_dict['track_meanlat']
    track_pflon = track_dict.get('track_pflon')
    track_pflat = track_dict.get('track_pflat')
    dt_thres = track_dict['dt_thres']
    time_res = track_dict['time_res']
    # Get plot info from dictionary
    fontsize = plot_info['fontsize']
    levels = plot_info['levels']
    cmaps = plot_info['cmaps']
    titles = plot_info.get('titles')
    remove_oob_low = plot_info.get('remove_oob_low', False)
    remove_oob_high = plot_info.get('remove_oob_high', False)
    cblabels = plot_info['cblabels']
    cbticks = plot_info['cbticks']
    marker_size = plot_info['marker_size']
    tracknumber_fontsize = plot_info['tracknumber_fontsize']
    trackpath_linewidth = plot_info['trackpath_linewidth']
    trackpath_color = plot_info['trackpath_color']
    xlabel = plot_info['xlabel']
    ylabel = plot_info['ylabel']
    timestr = plot_info['timestr']
    figname = plot_info['figname']
    figsize = plot_info['figsize']
    dpi = plot_info.get('dpi', 200)
    mask_alpha = plot_info.get('mask_alpha', 1)
    perim_plot = plot_info.get('perim_plot', 'pcolormesh')
    perim_linewidth = plot_info.get('perim_linewidth', None)
    perim_color = plot_info.get('perim_color', 'black')
    pbc = plot_info.get('pbc', False)
    show_pf_markers = plot_info.get('show_pf_markers', False)
    # Track style (the defaults are the look of the original scripts)
    trackpath_linestyle = plot_info.get('trackpath_linestyle', '-')
    trackpath_alpha = plot_info.get('trackpath_alpha', 1.0)
    track_marker = plot_info.get('track_marker', 'o')
    track_marker_color = plot_info.get('track_marker_color') or trackpath_color
    tracknumber_color = plot_info.get('tracknumber_color', 'k')
    tracknumber_weight = plot_info.get('tracknumber_weight', 'bold')
    tracknumber_alpha = plot_info.get('tracknumber_alpha', 1.0)
    show_tracknumbers = plot_info.get('show_tracknumbers', True)
    # Halo around the track paths, symbols and numbers (path effects; None: no halo)
    halo_path, halo_marker, halo_text = halo_effects(
        plot_info.get('track_halo_color'), plot_info.get('track_halo_width', 2.0),
        plot_info.get('track_halo_alpha', 1.0), trackpath_linewidth,
    )

    # Map domain
    map_extent = map_info['map_extent']

    # Time difference matching pixel-time and track time
    dt_match = 1  # [min]

    # Marker style for track center
    marker_style = dict(edgecolor=track_marker_color, facecolor=track_marker_color, linestyle='-', marker=track_marker)

    # Set up figure
    mpl.rcParams['font.size'] = fontsize
    fig = plt.figure(figsize=figsize, dpi=dpi)

    if pcp is not None:
        # Set GridSpec for left (plot) and right (colorbars)
        gs = gridspec.GridSpec(**GRID_TWO_FIELDS['gridspec'])
        gs.update(**GRID_TWO_FIELDS['update'])
        # Use GridSpecFromSubplotSpec for panel and colorbar
        gs_cb = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], height_ratios=[1], width_ratios=[0.01,0.01], wspace=5)
        ax1 = plt.subplot(gs[0])
        cax1 = plt.subplot(gs_cb[0])
        cax2 = plt.subplot(gs_cb[1])
        # Figure title: time
        fig.suptitle(timestr, fontsize=fontsize*1.2)
    else:
        gs = gridspec.GridSpec(**GRID_ONE_FIELD['gridspec'])
        gs.update(**GRID_ONE_FIELD['update'])
        ax1 = plt.subplot(gs[0])
        cax1 = plt.subplot(gs[1])

    #################################################################
    ax1.set_aspect('auto', adjustable=None)
    if pcp is not None:
        ax1.set_title(titles['var1_title'], loc='left')

    # Plot variable
    cmap = plt.get_cmap(cmaps['var1_cmap'])
    norm_ref = mpl.colors.BoundaryNorm(levels['var1_levels'], ncolors=cmap.N, clip=True)
    # Remove out-of-bounds values
    if (remove_oob_low):
        fvar = np.ma.masked_where(fvar < min(levels['var1_levels']), fvar)
    if (remove_oob_high):
        fvar = np.ma.masked_where(fvar > max(levels['var1_levels']), fvar)
    cf1 = ax1.pcolormesh(xx, yy, fvar, norm=norm_ref, cmap=cmap, zorder=2)

    # Overplot tracknumber perimeters
    if perim_plot == 'pcolormesh':
        Tn = np.ma.masked_where(tn_perim == 0, tn_perim)
        Tn[Tn > 0] = 10
        tn1 = ax1.pcolormesh(xx, yy, Tn, cmap='gray', zorder=3, alpha=mask_alpha)
    elif perim_plot == 'contour':
        Tn = np.copy(tn.data)
        Tn[Tn > 0] = 10
        tn1 = ax1.contour(xx, yy, Tn, levels=[9,11], colors=perim_color, linewidths=perim_linewidth, zorder=3)
    else:
        print(f"ERROR: undefined perim_plot method: {perim_plot}!")
        sys.exit()
    # Set axis
    ax1.set_xlim(map_extent[0], map_extent[1])
    ax1.set_ylim(map_extent[2], map_extent[3])
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    if pcp is not None:
        # Precipitation
        cmap = plt.get_cmap(cmaps['var2_cmap'])
        norm = mpl.colors.BoundaryNorm(levels['var2_levels'], ncolors=cmap.N, clip=True)
        pcp_masked = np.ma.masked_where(((pcp < min(levels['var2_levels']))), pcp)
        cf2 = ax1.pcolormesh(xx, yy, pcp_masked, norm=norm, cmap=cmap, zorder=2)
        # Tb Colorbar
        cb1 = plt.colorbar(cf1, cax=cax1, label=cblabels['var1_label'], ticks=cbticks['var1_ticks'],
                           extend='both', orientation='vertical')
        # Precipitation Colorbar
        cb2 = plt.colorbar(cf2, cax=cax2, label=cblabels['var2_label'], ticks=cbticks['var2_ticks'],
                           extend='both', orientation='vertical')
    else:
        # Variable colorbar
        cb1 = plt.colorbar(cf1, cax=cax1, label=cblabels['var1_label'], ticks=cbticks['var1_ticks'], extend='both')
        ax1.set_title(timestr)

    # Get domain maximum values
    domain_max_x = map_extent[1] - map_extent[0]
    domain_max_y = map_extent[3] - map_extent[2]

    # Plot track centroids and paths
    for itrack in range(0, ntracks):
        # Get duration of the track
        ilifetime = lifetime.values[itrack]
        itracknum = lifetime.tracks.data[itrack]+1
        idur = (ilifetime / time_res).astype(int)

        # Get basetime of the track and the last time
        ibt = track_bt.values[itrack,:idur]
        ibt_end = np.nanmax(ibt)
        # Compute time difference between current pixel-level data time and the last time of the track
        idt_end = (pixel_bt - ibt_end).astype('timedelta64[m]')
        # Proceed if time difference is <= threshold
        # This means for tracks that end longer than the time threshold are not plotted
        if (idt_end <= dt_thres):
            # Find times in track data <= current pixel-level file time
            idx_cut = np.where(ibt <= pixel_bt)[0]
            idur_cut = len(idx_cut)
            if (idur_cut > 0):
                color_vals = np.repeat(ilifetime, idur_cut)
                # Track path
                size_vals = np.repeat(marker_size, idur_cut)
                size_vals[0] = marker_size * 2   # Make start symbol size larger
                if pbc:
                    ### Handle tracks that cross domain for plotting
                    # Get adjusted positions
                    adjusted_lon = track_meanlon.values[itrack, idx_cut]
                    adjusted_lat = track_meanlat.values[itrack, idx_cut]

                    # Wrap positions back into domain for plotting
                    wrapped_lon = np.mod(adjusted_lon - map_extent[0], domain_max_x) + map_extent[0]
                    wrapped_lat = np.mod(adjusted_lat - map_extent[2], domain_max_y) + map_extent[2]

                    # Identify where wrap-around occurs to split the trajectory
                    lon_diff = np.abs(np.diff(wrapped_lon))
                    lat_diff = np.abs(np.diff(wrapped_lat))
                    wrap_indices = np.where((lon_diff > (domain_max_x / 2)) | (lat_diff > (domain_max_y / 2)))[0] + 1

                    # Split the trajectory at wrap-around points
                    split_lon = np.split(wrapped_lon, wrap_indices)
                    split_lat = np.split(wrapped_lat, wrap_indices)

                    # Plot each segment separately
                    for lon_seg, lat_seg in zip(split_lon, split_lat):
                        # Ensure there are at least two points to plot a line
                        if len(lon_seg) >= 2:
                            ax1.plot(lon_seg, lat_seg, lw=trackpath_linewidth, ls=trackpath_linestyle, color=trackpath_color,
                                     alpha=trackpath_alpha, path_effects=halo_path, zorder=3)
                        else:
                            # For single points, plot as markers
                            ax1.scatter(lon_seg, lat_seg, s=marker_size, color=trackpath_color, alpha=trackpath_alpha,
                                        path_effects=halo_marker, zorder=3,)

                    # PF positions
                    if show_pf_markers and track_pflon is not None:
                        adjusted_pflon = track_pflon.values[itrack, idx_cut]
                        adjusted_pflat = track_pflat.values[itrack, idx_cut]
                        wrapped_pflon = np.mod(adjusted_pflon - map_extent[0], domain_max_x) + map_extent[0]
                        wrapped_pflat = np.mod(adjusted_pflat - map_extent[2], domain_max_y) + map_extent[2]
                        pflon_diff = np.abs(np.diff(wrapped_pflon))
                        pflat_diff = np.abs(np.diff(wrapped_pflat))
                        pfwrap_indices = np.where((pflon_diff > (domain_max_x / 2)) | (pflat_diff > (domain_max_y / 2)))[0] + 1
                        split_pflon = np.split(wrapped_pflon, pfwrap_indices)
                        split_pflat = np.split(wrapped_pflat, pfwrap_indices)
                        for lon_seg, lat_seg in zip(split_pflon, split_pflat):
                            # Plot PF positions as markers
                            ax1.scatter(lon_seg, lat_seg, s=marker_size, marker='^', color='red', zorder=3,)

                    # Initiation location (adjusted)
                    init_lon = track_meanlon.values[itrack, 0]
                    init_lat = track_meanlat.values[itrack, 0]
                    wrapped_init_lon = np.mod(init_lon - map_extent[0], domain_max_x) + map_extent[0]
                    wrapped_init_lat = np.mod(init_lat - map_extent[2], domain_max_y) + map_extent[2]
                    ax1.scatter(wrapped_init_lon, wrapped_init_lat, s=marker_size*2, alpha=trackpath_alpha,
                                path_effects=halo_marker, zorder=4, **marker_style)
                else:
                    cc = ax1.plot(track_meanlon.values[itrack,idx_cut], track_meanlat.values[itrack,idx_cut],
                                  lw=trackpath_linewidth, ls=trackpath_linestyle, color=trackpath_color,
                                  alpha=trackpath_alpha, path_effects=halo_path, zorder=3)
                    # Initiation location
                    cl1 = ax1.scatter(track_meanlon.values[itrack,0], track_meanlat.values[itrack,0], s=marker_size*2,
                                      alpha=trackpath_alpha, path_effects=halo_marker, zorder=4, **marker_style)

        # Find the closest time from track times
        idt = np.abs((ibt - pixel_bt).astype('timedelta64[m]'))
        idx_match = np.argmin(idt)
        idt_match = idt[idx_match]
        # Get track data
        _imeanlon = track_meanlon.data[itrack, idx_match]
        _imeanlat = track_meanlat.data[itrack, idx_match]

        # Adjust positions for plotting
        if pbc:
            _imeanlon = np.mod(_imeanlon - map_extent[0], domain_max_x) + map_extent[0]
            _imeanlat = np.mod(_imeanlat - map_extent[2], domain_max_y) + map_extent[2]

        # Proceed if time difference is < dt_match
        if idt_match < dt_match:
            # Overplot tracknumbers at current frame
            if show_tracknumbers and (_imeanlon > map_extent[0]) & (_imeanlon < map_extent[1]) & \
                (_imeanlat > map_extent[2]) & (_imeanlat < map_extent[3]):
                ax1.text(_imeanlon+0.02, _imeanlat+0.02, f'{itracknum:.0f}', color=tracknumber_color,
                         size=tracknumber_fontsize, weight=tracknumber_weight, ha='left', va='center',
                         alpha=tracknumber_alpha, path_effects=halo_text, zorder=4)

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
            Dictionary containing plotting variables (figdir, figbasename, var1_varname, var2_varname (None for one
            field), perim_thick [km], pixel_radius, xscale, yscale, ...)
        config: dictionary
            Dictionary containing config parameters
        ifile: int
            Index of the file in the series

    Returns:
        1.
    """

    map_extent = map_info.get('map_extent', None)
    subset = map_info.get('subset', 0)
    perim_thick = plot_info.get('perim_thick')
    pixel_radius = plot_info.get('pixel_radius')
    figdir = plot_info.get('figdir')
    figbasename = plot_info.get('figbasename')
    figname_type = plot_info.get('figname_type', 'date_time')
    var1_varname = plot_info.get('var1_varname')
    var2_varname = plot_info.get('var2_varname')
    xscale = plot_info.get('xscale', 1)
    yscale = plot_info.get('yscale', 1)

    # Read pixel-level data
    ds = xr.open_dataset(datafile, mask_and_scale=False)
    pixel_bt = ds.time.data

    # Get map extent from data
    if map_extent is None:
        map_extent = get_data_extent(ds)
        # (the extent of the data is in the units of the file, the plot is in the scaled units)
        map_extent = [map_extent[0] * xscale, map_extent[1] * xscale, map_extent[2] * yscale, map_extent[3] * yscale]
        map_info['map_extent'] = map_extent

    dilationstructure = make_dilation_structure(perim_thick, pixel_radius, pixel_radius)

    # Only plot if there is feature in the frame
    # if (np.nanmax(tn) > 0):
    # Subset pixel data within the map domain
    window = None
    if subset == 1:
        # (the extent is in the scaled units)
        window = find_subset_window(ds, var1_varname, [map_extent[0]/xscale, map_extent[1]/xscale,
                                                       map_extent[2]/yscale, map_extent[3]/yscale], cartesian=True)
    if window is not None:
        fvar = ds[var1_varname].isel(window).squeeze()
        pcp_sub = ds[var2_varname].isel(window).squeeze() if var2_varname is not None else None
        tracknumber_sub = ds['cloudtracknumber'].isel(window).squeeze()
        lon_sub = ds['longitude'].isel(window).data
        lat_sub = ds['latitude'].isel(window).data
    else:
        fvar = ds[var1_varname].squeeze()
        pcp_sub = ds[var2_varname].squeeze() if var2_varname is not None else None
        tracknumber_sub = ds['cloudtracknumber'].squeeze()
        lon_sub = ds['longitude'].data
        lat_sub = ds['latitude'].data

    # Scale x, y (change units from [m] to [km])
    lon_sub = lon_sub * xscale
    lat_sub = lat_sub * yscale

    # Get object perimeters
    tn_perim = label_perimeter(tracknumber_sub.data, dilationstructure)

    # Calculates track center locations
    lon_tn, lat_tn, tn_unique = calc_track_center(tracknumber_sub.data, lon_sub, lat_sub)

    # Plotting variables
    figname, timestr, suptitle = make_figure_name(ds, figdir, figbasename, figname_type, ifile, '')

    # Put variables in dictionaries
    pixel_dict = {
        'pixel_bt': pixel_bt,
        'longitude': lon_sub,
        'latitude': lat_sub,
        'var1': fvar,
        'var2': pcp_sub,
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
