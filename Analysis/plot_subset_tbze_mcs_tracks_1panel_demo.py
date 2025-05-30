"""
Demonstrates ploting MCS tracks on Tb, reflectivity snapshots on a single panel for a subset domain.

>python plot_subset_tbze_mcs_tracks_1panel_demo.py -s STARTDATE -e ENDDATE -c CONFIG.yml
Optional arguments:
-p 0 (serial), 1 (parallel)
--extent lonmin lonmax latmin latmax (subset domain boundary)
--subset 0 (no), 1 (yes) (subset data before plotting)
--figsize width height (figure size in inches)
--output output_directory (output figure directory)
--figbasename figure base name (output figure base name)
--trackstats_file MCS track stats file name (optional, if different from robust MCS track stats file)
--pixel_path Pixel-level tracknumber mask files directory (optional, if different from robust MCS pixel files)

Zhe Feng, PNNL
contact: Zhe.Feng@pnnl.gov
"""

import argparse
import numpy as np
import os, sys
import xarray as xr
import pandas as pd
from scipy.ndimage import binary_erosion
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import colorcet as cc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# For non-gui matplotlib back end
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
mpl.use('agg')
import dask
from dask.distributed import Client, LocalCluster
import warnings
warnings.filterwarnings("ignore")
from pyflextrkr.ft_utilities import load_config, subset_files_timerange

#-----------------------------------------------------------------------
def parse_cmd_args():
    # Define and retrieve the command-line arguments...
    parser = argparse.ArgumentParser(
        description="Plot MCS tracks on radar reflectivity snapshots for a user-defined subset domain."
    )
    parser.add_argument("-s", "--start", help="first time in time series to plot, format=YYYY-mm-ddTHH:MM:SS", required=True)
    parser.add_argument("-e", "--end", help="last time in time series to plot, format=YYYY-mm-ddTHH:MM:SS", required=True)
    parser.add_argument("-c", "--config", help="yaml config file for tracking", required=True)
    parser.add_argument("-p", "--parallel", help="flag to run in parallel (0:serial, 1:parallel)", type=int, default=0)
    parser.add_argument("--extent", nargs='+', help="map extent (lonmin lonmax latmin latmax)", type=float, default=None)
    parser.add_argument("--subset", help="flag to subset data (0:no, 1:yes)", type=int, default=0)
    parser.add_argument("--figsize", nargs='+', help="figure size (width, height) in inches", type=float, default=[10,10])
    parser.add_argument("--output", help="ouput directory", default=None)
    parser.add_argument("--figbasename", help="output figure base name", default="")
    parser.add_argument("--trackstats_file", help="MCS track stats file name", default=None)
    parser.add_argument("--pixel_path", help="Pixel-level tracknumer mask files directory", default=None)
    parser.add_argument("--time_format", help="Pixel-level file datetime format", default=None)
    args = parser.parse_args()

    # Put arguments in a dictionary
    args_dict = {
        'start_datetime': args.start,
        'end_datetime': args.end,
        'run_parallel': args.parallel,
        'config_file': args.config,
        'extent': args.extent,
        'subset': args.subset,
        'figsize': args.figsize,
        'out_dir': args.output,
        'figbasename': args.figbasename,
        'trackstats_file': args.trackstats_file,
        'pixeltracking_path': args.pixel_path,
        'time_format': args.time_format,
    }

    return args_dict

#-----------------------------------------------------------------------
def label_perimeter(tracknumber, dilationstructure):
    """
    Labels the perimeter on a 2D map from object tracknumber masks.
    """
    # Get unique tracknumbers that is no nan
    tracknumber_unique = np.unique(tracknumber[~np.isnan(tracknumber)]).astype(np.int32)

    # Make an array to store the perimeter
    tracknumber_perim = np.zeros(tracknumber.shape, dtype=np.int32)

    # Loop over each tracknumbers
    for ii in tracknumber_unique:
        # Isolate the track mask
        itn = tracknumber == ii
        # Erode the mask by 1 pixel
        itn_erode = binary_erosion(itn, structure=dilationstructure).astype(itn.dtype)
        # Subtract the eroded area to get the perimeter
        iperim = np.logical_xor(itn, itn_erode)
        # Label the perimeter pixels with the track number
        tracknumber_perim[iperim == 1] = ii

    return tracknumber_perim

#-----------------------------------------------------------------------
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    """ 
    Truncate colormap.
    """
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

#-----------------------------------------------------------------------
def get_track_stats(trackstats_file, start_datetime, end_datetime, dt_thres):
    """
    Subset tracks statistics data within start/end datetime

    Args:
        trackstats_file: string
            Track statistics file name.
        start_datetime: string
            Start datetime to subset tracks.
        end_datetime: dstring
            End datetime to subset tracks.
        dt_thres: timedelta
            A timedelta threshold to retain tracks.
            
    Returns:
        track_dict: dictionary
            Dictionary containing track stats data.
    """
    # Read track stats file
    dss = xr.open_dataset(trackstats_file)
    stats_starttime = dss['base_time'].isel(times=0)
    # Convert input datetime to np.datetime64
    stime = np.datetime64(start_datetime)
    etime = np.datetime64(end_datetime)
    time_res = dss.attrs['time_resolution_hour']

    # Find tracks initiated within the time window
    idx = np.where((stats_starttime >= stime) & (stats_starttime <= etime))[0]
    ntracks = len(idx)
    print(f'Number of tracks within input period: {ntracks}')

    # Check if PF variables exist in the stats dataset
    if 'pf_lon' in list(dss.data_vars):
        # 'track_pf_lon': dss['pf_lon'].isel(tracks=idx, nmaxpf=0),
        # 'track_pf_lat': dss['pf_lat'].isel(tracks=idx, nmaxpf=0),
        track_pf_lon = dss['pf_lon_centroid'].isel(tracks=idx, nmaxpf=0)
        track_pf_lat = dss['pf_lat_centroid'].isel(tracks=idx, nmaxpf=0)
        track_pf_diam = 2 * np.sqrt(dss['pf_area'].isel(tracks=idx, nmaxpf=0) / np.pi)
    else:
        # If PF variables do not exist, use CCS variables
        track_pf_lon = dss['meanlon'].isel(tracks=idx)
        track_pf_lat = dss['meanlat'].isel(tracks=idx)
        # Replace all valid values as NaN (PF diameter circles will not be plotted)
        track_pf_diam = track_pf_lon.where(np.isnan(track_pf_lon), other=np.NAN)

    # Subset these tracks and put in a dictionary    
    track_dict = {
        'ntracks': ntracks,
        'lifetime': dss['track_duration'].isel(tracks=idx) * time_res,
        'track_bt': dss['base_time'].isel(tracks=idx),
        'track_ccs_lon': dss['meanlon'].isel(tracks=idx),
        'track_ccs_lat': dss['meanlat'].isel(tracks=idx),
        'track_pf_lon': track_pf_lon,
        'track_pf_lat': track_pf_lat,
        'track_pf_diam': track_pf_diam,
        'dt_thres': dt_thres,
        'time_res': time_res,
    }
    
    return track_dict


#-----------------------------------------------------------------------
def plot_map_1panel(pixel_dict, plot_info, map_info, track_dict):
    """
    Plot Tb, Precipitation and MCS tracks snapshot on a single panel.

    Args:
        pixel_dict: dictionary
            Dictionary containing pixel data variables.
        plot_info: dictionary
            Dictionary containing plotting setup variables.
        map_info: dictionary
            Dictionary containing map boundary info.
        track_dict: dictionary
            Dictionary containing tracking data variables.
            
    Returns:
        fig: object
            Figure handle.
    """

    # Get pixel data from dictionary
    lon = pixel_dict['lon']
    lat = pixel_dict['lat']
    tb = pixel_dict['tb']
    # pcp = pixel_dict['pcp']
    dbz = pixel_dict['dbz']
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
    dbz_alpha = plot_info['dbz_alpha']
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
    map_edgecolor = plot_info['map_edgecolor']
    map_resolution = plot_info['map_resolution']
    timestr = plot_info['timestr']
    figname = plot_info['figname']
    figsize = plot_info['figsize']
    # Map domain, lat/lon ticks, background map features
    map_extent = map_info['map_extent']
    lonv = map_info.get('lonv', None)
    latv = map_info.get('latv', None)
    draw_border = map_info.get('draw_border', False)
    draw_state = map_info.get('draw_state', False)
            
    # Time difference matching pixel-time and track time
    dt_match = 1  # [min]
    
    # Marker style for tracks
    marker_style = dict(edgecolor=trackpath_color, facecolor=trackpath_color, linestyle='-', marker='o')

    # Set up map projection
    proj = ccrs.PlateCarree(central_longitude=180)
    data_proj = ccrs.PlateCarree(central_longitude=0)
    land = cfeature.NaturalEarthFeature('physical', 'land', map_resolution)
    borders = cfeature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land', map_resolution)
    states = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lakes', map_resolution)

    # Set up figure
    mpl.rcParams['font.size'] = fontsize
    # mpl.rcParams['font.family'] = 'Helvetica'
    fig = plt.figure(figsize=figsize, dpi=300, facecolor='w')
    # gs = gridspec.GridSpec(1, 3, height_ratios=[1], width_ratios=[1, 0.03, 0.03])
    # gs.update(wspace=0.2, hspace=0.05, left=0.05, right=0.95, top=0.92, bottom=0.08)
    # ax1 = plt.subplot(gs[0], projection=proj)
    # cax1 = plt.subplot(gs[1])
    # cax2 = plt.subplot(gs[2])

    # Set GridSpec for left (plot) and right (colorbars)
    gs = gridspec.GridSpec(1, 2, height_ratios=[1], width_ratios=[1, 0.1])
    gs.update(wspace=0.05, left=0.05, right=0.95, top=0.92, bottom=0.08)
    # Use GridSpecFromSubplotSpec for panel and colorbar
    gs_cb = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], height_ratios=[1], width_ratios=[0.01,0.01], wspace=5)
    ax1 = plt.subplot(gs[0], projection=proj)
    cax1 = plt.subplot(gs_cb[0])
    cax2 = plt.subplot(gs_cb[1])
    # Figure title: time
    fig.text(0.5, 0.96, timestr, fontsize=fontsize*1.4, ha='center')

    #################################################################
    # Tb Panel
    ax1 = plt.subplot(gs[0,0], projection=proj)
    ax1.set_extent(map_extent, crs=data_proj)
    ax1.add_feature(land, facecolor='none', edgecolor=map_edgecolor, zorder=4)
    if draw_border == True:
        ax1.add_feature(borders, edgecolor=map_edgecolor, facecolor='none', linewidth=0.8, zorder=4)
    if draw_state == True:
        ax1.add_feature(states, edgecolor=map_edgecolor, facecolor='none', linewidth=0.8, zorder=4)
    ax1.set_aspect('auto', adjustable=None)
    ax1.set_title(titles['tb_title'], loc='left')
    gl = ax1.gridlines(crs=data_proj, draw_labels=True, linestyle='--', linewidth=0.5)
    gl.right_labels = False
    gl.top_labels = False
    if (lonv is not None) & (latv is not None):
        gl.xlocator = mpl.ticker.FixedLocator(lonv)
        gl.ylocator = mpl.ticker.FixedLocator(latv)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()        
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)

    # Tb
    cmap = plt.get_cmap(cmaps['tb_cmap'])
    norm = mpl.colors.BoundaryNorm(levels['tb_levels'], ncolors=cmap.N, clip=True)
    tb_masked = np.ma.masked_where((np.isnan(tb)), tb)
    cf1 = ax1.pcolormesh(lon, lat, tb_masked, norm=norm, cmap=cmap, transform=data_proj, zorder=2, alpha=tb_alpha)
    # Overplot cloudtracknumber boundary
    cmap = plt.get_cmap(cmaps['tn_cmap'])
    Tn = np.ma.masked_where(tn_perim == 0, tn_perim)
    tn1 = ax1.pcolormesh(lon, lat, Tn, vmin=min(levels['tn_levels']), vmax=max(levels['tn_levels']),
                         cmap=cmap, transform=data_proj, zorder=4, alpha=1)  
    # Reflectivity
    cmap = plt.get_cmap(cmaps['dbz_cmap'])
    norm = mpl.colors.BoundaryNorm(levels['dbz_levels'], ncolors=cmap.N, clip=True)
    dbz_masked = np.ma.masked_where(((dbz <= min(levels['dbz_levels']))), dbz)
    cf2 = ax1.pcolormesh(lon, lat, dbz_masked, norm=norm, cmap=cmap, transform=data_proj, zorder=3, alpha=dbz_alpha)
    # Tb Colorbar
    cb1 = plt.colorbar(cf1, cax=cax1, label=cblabels['tb_label'], ticks=cbticks['tb_ticks'],
                       extend='both', orientation='vertical')
    # Reflectivity Colorbar
    cb2 = plt.colorbar(cf2, cax=cax2, label=cblabels['dbz_label'], ticks=cbticks['dbz_ticks'],
                       extend='both', orientation='vertical')

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
                               lw=trackpath_linewidth, ls='-', color=trackpath_color, transform=data_proj, zorder=3)
                # Initiation location
                cl1 = ax1.scatter(track_ccs_lon.data[itrack,0], track_ccs_lat.data[itrack,0], s=marker_size*2,
                                  transform=data_proj, zorder=4, **marker_style)
                
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
            # # Plot PF diameter circle
            # if ~np.isnan(_irad):
            #     ipfcircle = ax1.tissot(rad_km=_irad*2, lons=_ilon, lats=_ilat, n_samples=100,
            #                            facecolor='None', edgecolor=pfdiam_color, lw=pfdiam_linewidth, zorder=3)
            # Overplot tracknumbers at current frame
            if (_iccslon > map_extent[0]) & (_iccslon < map_extent[1]) & \
                    (_iccslat > map_extent[2]) & (_iccslat < map_extent[3]):
                ax1.text(_iccslon+0.05, _iccslat+0.05, f'{itracknum:.0f}', color='k', size=tracknumber_fontsize, weight='bold',
                         ha='left', va='center', transform=data_proj, zorder=5)
    
    # Custom legend for track paths
    legend_elements1 = [
        mpl.lines.Line2D([0], [0], color=trackpath_color, marker='o', lw=trackpath_linewidth, label='MCS Tracks'),
        mpl.lines.Line2D([0], [0], marker='o', lw=0, markerfacecolor='None',
                         markeredgecolor=mcsperim_color, markersize=12, label='MCS Mask'),
    ]
    ax1.legend(handles=legend_elements1, loc='lower right')

    # Thread-safe figure output
    canvas = FigureCanvas(fig)
    canvas.print_png(figname)
    fig.savefig(figname)
    
    return fig

#-----------------------------------------------------------------------
def work_for_time_loop(datafile, track_dict, map_info, plot_info, config):
    """
    Work with a pixel-level file.

    Args:
        datafile: string
            Pixel-level file name.
        track_dict: dictionary
            Dictionary containing tracking data variables.
        map_info: dictionary
            Dictionary containing map boundary info.
        figdir: string
            Directory name for figures.
        figbasename: string, optional, default=''
            Base name for figures.
            
    Returns:
        1: success.            
    """

    map_extent = map_info.get('map_extent', None)
    figdir = plot_info.get('figdir')
    figbasename = plot_info.get('figbasename')
    tb_varname = plot_info.get('tb_varname')
    dbz_varname = plot_info.get('dbz_varname')
    perim_thick = plot_info.get('perim_thick')

    # Read pixel-level data
    ds = xr.open_dataset(datafile)
    pixel_bt = ds['time'].data

    # Get map extent from data
    if map_extent is None:
        lonmin = ds['longitude'].min().item()
        lonmax = ds['longitude'].max().item()
        latmin = ds['latitude'].min().item()
        latmax = ds['latitude'].max().item()
        map_extent = [lonmin, lonmax, latmin, latmax]
        map_info['map_extent'] = map_extent
        map_info['subset'] = subset

    # Make dilation structure (larger values make thicker outlines)
    dilationstructure = np.zeros((perim_thick+1,perim_thick+1), dtype=int)
    dilationstructure[1:perim_thick, 1:perim_thick] = 1

    # Get tracknumbers
    # tn = ds['cloudtracknumber'].squeeze()
    # Only plot if there is track in the frame
    # if (np.nanmax(tn) > 0):
        
    # Tracknumber color levels for MCS masks (limit to 256 to fit in a colormap)
    tracknumbers = track_dict['lifetime'].tracks.values
    tn_nlev = np.min([len(tracknumbers), 256])
    tn_levels = np.linspace(np.min(tracknumbers)+1, np.max(tracknumbers)+1, tn_nlev)
    # Add to plot_info dictionary
    plot_info['levels']['tn_levels'] = tn_levels

    # Subset pixel data within the map domain
    if subset == 1:
        lonmin, lonmax = map_extent[0], map_extent[1]
        latmin, latmax = map_extent[2], map_extent[3]
        mask = (ds['longitude'] >= lonmin) & (ds['longitude'] <= lonmax) & \
               (ds['latitude'] >= latmin) & (ds['latitude'] <= latmax)
        tb_sub = ds[tb_varname].where(mask == True, drop=True).squeeze()
        # pcp_sub = ds[pcp_varname].where(mask == True, drop=True).squeeze()
        dbz_sub = ds[dbz_varname].where(mask == True, drop=True).squeeze()
        tracknumber_sub = ds['cloudtracknumber'].where(mask == True, drop=True).squeeze()
        lon_sub = ds['longitude'].where(mask == True, drop=True)
        lat_sub = ds['latitude'].where(mask == True, drop=True)
    else:
        tb_sub = ds[tb_varname].squeeze()
        # pcp_sub = ds[pcp_varname].squeeze()
        dbz_sub = ds[dbz_varname].squeeze()
        tracknumber_sub = ds['cloudtracknumber'].squeeze()
        lon_sub = ds['longitude']
        lat_sub = ds['latitude']
    # Get object perimeters
    tn_perim = label_perimeter(tracknumber_sub.data, dilationstructure)

    # Plotting variables
    fdatetime = pd.to_datetime(ds['time'].data.item()).strftime('%Y%m%d_%H%M%S')
    timestr = pd.to_datetime(ds['time'].data.item()).strftime('%Y-%m-%d %H:%M:%S UTC')
    figname = f'{figdir}{figbasename}{fdatetime}.png'

    # Put pixel data in a dictionary
    pixel_dict = {
        'lon': lon_sub, 
        'lat': lat_sub, 
        'tb': tb_sub, 
        # 'pcp': pcp_sub,
        'dbz': dbz_sub,
        'tracknumber': tracknumber_sub, 
        'tracknumber_perim': tn_perim,
        'pixel_bt': pixel_bt,
    }
    plot_info['timestr'] = timestr
    plot_info['figname'] = figname

    fig = plot_map_1panel(pixel_dict, plot_info, map_info, track_dict)
    plt.close(fig)
    print(figname)

    ds.close()
    return 1



if __name__ == "__main__":

    # Get the command-line arguments...
    args_dict = parse_cmd_args()
    start_datetime = args_dict.get('start_datetime')
    end_datetime = args_dict.get('end_datetime')
    run_parallel = args_dict.get('run_parallel')
    config_file = args_dict.get('config_file')
    map_extent = args_dict.get('extent')
    subset = args_dict.get('subset')
    figsize = args_dict.get('figsize')
    out_dir = args_dict.get('out_dir')
    figbasename = args_dict.get('figbasename')
    trackstats_file = args_dict.get('trackstats_file')
    pixeltracking_path = args_dict.get('pixeltracking_path')
    time_format = args_dict.get('time_format')

    if time_format is None: time_format = "yyyymodd_hhmmss"

    # Specify plotting info
    # Reflectivity color levels
    # dbz_levels = np.arange(0, 60.1, 5)
    dbz_levels = np.arange(10, 70.1, 5)
    dbz_ticks = dbz_levels
    # Tb color levels
    tb_levels = np.arange(170, 300.1, 5)
    tb_ticks = np.arange(170, 300.1, 10)
    levels = {'tb_levels': tb_levels, 'dbz_levels': dbz_levels}
    # Colorbar ticks & labels
    cbticks = {'tb_ticks': tb_ticks, 'dbz_ticks': dbz_ticks}
    cblabels = {'tb_label': 'Tb (K)', 'dbz_label': 'Reflectivity (dBZ)'}
    # Colormaps
    tb_cmap = truncate_colormap(plt.get_cmap('Greys'), minval=0.01, maxval=0.99)
    dbz_cmap = truncate_colormap(plt.get_cmap('gist_ncar'), minval=0.0, maxval=1.0)
    # tn_cmap = cc.cm["glasbey_light"]
    tn_cmap = cc.cm["glasbey_dark"]
    cmaps = {'tb_cmap': tb_cmap, 'dbz_cmap': dbz_cmap, 'tn_cmap': tn_cmap}
    titles = {'tb_title': 'Tb, Reflectivity (Tracked MCSs)'}
    plot_info = {
        'tb_varname': 'tb',
        'dbz_varname': 'reflectivity_comp',
        'levels': levels,
        'cmaps': cmaps,
        'titles': titles,
        'cbticks': cbticks,
        'cblabels': cblabels,
        'dbz_alpha': 0.9,
        'tb_alpha': 0.7,
        'fontsize': 10,
        'marker_size': 10,  # MCS initiation location marker size
        'perim_thick': 6,   # MCS perimeter thickness (larger values make thicker outlines)
        'trackpath_linewidth': 1.5,
        'tracknumber_fontsize': 10,
        'pfdiam_linewidth': 1,
        'trackpath_color': 'purple',
        'mcsperim_color': 'magenta',
        'pfdiam_color': 'magenta',
        'map_edgecolor': 'k',
        'map_resolution': '50m',
        'map_central_lon': 180,
        'figsize': figsize,
        'figbasename': figbasename,
    }

    # Customize lat/lon labels
    lonv = None
    latv = None
    # Put map info in a dictionary
    map_info = {
        'map_extent': map_extent,
        'subset': subset,
        'lonv': lonv,
        'latv': latv,
        'draw_border': False,
        'draw_state': True,
    }

    # Create a timedelta threshold
    # Tracks that end longer than this threshold from the current pixel-level frame are not plotted
    # This treshold controls the time window to retain previous tracks
    dt_thres = datetime.timedelta(hours=1)

    # Track stats file
    config = load_config(config_file)
    stats_path = config["stats_outpath"]
    mcsfinal_filebase = config["mcsfinal_filebase"]
    startdate = config["startdate"]
    enddate = config["enddate"]
    if trackstats_file is None:
        trackstats_file = f"{stats_path}{mcsfinal_filebase}{startdate}_{enddate}.nc"
    if pixeltracking_path is None:
        pixeltracking_path = config["pixeltracking_outpath"]
    pixeltracking_filebase = config["pixeltracking_filebase"]
    n_workers = config["nprocesses"]

    # Output figure directory
    if out_dir is None:
        figdir = f'{pixeltracking_path}quicklooks_trackpaths/'
    else:
        figdir = f'{out_dir}/'
    os.makedirs(figdir, exist_ok=True)
    # Add to plot_info dictionary
    plot_info['figdir'] = figdir

    # Convert datetime string to Epoch time (base time)
    # These are for searching pixel-level files
    start_basetime = pd.to_datetime(start_datetime).timestamp()
    end_basetime = pd.to_datetime(end_datetime).timestamp()
    # Subtract start_datetime by TimeDelta to include tracks 
    # that start before the start_datetime but may not have ended yet
    TimeDelta = pd.Timedelta(days=4)
    start_datetime_4stats = (pd.to_datetime(start_datetime) - TimeDelta).strftime('%Y-%m-%dT%H:%M:%S')

    # Find all pixel-level files that match the input datetime
    datafiles, \
    datafiles_basetime, \
    datafiles_datestring, \
    datafiles_timestring = subset_files_timerange(
        pixeltracking_path,
        pixeltracking_filebase,
        start_basetime,
        end_basetime,
        time_format=time_format,
    )
    print(f'Number of pixel files: {len(datafiles)}')

    # Get track stats data
    track_dict = get_track_stats(trackstats_file, start_datetime_4stats, end_datetime, dt_thres)

    # Serial option
    if run_parallel == 0:
        for ifile in range(len(datafiles)):
            print(datafiles[ifile])
            result = work_for_time_loop(
                datafiles[ifile], track_dict, map_info, plot_info, config,
            )

    # Parallel option
    elif run_parallel == 1:
        # Set Dask temporary directory for workers
        dask_tmp_dir = config.get("dask_tmp_dir", "./")
        dask.config.set({'temporary-directory': dask_tmp_dir})
        # Initialize dask
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
        client = Client(cluster)
        results = []
        for ifile in range(len(datafiles)):
            print(datafiles[ifile])
            result = dask.delayed(work_for_time_loop)(
                datafiles[ifile], track_dict, map_info, plot_info, config,
            )
            results.append(result)

        # Trigger dask computation
        final_result = dask.compute(*results)