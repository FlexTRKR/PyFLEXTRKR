"""
Demonstrates ploting cell tracks on radar reflectivity snapshots for a single radar domain.

Usage:
>python plot_subset_cell_tracks_demo.py -s STARTDATE -e ENDDATE -c CONFIG.yml --radar_lat LAT --radar_lon LON

Required arguments:
-s, --start           Start time (format: YYYY-mm-ddTHH:MM:SS)
-e, --end             End time (format: YYYY-mm-ddTHH:MM:SS)
-c, --config          YAML config file for tracking

Optional arguments:
--radar_lat           Radar latitude (defaults to domain center)
--radar_lon           Radar longitude (defaults to domain center)
-p, --parallel        Run in parallel (0:serial, 1:parallel, default=0)
--workers             Number of Dask workers for parallel processing (default=4)
--extent              Map extent: lonmin lonmax latmin latmax
--subset              Subset data before plotting (0:no, 1:yes, default=0)
--figbasename         Output figure base name (default="")
--figsize             Figure size: width height in inches
--figsize_x           Figure width in inches (height auto-calculated, default=10)
--output              Output directory for figures
--time_format         Pixel-level file datetime format (default="yyyymodd_hhmmss")
--varname             Variable name for plotting in pixel files (default="dbz_comp")
--draw_land           Draw land features (0:no, 1:yes, default=0)
--draw_border         Draw country borders (0:no, 1:yes, default=0)
--draw_state          Draw state/province borders (0:no, 1:yes, default=0)
--show_rangecircle    Draw radar range circles (0:no, 1:yes, default=1)
--show_azimuth        Draw azimuth lines (0:no, 1:yes, default=1)
--show_tracks         Show track paths (0:no, 1:yes, default=1)
--show_paths          Show track path lines (0:no, 1:yes, None:auto, default=None)
--show_symbols        Show track centroid symbols (0:no, 1:yes, None:auto, default=None)
--fontsize            Font size for labels and text (default=13)
--fontsize_tracks     Font size for track numbers (default: fontsize*0.8)
--map_resolution      Map resolution for Natural Earth features: '10m', '50m', '110m' (default='50m')
--title_prefix        Prefix string to add to figure title (default="")
"""
__author__ = "Zhe.Feng@pnnl.gov"
__created_date__ = "08-Jun-2022"

import argparse
import numpy as np
import os, sys
import xarray as xr
import pandas as pd
import math
from scipy.ndimage import binary_erosion, generate_binary_structure
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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
from pyflextrkr.ft_utilities import load_config
from pyflextrkr.ft_utilities import load_config, subset_files_timerange

#-----------------------------------------------------------------------
def parse_cmd_args():
    # Define and retrieve the command-line arguments...
    parser = argparse.ArgumentParser(
        description="Plot cell tracks on radar reflectivity snapshots for a user-defined subset domain."
    )
    parser.add_argument("-s", "--start", help="first time in time series to plot, format=YYYY-mm-ddTHH:MM:SS", required=True)
    parser.add_argument("-e", "--end", help="last time in time series to plot, format=YYYY-mm-ddTHH:MM:SS", required=True)
    parser.add_argument("-c", "--config", help="yaml config file for tracking", required=True)
    parser.add_argument("-p", "--parallel", help="flag to run in parallel (0:serial, 1:parallel)", type=int, default=0)
    parser.add_argument("--workers", type=int, help="Number of Dask workers for parallel processing", default=4)
    parser.add_argument("--radar_lat", help="radar latitude (optional, defaults to domain center)", type=float, default=None)
    parser.add_argument("--radar_lon", help="radar longitude (optional, defaults to domain center)", type=float, default=None)
    parser.add_argument("--extent", nargs='+', help="map extent (lonmin, lonmax, latmin, latmax)", type=float, default=None)
    parser.add_argument("--subset", help="flag to subset data (0:no, 1:yes)", type=int, default=0)
    parser.add_argument("--figbasename", help="output figure base name", default="")
    parser.add_argument("--figsize", nargs='+', help="figure size (width, height) in inches", type=float, default=None)
    parser.add_argument("--figsize_x", type=float, help="figure size width in inches", default=10)
    parser.add_argument("--output", help="ouput directory", default=None)
    parser.add_argument("--time_format", help="Pixel-level file datetime format", default="yyyymodd_hhmmss")
    parser.add_argument("--varname", help="Variable name for plotting in pixel files", default="dbz_comp")
    parser.add_argument("--draw_land", type=int, help="Draw land features (0:no, 1:yes)", default=0)
    parser.add_argument("--draw_border", type=int, help="Draw country borders (0:no, 1:yes)", default=0)
    parser.add_argument("--draw_state", type=int, help="Draw state/province borders (0:no, 1:yes)", default=0)
    parser.add_argument("--show_rangecircle", type=int, help="Draw radar range circles (0:no, 1:yes)", default=1)
    parser.add_argument("--show_azimuth", type=int, help="Draw azimuth lines (0:no, 1:yes)", default=1)
    parser.add_argument("--show_tracks", type=int, help="Show track paths (0:no, 1:yes)", default=1)
    parser.add_argument("--show_paths", type=int, help="Show track path lines (0:no, 1:yes, None:auto from show_tracks)", default=None)
    parser.add_argument("--show_symbols", type=int, help="Show track centroid symbols (0:no, 1:yes, None:auto from show_tracks)", default=None)
    parser.add_argument("--fontsize", type=float, help="Font size for labels and text", default=13)
    parser.add_argument("--fontsize_tracks", type=float, help="Font size for track numbers (default: fontsize*0.8)", default=None)
    parser.add_argument("--map_resolution", help="Map resolution for Natural Earth features ('10m', '50m', '110m')", default='50m')
    parser.add_argument("--title_prefix", help="Prefix string to add to figure title", default="")
    args = parser.parse_args()

    # Put arguments in a dictionary
    args_dict = {
        'start_datetime': args.start,
        'end_datetime': args.end,
        'run_parallel': args.parallel,
        'workers': args.workers,
        'config_file': args.config,
        'radar_lat': args.radar_lat,
        'radar_lon': args.radar_lon,
        'extent': args.extent,
        'subset': args.subset,
        'figbasename': args.figbasename,
        'figsize': args.figsize,
        'figsize_x': args.figsize_x,
        'out_dir': args.output,
        'time_format': args.time_format,
        'varname': args.varname,
        'draw_land': args.draw_land,
        'draw_border': args.draw_border,
        'draw_state': args.draw_state,
        'show_rangecircle': args.show_rangecircle,
        'show_azimuth': args.show_azimuth,
        'show_tracks': args.show_tracks,
        'show_paths': args.show_paths,
        'show_symbols': args.show_symbols,
        'fontsize': args.fontsize,
        'fontsize_tracks': args.fontsize_tracks,
        'map_resolution': args.map_resolution,
        'title_prefix': args.title_prefix,
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
        # Isolate the cell mask
        itn = tracknumber == ii
        # Erode the cell by 1 pixel
        itn_erode = binary_erosion(itn, structure=dilationstructure).astype(itn.dtype)
        # Subtract the eroded area to get the perimeter
        iperim = np.logical_xor(itn, itn_erode)
        # Label the perimeter pixels with the track number
        tracknumber_perim[iperim == 1] = ii

    return tracknumber_perim

#-----------------------------------------------------------------------
def calc_cell_center(tracknumber, longitude, latitude, xx, yy):
    """
    Calculates the center location from labeled cells.
    """
    
    # Find unique tracknumbers
    tracknumber_uniqe = np.unique(tracknumber[~np.isnan(tracknumber)])
    num_tracknumber = len(tracknumber_uniqe)
    # Make arrays for cell center locations
    lon_c = np.full(num_tracknumber, np.nan, dtype=float)
    lat_c = np.full(num_tracknumber, np.nan, dtype=float)
    xx_c = np.full(num_tracknumber, np.nan, dtype=float)
    yy_c = np.full(num_tracknumber, np.nan, dtype=float)

    # Loop over each tracknumbers to calculate the mean lat/lon & x/y for their center locations
    for ii, itn in enumerate(tracknumber_uniqe):
        iyy, ixx = np.where(tracknumber == itn)
        lon_c[ii] = np.mean(longitude[tracknumber == itn])
        lat_c[ii] = np.mean(latitude[tracknumber == itn])
        xx_c[ii] = np.mean(xx[ixx])
        yy_c[ii] = np.mean(yy[iyy])
        
    return lon_c, lat_c, xx_c, yy_c, tracknumber_uniqe

#-----------------------------------------------------------------------
def calc_latlon(lon1, lat1, dist, angle):
    """
    Haversine formula to calculate lat/lon locations from distance and angle.
    
    lon1:   longitude in [degree]
    lat1:   latitude in [degree]
    dist:   distance in [km]
    angle:  angle in [degree]
    """

    # Earth radius
    # R_earth = 6378.39  # at Equator [km]
    R_earth = 6374.2  # at 40 degree latitude [km]
#     R_earth = 6356.91  # at the pole [km]

    # Conver degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    bearing = math.radians(angle)

    lat2 = math.asin(math.sin(lat1) * math.cos(dist/R_earth) +
                     math.cos(lat1) * math.sin(dist/R_earth) * math.cos(bearing))
    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(dist/R_earth) * math.cos(lat1),
                             math.cos(dist/R_earth) - math.sin(lat1) * math.sin(lat2))
    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)

    return lon2, lat2

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
    stats_starttime = dss.base_time.isel(times=0)
    # Convert input datetime to np.datetime64
    stime = np.datetime64(start_datetime)
    etime = np.datetime64(end_datetime)
    time_res = dss.attrs['time_resolution_hour']

    # Find track initiated within the time window
    idx = np.where((stats_starttime >= stime) & (stats_starttime <= etime))[0]
    ntracks = len(idx)
    print(f'Number of tracks within input period: {ntracks}')

    # Calculate cell lifetime
    lifetime = dss.track_duration.isel(tracks=idx) * time_res

    # # Select long-lived tracks
    # idx_long = np.where(lifetime > lifetime_longtracks)[0]
    # ntracks_long = len(idx_long)
    # lifetime_long = lifetime.isel(tracks=idx_long)
    # print(f'Number of long tracks within input period: {ntracks_long}')

    # Subset these tracks and put in a dictionary
    track_dict = {
        'ntracks' : ntracks,
        'lifetime' : lifetime,
        'cell_bt' : dss['base_time'].isel(tracks=idx),
        'cell_lon' : dss['cell_meanlon'].isel(tracks=idx),
        'cell_lat' : dss['cell_meanlat'].isel(tracks=idx),
        'start_split_tracknumber' : dss['start_split_tracknumber'].isel(tracks=idx),
        'end_merge_tracknumber' : dss['end_merge_tracknumber'].isel(tracks=idx),
        'dt_thres': dt_thres,
        'time_res': time_res,
        # # Long-lived tracks
        # 'lifetime_long': lifetime_long,
        # 'cell_bt_long' : dss['base_time'].isel(tracks=idx_long),
        # 'cell_lon_long' : dss['cell_meanlon'].isel(tracks=idx_long),
        # 'cell_lat_long' : dss['cell_meanlat'].isel(tracks=idx_long),
        # 'start_split_tracknumber_long' : dss['start_split_tracknumber'].isel(tracks=idx_long),
        # 'end_merge_tracknumber_long' : dss['end_merge_tracknumber'].isel(tracks=idx_long),
    }
    
    return track_dict

#-----------------------------------------------------------------------
def plot_map(pixel_dict, plot_info, map_info, track_dict):
    """
    Plotting function.

    Args:
        pixel_dict: dictionary
            Dictionary containing pixel-level variables
        plot_info: dictionary
            Dictionary containing plotting variables
        map_info: dictionary
            Dictionary containing mapping variables
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

    # Map domain, lat/lon ticks, background map features
    map_extent = map_info['map_extent']
    lonv = map_info['lonv']
    latv = map_info['latv']
    draw_land = map_info.get('draw_land', False)
    draw_border = map_info.get('draw_border', False)
    draw_state = map_info.get('draw_state', False)

    # Set up track lifetime colors
    cmap_lifetime = plt.get_cmap(cmap_tracks)
    norm_lifetime = mpl.colors.BoundaryNorm(lev_lifetime, ncolors=cmap_lifetime.N, clip=True)
    
    # Set up map projection.
    # Detect dateline-crossing by normalizing both extent lons to -180~+180:
    # if lon0_n > lon1_n the extent goes eastward across the dateline.
    # This works for both 0-360 notation (lon_max=220) and -180~+180 notation (lon_max=-140).
    _lon0_n = ((map_extent[0] + 180) % 360) - 180
    _lon1_n = ((map_extent[1] + 180) % 360) - 180
    if _lon0_n > _lon1_n:
        central_longitude = 180
    else:
        central_longitude = 0
    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    # data_proj always PlateCarree(0): data lons are in -180~+180 convention.
    data_proj = ccrs.PlateCarree(central_longitude=0)
    # Compute set_extent bounds in the proj frame (values relative to central_longitude).
    _ext_lon0 = ((map_extent[0] - central_longitude + 180) % 360) - 180
    _ext_lon1 = ((map_extent[1] - central_longitude + 180) % 360) - 180
    land = cfeature.NaturalEarthFeature('physical', 'land', map_resolution)
    borders = cfeature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land', map_resolution)
    states = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lakes', map_resolution)

    # Set up figure
    mpl.rcParams['font.size'] = fontsize
    # mpl.rcParams['font.family'] = 'Helvetica'
    fig = plt.figure(figsize=figsize, dpi=300, facecolor='w')
    gs = gridspec.GridSpec(1,2, height_ratios=[1], width_ratios=[1,0.03])
    gs.update(wspace=0.05, hspace=0.05, left=0.1, right=0.9, top=0.92, bottom=0.08)
    ax1 = plt.subplot(gs[0], projection=proj)
    cax1 = plt.subplot(gs[1])

    ax1.set_extent([_ext_lon0, _ext_lon1, map_extent[2], map_extent[3]], crs=proj)
    ax1.set_aspect('auto', adjustable=None) # Do not auto adjust aspect ratio
    # Add map features
    if draw_land == True:
        ax1.add_feature(land, edgecolor=map_edgecolor, facecolor='none', linewidth=1, zorder=4)
    if draw_border == True:
        ax1.add_feature(borders, edgecolor=map_edgecolor, facecolor='none', linewidth=0.8, zorder=4)
    if draw_state == True:
        ax1.add_feature(states, edgecolor='gray', facecolor='none', linewidth=0.5, zorder=4)
    # Gridlines and lat/lon labels
    gl = ax1.gridlines(crs=proj, draw_labels=True, linestyle='--', linewidth=0.)
    gl.right_labels = False
    gl.top_labels = False
    if (lonv is not None) & (latv is not None):
        gl.xlocator = mpl.ticker.FixedLocator(lonv)
        gl.ylocator = mpl.ticker.FixedLocator(latv)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()        
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
    cf1 = ax1.pcolormesh(xx, yy, var_fill, norm=norm_pcm, cmap=cmap, transform=proj, zorder=2, alpha=shade_alpha)
    # Overplot cell tracknumber perimeters
    Tn = np.ma.masked_where(tn_perim == 0, tn_perim)
    Tn[Tn > 0] = 10
    tn1 = ax1.pcolormesh(xx, yy, Tn, cmap='gray', transform=proj, zorder=3, alpha=mask_alpha)

    # Plot track centroids and paths
    if show_tracks:
        marker_style_s = dict(edgecolor='k', linestyle='-', marker='o')
        marker_style_m = dict(edgecolor='k', linestyle='-', marker='o')
        marker_style_l = dict(edgecolor='k', linestyle='-', marker='o')
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
                        cc = ax1.plot(cell_lon.values[itrack,idx_cut], cell_lat.values[itrack,idx_cut], lw=lw_c, ls='-', color='k', transform=proj, zorder=3)
                    if show_symbols:
                        cl = ax1.scatter(cell_lon.values[itrack,idx_cut], cell_lat.values[itrack,idx_cut], s=size_vals, c=color_vals, 
                                        norm=norm_lifetime, cmap=cmap_lifetime, transform=proj, zorder=4, **marker_style)

        # Plot colorbar for tracks
        if show_symbols:
            cax = inset_axes(ax1, width="100%", height="100%", bbox_to_anchor=(.04, .97, .3, .03), bbox_transform=ax1.transAxes)
            cbinset = mpl.colorbar.ColorbarBase(cax, cmap=cmap_lifetime, norm=norm_lifetime, orientation='horizontal', label=cblabel_tracks)
            cbinset.set_ticks(cbticks_tracks)
            
    # Overplot cell tracknumbers at current frame
    for ii in range(0, len(lon_tn)):
        # Normalize to 0-360 for dateline-safe lon check
        _lon_tn_norm = lon_tn[ii] % 360
        _lonmin_norm = map_extent[0] % 360
        _lonmax_norm = map_extent[1] % 360
        if _lonmin_norm <= _lonmax_norm:
            _in_lon = (_lon_tn_norm >= _lonmin_norm) and (_lon_tn_norm <= _lonmax_norm)
        else:  # domain crosses the dateline
            _in_lon = (_lon_tn_norm >= _lonmin_norm) or (_lon_tn_norm <= _lonmax_norm)
        if _in_lon and (lat_tn[ii] > map_extent[2]) and (lat_tn[ii] < map_extent[3]):
            ax1.text(lon_tn[ii]+0.02, lat_tn[ii]+0.02, f'{tracknumbers[ii]:.0f}', color='k', size=fontsize_tracks,
                    weight='bold', ha='left', va='center', transform=data_proj, zorder=4)

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

    # Thread-safe figure output
    canvas = FigureCanvas(fig)
    canvas.print_png(figname)
    fig.savefig(figname)
    
    return fig

#-----------------------------------------------------------------------
def work_for_time_loop(datafile, track_dict, map_info, plot_info):
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

    Returns:
        1.
    """
    
    map_extent = map_info.get('map_extent', None)
    figdir = plot_info.get('figdir')
    figbasename = plot_info.get('figbasename')
    varname_fill = plot_info.get('varname_fill')
    radar_lat = plot_info.get('radar_lat')
    radar_lon = plot_info.get('radar_lon')

    # Read pixel-level data
    ds = xr.open_dataset(datafile)
    pixel_bt = ds.time.data

    # Get map extent from data
    if map_extent is None:
        lonmin = ds['longitude'].min().item()
        lonmax = ds['longitude'].max().item()
        latmin = ds['latitude'].min().item()
        latmax = ds['latitude'].max().item()
        map_extent = [lonmin, lonmax, latmin, latmax]
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
    # perim_thick = 1
    # dilationstructure = np.zeros((perim_thick+1,perim_thick+1), dtype=int)
    # dilationstructure[1:perim_thick, 1:perim_thick] = 1
    dilationstructure = generate_binary_structure(2,1)

    # Get cell tracknumbers
    # tn = ds['tracknumber'].squeeze()

    # Subset pixel data within the map domain
    if subset == 1:
        lonmin, lonmax = map_extent[0], map_extent[1]
        latmin, latmax = map_extent[2], map_extent[3]
        
        # Use index slicing to subset - works for any 2D lat/lon grid without creating NaNs
        lon_vals = ds['longitude'].values
        lat_vals = ds['latitude'].values
        
        # Find all points within the domain
        mask = ((lon_vals >= lonmin) & (lon_vals <= lonmax) & 
                (lat_vals >= latmin) & (lat_vals <= latmax))
        
        # Find bounding box in index space
        rows, cols = np.where(mask)
        
        if len(rows) > 0:
            row_min, row_max = rows.min(), rows.max() + 1
            col_min, col_max = cols.min(), cols.max() + 1
            
            # Get dimension names
            dim_names = list(ds['longitude'].dims)
            dim_y = dim_names[0]
            dim_x = dim_names[1]
            
            # Subset using index slicing
            lon_sub = ds['longitude'].isel({dim_y: slice(row_min, row_max), dim_x: slice(col_min, col_max)}).squeeze().values
            lat_sub = ds['latitude'].isel({dim_y: slice(row_min, row_max), dim_x: slice(col_min, col_max)}).squeeze().values
            var_fill = ds[varname_fill].isel({dim_y: slice(row_min, row_max), dim_x: slice(col_min, col_max)}).squeeze()
            convmask_sub = ds['conv_mask'].isel({dim_y: slice(row_min, row_max), dim_x: slice(col_min, col_max)}).squeeze()
            tracknumber_sub = ds['tracknumber'].isel({dim_y: slice(row_min, row_max), dim_x: slice(col_min, col_max)}).squeeze()
            xx_sub = ds['lon'].isel({dim_x: slice(col_min, col_max)}).squeeze().values
            yy_sub = ds['lat'].isel({dim_y: slice(row_min, row_max)}).squeeze().values
        else:
            # No valid data in subset domain, use full domain
            xx_sub = ds['lon'].values
            yy_sub = ds['lat'].values
            lon_sub = ds['longitude'].squeeze().values
            lat_sub = ds['latitude'].squeeze().values
            var_fill = ds[varname_fill].squeeze()
            convmask_sub = ds['conv_mask'].squeeze()
            tracknumber_sub = ds['tracknumber'].squeeze()
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
    timestr = ds['time'].squeeze().dt.strftime("%Y-%m-%d %H:%M:%S UTC").data
    fignametimestr = ds['time'].squeeze().dt.strftime("%Y%m%d_%H%M%S").data.item()
    figname = f'{figdir}{figbasename}{fignametimestr}.png'

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



if __name__ == "__main__":

    # Get the command-line arguments...
    args_dict = parse_cmd_args()
    start_datetime = args_dict.get('start_datetime')
    end_datetime = args_dict.get('end_datetime')
    run_parallel = args_dict.get('run_parallel')
    n_workers = args_dict.get('workers')
    config_file = args_dict.get('config_file')
    radar_lat = args_dict.get('radar_lat')
    radar_lon = args_dict.get('radar_lon')
    map_extent = args_dict.get('extent')
    subset = args_dict.get('subset')
    figbasename = args_dict.get('figbasename')
    figsize = args_dict.get('figsize')
    figsize_x = args_dict.get('figsize_x', 10)
    out_dir = args_dict.get('out_dir')
    time_format = args_dict.get('time_format')
    varname_fill = args_dict.get('varname')
    draw_land = args_dict.get('draw_land', 0)
    draw_border = args_dict.get('draw_border', 0)
    draw_state = args_dict.get('draw_state', 0)
    show_rangecircle = args_dict.get('show_rangecircle', 1)
    show_azimuth = args_dict.get('show_azimuth', 1)
    show_tracks = args_dict.get('show_tracks', 1)
    # Set show_paths and show_symbols: if None, default to show_tracks value
    show_paths_arg = args_dict.get('show_paths', None)
    show_symbols_arg = args_dict.get('show_symbols', None)
    show_paths = show_paths_arg if show_paths_arg is not None else show_tracks
    show_symbols = show_symbols_arg if show_symbols_arg is not None else show_tracks
    title_prefix = args_dict.get('title_prefix', '')
    fontsize = args_dict.get('fontsize', 13)
    fontsize_tracks_arg = args_dict.get('fontsize_tracks', None)
    map_resolution = args_dict.get('map_resolution', '50m')
    
    # Determine the figsize based on lat/lon ratio
    if (figsize is None):
        if (map_extent is not None):
            # Calculate aspect ratio from map extent
            lon_span = (map_extent[1] - map_extent[0]) % 360 or 360
            lat_span = map_extent[3] - map_extent[2]
            # Use simple lat/lon ratio (works reasonably well for mid-latitudes)
            aspect_ratio = lat_span / lon_span
            figsize_y = figsize_x * aspect_ratio
            figsize = [figsize_x, figsize_y]
        else:
            # Default figsize if map_extent not provided
            figsize = [figsize_x, figsize_x * 0.9]
    print(f'Figure size (width, height) in inches: {figsize}')

    # Specify plotting info
    # varname_fill = 'dbz_comp'
    # varname_fill = 'echotop10'
    var_scale = 1     # scale factor for the variable
    # var_scale = 1e-3    # scale factor for the variable

    # Colorfill levels
    # levels = np.arange(-10, 60.1, 5)
    levels = np.arange(-10, 70.1, 5)
    # levels = [1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,9,10,12,14,16,18,20]
    lev_lifetime = np.arange(0.5, 4.01, 0.5)
    # Colorbar ticks & labels
    # cbticks = np.arange(-10, 60.1, 5)
    cbticks = levels
    cblabels = 'Composite Reflectivity (dBZ)'
    # cblabels = '10 dBZ ETH (km)'
    cblabel_tracks = 'Lifetime (hour)'
    cbticks_tracks = [1,2,3,4]
    # Colormaps
    cmaps = 'gist_ncar'     # Reflectivity
    # cmaps = 'nipy_spectral'     # Echo-top Height
    cmap_tracks = 'Spectral_r'  # Lifetime
    # Set fontsize_tracks: if None, default to fontsize * 0.8
    fontsize_tracks = fontsize_tracks_arg if fontsize_tracks_arg is not None else fontsize * 0.8
    
    # Put plot specifications in a dictionary
    plot_info = {
        'varname_fill': varname_fill,
        'var_scale': var_scale,
        'levels': levels,
        'lev_lifetime': lev_lifetime,
        'cbticks': cbticks,
        'cbticks_tracks': cbticks_tracks,
        'cblabels': cblabels,
        'cblabel_tracks': cblabel_tracks,
        'fontsize': fontsize,
        'fontsize_tracks': fontsize_tracks,
        'shade_alpha': 0.85,    # transparancy alpha for shading
        'mask_alpha': 0.6,   # transparancy alpha for cell perimeter mask
        'cmaps': cmaps,
        'cmap_tracks': cmap_tracks,
        'show_tracks': bool(show_tracks),
        'show_paths': bool(show_paths),
        'show_symbols': bool(show_symbols),
        'marker_size': [10,10,10],    # track centroid marker size (short, medium, long lived)
        'lw_centroid': [1,1,1],         # track path line width
        'radii': np.arange(20,101,20),  # radii for the radar range rings [km]
        'azimuths': np.arange(0,361,90),   # azimuth angles for HSRHI scans [degree]
        'map_edgecolor': 'k',
        'map_resolution': map_resolution,
        'figbasename': figbasename,
        'figsize': figsize,
        'radar_lon': radar_lon,
        'radar_lat': radar_lat,
        'show_rangecircle': bool(show_rangecircle),
        'show_azimuth': bool(show_azimuth),
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
        'draw_land': bool(draw_land),
        'draw_border': bool(draw_border),
        'draw_state': bool(draw_state),
    }

    # Tracks that end longer than this threshold from the current pixel-level frame are not plotted
    # This treshold controls the time window to retain previous tracks
    track_retain_time_min = 15

    # Create a timedelta threshold in minutes
    dt_thres = datetime.timedelta(minutes=track_retain_time_min)

    # Track stats file
    config = load_config(config_file)
    stats_path = config["stats_outpath"]
    pixeltracking_path = config["pixeltracking_outpath"]
    pixeltracking_filebase = config["pixeltracking_filebase"]
    trackstats_filebase = config["trackstats_filebase"]
    startdate = config["startdate"]
    enddate = config["enddate"]
    radar_sensitivity = config.get("radar_sensitivity", 0)
    plot_info['radar_sensitivity'] = radar_sensitivity

    trackstats_file = f"{stats_path}{trackstats_filebase}{startdate}_{enddate}.nc"
    # Use n_workers from command-line if provided, otherwise use config file
    if n_workers is None:
        n_workers = config["nprocesses"]
  
    # Convert datetime string to Epoch time (base time)
    start_basetime = pd.to_datetime(start_datetime).timestamp()
    end_basetime = pd.to_datetime(end_datetime).timestamp()

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

    # Output figure directory
    if out_dir is None:
        figdir = f'{pixeltracking_path}quicklooks_trackpaths/'
    else:
        figdir = out_dir
    os.makedirs(figdir, exist_ok=True)
    # Add to plot_info dictionary
    plot_info['figdir'] = figdir

    # Get track stats data
    track_dict = get_track_stats(trackstats_file, start_datetime, end_datetime, dt_thres)

    # Serial option
    if run_parallel == 0:
        for ifile in range(len(datafiles)):
            print(datafiles[ifile])
            result = work_for_time_loop(datafiles[ifile], track_dict, map_info, plot_info)

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
            result = dask.delayed(work_for_time_loop)(datafiles[ifile], track_dict, map_info, plot_info)
            results.append(result)

        # Trigger dask computation
        final_result = dask.compute(*results)

        cluster.close()
        client.close()    
    else:
        sys.exit('Valid parallelization flag not provided')
