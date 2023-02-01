"""
Demonstrates plotting generic tracks on input variable snapshots.

>python plot_subset_generic_tracks_demo.py -s STARTDATE -e ENDDATE -c CONFIG.yml
Optional arguments:
-p 0 (serial), 1 (parallel)
--extent lonmin lonmax latmin latmax (subset domain boundary)
--subset 0 (no), 1 (yes) (subset data before plotting)
--figsize width height (figure size in inches)
--output output_directory (output figure directory)
--figbasename figure base name (output figure base name)
"""
__author__ = "Zhe.Feng@pnnl.gov"
__created_date__ = "26-Jan-2023"

import argparse
import numpy as np
import os, sys
import xarray as xr
import pandas as pd
from scipy.ndimage import binary_erosion, generate_binary_structure
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
        description="Plot tracks on input data snapshots."
    )
    parser.add_argument("-s", "--start", help="first time in time series to plot, format=YYYY-mm-ddTHH:MM:SS", required=True)
    parser.add_argument("-e", "--end", help="last time in time series to plot, format=YYYY-mm-ddTHH:MM:SS", required=True)
    parser.add_argument("-c", "--config", help="yaml config file for tracking", required=True)
    parser.add_argument("-p", "--parallel", help="flag to run in parallel (0:serial, 1:parallel)", type=int, default=0)
    parser.add_argument("--extent", nargs='+', help="map extent (lonmin, lonmax, latmin, latmax)", type=float, default=None)
    parser.add_argument("--subset", help="flag to subset data (0:no, 1:yes)", type=int, default=0)
    parser.add_argument("--figsize", nargs='+', help="figure size (width, height) in inches", type=float, default=[8,7])
    parser.add_argument("--output", help="ouput directory", default=None)
    parser.add_argument("--figbasename", help="output figure base name", default="")
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
def calc_track_center(tracknumber, longitude, latitude):
    """
    Calculates the center location from labeled tracks.
    """
    
    # Find unique tracknumbers
    tracknumber_uniqe = np.unique(tracknumber[~np.isnan(tracknumber)])
    num_tracknumber = len(tracknumber_uniqe)
    # Make arrays for track center locations
    lon_c = np.full(num_tracknumber, np.nan, dtype=float)
    lat_c = np.full(num_tracknumber, np.nan, dtype=float)

    # Loop over each tracknumbers to calculate the mean lat/lon & x/y for their center locations
    for ii, itn in enumerate(tracknumber_uniqe):
        iyy, ixx = np.where(tracknumber == itn)
        lon_c[ii] = np.mean(longitude[tracknumber == itn])
        lat_c[ii] = np.mean(latitude[tracknumber == itn])
        
    return lon_c, lat_c, tracknumber_uniqe

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

    # Calculate track lifetime
    lifetime = dss.track_duration.isel(tracks=idx) * time_res

    # Subset these tracks and put in a dictionary
    track_dict = {
        'ntracks': ntracks,
        'lifetime': lifetime,
        'base_time': dss['base_time'].isel(tracks=idx),
        'meanlon': dss['meanlon'].isel(tracks=idx),
        'meanlat': dss['meanlat'].isel(tracks=idx),
        # 'start_split_tracknumber': dss['start_split_tracknumber'].isel(tracks=idx),
        # 'end_merge_tracknumber': dss['end_merge_tracknumber'].isel(tracks=idx),
        'dt_thres': dt_thres,
        'time_res': time_res,
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
    cblabels = plot_info['cblabels']
    cbticks = plot_info['cbticks']
    marker_size = plot_info['marker_size']
    trackpath_linewidth = plot_info['trackpath_linewidth']
    trackpath_color = plot_info['trackpath_color']
    map_edgecolor = plot_info['map_edgecolor']
    map_resolution = plot_info['map_resolution']
    map_central_lon = plot_info['map_central_lon']
    timestr = plot_info['timestr']
    figname = plot_info['figname']
    figsize = plot_info['figsize']
    dt_thres = track_dict['dt_thres']
    time_res = track_dict['time_res']
    # Map domain, lat/lon ticks, background map features
    map_extent = map_info['map_extent']
    lonv = map_info['lonv']
    latv = map_info['latv']
    draw_border = map_info.get('draw_border', False)
    draw_state = map_info.get('draw_state', False)

    # Marker style for track center
    marker_style = dict(edgecolor=trackpath_color, facecolor=trackpath_color, linestyle='-', marker='o')

    # Set up map projection
    proj = ccrs.PlateCarree(central_longitude=map_central_lon)
    data_proj = ccrs.PlateCarree()
    land = cfeature.NaturalEarthFeature('physical', 'land', map_resolution)
    borders = cfeature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land', map_resolution)
    states = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lakes', map_resolution)

    # Set up figure
    mpl.rcParams['font.size'] = fontsize
    mpl.rcParams['font.family'] = 'Helvetica'
    fig = plt.figure(figsize=figsize, dpi=200)
    gs = gridspec.GridSpec(1,2, height_ratios=[1], width_ratios=[1,0.03])
    gs.update(wspace=0.05, hspace=0.05, left=0.1, right=0.9, top=0.92, bottom=0.08)
    ax1 = plt.subplot(gs[0], projection=proj)
    cax1 = plt.subplot(gs[1])
    # Plot background map elements
    ax1.set_extent(map_extent, crs=data_proj)
    ax1.set_aspect('auto', adjustable=None)
    ax1.add_feature(land, facecolor='none', edgecolor=map_edgecolor, zorder=3)
    if draw_border == True:
        ax1.add_feature(borders, edgecolor=map_edgecolor, facecolor='none', linewidth=0.8, zorder=3)
    if draw_state == True:
        ax1.add_feature(states, edgecolor=map_edgecolor, facecolor='none', linewidth=0.8, zorder=3)
    # Grid lines
    gl = ax1.gridlines(crs=data_proj, draw_labels=True, linestyle='--', linewidth=0.)
    gl.right_labels = False
    gl.top_labels = False
    if (lonv is not None) & (latv is not None):
        gl.xlocator = mpl.ticker.FixedLocator(lonv)
        gl.ylocator = mpl.ticker.FixedLocator(latv)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()        
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)

    # Plot variable
    cmap = plt.get_cmap(cmap)
    norm_ref = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    # fvar = np.ma.masked_where(fvar < min(levels), fvar)
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
                              lw=trackpath_linewidth, ls='-', color=trackpath_color, transform=data_proj, zorder=3)
                cl = ax1.scatter(meanlon.values[itrack,idx_cut], meanlat.values[itrack,idx_cut],
                                 s=size_vals, transform=data_proj, zorder=4, **marker_style)

    # Overplot tracknumbers at current frame
    for ii in range(0, len(lon_tn)):
        ax1.text(lon_tn[ii]+0.02, lat_tn[ii]+0.02, f'{tracknumbers[ii]:.0f}', color='k', size=10, 
                 weight='bold', ha='left', va='center', transform=data_proj, zorder=4)

    # Thread-safe figure output
    canvas = FigureCanvas(fig)
    canvas.print_png(figname)
    fig.savefig(figname)
    
    return fig

#-----------------------------------------------------------------------
def work_for_time_loop(datafile, track_dict, map_info, plot_info, config):
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
            Dictionary containing config parameters

    Returns:
        1.
    """
    
    map_extent = map_info.get('map_extent', None)
    figdir = plot_info.get('figdir')
    figbasename = plot_info.get('figbasename')

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
        map_info['subset'] = subset

    # Make dilation structure (larger values make thicker outlines)
    # perim_thick = 1
    # dilationstructure = np.zeros((perim_thick+1,perim_thick+1), dtype=int)
    # dilationstructure[1:perim_thick, 1:perim_thick] = 1
    dilationstructure = generate_binary_structure(2,1)

    # Data variable names
    field_varname = config["field_varname"]

    # Get tracknumbers
    # tn = ds['cloudtracknumber'].squeeze()

    # Only plot if there is feature in the frame
    # if (np.nanmax(tn) > 0):
    # Subset pixel data within the map domain
    if subset == 1:
        map_extent = map_info['map_extent']
        buffer = 0.05  # buffer area for subset
        lonmin, lonmax = map_extent[0]-buffer, map_extent[1]+buffer
        latmin, latmax = map_extent[2]-buffer, map_extent[3]+buffer
        mask = (ds['longitude'] >= lonmin) & (ds['longitude'] <= lonmax) & (ds['latitude'] >= latmin) & (ds['latitude'] <= latmax)
        xmin = mask.where(mask == True, drop=True).lon.min().item()
        xmax = mask.where(mask == True, drop=True).lon.max().item()
        ymin = mask.where(mask == True, drop=True).lat.min().item()
        ymax = mask.where(mask == True, drop=True).lat.max().item()
        lon_sub = ds['longitude'].isel(lon=slice(xmin,xmax), lat=slice(ymin,ymax)).data
        lat_sub = ds['latitude'].isel(lon=slice(xmin,xmax), lat=slice(ymin,ymax)).data
        fvar = ds[field_varname].isel(lon=slice(xmin,xmax), lat=slice(ymin,ymax)).squeeze()
        tracknumber_sub = ds['cloudtracknumber'].isel(lon=slice(xmin,xmax), lat=slice(ymin,ymax)).squeeze()
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
    timestr = ds['time'].squeeze().dt.strftime("%Y-%m-%d %H:%M UTC").data
    # titles = [timestr]
    fignametimestr = ds['time'].squeeze().dt.strftime("%Y%m%d_%H%M").data.item()
    figname = f'{figdir}{figbasename}{fignametimestr}.png'

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

    # Specify plotting info
    plot_info = {
        'fontsize': 13,     # plot font size
        'cmap': 'RdBu_r',   # colormap
        'levels': np.arange(-4, 4.01, 0.2),  # shading levels
        'cbticks': np.arange(-4, 4.01, 1),  # colorbar ticks
        'cblabels': 'Z500 Anomaly (m$^{2}$ s$^{-1}$)',  # colorbar label
        'marker_size': 8,   # track symbol marker size
        'trackpath_linewidth': 1.5, # track path line width
        'trackpath_color': 'blueviolet',    # track path color
        'map_edgecolor': 'gray',    # background map edge color
        'map_resolution': '110m',   # background map resolution ('110m', '50m', 10m')
        'map_central_lon': 180,     # map projection central longitude (for global map)
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
        'draw_state': False,
    }

    # Tracks that end longer than this threshold from the current pixel-level frame are not plotted
    # This treshold controls the time window to retain previous tracks
    track_retain_time_min = 60

    # Create a timedelta threshold in minutes
    dt_thres = datetime.timedelta(minutes=track_retain_time_min)

    # Track stats file
    config = load_config(config_file)
    stats_path = config["stats_outpath"]
    pixeltracking_path = config["pixeltracking_outpath"]
    pixeltracking_filebase = config["pixeltracking_filebase"]
    trackstats_filebase = config["trackstats_filebase"]
    finalstats_filebase = config.get("finalstats_filebase", None)
    startdate = config["startdate"]
    enddate = config["enddate"]
    n_workers = config["nprocesses"]

    # If finalstats_filebase is present, use it (links merge/split tracks)
    if finalstats_filebase is None:
        trackstats_file = f"{stats_path}{trackstats_filebase}{startdate}_{enddate}.nc"
    else:
        trackstats_file = f"{stats_path}{finalstats_filebase}{startdate}_{enddate}.nc"

    # Output figure directory
    if out_dir is None:
        figdir = f'{pixeltracking_path}quicklooks_trackpaths/'
    else:
        figdir = out_dir
    os.makedirs(figdir, exist_ok=True)
    # Add to plot_info dictionary
    plot_info['figdir'] = figdir

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
        time_format="yyyymodd_hhmm",
    )
    print(f'Number of pixel files: {len(datafiles)}')

    # Get track stats data
    track_dict = get_track_stats(trackstats_file, start_datetime, end_datetime, dt_thres)

    # Serial option
    if run_parallel == 0:
        for ifile in range(len(datafiles)):
            print(datafiles[ifile])
            result = work_for_time_loop(datafiles[ifile], track_dict, map_info, plot_info, config)

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
            result = dask.delayed(work_for_time_loop)(datafiles[ifile], track_dict, map_info, plot_info, config)
            results.append(result)

        # Trigger dask computation
        final_result = dask.compute(*results)
    
    else:
        sys.exit('Valid parallelization flag not provided')
