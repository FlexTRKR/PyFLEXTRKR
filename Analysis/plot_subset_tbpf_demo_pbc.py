"""
Demonstrates plotting generic tracks on input variable snapshots.

>python plot_subset_generic_tracks_nomap.py -s STARTDATE -e ENDDATE -c CONFIG.yml
Optional arguments:
-p 0 (serial), 1 (parallel)
--extent lonmin lonmax latmin latmax (subset domain boundary)
--subset 0 (no), 1 (yes) (subset data before plotting)
--figsize width height (figure size in inches)
--output output_directory (output figure directory)
--figbasename figure base name (output figure base name)
"""


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
# For non-gui matplotlib back end
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
mpl.use('agg')
import dask
from dask.distributed import Client, LocalCluster
import warnings
warnings.filterwarnings("ignore")
from pyflextrkr.ft_utilities import load_config, subset_files_timerange

#-----------------------------------------------------------------------
def four_floats(value):
    # Split string by ' '
    values = value.split(' ')
    if len(values) != 4:
        raise argparse.ArgumentError
    # Convert list to array and type to float
    values = np.array(values).astype(float)
    return values

def parse_cmd_args():
    # Define and retrieve the command-line arguments...
    parser = argparse.ArgumentParser(
        description="Plot tracks on input data snapshots."
    )
    parser.add_argument("-s", "--start", help="first time in time series to plot, format=YYYY-mm-ddTHH:MM:SS", required=True)
    parser.add_argument("-e", "--end", help="last time in time series to plot, format=YYYY-mm-ddTHH:MM:SS", required=True)
    parser.add_argument("-c", "--config", help="yaml config file for tracking", required=True)
    parser.add_argument("-p", "--parallel", help="flag to run in parallel (0:serial, 1:parallel)", type=int, default=0)
    parser.add_argument("--extent", help="map extent (lonmin lonmax latmin latmax)", type=four_floats, action='store', default=None)
    parser.add_argument("--subset", help="flag to subset data (0:no, 1:yes)", type=int, default=0)
    parser.add_argument("--figsize", nargs='+', help="figure size (width, height) in inches", type=float, default=None)
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

#--------------------------------------------------------------------------
def make_dilation_structure(dilate_radius, dx, dy):
    """
    Make a circular dilation structure

    Args:
        dilate_radius: float
            Dilation radius [kilometer].
        dx: float
            Grid spacing in x-direction [kilometer].
        dy: float
            Grid spacing in y-direction [kilometer].

    Returns:
        struc: np.array
            Dilation structure array.
    """
    # Convert radius to number grids
    rad_gridx = int(dilate_radius / dx)
    rad_gridy = int(dilate_radius / dy)
    xgrd, ygrd = np.ogrid[-rad_gridx:rad_gridx+1, -rad_gridy:rad_gridy+1]
    # Make dilation structure
    strc = xgrd*xgrd + ygrd*ygrd <= (dilate_radius / dx) * (dilate_radius / dy)
    return strc

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
        'track_bt': dss['base_time'].isel(tracks=idx),
        'track_meanlon': dss['meanlon'].isel(tracks=idx) * xscale,
        'track_meanlat': dss['meanlat'].isel(tracks=idx) * yscale,
        'track_pflon': dss['pf_lon'].isel(tracks=idx, nmaxpf=0) * xscale,
        'track_pflat': dss['pf_lat'].isel(tracks=idx, nmaxpf=0) * yscale,
        # 'track_pflon': dss['pf_lon_weightedcentroid'].isel(tracks=idx, nmaxpf=0) * xscale,
        # 'track_pflat': dss['pf_lat_weightedcentroid'].isel(tracks=idx, nmaxpf=0) * yscale,
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
    pcp = pixel_dict['pcp']
    tn_perim = pixel_dict['tn_perim']
    tn = pixel_dict['tn']
    ntracks = track_dict['ntracks']
    lifetime = track_dict['lifetime']
    track_bt = track_dict['track_bt']
    track_meanlon = track_dict['track_meanlon']
    track_meanlat = track_dict['track_meanlat']
    track_pflon = track_dict['track_pflon']
    track_pflat = track_dict['track_pflat']
 
    dt_thres = track_dict['dt_thres']
    time_res = track_dict['time_res']
    # Get plot info from dictionary
    fontsize = plot_info['fontsize']
    levels = plot_info['levels']
    cmaps = plot_info['cmap']
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
    mask_alpha = plot_info.get('mask_alpha', 1)
    perim_plot = plot_info.get('perim_plot', 'pcolormesh')
    perim_linewidth = plot_info.get('perim_linewidth', None)

    # Map domain, lat/lon ticks, background map features
    map_extent = map_info['map_extent']

    # Time difference matching pixel-time and track time
    dt_match = 1  # [min]

    # Marker style for track center
    marker_style = dict(edgecolor=trackpath_color, facecolor=trackpath_color, linestyle='-', marker='o')

    # Set up figure
    mpl.rcParams['font.size'] = fontsize
    fig = plt.figure(figsize=figsize, dpi=200)


    # Set GridSpec for left (plot) and right (colorbars)
    gs = gridspec.GridSpec(1, 2, height_ratios=[1], width_ratios=[1, 0.1])
    gs.update(wspace=0.05, left=0.05, right=0.95, top=0.92, bottom=0.08)
    # Use GridSpecFromSubplotSpec for panel and colorbar
    gs_cb = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], height_ratios=[1], width_ratios=[0.01,0.01], wspace=5)
    ax1 = plt.subplot(gs[0])
    cax1 = plt.subplot(gs_cb[0])
    cax2 = plt.subplot(gs_cb[1])
    # Figure title: time
    fig.suptitle(timestr, fontsize=fontsize*1.2)


    #################################################################
    # Tb Panel
    ax1 = plt.subplot(gs[0,0])
    ax1.set_aspect('auto', adjustable=None)
    ax1.set_title(titles['tb_title'], loc='left')

    # Plot variable Tb
    cmap = plt.get_cmap(cmaps['tb_cmap'])
    norm_ref = mpl.colors.BoundaryNorm(levels['tb_levels'], ncolors=cmap.N, clip=True)
    # Remove out-of-bounds values
    if (remove_oob_low):
        fvar = np.ma.masked_where(fvar < min(levels['tb_levels']), fvar)
    if (remove_oob_high):
        fvar = np.ma.masked_where(fvar > max(levels['tb_levels']), fvar)
    cf1 = ax1.pcolormesh(xx, yy, fvar, norm=norm_ref, cmap=cmap, zorder=2)

    # Overplot tracknumber perimeters
    if perim_plot == 'pcolormesh':
        Tn = np.ma.masked_where(tn_perim == 0, tn_perim)
        Tn[Tn > 0] = 10
        tn1 = ax1.pcolormesh(xx, yy, Tn, cmap='gray', zorder=3, alpha=mask_alpha)
    elif perim_plot == 'contour':
        Tn = np.copy(tn.data)
        Tn[Tn > 0] = 10
        tn1 = ax1.contour(xx, yy, Tn, levels=[9,11], colors='orange', linewidths=perim_linewidth, zorder=3)
    else:
        print(f"ERROR: undefined perim_plot method: {perim_plot}!")
        sys.exit()
    # Set axis
    ax1.set_xlim(map_extent[0], map_extent[1])
    ax1.set_ylim(map_extent[2], map_extent[3])
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)


    # Precipitation
    cmap = plt.get_cmap(cmaps['pcp_cmap'])
    norm = mpl.colors.BoundaryNorm(levels['pcp_levels'], ncolors=cmap.N, clip=True)
    pcp_masked = np.ma.masked_where(((pcp < min(levels['pcp_levels']))), pcp)
    cf2 = ax1.pcolormesh(xx, yy, pcp_masked, norm=norm, cmap=cmap, zorder=2)
    # Tb Colorbar
    cb1 = plt.colorbar(cf1, cax=cax1, label=cblabels['tb_label'], ticks=cbticks['tb_ticks'],
                       extend='both', orientation='vertical')
    # Precipitation Colorbar
    cb2 = plt.colorbar(cf2, cax=cax2, label=cblabels['pcp_label'], ticks=cbticks['pcp_ticks'],
                       extend='both', orientation='vertical')


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
                ### Handle tracks that cross domain for plotting
                # Get adjusted positions
                adjusted_lon = track_meanlon.values[itrack, idx_cut]
                adjusted_lat = track_meanlat.values[itrack, idx_cut]
                adjusted_pflon = track_pflon.values[itrack, idx_cut]
                adjusted_pflat = track_pflat.values[itrack, idx_cut]

                # Wrap positions back into domain for plotting
                wrapped_lon = np.mod(adjusted_lon - map_extent[0], domain_max_x) + map_extent[0]
                wrapped_lat = np.mod(adjusted_lat - map_extent[2], domain_max_y) + map_extent[2]
                wrapped_pflon = np.mod(adjusted_pflon - map_extent[0], domain_max_x) + map_extent[0]
                wrapped_pflat = np.mod(adjusted_pflat - map_extent[2], domain_max_y) + map_extent[2]

                # Identify where wrap-around occurs to split the trajectory
                lon_diff = np.abs(np.diff(wrapped_lon))
                lat_diff = np.abs(np.diff(wrapped_lat))
                wrap_indices = np.where((lon_diff > (domain_max_x / 2)) | (lat_diff > (domain_max_y / 2)))[0] + 1
                pflon_diff = np.abs(np.diff(wrapped_pflon))
                pflat_diff = np.abs(np.diff(wrapped_pflat))
                pfwrap_indices = np.where((pflon_diff > (domain_max_x / 2)) | (pflat_diff > (domain_max_y / 2)))[0] + 1

                # Split the trajectory at wrap-around points
                split_lon = np.split(wrapped_lon, wrap_indices)
                split_lat = np.split(wrapped_lat, wrap_indices)
                split_pflon = np.split(wrapped_pflon, pfwrap_indices)
                split_pflat = np.split(wrapped_pflat, pfwrap_indices)

                # import pdb; pdb.set_trace()

                # Plot each segment separately
                for lon_seg, lat_seg in zip(split_lon, split_lat):
                    # Ensure there are at least two points to plot a line
                    if len(lon_seg) >= 2:
                        ax1.plot(lon_seg, lat_seg, lw=trackpath_linewidth, ls='-', color=trackpath_color, zorder=3)
                    else:
                        # For single points, plot as markers
                        ax1.scatter(lon_seg, lat_seg, s=marker_size, color=trackpath_color, zorder=3,)
                for lon_seg, lat_seg in zip(split_pflon, split_pflat):
                    # Ensure there are at least two points to plot a line
                    # if len(lon_seg) >= 2:
                    #     ax1.plot(lon_seg, lat_seg, lw=trackpath_linewidth, ls='-', color='orange', zorder=3)
                    # else:
                    # For single points, plot as markers
                    ax1.scatter(lon_seg, lat_seg, s=marker_size, marker='^', color='red', zorder=3,)

                # Initiation location (adjusted)
                init_lon = track_meanlon.values[itrack, 0]
                init_lat = track_meanlat.values[itrack, 0]
                wrapped_init_lon = np.mod(init_lon - map_extent[0], domain_max_x) + map_extent[0]
                wrapped_init_lat = np.mod(init_lat - map_extent[2], domain_max_y) + map_extent[2]
                ax1.scatter(wrapped_init_lon, wrapped_init_lat, s=marker_size*2, zorder=4, **marker_style)

        # Find the closest time from track times
        idt = np.abs((ibt - pixel_bt).astype('timedelta64[m]'))
        idx_match = np.argmin(idt)
        idt_match = idt[idx_match]
        # Get track data
        _imeanlon = track_meanlon.data[itrack, idx_match]
        _imeanlat = track_meanlat.data[itrack, idx_match]

        # Adjust positions for plotting
        _iwrapped_meanlon = np.mod(_imeanlon - map_extent[0], domain_max_x) + map_extent[0]
        _iwrapped_meanlat = np.mod(_imeanlat - map_extent[2], domain_max_y) + map_extent[2]

        # Proceed if time difference is < dt_match
        if idt_match < dt_match:
            # Overplot tracknumbers at current frame
            if (_iwrapped_meanlon > map_extent[0]) & (_iwrapped_meanlon < map_extent[1]) & \
                (_iwrapped_meanlat > map_extent[2]) & (_iwrapped_meanlat < map_extent[3]):
                ax1.text(_iwrapped_meanlon+0.02, _iwrapped_meanlat+0.02, f'{itracknum:.0f}',
                            color='r', size=tracknumber_fontsize, weight='bold', ha='left', va='center', zorder=4)
                             
      
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
    perim_thick = plot_info.get('perim_thick')
    figdir = plot_info.get('figdir')
    figbasename = plot_info.get('figbasename')

    # Read pixel-level data
    ds = xr.open_dataset(datafile, mask_and_scale=False)
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

  
    dilationstructure = make_dilation_structure(perim_thick, pixel_radius, pixel_radius)

    # Data variable names
    field_varname = 'tb'



    # Only plot if there is feature in the frame
    # if (np.nanmax(tn) > 0):
    # Subset pixel data within the map domain
    if subset == 1:
        map_extent = map_info['map_extent']
        buffer = 0.0  # buffer area for subset
        lonmin, lonmax = map_extent[0]-buffer, map_extent[1]+buffer
        latmin, latmax = map_extent[2]-buffer, map_extent[3]+buffer
        mask = (ds['longitude'] >= lonmin) & (ds['longitude'] <= lonmax) & \
               (ds['latitude'] >= latmin) & (ds['latitude'] <= latmax)
        fvar = ds[field_varname].where(mask == True, drop=True).squeeze()
        pcp_sub = ds['precipitation'].where(mask == True, drop=True).squeeze()
        tracknumber_sub = ds['cloudtracknumber'].where(mask == True, drop=True).squeeze()
        lon_sub = ds['longitude'].where(mask == True, drop=True).data
        lat_sub = ds['latitude'].where(mask == True, drop=True).data
    else:
        fvar = ds[field_varname].squeeze()
        pcp_sub = ds['precipitation'].squeeze()
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
    timestr = ds['time'].squeeze().dt.strftime("%Y-%m-%d %H:%M:%S UTC").data
    # titles = [timestr]
    fignametimestr = ds['time'].squeeze().dt.strftime("%Y%m%d_%H%M%S").data.item()
    figname = f'{figdir}{figbasename}{fignametimestr}.png'

    # Put variables in dictionaries
    pixel_dict = {
        'pixel_bt': pixel_bt,
        'longitude': lon_sub,
        'latitude': lat_sub,
        'fvar': fvar,
        'pcp': pcp_sub,
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
    trackstats_file = args_dict.get('trackstats_file')
    pixeltracking_path = args_dict.get('pixeltracking_path')
    time_format = args_dict.get('time_format')

    if time_format is None: time_format = "yyyymodd_hhmmss"

    # Determine the figsize based on lat/lon ratio
    if (figsize is None):
        # If map_extent is specified, calculate aspect ratio
        if (map_extent is not None):
            # Get map aspect ratio from map_extent (minlon, maxlon, minlat, maxlat)
            lon_span = map_extent[1] - map_extent[0]
            lat_span = map_extent[3] - map_extent[2]
            fig_ratio_yx = lat_span / lon_span

            figsize_x = 12
            figsize_y = figsize_x * fig_ratio_yx
            figsize_y = float("{:.2f}".format(figsize_y))  # round to 2 decimal digits
            figsize = [figsize_x, figsize_y]
        else:
            figsize = [10, 10]

    # Specify plotting info
    
    # Precipitation color levels
    pcp_levels = [2, 3, 4, 5, 6, 8, 10, 15, 20, 30]
    pcp_ticks = pcp_levels
    # Tb color levels
    tb_levels = np.arange(180, 300.1, 2)
    tb_ticks = np.arange(180, 300.1, 20)
    levels = {'tb_levels': tb_levels, 'pcp_levels': pcp_levels}
    # Colorbar ticks & labels
    cbticks = {'tb_ticks': tb_ticks, 'pcp_ticks': pcp_ticks}
    cblabels = {'tb_label': 'Tb (K)', 'pcp_label': 'Precipitation (mm h$^{-1}$)'}
    # Colormaps
    tb_cmap = 'Greys'
    pcp_cmap = 'YlGnBu'
    cmaps = {'tb_cmap': tb_cmap, 'pcp_cmap': pcp_cmap}
    titles = {'tb_title': 'IR Brightness Temperature, Precipitation, Tracked MCS (Outline)'}




    
    # Scaling factor for x, y coordinates
    xscale = 1 #10
    yscale = 1 #!0
   
    plot_info = {
        'fontsize': 14,     # plot font size
        'cmap': cmaps,
        'levels': levels,
        'cbticks': cbticks, 
        'cblabels': cblabels,
        'tb_alpha': 0.7,
        'pcp_alpha': 0.9,
        'remove_oob_low': True,   # mask out-of-bounds low values (< min(levels))
        'remove_oob_high': False,  # mask out-of-bounds high values (> max(levels))
        'mask_alpha': 0.6,   # transparancy alpha for perimeter mask
        'marker_size': 10,   # track symbol marker size
        'tracknumber_fontsize': 14,
        'perim_plot': 'contour',  # method to plot tracked feature perimeter ('contour', 'pcolormesh')
        'perim_linewidth': 1.5,  # perimeter line width for 'contour' method
        'perim_thick': 2,  # width of the tracked feature perimeter [km]
        'trackpath_linewidth': 1.5, # track path line width
        'trackpath_color': 'blueviolet',    # track path color
        'xlabel': 'X (km)',
        'ylabel': 'Y (km)',
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
        'draw_land': False,
        'draw_border': False,
        'draw_state': False,
    }

    # Track stats file
    config = load_config(config_file)
    stats_path = config["stats_outpath"]
    pixeltracking_path = config["pixeltracking_outpath"]
    pixeltracking_filebase = config["pixeltracking_filebase"]
    trackstats_filebase = config["trackstats_filebase"]
    # finalstats_filebase = config.get("finalstats_filebase", None)
    mcsfinal_filebase = config.get("mcsfinal_filebase", None)
    startdate = config["startdate"]
    enddate = config["enddate"]
    n_workers = config["nprocesses"]
    datatimeresolution = config["datatimeresolution"]  # hour
    pixel_radius = config["pixel_radius"]

    # Tracks that end longer than this threshold from the current pixel-level frame are not plotted
    # This treshold controls the time window to retain previous tracks
    track_retain_time_min = (datatimeresolution * 60)

    # Create a timedelta threshold in minutes
    dt_thres = datetime.timedelta(minutes=track_retain_time_min)

    # If trackstats_file is not specified
    if trackstats_file is None:
        # If finalstats_filebase is present, use it (links merge/split tracks)
        # if finalstats_filebase is None:
        if mcsfinal_filebase is None:
            trackstats_file = f"{stats_path}{trackstats_filebase}{startdate}_{enddate}.nc"
        else:
            trackstats_file = f"{stats_path}{mcsfinal_filebase}{startdate}_{enddate}.nc"
    if pixeltracking_path is None:
        pixeltracking_path = f"{config['root_path']}{config['pixel_path_name']}/{startdate}_{enddate}/"

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
    # Subtract start_datetime by TimeDelta to include tracks
    # that start before the start_datetime but may not have ended yet
    TimeDelta = pd.Timedelta(days=30)
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
        time_format="yyyymodd_hhmmss",
    )
    print(f'Number of pixel files: {len(datafiles)}')

    # Get track stats data
    track_dict = get_track_stats(trackstats_file, start_datetime_4stats, end_datetime, dt_thres)

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
