"""
Demonstrates ploting MCS tracks on Tb, precipitation snapshots for SAAG.

Zhe Feng, PNNL
contact: Zhe.Feng@pnnl.gov
"""

import numpy as np
import glob, os, sys
import xarray as xr
import pandas as pd
from scipy.ndimage import binary_erosion
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
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    """ 
    Truncate colormap.
    """
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

#-----------------------------------------------------------------------
def plot_map_2panels(lon, lat, dataarray, tn_perim, pixel_bt, levels, cmaps, titles, cblabels, cbticks, timestr, dt_thres, 
                     ntracks, lifetime, track_bt, track_ccs_lon, track_ccs_lat, track_pf_lon, track_pf_lat, track_pf_diam, figname):
    
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['font.family'] = 'Helvetica'
    
    # Time difference matching pixel-time and track time
    dt_match = 1  # [min]
    
    # Size, line width for tracks
    size_c = 10
    lw_c = 1.5
    lw_r = 1
    marker_style = dict(edgecolor='k', linestyle='-', marker='o')
    cmap_tracks = 'Spectral_r'
    cblabel_tracks = 'Lifetime (hour)'
    lev_lifetime = np.arange(5, 25.1, 5)
    cbticks_tracks = np.arange(5, 25.1, 5)
    cmap_lifetime = plt.get_cmap(cmap_tracks)
    norm_lifetime = mpl.colors.BoundaryNorm(lev_lifetime, ncolors=cmap_lifetime.N, clip=True)

    # Map domain, lat/lon ticks, background map features
    map_extend = [-82, -34, -56, 13]
    lonv = np.arange(-80,-30.1,10)
    latv = np.arange(-50,10.1,10)
    proj = ccrs.PlateCarree()
    levelshgt = [1000,6000]
    resolution = '50m'
    land = cfeature.NaturalEarthFeature('physical', 'land', resolution)
    ocean = cfeature.NaturalEarthFeature('physical', 'ocean', resolution)
    borders = cfeature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land', resolution)

    # Set up figure
    fig = plt.figure(figsize=[10,8], dpi=300, facecolor='w')
    gs = gridspec.GridSpec(2,2, height_ratios=[1,0.02], width_ratios=[1,1])
    gs.update(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.05, hspace=0.1)
    # Figure title: time
    fig.text(0.5, 0.94, timestr, fontsize=14, ha='center')

    #################################################################
    # Tb Panel
    ax1 = plt.subplot(gs[0,0], projection=proj)
    ax1.set_extent(map_extend, crs=proj)
    ax1.add_feature(borders, edgecolor='k', facecolor='none', linewidth=0.8, zorder=3)
    ax1.add_feature(land, facecolor='none', edgecolor='k', zorder=3)
    ax1.set_aspect('auto', adjustable=None)
    ax1.set_title(titles[0], loc='left')
    gl = ax1.gridlines(crs=proj, draw_labels=False, linestyle='--', linewidth=0.5)
    gl.xlocator = mpl.ticker.FixedLocator(lonv)
    gl.ylocator = mpl.ticker.FixedLocator(latv)        
    ax1.set_xticks(lonv, crs=ccrs.PlateCarree())
    ax1.set_yticks(latv, crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()        
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)

    cmap = plt.get_cmap(cmaps[0])
    norm = mpl.colors.BoundaryNorm(levels[0], ncolors=cmap.N, clip=True)
    data = dataarray[0]
    Zm = np.ma.masked_where((np.isnan(data)), data)
    cf1 = ax1.pcolormesh(lon, lat, Zm, norm=norm, cmap=cmap, transform=ccrs.PlateCarree(), zorder=2)
    # Overplot cloudtracknumber boundary
    tn = np.copy(dataarray[2].data)
    # Replace all valid cloudtracknumber with a constant, and invalid cloudtracknumber with 0
    tn[(tn >= 1)] = 10
    tn[np.isnan(tn)] = 0
    # Overlay boundary of cloudtracknumber on Tb
    # tn1 = ax1.contour(lon, lat, tn, colors='magenta', linewidths=1, alpha=0.5, transform=ccrs.PlateCarree(), zorder=5)
    # Overplot tracknumber perimeters
    Tn = np.ma.masked_where(tn_perim == 0, tn_perim)
    Tn[Tn > 0] = 10
    tn1 = ax1.pcolormesh(lon, lat, Tn, cmap='cool_r', transform=proj, zorder=3)
    # Tb Colorbar
    cax1 = plt.subplot(gs[1,0])
    cb1 = plt.colorbar(cf1, cax=cax1, label=cblabels[0], ticks=cbticks[0], extend='both', orientation='horizontal')
    # Terrain height
    # ct = ax1.contour(lon_ter, lat_ter, ter, levels=levelshgt, colors='dimgray', linewidths=1, transform=proj, zorder=3)
    
    #################################################################
    # Precipitation Panel
    ax2 = plt.subplot(gs[0,1], projection=ccrs.PlateCarree())
    ax2.set_extent(map_extend, crs=ccrs.PlateCarree())
    ax2.add_feature(borders, edgecolor='k', facecolor='none', linewidth=0.8, zorder=3)
    ax2.add_feature(land, facecolor='none', edgecolor='k', zorder=3)
    ax2.set_aspect('auto', adjustable=None)
    ax2.set_title(titles[1], loc='left')
    gl = ax2.gridlines(crs=proj, draw_labels=False, linestyle='--', linewidth=0.5)
    gl.xlocator = mpl.ticker.FixedLocator(lonv)
    gl.ylocator = mpl.ticker.FixedLocator(latv)        
    ax2.set_xticks(lonv, crs=ccrs.PlateCarree())
    # ax2.set_yticks(latv, crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()        
    ax2.xaxis.set_major_formatter(lon_formatter)
    # ax2.yaxis.set_major_formatter(lat_formatter)

    # MCS track number mask
    cmap = plt.get_cmap(cmaps[2])
    data = dataarray[2]
    norm = mpl.colors.BoundaryNorm(levels[2], ncolors=cmap.N, clip=True)
    Zm = np.ma.masked_invalid(data)
    cm1 = ax2.pcolormesh(lon, lat, Zm, norm=norm, cmap=cmap, transform=ccrs.PlateCarree(), zorder=2, alpha=0.7)
    
    # Precipitation
    cmap = plt.get_cmap(cmaps[1])
    norm = mpl.colors.BoundaryNorm(levels[1], ncolors=cmap.N, clip=True)
    data = dataarray[1]
    Zm = np.ma.masked_where(((data < 2)), data)
    cf2 = ax2.pcolormesh(lon, lat, Zm, norm=norm, cmap=cmap, transform=ccrs.PlateCarree(), zorder=2)
    # Colorbar
    cax2 = plt.subplot(gs[1,1])
    cb2 = plt.colorbar(cf2, cax=cax2, label=cblabels[1], ticks=cbticks[1], extend='both', orientation='horizontal')
    # Terrain height
    # ct = ax2.contour(lon_ter, lat_ter, ter, levels=levelshgt, colors='dimgray', linewidths=1, transform=proj, zorder=3)
    
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
                color_vals = np.repeat(ilifetime, idur_cut)
                size_vals = np.repeat(size_c, idur_cut)
                size_vals[0] = size_c * 2
                cc1 = ax1.plot(track_ccs_lon.data[itrack,idx_cut], track_ccs_lat.data[itrack,idx_cut], lw=lw_c, ls='-', color='k', transform=proj, zorder=3)
                cc2 = ax2.plot(track_ccs_lon.data[itrack,idx_cut], track_ccs_lat.data[itrack,idx_cut], lw=lw_c, ls='-', color='k', transform=proj, zorder=3)
                # cc2 = ax2.plot(track_pf_lon.data[itrack,idx_cut], track_pf_lat.data[itrack,idx_cut], lw=lw_c, ls='-', color='k', transform=proj, zorder=3)
                # Initiation location
                cl1 = ax1.scatter(track_ccs_lon.data[itrack,0], track_ccs_lat.data[itrack,0], s=size_c*2, c='k', transform=proj, zorder=4, **marker_style)
                cl2 = ax2.scatter(track_ccs_lon.data[itrack,0], track_ccs_lat.data[itrack,0], s=size_c*2, c='k', transform=proj, zorder=4, **marker_style)
                # cl = ax2.scatter(track_pf_lon.data[itrack,idx_cut], track_pf_lat.data[itrack,idx_cut], s=size_vals, c=color_vals, 
                #                  norm=norm_lifetime, cmap=cmap_lifetime, transform=proj, zorder=4, **marker_style)
                
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
            # Plot PF diameter circle
            if ~np.isnan(_irad):
                ipfcircle = ax2.tissot(rad_km=_irad*2, lons=_ilon, lats=_ilat, n_samples=100, facecolor='None', edgecolor='magenta', lw=lw_r, zorder=3)
            # Overplot tracknumbers at current frame
            ax1.text(_iccslon+0.05, _iccslat+0.05, f'{itracknum:.0f}', color='k', size=8, weight='bold', ha='left', va='center', transform=proj, zorder=3)
            ax2.text(_iccslon+0.05, _iccslat+0.05, f'{itracknum:.0f}', color='k', size=8, weight='bold', ha='left', va='center', transform=proj, zorder=3)

    # Custom legend
    legend_elements1 = [
        mpl.lines.Line2D([0], [0], color='k', marker='o', lw=lw_c, label='MCS Tracks'),
        mpl.lines.Line2D([0], [0], marker='o', lw=0, markerfacecolor='None', markeredgecolor='magenta', markersize=12, label='MCS Mask'),
    ]
    legend_elements2 = [
        mpl.lines.Line2D([0], [0], color='k', marker='o', lw=lw_c, label='MCS Tracks'),
        mpl.lines.Line2D([0], [0], marker='o', lw=0, markerfacecolor='None', markeredgecolor='magenta', markersize=12, label='PF Diam (x2)'),
    ]
    ax1.legend(handles=legend_elements1, fontsize=10, loc='lower right')
    ax2.legend(handles=legend_elements2, fontsize=10, loc='lower right')
    
    # fig.savefig(figname, dpi=300, bbox_inches='tight', facecolor='w')
    # Thread-safe figure output
    canvas = FigureCanvas(fig)
    canvas.print_png(figname)
    fig.savefig(figname)
    
    return fig

#-----------------------------------------------------------------------
def work_for_time_loop(datafile, ntracks, lifetime, track_bt, track_ccs_lon, track_ccs_lat, track_pf_lon, track_pf_lat, track_pf_diam, dt_thres, figdir):

    # Read pixel-level data
    ds = xr.open_dataset(datafile)
    longitude = ds['longitude'].data
    latitude = ds['latitude'].data
    pixel_bt = ds['time'].data
    
    # Make dilation structure (larger values make thicker outlines)
    perim_thick = 6
    dilationstructure = np.zeros((perim_thick+1,perim_thick+1), dtype=int)
    dilationstructure[1:perim_thick, 1:perim_thick] = 1

    # Get tracknumbers
    tn = ds['cloudtracknumber'].squeeze()
    # Only plot if there is track in the frame
    if (np.nanmax(tn) > 0):

        # Get object perimeters
        tn_perim = label_perimeter(tn.data, dilationstructure)
        
        # Precipitation color levels
        pcplev = [2,3,4,5,6,8,10,15]
        # Track number color levels for MCS masks
        tracknumbers = lifetime.tracks
        tnlev = np.arange(tracknumbers.min()+1, tracknumbers.max()+1, 1)
        levels = [np.arange(200, 320.1, 10), pcplev, tnlev]
        cbticks = [np.arange(200, 320.1, 20), pcplev]
        cblabels = ['Tb (K)', 'Precipitation (mm h$^{-1}$)']
        # Truncate colormaps
        cmap_tb = truncate_colormap(plt.get_cmap('jet'), minval=0.05, maxval=0.95)
        cmap_pcp = truncate_colormap(plt.get_cmap('viridis'), minval=0.2, maxval=1.0)
        cmap_mcs = truncate_colormap(plt.get_cmap('jet_r'), minval=0.05, maxval=0.95)
        cmaps = [cmap_tb, cmap_pcp, cmap_mcs]
        titles = ['(a) IR Brightness Temperature','(b) Precipitation (Tracked MCSs Shaded)']

        dataarr = [ds['tb'].squeeze(), ds['precipitation'].squeeze(), ds['cloudtracknumber'].squeeze()]
        fdatetime = pd.to_datetime(ds['time'].data.item()).strftime('%Y%m%d_%H%M')
        timestr = pd.to_datetime(ds['time'].data.item()).strftime('%Y-%m-%d %H:%M UTC')
        figname = f'{figdir}{fdatetime}.png'
        # print(timestr)
        fig = plot_map_2panels(longitude, latitude, dataarr, tn_perim, pixel_bt, levels, cmaps, titles, cblabels, cbticks, timestr, dt_thres, 
                               ntracks, lifetime, track_bt, track_ccs_lon, track_ccs_lat, track_pf_lon, track_pf_lat, track_pf_diam, figname)
        plt.close(fig)
        print(figname)

    ds.close()
    return 1


if __name__ == "__main__":

    start_datetime = sys.argv[1]
    end_datetime = sys.argv[2]
    run_parallel = int(sys.argv[3])
    config_file = sys.argv[4]

    # start_datetime = '2019-01-24T00'
    # end_datetime = '2019-01-26T00'
    # run_parallel = 1
    # config_file = '/global/homes/f/feng045/program/PyFLEXTRKR/config/config_gpm_mcs_saag.yml'

    # Create a timedelta threshold in minutes
    # Tracks that end longer than this threshold from the current pixel-level frame are not plotted
    # This treshold controls the time window to retain previous tracks
    dt_thres = datetime.timedelta(hours=1)

    # Track stats file
    config = load_config(config_file)
    stats_path = config["stats_outpath"]
    pixeltracking_path = config["pixeltracking_outpath"]
    pixeltracking_filebase = config["pixeltracking_filebase"]
    mcsfinal_filebase = config["mcsfinal_filebase"]
    startdate = config["startdate"]
    enddate = config["enddate"]
    trackstats_file = f"{stats_path}{mcsfinal_filebase}{startdate}_{enddate}.nc"
    n_workers = config["nprocesses"]

    # Output figure directory
    figdir = f'{pixeltracking_path}quicklooks_trackpaths/'
    os.makedirs(figdir, exist_ok=True)
    print(figdir)


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


    # Read track stats file
    dss = xr.open_dataset(trackstats_file)
    stats_starttime = dss['base_time'].isel(times=0)
    # Convert input datetime to np.datetime64
    stime = np.datetime64(start_datetime)
    etime = np.datetime64(end_datetime)

    # Find tracks initiated within the time window
    idx = np.where((stats_starttime >= stime) & (stats_starttime <= etime))[0]
    ntracks = len(idx)
    print(f'Number of tracks within input period: {ntracks}')

    # Subset these tracks
    time_res = dss.attrs['time_resolution_hour']
    lifetime = dss['track_duration'].isel(tracks=idx) * time_res
    track_bt = dss['base_time'].isel(tracks=idx)
    track_ccs_lon = dss['meanlon'].isel(tracks=idx)
    track_ccs_lat = dss['meanlat'].isel(tracks=idx)
    track_pf_lon = dss['pf_lon_centroid'].isel(tracks=idx, nmaxpf=0)
    track_pf_lat = dss['pf_lat_centroid'].isel(tracks=idx, nmaxpf=0)
    track_pf_diam = 2 * np.sqrt(dss['pf_area'].isel(tracks=idx, nmaxpf=0) / np.pi)

    # Serial option
    if run_parallel == 0:
        for ifile in range(len(datafiles)):
            print(datafiles[ifile])
            result = work_for_time_loop(datafiles[ifile], ntracks, lifetime, track_bt, track_ccs_lon, track_ccs_lat, track_pf_lon, track_pf_lat, track_pf_diam, dt_thres, figdir)

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
            result = dask.delayed(work_for_time_loop)(datafiles[ifile], ntracks, lifetime, track_bt, track_ccs_lon, track_ccs_lat, track_pf_lon, track_pf_lat, track_pf_diam, dt_thres, figdir)
            results.append(result)

        # Trigger dask computation
        final_result = dask.compute(*results)