import numpy as np
import glob, os, sys
import xarray as xr
import pandas as pd
from scipy.ndimage import label, binary_dilation, binary_erosion, generate_binary_structure
import time, datetime, calendar, pytz
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# For non-gui matplotlib back end
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
mpl.use('agg')
# Parallalization
import dask
from dask import delayed
from dask.distributed import Client, LocalCluster

import warnings
warnings.filterwarnings("ignore")


def label_perimeter(tracknumber):
    """
    Labels the perimeter on a 2D map from cell tracknumbers.
    """
    
    # Generate a cross structure
    dilationstructure = generate_binary_structure(2,1)
    
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
        # Label the perimeter pixels with the cell number
        tracknumber_perim[iperim == 1] = ii
    
    return tracknumber_perim

def calc_latlon(lon1, lat1, dist, angle):
    """
    Haversine formula to calculate lat/lon locations from distance and angle.
    
    lon1:   longitude in [degree]
    lat1:   latitude in [degree]
    dist:   distance in [km]
    angle:  angle in [degree]
    """

    import math

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


def plot_map(xx, yy, comp_ref, tn_perim, pixel_bt, levels, cmaps, cblabels, cbticks, timestr, dt_thres, 
             ntracks, lifetime, cell_bt, cell_lon, cell_lat, figname):

    mpl.rcParams['font.size'] = 14
    mpl.rcParams['font.family'] = 'Helvetica'

    size_centroid = 80
    lw_centroid = 1.5
    
    radii = np.arange(20,101,20)  # radii for the range rings [km]
    azimuths = np.arange(0,361,30)  # azimuth angles for HSRHI scans [degree]
    radar_lon, radar_lat = -64.7284, -32.1264  # CSAPR radar location

    topo_levs = [500,1000,1500,2000,2500]
    cmap_topo = 'Reds'

    radii = np.arange(20,101,20)  # radii for the range rings [km]
    azimuths = np.arange(0,361,30)  # azimuth angles for HSRHI scans [degree]
    radar_lon, radar_lat = -64.7284, -32.1264  # CSAPR radar location

    map_extend = [np.min(xx), np.max(xx), np.min(yy), np.max(yy)]
    # map_extend = [minlon, maxlon, minlat, maxlat]
    lonvals = mpl.ticker.FixedLocator(np.arange(-66,-63,0.5))
    latvals = mpl.ticker.FixedLocator(np.arange(-34,-30,0.5))
    proj = ccrs.PlateCarree()

    fig = plt.figure(figsize=[8,7.5], dpi=200)
    gs = gridspec.GridSpec(1,2, height_ratios=[1], width_ratios=[1,0.03])
    gs.update(wspace=0.05, hspace=0.05)

    # ax1 = plt.subplot(111, projection=proj)
    ax1 = plt.subplot(gs[0], projection=proj)
    ax1.set_extent(map_extend, crs=proj)
    ax1.set_aspect('auto', adjustable=None)
    gl = ax1.gridlines(crs=proj, draw_labels=True, linestyle='--', linewidth=0, zorder=5)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlocator = lonvals
    gl.ylocator = latvals
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # ax1.contour(topoLon, topoLat, topoZ, levels=topo_levs, cmap=cmap_topo, linewidths=1, transform=proj)

    # Plot reflectivity
    cmap = plt.get_cmap(cmaps)
    norm_ref = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    cf1 = ax1.pcolormesh(xx, yy, comp_ref, norm=norm_ref, cmap=cmap, transform=proj, zorder=2)
    # Overplot cell tracknumber perimeters
    Tn = np.ma.masked_where(tn_perim == 0, tn_perim)
    Tn[Tn > 0] = 10
    tn1 = ax1.pcolormesh(xx, yy, Tn, cmap='gray', transform=proj, zorder=3)

    # Plot track centroids and paths
    marker_style = dict(edgecolor='k', linestyle='-', marker='o')
    for itrack in range(0, ntracks):
        # Get duration of the track
        idur = (lifetime.values[itrack] / time_res).astype(int)
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
                color_vals = np.repeat(lifetime.values[itrack], idur_cut)
                size_vals = np.repeat(size_centroid, idur_cut)
                size_vals[0] = size_centroid * 2
                cc = ax1.plot(cell_lon.values[itrack,idx_cut], cell_lat.values[itrack,idx_cut], lw=lw_centroid, ls='-', color='k', transform=proj, zorder=3)
                cl = ax1.scatter(cell_lon.values[itrack,idx_cut], cell_lat.values[itrack,idx_cut], s=size_vals, c=color_vals, 
                                vmin=0.5, vmax=5, cmap='Spectral_r', transform=proj, zorder=4, **marker_style)

    # Plot range circles around radar
    for ii in range(0, len(radii)):
        rr = ax1.tissot(rad_km=radii[ii], lons=radar_lon, lats=radar_lat, n_samples=100, facecolor='None', edgecolor='k', lw=0.4, zorder=3)
    # Plot azimuth lines
    for ii in range(0, len(azimuths)):
        lon2, lat2 = calc_latlon(radar_lon, radar_lat, 200, azimuths[ii])
        ax1.plot([radar_lon,lon2], [radar_lat,lat2], color='k', lw=0.4, transform=ccrs.Geodetic(), zorder=5)
    # Reflectivity colorbar
    cax1 = plt.subplot(gs[1])
    cb1 = plt.colorbar(cf1, cax=cax1, label=cblabels, ticks=cbticks, extend='both')
    ax1.set_title(timestr)

#     fig.savefig(figname, dpi=300, bbox_inches='tight')

    # Thread-safe figure output
    canvas = FigureCanvas(fig)
    canvas.print_png(figname)
    fig.savefig(figname)
    
    return fig


def work_for_time_loop(datafile, ntracks, lifetime, cell_bt, cell_lon, cell_lat, dt_thres, figdir):

    # Read pixel-level data
    ds = xr.open_dataset(datafile)
    # Make x,y coordinates
    ds.coords['lon'] = ds.lon - 100
    ds.coords['lat'] = ds.lat - 100
    xx = ds.lon.data
    yy = ds.lat.data
    longitude = ds.longitude.data
    latitude = ds.latitude.data
    pixel_bt = ds.time.data

    # Get cell tracknumbers and cloudnumbers
    tn = ds.tracknumber.squeeze()
    cn = ds.cloudnumber.squeeze()

    # Find cells that are not tracked (tracknumber == nan)
    cn_notrack = cn.where(np.isnan(tn))

    # Get cell perimeters
    tn_perim = label_perimeter(tn.data)
    cn_perim = label_perimeter(cn.data)
    cn_notrack_perim = label_perimeter(cn_notrack.data)

    comp_ref = ds.comp_ref.squeeze()

    cmaps = 'gist_ncar'
    levels = np.arange(-10, 60.1, 5)
    cbticks = np.arange(-10, 60.1, 5)
    timestr = ds.time.squeeze().dt.strftime("%Y-%m-%d %H:%M UTC").data
    # titles = [timestr]
    cblabels = 'Composite Reflectivity (dBZ)'
    fignametimestr = ds.time.squeeze().dt.strftime("%Y%m%d_%H%M").data.item()
    figname = figdir + fignametimestr + '.png'

    fig = plot_map(longitude, latitude, comp_ref, tn_perim, pixel_bt, levels, cmaps, cblabels, cbticks, timestr, dt_thres, 
                   ntracks, lifetime, cell_bt, cell_lon, cell_lat, figname)

    # plt.close(fig)
    ds.close()


if __name__ == "__main__":

    start_datetime = sys.argv[1]
    end_datetime = sys.argv[2]
    run_parallel = int(sys.argv[3])

    # start_datetime = '2019-01-25T17'
    # end_datetime = '2019-01-26T06'
    # run_parallel = 0

    # Set up dask workers (if run_parallel = 1)
    n_workers = 32

    # Track stats file
    statsdir = os.path.expandvars('$ICLASS') + f'cacti/radar_processing/taranis_corcsapr2cfrppiqcM1_celltracking.c1/stats/'
    statsfile = f'{statsdir}stats_tracknumbersv1.0_20181015.0000_20190303.0000.nc'
    # Terrain file (not used)
    terrain_file = os.path.expandvars('$ICLASS') + f'cacti/radar_processing/corgridded_terrain.c0/topo_cacti_csapr2.nc'

    # Pixel-level files
    datadir = os.path.expandvars('$ICLASS') + f'cacti/radar_processing/taranis_corcsapr2cfrppiqcM1_celltracking.c1/celltracking/20181015.0000_20190303.0000/'
    # Generate 15min time marks within the start/end datetime
    input_datetimes = pd.date_range(start=start_datetime, end=end_datetime, freq='15min').strftime('%Y%m%d_%H%M')
    # Find all files that matches the input datetime
    datafiles = []
    for tt in range(0, len(input_datetimes)):
        datafiles.extend(sorted(glob.glob(f'{datadir}celltracks_{input_datetimes[tt]}*.nc')))
    print(f'Number of pixel files: {len(datafiles)}')
 
    # Output figure directory
    figdir = os.path.expandvars('$ICLASS') + f'cacti/radar_processing/taranis_corcsapr2cfrppiqcM1_celltracking.c1/celltracking/track_demo/'

    # Create a timedelta threshold in minutes
    # Tracks that end longer than this threshold from the current pixel-level frame are not plotted
    # This treshold controls the time window to retain previous tracks
    dt_thres = datetime.timedelta(minutes=30)

    # # Read topography file
    # terr = xr.open_dataset(terrain_file)
    # topoZ = terr['z']
    # topoLon = terr['x']
    # topoLat = terr['y']
    # topoZ.plot(vmin=0, vmax=2250, cmap='terrain')

    # Read track stats file
    dss = xr.open_dataset(statsfile)
    stats_starttime = dss.basetime.isel(times=0)
    # Convert input datetime to np.datetime64
    stime = np.datetime64(start_datetime)
    etime = np.datetime64(end_datetime)

    # Find track initiated within the time window
    idx = np.where((stats_starttime >= stime) & (stats_starttime <= etime))[0]
    print(f'Number of tracks within input period: {len(idx)}')

    # Subset these tracks
    time_res = dss.attrs['time_resolution_hour']
    lifetime = dss.lifetime.isel(tracks=idx) * time_res
    cell_bt = dss.basetime.isel(tracks=idx)
    cell_lon = dss.cell_meanlon.isel(tracks=idx)
    cell_lat = dss.cell_meanlat.isel(tracks=idx)
    start_split_tracknumber = dss.start_split_tracknumber.isel(tracks=idx)
    end_merge_tracknumber = dss.end_merge_tracknumber.isel(tracks=idx)

    # Select long-lived tracks
    idx_long = np.where(lifetime > 0.5)[0]
    ntracks_long = len(idx_long)
    print(f'Number of long tracks within input period: {ntracks_long}')
    lifetime_long = lifetime.isel(tracks=idx_long)
    cell_bt_long = cell_bt.isel(tracks=idx_long)
    cell_lon_long = cell_lon.isel(tracks=idx_long)
    cell_lat_long = cell_lat.isel(tracks=idx_long)
    start_split_tracknumber_long = start_split_tracknumber.isel(tracks=idx_long)
    end_merge_tracknumber_long = end_merge_tracknumber.isel(tracks=idx_long)

    # Serial option
    if run_parallel == 0:
        for ifile in range(len(datafiles)):
            print(datafiles[ifile])
            result = work_for_time_loop(datafiles[ifile], ntracks_long, lifetime_long, cell_bt_long, cell_lon_long, cell_lat_long, dt_thres, figdir)

    # Parallel option
    elif run_parallel == 1:

        # Initialize dask
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
        client = Client(cluster)
        results = []
        for ifile in range(len(datafiles)):
            print(datafiles[ifile])
            result = delayed(work_for_time_loop)(datafiles[ifile], ntracks_long, lifetime_long, cell_bt_long, cell_lon_long, cell_lat_long, dt_thres, figdir)
            results.append(result)

        # Trigger dask computation
        final_result = dask.compute(*results)
