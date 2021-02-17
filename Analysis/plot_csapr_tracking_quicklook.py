import numpy as np
import glob, os, sys
import xarray as xr
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

#-----------------------------------------------------------------------
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
        # lon_c[ii] = np.mean(longitude[iyy, ixx])
        # lat_c[ii] = np.mean(latitude[iyy, ixx])
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


def plot_map_2panels(xx, yy, comp_ref, tn_perim, notn_perim, xx_tn, yy_tn, tracknumbers, xx_cn, yy_cn, notracknumbers, 
                     levels, cmaps, titles, cblabels, cbticks, timestr, figname):
    """
    Plot with Cartopy map projection.
    """

    mpl.rcParams['font.size'] = 12
    mpl.rcParams['font.family'] = 'Helvetica'
    
    radii = np.arange(20,101,20)  # radii for the range rings [km]
    azimuths = np.arange(0,361,30)  # azimuth angles for HSRHI scans [degree]
    radar_lon, radar_lat = -64.7284, -32.1264  # CSAPR radar location
    
    map_extend = [np.min(xx), np.max(xx), np.min(yy), np.max(yy)]
    lonvals = mpl.ticker.FixedLocator(np.arange(-66,-63,0.5))
    latvals = mpl.ticker.FixedLocator(np.arange(-34,-30,0.5))
    proj = ccrs.PlateCarree()
    
    fig = plt.figure(figsize=[11,5], dpi=200)
    
    # Set up the two panels with GridSpec, use GridSpecFromSubplotSpec to make enough space between the two panels for colorbars
    # and make the colorbars right next to the panels
    # This may be overkill to use GridSpec but it's a good example to have complete control of the locations, 
    # which is good for making animations where the panel locations need to be locked
    # Set GridSpec for left and right panel
    gs = gridspec.GridSpec(1,2, height_ratios=[1], width_ratios=[0.5,0.5])
    gs.update(left=0.08, right=0.92, top=0.85, wspace=0.35, hspace=0.1)
    # Use GridSpecFromSubplotSpec for panel and colorbar
    gs_left = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], height_ratios=[1], width_ratios=[1,0.03], wspace=0.05, hspace=0.1)
    gs_right = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], height_ratios=[1], width_ratios=[1,0.03], wspace=0.05, hspace=0.1)
    
    fig.text(0.5, 0.93, timestr, fontsize=14, ha='center')
    
    ##########################################################
    # Panel 1
    ##########################################################
    ax1 = plt.subplot(gs_left[0], projection=proj)
    ax1.set_extent(map_extend, crs=proj)
    ax1.set_aspect('auto', adjustable=None)
    gl = ax1.gridlines(crs=proj, draw_labels=True, linestyle='--', linewidth=0, zorder=2)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlocator = lonvals
    gl.ylocator = latvals
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    # Plot reflectivity
    cmap = plt.get_cmap(cmaps)
    norm_ref = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    cf1 = ax1.pcolormesh(xx, yy, comp_ref, norm=norm_ref, cmap=cmap, transform=proj, zorder=2)
    # Overplot cell tracknumber perimeters
#     cmap_tn = plt.get_cmap('jet')
#     norm_tn = mpl.colors.BoundaryNorm(np.arange(0,len(tn_perim)+1,1), ncolors=cmap_tn.N, clip=True)
    Tn = np.ma.masked_where(tn_perim == 0, tn_perim)
    Tn[Tn > 0] = 10
    tn1 = ax1.pcolormesh(xx, yy, Tn, cmap='gray', transform=proj, zorder=3)
    # Overplot cell tracknumbers
    for ii in range(0, len(xx_tn)):
        ax1.text(xx_tn[ii], yy_tn[ii], f'{tracknumbers[ii]:.0f}', color='k', size=14, weight='bold', ha='left', va='center', transform=proj, zorder=3)
#         ax1.plot(xx_tn[ii], yy_tn[ii], marker='o', markersize=3, color='k', transform=proj, zorder=3)
#     ax1.scatter(xx_tn, yy_tn, s=20, marker='o', c='dodgerblue', edgecolors='k', linewidths=1, transform=proj, zorder=3)

    # Plot range circles around radar
    for ii in range(0, len(radii)):
        ax1.tissot(rad_km=radii[ii], lons=radar_lon, lats=radar_lat, n_samples=100, facecolor='None', edgecolor='k', lw=0.6, zorder=5)
    # Plot azimuth lines
    for ii in range(0, len(azimuths)):
        lon2, lat2 = calc_latlon(radar_lon, radar_lat, 200, azimuths[ii])
        ax1.plot([radar_lon,lon2], [radar_lat,lat2], color='k', lw=0.6, transform=ccrs.Geodetic(), zorder=5)
    # Reflectivity colorbar
    cax1 = plt.subplot(gs_left[1])
    cb1 = plt.colorbar(cf1, cax=cax1, label=cblabels, ticks=cbticks, extend='both')
    ax1.set_title(titles[0], loc='left')
    
    ##########################################################
    # Panel 2
    ##########################################################
    ax2 = plt.subplot(gs_right[0], projection=proj)
    ax2.set_extent(map_extend, crs=proj)
    ax2.set_aspect('auto', adjustable=None)
    gl = ax2.gridlines(crs=proj, draw_labels=True, linestyle='--', linewidth=0, zorder=2)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlocator = lonvals
    gl.ylocator = latvals
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    # Plot reflectivity
    cf2 = ax2.pcolormesh(xx, yy, comp_ref, norm=norm_ref, cmap='gist_ncar', transform=proj, zorder=2)
    # Overplot cells that are not tracked
    Tn = np.ma.masked_where(notn_perim == 0, notn_perim)
    Tn[Tn > 0] = 10
    tn2 = ax2.pcolormesh(xx, yy, Tn, cmap='gray', transform=proj, zorder=3)
    # Overplot cell tracknumbers
    for ii in range(0, len(xx_cn)):
        ax2.text(xx_cn[ii], yy_cn[ii], f'{notracknumbers[ii]:.0f}', color='k', transform=proj, zorder=3)
#         ax2.plot(xx_cn[ii], yy_cn[ii], marker='o', markersize=3, color='k', transform=proj, zorder=3)
#     ax2.scatter(xx_cn, yy_cn, s=20, marker='o', c='dodgerblue', edgecolors='k', linewidths=1, transform=proj, zorder=3)
    
    # Plot range circles around radar
    for ii in range(0, len(radii)):
        ax2.tissot(rad_km=radii[ii], lons=radar_lon, lats=radar_lat, n_samples=100, facecolor='None', edgecolor='k', lw=0.6, zorder=5)
    # Plot azimuth lines
    for ii in range(0, len(azimuths)):
        lon2, lat2 = calc_latlon(radar_lon, radar_lat, 200, azimuths[ii])
        ax2.plot([radar_lon,lon2], [radar_lat,lat2], color='k', lw=0.6, transform=ccrs.Geodetic(), zorder=5)
    # Reflectivity colorbar
    cax2 = plt.subplot(gs_right[1])
    cb2 = plt.colorbar(cf2, cax=cax2, label=cblabels, ticks=cbticks, extend='both')
    ax2.set_title(titles[1], loc='left')
    
#     fig.savefig(figname, dpi=300, bbox_inches='tight')

    # Thread-safe figure output
    canvas = FigureCanvas(fig)
    canvas.print_png(figname)
    fig.savefig(figname)
    return fig


#-----------------------------------------------------------------------
# @dask.delayed
def work_for_time_loop(datafile, figdir):
    # Read data file
    # ds = xr.open_mfdataset(datafiles, concat_dim='time', combine='nested')
    ds = xr.open_dataset(datafile)
    # Make x,y coordinates
    ds.coords['lon'] = ds.lon - 100
    ds.coords['lat'] = ds.lat - 100
    xx = ds.lon.data
    yy = ds.lat.data
    longitude = ds.longitude.data
    latitude = ds.latitude.data

    # # Find non NAN unique tracknumbers
    # cellnumber_unique = np.unique(ds.conv_mask_inflated.data)
    # cellnumber_unique = np.unique(cellnumber_unique[~np.isnan(cellnumber_unique) & (cellnumber_unique > 0)])
    # ntracks_unique = len(cellnumber_unique)

    # Get cell tracknumbers and cloudnumbers
    tn = ds.tracknumber.squeeze()
    cn = ds.cloudnumber.squeeze()

    # Find cells that are not tracked (tracknumber == nan)
    cn_notrack = cn.where(np.isnan(tn))

    # Get cell perimeters
    tn_perim = label_perimeter(tn.data)
    cn_perim = label_perimeter(cn.data)
    cn_notrack_perim = label_perimeter(cn_notrack.data)

    # Apply tracknumber to conv_mask1
    conv = ds.conv_mask.squeeze()
    tnconv1 = tn.where(conv > 0).data

    # Calculates cell center locations
    lon_tn1, lat_tn1, xx_tn1, yy_tn1, tnconv1_uniqe = calc_cell_center(tnconv1, longitude, latitude, xx, yy)
    lon_cn1, lat_cn1, xx_cn1, yy_cn1, cnnotrack_unique = calc_cell_center(cn_notrack.data, longitude, latitude, xx, yy)

    comp_ref = ds.comp_ref.squeeze()
    levels = np.arange(-10, 60.1, 5)
    cbticks = np.arange(-10, 60.1, 5)
    cmaps = 'gist_ncar'
    titles = ['(a) Tracked Cells', '(b) Not Tracked Cells']
    cblabels = 'Composite Reflectivity (dBZ)'
    timestr = ds.time.squeeze().dt.strftime("%Y-%m-%d %H:%M UTC").data
    fignametimestr = ds.time.squeeze().dt.strftime("%Y%m%d_%H%M").data.item()
    figname = figdir + fignametimestr + '.png'
    print(figname)

    fig = plot_map_2panels(longitude, latitude, comp_ref, tn_perim, cn_notrack_perim, lon_tn1, lat_tn1, tnconv1_uniqe, lon_cn1, lat_cn1, cnnotrack_unique, 
                            levels, cmaps, titles, cblabels, cbticks, timestr, figname)
    plt.close(fig)
    ds.close()

    return 1


if __name__ == "__main__":
    
    # Get start/end date/time from input
    startdate = sys.argv[1]
    # enddate = sys.argv[2]
    run_parallel = int(sys.argv[2])

    # Set parallel option - 0:serial, 1:parallal
    # run_parallel = 1
    # try:
    #     run_parallel
    # except NameError:
    #     print(f"run_parallel is not set. Will run in serial (run_parallel = 0).")
    #     run_parallel = 0
    # else:
    #     if (run_parallel == 0) | (run_parallel == 1):
    #         pass
    #     else:
    #         print(f"Error: unknown run_parallel option. 0:serial, 1:dask.")
    #         exit()

    # Input data files
    # datadir = f'/global/cscratch1/sd/feng045/iclass/cacti/arm/csapr/celltracking/{startdate}_{enddate}/'
    # datadir = os.path.expandvars('$ICLASS') + f'/cacti/radar_processing/taranis_corcsapr2cfrppiqcM1_celltracking.c1/celltracking/{startdate}_{enddate}/'
    datadir = os.path.expandvars('$ICLASS') + f'cacti/radar_processing/taranis_corcsapr2cfrppiqcM1_celltracking.c1/celltracking/20181015.0000_20190303.0000/'
    datafiles = sorted(glob.glob(f'{datadir}celltracks_{startdate}*.nc'))
    # datadir = '/global/cscratch1/sd/feng045/iclass/cacti/arm/csapr/celltracking/20181110.1800_20181112.2359/'
    # datafiles = sorted(glob.glob(f'{datadir}celltracks_2018111[0-2]_????.nc'))
    # datadir = '/global/cscratch1/sd/feng045/iclass/cacti/arm/csapr/celltracking/20181125.2200_20181127.2359/'
    # datafiles = sorted(glob.glob(f'{datadir}celltracks_2018112[5-7]_????*.nc'))
    print(f'Number of files: {len(datafiles)}')
    print(f'{datadir}celltracks_{startdate}*.nc')

    # Output figure directory  
    figdir = f'{datadir}/quicklooks/'
    print(f'Output dir: {figdir}')
    os.makedirs(figdir, exist_ok=True)

    # Serial option
    if run_parallel == 0:

        for ifile in range(len(datafiles)):
            print(datafiles[ifile])
            # Plot the current file
            # Serial
            result = work_for_time_loop(datafiles[ifile], figdir)

    # Parallel option
    elif run_parallel == 1:

        # Set up dask workers and threads
        n_workers = 32

        # Initialize dask
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
        client = Client(cluster)

        results = []
        for ifile in range(len(datafiles)):
            print(datafiles[ifile])
            # Plot the current file
            # Dask
            result = delayed(work_for_time_loop)(datafiles[ifile], figdir)
            results.append(result)

        # # Trigger dask computation...
        # thesum = delayed(sum)(results)
        # thesum = thesum.compute()

        # Trigger dask computation
        final_result = dask.compute(*results)
