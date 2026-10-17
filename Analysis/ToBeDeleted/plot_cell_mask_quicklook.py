import numpy as np
import glob, os, sys
import xarray as xr
import pandas as pd
from scipy.ndimage import label, binary_dilation, binary_erosion, generate_binary_structure
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
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
def plot_map_2panels(pixel_dict, plot_info, map_info):
    """
    Plot with Cartopy map projection.
    """

    # Get pixel data from dictionary
    xx = pixel_dict['longitude']
    yy = pixel_dict['latitude']
    comp_ref = pixel_dict['comp_ref']
    conv_mask = pixel_dict['conv_mask']
    conv_core = pixel_dict['conv_core']
    cn_perim = pixel_dict['cn_perim']
    # tn = pixel_dict['tn']
    # tn_perim = pixel_dict['tn_perim']
    # notn_perim = pixel_dict['cn_notrack_perim']
    # xx_tn = pixel_dict['lon_tn1']
    # yy_tn = pixel_dict['lat_tn1']
    # tracknumbers = pixel_dict['tnconv1_unique'] 
    # xx_cn = pixel_dict['lon_cn1']
    # yy_cn = pixel_dict['lat_cn1']
    # notracknumbers = pixel_dict['cnnotrack_unique']
    # Get plot info from dictionary
    levels = plot_info['levels']
    cmaps = plot_info['cmaps']
    titles = plot_info['titles'] 
    cblabels = plot_info['cblabels']
    cbticks = plot_info['cbticks']
    timestr = plot_info['timestr']
    figname = plot_info['figname']
    # Map domain, lat/lon ticks, background map features
    map_extend = map_info['map_extend']
    lonv = map_info['lonv']
    latv = map_info['latv']

    perim_color = 'k'

    radii = np.arange(20,101,20)  # radii for the range rings [km]
    # azimuths = np.arange(0,361,30)  # azimuth angles for HSRHI scans [degree]
    azimuths = np.arange(0,361,90)
    radar_lon, radar_lat = -64.7284, -32.1264  # CSAPR radar location
    
    proj = ccrs.PlateCarree()
    
    mpl.rcParams['font.size'] = 11
    mpl.rcParams['font.family'] = 'Helvetica'
    fig = plt.figure(figsize=[11,4.6], dpi=200, facecolor='w')
    
    # Set up the two panels with GridSpec, use GridSpecFromSubplotSpec to make enough space between the two panels for colorbars
    # and make the colorbars right next to the panels
    # This may be overkill to use GridSpec but it's a good example to have complete control of the locations, 
    # which is good for making animations where the panel locations need to be locked
    # Set GridSpec for left and right panel
    gs = gridspec.GridSpec(1,2, height_ratios=[1], width_ratios=[0.5,0.5])
    gs.update(left=0.05, right=0.94, top=0.88, bottom=0.05, wspace=0.35, hspace=0.1)
    # Use GridSpecFromSubplotSpec for panel and colorbar
    gs_left = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], height_ratios=[1], width_ratios=[1,0.03], wspace=0.05, hspace=0.1)
    gs_right = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], height_ratios=[1], width_ratios=[1,0.03], wspace=0.05, hspace=0.1)
    
    fig.text(0.5, 0.95, timestr, fontsize=14, ha='center')
    
    ##########################################################
    # Panel 1
    ##########################################################
    ax1 = plt.subplot(gs_left[0], projection=proj)
    ax1.set_extent(map_extend, crs=proj)
    ax1.set_aspect('auto', adjustable=None)
    gl = ax1.gridlines(crs=proj, draw_labels=False, linestyle='--', linewidth=0.)
    gl.xlocator = mpl.ticker.FixedLocator(lonv)
    gl.ylocator = mpl.ticker.FixedLocator(latv)
    ax1.set_xticks(lonv, crs=proj)
    ax1.set_yticks(latv, crs=proj)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()        
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    
    # Plot reflectivity
    cmap = plt.get_cmap(cmaps)
    norm_ref = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    comp_ref = np.ma.masked_where(comp_ref < min(levels), comp_ref)
    cf1 = ax1.pcolormesh(xx, yy, comp_ref, shading='nearest', norm=norm_ref, cmap=cmap, transform=proj, zorder=2)
    # Overplot cell tracknumber perimeters
#     cmap_tn = plt.get_cmap('jet')
#     norm_tn = mpl.colors.BoundaryNorm(np.arange(0,len(tn_perim)+1,1), ncolors=cmap_tn.N, clip=True)
    # # Replace all valid cloudtracknumber with a constant, and invalid cloudtracknumber with 0
    # tn = tn.data
    # tn[(tn >= 1)] = 10
    # tn[np.isnan(tn)] = 0
    # # Overlay boundary of cloudtracknumber on Tb
    # tn1 = ax1.contour(xx, yy, tn, colors=perim_color, levels=[0,1], linewidths=1, alpha=0.5, transform=proj, zorder=5)

    # Overplot cores
    # Create an all black colormap
    colors = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
    cmap_name = 'all_black'
    cm_black = mpl.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=3)
    core_mask = np.ma.masked_where(conv_core == 0, conv_core)
    cc1 = ax1.pcolormesh(xx, yy, core_mask, vmin=0, vmax=1, cmap=cm_black, transform=proj, zorder=3, alpha=0.7)

    # Plot cell tracknumber perimeters
    Cn = np.ma.masked_where(cn_perim == 0, cn_perim)
    Cn[Cn > 0] = 10
    cn1 = ax1.pcolormesh(xx, yy, Cn, shading='nearest', cmap='gray', transform=proj, alpha=1, zorder=3)

    # # Overplot cell tracknumbers
    # for ii in range(0, len(xx_tn)):
    #     ax1.text(xx_tn[ii], yy_tn[ii], f'{tracknumbers[ii]:.0f}', color='k', size=10, weight='bold', ha='left', va='center', transform=proj, zorder=3)
#         ax1.plot(xx_tn[ii], yy_tn[ii], marker='o', markersize=3, color='k', transform=proj, zorder=3)
#     ax1.scatter(xx_tn, yy_tn, s=20, marker='o', c='dodgerblue', edgecolors='k', linewidths=1, transform=proj, zorder=3)

    # Plot range circles around radar
    for ii in range(0, len(radii)):
        rr = ax1.tissot(rad_km=radii[ii], lons=radar_lon, lats=radar_lat, n_samples=100, facecolor='None', edgecolor='k', lw=0.4, zorder=3)
    # Plot azimuth lines
    for ii in range(0, len(azimuths)):
        lon2, lat2 = calc_latlon(radar_lon, radar_lat, 200, azimuths[ii])
        ax1.plot([radar_lon,lon2], [radar_lat,lat2], color='k', lw=0.4, transform=ccrs.Geodetic(), zorder=5)
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
    gl = ax2.gridlines(crs=proj, draw_labels=False, linestyle='--', linewidth=0.)
    gl.xlocator = mpl.ticker.FixedLocator(lonv)
    gl.ylocator = mpl.ticker.FixedLocator(latv)
    ax2.set_xticks(lonv, crs=proj)
    ax2.set_yticks(latv, crs=proj)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax2.xaxis.set_major_formatter(lon_formatter)
    ax2.yaxis.set_major_formatter(lat_formatter)
    
    # # Plot reflectivity
    # cf2 = ax2.pcolormesh(xx, yy, comp_ref, shading='nearest', norm=norm_ref, cmap='gist_ncar', transform=proj, zorder=2)
    # # Reflectivity colorbar
    # cax2 = plt.subplot(gs_right[1])
    # cb2 = plt.colorbar(cf2, cax=cax2, label=cblabels, ticks=cbticks, extend='both')
    # # Overplot cells that are not tracked
    # Tn = np.ma.masked_where(notn_perim == 0, notn_perim)
    # Tn[Tn > 0] = 10
    # tn2 = ax2.pcolormesh(xx, yy, Tn, shading='nearest', cmap='gray', transform=proj, zorder=3)
    # # Overplot cell tracknumbers
    # for ii in range(0, len(xx_cn)):
    #     ax2.text(xx_cn[ii], yy_cn[ii], f'{notracknumbers[ii]:.0f}', color='k', size=10, transform=proj, zorder=3)

    cm_nlev = np.min([len(conv_mask), 256])
    levels_convmask = np.linspace(np.nanmin(conv_mask)+1, np.nanmax(conv_mask)+1, cm_nlev)
    # Plot cell masks
    Cm = np.ma.masked_where(conv_mask == 0, conv_mask)
    # Cm[Cm > 0] = 10
    # cm2 = ax2.pcolormesh(xx, yy, Cm, shading='nearest', cmap='gray', transform=proj, alpha=0.5, zorder=3)
    cmap = plt.get_cmap('jet')
    norm_ref = mpl.colors.BoundaryNorm(levels_convmask, ncolors=cmap.N, clip=True)
    cm2 = ax2.pcolormesh(xx, yy, Cm, shading='nearest', cmap=cmap, transform=proj,  zorder=3)
    cax2 = plt.subplot(gs_right[1])
    cb2 = plt.colorbar(cm2, cax=cax2, label='', extend='both')
    
    # Plot range circles around radar
    for ii in range(0, len(radii)):
        rr = ax2.tissot(rad_km=radii[ii], lons=radar_lon, lats=radar_lat, n_samples=100, facecolor='None', edgecolor='k', lw=0.4, zorder=3)
    # Plot azimuth lines
    for ii in range(0, len(azimuths)):
        lon2, lat2 = calc_latlon(radar_lon, radar_lat, 200, azimuths[ii])
        ax2.plot([radar_lon,lon2], [radar_lat,lat2], color='k', lw=0.4, transform=ccrs.Geodetic(), zorder=5)
    
    ax2.set_title(titles[1], loc='left')
    
    # Thread-safe figure output
    canvas = FigureCanvas(fig)
    canvas.print_png(figname)
    fig.savefig(figname)
    return fig


#-----------------------------------------------------------------------
# @dask.delayed
def work_for_time_loop(datafile, map_info, figdir):
    # Read data file
    # ds = xr.open_mfdataset(datafiles, concat_dim='time', combine='nested')
    ds = xr.open_dataset(datafile)
    # Make x,y coordinates
    ds.coords['lon'] = ds.lon
    ds.coords['lat'] = ds.lat
    xx = ds.lon.data
    yy = ds.lat.data
    longitude = ds.longitude.data
    latitude = ds.latitude.data

    # Make dilation structure (larger values make thicker outlines)
    perim_thick = 10
    dilationstructure = np.zeros((perim_thick+1,perim_thick+1), dtype=int)
    dilationstructure[1:perim_thick, 1:perim_thick] = 1
    # dilationstructure = generate_binary_structure(2,1)

    # Get cell tracknumbers and cloudnumbers
    cn = ds['feature_number'].squeeze()
    # Only plot if there is cell in the frame

    if (np.nanmax(cn) > 0):

        # Subset pixel data within the map domain
        map_extend = map_info['map_extend']
        buffer = 0.05  # buffer area for subset
        lonmin, lonmax = map_extend[0]-buffer, map_extend[1]+buffer
        latmin, latmax = map_extend[2]-buffer, map_extend[3]+buffer
        mask = (ds['longitude'] >= lonmin) & (ds['longitude'] <= lonmax) & (ds['latitude'] >= latmin) & (ds['latitude'] <= latmax)
        xx_sub = mask.where(mask == True, drop=True).lon.data
        yy_sub = mask.where(mask == True, drop=True).lat.data
        xmin = mask.where(mask == True, drop=True).lon.min().item()
        xmax = mask.where(mask == True, drop=True).lon.max().item()
        ymin = mask.where(mask == True, drop=True).lat.min().item()
        ymax = mask.where(mask == True, drop=True).lat.max().item()
        lon_sub = ds['longitude'].sel(lon=slice(xmin,xmax), lat=slice(ymin,ymax)).data
        lat_sub = ds['latitude'].sel(lon=slice(xmin,xmax), lat=slice(ymin,ymax)).data
        comp_ref = ds['dbz_comp'].sel(lon=slice(xmin,xmax), lat=slice(ymin,ymax)).squeeze()
        convmask_sub = ds['conv_mask'].sel(lon=slice(xmin,xmax), lat=slice(ymin,ymax)).squeeze()
        convcore_sub = ds['conv_core'].sel(lon=slice(xmin,xmax), lat=slice(ymin,ymax)).squeeze()
        cloudnumber_sub = ds['feature_number'].sel(lon=slice(xmin,xmax), lat=slice(ymin,ymax)).squeeze()
        # cn_notrack = cloudnumber_sub.where(np.isnan(tracknumber_sub))
        # Get object perimeters
        # tn_perim = label_perimeter(tracknumber_sub.data, dilationstructure)
        cn_perim = label_perimeter(cloudnumber_sub.data, dilationstructure)
        # cn_notrack_perim = label_perimeter(cn_notrack.data, dilationstructure)

        # Apply tracknumber to conv_mask1
        # tnconv1 = tracknumber_sub.where(convmask_sub > 0).data

        # Calculates cell center locations
        # lon_tn1, lat_tn1, xx_tn1, yy_tn1, tnconv1_unique = calc_cell_center(tnconv1, lon_sub, lat_sub, xx_sub, yy_sub)
        # lon_cn1, lat_cn1, xx_cn1, yy_cn1, cnnotrack_unique = calc_cell_center(cn_notrack.data, lon_sub, lat_sub, xx_sub, yy_sub)

        # comp_ref = ds.comp_ref.squeeze()
        levels = np.arange(-10, 60.1, 5)
        cbticks = np.arange(-10, 60.1, 5)
        cmaps = 'gist_ncar'
        titles = ['(a) Composite Reflectivity', '(b) Cell Mask']
        cblabels = 'Composite Reflectivity (dBZ)'
        timestr = ds.time.squeeze().dt.strftime("%Y-%m-%d %H:%M UTC").data
        fignametimestr = ds.time.squeeze().dt.strftime("%Y%m%d_%H%M").data.item()
        figname = figdir + fignametimestr + '.png'
        
        pixel_dict = {
            'longitude': lon_sub, 
            'latitude': lat_sub, 
            'comp_ref': comp_ref, 
            # 'tn': tracknumber_sub,
            'conv_mask': convmask_sub,
            'conv_core': convcore_sub,
            'cn_perim': cn_perim,
            # 'tn_perim': tn_perim, 
            # 'cn_notrack_perim': cn_notrack_perim, 
            # 'lon_tn1': lon_tn1, 
            # 'lat_tn1': lat_tn1, 
            # 'tnconv1_unique': tnconv1_unique, 
            # 'lon_cn1': lon_cn1, 
            # 'lat_cn1': lat_cn1, 
            # 'cnnotrack_unique': cnnotrack_unique, 
        }
        plot_info = {
            'levels': levels, 
            'cmaps': cmaps, 
            'titles': titles, 
            'cblabels': cblabels, 
            'cbticks': cbticks, 
            'timestr': timestr, 
            'figname': figname,
        }
        fig = plot_map_2panels(pixel_dict, plot_info, map_info)
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

    # Set subset map domain
    map_extend = [-65.4, -63.2, -33.2, -30.8]
    # Set lat/lon labels
    lon_bin = 1
    lat_bin = 1
    lonv = np.arange(-65.4, -63.1, 1)
    latv = np.arange(-33, -31.1, 1)
    # lonv = np.arange(map_extend[0], map_extend[1]+0.001, lon_bin)
    # latv = np.arange(map_extend[2], map_extend[3]+0.001, lat_bin)
    # Put map info in a dictionary
    map_info = {
        'map_extend': map_extend,
        'lonv': lonv,
        'latv': latv,
    }

    # Get directory info from config file
    config = load_config(config_file)
    pixeltracking_path = config["tracking_outpath"]
    pixeltracking_filebase = config["cloudid_filebase"]
    n_workers = config["nprocesses"]

    # Output figure directory
    figdir = f'{pixeltracking_path}/quicklooks/'
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

    # Serial option
    if run_parallel == 0:

        for ifile in range(len(datafiles)):
            print(datafiles[ifile])
            # Plot the current file
            # Serial
            result = work_for_time_loop(datafiles[ifile], map_info, figdir)

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
            # Plot the current file
            # Dask
            result = dask.delayed(work_for_time_loop)(
                datafiles[ifile], map_info, figdir
            )
            results.append(result)

        # Trigger dask computation
        final_result = dask.compute(*results)
