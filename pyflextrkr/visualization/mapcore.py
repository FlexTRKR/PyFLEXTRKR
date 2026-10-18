"""
Map utilities shared by all plot families: projections by name, the dateline-aware PlateCarree setup of the original
scripts, the map domain (set_extent / set_global and the latitude-band crop), the figure size from the domain,
longitude helpers, and geodesic circles and lines (range rings, PF circles, azimuth lines).

Called by: mcs_map, cell_map, generic_map (the plot functions) and mcs_cli, cell_cli, generic_cli (figure size).
Calls: cartopy.crs, cartopy.geodesic, matplotlib (for the grid layout only).
"""
import numpy as np
import cartopy.crs as ccrs
from cartopy.geodesic import Geodesic
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure

# Map projections: name -> (cartopy class, the optional arguments it accepts)
PROJECTIONS = {
    'platecarree': (ccrs.PlateCarree, ('central_longitude',)),
    'robinson': (ccrs.Robinson, ('central_longitude',)),
    'mollweide': (ccrs.Mollweide, ('central_longitude',)),
    'equalearth': (ccrs.EqualEarth, ('central_longitude',)),
    'mercator': (ccrs.Mercator, ('central_longitude',)),
    'lambertconformal': (ccrs.LambertConformal, ('central_longitude', 'central_latitude', 'standard_parallels')),
    'albersequalarea': (ccrs.AlbersEqualArea, ('central_longitude', 'central_latitude', 'standard_parallels')),
    'orthographic': (ccrs.Orthographic, ('central_longitude', 'central_latitude')),
    'northpolarstereo': (ccrs.NorthPolarStereo, ('central_longitude',)),
    'southpolarstereo': (ccrs.SouthPolarStereo, ('central_longitude',)),
}

# Projections whose y coordinate depends on the latitude only (so a latitude band is a band of y; --lat_range)
LAT_BAND_PROJECTIONS = ('platecarree', 'robinson', 'mollweide', 'equalearth', 'mercator')
# Projections that show the whole globe in one shape: set_global() is the whole-globe domain (the others use set_extent of the box)
WORLD_PROJECTIONS = ('robinson', 'mollweide', 'equalearth', 'orthographic')

#-----------------------------------------------------------------------
def _is_projection(proj, keys):
    """True if proj is an instance of one of the projections of PROJECTIONS named in keys."""
    return isinstance(proj, tuple(PROJECTIONS[k][0] for k in keys))

#-----------------------------------------------------------------------
def projection_key(name):
    """Normalized key of a projection name, e.g. 'Lambert_Conformal' -> 'lambertconformal'."""
    key = str(name).lower().replace('_', '').replace('-', '').replace(' ', '')
    if key not in PROJECTIONS:
        raise ValueError(f"Unknown projection '{name}', choose from {sorted(PROJECTIONS)}")
    return key

#-----------------------------------------------------------------------
def check_lat_range(projection, lat_range):
    """
    --lat_range is only for the projections of LAT_BAND_PROJECTIONS (the band of latitudes is a band of y): a ValueError
    for the others, where it would cut the map at the wrong place. The command lines call this first, so it fails
    before a file is read.
    """
    if lat_range is not None and projection_key(projection) not in LAT_BAND_PROJECTIONS:
        raise ValueError(f"--lat_range is for the projections whose y depends on the latitude only "
                         f"({', '.join(LAT_BAND_PROJECTIONS)}), not for {projection}: use --extent for the latitudes")

#-----------------------------------------------------------------------
def lat_to_y(proj, lat):
    """
    Projected y of a latitude on the central meridian, for the projections of LAT_BAND_PROJECTIONS (y depends on the
    latitude only). The world projections give NaN within 0.1 degree of a pole: the pole is then the y limit of the projection.
    """
    _, y = proj.transform_point(0, lat, ccrs.PlateCarree())
    if not np.isfinite(y):
        y = proj.y_limits[1] if lat > 0 else proj.y_limits[0]
    return float(y)

#-----------------------------------------------------------------------
def make_projection(name, central_longitude=None, central_latitude=None, standard_parallels=None):
    """
    Make a cartopy projection. Only the parameters that are given (not None) are passed on, the others are the cartopy
    defaults of that projection. A parameter that the projection does not accept is an error (it would otherwise be
    silently ignored).
    """
    key = projection_key(name)
    cls, accepted = PROJECTIONS[key]
    given = {'central_longitude': central_longitude, 'central_latitude': central_latitude,
             'standard_parallels': None if standard_parallels is None else tuple(standard_parallels)}
    kwargs = {}
    for k, v in given.items():
        if v is None:
            continue
        if k not in accepted:
            raise ValueError(f"--{k} is not used by the {cls.__name__} projection (accepts: {', '.join(accepted)})")
        kwargs[k] = v
    if key == 'lambertconformal' and 'standard_parallels' in kwargs:
        # Cartopy draws the cone from its pole to the latitude `cutoff` (default -30). For a cone around the south pole (the
        # standard parallels in the southern hemisphere) that would show only the part south of 30S: the mirror is +30
        sp = kwargs['standard_parallels']
        poliest = max(sp, key=abs)
        if poliest < 0:
            kwargs['cutoff'] = 30
    return cls(**kwargs)

#-----------------------------------------------------------------------
def get_projections(map_extent, projection='PlateCarree', central_longitude=None, central_latitude=None,
                    standard_parallels=None):
    """
    Projection of the axes (proj), projection of the data (data_proj) and the set_extent longitudes in the proj frame.

    PlateCarree (the default) is set up as in the original scripts: the central longitude is 180 if the extent crosses
    the dateline and 0 otherwise (or central_longitude if it is given); data_proj is always PlateCarree(0) because the
    data longitudes are in the -180~+180 convention.
    Other projections come from make_projection(); their domain is set by set_map_domain (ext_lon0 = ext_lon1 = None).

    Args:
        map_extent: list [lonmin, lonmax, latmin, latmax]
        projection: string, name of the projection (see PROJECTIONS)
        central_longitude, central_latitude, standard_parallels: optional projection parameters

    Returns:
        proj, data_proj, ext_lon0, ext_lon1
    """
    # data_proj always PlateCarree(0): data lons are in -180~+180 convention.
    data_proj = ccrs.PlateCarree(central_longitude=0)
    if projection_key(projection) != 'platecarree':
        proj = make_projection(projection, central_longitude, central_latitude, standard_parallels)
        return proj, data_proj, None, None

    # Set up map projection.
    # Detect dateline-crossing by normalizing both extent lons to -180~+180:
    # if lon0_n > lon1_n the extent goes eastward across the dateline.
    # This works for both 0-360 notation (lon_max=220) and -180~+180 notation (lon_max=-140).
    _lon0_n = ((map_extent[0] + 180) % 360) - 180
    _lon1_n = ((map_extent[1] + 180) % 360) - 180
    if central_longitude is not None:
        pass
    elif _lon0_n > _lon1_n:
        central_longitude = 180
    else:
        central_longitude = 0
    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    # Compute set_extent bounds in the proj frame (values relative to central_longitude).
    _ext_lon0 = ((map_extent[0] - central_longitude + 180) % 360) - 180
    _ext_lon1 = ((map_extent[1] - central_longitude + 180) % 360) - 180
    return proj, data_proj, _ext_lon0, _ext_lon1

#-----------------------------------------------------------------------
def set_map_domain(ax, proj, map_extent, ext_lon0=None, ext_lon1=None, lat_range=None, extent_xy=None):
    """
    Set the visible domain of a map axes.

    PlateCarree (ext_lon0 / ext_lon1 given): set_extent in the proj frame, as in the original scripts; an extent that
    covers all longitudes (both ends equal in the proj frame) is the whole globe: set_global().
    Other projections: for the projections that show the whole world in one shape (Robinson, Mollweide, EqualEarth,
    Orthographic) an extent of 360 degrees of longitude is set_global(); everything else (regional extents, and the
    360-degree extent of Mercator, the conic and the polar projections) is set_extent of the lon/lat box (polar cap:
    --extent -180 180 20 90).
    lat_range: (lat_min, lat_max), crop the visible latitude band after that (the projected y-limits of the two
    latitudes), e.g. the 60S-60N band of a global Robinson map. Without lat_range a global map of Robinson, Mollweide or
    EqualEarth shows the latitudes of map_extent. Only for the projections whose y depends on the latitude only
    (LAT_BAND_PROJECTIONS); for the others it is an error (it would cut the map at the wrong place).

    Args:
        ax: cartopy GeoAxes
        proj: the projection of ax
        map_extent: list [lonmin, lonmax, latmin, latmax]
        ext_lon0, ext_lon1: set_extent longitudes in the proj frame (from get_projections), None for other projections
        lat_range: list or tuple (lat_min, lat_max), optional
        extent_xy: [xmin, xmax, ymin, ymax] in the coordinates of proj, optional (data_extent_xy): the domain is exactly this
            (no --extent was given: the domain of the data, e.g. a WRF Lambert conformal grid plotted in that projection)
    """
    is_platecarree = ext_lon0 is not None
    lon_span = (map_extent[1] - map_extent[0]) % 360 or 360
    check_lat_range(type(proj).__name__, lat_range)
    if extent_xy is not None:
        ax.set_extent(extent_xy, crs=proj)
    elif is_platecarree:
        if abs(ext_lon1 - ext_lon0) < 1e-6 or lon_span >= 360:
            ax.set_global()
        else:
            ax.set_extent([ext_lon0, ext_lon1, map_extent[2], map_extent[3]], crs=proj)
    else:
        whole_globe = lon_span >= 359
        if whole_globe and _is_projection(proj, WORLD_PROJECTIONS):
            ax.set_global()
            if lat_range is None and _is_projection(proj, LAT_BAND_PROJECTIONS):
                lat_range = (map_extent[2], map_extent[3])
        else:
            lon0, lon1 = (-180., 180.) if whole_globe else (float(norm_lon(map_extent[0])), float(norm_lon(map_extent[1])))
            if lon1 < lon0:
                lon1 += 360.
            ax.set_extent([lon0, lon1, map_extent[2], map_extent[3]], crs=ccrs.PlateCarree())

    # Crop the map to the specified latitude range if provided
    if lat_range is not None:
        lat_min, lat_max = lat_range
        y_min = lat_to_y(proj, lat_min)
        y_max = lat_to_y(proj, lat_max)
        ax.set_ylim([y_min, y_max])

#-----------------------------------------------------------------------
def data_extent_xy(proj, lon, lat, nsample=60):
    """
    The domain of a lon / lat grid in the coordinates of a projection: [xmin, xmax, ymin, ymax] around the pixels (and
    half a pixel more on each side). The extremes of a grid are on its edge: the whole edge and a coarse sample of the
    inside are transformed. For data on a grid of the same projection (a WRF Lambert conformal domain plotted with that
    Lambert conformal projection) this is the rectangle of the data, with no empty space around it.

    Args:
        proj: cartopy projection of the axes
        lon, lat: 2-D arrays (ny, nx) of the longitude and latitude of the pixels
        nsample: number of rows / columns sampled inside the grid

    Returns:
        [xmin, xmax, ymin, ymax]
    """
    lon = norm_lon(lon)
    lat = np.asarray(lat, dtype=float)
    ny, nx = lon.shape
    iy = np.unique(np.r_[np.arange(0, ny, max(ny // nsample, 1)), ny - 1])
    ix = np.unique(np.r_[np.arange(0, nx, max(nx // nsample, 1)), nx - 1])
    plon = np.concatenate([lon[np.ix_(iy, ix)].ravel(), lon[0, :], lon[-1, :], lon[:, 0], lon[:, -1]])
    plat = np.concatenate([lat[np.ix_(iy, ix)].ravel(), lat[0, :], lat[-1, :], lat[:, 0], lat[:, -1]])
    src = ccrs.PlateCarree(central_longitude=0)
    pts = proj.transform_points(src, plon, plat)[:, :2]
    pts = pts[np.isfinite(pts).all(axis=1)]
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    # half a pixel more on each side (the spacing of the middle row and column in the projected coordinates)
    mid_row = proj.transform_points(src, lon[ny // 2, :], lat[ny // 2, :])[:, :2]
    mid_col = proj.transform_points(src, lon[:, nx // 2], lat[:, nx // 2])[:, :2]
    dx = np.nanmedian(np.abs(np.diff(mid_row[:, 0]))) if nx > 1 else 0.
    dy = np.nanmedian(np.abs(np.diff(mid_col[:, 1]))) if ny > 1 else 0.
    return [float(xmin - dx / 2), float(xmax + dx / 2), float(ymin - dy / 2), float(ymax + dy / 2)]

#-----------------------------------------------------------------------
def auto_map_domain(map_extent, projection, central_longitude, central_latitude, standard_parallels, lat_range, lon, lat):
    """
    The shape of the map, and its domain when the data decide it (the command lines call this with the lon / lat of the
    first pixel-level file).

    With --extent the domain is that box. Without it the domain is the data: for a projection other than PlateCarree and
    regional data it is the box of the data in the coordinates of the projection (data_extent_xy: data on a Lambert conformal
    grid shown with that projection fill the map; the lon / lat box of such data would leave empty corners); otherwise the
    lon / lat box of the data.

    Args:
        map_extent: [lonmin, lonmax, latmin, latmax] (--extent) or None
        projection, central_longitude, central_latitude, standard_parallels: the projection (get_projections)
        lat_range: (lat_min, lat_max) of --lat_range or None
        lon, lat: 2-D arrays of the longitude and latitude of the pixels

    Returns:
        extent_xy: [xmin, xmax, ymin, ymax] in the coordinates of the projection, or None (the domain is map_extent or the
            lon / lat box of the data)
        aspect_yx: height / width of the map
    """
    data_extent = [float(np.nanmin(lon)), float(np.nanmax(lon)), float(np.nanmin(lat)), float(np.nanmax(lat))]
    extent = map_extent if map_extent is not None else data_extent
    proj, _, _, _ = get_projections(extent, projection, central_longitude, central_latitude, standard_parallels)
    regional = (data_extent[1] - data_extent[0]) < 359
    if map_extent is None and projection_key(projection) != 'platecarree' and regional:
        extent_xy = data_extent_xy(proj, lon, lat)
        return extent_xy, (extent_xy[3] - extent_xy[2]) / (extent_xy[1] - extent_xy[0])
    return None, projected_aspect(proj, extent, lat_range)

#-----------------------------------------------------------------------
def fit_colorbars_to_maps(pairs, orientation):
    """
    The maps keep their shape (aspect 'equal', adjustable 'box'), so a map that does not fill its cell of the figure is
    centered in it. Move the colorbars with the map so that each colorbar has the extent of its map: as long as the map is
    high (orientation 'vertical': the colorbars at the right of the map) or as wide as the map (orientation 'horizontal':
    below the map), at the same distance from the map as in the layout.

    Args:
        pairs: list of (map axes, [colorbar axes, ...])
        orientation: 'vertical' or 'horizontal', the orientation of the colorbars
    """
    for ax, caxes in pairs:
        ax.apply_aspect()
        p0 = ax.get_position(original=True)     # the cell of the layout
        p1 = ax.get_position()                  # the map, after its aspect was applied
        for cax in caxes:
            c = cax.get_position()
            if orientation == 'vertical':
                cax.set_position([c.x0 + (p1.x1 - p0.x1), p1.y0, c.width, p1.height])
            else:
                cax.set_position([p1.x0, c.y0 + (p1.y0 - p0.y0), p1.width, c.height])

#-----------------------------------------------------------------------
def keep_colorbar_text_inside(fig, pairs, orientation, spacing_auto=True, pad=4.0, gap=8.0):
    """
    The text of a colorbar (tick labels and label) can reach the next colorbar or the edge of the figure, depending on the
    font size, the length of the label and the size of the figure. Call this after fit_colorbars_to_maps: the text is
    measured with the renderer and
      1. a colorbar that is close to the text of the colorbar before it (2 colorbars side by side) is moved away from it
         (only with spacing_auto; a gap that was asked for with --cbar_spacing is kept),
      2. the maps are made smaller if the text of a colorbar would end outside the figure: the maps keep their shape, the
         colorbars move with them, and all maps shrink by the same amount, so the panels stay equal.

    Args:
        fig: the figure, with an Agg canvas (fig.canvas.get_renderer())
        pairs: list of (map axes, [colorbar axes, ...]) as in fit_colorbars_to_maps
        orientation: 'vertical' (colorbars at the right of the map, the text to the right of them) or 'horizontal'
            (colorbars below the map, the text below them)
        spacing_auto: move a colorbar away from the text of the previous one
        pad: smallest distance of the text from the edge of the figure [points]
        gap: smallest distance between the text of a colorbar and the next colorbar [points]
    """
    renderer = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()
    width_in, height_in = fig.get_size_inches()
    vertical = orientation == 'vertical'
    pad_f = pad / 72.0 / (width_in if vertical else height_in)
    gap_f = gap / 72.0 / width_in

    def text_box(cax):
        return to_fig.transform_bbox(cax.get_tightbbox(renderer))

    # 1. colorbars side by side: the second one starts after the text of the first one
    if spacing_auto and vertical:
        for ax, caxes in pairs:
            for prev, cax in zip(caxes[:-1], caxes[1:]):
                room = cax.get_position().x0 - text_box(prev).x1
                if room < gap_f:
                    c = cax.get_position()
                    cax.set_position([c.x0 + (gap_f - room), c.y0, c.width, c.height])

    # 2. the text of the last colorbar of each map ends inside the figure
    if vertical:
        over = max(text_box(caxes[-1]).x1 for ax, caxes in pairs) - (1.0 - pad_f)
    else:
        over = pad_f - min(text_box(cax).y0 for ax, caxes in pairs for cax in caxes)
    if over <= 0:
        return
    for ax, caxes in pairs:
        ax.apply_aspect()
        p = ax.get_position()
        # the map shrinks around its left edge (colorbars at the right) or its top edge (colorbars below), by `over`
        k = max((p.width if vertical else p.height) - over, 0.3 * (p.width if vertical else p.height)) / (p.width if vertical else p.height)
        w, h = p.width * k, p.height * k
        if vertical:
            box = [p.x0, p.y0 + (p.height - h) / 2, w, h]
            ax.set_position(box, which='both')
            for cax in caxes:
                c = cax.get_position()
                cax.set_position([c.x0 - (p.width - w), box[1], c.width, h])
        else:
            box = [p.x0 + (p.width - w) / 2, p.y1 - h, w, h]
            ax.set_position(box, which='both')
            for cax in caxes:
                c = cax.get_position()
                cax.set_position([box[0], c.y0 + (p.height - h), w, c.height])

#-----------------------------------------------------------------------
def lat_labels_on_boundary(ax, proj, gl, map_extent, lat_range=None, latv=None, pad=3):
    """
    Latitude labels along the left, curved edge of a whole-world map (Robinson, Mollweide, EqualEarth): one at the latitude of
    each gridline, e.g. 60S, 40S, ... 60N. The gridliner of cartopy puts a label where a gridline meets the rectangle of
    the axes. The gridlines of the middle latitudes end at the curved edge of the map, inside the rectangle, so the labels
    of +-40 and +-60 degrees were missing in some figure sizes.

    Args:
        ax: map axes; proj: its projection; gl: the gridliner of ax (its latitude locator, formatter and label style are used)
        map_extent: [lonmin, lonmax, latmin, latmax] of the map; lat_range: (lat_min, lat_max) of the visible band, or None
        latv: latitudes of the gridlines if they were set (--lat_ticks); pad: distance of the label from the edge [points]

    Returns:
        True if the labels were made (the left labels of the gridliner are then switched off), False for the other maps.
    """
    lon_span = (map_extent[1] - map_extent[0]) % 360 or 360
    if lon_span < 359 or not _is_projection(proj, ('robinson', 'mollweide', 'equalearth')):
        return False
    lat_lo, lat_hi = lat_range if lat_range is not None else (map_extent[2], map_extent[3])
    ticks = np.asarray(latv if latv is not None else gl.ylocator.tick_values(lat_lo, lat_hi), dtype=float)
    ticks = ticks[(ticks >= lat_lo - 1e-6) & (ticks <= lat_hi + 1e-6)]
    # the left edge of the map is the meridian 180 degrees west of the central longitude
    lon_edge = float(norm_lon(proj.proj4_params['lon_0'] - 180.0 + 1e-3))
    for lat in ticks:
        x, y = proj.transform_point(lon_edge, lat, ccrs.PlateCarree())
        if not (np.isfinite(x) and np.isfinite(y)):
            continue                                           # a pole
        ax.annotate(gl.yformatter(lat), xy=(x, y), xytext=(-pad, 0), textcoords='offset points', ha='right', va='center',
                    annotation_clip=False, **gl.ylabel_style)
    gl.left_labels = False
    # The label of the highest latitude sits on the top edge of the map: lift the title (left aligned) above it
    if len(ticks) and np.isclose(ticks.max(), lat_hi) and ax.get_title(loc='left'):
        ax.set_title(ax.get_title(loc='left'), loc='left', pad=0.5 * gl.ylabel_style.get('size', mpl.rcParams['font.size']) + 8)
    return True

#-----------------------------------------------------------------------
def extent_aspect(map_extent):
    """
    Height / width of a lon/lat box in degrees (the longitude span is taken modulo 360, so a box that crosses the
    dateline or is given in 0-360 notation works). This is the figure aspect used by the original scripts.
    """
    lon_span = (map_extent[1] - map_extent[0]) % 360 or 360
    lat_span = map_extent[3] - map_extent[2]
    return lat_span / lon_span

#-----------------------------------------------------------------------
def projected_aspect(proj, map_extent, lat_range=None, npts=100):
    """
    Height / width of a lon/lat box in the coordinates of a projection (the edges of the box are sampled, so the curved
    edges are included). A box that covers all longitudes uses the x-limits of Robinson, Mollweide and EqualEarth (the
    whole-world shapes; y from the two latitudes) and the limits of Orthographic; lat_range replaces the latitudes of
    the box (only for the projections of LAT_BAND_PROJECTIONS, as in set_map_domain).

    Args:
        proj: cartopy projection of the axes
        map_extent: list [lonmin, lonmax, latmin, latmax]
        lat_range: (lat_min, lat_max), optional
    """
    src = ccrs.PlateCarree(central_longitude=0)
    lat0, lat1 = (map_extent[2], map_extent[3]) if lat_range is None else lat_range
    lon_span = (map_extent[1] - map_extent[0]) % 360 or 360
    whole_globe = lon_span >= 359
    if whole_globe and _is_projection(proj, ('robinson', 'mollweide', 'equalearth')):
        # the whole world: x from the projection, y from the two latitudes
        xmin, xmax = proj.x_limits
        y0, y1 = lat_to_y(proj, lat0), lat_to_y(proj, lat1)
        ymin, ymax = min(y0, y1), max(y0, y1)
    elif whole_globe and _is_projection(proj, ('orthographic',)):
        xmin, xmax = proj.x_limits
        ymin, ymax = proj.y_limits
    else:
        lon0, lon1 = (-180., 180.) if whole_globe else (float(norm_lon(map_extent[0])), float(norm_lon(map_extent[1])))
        if lon1 < lon0:
            lon1 += 360.
        lons = np.linspace(lon0, lon1, npts)
        lats = np.linspace(lat0, lat1, npts)
        x = np.r_[lons, lons, np.full(npts, lon0), np.full(npts, lon1)]
        y = np.r_[np.full(npts, lat0), np.full(npts, lat1), lats, lats]
        pts = proj.transform_points(src, x, y)[:, :2]
        ok = np.isfinite(pts).all(axis=1)
        xmin, xmax, ymin, ymax = pts[ok, 0].min(), pts[ok, 0].max(), pts[ok, 1].min(), pts[ok, 1].max()
    return (ymax - ymin) / (xmax - xmin)

#-----------------------------------------------------------------------
def panel_fractions(nrows, ncols, height_ratios, width_ratios, row=0, col=0, **gs_adjust):
    """
    Width and height of one cell of a GridSpec as fractions of the figure (the layout arguments are those of
    gridspec.GridSpec and gs.update: left, right, bottom, top, wspace, hspace). The plot functions and the figure size
    use the same numbers, so the map axes get the aspect that was asked for.
    """
    fig = Figure(figsize=(1, 1))
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, height_ratios=height_ratios, width_ratios=width_ratios, **gs_adjust)
    bottoms, tops, lefts, rights = gs.get_grid_positions(fig)
    return rights[col] - lefts[col], tops[row] - bottoms[row]

#-----------------------------------------------------------------------
def figsize_for_aspect(figsize_x, aspect_yx, panel_w, panel_h):
    """
    Figure size [width, height] in inches such that a panel that is panel_w wide and panel_h high (fractions of the
    figure, see panel_fractions) has the height / width ratio aspect_yx.
    """
    return [float(figsize_x), float(round(figsize_x * aspect_yx * panel_w / panel_h, 2))]

#-----------------------------------------------------------------------
def min_figure_height(fontsize, top_margin):
    """
    Smallest height [inches] of a figure whose top margin (the fraction of the height above the maps) has room for the
    title of the figure (1.5 x fontsize) and the title of the map (1.2 x fontsize), with their spacing (0.15 inch). A map much
    wider than high (a global map between 60S and 60N) would otherwise get such a short figure that the two titles overlap.
    """
    return (0.045 * fontsize + 0.15) / top_margin

#-----------------------------------------------------------------------
def figure_margins(figsize, update, fontsize):
    """
    A map much wider than high (the 60S-60N band of the globe, the Sahel) gives a short figure, and the margins of the layout
    above and below the maps, fractions of the figure height, are then too small for the title of the figure (1.5 x fontsize),
    the title of the map (1.2 x fontsize) and the longitude labels. The figure is then made as high as the maps plus margins in
    inches: the maps keep their size and the figure is not taller than needed.

    Args:
        figsize: [width, height] in inches, from the shape of the map and the margins of the layout
        update: the layout, a dictionary with 'top' and 'bottom' (fractions of the figure height)
        fontsize: font size of the titles and labels

    Returns:
        figsize [width, height], and the margins (top, bottom) in inches for figure_layout (None: the margins of the layout
        are large enough, figsize is not changed)
    """
    width, height = figsize
    top_in = 0.045 * fontsize + 0.15
    if (1.0 - update['top']) * height >= top_in:
        return figsize, None
    bottom_in = max(update['bottom'] * height, (fontsize + 7.0) / 72.0 + 0.15)
    content = (update['top'] - update['bottom']) * height          # the maps, the gaps between them, the colorbars below them
    return [width, round(content + top_in + bottom_in, 2)], (top_in, bottom_in)

#-----------------------------------------------------------------------
def norm_lon(lon):
    """Longitude in -180..180 (works for scalars and arrays, any input convention)."""
    return ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0

#-----------------------------------------------------------------------
def in_map_extent(lon, lat, map_extent):
    """
    Check if a location is within the map extent (dateline-safe longitude check, strict latitude limits).
    """
    # Normalize to 0-360 for dateline-safe lon check
    _lon_norm = lon % 360
    _lonmin_norm = map_extent[0] % 360
    _lonmax_norm = map_extent[1] % 360
    if _lonmin_norm <= _lonmax_norm:
        _in_lon = (_lon_norm >= _lonmin_norm) and (_lon_norm <= _lonmax_norm)
    else:  # domain crosses the dateline
        _in_lon = (_lon_norm >= _lonmin_norm) or (_lon_norm <= _lonmax_norm)
    return _in_lon and (lat > map_extent[2]) and (lat < map_extent[3])

#-----------------------------------------------------------------------
def _continuous_lon(lon):
    """Longitudes [degree] without the jump of 360 degrees at the dateline (they can pass 180)."""
    return np.rad2deg(np.unwrap(np.deg2rad(lon)))

#-----------------------------------------------------------------------
def geodesic_circle(lon, lat, radius_km, n_samples=100):
    """
    Longitudes and latitudes [degree] of the circle of radius radius_km around (lon, lat): the points at that geodesic
    distance on the WGS84 ellipsoid (the first and the last point are the same). The longitudes are continuous, so a
    circle at the dateline is one curve.

    Draw it with the transform of the data (ax.plot(*geodesic_circle(...), transform=data_proj)), not with ax.tissot or
    transform=ccrs.Geodetic(): with PROJ 9.8 the transform from Geodetic to PlateCarree moves a point north or south
    by up to 0.19 degree (20 km, measured at 32S and 35N), so such circles do not sit on the cells that are drawn with
    the transform of the data.

    Args:
        lon, lat: center [degree]
        radius_km: radius [km]
        n_samples: number of points of the circle

    Returns:
        lons, lats: 1D arrays
    """
    circle = Geodesic().circle(lon, lat, radius_km * 1e3, n_samples=n_samples, endpoint=True)
    return _continuous_lon(circle[:, 0]), circle[:, 1]

#-----------------------------------------------------------------------
def geodesic_ray(lon, lat, azimuth, length_km, n_samples=21):
    """
    Longitudes and latitudes [degree] of the geodesic line that starts at (lon, lat) and goes length_km in the
    direction azimuth [degree clockwise from north] (WGS84 ellipsoid). Draw it with the transform of the data (see
    geodesic_circle).

    Args:
        lon, lat: start [degree]
        azimuth: direction [degree clockwise from north]
        length_km: length [km]
        n_samples: number of points of the line

    Returns:
        lons, lats: 1D arrays
    """
    distances = np.linspace(0., length_km * 1e3, n_samples)
    line = Geodesic().direct(np.tile([lon, lat], (n_samples, 1)), azimuth, distances)
    return _continuous_lon(line[:, 0]), line[:, 1]
