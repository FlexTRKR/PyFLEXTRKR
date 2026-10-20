"""
Tests of pyflextrkr.visualization.mapcore: projections by name, the dateline set-up of PlateCarree, the map domain with
the latitude-band crop, the figure size from the domain and the longitude helpers.
Skipped where cartopy / matplotlib are not installed. The CI environment has both (cartopy 0.26 comes with easygems), so these tests run
there with the cartopy of that day: the parameters of a projection are read through central_longitude() below, because PlateCarree has
`lon_0` in cartopy up to 0.25 and the prime meridian `pm` in 0.26.
"""
import numpy as np
import pytest

pytest.importorskip("matplotlib")
ccrs = pytest.importorskip("cartopy.crs")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyflextrkr.visualization import mapcore


#-----------------------------------------------------------------------
def central_longitude(proj):
    """The central longitude of a cartopy projection from its proj4 parameters: `lon_0`, or for the PlateCarree of cartopy 0.26 (a
    'latlong' CRS) the prime meridian `pm`."""
    params = proj.proj4_params
    return params['lon_0'] if 'lon_0' in params else params['pm']


#-----------------------------------------------------------------------
def test_projection_key_normalizes_names():
    assert mapcore.projection_key('Robinson') == 'robinson'
    assert mapcore.projection_key('Lambert_Conformal') == 'lambertconformal'
    assert mapcore.projection_key('plate carree') == 'platecarree'
    with pytest.raises(ValueError, match='Unknown projection'):
        mapcore.projection_key('NotAProjection')


def test_make_projection_passes_only_given_parameters():
    proj = mapcore.make_projection('Robinson', central_longitude=225)
    assert isinstance(proj, ccrs.Robinson)
    assert central_longitude(proj) == 225
    # the cartopy defaults apply when nothing is given
    assert central_longitude(mapcore.make_projection('Robinson')) == 0
    lcc = mapcore.make_projection('LambertConformal', central_longitude=-100, central_latitude=40, standard_parallels=[30, 50])
    assert central_longitude(lcc) == -100 and lcc.proj4_params['lat_0'] == 40
    # a parameter that the projection does not take is an error, not silently ignored
    with pytest.raises(ValueError, match='central_latitude'):
        mapcore.make_projection('Robinson', central_latitude=10)


def test_lambert_conformal_cone_around_the_south_pole_covers_the_northern_tropics():
    # cartopy draws the cone from its pole to the latitude `cutoff` (default -30): for standard parallels in the southern
    # hemisphere that shows only the part south of 30S, so make_projection mirrors it (+30)
    from shapely.geometry import Point, Polygon
    south = mapcore.make_projection('LambertConformal', central_longitude=-60, standard_parallels=[-40, -10])
    north = mapcore.make_projection('LambertConformal', central_longitude=-97, standard_parallels=[31, 35])
    assert south.cutoff == 30 and north.cutoff == -30
    x, y = south.transform_point(-60., 12., ccrs.PlateCarree())
    assert Polygon(south.boundary).contains(Point(x, y))
    x, y = north.transform_point(-97., 30., ccrs.PlateCarree())
    assert Polygon(north.boundary).contains(Point(x, y))
    # the single standard parallel of the south, and a user cutoff is not needed
    assert mapcore.make_projection('LambertConformal', standard_parallels=[-30]).cutoff == 30
    assert mapcore.make_projection('LambertConformal', standard_parallels=[40]).cutoff == -30


def test_get_projections_platecarree_dateline():
    # extent that crosses the dateline: central longitude 180, set_extent bounds in that frame (both 0-360 and -180~180)
    for extent in ([140., -140., -20., 20.], [140., 220., -20., 20.]):
        proj, data_proj, lon0, lon1 = mapcore.get_projections(extent)
        assert central_longitude(proj) == 180
        assert (lon0, lon1) == pytest.approx((-40., 40.))
        assert central_longitude(data_proj) == 0
    # regional extent: central longitude 0
    proj, _, lon0, lon1 = mapcore.get_projections([-80., -35., -55., 12.])
    assert central_longitude(proj) == 0
    assert (lon0, lon1) == pytest.approx((-80., -35.))
    # explicit central longitude
    proj, _, lon0, lon1 = mapcore.get_projections([100., 130., -10., 10.], central_longitude=115)
    assert central_longitude(proj) == 115
    assert (lon0, lon1) == pytest.approx((-15., 15.))


def test_get_projections_other_projection_has_no_set_extent_longitudes():
    proj, data_proj, lon0, lon1 = mapcore.get_projections([0., 360., -90., 90.], 'Robinson', central_longitude=225)
    assert central_longitude(proj) == 225
    assert lon0 is None and lon1 is None
    assert central_longitude(data_proj) == 0


def test_set_map_domain_lat_band_crop():
    # the 60S-60N band of a global Robinson map centered on 225E (the snippet of the global MCS maps)
    extent = [0., 360., -90., 90.]
    proj, _, lon0, lon1 = mapcore.get_projections(extent, 'Robinson', central_longitude=225)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    mapcore.set_map_domain(ax, proj, extent, lon0, lon1, lat_range=(-60, 60))
    _, y60 = proj.transform_point(0, 60, ccrs.PlateCarree())
    _, ym60 = proj.transform_point(0, -60, ccrs.PlateCarree())
    assert ax.get_ylim() == pytest.approx((ym60, y60))
    assert y60 > 0 and ym60 == pytest.approx(-y60)
    # the whole longitude range is kept
    assert ax.get_xlim() == pytest.approx(proj.x_limits)
    # without lat_range a global map shows the latitudes of the extent
    ax2 = fig.add_subplot(2, 1, 2, projection=proj)
    mapcore.set_map_domain(ax2, proj, [0., 360., -45., 45.], lon0, lon1)
    _, y45 = proj.transform_point(0, 45, ccrs.PlateCarree())
    assert ax2.get_ylim() == pytest.approx((-y45, y45))
    plt.close(fig)


def test_lat_band_only_for_projections_where_y_depends_on_latitude():
    mapcore.check_lat_range('Robinson', (-60, 60))
    mapcore.check_lat_range('Mercator', (-60, 60))
    mapcore.check_lat_range('LambertConformal', None)
    for name in ('NorthPolarStereo', 'Orthographic', 'LambertConformal', 'AlbersEqualArea'):
        with pytest.raises(ValueError, match='lat_range'):
            mapcore.check_lat_range(name, (-60, 60))
    proj = mapcore.make_projection('NorthPolarStereo')
    ax = plt.figure().add_subplot(1, 1, 1, projection=proj)
    with pytest.raises(ValueError, match='lat_range'):
        mapcore.set_map_domain(ax, proj, [-180., 180., 20., 90.], lat_range=(20, 90))
    plt.close('all')


def test_lat_to_y_uses_the_projection_limit_where_the_transform_is_nan():
    class NoPole:
        y_limits = (-5., 7.)
        def transform_point(self, x, y, src):
            return (float('nan'), float('nan')) if abs(y) > 89.5 else (0., 0.1 * y)
    assert mapcore.lat_to_y(NoPole(), 90.) == 7.
    assert mapcore.lat_to_y(NoPole(), -90.) == -5.
    assert mapcore.lat_to_y(NoPole(), 30.) == pytest.approx(3.)


@pytest.mark.parametrize("name", ['Robinson', 'Mollweide', 'EqualEarth'])
def test_world_projection_with_the_poles_in_the_extent(name):
    # the transform of a latitude within 0.1 degree of a pole is NaN: the pole is the y limit of the projection
    proj = mapcore.make_projection(name, central_longitude=180)
    extent = [0., 360., -90., 90.]
    assert mapcore.lat_to_y(proj, 90.) == pytest.approx(proj.y_limits[1])
    assert mapcore.lat_to_y(proj, -90.) == pytest.approx(proj.y_limits[0])
    assert mapcore.lat_to_y(proj, 45.) == pytest.approx(proj.transform_point(0, 45, ccrs.PlateCarree())[1])
    expected = (proj.y_limits[1] - proj.y_limits[0]) / (proj.x_limits[1] - proj.x_limits[0])
    assert mapcore.projected_aspect(proj, extent) == pytest.approx(expected, rel=1e-6)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    mapcore.set_map_domain(ax, proj, extent)
    assert ax.get_ylim() == pytest.approx(proj.y_limits)
    plt.close(fig)


def test_set_map_domain_polar_cap_world_and_mercator():
    # polar cap: the whole circle of latitude 20N, not half of it
    proj = mapcore.make_projection('NorthPolarStereo')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    extent = [-180., 180., 20., 90.]
    mapcore.set_map_domain(ax, proj, extent)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    assert x0 < 0 < x1 and y0 < 0 < y1
    assert (x1 - x0) == pytest.approx(y1 - y0, rel=0.02)
    assert mapcore.projected_aspect(proj, extent) == pytest.approx(1.0, rel=0.02)
    # orthographic: the whole disk
    proj = mapcore.make_projection('Orthographic', central_longitude=115, central_latitude=0)
    ax = fig.add_subplot(2, 1, 2, projection=proj)
    mapcore.set_map_domain(ax, proj, [0., 360., -90., 90.])
    assert ax.get_xlim() == pytest.approx(proj.x_limits) and ax.get_ylim() == pytest.approx(proj.y_limits)
    assert mapcore.projected_aspect(proj, [0., 360., -90., 90.]) == pytest.approx(1.0)
    # Mercator: all longitudes, the latitudes of the extent, and a latitude band
    proj = mapcore.make_projection('Mercator', central_longitude=180)
    ax = fig.add_subplot(3, 1, 3, projection=proj)
    mapcore.set_map_domain(ax, proj, [0., 360., -60., 60.])
    assert ax.get_xlim() == pytest.approx(proj.x_limits, rel=1e-3)
    _, y60 = proj.transform_point(0, 60, ccrs.PlateCarree())
    assert ax.get_ylim() == pytest.approx((-y60, y60), rel=1e-3)
    plt.close(fig)


def test_set_map_domain_platecarree_extent_and_global():
    extent = [140., -140., -20., 20.]
    proj, _, lon0, lon1 = mapcore.get_projections(extent)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    mapcore.set_map_domain(ax, proj, extent, lon0, lon1)
    assert ax.get_extent(crs=proj) == pytest.approx((-40., 40., -20., 20.))
    # all longitudes: set_global
    proj, _, lon0, lon1 = mapcore.get_projections([-180., 180., -60., 60.])
    ax2 = fig.add_subplot(2, 1, 2, projection=proj)
    mapcore.set_map_domain(ax2, proj, [-180., 180., -60., 60.], lon0, lon1)
    assert ax2.get_xlim() == pytest.approx((-180., 180.))
    plt.close(fig)


def test_aspect_of_the_map_domain():
    # lon/lat box in degrees, the longitude span is taken modulo 360
    assert mapcore.extent_aspect([140., -140., -20., 20.]) == pytest.approx(0.5)
    assert mapcore.extent_aspect([0., 180., 0., 90.]) == pytest.approx(0.5)
    assert mapcore.extent_aspect([-180., 180., -60., 60.]) == pytest.approx(120. / 360.)
    # global Robinson map of the 60S-60N band
    proj = mapcore.make_projection('Robinson', central_longitude=225)
    assert mapcore.projected_aspect(proj, [0., 360., -90., 90.], lat_range=(-60, 60)) == pytest.approx(0.374, abs=0.005)
    # a box on a projection that has no distortion in the middle: PlateCarree is the degrees ratio
    assert mapcore.projected_aspect(ccrs.PlateCarree(), [-30., 30., -10., 10.]) == pytest.approx(1. / 3., rel=1e-6)


def test_min_figure_height_for_the_titles():
    # the top margin (a fraction of the height) has room for the title of the figure (1.5 x font size) and of the map (1.2 x)
    assert mapcore.min_figure_height(10, 0.1) == pytest.approx(6.0)
    assert mapcore.min_figure_height(14, 0.1) == pytest.approx(7.8)
    assert mapcore.min_figure_height(10, 0.08) > mapcore.min_figure_height(10, 0.1)


def test_figure_margins_of_a_short_figure_are_inches():
    update = dict(top=0.9, bottom=0.08)
    # a figure that is high enough: the margins of the layout have room for the titles
    assert mapcore.figure_margins([10., 8.], update, 10) == ([10., 8.], None)
    # a map much wider than high: the maps keep their height, the margins above and below them are inches, not 10 % and 8 % of a short figure
    size, margins = mapcore.figure_margins([10., 3.7], update, 10)
    top, bottom = margins
    assert top == pytest.approx(0.6)                                            # 0.045 x 10 + 0.15: the 2 titles
    assert bottom == pytest.approx((10 + 7) / 72. + 0.15)                       # the longitude labels
    assert size[0] == 10. and size[1] == pytest.approx((0.9 - 0.08) * 3.7 + top + bottom, abs=0.01)
    assert size[1] < mapcore.min_figure_height(10, 0.1)                         # shorter than the old rule that made the figure taller
    # a larger font needs larger margins
    assert mapcore.figure_margins([10., 3.7], update, 14)[1][0] > top


def test_panel_fractions_and_figure_size():
    w, h = mapcore.panel_fractions(1, 1, [1], [1], left=0.1, right=0.9, bottom=0.1, top=0.9)
    assert (w, h) == pytest.approx((0.8, 0.8))
    # 2 rows of equal height with no space between: each panel is half of the plot height
    w, h = mapcore.panel_fractions(2, 1, [1, 1], [1], left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.0)
    assert (w, h) == pytest.approx((0.8, 0.4))
    # the figure is as high as the panel needs for the aspect that was asked for
    assert mapcore.figsize_for_aspect(10, 0.5, 0.8, 0.8) == [10.0, 5.0]
    assert mapcore.figsize_for_aspect(10, 0.5, 0.8, 0.4) == [10.0, 10.0]


def lambert_grid(nx=100, ny=80, dx=3000.):
    """lon / lat of a regular grid on the Lambert conformal projection of a WRF CONUS domain."""
    proj = ccrs.LambertConformal(central_longitude=-97., central_latitude=36., standard_parallels=(31., 35.))
    x = (np.arange(nx) - nx / 2) * dx
    y = (np.arange(ny) - ny / 2) * dx
    xx, yy = np.meshgrid(x, y)
    ll = ccrs.PlateCarree().transform_points(proj, xx, yy)
    return proj, xx, yy, ll[..., 0], ll[..., 1]


def test_data_extent_xy_of_data_on_the_grid_of_the_projection():
    proj, xx, yy, lon, lat = lambert_grid(nx=700, ny=500)          # 2100 x 1500 km
    extent = mapcore.data_extent_xy(proj, lon, lat)
    dx = 3000.
    # the box of the pixels and half a pixel on each side: no empty space around the data
    assert extent == pytest.approx([xx.min() - dx / 2, xx.max() + dx / 2, yy.min() - dx / 2, yy.max() + dx / 2], abs=5.)
    # the lon / lat box of the same data has another shape once projected (empty corners): 0.66 instead of 0.71
    box = [lon.min(), lon.max(), lat.min(), lat.max()]
    assert abs(mapcore.projected_aspect(proj, box) - 500. / 700.) > 0.03


def test_auto_map_domain_from_the_data():
    proj, xx, yy, lon, lat = lambert_grid()
    args = (-97., None, (31., 35.), None)
    extent_xy, aspect = mapcore.auto_map_domain(None, 'LambertConformal', *args, lon, lat)
    assert aspect == pytest.approx(80. / 100., rel=0.01)
    assert extent_xy[1] - extent_xy[0] == pytest.approx(100 * 3000., rel=0.01)
    # --extent: the box asked for, no domain from the data
    extent_xy, aspect = mapcore.auto_map_domain([-110., -90., 30., 40.], 'LambertConformal', *args, lon, lat)
    assert extent_xy is None and aspect > 0
    # PlateCarree: the lon / lat box of the data (the degrees ratio)
    extent_xy, aspect = mapcore.auto_map_domain(None, 'PlateCarree', None, None, None, None, lon, lat)
    assert extent_xy is None
    assert aspect == pytest.approx((lat.max() - lat.min()) / (lon.max() - lon.min()), rel=1e-6)
    # a global dataset: no domain in projected coordinates (the whole world), the shape of the whole map
    glon, glat = np.meshgrid(np.arange(-179.5, 180, 1.), np.arange(-60, 60.5, 1.))
    extent_xy, aspect = mapcore.auto_map_domain(None, 'Robinson', 225., None, None, (-60, 60), glon, glat)
    assert extent_xy is None and aspect == pytest.approx(0.374, abs=0.01)


def test_set_map_domain_with_the_extent_in_projected_coordinates():
    proj, xx, yy, lon, lat = lambert_grid()
    extent = mapcore.data_extent_xy(proj, lon, lat)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    mapcore.set_map_domain(ax, proj, [lon.min(), lon.max(), lat.min(), lat.max()], extent_xy=extent)
    assert ax.get_xlim() == pytest.approx(extent[:2], abs=1.) and ax.get_ylim() == pytest.approx(extent[2:], abs=1.)
    plt.close(fig)


def test_colorbars_follow_a_map_that_keeps_its_shape():
    # a tall map (30E x 60N: width / height 0.5) in a wide cell: it is centered in the cell, narrower than it
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.8], projection=ccrs.PlateCarree())
    ax.set_extent([0, 30, 0, 60], crs=ccrs.PlateCarree())
    ax.set_aspect('equal', adjustable='box')
    cax = fig.add_axes([0.82, 0.1, 0.03, 0.8])               # the colorbar of the cell: 0.02 of the figure from the cell
    mapcore.fit_colorbars_to_maps([(ax, [cax])], 'vertical')
    p, c = ax.get_position(), cax.get_position()
    assert p.width * 10 / (p.height * 4) == pytest.approx(0.5, rel=1e-3)         # the shape of the map
    assert (c.y0, c.height) == pytest.approx((p.y0, p.height))                  # the extent of the map
    assert c.x0 - p.x1 == pytest.approx(0.02, abs=1e-6) and c.width == pytest.approx(0.03)   # as far from the map as from the cell
    plt.close(fig)
    # a wide map (60E x 30N) in a tall cell, horizontal colorbar below it
    fig = plt.figure(figsize=(4, 10))
    ax = fig.add_axes([0.1, 0.2, 0.8, 0.7], projection=ccrs.PlateCarree())
    ax.set_extent([0, 60, 0, 30], crs=ccrs.PlateCarree())
    ax.set_aspect('equal', adjustable='box')
    cax = fig.add_axes([0.1, 0.12, 0.8, 0.02])
    mapcore.fit_colorbars_to_maps([(ax, [cax])], 'horizontal')
    p, c = ax.get_position(), cax.get_position()
    assert p.width * 4 / (p.height * 10) == pytest.approx(2.0, rel=1e-3)
    assert (c.x0, c.width) == pytest.approx((p.x0, p.width))
    assert p.y0 - (c.y0 + c.height) == pytest.approx(0.2 - (0.12 + 0.02), abs=1e-6)
    plt.close(fig)


def test_latitude_labels_on_the_curved_edge_of_a_world_map():
    # the 60S-60N band of a Robinson map centered on 225E in a tall figure: cartopy's own labels miss the middle latitudes
    extent = [0., 360., -90., 90.]
    proj, data_proj, l0, l1 = mapcore.get_projections(extent, 'Robinson', central_longitude=225)
    fig = plt.figure(figsize=(14, 11.5))
    ax = fig.add_axes([0.05, 0.3, 0.8, 0.4], projection=proj)
    ax.set_title('(a) IR Brightness Temperature', loc='left')
    mapcore.set_map_domain(ax, proj, extent, l0, l1, (-60, 60))
    gl = ax.gridlines(crs=data_proj, draw_labels=True)
    gl.right_labels = gl.top_labels = False
    assert mapcore.lat_labels_on_boundary(ax, proj, gl, extent, (-60, 60)) is True
    # the title is above the label of 60N (they do not overlap)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title_box = ax._left_title.get_window_extent(renderer)
    assert not any(title_box.overlaps(t.get_window_extent(renderer)) for t in ax.texts)
    assert gl.left_labels is False
    labels = [t.get_text() for t in ax.texts]
    assert labels == ['60°S', '40°S', '20°S', '0°', '20°N', '40°N', '60°N']
    # the labels are on the curved edge: further from the left edge of the axes for the high latitudes
    xs = [t.xy[0] for t in ax.texts]
    assert xs[0] > xs[1] > xs[2] > xs[3] and xs[3] < xs[4] < xs[5] < xs[6]
    # the ticks of --lat_ticks
    ax2 = fig.add_axes([0.05, 0.75, 0.8, 0.2], projection=proj)
    mapcore.set_map_domain(ax2, proj, extent, l0, l1, (-60, 60))
    gl2 = ax2.gridlines(crs=data_proj, draw_labels=True)
    assert mapcore.lat_labels_on_boundary(ax2, proj, gl2, extent, (-60, 60), latv=[-30, 0, 30, 80]) is True
    assert [t.get_text() for t in ax2.texts] == ['30°S', '0°', '30°N']
    plt.close(fig)
    # not for a regional map, nor for the other projections
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    gl = ax.gridlines(draw_labels=True)
    assert mapcore.lat_labels_on_boundary(ax, proj, gl, [100., 130., -10., 10.]) is False and not ax.texts
    pc = ccrs.PlateCarree()
    ax = fig.add_subplot(2, 1, 2, projection=pc)
    gl = ax.gridlines(draw_labels=True)
    assert mapcore.lat_labels_on_boundary(ax, pc, gl, extent) is False and not ax.texts
    plt.close(fig)


def test_longitude_helpers():
    assert mapcore.norm_lon(190.) == pytest.approx(-170.)
    assert mapcore.norm_lon(-190.) == pytest.approx(170.)
    np.testing.assert_allclose(mapcore.norm_lon(np.array([0., 180., 360., 545.])), [0., -180., 0., -175.])
    # dateline-safe, strict latitude limits
    extent = [160., -160., -10., 10.]
    assert mapcore.in_map_extent(170., 0., extent)
    assert mapcore.in_map_extent(-170., 0., extent)
    assert mapcore.in_map_extent(190., 5., extent)
    assert not mapcore.in_map_extent(0., 0., extent)
    assert not mapcore.in_map_extent(170., 10., extent)
    assert mapcore.in_map_extent(10., 0., [0., 30., -5., 5.])
    assert not mapcore.in_map_extent(-10., 0., [0., 30., -5., 5.])


def test_geodesic_circle_has_the_radius_around_the_center():
    from cartopy.geodesic import Geodesic
    lon0, lat0 = -64.72837, -32.12641                       # the CSAPR2 radar of CACTI
    for radius_km in (20., 100.):
        lons, lats = mapcore.geodesic_circle(lon0, lat0, radius_km, n_samples=100)
        assert len(lons) == len(lats) == 100
        assert (lons[0], lats[0]) == pytest.approx((lons[-1], lats[-1]))        # closed curve
        distance = Geodesic().inverse(np.tile([lon0, lat0], (len(lons), 1)), np.column_stack([lons, lats]))[:, 0]
        np.testing.assert_allclose(distance, radius_km * 1e3, atol=1.)           # on the ellipsoid, within 1 m
        # centered: the mean of the points is the center up to the curvature of the meridians (about 250 m for 100 km at 32S)
        assert np.mean(lons[:-1]) == pytest.approx(lon0, abs=5e-3)
        assert np.mean(lats[:-1]) == pytest.approx(lat0, abs=5e-3)
    # the center in 0-360 degrees (some radar locations are given that way): the circle is at the same place
    lons360, lats360 = mapcore.geodesic_circle(lon0 + 360., lat0, 100.)
    lons, lats = mapcore.geodesic_circle(lon0, lat0, 100.)
    np.testing.assert_allclose(lons360, lons, atol=1e-9)
    np.testing.assert_allclose(lats360, lats, atol=1e-9)


def test_geodesic_circle_at_the_dateline_is_one_continuous_curve():
    lons, lats = mapcore.geodesic_circle(179.9, 10., 150.)
    assert np.abs(np.diff(lons)).max() < 5.                       # no jump from +180 to -180
    assert lons.max() > 180. and lons.min() < 179.9               # it passes the dateline
    assert (np.mean(lons[:-1]), np.mean(lats[:-1])) == pytest.approx((179.9, 10.), abs=5e-3)
    # ... and is drawn on the other side of a PlateCarree map centered on the dateline
    proj = ccrs.PlateCarree(central_longitude=180)
    x = proj.transform_points(ccrs.PlateCarree(), lons, lats)[:, 0]
    assert np.abs(x).max() < 2.


def test_geodesic_ray_has_the_azimuth_and_the_length():
    from cartopy.geodesic import Geodesic
    lon0, lat0 = 100., -20.
    for azimuth in (0., 90., 225., 360.):
        lons, lats = mapcore.geodesic_ray(lon0, lat0, azimuth, 200., n_samples=21)
        assert len(lons) == len(lats) == 21
        assert (lons[0], lats[0]) == pytest.approx((lon0, lat0))
        out = Geodesic().inverse(np.array([[lon0, lat0]]), np.array([[lons[-1], lats[-1]]]))
        assert out[0, 0] == pytest.approx(200e3, abs=1.)
        assert (out[0, 1] % 360.) == pytest.approx(azimuth % 360., abs=1e-4)
        # equally spaced along the line
        step = Geodesic().inverse(np.column_stack([lons[:-1], lats[:-1]]), np.column_stack([lons[1:], lats[1:]]))[:, 0]
        np.testing.assert_allclose(step, 10e3, atol=1.)
    # north goes to higher latitude, east to higher longitude
    assert mapcore.geodesic_ray(lon0, lat0, 0., 200.)[1][-1] > lat0
    assert mapcore.geodesic_ray(lon0, lat0, 90., 200.)[0][-1] > lon0


def test_the_plot_code_has_no_tissot_circles_and_no_geodetic_transform():
    """
    ax.tissot and transform=ccrs.Geodetic() are moved north or south by up to 0.19 degree with PROJ 9.8 (Geodetic -> PlateCarree), so
    circles and lines made with them are not where the data are: the circles and lines are geodesic points in the transform of the data
    (mapcore.geodesic_circle, geodesic_ray). The calls are searched in the syntax tree (comments may mention them).
    """
    import ast
    import glob
    import os
    for path in sorted(glob.glob(os.path.join(os.path.dirname(mapcore.__file__), '*.py'))):
        for node in ast.walk(ast.parse(open(path, encoding='utf-8').read())):
            if isinstance(node, ast.Call):
                name = getattr(node.func, 'attr', None) or getattr(node.func, 'id', None)
                assert name not in ('tissot', 'Geodetic'), f"{os.path.basename(path)} line {node.lineno} calls {name}"
