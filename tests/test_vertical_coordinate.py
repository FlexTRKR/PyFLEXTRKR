"""
Unit tests for standardize_vertical_coordinate() in idcells_reflectivity.py.

This function was recently refactored to accept a config dict and support
scale_factor / units_override options. Tests here directly guard against
regressions in that logic.

No external data needed — all tests run on small numpy arrays.
"""

import numpy as np
import pytest

from pyflextrkr.idcells_reflectivity import standardize_vertical_coordinate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(**kwargs):
    """Build a minimal config dict; caller overrides individual keys."""
    base = {'z_coord_type': 'height'}
    base.update(kwargs)
    return base


METRES = np.array([0.0, 1000.0, 2000.0, 5000.0])
KILOMETRES = METRES / 1000.0
PASCALS = np.array([101325.0, 85000.0, 50000.0, 20000.0])
HPA    = PASCALS * 0.01


# ---------------------------------------------------------------------------
# Height coordinate (target: metres)
# ---------------------------------------------------------------------------

class TestHeightCoordinate:
    def test_metres_no_conversion(self):
        vals, msg = standardize_vertical_coordinate(
            METRES.copy(), {'units': 'm'}, 'z_coord', _cfg()
        )
        assert np.allclose(vals, METRES), "Metres input should be returned unchanged"

    def test_kilometres_converted_to_metres(self):
        vals, msg = standardize_vertical_coordinate(
            KILOMETRES.copy(), {'units': 'km'}, 'z_coord', _cfg()
        )
        assert np.allclose(vals, METRES), \
            f"km → m conversion failed: {vals} vs {METRES}"

    def test_meters_spelled_out(self):
        vals, _ = standardize_vertical_coordinate(
            METRES.copy(), {'units': 'meters'}, 'z_coord', _cfg()
        )
        assert np.allclose(vals, METRES)

    def test_kilometers_spelled_out(self):
        vals, _ = standardize_vertical_coordinate(
            KILOMETRES.copy(), {'units': 'kilometers'}, 'z_coord', _cfg()
        )
        assert np.allclose(vals, METRES)

    def test_geopotential_height_raises(self):
        with pytest.raises(ValueError, match="(?i)geopotential"):
            standardize_vertical_coordinate(
                METRES.copy(), {'units': 'gpm'}, 'z_coord', _cfg()
            )

    def test_unknown_units_raises(self):
        with pytest.raises(ValueError):
            standardize_vertical_coordinate(
                METRES.copy(), {'units': 'furlongs'}, 'z_coord', _cfg()
            )

    def test_no_units_attribute_returns_unchanged(self):
        """Missing 'units' key should warn and return values as-is (no crash)."""
        vals, msg = standardize_vertical_coordinate(
            METRES.copy(), {}, 'z_coord', _cfg()
        )
        assert np.allclose(vals, METRES), \
            "Missing units attribute should return values unchanged"


# ---------------------------------------------------------------------------
# Pressure coordinate (target: hPa)
# ---------------------------------------------------------------------------

class TestPressureCoordinate:
    def test_hpa_no_conversion(self):
        vals, _ = standardize_vertical_coordinate(
            HPA.copy(), {'units': 'hPa'}, 'z_coord', _cfg(z_coord_type='pressure')
        )
        assert np.allclose(vals, HPA)

    def test_mb_no_conversion(self):
        vals, _ = standardize_vertical_coordinate(
            HPA.copy(), {'units': 'mb'}, 'z_coord', _cfg(z_coord_type='pressure')
        )
        assert np.allclose(vals, HPA)

    def test_pascal_to_hpa(self):
        vals, _ = standardize_vertical_coordinate(
            PASCALS.copy(), {'units': 'Pa'}, 'z_coord', _cfg(z_coord_type='pressure')
        )
        assert np.allclose(vals, HPA), \
            f"Pa → hPa conversion failed: {vals} vs {HPA}"

    def test_millibar_no_conversion(self):
        vals, _ = standardize_vertical_coordinate(
            HPA.copy(), {'units': 'millibar'}, 'z_coord', _cfg(z_coord_type='pressure')
        )
        assert np.allclose(vals, HPA)

    def test_unknown_pressure_units_raises(self):
        with pytest.raises(ValueError):
            standardize_vertical_coordinate(
                PASCALS.copy(), {'units': 'atm'}, 'z_coord',
                _cfg(z_coord_type='pressure')
            )


# ---------------------------------------------------------------------------
# scale_factor override (highest priority)
# ---------------------------------------------------------------------------

class TestScaleFactor:
    def test_scale_factor_overrides_units(self):
        """scale_factor applies before unit detection — even if units are wrong."""
        vals, msg = standardize_vertical_coordinate(
            KILOMETRES.copy(),
            {'units': 'WRONG_UNITS'},   # would normally raise
            'z_coord',
            _cfg(z_coord_scale_factor=1000.0),
        )
        assert np.allclose(vals, METRES), \
            "scale_factor=1000 on km input should yield metres"
        assert "scale" in msg.lower() or "1000" in msg, \
            "Message should mention the scale factor applied"

    def test_scale_factor_one_is_identity(self):
        vals, _ = standardize_vertical_coordinate(
            METRES.copy(), {'units': 'm'}, 'z_coord',
            _cfg(z_coord_scale_factor=1.0)
        )
        assert np.allclose(vals, METRES)

    def test_sfc_elev_scale_factor_used_for_sfc_coord(self):
        """sfc_elev_scale_factor should be picked when coord_name contains 'sfc'."""
        vals, _ = standardize_vertical_coordinate(
            KILOMETRES.copy(),
            {'units': 'km'},
            'sfc_elev',                              # triggers sfc branch
            _cfg(sfc_elev_scale_factor=1000.0),
        )
        assert np.allclose(vals, METRES)

    def test_z_coord_scale_factor_not_used_for_sfc_coord(self):
        """z_coord_scale_factor should NOT apply when coord_name is 'sfc_elev'."""
        # sfc_elev_scale_factor is None → falls through to units detection
        # 'km' units → should still convert correctly via units detection
        vals, _ = standardize_vertical_coordinate(
            KILOMETRES.copy(),
            {'units': 'km'},
            'sfc_elev',
            _cfg(z_coord_scale_factor=1000.0),  # only z_coord_ prefix, not sfc_elev_
        )
        # Falls back to unit-based conversion (km → m ×1000)
        assert np.allclose(vals, METRES)


# ---------------------------------------------------------------------------
# units_override
# ---------------------------------------------------------------------------

class TestUnitsOverride:
    def test_units_override_replaces_file_units(self):
        """units_override='km' should cause km→m conversion ignoring file attrs."""
        vals, _ = standardize_vertical_coordinate(
            KILOMETRES.copy(),
            {'units': 'WRONG'},          # file says wrong unit
            'z_coord',
            _cfg(z_coord_units_override='km'),
        )
        assert np.allclose(vals, METRES), \
            "units_override='km' should convert km to m"

    def test_units_override_lower_priority_than_scale_factor(self):
        """scale_factor wins over units_override."""
        vals, _ = standardize_vertical_coordinate(
            KILOMETRES.copy(),
            {},
            'z_coord',
            _cfg(z_coord_scale_factor=1000.0, z_coord_units_override='m'),
        )
        assert np.allclose(vals, METRES)


# ---------------------------------------------------------------------------
# Invalid coord_type
# ---------------------------------------------------------------------------

def test_invalid_coord_type_raises():
    with pytest.raises(ValueError, match="coord_type"):
        standardize_vertical_coordinate(
            METRES.copy(), {'units': 'm'}, 'z_coord',
            {'z_coord_type': 'temperature'}
        )
