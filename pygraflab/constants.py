"""
Physical constants, reference ellipsoid parameters, and unit conversion factors.

Ported from GrafLab 2.1.4 (Bucha & Janak, 2013).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Gravitational constant and crustal density (Moritz, 2000; Pavlis et al.)
# ---------------------------------------------------------------------------
G_GRAV = 6.67259e-11        # Universal gravitational constant  [m^3 kg^-1 s^-2]
RHO_CRUST = 2670.0          # Mean crustal density              [kg m^-3]

# ---------------------------------------------------------------------------
# Unit conversion factors
# ---------------------------------------------------------------------------
M2S2 = 1.0                  # potential: m^2 s^-2 (no conversion)
MGAL = 1.0e5                # gravity:  1 m/s^2 = 10^5 mGal
EOTVOS = 1.0e9              # gradient: 1 s^-2  = 10^9 Eotvos
ARCSEC = (180.0 / np.pi) * 3600.0   # rad -> arcsec

# ---------------------------------------------------------------------------
# Reference ellipsoids
# ---------------------------------------------------------------------------
# Each ellipsoid is stored as a dict with keys:
#   name, GM, a, f, e, omega, C20
# where:
#   GM    = geocentric gravitational constant  [m^3 s^-2]
#   a     = semi-major axis                    [m]
#   f     = flattening                         [-]
#   e     = first eccentricity                 [-]
#   omega = angular velocity                   [rad s^-1]
#   C20   = fully-normalised C20 coefficient   [-]

def _make_ellipsoid(name, GM, a, f, omega, C20):
    e = np.sqrt(f * (2.0 - f))
    return {
        'name': name,
        'GM': GM,
        'a': a,
        'f': f,
        'e': e,
        'omega': omega,
        'C20': C20,
    }


GRS80 = _make_ellipsoid(
    name='GRS80',
    GM=3986005.0e8,                        # m^3 s^-2
    a=6378137.0,                           # m
    f=1.0 / 298.257222100882711,           # derived from GRS80 defining constants
    omega=7292115.0e-11,                   # rad s^-1
    C20=-108263.0e-8 / np.sqrt(5.0),       # fully normalised
)

WGS84 = _make_ellipsoid(
    name='WGS84',
    GM=3986004.418e8,                      # m^3 s^-2
    a=6378137.0,                           # m
    f=1.0 / 298.257223563,                 # NIMA 2000
    omega=7292115.0e-11,                   # rad s^-1
    C20=-0.484166774985e-3,                # fully normalised
)

# Lookup by integer code (matches GrafLab convention)
ELLIPSOIDS = {1: GRS80, 2: WGS84}


def get_ellipsoid(spec):
    """Return an ellipsoid dict.

    Parameters
    ----------
    spec : int or array-like
        1 -> GRS80, 2 -> WGS84.
        If a sequence of 5 values [GM, a, e, C20, omega], build a custom
        ellipsoid (the flattening *f* is derived from *e*).
    """
    if isinstance(spec, (int, np.integer)):
        return ELLIPSOIDS[int(spec)]
    arr = np.asarray(spec, dtype=float)
    if arr.size == 5:
        GM, a, e, C20, omega = arr
        f = 1.0 - np.sqrt(1.0 - e * e)
        return _make_ellipsoid('Custom', GM, a, f, omega, C20)
    raise ValueError("Ellipsoid spec must be 1, 2, or a 5-element vector "
                     "[GM, a, e, C20, omega].")
