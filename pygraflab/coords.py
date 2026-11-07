"""
Coordinate transformations between spherical, ellipsoidal, and Cartesian systems.

Ported from GrafLab 2.1.4 helper functions (lines 9597-9687).

All angles in **radians**, distances in **metres**.
"""

import numpy as np


def sph2cart(lam, fi, r):
    """Spherical (geocentric) -> Cartesian.

    Parameters
    ----------
    lam : float or ndarray  – longitude  [rad]
    fi  : float or ndarray  – spherical latitude [rad]
    r   : float or ndarray  – geocentric radius [m]

    Returns
    -------
    X, Y, Z : ndarray
    """
    cos_fi = np.cos(fi)
    X = r * cos_fi * np.cos(lam)
    Y = r * cos_fi * np.sin(lam)
    Z = r * np.sin(fi)
    return X, Y, Z


def cart2sph(X, Y, Z):
    """Cartesian -> spherical (geocentric).

    Returns
    -------
    lam, fi, r : ndarray
        longitude [rad], spherical latitude [rad], radius [m]
    """
    r = np.sqrt(X**2 + Y**2 + Z**2)
    fi = np.arctan2(Z, np.sqrt(X**2 + Y**2))
    lam = np.arctan2(Y, X)
    return lam, fi, r


def ell2cart(fi, lam, h, ellipsoid):
    """Ellipsoidal (geodetic) -> Cartesian.

    Parameters
    ----------
    fi        : float or ndarray – geodetic latitude  [rad]
    lam       : float or ndarray – longitude          [rad]
    h         : float or ndarray – ellipsoidal height [m]
    ellipsoid : dict             – must contain 'a' and 'e'

    Returns
    -------
    X, Y, Z : ndarray
    """
    a = ellipsoid['a']
    e = ellipsoid['e']
    sin_fi = np.sin(fi)
    cos_fi = np.cos(fi)
    N = a / np.sqrt(1.0 - e**2 * sin_fi**2)   # prime vertical radius
    X = (N + h) * cos_fi * np.cos(lam)
    Y = (N + h) * cos_fi * np.sin(lam)
    Z = ((1.0 - e**2) * N + h) * sin_fi
    return X, Y, Z


def cart2ell(X, Y, Z, ellipsoid):
    """Cartesian -> ellipsoidal (geodetic) via Bowring iteration.

    Returns
    -------
    fi, lam, h : ndarray
        geodetic latitude [rad], longitude [rad], ellipsoidal height [m]
    """
    a = ellipsoid['a']
    e = ellipsoid['e']
    e2 = e * e

    lam = np.arctan2(Y, X)
    p = np.sqrt(X**2 + Y**2)

    # Initial estimate
    fi = np.arctan2(Z, (1.0 - e2) * p)

    # Iterate until convergence
    for _ in range(20):
        sin_fi = np.sin(fi)
        N = a / np.sqrt(1.0 - e2 * sin_fi**2)
        h = p / np.cos(fi) - N
        fi_new = np.arctan2(Z, (1.0 - e2 * N / (N + h)) * p)
        if np.all(np.abs(fi_new - fi) < np.finfo(float).eps):
            fi = fi_new
            break
        fi = fi_new

    return fi, lam, h
