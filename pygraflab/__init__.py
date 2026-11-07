"""
pyGrafLab — Python port of GrafLab 2.1.4 spherical harmonic synthesis engine.

Provides script-level access to:
- GGM / DTM / covariance file I/O
- 24 geopotential functionals (gravity, potential, deflection, tensors, ...)
- Grid and pointwise synthesis modes
- Three fnALF computation methods (standard, modified, extended-range)
- Commission error propagation
- Data export (txt / mat) and computation reports

Example
-------
>>> import pygraflab
>>> ggm = pygraflab.load_ggm('EGM2008.gfc', nmax=360)
>>> results, lon, lat = pygraflab.compute(
...     ggm, functional_codes=[11, 17],
...     mode='grid', ellipsoid=1,
...     lon_min=0, lon_max=20, lon_step=1,
...     lat_min=-10, lat_max=10, lat_step=1,
...     h=0, method=1)
>>> pygraflab.export_txt(results, 'output/')
"""

from .io import load_ggm, load_dtm, load_covariance, GGMModel, DTMModel
from .constants import GRS80, WGS84, get_ellipsoid, G_GRAV, RHO_CRUST
from .synthesis import (
    synthesize_grid, synthesize_pointwise,
    FUNCTIONAL_NAMES,
)
from .commission import compute_commission
from .export import export_txt, export_grid_txt, export_mat, generate_report
from .ellipsoid import compute_ellipsoid_coefficients, normal_gravity
from .coords import sph2cart, cart2sph, ell2cart, cart2ell
from .fnalfs import compute_fnalfs

import time
import numpy as np


try:  # report the installed distribution's version, or "dev" from a checkout
    from importlib.metadata import PackageNotFoundError, version as _dist_version
    __version__ = _dist_version("pygraflab")
except Exception:  # pragma: no cover - metadata absent in a source tree
    __version__ = "dev"

__all__ = [
    # I/O
    'load_ggm', 'load_dtm', 'load_covariance',
    'GGMModel', 'DTMModel',
    # Synthesis
    'synthesize_grid', 'synthesize_pointwise', 'compute',
    'FUNCTIONAL_NAMES',
    # Commission
    'compute_commission',
    # Export
    'export_txt', 'export_grid_txt', 'export_mat', 'generate_report',
    # Constants & helpers
    'GRS80', 'WGS84', 'get_ellipsoid', 'G_GRAV', 'RHO_CRUST',
    'compute_ellipsoid_coefficients', 'normal_gravity',
    'sph2cart', 'cart2sph', 'ell2cart', 'cart2ell',
    'compute_fnalfs',
]


def compute(ggm, functional_codes, mode='grid', ellipsoid=1,
            lon_min=None, lon_max=None, lon_step=None,
            lat_min=None, lat_max=None, lat_step=None,
            h=0.0,
            lon_deg=None, lat_deg=None, h_or_r=None,
            coord_type=0, method=1,
            dtm=None, nmin=None,
            commission=False, covmat=None, nmax_cov=None):
    """High-level entry point for spherical harmonic synthesis.

    Dispatches to :func:`synthesize_grid` or :func:`synthesize_pointwise`
    depending on ``mode``.

    Parameters
    ----------
    ggm : GGMModel
        Loaded global geopotential model.
    functional_codes : list of int
        Functional codes (2–25). See FUNCTIONAL_NAMES for the full list.
    mode : str
        ``'grid'`` (regular grid) or ``'pointwise'`` (scattered points).
    ellipsoid : int or list
        1 = GRS80, 2 = WGS84, or [GM, a, e, C20, omega].
    lon_min, lon_max, lon_step : float
        Grid longitude range and step (degrees). Required if mode='grid'.
    lat_min, lat_max, lat_step : float
        Grid latitude range and step (degrees). Required if mode='grid'.
    h : float
        Ellipsoidal height for grid mode (metres). Default 0.
    lon_deg, lat_deg, h_or_r : array-like
        Point coordinates for pointwise mode. Required if mode='pointwise'.
    coord_type : int
        0 = ellipsoidal coordinates, 1 = spherical. Default 0.
    method : int
        fnALF method: 1 = standard, 2 = modified, 3 = extended-range.
    dtm : DTMModel or None
        Digital terrain model (needed for geoid/height anomaly).
    nmin : int or None
        Minimum harmonic degree (default: GGM's nmin).
    commission : bool
        Compute commission error (requires covmat).
    covmat : ndarray or None
        Coefficient covariance matrix.
    nmax_cov : int or None
        Maximum degree of the covariance matrix.

    Returns
    -------
    results : dict
        functional_code -> ndarray.
    lon : ndarray
        Longitude vector (grid) or input longitudes (pointwise), degrees.
    lat : ndarray
        Latitude vector (grid) or input latitudes (pointwise), degrees.
    commission_sigma : dict or None
        functional_code -> ndarray of standard deviations (only if commission=True).
    """
    t0 = time.time()

    if mode == 'grid':
        if any(v is None for v in (lon_min, lon_max, lon_step,
                                    lat_min, lat_max, lat_step)):
            raise ValueError(
                "Grid mode requires lon_min, lon_max, lon_step, "
                "lat_min, lat_max, lat_step.")

        results, lon_vec, lat_vec = synthesize_grid(
            ggm, functional_codes, ellipsoid,
            lon_min, lon_step, lon_max,
            lat_min, lat_step, lat_max,
            h, method=method, dtm=dtm, nmin=nmin, coord_type=coord_type)

        lon_out, lat_out = lon_vec, lat_vec

    elif mode == 'pointwise':
        if lon_deg is None or lat_deg is None or h_or_r is None:
            raise ValueError(
                "Pointwise mode requires lon_deg, lat_deg, h_or_r.")

        results, lon_out, lat_out = synthesize_pointwise(
            ggm, functional_codes, ellipsoid,
            lon_deg, lat_deg, h_or_r,
            coord_type=coord_type, method=method,
            dtm=dtm, nmin=nmin)

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'grid' or 'pointwise'.")

    # Commission error (pointwise only — grid commission not implemented)
    commission_sigma = None
    if commission and covmat is not None:
        if mode == 'pointwise':
            commission_sigma = compute_commission(
                lon_deg, lat_deg, h_or_r,
                functional_codes, covmat,
                nmax_cov if nmax_cov else ggm.nmax,
                ggm, ellipsoid,
                coord_type=coord_type, nmin=nmin)
        else:
            import warnings
            warnings.warn("Commission error for grid mode is not yet "
                          "supported. Use pointwise mode instead.")

    elapsed = time.time() - t0
    # Attach timing as attribute (not returned, but accessible)
    compute._last_elapsed = elapsed

    return results, lon_out, lat_out, commission_sigma
