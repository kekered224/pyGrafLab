"""
Commission error computation via covariance propagation.

Ported from GrafLab 2.1.4 (lines 7140-8922).

For a functional f = f(Cnm, Snm), the commission error (standard deviation) is:

    sigma_f = sqrt( (partial f / partial x)^T  Sigma  (partial f / partial x) )

where x is the vector of spherical harmonic coefficients ordered as
  [C20, S20, C21, S21, C22, S22, C30, S30, ...]  (GrafLab packed order),
and Sigma is the corresponding variance-covariance matrix.

Only pointwise mode is supported (the design matrix is built per-point).
LNOF tensors (codes 8, 9, 14, 15) are NOT supported, matching MATLAB.
Only the standard forward column method (method 1) is used for fnALFs.
"""

import numpy as np
from .constants import get_ellipsoid, G_GRAV, RHO_CRUST, ARCSEC, MGAL, EOTVOS
from .coords import sph2cart, cart2sph, ell2cart, cart2ell
from .ellipsoid import compute_ellipsoid_coefficients, normal_gravity
from .fnalfs import (compute_fnalfs, fnalfs_first_derivative,
                     fnalfs_second_derivative)
from .io import load_covariance
from .synthesis import (DISTURBING_FUNCTIONALS, NEED_GAMMA, NEED_DTM,
                        NEED_DALFS, NEED_DDALFS, FUNCTIONAL_NAMES)


# Functionals supported for commission error (matches GrafLab restrictions)
COMMISSION_SUPPORTED = {2, 3, 5, 6, 7, 10, 11, 12, 13, 16, 17, 18, 19, 22, 24, 25}


def _cov_index(nmax_cov):
    """Build mapping (n, m, type) -> column index in the covariance matrix.

    The packed order (matching GrafLab ASCII format, starting from degree 2):
      [C20, S20, C21, S21, C22, S22, C30, S30, C31, S31, ...]
    For each degree n, order m runs from 0 to n. Each (n, m) pair has
    C-type and S-type, EXCEPT that S_{n,0} is absent (no Sn0 terms).
    """
    mapping = {}
    idx = 0
    for n in range(2, nmax_cov + 1):
        for m in range(0, n + 1):
            mapping[(n, m, 'C')] = idx
            idx += 1
            if m > 0:  # S_{n,0} does not exist
                mapping[(n, m, 'S')] = idx
                idx += 1
    return mapping, idx


def _prepare_pointwise_coords(lon_deg, lat_deg, h_or_r, coord_type,
                              ellipsoid):
    """Convert input coordinates to spherical (fiG, r) and ellipsoidal (fi)."""
    fi = np.deg2rad(np.atleast_1d(np.asarray(lat_deg, dtype=float)))
    lam = np.deg2rad(np.atleast_1d(np.asarray(lon_deg, dtype=float)))
    hr = np.atleast_1d(np.asarray(h_or_r, dtype=float))
    n_pts = len(fi)

    if coord_type == 0:  # ellipsoidal input
        X, Y, Z = ell2cart(fi, np.zeros(n_pts), hr, ellipsoid)
        _, fiG, r = cart2sph(X, Y, Z)
    else:  # spherical input
        fiG = fi.copy()
        r = hr.copy()
        X, Y, Z = sph2cart(np.zeros(n_pts), fiG, r)
        fi_ell, _, _ = cart2ell(X, Y, Z, ellipsoid)
        fi = fi_ell

    return fi, fiG, r, lam


def compute_commission(lon_deg, lat_deg, h_or_r, functional_codes,
                       covmat, nmax_cov, ggm, ellipsoid_spec,
                       coord_type=0, nmin=None):
    """Compute commission error (standard deviation) at scattered points.

    Parameters
    ----------
    lon_deg, lat_deg : array-like – coordinates in degrees
    h_or_r : array-like – ellipsoidal height (coord_type=0) or radius (coord_type=1)
    functional_codes : list of int (codes 2-25, LNOF 8/9/14/15 not supported)
    covmat : ndarray (n_cov x n_cov) – full symmetric covariance matrix
    nmax_cov : int – maximum degree of the covariance matrix
    ggm : GGMModel – the GGM (used for GM, R)
    ellipsoid_spec : int or list
    coord_type : int – 0=ellipsoidal, 1=spherical
    nmin : int or None – minimum degree for the error propagation (default 2)

    Returns
    -------
    sigma : dict – key=functional code, value=ndarray of standard deviations.
            Multi-component functionals (4, 6, 7, 12, 13, 16, 20) return
            an array of shape (n_pts, n_components).
    lon, lat : ndarray (degrees)
    """
    # Validate functional codes
    unsupported = set(functional_codes) - COMMISSION_SUPPORTED
    if unsupported:
        raise ValueError(
            f"Commission error not supported for functionals: "
            f"{sorted(unsupported)}. LNOF tensors (8, 9, 14, 15) and "
            f"functionals 4, 20, 21, 23 are not supported.")

    ellipsoid = get_ellipsoid(ellipsoid_spec)
    GM = ggm.GM
    R = ggm.R

    if nmin is None:
        nmin = 2
    if nmin < 2:
        raise ValueError("nmin must be >= 2 for commission error computation.")
    if nmin > nmax_cov:
        raise ValueError("nmin exceeds nmax of covariance matrix.")

    fc_set = set(functional_codes)
    need_dALFs = bool(fc_set & NEED_DALFS)
    need_ddALFs = bool(fc_set & NEED_DDALFS)
    need_por = bool(fc_set & DISTURBING_FUNCTIONALS)
    need_gamma = bool(fc_set & NEED_GAMMA)

    # Coordinate preparation
    fi, fiG, r, lam = _prepare_pointwise_coords(
        lon_deg, lat_deg, h_or_r, coord_type, ellipsoid)
    n_pts = len(fi)
    q = R / r
    cos_fiG = np.cos(fiG)

    # Build covariance index mapping
    cov_map, n_cov_total = _cov_index(nmax_cov)

    # Truncate covariance to nmax_cov (already assumed done by caller)
    # Verify dimensions
    if covmat.shape != (n_cov_total, n_cov_total):
        raise ValueError(
            f"Covariance matrix shape {covmat.shape} does not match "
            f"expected ({n_cov_total}, {n_cov_total}) for nmax_cov={nmax_cov}")

    # Precompute ellipsoid coefficients if needed
    CEl = None
    if need_por:
        CEl = compute_ellipsoid_coefficients(ellipsoid, R, GM)

    # Precompute normal gravity if needed
    gammaP = None
    if need_gamma:
        gammaP = normal_gravity(fi, h_or_r if coord_type == 0 else
                                r - ellipsoid['a'], ellipsoid)

    # Precompute fnALFs and derivatives for all orders m
    # Use method 1 (standard forward column) only
    fnalfs_cache = {}
    dfnalfs_cache = {}
    ddfnalfs_cache = {}
    for m in range(0, nmax_cov + 1):
        Pnm = compute_fnalfs(fiG, q, nmax_cov, m, method=1, precomp=None)
        # Apply nmin truncation
        if nmin > 2:
            m_min = max(0, nmin - m)
            if m_min > 0 and m_min <= Pnm.shape[1]:
                Pnm = Pnm.copy()
                Pnm[:, :m_min] = 0.0
        fnalfs_cache[m] = Pnm

        if need_dALFs:
            dPnm = fnalfs_first_derivative(Pnm, fiG, q, nmax_cov, m)
            dfnalfs_cache[m] = dPnm
        if need_ddALFs:
            if not need_dALFs:
                dPnm = fnalfs_first_derivative(Pnm, fiG, q, nmax_cov, m)
            ddPnm = fnalfs_second_derivative(Pnm, dPnm, fiG, nmax_cov, m)
            ddfnalfs_cache[m] = ddPnm

    # Allocate output sigma dictionary
    sigma = {}
    # Multi-component functionals get per-component sigma
    multi_comp = {
        6: 3, 7: 3, 12: 3, 13: 3, 16: 3,
    }

    # For each functional, build the partial derivative matrix and propagate
    for fc in functional_codes:
        n_comp = multi_comp.get(fc, 1)
        sigma_fc = np.zeros((n_pts, n_comp)) if n_comp > 1 else np.zeros(n_pts)

        # Build partial derivatives per (n, m) and accumulate variance
        # Using: sigma^2 = sum_{nm,n'm'} A_{i,nm} Sigma_{nm,n'm'} A_{i,n'm'}
        # We build the full design matrix row-by-row for each point
        # For efficiency, compute partials for all points simultaneously

        # Design matrix A: shape (n_pts, n_cov_total) for scalar functionals
        # For multi-component, shape (n_pts, n_cov_total, n_comp)
        if n_comp > 1:
            A = np.zeros((n_pts, n_cov_total, n_comp))
        else:
            A = np.zeros((n_pts, n_cov_total))

        for m in range(0, nmax_cov + 1):
            Pnm = fnalfs_cache[m]
            dPnm = dfnalfs_cache.get(m)
            ddPnm = ddfnalfs_cache.get(m)

            cos_m = np.cos(m * lam)
            sin_m = np.sin(m * lam)

            n_arr = np.arange(nmax_cov + 1, dtype=float)
            ampl_np1 = n_arr + 1
            ampl_nm1 = n_arr - 1
            ampl_np1_np2 = ampl_np1 * (n_arr + 2)

            # For each degree n at this order m
            for n in range(max(m, nmin), nmax_cov + 1):
                col_C = cov_map.get((n, m, 'C'))
                col_S = cov_map.get((n, m, 'S'))

                # Column index in Pnm matrix (column j corresponds to degree n=m+j)
                j = n - m
                P_n = Pnm[:, j]  # shape (n_pts,)
                dP_n = dPnm[:, j] if dPnm is not None else None
                ddP_n = ddPnm[:, j] if ddPnm is not None else None

                # Compute partial derivatives for this (n, m)
                # partial f / partial C_{nm}  and  partial f / partial S_{nm}
                dC, dS = _functional_partials(
                    fc, P_n, dP_n, ddP_n, m, n, cos_m, sin_m,
                    ampl_np1, ampl_nm1, ampl_np1_np2,
                    GM, R, r, fiG, cos_fiG, gammaP, need_por, CEl)

                if col_C is not None:
                    if n_comp > 1:
                        A[:, col_C, :] = dC
                    else:
                        A[:, col_C] = dC
                if col_S is not None and dS is not None:
                    if n_comp > 1:
                        A[:, col_S, :] = dS
                    else:
                        A[:, col_S] = dS

        # Variance propagation: sigma^2 = diag(A Sigma A^T)
        # For efficiency: AS = A @ Sigma, then sigma^2 = sum(AS * A, axis=1)
        if n_comp > 1:
            for c in range(n_comp):
                Ac = A[:, :, c]
                AS = Ac @ covmat
                sigma_fc[:, c] = np.sqrt(np.maximum(
                    np.sum(AS * Ac, axis=1), 0.0))
        else:
            AS = A @ covmat
            sigma_fc = np.sqrt(np.maximum(np.sum(AS * A, axis=1), 0.0))

        sigma[fc] = sigma_fc

    return sigma, np.rad2deg(lam), np.rad2deg(fi)


def _functional_partials(fc, P_n, dP_n, ddP_n, m, n, cos_m, sin_m,
                         ampl_np1, ampl_nm1, ampl_np1_np2,
                         GM, R, r, fiG, cos_fiG, gammaP, need_por, CEl):
    """Compute partial derivatives of functional fc w.r.t. C_{nm} and S_{nm}.

    Returns (dC, dS) arrays of shape (n_pts,) for scalar functionals,
    or (n_pts, n_comp) for multi-component functionals.
    """
    n_pts = len(P_n)

    # Helper: base term Pnm * cos/sin
    Pcos = P_n * cos_m   # partial w.r.t. C: P_{nm} cos(m lambda)
    Psin = P_n * sin_m   # partial w.r.t. S: P_{nm} sin(m lambda)

    # For disturbing functionals, partials w.r.t. deltaC are the same
    # as w.r.t. C (since deltaC_{nm} = C_{nm} - CEl_{nm}, and d/dC = d/d(deltaC)).
    # However, for fc==24 (T_rr), the partial is w.r.t. deltaC.

    # Degree-indexed amplification
    np1 = float(n + 1)
    nm1 = float(n - 1)
    np1_np2 = float((n + 1) * (n + 2))

    # ---- 2: eta (arcsec) ----
    if fc == 2:
        # eta = -GM/(r^2 gammaP cos_fiG) * sum_n sum_m m*Pnm*(-deltaC sin + S cos) * ARCSEC
        # partial w.r.t. C_{nm}: -GM/(r^2 gammaP cos_fiG) * m * Pnm * (-sin_m) * ARCSEC
        # partial w.r.t. S_{nm}: -GM/(r^2 gammaP cos_fiG) * m * Pnm * cos_m * ARCSEC
        scale = -GM / (r**2 * gammaP * cos_fiG) * ARCSEC
        pole = np.abs(np.abs(fiG) - np.pi / 2) < 1e-15
        dC = scale * m * P_n * (-sin_m)
        dS = scale * m * P_n * cos_m
        dC[pole] = 0.0
        dS[pole] = 0.0
        return dC, dS

    # ---- 3: xi (arcsec) ----
    elif fc == 3:
        scale = -GM / (r**2 * gammaP) * ARCSEC
        dC = scale * dP_n * cos_m
        dS = scale * dP_n * sin_m
        return dC, dS

    # ---- 5: T (m^2/s^2) ----
    elif fc == 5:
        scale = GM / r
        dC = scale * Pcos
        dS = scale * Psin
        return dC, dS

    # ---- 6: Trr, Tff, Tll (Eotvos, 3 components) ----
    elif fc == 6:
        dC = np.zeros((n_pts, 3))
        dS = np.zeros((n_pts, 3))
        # Trr
        sc_rr = GM / r**3 * EOTVOS
        dC[:, 0] = sc_rr * Pcos * np1_np2
        dS[:, 0] = sc_rr * Psin * np1_np2
        # Tff
        sc_ff = GM / r**3 * EOTVOS
        dC[:, 1] = sc_ff * ddP_n * cos_m
        dS[:, 1] = sc_ff * ddP_n * sin_m
        # -Tll (note the negative sign in output)
        sc_ll = -GM / (r**3 * cos_fiG**2) * EOTVOS
        dC[:, 2] = sc_ll * Pcos * m**2
        dS[:, 2] = sc_ll * Psin * m**2
        return dC, dS

    # ---- 7: Trf, Trl, Tfl (Eotvos, 3 components) ----
    elif fc == 7:
        dC = np.zeros((n_pts, 3))
        dS = np.zeros((n_pts, 3))
        # -Trf
        sc_rf = -GM / r**3 * EOTVOS
        dC[:, 0] = sc_rf * dP_n * cos_m * np1
        dS[:, 0] = sc_rf * dP_n * sin_m * np1
        # -Trl
        sc_rl = -GM / (r**3 * cos_fiG) * EOTVOS
        dC[:, 1] = sc_rl * P_n * m * (-sin_m) * np1
        dS[:, 1] = sc_rl * P_n * m * cos_m * np1
        # Tfl
        sc_fl = GM / (r**3 * cos_fiG) * EOTVOS
        dC[:, 2] = sc_fl * dP_n * m * (-sin_m)
        dS[:, 2] = sc_fl * dP_n * m * cos_m
        return dC, dS

    # ---- 10: Geoid undulation N ----
    elif fc == 10:
        scale = GM / (r * gammaP)
        dC = scale * Pcos
        dS = scale * Psin
        return dC, dS

    # ---- 11: V (gravitational potential) ----
    elif fc == 11:
        scale = GM / r
        dC = scale * Pcos
        dS = scale * Psin
        return dC, dS

    # ---- 12: Vrr, Vff, Vll (Eotvos, 3 components) ----
    elif fc == 12:
        dC = np.zeros((n_pts, 3))
        dS = np.zeros((n_pts, 3))
        sc_rr = GM / r**3 * EOTVOS
        dC[:, 0] = sc_rr * Pcos * np1_np2
        dS[:, 0] = sc_rr * Psin * np1_np2
        sc_ff = GM / r**3 * EOTVOS
        dC[:, 1] = sc_ff * ddP_n * cos_m
        dS[:, 1] = sc_ff * ddP_n * sin_m
        sc_ll = -GM / (r**3 * cos_fiG**2) * EOTVOS
        dC[:, 2] = sc_ll * Pcos * m**2
        dS[:, 2] = sc_ll * Psin * m**2
        return dC, dS

    # ---- 13: Vrf, Vrl, Vfl (Eotvos, 3 components) ----
    elif fc == 13:
        dC = np.zeros((n_pts, 3))
        dS = np.zeros((n_pts, 3))
        sc_rf = -GM / r**3 * EOTVOS
        dC[:, 0] = sc_rf * dP_n * cos_m * np1
        dS[:, 0] = sc_rf * dP_n * sin_m * np1
        sc_rl = -GM / (r**3 * cos_fiG) * EOTVOS
        dC[:, 1] = sc_rl * P_n * m * (-sin_m) * np1
        dS[:, 1] = sc_rl * P_n * m * cos_m * np1
        sc_fl = GM / (r**3 * cos_fiG) * EOTVOS
        dC[:, 2] = sc_fl * dP_n * m * (-sin_m)
        dS[:, 2] = sc_fl * dP_n * m * cos_m
        return dC, dS

    # ---- 16: Gravity vector (mGal, 3 components: gfi, glam, gr) ----
    elif fc == 16:
        dC = np.zeros((n_pts, 3))
        dS = np.zeros((n_pts, 3))
        # gfi (MGAL): partial of Wfi/r component
        # Wfi = GM/r * sum (C cos + S sin) dPnm
        # gfi = (GM/r * dPnm * cos/sin + centrifugal) / r * MGAL
        # Centrifugal has no Cnm/Snm dependence
        sc_fi = GM / r**2 * MGAL
        dC[:, 0] = sc_fi * dP_n * cos_m
        dS[:, 0] = sc_fi * dP_n * sin_m
        # glam (MGAL): partial of -Wlam/(r cos_fiG)
        # Wlam = GM/r * sum m*Pnm*(-C sin + S cos)
        # glam partial C: -GM/(r^2 cos_fiG) * m * Pnm * (-sin) * MGAL
        sc_lam = -GM / (r**2 * cos_fiG) * MGAL
        dC[:, 1] = sc_lam * P_n * m * (-sin_m)
        dS[:, 1] = sc_lam * P_n * m * cos_m
        # gr (MGAL): partial of Wr component
        # Wr = -GM/r^2 * sum (n+1)*Pnm*(C cos + S sin) + centrifugal
        sc_r = -GM / r**2 * MGAL
        dC[:, 2] = sc_r * Pcos * np1
        dS[:, 2] = sc_r * Psin * np1
        return dC, dS

    # ---- 17: g_sa (mGal, scalar) ----
    elif fc == 17:
        # g_sa = |GM/r^2 * sum (n+1)*Pnm*(C cos+S sin) - centrifugal| * MGAL
        # For the partial, ignore centrifugal (no Cnm dependence), take absolute at end
        sc = GM / r**2 * MGAL
        dC = sc * Pcos * np1
        dS = sc * Psin * np1
        return dC, dS

    # ---- 18: W (gravity potential) ----
    elif fc == 18:
        scale = GM / r
        dC = scale * Pcos
        dS = scale * Psin
        return dC, dS

    # ---- 19: gravity anomaly sa (mGal) ----
    elif fc == 19:
        sc = GM / r**2 * MGAL
        dC = sc * Pcos * nm1
        dS = sc * Psin * nm1
        return dC, dS

    # ---- 22: height anomaly ell (m) ----
    elif fc == 22:
        scale = GM / (r * gammaP)
        dC = scale * Pcos
        dS = scale * Psin
        return dC, dS

    # ---- 24: T_rr (Eotvos) ----
    elif fc == 24:
        sc = GM / r**3 * EOTVOS
        dC = sc * Pcos * np1_np2
        dS = sc * Psin * np1_np2
        return dC, dS

    # ---- 25: W_rr (Eotvos) ----
    elif fc == 25:
        sc = GM / r**3 * EOTVOS
        dC = sc * Pcos * np1_np2
        dS = sc * Psin * np1_np2
        return dC, dS

    else:
        raise ValueError(f"Commission error not implemented for fc={fc}")
