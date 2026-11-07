"""
Spherical harmonic synthesis engine – grid and pointwise modes.

Combines fnALF computation, lumped-coefficient accumulation, and
final functional evaluation for all 24 geopotential functionals.

Ported from GrafLab 2.1.4 (lines 2083-4983, 4984-7139).
"""

import math

import numpy as np
from .constants import GRS80, WGS84, get_ellipsoid, G_GRAV, RHO_CRUST, ARCSEC, MGAL, EOTVOS
from .coords import sph2cart, cart2sph, ell2cart, cart2ell
from .ellipsoid import compute_ellipsoid_coefficients, normal_gravity
from .fnalfs import (compute_fnalfs, fnalfs_first_derivative,
                     fnalfs_second_derivative, _precompute_extended)
from .io import GGMModel, DTMModel


# ===================================================================
#  Functional metadata
# ===================================================================

FUNCTIONAL_NAMES = {
    2:  'Deflection_eta',
    3:  'Deflection_xi',
    4:  'Deflection_Theta',
    5:  'Disturbing_potential_T',
    6:  'Disturbing_tensor_Trr_Tpp_Tll',
    7:  'Disturbing_tensor_Trf_Trl_Tfl',
    8:  'Disturbing_tensor_LNOF_Txx_Tyy_Tzz',
    9:  'Disturbing_tensor_LNOF_Txy_Txz_Tyz',
    10: 'Geoid_undulation_N',
    11: 'Gravitational_potential_V',
    12: 'Gravitational_tensor_Vrr_Vpp_Vll',
    13: 'Gravitational_tensor_Vrf_Vrl_Vfl',
    14: 'Gravitational_tensor_LNOF_Vxx_Vyy_Vzz',
    15: 'Gravitational_tensor_LNOF_Vxy_Vxz_Vyz',
    16: 'Gravity_vector_gX_gY_gZ',
    17: 'Gravity_sa',
    18: 'Gravity_potential_W',
    19: 'Gravity_anomaly_sa',
    20: 'Gravity_disturbance',
    21: 'Gravity_disturbance_sa',
    22: 'Height_anomaly_ell',
    23: 'Height_anomaly',
    24: 'T_rr',
    25: 'W_rr',
}

# Which functionals use disturbing field (deltaC = C - CEl)
DISTURBING_FUNCTIONALS = {2, 3, 4, 5, 6, 7, 8, 9, 10, 19, 21, 22, 23, 24}

# Which functionals use full gravitational field (C)
GRAVITATIONAL_FUNCTIONALS = {11, 12, 13, 14, 15, 16, 17, 18, 20, 25}

# Which functionals need normal gravity gammaP
NEED_GAMMA = {2, 3, 4, 10, 22, 23}

# Which functionals need DTM
NEED_DTM = {10, 23}

# Which functionals need first-order fnALF derivatives
NEED_DALFS = {2, 3, 4, 7, 9, 13, 15, 16, 20}

# Which functionals need second-order fnALF derivatives
NEED_DDALFS = {6, 12}

# Which functionals need CEl (normal field coefficients)
NEED_NORMAL = {20}

# Guard rail: refuses grids that would exhaust memory instead of crashing
MAX_GRID_POINTS = 2_000_000


# ===================================================================
#  Index helper – extract coefficients for a given order m
# ===================================================================

def _extract_coeffs(C, S, index, m, nmin_GGM, nmax):
    """Extract Cnm, Snm for degree n = max(m,nmin_GGM)..nmax at order m."""
    n_start = max(m, nmin_GGM)
    if n_start > nmax:
        return np.array([]), np.array([])
    # Coefficients of order m for degrees n_start..nmax occupy, in the packed
    # layout, one contiguous slot per degree: index[n - nmin_GGM] + m.  The
    # per-degree start offsets are therefore consecutive, so a single slice
    # replaces the element-wise loop over the (up to 2190) degrees.
    rows = np.asarray(index[n_start - nmin_GGM:nmax - nmin_GGM + 1]) + m
    return C[rows], S[rows]


def _extract_coeffs_aligned(C, S, index, order, n_lo, nmin_GGM, nmax):
    """Extract coefficients at `order`, aligned to degree range n_lo..nmax.

    Degrees below max(order, nmin_GGM) do not exist and are returned as zero,
    so the result always has length nmax - n_lo + 1 and can be multiplied
    directly by fnALF matrices whose columns start at degree n_lo.
    """
    n_len = nmax - n_lo + 1
    C_out = np.zeros(n_len)
    S_out = np.zeros(n_len)
    if order < 0 or order > nmax:
        return C_out, S_out
    start = max(order, n_lo, nmin_GGM)
    if start > nmax:
        return C_out, S_out
    Cs, Ss = _extract_coeffs(C, S, index, order, nmin_GGM, nmax)
    # _extract_coeffs starts at max(order, nmin_GGM); drop the leading
    # degrees that lie below n_lo.
    skip = start - max(order, nmin_GGM)
    take = nmax - start + 1
    Cs = Cs[skip:skip + take]
    Ss = Ss[skip:skip + take]
    off = start - n_lo
    C_out[off:off + len(Cs)] = Cs
    S_out[off:off + len(Ss)] = Ss
    return C_out, S_out


def _extract_coeffs_shifted(C, S, index, m_shift, nmin_GGM, nmax, m_orig):
    """Extract Cnm, Snm for degree n = m_orig..nmax at shifted order m_shift.

    For LNOF tensors that need C_{n, m+/-2} or C_{n, m+/-1}.
    """
    n_start = max(m_shift, m_orig)
    if n_start > nmax:
        return np.array([]), np.array([])
    rows = np.asarray(index[n_start - nmin_GGM:nmax - nmin_GGM + 1]) + m_shift
    if np.any(rows >= len(C)):
        return np.array([]), np.array([])
    return C[rows], S[rows]


# ===================================================================
#  Main synthesis function – GRID mode
# ===================================================================

def synthesize_grid(ggm, functional_codes, ellipsoid_spec,
                    lon_min, lon_step, lon_max,
                    lat_min, lat_step, lat_max, h,
                    method=1, dtm=None, nmin=None, coord_type=0):
    """Compute geopotential functionals on a regular grid.

    Parameters
    ----------
    ggm        : GGMModel
    functional_codes : list of int (codes 2-25)
    ellipsoid_spec : int or list
    lon_min/step/max : float (degrees)
    lat_min/step/max : float (degrees)
    h          : float – height above reference surface [m]
    method     : int – 1, 2, or 3 (fnALF method)
    dtm        : DTMModel or None
    nmin       : int or None – minimum degree (default: ggm.nmin)

    Returns
    -------
    results : dict  – key=functional code, value=ndarray
    lon_vec, lat_vec : ndarray (degrees)
    """
    ellipsoid = get_ellipsoid(ellipsoid_spec)
    GM = ggm.GM
    R = ggm.R
    nmin_GGM = ggm.nmin
    nmax = ggm.nmax

    # ---- input validation ------------------------------------------------
    lon_min, lon_max = float(lon_min), float(lon_max)
    lat_min, lat_max = float(lat_min), float(lat_max)
    lon_step, lat_step = float(lon_step), float(lat_step)
    h = float(h)

    for label, value in (("lon_min", lon_min), ("lon_max", lon_max),
                         ("lat_min", lat_min), ("lat_max", lat_max),
                         ("lon_step", lon_step), ("lat_step", lat_step), ("h", h)):
        if not math.isfinite(value):
            raise ValueError(f"'{label}' must be finite.")
    if lon_step <= 0 or lat_step <= 0:
        raise ValueError("lon_step and lat_step must be positive.")
    if lon_min > lon_max or lat_min > lat_max:
        raise ValueError("Grid bounds must satisfy min <= max.")
    if not (-90.0 <= lat_min and lat_max <= 90.0):
        raise ValueError("Latitude bounds must lie within [-90, 90].")
    if method not in (1, 2, 3):
        raise ValueError("method must be 1 (standard), 2 (modified) or 3 (extended).")
    if coord_type not in (0, 1):
        raise ValueError("coord_type must be 0 (ellipsoidal) or 1 (spherical).")

    functional_codes = [int(code) for code in functional_codes]
    unknown = [code for code in functional_codes if code not in FUNCTIONAL_NAMES]
    if unknown:
        raise ValueError(f"Unknown functional code(s): {unknown}")
    if not functional_codes:
        raise ValueError("At least one functional code is required.")

    n_lon = int(round((lon_max - lon_min) / lon_step)) + 1
    n_lat = int(round((lat_max - lat_min) / lat_step)) + 1
    n_points = n_lon * n_lat
    if n_points > MAX_GRID_POINTS:
        raise ValueError(
            f"Grid too large: {n_lon} x {n_lat} = {n_points:,} points "
            f"(limit {MAX_GRID_POINTS:,}). Increase the step size.")
    if coord_type == 1 and h <= 0:
        raise ValueError("With coord_type=1 the height argument is a geocentric radius and must be > 0.")

    if nmin is None:
        nmin = nmin_GGM
    nmin = int(nmin)
    if nmin < 0 or nmin > nmax:
        raise ValueError(f"nmin must satisfy 0 <= nmin <= model nmax ({nmax}).")

    # Build latitude/longitude vectors
    lat_vec = np.arange(lat_min, lat_max + lat_step * 0.5, lat_step)
    lon_vec = np.arange(lon_min, lon_max + lon_step * 0.5, lon_step)
    length_fi = len(lat_vec)
    length_lam = len(lon_vec)

    fi_deg = lat_vec
    lam_deg = lon_vec
    fi = np.deg2rad(fi_deg)
    lam = np.deg2rad(lam_deg)
    h_val = float(h)

    # Method 2 is a point-by-point summation and does not produce Fourier
    # coefficients; evaluate it through the pointwise path.  The arrays are
    # returned flattened, exactly like the other methods.
    if method == 2:
        lat2d, lon2d = np.meshgrid(lat_vec, lon_vec, indexing='ij')
        results, _, _ = synthesize_pointwise(
            ggm, functional_codes, ellipsoid_spec,
            lon2d.ravel(), lat2d.ravel(), np.full(lat2d.size, h_val),
            coord_type=coord_type, method=2, dtm=dtm, nmin=nmin)
        return results, lam_deg, fi_deg

    # Coordinate conversion
    if coord_type == 0:  # ellipsoidal
        X, Y, Z = ell2cart(fi, np.zeros(length_fi), np.full(length_fi, h_val), ellipsoid)
        _, fiG, r = cart2sph(X, Y, Z)
    else:  # spherical (h is the geocentric radius)
        fiG = fi.copy()
        r = np.full(length_fi, h_val)
        X, Y, Z = sph2cart(np.zeros(length_fi), fiG, r)
        fi_ell, _, _ = cart2ell(X, Y, Z, ellipsoid)
        fi = fi_ell

    # Check equatorial symmetry
    symmetric_grid = False
    n_half = length_fi
    odd = 0
    if length_fi > 1:
        if length_fi % 2 == 0:
            cond1 = True
            half = length_fi // 2
        else:
            half = length_fi // 2
            cond1 = abs(fi[half]) < 100 * np.finfo(float).eps
            odd = 1
        if cond1:
            if length_fi % 2 == 0:
                cond2 = np.allclose(fi[:half], -fi[half:][::-1])
            else:
                cond2 = np.allclose(fi[:half], -fi[half+1:][::-1])
            if cond2:
                symmetric_grid = True
                n_half = half + odd

    length_fiG_ALFs = n_half if (symmetric_grid and method == 3) else length_fi
    fiG_ALFs = fiG[:length_fiG_ALFs] if (symmetric_grid and method == 3) else fiG
    r_ALFs = r[:length_fiG_ALFs] if (symmetric_grid and method == 3) else r

    t_all = np.sin(fiG)
    u_all = np.cos(fiG)
    tan_fiG = np.tan(fiG)[:, np.newaxis]
    cos_fiG = np.cos(fiG)[:, np.newaxis]
    q_all = R / r
    t = np.sin(fiG_ALFs)
    u = np.cos(fiG_ALFs)
    q = R / r_ALFs
    q2 = q ** 2

    # Ellipsoid coefficients
    need_por = any(fc in DISTURBING_FUNCTIONALS for fc in functional_codes)
    need_grav = any(fc in GRAVITATIONAL_FUNCTIONALS for fc in functional_codes)
    need_normal = any(fc in NEED_NORMAL for fc in functional_codes)
    need_dALFs = any(fc in NEED_DALFS for fc in functional_codes)
    need_ddALFs = any(fc in NEED_DDALFS for fc in functional_codes)

    CEl = compute_ellipsoid_coefficients(ellipsoid, R, GM) if (need_por or need_normal) else None

    # Precompute for method 3
    precomp = None
    if method == 3:
        precomp = _precompute_extended(nmax, fiG_ALFs, q)

    # Pole mask for singularity treatment
    sing_mask = np.abs(np.abs(fiG_ALFs) - np.pi / 2) < 1e-15

    # Pre-compute longitude matrices for method 1/3
    if method in (1, 3):
        m_arr = np.arange(0, nmax + 1, dtype=float)
        cosla = np.cos(np.outer(m_arr, lam))   # (nmax+1) x length_lam
        sinla = np.sin(np.outer(m_arr, lam))

    # Prepare deltaC
    C_all = ggm.C.copy()
    S_all = ggm.S.copy()
    if need_por and CEl is not None:
        deltaC_all = C_all.copy()
        for n in range(0, 11):
            deg = 2 * n
            idx_list = np.where((ggm.n == deg) & (ggm.m == 0))[0]
            if len(idx_list) > 0:
                deltaC_all[idx_list[0]] -= CEl[deg]

    # Prepare DTM coefficients
    HC_all = dtm.HC if dtm is not None else None
    HS_all = dtm.HS if dtm is not None else None

    # DTM amplification factor (de-modification for H)
    need_dtm = any(fc in NEED_DTM for fc in functional_codes)

    # Initialize accumulator matrices for method 1/3
    # Each functional has its own A/B matrices
    accum = {}
    for fc in functional_codes:
        if fc in (6, 12):   # Trr/Vrr, Tff/Vff, Tll/Vll
            accum[fc] = {f'A{k}': np.zeros((length_fi, nmax + 1))
                         for k in ('rr', 'ff', 'll')}
            accum[fc].update({f'B{k}': np.zeros((length_fi, nmax + 1))
                              for k in ('rr', 'ff', 'll')})
        elif fc in (7, 13):  # Trf/Vrf, Trl/Vrl, Tfl/Vfl
            accum[fc] = {f'A{k}': np.zeros((length_fi, nmax + 1))
                         for k in ('rf', 'rl', 'fl')}
            accum[fc].update({f'B{k}': np.zeros((length_fi, nmax + 1))
                              for k in ('rf', 'rl', 'fl')})
        elif fc in (8, 14):  # LNOF diagonal
            for k in ('zz', 'xx1', 'xx2', 'xx3', 'yy1'):
                accum.setdefault(fc, {})[f'A{k}'] = np.zeros((length_fi, nmax + 1))
                accum[fc][f'B{k}'] = np.zeros((length_fi, nmax + 1))
        elif fc in (9, 15):  # LNOF off-diagonal
            for k in ('ne', 'nr', 'er'):
                accum.setdefault(fc, {})[f'A{k}'] = np.zeros((length_fi, nmax + 1))
                accum[fc][f'B{k}'] = np.zeros((length_fi, nmax + 1))
        elif fc == 4:  # Theta needs eta + xi
            for k in ('eta', 'ksi'):
                accum.setdefault(fc, {})[f'A{k}'] = np.zeros((length_fi, nmax + 1))
                accum[fc][f'B{k}'] = np.zeros((length_fi, nmax + 1))
        elif fc == 10:  # Geoid
            for k in ('N1c', 'H'):
                accum.setdefault(fc, {})[f'A{k}'] = np.zeros((length_fi, nmax + 1))
                accum[fc][f'B{k}'] = np.zeros((length_fi, nmax + 1))
        elif fc == 16:  # Gravity vector
            for k in ('r', 'lam', 'fi'):
                accum.setdefault(fc, {})[f'A{k}'] = np.zeros((length_fi, nmax + 1))
                accum[fc][f'B{k}'] = np.zeros((length_fi, nmax + 1))
        elif fc == 20:  # Gravity disturbance
            for k in ('r', 'lam', 'fi', 'Ur', 'Ufi'):
                accum.setdefault(fc, {})[f'A{k}'] = np.zeros((length_fi, nmax + 1))
                accum[fc][f'B{k}'] = np.zeros((length_fi, nmax + 1))
        elif fc == 23:  # Height anomaly
            for k in ('N1c', 'dg', 'H', 'zeta'):
                accum.setdefault(fc, {})[f'A{k}'] = np.zeros((length_fi, nmax + 1))
                accum[fc][f'B{k}'] = np.zeros((length_fi, nmax + 1))
        else:
            accum[fc] = {'A': np.zeros((length_fi, nmax + 1)),
                         'B': np.zeros((length_fi, nmax + 1))}

    # Degree amplification arrays (precompute once)
    n_arr = np.arange(nmax + 1, dtype=float)
    ampl_np1 = n_arr + 1                    # (n+1)
    ampl_nm1 = n_arr - 1                    # (n-1)
    ampl_np1_np2 = (n_arr + 1) * (n_arr + 2)  # (n+1)(n+2)

    # =================================================================
    #  Main loop over order m (descending)
    # =================================================================
    index = ggm.index

    for m in range(nmax, -1, -1):
        # Effective minimum degree for this order
        n_start = max(m, nmin)
        m_min = max(0, n_start - m)  # index offset in Pnm matrix

        # Compute fnALFs
        Pnm = compute_fnalfs(fiG_ALFs, q, nmax, m, method, precomp)

        # Compute derivatives if needed
        dPnm = None
        ddPnm = None
        if need_dALFs:
            dPnm = fnalfs_first_derivative(Pnm, fiG_ALFs, q, nmax, m)
        if need_ddALFs:
            if dPnm is None:
                dPnm = fnalfs_first_derivative(Pnm, fiG_ALFs, q, nmax, m)
            ddPnm = fnalfs_second_derivative(Pnm, dPnm, fiG_ALFs, nmax, m)

        # Apply symmetric grid expansion for method 3
        if symmetric_grid and method == 3:
            Pnm_full = _expand_symmetric(Pnm, nmax, m, length_fi, n_half, odd)
            if dPnm is not None:
                dPnm_full = _expand_symmetric_deriv(dPnm, nmax, m, length_fi, n_half, odd)
            else:
                dPnm_full = None
            if ddPnm is not None:
                ddPnm_full = _expand_symmetric_deriv(ddPnm, nmax, m, length_fi, n_half, odd)
            else:
                ddPnm_full = None
        else:
            Pnm_full = Pnm
            dPnm_full = dPnm
            ddPnm_full = ddPnm

        # Extract coefficients for this order, aligned to degrees n_start..nmax
        Cm, Sm = _extract_coeffs_aligned(C_all, S_all, index, m, n_start, nmin_GGM, nmax)
        if need_por:
            deltaCm, _ = _extract_coeffs_aligned(deltaC_all, S_all, index, m,
                                                 n_start, nmin_GGM, nmax)
            # S is same for both
            deltaSm = Sm.copy()
        if need_normal and m == 0:
            CElm = CEl.copy()  # only zonal terms are non-zero
            # Trim to match
            CElm_trimmed = np.zeros_like(Cm)
            for j, n in enumerate(range(n_start, nmax + 1)):
                if n <= 20 and n % 2 == 0:
                    CElm_trimmed[j] = CEl[n]
        else:
            CElm_trimmed = None

        # Pnm columns to use (from m_min)
        P_use = Pnm_full[:, m_min:]
        dP_use = dPnm_full[:, m_min:] if dPnm_full is not None else None
        ddP_use = ddPnm_full[:, m_min:] if ddPnm_full is not None else None

        n_temp = np.arange(n_start, nmax + 1, dtype=float)

        # Accumulate for each functional
        for fc in functional_codes:
            use_delta = fc in DISTURBING_FUNCTIONALS
            C_use = deltaCm if use_delta else Cm
            S_use = Sm

            if method in (1, 3):
                # Cross-order tensor terms also belong to the disturbing field.
                C_cross = deltaC_all if use_delta else C_all
                _accumulate_grid(accum, fc, m, nmax, n_temp, P_use, dP_use, ddP_use,
                                 C_use, S_use, Cm, Sm, C_cross, S_all, index,
                                 nmin_GGM, n_start, ampl_np1, ampl_nm1, ampl_np1_np2,
                                 q_all, R, r, CElm_trimmed,
                                 HC_all, HS_all, dtm, need_dtm,
                                 tan_fiG, cos_fiG)

    # =================================================================
    #  Longitude synthesis + final scaling
    # =================================================================
    # Compute normal gravity if needed
    gammaP = None
    if any(fc in NEED_GAMMA for fc in functional_codes):
        gammaP = normal_gravity(fi, np.full(length_fi, h_val), ellipsoid)

    results = {}
    for fc in functional_codes:
        results[fc] = _finalize_grid(accum[fc], fc, cosla, sinla,
                                      GM, R, r, fi, fiG, gammaP,
                                      length_fi, length_lam,
                                      dtm, HC_all, HS_all, index,
                                      nmin_GGM, nmax, ellipsoid, h_val)

    return results, lam_deg, fi_deg


def _expand_symmetric(Pnm_half, nmax, m, n_full, n_half, odd):
    """Expand fnALFs from northern half to full grid using parity."""
    np_ = Pnm_half.shape[1]
    Pnm = np.zeros((n_full, np_))
    Pnm[:n_half, :] = Pnm_half
    src = Pnm_half[(n_half - odd - 1)::-1, :]
    n_south = n_full - n_half
    Pnm[n_half:n_half + n_south, :] = src
    if m == nmax:
        pass  # n-m=0 always even
    elif m == nmax - 1:
        Pnm[n_half:, 1] = -Pnm[n_half:, 1]
    else:
        Pnm[n_half:, 1::2] = -Pnm[n_half:, 1::2]
    return Pnm


def _expand_symmetric_deriv(dPnm_half, nmax, m, n_full, n_half, odd):
    """Expand derivatives using same parity as fnALFs (derivative parity is same
    when expressed in terms of sin/cos of geocentric latitude)."""
    np_ = dPnm_half.shape[1]
    dPnm = np.zeros((n_full, np_))
    dPnm[:n_half, :] = dPnm_half
    src = dPnm_half[(n_half - odd - 1)::-1, :]
    n_south = n_full - n_half
    dPnm[n_half:n_half + n_south, :] = src
    # Derivatives have opposite parity: dP(-x)/dx has sign flip for even (n-m)
    # Actually: dP_nm/dphi(-phi) = (-1)^(n-m+1) * dP_nm/dphi(phi)
    if m == nmax:
        dPnm[n_half:, 0] = -dPnm[n_half:, 0]
    elif m == nmax - 1:
        dPnm[n_half:, 0] = -dPnm[n_half:, 0]
        # column 1: n-m=1, parity = (-1)^(1+1) = +1, so no flip
    else:
        # For column j: n-m = j, parity = (-1)^(j+1)
        # Even j: flip sign; odd j: no flip
        dPnm[n_half:, 0::2] = -dPnm[n_half:, 0::2]
    return dPnm


def _accumulate_grid(accum, fc, m, nmax, n_temp, P_use, dP_use, ddP_use,
                     C_use, S_use, Cm, Sm, C_all, S_all, index,
                     nmin_GGM, n_start, ampl_np1, ampl_nm1, ampl_np1_np2,
                     q_all, R, r, CElm_trimmed,
                     HC_all, HS_all, dtm, need_dtm,
                     tan_fiG, cos_fiG):
    """Accumulate Fourier coefficients A, B for one order m."""
    n_slice = slice(n_start, nmax + 1)
    n_pts = P_use.shape[0]

    if fc in (2,):  # eta
        Lm = m * P_use
        accum[fc]['A'][:, m] = Lm @ C_use
        accum[fc]['B'][:, m] = Lm @ S_use

    elif fc in (3,):  # xi
        Lm = dP_use
        accum[fc]['A'][:, m] = Lm @ C_use
        accum[fc]['B'][:, m] = Lm @ S_use

    elif fc == 4:  # Theta (eta + xi)
        Lm_eta = m * P_use
        accum[fc]['Aeta'][:, m] = Lm_eta @ C_use
        accum[fc]['Beta'][:, m] = Lm_eta @ S_use
        Lm_ksi = dP_use
        accum[fc]['Aksi'][:, m] = Lm_ksi @ C_use
        accum[fc]['Bksi'][:, m] = Lm_ksi @ S_use

    elif fc == 5:  # T
        Lm = P_use
        accum[fc]['A'][:, m] = Lm @ C_use
        accum[fc]['B'][:, m] = Lm @ S_use

    elif fc in (6, 12):  # Trr/Vrr, Tff/Vff, Tll/Vll
        Lm_rr = P_use * ampl_np1_np2[n_slice]
        Lm_ff = ddP_use
        Lm_ll = m**2 * P_use
        accum[fc]['Arr'][:, m] = Lm_rr @ C_use
        accum[fc]['Brr'][:, m] = Lm_rr @ S_use
        accum[fc]['Aff'][:, m] = Lm_ff @ C_use
        accum[fc]['Bff'][:, m] = Lm_ff @ S_use
        accum[fc]['All'][:, m] = Lm_ll @ C_use
        accum[fc]['Bll'][:, m] = Lm_ll @ S_use

    elif fc in (7, 13):  # Trf/Vrf, Trl/Vrl, Tfl/Vfl
        Lm_rf = dP_use * ampl_np1[n_slice]
        Lm_rl = m * P_use * ampl_np1[n_slice]
        Lm_fl = m * dP_use
        accum[fc]['Arf'][:, m] = Lm_rf @ C_use
        accum[fc]['Brf'][:, m] = Lm_rf @ S_use
        accum[fc]['Arl'][:, m] = Lm_rl @ C_use
        accum[fc]['Brl'][:, m] = Lm_rl @ S_use
        accum[fc]['Afl'][:, m] = Lm_fl @ C_use
        accum[fc]['Bfl'][:, m] = Lm_fl @ S_use

    elif fc in (8, 14):  # LNOF diagonal Txx/Tyy/Tzz or Vxx/Vyy/Vzz
        # Tzz/Vzz
        Lm_zz = P_use * ampl_np1_np2[n_slice]
        accum[fc]['Azz'][:, m] = Lm_zz @ C_use
        accum[fc]['Bzz'][:, m] = Lm_zz @ S_use

        # Txx/Tyy coefficients
        n_t = n_temp
        if m < 2:
            bnm = (n_t + m + 1) * (n_t + m + 2) / 2.0 / (m + 1)
            if m == 0:
                anm = (np.sqrt(2) / 4.0 *
                       np.sqrt(np.maximum(n_t**2 - (m+1)**2, 0)) *
                       np.sqrt(n_t + m + 2) *
                       np.sqrt(np.maximum(n_t - m, 0)))
            else:
                anm = (1.0 / 4.0 *
                       np.sqrt(np.maximum(n_t**2 - (m+1)**2, 0)) *
                       np.sqrt(n_t + m + 2) *
                       np.sqrt(np.maximum(n_t - m, 0)))
            cnm = np.zeros_like(n_t)
        else:
            bnm = (n_t**2 + m**2 + 3*n_t + 2) / 2.0
            anm = (1.0 / 4.0 *
                   np.sqrt(np.maximum(n_t**2 - (m+1)**2, 0)) *
                   np.sqrt(n_t + m + 2) *
                   np.sqrt(np.maximum(n_t - m, 0)))
            if m == 2:
                cnm = (np.sqrt(2) / 4.0 *
                       np.sqrt(np.maximum(n_t**2 - (m-1)**2, 0)) *
                       np.sqrt(np.maximum(n_t - m + 2, 0)) *
                       np.sqrt(n_t + m))
            else:
                cnm = (1.0 / 4.0 *
                       np.sqrt(np.maximum(n_t**2 - (m-1)**2, 0)) *
                       np.sqrt(np.maximum(n_t - m + 2, 0)) *
                       np.sqrt(n_t + m))

        Lm_xx1 = P_use * (bnm - ampl_np1_np2[n_slice])
        Lm_yy1 = P_use * bnm
        Lm_xx2 = P_use * anm
        Lm_xx3 = P_use * cnm

        accum[fc]['Axx1'][:, m] = Lm_xx1 @ C_use
        accum[fc]['Bxx1'][:, m] = Lm_xx1 @ S_use
        accum[fc]['Ayy1'][:, m] = Lm_yy1 @ C_use
        accum[fc]['Byy1'][:, m] = Lm_yy1 @ S_use

        # xx2 uses C_{n,m+2}, S_{n,m+2}
        if m + 2 <= nmax:
            C_m2, S_m2 = _extract_coeffs_aligned(C_all, S_all, index, m + 2,
                                                 n_start, nmin_GGM, nmax)
            accum[fc]['Axx2'][:, m] = Lm_xx2 @ C_m2
            accum[fc]['Bxx2'][:, m] = Lm_xx2 @ S_m2
        else:
            accum[fc]['Axx2'][:, m] = 0.0
            accum[fc]['Bxx2'][:, m] = 0.0

        # xx3 uses C_{n,m-2}, S_{n,m-2}
        if m >= 2:
            C_mm2, S_mm2 = _extract_coeffs_aligned(C_all, S_all, index, m - 2,
                                                   n_start, nmin_GGM, nmax)
            accum[fc]['Axx3'][:, m] = Lm_xx3 @ C_mm2
            accum[fc]['Bxx3'][:, m] = Lm_xx3 @ S_mm2
        else:
            accum[fc]['Axx3'][:, m] = 0.0
            accum[fc]['Bxx3'][:, m] = 0.0

    elif fc == 10:  # Geoid
        Lm = P_use
        accum[fc]['AN1c'][:, m] = Lm @ C_use
        accum[fc]['BN1c'][:, m] = Lm @ S_use
        if need_dtm and HC_all is not None:
            HCm, HSm = _extract_coeffs(HC_all, HS_all, index, m, nmin_GGM, nmax)
            # De-modify: amplH = 1 / (R/r)^n = (r/R)^n
            amplH = np.zeros((P_use.shape[0], len(n_temp)))
            for j, n in enumerate(n_temp.astype(int)):
                amplH[:, j] = (r / R)**n
            LmH = P_use * amplH
            accum[fc]['AH'][:, m] = LmH @ HCm[:len(n_temp)]
            accum[fc]['BH'][:, m] = LmH @ HSm[:len(n_temp)]

    elif fc == 11:  # V
        Lm = P_use
        accum[fc]['A'][:, m] = Lm @ C_use
        accum[fc]['B'][:, m] = Lm @ S_use

    elif fc == 16:  # Gravity vector
        Lm_r = P_use * ampl_np1[n_slice]
        Lm_lam = m * P_use
        Lm_fi = dP_use
        accum[fc]['Ar'][:, m] = Lm_r @ C_use
        accum[fc]['Br'][:, m] = Lm_r @ S_use
        accum[fc]['Alam'][:, m] = Lm_lam @ C_use
        accum[fc]['Blam'][:, m] = Lm_lam @ S_use
        accum[fc]['Afi'][:, m] = Lm_fi @ C_use
        accum[fc]['Bfi'][:, m] = Lm_fi @ S_use

    elif fc == 17:  # g_sa
        Lm = P_use * ampl_np1[n_slice]
        accum[fc]['A'][:, m] = Lm @ C_use
        accum[fc]['B'][:, m] = Lm @ S_use

    elif fc == 18:  # W
        Lm = P_use
        accum[fc]['A'][:, m] = Lm @ C_use
        accum[fc]['B'][:, m] = Lm @ S_use

    elif fc == 19:  # gravity anomaly sa
        Lm = P_use * ampl_nm1[n_slice]
        accum[fc]['A'][:, m] = Lm @ C_use
        accum[fc]['B'][:, m] = Lm @ S_use

    elif fc == 20:  # Gravity disturbance
        Lm_r = P_use * ampl_np1[n_slice]
        Lm_lam = m * P_use
        Lm_fi = dP_use
        accum[fc]['Ar'][:, m] = Lm_r @ C_use
        accum[fc]['Br'][:, m] = Lm_r @ S_use
        accum[fc]['Alam'][:, m] = Lm_lam @ C_use
        accum[fc]['Blam'][:, m] = Lm_lam @ S_use
        accum[fc]['Afi'][:, m] = Lm_fi @ C_use
        accum[fc]['Bfi'][:, m] = Lm_fi @ S_use
        if m == 0 and CElm_trimmed is not None:
            accum[fc]['AUr'][:, m] = Lm_r @ CElm_trimmed
            accum[fc]['BUr'][:, m] = 0.0
            accum[fc]['AUfi'][:, m] = Lm_fi @ CElm_trimmed
            accum[fc]['BUfi'][:, m] = 0.0

    elif fc == 21:  # gravity disturbance sa
        Lm = P_use * ampl_np1[n_slice]
        accum[fc]['A'][:, m] = Lm @ C_use
        accum[fc]['B'][:, m] = Lm @ S_use

    elif fc == 22:  # height anomaly ell
        Lm = P_use
        accum[fc]['A'][:, m] = Lm @ C_use
        accum[fc]['B'][:, m] = Lm @ S_use

    elif fc == 23:  # height anomaly
        Lm = P_use
        accum[fc]['AN1c'][:, m] = Lm @ C_use
        accum[fc]['BN1c'][:, m] = Lm @ S_use
        accum[fc]['Azeta'][:, m] = Lm @ C_use
        accum[fc]['Bzeta'][:, m] = Lm @ S_use

        Lm_dg = P_use * ampl_np1[n_slice]
        accum[fc]['Adg'][:, m] = Lm_dg @ C_use
        accum[fc]['Bdg'][:, m] = Lm_dg @ S_use

        if need_dtm and HC_all is not None:
            HCm, HSm = _extract_coeffs(HC_all, HS_all, index, m, nmin_GGM, nmax)
            amplH = np.zeros((P_use.shape[0], len(n_temp)))
            for j, n in enumerate(n_temp.astype(int)):
                amplH[:, j] = (r / R)**n
            LmH = P_use * amplH
            accum[fc]['AH'][:, m] = LmH @ HCm[:len(n_temp)]
            accum[fc]['BH'][:, m] = LmH @ HSm[:len(n_temp)]

    elif fc == 24:  # T_rr
        Lm = P_use * ampl_np1_np2[n_slice]
        accum[fc]['A'][:, m] = Lm @ C_use
        accum[fc]['B'][:, m] = Lm @ S_use

    elif fc == 25:  # W_rr
        Lm = P_use * ampl_np1_np2[n_slice]
        accum[fc]['A'][:, m] = Lm @ C_use
        accum[fc]['B'][:, m] = Lm @ S_use

    # LNOF off-diagonal (9, 15) — direct same-order accumulation
    elif fc in (9, 15):
        Lm_ne = m * (dP_use + tan_fiG * P_use) / cos_fiG
        accum[fc]['Ane'][:, m] = Lm_ne @ S_use
        accum[fc]['Bne'][:, m] = Lm_ne @ (-C_use)

        # The zonal (m = 0) part of H_nr does not vanish.
        Lm_nr = -(n_temp + 2.0) * dP_use
        accum[fc]['Anr'][:, m] = Lm_nr @ C_use
        accum[fc]['Bnr'][:, m] = Lm_nr @ S_use

        Lm_er = -m * (n_temp + 2.0) * P_use / cos_fiG
        accum[fc]['Aer'][:, m] = Lm_er @ S_use
        accum[fc]['Ber'][:, m] = Lm_er @ (-C_use)


def _finalize_grid(accum, fc, cosla, sinla, GM, R, r, fi, fiG, gammaP,
                   length_fi, length_lam, dtm, HC_all, HS_all, index,
                   nmin_GGM, nmax, ellipsoid, h_val):
    """Final longitude synthesis and physical scaling for one functional."""
    omega = ellipsoid['omega']

    if fc == 2:  # eta
        eta = -accum['A'] @ sinla + accum['B'] @ cosla
        Pg = -GM / (r[:, None]**2 * gammaP[:, None] * np.cos(fiG)[:, None]) * eta * ARCSEC
        pole = np.abs(np.abs(fi) - np.pi / 2) < 1e-15
        Pg[pole, :] = 0.0
        return Pg.ravel()

    elif fc == 3:  # xi
        ksi = accum['A'] @ cosla + accum['B'] @ sinla
        Pg = -GM / (r[:, None]**2 * gammaP[:, None]) * ksi * ARCSEC
        return Pg.ravel()

    elif fc == 4:  # Theta
        eta = -accum['Aeta'] @ sinla + accum['Beta'] @ cosla
        ksi = accum['Aksi'] @ cosla + accum['Bksi'] @ sinla
        eta_s = -GM / (r[:, None]**2 * gammaP[:, None] * np.cos(fiG)[:, None]) * eta * ARCSEC
        pole = np.abs(np.abs(fi) - np.pi / 2) < 1e-15
        eta_s[pole, :] = 0.0
        ksi_s = -GM / (r[:, None]**2 * gammaP[:, None]) * ksi * ARCSEC
        eta_v = eta_s.ravel()
        ksi_v = ksi_s.ravel()
        alpha = np.arctan2(eta_v, ksi_v)
        alpha[alpha < 0] += 2 * np.pi
        Pg = np.column_stack([np.sqrt(eta_v**2 + ksi_v**2), np.rad2deg(alpha)])
        return Pg

    elif fc == 5:  # T
        T = accum['A'] @ cosla + accum['B'] @ sinla
        return (GM / r[:, None] * T).ravel()

    elif fc in (6, 12):  # Trr/Vrr, Tff/Vff, Tll/Vll
        Trr = accum['Arr'] @ cosla + accum['Brr'] @ sinla
        Tff = accum['Aff'] @ cosla + accum['Bff'] @ sinla
        # Pole treatment for Tll
        pole_mask = np.abs(fi) > np.deg2rad(89.9)
        All = accum['All'].copy()
        Bll = accum['Bll'].copy()
        All[pole_mask, :] = 0.0
        Bll[pole_mask, :] = 0.0
        Tll = All @ cosla + Bll @ sinla
        Pg_rr = (GM / r[:, None]**3 * Trr * EOTVOS).ravel()
        Pg_ff = (GM / r[:, None]**3 * Tff * EOTVOS).ravel()
        Pg_ll = (GM / (r[:, None]**3 * np.cos(fiG)[:, None]**2) * Tll * EOTVOS).ravel()
        return np.column_stack([Pg_rr, Pg_ff, -Pg_ll])

    elif fc in (7, 13):  # Trf/Vrf, Trl/Vrl, Tfl/Vfl
        Trf = accum['Arf'] @ cosla + accum['Brf'] @ sinla
        Trl = -accum['Arl'] @ sinla + accum['Brl'] @ cosla
        # Pole treatment for Tfl
        pole_mask = np.abs(fi) > np.deg2rad(89.5)
        Afl = accum['Afl'].copy()
        Bfl = accum['Bfl'].copy()
        Afl[pole_mask, :] = 0.0
        Bfl[pole_mask, :] = 0.0
        Tfl = -Afl @ sinla + Bfl @ cosla
        Pg_rf = (GM / r[:, None]**3 * Trf * EOTVOS).ravel()
        Pg_rl = (GM / (r[:, None]**3 * np.cos(fiG)[:, None]) * Trl * EOTVOS).ravel()
        Pg_fl = (GM / (r[:, None]**3 * np.cos(fiG)[:, None]) * Tfl * EOTVOS).ravel()
        return np.column_stack([-Pg_rf, -Pg_rl, Pg_fl])

    elif fc in (8, 14):  # LNOF diagonal
        Tzz = accum['Azz'] @ cosla + accum['Bzz'] @ sinla
        Txx1 = accum['Axx1'] @ cosla + accum['Bxx1'] @ sinla
        Txx2 = (accum['Axx2'] @ np.vstack([cosla[2:, :], np.zeros((2, length_lam))]) +
                accum['Bxx2'] @ np.vstack([sinla[2:, :], np.zeros((2, length_lam))]))
        Tyy1 = accum['Ayy1'] @ cosla + accum['Byy1'] @ sinla
        Txx3 = (accum['Axx3'] @ np.vstack([np.zeros((2, length_lam)), cosla[:-2, :]]) +
                accum['Bxx3'] @ np.vstack([np.zeros((2, length_lam)), sinla[:-2, :]]))
        Txx = Txx1 + Txx2 + Txx3
        Tyy = Tyy1 + Txx2 + Txx3
        Pg_xx = (GM / r[:, None]**3 * Txx * EOTVOS).ravel()
        Pg_yy = (GM / r[:, None]**3 * Tyy * EOTVOS).ravel()
        Pg_zz = (GM / r[:, None]**3 * Tzz * EOTVOS).ravel()
        return np.column_stack([Pg_xx, -Pg_yy, Pg_zz])

    elif fc in (9, 15):  # LNOF off-diagonal
        Txy = accum['Ane'] @ cosla + accum['Bne'] @ sinla
        Txz = accum['Anr'] @ cosla + accum['Bnr'] @ sinla
        Tyz = accum['Aer'] @ cosla + accum['Ber'] @ sinla
        Pg_xy = (GM / r[:, None]**3 * Txy * EOTVOS).ravel()
        Pg_xz = (GM / r[:, None]**3 * Txz * EOTVOS).ravel()
        Pg_yz = (GM / r[:, None]**3 * Tyz * EOTVOS).ravel()
        # ICGEM sign convention (GrafLab): xy and yz off-diagonals are
        # the negatives of the raw cross derivatives; xz keeps its sign.
        return np.column_stack([-Pg_xy, Pg_xz, -Pg_yz])

    elif fc == 10:  # Geoid
        N1c = accum['AN1c'] @ cosla + accum['BN1c'] @ sinla
        N1c = GM / (r[:, None] * gammaP[:, None]) * N1c
        if dtm is not None and 'AH' in accum:
            H = accum['AH'] @ cosla + accum['BH'] @ sinla
            H[H < 0] = 0.0
            Pg = N1c - (2 * np.pi * G_GRAV * RHO_CRUST * H**2) / gammaP[:, None]
        else:
            Pg = N1c
        return Pg.ravel()

    elif fc == 11:  # V
        V = accum['A'] @ cosla + accum['B'] @ sinla
        return (GM / r[:, None] * V).ravel()

    elif fc == 16:  # Gravity vector
        Wr = accum['Ar'] @ cosla + accum['Br'] @ sinla
        Wlam = -accum['Alam'] @ sinla + accum['Blam'] @ cosla
        Wfi = accum['Afi'] @ cosla + accum['Bfi'] @ sinla
        Wr = GM / r[:, None]**2 * Wr
        Wr = -Wr + omega**2 * r[:, None] * np.cos(fiG)[:, None]**2
        Wlam = GM / r[:, None] * Wlam
        Wlam = -Wlam / (r[:, None] * np.cos(fiG)[:, None])
        Wfi = GM / r[:, None] * Wfi
        Wfi = (Wfi + (-omega**2 * r[:, None]**2 * np.cos(fiG)[:, None] * np.sin(fiG)[:, None])) / r[:, None]
        return np.column_stack([(Wfi * MGAL).ravel(),
                                (Wlam * MGAL).ravel(),
                                (Wr * MGAL).ravel()])

    elif fc == 17:  # g_sa
        g_sa = accum['A'] @ cosla + accum['B'] @ sinla
        g_sa = GM / r[:, None]**2 * g_sa
        return (np.sqrt((-g_sa + omega**2 * r[:, None] * np.cos(fiG)[:, None]**2)**2) * MGAL).ravel()

    elif fc == 18:  # W
        W = accum['A'] @ cosla + accum['B'] @ sinla
        W = GM / r[:, None] * W
        return (W + 0.5 * omega**2 * r[:, None]**2 * np.cos(fiG)[:, None]**2).ravel()

    elif fc == 19:  # gravity anomaly sa
        sa = accum['A'] @ cosla + accum['B'] @ sinla
        return (GM / r[:, None]**2 * sa * MGAL).ravel()

    elif fc == 20:  # Gravity disturbance
        Wr = accum['Ar'] @ cosla + accum['Br'] @ sinla
        Wlam = -accum['Alam'] @ sinla + accum['Blam'] @ cosla
        Wfi = accum['Afi'] @ cosla + accum['Bfi'] @ sinla
        Wr = GM / r[:, None]**2 * Wr
        Wr_mag = np.sqrt((-Wr + omega**2 * r[:, None] * np.cos(fiG)[:, None]**2)**2)
        Wlam_s = GM / r[:, None] * Wlam / (r[:, None] * np.cos(fiG)[:, None])
        Wfi_s = GM / r[:, None] * Wfi
        Wfi_s = (Wfi_s - omega**2 * r[:, None]**2 * np.cos(fiG)[:, None] * np.sin(fiG)[:, None]) / r[:, None]
        grad_W = np.sqrt(Wr_mag**2 + Wlam_s**2 + Wfi_s**2)

        Ur = accum['AUr'] @ cosla
        Ufi = accum['AUfi'] @ cosla
        Ur = GM / r[:, None]**2 * Ur
        Ur_mag = np.sqrt((-Ur + omega**2 * r[:, None] * np.cos(fiG)[:, None]**2)**2)
        Ufi_s = GM / r[:, None] * Ufi
        Ufi_s = (Ufi_s - omega**2 * r[:, None]**2 * np.cos(fiG)[:, None] * np.sin(fiG)[:, None]) / r[:, None]
        grad_U = np.sqrt(Ur_mag**2 + Ufi_s**2)

        return ((grad_W - grad_U) * MGAL).ravel()

    elif fc == 21:  # gravity disturbance sa
        sa = accum['A'] @ cosla + accum['B'] @ sinla
        return (GM / r[:, None]**2 * sa * MGAL).ravel()

    elif fc == 22:  # height anomaly ell
        z = accum['A'] @ cosla + accum['B'] @ sinla
        return (GM / (r[:, None] * gammaP[:, None]) * z).ravel()

    elif fc == 23:  # height anomaly
        N1c = accum['AN1c'] @ cosla + accum['BN1c'] @ sinla
        N1c = GM / (r[:, None] * gammaP[:, None]) * N1c
        zeta_el = accum['Azeta'] @ cosla + accum['Bzeta'] @ sinla
        zeta_el = GM / (r[:, None] * gammaP[:, None]) * zeta_el
        dg = accum['Adg'] @ cosla + accum['Bdg'] @ sinla
        dg = GM / r[:, None]**2 * dg

        if dtm is not None and 'AH' in accum:
            H = accum['AH'] @ cosla + accum['BH'] @ sinla
            H[H < 0] = 0.0
        else:
            H = np.zeros((length_fi, length_lam))

        N = N1c - (2 * np.pi * G_GRAV * RHO_CRUST * H**2) / gammaP[:, None]
        Pg = zeta_el - dg * (H + N) / gammaP[:, None]
        return Pg.ravel()

    elif fc == 24:  # T_rr
        T_rr = accum['A'] @ cosla + accum['B'] @ sinla
        return (GM / r[:, None]**3 * T_rr * EOTVOS).ravel()

    elif fc == 25:  # W_rr
        Wrr = accum['A'] @ cosla + accum['B'] @ sinla
        Wrr = GM / r[:, None]**3 * Wrr
        return ((Wrr + omega**2 * np.cos(fiG)[:, None]**2) * EOTVOS).ravel()

    else:
        raise ValueError(f"Unknown functional code: {fc}")


# ===================================================================
#  Pointwise synthesis — all 24 functionals
# ===================================================================

def _get_cross_order_coeffs(C_all, S_all, index, order, nmin_GGM, nmax, n_start):
    """Safely extract coefficients at a given order, aligned to n_start..nmax."""
    return _extract_coeffs_aligned(C_all, S_all, index, order, n_start,
                                   nmin_GGM, nmax)


def synthesize_pointwise(ggm, functional_codes, ellipsoid_spec,
                         lon_deg, lat_deg, h_or_r, coord_type=0,
                         method=1, dtm=None, nmin=None):
    """Compute geopotential functionals at scattered points.

    Parameters
    ----------
    ggm             : GGMModel
    functional_codes: list of int (codes 2-25)
    ellipsoid_spec  : int or list
    lon_deg, lat_deg: array-like – coordinates in degrees
    h_or_r          : array-like – ellipsoidal height (coord_type=0) or radius (coord_type=1)
    coord_type      : int – 0=ellipsoidal, 1=spherical
    method          : int – 1, 2, or 3 (fnALF method)
    dtm             : DTMModel or None
    nmin            : int or None

    Returns
    -------
    results : dict  – key=functional code, value=ndarray
    lon, lat : ndarray (degrees)
    """
    ellipsoid = get_ellipsoid(ellipsoid_spec)
    GM = ggm.GM
    R = ggm.R
    nmin_GGM = ggm.nmin
    nmax = ggm.nmax

    lat = np.atleast_1d(np.asarray(lat_deg, dtype=float)).ravel()
    lon = np.atleast_1d(np.asarray(lon_deg, dtype=float)).ravel()
    hr = np.atleast_1d(np.asarray(h_or_r, dtype=float)).ravel()
    n_pts = lat.size

    # ---- input validation ------------------------------------------------
    if not (lon.size == n_pts and hr.size == n_pts):
        raise ValueError(
            "lon_deg, lat_deg and h_or_r must have the same length "
            f"(got {lon.size}, {n_pts}, {hr.size}).")
    if n_pts == 0:
        raise ValueError("At least one computation point is required.")
    if n_pts > MAX_GRID_POINTS:
        raise ValueError(
            f"Too many points: {n_pts:,} (limit {MAX_GRID_POINTS:,}).")
    if not (np.all(np.isfinite(lat)) and np.all(np.isfinite(lon))
            and np.all(np.isfinite(hr))):
        raise ValueError("Coordinates must be finite.")
    if np.any(np.abs(lat) > 90.0):
        raise ValueError("Latitudes must lie within [-90, 90] degrees.")
    if method not in (1, 2, 3):
        raise ValueError("method must be 1 (standard), 2 (modified) or 3 (extended).")
    if coord_type not in (0, 1):
        raise ValueError("coord_type must be 0 (ellipsoidal) or 1 (spherical).")
    if coord_type == 1 and np.any(hr <= 0):
        raise ValueError(
            "With coord_type=1 the height argument is a geocentric radius and must be > 0.")

    functional_codes = [int(code) for code in functional_codes]
    unknown = [code for code in functional_codes if code not in FUNCTIONAL_NAMES]
    if unknown:
        raise ValueError(f"Unknown functional code(s): {unknown}")
    if not functional_codes:
        raise ValueError("At least one functional code is required.")

    if nmin is None:
        nmin = nmin_GGM
    nmin = int(nmin)
    if nmin < 0 or nmin > nmax:
        raise ValueError(f"nmin must satisfy 0 <= nmin <= model nmax ({nmax}).")

    # Helper: sum element-wise product over degree axis (axis=1)
    def _S(P, V):
        """Compute sum_j P[i,j]*V[i,j] for each point i."""
        return np.einsum('ij,ij->i', P, V)


    # Coordinate conversion
    fi = np.deg2rad(lat)
    lam = np.deg2rad(lon)

    if coord_type == 0:  # ellipsoidal
        X, Y, Z = ell2cart(fi, np.zeros(n_pts), hr, ellipsoid)
        _, fiG, r = cart2sph(X, Y, Z)
    else:  # spherical
        fiG = fi.copy()
        r = hr.copy()
        X, Y, Z = sph2cart(np.zeros(n_pts), fiG, r)
        fi_ell, _, _ = cart2ell(X, Y, Z, ellipsoid)
        fi = fi_ell

    q = R / r
    u = np.cos(fiG)
    omega = ellipsoid['omega']

    # Which functionals are requested
    fc_set = set(functional_codes)

    # Disturbing-field tensors (6-9) and full-field tensors (12-15) draw on
    # different coefficient sets, so each keeps its own accumulators and both
    # are evaluated in a single pass over the orders. The gravity vector (16)
    # and the gravity disturbance (20) share their accumulators, so they form
    # one accumulation group.
    need_por = bool(fc_set & DISTURBING_FUNCTIONALS)
    need_grav_field = bool(fc_set & GRAVITATIONAL_FUNCTIONALS)
    need_normal = bool(fc_set & NEED_NORMAL)
    need_dALFs = bool(fc_set & NEED_DALFS)
    need_ddALFs = bool(fc_set & NEED_DDALFS)
    need_dtm = bool(fc_set & NEED_DTM)
    need_lnof_diag = bool(fc_set & {8, 14})

    CEl = compute_ellipsoid_coefficients(ellipsoid, R, GM) if (need_por or need_normal) else None

    C_all = ggm.C.copy()
    S_all = ggm.S.copy()
    deltaC_all = C_all.copy()
    if need_por and CEl is not None:
        for n in range(11):
            deg = 2 * n
            idx = np.where((ggm.n == deg) & (ggm.m == 0))[0]
            if len(idx) > 0:
                deltaC_all[idx[0]] -= CEl[deg]

    HC_all = dtm.HC if dtm is not None else None
    HS_all = dtm.HS if dtm is not None else None

    precomp = None
    if method == 3:
        precomp = _precompute_extended(nmax, fiG, q)

    # Normal gravity
    gammaP = None
    if any(fc in NEED_GAMMA for fc in functional_codes):
        gammaP = normal_gravity(fi, hr, ellipsoid)

    # Initialize accumulators (multi-component functionals keep their own)
    acc = {fc: np.zeros(n_pts) for fc in functional_codes
           if fc not in (4, 6, 7, 8, 9, 16, 20)}
    if 4 in fc_set:
        acc_eta4, acc_ksi4 = np.zeros(n_pts), np.zeros(n_pts)
    # Disturbing-field tensors
    if 6 in fc_set:
        acc_rr = np.zeros(n_pts)
        acc_ff = np.zeros(n_pts)
        acc_ll = np.zeros(n_pts)
    if 7 in fc_set:
        acc_rf = np.zeros(n_pts)
        acc_rl = np.zeros(n_pts)
        acc_fl = np.zeros(n_pts)
    if 8 in fc_set:
        acc_zz = np.zeros(n_pts)
        acc_xx = np.zeros(n_pts)
        acc_yy = np.zeros(n_pts)
    if 9 in fc_set:
        acc_xy = np.zeros(n_pts)
        acc_xz = np.zeros(n_pts)
        acc_yz = np.zeros(n_pts)
    # Full-field tensors (separate accumulators from the disturbing ones)
    if 12 in fc_set:
        acc_rr_f = np.zeros(n_pts)
        acc_ff_f = np.zeros(n_pts)
        acc_ll_f = np.zeros(n_pts)
    if 13 in fc_set:
        acc_rf_f = np.zeros(n_pts)
        acc_rl_f = np.zeros(n_pts)
        acc_fl_f = np.zeros(n_pts)
    if 14 in fc_set:
        acc_zz_f = np.zeros(n_pts)
        acc_xx_f = np.zeros(n_pts)
        acc_yy_f = np.zeros(n_pts)
    if 15 in fc_set:
        acc_xy_f = np.zeros(n_pts)
        acc_xz_f = np.zeros(n_pts)
        acc_yz_f = np.zeros(n_pts)
    # Gravity vector (16) and gravity disturbance (20) share accumulators
    if fc_set & {16, 20}:
        acc_Wr = np.zeros(n_pts)
        acc_Wlam = np.zeros(n_pts)
        acc_Wfi = np.zeros(n_pts)
    if 20 in fc_set:
        acc_Ur = np.zeros(n_pts)
        acc_Ufi = np.zeros(n_pts)
    if 10 in fc_set:
        acc_N1c = np.zeros(n_pts)
        acc_H = np.zeros(n_pts)
    if 23 in fc_set:
        acc_zN1c = np.zeros(n_pts)
        acc_zdg = np.zeros(n_pts)
        acc_zH = np.zeros(n_pts)

    # One accumulation task per distinct accumulator set: 20 reuses 16's
    acc_groups = sorted({16 if fc == 20 else fc for fc in functional_codes})

    # Latitude factors used by the LNOF off-diagonal tensor (code-independent)
    tan_f = np.tan(fiG)[:, np.newaxis] if fc_set & {9, 15} else None
    cos_row = np.cos(fiG) if fc_set & {9, 15} else None

    # Main loop over order m
    index = ggm.index
    for m in range(nmax, -1, -1):
        Pnm = compute_fnalfs(fiG, q, nmax, m, method, precomp)

        # Method 2 evaluates the recursion with a 1e-280 seed to avoid
        # underflow; restore the physical scale before forming functionals.
        if method == 2:
            Pnm = Pnm * 1e280

        dPnm = None
        ddPnm = None
        if need_dALFs:
            dPnm = fnalfs_first_derivative(Pnm, fiG, q, nmax, m)
        if need_ddALFs:
            if dPnm is None:
                dPnm = fnalfs_first_derivative(Pnm, fiG, q, nmax, m)
            ddPnm = fnalfs_second_derivative(Pnm, dPnm, fiG, nmax, m)

        # nmin truncation
        n_start = max(m, nmin)
        m_min = max(0, n_start - m)
        Cm, Sm = _extract_coeffs_aligned(C_all, S_all, index, m, n_start,
                                         nmin_GGM, nmax)
        if need_por:
            deltaCm, _ = _extract_coeffs_aligned(deltaC_all, S_all, index, m,
                                                 n_start, nmin_GGM, nmax)

        P_use = Pnm[:, m_min:]
        dP_use = dPnm[:, m_min:] if dPnm is not None else None
        ddP_use = ddPnm[:, m_min:] if ddPnm is not None else None

        if method == 2:
            # Method 2 recurses on Pnm / u**m (GrafLab's Horner scheme): the
            # 1e-280 seed keeps the tail of the series representable, and the
            # u**m factor is re-applied here, after the derivatives have been
            # formed from the scaled values.
            um = (u ** m)[:, np.newaxis]
            P_use = P_use * um
            if dP_use is not None:
                dP_use = dP_use * um
            if ddP_use is not None:
                ddP_use = ddP_use * um

        n_arr = np.arange(n_start, nmax + 1, dtype=float)
        ampl_np1 = n_arr + 1
        ampl_nm1 = n_arr - 1
        ampl_np1_np2 = ampl_np1 * (n_arr + 2)
        n_len = len(n_arr)

        cos_m = np.cos(m * lam)
        sin_m = np.sin(m * lam)
        cos_mc = cos_m[:, np.newaxis]
        sin_mc = sin_m[:, np.newaxis]

        # Cross-order coefficients for the LNOF diagonal tensor. Codes 8 and
        # 14 use different coefficient sets, so both are prepared when the two
        # are requested together.
        if 8 in fc_set:
            Cp2_d, Sp2_d = _get_cross_order_coeffs(
                deltaC_all, S_all, index, m + 2, nmin_GGM, nmax, n_start)
            Cm2_d, Sm2_d = _get_cross_order_coeffs(
                deltaC_all, S_all, index, m - 2, nmin_GGM, nmax, n_start)
        if 14 in fc_set:
            Cp2_f, Sp2_f = _get_cross_order_coeffs(
                C_all, S_all, index, m + 2, nmin_GGM, nmax, n_start)
            Cm2_f, Sm2_f = _get_cross_order_coeffs(
                C_all, S_all, index, m - 2, nmin_GGM, nmax, n_start)
        if need_lnof_diag:
            cos_m2 = np.cos((m + 2) * lam)
            cos_m2c = cos_m2[:, np.newaxis]
            sin_m2 = np.sin((m + 2) * lam)
            sin_m2c = sin_m2[:, np.newaxis]
            cos_mm2 = np.cos(max(0, m - 2) * lam) if m >= 2 else np.ones_like(cos_m)
            cos_mm2c = cos_mm2[:, np.newaxis]
            sin_mm2 = np.sin(max(0, m - 2) * lam) if m >= 2 else np.zeros_like(sin_m)
            sin_mm2c = sin_mm2[:, np.newaxis]

        # DTM coefficients for geoid/height anomaly
        if need_dtm and HC_all is not None:
            HC_use, HS_use = _extract_coeffs_aligned(HC_all, HS_all, index, m,
                                                     n_start, nmin_GGM, nmax)
            # De-modification factor (r/R)^n to undo q^n in modified fnALFs
            rn_factor = np.zeros((n_pts, n_len))
            for j in range(n_len):
                rn_factor[:, j] = (r / R) ** n_arr[j]

        # Normal field coefficients (only m=0)
        if need_normal and m == 0:
            CElm = np.zeros(n_len)
            for j in range(n_len):
                deg = int(n_arr[j])
                if deg <= 20 and deg % 2 == 0:
                    CElm[j] = CEl[deg]

        # Shared coefficient-set products: computed once per order and reused
        # by every functional that draws on the same coefficient set.
        if need_por:
            CcSs_d = deltaCm * cos_mc + Sm * sin_mc
            neg_d = deltaCm * (-sin_mc) + Sm * cos_mc
            mneg_d = m * neg_d
        if need_grav_field:
            CcSs_f = Cm * cos_mc + Sm * sin_mc
            neg_f = Cm * (-sin_mc) + Sm * cos_mc
            mneg_f = m * neg_f

        # Accumulate for each accumulation group (one per accumulator set)
        for fc in acc_groups:
            if fc in DISTURBING_FUNCTIONALS:
                Cu_cos_Su_sin = CcSs_d
                m_neg_Cu_sin_Sm_cos = mneg_d
            else:
                Cu_cos_Su_sin = CcSs_f
                m_neg_Cu_sin_Sm_cos = mneg_f

            # ---- 2: eta ----
            if fc == 2:
                acc[fc] += _S(P_use, m_neg_Cu_sin_Sm_cos)

            # ---- 3: xi ----
            elif fc == 3:
                acc[fc] += _S(dP_use, Cu_cos_Su_sin)

            # ---- 4: Theta ----
            elif fc == 4:
                acc_eta4 += _S(P_use, m_neg_Cu_sin_Sm_cos)
                acc_ksi4 += _S(dP_use, Cu_cos_Su_sin)

            # ---- 5: T ----
            elif fc == 5:
                acc[fc] += _S(P_use, Cu_cos_Su_sin)

            # ---- 6: Trr, Tff, Tll ----
            elif fc == 6:
                acc_rr += _S(P_use, Cu_cos_Su_sin * ampl_np1_np2)
                acc_ff += _S(ddP_use, Cu_cos_Su_sin)
                acc_ll += _S(P_use, Cu_cos_Su_sin * m**2)

            # ---- 7: Trf, Trl, Tfl ----
            elif fc == 7:
                acc_rf += _S(dP_use, Cu_cos_Su_sin * ampl_np1)
                acc_rl += _S(P_use, m_neg_Cu_sin_Sm_cos * ampl_np1)
                acc_fl += _S(dP_use, m_neg_Cu_sin_Sm_cos)

            # ---- 8: Txx, Tyy, Tzz (LNOF diagonal) ----
            elif fc == 8:
                acc_zz += _S(P_use, Cu_cos_Su_sin * ampl_np1_np2)

                if m < 2:
                    bnm = (n_arr + m + 1) * (n_arr + m + 2) / 2.0 / (m + 1)
                    if m == 0:
                        anm = (np.sqrt(2) / 4.0 *
                               np.sqrt(np.maximum(n_arr**2 - (m + 1)**2, 0)) *
                               np.sqrt(n_arr + m + 2) *
                               np.sqrt(np.maximum(n_arr - m, 0)))
                    else:
                        anm = (1.0 / 4.0 *
                               np.sqrt(np.maximum(n_arr**2 - (m + 1)**2, 0)) *
                               np.sqrt(n_arr + m + 2) *
                               np.sqrt(np.maximum(n_arr - m, 0)))
                    cnm = np.zeros(n_len)
                else:
                    bnm = (n_arr**2 + m**2 + 3 * n_arr + 2) / 2.0
                    anm = (1.0 / 4.0 *
                           np.sqrt(np.maximum(n_arr**2 - (m + 1)**2, 0)) *
                           np.sqrt(n_arr + m + 2) *
                           np.sqrt(np.maximum(n_arr - m, 0)))
                    if m == 2:
                        cnm = (np.sqrt(2) / 4.0 *
                               np.sqrt(np.maximum(n_arr**2 - (m - 1)**2, 0)) *
                               np.sqrt(np.maximum(n_arr - m + 2, 0)) *
                               np.sqrt(n_arr + m))
                    else:
                        cnm = (1.0 / 4.0 *
                               np.sqrt(np.maximum(n_arr**2 - (m - 1)**2, 0)) *
                               np.sqrt(np.maximum(n_arr - m + 2, 0)) *
                               np.sqrt(n_arr + m))

                Txx2_inner = Cu_cos_Su_sin * (bnm - ampl_np1_np2)
                Tyy2_inner = Cu_cos_Su_sin * bnm

                if m + 2 <= nmax:
                    Txx1_inner = (Cp2_d * cos_m2c + Sp2_d * sin_m2c) * anm
                else:
                    Txx1_inner = np.zeros((n_pts, n_len))
                if m >= 2:
                    Txx3_inner = (Cm2_d * cos_mm2c + Sm2_d * sin_mm2c) * cnm
                else:
                    Txx3_inner = np.zeros((n_pts, n_len))

                acc_xx += _S(P_use, Txx1_inner + Txx2_inner + Txx3_inner)
                acc_yy += _S(P_use, Txx1_inner + Tyy2_inner + Txx3_inner)

            # ---- 9: Txy, Txz, Tyz (disturbing field) ----
            elif fc == 9:
                # Direct Hessian components in the geocentric local frame
                # (north, east, up), Cu = deltaC for 9 / C for 15:
                #   H_ne = (1/(r^2 cos f)) T_fl + (tan f/(r^2 cos f)) T_l
                #   H_nr = (1/r) T_rf - (1/r^2) T_f
                #   H_er = (1/(r cos f)) T_rl - (1/(r^2 cos f)) T_l
                # P_use/dP_use already contain q^n, and the final GM/r^3
                # scaling cancels the remaining radius factors.
                acc_xy += _S(dP_use + tan_f * P_use,
                             m_neg_Cu_sin_Sm_cos) / cos_row
                acc_xz -= _S(dP_use, Cu_cos_Su_sin * (n_arr + 2.0))
                acc_yz -= _S(P_use, m_neg_Cu_sin_Sm_cos *
                             (n_arr + 2.0)) / cos_row

            # ---- 15: Vxy, Vxz, Vyz (same geometry, full field) ----
            elif fc == 15:
                acc_xy_f += _S(dP_use + tan_f * P_use,
                               m_neg_Cu_sin_Sm_cos) / cos_row
                acc_xz_f -= _S(dP_use, Cu_cos_Su_sin * (n_arr + 2.0))
                acc_yz_f -= _S(P_use, m_neg_Cu_sin_Sm_cos *
                               (n_arr + 2.0)) / cos_row

            # ---- 10: Geoid undulation ----
            elif fc == 10:
                acc_N1c += _S(P_use, Cu_cos_Su_sin)
                if HC_all is not None:
                    acc_H += _S(P_use * rn_factor, HC_use * cos_mc + HS_use * sin_mc)

            # ---- 11: V ----
            elif fc == 11:
                acc[fc] += _S(P_use, CcSs_f)

            # ---- 12: Vrr, Vff, Vll ----
            elif fc == 12:
                acc_rr_f += _S(P_use, CcSs_f * ampl_np1_np2)
                acc_ff_f += _S(ddP_use, CcSs_f)
                acc_ll_f += _S(P_use, CcSs_f * m**2)

            # ---- 13: Vrf, Vrl, Vfl ----
            elif fc == 13:
                acc_rf_f += _S(dP_use, CcSs_f * ampl_np1)
                acc_rl_f += _S(P_use, mneg_f * ampl_np1)
                acc_fl_f += _S(dP_use, mneg_f)

            # ---- 14: Vxx, Vyy, Vzz (LNOF diagonal, gravitational) ----
            elif fc == 14:
                acc_zz_f += _S(P_use, CcSs_f * ampl_np1_np2)

                if m < 2:
                    bnm = (n_arr + m + 1) * (n_arr + m + 2) / 2.0 / (m + 1)
                    if m == 0:
                        anm = (np.sqrt(2) / 4.0 *
                               np.sqrt(np.maximum(n_arr**2 - (m + 1)**2, 0)) *
                               np.sqrt(n_arr + m + 2) *
                               np.sqrt(np.maximum(n_arr - m, 0)))
                    else:
                        anm = (1.0 / 4.0 *
                               np.sqrt(np.maximum(n_arr**2 - (m + 1)**2, 0)) *
                               np.sqrt(n_arr + m + 2) *
                               np.sqrt(np.maximum(n_arr - m, 0)))
                    cnm = np.zeros(n_len)
                else:
                    bnm = (n_arr**2 + m**2 + 3 * n_arr + 2) / 2.0
                    anm = (1.0 / 4.0 *
                           np.sqrt(np.maximum(n_arr**2 - (m + 1)**2, 0)) *
                           np.sqrt(n_arr + m + 2) *
                           np.sqrt(np.maximum(n_arr - m, 0)))
                    if m == 2:
                        cnm = (np.sqrt(2) / 4.0 *
                               np.sqrt(np.maximum(n_arr**2 - (m - 1)**2, 0)) *
                               np.sqrt(np.maximum(n_arr - m + 2, 0)) *
                               np.sqrt(n_arr + m))
                    else:
                        cnm = (1.0 / 4.0 *
                               np.sqrt(np.maximum(n_arr**2 - (m - 1)**2, 0)) *
                               np.sqrt(np.maximum(n_arr - m + 2, 0)) *
                               np.sqrt(n_arr + m))

                Vxx2_inner = CcSs_f * (bnm - ampl_np1_np2)
                Vyy2_inner = CcSs_f * bnm

                if m + 2 <= nmax:
                    Vxx1_inner = (Cp2_f * cos_m2c + Sp2_f * sin_m2c) * anm
                else:
                    Vxx1_inner = np.zeros((n_pts, n_len))
                if m >= 2:
                    Vxx3_inner = (Cm2_f * cos_mm2c + Sm2_f * sin_mm2c) * cnm
                else:
                    Vxx3_inner = np.zeros((n_pts, n_len))

                acc_xx_f += _S(P_use, Vxx1_inner + Vxx2_inner + Vxx3_inner)
                acc_yy_f += _S(P_use, Vxx1_inner + Vyy2_inner + Vxx3_inner)

            # ---- 16 / 20: gravity vector and gravity disturbance ----
            elif fc == 16:
                # The two codes share these accumulators; code 20 removes the
                # normal field during finalisation.
                acc_Wr += _S(P_use, CcSs_f * ampl_np1)
                acc_Wlam += _S(P_use, mneg_f)
                acc_Wfi += _S(dP_use, CcSs_f)
                if need_normal and m == 0:
                    acc_Ur += _S(P_use, CElm * cos_mc * ampl_np1)
                    acc_Ufi += _S(dP_use, CElm * cos_mc)

            # ---- 17: g_sa ----
            elif fc == 17:
                acc[fc] += _S(P_use, CcSs_f * ampl_np1)

            # ---- 18: W ----
            elif fc == 18:
                acc[fc] += _S(P_use, CcSs_f)

            # ---- 19: gravity anomaly sa ----
            elif fc == 19:
                acc[fc] += _S(P_use, Cu_cos_Su_sin * ampl_nm1)

            # ---- 21: gravity disturbance sa ----
            elif fc == 21:
                acc[fc] += _S(P_use, Cu_cos_Su_sin * ampl_np1)

            # ---- 22: height anomaly ell ----
            elif fc == 22:
                acc[fc] += _S(P_use, Cu_cos_Su_sin)

            # ---- 23: height anomaly ----
            elif fc == 23:
                base = _S(P_use, Cu_cos_Su_sin)
                acc_zN1c += base
                acc_zdg += _S(P_use, Cu_cos_Su_sin * ampl_np1)
                if HC_all is not None:
                    acc_zH += _S(P_use * rn_factor, HC_use * cos_mc + HS_use * sin_mc)

            # ---- 24: T_rr ----
            elif fc == 24:
                acc[fc] += _S(P_use, Cu_cos_Su_sin * ampl_np1_np2)

            # ---- 25: W_rr ----
            elif fc == 25:
                acc[fc] += _S(P_use, CcSs_f * ampl_np1_np2)

    # Final scaling — convert accumulated sums to physical values
    results = {}
    pole = np.abs(np.abs(fi) - np.pi / 2) < 1e-15
    cos_fiG = np.cos(fiG)

    for fc in functional_codes:

        if fc == 2:  # eta
            eta = acc[fc]
            eta[pole] = 0.0
            results[fc] = -GM / (r**2 * gammaP * cos_fiG) * eta * ARCSEC

        elif fc == 3:  # xi
            results[fc] = -GM / (r**2 * gammaP) * acc[fc] * ARCSEC

        elif fc == 4:  # Theta
            eta = acc_eta4
            eta[pole] = 0.0
            eta_s = -GM / (r**2 * gammaP * cos_fiG) * eta * ARCSEC
            ksi_s = -GM / (r**2 * gammaP) * acc_ksi4 * ARCSEC
            alpha = np.arctan2(eta_s, ksi_s)
            alpha[alpha < 0] += 2 * np.pi
            results[fc] = np.column_stack([np.sqrt(eta_s**2 + ksi_s**2),
                                           np.rad2deg(alpha)])

        elif fc == 5:  # T
            results[fc] = GM / r * acc[fc]

        elif fc == 6:  # Trr, Tff, Tll
            Pg_rr = GM / r**3 * acc_rr * EOTVOS
            Pg_ff = GM / r**3 * acc_ff * EOTVOS
            Pg_ll = GM / (r**3 * cos_fiG**2) * acc_ll * EOTVOS
            results[fc] = np.column_stack([Pg_rr, Pg_ff, -Pg_ll])

        elif fc == 7:  # Trf, Trl, Tfl
            Pg_rf = GM / r**3 * acc_rf * EOTVOS
            Pg_rl = GM / (r**3 * cos_fiG) * acc_rl * EOTVOS
            Pg_fl = GM / (r**3 * cos_fiG) * acc_fl * EOTVOS
            results[fc] = np.column_stack([-Pg_rf, -Pg_rl, Pg_fl])

        elif fc == 8:  # Txx, Tyy, Tzz
            Pg_xx = GM / r**3 * acc_xx * EOTVOS
            Pg_yy = GM / r**3 * acc_yy * EOTVOS
            Pg_zz = GM / r**3 * acc_zz * EOTVOS
            results[fc] = np.column_stack([Pg_xx, -Pg_yy, Pg_zz])

        elif fc == 9:  # Txy, Txz, Tyz (ICGEM: xy/yz negated, cf. GrafLab)
            Pg_xy = GM / r**3 * acc_xy * EOTVOS
            Pg_xz = GM / r**3 * acc_xz * EOTVOS
            Pg_yz = GM / r**3 * acc_yz * EOTVOS
            results[fc] = np.column_stack([-Pg_xy, Pg_xz, -Pg_yz])

        elif fc == 10:  # Geoid
            N1c = GM / (r * gammaP) * acc_N1c
            if dtm is not None:
                H = acc_H.copy()
                H[H < 0] = 0.0
                N1c -= 2 * np.pi * G_GRAV * RHO_CRUST * H**2 / gammaP
            results[fc] = N1c

        elif fc == 11:  # V
            results[fc] = GM / r * acc[fc]

        elif fc == 12:  # Vrr, Vff, Vll
            Pg_rr = GM / r**3 * acc_rr_f * EOTVOS
            Pg_ff = GM / r**3 * acc_ff_f * EOTVOS
            Pg_ll = GM / (r**3 * cos_fiG**2) * acc_ll_f * EOTVOS
            results[fc] = np.column_stack([Pg_rr, Pg_ff, -Pg_ll])

        elif fc == 13:  # Vrf, Vrl, Vfl
            Pg_rf = GM / r**3 * acc_rf_f * EOTVOS
            Pg_rl = GM / (r**3 * cos_fiG) * acc_rl_f * EOTVOS
            Pg_fl = GM / (r**3 * cos_fiG) * acc_fl_f * EOTVOS
            results[fc] = np.column_stack([-Pg_rf, -Pg_rl, Pg_fl])

        elif fc == 14:  # Vxx, Vyy, Vzz
            Pg_xx = GM / r**3 * acc_xx_f * EOTVOS
            Pg_yy = GM / r**3 * acc_yy_f * EOTVOS
            Pg_zz = GM / r**3 * acc_zz_f * EOTVOS
            results[fc] = np.column_stack([Pg_xx, -Pg_yy, Pg_zz])

        elif fc == 15:  # Vxy, Vxz, Vyz (ICGEM: xy/yz negated, cf. GrafLab)
            Pg_xy = GM / r**3 * acc_xy_f * EOTVOS
            Pg_xz = GM / r**3 * acc_xz_f * EOTVOS
            Pg_yz = GM / r**3 * acc_yz_f * EOTVOS
            results[fc] = np.column_stack([-Pg_xy, Pg_xz, -Pg_yz])

        elif fc == 16:  # Gravity vector
            Wr = -GM / r**2 * acc_Wr + omega**2 * r * cos_fiG**2
            Wlam = -GM / (r**2 * cos_fiG) * acc_Wlam
            Wfi = ((GM / r * acc_Wfi +
                    (-omega**2 * r**2 * cos_fiG * np.sin(fiG))) / r)
            results[fc] = np.column_stack([Wfi * MGAL, Wlam * MGAL, Wr * MGAL])

        elif fc == 17:  # g_sa
            g_sa = GM / r**2 * acc[fc]
            results[fc] = np.abs(-g_sa + omega**2 * r * cos_fiG**2) * MGAL

        elif fc == 18:  # W
            results[fc] = GM / r * acc[fc] + 0.5 * omega**2 * r**2 * cos_fiG**2

        elif fc == 19:  # gravity anomaly sa
            results[fc] = GM / r**2 * acc[fc] * MGAL

        elif fc == 20:  # Gravity disturbance
            Wr = GM / r**2 * acc_Wr
            Wr_mag = np.sqrt((-Wr + omega**2 * r * cos_fiG**2)**2)
            Wlam_s = GM / r * acc_Wlam / (r * cos_fiG)
            Wfi_s = (GM / r * acc_Wfi -
                     omega**2 * r**2 * cos_fiG * np.sin(fiG)) / r
            grad_W = np.sqrt(Wr_mag**2 + Wlam_s**2 + Wfi_s**2)

            Ur = GM / r**2 * acc_Ur
            Ur_mag = np.sqrt((-Ur + omega**2 * r * cos_fiG**2)**2)
            Ufi_s = (GM / r * acc_Ufi -
                     omega**2 * r**2 * cos_fiG * np.sin(fiG)) / r
            grad_U = np.sqrt(Ur_mag**2 + Ufi_s**2)

            results[fc] = (grad_W - grad_U) * MGAL

        elif fc == 21:  # gravity disturbance sa
            results[fc] = GM / r**2 * acc[fc] * MGAL

        elif fc == 22:  # height anomaly ell
            results[fc] = GM / (r * gammaP) * acc[fc]

        elif fc == 23:  # height anomaly
            zeta_el = GM / (r * gammaP) * acc_zN1c
            dg = GM / r**2 * acc_zdg
            if dtm is not None:
                H = acc_zH.copy()
                H[H < 0] = 0.0
            else:
                H = np.zeros(n_pts)
            N = GM / (r * gammaP) * acc_zN1c - (
                2 * np.pi * G_GRAV * RHO_CRUST * H**2) / gammaP
            results[fc] = zeta_el - dg * (H + N) / gammaP

        elif fc == 24:  # T_rr
            results[fc] = GM / r**3 * acc[fc] * EOTVOS

        elif fc == 25:  # W_rr
            results[fc] = (GM / r**3 * acc[fc] +
                           omega**2 * cos_fiG**2) * EOTVOS

    return results, lon, lat
