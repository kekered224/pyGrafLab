"""
Fully normalised Associated Legendre Functions (fnALFs) and their
first / second order derivatives with respect to geocentric latitude.

Three computation methods, with the validity ranges stated in GrafLab:
  1 – Standard forward column method
      all latitudes up to degree 1800; degrees up to 2190 between 0–56°
      and 80–90°
  2 – Modified forward column method (×1e-280 scaling with Horner summation)
      all latitudes from degree 1801 to 2700; overflows above 2700
  3 – Extended-range arithmetic
      any latitude, any degree

All return *modified* fnALFs, i.e. multiplied by q^n = (R/r)^n.

Ported from GrafLab 2.1.4 (lines 2444-2992).
"""

import os

import numpy as np

# ---------------------------------------------------------------------------
# Optional Numba acceleration
# ---------------------------------------------------------------------------
# The n-recursion is inherently sequential, so the pure-NumPy version pays
# ~2.4e6 Python-loop iterations at degree 2190.  When Numba is available the
# same recurrence is executed by a compiled kernel (bit-identical float64
# arithmetic, same operation order) which is roughly two orders of magnitude
# faster.  Set PYGRAFLAB_NO_NUMBA=1 to force the pure-NumPy path.
try:  # pragma: no cover - environment dependent
    from numba import njit as _njit
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False
    _njit = None

USE_NUMBA = _HAVE_NUMBA and os.environ.get("PYGRAFLAB_NO_NUMBA", "") == ""

if USE_NUMBA:  # pragma: no cover - compiled path
    @_njit(cache=True, fastmath=False, nogil=True)
    def _standard_kernel(Pnm, t, q, q2, m, nmax):
        """Forward-column recurrence for one order m (in place)."""
        n_pts = Pnm.shape[0]
        n_cols = Pnm.shape[1]
        if m < nmax:
            n = m + 1
            anm = np.sqrt((2.0*n - 1.0) * (2.0*n + 1.0)
                          / ((n - m) * (n + m)))
            for i in range(n_pts):
                Pnm[i, 1] = anm * t[i] * Pnm[i, 0] * q[i]
        for j in range(2, n_cols):
            n = m + j
            anm = np.sqrt((2.0*n - 1.0) * (2.0*n + 1.0)
                          / ((n - m) * (n + m)))
            bnm = np.sqrt((2.0*n + 1.0) * (n + m - 1.0) * (n - m - 1.0)
                          / ((n - m) * (n + m) * (2.0*n - 3.0)))
            for i in range(n_pts):
                Pnm[i, j] = (anm * t[i] * Pnm[i, j-1] * q[i]
                             - bnm * Pnm[i, j-2] * q2[i])

    @_njit(cache=True, fastmath=False, nogil=True)
    def _d1_kernel(dPnm, Pnm, t, u, q, m, nmax):
        """First derivative w.r.t. geocentric latitude for one order m."""
        n_pts = Pnm.shape[0]
        n_cols = Pnm.shape[1]
        if m == 0:
            for i in range(n_pts):
                dPnm[i, 0] = 0.0
            if nmax >= 1:
                for i in range(n_pts):
                    dPnm[i, 1] = ROOT3 * u[i] * q[i]
            for j in range(2, n_cols):
                n = m + j
                enm = np.sqrt((2.0*n + 1.0) * (n - m) * (n + m)
                              / (2.0*n - 1.0))
                for i in range(n_pts):
                    if u[i] != 0.0:
                        tu = t[i] / u[i]
                        qu = q[i] / u[i]
                    else:
                        tu = 0.0
                        qu = 0.0
                    dPnm[i, j] = -n * tu * Pnm[i, j] + enm * qu * Pnm[i, j-1]
        else:
            for i in range(n_pts):
                if u[i] != 0.0:
                    tu = t[i] / u[i]
                else:
                    tu = 0.0
                dPnm[i, 0] = -m * tu * Pnm[i, 0]
            for j in range(1, n_cols):
                n = m + j
                enm = np.sqrt((2.0*n + 1.0) * (n - m) * (n + m)
                              / (2.0*n - 1.0))
                for i in range(n_pts):
                    if u[i] != 0.0:
                        tu = t[i] / u[i]
                        qu = q[i] / u[i]
                    else:
                        tu = 0.0
                        qu = 0.0
                    dPnm[i, j] = -n * tu * Pnm[i, j] + enm * qu * Pnm[i, j-1]

    @_njit(cache=True, fastmath=False, nogil=True)
    def _dd_kernel(ddPnm, Pnm, dPnm, t, u, m, nmax):
        """Second derivative w.r.t. geocentric latitude for one order m."""
        n_pts = Pnm.shape[0]
        n_cols = Pnm.shape[1]
        for i in range(n_pts):
            if u[i] != 0.0:
                tu = t[i] / u[i]
                m2u2 = (m * m) / (u[i] * u[i])
            else:
                tu = 0.0
                m2u2 = 0.0
            for j in range(n_cols):
                n = m + j
                nn1 = n * (n + 1.0)
                if m == 0:
                    ddPnm[i, j] = tu * dPnm[i, j] - nn1 * Pnm[i, j]
                else:
                    ddPnm[i, j] = (tu * dPnm[i, j] + m2u2 * Pnm[i, j]
                                   - nn1 * Pnm[i, j])

# ---------------------------------------------------------------------------
# Extended-range constants
# ---------------------------------------------------------------------------
IND = 960
BIG = 2.0**IND
BIGI = 2.0 ** (-IND)
BIGS = 2.0 ** (IND // 2)      # 2^480
BIGSI = 2.0 ** (-(IND // 2))  # 2^-480
ROOT3 = np.sqrt(3.0)


# ===================================================================
#  Method 1 – Standard forward column method
# ===================================================================

def fnalfs_standard(fiG, q, nmax, m):
    """Compute modified fnALFs using the standard forward column method.

    Parameters
    ----------
    fiG  : ndarray (N,) – geocentric latitude [rad]
    q    : ndarray (N,) – R/r for each point
    nmax : int
    m    : int – current order

    Returns
    -------
    Pnm : ndarray (N, nmax - m + 1)
    """
    t = np.sin(fiG)
    u = np.cos(fiG)
    q2 = q ** 2
    np_ = nmax - m + 1
    N = len(fiG)
    Pnm = np.zeros((N, np_))

    # --- sectorial seed ---
    if m == 0:
        Pnm[:, 0] = 1.0
    elif m == 1:
        Pnm[:, 0] = ROOT3 * u * q
    else:
        i_arr = np.arange(2, m + 1, dtype=float)
        prod_val = np.prod(np.sqrt((2 * i_arr + 1) / (2 * i_arr)))
        Pnm[:, 0] = u**m * ROOT3 * prod_val * q**m

    # --- n = m+1 ---
    if USE_NUMBA:
        _standard_kernel(Pnm, t, q, q2, m, nmax)
        return Pnm

    if m < nmax:
        n = m + 1
        anm = np.sqrt((2*n - 1) * (2*n + 1) / ((n - m) * (n + m)))
        Pnm[:, 1] = anm * t * Pnm[:, 0] * q

    # --- n = m+2 .. nmax ---
    for j, n in enumerate(range(m + 2, nmax + 1), start=2):
        anm = np.sqrt((2*n - 1) * (2*n + 1) / ((n - m) * (n + m)))
        bnm = np.sqrt((2*n + 1) * (n + m - 1) * (n - m - 1) /
                       ((n - m) * (n + m) * (2*n - 3)))
        Pnm[:, j] = anm * t * Pnm[:, j-1] * q - bnm * Pnm[:, j-2] * q2

    return Pnm


# ===================================================================
#  Method 2 – Modified forward column method (1e-280 scaling)
# ===================================================================

def fnalfs_modified(fiG, q, nmax, m):
    """Compute the fnALFs for the modified forward column method (method 2).

    Two deviations from the standard recurrence, both needed to keep the
    coefficients inside the float64 range at very high degree:

    * the sectorial seed carries a 1e-280 scale factor;
    * the ``u**m`` power of cos(latitude) is factored out of the seed.

    The returned array is therefore ``Pnm / u**m`` (times 1e-280), i.e. the
    quantity that GrafLab's Horner summation consumes order by order.  The
    caller has to restore both factors before the values are used as Legendre
    functions -- see ``synthesis.synthesize_pointwise``.
    """
    t = np.sin(fiG)
    q2 = q ** 2
    np_ = nmax - m + 1
    N = len(fiG)
    Pnm = np.zeros((N, np_))
    S = 1e-280

    # Sectorial seed without the u**m factor (GrafLab 2.1.4, method 2).
    if m == 0:
        Pnm[:, 0] = S
    elif m == 1:
        Pnm[:, 0] = ROOT3 * q * S
    else:
        i_arr = np.arange(2, m + 1, dtype=float)
        prod_val = np.prod(np.sqrt((2 * i_arr + 1) / (2 * i_arr)))
        Pnm[:, 0] = ROOT3 * prod_val * q**m * S

    if USE_NUMBA:
        _standard_kernel(Pnm, t, q, q2, m, nmax)
        return Pnm

    if m < nmax:
        n = m + 1
        anm = np.sqrt((2*n - 1) * (2*n + 1) / ((n - m) * (n + m)))
        Pnm[:, 1] = anm * t * Pnm[:, 0] * q

    for j, n in enumerate(range(m + 2, nmax + 1), start=2):
        anm = np.sqrt((2*n - 1) * (2*n + 1) / ((n - m) * (n + m)))
        bnm = np.sqrt((2*n + 1) * (n + m - 1) * (n - m - 1) /
                       ((n - m) * (n + m) * (2*n - 3)))
        Pnm[:, j] = anm * t * Pnm[:, j-1] * q - bnm * Pnm[:, j-2] * q2

    return Pnm


# ===================================================================
#  Method 3 – Extended-range arithmetic  (64-bit vectorised path)
# ===================================================================

def _precompute_extended(nmax, fiG_ALFs, q_ALFs):
    """Pre-compute lookup tables and sectorial seeds for method 3.

    Returns
    -------
    rr, ri : ndarray – sqrt and 1/sqrt lookup
    am_arr, bm_arr : dict-of-arrays keyed by m (filled lazily in the loop)
    ps1b, ips1b : ndarray (N_half, nmax) – sectorial seeds
    tq, q2 : ndarray
    temp helpers
    """
    N = len(fiG_ALFs)
    t = np.sin(fiG_ALFs)
    u = np.cos(fiG_ALFs)
    q = q_ALFs
    tq = t * q
    q2 = q ** 2

    nmax23 = 2 * nmax + 3
    m1 = np.arange(1, nmax23 + 1, dtype=float)
    rr = np.sqrt(m1)
    ri = 1.0 / rr

    m2 = np.arange(1, nmax + 1, dtype=float)
    dd = np.zeros(nmax + 1)
    dd[1:] = rr[2 * m2.astype(int) + 3 - 1] * ri[2 * m2.astype(int) + 2 - 1]
    # dd[k] = sqrt(2k+3) / sqrt(2k+2)  for k=1..nmax
    for k in range(1, nmax + 1):
        dd[k] = np.sqrt(2*k + 3) / np.sqrt(2*k + 2)

    # --- Sectorial seeds ---
    ps1b = np.zeros((N, nmax + 1))   # ps1b[:, m] for m=1..nmax
    ips1b = np.zeros((N, nmax + 1), dtype=int)

    xb = ROOT3 * u * q
    ixb = np.zeros(N, dtype=int)
    ps1b[:, 1] = xb.copy()
    ips1b[:, 1] = ixb.copy()

    for m3 in range(2, nmax + 1):
        xb = (dd[m3 - 1] * u) * xb * q
        yb = np.abs(xb)
        mask_big = yb >= BIGS
        mask_small = yb < BIGSI
        xb[mask_big] *= BIGI
        ixb[mask_big] += 1
        xb[mask_small] *= BIG
        ixb[mask_small] -= 1
        ps1b[:, m3] = xb.copy()
        ips1b[:, m3] = ixb.copy()

    return rr, ri, dd, ps1b, ips1b, tq, q2


def _recurrence_coefficients(rr, ri, nmax, m):
    """Compute am[m..nmax] and bm[m+2..nmax] for extended-range method."""
    am = np.zeros(nmax + 1)
    bm = np.zeros(nmax + 1)
    am[m + 1 - 1] = rr[2*m + 3 - 1]  # am(m+1) = sqrt(2m+3), using 1-based->0-based
    # Actually let's use 1-based indexing to match MATLAB
    am = np.zeros(nmax + 2)
    bm = np.zeros(nmax + 2)
    if m + 1 <= nmax:
        am[m + 1] = rr[2*m + 3 - 1]  # rr is 1-indexed from MATLAB
    for n in range(m + 2, nmax + 1):
        ww = rr[2*n + 1 - 1] * ri[n - m - 1] * ri[n + m - 1]
        am[n] = rr[2*n - 1 - 1] * ww
        bm[n] = rr[n - m - 1 - 1] * rr[n + m - 1 - 1] * ri[2*n - 3 - 1] * ww
    return am, bm


def fnalfs_extended(fiG, q, nmax, m, precomp):
    """Compute modified fnALFs using extended-range arithmetic (64-bit path).

    Parameters
    ----------
    fiG     : ndarray (N,)
    q       : ndarray (N,)
    nmax    : int
    m       : int – current order
    precomp : tuple from _precompute_extended

    Returns
    -------
    Pnm : ndarray (N, nmax - m + 1)
    """
    rr, ri, dd, ps1b, ips1b, tq, q2 = precomp
    N = len(fiG)
    np_ = nmax - m + 1
    Pnm = np.zeros((N, np_))

    # --- Recurrence coefficients ---
    am, bm = _recurrence_coefficients(rr, ri, nmax, m)

    if m == 0:
        # Zonal: simple vectorised recurrence (no extended-range needed)
        Pnm[:, 0] = 1.0
        Pnm[:, 1] = ROOT3 * tq
        for i in range(2, nmax + 1):
            Pnm[:, i] = (Pnm[:, i-1] * np.sqrt((2*i + 1) * (2*i - 1)) / i * tq
                         - q2 * Pnm[:, i-2] * (i - 1) * np.sqrt(2*i + 1) /
                         (i * np.sqrt(2*i - 3)))
    else:
        # Non-zonal: extended-range arithmetic
        xb = ps1b[:, m].copy()
        ixb = ips1b[:, m].copy()

        # Convert to real values
        temp5 = np.full(N, BIGI)
        temp5[ixb == 0] = 1.0
        temp5[ixb < -1] = 0.0
        temp5[ixb > 0] = BIG
        Pnm[:, 0] = xb * temp5
        temp5[:] = BIGI

        if m < nmax:
            yb = xb.copy()
            iyb = ixb.copy()

            xb = (am[m + 1] * tq) * yb
            ixb = iyb.copy()
            wb = np.abs(xb)

            mask_big = wb >= BIGS
            mask_small = wb < BIGSI
            temp3 = np.ones(N)
            temp4 = np.zeros(N, dtype=int)
            temp3[mask_big] = BIGI
            temp3[mask_small] = BIG
            temp4[mask_big] = 1
            temp4[mask_small] = -1
            xb *= temp3
            ixb += temp4

            temp5 = np.full(N, BIGI)
            temp5[ixb == 0] = 1.0
            temp5[ixb < -1] = 0.0
            temp5[ixb > 0] = BIG
            Pnm[:, 1] = xb * temp5
            temp5[:] = BIGI

            for n in range(m + 2, nmax + 1):
                idb = ixb - iyb

                temp1 = np.zeros(N)
                temp2 = np.ones(N)

                id0 = idb == 0
                id1 = idb == 1
                idm1 = idb == -1
                idv1 = idb > 1

                temp1[id0] = 1.0
                temp1[id1] = 1.0
                temp2[id1] = BIGI
                temp1[idm1] = BIGI
                temp1[idv1] = 1.0
                temp2[idv1] = 0.0

                zzb = (am[n] * tq) * (xb * temp1) - bm[n] * ((yb * q2) * temp2)
                izb = iyb.copy()
                mask_set = id0 | id1 | idv1
                izb[mask_set] = ixb[mask_set]

                temp1[:] = 0.0
                temp2[:] = 1.0

                wb = np.abs(zzb)
                mask_big = wb >= BIGS
                mask_small = wb < BIGSI
                temp3 = np.ones(N)
                temp4 = np.zeros(N, dtype=int)
                temp3[mask_big] = BIGI
                temp3[mask_small] = BIG
                temp4[mask_big] = 1
                temp4[mask_small] = -1

                zzb *= temp3
                izb += temp4

                temp5 = np.full(N, BIGI)
                temp5[izb == 0] = 1.0
                temp5[izb < -1] = 0.0
                temp5[izb > 0] = BIG
                Pnm[:, n - m] = zzb * temp5
                temp5[:] = BIGI

                yb = xb.copy()
                iyb = ixb.copy()
                xb = zzb.copy()
                ixb = izb.copy()

    return Pnm


# ===================================================================
#  Apply symmetric-grid optimisation to method 3
# ===================================================================

def apply_symmetric_grid(Pnm_half, nmax, m, n_half, odd):
    """Extend Pnm from northern half to full grid using parity property.

    Pnm for the full grid is built in-place (the caller must allocate
    the full-sized Pnm and fill the first n_half rows).

    Parameters
    ----------
    Pnm_half : ndarray (n_half, nmax - m + 1) – northern half values
    nmax, m  : int
    n_half   : int – number of northern latitudes
    odd      : int – 1 if total number of latitudes is odd (equator row shared)

    Returns
    -------
    Pnm_full : ndarray (2*n_half - odd, nmax - m + 1)
    """
    n_rows = 2 * n_half - odd
    np_ = nmax - m + 1
    Pnm = np.zeros((n_rows, np_))
    Pnm[:n_half, :] = Pnm_half

    # Mirror: southern = reversed northern
    src = Pnm_half[(n_half - odd - 1)::-1, :]  # reverse
    Pnm[n_half:, :] = src

    # Negate columns with odd (n - m)
    if m == nmax:
        # Only 1 column: n-m = 0 (even) — no negation needed
        pass
    elif m == nmax - 1:
        # 2 columns: n-m = 0 (even), 1 (odd)
        Pnm[n_half:, 1] = -Pnm[n_half:, 1]
    else:
        # Columns 1, 3, 5, ... (0-indexed: 1, 3, 5...) have odd (n-m)
        Pnm[n_half:, 1::2] = -Pnm[n_half:, 1::2]

    return Pnm


# ===================================================================
#  First-order derivatives of modified fnALFs
# ===================================================================

def fnalfs_first_derivative(Pnm, fiG, q, nmax, m):
    """Compute d(Pnm)/d(fiG) — first derivative w.r.t. geocentric latitude.

    Parameters
    ----------
    Pnm  : ndarray (N, nmax - m + 1) – already computed fnALFs
    fiG  : ndarray (N,)
    q    : ndarray (N,)
    nmax : int
    m    : int

    Returns
    -------
    dPnm : ndarray (N, nmax - m + 1)
    """
    t = np.sin(fiG)
    u = np.cos(fiG)
    N = len(fiG)
    np_ = nmax - m + 1
    dPnm = np.zeros((N, np_))

    if USE_NUMBA:
        _d1_kernel(dPnm, Pnm, t, u, q, m, nmax)
        pole_mask = np.abs(np.abs(fiG) - np.pi / 2) < 1e-15
        if np.any(pole_mask):
            dPnm[pole_mask, :] = 0.0
        return dPnm

    # Protect against division by zero at poles
    with np.errstate(divide='ignore', invalid='ignore'):
        tu = np.where(np.abs(u) > 0, t / u, 0.0)
        qu = np.where(np.abs(u) > 0, q / u, 0.0)

    # e_nm = sqrt( (2n+1)(n-m)(n+m) / (2n-1) )
    n_arr = np.arange(m, nmax + 1, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        enm = np.sqrt((2*n_arr + 1) * (n_arr - m) * (n_arr + m) / (2*n_arr - 1))
    # For n=m=0 the formula gives 0/(-1) which is fine (n-m=0 makes numerator 0)
    enm = np.nan_to_num(enm, nan=0.0)

    if m == 0:
        dPnm[:, 0] = 0.0
        if nmax >= 1:
            dPnm[:, 1] = ROOT3 * u * q
        for j, n in enumerate(range(2, nmax + 1), start=2):
            dPnm[:, j] = -n * tu * Pnm[:, j] + enm[j] * qu * Pnm[:, j-1]
    elif m == nmax:
        dPnm[:, 0] = -m * tu * Pnm[:, 0]
    else:
        dPnm[:, 0] = -m * tu * Pnm[:, 0]
        for j, n in enumerate(range(m + 1, nmax + 1), start=1):
            dPnm[:, j] = -n * tu * Pnm[:, j] + enm[j] * qu * Pnm[:, j-1]

    # Singularity at poles
    pole_mask = np.abs(np.abs(fiG) - np.pi / 2) < 1e-15
    if np.any(pole_mask):
        dPnm[pole_mask, :] = 0.0

    return dPnm


# ===================================================================
#  Second-order derivatives of modified fnALFs
# ===================================================================

def fnalfs_second_derivative(Pnm, dPnm, fiG, nmax, m):
    """Compute d²(Pnm)/d(fiG)² — second derivative w.r.t. geocentric latitude.

    Parameters
    ----------
    Pnm  : ndarray (N, nmax - m + 1)
    dPnm : ndarray (N, nmax - m + 1)
    fiG  : ndarray (N,)
    nmax : int
    m    : int

    Returns
    -------
    ddPnm : ndarray (N, nmax - m + 1)
    """
    u = np.cos(fiG)
    N = len(fiG)
    np_ = nmax - m + 1
    ddPnm = np.zeros((N, np_))

    if USE_NUMBA:
        _dd_kernel(ddPnm, Pnm, dPnm, np.sin(fiG), u, m, nmax)
        pole_mask = np.abs(np.abs(fiG) - np.pi / 2) < 1e-15
        if np.any(pole_mask):
            ddPnm[pole_mask, :] = 0.0
        return ddPnm

    with np.errstate(divide='ignore', invalid='ignore'):
        tu = np.where(np.abs(u) > 0, np.sin(fiG) / u, 0.0)
        m2_over_u2 = np.where(np.abs(u) > 0, m**2 / u**2, 0.0)

    if m == 0:
        n_arr = np.arange(0, nmax + 1, dtype=float)
        # ddPnm = tu * dPnm - n*(n+1) * Pnm
        ddPnm = tu[:, None] * dPnm - (n_arr * (n_arr + 1))[None, :] * Pnm
    else:
        n_arr = np.arange(m, nmax + 1, dtype=float)
        # ddPnm(:, 1:end-m) = tu*dPnm + m^2/u^2 * Pnm - n*(n+1)*Pnm
        cols = nmax - m + 1  # = end - m  when indexing from 0 to nmax-m
        ddPnm[:, :cols] = (tu[:, None] * dPnm[:, :cols]
                           + m2_over_u2[:, None] * Pnm[:, :cols]
                           - (n_arr * (n_arr + 1))[None, :] * Pnm[:, :cols])

    # Singularity at poles
    pole_mask = np.abs(np.abs(fiG) - np.pi / 2) < 1e-15
    if np.any(pole_mask):
        ddPnm[pole_mask, :] = 0.0

    return ddPnm


# ===================================================================
#  Unified interface
# ===================================================================

def compute_fnalfs(fiG, q, nmax, m, method, precomp=None):
    """Compute modified fnALFs using the specified method.

    Parameters
    ----------
    method  : int – 1, 2, or 3
    precomp : precomputation tuple (only needed for method 3)

    Returns
    -------
    Pnm : ndarray (N, nmax - m + 1)
    """
    if method == 1:
        return fnalfs_standard(fiG, q, nmax, m)
    elif method == 2:
        return fnalfs_modified(fiG, q, nmax, m)
    elif method == 3:
        if precomp is None:
            precomp = _precompute_extended(nmax, fiG, q)
        return fnalfs_extended(fiG, q, nmax, m, precomp)
    else:
        raise ValueError(f"Unknown fnALF method: {method}")
