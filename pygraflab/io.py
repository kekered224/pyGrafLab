"""
GGM / DTM / covariance-matrix file I/O.

Supports:
- ICGEM ``.gfc`` files (fully normalised coefficients)
- MATLAB ``.mat`` files (v4/v5 via scipy, v7.3/HDF5 via h5py)
- Plain ASCII text (4-column: n, m, Cnm, Snm)

Ported from GrafLab 2.1.4 (lines 1650-2082).
"""

import io as _io
import math
import os
import warnings

import numpy as np


# -----------------------------------------------------------------------
#  GGM (Global Geopotential Model) file reader
# -----------------------------------------------------------------------

class GGMModel:
    """Container for a loaded GGM.

    Attributes
    ----------
    GM   : float  – geocentric gravitational constant [m^3 s^-2]
    R    : float  – reference sphere radius [m]
    nmin : int    – minimum degree in file
    nmax : int    – maximum degree in file (after truncation)
    n    : ndarray – degree array  (ncoeffs,)
    m    : ndarray – order array   (ncoeffs,)
    C    : ndarray – Cnm coefficients
    S    : ndarray – Snm coefficients
    index : ndarray – index[i] = row of first coefficient with degree (nmin + i)
    """
    def __init__(self):
        self.GM = None
        self.R = None
        self.nmin = 0
        self.nmax = 0
        self.n = None
        self.m = None
        self.C = None
        self.S = None
        self.index = None


def load_ggm(path, nmax=None, GM_default=3986004.415e8, R_default=6378136.3):
    """Load a GGM file and return a :class:`GGMModel`.

    Parameters
    ----------
    path       : str   – path to .gfc, .mat, or plain-text file
    nmax       : int or None – truncate to this max degree (None = use file max)
    GM_default : float – default GM if not in file header
    R_default  : float – default R  if not in file header
    """
    path = str(path)
    ext = os.path.splitext(path)[1].lower()

    if ext == '.gfc':
        return _load_gfc(path, nmax, GM_default, R_default)
    elif ext == '.mat':
        return _load_mat(path, nmax, GM_default, R_default)
    else:
        return _load_txt(path, nmax, GM_default, R_default)


def _load_gfc(path, nmax, GM_default, R_default):
    """Parse ICGEM .gfc file."""
    GM = GM_default
    R = R_default

    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        # --- scan header ---
        past_product_type = False
        while True:
            line = fh.readline()
            if not line:
                break
            stripped = line.strip()

            if not past_product_type:
                if stripped.lower().startswith('product_type'):
                    past_product_type = True
                continue

            if stripped.lower().startswith('earth_gravity_constant'):
                GM = float(stripped.split()[1])
            elif stripped.lower().startswith('radius'):
                R = float(stripped.split()[1])
            elif stripped.lower().startswith('norm'):
                val = stripped.split()[1] if len(stripped.split()) > 1 else ''
                if not val.lower().startswith('fully_normalized'):
                    raise ValueError(
                        "GrafLab requires fully normalised coefficients; "
                        f"file says norm = '{val}'.")
            elif stripped.lower().startswith('end_of_head'):
                break

        # --- read data ---
        # Columns: gfc/gfct  n  m  Cnm  Snm  [sigma_C  sigma_S  ...]
        # Newer ICGEM files may also contain trnd, acos, asin lines (skip those)
        #
        # Vectorised path: keep only the coefficient records and let NumPy
        # parse them. At degree 2190 (~2.4 M records) this is about twice as
        # fast as a Python loop. Fortran-style exponents ("1.234D-05") are
        # normalised to "E" beforehand. Anything the fast parser rejects falls
        # back to the line-by-line reader below, which also keeps the memory
        # footprint bounded for irregular files.
        rest = fh.read()

    n_arr = m_arr = C_arr = S_arr = None
    try:
        coef_lines = [ln for ln in rest.splitlines() if ln[:3].lower() == 'gfc']
        if coef_lines:
            text = '\n'.join(coef_lines).replace('D', 'E').replace('d', 'e')
            arr = np.loadtxt(_io.StringIO(text), usecols=(1, 2, 3, 4),
                             dtype=np.float64)
            if arr.ndim == 1:               # single coefficient record
                arr = arr.reshape(1, -1)
            n_arr = arr[:, 0].astype(np.int64)
            m_arr = arr[:, 1].astype(np.int64)
            C_arr = np.ascontiguousarray(arr[:, 2])
            S_arr = np.ascontiguousarray(arr[:, 3])
            if nmax is not None:
                keep = n_arr <= nmax
                n_arr, m_arr = n_arr[keep], m_arr[keep]
                C_arr, S_arr = C_arr[keep], S_arr[keep]
    except Exception:
        n_arr = None

    if n_arr is None:
        # Fallback: line-by-line parsing with geometrically growing buffers
        capacity = 1 << 16
        n_buf = np.empty(capacity, dtype=np.int64)
        m_buf = np.empty(capacity, dtype=np.int64)
        c_buf = np.empty(capacity, dtype=np.float64)
        s_buf = np.empty(capacity, dtype=np.float64)
        count = 0

        for line in rest.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            tag = parts[0].lower()
            if tag not in ('gfc', 'gfct'):
                # Skip non-coefficient lines (trnd, acos, asin, etc.)
                continue
            n = int(parts[1])
            if nmax is not None and n > nmax:
                continue
            m = int(parts[2])
            try:
                C = float(parts[3])
                S = float(parts[4])
            except ValueError:
                # Fortran-style exponents ('1.234D-05') appear in some files.
                C = float(parts[3].replace('d', 'e').replace('D', 'E'))
                S = float(parts[4].replace('d', 'e').replace('D', 'E'))

            if count == capacity:
                capacity *= 2
                grown = np.empty(capacity, dtype=np.int64)
                grown[:count] = n_buf[:count]
                n_buf = grown
                grown = np.empty(capacity, dtype=np.int64)
                grown[:count] = m_buf[:count]
                m_buf = grown
                grown = np.empty(capacity, dtype=np.float64)
                grown[:count] = c_buf[:count]
                c_buf = grown
                grown = np.empty(capacity, dtype=np.float64)
                grown[:count] = s_buf[:count]
                s_buf = grown

            n_buf[count] = n
            m_buf[count] = m
            c_buf[count] = C
            s_buf[count] = S
            count += 1

        n_arr = n_buf[:count]
        m_arr = m_buf[:count]
        C_arr = c_buf[:count]
        S_arr = s_buf[:count]

    if n_arr.size == 0:
        raise ValueError(
            f"No 'gfc' coefficient records found in '{path}'. "
            "Check that the file is a valid ICGEM model"
            + (f" containing degree <= {nmax}." if nmax is not None else ".")
        )

    # Sort by degree (primary) then order (secondary), matching MATLAB sortrows
    sort_idx = np.lexsort((m_arr, n_arr))
    n_arr = n_arr[sort_idx]
    m_arr = m_arr[sort_idx]
    C_arr = C_arr[sort_idx]
    S_arr = S_arr[sort_idx]

    return _build_ggm(n_arr, m_arr, C_arr, S_arr, GM, R, nmax, is_single=True)


def _load_mat(path, nmax, GM_default, R_default):
    """Load from .mat file (v4/v5 via scipy, v7.3/HDF5 via h5py)."""
    try:
        import scipy.io as sio
    except ImportError:
        sio = None

    if sio is not None:
        try:
            mat = sio.loadmat(path)
        except NotImplementedError:
            mat = None  # MATLAB v7.3 (HDF5) — handled below
        except Exception as exc:
            raise ValueError(f"Cannot read MATLAB file '{path}': {exc}") from exc

        if mat is not None:
            mat_keys = [k for k in mat.keys() if not k.startswith('__')]
            if len(mat_keys) == 1:
                return _load_from_matrix(mat[mat_keys[0]], nmax,
                                         GM_default, R_default)
            if len(mat_keys) == 3:
                nmin_ggm = int(np.asarray(mat.get('nmin', 0)).flatten()[0])
                nmax_ggm = int(np.asarray(mat['nmax']).flatten()[0])
                coeff_key = [k for k in mat_keys if k not in ('nmin', 'nmax')][0]
                return _load_from_3field(mat[coeff_key], nmin_ggm, nmax_ggm,
                                         nmax, GM_default, R_default)
            raise ValueError(
                f"Unsupported .mat structure: expected 1 or 3 variables, "
                f"found {len(mat_keys)} ({', '.join(mat_keys[:6])}).")

    try:
        import h5py
    except ImportError:
        raise ValueError(
            "'scipy' or 'h5py' is required to read .mat files. "
            "Install with: pip install scipy h5py") from None

    with h5py.File(path, 'r') as f:
        keys = [k for k in f.keys() if not k.startswith('#')]
        if len(keys) == 1:
            return _load_from_matrix(np.array(f[keys[0]]), nmax,
                                     GM_default, R_default)
        if len(keys) == 3:
            nmin_ggm = int(np.array(f['nmin']).flatten()[0])
            nmax_ggm = int(np.array(f['nmax']).flatten()[0])
            coeff_key = [k for k in keys if k not in ('nmin', 'nmax')][0]
            return _load_from_3field(np.array(f[coeff_key]), nmin_ggm,
                                     nmax_ggm, nmax, GM_default, R_default)
        raise ValueError(
            f"Unsupported .mat structure: expected 1 or 3 variables, "
            f"found {len(keys)}.")


def _load_txt(path, nmax, GM_default, R_default):
    """Load plain ASCII (4+ columns: n, m, Cnm, Snm)."""
    data = np.loadtxt(path)
    return _load_from_matrix(data, nmax, GM_default, R_default)


def _load_from_matrix(data, nmax, GM_default, R_default):
    """Process a raw (n, m, C, S) matrix."""
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError("GGM matrix must have >= 4 columns (n, m, C, S).")
    data = data[:, :4]
    data = data[data[:, 0].argsort()]  # sort by degree
    n_arr = data[:, 0].astype(int)
    m_arr = data[:, 1].astype(int)
    C_arr = data[:, 2]
    S_arr = data[:, 3]
    return _build_ggm(n_arr, m_arr, C_arr, S_arr, GM_default, R_default, nmax,
                      is_single=True)


def _load_from_3field(data, nmin_ggm, nmax_ggm, nmax, GM_default, R_default):
    """Process a 2-column (C, S) matrix with separate nmin/nmax."""
    # Truncate before rebuilding the (n, m) table: rebuilding degree 2190 and
    # then discarding half of it wastes ~2.4 M rows of index arrays.
    if nmax is not None:
        nmax_ggm = min(nmax_ggm, nmax)

    counts = np.arange(nmin_ggm, nmax_ggm + 1) + 1
    n_rows = int(counts.sum())
    if data.shape[0] < n_rows:
        raise ValueError(
            f"GGM matrix has {data.shape[0]} rows but degrees "
            f"{nmin_ggm}..{nmax_ggm} require {n_rows}.")
    data = data[:n_rows]
    C_arr = data[:, 0]
    S_arr = data[:, 1]

    # Triangular structure without a Python double loop.
    n_arr = np.repeat(np.arange(nmin_ggm, nmax_ggm + 1), counts)
    offsets = np.concatenate(([0], np.cumsum(counts)[:-1]))
    m_arr = np.arange(n_rows) - np.repeat(offsets, counts)
    return _build_ggm(n_arr, m_arr, C_arr, S_arr, GM_default, R_default, nmax,
                      is_single=False)


def _build_ggm(n_arr, m_arr, C_arr, S_arr, GM, R, nmax, is_single):
    """Common post-processing: validate, fill zeros, truncate, build index."""
    if n_arr.size == 0:
        raise ValueError("GGM contains no coefficients.")

    nmin_ggm = int(n_arr.min())

    if int(n_arr.max()) < 2:
        raise ValueError("GGM maximum degree must be at least 2.")

    # Insert missing degree-1 coefficients (zero padding)
    if is_single and nmin_ggm == 0 and n_arr.size > 1 and n_arr[1] == 2:
        # Degree 1, order 0 missing?
        mask_10 = (n_arr == 1) & (m_arr == 0)
        if not mask_10.any():
            n_arr = np.insert(n_arr, 1, 1)
            m_arr = np.insert(m_arr, 1, 0)
            C_arr = np.insert(C_arr, 1, 0.0)
            S_arr = np.insert(S_arr, 1, 0.0)
        # Degree 1, order 1 missing?
        mask_11 = (n_arr == 1) & (m_arr == 1)
        if not mask_11.any():
            idx = np.where((n_arr == 1) & (m_arr == 0))[0][0] + 1
            n_arr = np.insert(n_arr, idx, 1)
            m_arr = np.insert(m_arr, idx, 1)
            C_arr = np.insert(C_arr, idx, 0.0)
            S_arr = np.insert(S_arr, idx, 0.0)

    nmax_ggm = int(n_arr.max())
    if nmax is None:
        nmax = nmax_ggm
    if nmax > nmax_ggm:
        raise ValueError(f"Requested nmax={nmax} exceeds model nmax={nmax_ggm}.")
    if nmax < nmin_ggm:
        raise ValueError(
            f"Requested nmax={nmax} is below the model's minimum degree {nmin_ggm}.")

    # Truncate
    keep = n_arr <= nmax
    n_arr = n_arr[keep]
    m_arr = m_arr[keep]
    C_arr = C_arr[keep]
    S_arr = S_arr[keep]

    # Build index from the *actual* record counts per degree. The closed-form
    # triangle (idx[i] = idx[i-1] + nmin + i) silently mis-indexes files with
    # missing or duplicated orders, which would produce wrong results.
    counts = np.bincount(n_arr - nmin_ggm, minlength=nmax + 1 - nmin_ggm)
    idx = np.zeros(nmax + 1 - nmin_ggm, dtype=int)
    if idx.size > 1:
        idx[1:] = np.cumsum(counts[:-1])

    expected = np.arange(nmin_ggm, nmax + 1) + 1
    if counts.size == expected.size and not np.array_equal(counts, expected):
        warnings.warn(
            f"GGM coefficient table is not a complete triangle for degrees "
            f"{nmin_ggm}..{nmax}; degree {int(nmin_ggm + np.argmax(counts != expected))} "
            f"has {int(counts[np.argmax(counts != expected)])} records instead of "
            f"{int(expected[np.argmax(counts != expected)])}. Results may be unreliable.",
            RuntimeWarning,
            stacklevel=2,
        )

    model = GGMModel()
    model.GM = GM
    model.R = R
    model.nmin = nmin_ggm
    model.nmax = nmax
    model.n = n_arr
    model.m = m_arr
    model.C = C_arr
    model.S = S_arr
    model.index = idx
    return model


# -----------------------------------------------------------------------
#  DTM (Digital Terrain Model) file reader
# -----------------------------------------------------------------------

class DTMModel:
    """Container for DTM spherical harmonic coefficients.

    Attributes
    ----------
    nmax_dtm : int
    HC, HS   : ndarray – cosine / sine coefficients of topographic height
    """
    def __init__(self):
        self.nmax_dtm = 0
        self.HC = None
        self.HS = None


def load_dtm(path, nmax_ggm):
    """Load a DTM file (same format as GGM) and extract H coefficients.

    Parameters
    ----------
    path       : str – path to DTM file (.mat or plain text)
    nmax_ggm   : int – maximum degree of the GGM (DTM must cover this)
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.mat':
        try:
            import scipy.io as sio
            mat = sio.loadmat(path)
            keys = [k for k in mat.keys() if not k.startswith('__')]
            data = mat[keys[0]]
        except Exception:
            import h5py
            with h5py.File(path, 'r') as f:
                keys = list(f.keys())
                data = np.array(f[keys[0]])
    else:
        data = np.loadtxt(path)

    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError("DTM file must have >= 4 columns (n, m, HC, HS).")
    data = data[:, :4]

    nmax_dtm = int(data[:, 0].max())
    if nmax_dtm < nmax_ggm:
        raise ValueError(
            f"DTM max degree ({nmax_dtm}) < GGM nmax ({nmax_ggm}).")

    # Sort by degree if needed
    data = data[data[:, 0].argsort()]

    dtm = DTMModel()
    dtm.nmax_dtm = nmax_dtm
    dtm.HC = data[:, 2]
    dtm.HS = data[:, 3]
    return dtm


# -----------------------------------------------------------------------
#  Covariance matrix reader (for commission error)
# -----------------------------------------------------------------------

def load_covariance(path, nmax=None):
    """Load variance-covariance matrix of GGM coefficients.

    Supports a packed upper-triangular ASCII vector, a full square ASCII
    matrix, and MATLAB ``.mat`` files (v4/v5 via scipy, v7.3 via h5py).

    Parameters
    ----------
    path : str
    nmax : int or None
        Requested maximum degree. The matrix is ordered by fully normalised
        coefficients, so a matrix larger than ``(nmax + 1) ** 2`` is truncated
        to its leading block. ``None`` keeps the matrix unmodified.

    Returns
    -------
    covmat : ndarray – full symmetric covariance matrix
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.mat':
        try:
            import scipy.io as sio
        except ImportError:
            sio = None

        if sio is not None:
            try:
                mat = sio.loadmat(path)
            except NotImplementedError:
                mat = None  # MATLAB v7.3 (HDF5) — handled below
            except Exception as exc:
                raise ValueError(f"Cannot read MATLAB file '{path}': {exc}") from exc

            if mat is not None:
                keys = [k for k in mat.keys() if not k.startswith('__')]
                if not keys:
                    raise ValueError(f"No variables found in MATLAB file '{path}'.")
                covmat = np.array(mat[keys[0]], dtype=float)
                return _truncate_covariance(covmat, nmax, path)

        try:
            import h5py
        except ImportError:
            raise ValueError(
                "'scipy' or 'h5py' is required to read .mat files. "
                "Install with: pip install scipy h5py") from None

        with h5py.File(path, 'r') as f:
            keys = [k for k in f.keys() if not k.startswith('#')]
            if not keys:
                raise ValueError(f"No variables found in MATLAB file '{path}'.")
            covmat = np.array(f[keys[0]], dtype=float)
            return _truncate_covariance(covmat, nmax, path)

    # ASCII packed upper-triangular
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        # Single long vector — solve dim*(dim+1)/2 = n_elem exactly.
        n_elem = int(raw.size)
        dim = (math.isqrt(8 * n_elem + 1) - 1) // 2
        if dim * (dim + 1) // 2 != n_elem:
            raise ValueError(
                f"Cannot determine covariance matrix dimension from {n_elem} values.")
        # Vectorised symmetric expansion (was an O(dim^2) Python loop).
        covmat = np.zeros((dim, dim), dtype=float)
        iu = np.triu_indices(dim)
        covmat[iu] = raw
        covmat[(iu[1], iu[0])] = raw
        return _truncate_covariance(covmat, nmax, path)

    # A square file is already a full covariance matrix
    if raw.shape[0] == raw.shape[1]:
        return _truncate_covariance((raw + raw.T) / 2.0, nmax, path)

    raise ValueError(
        "Unsupported covariance layout: expected a packed upper-triangular "
        f"vector or a square matrix, got {raw.shape[0]}x{raw.shape[1]}. "
        "Convert the file to a single packed column or a full square matrix."
    )


def _truncate_covariance(covmat, nmax, path="<covariance>"):
    """Truncate a degree-ordered covariance matrix to the leading (nmax+1)^2 block."""
    if nmax is None:
        return covmat
    if covmat.ndim != 2 or covmat.shape[0] != covmat.shape[1]:
        raise ValueError(
            f"Covariance matrix from '{path}' is not square: shape {covmat.shape}.")

    dim = int(nmax) + 1
    limit = dim * dim
    if covmat.shape[0] <= limit:
        return covmat

    warnings.warn(
        f"Covariance matrix from '{path}' has dimension {covmat.shape[0]} but "
        f"nmax={nmax} implies {limit}; truncating to the leading {limit}x{limit} "
        "block on the assumption that coefficients are ordered by degree.",
        RuntimeWarning,
        stacklevel=3,
    )
    return covmat[:limit, :limit]
