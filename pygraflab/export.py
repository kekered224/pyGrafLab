"""
Data export and report generation for pyGrafLab.

Supports:
- Plain-text export (12-digit scientific notation)
- MATLAB .mat export (via scipy.io or h5py)
- Human-readable computation report

Ported from GrafLab 2.1.4 (lines 7140-9700).
"""

import os
import time
import numpy as np

from .synthesis import FUNCTIONAL_NAMES


# ===================================================================
#  Text export
# ===================================================================

def export_txt(data, path, fmt='%.12e'):
    """Export synthesis results to plain-text files.

    Parameters
    ----------
    data : dict
        Keys are functional codes (int), values are ndarray.
        For grid mode: shape (n_lat, n_lon) or (n_lat, n_lon, n_comp).
        For pointwise: shape (n_pts,) or (n_pts, n_comp).
    path : str
        Output directory (must exist).
    fmt : str
        Numeric format (default 12-digit scientific).
    """
    path = str(path)
    os.makedirs(path, exist_ok=True)
    for code, arr in sorted(data.items()):
        name = FUNCTIONAL_NAMES.get(code, f'functional_{code}')
        fname = os.path.join(path, f'{name}.txt')
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            np.savetxt(fname, arr, fmt=fmt)
        elif arr.ndim == 2:
            np.savetxt(fname, arr, fmt=fmt)
        elif arr.ndim == 3:
            # Grid with components: flatten lat/lon and stack
            n_lat, n_lon, n_comp = arr.shape
            flat = arr.reshape(-1, n_comp)
            np.savetxt(fname, flat, fmt=fmt)
        else:
            np.savetxt(fname, arr.reshape(-1), fmt=fmt)


def export_grid_txt(results, lon_vec, lat_vec, path, fmt='%.12e'):
    """Export grid results with coordinate columns prepended.

    Each output file has columns: lon, lat, value(s).

    Parameters
    ----------
    results : dict – functional code -> ndarray (n_lat, n_lon) or (n_lat, n_lon, n_comp)
    lon_vec : ndarray – longitude vector (degrees)
    lat_vec : ndarray – latitude vector (degrees)
    path    : str – output directory
    fmt     : str – numeric format
    """
    path = str(path)
    os.makedirs(path, exist_ok=True)
    lat_grid, lon_grid = np.meshgrid(lat_vec, lon_vec, indexing='ij')
    coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    n_grid = len(lat_vec) * len(lon_vec)

    for code, arr in sorted(results.items()):
        name = FUNCTIONAL_NAMES.get(code, f'functional_{code}')
        fname = os.path.join(path, f'{name}.txt')
        arr = np.asarray(arr, dtype=float)
        # Handle both flat and grid-shaped arrays
        if arr.size == n_grid and arr.ndim <= 2:
            values = arr.ravel()[:, np.newaxis]
        elif arr.ndim == 2 and arr.shape[0] == n_grid:
            values = arr
        elif arr.ndim == 3:
            values = arr.reshape(-1, arr.shape[2])
        else:
            values = arr.ravel()[:, np.newaxis]
        out = np.hstack([coords, values])
        np.savetxt(fname, out, fmt=fmt)


# ===================================================================
#  MATLAB .mat export
# ===================================================================

def export_mat(data, path, compress=True):
    """Export synthesis results to a MATLAB .mat file.

    Parameters
    ----------
    data : dict
        Keys are functional codes (int), values are ndarray.
    path : str
        Output file path (.mat).
    compress : bool
        Use compression (scipy.io.savemat default).
    """
    path = str(path)
    mdict = {}
    for code, arr in sorted(data.items()):
        name = FUNCTIONAL_NAMES.get(code, f'functional_{code}')
        mdict[name] = np.asarray(arr, dtype=float)

    try:
        import scipy.io as sio
        sio.savemat(path, mdict, do_compression=compress)
    except ImportError:
        # Fallback: use h5py for HDF5-based .mat v7.3
        import h5py
        with h5py.File(path, 'w') as f:
            for key, val in mdict.items():
                f.create_dataset(key, data=val)


# ===================================================================
#  Report generation
# ===================================================================

def generate_report(params, output_path):
    """Generate a human-readable computation report.

    Parameters
    ----------
    params : dict
        Computation parameters. Expected keys:
        - ggm_file       : str
        - ggm_nmax       : int
        - ggm_GM         : float
        - ggm_R          : float
        - ellipsoid      : str
        - functional_codes: list of int
        - mode           : 'grid' or 'pointwise'
        - method         : int (1, 2, or 3)
        - nmin           : int (optional)
        - For grid: lon_min, lon_max, lon_step, lat_min, lat_max, lat_step, h
        - For pointwise: n_points
        - dtm_file       : str (optional)
        - commission     : bool
        - elapsed_seconds: float
    output_path : str
        Output file path (.txt or .log).
    """
    lines = []
    lines.append('=' * 70)
    lines.append('  pyGrafLab Computation Report')
    lines.append(f'  Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('=' * 70)
    lines.append('')

    # GGM info
    lines.append('--- Global Geopotential Model ---')
    lines.append(f'  File:       {params.get("ggm_file", "N/A")}')
    lines.append(f'  Max degree: {params.get("ggm_nmax", "N/A")}')
    lines.append(f'  GM:         {params.get("ggm_GM", "N/A"):.6e} m^3/s^2')
    lines.append(f'  R:          {params.get("ggm_R", "N/A"):.3f} m')
    lines.append('')

    # Reference ellipsoid
    lines.append('--- Reference Ellipsoid ---')
    lines.append(f'  Ellipsoid:  {params.get("ellipsoid", "N/A")}')
    lines.append('')

    # Computation settings
    lines.append('--- Computation Settings ---')
    lines.append(f'  Mode:       {params.get("mode", "N/A")}')
    lines.append(f'  Method:     {params.get("method", 1)} (fnALF computation)')
    nmin = params.get('nmin', None)
    if nmin is not None:
        lines.append(f'  Min degree: {nmin}')
    lines.append('')

    # Grid or pointwise specifics
    mode = params.get('mode', '')
    if mode == 'grid':
        lines.append('--- Grid Parameters ---')
        lines.append(f'  Longitude:  [{params.get("lon_min"):.4f}, {params.get("lon_max"):.4f}] deg, step {params.get("lon_step"):.4f} deg')
        lines.append(f'  Latitude:   [{params.get("lat_min"):.4f}, {params.get("lat_max"):.4f}] deg, step {params.get("lat_step"):.4f} deg')
        lines.append(f'  Height:     {params.get("h", 0):.3f} m')
    elif mode == 'pointwise':
        lines.append('--- Pointwise Parameters ---')
        lines.append(f'  Points:     {params.get("n_points", "N/A")}')
    lines.append('')

    # DTM
    dtm_file = params.get('dtm_file')
    if dtm_file:
        lines.append('--- Digital Terrain Model ---')
        lines.append(f'  File:       {dtm_file}')
        lines.append('')

    # Functionals
    codes = params.get('functional_codes', [])
    lines.append(f'--- Computed Functionals ({len(codes)}) ---')
    for code in sorted(codes):
        name = FUNCTIONAL_NAMES.get(code, f'Unknown ({code})')
        lines.append(f'  [{code:2d}] {name}')
    lines.append('')

    # Commission error
    if params.get('commission', False):
        lines.append('--- Commission Error ---')
        lines.append('  Enabled (covariance propagation)')
        lines.append('')

    # Timing
    elapsed = params.get('elapsed_seconds', None)
    if elapsed is not None:
        lines.append('--- Performance ---')
        lines.append(f'  Elapsed:    {elapsed:.2f} s')
        lines.append('')

    lines.append('=' * 70)
    lines.append('  End of report')
    lines.append('=' * 70)

    text = '\n'.join(lines)
    with open(str(output_path), 'w', encoding='utf-8') as f:
        f.write(text)
    return text
