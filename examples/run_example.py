"""
pyGrafLab usage example -- spherical harmonic synthesis with EGM2008.

Demonstrates:
    1. Loading a GGM file (ICGEM .gfc format)
    2. Grid synthesis for gravity disturbance and geoid undulation
    3. Point-wise synthesis at scattered stations
    4. The high-level ``compute()`` interface
    5. Report generation and .mat export

The model file is discovered automatically, in this order:

    1. ``--ggm PATH``
    2. ``$PYGRAFLAB_GGM``
    3. ``GGM/*.gfc``            -- any model you downloaded from ICGEM

No download is required: a degree-180 EIGEN-6C4 subset ships in ``GGM/`` and
is picked up automatically. Point ``--ggm`` at a full model for production
work.

The example evaluates up to degree 360 by default so that it finishes in a
few seconds.  Higher degrees are honoured with ``--nmax`` (e.g. ``--nmax 2190``
with the full model), at a cost that grows roughly with ``nmax**3``:

    python examples/run_example.py --nmax 720
"""

import argparse
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
GGM_DIR = os.path.join(REPO_ROOT, "GGM")
DEFAULT_NMAX = 360

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pygraflab                              # noqa: E402


def find_ggm(explicit):
    """Return the model path to use, or None if nothing is available."""
    if explicit:
        return explicit
    candidates = [os.environ.get("PYGRAFLAB_GGM"),
                  os.path.join(GGM_DIR, "EGM2008.gfc")]
    for cand in candidates:
        if cand and os.path.exists(cand):
            return cand
    if os.path.isdir(GGM_DIR):
        found = sorted(f for f in os.listdir(GGM_DIR) if f.lower().endswith(".gfc"))
        if found:
            return os.path.join(GGM_DIR, found[0])
    return None


def peek_max_degree(path):
    """Read the max_degree field from an ICGEM header without loading the model."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for _ in range(200):
                line = fh.readline()
                if not line:
                    break
                parts = line.split()
                if len(parts) == 2 and parts[0].lower() == "max_degree":
                    return int(parts[1])
    except (OSError, ValueError):
        pass
    return None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--ggm", default=None,
                   help="path to an ICGEM .gfc model file "
                        "(default: auto-discover under GGM/)")
    p.add_argument("--nmax", type=int, default=None,
                   help="maximum degree to evaluate "
                        "(default: %d, or the model's own nmax if lower)" % DEFAULT_NMAX)
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "output"),
                   help="output directory")
    return p.parse_args()


def main():
    args = parse_args()
    ggm_path = find_ggm(args.ggm)
    out_dir = args.out

    if not ggm_path:
        sys.exit("No GGM file found under %s.\n"
                 "Download a model from https://icgem.gfz.de/ (Table of Models)\n"
                 "and pass its path with --ggm." % GGM_DIR)
    if not os.path.exists(ggm_path):
        sys.exit("GGM file not found: %s" % ggm_path)
    os.makedirs(out_dir, exist_ok=True)

    model_nmax = peek_max_degree(ggm_path)
    nmax = args.nmax if args.nmax else min(model_nmax or DEFAULT_NMAX, DEFAULT_NMAX)

    # -- 1. Load the model --------------------------------------------------
    print("Loading GGM: %s" % ggm_path)
    if model_nmax and nmax < model_nmax:
        print("  model degree %d, evaluating to %d "
              "(pass --nmax %d for the full model)"
              % (model_nmax, nmax, model_nmax))
    print("  requested nmax = %d" % nmax)
    ggm = pygraflab.load_ggm(ggm_path, nmax=nmax)
    print("  GM = %.6e m^3/s^2" % ggm.GM)
    print("  R  = %.3f m" % ggm.R)
    print("  degrees: %d - %d" % (ggm.nmin, ggm.nmax))
    print("  coefficients: %d\n" % len(ggm.C))

    # -- 2. Grid synthesis --------------------------------------------------
    print("=== Grid synthesis ===")
    print("  region : 30-40 N, 100-110 E, 1 deg step")
    print("  height : 0 m (ellipsoid surface)")
    print("  codes  : 11 (potential V), 19 (gravity anomaly), 10 (geoid N)")
    t0 = time.time()
    results_grid, lon_vec, lat_vec = pygraflab.synthesize_grid(
        ggm,
        functional_codes=[11, 19, 10],
        ellipsoid_spec=1,            # GRS80
        lon_min=100.0, lon_step=1.0, lon_max=110.0,
        lat_min=30.0, lat_step=1.0, lat_max=40.0,
        h=0.0,
        method=1,
    )
    elapsed_grid = time.time() - t0
    print("  completed in %.2f s" % elapsed_grid)
    for code, arr in sorted(results_grid.items()):
        name = pygraflab.FUNCTIONAL_NAMES.get(code, "?")
        print("  [%2d] %-38s shape=%s  min=%.6e  max=%.6e"
              % (code, name, arr.shape, arr.min(), arr.max()))
    print()

    grid_dir = os.path.join(out_dir, "grid")
    os.makedirs(grid_dir, exist_ok=True)
    pygraflab.export_grid_txt(results_grid, lon_vec, lat_vec, grid_dir)
    print("  grid results exported to %s\n" % grid_dir)

    # -- 3. Point-wise synthesis -------------------------------------------
    print("=== Point-wise synthesis ===")
    stations_lon = [-105.0, -104.0, -106.0, -105.5, -104.5]
    stations_lat = [37.0, 38.0, 39.0, 40.0, 41.0]
    stations_h = [1600.0, 1800.0, 2000.0, 1500.0, 1700.0]
    print("  %d stations, codes: 5 (T), 20 (dg), 2/3 (deflections)"
          % len(stations_lon))

    t0 = time.time()
    results_pt, lon_pt, lat_pt = pygraflab.synthesize_pointwise(
        ggm,
        functional_codes=[5, 20, 2, 3],
        ellipsoid_spec=1,
        lon_deg=stations_lon,
        lat_deg=stations_lat,
        h_or_r=stations_h,
        coord_type=0,               # ellipsoidal heights
        method=1,
    )
    elapsed_pt = time.time() - t0
    print("  completed in %.2f s" % elapsed_pt)
    for code, arr in sorted(results_pt.items()):
        name = pygraflab.FUNCTIONAL_NAMES.get(code, "?")
        print("  [%2d] %-38s shape=%s" % (code, name, arr.shape))
    print()

    pt_dir = os.path.join(out_dir, "pointwise")
    os.makedirs(pt_dir, exist_ok=True)
    pygraflab.export_txt(results_pt, pt_dir)
    print("  point-wise results exported to %s\n" % pt_dir)

    # -- 4. High-level compute() interface ---------------------------------
    print("=== compute() interface ===")
    results_hl, lon_hl, lat_hl, _comm = pygraflab.compute(
        ggm,
        functional_codes=[11, 19],
        mode="grid",
        ellipsoid=1,
        lon_min=105.0, lon_max=106.0, lon_step=0.5,
        lat_min=35.0, lat_max=36.0, lat_step=0.5,
        h=0.0,
        method=1,
    )
    print("  elapsed: %.2f s" % pygraflab.compute._last_elapsed)
    for code, arr in sorted(results_hl.items()):
        print("  [%2d] %-38s shape=%s"
              % (code, pygraflab.FUNCTIONAL_NAMES.get(code, "?"), arr.shape))
    print()

    # -- 5. Report ----------------------------------------------------------
    report_path = os.path.join(out_dir, "report.txt")
    params = {
        "ggm_file": ggm_path,
        "ggm_nmax": ggm.nmax,
        "ggm_GM": ggm.GM,
        "ggm_R": ggm.R,
        "ellipsoid": "GRS80",
        "functional_codes": [11, 19, 10, 5, 20, 2, 3],
        "mode": "grid + pointwise",
        "method": 1,
        "lon_min": 100.0, "lon_max": 110.0, "lon_step": 1.0,
        "lat_min": 30.0, "lat_max": 40.0, "lat_step": 1.0,
        "h": 0.0,
        "elapsed_seconds": elapsed_grid + elapsed_pt,
    }
    pygraflab.generate_report(params, report_path)
    print("report saved to %s" % report_path)

    # -- 6. .mat export -----------------------------------------------------
    mat_path = os.path.join(out_dir, "results.mat")
    try:
        pygraflab.export_mat(results_grid, mat_path)
        print(".mat file saved to %s" % mat_path)
    except ImportError:
        print("scipy.io not available -- skipping .mat export")

    print("\nDone.")


if __name__ == "__main__":
    main()
