"""
Build a truncated ICGEM model file from a full-resolution GGM.

The truncated file is written locally and is git-ignored, except for the
degree-180 subset that ships as ``GGM/EIGEN-6C4_n180.gfc`` (built with
``--label EIGEN-6C4_n180``) so that a fresh clone can run the example and the
tests without downloading a 250 MB model.

Only ``gfc``/``gfct`` records with degree <= ``--nmax`` are kept; the header is
copied verbatim except for ``max_degree``, which is rewritten so that
``examples/run_example.py`` reports the degree of the file it actually loads.

Usage
-----
    python tools/make_sample_ggm.py                        # EGM2008 -> degree 180
    python tools/make_sample_ggm.py --src GGM/EGM2008.gfc \\
        --out GGM/EGM2008_n180.gfc --nmax 180

The source file is never modified.
"""

import argparse
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)


def build_sample(src, out, nmax, overwrite=False, label=None):
    if not os.path.exists(src):
        sys.exit("Source model not found: %s" % src)
    if os.path.exists(out) and not overwrite:
        sys.exit("Output already exists: %s (pass --force to overwrite)" % out)

    kept = 0
    header_done = False

    out_dir = os.path.dirname(os.path.abspath(out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(src, "r", encoding="utf-8", errors="ignore") as fh, \
            open(out, "w", encoding="utf-8", newline="\n") as fo:
        for line in fh:
            if not header_done:
                stripped = line.strip()
                key = stripped.lower()
                if label and key.startswith("modelname"):
                    fo.write("modelname".ljust(25) + label + "\n")
                    continue
                if key.startswith("max_degree"):
                    parts = stripped.split()
                    if len(parts) == 2:
                        fo.write("max_degree".ljust(28) + "%d\n" % nmax)
                        continue
                fo.write(line)
                if stripped.lower().startswith("end_of_head"):
                    header_done = True
                continue

            parts = line.split()
            if not parts:
                continue
            tag = parts[0].lower()
            if tag in ("gfc", "gfct"):
                if len(parts) < 5:
                    continue
                if int(parts[1]) > nmax:
                    continue
                kept += 1
            elif tag in ("trnd", "acos", "asin"):
                # Trend / periodic terms are tied to the full expansion.
                continue
            fo.write(line)

    if kept == 0:
        sys.exit("No coefficients <= degree %d found in %s" % (nmax, src))

    print("wrote %s" % out)
    print("  degrees : 0..%d" % nmax)
    print("  records : %d" % kept)
    print("  size    : %.2f MB" % (os.path.getsize(out) / 1e6))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--src", default=os.path.join(REPO_ROOT, "GGM", "EGM2008.gfc"),
                    help="full ICGEM model (default: GGM/EGM2008.gfc)")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "GGM", "EGM2008_n180.gfc"),
                    help="output file (default: GGM/EGM2008_n180.gfc)")
    ap.add_argument("--nmax", type=int, default=180,
                    help="maximum degree to keep (default: 180)")
    ap.add_argument("--label", default=None,
                    help="rewrite the modelname field, e.g. EIGEN-6C4_n180")
    ap.add_argument("--force", action="store_true", help="overwrite the output file")
    args = ap.parse_args()
    build_sample(args.src, args.out, args.nmax, args.force, args.label)


if __name__ == "__main__":
    main()
