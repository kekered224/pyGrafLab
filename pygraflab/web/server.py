"""
pyGrafLab Web Server
====================
Flask backend wrapping the pyGrafLab spherical harmonic synthesis engine.
Provides REST API endpoints for model loading, computation, and result retrieval.

Usage:
    python -m pygraflab.web [--host HOST] [--port PORT] [--no-browser]
    pygraflab-web [--host HOST] [--port PORT]        # after pip install

Endpoints:
    GET  /api/health            Liveness probe
    GET  /api/status            Computation status
    GET  /api/model-info        Metadata of the loaded model
    GET  /api/functionals       List available functional codes
    POST /api/load-model        Load a GGM file by path
    POST /api/upload-model      Upload and load a GGM file
    POST /api/upload-covariance Upload a covariance matrix
    POST /api/compute           Run a synthesis computation
    GET  /api/results           Retrieve computation results
    GET  /api/outputs           List exported output files
    GET  /api/download/<name>   Download an output file
"""

import argparse
import math
import os
import sys
import threading
import time
import traceback
import webbrowser
from threading import Timer

import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory

# Resolve the working root.  In a source checkout this is the project root
# (the directory holding pyproject.toml); a wheel install has no such file, so
# runtime artefacts fall back to the current working directory.
PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def _work_root(start):
    cur = start
    while True:
        if os.path.isfile(os.path.join(cur, "pyproject.toml")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.getcwd()
        cur = parent


BASE_DIR = _work_root(PKG_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pygraflab
from pygraflab import FUNCTIONAL_NAMES

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
STATIC_DIR = os.path.join(PKG_DIR, "static")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
UPLOAD_DIR = os.path.join(OUTPUT_DIR, "uploads")

# Serve the UI assets from the site root (styles.css, app.js, plot.js).
# The page is returned at "/" and loads everything with relative URLs, so the
# default "/static" prefix would 404 every asset.
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB model uploads
app.config["JSON_SORT_KEYS"] = False

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_GRID_POINTS = 2_000_000      # hard cap on requested grid nodes
MAX_JSON_POINTS = 400_000        # results larger than this are download-only
ALLOWED_MODEL_EXT = (".gfc", ".mat", ".txt", ".dat", ".grx", ".bin")


# ---------------------------------------------------------------------------
# State (guarded by a lock: one computation at a time)
# ---------------------------------------------------------------------------
_LOCK = threading.RLock()
state = {
    "ggm": None,
    "ggm_info": None,
    "computing": False,
    "progress": 0.0,
    "status_msg": "Ready",
    "results": None,
    "lon": None,
    "lat": None,
    "commission_sigma": None,
    "params": None,
    "error": None,
    "elapsed": 0.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _json_safe(obj):
    """Recursively convert numpy objects and non-finite floats to JSON values."""
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


try:  # Flask >= 2.2
    app.json.default = lambda o: _json_safe(o)
    app.json.ensure_ascii = False
except Exception:  # pragma: no cover - older Flask
    app.config["JSON_AS_ASCII"] = False


def _model_info(ggm, filename="", path=""):
    """Extract serialisable metadata from a loaded GGMModel."""
    return {
        "filename": filename,
        "path": path,
        "GM": float(ggm.GM),
        "R": float(ggm.R),
        "nmin": int(ggm.nmin),
        "nmax": int(ggm.nmax),
    }


def _results_to_json(results, lon, lat, sigma):
    """Convert numpy results to a JSON-safe dict.

    For grid mode, flat arrays are reshaped to 2D (n_lat x n_lon) so the
    frontend can render them directly.
    """
    lon_arr = np.asarray(lon).ravel()
    lat_arr = np.asarray(lat).ravel()
    n_lon = lon_arr.size
    n_lat = lat_arr.size

    out = {
        "lon": lon_arr.tolist(),
        "lat": lat_arr.tolist(),
        "functionals": {},
    }
    for code, data in results.items():
        arr = data if isinstance(data, np.ndarray) else np.asarray(data)
        shape = list(arr.shape)

        if arr.ndim == 1 and arr.size == n_lat * n_lon and n_lat > 1 and n_lon > 1:
            arr = arr.reshape(n_lat, n_lon)
            shape = list(arr.shape)
        elif arr.ndim == 2 and arr.shape[0] == n_lat * n_lon and n_lat > 1 and n_lon > 1:
            arr = arr.reshape(n_lat, n_lon, arr.shape[1])
            shape = list(arr.shape)

        out["functionals"][str(code)] = {
            "name": FUNCTIONAL_NAMES.get(code, f"Code {code}"),
            "data": _json_safe(arr),
            "shape": shape,
        }
    if sigma:
        out["commission"] = {
            str(code): _json_safe(s) for code, s in sigma.items()
        }
    return out


def _secure_upload_name(filename, fallback="upload.gfc"):
    """Strip directories and unsafe characters from an uploaded file name."""
    base = os.path.basename(str(filename).replace("\\", "/")).strip()
    base = "".join(ch for ch in base if ch.isprintable() and ch not in '<>:"|?*')
    return base or fallback


def _save_upload(storage, allowed_ext=ALLOWED_MODEL_EXT):
    """Persist an uploaded file under output/uploads and return its path."""
    name = _secure_upload_name(storage.filename, "upload")
    if allowed_ext and not name.lower().endswith(allowed_ext):
        raise ValueError(
            f"Unsupported file type. Allowed: {', '.join(allowed_ext)}"
        )
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    target = os.path.join(UPLOAD_DIR, name)
    base, ext = os.path.splitext(name)
    counter = 1
    while os.path.exists(target):
        name = f"{base}_{counter}{ext}"
        target = os.path.join(UPLOAD_DIR, name)
        counter += 1
    storage.save(target)
    return target, name


def _parse_nmax(raw):
    """Parse an optional nmax value; returns None or a positive int."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    try:
        nmax = int(raw)
    except (TypeError, ValueError):
        raise ValueError("nmax must be an integer.") from None
    if nmax < 2:
        raise ValueError("nmax must be >= 2.")
    return nmax


def _origin_allowed():
    """Reject cross-site browser requests to the local API.

    A malicious web page must not be able to drive a server bound to
    localhost, so any Origin that is not this host is refused.
    """
    origin = request.headers.get("Origin")
    if not origin:
        return True
    host = request.host
    return origin in (f"http://{host}", f"https://{host}")


@app.before_request
def _guard_cross_site():
    if request.method in ("POST", "PUT", "DELETE") and not _origin_allowed():
        return jsonify({"error": "Cross-origin request refused"}), 403
    return None


# ---------------------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/health", methods=["GET"])
def api_health():
    with _LOCK:
        return jsonify({
            "status": "ok",
            "version": getattr(pygraflab, "__version__", "unknown"),
            "model_loaded": state["ggm"] is not None,
            "computing": state["computing"],
        })


# ---------------------------------------------------------------------------
# API: model info & loading
# ---------------------------------------------------------------------------
@app.route("/api/functionals", methods=["GET"])
def api_functionals():
    """Return the mapping of functional codes to names."""
    items = [{"code": k, "name": v} for k, v in sorted(FUNCTIONAL_NAMES.items())]
    return jsonify(items)


@app.route("/api/model-info", methods=["GET"])
def api_model_info():
    """Return info about the currently loaded model (if any)."""
    with _LOCK:
        if state["ggm"] is None:
            return jsonify({"loaded": False})
        return jsonify({"loaded": True, **state["ggm_info"]})


@app.route("/api/load-model", methods=["POST"])
def api_load_model():
    """Load a GGM file.

    JSON body: { "path": "...", "nmax": int (optional) }
    """
    data = request.get_json(silent=True) or {}
    path = str(data.get("path", "")).strip()
    if not path:
        return jsonify({"error": "path is required"}), 400

    try:
        nmax = _parse_nmax(data.get("nmax"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # A bare filename is looked up in GGM/ and the upload directory
    if not os.path.isabs(path) and os.sep not in path and "/" not in path:
        for root in (os.path.join(BASE_DIR, "GGM"), UPLOAD_DIR):
            candidate = os.path.join(root, path)
            if os.path.isfile(candidate):
                path = candidate
                break

    if not os.path.isfile(path):
        return jsonify({"error": f"File not found: {path}"}), 400

    with _LOCK:
        if state["computing"]:
            return jsonify({"error": "Cannot load a model while computing"}), 409
        try:
            model = pygraflab.load_ggm(path, nmax=nmax)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"error": f"Failed to load model: {exc}"}), 500
        state["ggm"] = model
        state["ggm_info"] = _model_info(model, os.path.basename(path), path)
        state["results"] = None
        state["error"] = None
        state["status_msg"] = f"Loaded {os.path.basename(path)}"
        return jsonify({"success": True, **state["ggm_info"]})


@app.route("/api/upload-model", methods=["POST"])
def api_upload_model():
    """Upload and load a GGM file via multipart form.

    Form fields: 'file' (required), 'nmax' (optional).
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    storage = request.files["file"]
    if not storage.filename:
        return jsonify({"error": "Empty filename"}), 400

    try:
        nmax = _parse_nmax(request.form.get("nmax", ""))
        save_path, name = _save_upload(storage)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    with _LOCK:
        if state["computing"]:
            return jsonify({"error": "Cannot load a model while computing"}), 409
        try:
            model = pygraflab.load_ggm(save_path, nmax=nmax)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"error": f"Failed to load model: {exc}"}), 500
        state["ggm"] = model
        state["ggm_info"] = _model_info(model, name, save_path)
        state["results"] = None
        state["error"] = None
        state["status_msg"] = f"Loaded {name}"
        return jsonify({"success": True, **state["ggm_info"]})


@app.route("/api/upload-covariance", methods=["POST"])
def api_upload_covariance():
    """Upload a covariance matrix file via multipart form.

    Returns the server-side path so it can be used in /api/compute.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    storage = request.files["file"]
    if not storage.filename:
        return jsonify({"error": "Empty filename"}), 400

    try:
        save_path, name = _save_upload(storage, allowed_ext=(".mat", ".txt", ".dat", ".cov"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # Validate the contents now so a malformed matrix fails here with a clear
    # message instead of surfacing later as a compute-time error.
    try:
        covmat = pygraflab.load_covariance(save_path)
    except Exception as exc:
        try:
            os.remove(save_path)
        except OSError:
            pass
        return jsonify({"error": f"Cannot parse covariance file: {exc}"}), 400

    if covmat.ndim != 2 or covmat.shape[0] != covmat.shape[1]:
        try:
            os.remove(save_path)
        except OSError:
            pass
        return jsonify(
            {"error": f"Covariance matrix must be square, got shape {covmat.shape}."}), 400

    return jsonify({
        "success": True,
        "path": save_path,
        "filename": name,
        "dimension": int(covmat.shape[0]),
    })


# ---------------------------------------------------------------------------
# API: computation
# ---------------------------------------------------------------------------
def _validate_compute_params(payload):
    """Validate and normalise a /api/compute request body."""
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    codes = payload.get("functional_codes") or []
    if not isinstance(codes, list) or not codes:
        raise ValueError("No functional codes specified.")
    try:
        codes = [int(c) for c in codes]
    except (TypeError, ValueError):
        raise ValueError("functional_codes must be integers.") from None
    unknown = [c for c in codes if c not in FUNCTIONAL_NAMES]
    if unknown:
        raise ValueError(f"Unknown functional code(s): {unknown}")

    mode = payload.get("mode", "grid")
    if mode not in ("grid", "pointwise"):
        raise ValueError("mode must be 'grid' or 'pointwise'.")

    params = {
        "functional_codes": codes,
        "mode": mode,
        "ellipsoid": payload.get("ellipsoid", 1),
        "method": int(payload.get("method", 1)),
        "coord_type": int(payload.get("coord_type", 0)),
    }
    if params["method"] not in (1, 2, 3):
        raise ValueError("method must be 1, 2 or 3.")
    if params["coord_type"] not in (0, 1):
        raise ValueError("coord_type must be 0 (ellipsoidal) or 1 (spherical).")

    if mode == "grid":
        for key in ("lon_min", "lon_max", "lon_step", "lat_min", "lat_max", "lat_step"):
            if payload.get(key) in (None, ""):
                raise ValueError(f"Grid mode requires '{key}'.")
            try:
                params[key] = float(payload[key])
            except (TypeError, ValueError):
                raise ValueError(f"'{key}' must be a number.") from None
            if not math.isfinite(params[key]):
                raise ValueError(f"'{key}' must be finite.")

        if params["lon_step"] <= 0 or params["lat_step"] <= 0:
            raise ValueError("Grid steps must be positive.")
        if params["lon_min"] >= params["lon_max"]:
            raise ValueError("lon_min must be smaller than lon_max.")
        if params["lat_min"] >= params["lat_max"]:
            raise ValueError("lat_min must be smaller than lat_max.")
        if not (-90.0 <= params["lat_min"] and params["lat_max"] <= 90.0):
            raise ValueError("Latitude bounds must lie within [-90, 90].")

        n_lon = int(round((params["lon_max"] - params["lon_min"]) / params["lon_step"])) + 1
        n_lat = int(round((params["lat_max"] - params["lat_min"]) / params["lat_step"])) + 1
        n_points = n_lon * n_lat
        if n_points > MAX_GRID_POINTS:
            raise ValueError(
                f"Grid too large: {n_lon} x {n_lat} = {n_points:,} points "
                f"(limit {MAX_GRID_POINTS:,}). Increase the step size or reduce the area."
            )

        try:
            h = float(payload.get("h", 0.0))
        except (TypeError, ValueError):
            raise ValueError("'h' must be a number.") from None
        if not math.isfinite(h) or not (-10_000.0 <= h <= 1_000_000.0):
            raise ValueError("'h' must be between -10 km and 1000 km.")
        params["h"] = h

    else:  # pointwise
        lon = payload.get("lon_deg") or []
        lat = payload.get("lat_deg") or []
        h_or_r = payload.get("h_or_r") or []
        if not isinstance(lon, list) or not isinstance(lat, list) or not isinstance(h_or_r, list):
            raise ValueError("lon_deg, lat_deg and h_or_r must be lists.")
        if not (len(lon) == len(lat) == len(h_or_r)) or not lon:
            raise ValueError("lon_deg, lat_deg and h_or_r must have the same non-zero length.")
        if len(lon) > MAX_GRID_POINTS:
            raise ValueError(f"Too many points (limit {MAX_GRID_POINTS:,}).")
        try:
            params["lon_deg"] = [float(v) for v in lon]
            params["lat_deg"] = [float(v) for v in lat]
            params["h_or_r"] = [float(v) for v in h_or_r]
        except (TypeError, ValueError):
            raise ValueError("Point coordinates must be numbers.") from None
        for label, values in (("lon_deg", params["lon_deg"]), ("lat_deg", params["lat_deg"])):
            if any(not math.isfinite(v) for v in values):
                raise ValueError(f"'{label}' contains non-finite values.")
        if any(abs(v) > 90.0 for v in params["lat_deg"]):
            raise ValueError("Latitudes must lie within [-90, 90].")
        if params["coord_type"] == 1 and any(v <= 0 for v in params["h_or_r"]):
            raise ValueError("With spherical coordinates h_or_r is a radius and must be > 0.")

    nmin = payload.get("nmin")
    if nmin not in (None, ""):
        try:
            params["nmin"] = int(nmin)
        except (TypeError, ValueError):
            raise ValueError("nmin must be an integer.") from None
        if params["nmin"] < 0:
            raise ValueError("nmin must be >= 0.")

    params["commission"] = bool(payload.get("commission", False))
    if params["commission"]:
        cov_path = str(payload.get("covmat_path", "") or "").strip()
        if cov_path:
            if not os.path.isfile(cov_path):
                raise ValueError(f"Covariance file not found: {cov_path}")
            params["covmat_path"] = cov_path
            try:
                nmax_cov = _parse_nmax(payload.get("nmax_cov"))
            except ValueError as exc:
                raise ValueError(f"nmax_cov: {exc}") from None
            if nmax_cov:
                params["nmax_cov"] = nmax_cov

        # GrafLab rule: commission errors are propagated with the standard
        # forward column method only, whatever recursion is selected for the
        # functionals themselves.
        if params["method"] != 1:
            params["method"] = 1
            params["method_notice"] = (
                "Commission error is always evaluated with the standard "
                "forward column method."
            )

    return params


@app.route("/api/compute", methods=["POST"])
def api_compute():
    """Start a synthesis computation (runs in a background thread)."""
    with _LOCK:
        if state["computing"]:
            return jsonify({"error": "A computation is already running"}), 409
        if state["ggm"] is None:
            return jsonify({"error": "No GGM model loaded"}), 400

    try:
        params = _validate_compute_params(request.get_json(silent=True))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    with _LOCK:
        if state["computing"]:  # re-check: another request won the race
            return jsonify({"error": "A computation is already running"}), 409
        model = state["ggm"]

        kwargs = {k: v for k, v in params.items()
                  if k not in ("covmat_path", "method_notice")}
        kwargs["ggm"] = model
        if params.get("commission") and params.get("covmat_path"):
            try:
                kwargs["covmat"] = pygraflab.load_covariance(params["covmat_path"])
            except Exception as exc:
                return jsonify({"error": f"Failed to read covariance: {exc}"}), 400

        state["computing"] = True
        state["progress"] = 0.0
        state["status_msg"] = "Computing…"
        state["error"] = None
        state["results"] = None
        state["elapsed"] = 0.0

    mode = params["mode"]

    def _worker():
        t0 = time.time()
        try:
            results, lon, lat, sigma = pygraflab.compute(**kwargs)
            elapsed = time.time() - t0

            # Export to disk first so results survive a JSON size limit
            try:
                if mode == "grid":
                    pygraflab.export_grid_txt(results, lon, lat, OUTPUT_DIR)
                else:
                    pygraflab.export_txt(results, OUTPUT_DIR)
            except Exception:
                traceback.print_exc()

            # Grid mode returns longitude/latitude *vectors*, so the payload
            # size is nodes x functionals, not the length of either vector.
            lon_arr = np.asarray(lon)
            lat_arr = np.asarray(lat)
            if mode == "grid":
                n_points = int(lon_arr.size * lat_arr.size)
            else:
                n_points = int(lon_arr.size)
            n_values = n_points * max(len(results), 1)

            if n_values > MAX_JSON_POINTS:
                with _LOCK:
                    state["results"] = None
                    state["lon"] = None
                    state["lat"] = None
                    state["commission_sigma"] = None
                    state["elapsed"] = elapsed
                    state["progress"] = 1.0
                    state["error"] = (
                        f"Result has {n_points:,} grid nodes x "
                        f"{len(results)} functional(s) = {n_values:,} values, "
                        f"above the {MAX_JSON_POINTS:,}-value browser limit. "
                        "Download the exported .txt files instead."
                    )
                    state["status_msg"] = "Complete — result too large to display"
                return

            with _LOCK:
                state["results"] = results
                state["lon"] = lon
                state["lat"] = lat
                state["commission_sigma"] = sigma
                state["params"] = {
                    k: v for k, v in params.items()
                    if k not in ("covmat", "method_notice")
                }
                state["elapsed"] = elapsed
                state["progress"] = 1.0
                state["status_msg"] = f"Complete — {elapsed:.2f}s"
        except Exception as exc:
            traceback.print_exc()
            with _LOCK:
                state["error"] = str(exc)
                state["status_msg"] = f"Error: {exc}"
        finally:
            with _LOCK:
                state["computing"] = False

    threading.Thread(target=_worker, name="pygraflab-compute", daemon=True).start()
    reply = {"success": True, "message": "Computation started"}
    if params.get("method_notice"):
        reply["notice"] = params["method_notice"]
    return jsonify(reply), 202


@app.route("/api/status", methods=["GET"])
def api_status():
    """Poll computation status."""
    with _LOCK:
        return jsonify({
            "computing": state["computing"],
            "progress": state["progress"],
            "status": state["status_msg"],
            "error": state["error"],
            "elapsed": state["elapsed"],
            "has_results": state["results"] is not None,
        })


@app.route("/api/results", methods=["GET"])
def api_results():
    """Return computation results as JSON."""
    with _LOCK:
        results = state["results"]
        if results is None:
            return jsonify({"error": "No results available"}), 404
        payload = _results_to_json(
            results, state["lon"], state["lat"], state["commission_sigma"]
        )
        payload["elapsed"] = state["elapsed"]
        payload["params"] = state["params"]
    return jsonify(_json_safe(payload))


# ---------------------------------------------------------------------------
# API: file downloads
# ---------------------------------------------------------------------------
@app.route("/api/outputs", methods=["GET"])
@app.route("/api/download-all", methods=["GET"], endpoint="api_download_all_legacy")
def api_download_all():
    """List all available output files."""
    files = []
    if os.path.isdir(OUTPUT_DIR):
        for name in sorted(os.listdir(OUTPUT_DIR)):
            if name.startswith("."):
                continue
            path = os.path.join(OUTPUT_DIR, name)
            if os.path.isfile(path):
                files.append({"name": name, "size": os.path.getsize(path)})
    return jsonify(files)


@app.route("/api/download/<path:filename>", methods=["GET"])
def api_download(filename):
    """Download an output file by name (confined to output/)."""
    safe = _secure_upload_name(filename, "")
    if not safe:
        return jsonify({"error": "Invalid file name"}), 400
    path = os.path.realpath(os.path.join(OUTPUT_DIR, safe))
    if os.path.commonpath([path, os.path.realpath(OUTPUT_DIR)]) != os.path.realpath(OUTPUT_DIR):
        return jsonify({"error": "Invalid file name"}), 400
    if not os.path.isfile(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, as_attachment=True, download_name=safe)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="pyGrafLab Web Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=5002, help="Port number")
    parser.add_argument("--debug", action="store_true",
                        help="Enable the Werkzeug debugger (development only)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open a browser window")
    args = parser.parse_args(argv)

    url = f"http://{args.host}:{args.port}"
    print(f"\n  pyGrafLab Web Server\n  {url}\n")

    if not args.no_browser and not os.environ.get("WERKZEUG_RUN_MAIN"):
        Timer(1.5, lambda: webbrowser.open(url)).start()

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True,
        exclude_patterns=["output/*", "output/uploads/*"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
