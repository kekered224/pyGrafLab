"""Smoke tests for a fresh source checkout.

The numerical cases run against the degree-180 EIGEN-6C4 subset that ships in
``GGM/``, or against any other ``.gfc`` found there, and are skipped only when
the directory holds no model at all. The public API and the web server's
cross-site guard are always checked; full validation against MATLAB GrafLab
lives in the reference comparison reports.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pygraflab  # noqa: E402

GGM_DIR = os.path.join(ROOT, "GGM")
BUNDLED = os.path.join(GGM_DIR, "EIGEN-6C4_n180.gfc")
GGM = BUNDLED if os.path.isfile(BUNDLED) else os.path.join(GGM_DIR, "EGM2008.gfc")
if not os.path.isfile(GGM) and os.path.isdir(GGM_DIR):
    found = sorted(f for f in os.listdir(GGM_DIR) if f.lower().endswith(".gfc"))
    if found:
        GGM = os.path.join(GGM_DIR, found[0])
HAS_GGM = os.path.isfile(GGM)


def test_public_api():
    assert isinstance(pygraflab.__version__, str)
    assert 5 in pygraflab.FUNCTIONAL_NAMES    # disturbing potential
    assert 20 in pygraflab.FUNCTIONAL_NAMES   # gravity disturbance
    for name in ("load_ggm", "compute", "synthesize_grid",
                 "synthesize_pointwise", "export_txt"):
        assert callable(getattr(pygraflab, name))


@pytest.mark.skipif(not HAS_GGM, reason="no local .gfc model (see README.md)")
def test_pointwise_synthesis():
    ggm = pygraflab.load_ggm(GGM, nmax=12)
    results, lon, lat, sigma = pygraflab.compute(
        ggm, functional_codes=[5, 20], mode="pointwise", ellipsoid=1,
        lon_deg=[116.4, 121.5], lat_deg=[39.9, 31.2],
        h_or_r=[50.0, 20.0], method=1,
    )
    assert set(results) == {5, 20}
    for arr in results.values():
        assert np.asarray(arr).shape == (2,)
        assert np.all(np.isfinite(arr))
    assert np.allclose(lon, [116.4, 121.5])
    assert np.allclose(lat, [39.9, 31.2])
    assert sigma is None


@pytest.mark.skipif(not HAS_GGM, reason="no local .gfc model (see README.md)")
def test_grid_synthesis():
    ggm = pygraflab.load_ggm(GGM, nmax=12)
    results, lon, lat, sigma = pygraflab.compute(
        ggm, functional_codes=[20], mode="grid", ellipsoid=1,
        lon_min=100, lon_max=102, lon_step=1,
        lat_min=30, lat_max=32, lat_step=1, h=0.0, method=1,
    )
    arr = np.asarray(results[20])
    assert arr.size == 9
    assert np.all(np.isfinite(arr))
    assert len(lon) == 3 and len(lat) == 3
    assert sigma is None


def test_web_cross_site_guard():
    pytest.importorskip("flask")
    from pygraflab.web import server

    client = server.app.test_client()
    assert client.get("/api/health").status_code == 200

    # A cross-site POST is rejected before routing.
    bad = client.post("/api/__probe__", headers={"Origin": "https://evil.example"})
    assert bad.status_code == 403

    # A same-origin POST passes the guard (route simply does not exist).
    good = client.post("/api/__probe__", headers={"Origin": "http://localhost"})
    assert good.status_code != 403


def test_web_commission_pins_standard_method():
    """Commission errors always use the standard forward column method."""
    pytest.importorskip("flask")
    from pygraflab.web import server

    params = server._validate_compute_params({
        "functional_codes": [19],
        "mode": "pointwise",
        "lon_deg": [10.0],
        "lat_deg": [45.0],
        "h_or_r": [0.0],
        "method": 3,
        "commission": True,
    })
    assert params["method"] == 1
    assert "standard forward column" in params["method_notice"]

    # Without commission error the selection is left untouched.
    plain = server._validate_compute_params({
        "functional_codes": [19],
        "mode": "pointwise",
        "lon_deg": [10.0],
        "lat_deg": [45.0],
        "h_or_r": [0.0],
        "method": 3,
    })
    assert plain["method"] == 3
    assert "method_notice" not in plain


def test_fnalfs_method2_high_degree():
    """Regression: the modified forward column method must survive m > 470.

    Its sectorial seed carries a 1e-280 scale factor and drops the ``u**m``
    factor, which is re-applied afterwards (GrafLab's Horner scheme). Keeping
    ``u**m`` in the seed makes ``u**m * 1e-280`` underflow to zero from
    m ~ 470 upwards, silently truncating the series.
    """
    from pygraflab.fnalfs import fnalfs_standard, fnalfs_modified

    nmax = 600
    fiG = np.array([np.deg2rad(36.6)])
    q = np.array([6378136.46 / (6378136.46 + 1627.0)])
    u = np.cos(fiG)

    for m in (0, 100, 470, 600):
        ref = fnalfs_standard(fiG, q, nmax, m)
        got = fnalfs_modified(fiG, q, nmax, m) * 1e280 * (u ** m)[:, None]
        assert np.max(np.abs(got)) > 0.0, f"fnALFs underflowed to zero at m={m}"
        assert np.max(np.abs(got - ref)) <= 1e-12 * np.max(np.abs(ref)), f"m={m}"


@pytest.mark.skipif(not HAS_GGM, reason="no local .gfc model (see README.md)")
def test_method_agreement():
    """Methods 1 and 2 must give the same functionals."""
    ggm = pygraflab.load_ggm(GGM, nmax=180)
    kwargs = dict(functional_codes=[5, 20], mode="pointwise", ellipsoid=1,
                  lon_deg=[116.4, 121.5], lat_deg=[39.9, 31.2],
                  h_or_r=[0.0, 0.0])
    r1 = pygraflab.compute(ggm, method=1, **kwargs)[0]
    r2 = pygraflab.compute(ggm, method=2, **kwargs)[0]
    for code in (5, 20):
        a = np.asarray(r1[code], dtype=float)
        b = np.asarray(r2[code], dtype=float)
        assert np.all(np.isfinite(b))
        assert np.allclose(b, a, rtol=0.0, atol=1e-9 * max(1.0, np.max(np.abs(a))))
