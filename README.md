# pyGrafLab

**Python port of the GrafLab 2.1.4 spherical harmonic synthesis engine.**

`pyGrafLab` reproduces the calculation engine of GrafLab 2.1.4 (Bucha & Janák, 2013)
in pure Python (numpy / scipy); the algorithm and its formulations are the work of
the original authors, this project only re-implements them. It evaluates 24
geopotential functionals — gravity,
gravity disturbances and anomalies, deflections of the vertical, geoid undulation,
height anomalies, and the full Marussi tensor in local north-oriented and
spherical frames — from any ICGEM-format global geopotential model (GGM), on
regular grids or at scattered points, with three alternative associated
Legendre function (fnALF) recursion schemes and optional commission-error
propagation.

The project ships with a **zero-dependency web interface**: a Flask backend and a
plain HTML/CSS/JS front end that renders fields on a local `<canvas>` element.
No CDN, no Plotly, no web fonts — the UI works fully offline.

---

## Features

- **24 functional codes** (2–25) covering potential, gravity vector, Marussi
  tensor (LNOF and spherical), deflections, geoid undulation, height anomaly and
  their disturbing/actual variants.
- **Three fnALF methods** — standard (1), modified (2), extended-range (3) —
  selectable per run and used within GrafLab's stated ranges: method 1 up to
  degree 1800 at all latitudes (and 2190 between 0–56° and 80–90°), method 2
  from 1801 to 2700, method 3 at any degree and latitude.
- **Grid and point-wise synthesis** modes.
- **Ellipsoidal (GRS80 / WGS84) and spherical** coordinate handling, with the
  ellipsoid coefficients (`GM`, `a`, `e`, `C20`, `ω`) either taken from the
  built-in constants or supplied explicitly.
- **DTM support** for geoid undulation and height anomaly (Helmert vs.
  Heiskanen–Moritz height-anomaly variants).
- **Commission error propagation** from a coefficient covariance matrix
  (point-wise mode).
- **GFC / MAT model loading**, including h5py-based MAT v7.3 support and
  non-complete triangular coefficient files.
- **Export** to plain text, MATLAB `.mat`, and a human-readable computation
  report.
- **Offline web UI** with a self-contained canvas field plotter: diverging and
  sequential colour ramps, contour lines, hover read-out, colour bar and
  auto-scaled axes.

---

## Installation

### Requirements

- Python ≥ 3.9
- `numpy>=1.20`, `scipy>=1.7`; `flask>=2.3` for the web interface only

```bash
git clone https://github.com/kekered224/pyGrafLab.git pyGrafLab
cd pyGrafLab
pip install -e ".[web,fast]"   # or just `pip install -e .` for the Python API
```

`pygraflab` itself only needs `numpy` and `scipy`; Flask is required for the
web interface only.

### Optional

- `h5py` — required only to read MATLAB v7.3 `.mat` model files.
- `numba` (`[fast]`) — JIT-compiles the fnALF recursions (~35x faster synthesis
  at degree 2190). Entirely optional: without it the pure-NumPy
  implementation is used and results are identical.

---

## Quick start

### 1. Web interface (recommended)

```bash
python -m pygraflab.web                 # opens http://127.0.0.1:5002
python -m pygraflab.web --port 5010     # custom port
python -m pygraflab.web --no-browser    # headless
python -m pygraflab.web --debug         # enable the Werkzeug debugger (dev only)
```

After `pip install`, the same server is available as the `pygraflab-web`
command. On Windows, `start.bat` launches it on port 5002.

The launcher uses an isolated `.venv` (project-local first, then the workspace
`..\..\.venv`, otherwise it creates one) and only runs `pip install -e ".[web]"`
there — never into the global or user site-packages. It checks that
`pygraflab.web.server` actually imports instead of trusting pip's exit code,
and passes extra arguments through (`start.bat --no-browser --port 5003`).

Workflow: **Load model** (path or upload) → choose functional codes → set
grid/points and options → **Run**. Results are plotted on the canvas and
exported to `output/`.

The interface uses a VS Code-style layout (title bar, activity bar, collapsible
sections, editor tabs, status bar) with a light and a dark theme. The toggle
lives in the title bar; the choice is stored in `localStorage`, and
`?theme=dark` / `?theme=light` forces one for a single visit.

### 2. Python API

```python
import pygraflab

ggm = pygraflab.load_ggm('GGM/EGM2008.gfc', nmax=180)

results, lon, lat, sigma = pygraflab.compute(
    ggm,
    functional_codes=[10, 20],   # geoid undulation + gravity disturbance
    mode='grid', ellipsoid=1,    # GRS80
    lon_min=100, lon_max=120, lon_step=0.5,
    lat_min=20,  lat_max=40,  lat_step=0.5,
    h=0.0,
    method=1,
)

pygraflab.export_txt(results, 'output/')
```

Point-wise mode:

```python
results, lon, lat, sigma = pygraflab.compute(
    ggm, functional_codes=[20], mode='pointwise', ellipsoid=1,
    lon_deg=[116.4, 121.5], lat_deg=[39.9, 31.2], h_or_r=[50.0, 20.0],
    method=1,
)
```

### 3. Example

No download needed — the bundled degree-180 subset is picked up automatically:

```bash
python examples/run_example.py
python examples/run_example.py --ggm GGM/EGM2008.gfc    # or a full model of your own
```

Runs grid synthesis, point-wise synthesis, the high-level `compute()`
interface and the exporters, writing to `output/`.

The model is discovered in this order: `--ggm PATH`, `$PYGRAFLAB_GGM`, any
`GGM/*.gfc`. The example asks for degree 360 and falls back to the model's own
`max_degree` when that is lower, so the bundled subset runs in about a second;
pass `--nmax` to go higher (`--nmax 2190` with the full model, at a cost that
grows roughly with `nmax**3`).

```bash
python examples/run_example.py --nmax 720
```

---

## Model files

A global geopotential model in ICGEM `.gfc` format is a plain-text header
followed by one `gfc n m C S sigma_C sigma_S` record per coefficient.

**One truncated model ships with the repository**: `GGM/EIGEN-6C4_n180.gfc`,
EIGEN-6C4 (Förste et al., 2014) cut at degree 180 — 16 471 coefficients, 1.2 MB.
It exists so that a fresh clone can run `examples/run_example.py` and `pytest`
without downloading anything: the example evaluates in about one second and the
two numerical tests are no longer skipped. It is a header-preserving subset,
generated with `tools/make_sample_ggm.py`, and its coefficients are bit-identical
to the first 180 degrees of the full model.

Treat its output as a smoke test, not as a result: at degree 180 the geoid
undulation stays within ~0.5 m of the full-degree-2190 value, while gravity
disturbances, anomalies and deflections of the vertical lose their
short-wavelength part and can differ by tens of mGal or arc-seconds.

For production work use the full model. Download one from ICGEM and put it in
`GGM/` (or pass `--ggm PATH` / set `$PYGRAFLAB_GGM`); everything in `GGM/` other
than the bundled subset is git-ignored, and the example picks up any `.gfc`
file it finds there.

| Data | Download |
|------|----------|
| Global geopotential models (`EGM2008`, `EIGEN-6C4`, `XGM2019e`, ...) | <https://icgem.gfz.de/> — *Table of Models* → `gfc` / `zip` |
| MERIT DEM (topography, for pyTGF) | <https://global-hydrodynamics.github.io/MERIT_DEM/> |

A degree-2190 EGM2008 file is about 250 MB and needs roughly 2 GB while it is
parsed. Truncate a local copy to a working degree the same way the bundled
subset was built:

```bash
python tools/make_sample_ggm.py --nmax 180 --label EGM2008_n180   # GGM/EGM2008.gfc -> GGM/EGM2008_n180.gfc
python tools/make_sample_ggm.py --nmax 360 --force                # a larger subset
```

The script keeps the header (rewriting `max_degree`, and `modelname` when
`--label` is given), keeps `gfc` records with `n <= nmax`, and drops
trend/periodic terms, which belong to the full expansion.

`.mat` models (including MATLAB v7.3 / HDF5 files) are also accepted.


---

## Functional codes

| Code | Quantity | Code | Quantity |
|------|----------|------|----------|
| 2  | Deflection η | 14 | Gravitational tensor LNOF `Vxx`, `Vyy`, `Vzz` |
| 3  | Deflection ξ | 15 | Gravitational tensor LNOF `Vxy`, `Vxz`, `Vyz` |
| 4  | Deflection Θ | 16 | Gravity vector `gX`, `gY`, `gZ` |
| 5  | Disturbing potential `T` | 17 | Gravity `sa` |
| 6  | Disturbing tensor `Trr`, `Tφφ`, `Tλλ` | 18 | Gravity potential `W` |
| 7  | Disturbing tensor `Trφ`, `Trλ`, `Tφλ` | 19 | Gravity anomaly `sa` |
| 8  | Disturbing tensor LNOF `Txx`, `Tyy`, `Tzz` | 20 | Gravity disturbance |
| 9  | Disturbing tensor LNOF `Txy`, `Txz`, `Tyz` | 21 | Gravity disturbance `sa` |
| 10 | Geoid undulation `N` | 22 | Height anomaly (ellipsoidal) |
| 11 | Gravitational potential `V` | 23 | Height anomaly |
| 12 | Gravitational tensor `Vrr`, `Vφφ`, `Vλλ` | 24 | `Trr` |
| 13 | Gravitational tensor `Vrφ`, `Vrλ`, `Vφλ` | 25 | `Wrr` |

The authoritative mapping lives in `pygraflab.synthesis.FUNCTIONAL_NAMES` and is
exposed at `GET /api/functionals`.

---

## Web API

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET  | `/api/health` | Liveness probe with engine version |
| GET  | `/api/functionals` | Available functional codes and units |
| GET  | `/api/model-info` | Metadata of the currently loaded model |
| GET  | `/api/status` | Computation progress and status message |
| POST | `/api/load-model` | Load a GGM file by server-side path |
| POST | `/api/upload-model` | Upload and load a GGM file |
| POST | `/api/upload-covariance` | Upload a coefficient covariance matrix |
| POST | `/api/compute` | Run a synthesis computation (async worker) |
| GET  | `/api/results` | Retrieve results of the last computation |
| GET  | `/api/outputs` | List exported files in `output/` |
| GET  | `/api/download/<name>` | Download one exported file |
| GET  | `/api/download-all` | Download all exported files (legacy alias of `/api/outputs`) |

**Limits and guards**

- One computation runs at a time; concurrent `POST /api/compute` returns `409`.
- Grids above `MAX_GRID_POINTS = 2_000_000` nodes are rejected before allocation.
- Result payloads above `MAX_JSON_POINTS = 400_000` serialised values
  (grid nodes × functionals) are exported to file only, with a notice returned
  to the client and `/api/results` answering `404`.
- Model uploads are capped at 512 MB and restricted to
  `.gfc`, `.mat`, `.txt`, `.dat`, `.grx`, `.bin`.
- Covariance uploads are parsed on arrival: a malformed matrix is rejected with
  `400` and removed from disk, and the response reports the matrix dimension.
- All parameters are validated server-side (finite numbers, positive steps,
  `min < max`, latitude within ±90°, legal functional codes, etc.).
- Cross-site requests are rejected by an `Origin`/`Referer` guard unless they
  originate from the serving host.

---

## Front-end notes

The UI is plain ES2017 — no bundler, no framework, nothing to build. Edit
`pygraflab/web/static/` and reload. The tab icon lives at
`pygraflab/web/static/favicon.png`; `icons/app-icon.png` and
`icons/app-icon.ico` are the full-size artwork for desktop shortcuts.

`styles.css` holds the design tokens twice, under `:root` (light) and
`:root[data-theme="dark"]`; both declare `color-scheme` so native controls
follow along. The theme is set before first paint from an inline script, so
there is no flash.

Canvas rendering cannot read CSS variables, so `plot.js` keeps its own
`THEMES.light` / `THEMES.dark` object that `app.js` switches via
`FieldPlotUtil.setTheme()`. **A colour change must be applied in both the CSS
token and the matching `THEMES` entry**, otherwise the plot and the surrounding
chrome drift apart.

## Repository layout

```
pyGrafLab/
├── pygraflab/                  # Core engine (numpy / scipy only)
│   ├── io.py                   # GFC / MAT / covariance loaders
│   ├── synthesis.py            # Grid and point-wise synthesis
│   ├── fnalfs.py               # Three fnALF recursion methods
│   ├── commission.py           # Commission error propagation
│   ├── coords.py               # Spherical / ellipsoidal conversions
│   ├── ellipsoid.py            # Normal gravity and ellipsoid coefficients
│   ├── constants.py            # GRS80, WGS84, gravitational constant
│   ├── export.py               # txt / mat / report writers
│   └── web/                    # Flask backend + offline UI (installed with the package)
│       ├── server.py           # API endpoints
│       └── static/
│           ├── index.html      # UI markup
│           ├── styles.css      # Design tokens and components
│           ├── plot.js         # Dependency-free canvas field plotter
│           └── app.js          # UI logic and API client
├── GGM/                        # EIGEN-6C4_n180.gfc ships here; other models are git-ignored
├── tools/
│   └── make_sample_ggm.py      # Truncates a local model to a lower degree
├── examples/                   # run_example.py
├── output/                     # CLI and web export target (uploads/ holds uploads)
├── start.bat                   # Windows launcher (port 5002)
└── README.md
```

---

## Performance notes

- Synthesis is vectorised over latitude bands in grid mode; memory scales with
  the target grid, not with `nmax²`.
- `nmax` truncation at load time is the single most effective speed control —
  load only the degrees you need.
- Method 2 recurses on the fnALFs with the `u**m` factor and a 1e-280 seed
  factored out, and restores both during summation (GrafLab's Horner scheme).
  At degree 2190 it is slightly faster than method 1 and agrees with it to
  machine precision; method 3 trades speed for range at still higher degree.
- Point-wise mode avoids allocating the full grid and is preferable for a few
  hundred to a few thousand scattered points.
- All requested functionals are accumulated in a **single pass** over the
  spherical-harmonic orders, so combining disturbing-field and full-field
  outputs costs little more than one of them alone.
- `.gfc` loading uses a vectorised parser (falls back to a streaming reader for
  irregular files): a 2190-degree model reads in about 3 s.
- **Numba acceleration** (`pip install "pygraflab[fast]"`) JIT-compiles the
  sequential fnALF recursions and their derivatives. The kernels reproduce the
  reference float64 operation order bit for bit; results are identical to the
  pure-NumPy path. Without Numba the package still works — the pure-NumPy code
  is used as a fallback, and `PYGRAFLAB_NO_NUMBA=1` forces it.

Benchmark: EIGEN-6C4 at the full degree 2190, 20 points, **all 24 functionals**,
GRS80, standard fnALFs, single thread, Python 3.12 on Windows:

| Implementation | Wall time |
|---|---|
| MATLAB GrafLab 2.1.4 (7 calls, model reloaded before each) | 70 s |
| pyGrafLab, pure NumPy (before acceleration)                 | 190 s |
| pyGrafLab, with Numba — model load + synthesis              | **8.3 s** |
| pyGrafLab, with Numba — synthesis only                      | 5.1 s |

About 8x faster than the MATLAB reference end to end. Most of the MATLAB time
is spent reloading the 2190-degree model for each of its seven calls (~4.9 s
each); on the synthesis itself the difference is roughly 6x as well, and the
acceleration over the pure-NumPy path is about 35x. Grid mode
reuses the same fnALF path: 22 functionals on a 21 x 21 grid over Colorado take
about 6.7 s in total.

---

## Validation

pyGrafLab was validated against the original MATLAB GrafLab 2.1.4 on
point-wise synthesis of **all 24 functional codes** (2-25) using
EIGEN-6C4 at the full degree 2190, at 20 computation points in the
Colorado region (GRS80, standard fnALFs, heights 1.4-3.6 km). All
functionals agree to machine precision:

| Quantity group | max abs diff | max rel diff |
|----------------|--------------|--------------|
| Deflections eta / xi / Theta (arcsec) | 1.4e-11 | 4.0e-14 |
| Potentials T / V / W (m²/s²)          | 1.5e-7  | 6.7e-12 |
| Gravity, disturbances, anomalies (mGal)| 3.5e-9 | 5.7e-12 |
| Geoid N, height anomalies (m)         | 3.3e-13 | 4.7e-14 |
| Marussi tensor, all frames (E)        | 1.8e-11 | 2.5e-13 |

Codes 10/23 (geoid undulation, height anomaly) are compared at h = 0
with a zero DTM, mirroring the GrafLab requirements for these functionals.
This validation also uncovered and fixed a sign-convention bug in the
LNOF off-diagonal components (codes 9/15: Txy/Vxy and Tyz/Vyz follow the
ICGEM convention of negated cross derivatives, matching GrafLab).

Bundled smoke tests cover the model loaders, the fnALF recursions and synthesis
internals:

```bash
pip install -e ".[dev]"
pytest
```

---

## Citation, attribution and references

**Attribution.** All credit for the algorithm belongs to the original authors.
`pyGrafLab` is an independent Python re-implementation of GrafLab 2.1.4 by
**Blažej Bucha and Juraj Janák**. The set of functionals, the fnALF recursion
schemes, the input/output layout and the overall design of the software follow
their published work. No MATLAB source code from GrafLab is included in this
repository, and this project claims no authorship of the original software.

### Cite the original software

If you publish results obtained with this implementation, cite the original
GrafLab paper:

> Bucha, B., Janák, J. (2013). *A MATLAB-based graphical user interface
> program for computing functionals of the geopotential up to ultra-high
> degrees and orders.* Computers & Geosciences, 56, 186–196.
> https://doi.org/10.1016/j.cageo.2013.03.012

```bibtex
@article{bucha2013graflab,
  author  = {Bucha, Bla{\v{z}}ej and Jan{\'a}k, Juraj},
  title   = {A {MATLAB}-based graphical user interface program for computing
             functionals of the geopotential up to ultra-high degrees and orders},
  journal = {Computers \& Geosciences},
  volume  = {56},
  pages   = {186--196},
  year    = {2013},
  doi     = {10.1016/j.cageo.2013.03.012}
}
```

### Cite this implementation

Citation metadata for `pyGrafLab` itself is provided in `CITATION.cff`, which
GitHub renders through its "Cite this repository" button. Citing the port is
optional; citing Bucha & Janák (2013) is not.

### Method references

- Barthelmes, F. (2013). *Definition of Functionals of the Geopotential and
  Their Calculation from Spherical Harmonic Models.* Scientific Technical
  Report STR09/02, GFZ German Research Centre for Geosciences.
  https://doi.org/10.2312/GFZ.b103-0902-26
- Heiskanen, W. A., Moritz, H. (1967). *Physical Geodesy.* W. H. Freeman,
  San Francisco.
- Tscherning, C. C., Rapp, R. H., Goad, C. C. (1983). A comparison of methods
  for computing gravimetric quantities from high degree spherical harmonic
  expansions. *Manuscripta Geodaetica*, 8(3), 249–272.
- Ince, E. S., Barthelmes, F., Reißland, S., Elger, K., Förste, C., Flechtner,
  F., Schuh, H. (2019). ICGEM – 15 years of successful collection and
  distribution of global gravitational models, associated services and future
  plans. *Earth System Science Data*, 11, 647–674.
  https://doi.org/10.5194/essd-11-647-2019

### Models used in the validation

- Förste, C., Bruinsma, S. L., Abrikosov, O., Lemoine, J.-M., Marty, J. C.,
  Flechtner, F., Balmino, G., Barthelmes, F., Biancale, R. (2014).
  *EIGEN-6C4 – The latest combined global gravity field model including GOCE
  data up to degree and order 2190 of GFZ Potsdam and GRGS Toulouse.*
  GFZ Data Services. https://doi.org/10.5880/icgem.2015.1

Apart from the degree-180 EIGEN-6C4 subset in `GGM/`, model coefficients are
not redistributed with this repository; download the full models from ICGEM
(see [Model files](#model-files)).

---

## License

`pyGrafLab` is released under the MIT licence (see `LICENSE`). The Python code,
the front end and this documentation are Copyright (c) 2026 the pyGrafLab
contributors.

**The original GrafLab 2.1.4 software by Blažej Bucha and Juraj Janák (2013)
and its documentation remain the copyright of those authors and their rights
holders; this project claims no ownership of that work.** GrafLab remains the
reference for the algorithmic specification; cite Bucha & Janák (2013) in work
that uses this implementation. The MIT licence covers the independently written
Python code only — no model coefficients are distributed with this repository.
