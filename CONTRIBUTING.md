# Contributing

## Development setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev,web]"
python -m pytest tests -q
python examples/run_example.py --ggm GGM/EGM2008.gfc --nmax 180
```

The smoke tests skip their numerical cases when no local `.gfc` model is
present. Download one from ICGEM (see README.md) to run them and the example.

## Front-end changes

`pygraflab/web/static/` contains plain ES2017 JavaScript — no bundler, no
framework, nothing to build. After editing, check syntax:

```bash
node --check pygraflab/web/static/app.js
node --check pygraflab/web/static/plot.js
```

Canvas colours are duplicated between the CSS tokens in `styles.css` and
`THEMES.light/.dark` in `plot.js`. **Change both together.** Do not introduce
CDN links, web fonts or tile servers; the UI must stay fully offline.

## Model data

No model coefficients are committed. Download a `.gfc` file from ICGEM (see
README.md) into `GGM/`; the whole directory is git-ignored. Truncate a local
copy with `python tools/make_sample_ggm.py` if you need a smaller file.

## Pull requests

- Keep this repository self-contained; it is independent of pyTGF.
- Run `pytest tests -q` and `node --check` on both UI scripts before opening a
  PR. The example above needs a model you downloaded yourself.
- Do not commit files under `output/`, `GGM/`, virtual environments or build
  artefacts.
