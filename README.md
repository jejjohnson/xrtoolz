# xrtoolz

[![Tests](https://github.com/jejjohnson/xrtoolz/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/xrtoolz/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/xrtoolz/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/xrtoolz/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/xrtoolz/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/xrtoolz/actions/workflows/typecheck.yml)
[![Deploy Docs](https://github.com/jejjohnson/xrtoolz/actions/workflows/pages.yml/badge.svg)](https://github.com/jejjohnson/xrtoolz/actions/workflows/pages.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

**Composable operator library for geoprocessing Earth System Data Cubes.**

`xrtoolz` provides a uniform `Operator` abstraction for preprocessing, inference, and evaluation of xarray datasets. Pipelines compose linearly via `Sequential` or as DAGs via the functional `Graph` API — the same operator works in both. The composition core lives in the carrier-agnostic [`pipekit`](https://github.com/jejjohnson/pipekit) framework; `xrtoolz` is a direct consumer that adds the xarray-specific operator families.

## Package layout

```
xrtoolz/
├── combinators.py  # Augment, ApplyToEach (xarray-Dataset-flavoured)
├── signature.py    # dict-keyed Signature for shape inference
├── geo/    # Generic xarray geoprocessing (validation, subset, regrid,
│           # detrend, masks, metrics, spectral, ...)
├── ocn/    # Oceanography physics (streamfunction, geostrophic velocity, ...)
├── atm/    # Atmospheric physics (potential temperature, wind, ...)
│   └── gas/ch4/  # Trace-gas physics (column averaging kernel, ...)
├── rs/     # Remote sensing (NDVI, radiance/reflectance, ...)
└── ice/    # Cryosphere (reserved; no content yet)
```

`Operator`, `Sequential`, `Graph`, `Input`, `Node`, `ConfigMixin`, `Tap` are re-exported from `pipekit` at the top level. Rule: anything domain-agnostic about composition lives in `pipekit`; anything domain-agnostic about xarray lives in `xrtoolz` itself (`combinators`, `signature`); only true physics lives in `ocn`/`atm`/`rs`.

## Quick start

```bash
# Prerequisites: uv (https://github.com/astral-sh/uv)
git clone https://github.com/jejjohnson/xrtoolz.git
cd xrtoolz
make install      # install all dependency groups (resolves pipekit from GitHub)
make test         # run tests
make docs-serve   # preview docs locally
```

### Pre-PyPI install

`xrtoolz` depends on [`pipekit`](https://github.com/jejjohnson/pipekit) (also pre-PyPI), resolved via `[tool.uv.sources]`. Plain `pip install git+https://...` will fail until `pipekit` reaches PyPI — use `uv`:

```bash
uv pip install "git+https://github.com/jejjohnson/xrtoolz@main"
```

## Status

Pre-alpha. See the full design document in [`docs/design/`](docs/design/) for motivation, architecture, boundaries, and roadmap.

## License

MIT. See [LICENSE](LICENSE).
