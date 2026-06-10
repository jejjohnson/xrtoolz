# xrtoolz

[![Tests](https://github.com/jejjohnson/xrtoolz/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/xrtoolz/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/xrtoolz/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/xrtoolz/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/xrtoolz/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/xrtoolz/actions/workflows/typecheck.yml)
[![Deploy Docs](https://github.com/jejjohnson/xrtoolz/actions/workflows/pages.yml/badge.svg)](https://github.com/jejjohnson/xrtoolz/actions/workflows/pages.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

**Composable operators for geoprocessing Earth System Data Cubes.**

`xrtoolz` gives every preprocessing, physics, inference, and evaluation step
a single shape — an **`Operator`**: a callable that maps `xarray` in to
`xarray` out, carries its configuration, and reports its shape signature.
Operators chain linearly via `Sequential` or wire into a DAG via the
functional `Graph` API — *the same operator works in both*.

The carrier-agnostic composition core lives in
[`pipekit`](https://github.com/jejjohnson/pipekit); `xrtoolz` adds the
xarray-specific operator families on top.

📖 **[Documentation](https://jejjohnson.github.io/xrtoolz)** ·
🧩 **[API Reference](https://jejjohnson.github.io/xrtoolz/api/)** ·
🧪 **[Tutorials](https://jejjohnson.github.io/xrtoolz/notebooks/operators_pipeline_demo/)**

## Quick start

```python
import xrtoolz
from xrtoolz import Sequential
from xrtoolz.geo import RemoveMean
from xrtoolz.ocn.operators import Streamfunction, GeostrophicVelocities
from xrtoolz.viz.validation import SpatialMapPanel

# Geostrophic surface currents from sea-surface height, as one pipeline
pipeline = Sequential(
    RemoveMean(var="ssh"),       # de-mean the SSH field
    Streamfunction(),            # ψ = g·η / f
    GeostrophicVelocities(),     # (u_g, v_g) from ψ
)

currents = pipeline(ds)          # xr.Dataset in → xr.Dataset out
fig = SpatialMapPanel(var="u")(currents)
```

Need branching or shared intermediates? Wire the *same* operators as a DAG
with `Graph(inputs=…, outputs=…)`.

## Architecture — three layers

| Layer | Name | What it is |
|-------|------|------------|
| 0 | **Primitives** | Pure functions `(xr.Dataset, …) → xr.Dataset`. Backed by jaxtyped numpy kernels. |
| 1 | **Operators** | `Operator` subclasses with a uniform `__call__` / `get_config` / `__repr__`; compose with `Sequential`. |
| 2 | **Graph** | `Input` / `Node` / `Graph` DAG API and `ModelOp` inference wrappers. |

Drop to Layer 0 when you want the raw function; stay at Layer 1–2 to
compose. `Operator`, `Sequential`, `Graph`, `Input`, `Node`, `Tap` are
re-exported from `pipekit` at the top level.

## Module map

| Module | Scope |
|--------|-------|
| [`geo`](https://jejjohnson.github.io/xrtoolz/api/geo/coords/) | Generic geoprocessing — validation, subsetting, regions, masks, CRS, climatology, wavelets, extremes |
| [`calc`](https://jejjohnson.github.io/xrtoolz/api/calc/) | Finite-difference `gradient` / `divergence` / `curl` / `laplacian` |
| [`transforms`](https://jejjohnson.github.io/xrtoolz/api/transforms/) | Fourier / wavelet / DCT spectra, spectral fluxes, decompositions, morphology |
| [`interpolate`](https://jejjohnson.github.io/xrtoolz/api/interpolate/) | Regrid, coarsen, gap-fill, bin, smooth, point-sample |
| [`ocn`](https://jejjohnson.github.io/xrtoolz/api/ocn/) | Ocean physics — geostrophy, vorticity, KE, stratification, SSH |
| `atm` · `rs` | Atmospheric & remote-sensing physics (reserved; planned) |
| [`metrics`](https://jejjohnson.github.io/xrtoolz/api/metrics/) | Pixel, spectral, multiscale, physical, masked, distributional evaluation |
| [`budgets`](https://jejjohnson.github.io/xrtoolz/api/budgets/) | Conservation-budget residuals (heat, salt, volume, KE) |
| [`einx`](https://jejjohnson.github.io/xrtoolz/api/einx/) | Named-tensor algebra (einsum / rearrange / reduce by dim name) |
| [`inference`](https://jejjohnson.github.io/xrtoolz/api/inference/) | `ModelOp` wrappers for sklearn / JAX / framework-agnostic models |
| [`viz`](https://jejjohnson.github.io/xrtoolz/api/viz/) | Cartopy axes, colormap registry, V1–V5 validation panels |

> **Data acquisition** (CMEMS / CDS / AEMET archive readers) now lives in the
> companion [`xrreader`](https://github.com/jejjohnson/xrreader) package.

## Install

```bash
uv add xrtoolz
```

`xrtoolz` depends on [`pipekit`](https://github.com/jejjohnson/pipekit) and
[`xrreader`](https://github.com/jejjohnson/xrreader) (both pre-PyPI),
resolved via `[tool.uv.sources]`. A plain `pip install git+https://…` will
fail until those reach PyPI — use `uv`:

```bash
uv pip install "git+https://github.com/jejjohnson/xrtoolz@main"
```

### From source

```bash
git clone https://github.com/jejjohnson/xrtoolz.git
cd xrtoolz
make install      # install all dependency groups (resolves pipekit from GitHub)
make test         # run the test suite
make docs-serve   # preview the docs locally
```

## Status

Pre-alpha. The API is stabilising domain by domain — see the full
[design documentation](https://jejjohnson.github.io/xrtoolz/design/) for
motivation, architecture, boundaries, conventions, and roadmap.

## License

MIT. See [LICENSE](LICENSE).
