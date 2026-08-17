# xrtoolz-grad — finite-difference calculus

```bash
# Pre-PyPI, install from the workspace via git — xrgrad has no internal
# deps, so this resolves on its own; once published this becomes a
# plain `pip install xrtoolz-grad`.
uv pip install "xrtoolz-grad @ git+https://github.com/jejjohnson/xrtoolz@main#subdirectory=packages/xrtoolz-grad"
```

`xrgrad` provides partial derivatives and vector-calculus operators
(`∇`, `∇·`, `∇×`, `Δ`) on labeled xarray grids. It is a pure-function
package — no pipekit, no other xrtoolz dependencies — and the only
member of the workspace that carries the
[finitediffx](https://github.com/ASEM000/finitediffx) (JAX-backed)
stencil engine.

## Geometry dispatch

Every operator takes `geometry=`, which selects how coordinate values
become physical distances:

| Geometry | Assumes | Spacing |
|---|---|---|
| `"cartesian"` | uniform grid, coords in metres | single Δ read from the coord, validated uniform (`uniform_rtol`) |
| `"rectilinear"` | non-uniform 1-D coord per dim, in metres | true coordinate differences |
| `"spherical"` | lon/lat in **degrees** | metric factors below |

For the spherical case, derivatives are taken in radians and scaled by
the metric factors

$$
\frac{\partial}{\partial x}
  = \frac{1}{R\cos\varphi}\frac{\partial}{\partial\lambda},
\qquad
\frac{\partial}{\partial y}
  = \frac{1}{R}\frac{\partial}{\partial\varphi},
$$

with $\lambda$ longitude, $\varphi$ latitude, and $R$ the Earth radius
(`xrgrad.EARTH_RADIUS`, 6 371 000 m).

### Mixed geometry

`divergence` also accepts a **per-dim sequence** of geometries, one per
entry in `dims`. The common case is a 3-D ocean budget: spherical
horizontal, rectilinear vertical.

```python
div = xrgrad.divergence(
    ds,
    ("flux_e", "flux_n", "flux_w"),
    dims=("lon", "lat", "depth"),
    geometry=("spherical", "spherical", "rectilinear"),
)
```

The `-(F_y \tan\varphi)/R` curvature term is applied when **both** the
longitude and latitude axes are spherical. Marking only one of them (or
marking the vertical axis spherical) raises — the metric couples the
pair, so neither is meaningful alone. Geometry-specific keywords are
routed only to the backends that accept them, so `radius=` may
accompany a mixed call without reaching the vertical axis.

### Temporal coordinates

`datetime64` and `timedelta64` coordinates work anywhere a numeric one
does; they are converted to **seconds**, so `∂c/∂t` comes back per
second. Values are anchored at the first sample before conversion, which
keeps nanosecond steps representable in float64. A uniform time axis
takes the cartesian path; an irregular one (a skipped day, monthly
means) needs `geometry="rectilinear"`. `cftime`/object-dtype calendars
are rejected with an explicit error rather than silently mis-stepped.

All geometries expect the differentiated dimension to carry a **1-D
coordinate**: step sizes come from coordinate values, so a dimension
without one raises `ValueError`, and 2-D (curvilinear) coordinates are
rejected rather than silently producing a nonsense step. `"rectilinear"`
additionally requires the coordinate to be **strictly monotonic** —
repeated or reversing values would divide by a zero coordinate
difference. Strictly descending coordinates are fine, as are scalar
coordinates on axes a given derivative does not use (a `lat` derivative
still works after `da.sel(lon=...)`).

## Curvature corrections

On the sphere, divergence and curl are not plain sums of partials —
the unit vectors themselves rotate with longitude. `xrgrad` adds the
curvature terms so results match the equivalent `metpy.calc` operators
on lon/lat fields:

$$
\nabla\cdot\mathbf{F}
  = \frac{1}{R\cos\varphi}\frac{\partial F_x}{\partial\lambda}
  + \frac{1}{R}\frac{\partial F_y}{\partial\varphi}
  - \frac{F_y\tan\varphi}{R},
$$

$$
\zeta
  = \frac{1}{R\cos\varphi}\frac{\partial v}{\partial\lambda}
  - \frac{1}{R}\frac{\partial u}{\partial\varphi}
  + \frac{u\tan\varphi}{R}.
$$

On spherical and rectilinear geometries the Laplacian is composed as
$\Delta f = \nabla\cdot\nabla f$, so it inherits the corrections
automatically. Only on Cartesian grids, where there is no curvature term
to inherit, does `laplacian` instead sum direct second-derivative
stencils $\sum_d \partial^2 f/\partial d^2$ — see below.

## Accuracy and stencils

`accuracy=` is finitediffx's accuracy order; `method=` selects
`"central"` (default), `"forward"`, or `"backward"` stencils. Interior
points use the requested central stencil; boundary points fall back to
one-sided differences of the same order. Two practical notes:

- First derivatives of polynomials up to the stencil order are exact on
  interior points at `accuracy=1`.
- `partial` takes an `order=` argument for higher derivatives. `order=2`
  uses a single second-derivative stencil rather than two composed
  passes. It is available on Cartesian grids (and on rectilinear ones
  whose coordinate turns out to be uniform); spherical geometry and
  genuinely non-uniform coordinates raise `NotImplementedError`.
- `laplacian` sums `order=2` stencils **only** for
  `geometry="cartesian"`, so there it is already exact for quadratic
  fields at the default `accuracy=1` and degrades only a single boundary
  ring. Every other geometry — spherical, and *all* rectilinear grids,
  including ones whose coordinate is uniform enough that each partial
  delegates to the Cartesian backend — still composes two
  first-derivative stencils, so pass `accuracy=2` (or higher) there when
  the interior must reproduce quadratic fields exactly.
- `divergence`, `curl`, and `laplacian` accept a per-dim `accuracy`
  tuple as well as a scalar, matching `gradient`. The tuple pairs with
  `dims` in order, which is useful on anisotropic grids.

## Periodic boundaries

By default the first and last points of an axis fall back to one-sided
stencils. On a genuinely periodic axis — global longitude being the
usual case — that leaves a visible seam at the dateline. Pass
`periodic=` to wrap instead, so those points are differentiated against
the opposite edge and are as accurate as the interior:

```python
zeta = xrgrad.curl(
    ds, ("u", "v"), dims=("lon", "lat"),
    geometry="spherical", periodic="lon",
)
```

`partial` takes a bool (it acts on one axis); the multi-dim operators
take a dimension name or a sequence of names, which must be among the
dims being differentiated. Restrictions:

- **Spherical longitude** must span the full 360°; a regional grid has
  no wrap-around neighbour and raises.
- **Latitude never wraps** — the poles are a separate problem.
- **Non-uniform rectilinear** axes raise: wrapping the index grid would
  also have to wrap the spacing, which has no single period.

```python
import numpy as np
import xarray as xr
import xrgrad

x = np.linspace(0.0, 3.0, 4)
X, Y = np.meshgrid(x, x, indexing="xy")
coords = {"y": x, "x": x}
ds = xr.Dataset(
    {
        "u": xr.DataArray(-Y, dims=("y", "x"), coords=coords),
        "v": xr.DataArray(X, dims=("y", "x"), coords=coords),
    }
)

zeta = xrgrad.curl(ds, ("u", "v"), dims=("x", "y"))       # rigid rotation → 2
div = xrgrad.divergence(ds, ("v", "u"), dims=("x", "y"))  # ∇·(x, -y) → 0
```

For a worked spherical example (geostrophic currents and vorticity from
sea-surface height), see the
[quickstart notebook](../notebooks/xrgrad_quickstart.ipynb).

## Grid metrics and constants

`grid_metrics_from_coords` derives the cell-width / face-area Datasets
consumed by `xrtoolz.budgets` from a Dataset's lon/lat (and optional
depth) coordinates. The physical constants `EARTH_RADIUS`, `GRAVITY`,
and `OMEGA` are exported at the top level and replace the equivalent
`metpy.constants` lookups.

## API reference

- [Calculus](../api/calc.md)
