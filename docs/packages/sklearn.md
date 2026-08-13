# xrtoolz-sklearn — the scikit-learn bridge

```bash
pip install xrtoolz-sklearn
```

`xrsklearn` lets any sklearn-style estimator operate on N-D labeled
data. One marshalling core — **stack → delegate → unstack** — exposed
through three surfaces.

## The lifecycle

sklearn estimators want a 2-D `(n_samples, n_features)` matrix.
`XarrayEstimator` gets there and back:

1. **Stack**: the input is transposed so `sample_dim` leads, and every
   other dim is flattened into a feature `MultiIndex`
   (`(time, lat, lon)` → `(time, lat×lon)`). Dataset inputs
   column-concatenate their data_vars into one feature matrix.
2. **Delegate**: the wrapped estimator's own
   `fit / transform / predict / …` runs on the 2-D view.
3. **Unstack**: outputs are re-labeled. Same-feature-count outputs
   rebuild the original feature dims; reduced outputs (PCA scores,
   cluster labels) come back as `(sample_dim, component)` or
   `(sample_dim,)`.

The fitted wrapper stores the feature-grid metadata, which is what lets
`inverse_transform` rebuild the original `(sample, *feature_dims)`
layout. Fitted-estimator attributes (`components_`,
`cluster_centers_`, …) pass through untouched.

```python
from sklearn.decomposition import PCA
from xrsklearn import XarrayEstimator

wrap = XarrayEstimator(PCA(n_components=3), sample_dim="time")
scores = wrap.fit_transform(da)        # (time, lat, lon) → (time, component)
recon = wrap.inverse_transform(scores)  # back to (time, lat, lon)
```

## NaN policy

Real gridded data has land masks and gaps; sklearn raises on NaN.
`nan_policy=` decides what happens at the 2-D boundary:

| Policy | Behaviour |
|---|---|
| `"propagate"` (default) | hand NaNs to the estimator unchanged — fine for NaN-aware estimators, raises inside sklearn otherwise |
| `"mask"` | drop sample rows containing any NaN before delegating, re-insert NaN rows in the output |
| `"raise"` | fail fast with a labeled error before sklearn sees the data |

## The `.sklearn` accessors

Importing `xrsklearn` registers `da.sklearn` / `ds.sklearn` accessors —
thin sugar that constructs an `XarrayEstimator` and forwards:

```python
import xrsklearn  # registration side effect

scaled = da.sklearn.fit_transform(StandardScaler(), sample_dim="time")
```

Methods that need a *fitted* estimator (`transform`, `predict`,
`score`, `inverse_transform`) accept either a raw fitted sklearn
estimator or a fitted `XarrayEstimator` — but only the wrapper carries
the metadata to rebuild the original feature grid on
`inverse_transform`.

## SklearnOp — the pipeline operator

`SklearnOp` wraps an estimator as an `xrcore.Operator` so fit/transform
steps compose into `pipekit.Sequential` chains next to any other
operator, reading and writing named Dataset variables:

```python
from pipekit import Sequential
from xrsklearn import SklearnOp

pipeline = Sequential(
    [
        SklearnOp(StandardScaler(), variable="ssh", sample_dim="time",
                  method="fit_transform"),
        SklearnOp(fitted_pca, variable="ssh", output_variable="pcs",
                  sample_dim="time"),
    ]
)
```

## Why not the split-object pattern?

Stateful xrtoolz operations normally split into `CalculateX` (returns
state) and `ApplyX(state)`. The sklearn bridge is the deliberate
exception: sklearn's own fit/transform API *is* the state contract, and
wrapping it twice would force every estimator through a second,
redundant state object. `XarrayEstimator` keeps sklearn's lifecycle;
`SklearnOp` adapts it to pipelines. Operators holding live estimators
set `forbid_in_yaml` (pipekit convention), since a fitted model is not
YAML-serializable.

## API reference

- [Utilities (`XarrayEstimator`)](../api/utils.md)
- [Transforms (`SklearnOp`)](../api/transforms.md)
