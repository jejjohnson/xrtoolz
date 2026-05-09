"""Unstructured points → gridded value resampling."""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from scipy.stats import binned_statistic_2d
from sklearn.neighbors import KernelDensity

from xr_toolz.interpolate._src.binning import Grid


BandwidthRule = Literal["scott", "silverman"]
KernelName = Literal[
    "gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"
]
MetricName = Literal["euclidean", "haversine"]
AlgorithmName = Literal["auto", "kd_tree", "ball_tree"]
OutputMode = Literal["density", "counts", "counts_per_area"]


def points_to_grid(
    lons: np.ndarray,
    lats: np.ndarray,
    values: np.ndarray,
    grid: Grid,
    statistic: str = "mean",
) -> xr.DataArray:
    """Bin raw (lon, lat, value) tuples onto ``grid``.

    Thin wrapper around :func:`scipy.stats.binned_statistic_2d` that
    doesn't require constructing a scattered DataArray first.
    """
    finite = np.isfinite(values)
    lon_edges, lat_edges = grid.bin_edges()
    stat, _, _, _ = binned_statistic_2d(
        np.ravel(lons)[finite],
        np.ravel(lats)[finite],
        np.ravel(values)[finite],
        statistic=statistic,
        bins=[lon_edges, lat_edges],
    )
    return xr.DataArray(
        data=stat.T,
        dims=("lat", "lon"),
        coords={"lon": grid.lon, "lat": grid.lat},
    )


def _bandwidth_rule(pts: np.ndarray, rule: BandwidthRule) -> float:
    """Return Scott or Silverman bandwidth for ``pts``."""
    n, d = pts.shape
    coordinate_variance = np.var(pts, axis=0, ddof=1)
    sigma = float(np.sqrt(np.mean(coordinate_variance)))
    if rule == "scott":
        bandwidth = n ** (-1.0 / (d + 4)) * sigma
    elif rule == "silverman":
        bandwidth = (n * (d + 2) / 4.0) ** (-1.0 / (d + 4)) * sigma
    else:
        raise ValueError(f"unknown bandwidth rule {rule!r}")
    if bandwidth <= 0 or not np.isfinite(bandwidth):
        raise ValueError(f"bandwidth rule {rule!r} produced {bandwidth!r}")
    return float(bandwidth)


def kde_to_grid(
    lons: ArrayLike,
    lats: ArrayLike,
    grid: Grid,
    *,
    weights: ArrayLike | None = None,
    bandwidth: float | BandwidthRule = "scott",
    kernel: KernelName = "gaussian",
    metric: MetricName = "euclidean",
    algorithm: AlgorithmName = "auto",
    output: OutputMode = "density",
    rtol: float = 1e-4,
) -> xr.DataArray:
    """Evaluate a KDE of scattered lon/lat points on ``grid``.

    Args:
        lons: 1-D point longitudes. Non-finite entries are dropped.
        lats: 1-D point latitudes. Non-finite entries are dropped.
        grid: Target :class:`Grid`.
        weights: Optional per-point weights passed to
            :class:`sklearn.neighbors.KernelDensity`.
        bandwidth: Positive float, or ``"scott"`` / ``"silverman"``. For
            ``metric="haversine"``, float bandwidths are radians.
        kernel: sklearn KDE kernel name.
        metric: ``"euclidean"`` treats lon/lat as planar; ``"haversine"``
            converts lon/lat to radians and uses great-circle distances.
        algorithm: sklearn tree backend. ``"haversine"`` uses ``"ball_tree"``.
        output: ``"density"`` integrates to one; ``"counts"`` scales density
            by point count or weight sum; ``"counts_per_area"`` scales counts
            by the mean grid-cell area.
        rtol: Relative tolerance for sklearn tree pruning.

    Returns:
        DataArray on ``grid`` with ``("lat", "lon")`` dimensions.
    """
    if output not in ("density", "counts", "counts_per_area"):
        raise ValueError(f"unknown output mode {output!r}")
    if metric not in ("euclidean", "haversine"):
        raise ValueError(f"unknown metric {metric!r}")

    lon_values = np.ravel(np.asarray(lons, dtype=float))
    lat_values = np.ravel(np.asarray(lats, dtype=float))
    if lon_values.shape != lat_values.shape:
        raise ValueError("lons and lats must have the same shape")

    finite = np.isfinite(lon_values) & np.isfinite(lat_values)
    if weights is None:
        sample_weight = None
    else:
        weight_values = np.ravel(np.asarray(weights, dtype=float))
        if weight_values.shape != lon_values.shape:
            raise ValueError("weights must have the same shape as lons and lats")
        finite &= np.isfinite(weight_values)
        sample_weight = weight_values[finite]

    pts = np.column_stack([lon_values[finite], lat_values[finite]])
    if pts.shape[0] < 2:
        raise ValueError("KDE requires at least 2 finite points")

    if metric == "haversine":
        fit_pts = np.deg2rad(pts[:, [1, 0]])
        algorithm = "ball_tree"
    else:
        fit_pts = pts

    if isinstance(bandwidth, str):
        bandwidth_value = _bandwidth_rule(fit_pts, bandwidth)
    else:
        bandwidth_value = float(bandwidth)
        if bandwidth_value <= 0 or not np.isfinite(bandwidth_value):
            raise ValueError(f"bandwidth must be positive, got {bandwidth!r}")

    kde = KernelDensity(
        bandwidth=bandwidth_value,
        kernel=kernel,
        metric=metric,
        algorithm=algorithm,
        rtol=rtol,
    ).fit(fit_pts, sample_weight=sample_weight)

    lon_grid, lat_grid = np.meshgrid(grid.lon, grid.lat, indexing="xy")
    if metric == "haversine":
        queries = np.deg2rad(np.column_stack([lat_grid.ravel(), lon_grid.ravel()]))
    else:
        queries = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    density = np.exp(kde.score_samples(queries)).reshape(len(grid.lat), len(grid.lon))

    n_eff = pts.shape[0] if sample_weight is None else float(sample_weight.sum())
    if output == "density":
        data = density
    elif output == "counts":
        data = density * n_eff
    else:
        dlon = float(np.mean(np.abs(np.diff(np.asarray(grid.lon, dtype=float)))))
        dlat = float(np.mean(np.abs(np.diff(np.asarray(grid.lat, dtype=float)))))
        data = density * n_eff * dlon * dlat

    return xr.DataArray(
        data=data,
        dims=("lat", "lon"),
        coords={"lon": grid.lon, "lat": grid.lat},
    )
