"""Tests for the xarray ↔ sklearn bridge."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr


pytest.importorskip("sklearn")


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from xr_toolz.core import Sequential
from xr_toolz.geo.operators import ValidateCoords
from xr_toolz.transforms import SklearnOp
from xr_toolz.utils import XarrayEstimator


def _sample_da() -> xr.DataArray:
    values = np.array(
        [
            [0.0, 1.0, 2.0],
            [np.nan, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, np.nan, 6.0],
            [5.0, 6.0, 7.0],
        ]
    )
    return xr.DataArray(
        values,
        dims=("time", "feature"),
        coords={"time": np.arange(values.shape[0]), "feature": ["a", "b", "c"]},
        name="ssh",
        attrs={"units": "m"},
    )


def test_nan_policy_mask_standard_scaler_preserves_sample_axis() -> None:
    da = _sample_da()
    valid = ~np.isnan(da.values).any(axis=1)

    out = XarrayEstimator(
        StandardScaler(),
        sample_dim="time",
        nan_policy="mask",
    ).fit_transform(da)

    expected = StandardScaler().fit_transform(da.values[valid])
    assert out.dims == da.dims
    assert out.attrs == da.attrs
    np.testing.assert_array_equal(out["time"], da["time"])
    np.testing.assert_allclose(out.values[valid], expected)
    assert np.isnan(out.values[~valid]).all()


def test_nan_policy_mask_pca_inverse_round_trip_preserves_nan_rows() -> None:
    da = _sample_da()
    valid = ~np.isnan(da.values).any(axis=1)
    wrap = XarrayEstimator(
        PCA(n_components=2),
        sample_dim="time",
        nan_policy="mask",
    )

    scores = wrap.fit_transform(da)
    reconstructed = wrap.inverse_transform(scores)

    assert scores.dims == ("time", "component")
    assert reconstructed.dims == da.dims
    assert np.isnan(scores.values[~valid]).all()
    assert np.isnan(reconstructed.values[~valid]).all()
    np.testing.assert_array_equal(reconstructed["time"], da["time"])
    np.testing.assert_array_equal(reconstructed["feature"], da["feature"])


def test_nan_policy_mask_kmeans_predict_preserves_nan_rows() -> None:
    da = _sample_da()
    valid = ~np.isnan(da.values).any(axis=1)
    wrap = XarrayEstimator(
        KMeans(n_clusters=2, n_init="auto", random_state=0),
        sample_dim="time",
        nan_policy="mask",
    ).fit(da)

    labels = wrap.predict(da)

    expected = wrap.estimator_.predict(da.values[valid])
    assert labels.dims == ("time",)
    np.testing.assert_array_equal(labels.values[valid], expected)
    assert np.isnan(labels.values[~valid]).all()


def test_sklearn_accessor_matches_explicit_estimator() -> None:
    da = _sample_da().drop_sel(time=[1, 4])

    explicit = XarrayEstimator(StandardScaler(), sample_dim="time").fit_transform(da)
    via_accessor = da.sklearn.fit_transform(StandardScaler(), sample_dim="time")

    np.testing.assert_array_equal(via_accessor.values, explicit.values)
    assert via_accessor.dims == explicit.dims
    assert via_accessor.attrs == explicit.attrs


def test_sklearn_op_composes_in_sequential_and_config_is_json_safe() -> None:
    da = xr.DataArray(
        np.arange(4 * 3 * 2, dtype=float).reshape(4, 3, 2),
        dims=("time", "lat", "lon"),
        coords={
            "time": np.arange(4),
            "lat": np.array([-10.0, 0.0, 10.0]),
            "lon": np.array([150.0, 151.0]),
        },
        name="ssh",
        attrs={"long_name": "sea surface height"},
    )
    ds = xr.Dataset({"ssh": da}, attrs={"title": "synthetic ocean"})
    scaled = XarrayEstimator(StandardScaler(), sample_dim="time").fit_transform(da)
    fitted_pca = XarrayEstimator(PCA(n_components=2), sample_dim="time").fit(scaled)
    seq = Sequential(
        [
            ValidateCoords(),
            SklearnOp(StandardScaler(), variable="ssh", sample_dim="time"),
            SklearnOp(
                fitted_pca.estimator_,
                variable="ssh",
                output_variable="pcs",
                sample_dim="time",
                method="transform",
            ),
        ]
    )

    out = seq(ds)

    assert out.attrs == ds.attrs
    assert "ssh" in out
    assert "pcs" in out
    assert out["pcs"].dims == ("time", "component")
    np.testing.assert_array_equal(out["time"], ds["time"])
    np.testing.assert_array_equal(out["lat"], ds["lat"])
    np.testing.assert_array_equal(out["lon"], ds["lon"])
    assert out["ssh"].attrs == da.attrs
    json.dumps(seq.get_config())


def test_sklearn_op_config_handles_circular_estimator_params() -> None:
    class CircularEstimator:
        def get_params(self, deep: bool = False) -> dict[str, object]:
            params: dict[str, object] = {}
            params["self"] = params
            return params

    config = SklearnOp(CircularEstimator()).get_config()

    json.dumps(config)
