"""Operator wrapper for sklearn-style estimators."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, Literal

import xarray as xr

from xr_toolz.core import Operator
from xr_toolz.utils import XarrayEstimator
from xr_toolz.utils._src.sklearn_wrap import NanPolicy


SklearnMethod = Literal[
    "fit_transform",
    "transform",
    "predict",
    "predict_proba",
    "inverse_transform",
]


class SklearnOp(Operator):
    """Layer-1 operator that delegates sklearn marshalling to ``XarrayEstimator``.

    ``method="fit_transform"`` fits a fresh clone on each call. Other methods
    call the estimator as already fitted, matching sklearn's runtime contract.
    ``get_config()`` stores primitive estimator parameters directly and stores
    non-primitive parameters as ``repr(...)`` strings so the config remains JSON
    serializable.
    """

    def __init__(
        self,
        estimator: Any,
        *,
        variable: Hashable | None = None,
        output_variable: Hashable | None = None,
        sample_dim: Hashable | None = None,
        new_feature_dim: str = "component",
        nan_policy: NanPolicy = "propagate",
        method: SklearnMethod = "fit_transform",
    ) -> None:
        self.estimator = estimator
        self.variable = variable
        self.output_variable = output_variable
        self.sample_dim = sample_dim
        self.new_feature_dim = new_feature_dim
        self.nan_policy = nan_policy
        self.method = method

    def _apply(self, data: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
        target = (
            data[self.variable]
            if isinstance(data, xr.Dataset) and self.variable
            else data
        )
        out = self._run(target)
        if not isinstance(data, xr.Dataset):
            return out

        name = self.output_variable or self.variable
        if name is None:
            return out
        return data.assign({name: out})

    def _run(self, data: xr.DataArray | xr.Dataset) -> xr.DataArray:
        wrap = XarrayEstimator(
            self.estimator,
            sample_dim=self.sample_dim,
            new_feature_dim=self.new_feature_dim,
            nan_policy=self.nan_policy,
        )
        if self.method != "fit_transform":
            wrap.estimator_ = self.estimator
        method = getattr(wrap, self.method)
        return method(data)

    def get_config(self) -> dict[str, Any]:
        return {
            "estimator": self.estimator.__class__.__name__,
            "estimator_params": self._estimator_params(),
            "variable": self.variable,
            "output_variable": self.output_variable,
            "sample_dim": self.sample_dim,
            "new_feature_dim": self.new_feature_dim,
            "nan_policy": self.nan_policy,
            "method": self.method,
        }

    def _estimator_params(self) -> dict[str, Any]:
        if not hasattr(self.estimator, "get_params"):
            return {}
        return {
            key: _json_safe(value)
            for key, value in self.estimator.get_params(deep=False).items()
        }


def _json_safe(value: Any, *, _depth: int = 0) -> Any:
    if _depth > 10:
        return repr(value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, tuple):
        return [_json_safe(v, _depth=_depth + 1) for v in value]
    if isinstance(value, list):
        return [_json_safe(v, _depth=_depth + 1) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v, _depth=_depth + 1) for k, v in value.items()}
    return repr(value)
