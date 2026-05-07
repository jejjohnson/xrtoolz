"""Operator wrapper for sklearn-style estimators."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, Literal

import numpy as np
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
_MAX_JSON_RECURSION_DEPTH = 10


class SklearnOp(Operator):
    """Layer-1 operator that delegates sklearn marshalling to ``XarrayEstimator``.

    ``method="transform"`` (default) and ``"predict"`` / ``"predict_proba"`` /
    ``"inverse_transform"`` require ``estimator`` to be already fitted — pass an
    sklearn estimator that has been fitted, or a fitted ``XarrayEstimator``.
    Passing a fitted ``XarrayEstimator`` is the only way to get the original
    ``(sample_dim, *feature_dims)`` grid back from ``inverse_transform``, since
    that path needs the training metadata captured at ``fit`` time.

    ``method="fit_transform"`` fits a fresh clone on each call. Avoid inside
    ``Sequential`` — re-fitting on every pipeline invocation is rarely what you
    want.

    ``get_config()`` stores primitive estimator parameters directly and stores
    non-primitive parameters as ``repr(...)`` strings so the config remains JSON
    serializable.

    Args:
        estimator: A raw sklearn estimator (``BaseEstimator``) or an
            ``XarrayEstimator``. For methods other than ``"fit_transform"`` it
            must already be fitted.
        variable: Name of the Dataset variable to operate on. Required when
            input is a multi-variable Dataset (or any Dataset for which the
            whole-Dataset stacked path is not desired).
        output_variable: Name of the variable to assign the result to. Defaults
            to ``variable``. When input is a Dataset and neither ``variable``
            nor ``output_variable`` is set, ``__call__`` raises ``ValueError``
            because the result would no longer be a Dataset and would break
            ``Sequential`` chains.
        sample_dim: xarray dim indexing samples. Defaults to the first dim.
        new_feature_dim: Name for the feature dim when the estimator changes
            feature count (e.g. PCA component count).
        nan_policy: ``"propagate"`` (default), ``"raise"``, or ``"mask"``. See
            :class:`XarrayEstimator`.
        method: Which sklearn-style method to call on each invocation.
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
        method: SklearnMethod = "transform",
    ) -> None:
        self.estimator = estimator
        self.variable = variable
        self.output_variable = output_variable
        self.sample_dim = sample_dim
        self.new_feature_dim = new_feature_dim
        self.nan_policy = nan_policy
        self.method = method

    def _apply(self, data: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
        if (
            isinstance(data, xr.Dataset)
            and self.variable is None
            and self.output_variable is None
        ):
            raise ValueError(
                "SklearnOp received a Dataset input but neither `variable` nor "
                "`output_variable` is set. The whole-Dataset path produces a "
                "DataArray, which would break a Sequential chain. Pass "
                "`variable=...` to select a single variable, or "
                "`output_variable=...` to name the result when stacking all "
                "variables together."
            )
        target = (
            data[self.variable]
            if isinstance(data, xr.Dataset) and self.variable is not None
            else data
        )
        out = self._run(target)
        if not isinstance(data, xr.Dataset):
            return out

        name = self.output_variable or self.variable
        return data.assign({name: out})

    def _run(self, data: xr.DataArray | xr.Dataset) -> xr.DataArray:
        wrap = self._resolve_wrap()
        method = getattr(wrap, self.method)
        return method(data)

    def _resolve_wrap(self) -> XarrayEstimator:
        """Return an ``XarrayEstimator`` appropriate for ``self.method``.

        Pre-fitted ``XarrayEstimator`` instances are reused as-is so their
        ``_fitted_meta_`` is preserved (needed for ``inverse_transform`` to
        recover the original feature grid). Raw sklearn estimators are wrapped
        on each call; for ``method="fit_transform"`` the wrapper fits a fresh
        clone, otherwise the estimator is treated as already fitted.
        """
        if isinstance(self.estimator, XarrayEstimator):
            return self.estimator
        wrap = XarrayEstimator(
            self.estimator,
            sample_dim=self.sample_dim,
            new_feature_dim=self.new_feature_dim,
            nan_policy=self.nan_policy,
        )
        if self.method != "fit_transform":
            wrap.estimator_ = self.estimator
        return wrap

    def get_config(self) -> dict[str, Any]:
        return {
            "estimator": self._estimator_class_name(),
            "estimator_params": self._estimator_params(),
            "variable": self.variable,
            "output_variable": self.output_variable,
            "sample_dim": self.sample_dim,
            "new_feature_dim": self.new_feature_dim,
            "nan_policy": self.nan_policy,
            "method": self.method,
        }

    def _estimator_class_name(self) -> str:
        if isinstance(self.estimator, XarrayEstimator):
            return self.estimator.estimator.__class__.__name__
        return self.estimator.__class__.__name__

    def _estimator_params(self) -> dict[str, Any]:
        # XarrayEstimator wraps the underlying sklearn estimator; pull params
        # off the wrapped instance so the config reflects what sklearn sees.
        target = (
            self.estimator.estimator
            if isinstance(self.estimator, XarrayEstimator)
            else self.estimator
        )
        if not hasattr(target, "get_params"):
            return {}
        return {
            key: _json_safe(value)
            for key, value in target.get_params(deep=False).items()
        }


def _json_safe(value: Any, *, _depth: int = 0) -> Any:
    if _depth > _MAX_JSON_RECURSION_DEPTH:
        return repr(value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, np.generic):
        # numpy scalar (np.int64, np.float64, np.bool_, ...) → native Python.
        return value.item()
    if isinstance(value, tuple):
        return [_json_safe(v, _depth=_depth + 1) for v in value]
    if isinstance(value, list):
        return [_json_safe(v, _depth=_depth + 1) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v, _depth=_depth + 1) for k, v in value.items()}
    return repr(value)
