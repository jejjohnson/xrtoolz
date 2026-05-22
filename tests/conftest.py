from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal

import pytest
import xarray as xr


ArrayBackend = Literal["numpy", "dask"]
MaybeChunk = Callable[
    [xr.DataArray | xr.Dataset, ArrayBackend, Mapping[str, int] | None],
    xr.DataArray | xr.Dataset,
]


@pytest.fixture(params=["numpy", "dask"])
def array_backend(request: pytest.FixtureRequest) -> ArrayBackend:
    assert request.param in {"numpy", "dask"}
    return request.param


def _maybe_chunk(
    obj: xr.DataArray | xr.Dataset,
    backend: ArrayBackend,
    chunks: Mapping[str, int] | None = None,
) -> xr.DataArray | xr.Dataset:
    if backend == "dask":
        pytest.importorskip("dask.array")
        return obj.chunk(chunks or {})
    return obj


@pytest.fixture
def maybe_chunk() -> MaybeChunk:
    return _maybe_chunk
