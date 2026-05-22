from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import pytest
import xarray as xr


ArrayBackend = Literal["numpy", "dask"]


@pytest.fixture(params=["numpy", "dask"])
def array_backend(request: pytest.FixtureRequest) -> ArrayBackend:
    return request.param  # type: ignore[return-value]


def maybe_chunk(
    obj: xr.DataArray | xr.Dataset,
    backend: ArrayBackend,
    chunks: Mapping[str, int] | None = None,
) -> xr.DataArray | xr.Dataset:
    if backend == "dask":
        pytest.importorskip("dask.array")
        return obj.chunk(chunks or {})
    return obj
