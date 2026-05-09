"""Grid-to-grid value resampling.

Deterministic refinement (:func:`refine`), aggregation
(:func:`coarsen`), and target-grid resampling (:func:`regrid_like`)
along one or more dimensions. Learned counterparts
(``Downscale``/``Upscale``) live in :mod:`.downscale`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

import numpy as np
import xarray as xr


def coarsen(
    ds: xr.Dataset | xr.DataArray,
    factor: dict[str, int],
    method: str = "mean",
    boundary: str = "trim",
) -> xr.Dataset | xr.DataArray:
    """Coarsen ``ds`` along one or more dimensions by integer factors.

    Thin wrapper around ``xr.Dataset.coarsen``.
    """
    coarsened = ds.coarsen(dim=factor, boundary=boundary)
    return getattr(coarsened, method)()


def coarsen_conservative(
    ds: xr.Dataset | xr.DataArray,
    factor: Mapping[str, int],
    *,
    lat: str = "lat",
    boundary: Literal["trim", "exact", "pad"] = "trim",
) -> xr.Dataset | xr.DataArray:
    """Area-weighted coarsen using cosine-of-latitude weights.

    This preserves cosine-latitude-weighted integrals for aligned, integer
    coarsening of regular latitude/longitude grids. Non-latitude dimensions
    use uniform weights, and missing values are skipped with weights
    renormalized within each block.
    """
    factor_dict = _validate_coarsen_factor(factor)
    if isinstance(ds, xr.Dataset):
        return ds.map(
            lambda da: _coarsen_conservative_dataset_variable(
                da, factor_dict, lat=lat, boundary=boundary
            )
        )
    return _coarsen_conservative_dataarray(ds, factor_dict, lat=lat, boundary=boundary)


def _coarsen_conservative_dataset_variable(
    da: xr.DataArray,
    factor: dict[str, int],
    *,
    lat: str,
    boundary: Literal["trim", "exact", "pad"],
) -> xr.DataArray:
    variable_factor = {dim: value for dim, value in factor.items() if dim in da.dims}
    if not variable_factor:
        return da
    return _coarsen_conservative_dataarray(
        da, variable_factor, lat=lat, boundary=boundary
    )


def _coarsen_conservative_dataarray(
    da: xr.DataArray,
    factor: dict[str, int],
    *,
    lat: str,
    boundary: Literal["trim", "exact", "pad"],
) -> xr.DataArray:
    if lat not in da.dims or lat not in factor:
        return da.coarsen(dim=factor, boundary=boundary).mean()

    _validate_lat_chunks(da, factor[lat], lat=lat)

    cos_lat = np.cos(np.deg2rad(da[lat]))
    weights = xr.DataArray(cos_lat.data, dims=(lat,), coords={lat: da[lat]})
    mask = xr.apply_ufunc(np.isfinite, da, dask="allowed")
    weighted = weights * mask
    numerator = (
        (da.where(mask, 0.0) * weighted).coarsen(dim=factor, boundary=boundary).sum()
    )
    denominator = weighted.coarsen(dim=factor, boundary=boundary).sum()
    valid = denominator > 0
    return numerator.where(valid) / denominator.where(valid)


def _validate_coarsen_factor(factor: Mapping[str, int]) -> dict[str, int]:
    factor_dict = dict(factor)
    for dim, value in factor_dict.items():
        if not isinstance(value, int) or value < 1:
            raise ValueError(
                f"coarsen factor for {dim!r} must be a positive int, got {value!r}."
            )
    return factor_dict


def _validate_lat_chunks(da: xr.DataArray, factor: int, *, lat: str) -> None:
    chunks = da.chunks
    if chunks is None:
        return
    axis = da.get_axis_num(lat)
    bad_chunks = [chunk for chunk in chunks[axis] if chunk % factor != 0]
    if bad_chunks:
        raise ValueError(
            f"conservative coarsen requires chunks along {lat!r} to be multiples "
            f"of factor {factor}; got chunks {chunks[axis]!r}."
        )


def refine(
    ds: xr.Dataset | xr.DataArray,
    factor: dict[str, int],
    method: str = "linear",
) -> xr.Dataset | xr.DataArray:
    """Refine ``ds`` along one or more dimensions by integer factors.

    Produces a ``factor[dim]``-times-denser grid along each dimension
    via :meth:`xr.Dataset.interp`.
    """
    new_coords: dict[str, Sequence[float]] = {}
    for dim, f in factor.items():
        old = ds[dim].values
        if f <= 0:
            raise ValueError(f"refinement factor for {dim!r} must be positive.")
        new_coords[dim] = np.linspace(old.min(), old.max(), (len(old) - 1) * f + 1)
    return ds.interp(new_coords, method=method)


def regrid_like(
    ds: xr.Dataset | xr.DataArray,
    target: xr.Dataset | xr.DataArray,
    *,
    dims: Iterable[str] = ("lat", "lon"),
    method: str = "linear",
) -> xr.Dataset | xr.DataArray:
    """Resample ``ds`` onto ``target``'s coordinates along ``dims``.

    Thin :meth:`xr.Dataset.interp` wrapper for the common
    "regrid model output to observation grid" step. Coordinates listed
    in ``dims`` must exist on both ``ds`` and ``target``; values along
    other dims pass through.
    """
    dim_list = list(dims)
    missing_target = [d for d in dim_list if d not in target.coords]
    if missing_target:
        raise ValueError(
            f"target is missing requested dims {missing_target!r} as coords; "
            f"got coords {tuple(target.coords)}. Pass `dims=` explicitly to "
            "regrid only the dims that are actually shared."
        )
    missing_source = [d for d in dim_list if d not in ds.coords]
    if missing_source:
        raise ValueError(
            f"input is missing requested dims {missing_source!r} as coords; "
            f"got coords {tuple(ds.coords)}."
        )
    target_coords = {d: target[d] for d in dim_list}
    return ds.interp(target_coords, method=method)
