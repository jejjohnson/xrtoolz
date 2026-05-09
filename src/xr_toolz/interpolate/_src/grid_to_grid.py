"""Grid-to-grid value resampling.

Deterministic refinement (:func:`refine`), aggregation
(:func:`coarsen`), and target-grid resampling (:func:`regrid_like`)
along one or more dimensions. Learned counterparts
(``Downscale``/``Upscale``) live in :mod:`.downscale`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
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


def refine_2d(
    da: xr.DataArray,
    *,
    factor: Mapping[str, int | float],
    lat: str = "lat",
    lon: str = "lon",
    order: int = 3,
    anti_aliasing: bool | None = None,
    mode: Literal["reflect", "constant", "edge", "symmetric", "wrap"] = "reflect",
    cval: float = 0.0,
) -> xr.DataArray:
    """Resize a 2-D ``(lat, lon)`` plate via ``skimage.transform.resize``.

    Order follows scikit-image's spline convention: ``0`` nearest, ``1``
    bilinear, ``2`` biquadratic, ``3`` bicubic, ``4`` biquartic, and ``5``
    biquintic. Leading dimensions are broadcast independently with
    :func:`xarray.apply_ufunc`.
    """
    resize = _get_skimage_resize()
    if lat not in da.dims or lon not in da.dims:
        raise ValueError(f"da must have dims {lat!r} and {lon!r}.")
    if lat not in factor or lon not in factor:
        raise ValueError(f"factor must include both {lat!r} and {lon!r}.")
    if isinstance(order, bool) or not isinstance(order, int):
        raise ValueError(f"order must be an integer in 0..5, got {order!r}.")
    if order not in range(6):
        raise ValueError(f"order must be in 0..5, got {order!r}.")

    f_lat = factor[lat]
    f_lon = factor[lon]
    if f_lat <= 0 or f_lon <= 0:
        raise ValueError(
            f"refinement factors for {lat!r} and {lon!r} must be positive."
        )

    n_lat = max(1, round(da.sizes[lat] * f_lat))
    n_lon = max(1, round(da.sizes[lon] * f_lon))
    new_lat = _interp_coord(da[lat].values, n_lat)
    new_lon = _interp_coord(da[lon].values, n_lon)

    def _resize_slice(arr2d: np.ndarray) -> np.ndarray:
        arr2d = np.asarray(arr2d, dtype=np.float64)
        return resize(
            arr2d,
            (n_lat, n_lon),
            order=order,
            anti_aliasing=anti_aliasing,
            mode=mode,
            cval=cval,
            preserve_range=True,
        )

    out = xr.apply_ufunc(
        _resize_slice,
        da,
        input_core_dims=[[lat, lon]],
        output_core_dims=[[lat, lon]],
        exclude_dims={lat, lon},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={
            "output_sizes": {lat: n_lat, lon: n_lon},
            "allow_rechunk": False,
        },
    )
    return out.assign_coords({lat: new_lat, lon: new_lon})


def _get_skimage_resize() -> Callable[..., np.ndarray]:
    try:
        from skimage.transform import resize
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "refine_2d requires scikit-image. "
            "Install with: pip install 'xr_toolz[image]'"
        ) from exc
    return resize


def _interp_coord(coord: np.ndarray, size: int) -> np.ndarray:
    old = np.asarray(coord)
    old_idx = np.arange(len(old))
    new_idx = np.linspace(0, len(old) - 1, size)
    return np.interp(new_idx, old_idx, old)


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
