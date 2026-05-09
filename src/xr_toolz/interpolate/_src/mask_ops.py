"""Binary mask-cleanup primitives for interpolation workflows."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import xarray as xr


try:
    _morph: Any = importlib.import_module("skimage.morphology")
except ImportError:  # pragma: no cover - exercised only without the image extra
    _morph = None


Footprint = int | str | np.ndarray
_FOOTPRINT_NAMES = {"disk", "square", "diamond", "star"}


def _require_skimage() -> Any:
    if _morph is None:
        raise ImportError(
            "Mask morphology requires scikit-image. "
            "Install with: pip install 'xr_toolz[image]'"
        )
    return _morph


def _resolve_footprint(footprint: Footprint) -> np.ndarray:
    """Resolve a compact footprint specification to a scikit-image footprint."""
    morph = _require_skimage()
    if isinstance(footprint, np.ndarray):
        return footprint
    if isinstance(footprint, bool):
        raise TypeError("footprint must not be a boolean")
    if isinstance(footprint, int):
        return morph.disk(footprint)
    if isinstance(footprint, str) and footprint in _FOOTPRINT_NAMES:
        if footprint == "square" and hasattr(morph, "footprint_rectangle"):
            return morph.footprint_rectangle((1, 1))
        return getattr(morph, footprint)(1)
    raise TypeError(
        "footprint must be an int, one of "
        f"{sorted(_FOOTPRINT_NAMES)!r}, or a numpy.ndarray; got {footprint!r}"
    )


def _validate_bool_mask(mask: xr.DataArray, lon: str, lat: str) -> None:
    if mask.dtype != bool:
        raise TypeError(f"mask must be boolean, got {mask.dtype}")
    missing = [dim for dim in (lat, lon) if dim not in mask.dims]
    if missing:
        raise ValueError(
            f"mask must have dims {lat!r} and {lon!r}; missing {tuple(missing)!r}"
        )


def _validate_area(area: int) -> None:
    if isinstance(area, bool) or not isinstance(area, int):
        raise TypeError(f"area must be an int, got {type(area).__name__}")
    if area < 1:
        raise ValueError(f"area (minimum pixel count) must be >= 1, got {area}")


def _remove_small_holes(m: np.ndarray, *, area: int) -> np.ndarray:
    morph = _require_skimage()
    if "max_size" in inspect.signature(morph.remove_small_holes).parameters:
        # scikit-image 0.26+ removes components with size <= max_size.
        # xr_toolz keeps the historical "smaller than area" contract, so
        # use area - 1 (e.g. area=1 removes nothing).
        return morph.remove_small_holes(m, max_size=area - 1)
    return morph.remove_small_holes(m, area_threshold=area)


def _remove_small_objects(m: np.ndarray, *, area: int) -> np.ndarray:
    morph = _require_skimage()
    if "max_size" in inspect.signature(morph.remove_small_objects).parameters:
        # scikit-image 0.26+ removes components with size <= max_size.
        # xr_toolz keeps the historical "smaller than area" contract, so
        # use area - 1 (e.g. area=1 removes nothing).
        return morph.remove_small_objects(m, max_size=area - 1)
    return morph.remove_small_objects(m, min_size=area)


def _wrap2d(
    mask: xr.DataArray,
    *,
    lon: str,
    lat: str,
    fn: Callable[[np.ndarray], np.ndarray],
) -> xr.DataArray:
    """Apply a ``(lat, lon) -> (lat, lon)`` binary operation slice-by-slice."""
    _require_skimage()
    _validate_bool_mask(mask, lon=lon, lat=lat)
    return xr.apply_ufunc(
        fn,
        mask,
        input_core_dims=[[lat, lon]],
        output_core_dims=[[lat, lon]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[bool],
        dask_gufunc_kwargs={"allow_rechunk": False},
    )


def remove_small_holes_2d(
    mask: xr.DataArray,
    *,
    area: int = 4,
    lon: str = "lon",
    lat: str = "lat",
) -> xr.DataArray:
    """Flip False connected components smaller than ``area`` to True.

    ``True`` means "masked / needs filling" in xr_toolz interpolation
    workflows, so this removes small unmasked islands inside larger gaps.

    Args:
        mask: Boolean mask with ``lat`` and ``lon`` dimensions.
        area: False regions smaller than this pixel count are filled.
        lon: Longitude dimension name.
        lat: Latitude dimension name.

    Returns:
        Same-shaped boolean mask with small False holes filled.
    """
    _validate_area(area)
    return _wrap2d(
        mask,
        lon=lon,
        lat=lat,
        fn=lambda m: _remove_small_holes(m, area=area),
    )


def remove_small_objects_2d(
    mask: xr.DataArray,
    *,
    area: int = 4,
    lon: str = "lon",
    lat: str = "lat",
) -> xr.DataArray:
    """Flip True connected components smaller than ``area`` to False.

    ``True`` means "masked / needs filling" in xr_toolz interpolation
    workflows, so this removes isolated masked specks.

    Args:
        mask: Boolean mask with ``lat`` and ``lon`` dimensions.
        area: True regions smaller than this pixel count are dropped.
        lon: Longitude dimension name.
        lat: Latitude dimension name.

    Returns:
        Same-shaped boolean mask with small True objects removed.
    """
    _validate_area(area)
    return _wrap2d(
        mask,
        lon=lon,
        lat=lat,
        fn=lambda m: _remove_small_objects(m, area=area),
    )


def binary_opening_2d(
    mask: xr.DataArray,
    *,
    footprint: Footprint = 1,
    lon: str = "lon",
    lat: str = "lat",
) -> xr.DataArray:
    """Morphologically open a 2-D boolean mask slice-by-slice.

    Args:
        mask: Boolean mask with ``lat`` and ``lon`` dimensions.
        footprint: Structuring element specification.
        lon: Longitude dimension name.
        lat: Latitude dimension name.

    Returns:
        Same-shaped boolean mask after erosion followed by dilation.
    """
    morph = _require_skimage()
    fp = _resolve_footprint(footprint)
    return _wrap2d(
        mask,
        lon=lon,
        lat=lat,
        fn=lambda m: morph.opening(m, footprint=fp),
    )


def binary_closing_2d(
    mask: xr.DataArray,
    *,
    footprint: Footprint = 1,
    lon: str = "lon",
    lat: str = "lat",
) -> xr.DataArray:
    """Morphologically close a 2-D boolean mask slice-by-slice.

    Args:
        mask: Boolean mask with ``lat`` and ``lon`` dimensions.
        footprint: Structuring element specification.
        lon: Longitude dimension name.
        lat: Latitude dimension name.

    Returns:
        Same-shaped boolean mask after dilation followed by erosion.
    """
    morph = _require_skimage()
    fp = _resolve_footprint(footprint)
    return _wrap2d(
        mask,
        lon=lon,
        lat=lat,
        fn=lambda m: morph.closing(m, footprint=fp),
    )


def clean_mask(
    mask: xr.DataArray,
    *,
    fill_holes_area: int | None = 4,
    drop_objects_area: int | None = None,
    closing_footprint: Footprint | None = None,
    opening_footprint: Footprint | None = None,
    lon: str = "lon",
    lat: str = "lat",
) -> xr.DataArray:
    """Run the common mask-despeckling pipeline.

    Steps run in fixed order when their corresponding keyword is not
    ``None``: remove small holes, remove small objects, binary closing,
    then binary opening.

    Args:
        mask: Boolean mask with ``lat`` and ``lon`` dimensions.
        fill_holes_area: Optional hole area threshold.
        drop_objects_area: Optional object area threshold.
        closing_footprint: Optional binary-closing footprint.
        opening_footprint: Optional binary-opening footprint.
        lon: Longitude dimension name.
        lat: Latitude dimension name.

    Returns:
        Same-shaped cleaned boolean mask.
    """
    out = mask
    if fill_holes_area is not None:
        out = remove_small_holes_2d(out, area=fill_holes_area, lon=lon, lat=lat)
    if drop_objects_area is not None:
        out = remove_small_objects_2d(out, area=drop_objects_area, lon=lon, lat=lat)
    if closing_footprint is not None:
        out = binary_closing_2d(out, footprint=closing_footprint, lon=lon, lat=lat)
    if opening_footprint is not None:
        out = binary_opening_2d(out, footprint=opening_footprint, lon=lon, lat=lat)
    return out


__all__ = [
    "binary_closing_2d",
    "binary_opening_2d",
    "clean_mask",
    "remove_small_holes_2d",
    "remove_small_objects_2d",
]
