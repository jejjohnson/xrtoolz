"""Multi-granule mosaicking — merge georeferenced tiles into one cube.

Thin wrapper around :mod:`rioxarray.merge`. Inputs that carry differing
CRSs are reprojected onto the first input's CRS before the merge, so a
mixed-CRS granule list produces one coherent output rather than a
silently misaligned union.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import rioxarray  # noqa: F401  — needed so ds.rio is populated
import xarray as xr


MosaicMethod = Literal["first", "last", "min", "max", "sum"]

_VALID_METHODS: tuple[str, ...] = ("first", "last", "min", "max", "sum")


def _require_crs(obj: xr.DataArray | xr.Dataset, index: int):
    """Return ``obj``'s CRS, raising if it has none."""
    crs = obj.rio.crs
    if crs is None:
        raise ValueError(
            f"input {index} has no CRS; every mosaic input must carry one "
            "(set it with `xrtoolz.geo.assign_crs` or `ds.rio.write_crs`)."
        )
    return crs


def spatial_mosaic(
    datasets: Sequence[xr.DataArray] | Sequence[xr.Dataset],
    *,
    method: MosaicMethod = "first",
    resampling: str = "nearest",
    nodata: float | None = None,
) -> xr.DataArray | xr.Dataset:
    """Merge overlapping / adjacent georeferenced inputs into a single cube.

    Wraps :func:`rioxarray.merge.merge_arrays` (DataArray inputs) or
    :func:`rioxarray.merge.merge_datasets` (Dataset inputs). All inputs
    must carry a CRS (``ds.rio.crs``); any input whose CRS differs from
    the first input's is reprojected onto it before merging. Overlapping
    pixels are resolved by ``method``. The result spans the union of the
    input extents.

    Args:
        datasets: One or more DataArrays *or* one or more Datasets — the
            sequence must be homogeneous. Each must have a CRS attached.
            A single-element sequence is returned unchanged.
        method: Overlap resolution rule — one of ``"first"``, ``"last"``,
            ``"min"``, ``"max"``, ``"sum"``. ``"first"`` keeps the value
            from the earliest input covering a pixel.
        resampling: Name of the :class:`rasterio.enums.Resampling` member
            used when harmonising inputs whose CRS differs from the first
            input's. Ignored when all inputs already share a CRS. Default
            ``"nearest"`` (safe for categorical data; use ``"bilinear"``
            for continuous fields).
        nodata: Fill value for cells covered by no input. ``None`` defers
            to each input's own ``rio.nodata``.

    Returns:
        A single DataArray (for DataArray inputs) or Dataset (for Dataset
        inputs) spanning the union of the input extents, in the first
        input's CRS.

    Raises:
        ValueError: If ``datasets`` is empty, mixes DataArrays and
            Datasets, names an unknown ``method`` / ``resampling``, or if
            any input lacks a CRS.

    Example:
        >>> import numpy as np, xarray as xr
        >>> import rioxarray  # noqa: F401
        >>> from xrtoolz.geo import spatial_mosaic
        >>> def tile(x0):
        ...     # North-up: y descends, as in every georeferenced raster.
        ...     return xr.DataArray(
        ...         np.ones((4, 4)),
        ...         dims=("y", "x"),
        ...         coords={"y": np.arange(3.0, -1.0, -1.0), "x": x0 + np.arange(4.0)},
        ...     ).rio.write_crs("EPSG:4326")
        >>> mosaic = spatial_mosaic([tile(0.0), tile(4.0)])
        >>> int(mosaic.sizes["x"])
        8

    """
    from rasterio.enums import Resampling
    from rioxarray.merge import merge_arrays, merge_datasets

    items = list(datasets)
    if not items:
        raise ValueError("spatial_mosaic requires at least one input dataset.")

    if not hasattr(Resampling, resampling):
        valid = [r.name for r in Resampling]
        raise ValueError(f"Unknown resampling {resampling!r}; expected one of {valid}.")

    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown method {method!r}; expected one of {list(_VALID_METHODS)}."
        )

    is_dataset = [isinstance(item, xr.Dataset) for item in items]
    if any(is_dataset) and not all(is_dataset):
        raise ValueError(
            "spatial_mosaic requires a homogeneous sequence — either all "
            "DataArrays or all Datasets, not a mix."
        )

    target_crs = _require_crs(items[0], 0)
    harmonised = [items[0]]
    for index, item in enumerate(items[1:], start=1):
        crs = _require_crs(item, index)
        if crs == target_crs:
            harmonised.append(item)
        else:
            harmonised.append(
                item.rio.reproject(
                    target_crs, resampling=getattr(Resampling, resampling)
                )
            )

    if len(harmonised) == 1:
        return harmonised[0]

    # rioxarray's merge_* apply `method` on the already-aligned reprojected
    # sources, so `resampling` above is the only place it is consumed.
    merge = merge_datasets if all(is_dataset) else merge_arrays
    return merge(harmonised, method=method, nodata=nodata)


__all__ = ["MosaicMethod", "spatial_mosaic"]
