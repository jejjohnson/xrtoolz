"""Tier A — array kernels for axis remapping (D11, D12).

Per design decision D11, every arithmetic submodule grows a duck-array
``axis=`` entry point. ``remap_axis`` interpolates values from a source
1D coordinate vector to a target 1D coordinate vector along a chosen
axis, preserving all other dimensions.

Backend: numpy. Methods: ``"linear"`` (np.interp), ``"nearest"``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def remap_axis(
    values: ArrayLike,
    *,
    axis: int = -1,
    source_coords: ArrayLike,
    target_coords: ArrayLike,
    method: str = "linear",
) -> NDArray[np.floating]:
    """Interpolate values along ``axis`` from ``source_coords`` to ``target_coords``.

    Parameters
    ----------
    values
        Array with one axis whose length matches ``source_coords``.
    axis
        Axis to remap along.
    source_coords
        1D monotonic (ascending or descending) coordinate vector.
    target_coords
        1D coordinate vector to interpolate to.
    method
        ``"linear"`` or ``"nearest"``.

    Returns
    -------
    NDArray
        Same shape as ``values`` except along ``axis``, which is
        replaced by ``len(target_coords)``. Targets outside the source
        range produce NaN (no extrapolation).
    """
    arr = np.asarray(values, dtype=float)
    src = np.asarray(source_coords, dtype=float)
    tgt = np.asarray(target_coords, dtype=float)

    if src.ndim != 1 or tgt.ndim != 1:
        raise ValueError("source_coords and target_coords must be 1D")
    if arr.shape[axis] != src.size:
        raise ValueError(
            f"values.shape[axis]={arr.shape[axis]} but len(source_coords)={src.size}"
        )

    # Normalize to ascending source for np.interp / searchsorted.
    diffs = np.diff(src)
    if np.all(diffs > 0):
        ascending = True
    elif np.all(diffs < 0):
        ascending = False
        src = src[::-1]
    else:
        raise ValueError("source_coords must be strictly monotonic")

    moved = np.moveaxis(arr, axis, -1)
    if not ascending:
        moved = moved[..., ::-1]

    flat = moved.reshape(-1, src.size)
    out = np.empty((flat.shape[0], tgt.size), dtype=float)

    if method == "linear":
        for i in range(flat.shape[0]):
            out[i] = np.interp(tgt, src, flat[i], left=np.nan, right=np.nan)
    elif method == "nearest":
        idx = np.abs(src[None, :] - tgt[:, None]).argmin(axis=1)  # (M,)
        out[:] = flat[:, idx]
        oor = (tgt < src.min()) | (tgt > src.max())
        if oor.any():
            out[:, oor] = np.nan
    else:
        raise ValueError(f"unknown method {method!r}; expected 'linear' or 'nearest'")

    out = out.reshape(*moved.shape[:-1], tgt.size)
    return np.moveaxis(out, -1, axis)


__all__ = ["remap_axis"]
