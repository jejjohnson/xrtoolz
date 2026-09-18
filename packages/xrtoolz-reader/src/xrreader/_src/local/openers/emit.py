"""EMIT L2B methane-enhancement opener.

Assumes the NetCDF container (the raster on ``(y, x)`` with the plume
enhancement and, optionally, its uncertainty). The distributed
``EMIT_L2B_CH4ENH`` GeoTIFFs are not read here — ``rioxarray`` is not an
``xrreader`` dependency — convert them to NetCDF first (e.g. with
``rioxarray.open_rasterio(...).to_netcdf``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from xrreader._src.local.openers._common import attach_scene_time, promote_lonlat
from xrreader.types import CH4_ENHANCEMENT, apply_cf_attrs


_RASTER_DIMS = {"downtrack": "y", "crosstrack": "x", "ortho_y": "y", "ortho_x": "x"}
_UNCERTAINTY = "ch4_uncertainty"


def open_emit_ch4_l2b(
    path: str | Path,
    *,
    glt_path: str | Path | None = None,
) -> xr.Dataset:
    """Open an EMIT L2B CH4 enhancement scene as a flat CF Dataset.

    The raster carries ``ch4_enhancement`` (ppm m) and, when present,
    ``ch4_uncertainty`` on ``(y, x)``. With ``glt_path`` the scene is
    orthorectified through EMIT's geometric lookup table: ``glt_x`` /
    ``glt_y`` hold the 1-based raw-pixel column / row that fills each
    orthorectified cell (``0`` marks no data), and the 2-D ``lon`` /
    ``lat`` of the orthorectified grid come from the GLT file's own
    ``lon`` / ``lat`` variables or, failing that, its GDAL ``geotransform``
    attribute. Without a GLT, 2-D ``lon`` / ``lat`` are taken from the
    raster file itself when it has them.

    Args:
        path: NetCDF raster with ``ch4_enhancement`` on ``(y, x)``
            (EMIT's ``downtrack`` / ``crosstrack`` dims are renamed).
        glt_path: Optional geometric lookup table NetCDF.

    Returns:
        Scene Dataset on ``(time, y, x)`` (``time`` of length 1 when the
        file exposes a timestamp) with 2-D ``lon`` / ``lat`` coordinates.

    Raises:
        KeyError: If ``ch4_enhancement`` is missing from the raster.
        ValueError: If the GLT exposes neither ``lon``/``lat`` nor a
            ``geotransform``.
    """
    ds = xr.open_dataset(path)
    ds = ds.rename({k: v for k, v in _RASTER_DIMS.items() if k in ds.dims})
    name = CH4_ENHANCEMENT.for_source("emit")
    if name not in ds.data_vars:
        raise KeyError(f"{path!s} has no {name!r} variable")
    ds = ds.rename({name: CH4_ENHANCEMENT.name}) if name != CH4_ENHANCEMENT.name else ds
    if glt_path is not None:
        ds = _orthorectify(ds, xr.open_dataset(glt_path))
    ds = promote_lonlat(ds)
    ds[CH4_ENHANCEMENT.name] = apply_cf_attrs(ds[CH4_ENHANCEMENT.name], CH4_ENHANCEMENT)
    return attach_scene_time(ds)


def _orthorectify(ds: xr.Dataset, glt: xr.Dataset) -> xr.Dataset:
    """Resample every ``(y, x)`` variable of ``ds`` onto the GLT grid."""
    glt = glt.rename({k: v for k, v in _RASTER_DIMS.items() if k in glt.dims})
    gx = glt["glt_x"].values.astype(np.int64)
    gy = glt["glt_y"].values.astype(np.int64)
    valid = (gx > 0) & (gy > 0)
    iy = np.where(valid, gy - 1, 0)
    ix = np.where(valid, gx - 1, 0)
    out: dict[str, tuple[tuple[str, str], np.ndarray, dict]] = {}
    for name, da in ds.data_vars.items():
        if set(da.dims) != {"y", "x"}:
            continue
        arr = da.transpose("y", "x").values[iy, ix].astype(float)
        arr[~valid] = np.nan
        out[str(name)] = (("y", "x"), arr, dict(da.attrs))
    if "lon" in glt.variables and "lat" in glt.variables:
        lon = glt["lon"].values
        lat = glt["lat"].values
    elif "geotransform" in glt.attrs:
        ulx, dx, _, uly, _, dy = (float(v) for v in glt.attrs["geotransform"])
        jj, ii = np.indices(gx.shape)
        lon = ulx + (ii + 0.5) * dx
        lat = uly + (jj + 0.5) * dy
    else:
        raise ValueError("GLT carries neither lon/lat variables nor a geotransform")
    coords = {
        "lon": (
            ("y", "x"),
            lon,
            {"standard_name": "longitude", "units": "degrees_east"},
        ),
        "lat": (
            ("y", "x"),
            lat,
            {"standard_name": "latitude", "units": "degrees_north"},
        ),
    }
    return xr.Dataset(out, coords=coords, attrs=dict(ds.attrs))
