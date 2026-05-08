"""Along-track geodesic helpers and wavelength-domain filtering."""

from __future__ import annotations

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from pyproj import Geod

from xr_toolz.interpolate import fir_filter


_GEOD = Geod(ellps="WGS84")


def median_dx_km(lon: ArrayLike, lat: ArrayLike) -> float:
    """Median geodesic spacing between consecutive points in kilometres.

    Args:
        lon: Longitude samples in degrees.
        lat: Latitude samples in degrees, with the same shape as ``lon``.

    Returns:
        Median WGS-84 geodesic segment length in kilometres.
    """
    lon_arr = np.asarray(lon, dtype=float).ravel()
    lat_arr = np.asarray(lat, dtype=float).ravel()
    if lon_arr.shape != lat_arr.shape:
        raise ValueError(
            f"lon and lat must have the same shape; got {lon_arr.shape} and "
            f"{lat_arr.shape}"
        )
    if lon_arr.size < 2:
        raise ValueError("at least two lon/lat points are required")

    distances_m = np.asarray(_GEOD.line_lengths(lon_arr, lat_arr), dtype=float)
    finite = distances_m[np.isfinite(distances_m)]
    if finite.size == 0:
        raise ValueError("no finite geodesic segment lengths found")
    return float(np.median(finite) / 1000.0)


def _cutoff_from_wavelength(lambda_km: float, spacing_km: float) -> float:
    if lambda_km <= 0:
        raise ValueError(f"wavelength must be > 0 km, got {lambda_km}")
    cutoff = 2.0 * spacing_km / lambda_km
    if not (0.0 < cutoff < 1.0):
        raise ValueError(
            f"wavelength {lambda_km} km is below the Nyquist limit for spacing "
            f"{spacing_km} km; expected lambda_km > {2.0 * spacing_km}"
        )
    return cutoff


def bandpass_wavelength(
    ds: xr.Dataset,
    *,
    dim: str,
    lambda_min_km: float | None = None,
    lambda_max_km: float | None = None,
    spacing_km: float | None = None,
    method: str = "lanczos",
    num_taps: int | None = None,
    attenuation_db: float | None = None,
    lon: str = "longitude",
    lat: str = "latitude",
) -> xr.Dataset:
    """Filter an along-track dataset using wavelength cutoffs in kilometres.

    Args:
        ds: Input dataset.
        dim: Along-track dimension to filter.
        lambda_min_km: Short-wavelength edge in kilometres. With
            ``lambda_max_km`` this becomes the high-frequency band edge.
        lambda_max_km: Long-wavelength edge in kilometres. With
            ``lambda_min_km`` this becomes the low-frequency band edge.
        spacing_km: Optional along-track sample spacing. If omitted, it is
            derived as the median WGS-84 spacing from ``ds[lon]`` / ``ds[lat]``.
        method: FIR window family, ``"lanczos"`` or ``"kaiser"``.
        num_taps: Optional odd FIR tap count.
        attenuation_db: Kaiser attenuation target in decibels.
        lon: Longitude variable name used when deriving spacing.
        lat: Latitude variable name used when deriving spacing.

    Returns:
        Dataset filtered along ``dim``. Variables without ``dim`` or with
        non-numeric dtype pass through unchanged.
    """
    if lambda_min_km is None and lambda_max_km is None:
        raise ValueError("at least one of lambda_min_km or lambda_max_km is required")
    if (
        lambda_min_km is not None
        and lambda_max_km is not None
        and lambda_min_km >= lambda_max_km
    ):
        raise ValueError(
            f"lambda_min_km must be < lambda_max_km; got {lambda_min_km} >= "
            f"{lambda_max_km}"
        )
    if spacing_km is None:
        spacing_km = median_dx_km(ds[lon], ds[lat])
    if spacing_km <= 0:
        raise ValueError(f"spacing_km must be > 0, got {spacing_km}")

    if lambda_min_km is None:
        if lambda_max_km is None:
            raise ValueError("lambda_max_km is required for low-pass filtering")
        btype = "low"
        cutoff: float | tuple[float, float] = _cutoff_from_wavelength(
            lambda_max_km, spacing_km
        )
    elif lambda_max_km is None:
        btype = "high"
        cutoff = _cutoff_from_wavelength(lambda_min_km, spacing_km)
    else:
        btype = "bandpass"
        cutoff = (
            _cutoff_from_wavelength(lambda_max_km, spacing_km),
            _cutoff_from_wavelength(lambda_min_km, spacing_km),
        )

    return fir_filter(
        ds,
        dim=dim,
        cutoff=cutoff,
        method=method,
        btype=btype,
        num_taps=num_taps,
        attenuation_db=attenuation_db,
    )


__all__ = ["bandpass_wavelength", "median_dx_km"]
