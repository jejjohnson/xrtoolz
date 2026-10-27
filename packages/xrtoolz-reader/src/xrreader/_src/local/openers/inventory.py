"""Gridded methane emission inventories (EDGAR, GFEI, EPA) as one CF flux.

Inventories ship as lat/lon rasters in different units: EDGAR's ``flx``
files in kg m⁻² s⁻¹, GFEI's grids in Mg km⁻² a⁻¹. :func:`open_ch4_inventory`
returns every one of them as the same CF variable, ``ch4_emission_flux`` in
kg m⁻² s⁻¹, on ascending ``lat`` and ``lon`` in [-180, 180).

Only area fluxes are accepted. Per-cell totals (EDGAR's ``emi`` files, in
tonnes per cell per year or per month) need the cell area and the period,
and the period is not in the file, so they are rejected rather than
converted with a guessed period: use the ``flx`` files instead.
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from xrreader._src.local.openers._common import promote_lonlat
from xrreader.types import CH4_EMISSION_FLUX, apply_cf_attrs


#: Seconds in a Julian year (365.25 d), the year in the ``a⁻¹`` conversions.
SECONDS_PER_YEAR: float = 365.25 * 86_400.0

# 1 Mg km⁻² a⁻¹ = 10³ kg / 10⁶ m² / (365.25 · 86 400 s)
_PER_YEAR_MG_KM2 = 1.0e3 / 1.0e6 / SECONDS_PER_YEAR
#: Accepted flux units → factor to kg m⁻² s⁻¹ (spelling variants included).
UNIT_FACTORS: dict[str, float] = {
    "kg m-2 s-1": 1.0,
    "kg m**-2 s**-1": 1.0,
    "kg/m2/s": 1.0,
    "kg m^-2 s^-1": 1.0,
    "g m-2 s-1": 1.0e-3,
    "g/m2/s": 1.0e-3,
    "Mg km-2 a-1": _PER_YEAR_MG_KM2,
    "Mg km-2 yr-1": _PER_YEAR_MG_KM2,
    "Mg/km2/yr": _PER_YEAR_MG_KM2,
    "Mg/km2/a": _PER_YEAR_MG_KM2,
    "t km-2 a-1": _PER_YEAR_MG_KM2,
    "t km-2 yr-1": _PER_YEAR_MG_KM2,
}


def open_ch4_inventory(
    path: str | Path,
    *,
    variable: str | None = None,
    units: str | None = None,
    source: str | None = None,
) -> xr.Dataset:
    """Open a gridded CH₄ emission inventory as a flux in kg m⁻² s⁻¹.

    F [kg m⁻² s⁻¹] = f · F_file,  f = ``UNIT_FACTORS[units]``.

    Algorithm:
        1. Open the file and rename ``latitude`` / ``longitude`` to ``lat`` /
           ``lon``.
        2. Pick the flux variable: ``variable``, or the one data variable on
           both ``lat`` and ``lon``.
        3. Look up the factor f of its units (``units`` overrides the file's
           attribute) and scale.
        4. Wrap longitudes in [0, 360) to [-180, 180), sort both axes, and
           stamp the CF attributes over whatever the scaling kept.

    Args:
        path: NetCDF file of the inventory (one sector or a total).
        variable: Name of the flux variable; inferred when ``None``.
        units: Units of the flux, overriding the file's ``units`` attribute
            (for files that omit or misspell it).
        source: Free-text provenance (e.g. ``"EDGAR v8.0"``), stored in the
            ``source`` attribute.

    Returns:
        Dataset with ``ch4_emission_flux`` on ``([time,] lat, lon)`` in
        kg m⁻² s⁻¹, with ``units_in_file`` and ``conversion_factor``
        attributes recording what was converted.

    Raises:
        KeyError: If ``variable`` is not in the file, or no variable (or more
            than one) lies on ``lat`` and ``lon`` when inferring it.
        ValueError: If the units are missing or not an accepted area flux.

    Examples:
        A GFEI-style grid in Mg km⁻² a⁻¹ on 0–360° longitudes:

        >>> import tempfile, pathlib
        >>> import numpy as np
        >>> import xarray as xr
        >>> path = pathlib.Path(tempfile.mkdtemp()) / "gfei.nc"
        >>> attrs = {"units": "Mg km-2 a-1"}
        >>> xr.Dataset(
        ...     {"emis_ch4": (("lat", "lon"), np.ones((1, 2)), attrs)},
        ...     coords={"lat": [32.05], "lon": [257.85, 257.95]},
        ... ).to_netcdf(path)
        >>> flux = open_ch4_inventory(path)["ch4_emission_flux"]
        >>> flux["lon"].values.round(2).tolist(), flux.attrs["units"]
        ([-102.15, -102.05], 'kg m-2 s-1')
        >>> bool(np.allclose(flux, 1e3 / 1e6 / (365.25 * 86_400.0)))
        True

        Per-cell totals are refused rather than converted with a guessed
        period:

        >>> open_ch4_inventory(path, units="Tonnes")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: 'emis_ch4' is in 'Tonnes', which is not an accepted area flux
    """
    dataset = promote_lonlat(xr.open_dataset(path))
    name = _infer_flux_variable(dataset, path) if variable is None else variable
    if name not in dataset.data_vars:
        raise KeyError(f"{path!s} has no variable {name!r}")
    flux_in_file = dataset[name]
    units_in_file = units if units is not None else flux_in_file.attrs.get("units")
    factor = _factor(units_in_file, name)

    # F = f · F_file
    flux = flux_in_file * factor
    flux = _wrap_and_sort(flux)
    # Overwrite: under `xr.set_options(keep_attrs=True)` the product above
    # keeps the file's `units`, which would mislabel the converted values.
    flux = apply_cf_attrs(
        flux.rename(CH4_EMISSION_FLUX.name), CH4_EMISSION_FLUX, overwrite=True
    )
    flux.attrs["units_in_file"] = str(units_in_file)
    flux.attrs["conversion_factor"] = factor
    out = flux.to_dataset()
    out.attrs = dict(dataset.attrs)
    if source is not None:
        out.attrs["source"] = source
    return out


def _infer_flux_variable(dataset: xr.Dataset, path: str | Path) -> str:
    """The one data variable on both ``lat`` and ``lon``."""
    gridded = [
        str(name)
        for name, values in dataset.data_vars.items()
        if {"lat", "lon"} <= set(values.dims)
    ]
    if len(gridded) != 1:
        raise KeyError(
            f"{path!s}: expected one variable on (lat, lon) to infer the flux, "
            f"found {gridded!r}; pass `variable=`"
        )
    return gridded[0]


def _factor(units: object, name: str) -> float:
    """f such that F [kg m⁻² s⁻¹] = f · F_file, from the (normalised) units."""
    if units is None:
        raise ValueError(
            f"{name!r} has no `units` attribute; pass `units=` (one of "
            f"{sorted(UNIT_FACTORS)})"
        )
    # collapse runs of whitespace so "kg  m-2 s-1" matches
    key = " ".join(str(units).split())
    if key not in UNIT_FACTORS:
        raise ValueError(
            f"{name!r} is in {units!r}, which is not an accepted area flux "
            f"{sorted(UNIT_FACTORS)}. Per-cell totals (tonnes per cell) need a "
            "period the file does not state; use the inventory's flux files"
        )
    return UNIT_FACTORS[key]


def _wrap_and_sort(flux: xr.DataArray) -> xr.DataArray:
    """Longitudes into [-180, 180), both axes ascending."""
    lon = flux["lon"]
    if float(lon.max()) >= 180.0:
        # λ ← ((λ + 180) mod 360) − 180
        wrapped = ((lon + 180.0) % 360.0) - 180.0
        flux = flux.assign_coords(lon=wrapped)
    return flux.sortby("lat").sortby("lon")


__all__ = ["SECONDS_PER_YEAR", "UNIT_FACTORS", "open_ch4_inventory"]
