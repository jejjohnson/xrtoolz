"""Tests for the periodic-boundary option."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrgrad import curl, divergence, gradient, laplacian, partial


def _sine_on_full_period(n: int = 32, length: float = 4.0) -> xr.DataArray:
    """``sin(2πx/L)`` sampled over exactly one period (endpoint excluded)."""
    x = np.linspace(0.0, length, n, endpoint=False)
    return xr.DataArray(
        np.sin(2 * np.pi * x / length),
        dims=("x",),
        coords={"x": x},
        name="f",
    )


def _analytic_cosine(da: xr.DataArray, length: float = 4.0) -> np.ndarray:
    x = da["x"].values
    return (2 * np.pi / length) * np.cos(2 * np.pi * x / length)


@pytest.mark.parametrize("accuracy", [1, 2, 3, 4])
def test_periodic_edges_match_interior_accuracy(accuracy):
    """The seam must be as accurate as the interior — pins the halo size.

    An undersized halo would leave one-sided stencils at the first and
    last points, so their error would jump by orders of magnitude.
    """
    da = _sine_on_full_period()
    got = partial(da, "x", periodic=True, accuracy=accuracy)
    err = np.abs(got.values - _analytic_cosine(da))
    edge = max(err[0], err[-1])
    interior = err[2:-2].max()
    assert edge <= 5.0 * interior, (
        f"accuracy={accuracy}: edge error {edge:g} >> interior {interior:g}; "
        "halo is too small, seam kept one-sided stencils"
    )


def test_periodic_improves_on_non_periodic_at_the_seam():
    da = _sine_on_full_period()
    exact = _analytic_cosine(da)
    wrapped = np.abs(partial(da, "x", periodic=True).values - exact)
    plain = np.abs(partial(da, "x").values - exact)
    assert wrapped[0] < plain[0]
    assert wrapped[-1] < plain[-1]


def test_periodic_leaves_interior_unchanged():
    """Wrapping only changes points whose stencil reaches past an edge."""
    da = _sine_on_full_period()
    wrapped = partial(da, "x", periodic=True)
    plain = partial(da, "x")
    np.testing.assert_allclose(wrapped.values[2:-2], plain.values[2:-2], atol=0.0)


def test_non_periodic_default_is_unchanged():
    da = _sine_on_full_period()
    np.testing.assert_allclose(
        partial(da, "x").values, partial(da, "x", periodic=False).values, atol=0.0
    )


def _global_field(nlon: int = 72, nlat: int = 19) -> xr.Dataset:
    """A global lon/lat grid spanning the full 360°, plus a smooth field."""
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False)
    lat = np.linspace(-60.0, 60.0, nlat)
    grid_lon, grid_lat = np.meshgrid(lon, lat, indexing="xy")
    coords = {"lat": lat, "lon": lon}
    return xr.Dataset(
        {
            "u": xr.DataArray(
                np.sin(np.deg2rad(grid_lon)) * np.cos(np.deg2rad(grid_lat)),
                dims=("lat", "lon"),
                coords=coords,
            ),
            "v": xr.DataArray(
                np.cos(np.deg2rad(grid_lon)) * np.cos(np.deg2rad(grid_lat)),
                dims=("lat", "lon"),
                coords=coords,
            ),
        }
    )


def test_spherical_periodic_lon_removes_dateline_seam():
    ds = _global_field()
    plain = partial(ds["u"], "lon", geometry="spherical")
    wrapped = partial(ds["u"], "lon", geometry="spherical", periodic=True)
    # Analytic: ∂/∂x of sin(λ)cos(φ) = cos(λ)cos(φ) / (R cos φ) = cos(λ)/R.
    lon_rad = np.deg2rad(ds["lon"].values)
    exact = np.cos(lon_rad)[None, :] / 6_371_000.0
    seam = [0, -1]
    assert (
        np.abs(wrapped.values[:, seam] - exact[:, seam]).max()
        < np.abs(plain.values[:, seam] - exact[:, seam]).max()
    )


def test_spherical_periodic_curl_of_gradient_is_zero_at_the_seam():
    """``∇×∇f ≈ 0`` including the wrap columns."""
    ds = _global_field()
    grad = gradient(
        ds["u"].rename("f"), geometry="spherical", periodic="lon", dims=("lon", "lat")
    )
    z = curl(
        grad,
        tuple(grad.data_vars),
        dims=("lon", "lat"),
        geometry="spherical",
        periodic="lon",
    )
    seam = np.abs(z.values[1:-1, [0, -1]]).max()
    interior = np.abs(z.values[1:-1, 2:-2]).max()
    assert seam <= 5.0 * max(interior, 1e-30)


def test_partial_span_longitude_raises():
    ds = _global_field().isel(lon=slice(0, 30))  # regional, not global
    with pytest.raises(ValueError, match="full 360"):
        partial(ds["u"], "lon", geometry="spherical", periodic=True)


def test_periodic_latitude_raises():
    ds = _global_field()
    with pytest.raises(NotImplementedError, match="latitude does not wrap"):
        partial(ds["u"], "lat", geometry="spherical", periodic=True)


def test_periodic_on_non_uniform_rectilinear_raises():
    x = np.array([0.0, 0.5, 1.5, 3.5, 8.0])
    da = xr.DataArray(x**2, dims=("x",), coords={"x": x}, name="f")
    with pytest.raises(NotImplementedError, match="non-uniform"):
        partial(da, "x", geometry="rectilinear", periodic=True)


def test_periodic_on_uniform_rectilinear_delegates_to_cartesian():
    da = _sine_on_full_period()
    rect = partial(da, "x", geometry="rectilinear", periodic=True)
    cart = partial(da, "x", geometry="cartesian", periodic=True)
    np.testing.assert_allclose(rect.values, cart.values, atol=0.0)


def test_unknown_periodic_dim_name_raises():
    da = _sine_on_full_period()
    with pytest.raises(ValueError, match="not among"):
        gradient(da, dims=("x",), periodic="nope")


def test_periodic_accepts_str_and_sequence():
    ds = _global_field()
    one = divergence(
        ds, ("u", "v"), dims=("lon", "lat"), geometry="spherical", periodic="lon"
    )
    many = divergence(
        ds, ("u", "v"), dims=("lon", "lat"), geometry="spherical", periodic=["lon"]
    )
    np.testing.assert_allclose(one.values, many.values, atol=0.0)


def test_laplacian_accepts_periodic():
    """Cartesian laplacian wraps through its order-2 fast path."""
    da = _sine_on_full_period()
    wrapped = laplacian(da, dims=("x",), periodic="x")
    exact = -((2 * np.pi / 4.0) ** 2) * da.values
    err_edge = max(
        abs(wrapped.values[0] - exact[0]), abs(wrapped.values[-1] - exact[-1])
    )
    err_interior = np.abs(wrapped.values[2:-2] - exact[2:-2]).max()
    assert err_edge <= 5.0 * err_interior
