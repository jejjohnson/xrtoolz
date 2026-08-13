"""Tests for higher-order derivatives (``order=``) and the cartesian Laplacian."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import xarray as xr

from xrgrad import gradient, laplacian, partial


def _cubic(n: int = 32) -> xr.DataArray:
    """Return ``f(x) = x³`` on a uniform 1-D grid."""
    x = np.linspace(0.0, 4.0, n)
    return xr.DataArray(x**3, dims=("x",), coords={"x": x}, name="f")


def _paraboloid(n: int = 24) -> xr.DataArray:
    """Return ``f(x, y) = x² + y²`` on a uniform 2-D grid."""
    x = np.linspace(0.0, 3.0, n)
    y = np.linspace(-1.0, 2.0, n)
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    return xr.DataArray(
        grid_x**2 + grid_y**2,
        dims=("x", "y"),
        coords={"x": x, "y": y},
        name="f",
    )


def test_order_two_matches_analytic_second_derivative():
    """``d²(x³)/dx² = 6x`` on interior points at accuracy=2."""
    da = _cubic()
    d2 = partial(da, "x", order=2, accuracy=2)
    expected = 6.0 * da["x"].values
    np.testing.assert_allclose(d2.values[1:-1], expected[1:-1], atol=1e-8)


def test_order_two_is_exact_for_quadratic_at_default_accuracy():
    """The 3-point second-derivative stencil is exact for a parabola."""
    x = np.linspace(0.0, 5.0, 16)
    da = xr.DataArray(x**2, dims=("x",), coords={"x": x}, name="f")
    np.testing.assert_allclose(partial(da, "x", order=2).values, 2.0, atol=1e-9)


def test_order_two_output_name_uses_d2_prefix_and_suffix():
    d2 = partial(_cubic(), "x", order=2)
    assert d2.name == "d2f_dx2"


def test_order_one_output_name_unchanged():
    assert partial(_cubic(), "x").name == "df_dx"


def test_order_two_preserves_dims_coords_and_attrs():
    da = _cubic()
    da.attrs["units"] = "m"
    d2 = partial(da, "x", order=2)
    assert d2.dims == da.dims
    assert d2.attrs == {"units": "m"}
    np.testing.assert_array_equal(d2["x"].values, da["x"].values)


@pytest.mark.parametrize("bad_order", [0, -1, 1.5, True])
def test_invalid_order_raises(bad_order):
    with pytest.raises(ValueError, match="positive integer"):
        partial(_cubic(), "x", order=bad_order)


def test_order_two_on_spherical_raises_not_implemented():
    lon = np.linspace(0.0, 10.0, 8)
    lat = np.linspace(0.0, 10.0, 8)
    da = xr.DataArray(
        np.zeros((lat.size, lon.size)),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="f",
    )
    with pytest.raises(NotImplementedError, match="laplacian"):
        partial(da, "lon", order=2, geometry="spherical")


def test_order_two_on_non_uniform_rectilinear_raises_not_implemented():
    x = np.array([0.0, 0.5, 1.5, 3.5, 8.0])
    da = xr.DataArray(x**2, dims=("x",), coords={"x": x}, name="f")
    with pytest.raises(NotImplementedError, match="non-uniform"):
        partial(da, "x", order=2, geometry="rectilinear")


def test_order_two_on_uniform_rectilinear_delegates_to_cartesian():
    """A uniform coord takes the cartesian fast path, so order=2 works."""
    da = _cubic()
    rect = partial(da, "x", order=2, accuracy=2, geometry="rectilinear")
    cart = partial(da, "x", order=2, accuracy=2, geometry="cartesian")
    np.testing.assert_allclose(rect.values, cart.values, atol=0.0)


def test_gradient_has_no_order_parameter():
    """Gradients are first-order by definition — the signature stays clean."""
    assert "order" not in inspect.signature(gradient).parameters


def test_cartesian_laplacian_exact_for_paraboloid_at_default_accuracy():
    """``Δ(x² + y²) = 4`` on interior points without needing accuracy=2."""
    f = _paraboloid()
    lap = laplacian(f, dims=("x", "y"))
    np.testing.assert_allclose(lap.values[1:-1, 1:-1], 4.0, atol=1e-9)


def test_cartesian_laplacian_matches_sum_of_second_partials():
    f = _paraboloid()
    lap = laplacian(f, dims=("x", "y"), accuracy=2)
    expected = partial(f, "x", order=2, accuracy=2) + partial(
        f, "y", order=2, accuracy=2
    )
    np.testing.assert_allclose(lap.values, expected.values, atol=0.0)


def test_cartesian_laplacian_returns_unnamed_dataarray():
    """Matches the divergence-composition path, which renames to None."""
    assert laplacian(_paraboloid(), dims=("x", "y")).name is None


def test_laplacian_empty_dims_raises():
    with pytest.raises(ValueError, match="at least one dimension"):
        laplacian(_paraboloid(), dims=())


def test_float32_input_preserves_dtype():
    """Covers the ``_x64_scope(False)`` branch, previously untested."""
    x = np.linspace(0.0, 4.0, 32, dtype=np.float32)
    da = xr.DataArray((x**2).astype(np.float32), dims=("x",), coords={"x": x})
    assert partial(da, "x").dtype == np.float32
    assert partial(da, "x", order=2).dtype == np.float32


def test_float64_input_preserves_dtype():
    da = _cubic()
    assert partial(da, "x").dtype == np.float64
    assert partial(da, "x", order=2).dtype == np.float64
