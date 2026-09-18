"""Tests for :mod:`xrtoolz.atm._src.wrf`."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrtoolz.atm import (
    destagger,
    open_wrfout,
    wrf_height_agl,
    wrf_temperature,
    wrf_time,
)
from xrtoolz.atm._src.wrf import KAPPA, P_REFERENCE

from .conftest import write_synthetic_wrfout


# ---------- destagger ----------------------------------------------------


def test_destagger_linear_field_is_exact():
    x_stag = 10.0 * (np.arange(7) - 0.5)
    da = xr.DataArray(
        3.0 + 0.25 * x_stag[None, :] + np.arange(2)[:, None],
        dims=("t", "x_stag"),
        coords={"x_stag": x_stag, "t": [0, 1], "flag": ("x_stag", list("abcdefg"))},
        attrs={"units": "m s-1"},
        name="U",
    )
    out = destagger(da, "x_stag")
    x_mass = 10.0 * np.arange(6)
    assert out.dims == ("t", "x")
    assert out.sizes["x"] == da.sizes["x_stag"] - 1
    np.testing.assert_allclose(
        out, 3.0 + 0.25 * x_mass[None, :] + np.arange(2)[:, None], rtol=1e-14
    )
    np.testing.assert_allclose(out["x"], x_mass)  # coordinate averaged
    assert "flag" not in out.coords  # non-numeric coord on the dim is dropped
    assert list(out["t"]) == [0, 1]
    assert out.attrs == {"units": "m s-1"}
    assert out.name == "U"


def test_destagger_explicit_new_dim_and_errors():
    da = xr.DataArray(np.arange(4.0), dims="face")
    out = destagger(da, "face", new_dim="centre")
    assert out.dims == ("centre",)
    np.testing.assert_allclose(out, [0.5, 1.5, 2.5])
    with pytest.raises(ValueError, match="not in DataArray dims"):
        destagger(da, "x_stag")


def test_destagger_stays_lazy():
    pytest.importorskip("dask")
    da = xr.DataArray(np.ones((3, 5)), dims=("t", "x_stag")).chunk({"t": 1})
    out = destagger(da, "x_stag")
    assert out.chunks is not None
    np.testing.assert_allclose(out.compute(), 1.0)


# ---------- wrf_time -----------------------------------------------------


STAMPS = np.array(
    ["2024-06-01T00:00:00", "2024-06-01T01:00:00", "2024-06-01T02:00:00"],
    dtype="datetime64[s]",
)


def _times_rows() -> np.ndarray:
    return np.array([str(t).replace("T", "_").encode() for t in STAMPS], dtype="S19")


def test_wrf_time_from_times_only():
    ds = xr.Dataset({"Times": (("Time",), _times_rows())})
    out = wrf_time(ds)
    assert out.dims == ("Time",)
    assert out.dtype == np.dtype("datetime64[s]")
    np.testing.assert_array_equal(out.values, STAMPS)


def test_wrf_time_from_times_char_matrix():
    rows = np.array(
        [np.frombuffer(str(t).replace("T", "_").encode(), dtype="S1") for t in STAMPS]
    )
    assert rows.shape == (3, 19)
    ds = xr.Dataset({"Times": (("Time", "DateStrLen"), rows)})
    np.testing.assert_array_equal(wrf_time(ds).values, STAMPS)


def test_wrf_time_from_xtime_minutes():
    ds = xr.Dataset(
        {
            "XTIME": (
                ("Time",),
                [0.0, 60.0, 120.0],
                {"units": "minutes since 2024-06-01 00:00:00"},
            )
        },
        attrs={"SIMULATION_START_DATE": "2024-06-01_00:00:00"},
    )
    np.testing.assert_array_equal(wrf_time(ds).values, STAMPS)


def test_wrf_time_from_xtime_datetime64():
    ds = xr.Dataset({"XTIME": (("Time",), STAMPS.astype("datetime64[ns]"))})
    out = wrf_time(ds)
    assert out.dtype == np.dtype("datetime64[s]")
    np.testing.assert_array_equal(out.values, STAMPS)


def test_wrf_time_both_present_agree():
    ds = xr.Dataset(
        {"Times": (("Time",), _times_rows()), "XTIME": (("Time",), [0.0, 60.0, 120.0])},
        attrs={"SIMULATION_START_DATE": "2024-06-01_00:00:00"},
    )
    both = wrf_time(ds)
    np.testing.assert_array_equal(both.values, wrf_time(ds.drop_vars("XTIME")).values)
    np.testing.assert_array_equal(both.values, wrf_time(ds.drop_vars("Times")).values)


def test_wrf_time_errors():
    with pytest.raises(ValueError, match="neither"):
        wrf_time(xr.Dataset({"U": (("Time",), [0.0])}))
    with pytest.raises(ValueError, match="SIMULATION_START_DATE"):
        wrf_time(xr.Dataset({"XTIME": (("Time",), [0.0, 60.0])}))


# ---------- open_wrfout on the synthetic file ------------------------------


def test_open_wrfout_dims_coords_and_attrs(synthetic_wrf):
    ds = open_wrfout(synthetic_wrf.path)
    spec = synthetic_wrf
    assert dict(ds.sizes) == {
        "time": spec.n_time,
        "level": spec.n_lev,
        "y": spec.n_y,
        "x": spec.n_x,
    }
    for name in ("u", "v", "w", "temperature", "pressure"):
        assert ds[name].dims == ("time", "level", "y", "x")
    assert ds["pbl_height"].dims == ("time", "y", "x")
    assert ds["terrain"].dims == ("y", "x")
    assert ds["z_agl"].dims == ("time", "level", "y", "x")
    assert ds["time"].dtype == np.dtype("datetime64[s]")
    np.testing.assert_array_equal(ds["time"].values, spec.times)
    np.testing.assert_allclose(ds["x"], spec.x_mass)
    np.testing.assert_allclose(ds["y"], spec.y_mass)
    np.testing.assert_array_equal(ds["level"], np.arange(spec.n_lev))
    assert ds["lon"].dims == ("y", "x") and ds["lat"].dims == ("y", "x")
    np.testing.assert_allclose(ds["lon"].isel(y=0), -3.0 + 0.01 * np.arange(spec.n_x))
    np.testing.assert_allclose(ds["lat"].isel(x=0), 40.0 + 0.009 * np.arange(spec.n_y))
    # Raw WRF coordinate variables do not leak through.
    assert not {"XLONG", "XLAT", "XTIME", "Times"} & set(ds.variables)
    for key, value in (
        ("DX", 1000.0),
        ("DY", 1000.0),
        ("MAP_PROJ", 1),
        ("CEN_LAT", 40.05),
        ("STAND_LON", -2.945),
    ):
        assert ds.attrs[key] == value
    assert "TRUELAT1" in ds.attrs and "TRUELAT2" in ds.attrs and "CEN_LON" in ds.attrs
    assert "TITLE" not in ds.attrs


def test_open_wrfout_physical_fields(synthetic_wrf):
    ds = open_wrfout(synthetic_wrf.path)
    spec = synthetic_wrf
    tt, zz, yy, xx = spec.grid()
    np.testing.assert_allclose(ds["z_agl"], zz, rtol=1e-12)
    np.testing.assert_allclose(ds["pressure"], P_REFERENCE - 8.0 * zz, rtol=1e-12)
    np.testing.assert_allclose(
        ds["temperature"],
        spec.theta(zz) * ((P_REFERENCE - 8.0 * zz) / P_REFERENCE) ** KAPPA,
        rtol=1e-12,
    )
    np.testing.assert_allclose(ds["u"], spec.u(xx, yy, zz, tt), rtol=1e-12)
    np.testing.assert_allclose(ds["v"], spec.v(xx, yy, zz, tt), rtol=1e-12)
    np.testing.assert_allclose(ds["w"], spec.w(xx, yy, zz, tt), rtol=1e-12)
    np.testing.assert_allclose(
        ds["pbl_height"], spec.pbl_height(xx[:, 0], yy[:, 0], tt[:, 0]), rtol=1e-12
    )
    np.testing.assert_allclose(ds["terrain"], spec.terrain)


def test_open_wrfout_cf_attrs(synthetic_wrf):
    ds = open_wrfout(synthetic_wrf.path)
    expected = {
        "u": ("eastward_wind", "m s-1"),
        "v": ("northward_wind", "m s-1"),
        "w": ("upward_air_velocity", "m s-1"),
        "temperature": ("air_temperature", "K"),
        "pressure": ("air_pressure", "Pa"),
        "pbl_height": ("atmosphere_boundary_layer_thickness", "m"),
        "terrain": ("surface_altitude", "m"),
        "z_agl": ("height", "m"),
        "lon": ("longitude", "degrees_east"),
        "lat": ("latitude", "degrees_north"),
    }
    for name, (standard_name, units) in expected.items():
        assert ds[name].attrs["standard_name"] == standard_name, name
        assert ds[name].attrs["units"] == units, name
        assert "stagger" not in ds[name].attrs
    assert ds["z_agl"].attrs["positive"] == "up"
    assert ds["x"].attrs["units"] == "m" and ds["y"].attrs["units"] == "m"


def test_open_wrfout_times_slice_and_raw_variables(synthetic_wrf):
    ds = open_wrfout(synthetic_wrf.path, times=slice(1, 3), variables=["QVAPOR"])
    spec = synthetic_wrf
    assert ds.sizes["time"] == 2
    np.testing.assert_array_equal(ds["time"].values, spec.times[1:3])
    _, zz, _, _ = spec.grid()
    np.testing.assert_allclose(
        ds["u"].isel(time=0), open_wrfout(spec.path)["u"].isel(time=1)
    )
    assert ds["QVAPOR"].dims == ("time", "level", "y", "x")
    np.testing.assert_allclose(ds["QVAPOR"], spec.qvapor(zz)[1:3], rtol=1e-12)
    with pytest.raises(KeyError):
        open_wrfout(spec.path, variables=["NOPE"])


@pytest.mark.parametrize("hgt_time_axis", [True, False])
def test_open_wrfout_hgt_with_and_without_time_axis(tmp_path, hgt_time_axis):
    spec = write_synthetic_wrfout(tmp_path / "wrfout.nc", hgt_time_axis=hgt_time_axis)
    ds = open_wrfout(spec.path)
    _, zz, _, _ = spec.grid()
    assert ds["terrain"].dims == ("y", "x")
    np.testing.assert_allclose(ds["terrain"], spec.terrain)
    np.testing.assert_allclose(ds["z_agl"], zz, rtol=1e-12)
    np.testing.assert_allclose(
        wrf_height_agl(xr.open_dataset(spec.path, decode_times=False)), zz, rtol=1e-12
    )


@pytest.mark.parametrize(("with_times", "with_xtime"), [(True, False), (False, True)])
def test_open_wrfout_time_axis_from_either_source(tmp_path, with_times, with_xtime):
    spec = write_synthetic_wrfout(
        tmp_path / "wrfout.nc", with_times=with_times, with_xtime=with_xtime
    )
    np.testing.assert_array_equal(open_wrfout(spec.path)["time"].values, spec.times)


def test_wrf_temperature_and_height_helpers_on_raw_file(synthetic_wrf):
    raw = xr.open_dataset(synthetic_wrf.path, decode_times=False)
    _, zz, _, _ = synthetic_wrf.grid()
    np.testing.assert_allclose(
        wrf_temperature(raw), synthetic_wrf.temperature(zz), rtol=1e-12
    )
    np.testing.assert_allclose(wrf_height_agl(raw), zz, rtol=1e-12)


@pytest.mark.dask
def test_open_wrfout_chunks_keep_everything_lazy(synthetic_wrf):
    pytest.importorskip("dask")
    lazy = open_wrfout(synthetic_wrf.path, chunks={"time": 1})
    for name, da in lazy.data_vars.items():
        assert da.chunks is not None, name
        if "time" in da.dims:
            assert da.chunks[da.dims.index("time")] == (1,) * synthetic_wrf.n_time, name
    assert lazy["z_agl"].chunks is not None
    eager = open_wrfout(synthetic_wrf.path)
    xr.testing.assert_allclose(lazy.compute(), eager)
