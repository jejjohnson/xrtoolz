"""Shape-signature propagation and structural summary tests."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xr_toolz.core import Graph, Input, Operator, Sequential, Signature
from xr_toolz.geo.operators import (
    CalculateClimatology,
    Reduce,
    RenameCoords,
    RenameVariables,
    SelectVariables,
    SubsetBBox,
    SubsetTime,
)
from xr_toolz.interpolate.operators import Coarsen, Refine, RegridLike, ResampleTime
from xr_toolz.metrics.operators import RMSE


def test_signature_renders_unknown_dimensions() -> None:
    signature = Signature({"time": None, "lat": 3}, dtype="float32")

    assert signature.format() == "(time=?, lat=3); dtype=float32"


def test_operator_default_signature_is_shape_preserving() -> None:
    signature = Signature({"time": 10}, dtype="float32")

    assert Operator().compute_output_signature(signature) == signature


def test_geo_shape_changing_operator_signatures() -> None:
    signature = Signature({"time": 365, "lat": 181, "lon": 360}, dtype="float32")

    renamed = RenameCoords({"lon": "longitude"}).compute_output_signature(signature)
    assert renamed == Signature(
        {"time": 365, "lat": 181, "longitude": 360},
        dtype="float32",
    )
    assert (
        RenameVariables({"tas": "air_temperature"}).compute_output_signature(signature)
        == signature
    )

    subset = SubsetBBox((-125, -65), (25, 50)).compute_output_signature(signature)
    assert subset == Signature({"time": 365, "lat": None, "lon": None}, "float32")

    time_subset = SubsetTime("2000-01-01", "2000-01-31").compute_output_signature(
        signature
    )
    assert time_subset == Signature({"time": None, "lat": 181, "lon": 360}, "float32")

    reduced = Reduce(dim=("time", "lat")).compute_output_signature(signature)
    assert reduced == Signature({"lon": 360}, "float32")

    assert SelectVariables("tas").compute_output_signature(signature) == signature

    climatology = CalculateClimatology(freq="month").compute_output_signature(signature)
    assert climatology == Signature(
        {"month": None, "lat": 181, "lon": 360},
        "float32",
    )


def test_interpolation_shape_changing_operator_signatures() -> None:
    signature = Signature({"time": 24, "lat": 10, "lon": 20}, dtype="float32")
    target = xr.Dataset(
        coords={
            "lat": np.linspace(-90, 90, 5),
            "lon": np.linspace(0, 360, 8, endpoint=False),
        }
    )

    assert ResampleTime("1D").compute_output_signature(signature) == Signature(
        {"time": None, "lat": 10, "lon": 20},
        "float32",
    )
    assert RegridLike(target).compute_output_signature(signature) == Signature(
        {"time": 24, "lat": 5, "lon": 8},
        "float32",
    )
    assert Coarsen({"lat": 2, "lon": 5}).compute_output_signature(signature) == (
        Signature({"time": 24, "lat": 5, "lon": 4}, "float32")
    )
    assert Coarsen({"lat": 3}, boundary="pad").compute_output_signature(
        signature
    ) == Signature({"time": 24, "lat": 4, "lon": 20}, "float32")
    with pytest.raises(ValueError, match="boundary='exact'"):
        Coarsen({"lat": 3}, boundary="exact").compute_output_signature(signature)
    assert Refine({"lat": 2}).compute_output_signature(signature) == Signature(
        {"time": 24, "lat": 19, "lon": 20},
        "float32",
    )


def test_sequential_summary_rolls_up_step_signatures() -> None:
    pipeline = Sequential(
        [
            SubsetBBox((-125, -65), (25, 50)),
            ResampleTime("1D"),
        ]
    )
    signature = Signature({"time": 365, "lat": 181, "lon": 360}, "float32")

    text = pipeline.summary(signature)

    assert "Sequential (2 ops)" in text
    assert "SubsetBBox" in text
    assert "ResampleTime" in text
    assert "(time=365, lat=181, lon=360); dtype=float32" in text
    assert "(time=?, lat=?, lon=?); dtype=float32" in text


def test_graph_summary_propagates_multi_input_metric_signature() -> None:
    pred = Input("pred")
    ref = Input("ref")
    score = RMSE(variable="tas", dims=("time", "lat"))(pred, ref)
    graph = Graph(inputs={"pred": pred, "ref": ref}, outputs={"score": score})
    signature = Signature({"time": 12, "lat": 4, "lon": 6}, "float32")

    output = graph.compute_output_signature({"pred": signature, "ref": signature})
    text = graph.summary({"pred": signature, "ref": signature})

    assert output == {"score": Signature({"lon": 6}, "float")}
    assert "Graph (2 inputs, 1 outputs)" in text
    assert "RMSE" in text
    assert "(lon=6); dtype=float" in text


def test_metric_signature_rejects_mismatched_inputs() -> None:
    op = RMSE(variable="tas", dims=("time",))
    pred = Signature({"time": 12, "lat": 4}, "float32")
    ref = Signature({"time": 10, "lat": 4}, "float32")

    with pytest.raises(ValueError, match="sizes do not match"):
        op.compute_output_signature((pred, ref))
