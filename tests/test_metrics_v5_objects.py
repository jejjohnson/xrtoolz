from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from xrtoolz.metrics import (
    DetectAnomalyObjects,
    EventDefinition,
    LabelObjects,
    MatchObjects,
    detect_anomaly_objects,
    label_objects,
    match_objects,
)


def _blob(center_lat: float, center_lon: float) -> xr.DataArray:
    lat = np.arange(-10.0, 11.0)
    lon = np.arange(-10.0, 11.0)
    lon2d, lat2d = np.meshgrid(lon, lat, indexing="xy")
    values = np.exp(-(((lat2d - center_lat) ** 2) + ((lon2d - center_lon) ** 2)) / 8.0)
    return xr.DataArray(values, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))


def test_event_definition_has_json_serializable_representation():
    definition = EventDefinition(
        variable="ssh",
        threshold="p90",
        min_duration=1,
        min_area=2.0,
    )

    encoded = definition.to_json()
    assert json.loads(encoded)["threshold"] == "p90"
    assert definition.to_json_dict()["baseline"] is None


def test_gaussian_blob_above_threshold_detected_at_expected_location_and_area():
    blob = _blob(center_lat=2.0, center_lon=-3.0)
    ds = blob.to_dataset(name="ssh")
    definition = EventDefinition(variable="ssh", threshold=0.5, anomaly=False)

    objects = detect_anomaly_objects(ds, definition)
    expected_area = int((blob > 0.5).sum())

    assert objects.sizes["event"] == 1
    assert float(objects["centroid_lat"].isel(event=0, time=0)) == pytest.approx(2.0)
    assert float(objects["centroid_lon"].isel(event=0, time=0)) == pytest.approx(-3.0)
    assert float(objects["area"].isel(event=0, time=0)) == pytest.approx(expected_area)
    assert float(objects["intensity_max"].isel(event=0, time=0)) == pytest.approx(1.0)
    assert set(
        [
            "area",
            "centroid_lon",
            "centroid_lat",
            "intensity_max",
            "intensity_mean",
            "start_time",
            "end_time",
            "duration",
        ]
    ) <= set(objects.data_vars)


def test_label_objects_and_operator_match_layer0():
    mask = _blob(center_lat=0.0, center_lon=0.0) > 0.5

    direct = label_objects(mask)
    via_operator = LabelObjects()(mask)

    xr.testing.assert_equal(via_operator, direct)


def test_detect_anomaly_objects_operator_matches_layer0():
    ds = _blob(center_lat=1.0, center_lon=1.0).to_dataset(name="ssh")
    definition = EventDefinition(variable="ssh", threshold=0.5, anomaly=False)

    direct = detect_anomaly_objects(ds, definition)
    via_operator = DetectAnomalyObjects(definition)(ds)

    xr.testing.assert_equal(via_operator, direct)


def test_iou_matching_identical_inputs_scores_one():
    objects = label_objects(_blob(center_lat=0.0, center_lon=0.0) > 0.5)

    matches = match_objects(objects, objects)

    assert matches.sizes["match"] == 1
    assert float(matches["iou"].item()) == pytest.approx(1.0)
    assert bool(matches["matched"].item())


def test_iou_matching_disjoint_inputs_scores_zero():
    pred = label_objects(_blob(center_lat=-7.0, center_lon=-7.0) > 0.5)
    ref = label_objects(_blob(center_lat=7.0, center_lon=7.0) > 0.5)

    matches = MatchObjects(method="iou", threshold=0.1)(pred, ref)

    assert matches.sizes["match"] == 1
    assert float(matches["iou"].item()) == pytest.approx(0.0)
    assert not bool(matches["matched"].item())
