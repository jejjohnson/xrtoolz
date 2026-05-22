"""Object-event detection, labelling, and matching primitives (V5.1).

Layer-0 functions:

- :func:`detect_anomaly_objects` builds a threshold mask from an
  :class:`EventDefinition`, labels connected components, and filters objects.
- :func:`label_objects` labels connected components in a Boolean mask.
- :func:`match_objects` pairs two object datasets by IoU or centroid distance.

Layer-1 wrappers: :class:`DetectAnomalyObjects`, :class:`LabelObjects`,
:class:`MatchObjects`.

Object Dataset schema
---------------------
Detector and labeller outputs are :class:`xarray.Dataset` objects with an
``"event"`` dimension and, when the input has time, a ``"time"`` dimension.
The stable V5 object table fields are:

``area(event, time)``
    Count of active spatial cells per event/time slice.
``centroid_lon(event, time)``, ``centroid_lat(event, time)``
    Cell-count centroid coordinates. Empty slices are NaN.
``intensity_max(event, time)``, ``intensity_mean(event, time)``
    Per-slice extrema/mean of the detection field. ``label_objects`` has no
    intensity field, so these are NaN there.
``start_time(event)``, ``end_time(event)``, ``duration(event)``
    Temporal extent and number of active time slices. Non-temporal masks use a
    synthetic one-step ``time`` coordinate and have duration 1.

Outputs also carry ``label`` on the original mask dimensions (0 = background)
and ``object_mask(event, ...)`` for overlap metrics. Input latitude/longitude
coordinates are preserved as normal xarray coordinates named ``"lat"`` and
``"lon"`` when present.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import xarray as xr

from xrtoolz._operator import Operator
from xrtoolz.utils._src.optional_imports import _require_optional


MatchMethod = Literal["iou", "centroid"]


@dataclass(frozen=True)
class EventDefinition:
    """Reusable threshold/anomaly event specification.

    ``threshold`` may be an absolute numeric threshold or a percentile string
    such as ``"p90"``. Use :meth:`to_json_dict` or :meth:`to_json` for a stable
    JSON-serializable representation; baseline data are summarized rather than
    embedded.
    """

    variable: str
    threshold: float | str
    baseline: xr.Dataset | None = None
    min_duration: int | None = None
    min_area: float | None = None
    connectivity: int = 8
    anomaly: bool = True

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration summary."""
        baseline = None
        if self.baseline is not None:
            baseline = {
                "data_vars": sorted(self.baseline.data_vars),
                "coords": sorted(self.baseline.coords),
                "dims": {str(k): int(v) for k, v in self.baseline.sizes.items()},
            }
        return {
            "variable": self.variable,
            "threshold": self.threshold,
            "baseline": baseline,
            "min_duration": self.min_duration,
            "min_area": self.min_area,
            "connectivity": self.connectivity,
            "anomaly": self.anomaly,
        }

    def to_json(self) -> str:
        """Serialize :meth:`to_json_dict` with deterministic key order."""
        return json.dumps(self.to_json_dict(), sort_keys=True)


def _require_label():
    measure = _require_optional(
        "skimage.measure",
        extra="image",
        feature="object labelling",
        package="scikit-image",
    )
    return measure.label


def _label_connectivity(ndim: int, connectivity: int) -> int:
    """Map common neighborhood names to scikit-image connectivity ranks."""
    if connectivity in (4, 6):
        return 1
    if connectivity in (8, 18):
        return min(2, ndim)
    if connectivity in (26,):
        return ndim
    if 1 <= connectivity <= ndim:
        return connectivity
    raise ValueError(
        "connectivity must be one of 4, 8, 6, 18, 26 or a skimage "
        f"connectivity rank in [1, {ndim}], got {connectivity!r}."
    )


def _as_time_mask(
    mask: xr.DataArray,
    *,
    time_dim: str = "time",
) -> tuple[xr.DataArray, str, bool]:
    if time_dim in mask.dims:
        return mask, time_dim, True
    expanded = mask.expand_dims({time_dim: [0]})
    return expanded, time_dim, False


def _empty_object_dataset(
    mask: xr.DataArray,
    label: xr.DataArray,
    *,
    time_dim: str,
) -> xr.Dataset:
    event = np.array([], dtype=np.int64)
    time = mask[time_dim].values
    ds = xr.Dataset(
        {
            "label": label,
            "object_mask": (("event", *mask.dims), np.zeros((0, *mask.shape), bool)),
            "area": (("event", time_dim), np.zeros((0, time.size), dtype=float)),
            "centroid_lon": (
                ("event", time_dim),
                np.zeros((0, time.size), dtype=float),
            ),
            "centroid_lat": (
                ("event", time_dim),
                np.zeros((0, time.size), dtype=float),
            ),
            "intensity_max": (
                ("event", time_dim),
                np.zeros((0, time.size), dtype=float),
            ),
            "intensity_mean": (
                ("event", time_dim),
                np.zeros((0, time.size), dtype=float),
            ),
            "start_time": (("event",), np.array([], dtype=object)),
            "end_time": (("event",), np.array([], dtype=object)),
            "duration": (("event",), np.array([], dtype=np.int64)),
        },
        coords={"event": event, time_dim: time},
    )
    coords = {k: v for k, v in mask.coords.items() if k not in ds.coords}
    return ds.assign_coords(coords)


def _coord_values(mask: xr.DataArray, dim: str) -> np.ndarray:
    if dim in mask.coords:
        return np.asarray(mask.coords[dim].values, dtype=float)
    return np.arange(mask.sizes[dim], dtype=float)


def _summarize_objects(
    label_da: xr.DataArray,
    mask: xr.DataArray,
    *,
    dims: tuple[str, str],
    intensity: xr.DataArray | None = None,
    time_dim: str = "time",
) -> xr.Dataset:
    mask_t, time_dim, _ = _as_time_mask(mask, time_dim=time_dim)
    labels_t, _, _ = _as_time_mask(label_da, time_dim=time_dim)
    labels = labels_t.transpose(time_dim, *dims).values
    event_ids = np.unique(labels)
    event_ids = event_ids[event_ids > 0].astype(np.int64)

    if event_ids.size == 0:
        return _empty_object_dataset(mask_t, label_da, time_dim=time_dim)

    lat_dim, lon_dim = dims
    lat_values = _coord_values(mask_t, lat_dim)
    lon_values = _coord_values(mask_t, lon_dim)
    time_values = mask_t[time_dim].values
    n_event = event_ids.size
    n_time = time_values.size

    area = np.full((n_event, n_time), np.nan, dtype=float)
    centroid_lat = np.full_like(area, np.nan)
    centroid_lon = np.full_like(area, np.nan)
    intensity_max = np.full_like(area, np.nan)
    intensity_mean = np.full_like(area, np.nan)
    object_mask_values = np.zeros((n_event, *mask_t.shape), dtype=bool)

    if intensity is not None:
        intensity_t, _, _ = _as_time_mask(intensity, time_dim=time_dim)
        intensity_values = intensity_t.transpose(time_dim, *dims).values
    else:
        intensity_values = None

    start_time: list[object] = []
    end_time: list[object] = []
    duration: list[int] = []

    for i, event_id in enumerate(event_ids):
        event_sel = labels == event_id
        object_mask_values[i] = np.asarray(labels_t == event_id)
        active_times = np.any(event_sel, axis=(1, 2))
        duration.append(int(active_times.sum()))
        if active_times.any():
            active = np.flatnonzero(active_times)
            start_time.append(time_values[int(active[0])])
            end_time.append(time_values[int(active[-1])])
        else:
            start_time.append(np.nan)
            end_time.append(np.nan)

        for t in np.flatnonzero(active_times):
            spatial_sel = event_sel[t]
            lat_idx, lon_idx = np.where(spatial_sel)
            area[i, t] = float(spatial_sel.sum())
            # Cell-count centroid; callers with unequal cell areas should
            # preweight or regrid before object detection.
            centroid_lat[i, t] = float(lat_values[lat_idx].mean())
            centroid_lon[i, t] = float(lon_values[lon_idx].mean())
            if intensity_values is not None:
                vals = intensity_values[t][spatial_sel]
                intensity_max[i, t] = float(np.nanmax(vals))
                intensity_mean[i, t] = float(np.nanmean(vals))

    ds = xr.Dataset(
        {
            "label": label_da,
            "object_mask": (("event", *mask_t.dims), object_mask_values),
            "area": (("event", time_dim), area),
            "centroid_lon": (("event", time_dim), centroid_lon),
            "centroid_lat": (("event", time_dim), centroid_lat),
            "intensity_max": (("event", time_dim), intensity_max),
            "intensity_mean": (("event", time_dim), intensity_mean),
            "start_time": (("event",), np.asarray(start_time)),
            "end_time": (("event",), np.asarray(end_time)),
            "duration": (("event",), np.asarray(duration, dtype=np.int64)),
        },
        coords={"event": event_ids, time_dim: time_values},
    )
    coords = {k: v for k, v in mask_t.coords.items() if k not in ds.coords}
    return ds.assign_coords(coords)


def label_objects(
    mask: xr.DataArray,
    *,
    dims: tuple[str, str] = ("lat", "lon"),
    connectivity: int = 8,
) -> xr.Dataset:
    """Label connected components in a Boolean mask.

    Components are labelled across all mask dimensions, including ``"time"``
    when present, while ``dims`` identifies the two spatial axes used for
    object summaries and centroids.
    """
    missing = [d for d in dims if d not in mask.dims]
    if missing:
        raise ValueError(f"mask is missing spatial dims {missing!r}; got {mask.dims}.")

    mask_bool = mask.fillna(False).astype(bool)
    dim_order = tuple(mask_bool.dims)
    label = _require_label()
    values = mask_bool.transpose(*dim_order).values
    labelled = label(
        values,
        connectivity=_label_connectivity(values.ndim, connectivity),
        background=0,
    ).astype(np.int64)
    label_da = xr.DataArray(
        labelled,
        dims=dim_order,
        coords={
            k: v for k, v in mask_bool.coords.items() if set(v.dims) <= set(dim_order)
        },
        name="label",
    ).transpose(*mask_bool.dims)
    return _summarize_objects(label_da, mask_bool, dims=dims)


def _threshold_value(field: xr.DataArray, threshold: float | str) -> float:
    """Resolve numeric and percentile thresholds, including p0 and p100."""
    if isinstance(threshold, str):
        if not threshold.startswith("p"):
            raise ValueError(
                f"percentile thresholds must be strings like 'p90', got {threshold!r}."
            )
        percentile = float(threshold[1:])
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(f"percentile must be in [0, 100], got {percentile}.")
        return float(np.nanpercentile(field.values, percentile))
    return float(threshold)


def detect_anomaly_objects(ds: xr.Dataset, definition: EventDefinition) -> xr.Dataset:
    """Detect thresholded anomaly objects for ``definition.variable``."""
    if definition.variable not in ds:
        raise KeyError(f"Dataset is missing variable {definition.variable!r}.")
    field = ds[definition.variable]
    if definition.anomaly and definition.baseline is not None:
        if definition.variable not in definition.baseline:
            raise KeyError(f"Baseline is missing variable {definition.variable!r}.")
        field = field - definition.baseline[definition.variable]

    threshold = _threshold_value(field, definition.threshold)
    mask = field > threshold
    objects = label_objects(mask, connectivity=definition.connectivity)
    objects = _summarize_objects(
        objects["label"],
        mask,
        dims=("lat", "lon"),
        intensity=field,
    )

    duration_ok = xr.ones_like(objects["duration"], dtype=bool)
    if definition.min_duration is not None:
        duration_ok = objects["duration"] >= int(definition.min_duration)
    area_ok = xr.ones_like(objects["duration"], dtype=bool)
    if definition.min_area is not None:
        area_ok = objects["area"].max("time") >= float(definition.min_area)
    keep = duration_ok & area_ok
    objects = objects.sel(event=objects["event"].where(keep, drop=True))
    return objects.assign_attrs(event_definition=definition.to_json())


def _object_masks(objects: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    if "object_mask" in objects:
        event_ids = np.asarray(objects["event"].values, dtype=np.int64)
        masks = np.asarray(objects["object_mask"].values, dtype=bool)
        return event_ids, masks.reshape((event_ids.size, -1))
    if "label" in objects:
        labels = np.asarray(objects["label"].values)
        event_ids = np.unique(labels)
        event_ids = event_ids[event_ids > 0].astype(np.int64)
        masks = np.stack([(labels == event_id).ravel() for event_id in event_ids])
        return event_ids, masks
    raise ValueError("objects datasets must carry 'object_mask' or 'label'.")


def _event_centroids(objects: xr.Dataset) -> dict[int, tuple[float, float]]:
    if not {"centroid_lon", "centroid_lat"} <= set(objects.data_vars):
        raise ValueError("centroid matching requires centroid_lon and centroid_lat.")
    centroids: dict[int, tuple[float, float]] = {}
    for event in objects["event"].values:
        obj = objects.sel(event=event)
        lon = float(obj["centroid_lon"].mean("time", skipna=True))
        lat = float(obj["centroid_lat"].mean("time", skipna=True))
        centroids[int(event)] = (lon, lat)
    return centroids


def match_objects(
    objects_pred: xr.Dataset,
    objects_ref: xr.Dataset,
    *,
    method: MatchMethod = "iou",
    threshold: float = 0.1,
) -> xr.Dataset:
    """Pair predicted/reference objects by IoU or centroid distance.

    The result has one row per predicted event and reports the best reference
    candidate, its score, and whether it passes ``threshold``.
    """
    pred_ids, pred_masks = _object_masks(objects_pred)
    ref_ids, ref_masks = _object_masks(objects_ref)
    if pred_ids.size == 0 or ref_ids.size == 0:
        return xr.Dataset(
            {
                "pred_event": (("match",), np.array([], dtype=np.int64)),
                "ref_event": (("match",), np.array([], dtype=np.int64)),
                "score": (("match",), np.array([], dtype=float)),
                "iou": (("match",), np.array([], dtype=float)),
                "centroid_distance": (("match",), np.array([], dtype=float)),
                "matched": (("match",), np.array([], dtype=bool)),
            },
            coords={"match": np.array([], dtype=np.int64)},
        )

    if method == "iou":
        intersections = pred_masks.astype(int) @ ref_masks.astype(int).T
        pred_area = pred_masks.sum(axis=1)[:, None]
        ref_area = ref_masks.sum(axis=1)[None, :]
        unions = pred_area + ref_area - intersections
        scores = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections, dtype=float),
            where=unions > 0,
        )
        best = np.argmax(scores, axis=1)
        best_scores = scores[np.arange(pred_ids.size), best]
        centroid_distance = np.full(pred_ids.size, np.nan)
        iou = best_scores
        matched = best_scores >= threshold
    elif method == "centroid":
        pred_centroids = _event_centroids(objects_pred)
        ref_centroids = _event_centroids(objects_ref)
        distances = np.empty((pred_ids.size, ref_ids.size), dtype=float)
        for i, pred_id in enumerate(pred_ids):
            pred_lon, pred_lat = pred_centroids[int(pred_id)]
            for j, ref_id in enumerate(ref_ids):
                ref_lon, ref_lat = ref_centroids[int(ref_id)]
                distances[i, j] = float(
                    np.hypot(pred_lon - ref_lon, pred_lat - ref_lat)
                )
        best = np.argmin(distances, axis=1)
        centroid_distance = distances[np.arange(pred_ids.size), best]
        best_scores = centroid_distance
        iou = np.full(pred_ids.size, np.nan)
        matched = centroid_distance <= threshold
    else:
        raise ValueError("method must be 'iou' or 'centroid'.")

    return xr.Dataset(
        {
            "pred_event": (("match",), pred_ids),
            "ref_event": (("match",), ref_ids[best]),
            "score": (("match",), best_scores.astype(float)),
            "iou": (("match",), iou.astype(float)),
            "centroid_distance": (("match",), centroid_distance.astype(float)),
            "matched": (("match",), matched.astype(bool)),
        },
        coords={"match": np.arange(pred_ids.size, dtype=np.int64)},
    )


class DetectAnomalyObjects(Operator):
    """Layer-1 wrapper for :func:`detect_anomaly_objects`."""

    def __init__(self, definition: EventDefinition) -> None:
        self.definition = definition

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        return detect_anomaly_objects(ds, self.definition)

    def get_config(self) -> dict[str, Any]:
        return {"definition": self.definition.to_json_dict()}


class LabelObjects(Operator):
    """Layer-1 wrapper for :func:`label_objects`."""

    def __init__(
        self,
        *,
        dims: tuple[str, str] = ("lat", "lon"),
        connectivity: int = 8,
    ) -> None:
        self.dims = tuple(dims)
        self.connectivity = connectivity

    def _apply(self, mask: xr.DataArray) -> xr.Dataset:
        return label_objects(mask, dims=self.dims, connectivity=self.connectivity)

    def get_config(self) -> dict[str, Any]:
        return {"dims": list(self.dims), "connectivity": self.connectivity}


class MatchObjects(Operator):
    """Layer-1 wrapper for :func:`match_objects`."""

    def __init__(self, *, method: MatchMethod = "iou", threshold: float = 0.1) -> None:
        self.method = method
        self.threshold = threshold

    def _apply(self, objects_pred: xr.Dataset, objects_ref: xr.Dataset) -> xr.Dataset:
        return match_objects(
            objects_pred,
            objects_ref,
            method=self.method,
            threshold=self.threshold,
        )

    def get_config(self) -> dict[str, Any]:
        return {"method": self.method, "threshold": self.threshold}


class _ObjectMetricStub(Operator):
    """Common ``NotImplementedError`` shell for V5 metric reservations."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} is a name reservation; "
            "metric implementation lands with V5.3."
        )


class ProbabilityOfDetection(_ObjectMetricStub):
    """Hits / (hits + misses). Implementation pending V5.3."""


class FalseAlarmRatio(_ObjectMetricStub):
    """False alarms / (hits + false alarms). Implementation pending V5.3."""


class CriticalSuccessIndex(_ObjectMetricStub):
    """Threat score: ``hits / (hits + misses + false alarms)``. V5.3 stub."""


class IntersectionOverUnion(_ObjectMetricStub):
    """Object overlap metric class. Implementation pending V5.3."""


class DurationError(_ObjectMetricStub):
    """Bias in matched-event duration. Implementation pending V5.3."""


class IntensityBias(_ObjectMetricStub):
    """Bias in matched-event intensity. Implementation pending V5.3."""


class CentroidDistance(_ObjectMetricStub):
    """Centroid displacement metric class. Implementation pending V5.3."""


__all__ = [
    "CentroidDistance",
    "CriticalSuccessIndex",
    "DetectAnomalyObjects",
    "DurationError",
    "EventDefinition",
    "FalseAlarmRatio",
    "IntensityBias",
    "IntersectionOverUnion",
    "LabelObjects",
    "MatchObjects",
    "ProbabilityOfDetection",
    "detect_anomaly_objects",
    "label_objects",
    "match_objects",
]
