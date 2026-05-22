"""Object-event detection, labelling, matching, and metric class names."""

from xrtoolz.metrics._src.object import (
    CentroidDistance,
    CriticalSuccessIndex,
    DetectAnomalyObjects,
    DurationError,
    EventDefinition,
    FalseAlarmRatio,
    IntensityBias,
    IntersectionOverUnion,
    LabelObjects,
    MatchObjects,
    ProbabilityOfDetection,
    detect_anomaly_objects,
    label_objects,
    match_objects,
)


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
