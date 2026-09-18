"""Local-file adapter: open satellite L2 products already on disk."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from xrreader._src.base import DatasetInfo, DataSource
from xrreader._src.local.catalog import LOCAL_DATASETS
from xrreader._src.local.openers import (
    open_emit_ch4_l2b,
    open_ghgsat_ch4_l2,
    open_tropomi_ch4_l2,
)
from xrreader.types import (
    BBox,
    DepthRange,
    PressureLevels,
    TimeRange,
    Variable,
    subset_bbox,
    subset_time,
)


_OPENERS: dict[str, Callable[..., xr.Dataset]] = {
    "tropomi.ch4": open_tropomi_ch4_l2,
    "emit.ch4": open_emit_ch4_l2b,
    "ghgsat.ch4": open_ghgsat_ch4_l2,
}


class LocalL2Source(DataSource):
    """Open satellite L2 product files from the local filesystem.

    ``dataset_id`` selects a product opener from :data:`LOCAL_DATASETS`
    (``"tropomi.ch4"``, ``"emit.ch4"``, ``"ghgsat.ch4"``); the file path
    comes through ``path=`` (or ``paths=`` for multi-granule concatenation
    along ``time``). ``bbox`` is applied with
    :func:`xrreader.subset_bbox` on the 2-D ``lon``/``lat`` coordinates
    and ``time`` with :func:`xrreader.subset_time` (or a per-scanline mask
    for swaths). Nothing is fetched: :meth:`download` raises.
    """

    source_id = "local"

    # ---- DataSource API ---------------------------------------------------

    def list_datasets(self) -> list[DatasetInfo]:
        return list(LOCAL_DATASETS.values())

    def describe(self, dataset_id: str) -> DatasetInfo:
        try:
            return LOCAL_DATASETS[dataset_id]
        except KeyError:
            raise KeyError(
                f"Unknown local dataset {dataset_id!r}; known: {sorted(LOCAL_DATASETS)}"
            ) from None

    def download(
        self,
        dataset_id: str,
        output: Path,
        *,
        variables: list[str | Variable] | None = None,
        bbox: BBox | None = None,
        time: TimeRange | None = None,
        depth: DepthRange | None = None,
        levels: PressureLevels | None = None,
        **extras: Any,
    ) -> Path:
        """Not supported: the product files are already on disk.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("local files are already on disk")

    def open(
        self,
        dataset_id: str,
        *,
        path: str | Path | None = None,
        paths: Sequence[str | Path] | None = None,
        variables: list[str | Variable] | None = None,
        bbox: BBox | None = None,
        time: TimeRange | None = None,
        depth: DepthRange | None = None,
        levels: PressureLevels | None = None,
        qa_min: float | None = None,
        **extras: Any,
    ) -> xr.Dataset:
        """Open one or more product files as a single flat Dataset.

        Args:
            dataset_id: Catalog short name (``"tropomi.ch4"``, ``"emit.ch4"``
                or ``"ghgsat.ch4"``).
            path: The product file. Exactly one of ``path`` / ``paths``.
            paths: Several granules / scenes, concatenated along ``time``
                in the order given (they must share the swath or raster
                shape).
            variables: Registry names or :class:`Variable` instances to
                keep; ``None`` keeps everything the opener produced.
            bbox: Keep only pixels whose 2-D ``lon`` / ``lat`` fall inside
                the box (``where(..., drop=True)`` — scanlines / rows
                that fall entirely outside are dropped).
            time: Keep measurements inside the window. Swaths are masked
                on their per-scanline ``scanline_time``; scenes are sliced
                on the ``time`` index.
            depth: Ignored (no vertical axis in these products).
            levels: Ignored (no vertical axis in these products).
            qa_min: Forwarded to the opener when given (TROPOMI:
                ``qa_value < qa_min`` is masked). ``None`` leaves the
                opener's own default in force (``0.5`` for TROPOMI); pass
                ``0.0`` to keep every pixel.
            **extras: Forwarded to the opener (``keep_groups=``,
                ``glt_path=`` ...).

        Returns:
            The screened Dataset.

        Raises:
            KeyError: If ``dataset_id`` is not a local product, or a
                requested variable is absent from the opened file.
            ValueError: If neither or both of ``path`` / ``paths`` are
                given, or ``bbox`` crosses the antimeridian.
        """
        opener = _OPENERS.get(dataset_id)
        if opener is None:
            raise KeyError(
                f"Unknown local dataset {dataset_id!r}; known: {sorted(_OPENERS)}"
            )
        if (path is None) == (paths is None):
            raise ValueError("pass exactly one of path= or paths=")
        if qa_min is not None:
            extras["qa_min"] = qa_min
        files = [path] if path is not None else list(paths or ())
        parts = [opener(Path(p), **extras) for p in files]
        ds = parts[0] if len(parts) == 1 else xr.concat(parts, dim="time")
        if variables:
            keep = {v.name for v in self._resolve_variables(variables)}
            if missing := sorted(keep - set(ds.data_vars)):
                raise KeyError(f"{missing} not in dataset; have {sorted(ds.data_vars)}")
            ds = ds.drop_vars([n for n in ds.data_vars if n not in keep])
        if bbox is not None:
            if bbox.crosses_antimeridian:
                raise ValueError(
                    "BBox crosses the antimeridian; call .to_360() or split it"
                )
            ds = subset_bbox(
                ds, (bbox.lon_min, bbox.lon_max), (bbox.lat_min, bbox.lat_max)
            )
        if time is not None:
            ds = _subset_time_range(ds, time)
        return ds


def _subset_time_range(ds: xr.Dataset, time: TimeRange) -> xr.Dataset:
    """Apply a :class:`TimeRange` to a swath (per scanline) or scene."""
    start = _naive(time.start)
    end = _naive(time.end)
    if "scanline_time" in ds.coords:
        st = ds["scanline_time"]
        return ds.where((st >= start) & (st <= end), drop=True)
    return subset_time(ds, start, end)


def _naive(ts: pd.Timestamp) -> np.datetime64:
    """Drop the UTC tz so comparisons against ``datetime64`` coords work."""
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.to_datetime64()
