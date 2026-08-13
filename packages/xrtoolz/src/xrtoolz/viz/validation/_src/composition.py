"""Shared plumbing for the composable panel wrappers.

:class:`~xrtoolz.viz.validation.FacetPanel`,
:class:`~xrtoolz.viz.validation.PairwiseComparePanel` and
:class:`~xrtoolz.viz.validation.AnimatePanel` all wrap an *inner* panel
and render it into axes they own. The mechanics they share live here:
dispatching to the inner renderer, forwarding cartopy projections,
recovering a mappable for shared colorbars, and keeping the figure's
axes list from growing when a panel is rendered repeatedly.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Any

import matplotlib.cm as mpl_cm
import matplotlib.figure as mpl_figure
import numpy as np
import xarray as xr

from xrtoolz.viz.validation._src.base import _ValidationPanel


#: An inner panel is either a ``_ValidationPanel`` or a bare
#: ``(ds, ax) -> Any`` callable.
InnerPanel = _ValidationPanel | Callable[[xr.Dataset, Any], Any]


def _resolve_subplot_kw(
    panel: InnerPanel, override: dict[str, Any] | None
) -> dict[str, Any]:
    """Build ``subplot_kw`` for a grid hosting ``panel``.

    Panels that need cartopy axes advertise it through a ``projection``
    attribute (the convention :class:`SpatialMapPanel` established). An
    explicit ``override`` always wins — including an empty dict, which
    deliberately forces plain matplotlib axes.
    """
    if override is not None:
        return dict(override)
    projection = getattr(panel, "projection", None)
    if projection is None:
        return {}
    from xrtoolz.viz._src.projections import _resolve_projection

    crs = _resolve_projection(projection)
    return {} if crs is None else {"projection": crs}


def _render_into(panel: InnerPanel, fig: mpl_figure.Figure, ax: Any, ds: Any) -> Any:
    """Render ``ds`` into ``ax`` using whichever inner-panel flavour."""
    if isinstance(panel, _ValidationPanel):
        return panel._build(fig, ax, ds)
    return panel(ds, ax)


def _find_mappable(ax: Any, returned: Any = None) -> Any | None:
    """Recover a colorbar-able mappable produced by an inner panel.

    Prefers whatever ``_build`` returned, so a panel can name its own
    mappable explicitly. None of the shipped panels return one today —
    they call ``fig.colorbar`` internally — so fall back to scavenging
    the most recent artist off the axes, which is what actually makes
    ``sharebar`` work with the current panel suite.
    """
    if isinstance(returned, mpl_cm.ScalarMappable):
        return returned
    for artists in (getattr(ax, "collections", []), getattr(ax, "images", [])):
        for artist in reversed(list(artists)):
            if isinstance(artist, mpl_cm.ScalarMappable):
                return artist
    return None


def _drop_axes_added_since(fig: mpl_figure.Figure, keep: frozenset[Any]) -> None:
    """Remove figure axes created since ``keep`` was snapshotted.

    Panels such as :class:`SpatialMapPanel` call ``fig.colorbar`` inside
    ``_build``, which appends a new Axes to the figure every single call.
    Re-rendering into the same axes — every animation frame, or every
    cell when a shared colorbar supersedes the per-cell ones — would
    otherwise pile up Axes without bound.
    """
    for axis in [a for a in fig.axes if a not in keep]:
        axis.remove()


@contextlib.contextmanager
def _temporary_attrs(obj: Any, overrides: dict[str, Any] | None) -> Iterator[None]:
    """Temporarily set attributes on ``obj``, restoring them after.

    Used instead of the clone-via-``get_config`` round-trip the design
    sketches proposed: several panels stringify un-reconstructable state
    in ``get_config`` (``SpatialMapPanel`` emits ``repr(crs)`` for a
    cartopy instance), so rebuilding through ``__init__`` would fail on
    exactly the panels most likely to be wrapped.
    """
    if not overrides:
        yield
        return
    missing = object()
    previous = {name: getattr(obj, name, missing) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(obj, name, value)
        yield
    finally:
        for name, value in previous.items():
            if value is missing:
                delattr(obj, name)
            else:
                setattr(obj, name, value)


def _inner_cell_grid(panel: InnerPanel) -> tuple[int, int] | None:
    """Return the ``(nrows, ncols)`` a panel needs, if more than one Axes.

    Most panels draw into a single Axes and report ``None``. Wrappers
    that render several — :class:`PairwiseComparePanel`'s ref/study/diff
    triptych — advertise their shape through ``_cell_grid`` so an outer
    grid can hand them a subdivided cell instead of a bare Axes.
    """
    cell_grid = getattr(panel, "_cell_grid", None)
    if cell_grid is None:
        return None
    shape = cell_grid()
    return None if shape == (1, 1) else shape


def _build_axes_for(
    panel: InnerPanel,
    ds: Any,
    *,
    figsize: tuple[float, float],
    subplot_kw: dict[str, Any] | None,
) -> tuple[mpl_figure.Figure, Any]:
    """Create the figure and axes ``panel`` needs to render ``ds``.

    Panels whose layout depends on the data — :class:`FacetPanel` sizes
    its grid from the faceted dimension — expose ``_make_fig_axes_for``.
    That hook is what lets a wrapper host another wrapper, e.g.
    ``AnimatePanel(FacetPanel(...))``, where the axes must exist before
    the first frame is drawn.
    """
    import matplotlib.pyplot as plt

    if subplot_kw is None and isinstance(panel, _ValidationPanel):
        builder = getattr(panel, "_make_fig_axes_for", None)
        with _temporary_attrs(panel, {"figsize": figsize}):
            if builder is not None:
                return builder(ds)
            return panel._make_fig_axes()
    resolved = _resolve_subplot_kw(panel, subplot_kw)
    fig, axes = plt.subplots(figsize=figsize, subplot_kw=resolved or None)
    return fig, axes


def _clear_axes(axes: Any) -> None:
    """Clear one Axes or a whole array of them."""
    for ax in np.ravel(np.asarray(axes, dtype=object)).tolist():
        ax.clear()


def _inner_config(panel: InnerPanel) -> dict[str, Any]:
    """Serialise the inner panel for a wrapper's ``get_config``.

    ``_ValidationPanel`` inners recurse into their own config and stay
    round-trippable; bare callables cannot, and say so explicitly rather
    than pretending otherwise.
    """
    if isinstance(panel, _ValidationPanel):
        return {
            "panel": type(panel).__name__,
            "panel_config": panel.get_config(),
            "roundtrippable": True,
        }
    return {
        "panel": getattr(panel, "__name__", "<callable>"),
        "panel_config": None,
        "roundtrippable": False,
    }


__all__ = [
    "InnerPanel",
    "_build_axes_for",
    "_clear_axes",
    "_drop_axes_added_since",
    "_find_mappable",
    "_inner_cell_grid",
    "_inner_config",
    "_render_into",
    "_resolve_subplot_kw",
    "_temporary_attrs",
]
