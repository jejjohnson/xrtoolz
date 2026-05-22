"""Hovmoller validation panel."""

from __future__ import annotations

from typing import Any

import matplotlib.colors as mcolors
import matplotlib.figure as mpl_figure
import numpy as np
import xarray as xr

from xrtoolz.viz.validation._src.base import _ValidationPanel


class HovmollerPanel(_ValidationPanel):
    """Time × spatial-axis cross-section of a field.

    Args:
        var: Data-variable name when input is a Dataset. ``None``
            (default) auto-picks the first ``data_var``.
        time_dim: Time coordinate / dimension name. Default ``"time"``.
        keep_dim: Spatial coordinate / dimension to keep. Remaining
            dimensions are averaged before plotting. Default ``"lat"``.
        cmap: Matplotlib colormap. Default ``"RdBu_r"``.
        norm: Colour normalization, either ``"linear"`` or ``"log"``.
            Default ``"linear"``.
        vmin, vmax: Optional colour-scale limits.
    """

    _default_axes_layout = (1, 1)

    def __init__(
        self,
        *,
        var: str | None = None,
        time_dim: str = "time",
        keep_dim: str = "lat",
        cmap: str = "RdBu_r",
        norm: str = "linear",
        vmin: float | None = None,
        vmax: float | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        if norm not in {"linear", "log"}:
            msg = "norm must be 'linear' or 'log'"
            raise ValueError(msg)
        self.var = var
        self.time_dim = time_dim
        self.keep_dim = keep_dim
        self.cmap = cmap
        self.norm = norm
        self.vmin = vmin
        self.vmax = vmax

    def _default_title(self) -> str:
        return self.var or "Hovmoller"

    def _select_var(self, obj: xr.DataArray | xr.Dataset) -> xr.DataArray:
        if isinstance(obj, xr.Dataset):
            name = self.var or next(iter(obj.data_vars))
            return obj[name]
        return obj

    def _section(self, da: xr.DataArray) -> xr.DataArray:
        missing = [dim for dim in (self.time_dim, self.keep_dim) if dim not in da.dims]
        if missing:
            msg = f"Input is missing required dimension(s): {', '.join(missing)}"
            raise ValueError(msg)
        avg_dims = [dim for dim in da.dims if dim not in {self.time_dim, self.keep_dim}]
        if avg_dims:
            da = da.mean(dim=avg_dims)
        return da.transpose(self.keep_dim, self.time_dim)

    def _color_norm(self) -> mcolors.Normalize | None:
        if self.norm == "linear":
            return None
        return mcolors.LogNorm(vmin=self.vmin, vmax=self.vmax)

    def _build(
        self,
        fig: mpl_figure.Figure,
        axes: Any,
        field: xr.DataArray | xr.Dataset,
    ) -> None:
        ax = axes
        section = self._section(self._select_var(field))
        vals = np.asarray(section.values)
        pcm_kw: dict[str, Any] = {"cmap": self.cmap, "shading": "auto"}
        color_norm = self._color_norm()
        if color_norm is None:
            pcm_kw["vmin"] = self.vmin
            pcm_kw["vmax"] = self.vmax
        else:
            pcm_kw["norm"] = color_norm
            vals = np.ma.masked_less_equal(vals, 0.0)
        im = ax.pcolormesh(
            section[self.time_dim].values,
            section[self.keep_dim].values,
            vals,
            **pcm_kw,
        )
        ax.set_xlabel(self.time_dim)
        ax.set_ylabel(self.keep_dim)
        fig.colorbar(im, ax=ax)

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "var": self.var,
            "time_dim": self.time_dim,
            "keep_dim": self.keep_dim,
            "cmap": self.cmap,
            "norm": self.norm,
            "vmin": self.vmin,
            "vmax": self.vmax,
        }


__all__ = ["HovmollerPanel"]
