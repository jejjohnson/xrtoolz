"""xarray-aware :class:`Operator` — adds ``DataTree`` dispatch on top of pipekit.

``pipekit.Operator`` is carrier-agnostic and only knows two call modes:
eager ``_apply`` and symbolic ``Node``-construction. Earth-science data
also flows through ``xarray.DataTree`` (multi-group / multi-resolution
hierarchies). Rather than asking every diagnostic operator to opt in to
tree handling by hand, we widen the base class with one additional
branch: if any positional argument is a ``DataTree``, the operator is
mapped over every leaf via ``xr.map_over_datasets``.

The dispatch order matches the design doc (see
``docs/design/xarray-native-primitives.md``):

1. **Symbolic graph mode** — any ``Node`` argument routes to
   ``pipekit.Operator.__call__`` (which builds a ``Node`` recording this
   operator and its parents).
2. **DataTree mode** — any ``DataTree`` argument: apply
   ``xr.map_over_datasets`` to thread ``_apply`` over each matching
   leaf. ``xarray`` enforces the multi-input structural-match
   requirement; mixing a ``DataTree`` with a plain ``Dataset`` raises.
3. **Eager mode** — fall through to ``pipekit.Operator.__call__``,
   which runs ``_apply`` directly on the carrier(s).

Consequence: every operator that inherits from this class — every
``xrtoolz`` diagnostic, every combinator, every ``Sequential`` /
``Graph`` built out of them — gains ``DataTree`` support for free.
``Sequential`` and ``Graph`` themselves do not need changes: they are
carrier-agnostic in pipekit and simply thread whatever each step
returns to the next, so a ``DataTree`` in / ``DataTree`` out chain
composes naturally.

This is the implementation of "PR α" from the design doc.
"""

from __future__ import annotations

from typing import Any

import xarray as xr
from pipekit import Operator as _PipekitOperator


class Operator(_PipekitOperator):
    """``pipekit.Operator`` plus xarray ``DataTree`` dispatch.

    Subclasses implement ``_apply(carrier, *extra)`` exactly as they
    would against ``pipekit.Operator`` — the override only affects
    ``__call__`` dispatch, not ``_apply``.

    Example:
        A shape-preserving diagnostic works against any of the three
        xarray containers without per-class branching::

            class GaussianSmooth(Operator):
                def __init__(self, variable, *, dim, sigma):
                    self.variable, self.dim, self.sigma = variable, dim, sigma

                def _apply(self, ds: xr.Dataset) -> xr.Dataset:
                    return ds.assign({self.variable: ...})

            op = GaussianSmooth("ssh", dim="time", sigma=3)
            op(ds)   # → Dataset
            op(dt)   # → DataTree, each leaf smoothed independently
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Dispatch eager / symbolic / DataTree modes on positional args.

        - Any ``Node`` arg → pipekit's symbolic graph construction.
        - Any ``DataTree`` arg → leaf-wise map via
          ``xr.map_over_datasets``; the operator is applied to every
          matching ``Dataset`` leaf and the resulting leaves are
          reassembled into a ``DataTree`` with the input's structure.
          Empty-payload nodes (notably the implicit root of a tree
          built from ``DataTree.from_dict``) are skipped via a
          ``None``-return passthrough — there's nothing to operate on
          and most ``_apply`` implementations would raise a ``KeyError``
          trying to look up their variable.
        - Otherwise → eager ``_apply`` via the pipekit base class.
        """
        # Lazy import — Node lives in pipekit but pulling it eagerly at
        # module import would create a strict load order between the
        # base classes.
        from pipekit._base.graph import Node

        if any(isinstance(a, Node) for a in args):
            return super().__call__(*args, **kwargs)
        if any(isinstance(a, xr.DataTree) for a in args):
            apply = self._apply

            def _leaf(*leaves: xr.Dataset) -> xr.Dataset | None:
                # Skip empty-payload nodes (e.g. the synthetic root of
                # a tree assembled from a flat ``{path: Dataset}`` dict).
                # Returning ``None`` tells ``map_over_datasets`` to keep
                # the node in place without rewriting its contents.
                if any(len(leaf.data_vars) == 0 for leaf in leaves):
                    return None
                return apply(*leaves, **kwargs)

            return xr.map_over_datasets(_leaf, *args)
        return super().__call__(*args, **kwargs)
