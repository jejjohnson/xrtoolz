"""Lightweight shape descriptors for operator summary/introspection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Signature:
    """Shape-and-dtype descriptor propagated without executing data paths."""

    dims: dict[str, int | None]
    dtype: Any = None

    def __post_init__(self) -> None:
        self.dims = {
            name: (None if size is None else int(size))
            for name, size in self.dims.items()
        }

    def replace_dims(self, updates: dict[str, int | None]) -> Signature:
        """Return a copy with selected dimension sizes replaced."""
        dims = dict(self.dims)
        for name, size in updates.items():
            if name in dims:
                dims[name] = size
        return Signature(dims, dtype=self.dtype)

    def rename_dims(self, mapping: dict[str, str]) -> Signature:
        """Return a copy with dimension names renamed by ``mapping``."""
        return Signature(
            {mapping.get(name, name): size for name, size in self.dims.items()},
            dtype=self.dtype,
        )

    def drop_dims(self, names: tuple[str, ...]) -> Signature:
        """Return a copy without the requested dimensions."""
        drop = set(names)
        return Signature(
            {name: size for name, size in self.dims.items() if name not in drop},
            dtype=self.dtype,
        )

    def format(self) -> str:
        """Render dimensions and dtype as a compact summary string."""
        shape = ", ".join(
            f"{name}={size if size is not None else '?'}"
            for name, size in self.dims.items()
        )
        rendered = f"({shape})"
        if self.dtype is not None:
            rendered = f"{rendered}; dtype={self.dtype}"
        return rendered
