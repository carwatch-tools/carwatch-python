"""Tools for loading and analyzing CARWatch study data."""

import importlib

from .merge import merge_saliva

__all__ = ["io", "logs", "merge_saliva"]

__version__ = "0.1.0"


def __getattr__(name: str):
    """Lazily import public submodules."""
    if name in {"io", "logs"}:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
