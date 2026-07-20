"""Tools for loading and analyzing CARWatch study data."""

import importlib

__all__ = ["io", "logs", "study_manager"]

__version__ = "0.1.0"


def __getattr__(name: str):
    """Lazily import public submodules."""
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
