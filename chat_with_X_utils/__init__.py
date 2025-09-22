__all__ = ["metadata_mangement", "print_utils", "tool_utils"]

import importlib

def __getattr__(name):
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")