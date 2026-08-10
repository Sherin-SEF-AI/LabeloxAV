"""LabeloxAV Python SDK.

    import labelox
    labelox.configure("https://labelox.example.com", token="lbx2....")
    ds = labelox.load("night AND vru", version="2026.07.1")

`load` is re-exported here so a training config can reference a dataset by what it means rather than by
where somebody happened to put a zip.

Imported lazily: the generated REST client and the dataset client pull in httpx, and a module that raises on
import would take the whole package down for callers who only wanted one half of it.
"""

from __future__ import annotations

from typing import Any

__all__ = ["Dataset", "Labelox", "configure", "load"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from sdk import dataset

        return getattr(dataset, name)
    raise AttributeError(f"module 'sdk' has no attribute '{name}'")
