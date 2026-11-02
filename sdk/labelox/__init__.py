"""LabeloxAV Python client.

    pip install labelox-client

    from labelox import Client
    lbx = Client("http://labelox.lab:3000", api_key="lbxk_...")   # or set LBX_URL and LBX_API_KEY
    lbx.health()

    import labelox
    labelox.configure("http://labelox.lab:3000", token="lbxk_...")
    ds = labelox.load("night AND vru", version="2026.07.1")       # a dataset by what it means

`Client` maps the REST API to methods. `configure`, `load` and `Dataset` stream versioned datasets as shards.
The generated client covering every endpoint is `labelox.generated.LabeloxClient`.

Imported lazily, so a caller who wants one half never pays for importing the other.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.1"
__all__ = ["ApiError", "Client", "Dataset", "configure", "load"]


def __getattr__(name: str) -> Any:
    if name in ("Client", "ApiError"):
        from . import client

        return client.Labelox if name == "Client" else client.LabeloxError
    if name in ("Dataset", "configure", "load"):
        from . import dataset

        return getattr(dataset, name)
    raise AttributeError(f"module 'labelox' has no attribute '{name}'")
