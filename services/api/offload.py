"""Blocking work a request handler must not do on the event loop.

The API is one process with one event loop. An object-store read is a blocking HTTP call and a JPEG
decode is tens of milliseconds of CPU; done inline in an `async def` handler, either one stops every
other request the server is handling until it returns. A single slow read from storage was enough to
make the whole product feel hung.

These wrap the common cases so a handler reads the same as before with an `await` in front.
tests/test_event_loop_blocking.py fails on a new inline call.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

import cv2
import numpy as np
import numpy.typing as npt

from core.storage import get_object_store

T = TypeVar("T")
Image = npt.NDArray[Any]


async def run_blocking(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a blocking callable in the default thread pool."""
    return await asyncio.to_thread(fn, *args, **kwargs)


async def fetch_bytes(uri: str) -> bytes:
    """An object's bytes, read off the loop."""
    return await asyncio.to_thread(get_object_store().get_bytes, uri)


async def put_bytes(key: str, data: bytes, content_type: str) -> str:
    """Store bytes off the loop; returns the object's uri."""
    return await asyncio.to_thread(get_object_store().put_bytes, key, data, content_type)


def decode_image(raw: bytes) -> Image | None:
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


async def decode_image_async(raw: bytes) -> Image | None:
    return await asyncio.to_thread(decode_image, raw)


async def fetch_image(uri: str) -> Image | None:
    """Read and decode an image, both off the loop. None when the bytes are not an image."""
    def _load() -> Image | None:
        return decode_image(get_object_store().get_bytes(uri))

    return await asyncio.to_thread(_load)


async def encode_image(ext: str, img: Image, params: list[int] | None = None) -> bytes | None:
    """Encode an image off the loop. None when the encoder refuses it."""
    def _enc() -> bytes | None:
        ok, buf = cv2.imencode(ext, img, params or [])
        return buf.tobytes() if ok else None

    return await asyncio.to_thread(_enc)


async def ensure_bucket() -> None:
    await asyncio.to_thread(get_object_store().ensure_bucket)
