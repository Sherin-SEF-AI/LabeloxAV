"""The champion weights cache is content-addressed and integrity-checked.

The defect: the cache path was the blob's basename, so two different champions both exported as "best.pt"
shared one cache entry. A promotion could then keep serving the previous model's weights, silently, with the
registry insisting the new model was live. The blob was also loaded with no verification, so a truncated
download became a corrupt model rather than an error.

Pure unit tests: the object store is stubbed, so no MinIO and no database."""
from __future__ import annotations

import hashlib

import pytest

from services.autolabel import runner


class _Store:
    """Minimal object-store stub: uri -> bytes, counting reads so cache hits are observable."""

    def __init__(self, blobs: dict[str, bytes]):
        self.blobs = blobs
        self.reads = 0

    def get_bytes(self, uri: str) -> bytes:
        self.reads += 1
        if uri not in self.blobs:
            raise KeyError(uri)
        return self.blobs[uri]


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Point the scratch cache at a temp dir so the test never touches the real one. Settings is a pydantic
    model (no attribute assignment), so the module-level getter is swapped for a tiny stand-in."""

    class _Settings:
        def scratch_path(self):
            return tmp_path

    monkeypatch.setattr(runner, "get_settings", lambda: _Settings())
    return tmp_path


def _use_store(monkeypatch, store: _Store):
    monkeypatch.setattr(runner, "get_object_store", lambda: store)


def test_same_basename_different_champions_do_not_collide(cache_dir, monkeypatch):
    # The exact regression: both champions are exported as "best.pt" under different prefixes.
    store = _Store({
        "models/champion-a/best.pt": b"WEIGHTS-A",
        "models/champion-b/best.pt": b"WEIGHTS-B",
    })
    _use_store(monkeypatch, store)

    a = runner._local_champion_weights("models/champion-a/best.pt")
    b = runner._local_champion_weights("models/champion-b/best.pt")

    assert a != b, "distinct champions must not share a cache entry"
    from pathlib import Path
    assert Path(a).read_bytes() == b"WEIGHTS-A"
    assert Path(b).read_bytes() == b"WEIGHTS-B"


def test_second_resolve_is_served_from_cache(cache_dir, monkeypatch):
    store = _Store({"models/c/best.pt": b"WEIGHTS"})
    _use_store(monkeypatch, store)

    first = runner._local_champion_weights("models/c/best.pt")
    second = runner._local_champion_weights("models/c/best.pt")

    assert first == second
    assert store.reads == 1, "a cache hit must not re-download the blob"


def test_corrupted_cache_entry_is_detected_and_refetched(cache_dir, monkeypatch):
    from pathlib import Path

    store = _Store({"models/c/best.pt": b"GOOD-WEIGHTS"})
    _use_store(monkeypatch, store)

    path = Path(runner._local_champion_weights("models/c/best.pt"))
    path.write_bytes(b"TRUNCATED")            # simulate a corrupted/partial cache file

    again = runner._local_champion_weights("models/c/best.pt")
    assert Path(again).read_bytes() == b"GOOD-WEIGHTS"
    assert store.reads == 2, "a digest mismatch must force a re-download"


def test_digest_sidecar_matches_the_blob(cache_dir, monkeypatch):
    from pathlib import Path

    store = _Store({"models/c/best.pt": b"WEIGHTS"})
    _use_store(monkeypatch, store)

    path = Path(runner._local_champion_weights("models/c/best.pt"))
    recorded = path.with_suffix(".sha256").read_text().strip()
    assert recorded == hashlib.sha256(b"WEIGHTS").hexdigest()


def test_unreadable_blob_returns_none_rather_than_a_bad_path(cache_dir, monkeypatch):
    _use_store(monkeypatch, _Store({}))
    assert runner._local_champion_weights("models/missing/best.pt") is None


def test_empty_blob_is_refused(cache_dir, monkeypatch):
    # A zero-byte weights file is never a loadable model; returning it would fail deep inside ultralytics.
    _use_store(monkeypatch, _Store({"models/c/best.pt": b""}))
    assert runner._local_champion_weights("models/c/best.pt") is None
