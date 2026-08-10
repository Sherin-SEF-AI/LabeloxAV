"""A dataset you can put in a training config.

    import labelox
    ds = labelox.load("night AND vru", version="2026.07.1")
    for sample in ds:
        image_bytes, labels = sample["jpg"], sample["json"]

The point is what ends up written down. A path names where somebody put a zip; a query and a version name
what the data is and when it was fixed, which is the thing whoever inherits the config in a year actually
needs. Both are recorded on the returned object, so a run can stamp them into its own metadata.

Deliberately thin. `iter_samples` yields plain dicts decoded from ordinary tar members, and `to_webdataset`
hands the shard URLs to the `webdataset` package if it is installed. Neither this module nor the format
requires anything LabeloxAV-specific to read: the shards are tar, the index is Parquet, and a team that
stops paying can still open both.
"""

from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Iterator
from typing import Any
from urllib.parse import quote

import httpx


class Dataset:
    """A resolved query: its shards, its index, and what it was selected by."""

    def __init__(self, *, query: str, version: str | None, shards: list[str], index_uri: str | None,
                 samples: int, predicate: dict, sealed: bool, client: Labelox):
        self.query = query
        self.version = version
        self.shards = shards
        self.index_uri = index_uri
        self.samples = samples
        # Kept and shown, because a dataset whose contents cannot be explained is one nobody can defend in a
        # review, and "night AND vru" is a claim about a predicate somebody should be able to check.
        self.predicate = predicate
        self.sealed = sealed
        self._client = client

    def __repr__(self) -> str:
        pin = f", version={self.version!r}" if self.version else " (unpinned)"
        return f"<labelox.Dataset {self.query!r}{pin}: {self.samples} samples in {len(self.shards)} shards>"

    def __len__(self) -> int:
        return self.samples

    def iter_samples(self) -> Iterator[dict]:
        """Yield {"key", "jpg", "json"} per sample, decoding shards in order.

        Streams: the first sample is available before the last shard has been fetched, which is the whole
        reason for shards rather than one archive.
        """
        for url in self.shards:
            data = self._client._fetch_blob(url)
            tf = tarfile.open(fileobj=io.BytesIO(data), mode="r")
            grouped: dict[str, dict] = {}
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                key, _, ext = member.name.partition(".")
                payload = tf.extractfile(member).read()
                entry = grouped.setdefault(key, {"key": key})
                entry[ext] = json.loads(payload) if ext == "json" else payload
            # Only complete samples. WebDataset pairs by key, and half a sample is not one.
            for entry in grouped.values():
                if "jpg" in entry and "json" in entry:
                    yield entry

    def to_webdataset(self, **kwargs: Any):
        """The same shards as a `webdataset.WebDataset`, for dropping straight into a DataLoader."""
        try:
            import webdataset as wds
        except ImportError as exc:   # pragma: no cover - optional dependency
            raise ImportError("pip install webdataset to use to_webdataset(); iter_samples() needs nothing "
                              "beyond the standard library") from exc
        return wds.WebDataset([self._client._shard_url(u) for u in self.shards], **kwargs)

    def index(self):
        """The Parquet index as a pyarrow Table: what is in here, without downloading the images."""
        if not self.index_uri:
            return None
        import pyarrow.parquet as pq
        return pq.read_table(io.BytesIO(self._client._fetch_blob(self.index_uri)))


class Labelox:
    """Client for the dataset surface."""

    def __init__(self, base_url: str = "http://localhost:8000", token: str | None = None,
                 timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._timeout = timeout

    def vocabulary(self) -> dict:
        return self._get("/api/datasets/vocabulary")

    def compile(self, query: str) -> dict:
        """What a query means, without building it. Cheap, and the moment to catch a wrong expansion."""
        return self._get("/api/datasets/compile", params={"q": query})

    def preview(self, query: str, version: str | None = None) -> dict:
        """Frame and object counts plus the class mix, before committing to a build."""
        p = {"q": query}
        if version:
            p["version"] = version
        return self._get("/api/datasets/preview", params=p)

    def load(self, query: str, *, version: str | None = None, samples_per_shard: int = 256,
             limit: int = 200_000) -> Dataset:
        """Resolve a query into streamable shards."""
        body = {"query": query, "version": version, "samples_per_shard": samples_per_shard,
                "limit": limit}
        r = self._post("/api/datasets/shards", body)
        return Dataset(query=query, version=version, shards=r.get("shards") or [],
                       index_uri=r.get("index_uri"), samples=int(r.get("samples") or 0),
                       predicate=r.get("predicate") or {}, sealed=bool(r.get("sealed")), client=self)

    # ---- transport -------------------------------------------------------------------------------

    def _get(self, path: str, params: dict | None = None) -> dict:
        with httpx.Client(timeout=self._timeout) as c:
            resp = c.get(f"{self.base_url}{path}", params=params, headers=self._headers)
            resp.raise_for_status()
            return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        with httpx.Client(timeout=self._timeout) as c:
            resp = c.post(f"{self.base_url}{path}", json=body, headers=self._headers)
            resp.raise_for_status()
            return resp.json()

    def _shard_url(self, uri: str) -> str:
        """The URL a reader fetches a shard from: the API, so one token governs everything."""
        return f"{self.base_url}/api/datasets/blob?uri={quote(uri, safe='')}"

    def _fetch_blob(self, uri: str) -> bytes:
        """Shard or index bytes. Goes through the API so one token governs access to everything."""
        with httpx.Client(timeout=self._timeout) as c:
            resp = c.get(f"{self.base_url}/api/datasets/blob", params={"uri": uri},
                         headers=self._headers)
            resp.raise_for_status()
            return resp.content


_default: Labelox | None = None


def configure(base_url: str = "http://localhost:8000", token: str | None = None) -> Labelox:
    """Set the client `load` uses, so the common case is a one-liner."""
    global _default
    _default = Labelox(base_url=base_url, token=token)
    return _default


def load(query: str, *, version: str | None = None, **kwargs: Any) -> Dataset:
    """`labelox.load("night AND vru", version="2026.07.1")`."""
    if _default is None:
        raise RuntimeError("call labelox.configure(base_url=..., token=...) first")
    return _default.load(query, version=version, **kwargs)
