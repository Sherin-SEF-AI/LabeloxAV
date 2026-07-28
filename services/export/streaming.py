"""Stream a sealed dataset out as a ZIP, without ever holding it in memory.

Exports are fully materialised before they are served: the writers build the whole tree on disk, and the
only way to take delivery of it was to fetch each file through its own presigned URL. For a dataset of forty
thousand frames that is forty thousand round trips, and any naive attempt to bundle it inside the API
process would build the archive in memory and take the process down with it.

The generator below yields a valid ZIP a chunk at a time. Three things make that work:

- **ZIP64 is on unconditionally.** A dataset that crosses 4 GB, or 65535 entries, silently produces a
  corrupt archive under the classic format. Both are ordinary sizes here.
- **Stored, not deflated.** The bulk of an export is JPEG and PNG, which are already compressed; deflating
  them spends CPU proportional to the archive to save almost nothing, and it is the compression that would
  otherwise force buffering to know the compressed size.
- **One object in flight at a time.** Each file is fetched from the object store, written, and released
  before the next is fetched, so peak memory is one file rather than one dataset.

An unreadable file is skipped and recorded in a manifest inside the archive rather than aborting the stream.
Half a gigabyte into a download is the worst possible moment to discover that one blob is missing, and the
consumer needs to know which one rather than starting again.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import AsyncIterator, Iterator

from core.logging import get_logger

log = get_logger("export_stream")

# 1 MiB. Big enough that the per-chunk overhead is irrelevant, small enough that a slow consumer does not
# leave tens of megabytes pinned in the writer's buffer.
CHUNK_BYTES = 1024 * 1024


class _Sink(io.RawIOBase):
    """A file-like object that accumulates writes for the generator to drain.

    zipfile wants something to write to; an HTTP response wants something to read from. This is the join:
    zipfile writes here, the generator drains it after each entry, and nothing accumulates beyond one file.
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    def writable(self) -> bool:
        return True

    def write(self, b) -> int:  # type: ignore[override]
        self._buf.extend(b)
        return len(b)

    def drain(self) -> bytes:
        out = bytes(self._buf)
        self._buf.clear()
        return out


def stream_zip(entries: Iterator[tuple[str, bytes]], *,
               manifest: dict | None = None) -> Iterator[bytes]:
    """Yield a ZIP built from (arcname, content) pairs.

    Synchronous and iterator-based so it can be driven from either an async route or a script. The caller
    supplies content per entry, which is what keeps the fetch lazy: an entry is only produced when the
    stream is ready for it.
    """
    sink = _Sink()
    skipped: list[dict] = []
    written = 0

    # allowZip64 is not conditional. Guessing wrong produces an archive that unzips on the developer's
    # sample and fails on the real dataset.
    with zipfile.ZipFile(sink, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for arcname, content in entries:
            if content is None:
                skipped.append({"path": arcname, "reason": "unreadable"})
                continue
            try:
                zf.writestr(arcname, content)
                written += 1
            except Exception as exc:  # noqa: BLE001
                skipped.append({"path": arcname, "reason": type(exc).__name__})
                continue
            chunk = sink.drain()
            if chunk:
                yield chunk

        # The manifest goes inside the archive, so a consumer who receives a partial dataset learns which
        # files are absent from the archive itself rather than from a log they cannot see.
        body = {**(manifest or {}), "files_written": written, "files_skipped": skipped}
        zf.writestr("MANIFEST.json", json.dumps(body, indent=2))

    tail = sink.drain()
    if tail:
        yield tail
    log.info("export.streamed", files=written, skipped=len(skipped))


async def stream_commit(commit_id: str) -> AsyncIterator[bytes]:
    """Stream one sealed dataset commit as a ZIP.

    The object store is fetched lazily inside the generator, one file at a time, so peak memory is a single
    file however large the dataset is.
    """
    from core.storage import get_object_store
    from db.models import DatasetCommit
    from db.session import get_sessionmaker

    async with get_sessionmaker()() as db:
        commit = await db.get(DatasetCommit, commit_id)
        if commit is None:
            raise ValueError(f"dataset commit {commit_id!r} not found")
        uris = dict(commit.export_uris or {})
        spec = dict(commit.slice_spec or {})
        ontology_version = commit.ontology_version

    store = get_object_store()

    def _entries() -> Iterator[tuple[str, bytes]]:
        for rel, uri in sorted(uris.items()):
            try:
                yield rel, store.get_bytes(uri)
            except Exception as exc:  # noqa: BLE001
                # Skipped rather than fatal: aborting half a gigabyte in is the worst possible outcome, and
                # the consumer needs to know which file rather than starting again.
                log.warning("export.stream_file_failed", path=rel, error=str(exc))
                yield rel, None  # type: ignore[misc]

    manifest = {"commit_id": commit_id, "name": spec.get("name"),
                "formats": spec.get("formats", []), "ontology_version": ontology_version,
                "files_expected": len(uris)}
    for chunk in stream_zip(_entries(), manifest=manifest):
        yield chunk


def suggested_filename(commit_id: str, name: str | None) -> str:
    stem = "".join(c for c in (name or "dataset") if c.isalnum() or c in "-_") or "dataset"
    return f"{stem}-{commit_id[:12]}.zip"
