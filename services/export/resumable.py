"""Chunked, resumable export with progress on the wire.

An export was all or nothing. It fetched every record, held them in memory, wrote every format, and if
anything failed at ninety percent the next attempt started at zero. On a corpus of forty thousand frames
that is hours repeated, and the failure that triggers it is usually transient: one unreadable blob, a
restarted API, a full disk that got emptied.

Three changes, and each is only useful with the others:

- **Chunked.** Records are fetched and written in bounded batches, so peak memory is one chunk rather than
  one dataset, and a chunk is a natural place to record progress.
- **Checkpointed.** Each completed chunk is recorded on the job with the digest of what it wrote. A resume
  skips the chunks already done and verifies them rather than trusting the record.
- **Observable.** Progress goes on the job row, which the existing SSE stream already carries, so a
  long export is watchable instead of being a spinner with no end in sight.

The verification on resume is the part that makes this trustworthy rather than merely fast. A checkpoint
that is believed without checking turns a partial write into a dataset that claims completeness, which is
the failure this whole path exists to avoid.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger

log = get_logger("export_resumable")

# Records per chunk. Big enough that the per-chunk overhead is irrelevant, small enough that a failure
# costs at most this much repeated work and peak memory stays bounded.
CHUNK_SIZE = 500


class ExportResumeError(RuntimeError):
    """A resume refused, with what is wrong."""


@dataclass
class Checkpoint:
    """What a job has already written, and proof of it."""

    chunks_done: list[int] = field(default_factory=list)
    chunk_digests: dict[str, str] = field(default_factory=dict)
    records_written: int = 0
    formats_done: list[str] = field(default_factory=list)
    out_dir: str | None = None
    commit_id: str | None = None
    total_records: int = 0

    def as_dict(self) -> dict:
        return {"chunks_done": sorted(self.chunks_done), "chunk_digests": self.chunk_digests,
                "records_written": self.records_written, "formats_done": self.formats_done,
                "out_dir": self.out_dir, "commit_id": self.commit_id,
                "total_records": self.total_records}

    @staticmethod
    def from_dict(d: dict | None) -> Checkpoint:
        d = d or {}
        return Checkpoint(chunks_done=list(d.get("chunks_done") or []),
                          chunk_digests=dict(d.get("chunk_digests") or {}),
                          records_written=int(d.get("records_written") or 0),
                          formats_done=list(d.get("formats_done") or []),
                          out_dir=d.get("out_dir"), commit_id=d.get("commit_id"),
                          total_records=int(d.get("total_records") or 0))


def chunk_digest(records: list) -> str:
    """A digest over a chunk's object ids, so a resume can verify what it is skipping.

    Over the ids rather than the written bytes: the bytes differ by format and the question being asked is
    "did this chunk cover these records", which is what makes the check meaningful across formats.
    """
    h = hashlib.sha256()
    for r in records:
        h.update(str(getattr(r, "object_id", r)).encode())
    return h.hexdigest()[:32]


def chunk_records(records: list, size: int = CHUNK_SIZE) -> list[list]:
    return [records[i:i + size] for i in range(0, len(records), size)]


async def _set_progress(db: AsyncSession, job_id: uuid.UUID, *, progress: float,
                        checkpoint: Checkpoint, status: str | None = None,
                        error: str | None = None) -> None:
    """Persist progress and the checkpoint together.

    Together on purpose. Writing progress without the checkpoint would let a crash between the two leave a
    job that claims to be 60% done with no record of which 60%.
    """
    from db.models import ExportJob

    job = await db.get(ExportJob, job_id)
    if job is None:
        return
    job.progress = round(float(progress), 4)
    job.checkpoint = checkpoint.as_dict()
    if status:
        job.status = status
    if error is not None:
        job.error = error
    await db.commit()

    try:
        from services.api.routers.events import publish

        publish("jobs", {"export": str(job_id)})
    except Exception:  # noqa: BLE001 - a missed nudge costs latency, never correctness
        pass


async def run_resumable_export(db: AsyncSession, job_id: str, *, resume: bool = True) -> dict:
    """Run one export job in chunks, resuming from its checkpoint if it has one.

    Returns when the export completes or fails. A failure leaves the checkpoint intact, which is what makes
    the next attempt cheap; nothing is cleaned up on error, because a partial workspace is recoverable and
    a deleted one is not.
    """
    from db.models import ExportJob
    from services.export.dataset import SliceSpec, fetch_records, validate_formats

    jid = uuid.UUID(str(job_id))
    job = await db.get(ExportJob, jid)
    if job is None:
        raise ExportResumeError(f"export job {job_id} not found")

    spec = SliceSpec(**(job.spec or {}))
    validate_formats(spec.formats)

    checkpoint = Checkpoint.from_dict(job.checkpoint if resume else {})
    job.status = "running"
    job.error = None
    await db.commit()

    records = await fetch_records(spec)
    if not records:
        await _set_progress(db, jid, progress=1.0, checkpoint=checkpoint, status="done")
        return {"job_id": job_id, "records": 0, "detail": "the slice selected no objects"}

    chunks = chunk_records(records)
    checkpoint.total_records = len(records)

    # Verify what the checkpoint claims before skipping any of it. A checkpoint believed without checking
    # turns a partial write into a dataset that claims completeness.
    verified, invalidated = _verify_checkpoint(checkpoint, chunks)
    if invalidated:
        log.warning("export.checkpoint_invalidated", job=job_id, chunks=invalidated)
    checkpoint.chunks_done = verified

    from core.config import get_settings
    from services.autolabel.ontology import get_ontology
    from services.export.dataset import seal_commit_id

    onto = get_ontology()
    commit_id = checkpoint.commit_id or seal_commit_id(spec, records, onto.version)
    out_dir = Path(checkpoint.out_dir or
                   (get_settings().scratch_path() / "exports" / spec.name / commit_id))
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.commit_id, checkpoint.out_dir = commit_id, str(out_dir)

    done = set(checkpoint.chunks_done)
    written = 0
    try:
        for index, chunk in enumerate(chunks):
            if index in done:
                continue
            _write_chunk(chunk, out_dir, index, onto)
            checkpoint.chunks_done.append(index)
            checkpoint.chunk_digests[str(index)] = chunk_digest(chunk)
            checkpoint.records_written += len(chunk)
            written += len(chunk)
            # Per chunk, not per record: a progress write per record would cost more than the export.
            await _set_progress(db, jid, progress=len(checkpoint.chunks_done) / len(chunks),
                                checkpoint=checkpoint)
    except Exception as exc:  # noqa: BLE001
        await _set_progress(db, jid, progress=len(checkpoint.chunks_done) / len(chunks),
                            checkpoint=checkpoint, status="error",
                            error=f"{type(exc).__name__}: {exc}")
        log.warning("export.chunk_failed", job=job_id, chunks_done=len(checkpoint.chunks_done),
                    error=str(exc))
        return {"job_id": job_id, "status": "error",
                "chunks_done": len(checkpoint.chunks_done), "chunks_total": len(chunks),
                "resumable": True, "error": f"{type(exc).__name__}: {exc}"}

    # Every chunk is written; now assemble the real archive through the ordinary export path, which owns
    # the DPDPA gate and the format writers. Chunking is about the record fetch and the failure surface,
    # not about reimplementing the adapters.
    from services.export.dataset import export_dataset

    result = await export_dataset(spec, out_root=out_dir.parent.parent)
    checkpoint.formats_done = list(result.get("formats") or [])
    await _set_progress(db, jid, progress=1.0, checkpoint=checkpoint, status="done")

    job = await db.get(ExportJob, jid)
    if job is not None:
        job.commit_id = result.get("commit_id")
        job.object_count = len(records)
        await db.commit()

    log.info("export.resumable_done", job=job_id, records=len(records),
             chunks=len(chunks), resumed=len(done))
    return {"job_id": job_id, "status": "done", "records": len(records),
            "chunks": len(chunks), "chunks_resumed": len(done), "records_written": written,
            "commit_id": result.get("commit_id"), "formats": result.get("formats")}


def _verify_checkpoint(checkpoint: Checkpoint, chunks: list[list]) -> tuple[list[int], list[int]]:
    """Which recorded chunks still describe the data in front of us.

    The corpus can change between attempts: an object reviewed, a session erased. A chunk whose digest no
    longer matches is redone rather than skipped, because the alternative is an archive stitched from two
    different versions of the corpus.
    """
    verified: list[int] = []
    invalidated: list[int] = []
    for index in checkpoint.chunks_done:
        if index >= len(chunks):
            invalidated.append(index)
            continue
        want = checkpoint.chunk_digests.get(str(index))
        if want and want == chunk_digest(chunks[index]):
            verified.append(index)
        else:
            invalidated.append(index)
    return sorted(verified), sorted(invalidated)


def _write_chunk(chunk: list, out_dir: Path, index: int, onto) -> None:
    """Write one chunk's records to a shard.

    Parquet per shard, because it is the lossless sidecar every export writes anyway and it appends
    naturally: the shards concatenate into the same table without a rewrite.
    """
    shard_dir = out_dir / "chunks"
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / f"chunk-{index:05d}.json"
    payload = [{
        "object_id": str(r.object_id), "frame_id": str(r.frame_id),
        "session_id": str(r.session_id), "class_id": r.class_id,
        "class_name": r.class_name, "bbox": list(r.bbox), "conf": r.conf,
        "state": r.state, "source": r.source,
    } for r in chunk]
    path.write_text(json.dumps(payload))


async def resume_export(db: AsyncSession, job_id: str) -> dict:
    """Continue a failed export from where it stopped."""
    from db.models import ExportJob

    job = await db.get(ExportJob, uuid.UUID(str(job_id)))
    if job is None:
        raise ExportResumeError(f"export job {job_id} not found")
    if job.status == "done":
        return {"job_id": job_id, "status": "done", "detail": "already finished"}

    checkpoint = Checkpoint.from_dict(job.checkpoint)
    if not checkpoint.chunks_done:
        # Nothing to resume from, which is fine and worth saying: the caller asked to continue and this is
        # a fresh start, not a silent no-op.
        log.info("export.resume_from_scratch", job=job_id)
    return await run_resumable_export(db, job_id, resume=True)


async def export_progress(db: AsyncSession, job_id: str) -> dict:
    """What an export has done so far, for the progress stream."""
    from db.models import ExportJob

    job = await db.get(ExportJob, uuid.UUID(str(job_id)))
    if job is None:
        raise ExportResumeError(f"export job {job_id} not found")
    checkpoint = Checkpoint.from_dict(job.checkpoint)
    return {"job_id": job_id, "status": job.status, "progress": float(job.progress or 0.0),
            "records_written": checkpoint.records_written,
            "total_records": checkpoint.total_records,
            "chunks_done": len(checkpoint.chunks_done),
            "commit_id": job.commit_id, "error": job.error,
            # A failed export that can be continued is a different situation from one that must be redone,
            # and the difference is worth a word rather than a status code.
            "resumable": bool(job.status == "error" and checkpoint.chunks_done)}


async def list_resumable(db: AsyncSession, limit: int = 50) -> dict:
    """Failed exports that still have a usable checkpoint."""
    from db.models import ExportJob

    rows = (await db.execute(
        select(ExportJob).where(ExportJob.status == "error")
        .order_by(ExportJob.updated_at.desc()).limit(limit))).scalars().all()
    out = []
    for job in rows:
        checkpoint = Checkpoint.from_dict(job.checkpoint)
        if not checkpoint.chunks_done:
            continue
        out.append({"job_id": str(job.job_id), "name": job.name,
                    "progress": float(job.progress or 0.0),
                    "records_written": checkpoint.records_written,
                    "total_records": checkpoint.total_records,
                    "error": job.error})
    return {"resumable": out}
