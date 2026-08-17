"""Labels from an outside workforce, coming back into the corpus.

Work could be dispatched and could not return. `dispatch_job` POSTs a batch of frame ids to a vendor and
`submit_return` takes a *claimed* count of objects, and between those two there was no path by which the
vendor's actual annotations became rows. A Workforce carries an HMAC secret for signing its callbacks and
nothing else: no user, no role, no API key, so it cannot write through the object API either.

That hole reached further than the missing feature. `score_honeypots` grades "human-sourced objects on the
honeypot frames that are not the gold rows", which for an in-house annotator working in the editor is their
work and for a vendor is the empty set. The quality gate that decides whether to accept a batch was reading
rows the vendor had no way to create, so a vendor's accuracy was never a measurement of the vendor.

This is the return leg. It deliberately reuses the import adapters rather than parsing anything itself: the
same fifteen formats a dataset can arrive in are the formats a batch can come back in, and a second CVAT
parser would be a second thing to keep correct.

Three refusals matter more than the happy path.

A returned file may only touch the frames that were dispatched. Annotations naming any other frame are
refused, not merged: an outside party writing into the corpus must not be able to choose where.

A class name the ontology cannot place lands in a fallback bucket and is counted as one. It is not minted,
because a vendor typing a class name by hand is exactly how a sidecar class nothing can store gets created,
and it is not discarded either: silently dropping labelling somebody was paid for is the worse of the two
failures, and the count is what makes the fallback visible instead of looking like a real answer.

The whole batch is one reversible run, because the honeypot verdict arrives after the write. A batch that
fails the quality bar has to leave the corpus completely, and per-object undo of somebody else's work is how
half a batch stays behind forever.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.timebase import now_ns
from db.models import AgentRun, Frame, LabelJob, Object, WorkforceAssignment

log = get_logger("workforce.return_ingest")

RUN_KIND = "workforce_return"

# States in which a batch may still be ingested. A decided assignment is finished, and accepting more work
# against it would let a vendor keep writing after its verdict.
INGESTABLE_STATES = ("dispatched", "returned")


class ReturnIngestError(RuntimeError):
    pass


def _frame_id_of(image_ref: str) -> uuid.UUID | None:
    """The frame a returned image refers to, or None if the name does not name one.

    Every exporter writes a dispatched frame as `<frame_id>.<ext>`, so the reference the vendor sends back
    carries the identity we gave it. Matching on the name rather than on order is what makes a partially
    returned or reordered batch safe.
    """
    stem = Path(image_ref).name.rsplit(".", 1)[0]
    try:
        return uuid.UUID(stem)
    except (ValueError, AttributeError):
        return None


async def ingest_return(db: AsyncSession, *, assignment_id: str, fmt: str, root: Path,
                        created_by: str | None = None) -> dict:
    """Parse a returned batch and write its objects against the dispatched frames.

    Returns counts and the id of the reversible run that carries them. Idempotent per assignment: a vendor
    retrying an upload gets the first run's answer rather than a second copy of its work.
    """
    from services.imports.run import ADAPTERS

    asg = await db.get(WorkforceAssignment, uuid.UUID(assignment_id))
    if asg is None:
        raise ReturnIngestError("assignment not found")
    if asg.state not in INGESTABLE_STATES:
        raise ReturnIngestError(f"assignment is {asg.state}; a decided batch cannot be added to")

    existing = (await db.execute(
        select(AgentRun).where(AgentRun.kind == RUN_KIND,
                               AgentRun.scope["assignment_id"].astext == str(asg.assignment_id),
                               AgentRun.status == "committed"))).scalars().first()
    if existing is not None:
        return {"run_id": str(existing.run_id), "counts": existing.counts,
                "note": "this assignment was already ingested; the batch was not written twice"}

    job = await db.get(LabelJob, asg.job_id)
    if job is None:
        raise ReturnIngestError("job missing for this assignment")
    if fmt not in ADAPTERS:
        raise ReturnIngestError(f"unknown return format: {fmt} (choose from {sorted(ADAPTERS)})")

    dispatched = {uuid.UUID(str(f)) for f in (job.frame_ids or [])}
    if not dispatched:
        raise ReturnIngestError("this job dispatched no frames, so nothing can be returned against it")

    frames = ADAPTERS[fmt](root)
    counts = {"frames_seen": len(frames), "frames_matched": 0, "objects_written": 0,
              "unresolvable_names": 0, "foreign_frames": 0, "fallback_classes": 0}
    run_id = uuid.uuid4()
    changes: dict[str, dict] = {}

    from services.autolabel.ontology import get_ontology
    from services.imports.remap import remap_name

    onto = get_ontology()
    ts = now_ns()

    for rec in frames:
        fid = _frame_id_of(rec.image_ref)
        if fid is None:
            counts["unresolvable_names"] += 1
            continue
        if fid not in dispatched:
            # Refused rather than merged. The dispatch named the frames; anything else is the vendor
            # choosing where to write in somebody else's corpus.
            counts["foreign_frames"] += 1
            log.warning("workforce.return.foreign_frame", assignment=assignment_id, frame_id=str(fid))
            continue
        frame = await db.get(Frame, fid)
        if frame is None:
            counts["foreign_frames"] += 1
            continue
        counts["frames_matched"] += 1

        for ob in rec.objects:
            # remap_name always yields a storable class: a name it cannot place becomes a fallback bucket
            # rather than a new class. Counted so a batch that mapped badly is visible as one.
            class_id, _mapped_name, placed = remap_name(ob.name, onto)
            if not placed:
                counts["fallback_classes"] += 1
            oid = uuid.uuid4()
            db.add(Object(
                object_id=oid, frame_id=fid, class_id=class_id, bbox=[float(v) for v in ob.bbox],
                conf=float(ob.conf or 1.0),
                # A person made this label, which is what `human` records; who that person worked for is
                # provenance. Left in review because the honeypot verdict has not happened yet, and a batch
                # that has not passed its quality bar must not enter the corpus as settled truth.
                source="human", state="review", attrs=dict(ob.attrs or {}),
                provenance={"workforce_id": str(asg.workforce_id), "assignment_id": str(asg.assignment_id),
                            "job_id": str(job.job_id), "ingest_run_id": str(run_id),
                            "return_format": fmt, "external_class": ob.name},
                version=1, rot_deg=float(ob.rot_deg or 0.0),
            ))
            # The run records what to remove, not what to restore: these objects did not exist before it.
            changes[str(oid)] = {"created": True, "frame_id": str(fid), "class_id": class_id}
            counts["objects_written"] += 1

    db.add(AgentRun(run_id=run_id, kind=RUN_KIND,
                    scope={"assignment_id": str(asg.assignment_id), "job_id": str(job.job_id),
                           "workforce_id": str(asg.workforce_id), "format": fmt, "ts_ns": ts},
                    status="committed", policy={"format": fmt}, counts=counts, changes=changes,
                    critic={}, created_by=created_by))
    await db.commit()
    log.info("workforce.return.ingested", assignment=assignment_id, run_id=str(run_id), **counts)
    return {"run_id": str(run_id), "counts": counts}


async def revert_ingest(db: AsyncSession, run_id: str) -> dict:
    """Remove an ingested batch whole.

    The honeypot verdict lands after the write, so a rejected batch has to be able to leave the corpus
    completely. Deleting only the objects this run created, and only those still carrying its run id, so a
    label a reviewer has since edited is left alone rather than silently removed underneath them.
    """
    run = await db.get(AgentRun, uuid.UUID(run_id))
    if run is None or run.kind != RUN_KIND:
        raise ReturnIngestError("no such return-ingest run")

    removed = kept = 0
    for oid in (run.changes or {}):
        obj = await db.get(Object, uuid.UUID(oid))
        if obj is None:
            continue
        if (obj.provenance or {}).get("ingest_run_id") != run_id:
            kept += 1          # somebody has since rewritten this object; it is theirs now
            continue
        await db.delete(obj)
        removed += 1
    run.status = "reverted"
    await db.commit()
    log.info("workforce.return.reverted", run_id=run_id, removed=removed, kept=kept)
    return {"run_id": run_id, "removed": removed, "kept": kept}
