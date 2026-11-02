"""Re-check stored frames for plates and faces the first pass missed, across the corpus.

    python -m scripts.pii_recheck_sweep --types plate               # dry run: counts only, nothing written
    python -m scripts.pii_recheck_sweep --types plate --apply       # blur, permanently
    python -m scripts.pii_recheck_sweep --types face --min-score face=0.75 --apply

Auto-labelling re-checks each frame it labels (services/autolabel/runner.py), but frames labelled before that
existed never got the second look. This walks every frame that has vehicle or person boxes (the re-check
needs them for its crops) and runs services/anonymize/recheck.py on each.

A blur cannot be undone, so without --apply nothing is written. With it, progress is kept in a checkpoint
file, so a sweep stopped part way resumes where it left off rather than starting again; the checkpoint is
per type set, since a plates pass does not mean the faces pass has run. The GPU slot is taken one batch at
a time, so a queued auto-label job gets the card between batches instead of waiting hours.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

import click
from sqlalchemy import select

from core.gpu_slot import gpu_slot
from core.logging import get_logger, setup_logging
from core.storage import get_object_store
from db.models import Frame, Object
from db.session import get_sessionmaker

log = get_logger("pii_recheck_sweep")

BATCH = 50


def _checkpoint(types: frozenset[str], min_scores: dict[str, float]) -> Path:
    tag = "-".join(sorted(types)) + "".join(f"-{k}{v}" for k, v in sorted(min_scores.items()))
    return Path(".scratch") / f"pii_recheck_{tag}.done"


async def _frames() -> list[uuid.UUID]:
    """Every frame with at least one machine-labelled object, in a stable order."""
    async with get_sessionmaker()() as db:
        return list((await db.execute(
            select(Frame.frame_id).where(Frame.frame_id.in_(
                select(Object.frame_id).where(Object.source != "human").distinct()))
            .order_by(Frame.frame_id))).scalars().all())


async def sweep(types: frozenset[str], min_scores: dict[str, float], apply: bool, limit: int | None) -> dict:
    from services.anonymize.anonymizer import get_anonymizer
    from services.anonymize.recheck import recheck_frame

    store = get_object_store()
    anonymizer = get_anonymizer()
    ckpt = _checkpoint(types, min_scores)
    done: set[str] = set(ckpt.read_text().split()) if apply and ckpt.exists() else set()
    todo = [f for f in await _frames() if str(f) not in done]
    if limit:
        todo = todo[:limit]
    totals = {"frames": 0, "frames_changed": 0, "plates_added": 0, "faces_added": 0, "failed": 0,
              "unreadable": 0, "skipped_done": len(done), "to_do": len(todo), "apply": apply}
    t0 = time.time()
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    with ckpt.open("a") as out:
        for start in range(0, len(todo), BATCH):
            batch = todo[start:start + BATCH]
            async with gpu_slot("pii-recheck-sweep", timeout_s=None):
                async with get_sessionmaker()() as db:
                    for fid in batch:
                        frame = await db.get(Frame, fid)
                        if frame is None:
                            continue
                        try:
                            res = await recheck_frame(db, store, anonymizer, frame, apply=apply, types=types,
                                                      min_scores=min_scores or None)
                            if apply:
                                await db.commit()
                        except Exception as exc:  # noqa: BLE001  one frame's failure must not end the sweep
                            await db.rollback()
                            totals["failed"] += 1
                            log.warning("sweep.frame_failed", frame_id=str(fid), error=str(exc))
                            continue
                        totals["frames"] += 1
                        if res["status"] == "unreadable":
                            totals["unreadable"] += 1
                        if res["regions_added"]:
                            totals["frames_changed"] += 1
                        totals["plates_added"] += res["plates_added"]
                        totals["faces_added"] += res["faces_added"]
                        if apply:
                            out.write(f"{fid}\n")
                    if not apply:
                        await db.rollback()
            out.flush()
            n = totals["frames"]
            rate = n / max(1e-6, time.time() - t0)
            log.info("sweep.progress", done=n, of=len(todo), plates=totals["plates_added"],
                     faces=totals["faces_added"], failed=totals["failed"],
                     eta_min=round((len(todo) - n) / max(rate, 1e-6) / 60))
    totals["seconds"] = round(time.time() - t0)
    return totals


@click.command()
@click.option("--types", default="plate", help="comma-separated redaction targets: plate, face")
@click.option("--min-score", "min_score", multiple=True, help="type=score, e.g. face=0.75")
@click.option("--apply", is_flag=True, help="blur and record; without it nothing is written")
@click.option("--limit", type=int, default=None)
def main(types: str, min_score: tuple[str, ...], apply: bool, limit: int | None) -> None:
    setup_logging()
    ts = frozenset(t.strip() for t in types.split(",") if t.strip())
    scores = {k: float(v) for k, v in (m.split("=", 1) for m in min_score)}
    print(json.dumps(asyncio.run(sweep(ts, scores, apply, limit)), indent=1))


if __name__ == "__main__":
    main()
