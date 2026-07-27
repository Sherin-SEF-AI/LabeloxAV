"""Recover lost prediction provenance for objects reviewed before the prediction plane existed.

Human review overwrote the prediction row in place, so for historical objects the original detection is only
partially recoverable: Review.before carries its class_id and bbox, but never its source or conf. This script
reconstructs a Prediction for each such object from Review.before, under a synthetic reconstructed InferenceRun
with a null conf. Because conf is unavailable, the eval refuses to compute AP or a PR curve for a reconstructed
run and returns only fixed-threshold precision/recall with a caveat, so nothing downstream mistakes a
reconstructed run for a real inference. The run's params carry {"reconstructed": true, "conf_unavailable": true}.

    python scripts/backfill_prediction_from_review.py [--date-tag 20260727] [--dry-run] [--force]
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime

from sqlalchemy import select

from core.config import get_settings
from core.logging import get_logger, setup_logging
from core.version import code_sha
from db.models import InferenceRun, ModelRegistry, Object, Prediction, Review
from db.session import get_sessionmaker

log = get_logger("backfill_prediction")


async def backfill(date_tag: str | None = None, dry_run: bool = False, force: bool = False) -> dict:
    date_tag = date_tag or datetime.now(UTC).strftime("%Y%m%d")
    model_v = f"reconstructed-pre-{date_tag}"

    async with get_sessionmaker()() as db:
        if not force:
            existing = (await db.execute(select(InferenceRun).where(
                InferenceRun.model_version == model_v, InferenceRun.status == "complete"))).scalars().first()
            if existing is not None:
                log.info("backfill.already_done", run_id=str(existing.run_id), model_version=model_v)
                return {"run_id": str(existing.run_id), "reconstructed": 0, "note": "already reconstructed"}

        # The earliest review per object captured its origin (bbox + class_id); later reviews are re-edits.
        rows = (await db.execute(
            select(Review.object_id, Review.before, Review.ts_ns).order_by(Review.ts_ns.asc()))).all()
        first_before: dict = {}
        for oid, before, _ts in rows:
            first_before.setdefault(oid, before or {})
        recon = [(oid, b) for oid, b in first_before.items()
                 if b.get("bbox") and b.get("class_id") is not None]
        if not recon:
            return {"reconstructed": 0, "note": "no reviewed objects with a recoverable bbox + class_id"}

        obj_ids = [o for o, _ in recon]
        frame_of = dict((await db.execute(
            select(Object.object_id, Object.frame_id).where(Object.object_id.in_(obj_ids)))).all())

        if dry_run:
            return {"would_reconstruct": len(recon), "model_version": model_v, "dry_run": True}

        if not await db.get(ModelRegistry, model_v):
            db.add(ModelRegistry(model_version=model_v,
                                 notes="synthetic: predictions reconstructed from review history"))
            await db.flush()
        run = InferenceRun(model_version=model_v, gold_id=None, code_sha=code_sha(), status="running",
                           params={"reconstructed": True, "conf_unavailable": True})
        db.add(run)
        await db.flush()

        n = 0
        for oid, b in recon:
            fid = frame_of.get(oid)
            if fid is None:
                continue
            db.add(Prediction(run_id=run.run_id, frame_id=fid, class_id=int(b["class_id"]),
                              bbox=[float(x) for x in b["bbox"]], conf=None))
            n += 1
        run.status = "complete"
        run.finished_at = datetime.now(UTC)
        run.frame_count = len({frame_of[o] for o, _ in recon if frame_of.get(o)})
        await db.commit()
        result = {"run_id": str(run.run_id), "reconstructed": n, "model_version": model_v}
        log.info("backfill.done", **result)
        return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date-tag", default=None, help="tag for the synthetic model version (default: today)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="reconstruct even if a run for this tag exists")
    args = ap.parse_args()
    setup_logging(get_settings().log_level)
    print(asyncio.run(backfill(args.date_tag, args.dry_run, args.force)))


if __name__ == "__main__":
    main()
