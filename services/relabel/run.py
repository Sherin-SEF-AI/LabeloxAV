"""Relabel job orchestration (M4.2). compute_target='local' runs ontology-promotion relabeling inline
(deterministic, no model). compute_target='cloud' bursts the champion-model re-inference to the A100 via
the seam; the pod emits relabeled.jsonl which ingest_model_relabel applies through the same diff + apply
path. Each run records a relabel_run with its lakeFS branch and counts, and is reversible. Mirrors the
autolabel/map-fusion job lifecycle."""

from __future__ import annotations

import uuid

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import RelabelJob, RelabelRun
from db.session import get_sessionmaker
from services.relabel.apply import apply_relabel
from services.relabel.reinfer import parse_model_proposals, propose_ontology_promotion

log = get_logger("relabel_run")


async def start_relabel(model_version: str, session_ids: list[str] | None = None,
                        ontology_promotion: dict | None = None, compute_target: str = "local") -> dict:
    maker = get_sessionmaker()
    async with maker() as db:
        job = RelabelJob(model_version=model_version, session_ids=session_ids or [],
                         ontology_promotion=ontology_promotion, compute_target=compute_target, status="pending")
        db.add(job)
        await db.flush()
        job_id = job.job_id
        await db.commit()

    if compute_target == "cloud":
        from services.relabel.cloud import mark_queued_for_cloud_relabel

        await mark_queued_for_cloud_relabel(job_id, model_version, session_ids)
        return {"job_id": str(job_id), "compute_target": "cloud", "status": "queued_for_cloud"}

    res = await run_relabel_job(job_id)
    return {"job_id": str(job_id), "compute_target": "local", **res}


async def run_relabel_job(job_id: uuid.UUID) -> dict:
    """Local execution: ontology-promotion relabeling end to end (build -> diff -> apply -> seal run)."""
    maker = get_sessionmaker()
    async with maker() as db:
        job = await db.get(RelabelJob, job_id)
        if job is None:
            return {"error": "job not found"}
        await db.execute(update(RelabelJob).where(RelabelJob.job_id == job_id).values(status="running", stage="build", progress=0.1))
        await db.commit()

        if not job.ontology_promotion:
            await db.execute(update(RelabelJob).where(RelabelJob.job_id == job_id).values(
                status="done", stage="done", progress=1.0,
                result={"note": "no ontology promotion given; model re-inference runs on the cloud burst"}))
            await db.commit()
            return {"proposed": 0, "note": "local relabel handles ontology promotion; use compute_target=cloud for model re-inference"}

        prom = job.ontology_promotion
        proposals = await propose_ontology_promotion(db, prom["from_class"], prom["to_class"], list(job.session_ids) or None)
        await db.execute(update(RelabelJob).where(RelabelJob.job_id == job_id).values(stage="apply", progress=0.6,
                                                                                       counts={"proposed": len(proposals)}))
        await db.commit()

        # pre-mint the run id so the provenance history, the returned run_id, and the RelabelRow all match
        run_id = uuid.uuid4()
        res = await apply_relabel(db, proposals, job.model_version,
                                  branch=f"relabel-{prom['to_class']}-{str(job_id)[:8]}", run_id=str(run_id))
        run = RelabelRun(run_id=run_id, model_version=job.model_version, lakefs_branch=res["branch"],
                         proposed=res["proposed"], auto_applied=res["applied"], routed_to_review=res["routed_to_review"],
                         regressions_flagged=res["regressions"],
                         reason=f"ontology promotion: {prom['from_class']} -> {prom['to_class']}", job_id=job_id)
        db.add(run)
        await db.flush()
        await db.execute(update(RelabelJob).where(RelabelJob.job_id == job_id).values(
            status="done", stage="done", progress=1.0, run_id=run_id, result={**res, "run_id": str(run_id)}))
        await db.commit()

    log.info("relabel.job_done", job_id=str(job_id), **{k: res[k] for k in ("applied", "routed_to_review", "branch")})
    return {**res, "run_id": str(run_id)}


async def ingest_model_relabel(model_version: str, model_output: list[dict],
                               session_ids: list[str] | None = None) -> dict:
    """Apply champion-model re-inference output (the pod's relabeled.jsonl) through the diff + apply path."""
    maker = get_sessionmaker()
    async with maker() as db:
        proposals = await parse_model_proposals(db, model_output)
        run_id = uuid.uuid4()
        res = await apply_relabel(db, proposals, model_version, run_id=str(run_id))
        run = RelabelRun(run_id=run_id, model_version=model_version, lakefs_branch=res["branch"], proposed=res["proposed"],
                         auto_applied=res["applied"], routed_to_review=res["routed_to_review"],
                         regressions_flagged=res["regressions"], reason="champion model re-inference")
        db.add(run)
        await db.commit()
    return res
