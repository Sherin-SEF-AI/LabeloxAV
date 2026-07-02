"""The Documentation Agent: auto-drafts the buyer-diligence artifacts from the platform's own analytics and
governance surfaces, so datasheets, model cards, and weekly quality reports are never hand-written twice.

It only reads and renders (no label or model mutation), stores the Markdown artifact in the object store,
and records an audit entry. Reuses the analytics dashboards, the coverage report, the quality sheet, the
model registry, and the drift + audit trail.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.storage import get_object_store
from db.models import AuditDecision, DatasetCommit, DriftMetric, ModelRegistry
from services.export.datasheet import render_datasheet, render_model_card, render_weekly_report

log = get_logger("agent.doc_agent")

_KIND = "doc_agent"


def _store_doc(kind: str, ident: str, markdown: str) -> str:
    store = get_object_store()
    store.ensure_bucket()
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in ident)[:64] or "corpus"
    return store.put_bytes(f"docs/{kind}/{safe}.md", markdown.encode(), "text/markdown")


async def _audit(db: AsyncSession, decision: str, subject: str, uri: str) -> None:
    try:
        from services.govern.audit import record

        await record(db, _KIND, decision, subject, {"uri": uri})
    except Exception:  # noqa: BLE001
        pass


async def generate_datasheet(db: AsyncSession, gold_id: str | None = None, title: str | None = None) -> dict:
    from services.agent.coverage import analyze_coverage
    from services.analytics.dashboards import class_distribution, overview
    from services.autolabel.ontology import get_ontology

    ov = await overview()
    cd = await class_distribution()
    cov = await analyze_coverage(db)
    quality = None
    if gold_id:
        from services.analytics.quality import quality_sheet

        quality = await quality_sheet(gold_id)
    size = {"sessions": ov["sessions"], "frames": ov["frames"], "objects": ov["objects"],
            "human_touched": f"{ov['human_touched_pct']}%", "ontology_version": get_ontology().version}
    md = render_datasheet(title=title or "LabeloxAV corpus", size=size, coverage=cov, class_dist=cd,
                          scene=cov.get("scene_coverage", {}), geo=cov.get("geo", {}), quality=quality)
    uri = _store_doc("datasheet", gold_id or "corpus", md)
    await _audit(db, "datasheet", gold_id or "corpus", uri)
    log.info("doc.datasheet", uri=uri, gold_id=gold_id)
    return {"uri": uri, "markdown": md}


async def generate_model_card(db: AsyncSession, model_version: str) -> dict:
    m = await db.get(ModelRegistry, model_version)
    if m is None:
        raise ValueError("model not found")
    commit = await db.get(DatasetCommit, m.dataset_commit) if m.dataset_commit else None
    model = {"model_version": m.model_version, "task": m.task, "is_champion": m.is_champion,
             "promoted_from": m.promoted_from, "weights_uri": m.weights_uri, "dataset_commit": m.dataset_commit,
             "gold_metrics": m.gold_metrics}
    dc = None
    if commit is not None:
        dc = {"commit_id": commit.commit_id, "object_count": commit.object_count,
              "ontology_version": commit.ontology_version}
    md = render_model_card(model=model, dataset_commit=dc)
    uri = _store_doc("model-card", m.model_version, md)
    await _audit(db, "model_card", m.model_version, uri)
    log.info("doc.model_card", uri=uri, model=model_version)
    return {"uri": uri, "markdown": md}


async def generate_weekly_report(db: AsyncSession) -> dict:
    from services.agent.coverage import analyze_coverage
    from services.govern.control_sample import measured_precision

    week_ago = datetime.now(timezone.utc) - timedelta(days=7)
    prec = await measured_precision(db)
    drift = [{"metric": d.metric, "value": d.value, "breach": d.breach} for d in (await db.execute(
        select(DriftMetric).where(DriftMetric.created_at >= week_ago)
        .order_by(DriftMetric.created_at.desc()).limit(50))).scalars().all()]
    promos = [{"subject": a.subject, "decision": a.decision} for a in (await db.execute(
        select(AuditDecision).where(AuditDecision.decision.in_(["promote", "evaluate_challenger", "rollback"]),
                                    AuditDecision.created_at >= week_ago).limit(30))).scalars().all()]
    cov = await analyze_coverage(db)
    md = render_weekly_report(precision=prec, drift=drift, promotions=promos, coverage_gaps=cov.get("gaps", []))
    uri = _store_doc("weekly", "latest", md)
    await _audit(db, "weekly_report", "latest", uri)
    log.info("doc.weekly", uri=uri)
    return {"uri": uri, "markdown": md}
