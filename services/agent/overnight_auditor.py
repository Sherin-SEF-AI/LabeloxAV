"""The Overnight Auditor: a scheduled watchdog that patrols the day's auto-accepted labels while the team
sleeps and writes a morning report.

It samples the labels the gate auto-accepted in the last window, spends a bounded VLM budget spot-checking
them against an independent model, runs the cross-frame consistency critic, folds in the true auto-accept
precision measured from the control sample, and surfaces per-class confusion movers correlated with scene
(e.g. "e_auto confusion up in yesterday's two rain sessions"). Suspect labels are demoted to review as one
reversible AgentRun -- the auditor proposes, the review queue disposes; it never edits a label in place.

Reuses the existing measurement stack: the Path C VLM verifier, the agent critic, the control-sample
precision, and the AgentRun/AuditDecision spine. Scheduled once per off-hours window by controller.tick,
using the AgentRun table itself as the "already ran today" marker (no new schema).
"""

from __future__ import annotations

import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.storage import get_object_store
from db.models import Frame, Object
from db.session import get_sessionmaker
from services.agent.runtime.budget import TokenBudget

log = get_logger("agent.overnight_auditor")

_KIND = "overnight_auditor"


async def _sample_auto_accepts(db: AsyncSession, since: datetime, limit: int):
    """A bounded random sample of the window's auto-accepted objects, with their frame for context."""
    rows = (await db.execute(
        select(Object, Frame).join(Frame, Frame.frame_id == Object.frame_id)
        .where(Object.state == "auto_accept", Object.created_at >= since)
        .order_by(func.random()).limit(limit))).all()
    return rows


async def _frame_objects(db: AsyncSession, frame_id) -> list[Object]:
    return list((await db.execute(select(Object).where(Object.frame_id == frame_id))).scalars().all())


def _scene_tag(frame: Frame) -> str:
    sc = frame.scene or {}
    return sc.get("weather") or sc.get("time_of_day") or "unknown"


async def run_audit(run_id: uuid.UUID, *, sample_size: int = 200, vlm_calls: int = 60,
                    since_hours: int = 24) -> None:
    """Background worker: audit the window's auto-accepts, queue suspects to review, write the report onto
    the AgentRun and an AuditDecision. Idempotent per run_id."""
    from services.agent.critic import critique_frame
    from services.agent.frame_agent import _build_context
    from services.autolabel.ontology import get_ontology
    from services.recall.backends import load_image_bgr

    onto = get_ontology()
    store = get_object_store()
    since = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    budget = TokenBudget(vlm_calls)

    # VLM verifier (independent-model spot check). Absent/disabled Ollama -> critic + control only.
    verifier = None
    try:
        from core.config import get_settings
        from services.autolabel.grounding import supported_concept_ids
        from services.autolabel.paths.path_c_qwen3vl import VlmVerifier, make_vlm_client

        settings = get_settings()
        if settings.models.vlm.enabled:
            verifier = VlmVerifier(make_vlm_client(settings), onto, settings,
                                   supported_ids=await supported_concept_ids())
    except Exception as exc:  # noqa: BLE001 - the auditor still runs without the VLM
        log.warning("audit.vlm_unavailable", error=str(exc))

    maker = get_sessionmaker()
    async with maker() as db:
        sampled = await _sample_auto_accepts(db, since, sample_size)

    if not sampled:
        report = {"window_hours": since_hours, "sampled": 0, "note": "no auto-accepted labels in the window"}
        await _finish(run_id, "committed", report, {})
        return

    # group by frame so we load each image + build each critic context once
    by_frame: dict = {}
    for obj, frame in sampled:
        by_frame.setdefault(frame.frame_id, {"frame": frame, "objs": []})["objs"].append(obj)

    vlm_checked = 0
    confusion: Counter = Counter()          # (from_name -> to_name)
    confusion_scene: dict = {}              # (from,to) -> Counter(scene)
    critic_flags: Counter = Counter()
    suspects: dict = {}                     # object_id(str) -> {reason, from, to, scene}

    for fid, grp in by_frame.items():
        frame = grp["frame"]
        async with maker() as db:
            all_objs = await _frame_objects(db, fid)
            try:
                ctx = await _build_context(db, frame, all_objs)
                verdicts = critique_frame(ctx)
            except Exception:  # noqa: BLE001 - critic is a bonus signal; never fail the audit over it
                verdicts = {}
        img = None
        for obj in grp["objs"]:
            oid = str(obj.object_id)
            cur = onto.by_id(int(obj.class_id)).name
            reason = None
            # 1) critic consistency flag
            v = verdicts.get(oid)
            if v is not None and not v.ok:
                for r in v.reasons:
                    critic_flags[r.split(":")[0]] += 1
                reason = "critic:" + ",".join(sorted({r.split(":")[0] for r in v.reasons}))
            # 2) VLM independent-model spot check (budgeted)
            if verifier is not None and not budget.exhausted:
                try:
                    if img is None:
                        img = load_image_bgr(store, frame.img_uri)
                    res = verifier.verify_object(img, tuple(float(x) for x in obj.bbox), int(obj.class_id))
                    budget.charge(max(1, res.votes))
                    vlm_checked += 1
                    if res.class_name and res.confident and res.class_name != cur:
                        confusion[(cur, res.class_name)] += 1
                        confusion_scene.setdefault((cur, res.class_name), Counter())[_scene_tag(frame)] += 1
                        reason = (reason + "; " if reason else "") + f"vlm:{cur}->{res.class_name}"
                except Exception as exc:  # noqa: BLE001
                    log.warning("audit.vlm_call_failed", object_id=oid, error=str(exc))
            if reason:
                suspects[oid] = {"reason": reason, "from": cur, "scene": _scene_tag(frame)}

    # queue suspects to review as one reversible run (never touch a human-owned object)
    changes = await _queue_suspects(run_id, list(suspects.keys()))

    async with maker() as db:
        precision = await _control_precision(db)

    # named confusion movers + scene correlation
    movers = []
    for (frm, to), n in confusion.most_common(6):
        scenes = confusion_scene.get((frm, to), Counter())
        top_scene, top_n = (scenes.most_common(1)[0] if scenes else ("unknown", 0))
        movers.append({"from": frm, "to": to, "n": n,
                       "concentrated_in": top_scene if top_n >= max(2, n // 2) else None})

    among_sample = None if vlm_checked == 0 else round(1.0 - len(confusion) / vlm_checked, 4)
    report = {
        "window_hours": since_hours,
        "sampled": len(sampled),
        "vlm_checked": vlm_checked,
        "vlm_disagreements": int(sum(confusion.values())),
        "among_sample_agreement": among_sample,
        "control_precision": precision,
        "confusion_movers": movers,
        "critic_flags": dict(critic_flags),
        "suspects_queued": len(changes),
        "budget": budget.as_dict(),
        "notes": _narrate(movers, precision, len(changes)),
    }
    await _finish(run_id, "committed", report, changes)
    log.info("audit.done", run_id=str(run_id), sampled=len(sampled), suspects=len(changes),
             precision=precision.get("precision"))


def _narrate(movers: list[dict], precision: dict, queued: int) -> list[str]:
    notes = []
    p = precision.get("precision")
    if p is not None:
        notes.append(f"auto-accept precision holding at {round(p * 100, 1)} over {precision.get('reviewed')} reviewed controls")
    elif precision.get("pending"):
        notes.append(f"auto-accept precision not yet measurable ({precision.get('pending')} controls awaiting review)")
    for m in movers[:3]:
        loc = f" concentrated in {m['concentrated_in']} sessions" if m.get("concentrated_in") else ""
        notes.append(f"{m['from']} -> {m['to']} confusion up ({m['n']}){loc}")
    notes.append(f"{queued} suspect labels queued for review")
    return notes


async def _queue_suspects(run_id: uuid.UUID, object_ids: list[str]) -> dict:
    """Demote each still-machine suspect to review, stamped for exact revert."""
    maker = get_sessionmaker()
    changes: dict = {}
    async with maker() as db:
        for oid in object_ids:
            obj = await db.get(Object, uuid.UUID(oid))
            if obj is None or obj.source == "human" or obj.state != "auto_accept":
                continue
            changes[oid] = {"from_state": obj.state, "from_source": obj.source}
            obj.state = "review"
            obj.version = (obj.version or 0) + 1
            prov = dict(obj.provenance or {})
            prov["agent_run_id"] = str(run_id)
            prov.setdefault("audit_flag", True)
            obj.provenance = prov
        await db.commit()
    return changes


async def _control_precision(db: AsyncSession) -> dict:
    try:
        from services.govern.control_sample import measured_precision

        return await measured_precision(db)
    except Exception as exc:  # noqa: BLE001
        return {"precision": None, "error": str(exc)}


async def _finish(run_id: uuid.UUID, status: str, report: dict, changes: dict) -> None:
    from services.agent.runtime.report import finish_run

    await finish_run(run_id, status=status, report=report, changes=changes, decision="audit_summary")


async def launch_audit(db: AsyncSession, *, created_by: str = _KIND, **kw) -> dict:
    """Create the AgentRun and fire the background audit. Returns the run id immediately."""
    from services.agent.runtime.report import launch

    async def _worker(run_id: uuid.UUID) -> None:
        await run_audit(run_id, **kw)

    return await launch(db, _KIND, _worker, created_by=created_by, policy=kw)


async def maybe_run_nightly(db: AsyncSession) -> dict:
    """Off-hours hook for the runtime scheduler: run once per calendar day (AgentRun as the marker)."""
    from services.agent.runtime.report import ran_since

    day_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    if await ran_since(db, _KIND, day_start):
        return {"ran": False, "reason": "already ran today"}
    res = await launch_audit(db, created_by="scheduler")
    return {"ran": True, **res}


async def latest_report(db: AsyncSession) -> dict | None:
    from services.agent.runtime.report import latest_run

    return await latest_run(db, _KIND)
