"""The frame agent: decide, for every machine-produced object on a frame, whether the system is sure
enough to auto-accept it or should hand it to a person. It reuses what the pipeline already computed
(calibrated confidence, cross-path agreement, single-frame quality flags), adds the self-consistency
critic, and applies the accept/route policy.

Two entry points:
- plan_frame: pure read, writes nothing, returns exactly what a commit WOULD do (the dry-run).
- commit_frame: applies the plan, recording every state transition in one reversible AgentRun.

It only ever touches machine objects (source != "human"); a person's work is never altered. Auto-accept is
the sole autonomous write; everything else routes to review/annotate.
"""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import AgentRun, Frame, Object, ObjectDynamics
from services.agent.critic import CriticContext, critique_frame
from services.agent.policy import Decision, PolicyThresholds, decide

log = get_logger("agent.frame")


async def _load_objects(db: AsyncSession, frame_id: uuid.UUID) -> list[Object]:
    # Only machine objects are in scope; human-owned labels are never touched by the agent.
    rows = await db.execute(select(Object).where(Object.frame_id == frame_id, Object.source != "human"))
    return list(rows.scalars().all())


async def _build_context(db: AsyncSession, frame: Frame, objs: list[Object]) -> CriticContext:
    from services.autolabel.ontology import get_ontology

    obj_ids = [o.object_id for o in objs]
    # dynamics (motion consistency), if a session dynamics pass has run
    dynamics: dict[str, dict] = {}
    if obj_ids:
        drows = await db.execute(select(ObjectDynamics).where(ObjectDynamics.object_id.in_(obj_ids)))
        for d in drows.scalars().all():
            dynamics[str(d.object_id)] = {"speed_kmh": getattr(d, "speed_kmh", None),
                                          "risk_level": getattr(d, "risk_level", None)}
    # track history (temporal consistency): every object sharing a track, with its frame timestamp
    track_ids = [o.track_id for o in objs if o.track_id is not None]
    track_history: dict[str, list] = {}
    if track_ids:
        trows = await db.execute(
            select(Object.track_id, Frame.ts_ns, Object.class_id, Object.bbox)
            .join(Frame, Frame.frame_id == Object.frame_id)
            .where(Object.track_id.in_(track_ids))
        )
        for tid, ts_ns, class_id, bbox in trows.all():
            cx = (float(bbox[0]) + float(bbox[2])) / 2.0
            cy = (float(bbox[1]) + float(bbox[3])) / 2.0
            track_history.setdefault(str(tid), []).append((int(ts_ns), int(class_id), cx, cy))
    # LiDAR cloud (cross-modal), only when this frame actually has one
    cloud_xyz = None
    if getattr(frame, "lidar", None):
        try:
            from db.models import PointCloud
            from services.lidar.ingest.store import load_cloud
            pc = (await db.execute(
                select(PointCloud).where(PointCloud.session_id == frame.session_id)
                .order_by(func.abs(PointCloud.ts_ns - frame.ts_ns)).limit(1)
            )).scalar_one_or_none()
            if pc is not None:
                cloud_xyz = load_cloud(pc.cloud_uri).xyz
        except Exception:  # noqa: BLE001 -- cross-modal is a bonus check; never fail the run over it
            cloud_xyz = None
    return CriticContext(
        onto=get_ontology(), cam_id=frame.cam_id, width=frame.width, height=frame.height,
        frame_objects=objs, dynamics=dynamics, track_history=track_history, cloud_xyz=cloud_xyz,
    )


def _obj_decision(obj: Object, critic_ok: bool, th: PolicyThresholds) -> Decision:
    prov = obj.provenance or {}
    agreement = bool(prov.get("agreement"))
    quality_ok = not (prov.get("quality_flags") or [])
    return decide(float(obj.conf), agreement, quality_ok, critic_ok, th)


async def plan_frame(db: AsyncSession, frame_id: uuid.UUID, th: PolicyThresholds | None = None) -> dict:
    """Dry-run: what the agent would do to this frame's machine objects. Writes nothing."""
    th = th or PolicyThresholds()
    frame = await db.get(Frame, frame_id)
    if frame is None:
        raise ValueError("frame not found")
    objs = await _load_objects(db, frame_id)
    ctx = await _build_context(db, frame, objs)
    verdicts = critique_frame(ctx)

    from services.autolabel.ontology import get_ontology
    onto = get_ontology()

    items: list[dict] = []
    counts = {"total": len(objs), "auto_accept": 0, "review": 0, "annotate": 0,
              "unchanged": 0, "demoted_by_critic": 0}
    check_tally: dict[str, int] = {}
    for obj in objs:
        v = verdicts[str(obj.object_id)]
        dec = _obj_decision(obj, v.ok, th)
        counts[dec.action] = counts.get(dec.action, 0) + 1
        if not v.ok:
            counts["demoted_by_critic"] += 1
        for chk, status in v.checks.items():
            if status == "flag":
                check_tally[chk] = check_tally.get(chk, 0) + 1
        if dec.action == obj.state:
            counts["unchanged"] += 1
        try:
            cname = onto.by_id(int(obj.class_id)).name
        except Exception:  # noqa: BLE001
            cname = str(obj.class_id)
        items.append({
            "object_id": str(obj.object_id), "class_name": cname, "conf": round(float(obj.conf), 3),
            "current_state": obj.state, "action": dec.action, "changes_state": dec.action != obj.state,
            "reason": dec.reason, "tier": dec.tier, "critic_ok": v.ok, "critic_reasons": v.reasons,
        })
    return {"frame_id": str(frame_id), "policy": th.to_dict(), "counts": counts,
            "critic_flags": check_tally, "items": items}


async def commit_frame(db: AsyncSession, frame_id: uuid.UUID, th: PolicyThresholds | None = None,
                       created_by: str | None = None) -> dict:
    """Apply the plan and record it as one reversible AgentRun."""
    th = th or PolicyThresholds()
    plan = await plan_frame(db, frame_id, th)

    run_id = uuid.uuid4()
    changes: dict[str, dict] = {}
    # index objects for mutation
    objs = {str(o.object_id): o for o in await _load_objects(db, frame_id)}
    for item in plan["items"]:
        if not item["changes_state"]:
            continue
        obj = objs.get(item["object_id"])
        if obj is None:
            continue
        from_state, from_source = obj.state, obj.source
        to_state = item["action"]
        to_source = "auto_accept" if to_state == "auto_accept" else from_source
        obj.state = to_state
        obj.source = to_source
        obj.version = (obj.version or 0) + 1
        prov = dict(obj.provenance or {})
        prov["agent_run_id"] = str(run_id)
        if item["critic_reasons"]:
            prov.setdefault("agent_critic", []).extend(item["critic_reasons"])
        obj.provenance = prov
        changes[str(obj.object_id)] = {"from_state": from_state, "to_state": to_state,
                                       "from_source": from_source, "to_source": to_source}

    run = AgentRun(
        run_id=run_id, kind="frame", scope={"frame_id": str(frame_id)}, status="committed",
        policy=th.to_dict(), counts=plan["counts"], changes=changes, critic=plan["critic_flags"],
        created_by=created_by,
    )
    db.add(run)
    await db.commit()
    log.info("agent.frame.commit", frame_id=str(frame_id), run_id=str(run_id),
             changed=len(changes), auto_accept=plan["counts"].get("auto_accept", 0))
    return {"run_id": str(run_id), "frame_id": str(frame_id), "applied": len(changes),
            "counts": plan["counts"], "policy": th.to_dict()}
