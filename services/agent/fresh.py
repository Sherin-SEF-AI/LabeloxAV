"""Fresh-inference mode: run detection on a single frame and decide in one shot. The frame agent normally
acts on objects a prior batch autolabel already persisted; this closes the gap for a frame that has never
been labelled -- it runs the same detect -> fuse -> quality-review -> gate -> persist pipeline the batch
runner uses (Path A + Path B; the optional Path C VLM is skipped here), then runs the agent (critic +
policy) on the fresh objects.

Reuses the batch primitives directly (StagedRunner, FusionEngine, review_object_quality, gate_object,
persist_frame_objects) so a single frame goes through exactly the same logic as a whole session. GPU-bound:
it yields to a running training job, same single-GPU discipline as interactive segmentation.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, TrainingJob
from services.agent.frame_agent import commit_frame, plan_frame
from services.agent.policy import PolicyThresholds

log = get_logger("agent.fresh")


async def label_frame(db: AsyncSession, frame_id: uuid.UUID, *, auto_accept_enabled: bool = True) -> dict:
    """Detect + fuse + quality-review + gate + persist one frame, reusing the batch pipeline. Returns
    detection/object counts and the by-state breakdown."""
    from core.bus import EventBus
    from core.config import get_settings
    from core.schemas import FrameMeta
    from core.storage import get_object_store
    from services.autolabel.fusion import FusionEngine
    from services.autolabel.gate import gate_object
    from services.autolabel.grounding import supported_concept_ids
    from services.autolabel.ontology import get_ontology
    from services.autolabel.persist import persist_frame_objects
    from services.autolabel.quality_reviewer import review_object_quality
    from services.autolabel.runner import StagedRunner, load_image, resolve_detector_weights

    settings = get_settings()
    store = get_object_store()
    store.ensure_bucket()
    onto = get_ontology()

    frame = await db.get(Frame, frame_id)
    if frame is None:
        raise ValueError("frame not found")
    fmeta = FrameMeta(frame_id=frame.frame_id, session_id=frame.session_id, ts_ns=frame.ts_ns,
                      cam_id=frame.cam_id, img_uri=frame.img_uri, width=frame.width, height=frame.height,
                      ego_speed=frame.ego_speed, quality=frame.quality)

    supported_ids = await supported_concept_ids()
    champion = await resolve_detector_weights(db)
    img = load_image(frame.img_uri)

    runner = StagedRunner(champion, supported_ids)
    runner.open_stage1()
    try:
        dets_a, dets_b = runner.run_stage1_frame(img)
    finally:
        runner.close_stage1()

    engine = FusionEngine(settings, onto)
    fused = engine.fuse_frame(frame.frame_id, dets_a, dets_b)
    objs = [g.obj for g in fused]
    for fo in fused:
        others = [o for o in objs if o is not fo.obj]
        qv = review_object_quality(fo.obj, others, onto, frame.width, frame.height, settings.quality)
        fo.obj.provenance.quality_flags = qv.reasons
        fo.obj.state = gate_object(fo.obj, onto, settings.gate,
                                   auto_accept_enabled=auto_accept_enabled, quality_ok=qv.ok)

    bus = EventBus()
    await bus.start()
    try:
        by_state = await persist_frame_objects(db, store, bus, fmeta, fused)
        await db.commit()
    finally:
        await bus.stop()

    log.info("agent.fresh.label", frame_id=str(frame_id), detections=len(dets_a) + len(dets_b),
             objects=len(fused))
    return {"detections": len(dets_a) + len(dets_b), "objects": len(fused), "by_state": by_state}


async def label_and_decide(db: AsyncSession, frame_id: uuid.UUID, *, commit: bool = False,
                           policy: PolicyThresholds | None = None, created_by: str | None = None) -> dict:
    """One shot: fresh-detect the frame, then run the agent on the new objects. commit=False plans (writes
    the detected objects but takes no agent action); commit=True also applies the agent's decision."""
    if (await db.execute(select(TrainingJob.job_id).where(TrainingJob.status == "running").limit(1))).first():
        raise RuntimeError("GPU reserved for a training job; fresh inference is paused")
    policy = policy or PolicyThresholds()
    label = await label_frame(db, frame_id)
    if commit:
        agent = await commit_frame(db, frame_id, policy, created_by=created_by)
        return {"labeled": label, "agent": agent}
    plan = await plan_frame(db, frame_id, policy)
    return {"labeled": label, "agent_plan": plan["counts"]}
