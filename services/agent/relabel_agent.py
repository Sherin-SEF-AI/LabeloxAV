"""Relabel agent: a reasoning layer that re-examines existing labels and improves the class where an
independent model is confident the current one is wrong.

The reasoning is a margin, not a coin flip. For each object it reads the whole SigLIP 2 class distribution
over the crop, finds where the CURRENT class sits in it, and only proposes a relabel when a different class
both clears an absolute-confidence floor AND beats the current class by a clear margin. A strong, decisive
disagreement is applied and kept (the accuracy actually improves); a moderate one is applied but routed to
review for a human to confirm; a weak one is left alone. Every change records the original class so the run
reverts exactly. Runs on a single frame (from the editor) or across the whole corpus in the background
('relabel all frames').
"""

from __future__ import annotations

import uuid

from sqlalchemy import Text, distinct, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.storage import get_object_store
from db.models import AgentRun, Frame, Object, OntologyClass
from services.training.gpu_lease import training_holds_gpu

log = get_logger("agent.relabel")


def _decide(crop_bgr, current_id: int, *, min_conf: float, margin: float, strong_conf: float, strong_margin: float):
    """Return (suggested_id, suggested_name, top_conf, action) or None. action: relabel_keep | relabel_review."""
    from services.autolabel.classify_crop import classify_crop

    preds = classify_crop(crop_bgr, topk=20)
    if not preds:
        return None
    top = preds[0]
    cur_conf = next((float(p["conf"]) for p in preds if int(p["class_id"]) == int(current_id)), 0.0)
    if int(top["class_id"]) == int(current_id):
        return None
    # Never relabel a specific class down into a generic catch-all bucket: that loses information rather
    # than improving it (a 'sedan' must not become 'vehicle_fallback'). Upgrading a fallback is fine.
    if str(top["class_name"]).endswith("_fallback"):
        return None
    gap = float(top["conf"]) - cur_conf
    if float(top["conf"]) < min_conf or gap < margin:
        return None
    action = "relabel_keep" if (float(top["conf"]) >= strong_conf and gap >= strong_margin) else "relabel_review"
    return int(top["class_id"]), top["class_name"], round(float(top["conf"]), 3), action


async def plan_relabel(db: AsyncSession, frame_id: uuid.UUID, *, min_conf: float = 0.45, margin: float = 0.15,
                       strong_conf: float = 0.60, strong_margin: float = 0.30) -> dict:
    """Dry-run: which objects the reasoning layer would relabel, and how. No writes."""
    from services.autolabel.ontology import get_ontology
    from services.recall.backends import load_image_bgr

    frame = await db.get(Frame, frame_id)
    if frame is None:
        raise ValueError("frame not found")
    onto = get_ontology()
    objs = (await db.execute(select(Object).where(Object.frame_id == frame_id, Object.source != "human"))).scalars().all()
    if not objs:
        return {"frame_id": str(frame_id), "counts": {"total": 0, "relabel_keep": 0, "relabel_review": 0}, "items": []}
    img = load_image_bgr(get_object_store(), frame.img_uri)
    h, w = img.shape[:2]
    items = []
    counts = {"total": len(objs), "relabel_keep": 0, "relabel_review": 0}
    for o in objs:
        x1, y1, x2, y2 = (int(round(float(v))) for v in o.bbox)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        d = _decide(img[y1:y2, x1:x2], int(o.class_id), min_conf=min_conf, margin=margin,
                    strong_conf=strong_conf, strong_margin=strong_margin)
        if d is None:
            continue
        sug_id, sug_name, conf, action = d
        counts[action] += 1
        try:
            cur = onto.by_id(int(o.class_id)).name
        except Exception:  # noqa: BLE001
            cur = str(o.class_id)
        items.append({"object_id": str(o.object_id), "from_name": cur, "to_class": sug_id, "to_name": sug_name,
                      "conf": conf, "action": action})
    return {"frame_id": str(frame_id), "counts": counts, "items": items}


async def commit_relabel(db: AsyncSession, frame_id: uuid.UUID, *, created_by: str | None = None, **kw) -> dict:
    """Apply the reasoning-layer relabels on one frame as a reversible run."""
    plan = await plan_relabel(db, frame_id, **kw)
    run_id = uuid.uuid4()
    changes: dict[str, dict] = {}

    # The classifier scores every class the ontology object knows, and that object merges the governed YAML
    # with the custom-class sidecar. A sidecar class that was never seeded into `ontology_class` is therefore
    # proposable and unstorable at once, and writing it raises a foreign-key violation from inside a
    # background task: the corpus-wide run died at frame 13 of 25 because one crop scored highest as
    # `test_new_vehicle`, a leftover from a test.
    #
    # Skipping the proposal rather than letting it kill the run is the right trade. The frame is still read,
    # the other objects on it are still corrected, and the drift is counted instead of being a traceback in a
    # log nobody is reading.
    proposed = {int(i["to_class"]) for i in plan["items"]}
    storable: set[int] = set()
    if proposed:
        storable = set((await db.execute(
            select(OntologyClass.id).where(OntologyClass.id.in_(proposed)))).scalars().all())
    unstorable = proposed - storable
    if unstorable:
        log.warning("agent.relabel.unstorable_class", frame_id=str(frame_id), class_ids=sorted(unstorable))

    for item in plan["items"]:
        if int(item["to_class"]) not in storable:
            plan["counts"]["skipped_unknown_class"] = plan["counts"].get("skipped_unknown_class", 0) + 1
            continue
        obj = await db.get(Object, uuid.UUID(item["object_id"]))
        if obj is None or obj.source == "human" or int(obj.class_id) == item["to_class"]:
            continue
        changes[item["object_id"]] = {"from_class": int(obj.class_id), "from_state": obj.state, "from_source": obj.source}
        obj.class_id = item["to_class"]
        if item["action"] == "relabel_review":
            obj.state = "review"   # moderate confidence: a human confirms the relabel
        obj.version = (obj.version or 0) + 1
        prov = dict(obj.provenance or {})
        prov["agent_run_id"] = str(run_id)
        prov.setdefault("agent_relabel", []).append(f"{item['from_name']} -> {item['to_name']} ({item['conf']})")
        obj.provenance = prov
    db.add(AgentRun(run_id=run_id, kind="relabel", scope={"frame_id": str(frame_id)}, status="committed",
                    policy=kw, counts=plan["counts"], changes=changes, critic={}, created_by=created_by))
    await db.commit()
    log.info("agent.relabel.commit", frame_id=str(frame_id), run_id=str(run_id), relabeled=len(changes))
    return {"run_id": str(run_id), "frame_id": str(frame_id), "relabeled": len(changes), "counts": plan["counts"]}


async def run_relabel_all(run_id: uuid.UUID, *, max_frames: int = 200, created_by: str | None = None,
                          session_id: str | None = None, min_conf: float = 0.45, margin: float = 0.15) -> None:
    """Background: relabel every machine-labelled frame (bounded), one reversible child run per frame, the
    parent run aggregating counts. Yields to a running training job (GPU discipline)."""
    from db.session import get_sessionmaker
    from services.agent.resume import beat, done_set

    maker = get_sessionmaker()
    async with maker() as db:
        if await training_holds_gpu(db):
            run = await db.get(AgentRun, run_id)
            if run:
                run.status, run.counts = "committed", {"skipped": "training job holds the GPU"}
                await db.commit()
            return
        # Frames this job has never read before, oldest first.
        #
        # It used to be `distinct(frame_id) ... limit N` with no ordering and no exclusion, and Postgres
        # returns the same rows for that query every time. 34,121 frames are eligible, so the default
        # max_frames=200 covered 0.59% of them and covered the identical 0.59% on every run: pressing
        # "relabel all frames" a hundred times re-read the same 200 frames and never reached the other
        # 33,921. It looked like a job that found nothing rather than a job that could not see anything.
        #
        # A committed child run of kind `relabel` is the record that a frame was read, so excluding those
        # makes successive runs walk forward through the corpus instead of standing still.
        seen = (select(AgentRun.scope["frame_id"].astext)
                .where(AgentRun.kind == "relabel", AgentRun.scope["frame_id"].astext.isnot(None)))
        q = (select(distinct(Object.frame_id))
             .where(Object.source != "human", Object.frame_id.cast(Text).notin_(seen)))
        if session_id:
            q = q.join(Frame, Frame.frame_id == Object.frame_id).where(Frame.session_id == uuid.UUID(session_id))
        # Ordered so a run is reproducible and so an operator can tell two runs apart by what they covered.
        frame_ids = list((await db.execute(q.order_by(Object.frame_id).limit(max_frames))).scalars().all())
        if not frame_ids:
            run = await db.get(AgentRun, run_id)
            if run:
                run.status = "committed"
                run.counts = {"frames": 0, "detail": "every eligible frame has already been relabelled"}
                await db.commit()
            log.info("agent.relabel_all.nothing_left", run_id=str(run_id))
            return
        # Resume support: the unit of work is one frame, so the cursor is the frames already relabelled.
        # Re-running a frame would mint a second child run for it and double-count its totals, so skipping
        # is about correctness of the numbers, not only about the wasted GPU time.
        prior = await db.get(AgentRun, run_id)
        done = done_set(dict(prior.progress or {})) if prior is not None else set()
        totals = dict(prior.counts or {}) if prior is not None else {}
        child_runs: list[str] = list((prior.changes or {}).get("child_runs") or []) if prior is not None else []

    for k in ("frames", "relabel_keep", "relabel_review"):
        totals.setdefault(k, 0)
    if done:
        log.info("agent.relabel_all.resuming", run_id=str(run_id), already_done=len(done))
    try:
        for fid in frame_ids:
            if str(fid) in done:
                continue
            async with maker() as db:
                res = await commit_relabel(db, fid, created_by=created_by or "relabel-all",
                                           min_conf=min_conf, margin=margin)
            totals["frames"] += 1
            totals["relabel_keep"] += res["counts"].get("relabel_keep", 0)
            totals["relabel_review"] += res["counts"].get("relabel_review", 0)
            if res["relabeled"]:
                child_runs.append(res["run_id"])
            done.add(str(fid))
            async with maker() as db:
                run = await db.get(AgentRun, run_id)
                if run:
                    run.changes = {"child_runs": child_runs}
                    await db.commit()
                # Cursor, counts and heartbeat in one write, after the child runs are recorded: a frame in
                # the cursor whose child run was never saved would be skipped on resume and its relabels
                # would then be unrevertable, because revert walks child_runs.
                await beat(db, run_id, progress={"done": sorted(done), "total": len(frame_ids)},
                           counts=dict(totals))
        async with maker() as db:
            run = await db.get(AgentRun, run_id)
            if run:
                run.status, run.counts = "committed", dict(totals)
                run.changes = {"child_runs": child_runs}
                await db.commit()
        log.info("agent.relabel_all.done", run_id=str(run_id), **totals)
    except Exception as exc:  # noqa: BLE001
        log.error("agent.relabel_all.failed", run_id=str(run_id), error=str(exc))
        async with maker() as db:
            run = await db.get(AgentRun, run_id)
            if run:
                run.status, run.error = "error", str(exc)
                await db.commit()
