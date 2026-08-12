"""Relabel agent: a reasoning layer that re-examines existing labels and improves the class where an
independent model is confident the current one is wrong.

The reasoning is a margin, not a coin flip. For each object it reads the whole SigLIP 2 class distribution
over the crop, finds where the CURRENT class sits in it, and only proposes a relabel when a different class
both clears an absolute-confidence floor AND beats the current class by a clear margin. Every change records
the original class so the run reverts exactly. Runs on a single frame (from the editor) or across the whole
corpus in the background ('relabel all frames').

**Everything routes to review by default.** This module used to keep a decisive disagreement without a human
seeing it, on the reasoning that the accuracy improves. It does not. Measured against the 302 objects a
person had verified in this corpus, all 10 changes it would have applied unreviewed overruled that person,
including traffic_sign -> milestone at 0.985 and minivan -> mpv at 0.945. The confidence is a softmax over
how well a crop matches each class *name*, so a high number says the name fits, not that the label is wrong,
and it is highest exactly where two names are close. Auto-keep is therefore opt-in and off, and the agent is
treated as what it measurably is: good at finding candidates, not at deciding them.
"""

from __future__ import annotations

import uuid

from sqlalchemy import Text, distinct, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.gpu_slot import gpu_slot
from core.logging import get_logger
from core.storage import FETCHABLE_URI_SQL, get_object_store
from db.models import AgentRun, Frame, Object, OntologyClass
from services.agent.relabel_reasoning import adjudicate, corpus_pairs
from services.training.gpu_lease import training_holds_gpu

log = get_logger("agent.relabel")


# A crop this small carries too few pixels for a zero-shot classifier to say anything worth acting on, and
# a distant object is exactly where its confidence is least earned.
MIN_CROP_PX = 24

# Crossing a superclass is a different claim from refining within one, and the corpus says so. A judge
# calibrated against human review rulings put label precision at 0.958 by superclass with only 2 of 285
# cross-superclass errors: the existing labels get the superclass right almost always, so a model proposing
# to cross one is usually the thing that is wrong. Measured on 302 human-accepted objects, the old thresholds
# would have changed 73 of them, 50 of those across a superclass, including motorcycle -> pedestrian at 0.744
# and motorcycle -> street_vendor at 0.747, both auto-kept without review.
CROSS_L1_CONF = 0.90
CROSS_L1_MARGIN = 0.55

# How many frames may fail back to back before a corpus run gives up. One unreadable frame is a data problem
# and should be skipped; a run of them is the object store or the GPU being gone, and carrying on would mark
# thousands of frames done having read none of them.
MAX_CONSECUTIVE_FRAME_FAILURES = 20


def _decide(crop_bgr, current_id: int, *, min_conf: float, margin: float, strong_conf: float,
            strong_margin: float, cross_l1_conf: float = CROSS_L1_CONF,
            cross_l1_margin: float = CROSS_L1_MARGIN, auto_keep: bool = False):
    """Return (suggested_id, suggested_name, top_conf, action) or None. action: relabel_keep | relabel_review.

    Two bars, not one. Within a superclass (`truck` to `tempo`, both heavy) the model is refining a
    distinction the taxonomy makes and the ordinary thresholds apply. Across one (`motorcycle` to
    `pedestrian`) it is contradicting the part of the label most likely to be right, so the bar is much
    higher and the result always goes to a human rather than being kept.
    """
    from services.autolabel.classify_crop import classify_crop
    from services.autolabel.ontology import get_ontology

    h, w = crop_bgr.shape[:2]
    if h < MIN_CROP_PX or w < MIN_CROP_PX:
        return None

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

    conf = float(top["conf"])
    gap = conf - cur_conf
    # Returned alongside the decision so the second stage can reason about the same numbers rather than
    # recomputing them from a crop it would have to re-encode.

    onto = get_ontology()
    try:
        same_l1 = onto.by_id(int(current_id)).l1 == onto.by_id(int(top["class_id"])).l1
    except Exception:  # noqa: BLE001 - an unresolvable class is treated as the riskier case
        same_l1 = False

    if not same_l1:
        if conf < cross_l1_conf or gap < cross_l1_margin:
            return None
        # Never auto-kept. A model that wants to move an object between superclasses may be right, and it is
        # not right often enough to change the corpus without somebody looking.
        return int(top["class_id"]), top["class_name"], round(conf, 3), "relabel_review", round(gap, 3), False

    if conf < min_conf or gap < margin:
        return None
    decisive = conf >= strong_conf and gap >= strong_margin
    action = "relabel_keep" if (decisive and auto_keep) else "relabel_review"
    return int(top["class_id"]), top["class_name"], round(conf, 3), action, round(gap, 3), True


async def plan_relabel(db: AsyncSession, frame_id: uuid.UUID, *, min_conf: float = 0.45, margin: float = 0.15,
                       strong_conf: float = 0.60, strong_margin: float = 0.30,
                       auto_keep: bool = False, reason: bool = False, judge=None) -> dict:
    """Dry-run: which objects the reasoning layer would relabel, and how. No writes.

    `reason` turns on the second-stage reasoning in relabel_reasoning.py, and it is off by default for the
    same reason `auto_keep` is. It only ever removes proposals, and nothing has yet measured that the ones it
    removes are the wrong ones. On the current corpus that claim cannot be tested at all: the classifier
    proposes 3 changes in 3,227 objects and 0 on the 332 human-verified labels, because the corpus relabel
    pass already converged. Defaulting a filter to on before it has been shown to filter correctly is the
    mistake this module already made once with auto_keep.

    `auto_keep` is off by default, so every proposal lands in review. Measured against the 302 objects a
    human verified in this corpus, all 10 changes the agent would have applied without review overruled a
    label a person had checked, among them traffic_sign -> milestone at 0.985 and minivan -> mpv at 0.945.
    Confidence here reflects how well a crop matches a class *name*, not whether the label is wrong, so a
    high number is not permission to skip the human. The caller can opt back in.
    """
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
    suppressed: list[dict] = []
    counts = {"total": len(objs), "relabel_keep": 0, "relabel_review": 0, "rejected_by_reasoning": 0}
    # Read once per frame, not per object: a scan of the whole review trail, whose answer changes on the
    # timescale of a review session.
    pairs = await corpus_pairs(db) if reason else {}
    for o in objs:
        x1, y1, x2, y2 = (int(round(float(v))) for v in o.bbox)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        crop = img[y1:y2, x1:x2]
        d = _decide(crop, int(o.class_id), min_conf=min_conf, margin=margin,
                    strong_conf=strong_conf, strong_margin=strong_margin, auto_keep=auto_keep)
        if d is None:
            continue
        sug_id, sug_name, conf, action, gap, same_l1 = d
        try:
            cur = onto.by_id(int(o.class_id)).name
        except Exception:  # noqa: BLE001
            cur = str(o.class_id)

        item = {"object_id": str(o.object_id), "from_name": cur, "to_class": sug_id, "to_name": sug_name,
                "conf": conf, "action": action}

        if reason:
            verdict, why = await adjudicate(
                crop, cur, sug_name, pairs, conf=conf, margin=gap, cross_l1=not same_l1,
                cross_conf=CROSS_L1_CONF, cross_margin=CROSS_L1_MARGIN, judge=judge)
            item.update(why)
            if not why["kept"]:
                # Dropped before it reaches a queue somebody has to work, but kept visible. A second stage
                # that silently removes proposals is indistinguishable from a first stage that got quieter,
                # and the dry run is exactly where somebody is trying to decide whether to trust it. These
                # never enter `items`, so nothing downstream sees them as proposals.
                counts["rejected_by_reasoning"] += 1
                suppressed.append(item)
                continue

        counts[action] += 1
        items.append(item)
    return {"frame_id": str(frame_id), "counts": counts, "items": items, "suppressed": suppressed}


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
                          session_id: str | None = None, min_conf: float = 0.45, margin: float = 0.15,
                          auto_keep: bool = False) -> None:
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
        #
        # Frames whose img_uri names no object key are excluded here rather than failed later. Fifteen of
        # them carry `s3://x.jpg`, which parses to a bucket and an empty key, and they are unfetchable
        # permanently. Left in, they burn a slot in every batch and, worse, a run of them trips the
        # consecutive-failure guard: the corpus pass stopped at batch 30 with 3,707 frames left, having
        # decided from twenty dead fixtures in a row that something systemic was wrong.
        q = (select(distinct(Object.frame_id))
             .join(Frame, Frame.frame_id == Object.frame_id)
             .where(Object.source != "human", Object.frame_id.cast(Text).notin_(seen),
                    Frame.img_uri.op("~")(FETCHABLE_URI_SQL)))
        if session_id:
            q = q.where(Frame.session_id == uuid.UUID(session_id))
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
        consecutive_failures = 0
        # One job on the card at a time. This loop is hours of batched inference, and a training job or an
        # autolabel pass starting beside it does not fail cleanly: it runs the card out of memory part way
        # through a batch, which arrives here as a failed frame and, twenty in a row, as a stopped corpus
        # pass with nothing actually wrong with it. Waiting is right for a batch job; it has nowhere to be.
        async with gpu_slot(f"relabel_all:{run_id}", timeout_s=None):
            for fid in frame_ids:
                if str(fid) in done:
                    continue
                try:
                    async with maker() as db:
                        res = await commit_relabel(db, fid, created_by=created_by or "relabel-all",
                                                   min_conf=min_conf, margin=margin, auto_keep=auto_keep)
                except Exception as exc:  # noqa: BLE001
                    # One frame must not end a corpus-wide job. A frame whose image is missing from the object
                    # store took out a whole 1,000-frame batch at frame 209, losing the rest of the work for a
                    # single unreadable row. It is marked done so a resume does not stop on it again.
                    consecutive_failures += 1
                    totals["skipped_error"] = totals.get("skipped_error", 0) + 1
                    done.add(str(fid))
                    log.warning("agent.relabel_all.frame_failed", run_id=str(run_id), frame_id=str(fid),
                                error=str(exc)[:200])
                    # Marking it done in *this* run is not enough. The cross-run cursor is the set of committed
                    # child `relabel` runs, and a frame that failed never gets one, so the next run selects it
                    # again and fails on it again forever. Eleven frames whose images are absent from storage sat
                    # at the head of the queue doing exactly that: successive batches read 0 frames and reported
                    # success, and the corpus pass could never reach zero remaining.
                    #
                    # The child run says the frame was read and could not be used, which is true and is what the
                    # cursor actually means. It carries the error so the skip is auditable rather than silent,
                    # and a frame whose image comes back can be requeued by deleting these rows.
                    async with maker() as db:
                        db.add(AgentRun(run_id=uuid.uuid4(), kind="relabel",
                                        scope={"frame_id": str(fid)}, status="skipped",
                                        policy={"reason": "unreadable"}, counts={"skipped_error": 1},
                                        changes={}, critic={"error": str(exc)[:400]},
                                        created_by=created_by or "relabel-all"))
                        await db.commit()
                    if consecutive_failures >= MAX_CONSECUTIVE_FRAME_FAILURES:
                        # Tolerating a bad frame is right; tolerating an outage is not. A run of failures means
                        # the object store or the GPU has gone, and continuing would quietly mark thousands of
                        # frames done having read none of them.
                        raise RuntimeError(
                            f"{consecutive_failures} frames in a row failed; stopping rather than marking the "
                            f"rest done unread. Last error: {exc}") from exc
                    async with maker() as db:
                        await beat(db, run_id, progress={"done": sorted(done), "total": len(frame_ids)},
                                   counts=dict(totals))
                    continue
                consecutive_failures = 0
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
