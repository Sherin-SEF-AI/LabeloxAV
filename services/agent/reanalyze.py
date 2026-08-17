"""Reanalyse: look again at one frame's redaction and at its labels, in a single action.

Two things go wrong on a frame long after it was written, and until now they were checked by two different
people at two different times, if at all. A face or a plate the redactor could not see at frame resolution
stays unblurred, and a box that is a duplicate, a speck, a piece of sky or a rectangle hanging off the edge
of the image stays in the corpus. Both are visible from the frame alone, both are cheap to find, and neither
had a way to be asked for.

The two halves are deliberately not symmetric, because their mistakes are not symmetric:

  - **Redaction is fixed on the spot.** A face nobody blurred is a live exposure under the DPDPA, and a blur
    that turns out to be unnecessary costs a few hundred pixels of a frame. Waiting for a human to confirm
    each one is the expensive direction, so this half applies.
  - **Labels are proposed, never applied.** Auto-applying a class change is what put 1,047 buses inside a
    bus shelter in this corpus. Findings become `ErrorCandidate` rows on the queue that already has a
    reviewer, bulk verdicts, and a per-detector precision measurement to keep it honest.

The consequence is worth stating because it is what makes the action safe to run autonomously over the whole
corpus: **reanalyse never changes a label.** Its only writes are more blur and rows in a review queue. There
is no object mutation to revert and nothing that can contaminate a training set.

It runs on CPU. The PII detectors are configured for CPU precisely so redaction never competes with the
autolabel GPU budget, and every label check here is geometry over rows, so a corpus sweep can run beside a
training job instead of waiting for one.
"""

from __future__ import annotations

import uuid

from sqlalchemy import delete, distinct, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.schemas import BBox, UnifiedObject
from db.models import AgentRun, ErrorCandidate, Frame, Object
from services.errordetect.score import as_suspicion

log = get_logger("agent.reanalyze")

# One kind for the whole action, with the specific rule in `detail`, exactly as `policy_violation` does.
# `ErrorCandidate.kind` is String(24), and a kind per rule would fragment the precision measurement into
# a dozen samples too small to measure.
FINDING_KIND = "reanalyze"

# The one source this never questions. A person looked at that box, which is more evidence than any check
# here can offer.
#
# Everything else is in scope, which is wider than the `cleanup_sweep` tuple this borrows its checks from.
# That sweep DELETES, so restricting it to the three sources it was written for is right. This only ever
# proposes, and 60,171 objects in this corpus carry `imported`: excluding them would leave whole sessions
# whose every object came from a competitor export permanently unexamined, which is the opposite of what a
# re-check is for. `policy.detect_policy_violations` already draws the line in exactly this place.
_HUMAN = "human"

# How far outside the image a box may reach before it is a finding rather than a rounding artifact. Boxes
# are stored in pixels and a detector legitimately puts an edge a fraction of a pixel past the boundary.
_BOUNDS_TOLERANCE_PX = 1.0

# Fraction of a box that must lie outside the image for the finding to be worth a reviewer's time. A truly
# truncated object at the frame edge is normal and is already recorded as a truncation attribute; a box that
# is mostly outside the image is a coordinate error.
_BOUNDS_MIN_OUTSIDE = 0.2

_SCORES = {"out_of_bounds": 0.7, "stuff": 0.8, "oversize": 0.75, "ego_hood": 0.7, "duplicate": 0.65,
           "critic_flag": 0.6, "quality": 0.55}

# How many findings one frame may put on the review queue, highest suspicion first.
#
# Measured on six live frames: 8, 10, 0, 27, 26 and 30 findings, and 30 rows per frame across the 33,547
# frames in scope would be a million candidates nobody can work through. The cap is reported rather than
# applied quietly, because a queue that silently drops two thirds of what it found reads as a clean frame.
_MAX_FINDINGS_PER_FRAME = 12

# When a rule objects to at least this share of a frame's objects, it is describing the pipeline rather than
# the objects, and its findings are counted instead of queued. See `_drop_systemic` for the measurement.
_SYSTEMIC_FRACTION = 0.8
# Below a handful of objects, "fires on 80% of them" is two boxes and says nothing.
_SYSTEMIC_MIN_OBJECTS = 5


def out_of_bounds(bbox: list[float], w: int, h: int) -> tuple[float, str] | None:
    """Score and reason for a box that leaves the image, or None if it is inside it.

    This is the one check here with no existing implementation. Five call sites in this codebase silently
    clamp a box to the frame (`reconcile`, `relabel_agent`, and three others), and `attribute_agent`
    computes the out-of-frame fraction only to store it as a truncation attribute. Nothing ever said that a
    box mostly outside its own image is wrong, so nothing ever surfaced one.
    """
    if not bbox or len(bbox) < 4 or w <= 0 or h <= 0:
        return None
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if area <= 0:
        return None
    over = max(0.0, -x1) + max(0.0, x2 - w) + max(0.0, -y1) + max(0.0, y2 - h)
    if over <= _BOUNDS_TOLERANCE_PX:
        return None
    # Clip and compare areas rather than summing the overhangs, so a box overhanging two edges is not
    # double-counted at the corner.
    ix1, iy1 = max(0.0, x1), max(0.0, y1)
    ix2, iy2 = min(float(w), x2), min(float(h), y2)
    inside = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    outside_frac = 1.0 - inside / area
    if outside_frac < _BOUNDS_MIN_OUTSIDE:
        return None
    return (_SCORES["out_of_bounds"],
            f"{outside_frac * 100:.0f}% of the box lies outside the {w}x{h} image")


def _as_unified(o: Object, onto) -> UnifiedObject | None:
    """An ORM object as the contract `review_object_quality` reads. None when the class is unknown."""
    try:
        name = onto.by_id(int(o.class_id)).name
    except Exception:  # noqa: BLE001
        return None
    try:
        bbox = BBox.from_list([float(v) for v in o.bbox])
    except Exception:  # noqa: BLE001
        return None
    return UnifiedObject(object_id=o.object_id, frame_id=o.frame_id, track_id=o.track_id,
                         class_id=int(o.class_id), class_name=name, bbox=bbox, attrs=dict(o.attrs or {}),
                         conf=float(o.conf or 0.0))


def _finding(object_id, rule: str, score: float, reason: str) -> dict:
    return {"object_id": str(object_id), "kind": FINDING_KIND, "score": as_suspicion(score),
            "proposed_label": None, "detail": {"rule": rule, "reason": reason}}


async def check_labels(db: AsyncSession, frame: Frame) -> tuple[list[dict], dict[str, int]]:
    """Every label finding on one frame, plus the rules that fired on the whole frame and were set aside.

    Composed rather than reimplemented: each of these was written for a different sweep that a person had to
    know to run. The point of this function is that one press runs all of them on the frame in front of you.
    """
    from core.config import get_settings
    from services.agent.cleanup_sweep import _dup_removals, _reason
    from services.agent.critic import CriticContext, critique_frame
    from services.autolabel.ego_mask import get_ego_mask
    from services.autolabel.ontology import get_ontology
    from services.autolabel.quality_reviewer import review_object_quality
    from services.errordetect.policy import check_object

    objs = list((await db.execute(select(Object).where(
        Object.frame_id == frame.frame_id, Object.source != _HUMAN))).scalars().all())
    if not objs:
        return [], {}

    onto = get_ontology()
    settings = get_settings()
    w, h = int(frame.width or 0), int(frame.height or 0)
    found: list[dict] = []

    # The critic needs the frame's dynamics, track history and point cloud, which the frame agent already
    # knows how to assemble. Reusing it keeps the checks identical to the ones the per-frame agent runs.
    try:
        from services.agent.frame_agent import _build_context

        ctx: CriticContext | None = await _build_context(db, frame, objs)
    except Exception as exc:  # noqa: BLE001
        # A missing point cloud or dynamics row must not cost the frame its other nine checks.
        log.warning("reanalyze.context_failed", frame_id=str(frame.frame_id), error=str(exc)[:200])
        ctx = None
    if ctx is not None:
        for object_id, verdict in critique_frame(ctx).items():
            for reason in verdict.reasons:
                found.append(_finding(object_id, "critic_flag", _SCORES["critic_flag"], reason))

    # The ego hood mask is per vehicle and per camera, and the vehicle is on the session rather than the
    # frame. Without it the hood check simply does not fire, which is the right degradation: a session with
    # no recorded vehicle has no calibrated hood to test against.
    ego = None
    try:
        from db.models import Session as DbSession

        vehicle_id = (await db.execute(
            select(DbSession.vehicle_id).where(DbSession.session_id == frame.session_id))).scalars().first()
        ego = get_ego_mask(vehicle_id, frame.cam_id) if vehicle_id else None
    except Exception:  # noqa: BLE001
        ego = None

    unified = {o.object_id: _as_unified(o, onto) for o in objs}
    for o in objs:
        for rule, score, reason in check_object(o, objs, onto, w, h):
            found.append(_finding(o.object_id, rule, score, reason))

        bounds = out_of_bounds([float(v) for v in o.bbox], w, h)
        if bounds is not None:
            found.append(_finding(o.object_id, "out_of_bounds", bounds[0], bounds[1]))

        cleanup = _reason(o, onto, w, h, ego, settings.quality.max_area_frac)
        if cleanup:
            found.append(_finding(o.object_id, cleanup, _SCORES.get(cleanup, 0.7),
                                  f"cleanup rule '{cleanup}' applies to this box"))

        u = unified.get(o.object_id)
        if u is not None:
            others = [v for k, v in unified.items() if v is not None and k != o.object_id]
            quality = review_object_quality(u, others, onto, w, h, settings.quality)
            for r in quality.reasons:
                found.append(_finding(o.object_id, r, _SCORES["quality"],
                                      f"quality review flagged '{r}'"))

    # Duplicates are judged among the boxes that survive the other rules, so a stuff box removed above is
    # not also reported as somebody else's duplicate.
    flagged = {uuid.UUID(f["object_id"]) for f in found
               if f["detail"]["rule"] in ("stuff", "oversize", "ego_hood")}
    survivors = [o for o in objs if o.object_id not in flagged]
    for oid in _dup_removals(survivors, onto):
        found.append(_finding(oid, "duplicate", _SCORES["duplicate"],
                              "another box on this frame covers the same object with higher confidence"))

    # Keep the strongest finding per (object, rule): several checks reach the same conclusion about a
    # duplicate box, and a reviewer should see one row for it rather than four.
    best: dict[tuple[str, str], dict] = {}
    for f in found:
        key = (f["object_id"], f["detail"]["rule"])
        if key not in best or f["score"] > best[key]["score"]:
            best[key] = f
    kept, systemic = _drop_systemic(list(best.values()), len(objs))
    return sorted(kept, key=lambda f: f["score"], reverse=True), systemic


def _drop_systemic(findings: list[dict], n_objects: int) -> tuple[list[dict], dict[str, int]]:
    """Split findings into ones a reviewer can act on and rules that fired on the whole frame.

    A rule that objects to every object on a frame is reporting one fact, and the queue is the wrong place to
    say it a hundred times. Both rules this caught turned out to be real, and neither is fixable box by box:

      - `attr_validity` fired on 39 of 39, 122 of 122 and 64 of 64 objects, because the VLM path asked the
        model for every attribute in the ontology on every crop and validated the reply without a class id.
        7,477 objects that are not traffic signals carried a `signal_state`. Fixed at the writer and
        migrated out of the stored labels, so this one no longer fires at all.
      - `critic_flag` still fires on nearly every object, and it is right to: 10,006 of 11,287 tracks in
        this corpus (89%) change class somewhere along their length. That is a real and large problem with
        tracking or classification, and one number saying so is worth more than 2,433 rows saying it once
        per object.

    Counting them rather than queuing them is the same lesson the reasoning layer already learned here: a
    check that fires more often on the objects that were fine is not a weak check, it is a harmful one. The
    count is what tells somebody where to go and look.
    """
    if n_objects < _SYSTEMIC_MIN_OBJECTS:
        return findings, {}
    per_rule: dict[str, int] = {}
    for f in findings:
        per_rule[f["detail"]["rule"]] = per_rule.get(f["detail"]["rule"], 0) + 1
    systemic = {rule: n for rule, n in per_rule.items() if n / n_objects >= _SYSTEMIC_FRACTION}
    if not systemic:
        return findings, {}
    return [f for f in findings if f["detail"]["rule"] not in systemic], systemic


async def _persist(db: AsyncSession, frame_id: uuid.UUID, findings: list[dict]) -> int:
    """Replace this frame's pending reanalyse candidates with the current ones.

    Scoped to the frame rather than to the kind, unlike `run_detection`, which clears every pending candidate
    of the kinds it runs. That is right for a corpus sweep and wrong here: pressing the button on one frame
    must not discard a hundred other frames' pending findings.
    """
    frame_objects = select(Object.object_id).where(Object.frame_id == frame_id)
    await db.execute(delete(ErrorCandidate).where(
        ErrorCandidate.kind == FINDING_KIND, ErrorCandidate.status == "pending",
        ErrorCandidate.object_id.in_(frame_objects)))
    for f in findings:
        db.add(ErrorCandidate(object_id=uuid.UUID(f["object_id"]), kind=f["kind"], score=f["score"],
                              proposed_label=f["proposed_label"], detail=f["detail"], status="pending"))
    return len(findings)


async def reanalyze_frame(db: AsyncSession, frame_id: uuid.UUID, *, apply: bool = True) -> dict:
    """Re-check one frame end to end. With `apply=False` nothing is written anywhere.

    The plan form exists because the redaction half cannot be undone: the unredacted original is
    deliberately never stored, so a blur is permanent and has to be inspectable before it is taken.
    """
    from core.storage import get_object_store
    from services.anonymize.anonymizer import get_anonymizer
    from services.anonymize.recheck import recheck_frame

    frame = await db.get(Frame, frame_id)
    if frame is None:
        raise ValueError(f"no such frame: {frame_id}")

    redaction = await recheck_frame(db, get_object_store(), get_anonymizer(), frame, apply=apply)
    all_findings, systemic = await check_labels(db, frame)
    findings = all_findings[:_MAX_FINDINGS_PER_FRAME]
    dropped = len(all_findings) - len(findings)
    persisted = await _persist(db, frame_id, findings) if apply else 0
    if apply:
        await db.commit()

    log.info("reanalyze.frame", frame_id=str(frame_id), applied=apply,
             faces=redaction["faces_added"], plates=redaction["plates_added"],
             findings=len(findings), dropped=dropped, systemic=systemic)
    return {"frame_id": str(frame_id), "applied": apply, "redaction": redaction,
            "findings": findings, "findings_dropped": dropped, "systemic": systemic,
            "persisted": persisted}


async def run_reanalyze_all(run_id: uuid.UUID, *, session_id: str | None = None, max_frames: int = 500,
                            created_by: str | None = None) -> None:
    """Background: reanalyse every frame in scope, one at a time, reporting progress as it goes.

    Shaped like `run_relabel_all`, for the reasons that module records in its own comments: a per-frame
    try/except so one unreadable image cannot end a corpus pass, a consecutive-failure breaker so a dead
    object store stops the run instead of burning through it, and a heartbeat carrying `done`/`total` after
    every frame, which is what gives the console a progress bar rather than a spinner.

    Unlike the relabel sweep there is no GPU slot: every check here is CPU-bound geometry or a CPU detector,
    so this can run beside a training job rather than queueing behind one.
    """
    from db.session import get_sessionmaker
    from services.agent.resume import beat, done_set

    maker = get_sessionmaker()
    async with maker() as db:
        # Frames that carry a machine object worth checking. A frame with no objects has nothing for the
        # label half and no annotations to crop for the redaction half, so it is not in scope for either.
        q = (select(distinct(Object.frame_id))
             .join(Frame, Frame.frame_id == Object.frame_id)
             .where(Object.source != _HUMAN))
        if session_id:
            q = q.where(Frame.session_id == uuid.UUID(session_id))
        frame_ids = list((await db.execute(q.order_by(Object.frame_id).limit(max_frames))).scalars().all())

        prior = await db.get(AgentRun, run_id)
        done = done_set(dict(prior.progress or {})) if prior is not None else set()
        totals = dict(prior.counts or {}) if prior is not None else {}

    for k in ("frames", "faces_added", "plates_added", "findings", "findings_dropped", "skipped_error"):
        totals.setdefault(k, 0)
    # Rules that objected to a whole frame, counted across the sweep. One number per rule is what says "go
    # and fix the attribute writer" rather than "here are 5,825 boxes to look at". Resumed from `critic`
    # rather than `counts`, since that is where the breakdown is written.
    async with maker() as db:
        _prior = await db.get(AgentRun, run_id)
        systemic_totals: dict[str, int] = dict((_prior.critic or {}) if _prior is not None else {})
    if not frame_ids:
        async with maker() as db:
            run = await db.get(AgentRun, run_id)
            if run:
                run.status = "committed"
                run.counts = {**totals, "detail": "no frames in scope carry a machine-written object"}
                await db.commit()
        return
    if done:
        log.info("reanalyze.resuming", run_id=str(run_id), already_done=len(done))

    consecutive_failures = 0
    try:
        for fid in frame_ids:
            if str(fid) in done:
                continue
            try:
                async with maker() as db:
                    res = await reanalyze_frame(db, fid, apply=True)
                totals["frames"] += 1
                totals["faces_added"] += res["redaction"]["faces_added"]
                totals["plates_added"] += res["redaction"]["plates_added"]
                totals["findings"] += res["persisted"]
                totals["findings_dropped"] += res["findings_dropped"]
                for rule, n in res["systemic"].items():
                    systemic_totals[rule] = systemic_totals.get(rule, 0) + n
                # Flat in `counts`, because the console renders counts as "key value" pairs and a nested
                # dict there arrives as "[object Object]". The per-rule breakdown belongs in `critic`, which
                # is the column already defined as the run's findings summary by check.
                totals["systemic"] = sum(systemic_totals.values())
                consecutive_failures = 0
            except Exception as exc:  # noqa: BLE001
                # One frame must not end a corpus pass. It is marked done so a resume does not stop on the
                # same unreadable image forever.
                consecutive_failures += 1
                totals["skipped_error"] += 1
                log.warning("reanalyze.frame_failed", run_id=str(run_id), frame_id=str(fid),
                            error=str(exc)[:200])
            done.add(str(fid))
            async with maker() as db:
                await beat(db, run_id, progress={"done": sorted(done), "total": len(frame_ids)},
                           counts=dict(totals))
                run = await db.get(AgentRun, run_id)
                if run is not None and systemic_totals:
                    run.critic = dict(systemic_totals)
                    await db.commit()
            if consecutive_failures >= 20:
                # Twenty in a row is the object store being down or full, not twenty unlucky frames.
                log.error("reanalyze.aborting", run_id=str(run_id), **totals)
                break
    except Exception as exc:  # noqa: BLE001
        async with maker() as db:
            run = await db.get(AgentRun, run_id)
            if run:
                run.status, run.counts, run.error = "error", dict(totals), str(exc)[:500]
                await db.commit()
        raise

    async with maker() as db:
        run = await db.get(AgentRun, run_id)
        if run:
            run.status = "committed"
            run.counts = dict(totals)
            await db.commit()
    log.info("reanalyze.done", run_id=str(run_id), **totals)
