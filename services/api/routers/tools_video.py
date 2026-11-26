"""API for the video annotation tools, agent A: video and editor (#1 SAM 2 video tracking, #2 text prompt, #4 correction fan-out).

Mounted under /api; every path here starts with /tools/video/ so the area's routes stay together. Set the role
each route needs with `Depends(require_role(...))` (annotator / reviewer / admin) or `Depends(require_user)`.

- `POST /tools/video/track` queues a SAM 2 video tracking job (kind `sam2_video`) from a click, a box or an
  existing label, and re-prompts a track this tool made with a corrective click on another frame.
- `POST /tools/video/frames/{id}/describe/accept` writes the proposals a person ticked from the editor's Describe tool,
  as `source='described'` with the phrase on the provenance.
- `POST /tools/video/describe/session` queues the phrase over a whole session (kind `describe_session`).
- `POST /tools/video/fanout/preview` and `/apply`: after a class correction, the rest of the track, the
  embedding look-alikes and the same model confusion in the session, applied as one undoable batch.
- `POST /tools/video/fanout/{run_id}/undo` takes such a batch back (the person who made it, or a reviewer).
"""

from __future__ import annotations

import uuid
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.timebase import now_ns
from db.models import Frame, Object, Review, Track
from services.api.deps import current_user, db_session, require_role
from services.autolabel.ontology import get_ontology

log = get_logger("tools_video")

router = APIRouter()

# The most objects one fan-out may touch. A correction is one decision; one decision rewriting more than
# this is more likely a mis-linked track or an over-broad similarity threshold than a real systematic error.
MAX_FANOUT = 600
# Machine states a fan-out may move. A person's accepted or rejected verdict is not re-ruled by a fan-out.
OPEN_STATES = ("review", "annotate", "submitted", "auto_accept", "settled")


def _job_dict(job) -> dict:
    return {"job_id": str(job.job_id), "kind": job.kind, "status": job.status,
            "session_id": str(job.session_id) if job.session_id else None}


# ---- #1 SAM 2 video tracking -------------------------------------------------------------------------------

class PromptIn(BaseModel):
    frame_id: UUID
    points: list[list[float]] = Field(default_factory=list)   # [[x, y], ...] image pixels
    labels: list[int] = Field(default_factory=list)            # 1 positive, 0 negative, per point
    box: list[float] | None = None                             # [x1, y1, x2, y2]


class TrackIn(BaseModel):
    frame_id: UUID                       # the frame the person is looking at; the window centres on it
    object_id: UUID | None = None        # track an existing label (its box is the prompt on its frame)
    track_id: UUID | None = None         # re-prompt a track this tool made
    class_name: str | None = None        # for a fresh click with no label behind it
    prompts: list[PromptIn] = Field(default_factory=list)
    before: int = Field(10, ge=0, le=120)
    after: int = Field(10, ge=0, le=120)


@router.post("/tools/video/track", dependencies=[Depends(require_role("annotator"))])
async def track_mask(body: TrackIn, db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Queue SAM 2 video mask tracking for one object. Returns the job; follow it on /api/tools/jobs/{id}."""
    from services.tools.registry import enqueue

    frame = await db.get(Frame, body.frame_id)
    if frame is None:
        raise HTTPException(404, "frame not found")
    for p in body.prompts:
        if p.box is not None and len(p.box) != 4:
            raise HTTPException(400, "box must be [x1,y1,x2,y2]")
        if any(len(pt) != 2 for pt in p.points):
            raise HTTPException(400, "each point must be [x, y]")
        if p.labels and len(p.labels) != len(p.points):
            raise HTTPException(400, "labels must have one entry per point")
    if body.object_id is not None:
        obj = await db.get(Object, body.object_id)
        if obj is None:
            raise HTTPException(404, "object not found")
    elif body.track_id is not None:
        if await db.get(Track, body.track_id) is None:
            raise HTTPException(404, "track not found")
        if not body.prompts:
            raise HTTPException(400, "a re-prompt needs a click or box on the frame to correct")
    else:
        if not body.class_name or not get_ontology().has_name(body.class_name):
            raise HTTPException(400, f"unknown class '{body.class_name}'")
        if not any(p.points or p.box for p in body.prompts):
            raise HTTPException(400, "click on the object (or drag a box) to start tracking")
    job = await enqueue(db, "sam2_video", body.model_dump(mode="json"), session_id=frame.session_id,
                        user_id=getattr(user, "user_id", None))
    return _job_dict(job)


# ---- #2 text prompt to labels ------------------------------------------------------------------------------

class ProposalIn(BaseModel):
    bbox: list[float]
    polygons: list[list[float]] = Field(default_factory=list)
    conf: float | None = None


class DescribeAcceptIn(BaseModel):
    phrase: str
    class_name: str
    proposals: list[ProposalIn] = Field(min_length=1, max_length=50)


@router.post("/tools/video/frames/{frame_id}/describe/accept", dependencies=[Depends(require_role("annotator"))])
async def describe_accept(frame_id: UUID, body: DescribeAcceptIn, db: AsyncSession = Depends(db_session),
                          user=Depends(current_user), x_lbx_tab: Annotated[str | None, Header()] = None):
    """Write the described proposals a person ticked, as `source='described'` labels of the chosen class.

    The state is the one a person creating a label would get (role-clamped exactly as the object create is),
    because a person looked at each of these before ticking it; the source still says the geometry came from
    the phrase, and the phrase is on the provenance.
    """
    from core.storage import get_object_store
    from services.api.routers.objects import _write_mask
    from services.autolabel.describe import normalize_phrase
    from services.collab.presence import assert_editor_may_save
    from services.review_policy import ReviewStateError, state_for

    frame = await db.get(Frame, frame_id)
    if frame is None:
        raise HTTPException(404, "frame not found")
    await assert_editor_may_save([frame.frame_id], user, x_lbx_tab)
    onto = get_ontology()
    if not onto.has_name(body.class_name):
        raise HTTPException(400, f"unknown class '{body.class_name}'")
    phrase = normalize_phrase(body.phrase)
    if not phrase:
        raise HTTPException(400, "the phrase the proposals came from is required")
    for p in body.proposals:
        if len(p.bbox) != 4:
            raise HTTPException(400, "bbox must be [x1,y1,x2,y2]")
    try:
        state = state_for(None, "accepted", getattr(user, "role", None), None) or "submitted"
    except ReviewStateError as exc:  # pragma: no cover - a fixed, valid request
        raise HTTPException(400, str(exc)) from exc
    cid = onto.by_name(body.class_name).id
    store = get_object_store()
    made: list[str] = []
    for p in body.proposals:
        oid = uuid.uuid4()
        mask_uri = (_write_mask(store, frame.session_id, frame.frame_id, oid, p.polygons, frame.width, frame.height)
                    if p.polygons else None)
        obj = Object(object_id=oid, frame_id=frame.frame_id, class_id=cid, bbox=[float(v) for v in p.bbox],
                     conf=float(p.conf) if p.conf is not None else 1.0, source="described", state=state,
                     mask_uri=mask_uri, mask_encoding="polygon" if mask_uri else None,
                     attrs=onto.derive_attrs({}, cid),
                     annotator_id=getattr(user, "user_id", None),
                     provenance={"described_as": phrase, "method": "describe_frame",
                                 "model_conf": p.conf, "created_by": "describe-tool"})
        db.add(obj)
        db.add(Review(object_id=oid, reviewer=getattr(user, "name", None) or "anon",
                      user_id=getattr(user, "user_id", None), action="create", before=None,
                      after={"class_id": cid, "bbox": obj.bbox, "state": state, "described_as": phrase},
                      time_spent_ms=0, ts_ns=now_ns()))
        made.append(str(oid))
    await db.commit()
    return {"created": made, "state": state, "class_name": body.class_name, "phrase": phrase}


class DescribeSessionIn(BaseModel):
    session_id: UUID
    phrase: str
    class_name: str
    min_conf: float | None = Field(None, ge=0.01, le=0.99)
    cam_id: str | None = None
    max_frames: int | None = Field(None, ge=1, le=5000)
    dry_run: bool = False


@router.post("/tools/video/describe/session", dependencies=[Depends(require_role("reviewer"))])
async def describe_session(body: DescribeSessionIn, db: AsyncSession = Depends(db_session),
                           user=Depends(current_user)):
    """Queue one phrase over every frame of a session; finds are written for review (`state='review'`).

    Reviewer: it spends the GPU on a whole session and writes labels on every frame of it. An auto-label
    kind, so it waits out office hours.
    """
    from services.autolabel.describe import normalize_phrase
    from services.tools.registry import enqueue

    if not normalize_phrase(body.phrase):
        raise HTTPException(400, "a phrase is required; describe what you are looking for in a few words")
    if not get_ontology().has_name(body.class_name):
        raise HTTPException(400, f"unknown class '{body.class_name}'")
    params = {k: v for k, v in body.model_dump(mode="json").items() if v is not None}
    job = await enqueue(db, "describe_session", params, session_id=body.session_id,
                        user_id=getattr(user, "user_id", None))
    return _job_dict(job)


# ---- #4 correction fan-out ---------------------------------------------------------------------------------

class FanoutPreviewIn(BaseModel):
    object_id: UUID              # the object the person just corrected
    from_class: str              # what it was
    to_class: str                # what they made it
    threshold: float = Field(0.85, ge=0.5, le=0.99)
    limit: int = Field(60, ge=1, le=200)


def _row(o: Object, onto, *, score: float | None = None) -> dict:
    try:
        cname = onto.by_id(o.class_id).name
    except KeyError:
        cname = f"class {o.class_id}"
    return {"object_id": str(o.object_id), "frame_id": str(o.frame_id), "class_name": cname,
            "state": o.state, "source": o.source, "conf": o.conf, "version": o.version,
            "track_id": str(o.track_id) if o.track_id else None, "score": score,
            "crop_url": f"/api/objects/{o.object_id}/crop", "human": o.source == "human"}


def confusion_match(prov: dict | None, to_class: str) -> bool:
    """Whether a machine label's own record shows the model weighing the class the person corrected it to.

    The fused label keeps each path's vote (`proposals[].class_name`) and the reasoner's suggestion
    (`reasoning.suggested_class`). A label that is the old class while one of its own voters said the new
    class is the same confusion the person just ruled on, not merely the same class.
    """
    prov = prov or {}
    votes = {str(p.get("class_name")) for p in (prov.get("proposals") or []) if isinstance(p, dict)}
    suggested = (prov.get("reasoning") or {}).get("suggested_class") if isinstance(prov.get("reasoning"), dict) else None
    return to_class in votes or suggested == to_class


async def fanout_candidates(db: AsyncSession, body: FanoutPreviewIn) -> dict:
    """The three groups a correction can fan out to, each object in at most one group."""
    from services.intelligence.corrections import correction_candidates

    onto = get_ontology()
    for n in (body.from_class, body.to_class):
        if not onto.has_name(n):
            raise HTTPException(400, f"unknown class '{n}'")
    from_id, to_id = onto.by_name(body.from_class).id, onto.by_name(body.to_class).id
    origin = await db.get(Object, body.object_id)
    if origin is None:
        raise HTTPException(404, "object not found")
    frame = await db.get(Frame, origin.frame_id)
    if frame is None:
        raise HTTPException(404, "the object's frame is gone")
    seen = {origin.object_id}

    # (a) the rest of the track
    track_rows: list[dict] = []
    if origin.track_id is not None:
        rows = (await db.execute(select(Object).where(
            Object.track_id == origin.track_id, Object.object_id != origin.object_id))).scalars().all()
        for o in rows:
            seen.add(o.object_id)
            r = _row(o, onto)
            r["already"] = o.class_id == to_id
            r["selectable"] = not r["human"] and not r["already"]
            track_rows.append(r)

    # (b) embedding look-alikes, from the correction tool's own search
    look = await correction_candidates(db, str(body.object_id), kind="class", old_class_id=from_id,
                                       new_value=body.to_class, limit=body.limit, threshold=body.threshold)
    look_rows: list[dict] = []
    for c in look.get("candidates", []):
        oid = UUID(c["object_id"])
        if oid in seen:
            continue
        seen.add(oid)
        c = dict(c)
        c["selectable"] = not c.get("human") and not c.get("already")
        look_rows.append(c)

    # (c) the same model confusion in this session: machine labels still the old class whose own votes
    # included the new one
    conf_rows: list[dict] = []
    q = (select(Object).join(Frame, Frame.frame_id == Object.frame_id)
         .where(Frame.session_id == frame.session_id, Object.class_id == from_id,
                Object.source != "human", Object.state.in_(OPEN_STATES))
         .order_by(Object.conf.desc()).limit(2000))
    for o in (await db.execute(q)).scalars().all():
        if o.object_id in seen or not confusion_match(o.provenance, body.to_class):
            continue
        seen.add(o.object_id)
        r = _row(o, onto)
        r["already"] = False
        r["selectable"] = True
        conf_rows.append(r)
        if len(conf_rows) >= body.limit:
            break

    return {"object_id": str(body.object_id), "from_class": body.from_class, "to_class": body.to_class,
            "track": {"track_id": str(origin.track_id) if origin.track_id else None, "objects": track_rows},
            "lookalikes": {"objects": look_rows, "reason": look.get("reason"), "threshold": body.threshold},
            "confusion": {"objects": conf_rows,
                          "rule": f"machine labels in this session still '{body.from_class}' whose own model "
                                  f"votes included '{body.to_class}'"}}


@router.post("/tools/video/fanout/preview", dependencies=[Depends(require_role("annotator"))])
async def fanout_preview(body: FanoutPreviewIn, db: AsyncSession = Depends(db_session)):
    """What one class correction could also fix: the track, the look-alikes and the same confusion."""
    return await fanout_candidates(db, body)


class FanoutApplyIn(BaseModel):
    object_id: UUID
    to_class: str
    track_object_ids: list[UUID] = Field(default_factory=list)
    lookalike_object_ids: list[UUID] = Field(default_factory=list)
    confusion_object_ids: list[UUID] = Field(default_factory=list)
    expected_versions: dict[str, int] | None = None
    time_spent_ms: int = 0


@router.post("/tools/video/fanout/apply", dependencies=[Depends(require_role("annotator"))])
async def fanout_apply(body: FanoutApplyIn, db: AsyncSession = Depends(db_session), user=Depends(current_user),
                       x_lbx_tab: Annotated[str | None, Header()] = None):
    """Apply one class correction to the ticked objects of every group, as ONE undoable batch.

    The track group goes through exactly what `POST /tracks/{id}/relabel` does (`reclassify_track`, source
    `propagated`, human frames skipped, the ontology guard on, the track's own class moved). The look-alike
    and confusion groups go through what the correction dialog's bulk review does (`reclassify`, source
    `human`: a person looked at each crop and ticked it). Both land in one `record_batch` run, so a single
    undo takes the whole fan-out back, the track's class included.
    """
    from services.collab.presence import assert_editor_may_save
    from services.object_access import assert_may_change
    from services.review_apply import AttrRejected, apply_review_batch
    from services.review_batch import record_batch
    from services.review_policy import ReviewStateError

    onto = get_ontology()
    if not onto.has_name(body.to_class):
        raise HTTPException(400, f"unknown class '{body.to_class}'")
    cid = onto.by_name(body.to_class).id
    origin = await db.get(Object, body.object_id)
    if origin is None:
        raise HTTPException(404, "object not found")
    groups = {"track": body.track_object_ids, "lookalike": body.lookalike_object_ids,
              "confusion": body.confusion_object_ids}
    all_ids = {i for ids in groups.values() for i in ids} - {origin.object_id}
    if not all_ids:
        raise HTTPException(400, "nothing was ticked")
    if len(all_ids) > MAX_FANOUT:
        raise HTTPException(409, f"{len(all_ids)} objects is above the {MAX_FANOUT} limit for one fan-out")
    found = {o.object_id: o for o in (await db.execute(
        select(Object).where(Object.object_id.in_(all_ids)))).scalars()}
    track_objs = [found[i] for i in dict.fromkeys(body.track_object_ids) if i in found
                  and found[i].track_id is not None and found[i].track_id == origin.track_id]
    track_ids = {o.object_id for o in track_objs}
    other_objs = [found[i] for i in dict.fromkeys([*body.lookalike_object_ids, *body.confusion_object_ids])
                  if i in found and i not in track_ids and i != origin.object_id]
    missing = sorted(str(i) for i in all_ids if i not in found)

    await assert_may_change(db, [*track_objs, *other_objs], user)
    await assert_editor_may_save(list({o.frame_id for o in [*track_objs, *other_objs]}), user, x_lbx_tab)

    reviewer = getattr(user, "name", None) or "anon"
    uid = getattr(user, "user_id", None)
    role = getattr(user, "role", None)
    prov = {"fanout_from": str(origin.object_id)}
    changes: dict[str, dict] = {}
    out: dict = {"track": 0, "others": 0}
    try:
        if track_objs:
            res_t = await apply_review_batch(
                db, track_objs, action="reclassify_track", onto=onto, class_id=cid, role=role,
                source="propagated", reviewer=reviewer, uid=uid, expected_versions=body.expected_versions,
                time_spent_ms=body.time_spent_ms // 2, provenance_extra={**prov, "track_relabel": True},
                skip_human=True, guard_class_move=True, revalidate_attrs=True)
            changes.update(res_t.changes)
            out.update(track=res_t.n, skipped_human=res_t.skipped_human, refused=res_t.refused,
                       state=res_t.new_state, clamped=res_t.clamped)
            out.setdefault("skipped_stale", []).extend(res_t.stale)
        if other_objs:
            res_o = await apply_review_batch(
                db, other_objs, action="reclassify", onto=onto, class_id=cid, role=role, source="human",
                reviewer=reviewer, uid=uid, expected_versions=body.expected_versions,
                time_spent_ms=body.time_spent_ms - body.time_spent_ms // 2, provenance_extra=prov,
                revalidate_attrs=True)
            changes.update(res_o.changes)
            out["others"] = res_o.n
            out.setdefault("skipped_stale", []).extend(res_o.stale)
    except AttrRejected as exc:
        raise HTTPException(400, {"attr_errors": exc.errors, "object_id": exc.object_id}) from exc
    except ReviewStateError as exc:
        raise HTTPException(400, str(exc)) from exc

    policy: dict = {"action": "fanout", "class_name": body.to_class, "origin_object_id": str(origin.object_id),
                    "groups": {k: len(v) for k, v in groups.items()}}
    if out["track"] and origin.track_id is not None:
        track = await db.get(Track, origin.track_id)
        if track is not None:
            policy.update(track_id=str(track.track_id), track_class_from=int(track.class_id), track_class_to=cid)
            track.class_id = cid
    run_id = await record_batch(db, changes, created_by=reviewer, commit=False, policy=policy) if changes else None
    await db.commit()
    return {**out, "updated": len(changes), "run_id": run_id, "skipped_missing": missing,
            "class_name": body.to_class}


@router.post("/tools/video/fanout/{run_id}/undo", dependencies=[Depends(require_role("annotator"))])
async def fanout_undo(run_id: UUID, db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Take a fan-out back. The person who applied it may; anybody else needs to be a reviewer."""
    from db.models import AgentRun
    from services.api.deps import role_rank
    from services.object_access import REVIEWER_RANK
    from services.review_batch import revert_batch

    run = await db.get(AgentRun, run_id)
    if run is None or (run.policy or {}).get("action") != "fanout":
        raise HTTPException(404, "fan-out not found")
    if run.status == "reverted":
        raise HTTPException(409, "this fan-out was already undone")
    if user is not None and role_rank(getattr(user, "role", None)) < REVIEWER_RANK \
            and run.created_by != getattr(user, "name", None):
        raise HTTPException(403, "only the person who applied this fan-out, or a reviewer, can undo it")
    return await revert_batch(db, run)

