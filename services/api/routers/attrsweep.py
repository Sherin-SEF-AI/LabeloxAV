"""Attribute sweep: what is missing, a page of crops to answer it on, and one write for the answer.

Three endpoints because the mode has three questions. `coverage` answers "what work exists", `queue`
answers "show me the next screenful", and `apply` lands the answer. The write is not a new code path: it
delegates to `services/review_apply.py::apply_review_batch` with `action="set_attrs"`, which is the same
validated, derived, version-bumping, revertible path bulk review and the correction modal already use.

The one thing this layer adds is the track expansion. An attribute that describes the object rather than
the moment gets answered once and written to every member of the track, because a truck's load does not
change between frames and fifty separate answers about it are forty-nine opportunities to disagree.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from services.api.deps import current_user, db_session
from services.autolabel.ontology import get_ontology
from services.labelops.attr_sweep import DEFAULT_LIMIT, coverage, expand_targets, sweep_queue
from services.review_apply import AttrRejected, apply_review_batch
from services.review_batch import record_batch
from services.review_policy import ReviewStateError

log = get_logger("api_attrsweep")
router = APIRouter()

# A track-wide answer can touch every frame of a long track, and a page of sixty of them is a very large
# write for one keystroke. The cap is per request, so an annotator sweeping a busy session still gets
# through it; it exists so that one keystroke cannot rewrite a quarter of the corpus by accident.
MAX_SWEEP_OBJECTS = 5000


@router.get("/attrsweep/coverage", dependencies=[Depends(current_user)])
async def attr_coverage(session_id: UUID | None = None, db: AsyncSession = Depends(db_session)):
    """Per attribute: objects in scope, objects carrying it, objects missing it, and the worst classes.

    One table scan for all of them. Roughly a second over 578,436 objects, so it is a page load and not a
    background job, but it is not something to poll.
    """
    return await coverage(db, get_ontology(), session_id=session_id)


@router.get("/attrsweep/queue", dependencies=[Depends(current_user)])
async def attr_queue(attr: str, class_name: str | None = None, session_id: UUID | None = None,
                     limit: int = DEFAULT_LIMIT, unit: str = "auto",
                     db: AsyncSession = Depends(db_session)):
    """A page of crops missing one attribute, largest box first, with what each answer will cover."""
    try:
        return await sweep_queue(db, get_ontology(), attr=attr, class_name=class_name,
                                 session_id=session_id, limit=limit, unit=unit)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


class SweepApplyIn(BaseModel):
    attr: str
    # Typed by the ontology, not here: an enum takes a string, occupant_count an int, helmet a list of
    # bools. validate_attrs is what refuses a wrong one, with the reason.
    value: object
    unit: str = "object"          # "track" writes to every live member of each id given
    ids: list[str]
    # Off by default, so a sweep fills holes and never replaces an answer somebody gave deliberately on
    # one frame. A caller correcting a wrong track-wide answer sets it and is doing so on purpose.
    overwrite: bool = False
    state: str | None = None
    time_spent_ms: int = 0
    reviewer: str = "anon"


@router.post("/attrsweep/apply")
async def attr_apply(payload: SweepApplyIn, db: AsyncSession = Depends(db_session),
                     user=Depends(current_user)):
    """Write one attribute value across the objects or tracks given, as one revertible run."""
    onto = get_ontology()
    if payload.attr not in onto.attributes:
        raise HTTPException(400, f"unknown attribute '{payload.attr}'")
    if onto.attributes[payload.attr].derived_from:
        raise HTTPException(400, f"'{payload.attr}' is derived from "
                                 f"'{onto.attributes[payload.attr].derived_from}' and is computed, not written")
    if payload.unit not in ("track", "object"):
        raise HTTPException(400, "unit must be 'track' or 'object'")
    if not payload.ids:
        raise HTTPException(400, "no ids given")

    try:
        targets = await expand_targets(db, attr=payload.attr, unit=payload.unit, ids=payload.ids,
                                       overwrite=payload.overwrite)
    except ValueError as exc:
        raise HTTPException(400, f"malformed id: {exc}") from exc

    if not targets:
        # Not an error, and not a silent success either. Re-sweeping a track whose members already carry
        # the attribute lands here, and saying so is the difference between "done" and "did nothing".
        return {"updated": 0, "objects": 0, "run_id": None,
                "reason": "every target already carries this attribute" if not payload.overwrite
                          else "no live objects matched those ids"}
    if len(targets) > MAX_SWEEP_OBJECTS:
        raise HTTPException(409, {"objects": len(targets), "limit": MAX_SWEEP_OBJECTS,
                                  "hint": "narrow the selection; a page this large is more likely a "
                                          "mis-linked track than a real answer"})

    reviewer = user.name if user is not None else payload.reviewer
    uid = user.user_id if user is not None else None
    try:
        res = await apply_review_batch(
            db, targets, action="set_attrs", onto=onto, attrs={payload.attr: payload.value},
            requested_state=payload.state, role=getattr(user, "role", None),
            # The attribute is the person's answer, so the object's own source is left alone: a machine
            # box that a human has described is still a machine box, and rewriting source to "human" here
            # would set this repo's "an agent must not touch this" flag on every frame of the track.
            source=None,
            reviewer=reviewer, uid=uid, time_spent_ms=payload.time_spent_ms,
            provenance_extra={"attr_sweep": payload.attr},
            skip_human=False)
    except AttrRejected as exc:
        raise HTTPException(400, {"attr_errors": exc.errors, "object_id": exc.object_id}) from exc
    except ReviewStateError as exc:
        raise HTTPException(400, str(exc)) from exc

    run_id = await record_batch(
        db, res.changes, created_by=reviewer, commit=False,
        policy={"action": "attr_sweep", "attr": payload.attr, "value": payload.value,
                "unit": payload.unit, "ids": payload.ids[:200]}) if res.changes else None
    await db.commit()

    log.info("attrsweep.applied", attr=payload.attr, unit=payload.unit,
             ids=len(payload.ids), objects=res.n)
    return {"updated": res.n, "objects": len(targets), "attr": payload.attr, "value": payload.value,
            "unit": payload.unit, "run_id": run_id, "state": res.new_state, "clamped": res.clamped,
            "attrs_dropped": res.attrs_dropped}
