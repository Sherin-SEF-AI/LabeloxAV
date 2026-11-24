"""Which objects an annotator may change.

The workflow is two steps: an annotator's work goes to a reviewer before it counts. The server enforced
the second half (services/review_policy.py turns an annotator's "accept" into a submission) and got the
first half wrong by accident: every route under /api/objects sat behind a reviewer floor, so an annotator
could draw a box and then could not move it, relabel it or delete it. A box drawn in the wrong place
stayed wrong until a reviewer found the time, and the clamp in review_policy, written for exactly these
requests, could never run because they never arrived.

The floor is lifted for the routes that edit one object (services/api/main.py), and this is the rule that
replaces it. An annotator may change an object when it is still theirs to change:

  * nobody else has ruled on it. A reviewer's verdict stands until a reviewer changes it, and one
    annotator does not rework another's corrections;
  * it is not accepted or settled. Those are the states that mean "this counts", and only a reviewer or
    the settlement engine writes them;
  * if a person drew it, that person is the one asking. A machine-made label nobody has reviewed belongs
    to whoever picks it up, because correcting auto-labels is most of what an annotator does.

Reviewers and admins are not restricted, and neither is a deployment with authentication off, which has
no roles to apply.
"""

from __future__ import annotations

from collections.abc import Iterable

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Object, Review
from services.api.deps import role_rank

REVIEWER_RANK = role_rank("reviewer")
# States that mean the label counts. An annotator's own approval is `submitted` (review_policy clamps it),
# so an object in one of these was put there by a reviewer or by the settlement engine.
FINAL_STATES = frozenset({"accepted", "settled"})

NOT_YOURS = "this label was drawn by someone else; a reviewer can change it"
REVIEWED = "this label has been reviewed by someone else; a reviewer can change it"
FINAL = "this label is already accepted; a reviewer can change it"


def is_restricted(user) -> bool:
    """Whether the per-object rule applies to this caller at all."""
    return user is not None and role_rank(getattr(user, "role", None)) < REVIEWER_RANK


def refusal(obj: Object, user, ruled_by_others: bool) -> str | None:
    """Why this annotator may not change this object, or None when they may. Pure, so it is testable
    without a database: `ruled_by_others` is whether anyone else has a review on the object."""
    if not is_restricted(user):
        return None
    if obj.state in FINAL_STATES:
        return FINAL
    if obj.annotator_id is not None and obj.annotator_id != user.user_id:
        return NOT_YOURS
    if ruled_by_others:
        return REVIEWED
    return None


async def ruled_by_others(db: AsyncSession, object_ids: Iterable, user) -> set:
    """The objects, of those given, that somebody other than `user` has a review on.

    A review row without a user id is from before reviews recorded one (or from a machine path that
    writes through the review trail under its own name); it is somebody else's unless the name on it is
    this user's.
    """
    ids = list(object_ids)
    if not ids:
        return set()
    rows = (await db.execute(
        select(Review.object_id, Review.user_id, Review.reviewer).where(Review.object_id.in_(ids)))).all()
    return {oid for oid, uid, name in rows
            if (uid is not None and uid != user.user_id) or (uid is None and name != user.name)}


async def assert_may_change(db: AsyncSession, objects: Iterable[Object], user) -> None:
    """Raise 403 unless this caller may change every one of these objects.

    All or nothing: a bulk action that silently applied to some of a selection and skipped the rest would
    leave the person believing they had done something they had not.
    """
    if not is_restricted(user):
        return
    objs = list(objects)
    others = await ruled_by_others(db, [o.object_id for o in objs], user)
    refused = [(o, why) for o in objs if (why := refusal(o, user, o.object_id in others)) is not None]
    if not refused:
        return
    if len(objs) == 1:
        raise HTTPException(status_code=403, detail=refused[0][1])
    raise HTTPException(status_code=403, detail=(
        f"{len(refused)} of these {len(objs)} labels are not yours to change (drawn or reviewed by someone "
        "else, or already accepted); a reviewer can change them. Nothing was changed."))
