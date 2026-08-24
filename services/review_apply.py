"""Applying one review decision to a set of objects, in one place.

`services/review_policy.py` opens by saying the state rule "lives here, the router calls it, and both the
single and the bulk review path go through it". That was true of two paths and not of the third:
`POST /tracks/{id}/relabel` wrote `o.state = payload.state` straight from the request body with a default of
`accepted`, so an annotator could confirm an entire track and skip the QA step the whole two-stage workflow
is built on. It also never advanced `version`, never recorded a revertible batch, and never revalidated
attributes, so it was the one bulk write in the repo with no undo and no lock.

Rather than copy four rules into a third router, the per-object loop bulk review had already grown lives
here and every caller gets the same guarantees. Bulk review's observable behaviour is unchanged by the
move, which `tests/test_bulk_review_hardening.py` is the proof of.

Two behaviours are opt-in because the two callers genuinely differ, not because one of them is lazy:

`skip_human` - a track relabel must not overwrite a frame somebody else already ruled on, the same rule
`services/agent/temporal_repair.py` applies and the correction dialog shows as an "already" badge. Bulk
review is an explicit list of ids a person just ticked, so there is nothing to protect them from.

`guard_class_move` - the ontology guard from `services/agent/class_move.py` refuses a move that changes what
kind of thing something is. A track relabel is one decision fanned across ninety frames and is worth
guarding. Bulk review deliberately spans classes: the correction dialog exists to gather one systematic
error that is usually spread over several source classes, so guarding it there would break the feature
commit 38b28dd built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.timebase import now_ns
from db.models import Object, Review
from services.agent.class_move import refuse_reason
from services.review_batch import change_record
from services.review_policy import state_for, was_clamped


class AttrRejected(Exception):
    """Incoming attributes that do not validate against an object's effective class.

    Raised rather than dropped because the caller supplied them deliberately: silently discarding what
    somebody just typed is worse than telling them it does not apply.
    """

    def __init__(self, object_id: str, errors: list[str]) -> None:
        super().__init__(f"attr errors on {object_id}")
        self.object_id = object_id
        self.errors = errors


@dataclass
class ApplyResult:
    """What the batch did, including everything it declined to do and why."""

    n: int = 0
    new_state: str | None = None
    clamped: bool = False
    #: objects whose version moved under the caller, skipped rather than clobbered
    stale: list[dict] = field(default_factory=list)
    #: objects a person had already ruled on, left alone
    skipped_human: list[str] = field(default_factory=list)
    #: objects whose class move the ontology refuses, with the reason
    refused: list[dict] = field(default_factory=list)
    #: attributes dropped because they do not apply to the new class, by object id
    attrs_dropped: dict[str, list[str]] = field(default_factory=dict)
    #: the undo record, keyed by object id
    changes: dict[str, dict] = field(default_factory=dict)


async def apply_review_batch(
    db,
    objects: list[Object],
    *,
    action: str,
    onto: Any,
    class_id: int | None = None,
    attrs: dict | None = None,
    requested_state: str | None = None,
    role: str | None = None,
    source: str = "human",
    reviewer: str = "anon",
    uid: Any = None,
    expected_versions: dict[str, int] | None = None,
    time_spent_ms: int = 0,
    provenance_extra: dict | None = None,
    skip_human: bool = False,
    guard_class_move: bool = False,
    revalidate_attrs: bool = False,
) -> ApplyResult:
    """Apply one decision to every object given, returning what happened to each.

    Adds rows to the session and does not commit: the caller lands the edits and the batch stamp in one
    transaction, for the reason `services/review_batch.py` records at length.
    """
    res = ApplyResult()
    res.new_state = state_for(action, requested_state, role, None)
    res.clamped = was_clamped(action, requested_state, role, None)

    expected = expected_versions or {}
    # The batch's time is divided across its members rather than attributed to any one of them, so a grid
    # that triages sixty crops in a minute reports sixty seconds of work and not sixty minutes or zero.
    per_item_ms = int(time_spent_ms / len(objects)) if objects and time_spent_ms > 0 else 0

    for obj in objects:
        oid = str(obj.object_id)

        # Optimistic lock, per object. A stale member is skipped and named rather than failing the whole
        # batch, because one contended object should not discard fifty-nine good verdicts.
        want = expected.get(oid)
        if want is not None and obj.version != want:
            res.stale.append({"object_id": oid, "expected": want, "current": obj.version})
            continue

        # Never overwrite a decision a person already made. `source == "human"` is this repo's universal
        # marker for that, which is why the propagated rows below are not given it.
        if skip_human and obj.source == "human":
            res.skipped_human.append(oid)
            continue

        if guard_class_move and class_id is not None:
            reason = refuse_reason(onto, obj.class_id, class_id)
            if reason is not None:
                res.refused.append({"object_id": oid, "reason": reason})
                continue

        before = {"class_id": obj.class_id, "bbox": list(obj.bbox), "attrs": dict(obj.attrs or {}),
                  "state": obj.state, "source": obj.source, "conf": obj.conf,
                  "provenance": dict(obj.provenance or {})}
        # What the undo needs, captured before the edit. A Review row per object is the audit trail and
        # answers what changed; it is not an undo, because taking back a fifty-object batch through it means
        # fifty manual reversals with the operator remembering each prior value.
        res.changes[oid] = change_record(obj)

        if class_id is not None:
            obj.class_id = class_id

        if attrs:
            errors = onto.validate_attrs(attrs, obj.class_id)   # against the effective (possibly new) class
            if errors:
                raise AttrRejected(oid, errors)
            merged = dict(obj.attrs or {})
            merged.update(attrs)
            obj.attrs = merged

        if revalidate_attrs and obj.attrs:
            # A class change can make a previously valid attribute not applicable, which
            # services/quality/attr_audit.py names as a corpus-corruption source and attributes to this
            # exact path. Dropping the offending keys rather than refusing, because one stale attribute on
            # one frame must not make a ninety-frame track unfixable. change_record already captured the
            # prior attrs, so the drop comes back on revert.
            bad = _inapplicable(onto, obj.attrs, obj.class_id)
            if bad:
                obj.attrs = {k: v for k, v in obj.attrs.items() if k not in bad}
                res.attrs_dropped[oid] = sorted(bad)

        if res.new_state is not None:
            obj.state = res.new_state
        obj.source = source
        if provenance_extra:
            obj.provenance = {**(obj.provenance or {}), **provenance_extra}
        # Advance the lock version, exactly as single review does. Without this a bulk edit was invisible to
        # every other client's optimistic check, so an editor holding the object would overwrite it back.
        obj.version = (obj.version or 1) + 1

        db.add(Review(object_id=obj.object_id, reviewer=reviewer, user_id=uid, action=action,
                      before=before,
                      after={"class_id": obj.class_id, "bbox": list(obj.bbox),
                             "attrs": dict(obj.attrs or {}), "state": obj.state},
                      time_spent_ms=per_item_ms, ts_ns=now_ns()))
        res.n += 1

    return res


def _inapplicable(onto: Any, attrs: dict, class_id: int) -> set[str]:
    """Attribute keys this class cannot carry. Unknown keys are left alone, because that is a different
    fault and one this function silently deleting would hide."""
    errors = onto.validate_attrs(attrs, class_id)
    bad: set[str] = set()
    for key in attrs:
        marker = f"attribute '{key}' not applicable"
        if any(e.startswith(marker) for e in errors):
            bad.add(key)
    return bad


__all__ = ["ApplyResult", "AttrRejected", "apply_review_batch"]
