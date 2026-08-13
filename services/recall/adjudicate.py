"""Ruling on a recall candidate, and the loop that could not close without it.

Recall mining finds objects the detector missed: a gap in a track, an open-vocabulary hit with no box, a
region proposal nothing claimed. Each becomes a `RecallCandidate` at `status="pending"`.

Nothing could ever move one off `pending`. There was no route and no service function that wrote `confirmed`
or `rejected`, so the mining wrote rows that could be listed and never judged. Two things followed, and the
second is the expensive one.

`fit_channel_reliability` (`services/recall/recover.py`) exists to replace the hand-guessed per-channel
priors with measured ones, and it selects on exactly those two statuses. Its query was guaranteed empty, so
the priors stayed guesses forever and the ranking never learned which channel was worth trusting. A feature
whose whole point is to improve with use could not.

Confirming a candidate is a statement about the object, so it moves the object too: a recovered detection
somebody has agreed is real belongs in the review queue as human-sourced work, not left as a machine guess.
Rejecting one leaves the object alone. It says the CANDIDATE was a bad suggestion, which is a claim about
the miner rather than about the label, and quietly deleting somebody's object on the strength of that would
be the mining marking its own homework in the other direction.
"""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Object, RecallCandidate

log = get_logger("recall.adjudicate")

PENDING = "pending"
CONFIRMED = "confirmed"
REJECTED = "rejected"
VERDICTS = (CONFIRMED, REJECTED)


class AdjudicationError(ValueError):
    """A verdict that is not a verdict, or a candidate that is not there."""


async def adjudicate(db: AsyncSession, candidate_id: str, verdict: str, *,
                     reviewer: str | None = None) -> dict:
    """Rule on one candidate. Returns what changed.

    Re-ruling is allowed and is not an error: a reviewer who confirms and then thinks again should not have
    to find an administrator. The reliability fit reads the current status, so the last word is the one that
    counts.
    """
    if verdict not in VERDICTS:
        raise AdjudicationError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
    cand = await db.get(RecallCandidate, uuid.UUID(candidate_id))
    if cand is None:
        raise AdjudicationError("candidate not found")

    was = cand.status
    cand.status = verdict
    moved = False

    if verdict == CONFIRMED:
        obj = await db.get(Object, cand.object_id)
        # Only a machine label is promoted. An object a person has already ruled on is theirs, and a recall
        # confirmation is a statement about the miner having been right, not permission to overwrite them.
        if obj is not None and obj.source != "human":
            obj.state = "review"
            obj.version = (obj.version or 0) + 1
            prov = dict(obj.provenance or {})
            prov["recall_confirmed"] = {"candidate_id": str(cand.candidate_id),
                                        "channels": list(cand.channels or []),
                                        "by": reviewer or "reviewer"}
            obj.provenance = prov
            moved = True

    await db.commit()
    log.info("recall.adjudicated", candidate_id=candidate_id, verdict=verdict, was=was, moved=moved)
    return {"candidate_id": candidate_id, "verdict": verdict, "was": was,
            "object_id": str(cand.object_id), "object_routed_to_review": moved}


async def adjudication_progress(db: AsyncSession) -> dict:
    """How much of the mining has been judged, which is what says whether the priors can be fitted yet.

    Reported per channel as well as in total: the fit needs a minimum number of verdicts per channel before
    it will apply a measurement, so a total that looks healthy can still hide a channel nobody has ruled on.
    """
    rows = (await db.execute(
        select(RecallCandidate.status, func.count()).group_by(RecallCandidate.status))).all()
    by_status = {s: int(n) for s, n in rows}

    per_channel: dict[str, dict[str, int]] = {}
    for channels, status in (await db.execute(
            select(RecallCandidate.channels, RecallCandidate.status))).all():
        for ch in (channels or []):
            t = per_channel.setdefault(ch, {PENDING: 0, CONFIRMED: 0, REJECTED: 0})
            if status in t:
                t[status] += 1

    judged = by_status.get(CONFIRMED, 0) + by_status.get(REJECTED, 0)
    total = sum(by_status.values())
    return {"total": total, "judged": judged, "pending": by_status.get(PENDING, 0),
            "confirmed": by_status.get(CONFIRMED, 0), "rejected": by_status.get(REJECTED, 0),
            "per_channel": per_channel}
