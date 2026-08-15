"""What a past relabel run did to the corpus, grouped by the mistake rather than by the object.

A relabel run rewrote 50,812 objects across 611 class pairs. Most were refinements. Some were category
errors, and finding them one object at a time is hopeless: the operator who spots a bus labelled as a bus
shelter has no way to learn that 1,046 more share it, or that 708 traffic signs and 522 hoardings landed in
the same class by different routes.

The evidence is already in the corpus. Every rewrite stamped `provenance.agent_relabel` with the move it
made, so the lineages can be counted directly rather than inferred. Grouping them turns fifty thousand
individual edits into a short list of decisions, each of which a person can accept or reject once.

`services/agent/class_move.py` then says which of those decisions the system would now refuse. That is the
useful column: a lineage the guard rejects is one the ontology itself says cannot exist, so it needs no
judgement call to identify, only the time to clean up.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Object
from services.agent.class_move import refuse_reason
from services.autolabel.ontology import get_ontology

log = get_logger("contamination")

# The stamp the relabel agent writes, e.g. "bus -> bmtc_bus_shelter (0.989)". Parsed rather than stored
# structured because the corpus already carries fifty thousand of them in this form.
_MOVE = re.compile(r"^(?P<src>.+?) -> (?P<dst>.+?) \((?P<conf>[0-9.]+)\)$")

# The frame comes back with the object because the only useful thing to do with an example is open it, and
# the correction flow lives in the frame editor. An example that cannot be opened is a number, not a finding.
#
# The current class and source come back too, and they are what makes the panel finishable. A stamp is
# history: it records that a move happened and stays there after the object is put right. Counting stamps
# means the list still reads 3,057 outstanding immediately after 3,010 of them have been restored, which is
# a board that can never go green. What needs work is the objects that STILL carry the class the move
# produced and that no person has since ruled on.
_SQL = """
    SELECT e AS move, o.object_id, o.frame_id, o.class_id, o.source
    FROM object o, LATERAL jsonb_array_elements_text(o.provenance -> 'agent_relabel') e
"""


async def agent_relabel_lineages(db: AsyncSession, *, min_count: int = 25,
                                 refused_only: bool = False) -> list[dict]:
    """Every class move a relabel run made, largest first, with whether it would be allowed today.

    `min_count` keeps the list to systematic decisions. A move that happened twice is a coincidence; one that
    happened a thousand times is a policy somebody should get to review.
    """
    onto = get_ontology()
    counts: Counter = Counter()
    outstanding: Counter = Counter()
    examples: dict = defaultdict(list)

    def _cid(name: str) -> int | None:
        try:
            return onto.by_name(name).id
        except KeyError:
            return None

    for move, oid, fid, class_id, source in (await db.execute(text(_SQL))).all():
        m = _MOVE.match(move or "")
        if not m:
            continue
        pair = (m.group("src"), m.group("dst"))
        counts[pair] += 1
        # Still wrong: carries the class the move produced, and nobody has ruled on it since.
        still = _cid(pair[1]) == int(class_id) and source != "human"
        if still:
            outstanding[pair] += 1
            # Examples come from the outstanding ones, so "open" lands on something that needs work rather
            # than on an object somebody already fixed.
            if len(examples[pair]) < 8:
                examples[pair].append({"object_id": str(oid), "frame_id": str(fid)})

    out: list[dict] = []
    for (src, dst), n in counts.most_common():
        if n < min_count:
            continue
        reason = None
        try:
            reason = refuse_reason(onto, onto.by_name(src).id, onto.by_name(dst).id)
        except KeyError:
            # A class the ontology no longer names. The move still happened and the objects still carry it,
            # so it is reported rather than dropped; it simply cannot be judged.
            reason = None
        row = {"from_name": src, "to_name": dst, "count": n,
               "outstanding": outstanding[(src, dst)],
               "refused_now": reason is not None, "reason": reason,
               "examples": examples[(src, dst)]}
        if refused_only and not row["refused_now"]:
            continue
        out.append(row)

    log.info("contamination.lineages", lineages=len(out), refused=sum(1 for r in out if r["refused_now"]))
    return out


def summarize(lineages: list[dict]) -> dict:
    """What was rewritten, and what of it is still wrong.

    `objects` and `refused_objects` count history, which is what the run did. `outstanding` counts what is
    left to fix, and it is the number that has to be able to reach zero.
    """
    return {
        "lineages": len(lineages),
        "objects": sum(r["count"] for r in lineages),
        "refused_lineages": sum(1 for r in lineages if r["refused_now"]),
        "refused_objects": sum(r["count"] for r in lineages if r["refused_now"]),
        "outstanding": sum(r["outstanding"] for r in lineages),
        "refused_outstanding": sum(r["outstanding"] for r in lineages if r["refused_now"]),
    }


async def revert_lineage(db: AsyncSession, from_name: str, to_name: str, *,
                         limit: int | None = None, created_by: str | None = None) -> dict:
    """Put one class move back, as a single reversible action.

    Only for a lineage the ontology guard refuses. A refinement is a judgement call and reverting it in bulk
    would substitute one bulk opinion for another; a category error is not a judgement call, because the
    ontology says the resulting object cannot exist.

    Three conditions decide what is touched, and each excludes a way of doing harm:

      the object still carries the class the move produced, so anything corrected since is left alone
      its source is not `human`, so nobody's decision is overwritten
      its provenance names this exact move, so a class reached by another route is not swept up

    The prior state is recorded per object, which makes this revert itself revertible. Objects go back to
    `review` rather than to whatever they held before: a class that has been wrong and then mechanically
    restored has not been looked at by anyone, and saying so is cheaper than assuming it is now right.
    """
    from services.review_batch import record_batch

    onto = get_ontology()
    reason = refuse_reason(onto, onto.by_name(from_name).id, onto.by_name(to_name).id)
    if reason is None:
        raise ValueError(
            f"{from_name} -> {to_name} is a refinement, not a category error; "
            "reverting it in bulk would replace one bulk opinion with another")

    from_id, to_id = onto.by_name(from_name).id, onto.by_name(to_name).id
    stmt = (select(Object)
            .where(Object.class_id == to_id, Object.source != "human",
                   Object.provenance["agent_relabel"].astext.like(f'%"{from_name} -> {to_name} (%'))
            .order_by(Object.object_id))
    if limit:
        stmt = stmt.limit(limit)
    rows = (await db.execute(stmt)).scalars().all()

    changes: dict[str, dict] = {}
    for obj in rows:
        changes[str(obj.object_id)] = {"from_class": int(obj.class_id), "from_state": obj.state,
                                       "from_source": obj.source, "from_attrs": dict(obj.attrs or {})}
        obj.class_id = from_id
        obj.state = "review"
        obj.version = (obj.version or 0) + 1
    await db.commit()

    run_id = await record_batch(db, changes, created_by=created_by or "contamination",
                                policy={"revert_lineage": f"{from_name} -> {to_name}", "reason": reason})
    log.info("contamination.reverted", move=f"{from_name} -> {to_name}", objects=len(changes),
             run_id=run_id)
    return {"from_name": from_name, "to_name": to_name, "reverted": len(changes),
            "run_id": run_id, "reason": reason}
