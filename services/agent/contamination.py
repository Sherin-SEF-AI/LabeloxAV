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

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from services.agent.class_move import refuse_reason
from services.autolabel.ontology import get_ontology

log = get_logger("contamination")

# The stamp the relabel agent writes, e.g. "bus -> bmtc_bus_shelter (0.989)". Parsed rather than stored
# structured because the corpus already carries fifty thousand of them in this form.
_MOVE = re.compile(r"^(?P<src>.+?) -> (?P<dst>.+?) \((?P<conf>[0-9.]+)\)$")

# The frame comes back with the object because the only useful thing to do with an example is open it, and
# the correction flow lives in the frame editor. An example that cannot be opened is a number, not a finding.
_SQL = """
    SELECT e AS move, o.object_id, o.frame_id
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
    examples: dict = defaultdict(list)

    for move, oid, fid in (await db.execute(text(_SQL))).all():
        m = _MOVE.match(move or "")
        if not m:
            continue
        pair = (m.group("src"), m.group("dst"))
        counts[pair] += 1
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
               "refused_now": reason is not None, "reason": reason,
               "examples": examples[(src, dst)]}
        if refused_only and not row["refused_now"]:
            continue
        out.append(row)

    log.info("contamination.lineages", lineages=len(out), refused=sum(1 for r in out if r["refused_now"]))
    return out


def summarize(lineages: list[dict]) -> dict:
    """The two numbers worth putting at the top: how much was rewritten, and how much of it is now refused."""
    return {
        "lineages": len(lineages),
        "objects": sum(r["count"] for r in lineages),
        "refused_lineages": sum(1 for r in lineages if r["refused_now"]),
        "refused_objects": sum(r["count"] for r in lineages if r["refused_now"]),
    }
