"""M-Q.0 ontology pullback: which classes are grounded enough to prompt the open-vocab models with.

The open-vocab detector (Path B) and the VLM (Path C) can only label what they are prompted with, so an
ungrounded class name in the prompt is an invitation to hallucinate it. The grounded set is the curated
supported core (IDD-anchored road actors and safety primitives) plus the structural fallback classes plus
any class that has EARNED re-entry with promotion_min_instances gate-accepted instances. Acceptance, not a
single human draw, is the bar, so a class someone sketched a few times (water_bottles, buildings) stays out
until it is genuinely supported. Everything outside the set folds into object_fallback / vehicle_fallback.
"""

from __future__ import annotations

from sqlalchemy import func, select

from core.config import get_settings
from core.logging import get_logger
from db.models import Object
from db.session import get_sessionmaker
from services.autolabel.ontology import Ontology, get_ontology

log = get_logger("grounding")


def supported_core_ids(onto: Ontology | None = None) -> set[int]:
    """The curated supported core (config) plus the fallback classes, resolved to ids. Names absent from
    the active ontology are skipped, so the list is safe to evolve independently of the YAML."""
    onto = onto or get_ontology()
    s = get_settings()
    ids = {onto.by_name(n).id for n in s.ontology.supported_core if onto.has_name(n)}
    ids |= set(onto.fallback_ids())
    return ids


async def promoted_ids(min_instances: int | None = None) -> set[int]:
    """Classes that have earned re-entry: at least promotion_min_instances gate-accepted instances. Uses
    accepted / auto_accept (gold-quality), not raw human draws, so a few sketches do not promote a class."""
    s = get_settings()
    n = min_instances if min_instances is not None else s.ontology.promotion_min_instances
    async with get_sessionmaker()() as db:
        rows = (await db.execute(
            select(Object.class_id, func.count())
            .where(Object.state.in_(("accepted", "auto_accept")))
            .group_by(Object.class_id))).all()
    return {int(cid) for cid, count in rows if count >= n}


async def supported_concept_ids(min_instances: int | None = None) -> set[int]:
    """The full set of class ids safe to prompt the open-vocab models with: supported core + fallback +
    promoted. Logged so the active prompt surface is always inspectable."""
    onto = get_ontology()
    ids = supported_core_ids(onto) | await promoted_ids(min_instances)
    ids = {cid for cid in ids if cid in onto._by_id}  # guard against stale config names
    log.info("grounding.supported_concepts", count=len(ids), total_classes=len(onto.classes))
    return ids


async def promotion_candidates(min_instances: int | None = None) -> list[dict]:
    """Classes outside the supported core that have earned promotion (>= threshold accepted instances), for
    the governor to review and add to ontology.supported_core. Promotion is earned by data, not by hope."""
    onto = get_ontology()
    core = supported_core_ids(onto)
    promoted = await promoted_ids(min_instances)
    out = []
    for cid in sorted(promoted - core):
        if cid in onto._by_id:
            c = onto.by_id(cid)
            out.append({"class_id": cid, "name": c.name, "l1": c.l1})
    return out
