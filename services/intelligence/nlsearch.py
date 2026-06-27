"""Natural-language scenario search. A plain-language query is parsed into structured filters over
the scenario index (event type, actor class, light/surface tags). The semantic/embedding match
(CLIP/SigLIP over scene captions) is the next layer; the structured parse is the deterministic core.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Scenario
from db.models import Session as DbSession
from services.autolabel.ontology import get_ontology

EVENT_PHRASES = {
    "cut in": "cut_in", "cut-in": "cut_in", "cutting in": "cut_in",
    "wrong side": "wrong_side", "wrong-side": "wrong_side", "oncoming": "wrong_side",
    "near miss": "near_miss", "near-miss": "near_miss",
    "hard brake": "hard_brake", "braking": "hard_brake", "hard-brake": "hard_brake",
    "congestion": "congestion", "traffic jam": "congestion", "jam": "congestion",
    "animal": "animal_on_road", "cattle on road": "animal_on_road",
    "illegal park": "illegal_park", "parked": "illegal_park", "parking": "illegal_park",
}

CLASS_SYNONYMS = {
    "auto rickshaw": "autorickshaw", "auto-rickshaw": "autorickshaw", "rickshaw": "autorickshaw",
    "auto": "autorickshaw", "cow": "cattle", "tanker": "water_tanker", "water tanker": "water_tanker",
    "bike": "motorcycle", "two wheeler": "motorcycle", "pushcart": "push_cart",
}

LIGHT_TAGS = {"night", "dusk", "dawn", "day"}
SURFACE_TAGS = {"wet": "wet", "rain": "wet", "rainy": "wet", "dry": "dry"}


@dataclass
class ParsedQuery:
    types: set[str] = field(default_factory=set)
    actor_classes: set[str] = field(default_factory=set)
    tags: set[str] = field(default_factory=set)
    raw: str = ""


def parse_query(q: str) -> ParsedQuery:
    text = q.lower()
    onto = get_ontology()
    parsed = ParsedQuery(raw=q)

    for phrase, etype in EVENT_PHRASES.items():
        if phrase in text:
            parsed.types.add(etype)

    for syn, name in CLASS_SYNONYMS.items():
        if syn in text:
            parsed.actor_classes.add(name)
    for c in onto.classes:
        if c.name.replace("_", " ") in text:
            parsed.actor_classes.add(c.name)

    for t in LIGHT_TAGS:
        if t in text:
            parsed.tags.add(t)
    for word, tag in SURFACE_TAGS.items():
        if word in text:
            parsed.tags.add(tag)
    return parsed


def _matches(scn: Scenario, p: ParsedQuery) -> bool:
    if p.types and scn.type not in p.types:
        return False
    if p.tags and not p.tags.issubset(set(scn.tags or [])):
        return False
    if p.actor_classes:
        actor_classes = set((scn.meta or {}).get("actor_classes", []))
        single = (scn.meta or {}).get("class")
        if single:
            actor_classes.add(single)
        if not (p.actor_classes & actor_classes):
            return False
    return True


async def search_scenarios(
    db: AsyncSession,
    query: str,
    city: str | None = None,
    session_id: str | None = None,
    limit: int = 100,
    semantic: bool = False,
) -> list[dict]:
    parsed = parse_query(query)
    stmt = select(Scenario, DbSession.city, DbSession.vehicle_id).join(
        DbSession, Scenario.session_id == DbSession.session_id
    ).order_by(Scenario.criticality.desc())
    if parsed.types:
        stmt = stmt.where(Scenario.type.in_(parsed.types))
    if city:
        stmt = stmt.where(DbSession.city == city)
    if session_id:
        from uuid import UUID

        stmt = stmt.where(Scenario.session_id == UUID(session_id))

    rows = (await db.execute(stmt)).all()
    matched = [(scn, scity, vehicle) for scn, scity, vehicle in rows if _matches(scn, parsed)]

    if semantic and query.strip():
        # Blend structured match with a CLIP text-vs-actor-crop cosine, when embeddings exist.
        from services.intelligence.embeddings import encode_text, scenario_embedding

        qv = encode_text(query)
        scored = []
        for scn, scity, vehicle in matched:
            emb = await scenario_embedding(db, scn.actors)
            sem = float(emb @ qv) if emb is not None else 0.0
            d = _scenario_dict(scn, scity, vehicle)
            d["semantic_score"] = round(sem, 4)
            d["rank_score"] = round(0.5 * scn.criticality + 0.5 * sem, 4)
            scored.append(d)
        scored.sort(key=lambda d: d["rank_score"], reverse=True)
        return scored[:limit]

    return [_scenario_dict(scn, scity, vehicle) for scn, scity, vehicle in matched][:limit]


def _scenario_dict(scn: Scenario, city: str | None, vehicle: str | None) -> dict:
    return {
        "scenario_id": str(scn.scenario_id),
        "session_id": str(scn.session_id),
        "type": scn.type,
        "t_in_ns": scn.t_in_ns,
        "t_out_ns": scn.t_out_ns,
        "actors": scn.actors,
        "criticality": scn.criticality,
        "tags": scn.tags,
        "meta": scn.meta,
        "city": city,
        "vehicle_id": vehicle,
    }
