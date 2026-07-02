"""The Buyer Curation Agent: a buyer spec in plain language becomes a composed slice, an honest coverage
check, a sealed commit, and a drafted quality sheet.

It parses the spec into curation facets, counts exactly how many frames the corpus can actually fulfill,
and reports the shortfall honestly ("only 1.2k night-rain frames with an autorickshaw exist, here is the
gap") instead of over-promising. On confirmation it composes the slice, launches the export as a sealed
commit, and drafts the datasheet. The gap report doubles as collection guidance for the Fleet Dispatch
agent. Converts a day of sales engineering into minutes.
"""

from __future__ import annotations

import re

from sqlalchemy import Integer, and_, distinct, exists, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, Object, ObjectDynamics

log = get_logger("agent.buyer_agent")

_KIND = "buyer_curation"


def _target_count(text: str) -> int | None:
    """The requested volume, e.g. '10k frames' -> 10000, '1,200 frames' -> 1200, '3 frames' -> 3. A number
    counts when it is k-suffixed, at least 100, or directly followed by frames/images; small incidental
    numbers (a 2.5s TTC) are ignored."""
    best = None
    for m in re.finditer(r"(\d[\d,]*)\s*([kK])?", text):
        raw = m.group(1).replace(",", "")
        if not raw.isdigit():
            continue
        n = int(raw) * (1000 if m.group(2) else 1)
        followed = bool(re.match(r"\s*(frames?|images?|clips?)", text[m.end():m.end() + 16], re.I))
        if m.group(2) or n >= 100 or followed:
            best = n if best is None else max(best, n)
    return best


async def _available(db: AsyncSession, facets: dict) -> int:
    """True count of frames the corpus can fulfil for these facets (uncapped, unlike the copilot preview)."""
    q = select(func.count(distinct(Frame.frame_id))).select_from(Frame)
    for axis, val in facets["scene"].items():
        q = q.where(Frame.scene[axis].astext == val)
    if facets["class_ids"] or facets["attrs"] or facets["safety"]:
        oconds = [Object.frame_id == Frame.frame_id, Object.source != "human"]
        if facets["class_ids"]:
            oconds.append(Object.class_id.in_(list(facets["class_ids"])))
        for k, v in facets["attrs"].items():
            if v == ">0":
                oconds.append(Object.attrs[k].astext.cast(Integer) > 0)
            elif isinstance(v, bool):
                oconds.append(Object.attrs[k].astext == ("true" if v else "false"))
            else:
                oconds.append(Object.attrs[k].astext == v)
        q = q.where(exists(select(Object.object_id).where(and_(*oconds))))
        if facets["safety"]:
            q = q.where(exists(select(ObjectDynamics.object_id).join(Object, Object.object_id == ObjectDynamics.object_id)
                               .where(Object.frame_id == Frame.frame_id, ObjectDynamics.ttc_s.isnot(None),
                                      ObjectDynamics.ttc_s < 2.5)))
    return int((await db.execute(q)).scalar() or 0)


def _predicate(facets: dict) -> dict:
    pred: dict = {"states": ["accepted", "auto_accept"]}
    if facets["scene"]:
        pred["scene"] = facets["scene"]
    if facets["classes"]:
        pred["class_names"] = facets["classes"]
    return pred


async def analyze_spec(db: AsyncSession, text: str) -> dict:
    """Parse the spec, count what the corpus can fulfil, and report the shortfall + collection guidance."""
    from services.agent.copilot import parse_query
    from services.autolabel.ontology import get_ontology

    facets = parse_query(text, get_ontology())
    target = _target_count(text)
    available = await _available(db, facets)
    parts = []
    if facets["classes"]:
        parts.append(", ".join(facets["classes"][:4]))
    if facets["scene"]:
        parts.append(", ".join(f"{a}={v}" for a, v in facets["scene"].items()))
    if facets["safety"]:
        parts.append("near-miss")
    understood = " · ".join(parts) or "everything"

    fulfillment = {"requested": target, "available": available,
                   "fulfillable": (min(target, available) if target else available),
                   "shortfall": (max(0, target - available) if target else 0)}
    guidance = None
    if target and available < target:
        cov = {}
        from services.agent.coverage import analyze_coverage

        cov = await analyze_coverage(db)
        rel = [g for g in cov.get("gaps", []) if any(str(v) in g for v in facets["scene"].values())]
        guidance = (f"corpus can fulfil {available} of {target} for '{understood}'. "
                    f"Gap of {target - available}." + (" Thin: " + "; ".join(rel[:3]) if rel else ""))
    return {"understood": understood, "facets": {k: facets[k] for k in ("scene", "classes", "safety")},
            "predicate": _predicate(facets), "fulfillment": fulfillment, "guidance": guidance}


async def fulfill(db: AsyncSession, text: str, name: str, *, created_by: str | None = None) -> dict:
    """Compose the slice, launch the sealed export, and draft the datasheet."""
    import asyncio

    from services.agent.doc_agent import generate_datasheet
    from services.curation.slices import create_slice
    from services.export.dataset import SliceSpec, export_dataset

    analysis = await analyze_spec(db, text)
    pred = analysis["predicate"]
    slice_row = await create_slice(name, pred, description=f"buyer spec: {text[:200]}")
    spec = SliceSpec(name=name, class_names=pred.get("class_names"), states=pred.get("states"),
                     formats=["coco", "parquet"])

    async def _run() -> None:
        try:
            await export_dataset(spec)
        except Exception as exc:  # noqa: BLE001
            log.error("buyer.export_failed", error=str(exc))

    asyncio.create_task(_run())
    datasheet = await generate_datasheet(db, title=name)
    log.info("buyer.fulfill", name=name, slice_id=slice_row.get("slice_id"))
    return {"slice": slice_row, "export": "started (sealed commit)", "datasheet_uri": datasheet["uri"],
            "analysis": analysis}


async def spec(db: AsyncSession, text: str, *, name: str | None = None, confirm: bool = False,
               created_by: str | None = None) -> dict:
    """Analyze always; fulfil (compose + export + datasheet) only on confirm."""
    analysis = await analyze_spec(db, text)
    if not confirm:
        return {"status": "analyzed", **analysis}
    out = await fulfill(db, text, name or "buyer-slice", created_by=created_by)
    return {"status": "fulfilled", **out}
