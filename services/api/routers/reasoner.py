"""The reasoning layer as an API: what it decided, why, and whether it is any good.

The last of those is the point. A reasoning layer added on faith is one nobody can tune, and every weight
in the evidence collectors is currently a guess. These routes exist so the guesses can be replaced with
measurements taken against what humans actually decided.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.deps import db_session, require_role

router = APIRouter()


@router.get("/reasoner/trace/{object_id}")
async def trace(object_id: str, db: AsyncSession = Depends(db_session)):
    """Why this object is in the state it is in.

    What makes rapid review fast: the reviewer sees the reasoning rather than working it out from the crop.
    """
    from services.autolabel.reasoner.attribution import trace_for

    return await trace_for(db, object_id)


@router.get("/reasoner/attribution", dependencies=[Depends(require_role("reviewer"))])
async def attribution(since_hours: int | None = None, db: AsyncSession = Depends(db_session)):
    """Per check: how often it argued against a label, and how often it was right to.

    Measured over reviewed objects only, because an object nobody looked at is not evidence a check was
    right; counting it would let a check earn precision on things nobody examined.
    """
    from services.autolabel.reasoner.attribution import measure_checks

    return await measure_checks(db, since_hours=since_hours)


@router.get("/reasoner/outcomes", dependencies=[Depends(require_role("reviewer"))])
async def outcomes(since_hours: int | None = None, db: AsyncSession = Depends(db_session)):
    """Did the reasoner's own decisions hold up?

    The number the layer is accountable to: of the objects it accepted, how many did a human later reject.
    """
    from services.autolabel.reasoner.attribution import decision_outcomes

    return await decision_outcomes(db, since_hours=since_hours)


@router.get("/reasoner/weights", dependencies=[Depends(require_role("reviewer"))])
async def suggested_weights(since_hours: int | None = None,
                            db: AsyncSession = Depends(db_session)):
    """What the weights would be if they followed the measurements.

    Reported, never applied. A scoring function that silently rewrites itself from a few hundred verdicts
    drifts in a way nobody notices until a class collapses.
    """
    from services.autolabel.reasoner.attribution import suggest_weights

    return await suggest_weights(db, since_hours=since_hours)


@router.get("/reasoner/coverage")
async def coverage(db: AsyncSession = Depends(db_session)):
    """How much of the corpus carries a reasoning trace at all."""
    from services.autolabel.reasoner.attribution import coverage as _coverage

    return await _coverage(db)


@router.get("/reasoner/priors")
async def priors():
    """The physical and contextual priors the checks read.

    Exposed so a domain expert can see what the system believes about India's roads and argue with it. A
    height band nobody can inspect is a magic number.
    """
    from services.autolabel.reasoner.evidence import load_priors

    p = load_priors()
    return {
        "classes_with_height": sorted(p.get("heights_m") or {}),
        "classes_with_aspect": sorted(p.get("aspect_wh") or {}),
        "never_on_road": p.get("never_on_road") or [],
        "confusable_pairs": p.get("confusable_pairs") or [],
        "overhead_classes": p.get("overhead_classes") or [],
        "loaded": bool(p),
        "detail": ("absent means unknown, never impossible: a class with no prior produces no evidence "
                   "rather than evidence against it"),
    }


class ExplainIn(BaseModel):
    """One hypothetical detection, for trying the reasoner without running a session."""

    class_name: str
    bbox: list[float]
    conf: float = 0.8
    frame_w: int = 1280
    frame_h: int = 960
    depth_m: float | None = None
    focal_px: float | None = None
    scene: dict = {}
    proposals: list[dict] = []


@router.post("/reasoner/explain")
async def explain(payload: ExplainIn):
    """Run Tier 1 over a hypothetical detection and return every finding.

    The tuning surface. Adjusting a height band or a scene rule and seeing what it does to a known-bad
    detection is far more useful than reading the weights, and needs no session to try.
    """
    from core.schemas import BBox, PathProposal, Provenance, UnifiedObject
    from services.autolabel.ontology import get_ontology
    from services.autolabel.reasoner.evidence import EvidenceContext
    from services.autolabel.reasoner.verdict import reason_about

    onto = get_ontology()
    if len(payload.bbox) != 4:
        raise HTTPException(400, "bbox must be [x1, y1, x2, y2]")

    class_id = onto.by_name(payload.class_name).id if onto.has_name(payload.class_name) else 1
    proposals = [PathProposal(path=str(p.get("path", "a")), class_name=p.get("class_name"),
                              conf=p.get("conf"), verdict="proposed", model_version="explain")
                 for p in payload.proposals]

    obj = UnifiedObject(class_id=class_id, class_name=payload.class_name,
                        bbox=BBox(x1=payload.bbox[0], y1=payload.bbox[1],
                                  x2=payload.bbox[2], y2=payload.bbox[3]),
                        conf=payload.conf,
                        provenance=Provenance(proposals=proposals,
                                              agreement=len({p.class_name for p in proposals}) == 1))
    verdict = reason_about(EvidenceContext(
        obj=obj, onto=onto, frame_w=payload.frame_w, frame_h=payload.frame_h,
        scene=payload.scene, depth_m=payload.depth_m, focal_px=payload.focal_px))
    return {"class_name": payload.class_name, **verdict.as_trace(),
            "reasons": verdict.reasons}


@router.post("/reasoner/rerun/{session_id}", dependencies=[Depends(require_role("reviewer"))])
async def rerun(session_id: str, limit: int = Query(500, ge=1, le=5000),
                apply: bool = False, db: AsyncSession = Depends(db_session)):
    """Reason over a session's existing objects, without re-detecting.

    The way to apply an improved prior to work already annotated, and the way to see what the reasoner
    would have caught before trusting it with new sessions. `apply` is off by default: seeing what it
    would do is a different act from doing it, and the second should be deliberate.
    """
    from services.autolabel.reasoner.rerun import rerun_session

    return await rerun_session(db, session_id, limit=limit, apply=apply)
