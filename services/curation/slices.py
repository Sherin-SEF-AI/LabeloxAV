"""Milestone I: curation slices. A named, persisted dataset cohort defined once and reused for export,
training, and review. The membership predicate is a conjunction over the SigLIP2 scene axes (weather,
time_of_day, road_type, density), the frame's city, and its objects' classes / states / confidence: a frame
is in the slice only if it satisfies every clause present. An empty predicate is the universal slice. The
predicate test is pure so cohort membership is verified without infra; the same slice converts to an export
SliceSpec so curation and export stay one definition.
"""

from __future__ import annotations

from core.logging import get_logger

log = get_logger("curation_slices")

_SCENE_AXES = ("weather", "time_of_day", "road_type", "density")


def matches_predicate(record: dict, predicate: dict) -> bool:
    """record: {scene:{weather,time_of_day,road_type,density}, city, classes:[...], states:[...], max_conf}.
    predicate clauses: any of the scene axes (list), cities (list), class_names (list), states (list),
    min_conf (float). Every clause present must hold (AND); a missing clause is unconstrained."""
    scene = record.get("scene") or {}
    for axis in _SCENE_AXES:
        want = predicate.get(axis)
        if want and scene.get(axis) not in want:
            return False
    if predicate.get("cities") and record.get("city") not in predicate["cities"]:
        return False
    if predicate.get("class_names") and not (set(record.get("classes") or []) & set(predicate["class_names"])):
        return False
    if predicate.get("states") and not (set(record.get("states") or []) & set(predicate["states"])):
        return False
    min_conf = predicate.get("min_conf")
    if min_conf is not None and (record.get("max_conf") or 0.0) < min_conf:
        return False
    return True


def slice_to_export_spec(slice_row, formats: list | None = None) -> dict:
    """Convert a saved slice to the fields of an export SliceSpec, so a cohort exports without redefining it.
    The scene-axis clauses have no SliceSpec column, so they are carried as a note for the caller to apply
    via the frame query; the column-backed clauses map directly."""
    p = slice_row.predicate or {}
    spec = {"name": slice_row.name, "class_names": p.get("class_names"), "states": p.get("states"),
            "cities": p.get("cities"), "min_conf": p.get("min_conf"),
            "formats": formats or ["coco", "parquet"]}
    scene = {axis: p[axis] for axis in _SCENE_AXES if p.get(axis)}
    return {"spec": {k: v for k, v in spec.items() if v is not None}, "scene_filter": scene}


async def create_slice(name: str, predicate: dict, description: str | None = None) -> dict:
    from db.models import CurationSlice
    from db.session import get_sessionmaker
    async with get_sessionmaker()() as db:
        row = CurationSlice(name=name, predicate=predicate or {}, description=description)
        db.add(row)
        await db.commit()
        await db.refresh(row)
        sid = str(row.slice_id)
    log.info("curation.slice_created", name=name, slice=sid)
    return {"slice_id": sid, "name": name, "version": 1}


async def materialize_slice(slice_id, sample: int = 20) -> dict:
    """Count the frames in the cohort and return a small sample, by streaming each frame's scene + object
    rollup through the pure predicate. The count is the cohort size a curator sees before exporting."""
    from sqlalchemy import select

    from db.models import CurationSlice, Frame, Object
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    async with get_sessionmaker()() as db:
        slice_row = await db.get(CurationSlice, slice_id)
        if slice_row is None:
            return {"error": "slice not found"}
        pred = slice_row.predicate or {}
        rows = (await db.execute(
            select(Frame.frame_id, Frame.scene, DbSession.city, Object.class_id, Object.state, Object.conf)
            .join(DbSession, Frame.session_id == DbSession.session_id)
            .join(Object, Object.frame_id == Frame.frame_id, isouter=True))).all()
    from services.autolabel.ontology import get_ontology
    onto = get_ontology()
    by_frame: dict = {}
    for fid, scene, city, class_id, state, conf in rows:
        rec = by_frame.setdefault(str(fid), {"scene": scene or {}, "city": city, "classes": set(),
                                             "states": set(), "max_conf": 0.0})
        if class_id is not None:
            rec["classes"].add(onto.by_id(class_id).name)
            if state:
                rec["states"].add(state)
            rec["max_conf"] = max(rec["max_conf"], float(conf or 0.0))
    matched = [fid for fid, rec in by_frame.items()
               if matches_predicate({**rec, "classes": list(rec["classes"]), "states": list(rec["states"])}, pred)]
    return {"slice_id": str(slice_id), "name": slice_row.name, "count": len(matched),
            "sample": matched[:sample]}
