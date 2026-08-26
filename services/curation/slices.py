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
    min_conf (float), regions (list of city or state names), context (dict of pack context axis to values),
    rarity_min / rarity_max (float), track_event_types (list). Every clause present must hold (AND); a
    missing clause is unconstrained.

    The record may also carry `track_event_types`, the accepted event types on this frame's tracks."""
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

    # The clauses below have twins in services/export/dataset.py's SliceSpec, and the two are required to
    # agree: a cohort that previews as 900 frames and exports as 40,000 is worse than one that cannot be
    # exported at all. Each is the pure-Python reading of the same SQL clause there.

    # Region, resolved rather than matched. `cities` compares the raw string, so a predicate asking for
    # Bengaluru misses the 372 sessions recorded as `BLR`; this asks the pack's alias table instead.
    if predicate.get("regions"):
        from services.context.region import resolve_region

        r = resolve_region(record.get("city"))
        wanted = {w.strip().lower() for w in predicate["regions"]}
        if not ({(r.city or "").lower(), (r.state or "").lower()} & wanted):
            return False

    # The pack's frame-context axes, beyond the four the ingest classifier writes. Read from the pack rather
    # than listed here so a domain that declares a new axis becomes filterable without editing this file.
    for key, want in (predicate.get("context") or {}).items():
        if want and scene.get(key) not in want:
            return False

    rarity = scene.get("rarity")
    lo, hi = predicate.get("rarity_min"), predicate.get("rarity_max")
    if lo is not None or hi is not None:
        # An unscored frame is excluded from a rarity band rather than treated as rarity zero. Zero is a
        # real score meaning "nothing unusual here", and conflating it with "not yet measured" would fill
        # every low-rarity cohort with frames nobody has scored.
        if rarity is None:
            return False
        if lo is not None and rarity < lo:
            return False
        if hi is not None and rarity > hi:
            return False

    if predicate.get("track_event_types"):
        have = set(record.get("track_event_types") or [])
        if not (have & set(predicate["track_event_types"])):
            return False
    return True


def slice_to_export_spec(slice_row, formats: list | None = None) -> dict:
    """Convert a saved slice to the fields of an export SliceSpec, so a cohort exports without redefining it.
    The scene-axis and tag clauses have no SliceSpec column, so they are carried as separate filters for the
    caller to apply via the frame/object query; the column-backed clauses map directly.

    `unsupported` names any clause that neither the spec nor the side filters can express. It must be checked
    by the caller: a cohort defined purely by such a clause would otherwise export as the WHOLE corpus rather
    than the intended subset, which is a silent data-correctness failure, not a cosmetic one."""
    p = slice_row.predicate or {}
    spec = {"name": slice_row.name, "class_names": p.get("class_names"), "states": p.get("states"),
            "cities": p.get("cities"), "min_conf": p.get("min_conf"),
            "formats": formats or ["coco", "parquet"]}
    scene = {axis: p[axis] for axis in _SCENE_AXES if p.get(axis)}
    tags = {k: p[k] for k in ("tags", "frame_tags") if p.get(k)}
    handled = set(_SCENE_AXES) | {"class_names", "states", "cities", "min_conf", "tags", "frame_tags",
                                  "session_id", "sources", "max_conf", "object_ids", "frame_ids"}
    unsupported = sorted(k for k, v in p.items() if v not in (None, [], {}) and k not in handled)
    return {"spec": {k: v for k, v in spec.items() if v is not None},
            "scene_filter": scene, "tag_filter": tags,
            "session_id": p.get("session_id"), "sources": p.get("sources"),
            "max_conf": p.get("max_conf"), "unsupported": unsupported}


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
