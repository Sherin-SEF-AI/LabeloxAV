"""The database side of the top-down lane editor (#7): lanes in BEV, their relations, ego-pose propagation, export.

Geometry lives in `bev_view.py` (the warp) and `bev_lanes.py` (splines, ground, world); this module reads and
writes `lane` and `lane_relation` rows and puts the two together. Lanes stay in image space in the database, as
they always have, so the camera-view editor, optical-flow propagation and every consumer keep working unchanged.
"""

from __future__ import annotations

import asyncio
import uuid

import numpy as np
from sqlalchemy import delete, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import EgoPose, Frame, Lane, LaneRelation, MapElement, Object, OntologyClass
from db.models import Session as DbSession

log = get_logger("lane_map")

SIGNAL_CLASSES = ("traffic_signal", "traffic_light")
POSE_MATCH_NS = 60_000_000


async def frame_context(db: AsyncSession, frame_id: uuid.UUID):
    from services.calibration.resolve import resolve_calibration
    from services.hdmap.bev_view import view_for

    frame = await db.get(Frame, frame_id)
    if frame is None:
        raise LookupError("frame not found")
    cal = await resolve_calibration(frame.session_id, frame.cam_id, frame.width, frame.height)
    return frame, cal, view_for(cal)


def _rel_out(r: LaneRelation) -> dict:
    return {"relation_id": str(r.relation_id), "lane_id": str(r.lane_id), "relation": r.relation,
            "target_lane_id": str(r.target_lane_id) if r.target_lane_id else None,
            "target_object_id": str(r.target_object_id) if r.target_object_id else None,
            "target_element_id": str(r.target_element_id) if r.target_element_id else None}


async def _pose_at(db: AsyncSession, session_id, ts_ns: int) -> EgoPose | None:
    rows = (await db.execute(select(EgoPose).where(
        EgoPose.session_id == session_id, EgoPose.ts_ns >= ts_ns - POSE_MATCH_NS,
        EgoPose.ts_ns <= ts_ns + POSE_MATCH_NS))).scalars().all()
    return min(rows, key=lambda p: abs(p.ts_ns - ts_ns)) if rows else None


async def bev_state(db: AsyncSession, frame_id: uuid.UUID) -> dict:
    """Everything the top-down editor shows for one frame: lanes in BEV pixels, relations, signals, poses."""
    from services.hdmap.bev_view import image_to_bev

    frame, cal, view = await frame_context(db, frame_id)
    lanes = (await db.execute(select(Lane).where(Lane.frame_id == frame_id)
                              .order_by(Lane.created_at))).scalars().all()
    ids = [ln.lane_id for ln in lanes]
    rels = (await db.execute(select(LaneRelation).where(LaneRelation.lane_id.in_(ids)))).scalars().all() if ids else []
    sig = (await db.execute(
        select(Object.object_id, Object.bbox, OntologyClass.name).join(OntologyClass, OntologyClass.id == Object.class_id)
        .where(Object.frame_id == frame_id, OntologyClass.name.in_(SIGNAL_CLASSES)))).all()
    pose = await _pose_at(db, frame.session_id, frame.ts_ns)
    n_next = len((await db.execute(select(Frame.frame_id).where(
        Frame.session_id == frame.session_id, Frame.cam_id == frame.cam_id, Frame.ts_ns > frame.ts_ns)
        .limit(200))).all())
    out = []
    for ln in lanes:
        pts = image_to_bev(ln.control_points or [], cal, view)
        out.append({"lane_id": str(ln.lane_id), "track_ref": str(ln.track_ref) if ln.track_ref else None,
                    "control_points": ln.control_points, "bev_points": pts,
                    "dropped": max(0, len(ln.control_points or []) - len(pts)), "lane_type": ln.lane_type,
                    "is_ego": ln.is_ego, "source": ln.source, "marking_conf": ln.marking_conf,
                    "method": (ln.provenance or {}).get("method")})
    return {"frame_id": str(frame_id), "session_id": str(frame.session_id), "cam_id": frame.cam_id,
            "view": view.as_dict(),
            "calibration": {"source": cal.source, "quality": round(float(cal.quality), 3), "model": cal.model},
            "lanes": out, "relations": [_rel_out(r) for r in rels],
            "signals": [{"object_id": str(o), "bbox": b, "class_name": n} for o, b, n in sig],
            "has_pose": pose is not None, "pose_source": pose.source if pose else None,
            "next_frames": n_next}


async def save_bev_lane(db: AsyncSession, frame_id: uuid.UUID, bev_points: list, lane_type: str, is_ego: bool,
                        lane_id: uuid.UUID | None = None) -> dict:
    """Create or replace a lane from BEV pixels. Stored in image space; source becomes human."""
    from services.hdmap.bev_lanes import LANE_TYPES
    from services.hdmap.bev_view import bev_to_image

    if lane_type not in LANE_TYPES:
        raise ValueError(f"lane type must be one of {', '.join(LANE_TYPES)}")
    frame, cal, view = await frame_context(db, frame_id)
    pts = bev_to_image(bev_points or [], cal, view)
    if len(pts) < 2:
        raise ValueError("fewer than two of those points land on the road ahead of the camera")
    if lane_id is not None:
        lane = await db.get(Lane, lane_id)
        if lane is None or lane.frame_id != frame_id:
            raise LookupError("lane not found on this frame")
    else:
        lane = Lane(frame_id=frame.frame_id, session_id=frame.session_id, source="human")
        db.add(lane)
    lane.control_points, lane.lane_type, lane.is_ego, lane.source = pts, lane_type, bool(is_ego), "human"
    # A person drew this on the flattened road: an assertion about where the lane runs, not a measurement of
    # the paint, and not the proposer's any more.
    lane.marking_conf, lane.model_version = None, None
    lane.provenance = {**(lane.provenance or {}), "drawn_in": "bev"}
    await db.flush()
    out = {"lane_id": str(lane.lane_id), "control_points": pts, "lane_type": lane_type, "is_ego": bool(is_ego)}
    await db.commit()
    return out


async def add_relation(db: AsyncSession, lane_id: uuid.UUID, relation: str, *, target_lane_id=None,
                       target_object_id=None, target_element_id=None, user_id=None) -> dict:
    from db.models_tools_driving import LANE_RELATIONS

    if relation not in LANE_RELATIONS:
        raise ValueError(f"relation must be one of {', '.join(LANE_RELATIONS)}")
    targets = [t for t in (target_lane_id, target_object_id, target_element_id) if t is not None]
    if len(targets) != 1:
        raise ValueError("a relation names exactly one target: a lane, an object or a map element")
    lane = await db.get(Lane, lane_id)
    if lane is None:
        raise LookupError("lane not found")
    if relation in ("predecessor", "successor", "left_neighbour", "right_neighbour", "stops_at"):
        if target_lane_id is None:
            raise ValueError(f"{relation} needs a target lane")
        other = await db.get(Lane, target_lane_id)
        if other is None or other.session_id != lane.session_id:
            raise ValueError("the target lane must be in the same session")
        if target_lane_id == lane_id:
            raise ValueError("a lane cannot relate to itself")
        if relation == "stops_at" and other.lane_type != "stop_line":
            raise ValueError("stops_at needs a lane typed stop_line")
    if relation == "controlled_by":
        if target_object_id is not None:
            obj = await db.get(Object, target_object_id)
            if obj is None:
                raise ValueError("signal object not found")
        elif target_element_id is not None:
            el = await db.get(MapElement, target_element_id)
            if el is None or el.kind not in ("signal", "sign"):
                raise ValueError("the map element must be a signal or sign")
        else:
            raise ValueError("controlled_by needs a signal object or a map element")
    existing = (await db.execute(select(LaneRelation).where(
        LaneRelation.lane_id == lane_id, LaneRelation.relation == relation,
        LaneRelation.target_lane_id.is_(None) if target_lane_id is None else LaneRelation.target_lane_id == target_lane_id,
        LaneRelation.target_object_id.is_(None) if target_object_id is None
        else LaneRelation.target_object_id == target_object_id,
        LaneRelation.target_element_id.is_(None) if target_element_id is None
        else LaneRelation.target_element_id == target_element_id))).scalar_one_or_none()
    if existing is not None:
        return _rel_out(existing)
    row = LaneRelation(session_id=lane.session_id, lane_id=lane_id, relation=relation, target_lane_id=target_lane_id,
                       target_object_id=target_object_id, target_element_id=target_element_id, created_by=user_id)
    db.add(row)
    await db.flush()
    out = _rel_out(row)
    await db.commit()
    return out


async def delete_relation(db: AsyncSession, relation_id: uuid.UUID) -> bool:
    row = await db.get(LaneRelation, relation_id)
    if row is None:
        return False
    await db.delete(row)
    await db.commit()
    return True


async def propagate_by_pose(db: AsyncSession, frame_id: uuid.UUID, frames: int = 10,
                            lane_ids: list[uuid.UUID] | None = None, progress=None) -> dict:
    """Carry a frame's lanes onto the next frames through the ego poses (flat road, measured or visual poses).

    Re-running replaces this method's earlier copies of the same lanes; a human lane with the same track_ref on
    a target frame is never overwritten (that frame is skipped for that lane).
    """
    from services.calibration.resolve import resolve_calibration
    from services.hdmap.bev_lanes import carry_lane, ego_transform
    from services.hdmap.bev_view import view_for

    frame, cal, _view = await frame_context(db, frame_id)
    sess = await db.get(DbSession, frame.session_id)
    sensors = sess.sensors if sess is not None else None
    src_pose = await _pose_at(db, frame.session_id, frame.ts_ns)
    if src_pose is None:
        raise ValueError("this frame has no ego pose, so its lanes cannot be carried by geometry; "
                         "use optical-flow propagation instead")
    q = select(Lane).where(Lane.frame_id == frame_id, Lane.lane_type != "stop_line")
    if lane_ids:
        q = q.where(Lane.lane_id.in_(lane_ids))
    lanes = (await db.execute(q)).scalars().all()
    if not lanes:
        return {"created": 0, "frames": 0, "reason": "no lanes on the source frame"}
    nexts = (await db.execute(select(Frame).where(
        Frame.session_id == frame.session_id, Frame.cam_id == frame.cam_id, Frame.ts_ns > frame.ts_ns)
        .order_by(Frame.ts_ns).limit(max(1, min(int(frames), 500))))).scalars().all()
    t_src = ego_transform(src_pose, sensors, cal)
    for ln in lanes:
        if ln.track_ref is None:
            ln.track_ref = ln.lane_id
    refs = [ln.track_ref for ln in lanes]
    created, skipped_no_pose, skipped_human, out_of_view, touched = 0, 0, 0, 0, 0
    for k, nf in enumerate(nexts):
        pose = await _pose_at(db, frame.session_id, nf.ts_ns)
        if pose is None:
            skipped_no_pose += 1
            continue
        cal_n = await resolve_calibration(nf.session_id, nf.cam_id, nf.width, nf.height)
        t_dst = ego_transform(pose, sensors, cal_n)
        existing = (await db.execute(select(Lane).where(Lane.frame_id == nf.frame_id,
                                                        Lane.track_ref.in_(refs)))).scalars().all()
        human_refs = {e.track_ref for e in existing if e.source == "human"}
        await db.execute(delete(Lane).where(
            Lane.frame_id == nf.frame_id, Lane.track_ref.in_(refs), Lane.source == "propagated",
            Lane.provenance["method"].astext == "ego_pose"))
        far = view_for(cal_n).far_m
        any_here = False
        for ln in lanes:
            if ln.track_ref in human_refs:
                skipped_human += 1
                continue
            pts = await asyncio.to_thread(carry_lane, ln.control_points or [], cal, cal_n, t_src, t_dst,
                                          nf.width, nf.height, far)
            if pts is None:
                out_of_view += 1
                continue
            db.add(Lane(frame_id=nf.frame_id, session_id=nf.session_id, control_points=pts, lane_type=ln.lane_type,
                        is_ego=ln.is_ego, source="propagated", track_ref=ln.track_ref, marking_conf=None,
                        model_version="ego-pose", provenance={
                            "method": "ego_pose", "from_frame": str(frame_id), "from_lane": str(ln.lane_id),
                            "pose_source": pose.source, "pose_measured": bool(pose.measured),
                            "inherited_type": ln.lane_type}))
            created += 1
            any_here = True
        touched += int(any_here)
        if progress is not None:
            await progress((k + 1) / max(1, len(nexts)))
    await db.commit()
    log.info("lanes.propagated_by_pose", frame=str(frame_id), created=created, frames=len(nexts))
    return {"created": created, "frames": len(nexts), "frames_with_lanes": touched,
            "skipped_no_pose": skipped_no_pose, "skipped_human": skipped_human, "out_of_view": out_of_view,
            "pose_source": src_pose.source}


async def _origin(db: AsyncSession, session_id) -> tuple[float, float] | None:
    from services.tools.driving_paths import _origin as origin

    return await origin(db, session_id)


async def export_frames(db: AsyncSession, frame_ids: list[uuid.UUID]) -> dict:
    """The lanes of one or more frames of a session as Lanelet2 OSM and OpenDRIVE, with the drawn topology.

    Each frame's lanes are placed in the world through its ego pose and the session's GNSS origin. A frame
    without a pose can only be exported alone, in its own vehicle frame, and the result says it is not
    georeferenced.
    """
    from geoalchemy2 import Geometry
    from sqlalchemy import cast, func

    from services.calibration.resolve import resolve_calibration
    from services.hdmap.bev_lanes import (
        ego_transform,
        hints_from_relations,
        lane_world_polyline,
        wkt_line,
    )
    from services.hdmap.export import to_lanelet2_osm, to_opendrive

    frames = (await db.execute(select(Frame).where(Frame.frame_id.in_(frame_ids)).order_by(Frame.ts_ns))).scalars().all()
    if not frames:
        raise LookupError("no such frames")
    sids = {f.session_id for f in frames}
    if len(sids) != 1:
        raise ValueError("export frames of one session at a time")
    sid = sids.pop()
    sess = await db.get(DbSession, sid)
    origin = await _origin(db, sid)
    poses = {f.frame_id: await _pose_at(db, sid, f.ts_ns) for f in frames}
    if len(frames) > 1 and any(p is None for p in poses.values()):
        raise ValueError("some of these frames have no ego pose; without one they cannot share a map frame. "
                         "Export a single frame instead")
    georef = origin is not None and all(p is not None for p in poses.values())

    lanes_all = (await db.execute(select(Lane).where(Lane.frame_id.in_([f.frame_id for f in frames])))).scalars().all()
    lane_ids = [ln.lane_id for ln in lanes_all]
    refs = {ln.lane_id: (ln.track_ref or ln.lane_id) for ln in lanes_all}
    rels = (await db.execute(select(LaneRelation).where(or_(
        LaneRelation.lane_id.in_(lane_ids), LaneRelation.session_id == sid)))).scalars().all() if lane_ids else []
    # Relations are drawn on one frame's lanes and hold for the same lane on every frame (the track_ref), so
    # they are restated per frame over that frame's own lane rows.
    ref_lane_ids: dict[uuid.UUID, uuid.UUID] = {}
    for r in rels:
        for lid in (r.lane_id, r.target_lane_id):
            if lid is not None and lid not in refs:
                ln = await db.get(Lane, lid)
                if ln is not None:
                    ref_lane_ids[lid] = ln.track_ref or ln.lane_id
    refs_all = {**ref_lane_ids, **refs}

    fused: list[dict] = []
    rel_dicts: list[dict] = []
    cal_cache: dict = {}
    for f in frames:
        key = (f.cam_id, f.width, f.height)
        if key not in cal_cache:
            cal_cache[key] = await resolve_calibration(sid, f.cam_id, f.width, f.height)
        cal = cal_cache[key]
        pose = poses[f.frame_id]
        t_world = ego_transform(pose, sess.sensors if sess else None, cal) if pose is not None else None
        frame_lanes = [ln for ln in lanes_all if ln.frame_id == f.frame_id]
        by_ref = {refs[ln.lane_id]: ln for ln in frame_lanes}
        for ln in frame_lanes:
            line = await asyncio.to_thread(lane_world_polyline, ln.control_points or [], cal, t_world,
                                           origin if georef else None)
            if len(line) < 2:
                continue
            kind = "stop_line" if ln.lane_type == "stop_line" else "lane"
            fused.append({"element_id": str(ln.lane_id), "kind": kind, "wkt": wkt_line(line),
                          "attrs": {"lane_type": ln.lane_type, "group": str(f.frame_id), "source": ln.source},
                          "confidence": 1.0 if ln.source == "human" else float(ln.marking_conf or 0.5)})
        for r in rels:
            ref_a = refs_all.get(r.lane_id)
            a = by_ref.get(ref_a) if ref_a is not None else None
            if a is None:
                continue
            tgt = None
            if r.target_lane_id is not None:
                ref_b = refs_all.get(r.target_lane_id)
                b = by_ref.get(ref_b) if ref_b is not None else None
                if b is None:
                    continue
                tgt = str(b.lane_id)
            rel_dicts.append({"lane_id": str(a.lane_id), "relation": r.relation, "target_lane_id": tgt,
                              "target_object_id": r.target_object_id, "target_element_id": r.target_element_id})
    hints = hints_from_relations(rel_dicts)

    # Place each signal: a mapped signal where the map has it; a signal seen only as a box in the image where
    # its lane meets the stop line (or the lane's far end), flagged approximate.
    lines = {f["element_id"]: f["wkt"] for f in fused}
    from services.hdmap.export import parse_wkt

    for reg in hints["regulatory"]:
        ref = reg.pop("signal_ref", None)
        if ref is None:
            continue
        el = None
        try:
            el = (await db.execute(select(func.ST_X(cast(MapElement.geometry, Geometry)),
                                          func.ST_Y(cast(MapElement.geometry, Geometry)))
                                   .where(MapElement.element_id == uuid.UUID(ref)))).first()
        except ValueError:
            el = None
        if el and el[0] is not None:
            reg["signal"], reg["position"] = (float(el[0]), float(el[1])), "mapped"
            continue
        anchor = lines.get(str(reg.get("stop_line_id"))) or lines.get(reg["lane_ids"][0])
        if anchor:
            _shape, coords = parse_wkt(anchor)
            pt = np.mean(coords, axis=0) if reg.get("stop_line_id") else coords[-1]
            reg["signal"], reg["position"] = (float(pt[0]), float(pt[1])), "approximate"

    if not any(f["kind"] == "lane" for f in fused):
        raise ValueError("none of these lanes has two control points on the road, so there is nothing to export")
    center = origin if (georef and origin is not None) else (0.0, 0.0)
    osm = to_lanelet2_osm(fused, hints)
    xodr = to_opendrive(fused, center, hints)
    from services.hdmap.export import _topology

    topo = _topology(fused, hints)
    return {"lanelet2": osm, "opendrive": xodr, "georeferenced": georef,
            "origin": {"lat": origin[0], "lon": origin[1]} if origin else None,
            "frames": len(frames), "boundaries": sum(1 for f in fused if f["kind"] == "lane"),
            "stop_lines": sum(1 for f in fused if f["kind"] == "stop_line"),
            "lanelets": len(topo["lanelets"]), "successors": len(topo["successors"]),
            "unpaired": topo["unpaired"][:20], "regulatory": len([r for r in hints["regulatory"] if r.get("signal")
                                                                  or r.get("stop_line_id")]),
            "hints": {"pairs": len(hints["pairs"]), "successors": len(hints["successors"])},
            "note": None if georef else ("not georeferenced: coordinates are metres in the vehicle frame of this "
                                         "frame, written about latitude 0, longitude 0")}
