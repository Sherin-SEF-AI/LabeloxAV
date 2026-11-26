"""Paths and intent (#8, /trajectories): where the ego went, where other road users went and will go, and what
they meant to do.

Every position here comes from one of two places, and the response always says which:

- **world**: the session-local ENU frame of `ego_pose` (metres east and north of the first GNSS fix). A road
  user's position is its box's ground contact lifted to the road through the resolved calibration, then placed
  in the world through the ego pose at that frame. Ego motion is removed, so a parked car stays put.
- **ego**: the vehicle frame at each instant (x forward, y left), used when the session has no poses. The ego's
  own motion is still in these numbers: a parked car appears to drive backwards past a moving ego. The page
  says so rather than pretending otherwise.

Ground contact is the bottom centre of the 2D box on a flat road (the monocular IPM every dynamics number in
this repo uses), so lateral and longitudinal error grow with distance.

The future-path seed is a constant-turn-rate-and-velocity roll-out (`services/temporal/trajectory.py`), labelled
a kinematic baseline everywhere it appears. There is no learned forecaster in this repository.

Labels are stored beside the measurement: `path_label` for an edited ego path or a drawn future, `track_event` for
an intent span, mirrored onto `Track.intents` for the governed intent words so the older readers see it too.
`ego_pose` is never modified.
"""

from __future__ import annotations

import bisect
import math
import uuid
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import EgoPose, Frame, Object, OntologyClass, PathLabel, Track, TrackEvent
from db.models import Session as DbSession

log = get_logger("driving_paths")

POSE_MATCH_NS = 60_000_000
MIN_OBS = 3
# The intent words the page offers first, in the order the brief named them. Each is offered only if the
# session's pack vocabulary has it; the full vocabulary is always available below them.
QUICK_INTENTS = ("cut_in", "yielding", "jaywalking", "overtaking", "turning_left", "turning_right", "u_turn",
                 "stopping", "stopping_in_live_lane")
BASELINE_NOTE = ("Kinematic baseline: constant turn rate and velocity rolled out from the last observed motion. "
                 "There is no learned forecaster; drag the points to where the road user actually goes.")


def _movable(onto, class_id: int) -> bool:
    from services.export.scenario_build import MOVABLE_EXTRA_NAMES, MOVABLE_L1

    try:
        c = onto.by_id(class_id)
    except KeyError:
        return False
    return c.l1 in MOVABLE_L1 or c.name in MOVABLE_EXTRA_NAMES


async def _origin(db: AsyncSession, session_id: uuid.UUID) -> tuple[float, float] | None:
    """The session's ENU origin (lat, lon): its first GNSS fix, as `services/intelligence/ego_pose` defines it."""
    from geoalchemy2 import Geometry
    from sqlalchemy import cast

    geom = cast(Frame.gnss, Geometry)
    row = (await db.execute(select(func.ST_Y(geom), func.ST_X(geom)).where(
        Frame.session_id == session_id, Frame.gnss.isnot(None)).order_by(Frame.ts_ns).limit(1))).first()
    return (float(row[0]), float(row[1])) if row and row[0] is not None else None


class PoseIndex:
    """The session's ego poses, matched to an instant within POSE_MATCH_NS (no interpolation: a pose between two
    rows would be a guess)."""

    def __init__(self, poses: list[EgoPose]):
        self.poses = sorted(poses, key=lambda p: p.ts_ns)
        self.ts = [p.ts_ns for p in self.poses]

    def at(self, ts_ns: int) -> EgoPose | None:
        if not self.ts:
            return None
        i = bisect.bisect_left(self.ts, ts_ns)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(self.ts) and abs(self.ts[j] - ts_ns) <= POSE_MATCH_NS:
                if best is None or abs(self.ts[j] - ts_ns) < abs(best.ts_ns - ts_ns):
                    best = self.poses[j]
        return best


async def _context(db: AsyncSession, session_id: uuid.UUID) -> dict:
    sess = await db.get(DbSession, session_id)
    if sess is None:
        raise LookupError("session not found")
    poses = (await db.execute(select(EgoPose).where(EgoPose.session_id == session_id))).scalars().all()
    return {"session": sess, "poses": PoseIndex(list(poses)), "origin": await _origin(db, session_id), "cals": {}}


async def _cal(ctx: dict, session_id, cam_id: str, w: int, h: int):
    from services.calibration.resolve import resolve_calibration

    key = (cam_id, w, h)
    if key not in ctx["cals"]:
        ctx["cals"][key] = await resolve_calibration(session_id, cam_id, w, h)
    return ctx["cals"][key]


def _far_limit(ctx: dict, cal) -> float:
    from services.hdmap.bev_view import view_for

    key = ("far", cal.cam_id, round(cal.fx, 3), round(cal.cy, 3))
    if key not in ctx["cals"]:
        ctx["cals"][key] = view_for(cal).far_m
    return ctx["cals"][key]


def _world(ctx: dict, cal, ts_ns: int, ground: list[float]) -> list[float] | None:
    from services.hdmap.bev_lanes import ego_transform, transfer

    pose = ctx["poses"].at(ts_ns)
    if pose is None:
        return None
    t = ego_transform(pose, ctx["session"].sensors, cal)
    return transfer([ground], t, np.eye(4))[0]


def _ego_point(cal, ground: list[float]) -> list[float]:
    """Camera-ground (forward, left) to the vehicle frame (adds the mount's x, y)."""
    return [round(ground[0] + float(cal.xyz_m[0]), 4), round(ground[1] + float(cal.xyz_m[1]), 4)]


async def observed_paths(db: AsyncSession, session_id: uuid.UUID, ctx: dict | None = None,
                         track_ids: list[uuid.UUID] | None = None) -> dict:
    """Every movable track's observed ground path, plus the ego path, in one coordinate frame."""
    from services.autolabel.ontology import get_ontology
    from services.hdmap.bev_lanes import image_to_ground

    ctx = ctx or await _context(db, session_id)
    onto = get_ontology()
    q = (select(Object.object_id, Object.track_id, Object.bbox, Object.class_id, Frame.frame_id, Frame.ts_ns,
                Frame.cam_id, Frame.width, Frame.height)
         .join(Frame, Frame.frame_id == Object.frame_id)
         .where(Frame.session_id == session_id, Object.track_id.isnot(None)))
    if track_ids:
        q = q.where(Object.track_id.in_(track_ids))
    rows = (await db.execute(q.order_by(Object.track_id, Frame.ts_ns))).all()
    tq = select(Track).where(Track.session_id == session_id)
    if track_ids:
        tq = tq.where(Track.track_id.in_(track_ids))
    tracks = {t.track_id: t for t in (await db.execute(tq)).scalars().all()}
    names: dict[int, str] = {int(i): str(n) for i, n in (await db.execute(select(OntologyClass.id, OntologyClass.name))).all()}

    world_ok = bool(ctx["poses"].ts)
    out: dict[str, dict] = {}
    for oid, tid, bbox, _cid, fid, ts, cam, w, h in rows:
        tr = tracks.get(tid)
        if tr is None or not _movable(onto, tr.class_id):
            continue
        if not bbox or len(bbox) < 4:
            continue
        cal = await _cal(ctx, session_id, cam, int(w), int(h))
        g = image_to_ground([[(bbox[0] + bbox[2]) / 2.0, bbox[3]]], cal)
        if not g:
            continue
        if g[0][0] > _far_limit(ctx, cal):
            # Beyond the range a flat-road lift of this camera supports (bev_view.view_for): a box bottom a few
            # pixels under the horizon is worth tens of metres there, and a path through such points has speeds
            # no road user drives at. Counted, not drawn.
            ctx["beyond_range"] = ctx.get("beyond_range", 0) + 1
            continue
        rec = out.setdefault(str(tid), {"track_id": str(tid), "class_id": tr.class_id,
                                        "class_name": names.get(tr.class_id, str(tr.class_id)),
                                        "intents": tr.intents or [], "points": []})
        ego = _ego_point(cal, g[0])
        world = _world(ctx, cal, int(ts), g[0]) if world_ok else None
        rec["points"].append({"frame_id": str(fid), "object_id": str(oid), "ts_ns": int(ts), "ego": ego,
                              "world": world})

    # A track is drawn in the world only if every one of its points has a pose; a path half in one frame and
    # half in another is not a path.
    coord = "world" if world_ok else "ego"
    keep = {}
    for k, rec in out.items():
        if len(rec["points"]) < MIN_OBS:
            continue
        if coord == "world" and any(p["world"] is None for p in rec["points"]):
            continue
        keep[k] = rec
    return {"coord_frame": coord, "tracks": keep, "ctx": ctx, "beyond_range": int(ctx.get("beyond_range", 0))}


def _xy(p: dict, coord: str) -> list[float]:
    return p["world"] if coord == "world" else p["ego"]


def ego_path(ctx: dict) -> list[dict]:
    from services.intelligence.ego_pose import quat_yaw

    return [{"ts_ns": int(p.ts_ns), "frame_id": str(p.frame_id) if p.frame_id else None,
             "x": round(float(p.x), 4), "y": round(float(p.y), 4), "yaw": round(quat_yaw(p.qw, p.qz), 5),
             "source": p.source, "measured": bool(p.measured)} for p in ctx["poses"].poses]


def _path_out(p: PathLabel) -> dict:
    return {"path_id": str(p.path_id), "kind": p.kind, "track_id": str(p.track_id) if p.track_id else None,
            "anchor_frame_id": str(p.anchor_frame_id) if p.anchor_frame_id else None,
            "anchor_ts_ns": int(p.anchor_ts_ns), "coord_frame": p.coord_frame, "points": p.points or [],
            "source": p.source, "seed": p.seed or {}, "notes": p.notes,
            "updated_at": p.updated_at.isoformat() if p.updated_at else None}


async def session_view(db: AsyncSession, session_id: uuid.UUID) -> dict:
    """The page's whole model of one session: ego path, every movable track's path, labels and event counts."""
    obs = await observed_paths(db, session_id)
    ctx, coord = obs["ctx"], obs["coord_frame"]
    labels = (await db.execute(select(PathLabel).where(PathLabel.session_id == session_id))).scalars().all()
    counts: dict[uuid.UUID, int] = {tid: int(n) for tid, n in (await db.execute(
        select(TrackEvent.track_id, func.count()).join(Track, Track.track_id == TrackEvent.track_id)
        .where(Track.session_id == session_id).group_by(TrackEvent.track_id))).all()}
    t0 = min([p.ts_ns for p in ctx["poses"].poses] + [pt["ts_ns"] for r in obs["tracks"].values()
                                                      for pt in r["points"][:1]], default=0)
    tracks = []
    for k, rec in obs["tracks"].items():
        pts = rec["points"]
        tracks.append({**{x: rec[x] for x in ("track_id", "class_id", "class_name", "intents")},
                       "n": len(pts), "first_ts_ns": pts[0]["ts_ns"], "last_ts_ns": pts[-1]["ts_ns"],
                       "path": [[round((p["ts_ns"] - t0) / 1e9, 3), *_xy(p, coord)] for p in pts],
                       "frames": [p["frame_id"] for p in pts],
                       "events": int(counts.get(uuid.UUID(k), 0)),
                       "futures": [_path_out(lb) for lb in labels if lb.track_id and str(lb.track_id) == k]})
    tracks.sort(key=lambda t: -t["n"])
    ego_label = next((lb for lb in labels if lb.kind == "ego_path"), None)
    s = ctx["session"]
    return {
        "session_id": str(session_id), "route": s.route, "vehicle_id": s.vehicle_id, "city": s.city,
        "coord_frame": coord, "t0_ns": int(t0), "origin": ctx["origin"],
        "frame_note": ("world: metres east and north of the session's first GNSS fix, ego motion removed"
                       if coord == "world" else
                       "ego: metres from the vehicle at each instant (x forward, y left). This session has no ego "
                       "poses, so the ego's own motion is still in these paths."),
        "ego": {"poses": ego_path(ctx), "label": _path_out(ego_label) if ego_label else None},
        "tracks": tracks,
        "positions_beyond_range": obs["beyond_range"],
        "calibration": sorted({c.source for c in ctx["cals"].values() if hasattr(c, "source")}),
        "baseline_note": BASELINE_NOTE,
    }


async def forecast(db: AsyncSession, track_id: uuid.UUID, anchor_frame_id: uuid.UUID | None = None,
                   horizon_s: float = 3.0, dt_s: float = 0.25) -> dict:
    """The CTRV baseline from an anchor frame of a track, and how it compares to what the track then did."""
    from services.temporal.trajectory import baseline_error, ctrv_forecast, estimate_ctrv_state

    tr = await db.get(Track, track_id)
    if tr is None:
        raise LookupError("track not found")
    obs = await observed_paths(db, tr.session_id, track_ids=[track_id])
    rec = obs["tracks"].get(str(track_id))
    if rec is None:
        raise ValueError(f"this track has fewer than {MIN_OBS} positions on the road within the range this camera's "
                         f"flat-road lift supports ({obs['beyond_range']} were further out), so it has no path")
    coord = obs["coord_frame"]
    pts = rec["points"]
    ai = len(pts) - 1
    if anchor_frame_id is not None:
        found = [i for i, p in enumerate(pts) if p["frame_id"] == str(anchor_frame_id)]
        if not found:
            raise ValueError("the anchor frame is not one of this track's frames")
        ai = found[0]
    if ai < 1:
        raise ValueError("the anchor needs at least one earlier position to measure motion from")
    a_ts = pts[ai]["ts_ns"]
    hist = pts[: ai + 1]
    state = estimate_ctrv_state([(p["ts_ns"] - a_ts) / 1e9 for p in hist], [_xy(p, coord) for p in hist])
    if state is None:
        raise ValueError("the positions before the anchor do not span any time")
    fc = ctrv_forecast(state, horizon_s=horizon_s, dt_s=dt_s)
    after = [[round((p["ts_ns"] - a_ts) / 1e9, 3), *_xy(p, coord)] for p in pts[ai + 1:]]
    return {"track_id": str(track_id), "anchor_frame_id": pts[ai]["frame_id"], "anchor_ts_ns": a_ts,
            "coord_frame": coord, "state": {k: round(v, 4) for k, v in state.items()},
            "forecast": fc, "observed_after": after, "error_vs_observed": baseline_error(fc, after),
            "model": "ctrv", "note": BASELINE_NOTE}


async def save_future(db: AsyncSession, track_id: uuid.UUID, anchor_frame_id: uuid.UUID, points: list,
                      source: str, seed: dict | None, user_id, notes: str | None = None) -> dict:
    tr = await db.get(Track, track_id)
    if tr is None:
        raise LookupError("track not found")
    fr = await db.get(Frame, anchor_frame_id)
    if fr is None or fr.session_id != tr.session_id:
        raise ValueError("the anchor frame must be a frame of the track's session")
    pts = _clean_points(points)
    if len(pts) < 2:
        raise ValueError("a future path needs at least two points")
    coord = "world" if (await db.execute(select(func.count()).select_from(EgoPose).where(
        EgoPose.session_id == tr.session_id))).scalar() else "ego"
    row = (await db.execute(select(PathLabel).where(PathLabel.track_id == track_id, PathLabel.kind == "future",
                                                    PathLabel.anchor_ts_ns == fr.ts_ns))).scalar_one_or_none()
    if row is None:
        row = PathLabel(session_id=tr.session_id, track_id=track_id, kind="future", anchor_ts_ns=fr.ts_ns)
        db.add(row)
    row.anchor_frame_id, row.coord_frame, row.points, row.source = fr.frame_id, coord, pts, source
    row.seed, row.notes, row.created_by = seed or {}, notes, user_id
    row.updated_at = datetime.now(UTC)
    await db.commit()
    return _path_out(row)


async def save_ego_path(db: AsyncSession, session_id: uuid.UUID, points: list, source: str, user_id,
                        notes: str | None = None) -> dict:
    ctx = await _context(db, session_id)
    if not ctx["poses"].ts:
        raise ValueError("this session has no ego poses, so there is no ego path to correct")
    pts = _clean_points(points)
    if len(pts) < 2:
        raise ValueError("an ego path needs at least two points")
    t0 = ctx["poses"].ts[0]
    row = (await db.execute(select(PathLabel).where(PathLabel.session_id == session_id,
                                                    PathLabel.kind == "ego_path"))).scalar_one_or_none()
    if row is None:
        row = PathLabel(session_id=session_id, kind="ego_path", anchor_ts_ns=t0, coord_frame="world")
        db.add(row)
    row.points, row.source, row.notes, row.created_by = pts, source, notes, user_id
    row.anchor_frame_id = ctx["poses"].poses[0].frame_id
    row.seed = {"from": "ego_pose", "poses": len(ctx["poses"].ts)}
    row.updated_at = datetime.now(UTC)
    await db.commit()
    return _path_out(row)


def smooth_ego(poses: list[dict], window: int = 5) -> list[list[float]]:
    """The measured ego path through `smooth_path`, as [[t, x, y], ...] on the poses' own time origin."""
    from services.temporal.trajectory import smooth_path

    if not poses:
        return []
    t0 = poses[0]["ts_ns"]
    xy = smooth_path([[p["x"], p["y"]] for p in poses], window)
    return [[round((p["ts_ns"] - t0) / 1e9, 3), q[0], q[1]] for p, q in zip(poses, xy, strict=True)]


def _clean_points(points: list) -> list[list[float]]:
    out = []
    for p in points or []:
        if not isinstance(p, list | tuple) or len(p) < 3:
            continue
        t, x, y = (float(v) for v in p[:3])
        if not all(math.isfinite(v) for v in (t, x, y)):
            continue
        out.append([round(t, 4), round(x, 4), round(y, 4)])
    out.sort(key=lambda q: q[0])
    return out


async def add_intent(db: AsyncSession, track_id: uuid.UUID, event_type: str, start_frame_id: uuid.UUID,
                     end_frame_id: uuid.UUID, *, state: str, user, notes: str | None = None) -> dict:
    """An intent span on a track: a TrackEvent, mirrored onto Track.intents when the word is a governed intent."""
    from services.autolabel.ontology import get_ontology
    from services.domain import pack_id_for_session, validate_track_event_type
    from services.intelligence.intent import VEHICLE_INTENTS, VRU_INTENTS, _kind

    tr = await db.get(Track, track_id)
    if tr is None:
        raise LookupError("track not found")
    onto = get_ontology()
    try:
        l1 = onto.by_id(tr.class_id).l1
    except KeyError:
        l1 = None
    errors = validate_track_event_type(event_type, l1, await pack_id_for_session(db, tr.session_id))
    if errors:
        raise ValueError("; ".join(errors))
    frames = {f.frame_id: f for f in (await db.execute(
        select(Frame).where(Frame.frame_id.in_({start_frame_id, end_frame_id})))).scalars()}
    if len(frames) != len({start_frame_id, end_frame_id}) or any(f.session_id != tr.session_id
                                                                 for f in frames.values()):
        raise ValueError("the span's frames must be frames of the track's session")
    a, b = frames[start_frame_id], frames[end_frame_id]
    if a.ts_ns > b.ts_ns:
        a, b = b, a
    ev = TrackEvent(track_id=track_id, event_type=event_type, start_frame_id=a.frame_id, end_frame_id=b.frame_id,
                    start_ts_ns=a.ts_ns, end_ts_ns=b.ts_ns, source="human", state=state, notes=notes,
                    created_by=getattr(user, "name", None), ontology_version=onto.version,
                    evidence={"tool": "trajectories"})
    db.add(ev)
    kind = _kind(tr.class_id, onto)
    governed = (kind == "vehicle" and event_type in VEHICLE_INTENTS) or (kind == "vru" and event_type in VRU_INTENTS)
    if governed:
        intents = [i for i in (tr.intents or []) if not (i.get("intent") == event_type and i.get("source") == "human")]
        intents.append({"intent": event_type, "kind": kind, "source": "human",
                        "status": "confirmed" if state == "accepted" else "proposed", "confidence": 1.0,
                        "evidence": {"track_event": True}, "ontology_version": onto.version})
        tr.intents = intents
    await db.flush()
    out = {"event_id": str(ev.event_id), "track_id": str(track_id), "event_type": event_type, "state": state,
           "start_frame_id": str(a.frame_id), "end_frame_id": str(b.frame_id), "start_ts_ns": a.ts_ns,
           "end_ts_ns": b.ts_ns, "mirrored_to_intents": governed}
    await db.commit()
    return out


async def list_intents(db: AsyncSession, track_id: uuid.UUID) -> dict:
    from services.autolabel.ontology import get_ontology
    from services.domain import pack_id_for_session, track_event_spec, validate_track_event_type

    tr = await db.get(Track, track_id)
    if tr is None:
        raise LookupError("track not found")
    try:
        l1 = get_ontology().by_id(tr.class_id).l1
    except KeyError:
        l1 = None
    pack = await pack_id_for_session(db, tr.session_id)
    spec = track_event_spec(pack)
    types = [] if spec is None else [
        {"name": t.name, "definition": t.definition, "applies_to": t.applies_to,
         "applicable": not validate_track_event_type(t.name, l1, pack)} for t in spec.types]
    names = {t["name"] for t in types if t["applicable"]}
    rows = (await db.execute(select(TrackEvent).where(TrackEvent.track_id == track_id)
                             .order_by(TrackEvent.start_ts_ns))).scalars().all()
    return {"track_id": str(track_id), "class_l1": l1, "intents": tr.intents or [],
            "quick": [q for q in QUICK_INTENTS if q in names],
            "missing_quick": [q for q in ("turning_left", "turning_right", "stopping") if q not in names],
            "event_types": types,
            "events": [{"event_id": str(e.event_id), "event_type": e.event_type, "state": e.state,
                        "source": e.source, "start_frame_id": str(e.start_frame_id),
                        "end_frame_id": str(e.end_frame_id), "start_ts_ns": e.start_ts_ns,
                        "end_ts_ns": e.end_ts_ns, "notes": e.notes} for e in rows]}


async def scenario_export(db: AsyncSession, session_id: uuid.UUID) -> dict:
    """The labelled paths and intents of a session as a documented JSON scenario plus an OpenSCENARIO replay.

    JSON (`labeloxav.paths_intents/1`): coord_frame, origin, t0_ns, ego {source, points [[t, x, y]]}, actors
    [{track_id, class_name, observed [[t, x, y]], frames, futures [{anchor_t, source, points [[t, x, y]]}],
    intents, events [{type, start_t, end_t, state, source}]}]. Times in seconds after t0_ns; metres in
    coord_frame. The OpenSCENARIO document replays the observed paths (FollowTrajectoryAction, positions as
    observed) through services/export/adapter_openscenario.py; futures and intents ride in the JSON, because
    OpenSCENARIO has no element for a label about what an actor meant to do.
    """
    from services.export.adapter_openscenario import Actor, TrajectoryPoint, build_scenario

    view = await session_view(db, session_id)
    t0 = view["t0_ns"]
    events = (await db.execute(select(TrackEvent).join(Track, Track.track_id == TrackEvent.track_id)
                               .where(Track.session_id == session_id))).scalars().all()
    by_track: dict[str, list] = {}
    for e in events:
        by_track.setdefault(str(e.track_id), []).append(
            {"type": e.event_type, "start_t": round((e.start_ts_ns - t0) / 1e9, 3),
             "end_t": round((e.end_ts_ns - t0) / 1e9, 3), "state": e.state, "source": e.source})
    ego_label = view["ego"]["label"]
    poses = view["ego"]["poses"]
    if ego_label:
        a = (ego_label["anchor_ts_ns"] - t0) / 1e9
        ego_pts = [[round(p[0] + a, 3), p[1], p[2]] for p in ego_label["points"]]
        ego_src = "path_label"
    elif poses:
        ego_pts = [[round((p["ts_ns"] - t0) / 1e9, 3), p["x"], p["y"]] for p in poses]
        ego_src = "ego_pose"
    else:
        ego_pts, ego_src = [], "none"
    actors = []
    for t in view["tracks"]:
        futures = []
        for f in t["futures"]:
            a = (f["anchor_ts_ns"] - t0) / 1e9
            futures.append({"anchor_t": round(a, 3), "source": f["source"],
                            "points": [[round(p[0] + a, 3), p[1], p[2]] for p in f["points"]]})
        actors.append({"track_id": t["track_id"], "class_name": t["class_name"], "observed": t["path"],
                       "frames": t["frames"], "futures": futures, "intents": t["intents"],
                       "events": by_track.get(t["track_id"], [])})
    doc = {"format": "labeloxav.paths_intents/1", "session_id": str(session_id), "coord_frame": view["coord_frame"],
           "frame_note": view["frame_note"], "origin": ({"lat": view["origin"][0], "lon": view["origin"][1]}
                                                        if view["origin"] else None),
           "t0_ns": t0, "ego": {"source": ego_src, "points": ego_pts}, "actors": actors,
           "baseline": BASELINE_NOTE}

    def traj(pts: list) -> list:
        out = []
        for i, p in enumerate(pts):
            a, b = (p, pts[i + 1]) if i + 1 < len(pts) else (pts[i - 1], p) if i else (p, p)
            h = math.atan2(b[2] - a[2], b[1] - a[1]) if a is not b else 0.0
            out.append(TrajectoryPoint(t_s=max(0.0, float(p[0])), x=float(p[1]), y=float(p[2]), heading_rad=h))
        return out

    osc_actors = [Actor(name=f"actor_{a['track_id'][:8]}", class_name=a["class_name"], track_id=a["track_id"],
                        points=traj(a["observed"])) for a in actors if len(a["observed"]) >= MIN_OBS]
    ego = Actor(name="Ego", class_name="sedan", points=traj(ego_pts)) if len(ego_pts) >= 2 else None
    xml = build_scenario(name=f"session_{str(session_id)[:8]}", actors=osc_actors, ego=ego,
                         description=(f"Labelled paths and intents, {len(osc_actors)} actor(s), positions in the "
                                      f"{view['coord_frame']} frame."),
                         date=datetime.fromtimestamp(t0 / 1e9, tz=UTC).isoformat() if t0 else "1970-01-01T00:00:00")
    log.info("paths.scenario_export", session=str(session_id), actors=len(actors), events=len(events))
    return {"json": doc, "openscenario": xml, "actors": len(actors), "events": len(events)}
