"""Behavior and intent annotation (M-F.2): track-level typed intent from a closed, governed vocabulary. Intent
is temporal, so it lives on the track, not the frame. Two assist paths propose; a human always disposes:

  - trajectory-derived intent, computed from the stored Track.trajectory (centroid, area, drift, ego speed) and
    the session's traffic flow. The geometric intents (cut_in, hard_brake, u_turn, overtaking, wrong_side,
    parking; crossing, waiting, running, entering_lane) are computable and reuse the same signals as the
    Phase-1 event detector.
  - VLM-derived intent for the pose-and-context cases a trajectory cannot show (looking_at_vehicle, hesitating,
    jaywalking), duty-cycled on VRU tracks.

Every proposal lands as status "proposed" for human confirmation; an unclear case is left unknown, never
guessed. Confirmed intents feed scenario mining and the VLM dataset generation (M-F.5).
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from db.models import Frame, Object, Track
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology

log = get_logger("intelligence.intent")

# Closed, governed vocabularies (stamped with the ontology version on write). Underscored canonical values.
VRU_INTENTS = ["crossing", "waiting", "running", "jaywalking", "looking_at_vehicle", "hesitating", "entering_lane"]
VEHICLE_INTENTS = ["changing_lane", "u_turn", "parking", "overtaking", "merging", "cut_in", "hard_brake",
                   "yielding", "wrong_side"]
# The subset each assist path can propose; everything else stays unknown until a human sets it.
_TRAJECTORY_VRU = {"crossing", "waiting", "running", "entering_lane"}
_TRAJECTORY_VEHICLE = {"cut_in", "hard_brake", "u_turn", "overtaking", "wrong_side", "parking"}
_VLM_VRU = ["looking_at_vehicle", "hesitating", "jaywalking", "crossing", "waiting"]

_VEHICLE_L1 = {"two_wheeler", "three_wheeler", "four_wheeler", "heavy"}
_VRU_L1 = {"vru"}


def vocab() -> dict:
    """The governed closed vocabularies + which intents each assist path may propose."""
    return {"ontology_version": get_ontology().version, "vru": VRU_INTENTS, "vehicle": VEHICLE_INTENTS,
            "trajectory_vru": sorted(_TRAJECTORY_VRU), "trajectory_vehicle": sorted(_TRAJECTORY_VEHICLE),
            "vlm_vru": _VLM_VRU}


def _kind(class_id: int, onto) -> str | None:
    try:
        l1 = onto.by_id(class_id).l1
    except Exception:  # noqa: BLE001
        return None
    if l1 in _VEHICLE_L1:
        return "vehicle"
    if l1 in _VRU_L1:
        return "vru"
    return None


def _rec(intent: str, kind: str, source: str, conf: float, evidence: dict, onto) -> dict:
    return {"intent": intent, "kind": kind, "source": source, "status": "proposed",
            "confidence": round(float(conf), 3), "evidence": evidence, "ontology_version": onto.version}


def propose_from_trajectory(track: Track, onto, cfg, flow_sign: float = 0.0, frame_width: float = 1920.0) -> list[dict]:
    """Geometric intents from the stored trajectory. Returns proposed records; an unclear track returns [].
    cx is stored in pixels, so it is normalized by the frame width to a [0,1] lateral position."""
    kind = _kind(track.class_id, onto)
    traj = track.trajectory or {}
    pts = traj.get("points") or []
    s = traj.get("summary") or {}
    if kind is None or len(pts) < 3 or not s:
        return []
    e = cfg.intelligence.events
    out: list[dict] = []
    fw = frame_width if frame_width and frame_width > 1 else 1920.0
    cxs = [p.get("cx", 0.0) / fw for p in pts]     # normalized lateral position in [0,1]
    last_cx = cxs[-1]
    center_dist = abs(last_cx - 0.5)
    drift = float(s.get("x_drift_frac", 0.0))
    growth = float(s.get("area_growth", 1.0))
    mean_speed = float(s.get("mean_speed_px", 0.0))
    net_disp = float(s.get("net_disp_px", 0.0))
    approaching = bool(s.get("approaching"))
    n = int(s.get("n", len(pts)))

    if kind == "vehicle":
        if growth >= e.cut_in_area_growth and center_dist < e.cut_in_center_frac:
            out.append(_rec("cut_in", kind, "trajectory", min(1.0, 0.4 + (growth - 1.0)),
                            {"area_growth": round(growth, 3), "center_dist": round(center_dist, 3)}, onto))
        # ego hard-brake while this vehicle was tracked (a real deceleration in the ego speed series)
        hb = _ego_hard_brake(pts, e.hard_brake_decel)
        if hb is not None:
            out.append(_rec("hard_brake", kind, "trajectory", min(1.0, abs(hb) / 8.0),
                            {"decel_mps2": round(hb, 2)}, onto))
        if _is_uturn(cxs):
            out.append(_rec("u_turn", kind, "trajectory", 0.6, {"lateral_reversal": True}, onto))
        if abs(drift) > 0.12 and growth < e.cut_in_area_growth and center_dist > 0.25:
            out.append(_rec("overtaking", kind, "trajectory", min(1.0, 0.4 + abs(drift)),
                            {"x_drift_frac": round(drift, 3)}, onto))
        if flow_sign != 0 and abs(drift) > 0.05 and (1.0 if drift > 0 else -1.0) == -flow_sign and n >= e.wrong_side_frames:
            out.append(_rec("wrong_side", kind, "trajectory", min(1.0, 0.5 + abs(drift)),
                            {"x_drift_frac": round(drift, 3), "against_flow": True}, onto))
        static = net_disp < e.static_disp_frac and n >= e.static_min_frames
        on_shoulder = last_cx < e.shoulder_margin_frac or last_cx > 1 - e.shoulder_margin_frac
        if static and on_shoulder:
            out.append(_rec("parking", kind, "trajectory", 0.5, {"static": True, "on_shoulder": True}, onto))
    else:  # vru
        if abs(drift) > 0.10:
            out.append(_rec("crossing", kind, "trajectory", min(1.0, 0.4 + abs(drift)),
                            {"x_drift_frac": round(drift, 3)}, onto))
        if mean_speed > 6.0:
            out.append(_rec("running", kind, "trajectory", min(1.0, mean_speed / 12.0),
                            {"mean_speed_px": round(mean_speed, 2)}, onto))
        elif net_disp < e.static_disp_frac and mean_speed < 1.0:
            out.append(_rec("waiting", kind, "trajectory", 0.5, {"static": True}, onto))
        if approaching and center_dist < e.cut_in_center_frac:
            out.append(_rec("entering_lane", kind, "trajectory", 0.5,
                            {"approaching": True, "center_dist": round(center_dist, 3)}, onto))
    return out


def _ego_hard_brake(pts: list[dict], decel_thresh: float) -> float | None:
    """The sharpest ego deceleration (m/s^2) over the track's ego-speed series, or None if never hard."""
    worst = 0.0
    prev = None
    for p in pts:
        sp = p.get("ego_speed")
        ts = p.get("ts_ns")
        if not isinstance(sp, (int, float)) or ts is None:
            prev = None
            continue
        if prev is not None:
            dt = (ts - prev[1]) / 1e9
            if dt > 1e-3:
                a = (sp - prev[0]) / dt
                if a < worst:
                    worst = a
        prev = (sp, ts)
    return worst if worst <= decel_thresh else None


def _is_uturn(cxs: list[float]) -> bool:
    """A lateral sweep that reverses direction with real amplitude (a coarse u-turn proxy from a forward cam)."""
    if len(cxs) < 6:
        return False
    lo, hi = min(cxs), max(cxs)
    if (hi - lo) < 0.35 * (max(abs(hi), abs(lo), 1.0)):
        return False
    i_max = cxs.index(hi)
    return 1 < i_max < len(cxs) - 2  # peak in the interior => went one way then came back


async def _frame_width(db: AsyncSession, session_id: UUID) -> float:
    w = (await db.execute(select(Frame.width).where(Frame.session_id == session_id).limit(1))).scalar_one_or_none()
    return float(w) if w else 1920.0


async def _session_flow_sign(db: AsyncSession, session_id: UUID, onto) -> float:
    """The majority lateral drift of the session's vehicle tracks, for wrong-side detection."""
    rows = (await db.execute(select(Track.class_id, Track.trajectory).where(Track.session_id == session_id))).all()
    drifts = []
    for cid, traj in rows:
        if _kind(cid, onto) == "vehicle" and traj:
            d = float((traj.get("summary") or {}).get("x_drift_frac", 0.0))
            if abs(d) > 0.02:
                drifts.append(d)
    if not drifts:
        return 0.0
    return 1.0 if sum(1 for d in drifts if d > 0) >= len(drifts) / 2 else -1.0


def _merge(existing: list[dict], proposals: list[dict]) -> list[dict]:
    """Add proposals that are not already present (by intent+source), keeping human/confirmed entries intact."""
    have = {(r.get("intent"), r.get("source")) for r in existing}
    return existing + [p for p in proposals if (p["intent"], p["source"]) not in have]


async def propose_track(track_id: UUID, db: AsyncSession | None = None) -> dict:
    """Propose trajectory-derived intents for one track (idempotent: re-running does not duplicate)."""
    own = db is None
    maker = get_sessionmaker()
    db = db or maker()
    try:
        t = await db.get(Track, track_id)
        if t is None:
            return {"error": "track not found"}
        onto = get_ontology()
        flow = await _session_flow_sign(db, t.session_id, onto)
        fw = await _frame_width(db, t.session_id)
        proposals = propose_from_trajectory(t, onto, get_settings(), flow, fw)
        t.intents = _merge(list(t.intents or []), proposals)
        await db.commit()
        return {"track_id": str(track_id), "proposed": [p["intent"] for p in proposals], "intents": t.intents}
    finally:
        if own:
            await db.close()


async def propose_session(session_id: UUID) -> dict:
    """Trajectory-derived intent proposals across every track in a session."""
    maker = get_sessionmaker()
    onto = get_ontology()
    async with maker() as db:
        flow = await _session_flow_sign(db, session_id, onto)
        fw = await _frame_width(db, session_id)
        tracks = (await db.execute(select(Track).where(Track.session_id == session_id))).scalars().all()
        n_prop = 0
        for t in tracks:
            proposals = propose_from_trajectory(t, onto, get_settings(), flow, fw)
            if proposals:
                t.intents = _merge(list(t.intents or []), proposals)
                n_prop += len(proposals)
        await db.commit()
    log.info("intent.session_proposed", session_id=str(session_id), tracks=len(tracks), proposals=n_prop)
    return {"session_id": str(session_id), "tracks": len(tracks), "proposals": n_prop}


async def propose_vlm(track_id: UUID, max_frames: int = 3) -> dict:
    """VLM-derived intent for a VRU track: the pose-and-context cases a trajectory cannot show. Unclear -> no
    proposal (never guessed)."""
    import cv2
    import numpy as np

    from core.storage import get_object_store
    from services.autolabel.paths.path_c_qwen3vl import OllamaVlmClient, crop_object

    maker = get_sessionmaker()
    onto = get_ontology()
    async with maker() as db:
        t = await db.get(Track, track_id)
        if t is None:
            return {"error": "track not found"}
        if _kind(t.class_id, onto) != "vru":
            return {"track_id": str(track_id), "proposed": None, "reason": "VLM intent is for VRU tracks only"}
        rows = (await db.execute(
            select(Object.bbox, Frame.img_uri).join(Frame, Frame.frame_id == Object.frame_id)
            .where(Object.track_id == track_id).order_by(Frame.ts_ns))).all()
        if not rows:
            return {"track_id": str(track_id), "proposed": None, "reason": "no frames"}
        picks = rows[:: max(1, len(rows) // max_frames)][:max_frames]

    vlm = OllamaVlmClient()
    votes: dict[str, int] = {}
    for bbox, img_uri in picks:
        try:
            buf = np.frombuffer(get_object_store().get_bytes(img_uri), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                continue
            res = vlm.verify(crop_object(img, tuple(float(x) for x in bbox), 0.25), [*_VLM_VRU, "unknown"], {})
            name = (res.class_name or "").strip().lower().replace("-", "_").replace(" ", "_")
            if name in _VLM_VRU:
                votes[name] = votes.get(name, 0) + 1
        except Exception as exc:  # noqa: BLE001
            log.info("intent.vlm_frame_failed", error=str(exc))
    if not votes:
        return {"track_id": str(track_id), "proposed": None, "reason": "unclear (left unknown)"}
    intent, v = max(votes.items(), key=lambda kv: kv[1])
    async with maker() as db:
        t = await db.get(Track, track_id)
        rec = _rec(intent, "vru", "vlm", v / len(picks), {"votes": votes, "n_frames": len(picks)}, onto)
        t.intents = _merge(list(t.intents or []), [rec])
        await db.commit()
    log.info("intent.vlm_proposed", track_id=str(track_id), intent=intent)
    return {"track_id": str(track_id), "proposed": intent, "confidence": round(v / len(picks), 3)}


async def set_intent(track_id: UUID, intent: str, kind: str, status: str = "confirmed") -> dict:
    """Human sets or confirms a track's intent from the closed vocabulary. Validates against the vocab; an
    explicit 'unknown' is allowed and clears any proposal for that kind."""
    vocab_for = VEHICLE_INTENTS if kind == "vehicle" else VRU_INTENTS
    if intent != "unknown" and intent not in vocab_for:
        return {"error": f"intent '{intent}' not in the {kind} vocabulary"}
    maker = get_sessionmaker()
    onto = get_ontology()
    async with maker() as db:
        t = await db.get(Track, track_id)
        if t is None:
            return {"error": "track not found"}
        # drop any prior human record of this kind, keep machine proposals for context
        kept = [r for r in (t.intents or []) if not (r.get("source") == "human" and r.get("kind") == kind)]
        if intent != "unknown":
            kept.append({"intent": intent, "kind": kind, "source": "human", "status": status,
                         "confidence": 1.0, "evidence": {}, "ontology_version": onto.version})
        # mark a matching machine proposal confirmed
        for r in kept:
            if r.get("intent") == intent and r.get("source") != "human":
                r["status"] = "confirmed"
        t.intents = kept
        await db.commit()
        return {"track_id": str(track_id), "intents": t.intents}
