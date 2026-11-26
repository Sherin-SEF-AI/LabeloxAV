"""The database side of camera-LiDAR alignment (#9): which frames have a scan, the calibration in force, scoring,
the refinement job's body, and saving a manual calibration beside the dataset's.

A saved alignment is a `camera_calibration` row with `source = 'manual'` next to the row it was aligned from,
which is never modified (migration 0131 allows one manual row per camera beside the others). Deleting the manual
row restores the dataset calibration exactly. Every save and every refinement is also a `calibration_alignment`
row with the scores before and after, so the calibration in force can be traced to the evidence it was saved on.
"""

from __future__ import annotations

import asyncio
import base64
import uuid

import cv2
import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.storage import get_object_store
from db.models import CalibrationAlignment, CameraCalibration, Frame, PointCloud
from db.models import Session as DbSession

log = get_logger("calib_align")

PAIR_MATCH_NS = 60_000_000
MANUAL_QUALITY = 0.85
# The alignment page draws points the browser projects itself; these bound what it is sent.
VIEW_MAX_POINTS = 40000


async def lidar_sessions(db: AsyncSession) -> list[dict]:
    """Sessions with LiDAR scans and camera frames at the same instants, most scans first."""
    rows = (await db.execute(
        select(PointCloud.session_id, func.count(PointCloud.cloud_id))
        .where(PointCloud.source == "lidar").group_by(PointCloud.session_id)
        .order_by(func.count(PointCloud.cloud_id).desc()).limit(50))).all()
    out = []
    for sid, n in rows:
        s = await db.get(DbSession, sid)
        cams = (await db.execute(select(Frame.cam_id, func.count()).where(
            Frame.session_id == sid, Frame.ts_ns.in_(select(PointCloud.ts_ns).where(PointCloud.session_id == sid)))
            .group_by(Frame.cam_id))).all()
        if not cams:
            continue
        cal = (await db.execute(select(CameraCalibration.cam_id, CameraCalibration.source).where(
            CameraCalibration.session_id == sid))).all()
        out.append({"session_id": str(sid), "route": s.route if s else None, "vehicle_id": s.vehicle_id if s else None,
                    "scans": int(n), "cameras": [{"cam_id": c, "synced_frames": int(k)} for c, k in cams],
                    "calibrations": [{"cam_id": c, "source": src} for c, src in cal]})
    return out


async def synced_frames(db: AsyncSession, session_id: uuid.UUID, cam_id: str | None = None) -> list[dict]:
    """Camera frames of the session with a LiDAR scan at the same instant (within 60 ms), in time order."""
    frames = (await db.execute(select(Frame.frame_id, Frame.ts_ns, Frame.cam_id, Frame.width, Frame.height).where(
        Frame.session_id == session_id, *([Frame.cam_id == cam_id] if cam_id else []))
        .order_by(Frame.ts_ns))).all()
    clouds = (await db.execute(select(PointCloud.cloud_id, PointCloud.ts_ns).where(
        PointCloud.session_id == session_id, PointCloud.source == "lidar").order_by(PointCloud.ts_ns))).all()
    if not clouds:
        return []
    cts = np.array([c[1] for c in clouds], dtype=np.int64)
    out = []
    for fid, ts, cam, w, h in frames:
        i = int(np.clip(np.searchsorted(cts, ts), 0, len(cts) - 1))
        j = min((i - 1, i), key=lambda k: abs(int(cts[max(0, k)]) - ts))
        j = max(0, j)
        if abs(int(cts[j]) - ts) <= PAIR_MATCH_NS:
            out.append({"frame_id": str(fid), "cloud_id": str(clouds[j][0]), "ts_ns": int(ts), "cam_id": cam,
                        "width": int(w), "height": int(h)})
    return out


async def calibrations(db: AsyncSession, session_id: uuid.UUID, cam_id: str, img_w: int, img_h: int) -> dict:
    """The calibration in force (the manual row when there is one), and the one it was aligned from."""
    from services.calibration.resolve import (
        calibration_from_row,
        ego_height_from_sensors,
        nominal_calibration,
    )

    rows = (await db.execute(select(CameraCalibration).where(
        CameraCalibration.session_id == session_id, CameraCalibration.cam_id == cam_id))).scalars().all()
    sess = await db.get(DbSession, session_id)
    eh = ego_height_from_sensors(sess.sensors if sess else None)

    def build(row):
        c = calibration_from_row(row, img_w, img_h)
        c.ego_height_m = eh
        return c

    base_row = next((r for r in rows if r.source != "manual"), None)
    manual_row = next((r for r in rows if r.source == "manual"), None)
    base = build(base_row) if base_row else nominal_calibration(cam_id, img_w, img_h)
    current = build(manual_row) if manual_row else base
    return {"base": base, "current": current, "base_row": base_row, "manual_row": manual_row}


def _cal_dict(c) -> dict:
    return {"source": c.source, "quality": round(float(c.quality), 3), "model": c.model,
            "fx": c.fx, "fy": c.fy, "cx": c.cx, "cy": c.cy,
            "rpy_deg": [round(float(v), 6) for v in c.rpy_deg], "xyz_m": [round(float(v), 6) for v in c.xyz_m]}


def _decode(uri: str) -> np.ndarray | None:
    raw = get_object_store().get_bytes(uri)
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


async def load_view(db: AsyncSession, frame_id: uuid.UUID, *, max_points: int = 60000):
    """(frame, cloud row, AlignView) for one synced frame."""
    from services.lidar.calib.align import prepare_view
    from services.lidar.ingest.store import load_cloud

    frame = await db.get(Frame, frame_id)
    if frame is None:
        raise LookupError("frame not found")
    pc = (await db.execute(select(PointCloud).where(
        PointCloud.session_id == frame.session_id, PointCloud.source == "lidar",
        PointCloud.ts_ns >= frame.ts_ns - PAIR_MATCH_NS, PointCloud.ts_ns <= frame.ts_ns + PAIR_MATCH_NS)
        .order_by(func.abs(PointCloud.ts_ns - frame.ts_ns)).limit(1))).scalar_one_or_none()
    if pc is None:
        raise ValueError("no LiDAR scan within 60 ms of this frame")

    def work():
        cloud = load_cloud(pc.cloud_uri)
        img = _decode(frame.img_uri)
        if img is None:
            raise ValueError("frame image unavailable")
        return prepare_view(cloud.xyz, cloud.intensity, img, frame_id=str(frame.frame_id), max_points=max_points)

    return frame, pc, await asyncio.to_thread(work)


async def view_points(db: AsyncSession, frame_id: uuid.UUID, max_points: int = VIEW_MAX_POINTS) -> dict:
    """What the browser needs to project live: points (x, y, z, intensity, edge weight) that can reach the image
    under any extrinsic the sliders allow, the edge field, and both calibrations."""
    from services.lidar.calib.align import project, score_view, with_extrinsics

    frame, pc, view = await load_view(db, frame_id, max_points=200000)
    cals = await calibrations(db, frame.session_id, frame.cam_id, frame.width, frame.height)
    cur = cals["current"]
    # Keep points that land in a frame 40% larger than the image under the current calibration: the sliders
    # move a few degrees, so nothing outside that can be dragged into view.
    big = with_extrinsics(cur, cur.rpy_deg, cur.xyz_m)
    uv, _inside, z = project(view.xyz, big, frame.width, frame.height)
    mx, my = 0.4 * frame.width, 0.4 * frame.height
    keep = (z > 0.5) & (uv[:, 0] > -mx) & (uv[:, 0] < frame.width + mx) & (uv[:, 1] > -my) & (uv[:, 1] < frame.height + my)
    idx = np.flatnonzero(keep)
    edge_idx, rest_idx = idx[view.weights[idx] > 0], idx[view.weights[idx] <= 0]
    budget = max(0, max_points - len(edge_idx))
    if len(rest_idx) > budget:
        rest_idx = np.random.default_rng(0).choice(rest_idx, budget, replace=False)
    sel = np.sort(np.concatenate([edge_idx, rest_idx]))
    arr = np.c_[view.xyz[sel], view.intensity[sel], view.weights[sel]].astype(np.float32)
    ok, png = cv2.imencode(".png", np.clip(view.emap * 255.0, 0, 255).astype(np.uint8))
    s_cur = score_view(view, cur)
    s_base = score_view(view, cals["base"])
    return {"frame_id": str(frame_id), "cloud_id": str(pc.cloud_id), "session_id": str(frame.session_id),
            "cam_id": frame.cam_id, "width": frame.width, "height": frame.height,
            "points": arr, "edge_png": "data:image/png;base64," + base64.b64encode(png.tobytes()).decode() if ok else None,
            "current": _cal_dict(cur), "base": _cal_dict(cals["base"]),
            "has_manual": cals["manual_row"] is not None,
            "score_current": s_cur, "score_base": s_base, "scan_points": int(pc.point_count)}


async def score_extrinsic(db: AsyncSession, frame_ids: list[uuid.UUID], rpy, xyz) -> dict:
    from services.lidar.calib.align import score, with_extrinsics

    views, cal = [], None
    for fid in frame_ids:
        frame, _pc, v = await load_view(db, fid)
        views.append(v)
        if cal is None:
            cal = (await calibrations(db, frame.session_id, frame.cam_id, frame.width, frame.height))["current"]
    cand = with_extrinsics(cal, rpy, xyz)
    return {"current": await asyncio.to_thread(score, views, cal),
            "candidate": await asyncio.to_thread(score, views, cand)}


async def refine_job(params: dict, progress=None, job_id=None) -> dict:
    """The body of the `driving_calib_refine` tool job: refine from a start extrinsic over a few frames."""
    from db.session import get_sessionmaker
    from services.lidar.calib.align import projection_shift, refine, score, with_extrinsics

    frame_ids = [uuid.UUID(f) for f in params["frame_ids"]][:6]
    if not frame_ids:
        raise ValueError("pick at least one frame")
    async with get_sessionmaker()() as db:
        views, frames = [], []
        for fid in frame_ids:
            frame, _pc, v = await load_view(db, fid)
            views.append(v)
            frames.append(frame)
        f0 = frames[0]
        cals = await calibrations(db, f0.session_id, f0.cam_id, f0.width, f0.height)
    start = cals["current"]
    if params.get("rpy_deg") and params.get("xyz_m"):
        start = with_extrinsics(start, params["rpy_deg"], params["xyz_m"])
    if progress is not None:
        await progress(0.05, "scoring the start")
    res = await asyncio.to_thread(refine, views, start)
    if progress is not None:
        await progress(0.85, "checking on frames it did not see")
    # A refinement fitted to a few frames can fit their noise: the forward translation in particular is barely
    # observable from a forward camera and drifts to whatever those frames like. So the result is also scored on
    # frames held out of the fit, and only a result that improves there too is recommended for saving. Measured on
    # KITTI 0015: refining the dataset calibration "improved" the fit frames by 1% and made held-out frames 5%
    # worse; refining a 25 px misalignment improved both.
    held = await _held_out(f0.session_id, f0.cam_id, set(map(str, frame_ids)), int(params.get("held_out", 3)))
    refined = with_extrinsics(start, res["rpy_deg"], res["xyz_m"])
    if held:
        hb = await asyncio.to_thread(score, held, start)
        ha = await asyncio.to_thread(score, held, refined)
        res["held_out"] = {"frames": [v.frame_id for v in held], "before": hb, "after": ha,
                           "improved": ha["objective"] > hb["objective"] + 1e-6}
    else:
        res["held_out"] = None
    res["recommend_save"] = bool(res["improved"] and (res["held_out"] is None or res["held_out"]["improved"]))
    if progress is not None:
        await progress(0.95, "recording")
    res["shift_from_start"] = projection_shift(views[0].xyz, start, refined, f0.width, f0.height)
    res["shift_from_base"] = projection_shift(views[0].xyz, cals["base"], refined, f0.width, f0.height)
    res["start"] = {"rpy_deg": list(start.rpy_deg), "xyz_m": list(start.xyz_m)}
    async with get_sessionmaker()() as db:
        row = CalibrationAlignment(
            session_id=f0.session_id, cam_id=f0.cam_id, frame_ids=[str(f) for f in frame_ids], method="auto",
            base_source=cals["base"].source, base_rpy_deg=list(cals["base"].rpy_deg),
            base_xyz_m=list(cals["base"].xyz_m), rpy_deg=res["rpy_deg"], xyz_m=res["xyz_m"],
            score_before=res["before"], score_after=res["after"], saved=False, tool_job_id=job_id,
            created_by=params.get("user_id") and uuid.UUID(params["user_id"]))
        db.add(row)
        await db.flush()
        res["alignment_id"] = str(row.alignment_id)
        await db.commit()
    log.info("calib.refined", session=str(f0.session_id), before=res["before"]["objective"],
             after=res["after"]["objective"], evals=res["evaluations"])
    return res


async def _held_out(session_id, cam_id: str, exclude: set[str], n: int) -> list:
    """Up to n synced frames not in the fit, spread evenly over the session, prepared for scoring."""
    from db.session import get_sessionmaker

    if n <= 0:
        return []
    async with get_sessionmaker()() as db:
        frames = [f for f in await synced_frames(db, session_id, cam_id) if f["frame_id"] not in exclude]
        if not frames:
            return []
        pick = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, min(n, len(frames)) + 2)[1:-1]] \
            if len(frames) > n else frames
        out = []
        for f in pick:
            _fr, _pc, v = await load_view(db, uuid.UUID(f["frame_id"]))
            out.append(v)
    return out


class ResolverNotManualAware(RuntimeError):
    """The shared resolver still assumes one calibration row per camera; a manual row would break it."""


async def save_manual(db: AsyncSession, frame_ids: list[uuid.UUID], rpy, xyz, *, method: str, user_id=None,
                      notes: str | None = None, pairs: dict | None = None) -> dict:
    """Store an alignment as the camera's `manual` calibration beside the dataset row, and record why."""
    from services.calibration.resolve import resolve_calibration
    from services.lidar.calib.align import pairs_error, score, with_extrinsics

    if method not in ("manual", "auto", "pairs"):
        raise ValueError("method must be manual, auto or pairs")
    views, frames = [], []
    for fid in frame_ids[:6]:
        frame, _pc, v = await load_view(db, fid)
        views.append(v)
        frames.append(frame)
    if not frames:
        raise ValueError("pick at least one frame to score the alignment on")
    f0 = frames[0]
    cals = await calibrations(db, f0.session_id, f0.cam_id, f0.width, f0.height)
    cur, base = cals["current"], cals["base"]
    new = with_extrinsics(cur, rpy, xyz)
    before = await asyncio.to_thread(score, views, cur)
    after = await asyncio.to_thread(score, views, new)
    if pairs and pairs.get("points"):
        before["pairs"] = pairs_error(pairs["points"], pairs["uv"], cur, f0.width, f0.height)
        after["pairs"] = pairs_error(pairs["points"], pairs["uv"], new, f0.width, f0.height)

    row = cals["manual_row"]
    if row is None:
        b = cals["base_row"]
        row = CameraCalibration(session_id=f0.session_id, cam_id=f0.cam_id, source="manual")
        if b is not None:
            row.model, row.fx, row.fy, row.cx, row.cy = b.model, b.fx, b.fy, b.cx, b.cy
            row.dist, row.ref_width = list(b.dist or []), b.ref_width
        else:
            row.model, row.fx, row.fy, row.cx, row.cy = base.model, base.fx, base.fy, base.cx, base.cy
            row.dist, row.ref_width = list(base.dist or []), f0.width
        db.add(row)
    row.rpy_deg = [round(float(v), 6) for v in rpy]
    row.xyz_m = [round(float(v), 6) for v in xyz]
    row.quality = MANUAL_QUALITY
    await db.flush()
    al = CalibrationAlignment(
        session_id=f0.session_id, cam_id=f0.cam_id, frame_ids=[str(f.frame_id) for f in frames], method=method,
        base_source=base.source, base_rpy_deg=list(base.rpy_deg), base_xyz_m=list(base.xyz_m),
        rpy_deg=row.rpy_deg, xyz_m=row.xyz_m, score_before=before, score_after=after, saved=True,
        calibration_id=row.id, notes=notes, created_by=user_id)
    db.add(al)
    await db.flush()
    out = {"calibration_id": str(row.id), "alignment_id": str(al.alignment_id), "before": before, "after": after,
           "improved": after["objective"] > before["objective"] + 1e-6, "base_source": base.source}
    await db.commit()
    # The shared resolver must pick the manual row; if it still assumes one row per camera it raises on two, and
    # every 3D consumer of this session would break. Checked here, and undone, rather than left to surface there.
    try:
        got = await resolve_calibration(f0.session_id, f0.cam_id, f0.width, f0.height)
    except Exception as exc:  # noqa: BLE001  MultipleResultsFound, from a resolver that predates 0131
        await db.delete(await db.get(CameraCalibration, row.id))
        al2 = await db.get(CalibrationAlignment, al.alignment_id)
        if al2 is not None:
            al2.saved, al2.calibration_id = False, None
            al2.notes = (al2.notes or "") + " [not saved: resolver not manual-aware]"
        await db.commit()
        raise ResolverNotManualAware(
            "the calibration resolver does not yet prefer a manual row beside the dataset row "
            f"({type(exc).__name__}); nothing was saved") from exc
    out["resolver_source"] = got.source
    log.info("calib.manual_saved", session=str(f0.session_id), cam=f0.cam_id, improved=out["improved"])
    return out


async def revert_manual(db: AsyncSession, session_id: uuid.UUID, cam_id: str) -> bool:
    row = (await db.execute(select(CameraCalibration).where(
        CameraCalibration.session_id == session_id, CameraCalibration.cam_id == cam_id,
        CameraCalibration.source == "manual"))).scalar_one_or_none()
    if row is None:
        return False
    await db.delete(row)
    await db.commit()
    return True


async def history(db: AsyncSession, session_id: uuid.UUID, cam_id: str | None = None, limit: int = 20) -> list[dict]:
    q = select(CalibrationAlignment).where(CalibrationAlignment.session_id == session_id)
    if cam_id:
        q = q.where(CalibrationAlignment.cam_id == cam_id)
    rows = (await db.execute(q.order_by(CalibrationAlignment.created_at.desc()).limit(limit))).scalars().all()
    return [{"alignment_id": str(r.alignment_id), "cam_id": r.cam_id, "method": r.method, "saved": r.saved,
             "base_source": r.base_source, "rpy_deg": r.rpy_deg, "xyz_m": r.xyz_m,
             "score_before": r.score_before, "score_after": r.score_after, "frames": len(r.frame_ids or []),
             "notes": r.notes, "created_at": r.created_at.isoformat() if r.created_at else None} for r in rows]
