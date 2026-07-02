"""Synchronized multi-view frame groups (M3.1 / M-MC.0): group the rig's frames by ts_ns within a tolerance
(the STM32 PPS hardware sync keeps cameras mid-exposure centered), so all cameras at a given instant are
annotated together. Single-camera sessions yield one frame per group (degrades gracefully).

M-MC.0 extends the original in-memory grouping into a persisted `frame_group` table: each row records the
per-camera frame ids, the cameras that dropped a frame in this window (missing_cams), and the sync spread
(max pairwise timestamp difference across members), so the multi-camera canvas can navigate whole groups and
surface dropouts and out-of-tolerance windows instead of silently omitting them.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Frame, FrameGroup
from db.session import get_sessionmaker


def _cluster(rows: list, session_cams: list[str], tol_ns: int) -> list[dict]:
    """Greedily cluster time-ordered (frame_id, cam, ts, uri) rows into rig groups within tol_ns.

    A group opens on the first frame and admits later frames until one falls more than tol_ns beyond the
    group's earliest timestamp. Within a group each camera keeps its latest frame; sync_spread_ns is the span
    from the earliest to the latest member, and missing_cams are the session cameras with no frame here.
    """
    groups: list[dict] = []
    for fid, cam, ts, uri in rows:
        if not groups or ts - groups[-1]["ts0"] > tol_ns:
            groups.append({"ts0": ts, "ts_ns": ts, "frames": {}, "_ts": {}})
        g = groups[-1]
        g["frames"][cam] = {"frame_id": str(fid), "img_uri": uri, "ts_ns": int(ts)}
        g["_ts"][cam] = int(ts)
    for g in groups:
        member_ts = list(g["_ts"].values())
        g["sync_spread_ns"] = (max(member_ts) - min(member_ts)) if member_ts else 0
        g["missing_cams"] = [c for c in session_cams if c not in g["frames"]]
        g["n_cams"] = len(g["frames"])
        g.pop("_ts", None)
    return groups


async def frame_groups(session_id: UUID, tol_ns: int = 20_000_000) -> dict:
    """In-memory rig groups for a session (no writes): the shape the canvas and API read for navigation."""
    maker = get_sessionmaker()
    async with maker() as db:
        rows = (await db.execute(
            select(Frame.frame_id, Frame.cam_id, Frame.ts_ns, Frame.img_uri)
            .where(Frame.session_id == session_id).order_by(Frame.ts_ns))).all()
    cams = sorted({r[1] for r in rows})
    groups = _cluster(rows, cams, tol_ns)
    return {"cameras": cams, "multicamera": len(cams) > 1, "n_groups": len(groups), "groups": groups}


async def persist_groups(session_id: UUID, tol_ns: int = 20_000_000, db: AsyncSession | None = None) -> dict:
    """Assemble and persist the session's frame groups (idempotent backfill): recompute from the frames and
    replace any existing rows for the session. Returns a summary with the spread distribution and any groups
    that exceeded tolerance or dropped a camera, so ingestion and the backfill endpoint can report health."""
    own = db is None
    maker = get_sessionmaker()
    db = db or maker()
    try:
        rows = (await db.execute(
            select(Frame.frame_id, Frame.cam_id, Frame.ts_ns, Frame.img_uri)
            .where(Frame.session_id == session_id).order_by(Frame.ts_ns))).all()
        cams = sorted({r[1] for r in rows})
        groups = _cluster(rows, cams, tol_ns)

        await db.execute(delete(FrameGroup).where(FrameGroup.session_id == session_id))
        out_of_tol = 0
        with_missing = 0
        for g in groups:
            frame_ids = {cam: f["frame_id"] for cam, f in g["frames"].items()}
            if g["sync_spread_ns"] > tol_ns:
                out_of_tol += 1
            if g["missing_cams"]:
                with_missing += 1
            db.add(FrameGroup(
                session_id=session_id, ts_ns=g["ts_ns"], frame_ids=frame_ids,
                missing_cams=g["missing_cams"], sync_spread_ns=g["sync_spread_ns"], n_cams=g["n_cams"]))
        await db.commit()
        return {"session_id": str(session_id), "cameras": cams, "multicamera": len(cams) > 1,
                "n_groups": len(groups), "groups_out_of_tolerance": out_of_tol,
                "groups_with_missing_cam": with_missing, "tol_ns": tol_ns}
    finally:
        if own:
            await db.close()


def _group_dict(g: FrameGroup) -> dict:
    return {"group_id": str(g.group_id), "ts_ns": int(g.ts_ns), "frame_ids": g.frame_ids or {},
            "missing_cams": list(g.missing_cams or []), "sync_spread_ns": int(g.sync_spread_ns or 0),
            "n_cams": int(g.n_cams or 0), "confirmed": bool(g.confirmed)}


async def list_groups(session_id: UUID) -> dict:
    """The session's persisted groups in time order, for the canvas group navigator."""
    maker = get_sessionmaker()
    async with maker() as db:
        rows = (await db.execute(
            select(FrameGroup).where(FrameGroup.session_id == session_id).order_by(FrameGroup.ts_ns))).scalars().all()
        cams = sorted({c for g in rows for c in (g.frame_ids or {})})
        return {"session_id": str(session_id), "cameras": cams, "multicamera": len(cams) > 1,
                "n_groups": len(rows), "groups": [_group_dict(g) for g in rows]}


async def group_at_ts(session_id: UUID, ts_ns: int) -> dict | None:
    """The persisted group nearest a timestamp: how the annotation workspace opens the rig view for a frame."""
    from sqlalchemy import func

    maker = get_sessionmaker()
    async with maker() as db:
        g = (await db.execute(
            select(FrameGroup).where(FrameGroup.session_id == session_id)
            .order_by(func.abs(FrameGroup.ts_ns - ts_ns)).limit(1))).scalar_one_or_none()
        return _group_dict(g) if g else None


async def adjacent_group(session_id: UUID, group_id: UUID, direction: str) -> dict | None:
    """The next or previous group in time (group-aware prev/next navigation)."""
    from sqlalchemy import and_

    maker = get_sessionmaker()
    async with maker() as db:
        cur = await db.get(FrameGroup, group_id)
        if cur is None:
            return None
        if direction == "prev":
            q = (select(FrameGroup).where(and_(FrameGroup.session_id == session_id, FrameGroup.ts_ns < cur.ts_ns))
                 .order_by(FrameGroup.ts_ns.desc()).limit(1))
        else:
            q = (select(FrameGroup).where(and_(FrameGroup.session_id == session_id, FrameGroup.ts_ns > cur.ts_ns))
                 .order_by(FrameGroup.ts_ns).limit(1))
        g = (await db.execute(q)).scalar_one_or_none()
        return _group_dict(g) if g else None


async def confirm_group(group_id: UUID, confirmed: bool = True) -> dict | None:
    """Confirm (or unconfirm) a whole group at once: the group-level analogue of confirming a single frame."""
    maker = get_sessionmaker()
    async with maker() as db:
        g = await db.get(FrameGroup, group_id)
        if g is None:
            return None
        g.confirmed = confirmed
        await db.commit()
        await db.refresh(g)
        return _group_dict(g)
