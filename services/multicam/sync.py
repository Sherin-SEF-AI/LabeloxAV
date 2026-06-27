"""Synchronized multi-view frame groups (M3.1): group the rig's frames by ts_ns within a tolerance (the
STM32 PPS hardware sync keeps cameras mid-exposure centered), so all cameras at a given instant are
annotated together. Single-camera sessions yield one frame per group (degrades gracefully)."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select

from db.models import Frame
from db.session import get_sessionmaker


async def frame_groups(session_id: UUID, tol_ns: int = 20_000_000) -> dict:
    maker = get_sessionmaker()
    async with maker() as db:
        rows = (await db.execute(
            select(Frame.frame_id, Frame.cam_id, Frame.ts_ns, Frame.img_uri)
            .where(Frame.session_id == session_id).order_by(Frame.ts_ns))).all()
    cams = sorted({r[1] for r in rows})
    groups: list[dict] = []
    for fid, cam, ts, uri in rows:
        if not groups or ts - groups[-1]["ts0"] > tol_ns:
            groups.append({"ts0": ts, "ts_ns": ts, "frames": {}})
        groups[-1]["frames"][cam] = {"frame_id": str(fid), "img_uri": uri}
    return {"cameras": cams, "multicamera": len(cams) > 1, "n_groups": len(groups), "groups": groups}
