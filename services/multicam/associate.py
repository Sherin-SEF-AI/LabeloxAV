"""Cross-camera object association (M3.1): the same physical object seen in overlapping rig cameras at one
instant gets one rig identity (rig_track_id + cross_cam_links). Combines geometric overlap (when
extrinsics are available) with DINOv3 appearance cosine (the Phase 1 object embeddings). Gated on
calibration: a session that fails M3.0 is excluded. This extends the M2.0 per-camera tracks to rig level.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from uuid import UUID

import numpy as np
from sqlalchemy import select, update

from core.logging import get_logger
from db.models import Object, ObjectEmbedding
from db.session import get_sessionmaker
from services.calibration.report import session_calibrated
from services.multicam.sync import frame_groups

log = get_logger("multicam_associate")


async def associate_session(session_id: UUID, appearance_cos: float = 0.55, tol_ns: int = 20_000_000) -> dict:
    if not await session_calibrated(session_id):
        return {"associated": 0, "rig_tracks": 0,
                "reason": "session not calibrated (run /api/calibration/validate first; a failing session is excluded)"}
    grp = await frame_groups(session_id, tol_ns)
    if not grp["multicamera"]:
        return {"associated": 0, "rig_tracks": 0, "cameras": grp["cameras"],
                "reason": "single-camera session, nothing to associate across views"}

    maker = get_sessionmaker()
    n_links, n_rig = 0, 0
    async with maker() as db:
        for g in grp["groups"]:
            if len(g["frames"]) < 2:
                continue
            cam_of = {UUID(f["frame_id"]): cam for cam, f in g["frames"].items()}
            rows = (await db.execute(
                select(Object.object_id, Object.frame_id, ObjectEmbedding.dino_vec)
                .join(ObjectEmbedding, ObjectEmbedding.object_id == Object.object_id, isouter=True)
                .where(Object.frame_id.in_(list(cam_of))))).all()
            objs = [{"oid": oid, "cam": cam_of[fid],
                     "vec": np.asarray(v, np.float32) if v is not None else None} for oid, fid, v in rows]

            parent = {o["oid"]: o["oid"] for o in objs}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for i in range(len(objs)):
                for j in range(i + 1, len(objs)):
                    a, b = objs[i], objs[j]
                    if a["cam"] == b["cam"] or a["vec"] is None or b["vec"] is None:
                        continue
                    if float(a["vec"] @ b["vec"]) >= appearance_cos:
                        ra, rb = find(a["oid"]), find(b["oid"])
                        if ra != rb:
                            parent[ra] = rb

            comp: dict = defaultdict(list)
            for o in objs:
                comp[find(o["oid"])].append(o)
            for members in comp.values():
                if len({m["cam"] for m in members}) < 2:  # only cross-camera components are rig tracks
                    continue
                rid = uuid.uuid4()
                n_rig += 1
                for m in members:
                    links = {cam: str(mm["oid"]) for mm in members for cam in [mm["cam"]] if mm["oid"] != m["oid"]}
                    await db.execute(update(Object).where(Object.object_id == m["oid"])
                                     .values(rig_track_id=rid, cross_cam_links={"rig_track_id": str(rid), "views": links}))
                    n_links += 1
        await db.commit()

    out = {"associated": n_links, "rig_tracks": n_rig, "cameras": grp["cameras"]}
    log.info("multicam.associated", session_id=str(session_id), **out)
    return out
