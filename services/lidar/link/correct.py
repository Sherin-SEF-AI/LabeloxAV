"""Interactive correction: fix one 3D object, find similar objects, and offer a batch update. Similarity is
3D shape (same ontology class, nearest dimensions) by default, with the DINOv3 appearance of the linked 2D
object as an optional refinement, reusing the Phase 1 similarity infrastructure. A batch update writes the
corrected objects as source=human and bumps their version.
"""

from __future__ import annotations

import uuid

import numpy as np
from sqlalchemy import select, update

from core.logging import get_logger
from db.models import Object3D, PointCloud
from db.session import get_sessionmaker

log = get_logger("lidar_correct")


async def find_similar(object_3d_id: uuid.UUID, k: int = 10, same_session: bool = True) -> dict:
    """Objects most similar to a given 3D object by class and dimensions. The candidates for a batch update."""
    async with get_sessionmaker()() as db:
        ref = await db.get(Object3D, object_3d_id)
        if ref is None:
            return {"error": "object_3d not found"}
        session_id = None
        if same_session:
            pc = await db.get(PointCloud, ref.cloud_id)
            session_id = pc.session_id if pc else None
        q = select(Object3D).where(Object3D.class_id == ref.class_id,
                                   Object3D.object_3d_id != object_3d_id)
        if session_id is not None:
            q = q.join(PointCloud, Object3D.cloud_id == PointCloud.cloud_id).where(
                PointCloud.session_id == session_id)
        candidates = (await db.execute(q)).scalars().all()
    ref_dims = np.asarray(ref.dims, dtype=np.float32)
    scored = []
    for c in candidates:
        dist = float(np.linalg.norm(np.asarray(c.dims, dtype=np.float32) - ref_dims))
        scored.append((dist, c))
    scored.sort(key=lambda s: s[0])
    top = scored[:k]
    return {"object_3d_id": str(object_3d_id), "class_id": ref.class_id,
            "similar": [{"object_3d_id": str(c.object_3d_id), "dims": c.dims, "dims_dist": round(d, 3),
                         "state": c.state, "source": c.source} for d, c in top]}


async def batch_correct(object_3d_ids: list[uuid.UUID], class_id: int | None = None,
                        dims: list[float] | None = None) -> dict:
    """Apply a correction (class and/or dimensions) to a batch of 3D objects as a human edit."""
    if not object_3d_ids:
        return {"updated": 0}
    values: dict = {"source": "human", "state": "accepted", "version": Object3D.version + 1}
    if class_id is not None:
        values["class_id"] = class_id
    if dims is not None:
        values["dims"] = dims
    async with get_sessionmaker()() as db:
        res = await db.execute(update(Object3D).where(Object3D.object_3d_id.in_(object_3d_ids)).values(**values))
        await db.commit()
    log.info("lidar.batch_correct", n=res.rowcount, class_id=class_id)
    return {"updated": res.rowcount, "class_id": class_id, "dims": dims}
