"""Extend the Phase 1 rare-event mining to 3D cues: animal crossing, emergency vehicle, debris on the road,
and flooded road. Writes ScenarioCandidate rows (the same table and review queue as the 2D discovery), so the
existing intelligence surfaces 3D-specific rare events too.
"""

from __future__ import annotations

import uuid

import numpy as np
from sqlalchemy import select

from core.logging import get_logger
from db.models import Frame, Object3D, PointCloud, ScenarioCandidate
from db.session import get_sessionmaker
from services.autolabel.ontology import Ontology, get_ontology
from services.lidar.clean.ground import segment_ground
from services.lidar.extract.common import cluster_dbscan, height_above_plane
from services.lidar.ingest.normalize import Cloud
from services.lidar.ingest.store import load_cloud
from services.lidar.traverse.surface import classify_surface

log = get_logger("lidar_rare3d")

_ANIMAL = {"cattle", "dog", "goat", "buffalo", "pig", "monkey", "horse", "cow"}
_EMERGENCY = {"ambulance", "police_van", "fire_truck"}


def mine_3d_cues(cloud: Cloud, semantic: np.ndarray | None, plane: list[float], cuboids: list[dict],
                 road_class_id: int, onto: Ontology | None = None) -> list[dict]:
    """Return rare 3D cues for one cloud: flooded road, animal crossing, emergency vehicle, road debris."""
    onto = onto or get_ontology()
    cues: list[dict] = []

    surf = classify_surface(cloud, semantic, road_class_id, plane)
    if surf["surface"] == "water" and surf.get("confidence", 0) > 0.2:
        cues.append({"kind": "3d_flooded_road", "score": surf["confidence"], "classes": ["water"]})

    for c in cuboids:
        try:
            name = onto.by_id(c["class_id"]).name
            l1 = onto.by_id(c["class_id"]).l1
        except Exception:
            continue
        if name in _ANIMAL or l1 == "animal":
            cues.append({"kind": "3d_animal_crossing", "score": 0.9, "classes": [name]})
        elif name in _EMERGENCY:
            cues.append({"kind": "3d_emergency_vehicle", "score": 0.95, "classes": [name]})

    # debris: small low non-ground clusters sitting on the road surface
    if semantic is not None:
        road = cloud.xyz[semantic == road_class_id]
        if len(road) > 50:
            above = height_above_plane(road, plane)
            low_obs = road[(above > 0.1) & (above < 0.6)]
            if len(low_obs) >= 15:
                labels = cluster_dbscan(low_obs, eps=0.5)
                small = [cl for cl in set(labels.tolist()) - {-1} if 15 <= (labels == cl).sum() < 200]
                if small:
                    cues.append({"kind": "3d_road_debris", "score": round(min(1.0, len(small) / 3.0), 2),
                                 "classes": ["debris"]})
    return cues


async def mine_session_3d(session_id: uuid.UUID) -> dict:
    """Mine 3D rare cues across a session's clouds and write ScenarioCandidate rows for review."""
    onto = get_ontology()
    from services.lidar.segment3d.semantic import road_class_id
    road_id = road_class_id(onto)
    async with get_sessionmaker()() as db:
        clouds = (await db.execute(select(PointCloud).where(PointCloud.session_id == session_id)
                  .order_by(PointCloud.ts_ns))).scalars().all()
    written, by_kind = 0, {}
    for pc in clouds:
        cloud = load_cloud(pc.cloud_uri)
        _, plane, _ = segment_ground(cloud)
        async with get_sessionmaker()() as db:
            objs = (await db.execute(select(Object3D).where(Object3D.cloud_id == pc.cloud_id))).scalars().all()
            cubs = [{"class_id": o.class_id, "center": o.center, "dims": o.dims, "yaw": o.yaw} for o in objs]
            frame = (await db.execute(select(Frame.frame_id).where(Frame.session_id == session_id,
                     Frame.ts_ns == pc.ts_ns).order_by(Frame.cam_id).limit(1))).scalar_one_or_none()
            for cue in mine_3d_cues(cloud, None, plane, cubs, road_id, onto):
                db.add(ScenarioCandidate(session_id=session_id, frame_id=frame, kind=cue["kind"],
                                         score=cue["score"], rare_classes=cue.get("classes")))
                written += 1
                by_kind[cue["kind"]] = by_kind.get(cue["kind"], 0) + 1
            await db.commit()
    log.info("lidar.rare3d", session=str(session_id), clouds=len(clouds), candidates=written, by=by_kind)
    return {"session_id": str(session_id), "clouds": len(clouds), "candidates": written, "by_kind": by_kind}
