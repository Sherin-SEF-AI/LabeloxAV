"""3D analytics and natural-language search, extending the Phase 1 dashboard and search to point clouds. The
metrics summarize 3D object counts, point density, and 3D scene coverage; the search parses ontology class
names from a query and returns the clouds whose 3D objects co-occur (for example pedestrians near buses).
"""

from __future__ import annotations

from sqlalchemy import distinct, func, select

from core.logging import get_logger
from db.models import Frame, Object3D, PointCloud
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology

log = get_logger("lidar_analytics3d")


async def metrics_3d() -> dict:
    """3D coverage metrics for the analytics dashboard."""
    onto = get_ontology()
    async with get_sessionmaker()() as db:
        n_objects = (await db.execute(select(func.count()).select_from(Object3D))).scalar() or 0
        n_clouds = (await db.execute(select(func.count()).select_from(PointCloud))).scalar() or 0
        mean_points = (await db.execute(select(func.avg(PointCloud.point_count)))).scalar_one_or_none() or 0
        by_state = dict((await db.execute(
            select(Object3D.state, func.count()).group_by(Object3D.state))).all())
        by_class_rows = (await db.execute(
            select(Object3D.class_id, func.count()).group_by(Object3D.class_id)
            .order_by(func.count().desc()).limit(15))).all()
        scenes = (await db.execute(select(Frame.scene).where(Frame.scene.isnot(None)))).scalars().all()
    by_class = {onto.by_id(int(cid)).name: int(n) for cid, n in by_class_rows}
    structure_cov: dict[str, int] = {}
    for s in scenes:
        st = (s or {}).get("3d_structure")
        if st:
            structure_cov[st] = structure_cov.get(st, 0) + 1
    return {"object_3d_count": int(n_objects), "cloud_count": int(n_clouds),
            "mean_point_density": round(float(mean_points), 1),
            "objects_by_state": {k: int(v) for k, v in by_state.items()},
            "objects_by_class": by_class, "scene_3d_coverage": structure_cov}


def _query_classes(query: str) -> list[str]:
    """Ontology class names mentioned in the query (singular and simple plural)."""
    q = query.lower()
    names = []
    for c in get_ontology().classes:
        n = c.name.lower()
        if n in q or (n + "s") in q or n.replace("_", " ") in q:
            names.append(c.name)
    return sorted(set(names))


async def search_clouds_3d(query: str, limit: int = 50) -> dict:
    """Clouds whose 3D objects include every class named in the query (co-occurrence = 'near')."""
    onto = get_ontology()
    names = _query_classes(query)
    if not names:
        return {"query": query, "classes": [], "clouds": [], "reason": "no ontology class in query"}
    class_ids = [onto.by_name(n).id for n in names]
    async with get_sessionmaker()() as db:
        # clouds that contain ALL requested classes (one HAVING per class via count of distinct matches)
        rows = (await db.execute(
            select(Object3D.cloud_id, func.count(distinct(Object3D.class_id)))
            .where(Object3D.class_id.in_(class_ids))
            .group_by(Object3D.cloud_id)
            .having(func.count(distinct(Object3D.class_id)) == len(class_ids))
            .limit(limit))).all()
    clouds = [str(cid) for cid, _ in rows]
    log.info("lidar.search3d", query=query, classes=names, clouds=len(clouds))
    return {"query": query, "classes": names, "clouds": clouds, "count": len(clouds)}
