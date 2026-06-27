"""Multi-drive map fusion (M3.3): align overlapping map elements from several drives into one consistent
layer. The heavy path (GTSAM trajectory pose-graph alignment) runs as the map_fusion A100 burst
(services/hdmap/cloud.py + cloud/mapfusion_pod.py); this is the local averaging-fusion fallback that keeps
M3.3 testable without the pod: cluster same-kind elements by proximity and merge into consensus geometry,
boosting confidence where independent drives agree."""

from __future__ import annotations

import math

from geoalchemy2 import Geometry
from sqlalchemy import cast, func, select

from core.config import get_settings
from core.logging import get_logger
from db.models import MapElement
from db.session import get_sessionmaker

log = get_logger("hdmap_fuse")


async def fuse_local(session_ids: list[str], region: str | None = None) -> dict:
    """Cluster per-drive map_elements (within fuse_cluster_m, per kind) into consensus elements."""
    cfg = get_settings()
    region = region or cfg.spatial.map_region
    radius = cfg.spatial.fuse_cluster_m

    maker = get_sessionmaker()
    async with maker() as db:
        g = cast(MapElement.geometry, Geometry)
        rows = (await db.execute(select(
            MapElement.element_id, MapElement.kind, MapElement.confidence, MapElement.source_frames,
            MapElement.source_sessions, MapElement.attrs, MapElement.calibration_version,
            func.ST_Y(func.ST_Centroid(g)), func.ST_X(func.ST_Centroid(g)), func.ST_AsText(g))
            .where(MapElement.commit_id.is_(None)))).all()  # fuse only un-committed per-drive elements

    elems = [{"id": str(r[0]), "kind": r[1], "conf": float(r[2] or 0.5), "frames": r[3] or [],
              "sessions": r[4] or [], "attrs": r[5] or {}, "calib": r[6], "lat": r[7], "lon": r[8], "wkt": r[9]}
             for r in rows if any(s in (r[4] or []) for s in session_ids)]

    fused, used = [], set()
    for kind in sorted({e["kind"] for e in elems}):
        ke = [e for e in elems if e["kind"] == kind]
        for e in ke:
            if e["id"] in used:
                continue
            cluster = [e]
            used.add(e["id"])
            for f in ke:
                if f["id"] in used:
                    continue
                dm = math.hypot((e["lat"] - f["lat"]) * 111320.0,
                                (e["lon"] - f["lon"]) * 111320.0 * math.cos(math.radians(e["lat"])))
                if dm <= radius:
                    cluster.append(f)
                    used.add(f["id"])
            rep = max(cluster, key=lambda x: x["conf"])
            conf = min(0.99, (sum(c["conf"] for c in cluster) / len(cluster)) * (1.0 + 0.1 * (len(cluster) - 1)))
            fused.append({
                "kind": kind, "wkt": rep["wkt"],
                "attrs": {**rep["attrs"], "fused_count": len(cluster)},
                "frames": sorted({fr for c in cluster for fr in c["frames"]}),
                "sessions": sorted({s for c in cluster for s in c["sessions"]}),
                "calib": rep["calib"], "confidence": conf,
            })

    out = {"input_elements": len(elems), "fused_elements": len(fused),
           "consensus": sum(1 for f in fused if f["attrs"]["fused_count"] > 1), "region": region, "fused": fused}
    log.info("hdmap.fuse_local", **{k: out[k] for k in ("input_elements", "fused_elements", "consensus")})
    return out
