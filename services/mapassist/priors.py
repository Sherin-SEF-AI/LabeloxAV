"""Map-derived annotation priors (M3.2): turn a frame's matched road into editable hints (lane priors,
drivable corridor, road context) that seed Phase-2 annotation. Confidence-weighted hints, never ground
truth; empty where no road was matched (variable Indian OSM coverage)."""

from __future__ import annotations

from uuid import UUID

from db.models import Frame
from db.session import get_sessionmaker


async def frame_priors(frame_id: UUID) -> dict:
    maker = get_sessionmaker()
    async with maker() as db:
        f = await db.get(Frame, frame_id)
    if f is None:
        return {"found": False}
    if not f.road_segment_id:
        return {"found": True, "has_map": False, "hints": []}

    hints = [{"kind": "road_context", "road_class": f.road_class, "lane_count": f.lane_count,
              "speed_limit": f.speed_limit, "confidence": 0.6}]
    if f.lane_count:
        hints.append({"kind": "lane_prior", "suggested_lanes": f.lane_count,
                      "note": "OSM lane count, confirm against the markings", "confidence": 0.5})
    hints.append({"kind": "drivable_prior", "note": "road corridor present in OSM", "confidence": 0.6})
    return {"found": True, "has_map": True, "road_segment_id": f.road_segment_id, "road_class": f.road_class,
            "lane_count": f.lane_count, "speed_limit": f.speed_limit, "hints": hints}
