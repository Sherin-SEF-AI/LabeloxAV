"""API for the driving annotation tools, agent D: driving (#7 lanes and HD map, #8 paths and intent, #9 camera-LiDAR alignment).

Mounted under /api; every path here starts with /tools/driving/ so the area's routes stay together. Set the role
each route needs with `Depends(require_role(...))` (annotator / reviewer / admin) or `Depends(require_user)`.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()
