"""API for the sensors annotation tools, agent E: sensors (#10 radar and event cameras, #15 reconstruction, #16 synthetic variants).

Mounted under /api; every path here starts with /tools/sensors/ so the area's routes stay together. Set the role
each route needs with `Depends(require_role(...))` (annotator / reviewer / admin) or `Depends(require_user)`.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()
