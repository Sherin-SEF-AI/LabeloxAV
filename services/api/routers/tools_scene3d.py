"""API for the scene3d annotation tools, agent C: 3D (#3 label once in 3D, #5 point segmentation, #6 occupancy).

Mounted under /api; every path here starts with /tools/scene3d/ so the area's routes stay together. Set the role
each route needs with `Depends(require_role(...))` (annotator / reviewer / admin) or `Depends(require_user)`.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()
