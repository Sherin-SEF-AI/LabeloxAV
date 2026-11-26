"""API for the video annotation tools, agent A: video and editor (#1 SAM 2 video tracking, #2 text prompt, #4 correction fan-out).

Mounted under /api; every path here starts with /tools/video/ so the area's routes stay together. Set the role
each route needs with `Depends(require_role(...))` (annotator / reviewer / admin) or `Depends(require_user)`.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()
