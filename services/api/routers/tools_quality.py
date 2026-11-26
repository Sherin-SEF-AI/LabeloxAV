"""API for the quality annotation tools, agent B: quality (#11 consensus, #12 explained errors, #13 VLM reasons, #14 rare scenes).

Mounted under /api; every path here starts with /tools/quality/ so the area's routes stay together. Set the role
each route needs with `Depends(require_role(...))` (annotator / reviewer / admin) or `Depends(require_user)`.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()
