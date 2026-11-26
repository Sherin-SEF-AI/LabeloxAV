"""API for the agents annotation tools, agent F: agents (#17 language bulk edits, #18 overnight autopilot).

Mounted under /api; every path here starts with /tools/agents/ so the area's routes stay together. Set the role
each route needs with `Depends(require_role(...))` (annotator / reviewer / admin) or `Depends(require_user)`.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()
