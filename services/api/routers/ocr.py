"""OCR endpoints (M2.4): read road text (PaddleOCR Indic on pod / Qwen local), excluding license plates."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter

router = APIRouter()


@router.post("/ocr/run")
async def run(session_id: str, limit: int | None = None):
    """OCR a session's text-bearing sign/board objects; plate-overlapping regions are excluded."""
    from services.autolabel.ocr.reader import ocr_session

    return await ocr_session(UUID(session_id), limit)
