"""LABELOX core, wired onto the data-engine spine (M4). The three-path auto-label, ontology, active learning,
multi-camera workspace, champion-challenger governance, and recall recovery already exist; the M4 integration
is the label queue that closes the loop between the upstream gates and the human's attention:

  SIEVYX priority (what is worth labeling) MINUS the SANYX/CALYX gates (what is safe to label)

A sample from a session SANYX quarantined or CALYX blocked never enters the queue, so bad data never costs a
label. What remains is ranked by the SIEVYX combined priority. apply_gates() is pure so the exclusion is
unit-testable; build_label_queue() wires it to the live scores and gate state."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import CalibrationValidation, Frame, SessionHealth
from services.activelearn.selector import score_candidates


def apply_gates(items: list[dict], blocked_sessions: set[str]) -> list[dict]:
    """Drop items whose session is gated out (SANYX quarantine or CALYX block), keep the rest in priority
    order. Each item must carry ``session_id``."""
    return [it for it in sorted(items, key=lambda x: x.get("value", 0.0), reverse=True)
            if it.get("session_id") not in blocked_sessions]


async def blocked_session_ids(db: AsyncSession) -> set[str]:
    """The sessions gated out of labeling: SANYX quarantine (latest health decision) or CALYX block."""
    quarantined = set((await db.execute(
        select(SessionHealth.session_id).where(SessionHealth.decision == "quarantine"))).scalars())
    blocked = set((await db.execute(
        select(CalibrationValidation.session_id).where(CalibrationValidation.severity == "block"))).scalars())
    return {str(s) for s in (quarantined | blocked)}


async def build_label_queue(db: AsyncSession, limit: int = 100) -> dict:
    """The gated, SIEVYX-ranked label queue the workspace pulls from."""
    items = await score_candidates(db)
    # annotate each item with its session_id (score_candidates returns frame_id only)
    frame_ids = list({it["frame_id"] for it in items})
    sess_by_frame: dict[str, str] = {}
    if frame_ids:
        for fid, sid in (await db.execute(
                select(Frame.frame_id, Frame.session_id).where(Frame.frame_id.in_(frame_ids)))).all():
            sess_by_frame[str(fid)] = str(sid)
    for it in items:
        it["session_id"] = sess_by_frame.get(it["frame_id"])

    blocked = await blocked_session_ids(db)
    kept = apply_gates(items, blocked)
    return {
        "pool": len(items), "blocked_sessions": len(blocked), "excluded": len(items) - len(kept),
        "items": kept[: min(max(limit, 1), 1000)],
    }
