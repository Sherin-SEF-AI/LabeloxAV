"""Milestone D: the unified DPDPA pre-sale compliance gate. Face, plate, and now human speech are ONE
fail-closed gate in the export path: a clip with any un-redacted detected face, plate, or personal speech
segment is refused, not warned. The gate is fail-closed by construction: a frame with no anonymization audit
is treated as un-redacted (refuse), never assumed clean.

The speech detector is a runtime seam (a VAD/speech model), so detection is not faked here; the gate logic
that refuses on un-redacted speech is real and tested.
"""

from __future__ import annotations

from core.logging import get_logger

log = get_logger("dpdpa")


def evaluate_dpdpa(export_frame_ids: set, audited_frame_ids: set, speech_segments: list[dict]) -> dict:
    """Fail-closed DPDPA verdict. Blocks if any frame in scope was never anonymized (no PiiAudit row means an
    un-redacted face or plate, refuse not warn) or any personal speech segment is not redacted. speech_segments:
    [{is_personal, redacted}]. Returns {pass, blockers}."""
    blockers = []
    unaudited = export_frame_ids - audited_frame_ids
    if unaudited:
        blockers.append({"kind": "unredacted_visual_pii", "count": len(unaudited),
                         "detail": "frames with no face/plate anonymization audit"})
    unredacted_speech = [s for s in speech_segments
                         if s.get("is_personal", True) and not s.get("redacted", False)]
    if unredacted_speech:
        blockers.append({"kind": "unredacted_speech", "count": len(unredacted_speech),
                         "detail": "personal speech segments not masked"})
    return {"pass": len(blockers) == 0, "blockers": blockers}


async def dpdpa_export_gate(session_id, export_frame_ids: list) -> dict:
    """Load the anonymization audits and speech segments for the frames in scope and run the fail-closed
    DPDPA verdict. Called by the export path before any clip is written or delivered."""
    from sqlalchemy import select

    from db.models import PiiAudit, SpeechSegment
    from db.session import get_sessionmaker
    fids = {str(f) for f in export_frame_ids}
    async with get_sessionmaker()() as db:
        audited = {str(f) for f in (await db.execute(
            select(PiiAudit.frame_id).where(PiiAudit.session_id == session_id))).scalars().all()}
        speech = [{"is_personal": s.is_personal, "redacted": s.redacted} for s in (await db.execute(
            select(SpeechSegment).where(SpeechSegment.session_id == session_id))).scalars().all()]
    verdict = evaluate_dpdpa(fids, audited, speech)
    log.info("dpdpa.gate", session=str(session_id), frames=len(fids), passed=verdict["pass"],
             blockers=len(verdict["blockers"]))
    return verdict
