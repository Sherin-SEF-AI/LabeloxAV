"""Milestone D: human-speech detection on the 48kHz (or 16kHz dashcam) audio for DPDPA redaction. Detection
needs a voice-activity / speech model, so it is a runtime seam: this module owns the segment store and the
redaction-zone bookkeeping (real), and the model call is a WIRE adapter, not a fabricated detector. Default
is redact: a detected speech segment is personal until a human confirms it is non-personal.
"""

from __future__ import annotations

from core.logging import get_logger

log = get_logger("speech")


def detect_speech_regions(audio_signal, sample_rate: int) -> list[dict]:
    """Speech regions [{t_start_ns, t_end_ns}] on the audio. WIRE: a VAD/speech model (e.g. silero-vad or a
    panns speech head) lives in an adapter and fills this in. Returns an empty list until wired, so the
    caller stores nothing rather than inventing speech."""
    # WIRE: services/anonymize adapter for a voice-activity / speech detector. No model runtime is invented
    # here; the gate and store below are complete and run without it.
    return []


async def persist_speech_segments(session_id, regions: list[dict], method_version: str = "wire") -> dict:
    """Store detected speech regions as personal, un-redacted segments (default redact). The export gate then
    refuses the clip until each is masked or confirmed non-personal."""
    from db.models import SpeechSegment
    from db.session import get_sessionmaker
    async with get_sessionmaker()() as db:
        for r in regions:
            db.add(SpeechSegment(session_id=session_id, t_start_ns=int(r["t_start_ns"]),
                                 t_end_ns=int(r["t_end_ns"]), is_personal=True, redacted=False,
                                 method_version=method_version))
        await db.commit()
    log.info("speech.persisted", session=str(session_id), segments=len(regions))
    return {"session_id": str(session_id), "segments": len(regions), "default": "personal_unredacted"}
