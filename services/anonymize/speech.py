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
    panns speech head) lives in an adapter and fills this in.

    Fail-closed until wired: with no detector we cannot prove a clip is speech-free, so the WHOLE clip is
    returned as one personal segment (default redact) rather than an empty list that would silently let audio
    with speech through the DPDPA export gate. A human confirms non-personal or masks it; a wired detector
    replaces this with real segments."""
    n = len(audio_signal) if audio_signal is not None else 0
    if n == 0 or sample_rate <= 0:
        return []
    duration_ns = int(n / sample_rate * 1_000_000_000)
    if duration_ns <= 0:
        return []
    # WIRE: a real VAD adapter returns precise segments here; until then, treat the entire clip as personal.
    return [{"t_start_ns": 0, "t_end_ns": duration_ns, "method": "fail_closed_wholeclip"}]


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
