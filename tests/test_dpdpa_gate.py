"""Milestone D: the unified fail-closed DPDPA gate. Face, plate, and speech are one gate; it refuses on each
condition and passes only when all three are clear. A frame with no anonymization audit is treated as
un-redacted (fail-closed), and confirmed non-personal speech does not block."""

from __future__ import annotations

from services.anonymize.compliance import evaluate_dpdpa


def test_passes_when_all_clear():
    v = evaluate_dpdpa({"f1", "f2"}, {"f1", "f2"}, [{"is_personal": True, "redacted": True}])
    assert v["pass"] and v["blockers"] == []


def test_refuses_an_unaudited_frame_fail_closed():
    v = evaluate_dpdpa({"f1", "f2"}, {"f1"}, [])    # f2 was never anonymized
    assert not v["pass"]
    assert any(b["kind"] == "unredacted_visual_pii" for b in v["blockers"])


def test_refuses_unredacted_personal_speech():
    v = evaluate_dpdpa({"f1"}, {"f1"}, [{"is_personal": True, "redacted": False}])
    assert not v["pass"]
    assert any(b["kind"] == "unredacted_speech" for b in v["blockers"])


def test_confirmed_non_personal_speech_does_not_block():
    v = evaluate_dpdpa({"f1"}, {"f1"}, [{"is_personal": False, "redacted": False}])
    assert v["pass"]


def test_all_three_conditions_block_as_one_gate():
    v = evaluate_dpdpa({"f1", "f2"}, {"f1"}, [{"is_personal": True, "redacted": False}])
    assert not v["pass"]
    assert {b["kind"] for b in v["blockers"]} == {"unredacted_visual_pii", "unredacted_speech"}


async def test_export_gate_on_real_data():
    import uuid

    import pytest
    from sqlalchemy import delete, select

    from db.models import PiiAudit, SpeechSegment
    from db.session import get_sessionmaker
    from services.anonymize.compliance import dpdpa_export_gate
    async with get_sessionmaker()() as db:
        row = (await db.execute(select(PiiAudit.session_id, PiiAudit.frame_id).limit(1))).first()
    if row is None:
        pytest.skip("no PiiAudit data in the corpus")
    sid, fid = row
    assert (await dpdpa_export_gate(sid, [fid]))["pass"]                       # an audited frame passes
    refuse = await dpdpa_export_gate(sid, [fid, uuid.uuid4()])                 # an un-audited frame refuses
    assert not refuse["pass"] and any(b["kind"] == "unredacted_visual_pii" for b in refuse["blockers"])

    async with get_sessionmaker()() as db:
        seg = SpeechSegment(session_id=sid, t_start_ns=0, t_end_ns=1000, is_personal=True, redacted=False)
        db.add(seg)
        await db.commit()
        await db.refresh(seg)
        seg_id = seg.segment_id
    try:
        v = await dpdpa_export_gate(sid, [fid])                               # personal un-redacted speech refuses
        assert not v["pass"] and any(b["kind"] == "unredacted_speech" for b in v["blockers"])
        async with get_sessionmaker()() as db:
            (await db.get(SpeechSegment, seg_id)).redacted = True
            await db.commit()
        assert (await dpdpa_export_gate(sid, [fid]))["pass"]                  # redacted -> clear
    finally:
        async with get_sessionmaker()() as db:
            await db.execute(delete(SpeechSegment).where(SpeechSegment.segment_id == seg_id))
            await db.commit()
