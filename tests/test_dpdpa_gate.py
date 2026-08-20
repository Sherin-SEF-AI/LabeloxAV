"""Milestone D: the unified fail-closed DPDPA gate. Face, plate, and speech are one gate; it refuses on each
condition and passes only when all three are clear. A frame with no anonymization audit is treated as
un-redacted (fail-closed), and confirmed non-personal speech does not block."""

from __future__ import annotations

import pytest

from services.anonymize.compliance import evaluate_dpdpa

pytestmark = pytest.mark.db


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
    """The gate against database rows rather than in-memory arguments, so the queries are exercised too.

    This used to take the first PiiAudit row it found in the corpus and skip when there was none. Once the
    suite stopped inheriting residue from previous runs there never was one, so a compliance gate that
    refuses exports went from covered to silently skipped. It seeds its own session, frame and audit now,
    which also makes the passing case deterministic: it no longer depends on whichever session happened to
    be first, or on that session having no unredacted speech attached by some other test.
    """
    import uuid

    from sqlalchemy import delete

    from core.timebase import now_ns, seconds_to_ns
    from db.models import Frame, PiiAudit, SpeechSegment
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.anonymize.compliance import dpdpa_export_gate
    from services.autolabel.ontology import get_ontology

    sid, fid, ts = uuid.uuid4(), uuid.uuid4(), now_ns()
    async with get_sessionmaker()() as db:
        db.add(DbSession(session_id=sid, vehicle_id="CP-01", start_ts_ns=ts, end_ts_ns=ts + seconds_to_ns(1),
                         city="BLR", sensors={}, ontology_version=get_ontology().version))
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri="s3://x/f.jpg",
                     width=320, height=240, quality=0.9, scene={}))
        await db.flush()
        db.add(PiiAudit(frame_id=fid, session_id=sid, n_faces=1, n_plates=0,
                        regions=[{"type": "face", "bbox": [1.0, 1.0, 10.0, 10.0], "score": 0.9}],
                        method_version="test", ts_ns=ts))
        await db.commit()
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


def test_all_four_conditions_block_as_one_gate():
    """The coverage gap is a blocker like any other when the gate is enforcing.

    The three-condition test above keeps its `==` deliberately: adding a blocker kind has to be a decision
    someone makes here, and it also pins that the coverage gate stays out of a verdict that did not ask for
    it. That test needed no edit when this landed, which is the evidence that mode="off" is inert.
    """
    v = evaluate_dpdpa({"f1", "f2"}, {"f1"}, [{"is_personal": True, "redacted": False}],
                       coverage_gaps=[{"frame_id": "f1", "missing": {"face": 1},
                                       "classes": ["pedestrian"], "examined": True}],
                       mode="enforcing")
    assert not v["pass"]
    assert {b["kind"] for b in v["blockers"]} == {"unredacted_visual_pii", "unredacted_speech",
                                                  "unverified_blur_coverage"}


def test_advisory_measures_without_refusing():
    v = evaluate_dpdpa({"f1"}, {"f1"}, [],
                       coverage_gaps=[{"frame_id": "f1", "missing": {"face": 1},
                                       "classes": ["pedestrian"], "examined": False}],
                       mode="advisory")
    assert v["pass"], "advisory must not refuse; it is a measurement while the corpus is swept"
    assert [w["kind"] for w in v["warnings"]] == ["unverified_blur_coverage"]
    assert v["blockers"] == []


def test_the_blocker_frame_list_is_capped_but_the_count_is_not():
    # A whole-slice refusal must not put 40,000 frame records into a 422 body.
    gaps = [{"frame_id": f"f{i}", "missing": {"face": 1}, "classes": ["pedestrian"], "examined": False}
            for i in range(200)]
    v = evaluate_dpdpa({"f1"}, {"f1"}, [], coverage_gaps=gaps, mode="enforcing")
    blocker = next(b for b in v["blockers"] if b["kind"] == "unverified_blur_coverage")
    assert blocker["count"] == 200
    assert len(blocker["frames"]) == 25 and blocker["truncated"] is True


async def test_coverage_gate_on_real_frames():
    """Three frames that separate the ways this predicate can be wrong.

    A - a pedestrian with a face region actually covering them: passes.
    B - the same pedestrian, an audit row with no regions at all: refuses. This is the 82.4% population,
        and under the old row-existence gate it passed identically to A.
    C - no annotations and a zero-count audit: passes. recheck.py writes exactly this row for "looked,
        found nothing", so a gate keyed on n_faces > 0 would refuse every clean highway frame in the
        corpus. This is the assertion that matters most.
    """
    import uuid

    from core.timebase import now_ns, seconds_to_ns
    from db.models import Frame, Object, PiiAudit
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.anonymize.compliance import blur_coverage_gaps
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    ped = onto.by_name("pedestrian").id
    person = [600.0, 300.0, 760.0, 900.0]
    face_on_person = [650.0, 330.0, 720.0, 410.0]

    sid, ts = uuid.uuid4(), now_ns()
    fa, fb, fc = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    async with get_sessionmaker()() as db:
        db.add(DbSession(session_id=sid, vehicle_id="COV-01", start_ts_ns=ts,
                         end_ts_ns=ts + seconds_to_ns(1), city="BLR", sensors={},
                         ontology_version=onto.version))
        for fid in (fa, fb, fc):
            db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri="s3://x/f.jpg",
                         width=1920, height=1080, quality=0.9, scene={}))
        await db.flush()
        for fid in (fa, fb):
            db.add(Object(object_id=uuid.uuid4(), frame_id=fid, class_id=ped, bbox=person,
                          conf=0.9, source="human", state="accepted"))
        db.add(PiiAudit(frame_id=fa, session_id=sid, n_faces=1, n_plates=0,
                        regions=[{"type": "face", "bbox": face_on_person, "score": 0.9}],
                        method_version="test", ts_ns=ts))
        db.add(PiiAudit(frame_id=fb, session_id=sid, n_faces=0, n_plates=0, regions=[],
                        method_version="test", ts_ns=ts))
        db.add(PiiAudit(frame_id=fc, session_id=sid, n_faces=0, n_plates=0, regions=[],
                        method_version="test", ts_ns=ts))
        await db.commit()

        gaps = await blur_coverage_gaps(db, [fa, fb, fc])

    by_frame = {g["frame_id"]: g for g in gaps}
    assert str(fa) not in by_frame, "a covering face region must satisfy the person it covers"
    assert str(fc) not in by_frame, "a frame with no annotated people cannot be short a redaction"
    assert str(fb) in by_frame, "an annotated pedestrian with no region at all must be reported"
    assert by_frame[str(fb)]["missing"] == {"face": 1}
    assert by_frame[str(fb)]["classes"] == ["pedestrian"]
    assert by_frame[str(fb)]["examined"] is False


async def test_the_coverage_query_does_not_scale_with_frames():
    """One query pair for the whole export, not one per frame.

    export_dataset hands this whole slices, so an N+1 here would be an N+1 inside the gate that blocks
    delivery. Counting statements rather than timing, so it fails for the right reason.
    """
    import uuid

    from sqlalchemy import event

    from core.timebase import now_ns, seconds_to_ns
    from db.models import Frame, Object, PiiAudit
    from db.models import Session as DbSession
    from db.session import get_engine, get_sessionmaker
    from services.anonymize.compliance import blur_coverage_gaps
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    ped = onto.by_name("pedestrian").id
    sid, ts = uuid.uuid4(), now_ns()

    async with get_sessionmaker()() as db:
        db.add(DbSession(session_id=sid, vehicle_id="COV-N", start_ts_ns=ts,
                         end_ts_ns=ts + seconds_to_ns(1), city="BLR", sensors={},
                         ontology_version=onto.version))
        await db.commit()

    async def seed(n):
        fids = []
        async with get_sessionmaker()() as db:
            for _ in range(n):
                fid = uuid.uuid4()
                db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f",
                             img_uri="s3://x/f.jpg", width=1920, height=1080, quality=0.9, scene={}))
                await db.flush()
                db.add(Object(object_id=uuid.uuid4(), frame_id=fid, class_id=ped,
                              bbox=[600.0, 300.0, 760.0, 900.0], conf=0.9, source="human",
                              state="accepted"))
                db.add(PiiAudit(frame_id=fid, session_id=sid, n_faces=0, n_plates=0, regions=[],
                                method_version="test", ts_ns=ts))
                fids.append(fid)
            await db.commit()
        return fids

    few, many = await seed(3), await seed(40)

    counts = []
    for batch in (few, many):
        n = 0

        def _count(*_a, **_k):
            nonlocal n
            n += 1

        engine = get_engine().sync_engine
        event.listen(engine, "before_cursor_execute", _count)
        try:
            async with get_sessionmaker()() as db:
                await blur_coverage_gaps(db, batch)
        finally:
            event.remove(engine, "before_cursor_execute", _count)
        counts.append(n)

    assert counts[0] == counts[1], (
        f"statement count grew with the slice ({counts[0]} for 3 frames, {counts[1]} for 40); "
        "the coverage gate has become an N+1")
