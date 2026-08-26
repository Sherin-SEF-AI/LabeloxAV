"""Gap filling, and the accounting that would have caught the last one.

An earlier corpus fill created 137,913 objects and reported "137,947 gaps filled" - a count of boxes made,
with no way to tell it from a run that made the wrong boxes. These tests pin the two properties that make
that impossible to repeat: a hole whose anchors are not one object is refused with a named reason, and the
result says what it declined as well as what it wrote.
"""

import uuid

import pytest

from core.timebase import now_ns, seconds_to_ns

pytestmark = pytest.mark.db

DET = "fused"


async def _track(db, boxes, *, sources=None, classes=None, dt_s=0.33, cam="cam_f"):
    """A session, a track, and one frame per box. `boxes[i] is None` leaves that frame empty (a hole)."""
    from db.models import Frame, Object, OntologyClass, OntologyVersion
    from db.models import Session as DbSession
    from db.models import Track
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    if await db.get(OntologyVersion, onto.version) is None:
        db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
        await db.flush()
        for c in onto.classes:
            db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                 india=c.india, map_to={}))
        await db.flush()

    t0, sid, tid = now_ns(), uuid.uuid4(), uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id="INT-1", start_ts_ns=t0,
                     end_ts_ns=t0 + seconds_to_ns(60), city="BLR", sensors={},
                     ontology_version=onto.version))
    await db.flush()
    db.add(Track(track_id=tid, session_id=sid, class_id=onto.by_name("sedan").id,
                 first_ts_ns=t0, last_ts_ns=t0 + seconds_to_ns(60)))
    await db.flush()
    for i, bb in enumerate(boxes):
        fid = uuid.uuid4()
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=t0 + int(i * dt_s * 1e9), cam_id=cam,
                     img_uri=f"s3://x/{i}.jpg", width=1920, height=1080, quality=0.9))
        await db.flush()
        if bb is None:
            continue
        cname = (classes or {}).get(i, "sedan")
        db.add(Object(frame_id=fid, track_id=tid, class_id=onto.by_name(cname).id,
                      bbox=[float(v) for v in bb], conf=0.5,
                      source=(sources or {}).get(i, DET), state="review"))
    await db.commit()
    return tid


class TestItFillsWhatItShould:
    @pytest.mark.asyncio
    async def test_a_clean_hole_between_two_detections_is_filled(self):
        from db.session import get_sessionmaker
        from services.temporal.interpolate import interpolate_track_keyframed

        async with get_sessionmaker()() as db:
            tid = await _track(db, [[100, 100, 180, 200], None, None, [160, 110, 245, 215]])
        res = await interpolate_track_keyframed(tid, "linear", anchor_policy="detection")
        assert res["created"] == 2, res
        assert res["refused"] == {}

    @pytest.mark.asyncio
    async def test_the_fill_lands_between_its_anchors(self):
        """The arithmetic was never the defect, and this is the assertion that keeps it that way."""
        from sqlalchemy import select

        from db.models import Object
        from db.session import get_sessionmaker
        from services.temporal.interpolate import interpolate_track_keyframed

        async with get_sessionmaker()() as db:
            tid = await _track(db, [[100, 100, 200, 200], None, [300, 100, 400, 200]])
        await interpolate_track_keyframed(tid, "linear", anchor_policy="detection")
        async with get_sessionmaker()() as db:
            fills = (await db.execute(select(Object).where(
                Object.track_id == tid, Object.source == "interpolated"))).scalars().all()
        assert len(fills) == 1
        # Anchor centres are 150 and 350, so a single fill halfway between them belongs at 250.
        cx = (fills[0].bbox[0] + fills[0].bbox[2]) / 2
        assert 240 < cx < 260, fills[0].bbox

    @pytest.mark.asyncio
    async def test_confidence_decays_into_the_middle_of_a_gap(self):
        """A flat conf on every fill is what let 109,000 bad boxes look like the good ones."""
        from sqlalchemy import select

        from db.models import Frame, Object
        from db.session import get_sessionmaker
        from services.temporal.interpolate import interpolate_track_keyframed

        async with get_sessionmaker()() as db:
            tid = await _track(db, [[100, 100, 180, 200], None, None, None, None,
                                    [180, 110, 265, 215]])
        await interpolate_track_keyframed(tid, "linear", anchor_policy="detection")
        async with get_sessionmaker()() as db:
            rows = (await db.execute(
                select(Object.conf).join(Frame, Frame.frame_id == Object.frame_id)
                .where(Object.track_id == tid, Object.source == "interpolated")
                .order_by(Frame.ts_ns))).scalars().all()
        assert len(rows) == 4
        assert min(rows) < max(rows), f"confidence is flat across the gap: {rows}"

    @pytest.mark.asyncio
    async def test_a_fill_is_never_written_as_accepted(self):
        from sqlalchemy import select

        from db.models import Object
        from db.session import get_sessionmaker
        from services.temporal.interpolate import interpolate_track_keyframed

        async with get_sessionmaker()() as db:
            tid = await _track(db, [[100, 100, 180, 200], None, [160, 110, 245, 215]])
        await interpolate_track_keyframed(tid, "linear", anchor_policy="detection")
        async with get_sessionmaker()() as db:
            states = set((await db.execute(select(Object.state).where(
                Object.track_id == tid, Object.source == "interpolated"))).scalars().all())
        assert states <= {"annotate"}, states


class TestItRefusesWhatItShould:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("boxes,classes,reason", [
        ([[100, 100, 180, 200], None, [1500, 100, 1580, 200]], None, "endpoints_teleport"),
        ([[800, 400, 880, 500], None, [700, 300, 1000, 620]], None, "endpoints_scale_jump"),
        ([[100, 100, 180, 200], None, [160, 110, 245, 215]], {2: "pedestrian"},
         "endpoints_class_mismatch"),
    ])
    async def test_an_implausible_hole_is_refused_by_name(self, boxes, classes, reason):
        from db.session import get_sessionmaker
        from services.temporal.interpolate import interpolate_track_keyframed

        async with get_sessionmaker()() as db:
            tid = await _track(db, boxes, classes=classes)
        res = await interpolate_track_keyframed(tid, "linear", anchor_policy="detection")
        assert res["created"] == 0, res
        assert res["refused"] == {reason: 1}, res
        assert res["refused_frames"] == 1

    @pytest.mark.asyncio
    async def test_gate_false_lets_a_person_override_it(self):
        """The editor's "interpolate between the two boxes I just drew" is an assertion by a human that
        they are one object, and the gate exists to second-guess the tracker, not the annotator."""
        from db.session import get_sessionmaker
        from services.temporal.interpolate import interpolate_track_keyframed

        async with get_sessionmaker()() as db:
            tid = await _track(db, [[100, 100, 180, 200], None, [1500, 100, 1580, 200]])
        res = await interpolate_track_keyframed(tid, "linear", anchor_policy="detection", gate=False)
        assert res["created"] == 1, res


class TestAnchorPolicy:
    @pytest.mark.asyncio
    async def test_a_fill_never_anchors_on_another_fill(self):
        """The old implementation treated every object as an anchor, so its own output re-anchored the next
        run and the error compounded."""
        from db.session import get_sessionmaker
        from services.temporal.interpolate import interpolate_track_keyframed

        async with get_sessionmaker()() as db:
            # The middle box is a previous fill sitting far away; anchoring on it would bridge to it.
            tid = await _track(db, [[100, 100, 180, 200], [1500, 100, 1580, 200], None,
                                    [160, 110, 245, 215]],
                               sources={1: "interpolated"})
        res = await interpolate_track_keyframed(tid, "linear", anchor_policy="detection")
        # Anchors are the two detections only, so this is one clean hole of two frames.
        assert res["anchors"] == 2, res
        assert res["created"] == 2, res

    @pytest.mark.asyncio
    async def test_the_keyframe_policy_needs_human_anchors_and_says_so(self):
        from db.session import get_sessionmaker
        from services.temporal.interpolate import interpolate_track_keyframed

        async with get_sessionmaker()() as db:
            tid = await _track(db, [[100, 100, 180, 200], None, [160, 110, 245, 215]])
        res = await interpolate_track_keyframed(tid, "linear", anchor_policy="keyframe")
        assert res["created"] == 0
        assert "keyframe anchors" in res["reason"]

    @pytest.mark.asyncio
    async def test_an_unknown_policy_raises_rather_than_filling_everything(self):
        from db.session import get_sessionmaker
        from services.temporal.interpolate import interpolate_track_keyframed

        async with get_sessionmaker()() as db:
            tid = await _track(db, [[100, 100, 180, 200], None, [160, 110, 245, 215]])
        with pytest.raises(ValueError, match="unknown anchor policy"):
            await interpolate_track_keyframed(tid, "linear", anchor_policy="everything")


class TestTheTrackerDoesNotEatItsOwnOutput:
    def test_retrack_selects_detector_sources_only(self):
        """A bad association creates a fill; the fill becomes evidence for the same association on the next
        retrack; nothing in the loop can tell the difference."""
        import inspect

        from services.autolabel.track import assign

        assert "interpolated" not in assign._DETECTION_SOURCES
        assert "propagated" not in assign._DETECTION_SOURCES
        assert "_DETECTION_SOURCES" in inspect.getsource(assign.retrack_session)
