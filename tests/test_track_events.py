"""Track events: typed spans within a track.

`Track.intents` is the same idea without extent and holds 0 rows across 11,406 tracks. An intent can say a
track cut in; it cannot say a 93-frame track cut in over frames 40 to 55. The assertions that matter most
here are the two that guard against silent wrongness rather than failure: that the event vocabulary cannot
fork from the intent vocabulary it was built on, and that a span cannot quietly straddle two drives.
"""

import uuid

import pytest

from core.timebase import now_ns, seconds_to_ns

pytestmark = pytest.mark.db


async def _seed(db, *, n_frames=6, class_name="bus", dt_s=0.5):
    """A session, a track and n frames half a second apart. Returns (session_id, track_id, [frame_ids])."""
    from db.models import Frame, OntologyClass, OntologyVersion
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

    t0, sid = now_ns(), uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id="EVT-1", start_ts_ns=t0,
                     end_ts_ns=t0 + seconds_to_ns(60), city="BLR", sensors={},
                     ontology_version=onto.version))
    await db.flush()
    fids = []
    for i in range(n_frames):
        fid = uuid.uuid4()
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=t0 + int(i * dt_s * 1e9), cam_id="cam_f",
                     img_uri=f"s3://x/{i}.jpg", width=1920, height=1080, quality=0.9))
        fids.append(fid)
    tid = uuid.uuid4()
    db.add(Track(track_id=tid, session_id=sid, class_id=onto.by_name(class_name).id,
                 first_ts_ns=t0, last_ts_ns=t0 + int(n_frames * dt_s * 1e9)))
    await db.flush()
    return sid, tid, fids


def _user(role="reviewer"):
    from types import SimpleNamespace

    return SimpleNamespace(name=f"evt-{uuid.uuid4().hex[:6]}", user_id=None, role=role)


class TestTheVocabularyCannotFork:
    def test_every_governed_intent_is_an_event_type(self):
        """The reason the event vocabulary reuses the intent spellings verbatim.

        `cut_in` means one thing in this system. A second vocabulary spelling it `cutting_in` would split
        every query that ever asks for it, and nothing would report the split as a problem. If this fails,
        one of the two lists was edited without the other.
        """
        from services.domain import track_event_spec
        from services.intelligence.intent import VEHICLE_INTENTS, VRU_INTENTS

        assert set(VRU_INTENTS) | set(VEHICLE_INTENTS) <= track_event_spec("av").names()

    def test_a_domain_may_declare_no_events_at_all(self):
        from services.domain import track_event_spec, validate_track_event_type

        assert track_event_spec("sec") is None
        assert validate_track_event_type("hard_brake", "vru", "sec") != []

    def test_every_type_carries_a_definition_an_annotator_can_act_on(self):
        """The definition is the interface: it is what somebody reads while deciding whether a span starts.
        A blank or one-word entry produces labels two people disagree about silently."""
        from services.domain import track_event_spec

        for t in track_event_spec("av").types:
            assert len(t.definition.split()) >= 8, t.name
            assert t.definition.endswith("."), t.name

    def test_exactly_two_types_are_marked_proposable(self):
        """Marked, not inferred. The flag is what stops a later pass writing twenty-one more heuristics for
        manoeuvres a monocular estimate cannot see."""
        from services.domain import track_event_spec

        assert sorted(t.name for t in track_event_spec("av").types if t.proposable) == [
            "hard_brake", "stopping_in_live_lane"]

    def test_applicability_is_enforced_not_merely_displayed(self):
        from services.domain import validate_track_event_type

        assert validate_track_event_type("lane_splitting", "vru") != []
        assert validate_track_event_type("lane_splitting", "two_wheeler") == []
        assert validate_track_event_type("crossing", "vru") == []      # applies_to any
        assert validate_track_event_type("not_a_thing", "vru") != []


class TestWriting:
    @pytest.mark.asyncio
    async def test_a_span_is_stored_with_its_timestamps_resolved(self):
        from db.session import get_sessionmaker
        from services.api.routers.track_events import TrackEventIn, create_event

        async with get_sessionmaker()() as db:
            _, tid, fids = await _seed(db)
            await db.commit()
            out = await create_event(tid, TrackEventIn(event_type="stopping_in_live_lane",
                                                       start_frame_id=fids[1], end_frame_id=fids[4]),
                                     db=db, user=_user())
        assert out["state"] == "accepted" and out["source"] == "human"
        assert out["end_ts_ns"] > out["start_ts_ns"]

    @pytest.mark.asyncio
    async def test_a_backwards_drag_is_ordered_rather_than_refused(self):
        """A drag runs whichever way the annotator moved. Refusing right-to-left is a bug report from every
        annotator who makes one."""
        from db.session import get_sessionmaker
        from services.api.routers.track_events import TrackEventIn, create_event

        async with get_sessionmaker()() as db:
            _, tid, fids = await _seed(db)
            await db.commit()
            out = await create_event(tid, TrackEventIn(event_type="hard_brake",
                                                       start_frame_id=fids[4], end_frame_id=fids[1]),
                                     db=db, user=_user())
        assert out["start_frame_id"] == str(fids[1]) and out["end_frame_id"] == str(fids[4])

    @pytest.mark.asyncio
    async def test_a_span_cannot_straddle_two_sessions(self):
        """No foreign key can say this. A frame and a track can both exist and belong to different drives,
        and the resulting event is not wrong-looking in the database, it is unqueryable."""
        from fastapi import HTTPException

        from db.session import get_sessionmaker
        from services.api.routers.track_events import TrackEventIn, create_event

        async with get_sessionmaker()() as db:
            _, tid, fids = await _seed(db)
            _, _, other = await _seed(db)
            await db.commit()
            with pytest.raises(HTTPException) as ex:
                await create_event(tid, TrackEventIn(event_type="hard_brake", start_frame_id=fids[0],
                                                     end_frame_id=other[3]), db=db, user=_user())
        assert ex.value.status_code == 400
        assert "session" in str(ex.value.detail)

    @pytest.mark.asyncio
    async def test_an_event_the_track_class_cannot_carry_is_refused(self):
        from fastapi import HTTPException

        from db.session import get_sessionmaker
        from services.api.routers.track_events import TrackEventIn, create_event

        async with get_sessionmaker()() as db:
            _, tid, fids = await _seed(db, class_name="pedestrian")
            await db.commit()
            with pytest.raises(HTTPException) as ex:
                await create_event(tid, TrackEventIn(event_type="lane_splitting", start_frame_id=fids[0],
                                                     end_frame_id=fids[2]), db=db, user=_user())
        assert ex.value.status_code == 400

    @pytest.mark.asyncio
    async def test_accepting_a_proposal_keeps_the_record_of_who_proposed_it(self):
        """Overwriting source with `human` on accept would make every accepted proposal indistinguishable
        from a hand-drawn span, and the proposer's precision unmeasurable."""
        from db.models import TrackEvent
        from db.session import get_sessionmaker
        from services.api.routers.track_events import TrackEventPatch, update_event

        async with get_sessionmaker()() as db:
            _, tid, fids = await _seed(db)
            ev = TrackEvent(track_id=tid, event_type="hard_brake", start_frame_id=fids[0],
                            end_frame_id=fids[2], start_ts_ns=1, end_ts_ns=2, source="heuristic",
                            state="proposed", confidence=0.5)
            db.add(ev)
            await db.commit()
            out = await update_event(ev.event_id, TrackEventPatch(state="accepted"), db=db, user=_user())
        assert out["state"] == "accepted" and out["source"] == "heuristic"

    @pytest.mark.asyncio
    async def test_the_listing_carries_the_vocabulary_with_applicability_resolved(self):
        from db.session import get_sessionmaker
        from services.api.routers.track_events import list_events

        async with get_sessionmaker()() as db:
            _, tid, _ = await _seed(db, class_name="pedestrian")
            await db.commit()
            out = await list_events(tid, db=db)
        by = {t["name"]: t for t in out["event_types"]}
        assert out["class_l1"] == "vru"
        assert by["crossing"]["applicable"] is True
        assert by["lane_splitting"]["applicable"] is False


class TestProposers:
    """Pure-function tests. The heuristics are the part worth pinning; the write path is covered above."""

    def _samples(self, speeds, *, dt_s=0.2, lateral=1.0):
        from services.autolabel.event_proposals import Sample

        return [Sample(uuid.uuid4(), int(i * dt_s * 1e9), s, lateral) for i, s in enumerate(speeds)]

    def test_a_steady_speed_proposes_nothing(self):
        from core.config import get_settings
        from services.autolabel.event_proposals import propose_hard_brake

        assert propose_hard_brake(self._samples([40.0] * 10), get_settings().track_events) == []

    def test_a_sharp_drop_proposes_one_event_not_one_per_sample(self):
        """A long deceleration satisfies the window at many starting points. One brake is one event, and the
        version of this that emitted a span per qualifying pair gave a reviewer nine rejections to make."""
        from core.config import get_settings
        from services.autolabel.event_proposals import propose_hard_brake

        spans = propose_hard_brake(self._samples([50, 45, 38, 28, 16, 6, 5, 5, 5, 5]),
                                   get_settings().track_events)
        assert len(spans) == 1
        assert spans[0].evidence["drop_kmh"] > 12

    def test_the_span_ends_when_the_speed_stops_falling(self):
        from core.config import get_settings
        from services.autolabel.event_proposals import propose_hard_brake

        s = self._samples([50, 40, 30, 10, 10, 10, 10])
        spans = propose_hard_brake(s, get_settings().track_events)
        # The event is the deceleration, not the low speed afterwards.
        assert spans[0].end_ts_ns == s[3].ts_ns

    def test_a_two_sample_drop_is_not_a_brake(self):
        """5,553 of the first sweep's 16,436 proposals were built on two samples, which is exactly the shape
        one bad estimate makes."""
        from core.config import get_settings
        from services.autolabel.event_proposals import propose_hard_brake

        assert propose_hard_brake(self._samples([50, 10, 10, 10, 10, 10]),
                                  get_settings().track_events) == []

    def test_a_spike_that_rebounds_is_not_a_brake(self):
        """The measured noise floor is a median 9.1 km/h between consecutive samples, so a real brake is
        about one sigma and size alone cannot separate them. A vehicle that braked is still slow a moment
        later; an estimator that glitched is back where it was."""
        from core.config import get_settings
        from services.autolabel.event_proposals import propose_hard_brake

        assert propose_hard_brake(self._samples([50, 44, 38, 20, 50, 50, 50]),
                                  get_settings().track_events) == []

    def test_a_real_brake_that_holds_is_still_proposed(self):
        """The check must not reject the thing it exists to find."""
        from core.config import get_settings
        from services.autolabel.event_proposals import propose_hard_brake

        spans = propose_hard_brake(self._samples([50, 44, 36, 22, 8, 6, 6, 6, 6, 6]),
                                   get_settings().track_events)
        assert len(spans) == 1

    def test_an_implausible_speed_is_not_a_measurement(self):
        """23% of dynamics rows read over 60 km/h on dashcam footage of Bengaluru city traffic, and the
        column is clipped at 150."""
        from types import SimpleNamespace

        from core.config import get_settings
        from services.autolabel.event_proposals import _usable

        rows = [SimpleNamespace(frame_id=uuid.uuid4(), ts_ns=i, speed_kmh=140.0, lateral_m=1.0,
                                confidence=0.9) for i in range(10)]
        assert _usable(rows, get_settings().track_events) == []

    def test_too_few_samples_proposes_nothing(self):
        from core.config import get_settings
        from services.autolabel.event_proposals import propose_hard_brake

        assert propose_hard_brake(self._samples([50, 10]), get_settings().track_events) == []

    def test_a_long_stop_near_the_ego_path_is_proposed(self):
        from core.config import get_settings
        from services.autolabel.event_proposals import propose_stopping_in_live_lane

        spans = propose_stopping_in_live_lane(
            self._samples([1.0] * 20, dt_s=0.25, lateral=1.5), get_settings().track_events)
        assert len(spans) == 1 and spans[0].evidence["held_s"] >= 2.0

    def test_the_same_stop_at_the_kerb_is_not(self):
        """The lateral bound is the whole discriminator: a vehicle stopped 8m to the side is parked."""
        from core.config import get_settings
        from services.autolabel.event_proposals import propose_stopping_in_live_lane

        assert propose_stopping_in_live_lane(
            self._samples([1.0] * 20, dt_s=0.25, lateral=8.0), get_settings().track_events) == []

    def test_a_missing_lateral_estimate_ends_the_run_rather_than_extending_it(self):
        """An unbounded stop is exactly the parked case. Guessing it into the live lane is the wrong
        direction to be wrong in."""
        from core.config import get_settings
        from services.autolabel.event_proposals import Sample, propose_stopping_in_live_lane

        s = [Sample(uuid.uuid4(), int(i * 0.25e9), 1.0, None) for i in range(20)]
        assert propose_stopping_in_live_lane(s, get_settings().track_events) == []

    def test_a_low_confidence_dynamics_row_is_dropped(self):
        from types import SimpleNamespace

        from core.config import get_settings
        from services.autolabel.event_proposals import _usable

        rows = [SimpleNamespace(frame_id=uuid.uuid4(), ts_ns=i, speed_kmh=10.0, lateral_m=1.0,
                                confidence=0.1) for i in range(10)]
        assert _usable(rows, get_settings().track_events) == []

    @pytest.mark.asyncio
    async def test_rerunning_over_a_track_does_not_stack_duplicates(self):
        from sqlalchemy import select

        from db.models import Frame, Object, ObjectDynamics
        from db.session import get_sessionmaker
        from services.autolabel.ontology import get_ontology
        from services.autolabel.event_proposals import propose_for_track

        async with get_sessionmaker()() as db:
            sid, tid, fids = await _seed(db, n_frames=12, dt_s=0.25)
            frames = sorted((await db.execute(select(Frame).where(Frame.session_id == sid))).scalars(),
                            key=lambda f: f.ts_ns)
            # Real objects: object_dynamics keys on object_id, so a synthetic id would only prove the
            # proposer runs on rows that could not exist.
            cid = get_ontology().by_name("bus").id
            speeds = [50, 45, 38, 28, 16, 6, 5, 5, 5, 5, 5, 5]
            for f, sp in zip(frames, speeds, strict=False):
                oid = uuid.uuid4()
                db.add(Object(object_id=oid, frame_id=f.frame_id, track_id=tid, class_id=cid,
                              bbox=[10.0, 10.0, 60.0, 60.0], conf=0.9, source="fused", state="review"))
                await db.flush()
                db.add(ObjectDynamics(object_id=oid, track_id=tid, frame_id=f.frame_id,
                                      ts_ns=f.ts_ns, speed_kmh=float(sp), lateral_m=1.0, confidence=0.6))
            await db.commit()
            first = await propose_for_track(db, tid)
            second = await propose_for_track(db, tid)
        assert first["proposed"], first
        assert second["proposed"] == []
