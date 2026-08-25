"""WP5: what a COCO export carries beyond boxes.

Deliberately its own file rather than an addition to `test_export_import_roundtrip`, which is
`xfail(strict=True)` on an environmental fixture problem: adding assertions there would flip a strict xfail
the moment they passed and turn a working test into a failing one.

The theme is that a fact already in the database is not a fact in the export. Relations have been on
`ExportRecord` since they were added and only Parquet ever wrote them, so every COCO consumer has read a
rider and their motorcycle as two unrelated boxes.
"""

import json
import uuid

import pytest

from core.timebase import now_ns, seconds_to_ns

pytestmark = pytest.mark.db


async def _seed(db, *, with_event=True):
    """A frame with scene context, two related objects on a track, and one accepted track event."""
    from db.models import Frame, Object, ObjectRelationship, OntologyClass, OntologyVersion
    from db.models import Session as DbSession
    from db.models import Track, TrackEvent
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    if await db.get(OntologyVersion, onto.version) is None:
        db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
        await db.flush()
        for c in onto.classes:
            db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                 india=c.india, map_to={}))
        await db.flush()

    t0, sid, fid, tid = now_ns(), uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id="WP5-1", start_ts_ns=t0,
                     end_ts_ns=t0 + seconds_to_ns(5), city="BLR", sensors={},
                     ontology_version=onto.version))
    await db.flush()
    db.add(Frame(frame_id=fid, session_id=sid, ts_ns=t0, cam_id="cam_f",
                 img_uri=f"s3://x/{fid}.jpg", width=640, height=480, quality=0.9,
                 scene={"weather": "rain", "waterlogging": True, "rarity": 0.77}))
    db.add(Track(track_id=tid, session_id=sid, class_id=onto.by_name("motorcycle").id,
                 first_ts_ns=t0, last_ts_ns=t0 + seconds_to_ns(5)))
    await db.flush()

    bike, rider = uuid.uuid4(), uuid.uuid4()
    db.add(Object(object_id=bike, frame_id=fid, track_id=tid, class_id=onto.by_name("motorcycle").id,
                  bbox=[10.0, 10.0, 90.0, 90.0], conf=0.9, source="fused", state="accepted"))
    db.add(Object(object_id=rider, frame_id=fid, class_id=onto.by_name("rider").id,
                  bbox=[20.0, 5.0, 70.0, 60.0], conf=0.9, source="fused", state="accepted"))
    await db.flush()
    db.add(ObjectRelationship(from_object_id=rider, to_object_id=bike, frame_id=fid, kind="rider_of"))
    if with_event:
        db.add(TrackEvent(track_id=tid, event_type="lane_splitting", start_frame_id=fid, end_frame_id=fid,
                          start_ts_ns=t0 - 1, end_ts_ns=t0 + 1, source="human", state="accepted"))
        # A proposal on the same track, which must NOT ship: training on an unreviewed heuristic is how a
        # threshold becomes a label.
        db.add(TrackEvent(track_id=tid, event_type="hard_brake", start_frame_id=fid, end_frame_id=fid,
                          start_ts_ns=t0 - 1, end_ts_ns=t0 + 1, source="heuristic", state="proposed",
                          confidence=0.6))
    await db.commit()
    return sid, fid, str(bike), str(rider)


async def _records(session_id):
    from services.export.dataset import SliceSpec, fetch_records

    return await fetch_records(SliceSpec(name="wp5", session_id=str(session_id)))


class TestTheRecordCarriesWhatTheDatabaseKnows:
    @pytest.mark.asyncio
    async def test_relations_reach_the_export_record(self):
        from db.session import get_sessionmaker

        async with get_sessionmaker()() as db:
            sid, _, bike, rider = await _seed(db)
        recs = {str(r.object_id): r for r in await _records(sid)}
        assert recs[rider].relationships == [{"to_object_id": bike, "kind": "rider_of"}]
        assert recs[bike].relationships == []

    @pytest.mark.asyncio
    async def test_frame_context_rides_on_every_record_from_that_frame(self):
        from db.session import get_sessionmaker

        async with get_sessionmaker()() as db:
            sid, _, _, _ = await _seed(db)
        recs = await _records(sid)
        assert recs and all(r.context["weather"] == "rain" for r in recs)
        assert all(r.context["waterlogging"] is True for r in recs)

    @pytest.mark.asyncio
    async def test_only_accepted_events_ship(self):
        from db.session import get_sessionmaker

        async with get_sessionmaker()() as db:
            sid, _, bike, _ = await _seed(db)
        recs = {str(r.object_id): r for r in await _records(sid)}
        types = {e["event_type"] for e in recs[bike].track_events}
        assert types == {"lane_splitting"}, types

    @pytest.mark.asyncio
    async def test_an_event_that_does_not_cover_this_frame_is_not_attached(self):
        """A track-level list would say this object was lane-splitting at some point in its life, which is
        not what a consumer filtering for lane-splitting frames is asking."""
        from sqlalchemy import update

        from db.models import TrackEvent
        from db.session import get_sessionmaker

        async with get_sessionmaker()() as db:
            sid, fid, bike, _ = await _seed(db)
            # Move the accepted event's span well past this frame.
            await db.execute(update(TrackEvent).where(TrackEvent.event_type == "lane_splitting")
                             .values(start_ts_ns=now_ns() + seconds_to_ns(60),
                                     end_ts_ns=now_ns() + seconds_to_ns(70)))
            await db.commit()
        recs = {str(r.object_id): r for r in await _records(sid)}
        assert recs[bike].track_events == []


class TestCoco:
    @pytest.mark.asyncio
    async def test_the_written_file_carries_relations_events_and_context(self, tmp_path):
        """The round trip that matters: a fact in the database is not a fact in the export until a writer
        emits it, and only Parquet ever emitted relations."""
        from core.storage import get_object_store
        from db.session import get_sessionmaker
        from services.autolabel.ontology import get_ontology
        from services.export.adapter_coco import write_coco

        async with get_sessionmaker()() as db:
            sid, _, bike, rider = await _seed(db)
        recs = await _records(sid)
        path = write_coco(recs, get_ontology(), get_object_store(), tmp_path)
        coco = json.loads(path.read_text())

        by_obj = {a["labelox"]["object_id"]: a for a in coco["annotations"]}
        assert by_obj[rider]["labelox"]["relationships"] == [{"to_object_id": bike, "kind": "rider_of"}]
        # Keyed on the object_id the same block emits, so a reader joins them without an index of its own.
        assert by_obj[bike]["labelox"]["object_id"] == bike

        events = by_obj[bike]["labelox"]["track_events"]
        assert [e["event_type"] for e in events] == ["lane_splitting"]

        # Context on the image, not on each annotation: N copies per frame is a file that can contradict
        # itself, which is the same reason `split` lives there.
        assert len(coco["images"]) == 1
        assert coco["images"][0]["labelox_context"]["weather"] == "rain"
        assert all("labelox_context" not in a["labelox"] for a in coco["annotations"])

    @pytest.mark.asyncio
    async def test_it_stays_valid_coco(self, tmp_path):
        """Everything added lives under the `labelox` extension block or an underscored image key, so a
        stock COCO reader is unaffected."""
        from core.storage import get_object_store
        from db.session import get_sessionmaker
        from services.autolabel.ontology import get_ontology
        from services.export.adapter_coco import write_coco

        async with get_sessionmaker()() as db:
            sid, _, _, _ = await _seed(db)
        coco = json.loads(write_coco(await _records(sid), get_ontology(),
                                     get_object_store(), tmp_path).read_text())
        assert {"info", "images", "annotations", "categories"} <= set(coco)
        for a in coco["annotations"]:
            assert {"id", "image_id", "category_id", "bbox", "area", "iscrowd"} <= set(a)
            assert len(a["bbox"]) == 4 and a["bbox"][2] >= 0 and a["bbox"][3] >= 0
        for im in coco["images"]:
            assert {"id", "file_name", "width", "height"} <= set(im)


class TestTheNewSliceFilters:
    @pytest.mark.asyncio
    async def test_a_region_filter_matches_the_code_the_fleet_records(self):
        """The session's city is `BLR`. `cities: ["Bengaluru"]` misses it; `regions` does not."""
        from services.export.dataset import SliceSpec, fetch_records

        from db.session import get_sessionmaker

        async with get_sessionmaker()() as db:
            sid, _, _, _ = await _seed(db)
        got = await fetch_records(SliceSpec(name="r", session_id=str(sid), regions=["Bengaluru"]))
        assert got
        assert not await fetch_records(SliceSpec(name="r", session_id=str(sid), cities=["Bengaluru"]))

    @pytest.mark.asyncio
    async def test_a_context_filter_selects_on_a_scene_axis(self):
        from services.export.dataset import SliceSpec, fetch_records

        from db.session import get_sessionmaker

        async with get_sessionmaker()() as db:
            sid, _, _, _ = await _seed(db)
        assert await fetch_records(SliceSpec(name="c", session_id=str(sid),
                                             context={"weather": ["rain"]}))
        assert not await fetch_records(SliceSpec(name="c", session_id=str(sid),
                                                 context={"weather": ["clear"]}))

    @pytest.mark.asyncio
    async def test_a_rarity_band_selects_on_the_persisted_score(self):
        from services.export.dataset import SliceSpec, fetch_records

        from db.session import get_sessionmaker

        async with get_sessionmaker()() as db:
            sid, _, _, _ = await _seed(db)
        assert await fetch_records(SliceSpec(name="k", session_id=str(sid), rarity_min=0.5))
        assert not await fetch_records(SliceSpec(name="k", session_id=str(sid), rarity_max=0.5))

    @pytest.mark.asyncio
    async def test_an_event_filter_selects_objects_on_the_tracks_that_carry_it(self):
        from services.export.dataset import SliceSpec, fetch_records

        from db.session import get_sessionmaker

        async with get_sessionmaker()() as db:
            sid, _, bike, _ = await _seed(db)
        got = await fetch_records(SliceSpec(name="e", session_id=str(sid),
                                            track_event_types=["lane_splitting"]))
        # Only the object on the track carrying the event; the rider has no track.
        assert {str(r.object_id) for r in got} == {bike}
        # The proposed hard_brake is not selectable by default, for the same reason it does not ship.
        assert not await fetch_records(SliceSpec(name="e", session_id=str(sid),
                                                 track_event_types=["hard_brake"]))
