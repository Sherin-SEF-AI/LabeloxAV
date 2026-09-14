"""Which object is in front, from depth that was already there, and an amodal box that is never guessed.

Both halves of the occlusion work existed and were never joined. `ObjectRelationship` has allowed
`kind="occludes"` with `source="geometry"` since relations were added and nothing ever wrote one: zero
rows corpus-wide. `ObjectDynamics.distance_m` holds a ground-contact depth for 367,000 objects across
33,767 frames. Two boxes that overlap in the image and differ in depth have an order, and nothing asked.

The threshold is the interesting part, and getting it right needed the corpus rather than intuition. A
flat 40 m cutoff, which sounds reasonable, ordered ZERO pairs on the first three real frames tried: the
overlapping objects here sit at 58, 76 and 99 m, well inside what `ipm_max_range_m` already permits. The
rule that works is the one a stored depth already obeys, scaled with the estimate: `compute.py` derives
the lift's error as df/f = f/(fy*h) and only stores a distance while that stays under 25%, so two depths
are distinguishable only when they differ by more than the sum of their errors. That orders 34% of
overlapping pairs and leaves 66% open.

Leaving them open is the point. A wrong occlusion order is invisible in the label and wrong in every
consumer that reads it, so "these two overlap and this method cannot tell which is in front" has to be a
returnable answer rather than a coin toss.

The amodal tests say the same thing about a different field: `bbox` stays the visible extent, and
`bbox_amodal` is null until a person judges it. Nothing derives one from a class prior, because a guess
and an observation would be indistinguishable afterwards.
"""

import uuid

import pytest

from core.timebase import now_ns

pytestmark = pytest.mark.db


async def _frame(db, onto):
    from db.models import Frame, OntologyClass, OntologyVersion
    from db.models import Session as DbSession

    if await db.get(OntologyVersion, onto.version) is None:
        db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
        await db.flush()
        for c in onto.classes:
            db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                 india=c.india, map_to={}))
        await db.flush()
    ts, sid, fid = now_ns(), uuid.uuid4(), uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id="OCC-1", start_ts_ns=ts, end_ts_ns=ts + 1,
                     city="BLR", sensors={}, ontology_version=onto.version))
    db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri="s3://o/1.jpg",
                 width=1920, height=1080, quality=0.9, scene={}))
    await db.flush()
    return fid


async def _obj(db, fid, onto, bbox, *, distance=None, conf=0.8, class_name="sedan"):
    from db.models import Object, ObjectDynamics

    o = Object(object_id=uuid.uuid4(), frame_id=fid, class_id=onto.by_name(class_name).id,
               bbox=list(bbox), conf=0.9, source="fused", state="review", attrs={}, provenance={},
               version=1)
    db.add(o)
    await db.flush()
    if distance is not None:
        db.add(ObjectDynamics(object_id=o.object_id, frame_id=fid, distance_m=float(distance),
                              method="ipm_mono_v1", confidence=conf))
        await db.flush()
    return o


@pytest.mark.asyncio
async def test_the_nearer_of_two_overlapping_objects_is_recorded_as_occluding_the_further():
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.quality.occlusion import propose_occlusion

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        fid = await _frame(db, onto)
        near = await _obj(db, fid, onto, [100, 400, 300, 700], distance=8.0)
        far = await _obj(db, fid, onto, [200, 380, 500, 660], distance=25.0)

        res = await propose_occlusion(db, fid)
        assert len(res["pairs"]) == 1
        p = res["pairs"][0]
        assert p["from_object_id"] == str(near.object_id)
        assert p["to_object_id"] == str(far.object_id)
        assert p["depth_gap_m"] == pytest.approx(17.0)
        await db.rollback()


@pytest.mark.asyncio
async def test_two_objects_at_the_same_depth_are_left_unordered_rather_than_guessed():
    """The claim that makes this usable. A wrong occlusion order is invisible in the label and wrong in
    every consumer, so no order is a better answer than a coin toss."""
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.quality.occlusion import propose_occlusion

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        fid = await _frame(db, onto)
        await _obj(db, fid, onto, [100, 400, 300, 700], distance=12.0)
        await _obj(db, fid, onto, [200, 380, 500, 660], distance=12.4)

        res = await propose_occlusion(db, fid)
        assert res["pairs"] == []
        assert len(res["unordered"]) == 1
        assert "apart" in res["unordered"][0]["reason"]
        await db.rollback()


@pytest.mark.asyncio
async def test_the_gap_required_grows_with_distance():
    """A four-metre gap settles it at 10 m and does not at 90 m, because the error of a flat-road lift
    grows with the distance it is estimating. A flat threshold gets one of those two wrong whichever value
    it picks: 40 m ordered zero pairs on the real corpus."""
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.quality.occlusion import propose_occlusion

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        near_fid = await _frame(db, onto)
        await _obj(db, near_fid, onto, [100, 400, 300, 700], distance=6.0)
        await _obj(db, near_fid, onto, [200, 380, 500, 660], distance=10.0)
        near_res = await propose_occlusion(db, near_fid)

        far_fid = await _frame(db, onto)
        await _obj(db, far_fid, onto, [100, 400, 300, 700], distance=86.0)
        await _obj(db, far_fid, onto, [200, 380, 500, 660], distance=90.0)
        far_res = await propose_occlusion(db, far_fid)

        assert len(near_res["pairs"]) == 1, "4 m apart at 6 and 10 m is a real separation"
        assert far_res["pairs"] == [], "the same 4 m at 86 and 90 m is inside the estimate's own error"
        await db.rollback()


@pytest.mark.asyncio
async def test_boxes_that_barely_touch_are_not_an_occlusion():
    """Two vehicles side by side in adjacent lanes clip each other's boxes by a few pixels, and neither
    hides the other."""
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.quality.occlusion import propose_occlusion

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        fid = await _frame(db, onto)
        await _obj(db, fid, onto, [100, 400, 300, 700], distance=8.0)
        await _obj(db, fid, onto, [295, 400, 500, 700], distance=25.0)   # 5px of overlap
        res = await propose_occlusion(db, fid)
        assert res["pairs"] == [] and res["unordered"] == []
        await db.rollback()


@pytest.mark.asyncio
async def test_an_object_with_no_depth_cannot_be_ordered_and_is_not_silently_dropped():
    """A frame where nothing has been through the dynamics pass has to say so, because zero relations and
    'this was never measured' look identical otherwise."""
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.quality.occlusion import depth_order, propose_occlusion

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        fid = await _frame(db, onto)
        a = await _obj(db, fid, onto, [100, 400, 300, 700], distance=None)
        b = await _obj(db, fid, onto, [200, 380, 500, 660], distance=None)

        res = await propose_occlusion(db, fid)
        assert res["pairs"] == []
        assert "depth estimate" in res["reason"]
        assert res["n_objects"] == 2 and res["n_with_depth"] == 0

        order = await depth_order(db, fid)
        assert order["order"] == []
        assert set(order["no_depth"]) == {str(a.object_id), str(b.object_id)}, (
            "objects with no depth belong in their own list, not sorted to one end of the order")
        await db.rollback()


@pytest.mark.asyncio
async def test_committed_relations_are_proposed_and_never_confirmed():
    """The depth behind them is a monocular ground-plane estimate. A confirmed relation would be asserting
    a fact about the world on the strength of an approximation."""
    from sqlalchemy import select

    from db.models import ObjectRelationship
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.quality.occlusion import commit_occlusion

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        fid = await _frame(db, onto)
        await _obj(db, fid, onto, [100, 400, 300, 700], distance=8.0)
        await _obj(db, fid, onto, [200, 380, 500, 660], distance=25.0)

        res = await commit_occlusion(db, fid)
        assert res["created"] == 1
        rows = (await db.execute(select(ObjectRelationship)
                                 .where(ObjectRelationship.frame_id == fid))).scalars().all()
        assert len(rows) == 1
        assert rows[0].status == "proposed" and rows[0].source == "geometry"
        assert rows[0].kind == "occludes"
        # The evidence travels with it, so a reviewer can weigh the claim rather than take it on trust.
        assert rows[0].evidence["depth_gap_m"] == pytest.approx(17.0)
        assert rows[0].evidence["method"] == "ipm_mono_v1"
        await db.rollback()


@pytest.mark.asyncio
async def test_committing_twice_does_not_duplicate_a_relation():
    """This runs after every dynamics pass, so it has to be safe to repeat."""
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.quality.occlusion import commit_occlusion

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        fid = await _frame(db, onto)
        await _obj(db, fid, onto, [100, 400, 300, 700], distance=8.0)
        await _obj(db, fid, onto, [200, 380, 500, 660], distance=25.0)
        assert (await commit_occlusion(db, fid))["created"] == 1
        await db.flush()
        assert (await commit_occlusion(db, fid))["created"] == 0
        await db.rollback()


@pytest.mark.asyncio
async def test_an_amodal_box_is_stored_beside_the_visible_one_and_never_replaces_it():
    """`bbox` means the visible extent to every existing consumer, and that must not change under them."""
    from db.models import Object
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        fid = await _frame(db, onto)
        o = await _obj(db, fid, onto, [100, 400, 300, 700])
        assert o.bbox_amodal is None, "nobody has said, and nothing should have guessed"

        o.bbox_amodal = [80.0, 390.0, 420.0, 710.0]
        await db.flush()
        fresh = await db.get(Object, o.object_id)
        assert list(fresh.bbox) == [100.0, 400.0, 300.0, 700.0], "the visible box must be untouched"
        assert list(fresh.bbox_amodal) == [80.0, 390.0, 420.0, 710.0]
        await db.rollback()


@pytest.mark.asyncio
async def test_an_amodal_box_reaches_the_export_record_and_absence_stays_absent():
    """An adapter that fell back to the visible box would let a consumer read a copy as though it were a
    judgement about what is hidden, and train on the second as if it were the first."""
    from services.export.records import ExportRecord

    r = ExportRecord(
        object_id=uuid.uuid4(), frame_id=uuid.uuid4(), session_id=uuid.uuid4(), ts_ns=1, cam_id="cam_f",
        img_uri="s3://x/1.jpg", width=1920, height=1080, vehicle_id="V", city="BLR",
        class_id=1, class_name="sedan", bbox=[1.0, 2.0, 3.0, 4.0], conf=0.9, state="accepted",
        source="human")
    assert r.bbox_amodal is None, "the default has to be absent, not a copy of bbox"
