"""The two Tier 3 features whose machinery is complete and whose data is not, and what they say instead.

Cross-view linking and the LiDAR quality bridge are both finished code sitting in front of a corpus that
cannot exercise them. Measured:

    sessions with more than one camera            6 of 377
    the only five-camera session                  1,928 frames, ZERO labelled objects, no validated calibration
    camera_calibration rows                       101, every one source='estimated'
    Object3D rows                                 56, of which 14 are linked to a 2D object
    open quality_flag_3d rows                     8, every one on an unlinked cuboid

An inert page with no explanation is the worst state a working tool can be in, because the reasonable
conclusion is that it is broken. So both features answer with what is missing, in the order it has to be
fixed, and these tests pin that the answers are specific rather than a shrug.

The ordering matters and is tested: labelling objects on a session whose calibration has never been
validated produces labels that cannot be projected anywhere, so calibration is named before labelling.
"""

import uuid

import pytest

from core.timebase import now_ns, seconds_to_ns

pytestmark = pytest.mark.db


async def _session(db, onto, *, cams=("cam_f",), frames_per_cam=3, objects=0):
    from db.models import Frame, Object, OntologyClass, OntologyVersion
    from db.models import Session as DbSession

    if await db.get(OntologyVersion, onto.version) is None:
        db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
        await db.flush()
        for c in onto.classes:
            db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                 india=c.india, map_to={}))
        await db.flush()
    ts, sid = now_ns(), uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id="RIG-1", start_ts_ns=ts,
                     end_ts_ns=ts + seconds_to_ns(frames_per_cam + 1), city="BLR", sensors={},
                     ontology_version=onto.version))
    await db.flush()
    made = 0
    for cam in cams:
        for i in range(frames_per_cam):
            fid = uuid.uuid4()
            db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts + seconds_to_ns(i), cam_id=cam,
                         img_uri=f"s3://r/{cam}-{i}.jpg", width=1920, height=1080, quality=0.9, scene={}))
            await db.flush()
            if made < objects:
                db.add(Object(object_id=uuid.uuid4(), frame_id=fid, class_id=onto.by_name("sedan").id,
                              bbox=[1.0, 1.0, 50.0, 90.0], conf=0.8, source="fused", state="review",
                              attrs={}, provenance={}, version=1))
                made += 1
    await db.flush()
    return sid


@pytest.mark.asyncio
async def test_a_single_camera_session_says_it_has_no_second_view():
    """Not "not ready". The reason is a capture-side fact that no amount of annotation changes, and saying
    so stops somebody looking for a setting."""
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.multicam.readiness import session_readiness

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        sid = await _session(db, onto, cams=("cam_f",))
        r = await session_readiness(db, sid)
        assert r["ready"] is False
        codes = [b["code"] for b in r["blockers"]]
        assert "single_camera" in codes
        blocker = next(b for b in r["blockers"] if b["code"] == "single_camera")
        assert "rig" in blocker["fix"]
        await db.rollback()


@pytest.mark.asyncio
async def test_calibration_is_named_before_labelling():
    """The ordering is the useful part. Labelling a session whose calibration has never been validated
    produces labels that cannot be projected anywhere, so a reader working down the list does the thing
    that unblocks the other."""
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.multicam.readiness import session_readiness

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        sid = await _session(db, onto, cams=("cam_f", "cam_l"), objects=0)
        r = await session_readiness(db, sid)
        codes = [b["code"] for b in r["blockers"]]
        assert "calibration_not_validated" in codes and "nothing_to_link" in codes
        assert codes.index("calibration_not_validated") < codes.index("nothing_to_link")
        await db.rollback()


@pytest.mark.asyncio
async def test_a_validated_two_camera_session_with_objects_is_only_short_of_sync_groups():
    """Each blocker has to disappear when its own precondition is met, or the list is decoration."""
    from db.models import CalibrationValidation
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.multicam.readiness import session_readiness

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        sid = await _session(db, onto, cams=("cam_f", "cam_l"), objects=4)
        for cam in ("cam_f", "cam_l"):
            db.add(CalibrationValidation(session_id=sid, cam_id=cam, model="pinhole", status="pass"))
        await db.flush()

        r = await session_readiness(db, sid)
        codes = [b["code"] for b in r["blockers"]]
        assert "calibration_not_validated" not in codes
        assert "nothing_to_link" not in codes
        assert "single_camera" not in codes
        # What remains is real: no frame group holds both cameras, so there is no synchronized instant.
        assert codes == ["no_synchronized_groups"]
        await db.rollback()


@pytest.mark.asyncio
async def test_a_flag_on_an_unlinked_cuboid_is_reported_not_dropped():
    """The state the corpus is actually in: 42 of 56 cuboids have no 2D object, so most 3D findings have
    nowhere to surface. "No candidates were created" and "these findings have nowhere to go" are different
    facts, and only the second tells somebody what to fix."""
    from db.models import Object3D, PointCloud, QualityFlag3D
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.lidar.quality3d.bridge import bridge_flags

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        sid = await _session(db, onto, cams=("cam_f",))
        sess = await db.get(DbSession, sid)
        cloud = PointCloud(cloud_id=uuid.uuid4(), session_id=sess.session_id, ts_ns=now_ns(),
                           cloud_uri="s3://c/1.npz", point_count=1000, source="lidar")
        db.add(cloud)
        await db.flush()
        o3d = Object3D(object_3d_id=uuid.uuid4(), cloud_id=cloud.cloud_id, object_id=None,
                       class_id=onto.by_name("sedan").id, center=[10.0, 0.0, 0.8],
                       dims=[4.2, 1.8, 1.5], yaw=0.0, conf=0.8, box_source="fitted", source="fused", state="review")
        db.add(o3d)
        await db.flush()
        db.add(QualityFlag3D(object_3d_id=o3d.object_3d_id, cloud_id=cloud.cloud_id,
                             kind="floating", score=0.8, detail={"height_above_ground_m": 1.2},
                             status="open"))
        await db.flush()

        res = await bridge_flags(db, cloud_id=cloud.cloud_id)
        assert res["created"] == 0
        assert len(res["unbridgeable"]) == 1
        assert "not linked to a 2D object" in res["unbridgeable"][0]["reason"]
        await db.rollback()


@pytest.mark.asyncio
async def test_a_flag_on_a_linked_cuboid_reaches_the_queue_annotators_work():
    """The point of the bridge. quality_flag_3d has its own table, its own review endpoint and no reader
    anywhere a person goes; error_candidate has a page, a keymap and a throughput counter."""
    from sqlalchemy import select

    from db.models import ErrorCandidate, Frame, Object, Object3D, PointCloud, QualityFlag3D
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.lidar.quality3d.bridge import bridge_flags

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        sid = await _session(db, onto, cams=("cam_f",), objects=1)
        obj = (await db.execute(select(Object).join(Frame, Frame.frame_id == Object.frame_id)
                                .where(Frame.session_id == sid))).scalars().first()
        cloud = PointCloud(cloud_id=uuid.uuid4(), session_id=sid, ts_ns=now_ns(),
                           cloud_uri="s3://c/2.npz", point_count=1000, source="lidar")
        db.add(cloud)
        await db.flush()
        o3d = Object3D(object_3d_id=uuid.uuid4(), cloud_id=cloud.cloud_id, object_id=obj.object_id,
                       class_id=obj.class_id, center=[10.0, 0.0, 0.8], dims=[4.2, 1.8, 1.5], yaw=0.0,
                       conf=0.8, box_source="fitted", source="fused", state="review")
        db.add(o3d)
        await db.flush()
        db.add(QualityFlag3D(object_3d_id=o3d.object_3d_id, cloud_id=cloud.cloud_id,
                             kind="floating", score=0.8, detail={"height_above_ground_m": 1.2},
                             status="open"))
        await db.flush()

        res = await bridge_flags(db, cloud_id=cloud.cloud_id)
        assert res["created"] == 1
        await db.flush()
        cand = (await db.execute(select(ErrorCandidate)
                                 .where(ErrorCandidate.object_id == obj.object_id))).scalars().one()
        assert cand.kind == "lidar_floating", "the prefix says the evidence came from the cloud"
        assert cand.status == "pending", "nothing here is auto-confirmed"
        assert cand.proposed_label is None, (
            "a floating cuboid is a geometry problem; a class proposal it cannot justify would be worse "
            "than none, because the queue binds a key to applying it")
        assert cand.detail["height_above_ground_m"] == 1.2, "the checker's evidence has to travel with it"
        assert "not in this image" in cand.detail["note"]

        # And a second run adds nothing: this is meant to follow every consistency pass.
        assert (await bridge_flags(db, cloud_id=cloud.cloud_id))["created"] == 0
        await db.rollback()
