"""Whether a machine-filled box is still on the object its anchor was drawn around.

The check has existed inside `services/agent/propagate_agent.py` since propagation was written and only
ever ran at creation time, on boxes that agent itself made. Nothing could ask it about the boxes already
in the corpus, which is where it matters: interpolated and propagated objects are most of the machine
fill, and `errordetect/embedding_outlier.py` compares an object to its class centroid, which answers a
different question. A box that has slid off a scooter onto the road behind it still resembles the scooter
centroid and resembles the crop it started from not at all.

Measured over the real corpus once the pass existed: 64.6% of machine fill drifts, against 32.4% of
detector output judged the same way over a longer baseline. So the check discriminates rather than firing
on everything, which is the bar `reanalyze.py::_drop_systemic` sets at 80%.

The tests here are about the two ways this becomes untrustworthy rather than about the cosine, which is
numpy. First, `unknown` must never be folded into `drifted`: DINOv3 is absent on a CPU-only host and in
CI, and a pass that reported every box as drifted there would tell an annotator a hundred good boxes are
wrong because a GPU is busy. Second, an anchor is a box somebody drew or a detector fired on, never
another fill, because a fill judged against a fill compounds the error it was meant to catch.
"""

import uuid

import pytest

from core.timebase import now_ns, seconds_to_ns

pytestmark = pytest.mark.db


async def _track(db, onto, spec, *, class_name="sedan"):
    """One track. `spec` is a list of (source, is_keyframe) in time order."""
    from db.models import Frame, Object, OntologyClass, OntologyVersion, Track
    from db.models import Session as DbSession

    cid = onto.by_name(class_name).id
    if await db.get(OntologyVersion, onto.version) is None:
        db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
        await db.flush()
        for c in onto.classes:
            db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                 india=c.india, map_to={}))
        await db.flush()

    ts, sid, tid = now_ns(), uuid.uuid4(), uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id="DRF-1", start_ts_ns=ts,
                     end_ts_ns=ts + seconds_to_ns(len(spec) + 1), city="BLR", sensors={},
                     ontology_version=onto.version))
    await db.flush()
    db.add(Track(track_id=tid, session_id=sid, class_id=cid, first_ts_ns=ts,
                 last_ts_ns=ts + seconds_to_ns(len(spec)), trajectory={}, id_switch_flags={},
                 tracker_version="test", intents={}))
    await db.flush()
    oids = []
    for i, (source, keyframe) in enumerate(spec):
        fid, oid = uuid.uuid4(), uuid.uuid4()
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts + seconds_to_ns(i), cam_id="cam_f",
                     img_uri=f"s3://drift/{i}.jpg", width=1920, height=1080, quality=0.9, scene={}))
        await db.flush()
        db.add(Object(object_id=oid, frame_id=fid, track_id=tid, class_id=cid,
                      bbox=[0.0, 0.0, 40.0, 40.0], conf=0.6, source=source, state="review",
                      is_keyframe=keyframe, attrs={}, provenance={}, version=1))
        oids.append(oid)
    await db.flush()
    return tid, oids


def test_an_unreadable_crop_is_unknown_and_never_drifted():
    """The claim that keeps this usable off a GPU. DINOv3 is absent in CI and on a CPU-only host, and a
    pass that called every box drifted there would be worse than no pass at all."""
    from services.temporal.drift import cosine_to, encode_crop

    class _NoStore:
        def get_bytes(self, uri):
            raise FileNotFoundError(uri)

    assert encode_crop(_NoStore(), "s3://x/none.jpg", [0, 0, 10, 10]) is None
    # And with no reference vector there is nothing to compare against, which is also unknown, not zero.
    assert cosine_to(_NoStore(), "s3://x/none.jpg", [0, 0, 10, 10], None) is None


def test_a_degenerate_box_is_unknown_rather_than_a_similarity():
    """A one-pixel box has no appearance. Encoding it would produce a vector, and that vector would be
    noise wearing the shape of an answer."""
    import numpy as np

    from services.temporal.drift import crop_box

    class _Store:
        def get_bytes(self, uri):
            import cv2

            img = np.zeros((64, 64, 3), np.uint8)
            return cv2.imencode(".jpg", img)[1].tobytes()

    assert crop_box(_Store(), "s3://x/a.jpg", [10, 10, 11, 11]) is None
    assert crop_box(_Store(), "s3://x/a.jpg", [10, 10, 40, 40]) is not None


def test_the_size_band_is_symmetric():
    """A box three times the anchor's area and one a third of it are the same amount wrong."""
    from services.temporal.drift import SIZE_TOL, size_ok

    anchor = [0, 0, 10, 10]
    assert size_ok([0, 0, 10, 10], anchor)
    assert not size_ok([0, 0, 100, 100], anchor)      # 100x
    assert not size_ok([0, 0, 1, 1], anchor)          # 1/100
    assert SIZE_TOL == 3.0, "the band propagate_agent measured; changing it changes both callers"


@pytest.mark.asyncio
async def test_a_fill_is_never_judged_against_another_fill():
    """The compounding failure. `interpolate.py` already refuses to anchor a fill on a fill, because the
    old implementation treated every object as an anchor and its own output re-anchored the next run. The
    same rule has to hold here, or a drifted box becomes the reference that certifies the next one."""
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.temporal.drift import _anchors_and_fill

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        tid, oids = await _track(db, onto, [
            ("human", True), ("interpolated", False), ("propagated", False),
            ("fused", False), ("recall", False)])
        anchors, fill = await _anchors_and_fill(db, tid)
        assert {o.source for o, _, _ in anchors} == {"human", "fused"}
        assert {o.source for o, _, _ in fill} == {"interpolated", "propagated", "recall"}
        await db.rollback()


@pytest.mark.asyncio
async def test_a_track_with_no_anchor_says_so_rather_than_reporting_nothing_wrong():
    """A track made entirely of machine fill has nothing to compare against. Returning an empty clean
    result would read as "checked, all fine", which is the opposite of the truth."""
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.temporal.drift import track_drift

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        tid, _ = await _track(db, onto, [("interpolated", False)] * 3)
        r = await track_drift(db, tid, take_slot=False)
        assert r["checked"] == 0
        assert "no anchor" in r["reason"]
        await db.rollback()


@pytest.mark.asyncio
async def test_a_track_with_no_machine_fill_says_that_instead():
    """The other empty case, and a different sentence: there was nothing to check, not nothing to find."""
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.temporal.drift import track_drift

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        tid, _ = await _track(db, onto, [("fused", False), ("human", True), ("auto_accept", False)])
        r = await track_drift(db, tid, take_slot=False)
        assert r["checked"] == 0
        assert "no interpolated or propagated" in r["reason"]
        await db.rollback()


@pytest.mark.asyncio
async def test_every_fill_is_judged_against_the_nearest_anchor_in_time():
    """Nearest rather than preceding. A fill in the middle of a gap is better judged against whichever end
    is closer, and a rule that always looked backwards would compare the last frame of a long hole against
    an anchor several seconds stale."""
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.temporal.drift import track_drift

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        # anchor, fill, fill, fill, anchor: the last fill should pick the second anchor.
        tid, oids = await _track(db, onto, [
            ("human", True), ("interpolated", False), ("interpolated", False),
            ("interpolated", False), ("human", True)])
        r = await track_drift(db, tid, take_slot=False)
        assert r["checked"] == 3
        by_id = {row["object_id"]: row for row in r["rows"]}
        assert by_id[str(oids[1])]["anchor_object_id"] == str(oids[0])
        assert by_id[str(oids[3])]["anchor_object_id"] == str(oids[4])
        # No image store behind these fixtures, so every verdict is unknown, which is the point of the
        # first test in this file: the pass still ran and still reported what it could not measure.
        assert {row["verdict"] for row in r["rows"]} == {"unknown"}
        assert r["counts"]["drifted"] == 0
        await db.rollback()
