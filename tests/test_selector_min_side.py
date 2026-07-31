"""The scorer must not queue objects a human cannot judge.

The value ranking rewards uncertainty, and a distant object is small, low confidence and uncertain, so it
sorts to the top of every pool while being precisely the object a reviewer cannot rule on. A batch mined
without a size floor came back at 12 to 40 pixels on the shorter side: the reviewer guesses, and a guessed
verdict enters the corpus indistinguishable from a considered one.

The floor is applied in the pool query rather than after ranking, which is what these tests pin. Filtering
after the fact would let unreviewable objects consume `pool_limit` and leave the ranking choosing among
whatever few judgeable objects happened to survive the truncation.
"""

import uuid

import pytest

from core.timebase import now_ns, seconds_to_ns


async def _seed(db, onto, cid, sid, boxes, confs=None):
    """One frame per box.

    Confidence is settable because the pool is ordered by distance from the 0.5 decision boundary before it
    is truncated. Leaving every object at 0.5 makes that ordering a tie broken on a random uuid, which is
    fine for a test that only checks membership and useless for one that checks what truncation keeps.
    """
    from db.models import Frame, Object, OntologyClass, OntologyVersion
    from db.models import Session as DbSession

    ts = now_ns()
    if await db.get(OntologyVersion, onto.version) is None:
        db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
        await db.flush()
        for c in onto.classes:
            db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                 india=c.india, map_to={}))
        await db.flush()
    db.add(DbSession(session_id=sid, vehicle_id="MINSIDE-1", start_ts_ns=ts,
                     end_ts_ns=ts + seconds_to_ns(1), city="BLR", sensors={},
                     ontology_version=onto.version))
    await db.flush()
    made = []
    for k, bbox in enumerate(boxes):
        fid = uuid.uuid4()
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts + k * 1000, cam_id="cam_f",
                     img_uri=f"s3://x/{k}.jpg", width=1920, height=1080, quality=0.9, scene={}))
        await db.flush()
        oid = uuid.uuid4()
        db.add(Object(object_id=oid, frame_id=fid, class_id=cid, bbox=list(bbox),
                      conf=(confs[k] if confs else 0.5),
                      source="fused", state="review", attrs={}, provenance={}, version=1))
        made.append(oid)
    await db.flush()
    return made


@pytest.mark.asyncio
async def test_min_side_excludes_objects_too_small_to_judge():
    from db.session import get_sessionmaker
    from services.activelearn.selector import score_candidates
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    cid = next(c.id for c in onto.classes if c.name == "rider")
    sid = uuid.uuid4()

    # A 12x20 crop is a handful of pixels; a 60x90 one is a person on a motorcycle.
    boxes = [(100.0, 100.0, 112.0, 120.0),    # shorter side 12, unjudgeable
             (200.0, 200.0, 224.0, 240.0),    # shorter side 24, marginal
             (300.0, 300.0, 360.0, 390.0)]    # shorter side 60, judgeable

    async with get_sessionmaker()() as db:
        tiny, marginal, big = await _seed(db, onto, cid, sid, boxes)

        no_floor = await score_candidates(db, session_id=str(sid), class_ids=[cid], min_side_px=0)
        got = {str(r["object_id"]) for r in no_floor}
        assert {str(tiny), str(marginal), str(big)} <= got, "no floor should keep everything"

        floored = await score_candidates(db, session_id=str(sid), class_ids=[cid], min_side_px=48)
        kept = {str(r["object_id"]) for r in floored}
        assert str(big) in kept
        assert str(tiny) not in kept, "a 12px object cannot be reviewed and must not be dispatched"
        assert str(marginal) not in kept, "a 24px object is below the floor too"

        await db.rollback()


@pytest.mark.asyncio
async def test_floor_applies_before_the_pool_is_truncated():
    """The judgeable object must survive even when small ones would have filled the pool.

    This is the whole reason the filter lives in the query. With `pool_limit=2`, two unreviewable objects
    would otherwise consume the pool and the one object worth a reviewer's time would never be ranked.
    """
    from db.session import get_sessionmaker
    from services.activelearn.selector import score_candidates
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    cid = next(c.id for c in onto.classes if c.name == "rider")
    sid = uuid.uuid4()
    boxes = [(10.0, 10.0, 26.0, 30.0), (50.0, 50.0, 66.0, 70.0), (300.0, 300.0, 380.0, 400.0)]
    # The two small ones sit exactly on the decision boundary and so sort first, which is not a contrivance:
    # it is why the problem exists. Distant objects are the uncertain ones, so they head every pool.
    confs = [0.50, 0.50, 0.85]

    async with get_sessionmaker()() as db:
        _tiny_a, _tiny_b, big = await _seed(db, onto, cid, sid, boxes, confs)

        starved = await score_candidates(db, session_id=str(sid), class_ids=[cid],
                                         pool_limit=2, min_side_px=0)
        assert str(big) not in {str(r["object_id"]) for r in starved}, \
            "precondition: without the floor the small objects crowd out the judgeable one"

        floored = await score_candidates(db, session_id=str(sid), class_ids=[cid],
                                         pool_limit=2, min_side_px=48)
        assert str(big) in {str(r["object_id"]) for r in floored}

        await db.rollback()


@pytest.mark.asyncio
async def test_default_is_unfiltered_so_existing_callers_are_unchanged():
    """Ranking and dispatching are different jobs, and only dispatching needs the floor.

    The queue page ranks the whole corpus for display; silently hiding small objects there would change what
    the corpus looks like, not just what gets handed out as work.
    """
    from db.session import get_sessionmaker
    from services.activelearn.selector import MIN_REVIEWABLE_SIDE_PX, score_candidates
    from services.autolabel.ontology import get_ontology

    assert MIN_REVIEWABLE_SIDE_PX == 0.0

    onto = get_ontology()
    cid = next(c.id for c in onto.classes if c.name == "rider")
    sid = uuid.uuid4()

    async with get_sessionmaker()() as db:
        tiny, = await _seed(db, onto, cid, sid, [(10.0, 10.0, 22.0, 30.0)])
        rows = await score_candidates(db, session_id=str(sid), class_ids=[cid])
        assert str(tiny) in {str(r["object_id"]) for r in rows}
        await db.rollback()
