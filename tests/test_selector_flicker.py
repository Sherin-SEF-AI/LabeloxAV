"""Integration: the active-learning scorer's per-track flicker term. A jittery high-confidence track must score
higher flicker than a stable one, and the value score must surface a 'flicker' component."""

import uuid

import numpy as np
import pytest

from core.timebase import now_ns, seconds_to_ns


@pytest.mark.asyncio
async def test_track_flicker_distinguishes_jittery_from_stable():
    from sqlalchemy import delete

    from db.models import Frame, Object, OntologyClass, OntologyVersion, Track
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.activelearn.selector import _track_flicker, score_candidates
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    cid = next(c.id for c in onto.classes if c.name == "pedestrian")
    maker = get_sessionmaker()
    ts = now_ns()
    sid = uuid.uuid4()
    track_stable, track_jitter = uuid.uuid4(), uuid.uuid4()
    T = 8

    async with maker() as db:
        if await db.get(OntologyVersion, onto.version) is None:
            db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
            await db.flush()
            for c in onto.classes:
                db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                     india=c.india, map_to={}))
            await db.flush()
        db.add(DbSession(session_id=sid, vehicle_id="FLK-1", start_ts_ns=ts, end_ts_ns=ts + seconds_to_ns(1),
                         city="BLR", sensors={}, ontology_version=onto.version))
        db.add(Track(track_id=track_stable, session_id=sid, class_id=cid, first_ts_ns=ts,
                     last_ts_ns=ts + T * 1000, tracker_version="test"))
        db.add(Track(track_id=track_jitter, session_id=sid, class_id=cid, first_ts_ns=ts,
                     last_ts_ns=ts + T * 1000, tracker_version="test"))
        await db.flush()
        rng = np.random.default_rng(0)
        for k in range(T):
            fid = uuid.uuid4()
            db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts + k * 1000, cam_id="cam_f",
                         img_uri=f"s3://x/{k}.jpg", width=1920, height=1080, quality=0.9, scene={}))
            await db.flush()
            # stable: same box every frame; jitter: box jumps around, both high confidence
            stable = [100.0, 100, 160, 220]
            jitter = np.array([100.0, 100, 160, 220]) + rng.normal(0, 30, size=4)
            db.add(Object(object_id=uuid.uuid4(), frame_id=fid, class_id=cid, track_id=track_stable,
                          bbox=stable, conf=0.9, source="fused", state="review", attrs={}, provenance={}, version=1))
            db.add(Object(object_id=uuid.uuid4(), frame_id=fid, class_id=cid, track_id=track_jitter,
                          bbox=[float(x) for x in jitter], conf=0.9, source="fused", state="review",
                          attrs={}, provenance={}, version=1))
        await db.commit()

    async with maker() as db:
        fl = await _track_flicker(db, {track_stable, track_jitter})
        assert fl[track_jitter] > fl[track_stable]           # the jittery track flickers more
        assert fl[track_stable] < 0.02                       # the stable track is near zero

    async with maker() as db:
        scored = await score_candidates(db, session_id=str(sid))
        assert scored and "flicker" in scored[0]["scores"]   # the value score exposes the flicker component
        # the jittery track's objects carry a higher flicker score than the stable track's
        by_track_flicker = {}
        # re-read track per object to attribute (objects in `scored` do not carry track_id, so check the spread)
        flickers = [it["scores"]["flicker"] for it in scored]
        assert max(flickers) > min(flickers)                 # flicker actually varies across the pool
