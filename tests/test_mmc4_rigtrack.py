"""M-MC.4 cross-view track handoff + consistency: two rig identities at two instants that share per-camera
tracks chain into ONE rig track; when a member's class disagrees with the track vote it becomes a
cross_cam_inconsistent error candidate proposing the voted class. Single asyncio.run so the cached engine
binds to one loop.

Rig: cameras cam_f and cam_r, two instants. Tracks T1 (front) and T2 (right) run through both instants, so
the two per-instant rig identities are the same rig track. The right camera mislabels the object at the second
instant (class 6 vs the track's class 5), which the consistency check must flag."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from core.config import get_settings


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


@requires_infra
def test_rig_track_handoff_and_consistency():
    from sqlalchemy import delete, select
    from db.models import ErrorCandidate, Frame, FrameGroup, Object, RigObject, Track
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.multicam.rigident import link_objects
    from services.multicam.rigtrack import build_rig_tracks, check_consistency, rig_track_timeline, rig_tracks
    from services.multicam.sync import persist_groups

    sid = uuid.uuid4()
    t1, t2 = uuid.uuid4(), uuid.uuid4()   # per-camera tracks: front, right
    ms = 1_000_000

    async def run():
        maker = get_sessionmaker()
        async with maker() as db:
            db.add(DbSession(session_id=sid, vehicle_id="RIG-MMC4", start_ts_ns=0, end_ts_ns=10**9,
                             ontology_version="labelox-in-0.1.0"))
            await db.flush()
            db.add(Track(track_id=t1, session_id=sid, class_id=5, first_ts_ns=0, last_ts_ns=100 * ms))
            db.add(Track(track_id=t2, session_id=sid, class_id=5, first_ts_ns=0, last_ts_ns=100 * ms))
            await db.flush()
            frames = {}
            for inst, base in (("a", 0), ("b", 100 * ms)):
                for cam in ("cam_f", "cam_r"):
                    f = Frame(session_id=sid, ts_ns=base, cam_id=cam, img_uri=f"s3://x/{inst}-{cam}.jpg",
                              width=1920, height=1080)
                    db.add(f)
                    await db.flush()
                    frames[(inst, cam)] = f.frame_id
            # front says class 5 throughout; right says 5 then (wrongly) 6 at the second instant
            specs = {("a", "cam_f"): (5, t1), ("a", "cam_r"): (5, t2),
                     ("b", "cam_f"): (5, t1), ("b", "cam_r"): (6, t2)}
            oid = {}
            for key, (cls, trk) in specs.items():
                o = Object(frame_id=frames[key], class_id=cls, track_id=trk, bbox=[10, 10, 60, 60],
                           conf=0.9, state="auto_accept", source="fused")
                db.add(o)
                await db.flush()
                oid[key] = o.object_id
            await db.commit()

        await persist_groups(sid)
        async with maker() as db:
            groups = {int(g.ts_ns): g.group_id for g in (await db.execute(
                select(FrameGroup).where(FrameGroup.session_id == sid))).scalars().all()}
        g_a, g_b = groups[0], groups[100 * ms]

        # link each instant's two views into a rig identity
        await link_objects(sid, g_a, [oid[("a", "cam_f")], oid[("a", "cam_r")]])
        await link_objects(sid, g_b, [oid[("b", "cam_f")], oid[("b", "cam_r")]])

        # the two rig identities share tracks T1/T2 -> one rig track across time
        bt = await build_rig_tracks(sid)
        assert bt["rig_objects"] == 2 and bt["rig_tracks"] == 1, bt

        tracks = await rig_tracks(sid)
        assert tracks["n_tracks"] == 1
        trk = tracks["tracks"][0]
        assert trk["instants"] == 2 and trk["inconsistent"] is True
        assert sorted(trk["cameras"]) == ["cam_f", "cam_r"]

        tl = await rig_track_timeline(sid, uuid.UUID(trk["rig_track_id"]))
        assert tl["n_instants"] == 2 and tl["instants"][0]["ts_ns"] <= tl["instants"][1]["ts_ns"]

        # consistency check flags the mislabeled right-camera object at the second instant
        cc = await check_consistency(sid)
        assert cc["inconsistent_objects"] == 1, cc
        async with maker() as db:
            ec = (await db.execute(select(ErrorCandidate).where(ErrorCandidate.kind == "cross_cam_inconsistent",
                  ErrorCandidate.object_id == oid[("b", "cam_r")]))).scalar_one_or_none()
            assert ec is not None and ec.proposed_label["class_id"] == 5  # proposes the voted (front) class

        async with maker() as db:
            await db.execute(delete(ErrorCandidate).where(ErrorCandidate.object_id.in_(list(oid.values()))))
            await db.execute(delete(RigObject).where(RigObject.session_id == sid))
            await db.execute(delete(FrameGroup).where(FrameGroup.session_id == sid))
            await db.delete(await db.get(DbSession, sid))  # cascades frames/objects/tracks
            await db.commit()

    asyncio.run(run())
