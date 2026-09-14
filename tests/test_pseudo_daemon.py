"""The coverage daemon: what it picks, when it refuses, and that a batch takes itself back.

The lift itself is a GPU path and is not run here. What is tested is everything around it, which is what
was actually missing: the selection, the guards that keep it from taking the host down, and the revert.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import delete, select

from core.origin import REAL
from core.timebase import now_ns
from db.models import AgentRun, EgoPose, Frame, PointCloud
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology
from services.lidar.pseudo_daemon import (
    KIND,
    MEMORY_CEILING_FRAC,
    _pending_frames,
    _uncovered_sessions,
    coverage,
    maybe_lift_pending,
)

pytestmark = pytest.mark.db


async def _seed(db, *, frames: int = 6, covered: int = 2) -> dict:
    onto = get_ontology()
    tag = uuid.uuid4().hex[:6]
    sid = uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id=f"PSD-{tag}", start_ts_ns=0, end_ts_ns=1, city="BLR",
                     route=f"pseudo-fixture-{tag}", sensors={}, ontology_version=onto.version,
                     origin=REAL))
    await db.flush()
    base = now_ns()
    ts_list = []
    for i in range(frames):
        ts = base + i * 100_000_000
        ts_list.append(ts)
        db.add(Frame(frame_id=uuid.uuid4(), session_id=sid, ts_ns=ts, cam_id="front", width=1280,
                     height=960, img_uri=f"frames/{sid}/front/{ts}.jpg", origin=REAL, selected=True))
    for ts in ts_list[:covered]:
        db.add(PointCloud(session_id=sid, ts_ns=ts, source="pseudo",
                          cloud_uri=f"clouds/{sid}/{ts}.npz", point_count=1000))
    await db.commit()
    return {"session_id": sid, "ts": ts_list, "frames": frames, "covered": covered}


async def _cleanup(sid):
    async with get_sessionmaker()() as db:
        await db.execute(delete(DbSession).where(DbSession.session_id == sid))
        await db.commit()


class TestSelection:
    async def test_a_session_is_ranked_by_how_much_of_it_is_uncovered(self):
        async with get_sessionmaker()() as db:
            small = await _seed(db, frames=4, covered=3)
            big = await _seed(db, frames=12, covered=0)
            rows = await _uncovered_sessions(db, 5000)
        # Restricted to the two fixtures: the rest of the suite leaves its own uncovered sessions in
        # this database, and a test that asserted on absolute positions would be asserting on them.
        order = [str(sid) for sid, _n, _c in rows
                 if sid in (big["session_id"], small["session_id"])]
        assert order == [str(big["session_id"]), str(small["session_id"])], \
            "the least covered session comes first: coverage is the point"
        await _cleanup(small["session_id"])
        await _cleanup(big["session_id"])

    async def test_a_fully_covered_session_is_not_offered(self):
        async with get_sessionmaker()() as db:
            fx = await _seed(db, frames=3, covered=3)
            rows = await _uncovered_sessions(db, 100)
        assert str(fx["session_id"]) not in [str(s) for s, _n, _c in rows]
        await _cleanup(fx["session_id"])

    async def test_pending_frames_skips_the_ones_that_already_have_a_cloud(self):
        async with get_sessionmaker()() as db:
            fx = await _seed(db, frames=6, covered=2)
            pend = await _pending_frames(db, fx["session_id"], 100)
        assert len(pend) == 4
        assert set(fx["ts"][:2]).isdisjoint({p["ts_ns"] for p in pend})
        await _cleanup(fx["session_id"])

    async def test_the_batch_ceiling_bounds_the_work(self):
        async with get_sessionmaker()() as db:
            fx = await _seed(db, frames=10, covered=0)
            pend = await _pending_frames(db, fx["session_id"], 3)
        assert len(pend) == 3, "nothing here runs as one unbounded block"
        await _cleanup(fx["session_id"])

    async def test_a_synthetic_session_is_never_lifted(self):
        """A composite has no real depth to recover and spending the card on one would be spending it
        on the copy-paste generator's own output."""
        from core.origin import SYNTHETIC

        async with get_sessionmaker()() as db:
            fx = await _seed(db, frames=5, covered=0)
            await db.execute(DbSession.__table__.update()
                             .where(DbSession.session_id == fx["session_id"])
                             .values(origin=SYNTHETIC))
            await db.execute(Frame.__table__.update()
                             .where(Frame.session_id == fx["session_id"]).values(origin=SYNTHETIC))
            await db.commit()
            rows = await _uncovered_sessions(db, 100)
        assert str(fx["session_id"]) not in [str(s) for s, _n, _c in rows]
        await _cleanup(fx["session_id"])


class TestGuards:
    async def test_the_killswitch_stops_it(self):
        from services.govern.killswitch import get_state

        async with get_sessionmaker()() as db:
            st = await get_state(db)
            was = st.loop_enabled
            st.loop_enabled = False
            await db.commit()
            try:
                res = await maybe_lift_pending(db)
                assert res["ran"] is False and "killswitch" in res["reason"]
            finally:
                st.loop_enabled = was
                await db.commit()

    async def test_it_declines_once_it_has_run_today(self):
        from services.govern.killswitch import get_state

        async with get_sessionmaker()() as db:
            st = await get_state(db)
            was = st.loop_enabled
            st.loop_enabled = True          # the killswitch is a separate refusal, tested above
            rid = uuid.uuid4()
            db.add(AgentRun(run_id=rid, kind=KIND, scope={}, status="committed", policy={}, counts={},
                            changes={}, critic={}, created_by="test"))
            await db.commit()
            try:
                res = await maybe_lift_pending(db)
                assert res["ran"] is False and "today" in res["reason"]
            finally:
                st.loop_enabled = was
                await db.execute(delete(AgentRun).where(AgentRun.run_id == rid))
                await db.commit()

    def test_the_memory_ceiling_is_below_the_host_falling_over(self):
        assert 0.5 < MEMORY_CEILING_FRAC < 1.0


class TestRevert:
    async def test_a_batch_takes_its_clouds_back(self):
        from services.agent.runs import revert_run

        async with get_sessionmaker()() as db:
            fx = await _seed(db, frames=4, covered=0)
            ts = fx["ts"][0]
            pc = PointCloud(session_id=fx["session_id"], ts_ns=ts, source="pseudo",
                            cloud_uri=f"clouds/{fx['session_id']}/{ts}.npz", point_count=10)
            db.add(pc)
            await db.flush()
            rid = uuid.uuid4()
            db.add(AgentRun(run_id=rid, kind="pseudo_batch",
                            scope={"session_id": str(fx["session_id"])}, status="committed", policy={},
                            counts={}, changes={"cloud_ids": [str(pc.cloud_id)],
                                                "session_id": str(fx["session_id"])},
                            critic={}, created_by="test"))
            await db.commit()

            res = await revert_run(db, rid)
            assert res["reverted"] == 1
            left = (await db.execute(select(PointCloud).where(
                PointCloud.cloud_id == pc.cloud_id))).scalar_one_or_none()
            assert left is None
            run = await db.get(AgentRun, rid)
            assert run.status == "reverted"
        await _cleanup(fx["session_id"])

    async def test_a_run_that_recorded_nothing_says_so_rather_than_failing(self):
        from services.lidar.pseudo_daemon import revert_batch

        async with get_sessionmaker()() as db:
            run = AgentRun(run_id=uuid.uuid4(), kind="pseudo_batch", scope={}, status="committed",
                           policy={}, counts={}, changes={}, critic={}, created_by="test")
            db.add(run)
            await db.commit()
            res = await revert_batch(db, run)
            assert res["reverted"] == 0 and "reason" in res
            await db.execute(delete(AgentRun).where(AgentRun.run_id == run.run_id))
            await db.commit()


class TestCoverage:
    async def test_it_counts_real_frames_clouds_and_poses(self):
        async with get_sessionmaker()() as db:
            fx = await _seed(db, frames=5, covered=2)
            db.add(EgoPose(session_id=fx["session_id"], ts_ns=fx["ts"][0], x=0.0, y=0.0, z=0.0,
                           source="visual", measured=False, quality=0.4))
            await db.commit()
            cov = await coverage(db)
        assert cov["real_selected_frames"] >= 5
        assert cov["pseudo_clouds"] >= 2
        assert cov["ego_poses"] >= 1
        # The distinction the whole table exists for: an inferred pose is not a measured one.
        assert cov["ego_poses_measured"] <= cov["ego_poses"]
        await _cleanup(fx["session_id"])
