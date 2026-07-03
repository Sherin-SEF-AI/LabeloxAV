"""M-F.4 productivity + QA analytics: per-reviewer correction rate and agreement, inter-annotator agreement on
shared objects, and the cost view, computed from real Review rows. The metrics are corpus-wide, so the test
seeds reviews with unique reviewer ids and asserts those reviewers' computed numbers plus the report structure
and the non-punitive ordering (by agreement, not speed). Single asyncio.run so the cached engine binds to one
loop."""

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
def test_productivity_metrics_from_reviews():
    from sqlalchemy import delete
    from db.models import Frame, Object, Review, User
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.analytics.productivity import (
        cost_metrics,
        interannotator_agreement,
        reviewer_metrics,
    )

    sid = uuid.uuid4()
    fid = uuid.uuid4()
    u_good = uuid.uuid4()   # confirms everything -> high agreement
    u_fixer = uuid.uuid4()  # corrects everything -> high correction rate
    oid_shared = uuid.uuid4()

    async def run():
        maker = get_sessionmaker()
        async with maker() as db:
            db.add(DbSession(session_id=sid, vehicle_id="PROD", start_ts_ns=0, end_ts_ns=1, ontology_version="labelox-in-0.1.0"))
            db.add(User(user_id=u_good, name=f"good-{u_good.hex[:8]}", role="reviewer"))
            db.add(User(user_id=u_fixer, name=f"fixer-{u_fixer.hex[:8]}", role="reviewer"))
            await db.flush()
            db.add(Frame(frame_id=fid, session_id=sid, ts_ns=0, cam_id="cam_f", img_uri="s3://x/1.jpg", width=1920, height=1080))
            await db.flush()
            objs = []
            for _ in range(6):
                o = Object(frame_id=fid, class_id=5, bbox=[10, 10, 60, 60], conf=0.5, state="review", source="fused")
                db.add(o)
                await db.flush()
                objs.append(o.object_id)
            shared_obj = Object(object_id=oid_shared, frame_id=fid, class_id=7, bbox=[10, 10, 60, 60],
                                conf=0.5, state="review", source="fused")
            db.add(shared_obj)
            await db.flush()
            # u_good: 5 confirms (class unchanged)
            for o in objs[:5]:
                db.add(Review(object_id=o, reviewer="u-good", user_id=u_good, action="confirm",
                              before={"class_id": 5}, after={"class_id": 5}, time_spent_ms=1000, ts_ns=10 ** 12))
            # u_fixer: 5 reclassifies (class changed)
            for o in objs[:5]:
                db.add(Review(object_id=o, reviewer="u-fix", user_id=u_fixer, action="reclassify",
                              before={"class_id": 5}, after={"class_id": 6}, time_spent_ms=2000, ts_ns=10 ** 12))
            # a shared object both reviewers AGREE on (same final class) -> inter-annotator agreement counts it
            db.add(Review(object_id=oid_shared, reviewer="u-good", user_id=u_good, action="confirm",
                          before={"class_id": 7}, after={"class_id": 7}, time_spent_ms=500, ts_ns=10 ** 12))
            db.add(Review(object_id=oid_shared, reviewer="u-fix", user_id=u_fixer, action="confirm",
                          before={"class_id": 7}, after={"class_id": 7}, time_spent_ms=500, ts_ns=10 ** 12))
            await db.commit()

        metrics = {m["reviewer"]: m for m in await reviewer_metrics()}
        g = metrics[str(u_good)[:12]]
        f = metrics[str(u_fixer)[:12]]
        # u_good: 6 confirms -> all agreement, no corrections
        assert g["agreement"] == 1.0 and g["correction_rate"] == 0.0
        # u_fixer: 5 reclassifies + 1 (shared) confirm -> mostly corrections, low agreement
        assert f["correction_rate"] > 0.7 and f["agreement"] < 0.3

        ia = await interannotator_agreement()
        assert ia["shared_objects"] >= 1 and ia["agreement_rate"] is not None

        cost = await cost_metrics()
        for k in ("human_hours", "gpu_hours", "cost_per_frame_usd", "auto_accept_saved_usd", "assumptions"):
            assert k in cost

        # non-punitive ordering: results are sorted by agreement (quality), so the confirmer is not below the
        # fixer just because the fixer was slower/faster
        ordered = [m["reviewer"] for m in await reviewer_metrics() if m["reviewer"] in (str(u_good)[:12], str(u_fixer)[:12])]
        assert ordered.index(str(u_good)[:12]) < ordered.index(str(u_fixer)[:12])

        async with maker() as db:
            await db.execute(delete(Review).where(Review.user_id.in_([u_good, u_fixer])))
            await db.delete(await db.get(DbSession, sid))
            await db.execute(delete(User).where(User.user_id.in_([u_good, u_fixer])))
            await db.commit()

    asyncio.run(run())
