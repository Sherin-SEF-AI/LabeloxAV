"""M-MC.0 frame group assembly: a multi-camera session's frames cluster into synchronized rig groups within
tolerance, a dropped camera shows up in missing_cams, the sync spread is the member timestamp span, and
group-aware prev / next / confirm operate on whole groups. Single asyncio.run so the cached engine binds to
one loop (conftest clears caches around the test)."""

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
def test_frame_groups_assemble_with_dropout_and_navigation():
    from db.models import Frame
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.multicam.sync import (
        adjacent_group,
        confirm_group,
        group_at_ts,
        list_groups,
        persist_groups,
    )

    sid = uuid.uuid4()
    ms = 1_000_000  # 1 ms in ns
    cams = ["cam_f", "cam_l", "cam_r"]

    async def run():
        maker = get_sessionmaker()
        async with maker() as db:
            db.add(DbSession(session_id=sid, vehicle_id="RIG-MMC0", start_ts_ns=0, end_ts_ns=10**9,
                             ontology_version="labelox-in-0.1.0"))
            await db.flush()
            # three instants, ~100 ms apart, each within-group jitter a few ms (inside 20 ms tolerance).
            # instant 2 (t=100 ms) drops cam_r -> it must appear in missing_cams for that group only.
            plan = [
                (0, cams),
                (100 * ms, ["cam_f", "cam_l"]),   # cam_r dropped here
                (200 * ms, cams),
            ]
            for base, present in plan:
                for i, cam in enumerate(present):
                    db.add(Frame(session_id=sid, ts_ns=base + i * 2 * ms, cam_id=cam,
                                 img_uri=f"s3://x/{base}-{cam}.jpg", width=1920, height=1080))
            await db.commit()

        summary = await persist_groups(sid)
        assert summary["n_groups"] == 3, summary
        assert summary["multicamera"] is True
        assert summary["groups_out_of_tolerance"] == 0, summary
        assert summary["groups_with_missing_cam"] == 1, summary

        lst = await list_groups(sid)
        assert lst["n_groups"] == 3
        gs = lst["groups"]
        # group ordering by ts, dropout only on the middle group
        assert gs[0]["missing_cams"] == [] and gs[0]["n_cams"] == 3
        assert gs[1]["missing_cams"] == ["cam_r"] and gs[1]["n_cams"] == 2
        assert gs[2]["missing_cams"] == []
        # spread = span of members within the group (two 2 ms steps across 3 cams = 4 ms)
        assert gs[0]["sync_spread_ns"] == 4 * ms, gs[0]

        # group-at-ts snaps to the nearest group (the middle one)
        at = await group_at_ts(sid, 100 * ms)
        assert at["group_id"] == gs[1]["group_id"]

        # prev / next navigate whole groups in time order
        nxt = await adjacent_group(sid, uuid.UUID(gs[0]["group_id"]), "next")
        assert nxt["group_id"] == gs[1]["group_id"]
        prv = await adjacent_group(sid, uuid.UUID(gs[2]["group_id"]), "prev")
        assert prv["group_id"] == gs[1]["group_id"]
        assert await adjacent_group(sid, uuid.UUID(gs[0]["group_id"]), "prev") is None

        # confirm operates on the whole group
        c = await confirm_group(uuid.UUID(gs[1]["group_id"]))
        assert c["confirmed"] is True

        # idempotent backfill: rebuilding does not duplicate
        again = await persist_groups(sid)
        assert again["n_groups"] == 3

        async with maker() as db:
            from sqlalchemy import delete

            from db.models import FrameGroup

            await db.execute(delete(FrameGroup).where(FrameGroup.session_id == sid))
            await db.delete(await db.get(DbSession, sid))  # cascades frames/objects
            await db.commit()

    asyncio.run(run())
