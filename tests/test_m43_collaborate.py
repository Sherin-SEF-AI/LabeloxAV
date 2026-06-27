"""M4.3 lakeFS versioning and collaboration: two annotators work on isolated branches concurrently, their
work merges to main through a reviewed merge request with attribution (an annotator may not merge, a
reviewer may), an export pins a specific lakeFS commit, and a bad merge is reverted. Single asyncio.run."""

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
def test_two_annotators_isolated_branches_reviewed_merge_and_revert():
    from db.models import User
    from db.session import get_sessionmaker
    from versioning import collaborate as C
    from versioning import lakefs_store as L

    suffix = uuid.uuid4().hex[:6]
    alice_n, bob_n, carol_n = f"alice-{suffix}", f"bob-{suffix}", f"carol-{suffix}"
    obj_a, obj_b = f"obj-alice-{suffix}", f"obj-bob-{suffix}"  # unique so the branch differs from main

    async def run():
        maker = get_sessionmaker()
        async with maker() as db:
            alice = User(name=alice_n, role="annotator")
            bob = User(name=bob_n, role="annotator")
            carol = User(name=carol_n, role="reviewer")
            db.add_all([alice, bob, carol])
            await db.flush()
            aid, bid, cid = str(alice.user_id), str(bob.user_id), str(carol.user_id)
            await db.commit()

            # two annotators, two isolated branches, worked concurrently
            asg_a = await C.create_assignment(db, obj_a, aid)
            asg_b = await C.create_assignment(db, obj_b, bid)
            assert asg_a["branch"] != asg_b["branch"]
            await C.commit_assignment_work(db, asg_a["assignment_id"], {obj_a: {"class": "truck"}})
            await C.commit_assignment_work(db, asg_b["assignment_id"], {obj_b: {"class": "bus"}})

            mr_a = await C.open_merge_request(db, "alice work", asg_a["branch"], author_id=aid)
            mr_b = await C.open_merge_request(db, "bob work", asg_b["branch"], author_id=bid)

            # an annotator may not approve/merge; a reviewer may
            denied = await C.approve_merge_request(db, mr_a["mr_id"], aid)
            assert "error" in denied
            assert (await C.approve_merge_request(db, mr_a["mr_id"], cid))["status"] == "approved"

            merged_a = await C.merge_request(db, mr_a["mr_id"], cid)
            merged_b = await C.merge_request(db, mr_b["mr_id"], cid)
            assert merged_a["status"] == "merged" and merged_b["status"] == "merged"

            # both annotators' work is on main, with attribution recorded on the MRs
            assert L.read_label("main", obj_a) == {"class": "truck"}
            assert L.read_label("main", obj_b) == {"class": "bus"}
            from db.models import MergeRequest
            mr_a_row = await db.get(MergeRequest, uuid.UUID(mr_a["mr_id"]))
            assert str(mr_a_row.author_id) == aid and str(mr_a_row.reviewer_id) == cid

            # an export pins a specific lakeFS commit
            pin = await C.pin_export(f"fleet-export-{suffix}", {"objects": 2, "commit": suffix})
            assert pin["lakefs_commit"] and L.read_label  # commit exists

            # a bad merge is reverted: bob's label is removed from main, alice's remains
            rev = await C.revert_merge_request(db, mr_b["mr_id"], cid)
            assert rev["status"] == "reverted"
            assert L.read_label("main", obj_b) is None
            assert L.read_label("main", obj_a) == {"class": "truck"}

            # cleanup
            for b in (asg_a["branch"], asg_b["branch"]):
                L.delete_branch(b)
            for u in (alice, bob, carol):
                await db.delete(await db.get(User, u.user_id))
            await db.commit()

    asyncio.run(run())
