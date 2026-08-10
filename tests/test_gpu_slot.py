"""One GPU, and until now no agreement about who was using it.

The training worker takes a Postgres advisory lock for its lifetime and exits if another worker holds it.
Nothing else took any lock. A corpus relabel is hours of batched inference, autolabel is the same, embedding
backfills and ONNX exports load their own weights, and a training job wants most of the card. Any two could
run at once.

What that produces is not a clean failure. It is an out-of-memory part way through a batch, which the relabel
loop counts as a failed frame, and twenty failed frames in a row trip the consecutive-failure guard and stop
a corpus pass that had nothing wrong with it.

The tests here are about the properties a lock is worth having only if it actually has: it excludes, it
releases when the body raises, and it does not confuse itself with the worker-lifetime lock that already
exists for a different question.
"""

from __future__ import annotations

import asyncio

import pytest

from core.gpu_slot import GPU_SLOT_KEY, GpuBusy, gpu_slot, slot_is_free

pytestmark = pytest.mark.db

# A key of this test's own, so a training worker or a real job running beside the suite cannot make these
# pass or fail by holding the production slot.
TEST_KEY = 0x7E57_0001


async def test_the_slot_excludes_a_second_holder():
    """The whole point. Two GPU jobs must not both believe they have the card."""
    async with gpu_slot("first", key=TEST_KEY):
        with pytest.raises(GpuBusy):
            async with gpu_slot("second", timeout_s=0.0, key=TEST_KEY):
                pytest.fail("a second holder acquired the slot while the first held it")


async def test_the_slot_is_free_again_after_the_block():
    async with gpu_slot("first", key=TEST_KEY):
        assert await slot_is_free(TEST_KEY) is False
    assert await slot_is_free(TEST_KEY) is True


async def test_a_failing_job_does_not_keep_the_card():
    """Leaking the slot idles the GPU until the process exits, and a long-running service does not exit."""
    with pytest.raises(ValueError):
        async with gpu_slot("boom", key=TEST_KEY):
            raise ValueError("job failed")
    assert await slot_is_free(TEST_KEY) is True


async def test_a_waiter_gets_the_slot_once_it_is_released():
    """A batch job waits rather than failing, so the queue has to actually drain."""
    order: list[str] = []

    async def holder():
        async with gpu_slot("holder", key=TEST_KEY):
            order.append("holder-in")
            await asyncio.sleep(0.3)
            order.append("holder-out")

    async def waiter():
        await asyncio.sleep(0.05)
        async with gpu_slot("waiter", timeout_s=30.0, key=TEST_KEY):
            order.append("waiter-in")

    await asyncio.gather(holder(), waiter())
    assert order == ["holder-in", "holder-out", "waiter-in"]


async def test_the_timeout_reports_which_job_did_not_start():
    """An operator reading "the GPU slot was held" needs to know what was denied."""
    async with gpu_slot("incumbent", key=TEST_KEY):
        with pytest.raises(GpuBusy, match="autolabel"):
            async with gpu_slot("autolabel", timeout_s=0.0, key=TEST_KEY):
                pass


async def test_the_gpu_slot_is_not_the_training_worker_lock():
    """They answer different questions. Sharing a key would make a running worker look permanently busy and
    block every other GPU consumer for as long as it was up, which is forever."""
    from core.config import get_settings

    assert GPU_SLOT_KEY != get_settings().training.advisory_lock_key


async def test_holding_the_training_lock_does_not_block_the_gpu_slot():
    """The behaviour that distinction buys: a worker being alive must not stop a relabel from running."""
    from sqlalchemy import text

    from core.config import get_settings
    from db.session import get_engine

    worker_key = get_settings().training.advisory_lock_key
    async with get_engine().connect() as conn:
        got = (await conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": worker_key})).scalar()
        assert got, "another training worker is holding the lock; cannot run this test"
        try:
            async with gpu_slot("relabel", timeout_s=1.0, key=TEST_KEY) as slot:
                assert slot["holder"] == "relabel"
        finally:
            await conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": worker_key})
