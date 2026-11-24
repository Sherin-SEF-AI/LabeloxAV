"""One worker per card, each minding its own jobs.

The slot, the worker's lifetime lock and the training lease were one each for the whole box, so a second
GPU could not be used: a job on card 1 waited for card 0, and a second worker refused to start. They are
per card now. The dangerous half of that change is the orphan sweep, which used to be entitled to treat
every running row as its own; with two workers it must leave the other card's live jobs alone.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import delete

from core.config import get_settings
from core.gpu_slot import GPU_SLOT_KEY, gpu_slot, slot_key
from db.models import AutolabelJob, TrainingJob
from db.session import get_sessionmaker

pytestmark = pytest.mark.db


@pytest.fixture
def device(monkeypatch):
    def _set(name: str) -> None:
        monkeypatch.setattr(get_settings().gpu, "device", name)
    return _set


@pytest.fixture
async def clean():
    async def _wipe():
        async with get_sessionmaker()() as db:
            await db.execute(delete(AutolabelJob))
            await db.execute(delete(TrainingJob))
            await db.commit()
    await _wipe()
    yield
    await _wipe()


def test_card_zero_keeps_the_original_slot_key_and_each_card_has_its_own(device):
    device("cuda:0")
    assert slot_key() == GPU_SLOT_KEY, "a worker from before this change must still exclude one from after"
    assert slot_key("cuda:1") == GPU_SLOT_KEY + 1
    assert slot_key("cuda") == GPU_SLOT_KEY and slot_key(3) == GPU_SLOT_KEY + 3
    device("cuda:2")
    assert slot_key() == GPU_SLOT_KEY + 2, "a worker started for card 2 takes card 2's slot by default"


def test_each_card_has_its_own_worker_lock(device):
    from services.training.worker import worker_lock_key

    base = get_settings().training.advisory_lock_key
    device("cuda:0")
    assert worker_lock_key() == base
    device("cuda:1")
    assert worker_lock_key() == base + 1


@pytest.mark.asyncio
async def test_two_cards_can_be_held_at_once_and_one_card_cannot_be_held_twice():
    from core.gpu_slot import GpuBusy

    async with gpu_slot("job-on-0", device="cuda:0", timeout_s=1):
        async with gpu_slot("job-on-1", device="cuda:1", timeout_s=1):
            pass                                        # both held: this is the second GPU being usable
        with pytest.raises(GpuBusy):
            async with gpu_slot("second-job-on-0", device="cuda:0", timeout_s=0.1):
                pass


async def _autolabel(status: str, device: str | None) -> uuid.UUID:
    jid = uuid.uuid4()
    params = {"executor": "worker"}
    if device:
        params["device"] = device
    async with get_sessionmaker()() as db:
        db.add(AutolabelJob(job_id=jid, session_id=uuid.uuid4(), status=status, params=params))
        await db.commit()
    return jid


async def _status(model, jid) -> str:
    async with get_sessionmaker()() as db:
        return (await db.get(model, jid)).status


@pytest.mark.asyncio
async def test_a_claimed_job_records_the_card_it_was_claimed_for(clean, device):
    from services.autolabel.jobs import claim_autolabel_job

    jid = await _autolabel("pending", None)
    device("cuda:1")
    async with get_sessionmaker()() as db:
        assert await claim_autolabel_job(db) == jid
    async with get_sessionmaker()() as db:
        assert (await db.get(AutolabelJob, jid)).params["device"] == "cuda:1"


@pytest.mark.asyncio
async def test_a_starting_worker_requeues_its_own_cards_orphans_and_not_the_other_cards_live_job(clean, device):
    from services.autolabel.jobs import requeue_orphans

    mine = await _autolabel("running", "cuda:1")
    theirs = await _autolabel("running", "cuda:0")
    legacy = await _autolabel("running", None)          # claimed before devices were recorded: card 0
    device("cuda:1")
    async with get_sessionmaker()() as db:
        out = await requeue_orphans(db, max_attempts=3)
    assert out["requeued"] == [str(mine)]
    assert await _status(AutolabelJob, mine) == "pending"
    assert await _status(AutolabelJob, theirs) == "running", "card 0's live job was taken from under it"
    assert await _status(AutolabelJob, legacy) == "running"

    device("cuda:0")
    async with get_sessionmaker()() as db:
        out = await requeue_orphans(db, max_attempts=3)
    assert sorted(out["requeued"]) == sorted([str(theirs), str(legacy)])


async def _training(status: str, device: str | None) -> uuid.UUID:
    jid = uuid.uuid4()
    async with get_sessionmaker()() as db:
        db.add(TrainingJob(job_id=jid, status=status, purpose="test", task_type="detection",
                           compute_target="local",
                           config={"runtime": {"device": device}} if device else {}))
        await db.commit()
    return jid


@pytest.mark.asyncio
async def test_training_on_one_card_does_not_make_the_other_busy(clean, device):
    from services.training.gpu_lease import training_holds_gpu

    await _training("running", "cuda:1")
    async with get_sessionmaker()() as db:
        device("cuda:0")
        assert not await training_holds_gpu(db), "card 0 refused work because card 1 was training"
        device("cuda:1")
        assert await training_holds_gpu(db)
        assert await training_holds_gpu(db, device="cuda:1")
        assert not await training_holds_gpu(db, device="cuda:0")


@pytest.mark.asyncio
async def test_a_training_job_with_no_recorded_card_is_on_card_zero(clean, device):
    from services.training.gpu_lease import training_holds_gpu

    await _training("running", None)
    async with get_sessionmaker()() as db:
        device("cuda:0")
        assert await training_holds_gpu(db)
        device("cuda:1")
        assert not await training_holds_gpu(db)


@pytest.mark.asyncio
async def test_the_training_claim_stamps_the_card_and_the_orphan_sweep_is_scoped_to_it(clean, device):
    from services.training.worker import _claim, _reset_orphans

    jid = await _training("pending", None)
    device("cuda:1")
    assert await _claim() == str(jid)
    async with get_sessionmaker()() as db:
        assert (await db.get(TrainingJob, jid)).config["runtime"]["device"] == "cuda:1"

    device("cuda:0")
    await _reset_orphans()
    assert await _status(TrainingJob, jid) == "running", "card 0's worker reset card 1's live run"
    device("cuda:1")
    await _reset_orphans()
    assert await _status(TrainingJob, jid) == "pending"
