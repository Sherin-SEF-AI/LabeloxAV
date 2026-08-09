"""One GPU, one job at a time, across every process that wants it.

The training worker already refuses to start beside another training worker: it takes a Postgres advisory
lock for its whole lifetime and exits if somebody else holds it. That protects training from training and
from nothing else.

Everything else that touches the GPU takes no lock at all. A corpus relabel is hours of batched inference, an
autolabel pass is the same, embedding backfills and ONNX exports load their own weights, and a training job
wants most of the card. Any two of those can run at once today, and what they produce when they do is not a
clean failure. It is an out-of-memory part way through a batch, which the relabel loop counts as a failed
frame, twenty of which trip the consecutive-failure guard and stop a corpus pass that had nothing wrong with
it. The 3,707-frame stall had a different cause, but it is exactly the shape this would produce.

So the slot is a lock every GPU consumer takes, for the duration of the work rather than the lifetime of the
process. That distinction is the whole design: the worker-lifetime lock answers "is another worker running",
which must stay as it is, while this answers "is the card busy", which is a different question with a
different answer most of the time.

Postgres advisory locks rather than a file lock or a semaphore, because the consumers are separate processes
that may be separate containers, and they already share exactly one thing: this database. A lock held on a
connection is released if the process dies, which is the property that matters most. A file lock survives the
process that abandoned it and needs a reaper of its own.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections.abc import AsyncIterator

from sqlalchemy import text

from core.logging import get_logger

log = get_logger("gpu_slot")

# Distinct from `training.advisory_lock_key`, deliberately. That one means "one training worker"; this one
# means "one job on the card". Sharing a key would make a running worker look permanently busy and every
# other GPU consumer would block for as long as the worker was up, which is forever.
GPU_SLOT_KEY = 0x4C42_4750  # ASCII "LBGP", so a stray lock in pg_locks is identifiable by eye

# How often to retry while somebody else holds it. Short enough to pick the card up promptly, long enough
# that a queue of waiters is not a busy loop against the database.
POLL_S = 2.0


class GpuBusy(RuntimeError):
    """Raised when the slot could not be acquired inside the timeout."""


@contextlib.asynccontextmanager
async def gpu_slot(holder: str, *, timeout_s: float | None = 3600.0,
                   key: int = GPU_SLOT_KEY) -> AsyncIterator[dict]:
    """Hold the GPU slot for the duration of the block.

    `holder` names the work, so a log line says which job is on the card rather than only that something is.
    `timeout_s=None` waits forever, which is right for a batch job that has nowhere else to be; a request
    handler should pass something small and let the caller be told the card is busy.

    Acquired on a dedicated connection, not on a caller's session, because an advisory lock lives on the
    connection that took it and a pooled session can be returned to the pool mid-work.
    """
    from db.session import get_engine

    waited = 0.0
    started = time.monotonic()
    async with get_engine().connect() as conn:
        while True:
            got = (await conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": key})).scalar()
            if got:
                break
            if timeout_s is not None and (time.monotonic() - started) >= timeout_s:
                raise GpuBusy(
                    f"the GPU slot was held by another job for {timeout_s:.0f}s; {holder} did not start")
            if waited == 0.0:
                log.info("gpu_slot.waiting", holder=holder, pid=os.getpid())
            await asyncio.sleep(POLL_S)
            waited = time.monotonic() - started

        log.info("gpu_slot.acquired", holder=holder, pid=os.getpid(), waited_s=round(waited, 1))
        t0 = time.monotonic()
        try:
            yield {"holder": holder, "waited_s": round(waited, 1), "pid": os.getpid()}
        finally:
            # Released even when the body raises. Leaking the slot would idle the card until the process
            # exits, and a long-running service does not exit.
            await conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": key})
            log.info("gpu_slot.released", holder=holder, held_s=round(time.monotonic() - t0, 1))


async def slot_is_free(key: int = GPU_SLOT_KEY) -> bool:
    """Whether the slot is currently unheld.

    Advisory only: the answer can be stale the instant it is returned, so it is for reporting to an operator,
    never for deciding whether to start work. Deciding is what `gpu_slot` is for.
    """
    from db.session import get_engine

    async with get_engine().connect() as conn:
        got = (await conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": key})).scalar()
        if got:
            await conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": key})
        return bool(got)
