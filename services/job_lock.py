"""A lock that says, exactly, whether the process running a job is still alive.

In-process jobs (imports, exports) leave a row reading `running` when the API restarts, and the only way
to tell a dead one from a live one was to wait ten minutes for its `updated_at` to go stale
(services/job_reaper.py). That window is why an interrupted job was failed instead of continued: nothing
could say "its writer is gone" at the moment the new process started.

A Postgres session-level advisory lock can. It is held on a connection the job keeps open for as long as
it runs, and Postgres releases it the instant that connection goes away, whether the process exited,
crashed or was killed. So a job whose lock can be taken is a job nobody is running, with no window and
no heartbeat to tune, and taking it is also what stops two processes from resuming the same job.

The lock is taken outside a transaction on purpose. A connection left "idle in transaction" for the
length of an import holds a snapshot that blocks vacuum for as long (see db/session.py).
"""

from __future__ import annotations

import contextlib
import hashlib
import uuid
from collections.abc import AsyncIterator

from sqlalchemy import text

from db.session import get_engine

# Namespaces the key away from the GPU worker's lifetime lock and anything else using advisory locks.
_NAMESPACE = b"labelox.job:"


def lock_key(job_id: uuid.UUID | str) -> int:
    """The signed 64-bit advisory key for a job id."""
    raw = _NAMESPACE + uuid.UUID(str(job_id)).bytes
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big", signed=True)


@contextlib.asynccontextmanager
async def job_lock(job_id: uuid.UUID | str) -> AsyncIterator[bool]:
    """Hold the job's lock for the body. Yields False, holding nothing, if another process has it."""
    key = lock_key(job_id)
    async with get_engine().connect() as conn:
        got = bool((await conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": key})).scalar())
        await conn.commit()
        try:
            yield got
        finally:
            if got:
                with contextlib.suppress(Exception):
                    await conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": key})
                    await conn.commit()


async def is_held(job_id: uuid.UUID | str) -> bool:
    """Whether some live process holds this job's lock right now."""
    async with job_lock(job_id) as got:
        return not got
