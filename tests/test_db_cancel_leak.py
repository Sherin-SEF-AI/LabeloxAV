"""A request cancelled mid-query must not leave its connection open on the server.

Reproduced from a live leak: ten API sessions sat "idle in transaction" with a snapshot for 13 hours after a
browser navigated away mid-request, until the API process exited. See db/session.py::_close_socket_on_terminate
for the chain. This drives the request-scoped dependency the way Starlette does on a client disconnect (an anyio
cancel scope around a running query) and then asks Postgres whether the session is still there.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid

import anyio
import pytest
from sqlalchemy import text

from db.session import get_sessionmaker
from services.api.deps import db_session

pytestmark = pytest.mark.db


async def _left_behind(tag: str) -> int:
    async with get_sessionmaker()() as db:
        return (await db.execute(text("select count(*) from pg_stat_activity where application_name = :t"),
                                 {"t": tag})).scalar_one()


async def test_a_cancelled_query_does_not_leave_its_session_on_the_server():
    tag = f"cancel-{uuid.uuid4().hex[:8]}"

    async def request():
        async with contextlib.asynccontextmanager(db_session)() as s:
            await s.execute(text("select set_config('application_name', :t, false)"), {"t": tag})
            await s.execute(text("select pg_sleep(2)"))

    for _ in range(3):
        async with anyio.create_task_group() as tg:
            tg.start_soon(request)
            await anyio.sleep(0.3)
            tg.cancel_scope.cancel()

    # Long enough for pg_sleep to finish server-side: a leaked session is then idle in transaction.
    await asyncio.sleep(2.5)
    assert await _left_behind(tag) == 0
