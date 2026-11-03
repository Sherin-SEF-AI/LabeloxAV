"""Async engine and session factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from core.config import get_settings


def _close_socket_on_terminate() -> None:
    """Make a terminated asyncpg connection actually close its socket.

    A request cancelled while its query runs (the browser navigated away) leaked its connection for the
    life of the process: Postgres kept the session open in its transaction, holding its snapshot, which
    blocks vacuum and made CREATE INDEX CONCURRENTLY wait forever. Ten such sessions sat on the API for
    13 hours. The chain, reproduced: SQLAlchemy invalidates the connection and tries a graceful close()
    first; under Starlette's anyio cancel scope that close is itself cancelled after asyncpg has marked
    the protocol closing; SQLAlchemy then calls asyncpg's terminate(), which skips the abort because a
    closing connection already reports is_closed(). Nothing closes the transport. So after terminate, a
    transport still open is aborted here. (SQLAlchemy 2.0.36, asyncpg 0.30.0; tests/test_db_cancel_leak.py.)
    """
    from sqlalchemy.dialects.postgresql.asyncpg import AsyncAdapt_asyncpg_connection as Conn

    if getattr(Conn.terminate, "_lbx_closes_socket", False):
        return
    terminate = Conn.terminate

    def terminate_and_close(self):
        try:
            terminate(self)
        finally:
            transport = getattr(self._connection, "_transport", None)
            if transport is not None and not transport.is_closing():
                transport.abort()

    terminate_and_close._lbx_closes_socket = True
    Conn.terminate = terminate_and_close


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    _close_socket_on_terminate()
    settings = get_settings()
    if settings.postgres.pool == "null":
        from sqlalchemy.pool import NullPool

        return create_async_engine(settings.postgres.async_dsn, echo=False, poolclass=NullPool)
    return create_async_engine(
        settings.postgres.async_dsn,
        echo=False,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
    )


@lru_cache(maxsize=1)
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), expire_on_commit=False, class_=AsyncSession)


def database_unavailable(exc: BaseException) -> bool:
    """True when `exc` means the database could not be reached or had no connection to give, as opposed to
    a query that ran and failed.

    Identity lookups need the difference: "this token names nobody" is a 401, and the web client answers a
    401 by signing the person out, so a database at its connection limit must not be reported that way.
    """
    import asyncio

    from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError
    from sqlalchemy.exc import TimeoutError as PoolTimeout

    if isinstance(exc, OperationalError | InterfaceError | PoolTimeout | ConnectionError | OSError
                  | asyncio.TimeoutError):
        return True
    if isinstance(exc, DBAPIError) and exc.connection_invalidated:
        return True
    try:
        import asyncpg
    except ImportError:  # pragma: no cover - asyncpg is the driver everywhere this runs
        return False
    e = asyncpg.exceptions
    return isinstance(exc, e.TooManyConnectionsError | e.CannotConnectNowError | e.PostgresConnectionError
                      | e.InterfaceError)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency and general async context."""
    maker = get_sessionmaker()
    async with maker() as session:
        yield session
