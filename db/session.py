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


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
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
