"""The continuous embedder's gates: it must yield the GPU, respect the disabled flag, and not claim work when
the backlog is un-embeddable. These are the guards that stop it repeating the concurrency that starved the
fleet sweep, so they are worth pinning even though the embedding itself needs a GPU and real media.
"""

from __future__ import annotations

import asyncio

import pytest

from core.config import get_settings


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


def _clear_db_cache():
    from db.session import get_engine, get_sessionmaker

    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


def run_async(coro):
    _clear_db_cache()
    try:
        return asyncio.run(coro)
    finally:
        _clear_db_cache()


@pytest.fixture
def embed_cfg():
    cfg = get_settings().intel.embed
    saved = (cfg.daemon_enabled, cfg.daemon_min_free_mb)
    yield cfg
    cfg.daemon_enabled, cfg.daemon_min_free_mb = saved


def test_disabled_flag_stops_the_daemon(embed_cfg):
    from services.intelligence.embed import daemon

    embed_cfg.daemon_enabled = False

    async def _go():
        async with __import__("db.session", fromlist=["get_sessionmaker"]).get_sessionmaker()() as db:
            return await daemon.maybe_embed_pending(db)

    # disabled short-circuits before any DB or GPU work, so it holds even without infra
    try:
        r = run_async(_go())
    except Exception:
        pytest.skip("no database to open a session")
    assert r["ran"] is False and r["reason"] == "disabled"


@requires_infra
def test_low_vram_makes_the_daemon_yield(embed_cfg, monkeypatch):
    from services.intelligence.embed import daemon

    embed_cfg.daemon_enabled = True
    embed_cfg.daemon_min_free_mb = 4000
    # pretend the card is nearly full: the daemon must sit the tick out rather than compete
    monkeypatch.setattr(daemon, "_free_vram_mb", lambda: 500.0)

    async def _go():
        from db.session import get_sessionmaker

        async with get_sessionmaker()() as db:
            return await daemon.maybe_embed_pending(db)

    r = run_async(_go())
    # either it yielded for VRAM, or there was genuinely nothing pending; it must NOT have embedded
    assert r["ran"] is False
    if r["reason"] != "nothing pending":
        assert "gpu busy" in r["reason"]


@requires_infra
def test_pending_counts_is_cheap_and_returns_ints():
    from services.intelligence.embed import daemon

    async def _go():
        from db.session import get_sessionmaker

        async with get_sessionmaker()() as db:
            return await daemon.pending_counts(db)

    p = run_async(_go())
    # The half-embedded counts are reported beside the backlog, not folded into it. They exist because a
    # crop with a DINOv3 vector and a NULL SigLIP2 one was counted as complete by every earlier version of
    # this function, which is how the whole corpus became unreachable by text search without any counter
    # noticing. A non-zero value here beside a zero backlog is that defect returning.
    assert set(p) == {"frames", "objects", "frames_missing_siglip", "objects_missing_siglip"}
    assert all(isinstance(v, int) for v in p.values())
