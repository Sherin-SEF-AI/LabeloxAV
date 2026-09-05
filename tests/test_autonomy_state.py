"""Phase 3 of the autonomy work: one endpoint answers "what may the machine do right now, and why".

The daemon-liveness half is the part that earns a test: a dead daemon looks exactly like a healthy
idle one on every other surface, and the whole point of the dot is that stale is SHOWN, not guessed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete

from db.models import AuditDecision
from db.session import get_sessionmaker

pytestmark = pytest.mark.db


async def _tick_row(db, *, age_seconds: int, decision: str = "tick"):
    row = AuditDecision(actor="controller", decision=decision, subject=None, rationale={})
    db.add(row)
    await db.flush()
    row.created_at = datetime.now(UTC) - timedelta(seconds=age_seconds)
    await db.commit()


async def test_a_fresh_tick_is_alive_and_a_missed_pair_is_stale():
    from services.api.routers.autonomy import STALE_AFTER_TICKS, TICK_SECONDS, _daemon_liveness

    async with get_sessionmaker()() as db:
        await db.execute(delete(AuditDecision).where(AuditDecision.actor == "controller"))
        await db.commit()

        none = await _daemon_liveness(db)
        assert none["alive"] is False and "has not run" in none["detail"], \
            "no tick ever is not the same claim as a recent tick; the dot must not guess"

        await _tick_row(db, age_seconds=5)
        fresh = await _daemon_liveness(db)
        assert fresh["alive"] is True and fresh["seconds_since"] <= 10

        await db.execute(delete(AuditDecision).where(AuditDecision.actor == "controller"))
        await db.commit()
        await _tick_row(db, age_seconds=TICK_SECONDS * STALE_AFTER_TICKS + 30)
        stale = await _daemon_liveness(db)
        assert stale["alive"] is False, \
            "two missed ticks is the line: after it, 'live' would be a lie the page tells forever"
        assert stale["last_tick_at"] is not None, "stale still says WHEN, so the reader can judge"


async def test_the_state_endpoint_carries_every_section():
    from services.api.routers.autonomy import autonomy_state

    async with get_sessionmaker()() as db:
        res = await autonomy_state(db=db)
    for key in ("switches", "daemon", "ladder", "measurements", "settlement", "last_digest",
                "journal"):
        assert key in res, f"missing section {key}: the page reads this one aggregation"
    assert "settlement_enabled" in res["switches"]
    levels = [r["level"] for r in res["ladder"]]
    assert levels == sorted(levels, reverse=True), "highest rungs first; that is what the page shows"
    assert "control_precision" in res["measurements"]
    assert "measured_at" in res["measurements"]["control_precision"], \
        "a measurement without an age cannot be refused for staleness"


async def test_a_paused_tick_still_counts_as_a_heartbeat():
    """tick_paused means the daemon is alive and obeying the kill switch - the opposite of dead."""
    from services.api.routers.autonomy import _daemon_liveness

    async with get_sessionmaker()() as db:
        await db.execute(delete(AuditDecision).where(AuditDecision.actor == "controller"))
        await db.commit()
        await _tick_row(db, age_seconds=5, decision="tick_paused")
        res = await _daemon_liveness(db)
        assert res["alive"] is True and res["last_status"] == "tick_paused"
