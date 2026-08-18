"""`promote: true` did not promote anything.

A training job asked to promote its result set `promoted=True` on the model_run row and logged a line
telling an operator to export an environment variable by hand. Nothing became champion, nothing served the
new weights, and the run reported itself promoted while the old model kept answering every request. The
governance path that actually promotes (the gold-set champion gate, the auto-promote kill switch, the
single-champion constraint) existed the whole time behind POST /govern/promote and was never called.

The gate that the training job does run is a different question: it compares the candidate against its own
baseline. Passing it is permission to ask for promotion, not the answer.
"""

from __future__ import annotations

import uuid

import pytest

from db.models import Notification
from db.session import get_sessionmaker
from services.training import jobs as jobs_mod

pytestmark = pytest.mark.db


@pytest.mark.asyncio
async def test_a_promotion_that_the_champion_gate_refuses_is_reported_as_refused():
    """Not "promoted", which is what a boolean set from the training gate alone would have said."""
    async with get_sessionmaker()() as db:
        # No such challenger is registered, so the champion path refuses it. Any refusal reaches the same
        # branch; what matters is that the verdict comes from there rather than from the training gate.
        out = await jobs_mod._promote_run(db, f"never-registered-{uuid.uuid4().hex[:8]}", "detection")

    assert out.get("promoted") is not True
    assert out.get("error") or out.get("reason") or out.get("detail"), "a refusal has to say why"


@pytest.mark.asyncio
async def test_a_blocked_promotion_tells_somebody():
    """A promotion that quietly does not happen is how the flywheel idles for a day unnoticed."""
    run_id = f"never-registered-{uuid.uuid4().hex[:8]}"
    async with get_sessionmaker()() as db:
        await jobs_mod._promote_run(db, run_id, "detection")

    async with get_sessionmaker()() as db:
        from sqlalchemy import select
        rows = (await db.execute(
            select(Notification).where(Notification.subject_id == run_id))).scalars().all()
    assert [r.kind for r in rows] == ["promotion_blocked"]


@pytest.mark.asyncio
async def test_a_governance_failure_does_not_discard_the_finished_training_run(monkeypatch):
    """The weights are already stored and the run row is already written by this point. Losing all of that
    to an error inside promotion would be worse than the missed promotion."""
    async def _boom(*a, **k):
        raise RuntimeError("registry unavailable")

    import services.govern.champion as champ_mod
    monkeypatch.setattr(champ_mod, "evaluate_and_promote", _boom)

    async with get_sessionmaker()() as db:
        out = await jobs_mod._promote_run(db, "any-run", "detection")

    assert out == {"promoted": False, "error": "registry unavailable"}
