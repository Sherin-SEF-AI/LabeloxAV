"""Metering: 346 datasets shipped from this system with no record of what was delivered to whom.

Two properties carry the weight and both are easy to get wrong in ways nobody notices until a customer
does: a delivery must be billed once even when the export is re-run, and a delivery sold without a measured
quality claim must be visible as such on the invoice.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from core.config import get_settings

pytestmark = pytest.mark.db


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


def _clear():
    from db.session import get_engine, get_sessionmaker

    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


def run_async(coro):
    _clear()
    try:
        return asyncio.run(coro)
    finally:
        _clear()


@requires_infra
def test_re_exporting_the_same_commit_does_not_bill_twice():
    """The property the uniqueness constraint exists for.

    A commit id is content-addressed, so exporting the same slice twice produces the same commit and the
    same bytes. Re-running after a network failure is normal. Charging per call would bill twice for one
    artifact, and the customer would be right.
    """
    from db.session import get_sessionmaker
    from services.billing.meter import invoice, record_delivery

    account = f"acct-{uuid.uuid4().hex[:8]}"
    commit = f"commit-{uuid.uuid4().hex[:12]}"

    async def _flow():
        async with get_sessionmaker()() as db:
            first = await record_delivery(db, kind="export", subject_id=commit, quantity=1000,
                                          account=account)
        async with get_sessionmaker()() as db:
            second = await record_delivery(db, kind="export", subject_id=commit, quantity=1000,
                                           account=account)
        assert first["created"] is True
        assert second["created"] is False
        assert second["record_id"] == first["record_id"]

        async with get_sessionmaker()() as db:
            inv = await invoice(db, account=account)
        assert len(inv["lines"]) == 1

    run_async(_flow())


@requires_infra
def test_a_repeat_delivery_keeps_its_original_timestamp():
    """DO NOTHING rather than DO UPDATE on conflict.

    Overwriting would move the record into a later billing period, shifting revenue between months for a
    reason no customer could see and no auditor could follow.
    """
    from db.session import get_sessionmaker
    from services.billing.meter import record_delivery

    account = f"acct-{uuid.uuid4().hex[:8]}"
    commit = f"commit-{uuid.uuid4().hex[:12]}"

    async def _flow():
        async with get_sessionmaker()() as db:
            first = await record_delivery(db, kind="export", subject_id=commit, quantity=10, account=account)
        async with get_sessionmaker()() as db:
            again = await record_delivery(db, kind="export", subject_id=commit, quantity=10, account=account)
        assert again["ts_ns"] == first["ts_ns"]

    run_async(_flow())


@requires_infra
def test_an_export_is_metered_uncertified_and_the_invoice_says_so():
    """Most of this corpus cannot be certified: five gold sets against 570,379 objects.

    Refusing to meter uncertified deliveries would be dishonest bookkeeping; metering them as though they
    were measured would be worse. The invoice has to be able to show the difference.
    """
    from db.session import get_sessionmaker
    from services.billing.meter import invoice, record_delivery, usage_summary

    account = f"acct-{uuid.uuid4().hex[:8]}"

    async def _flow():
        async with get_sessionmaker()() as db:
            await record_delivery(db, kind="export", subject_id=f"c-{uuid.uuid4().hex[:12]}",
                                  quantity=500, account=account)
            await record_delivery(db, kind="export", subject_id=f"c-{uuid.uuid4().hex[:12]}",
                                  quantity=700, account=account, certified=True,
                                  certificate_signature="deadbeef")
        async with get_sessionmaker()() as db:
            inv = await invoice(db, account=account)
            summary = await usage_summary(db, account=account)

        assert inv["uncertified_lines"] == 1
        assert summary["by_kind"]["export"]["certified"]["quantity"] == 700
        assert summary["by_kind"]["export"]["uncertified"]["quantity"] == 500
        assert summary["certified_fraction"] == 0.5

    run_async(_flow())


@requires_infra
def test_a_certificate_can_be_attached_after_the_export_shipped():
    """Exports ship before they are evaluated. Without this, an export that was later measured stays marked
    unmeasured forever."""
    from db.session import get_sessionmaker
    from services.billing.meter import certify_delivery, record_delivery

    commit = f"commit-{uuid.uuid4().hex[:12]}"

    async def _flow():
        async with get_sessionmaker()() as db:
            r = await record_delivery(db, kind="export", subject_id=commit, quantity=100)
        assert r["certified"] is False
        async with get_sessionmaker()() as db:
            after = await certify_delivery(db, kind="export", subject_id=commit,
                                           certificate_signature="sig-abc")
        assert after["certified"] is True and after["certificate_signature"] == "sig-abc"

    run_async(_flow())


@requires_infra
def test_certifying_a_delivery_that_was_never_metered_reports_rather_than_inventing_one():
    """Creating the usage record here would bill for a delivery nobody made."""
    from db.session import get_sessionmaker
    from services.billing.meter import certify_delivery

    async def _flow():
        async with get_sessionmaker()() as db:
            r = await certify_delivery(db, kind="export", subject_id="never-exported",
                                       certificate_signature="x")
        assert "error" in r

    run_async(_flow())


@requires_infra
def test_an_unpriced_delivery_is_counted_as_unpriced_not_as_free():
    """A silent zero under-bills and is indistinguishable from a free tier."""
    from db.session import get_sessionmaker
    from services.billing.meter import invoice, record_delivery

    account = f"acct-{uuid.uuid4().hex[:8]}"
    s = get_settings()
    saved = dict(s.billing.unit_price_inr)

    async def _flow():
        try:
            s.billing.unit_price_inr = {}          # nothing priced
            async with get_sessionmaker()() as db:
                r = await record_delivery(db, kind="export", subject_id=f"c-{uuid.uuid4().hex[:12]}",
                                          quantity=100, account=account)
            assert r["amount_inr"] is None
            async with get_sessionmaker()() as db:
                inv = await invoice(db, account=account)
            assert inv["unpriced_lines"] == 1
            assert inv["total_inr"] == 0.0
        finally:
            s.billing.unit_price_inr = saved

    run_async(_flow())


@requires_infra
def test_price_is_stamped_at_write_time_not_recomputed_at_read_time():
    """Recomputing an old invoice against today's prices rewrites what a customer was quoted."""
    from db.session import get_sessionmaker
    from services.billing.meter import invoice, record_delivery

    account = f"acct-{uuid.uuid4().hex[:8]}"
    s = get_settings()
    saved = dict(s.billing.unit_price_inr)

    async def _flow():
        try:
            s.billing.unit_price_inr = {**saved, "export": 2.0}
            async with get_sessionmaker()() as db:
                await record_delivery(db, kind="export", subject_id=f"c-{uuid.uuid4().hex[:12]}",
                                      quantity=100, account=account)
            s.billing.unit_price_inr = {**saved, "export": 50.0}   # price rises afterwards
            async with get_sessionmaker()() as db:
                inv = await invoice(db, account=account)
            assert inv["lines"][0]["unit_price_inr"] == 2.0
            assert inv["total_inr"] == 200.0, "the old invoice must not be repriced"
        finally:
            s.billing.unit_price_inr = saved

    run_async(_flow())


def test_an_unknown_billable_kind_is_refused():
    """A typo would otherwise create a whole category of usage nobody is watching."""
    from db.session import get_sessionmaker
    from services.billing.meter import record_delivery

    async def _flow():
        async with get_sessionmaker()() as db:
            with pytest.raises(ValueError, match="unknown billable kind"):
                await record_delivery(db, kind="exprot", subject_id="x", quantity=1)

    run_async(_flow())
