"""Metering: every delivery recorded once, priced at the time, and marked with whether it was measured.

346 dataset commits have left this system with no usage record of any kind. `dataset_commit` records
lineage, which is a different question from what anybody owes.

Two decisions carry most of the weight here.

**A delivery is metered once, not once per call.** A commit id is content-addressed, so exporting the same
slice twice legitimately produces the same commit and the same bytes. Charging per call would bill twice for
one artifact, and re-running an export after a network failure is a normal thing to do. The uniqueness
constraint makes the meter idempotent at the database rather than in a caller nobody will remember to write
carefully.

**An uncertified delivery is billable and says so.** Most of this corpus cannot be certified today: a
certificate needs a sealed gold set and a scored evaluation run, and there are five gold sets against
570,379 objects. Refusing to meter uncertified exports would be dishonest bookkeeping, and silently metering
them as though they were measured would be worse. So the record carries `certified` and the certificate's
signature, and the invoice reports the split. A buyer should be able to see which lines came with a quality
claim they can verify.

Prices are stamped onto the record when it is written. An invoice recomputed against a later price list
would rewrite what a customer was quoted, which is the kind of quietly wrong that gets discovered by the
customer.
"""

from __future__ import annotations

import uuid as _uuid

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from core.timebase import now_ns
from db.models import UsageRecord

log = get_logger("billing.meter")

# What each kind of delivery is counted in. The unit is recorded on every row because an invoice that says
# "quantity 40,000" without saying of what is not an invoice.
UNITS = {"export": "objects", "inference": "frames", "judge": "crops"}


def _price_for(kind: str) -> float | None:
    prices = get_settings().billing.unit_price_inr
    return prices.get(kind)


async def record_delivery(db: AsyncSession, *, kind: str, subject_id: str, quantity: float,
                          account: str = "default", user_id: str | None = None,
                          certified: bool = False, certificate_signature: str | None = None,
                          detail: dict | None = None) -> dict:
    """Meter one delivery. Idempotent on (kind, subject_id): a repeat is recorded, never charged twice.

    Returns the record as it now stands, including whether this call was the one that created it, so a
    caller can tell a fresh delivery from a re-run without querying again.
    """
    if kind not in UNITS:
        raise ValueError(f"unknown billable kind '{kind}'; known: {sorted(UNITS)}")

    price = _price_for(kind)
    amount = round(price * quantity, 4) if price is not None else None
    values = {
        "kind": kind, "subject_id": subject_id, "account": account,
        "user_id": _uuid.UUID(user_id) if user_id else None,
        "quantity": float(quantity), "unit": UNITS[kind],
        "unit_price_inr": price, "amount_inr": amount,
        "certified": certified, "certificate_signature": certificate_signature,
        "detail": detail or {}, "ts_ns": now_ns(),
    }

    # DO NOTHING rather than DO UPDATE. A second export of the same commit is the same artifact at the same
    # price; overwriting would move the record's timestamp into a later billing period and could shift
    # revenue between months for no reason a customer could see.
    stmt = (pg_insert(UsageRecord).values(**values)
            .on_conflict_do_nothing(constraint="uq_usage_record_kind_subject")
            .returning(UsageRecord.record_id))
    created = (await db.execute(stmt)).scalar_one_or_none()
    await db.commit()

    if created is None:
        row = (await db.execute(select(UsageRecord).where(
            UsageRecord.kind == kind, UsageRecord.subject_id == subject_id))).scalar_one()
        log.info("billing.already_metered", kind=kind, subject_id=subject_id)
        return {**_row(row), "created": False}

    log.info("billing.metered", kind=kind, subject_id=subject_id, quantity=quantity,
             amount_inr=amount, certified=certified)
    row = (await db.execute(select(UsageRecord).where(UsageRecord.record_id == created))).scalar_one()
    return {**_row(row), "created": True}


async def certify_delivery(db: AsyncSession, *, kind: str, subject_id: str,
                           certificate_signature: str) -> dict:
    """Attach a quality certificate to a delivery that was metered before one existed.

    Exports usually ship before they are evaluated, so the alternative to this is a permanently uncertified
    line for a dataset that did eventually get measured. Only ever moves a record from uncertified to
    certified: there is no path here that removes a certificate, because an invoice line that quietly stops
    carrying a quality claim is exactly the change a customer would need to be told about.
    """
    row = (await db.execute(select(UsageRecord).where(
        UsageRecord.kind == kind, UsageRecord.subject_id == subject_id))).scalar_one_or_none()
    if row is None:
        return {"error": "no usage record for that delivery", "kind": kind, "subject_id": subject_id}
    row.certified = True
    row.certificate_signature = certificate_signature
    await db.commit()
    log.info("billing.certified", kind=kind, subject_id=subject_id)
    return _row(row)


async def usage_summary(db: AsyncSession, *, account: str | None = None,
                        since_ns: int | None = None, until_ns: int | None = None) -> dict:
    """Usage totals by kind, and the certified/uncertified split within each.

    The split is reported rather than left to be derived, because "you delivered 4.2M objects" and "0.3% of
    them came with a verifiable quality claim" are the two halves of the same sentence and only one of them
    is flattering.
    """
    q = select(UsageRecord.kind, UsageRecord.certified,
               func.count(UsageRecord.record_id), func.sum(UsageRecord.quantity),
               func.sum(UsageRecord.amount_inr)).group_by(UsageRecord.kind, UsageRecord.certified)
    if account:
        q = q.where(UsageRecord.account == account)
    if since_ns is not None:
        q = q.where(UsageRecord.ts_ns >= since_ns)
    if until_ns is not None:
        q = q.where(UsageRecord.ts_ns < until_ns)

    by_kind: dict[str, dict] = {}
    for kind, certified, n, qty, amount in (await db.execute(q)).all():
        d = by_kind.setdefault(kind, {"unit": UNITS.get(kind, "units"), "deliveries": 0, "quantity": 0.0,
                                      "amount_inr": 0.0, "certified": {"deliveries": 0, "quantity": 0.0},
                                      "uncertified": {"deliveries": 0, "quantity": 0.0}})
        d["deliveries"] += int(n)
        d["quantity"] += float(qty or 0)
        d["amount_inr"] += float(amount or 0)
        d["certified" if certified else "uncertified"]["deliveries"] += int(n)
        d["certified" if certified else "uncertified"]["quantity"] += float(qty or 0)

    total = round(sum(d["amount_inr"] for d in by_kind.values()), 2)
    all_deliveries = sum(d["deliveries"] for d in by_kind.values())
    all_certified = sum(d["certified"]["deliveries"] for d in by_kind.values())
    return {
        "account": account, "by_kind": by_kind, "total_inr": total,
        "deliveries": all_deliveries, "certified_deliveries": all_certified,
        "certified_fraction": round(all_certified / all_deliveries, 4) if all_deliveries else None,
    }


async def invoice(db: AsyncSession, *, account: str = "default", since_ns: int | None = None,
                  until_ns: int | None = None) -> dict:
    """An invoice: one line per delivery, with what it was and whether its quality was measured.

    Line-per-delivery rather than a rolled-up total, because the thing being sold is a specific sealed
    commit and a customer disputing a line needs to be able to point at which one.
    """
    q = select(UsageRecord).where(UsageRecord.account == account).order_by(UsageRecord.ts_ns)
    if since_ns is not None:
        q = q.where(UsageRecord.ts_ns >= since_ns)
    if until_ns is not None:
        q = q.where(UsageRecord.ts_ns < until_ns)
    rows = list((await db.execute(q)).scalars().all())

    lines = [_row(r) for r in rows]
    priced = [line for line in lines if line["amount_inr"] is not None]
    unpriced = len(lines) - len(priced)
    return {
        "account": account,
        "lines": lines,
        "total_inr": round(sum(line["amount_inr"] for line in priced), 2),
        # Named rather than treated as zero. An unpriced line is a delivery whose kind has no configured
        # price, and quietly summing it as 0 would under-bill without anybody noticing.
        "unpriced_lines": unpriced,
        "uncertified_lines": sum(1 for line in lines if not line["certified"]),
    }


def _row(r: UsageRecord) -> dict:
    return {
        "record_id": str(r.record_id), "kind": r.kind, "account": r.account,
        "subject_id": r.subject_id, "quantity": r.quantity, "unit": r.unit,
        "unit_price_inr": r.unit_price_inr, "amount_inr": r.amount_inr,
        "certified": r.certified, "certificate_signature": r.certificate_signature,
        "detail": r.detail, "ts_ns": r.ts_ns,
    }
