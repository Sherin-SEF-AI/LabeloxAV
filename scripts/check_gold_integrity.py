"""Report gold sets whose objects have been deleted out from under them.

A gold set seals a membership list as JSONB, not as foreign keys. Sealing protects the list from being
edited; it does not protect the rows it names. The fixture purge deleted 13,172 objects, nothing cascaded,
nothing warned, and 599 of this corpus's 646 sealed gold objects no longer exist: four of five sets are
entirely dangling and the largest holds 47 of its 400.

That is worth a standing check rather than a discovery, because every quality claim in the system is
measured against these sets and a shrunken one weakens every interval on a certificate without changing a
single number's appearance.

    .venv/bin/python -m scripts.check_gold_integrity
"""

from __future__ import annotations

import asyncio

from sqlalchemy import select

from core.logging import setup_logging
from db.models import GoldSet
from db.session import get_sessionmaker
from services.export.certificate import _resolvable_gold


async def _run() -> int:
    async with get_sessionmaker()() as db:
        sets = list((await db.execute(select(GoldSet).order_by(GoldSet.gold_id))).scalars().all())
        rows = []
        for g in sets:
            declared = list(g.object_ids or [])
            resolvable = await _resolvable_gold(db, declared)
            rows.append((g.gold_id, len(declared), resolvable))

    print(f"{'gold set':<38}{'declared':>10}{'resolvable':>12}{'missing':>9}")
    dangling = 0
    for gid, dec, res in rows:
        missing = dec - res
        if missing:
            dangling += 1
        mark = "" if not missing else ("  ALL GONE" if res == 0 else "  degraded")
        print(f"{gid:<38}{dec:>10}{res:>12}{missing:>9}{mark}")

    total_dec = sum(d for _, d, _ in rows)
    total_res = sum(r for _, _, r in rows)
    print()
    print(f"{dangling} of {len(rows)} gold sets have missing objects; "
          f"{total_res} of {total_dec} sealed objects remain")
    # Non-zero exit when anything is dangling, so this can gate a pipeline rather than only inform a reader.
    return 1 if dangling else 0


def main() -> None:
    setup_logging("INFO")
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
