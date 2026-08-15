"""Recover who drew each hand-made label, from the review trail that already recorded it.

`Object.annotator_id` arrived with migration 0091, so every label made before that has none and the
scorecards show a corpus of people who have apparently never labelled anything. The information is not lost:
`create_object` has always written a `Review` row with `action="create"` and the authenticated user, so for
every human-drawn object the creator is one join away.

This is a script rather than part of the migration on purpose. A migration should not spend minutes walking
a 576,393-row table, and this is recovery of a derived fact rather than a schema change: it can be re-run,
it can be skipped, and nothing depends on it having happened.

It only fills nulls. An object already carrying an annotator was attributed at creation time by the current
code, and that is the better source.
"""

from __future__ import annotations

import asyncio

from sqlalchemy import text

from core.logging import get_logger
from db.session import get_sessionmaker

log = get_logger("backfill.attribution")

# The earliest create-review per object: an object created once and re-reviewed later has one creator, and
# a later reviewer is not it.
SQL = """
UPDATE object o
SET annotator_id = c.user_id
FROM (
    SELECT DISTINCT ON (object_id) object_id, user_id
    FROM review
    WHERE action = 'create' AND user_id IS NOT NULL
    ORDER BY object_id, ts_ns ASC
) c
WHERE o.object_id = c.object_id
  AND o.annotator_id IS NULL
"""


async def main() -> None:
    async with get_sessionmaker()() as db:
        before = (await db.execute(
            text("SELECT count(*) FROM object WHERE annotator_id IS NOT NULL"))).scalar()
        result = await db.execute(text(SQL))
        await db.commit()
        after = (await db.execute(
            text("SELECT count(*) FROM object WHERE annotator_id IS NOT NULL"))).scalar()

    log.info("backfill.attribution_done", filled=result.rowcount, before=before, after=after)
    print(f"attributed {result.rowcount} objects from the review trail ({before} -> {after})")


if __name__ == "__main__":
    asyncio.run(main())
