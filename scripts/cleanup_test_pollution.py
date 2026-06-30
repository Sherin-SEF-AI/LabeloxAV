"""Delete test-pollution sessions from the dev database.

Infra tests commit Session/Frame/Object rows to the shared dev DB and do not roll them back, so test
sessions accumulate and clutter the editor's session list. Every test frame uses a dummy image URI in the
non-existent bucket "x" (s3://x/...); real ingestion always writes s3://labeloxav/frames/..., so any session
with an s3://x frame is unambiguously a test session. This deletes exactly those (cascade removes their
frames, objects, clouds, etc). Real data is never touched.

Dry run (default) prints what would be deleted; pass --apply to delete.

    python -m scripts.cleanup_test_pollution            # dry run
    python -m scripts.cleanup_test_pollution --apply     # delete
"""

from __future__ import annotations

import asyncio
import sys

from sqlalchemy import delete, func, select

from db.models import Frame
from db.models import Session as DbSession
from db.session import get_engine, get_sessionmaker


async def main(apply: bool) -> None:
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()
    async with get_sessionmaker()() as db:
        test_sids = (await db.execute(
            select(Frame.session_id).where(Frame.img_uri.like("s3://x%")).distinct())).scalars().all()
        # safety: a real session never has an s3://x frame, so none of these should also have a real frame
        mixed = (await db.execute(
            select(func.count(func.distinct(Frame.session_id)))
            .where(Frame.session_id.in_(test_sids), ~Frame.img_uri.like("s3://x%")))).scalar()
        total_s = (await db.execute(select(func.count()).select_from(DbSession))).scalar()

        print(f"total sessions: {total_s}")
        print(f"test sessions (with an s3://x dummy frame): {len(test_sids)}")
        print(f"of those, sessions that ALSO have a real frame (would be wrongly deleted): {mixed}")
        if mixed:
            print("ABORTING: some test sessions also hold real frames. Investigate before deleting.")
            return
        if not apply:
            print("\ndry run. Re-run with --apply to delete these sessions (cascade).")
            return

        res = await db.execute(delete(DbSession).where(DbSession.session_id.in_(test_sids)))
        await db.commit()
        print(f"\ndeleted {res.rowcount} test sessions and their cascaded frames/objects/clouds.")


if __name__ == "__main__":
    asyncio.run(main(apply="--apply" in sys.argv))
