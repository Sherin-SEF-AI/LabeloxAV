"""Delete test-pollution sessions from the dev database.

Infra tests commit Session/Frame/Object rows to the shared dev DB and do not roll them back, so test
sessions accumulate. Tests pollute through TWO channels:
  1. dummy-URI frames in the non-existent bucket "x" (s3://x/...); real ingestion writes s3://labeloxav/...
  2. real-bucket frames that are 640x480 random-noise images, uploaded by ingest/editor tests that actually
     store to MinIO (these have a real-looking s3://labeloxav URI but are synthetic noise).
A frame is synthetic if it matches either channel. A session is a test session only when EVERY frame in it
is synthetic (it has no real >=1280-wide dashcam frame), so a real session is never deleted even if a stray
test frame landed in it. Deletes those sessions (cascade removes frames, objects, clouds, etc).

The durable fix for the root cause is test isolation (tests/conftest.py points the suite at labeloxav_test);
this script only cleans up pollution that accumulated before that. A reversible alternative is to flag the
synthetic frames selected=false instead of deleting (see --quarantine).

Dry run (default) prints what would change; --apply deletes; --quarantine flags selected=false instead.

    python -m scripts.cleanup_test_pollution               # dry run
    python -m scripts.cleanup_test_pollution --quarantine  # reversible: selected=false on synthetic frames
    python -m scripts.cleanup_test_pollution --apply        # delete fully-synthetic sessions (cascade)
"""

from __future__ import annotations

import asyncio
import sys

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy import delete as sa_delete

from db.models import Frame
from db.models import Session as DbSession
from db.session import get_engine, get_sessionmaker

# a frame is synthetic if it is a dummy-URI test frame OR a 640x480 noise image
_SYNTHETIC = or_(Frame.img_uri.like("s3://x%"), and_(Frame.width == 640, Frame.height == 480))
# a frame is real dashcam if it is in the real bucket and full resolution
_REAL = and_(Frame.img_uri.like("s3://labeloxav%"), Frame.width >= 1280)


async def main(apply: bool, quarantine: bool) -> None:
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()
    async with get_sessionmaker()() as db:
        synth_frames = (await db.execute(select(func.count(Frame.frame_id)).where(_SYNTHETIC))).scalar()
        synth_sids = set((await db.execute(select(Frame.session_id).where(_SYNTHETIC).distinct())).scalars().all())
        real_sids = set((await db.execute(select(Frame.session_id).where(_REAL).distinct())).scalars().all())
        # a session is safe to delete only if it has synthetic frames and NO real frame
        deletable = synth_sids - real_sids
        mixed = synth_sids & real_sids
        total_s = (await db.execute(select(func.count()).select_from(DbSession))).scalar()

        print(f"total sessions: {total_s}")
        print(f"synthetic frames (dummy-URI or 640x480 noise): {synth_frames}")
        print(f"fully-synthetic sessions (safe to delete): {len(deletable)}")
        print(f"mixed sessions (synthetic + real frames, kept): {len(mixed)}")

        if quarantine:
            r = await db.execute(update(Frame).where(and_(_SYNTHETIC, Frame.selected.is_(True))).values(selected=False))
            await db.commit()
            print(f"\nquarantined {r.rowcount} synthetic frames (selected=false). Reversible:")
            print("  UPDATE frame SET selected=true WHERE img_uri LIKE 's3://x%' OR (width=640 AND height=480);")
            return
        if not apply:
            print("\ndry run. --quarantine to flag (reversible), or --apply to delete (cascade).")
            return
        res = await db.execute(sa_delete(DbSession).where(DbSession.session_id.in_(deletable)))
        await db.commit()
        print(f"\ndeleted {res.rowcount} fully-synthetic sessions and their cascaded frames/objects/clouds.")


if __name__ == "__main__":
    asyncio.run(main(apply="--apply" in sys.argv, quarantine="--quarantine" in sys.argv))
