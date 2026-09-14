"""Propose lanes across a session, batch by batch, through the same endpoint the editor uses.

Written because the lane proposer had exactly two callers: one frame at a time from the editor, and a
cloud pod that costs money. Neither can put lanes on a whole drive, which is why 1,385 frames in this
corpus have lanes and none of them is a frame that also has a position fix. Without an overlap the HD map
georeferencer cannot run at all.

Goes through the HTTP route rather than importing the detector, so what lands is exactly what a person
clicking "propose" would get: the same drivable-surface filter, the same line-type classification, the
same provenance. A second code path here would be a second answer.

Batched with a small concurrency limit for the reason everything here is batched: the proposer decodes a
full frame per call and the machine has one GPU that other work is also asking for.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from collections import Counter

import httpx

API = "http://localhost:8000"


async def frames_of(session_id: str, *, only_missing: bool, require_gnss: bool) -> list[str]:
    from sqlalchemy import text

    from db.session import get_sessionmaker

    where = ["f.session_id = :s"]
    if require_gnss:
        where.append("f.gnss is not null")
    if only_missing:
        where.append("not exists (select 1 from lane l where l.frame_id = f.frame_id)")
    sql = f"select f.frame_id from frame f where {' and '.join(where)} order by f.ts_ns"  # noqa: S608
    async with get_sessionmaker()() as db:
        return [str(r[0]) for r in await db.execute(text(sql), {"s": session_id})]


async def run(session_id: str, token: str, *, batch: int, concurrency: int,
              only_missing: bool, require_gnss: bool) -> dict:
    ids = await frames_of(session_id, only_missing=only_missing, require_gnss=require_gnss)
    if not ids:
        return {"frames": 0, "reason": "no frame in this session matches (already has lanes, or no fix)"}
    print(f"{len(ids)} frames to propose on")

    tally = Counter()
    lanes = rejected = 0
    sem = asyncio.Semaphore(concurrency)
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    async with httpx.AsyncClient(base_url=API, timeout=120.0, headers=headers) as c:
        async def one(fid: str) -> None:
            nonlocal lanes, rejected
            async with sem:
                try:
                    r = await c.post(f"/api/frames/{fid}/lanes/propose")
                except Exception as exc:
                    tally[f"error {type(exc).__name__}"] += 1
                    return
                if r.status_code != 200:
                    tally[f"HTTP {r.status_code}"] += 1
                    return
                d = r.json()
                lanes += int(d.get("proposed") or 0)
                rejected += int(d.get("rejected_off_surface") or 0)
                tally["ok"] += 1

        t0 = time.time()
        for i in range(0, len(ids), batch):
            await asyncio.gather(*(one(f) for f in ids[i:i + batch]))
            done = min(i + batch, len(ids))
            print(f"  {done}/{len(ids)}  lanes {lanes}  off-surface rejects {rejected}"
                  f"  {time.time() - t0:.0f}s")
            await asyncio.sleep(0)

    return {"session_id": session_id, "frames": len(ids), "lanes": lanes,
            "rejected_off_surface": rejected, "outcomes": dict(tally),
            "secs": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--token", default="")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--redo", action="store_true", help="propose again on frames that already have lanes")
    ap.add_argument("--any-frame", action="store_true",
                    help="do not require a position fix; lanes without one cannot be georeferenced")
    a = ap.parse_args()
    out = asyncio.run(run(a.session, a.token, batch=a.batch, concurrency=a.concurrency,
                          only_missing=not a.redo, require_gnss=not a.any_frame))
    print(out)


if __name__ == "__main__":
    main()
