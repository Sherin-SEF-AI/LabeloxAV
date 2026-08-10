"""Walk the relabel agent across the whole corpus, in independently revertible batches.

Outside the API process on purpose. The corpus is 34,000 frames and roughly two hours of GPU work, and a
task inside the API dies with any restart, holds the card the promotion gate and training also want, and
puts a multi-hour job inside one HTTP request. Here it survives a restart and can be stopped with Ctrl-C
between batches.

One AgentRun per batch rather than one for everything. A batch is a unit somebody can look at, keep or
revert on its own, which matters because a relabel is a change to the corpus and 34,000 frames of them is
not a decision anybody should have to take or undo all at once.

Batches advance because `run_relabel_all` selects frames no committed `relabel` child run has covered, so
this loop makes progress rather than re-reading the same frames.

    .venv/bin/python -m scripts.relabel_corpus --batch 500 --batches 4
    .venv/bin/python -m scripts.relabel_corpus --batch 1000 --all
"""

from __future__ import annotations

import argparse
import asyncio
import time
import uuid

from sqlalchemy import Text, distinct, func, select

from core.logging import get_logger, setup_logging
from db.models import AgentRun, Object
from db.session import get_sessionmaker

log = get_logger("relabel_corpus")


async def _remaining() -> int:
    async with get_sessionmaker()() as db:
        seen = (select(AgentRun.scope["frame_id"].astext)
                .where(AgentRun.kind == "relabel", AgentRun.scope["frame_id"].astext.isnot(None)))
        return (await db.execute(select(func.count()).select_from(
            select(distinct(Object.frame_id))
            .where(Object.source != "human", Object.frame_id.cast(Text).notin_(seen))
            .subquery()))).scalar() or 0


async def _run_batch(n: int) -> dict:
    from services.agent.relabel_agent import run_relabel_all

    run_id = uuid.uuid4()
    async with get_sessionmaker()() as db:
        db.add(AgentRun(run_id=run_id, kind="relabel_all", status="running",
                        scope={"max_frames": n, "session_id": None},
                        policy={"min_conf": 0.45, "margin": 0.15}, counts={}, changes={}, critic={},
                        created_by="relabel_corpus"))
        await db.commit()

    await run_relabel_all(run_id, max_frames=n, created_by="relabel_corpus")

    async with get_sessionmaker()() as db:
        run = await db.get(AgentRun, run_id)
        return {"run_id": str(run_id), "status": run.status, "counts": dict(run.counts or {}),
                "error": run.error}


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=500, help="frames per batch")
    ap.add_argument("--batches", type=int, default=1, help="how many batches to run")
    ap.add_argument("--all", action="store_true", help="keep going until nothing is left")
    args = ap.parse_args()
    setup_logging("INFO")

    left = await _remaining()
    print(f"  frames not yet relabelled: {left:,}")
    if not left:
        print("  nothing to do")
        return 0

    done_frames = keep = review = 0
    t0 = time.perf_counter()
    i = 0
    while True:
        i += 1
        if not args.all and i > args.batches:
            break
        if await _remaining() == 0:
            print("  every eligible frame has been relabelled")
            break

        bt = time.perf_counter()
        out = await _run_batch(args.batch)
        c = out["counts"]
        done_frames += int(c.get("frames", 0))
        keep += int(c.get("relabel_keep", 0))
        review += int(c.get("relabel_review", 0))
        rate = (time.perf_counter() - t0) / max(done_frames, 1)
        rem = await _remaining()
        print(f"  batch {i:>3}  {out['status']:<10} frames={c.get('frames', 0):<5} "
              f"keep={c.get('relabel_keep', 0):<5} review={c.get('relabel_review', 0):<5} "
              f"{time.perf_counter() - bt:>5.0f}s   run {out['run_id'][:8]}   "
              f"{rem:,} left, ~{rem * rate / 3600:.1f}h")
        if out["status"] == "error":
            # Stop rather than grind on: a batch that failed means the next one probably fails the same way,
            # and the run id above is the thing to look at.
            print(f"  stopping: {(out['error'] or '')[:160]}")
            return 1

    dt = time.perf_counter() - t0
    print(f"\n  {done_frames:,} frames in {dt / 60:.1f} min   "
          f"{keep:,} auto-kept, {review:,} routed to review")
    print(f"  {await _remaining():,} frames still to go")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
