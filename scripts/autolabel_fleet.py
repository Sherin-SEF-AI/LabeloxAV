"""Auto-label every already-ingested session of a fleet, one session at a time.

The corpus had 32,455 operational DASHCAM-01 frames sitting at 0.6 percent labeled, because ingest ran with a
small --label-limit and the sweep was never done. Every retrain iteration so far trained on public-dataset
imports plus a sliver of operational data, which is why the detector is domain-shifted. This closes that gap.

Resumable at session granularity: a session that already has objects is skipped, so an interrupted run picks
up where it stopped rather than redoing work or double-writing. Models load per session (about 6 s) instead of
once for the whole sweep, which costs ~18 minutes over 184 sessions but keeps VRAM clean between sessions and
means one session's failure cannot poison the next.

    python scripts/autolabel_fleet.py --vehicle DASHCAM-01
    python scripts/autolabel_fleet.py --vehicle DASHCAM-01 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from uuid import UUID

from sqlalchemy import distinct, func, select

from core.logging import get_logger, setup_logging
from db.models import Frame, Object
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.autolabel.runner import autolabel_session

log = get_logger("autolabel_fleet")


async def pending_sessions(vehicle: str, *, redo: bool) -> list[tuple[UUID, int, int]]:
    """Sessions for the fleet as (session_id, n_frames, n_labeled_frames), least-labeled first.

    Ordering by labeled-frame count ascending puts the completely untouched sessions first, so the coverage
    number moves fastest early and an interrupted sweep has still done the most valuable work.
    """
    async with get_sessionmaker()() as db:
        rows = (
            await db.execute(
                select(
                    DbSession.session_id,
                    func.count(distinct(Frame.frame_id)),
                    func.count(distinct(Object.frame_id)),
                )
                .select_from(DbSession)
                .join(Frame, Frame.session_id == DbSession.session_id)
                .outerjoin(Object, Object.frame_id == Frame.frame_id)
                .where(DbSession.vehicle_id == vehicle)
                .group_by(DbSession.session_id)
                .order_by(func.count(distinct(Object.frame_id)), func.count(distinct(Frame.frame_id)).desc())
            )
        ).all()
    return [(s, nf, nl) for s, nf, nl in rows if redo or nl == 0]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicle", default="DASHCAM-01")
    ap.add_argument("--limit-sessions", type=int, default=None, help="cap sessions this run")
    ap.add_argument("--frame-limit", type=int, default=None, help="cap frames per session")
    ap.add_argument("--redo", action="store_true", help="include sessions that already have objects")
    ap.add_argument("--progress", default=".scratch/autolabel_fleet_progress.jsonl")
    ap.add_argument("--dry-run", action="store_true", help="report the plan and exit")
    args = ap.parse_args()

    setup_logging()
    todo = await pending_sessions(args.vehicle, redo=args.redo)
    if args.limit_sessions:
        todo = todo[: args.limit_sessions]

    total_frames = sum(nf for _, nf, _ in todo)
    print(f"fleet {args.vehicle}: {len(todo)} sessions pending, {total_frames:,} frames")
    # 2.0 s/frame measured on a full 180-frame session, VLM pass included.
    print(f"estimated wall clock at 2.0 s/frame: {total_frames * 2.0 / 3600:.1f} h")
    if args.dry_run or not todo:
        return

    prog = Path(args.progress)
    prog.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    done = failed = objects = auto = 0

    for i, (sid, nf, _) in enumerate(todo, 1):
        t0 = time.time()
        try:
            summary = await autolabel_session(sid, args.frame_limit)
            by_state = summary.get("by_state", {})
            objects += int(summary.get("objects", 0))
            auto += int(by_state.get("auto_accept", 0))
            done += 1
            rec = {"session": str(sid), "frames": nf, "ok": True, "secs": round(time.time() - t0, 1), **summary}
        except Exception as exc:  # noqa: BLE001
            # One bad session (missing image, decode failure) must not end an 18-hour sweep.
            failed += 1
            rec = {"session": str(sid), "frames": nf, "ok": False, "error": str(exc)[:300]}
            log.warning("fleet.session_failed", session=str(sid), error=str(exc)[:200])

        with prog.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

        elapsed = time.time() - started
        frames_done = sum(f for _, f, _ in todo[:i])
        eta = (total_frames - frames_done) * (elapsed / max(frames_done, 1)) / 3600
        print(
            f"[{i}/{len(todo)}] {str(sid)[:8]} {nf:>4}f  "
            f"objects={objects:,} auto_accept={auto:,} failed={failed}  "
            f"elapsed={elapsed / 3600:.2f}h eta={eta:.2f}h",
            flush=True,
        )

    print(f"\ndone: {done} sessions, {failed} failed, {objects:,} objects, {auto:,} auto-accepted")


if __name__ == "__main__":
    asyncio.run(main())
