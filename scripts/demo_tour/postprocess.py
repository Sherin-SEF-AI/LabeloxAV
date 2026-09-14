"""Run the rest of the pipeline over the tour's sessions, so the tour shows processed data.

Auto-labelling produces boxes. Almost every page after the review chapter reads something built on top
of those boxes: tracks, embeddings, scenarios, three dimensional cuboids, occupancy grids. Without this
pass the later chapters would narrate features over empty pages.

Each stage is separate and failures do not cascade. A session that cannot be lifted into three
dimensions still gets its tracks, and the stage that failed prints why. The tour then narrates whatever
is actually there, which is the point of recording against a live system rather than a fixture.
"""

from __future__ import annotations

import argparse
import asyncio
import time
import uuid

from sqlalchemy import text

from core.logging import get_logger, setup_logging
from db.session import get_sessionmaker

log = get_logger("demo.post")


async def _sessions(routes: list[str]) -> list[uuid.UUID]:
    async with get_sessionmaker()() as db:
        rows = await db.execute(text(
            "select session_id from session where route = any(:r) or route like 'KITTI%' order by created_at"
        ), {"r": routes})
        return [r[0] for r in rows]


async def stage(name: str, coro) -> dict:
    """Run one stage, time it, and turn a failure into a reported reason rather than a stack trace."""
    t0 = time.time()
    try:
        out = await coro
        secs = round(time.time() - t0, 1)
        print(f"    {name:22s} ok   {secs:7.1f}s  {out if isinstance(out, dict) else ''}")
        return {"stage": name, "ok": True, "secs": secs, "result": out}
    except Exception as exc:
        secs = round(time.time() - t0, 1)
        # An exception with an empty message is common here (a bare AttributeError from an optional
        # dependency), and the type name alone is still the useful half of the report.
        first = next((ln for ln in str(exc).splitlines() if ln.strip()), "")
        reason = f"{type(exc).__name__}: {first[:140]}" if first else type(exc).__name__
        print(f"    {name:22s} FAIL {secs:7.1f}s  {reason}")
        return {"stage": name, "ok": False, "secs": secs, "reason": reason}


async def lift_frames(session_id: uuid.UUID, batch: int = 32) -> dict:
    """Lift this session's 2D boxes into cuboids, for the frames that have a cloud to lift against.

    Done here rather than through `maybe_lift_pending`, which chooses the least covered sessions across
    the whole corpus and refuses to run twice in a day. That is right for a nightly daemon and wrong for
    preparing three named sessions.

    Batched, with a yield between batches, for the reason every heavy path in this system is batched: one
    unbounded loop over a session is how a desktop with one card stops responding.
    """
    from sqlalchemy import text as _t

    from services.lidar.detect3d.run import lift_frame

    async with get_sessionmaker()() as db:
        frames = [r[0] for r in await db.execute(_t("""
            select distinct f.frame_id from frame f
            join point_cloud pc on pc.session_id = f.session_id and pc.ts_ns = f.ts_ns
            join object o on o.frame_id = f.frame_id
            where f.session_id = :s
            order by f.frame_id"""), {"s": session_id})]
    if not frames:
        return {"frames": 0, "reason": "no frame in this session has both a point cloud and a 2D box"}

    lifted, failed = 0, 0
    for i in range(0, len(frames), batch):
        for fid in frames[i:i + batch]:
            try:
                r = await lift_frame(fid)
                lifted += int(r.get("cuboids") or 0)
            except Exception:
                failed += 1
        await asyncio.sleep(0)
    return {"frames": len(frames), "cuboids": lifted, "frames_failed": failed}


async def run(session_id: uuid.UUID, do_3d: bool) -> list[dict]:
    from services.intelligence.embeddings import compute_session_embeddings
    from services.intelligence.run import mine_session

    print(f"  session {session_id}")
    out = [
        # Tracking first: scenarios and the three dimensional lift are both defined over tracks, so
        # running them before this would produce nothing and report success.
        await stage("tracks + scenarios", mine_session(session_id)),
        await stage("frame embeddings", compute_session_embeddings(session_id)),
    ]
    if not do_3d:
        return out

    from services.intelligence.ego_pose import build_ego_pose
    from services.lidar.occupancy4d import build_occupancy_window
    from services.lidar.track3d.from2d import lift_tracks_for_session

    # Ego pose before the lift. A cuboid is lifted into the ego frame and only becomes a world position
    # once there is a trajectory to place it on, and the occupancy grid is built in world coordinates.
    out.append(await stage("ego pose", build_ego_pose(session_id)))
    # Per-frame cuboids before per-track ones: `lift_tracks_for_session` selects tracks that already have
    # lifted boxes, so running it first returns zero and looks like success.
    out.append(await stage("3d per frame", lift_frames(session_id)))
    out.append(await stage("3d from tracks", lift_tracks_for_session(session_id, limit=40)))
    out.append(await stage("occupancy grids", build_occupancy_window(session_id)))
    return out


async def main_async(routes: list[str], do_3d: bool) -> None:
    sessions = await _sessions(routes)
    if not sessions:
        raise SystemExit(f"no sessions matched routes {routes}")
    print(f"{len(sessions)} sessions")
    failed = []
    for sid in sessions:
        for r in await run(sid, do_3d):
            if not r["ok"]:
                failed.append((str(sid)[:8], r["stage"], r["reason"]))
    print(f"\n{len(failed)} stage failures")
    for sid, st, why in failed:
        print(f"  {sid} {st}: {why}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", default="demo-tour-2026")
    ap.add_argument("--no-3d", action="store_true")
    a = ap.parse_args()
    setup_logging("warning")
    asyncio.run(main_async([r for r in a.routes.split(",") if r], not a.no_3d))


if __name__ == "__main__":
    main()
