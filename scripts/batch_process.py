"""Batch-process a directory of dashcam clips through the full pipeline:
ingest -> auto-label -> scenario-mine -> embed. Builds a real curated corpus.

    python scripts/batch_process.py --dir /path/to/clips --count 12 --label-limit 20 --fps 1.0
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
from uuid import UUID

from core.config import get_settings
from core.logging import get_logger, setup_logging
from core.timebase import now_ns
from services.autolabel.runner import autolabel_session
from services.ingest.reader_video import read_video
from services.ingest.run import _upload_raw, ingest
from services.intelligence.embeddings import compute_session_embeddings
from services.intelligence.run import mine_session

log = get_logger("batch")


def _pick(clip_dir: Path, count: int, hour_lo: int, hour_hi: int) -> list[Path]:
    clips = sorted(clip_dir.glob("*.MP4")) + sorted(clip_dir.glob("*.mp4"))
    kept = []
    for c in clips:
        # filename like 20260624101641_000035F.MP4 -> hour at index 8:10
        stem = c.name
        hh = stem[8:10] if stem[:8].isdigit() else None
        if hh is not None and hh.isdigit() and not (hour_lo <= int(hh) <= hour_hi):
            continue
        kept.append(c)
    return kept[:count]


async def process_clip(path: Path, fps: float, label_limit: int | None, city: str) -> dict:
    t0 = time.time()
    raw_uri = _upload_raw(path, "TIGOR-07")
    frame_iter = read_video(path, "cam_f", now_ns(), fps, None)
    ing = await ingest(
        frame_iter=frame_iter, vehicle="TIGOR-07", city=city, route=path.stem,
        raw_uri=raw_uri, mcap_uri=None, source_streams=["cam_f"],
    )
    sid = UUID(ing["session_id"])
    lab = await autolabel_session(sid, limit=label_limit)
    mine = await mine_session(sid)
    emb = await compute_session_embeddings(sid)
    return {
        "clip": path.name, "session_id": str(sid), "frames": ing["n_frames"],
        "objects": lab["objects"], "by_state": lab["by_state"], "vlm_calls": lab["vlm_calls"],
        "tracks": mine["tracks"], "scenarios": mine["scenarios"], "scenario_types": mine["by_type"],
        "embedded": emb["embedded"], "secs": round(time.time() - t0, 1),
    }


async def main_async(args) -> None:
    setup_logging("ERROR")
    clips = _pick(Path(args.dir), args.count, args.hour_lo, args.hour_hi)
    print(f"selected {len(clips)} clips from {args.dir}", flush=True)
    agg = {"clips": 0, "frames": 0, "objects": 0, "scenarios": 0, "secs": 0.0}
    for i, c in enumerate(clips, 1):
        try:
            r = await process_clip(c, args.fps, args.label_limit, args.city)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(clips)}] {c.name} FAILED: {exc}", flush=True)
            continue
        agg["clips"] += 1
        agg["frames"] += r["frames"]
        agg["objects"] += r["objects"]
        agg["scenarios"] += r["scenarios"]
        agg["secs"] += r["secs"]
        print(
            f"[{i}/{len(clips)}] {r['clip']}: {r['frames']}f -> {r['objects']}obj "
            f"{r['by_state']} | {r['tracks']}trk {r['scenarios']}scn {r['scenario_types']} "
            f"| vlm {r['vlm_calls']} | {r['secs']}s  (session {r['session_id'][:8]})",
            flush=True,
        )
    print(f"\nDONE: {agg['clips']} clips, {agg['frames']} frames, {agg['objects']} objects, "
          f"{agg['scenarios']} scenarios in {round(agg['secs']/60,1)} min", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--label-limit", type=int, default=20)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--city", default="BLR")
    ap.add_argument("--hour-lo", type=int, default=0)
    ap.add_argument("--hour-hi", type=int, default=23)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
