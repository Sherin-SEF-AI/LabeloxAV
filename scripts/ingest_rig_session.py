"""Ingest a synchronised multi-camera rig capture, using the session's own alignment index.

The reader is services/ingest/reader_rig.py, which is where the correctness argument lives: session-global
time rather than per-camera PTS, decoded frame positions rather than sidecar row numbers, corrected camera
positions rather than the manifest's, and each camera keeping its own instant so the physical inter-camera
spread is preserved instead of being erased.

This only wires that to the standard ingest, so the rig capture goes through exactly the same Gate A
privacy blur, quality gate and storage path as every other session.

    .venv/bin/python -m scripts.ingest_rig_session /path/to/session --stride 10
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from core.config import get_settings
from core.logging import setup_logging
from services.ingest.reader_rig import read_index, rig_frames
from services.ingest.run import ingest


async def _run(root: Path, vehicle: str, city: str | None, stride: int,
               max_groups: int | None) -> None:
    idx = read_index(root)
    sdir = idx["session_dir"]
    present = {c: len(list((sdir / c).glob("*.mp4"))) for c in sorted(idx["cameras"])}
    print(f"session {sdir.name}")
    print(f"  segments present per camera: {present}")
    print(f"  sync groups {len(idx['groups'])}, sampling every {stride}")

    result = await ingest(
        frame_iter=rig_frames(root, stride=stride, max_groups=max_groups),
        vehicle=vehicle, city=city, route=None,
        raw_uri=str(sdir), mcap_uri=None,
        source_streams=sorted(idx["cameras"]),
    )
    print(f"  ingested: {result}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--vehicle", default="BLURABBIT-RIG-01")
    ap.add_argument("--city", default=None)
    # Default 10 gives 3 fps from a 30 fps rig, which is ingest.target_fps. Sampling groups rather than
    # frames keeps each sampled instant as complete across the rig as the footage allows.
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--max-groups", type=int, default=None)
    args = ap.parse_args()
    setup_logging(get_settings().log_level)
    asyncio.run(_run(args.root, args.vehicle, args.city, args.stride, args.max_groups))


if __name__ == "__main__":
    main()
