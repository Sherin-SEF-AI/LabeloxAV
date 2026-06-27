"""Backfill DINOv3 + SigLIP 2 embeddings for all existing frames and object crops into pgvector.
Idempotent and resumable: skips frames/objects already embedded, so it can be re-run or interrupted.
Logs progress and peak VRAM to keep the hardware envelope honest.

    python -m scripts.backfill_embeddings --all
    python -m scripts.backfill_embeddings --session <uuid> --frames-only
"""

from __future__ import annotations

import asyncio
from uuid import UUID

import click

from core.config import get_settings
from core.logging import get_logger, setup_logging
from services.intelligence.embed.service import embed_frames, embed_objects

log = get_logger("backfill")


@click.command()
@click.option("--session", "session_id", default=None, help="one session (default: whole corpus)")
@click.option("--all", "do_all", is_flag=True, default=False)
@click.option("--limit", type=int, default=None)
@click.option("--frames-only", is_flag=True, default=False)
@click.option("--objects-only", is_flag=True, default=False)
@click.option("--reembed", is_flag=True, default=False, help="re-embed even if already present")
def main(session_id, do_all, limit, frames_only, objects_only, reembed) -> None:
    setup_logging(get_settings().log_level)
    if not session_id and not do_all:
        raise SystemExit("pass --session <uuid> or --all")
    sid = UUID(session_id) if session_id else None
    only_missing = not reembed

    async def run() -> dict:
        out: dict = {}
        if not objects_only:
            out.update(await embed_frames(sid, limit, only_missing=only_missing))
        if not frames_only:
            out.update(await embed_objects(sid, limit, only_missing=only_missing))
        return out

    res = asyncio.run(run())
    try:
        import torch

        if torch.cuda.is_available():
            res["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1e6)
    except Exception:  # noqa: BLE001
        pass
    log.info("backfill.done", **res)
    click.echo(res)


if __name__ == "__main__":
    main()
