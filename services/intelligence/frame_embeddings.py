"""Frame embedding CLI (compatibility shim).

The real DINOv3 + SigLIP 2 frame embedder lives in services.intelligence.embed.service, which writes the
frame_embedding.dino_vec (768) and siglip_vec (1152) columns. This module previously carried a stale
single-vector implementation that wrote columns the schema no longer has (model/dim/vec) and therefore
crashed on every frame. It is kept only as a CLI entry point and now delegates to embed_frames.

    python -m services.intelligence.frame_embeddings --session <uuid>
"""

from __future__ import annotations

import asyncio
from uuid import UUID

import click

from core.config import get_settings
from core.logging import setup_logging
from services.intelligence.embed.service import embed_frames


async def compute_frame_embeddings(session_id: UUID | None = None, limit: int | None = None,
                                   only_missing: bool = True) -> dict:
    """Embed a session's frames (or all frames) with the real DINOv3 + SigLIP 2 pipeline."""
    return await embed_frames(session_id=session_id, limit=limit, only_missing=only_missing)


@click.command()
@click.option("--session", "session_id", default=None)
@click.option("--all", "do_all", is_flag=True, default=False)
@click.option("--limit", type=int, default=None)
def main(session_id, do_all, limit) -> None:
    setup_logging(get_settings().log_level)
    sid = UUID(session_id) if session_id else None
    if not sid and not do_all:
        raise SystemExit("pass --session <uuid> or --all")
    click.echo(asyncio.run(compute_frame_embeddings(sid, limit)))


if __name__ == "__main__":
    main()
