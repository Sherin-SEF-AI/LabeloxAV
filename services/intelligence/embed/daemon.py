"""Continuous embedder. Every controller tick, embed a bounded slice of whatever is still unembedded, so
find-similar coverage tracks new data instead of drifting to zero.

Nothing embedded new data on its own: autolabel wrote 122K operational objects and none got vectors until a
manual backfill, so similar search went blind on the whole fleet. This closes that loop. It is deliberately
modest and polite:

  - bounded per tick (a few hundred frames, a few thousand crops) so one tick never runs for minutes;
  - VRAM-gated, so when autolabel or training is on the card the embedder yields rather than competing. That
    competition is exactly what starved the fleet sweep of memory, so the daemon refuses to repeat it;
  - only_missing, so it is idempotent and simply drains the backlog a slice at a time until it is empty, then
    no-ops cheaply.

maybe_embed_pending is the scheduler hook; pending_counts is the cheap check it and the UI can both read.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from db.models import Frame, Object
from services.intelligence.embed.pending import (
    frame_missing_siglip,
    frame_needs_embedding,
    object_missing_siglip,
    object_needs_embedding,
)

log = get_logger("embed.daemon")


async def pending_counts(db: AsyncSession) -> dict:
    """How many frames and objects still have no embedding.

    Uses NOT EXISTS, not NOT IN: `object_id NOT IN (SELECT object_id FROM object_embedding)` builds the whole
    id set and does a row-by-row membership test that takes minutes over 220K rows (and mis-handles a NULL in
    the subquery). NOT EXISTS is a hash anti-join on the embedding table's primary key, sub-second at the same
    scale.

    Both vectors are required, not just a row. See services/intelligence/embed/pending.py: a crop with a
    DINOv3 vector and a NULL SigLIP2 one is unreachable by text search, and the row-existence test that used
    to live here counted the entire corpus as complete while object text search returned nothing.
    """
    frames = (await db.execute(
        select(func.count()).select_from(Frame).where(frame_needs_embedding()))).scalar_one()
    objects = (await db.execute(
        select(func.count()).select_from(Object).where(object_needs_embedding()))).scalar_one()
    # Reported separately because it is the population every previous counter called done. A non-zero value
    # here beside a zero backlog is the signature of that defect returning.
    half_frames = (await db.execute(
        select(func.count()).select_from(Frame).where(frame_missing_siglip()))).scalar_one()
    half_objects = (await db.execute(
        select(func.count()).select_from(Object).where(object_missing_siglip()))).scalar_one()
    return {"frames": int(frames), "objects": int(objects),
            "frames_missing_siglip": int(half_frames), "objects_missing_siglip": int(half_objects)}


def _free_vram_mb() -> float | None:
    """Free VRAM in MB, or None when there is no CUDA device (a CPU box embeds without a memory gate)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return free / (1024 * 1024)
    except Exception:  # noqa: BLE001 - treat any probe failure as "cannot confirm free VRAM"
        return None


async def maybe_embed_pending(db: AsyncSession) -> dict:
    """Embed one bounded, VRAM-gated slice of the backlog. Safe to call every tick.

    Returns {ran, ...}. ran is False (with a reason) when the daemon is disabled, nothing is pending, or the
    GPU is too busy to embed without competing; True with the counts when it did work.
    """
    cfg = get_settings().intel.embed
    if not cfg.daemon_enabled:
        return {"ran": False, "reason": "disabled"}

    pending = await pending_counts(db)
    if pending["frames"] == 0 and pending["objects"] == 0:
        return {"ran": False, "reason": "nothing pending", "pending": pending}

    # Yield the GPU. If a detector or a training run holds the card, free VRAM is low and we sit this tick
    # out rather than repeat the contention that failed the sweep. None (no CUDA) means embed on CPU, no gate.
    free = _free_vram_mb()
    if free is not None and free < cfg.daemon_min_free_mb:
        log.info("embed.daemon.yield", free_mb=round(free), floor=cfg.daemon_min_free_mb, pending=pending)
        return {"ran": False, "reason": f"gpu busy (free {free:.0f} MB < {cfg.daemon_min_free_mb} MB)",
                "pending": pending}

    from services.intelligence.embed.service import embed_frames, embed_objects

    ef = eo = 0
    # Frames first: the frame index backs image/text/frame search, the smaller and higher-leverage backlog.
    if pending["frames"] > 0:
        ef = (await embed_frames(limit=cfg.daemon_max_frames_per_tick, only_missing=True)).get("embedded_frames", 0)
    if pending["objects"] > 0:
        eo = (await embed_objects(limit=cfg.daemon_max_objects_per_tick, only_missing=True)).get("embedded_objects", 0)

    # Zero progress with a non-empty backlog means the remainder is un-embeddable: its source media is gone
    # (a NoSuchKey URI, a fixture row). Report that honestly instead of claiming work every tick forever, so
    # the tick does not log a phantom action and an operator can see the backlog is stuck, not draining.
    if ef == 0 and eo == 0:
        return {"ran": False, "reason": "backlog unembeddable (source media missing)", "pending": pending}

    after = await pending_counts(db)
    log.info("embed.daemon.ran", embedded_frames=ef, embedded_objects=eo, remaining=after)
    return {"ran": True, "embedded_frames": ef, "embedded_objects": eo, "pending_after": after}
