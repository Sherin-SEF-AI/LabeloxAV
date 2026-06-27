"""Whole-frame DINOv2 embeddings for active-learning curation. DINOv2 self-supervised features are far
stronger than CLIP for image-to-image similarity, which is what dedup / coverage-gap / novelty / diversity
sampling need. Small model (~88 MB, 384-dim CLS), fast on GPU or CPU.

    python -m services.intelligence.frame_embeddings --all
    python -m services.intelligence.frame_embeddings --session <uuid>
"""

from __future__ import annotations

import asyncio
from uuid import UUID

import click
import cv2
import numpy as np

from core.config import get_settings
from core.logging import get_logger, setup_logging
from core.storage import get_object_store
from db.models import Frame, FrameEmbedding
from db.session import get_sessionmaker

log = get_logger("frame_embed")

MODEL_NAME = "facebook/dinov2-small"
_state: dict = {}


def model_tag() -> str:
    return "dinov2-small"


def _model():
    if "model" not in _state:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        proc = AutoImageProcessor.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME).to(dev).eval()
        _state.update(model=model, proc=proc, device=dev, torch=torch)
        log.info("dinov2.loaded", model=MODEL_NAME, device=dev)
    return _state


def encode_frame(image_bgr: np.ndarray) -> np.ndarray:
    from PIL import Image

    s = _model()
    pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    inputs = s["proc"](images=pil, return_tensors="pt").to(s["device"])
    with s["torch"].no_grad():
        out = s["model"](**inputs)
    cls = out.last_hidden_state[:, 0].squeeze(0).float().cpu().numpy()  # CLS token = global descriptor
    return (cls / (np.linalg.norm(cls) + 1e-8)).astype(np.float32)


async def _embed(frames: list, store) -> int:
    n = 0
    maker = get_sessionmaker()
    async with maker() as db:
        for fid, img_uri in frames:
            try:
                buf = np.frombuffer(store.get_bytes(img_uri), dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                vec = encode_frame(img)
                await db.merge(FrameEmbedding(frame_id=fid, model=model_tag(), dim=int(vec.shape[0]), vec=vec.tolist()))
                n += 1
                if n % 50 == 0:
                    await db.commit()
            except Exception as exc:  # noqa: BLE001
                log.warning("frame_embed.failed", frame_id=str(fid), error=str(exc))
        await db.commit()
    return n


async def compute_frame_embeddings(session_id: UUID | None = None, limit: int | None = None, only_missing: bool = True) -> dict:
    from sqlalchemy import select

    store = get_object_store()
    maker = get_sessionmaker()
    async with maker() as db:
        stmt = select(Frame.frame_id, Frame.img_uri)
        if session_id is not None:
            stmt = stmt.where(Frame.session_id == session_id)
        if only_missing:
            sub = select(FrameEmbedding.frame_id)
            stmt = stmt.where(Frame.frame_id.notin_(sub))
        if limit:
            stmt = stmt.limit(limit)
        frames = (await db.execute(stmt)).all()
    n = await _embed([(f, u) for f, u in frames], store)
    log.info("frame_embed.done", embedded=n, model=model_tag())
    return {"embedded": n, "model": model_tag()}


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
