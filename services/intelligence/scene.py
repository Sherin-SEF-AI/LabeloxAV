"""Zero-shot scene classification (M1.3). For each frame, score the SigLIP 2 image embedding against
text-prompt sets per axis (weather, time_of_day, road_type, density), softmax per axis, and store the
argmax plus per-axis confidence in frame.scene. No labeled scene data required. Reuses the SigLIP 2
frame vectors already in frame_embedding (so classification is a cheap matmul, no image decode). Low
confidence axes can optionally be confirmed by the duty-cycled Qwen-VL path. Emits scene.ready.
"""

from __future__ import annotations

from uuid import UUID

import numpy as np
from sqlalchemy import select, update

from core.bus import TOPIC_SCENE_READY, EventBus
from core.config import get_settings
from core.logging import get_logger
from db.models import Frame, FrameEmbedding
from db.session import get_sessionmaker

log = get_logger("scene")

# India-weighted prompt sets. Labels are the stored values; prompts are the SigLIP 2 text queries.
SCENE_AXES: dict[str, list[tuple[str, str]]] = {
    "weather": [("clear", "a photo of a road in clear sunny weather"), ("rain", "a photo of a road in the rain, wet road"),
                ("fog", "a photo of a road in fog or haze"), ("overcast", "a photo of a road on a grey overcast day")],
    "time_of_day": [("day", "a photo of a road in daylight"), ("night", "a photo of a road at night in the dark"),
                    ("dusk", "a photo of a road at dusk in the evening"), ("dawn", "a photo of a road at dawn in early morning")],
    "road_type": [("urban", "a busy urban city street"), ("highway", "a wide highway or expressway"),
                  ("residential", "a narrow residential neighbourhood street"), ("rural", "a rural country road")],
    "density": [("sparse", "a road with little or no traffic"), ("moderate", "a road with moderate traffic"),
                ("dense", "a road with dense heavy traffic congestion")],
}

_prompt_cache: dict = {}


def _axis_text_vecs() -> dict:
    if not _prompt_cache:
        from services.intelligence.embed import siglip2

        for axis, pairs in SCENE_AXES.items():
            labels = [lbl for lbl, _ in pairs]
            vecs = siglip2.encode_texts([p for _, p in pairs])
            _prompt_cache[axis] = (labels, vecs)
    return _prompt_cache


def classify_vec(frame_siglip_vec, scale: float = 100.0) -> dict:
    """Argmax label + softmax confidence per axis from a frame's SigLIP 2 image vector."""
    fv = np.asarray(frame_siglip_vec, dtype=np.float32)
    scene, conf = {}, {}
    for axis, (labels, tvecs) in _axis_text_vecs().items():
        logits = (tvecs @ fv) * scale
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        i = int(p.argmax())
        scene[axis] = labels[i]
        conf[axis] = round(float(p[i]), 3)
    scene["confidence_per_axis"] = conf
    return scene


async def classify_session(session_id: UUID | None = None, limit: int | None = None, only_missing: bool = True) -> dict:
    maker = get_sessionmaker()
    async with maker() as db:
        stmt = (select(Frame.frame_id, FrameEmbedding.siglip_vec)
                .join(FrameEmbedding, FrameEmbedding.frame_id == Frame.frame_id)
                .where(FrameEmbedding.siglip_vec.isnot(None)))
        if session_id is not None:
            stmt = stmt.where(Frame.session_id == session_id)
        if only_missing:
            stmt = stmt.where(Frame.scene.is_(None))
        if limit:
            stmt = stmt.limit(limit)
        rows = (await db.execute(stmt)).all()

    _axis_text_vecs()  # warm the prompt cache once
    n = 0
    async with maker() as db:
        for fid, vec in rows:
            await db.execute(update(Frame).where(Frame.frame_id == fid).values(scene=classify_vec(vec)))
            n += 1
            if n % 200 == 0:
                await db.commit()
        await db.commit()

    if session_id is not None and n:
        try:
            await EventBus().emit(TOPIC_SCENE_READY, {"session_id": str(session_id), "frames": n}, key=str(session_id))
        except Exception:  # noqa: BLE001
            pass
    log.info("scene.classified", frames=n, session_id=str(session_id) if session_id else "all")
    return {"classified": n}
