"""Bus consumer: embed each new frame (DINOv3 + SigLIP 2) as it lands, so the index stays current
without a manual backfill. Run as a lightweight single-box worker:

    python -m services.intelligence.embed.consumer

Subscribes to frame.ready (emitted by the ingest service per accepted frame). GPU-light; the heavy
detect/segment pass is a separate process, so there is no contention.
"""

from __future__ import annotations

import asyncio
from uuid import UUID

import cv2
import numpy as np

from core.bus import TOPIC_FRAME_READY, EventBus
from core.config import get_settings
from core.embeddings import model_versions
from core.logging import get_logger, setup_logging
from core.storage import get_object_store
from db.models import FrameEmbedding
from db.session import get_sessionmaker
from services.intelligence.embed import dinov3, siglip2

log = get_logger("embed_consumer")


async def run() -> None:
    bus, store, maker, mv = EventBus(), get_object_store(), get_sessionmaker(), model_versions()
    log.info("embed.consumer_start", topic=TOPIC_FRAME_READY, **mv)
    async for msg in bus.consume([TOPIC_FRAME_READY], group_id="embed-worker"):
        v = msg["value"]
        try:
            img = cv2.imdecode(np.frombuffer(store.get_bytes(v["img_uri"]), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            dv, sv = dinov3.encode_image(img), siglip2.encode_image(img)
            async with maker() as db:
                await db.merge(FrameEmbedding(frame_id=UUID(v["frame_id"]), dino_vec=dv.tolist(),
                                              siglip_vec=sv.tolist(), model_versions=mv))
                await db.commit()
            log.info("embed.frame_ready", frame_id=v["frame_id"])
        except Exception as exc:  # noqa: BLE001
            log.error("embed.consumer_failed", error=str(exc))


def main() -> None:
    setup_logging(get_settings().log_level)
    asyncio.run(run())


if __name__ == "__main__":
    main()
