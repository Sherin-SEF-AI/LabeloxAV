"""Staged GPU runner. A single resource manager owns model lifecycle and enforces the 16 GB
ceiling (Constraint 4).

Stage 1: YOLO26 (FP16) + SAM 3.1 (FP16) co-resident (~8 GB), detect + segment + (M3) fuse + gate.
Stage 2: Qwen3-VL-4B (Q4, ~3.3 GB) over the uncertain subset (M4).

Run modes: sequential (Stage 1 fully unloads before Stage 2 loads; safest on 16 GB) and
concurrent (both resident, ~11 GB; only if measured free VRAM allows). Every model load is
guarded by a free-VRAM check and fails loudly instead of OOM-crashing.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from uuid import UUID

import click
import cv2
import numpy as np
from sqlalchemy import select

from core.bus import EventBus
from core.config import get_settings
from core.logging import get_logger, setup_logging
from core.schemas import FrameMeta
from core.storage import get_object_store
from db.models import Frame
from db.session import get_sessionmaker
from services.autolabel.paths.base import RawDetection
from services.autolabel.paths.path_a_yolo26 import YoloPath
from services.autolabel.paths.path_b_sam3 import Sam3Path

log = get_logger("runner")


class GpuCapacityError(RuntimeError):
    pass


@dataclass
class FrameDetections:
    frame: FrameMeta
    image_bgr: np.ndarray
    dets_a: list[RawDetection] = field(default_factory=list)
    dets_b: list[RawDetection] = field(default_factory=list)


class VramGuard:
    """Reads real free/total VRAM from the driver and refuses loads that would breach headroom."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.device = self.settings.gpu.device
        self._index = int(self.device.split(":")[1]) if ":" in self.device else 0
        import torch

        self.torch = torch
        if not torch.cuda.is_available():
            raise GpuCapacityError("CUDA not available; the autolabel plane requires a GPU")
        # Force the CUDA context to exist before any memory-stats call (those raise
        # "Invalid device argument" if invoked before the context is initialized).
        torch.cuda.set_device(self._index)
        torch.cuda.init()
        self._warm = torch.zeros(1, device=self.device)

    def free_mb(self) -> float:
        free, _total = self.torch.cuda.mem_get_info(self._index)
        return free / (1024 * 1024)

    def reset_peak(self) -> None:
        self.torch.cuda.reset_peak_memory_stats(self._index)
        self.torch.cuda.empty_cache()

    def peak_mb(self) -> float:
        return self.torch.cuda.max_memory_reserved(self._index) / (1024 * 1024)

    def empty_cache(self) -> None:
        self.torch.cuda.empty_cache()

    def require(self, need_mb: float, name: str) -> None:
        free = self.free_mb()
        head = self.settings.gpu.vram_headroom_mb
        if free - need_mb < head:
            raise GpuCapacityError(
                f"refusing to load {name}: need ~{need_mb:.0f} MB, free {free:.0f} MB, "
                f"headroom {head} MB. Use gpu.mode=sequential or a smaller model."
            )
        log.info("vram.check", model=name, need_mb=round(need_mb), free_mb=round(free))


# Rough resident-set estimates for the guard. The guard also reads actual free VRAM, so these are
# advisory; they exist to fail before a load that obviously will not fit. Measured peak for the
# realized stack (YOLO11 + YOLO-World + SAM) is ~3.5 GB; estimates are deliberately conservative.
EST_YOLO_MB = 1800
EST_PATHB_MB = 3500


class StagedRunner:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.guard = VramGuard()
        self.yolo: YoloPath | None = None
        self.sam: Sam3Path | None = None

    def open_stage1(self) -> None:
        self.guard.reset_peak()
        self.guard.require(EST_YOLO_MB, "path_a_detector")
        self.yolo = YoloPath()
        self.yolo.load()
        self.guard.require(EST_PATHB_MB, "path_b_openvocab")
        self.sam = Sam3Path()
        self.sam.load()
        log.info("stage1.open", free_mb=round(self.guard.free_mb()))

    def close_stage1(self) -> None:
        if self.yolo:
            self.yolo.unload()
        if self.sam:
            self.sam.unload()
        self.yolo = None
        self.sam = None
        self.guard.empty_cache()
        log.info("stage1.close", peak_mb=round(self.guard.peak_mb()), free_mb=round(self.guard.free_mb()))

    def run_stage1_frame(self, image_bgr: np.ndarray) -> tuple[list[RawDetection], list[RawDetection]]:
        if self.yolo is None or self.sam is None:
            raise RuntimeError("stage1 not open")
        dets_a = self.yolo.infer(image_bgr)
        dets_b = self.sam.infer(image_bgr)
        return dets_a, dets_b


def load_image(img_uri: str) -> np.ndarray:
    store = get_object_store()
    buf = np.frombuffer(store.get_bytes(img_uri), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"failed to decode image {img_uri}")
    return img


async def fetch_frames(session_id: UUID, limit: int | None) -> list[FrameMeta]:
    maker = get_sessionmaker()
    async with maker() as db:
        stmt = select(Frame).where(Frame.session_id == session_id).order_by(Frame.ts_ns)
        if limit:
            stmt = stmt.limit(limit)
        rows = (await db.execute(stmt)).scalars().all()
    return [
        FrameMeta(
            frame_id=f.frame_id,
            session_id=f.session_id,
            ts_ns=f.ts_ns,
            cam_id=f.cam_id,
            img_uri=f.img_uri,
            width=f.width,
            height=f.height,
            ego_speed=f.ego_speed,
            quality=f.quality,
        )
        for f in rows
    ]


async def process_session(
    session_id: UUID,
    limit: int | None,
    on_frame: Callable[[FrameDetections], Awaitable[None]],
) -> dict:
    """Stream a session's frames through Stage 1, invoking on_frame per frame. The callback is the
    plug point for fusion + gate + persistence (M3) and the VLM pass (M4)."""
    frames = await fetch_frames(session_id, limit)
    if not frames:
        raise RuntimeError(f"no frames for session {session_id}")

    runner = StagedRunner()
    runner.open_stage1()
    n_a = 0
    n_b = 0
    try:
        for fm in frames:
            img = load_image(fm.img_uri)
            dets_a, dets_b = runner.run_stage1_frame(img)
            n_a += len(dets_a)
            n_b += len(dets_b)
            await on_frame(FrameDetections(frame=fm, image_bgr=img, dets_a=dets_a, dets_b=dets_b))
    finally:
        runner.close_stage1()

    summary = {
        "session_id": str(session_id),
        "frames": len(frames),
        "path_a_detections": n_a,
        "path_b_detections": n_b,
        "peak_vram_mb": round(runner.guard.peak_mb()),
        "vram_ceiling_mb": runner.settings.gpu.vram_total_mb,
    }
    log.info("stage1.summary", **summary)
    if summary["peak_vram_mb"] > summary["vram_ceiling_mb"]:
        log.error("vram.ceiling_exceeded", **summary)
    return summary


async def autolabel_session(session_id: UUID, limit: int | None, vlm_client=None) -> dict:
    """Full pipeline: detect + segment -> fuse -> calibrate -> gate -> (Path C VLM on the uncertain
    subset) -> persist objects.

    Path C is duty-cycled (Principle 08): the VLM runs only on objects the gate would not
    auto-accept, capped by a per-session budget. The VLM call rate is tracked as a first-class
    metric. vlm_client is injectable for tests; otherwise built from config when the VLM is enabled.
    """
    from services.autolabel.fusion import FusionEngine
    from services.autolabel.gate import gate_object, needs_vlm
    from services.autolabel.ontology import get_ontology
    from services.autolabel.paths.path_c_qwen3vl import VlmVerifier, apply_vlm, make_vlm_client
    from services.autolabel.persist import persist_frame_objects

    settings = get_settings()
    store = get_object_store()
    store.ensure_bucket()
    onto = get_ontology()
    engine = FusionEngine(settings, onto)
    maker = get_sessionmaker()
    bus = EventBus()
    await bus.start()

    verifier = None
    if settings.models.vlm.enabled:
        client = vlm_client or make_vlm_client(settings)
        verifier = VlmVerifier(client, onto, settings)

    totals = {"objects": 0, "by_state": {}, "vlm_calls": 0, "vlm_eligible": 0}
    budget = settings.models.vlm.max_calls_per_session

    async with maker() as db:
        per_frame_cap = settings.models.vlm.max_calls_per_frame

        async def on_frame(fd: FrameDetections) -> None:
            fused = engine.fuse_frame(fd.frame.frame_id, fd.dets_a, fd.dets_b)
            # Spend the per-frame VLM budget on the most uncertain objects first (lowest conf).
            order = sorted(range(len(fused)), key=lambda i: fused[i].obj.conf)
            frame_vlm = 0
            for i in order:
                fo = fused[i]
                fo.obj.state = gate_object(fo.obj, onto, settings.gate)
                if verifier and needs_vlm(fo.obj, onto, settings.gate):
                    totals["vlm_eligible"] += 1
                    if totals["vlm_calls"] < budget and frame_vlm < per_frame_cap:
                        res = verifier.verify_object(fd.image_bgr, tuple(fo.obj.bbox.as_list()), fo.obj.class_id)
                        apply_vlm(fo.obj, res, onto, settings.models.vlm.ollama_tag)
                        fo.obj.state = gate_object(fo.obj, onto, settings.gate)  # re-gate post-VLM
                        totals["vlm_calls"] += 1
                        frame_vlm += 1
            by_state = await persist_frame_objects(db, store, bus, fd.frame, fused)
            totals["objects"] += len(fused)
            for k, v in by_state.items():
                totals["by_state"][k] = totals["by_state"].get(k, 0) + v

        try:
            summary = await process_session(session_id, limit, on_frame)
            await db.commit()
        finally:
            await bus.stop()

    objects = max(totals["objects"], 1)
    totals["vlm_call_rate"] = round(totals["vlm_calls"] / objects, 4)
    summary.update(totals)
    store.put_bytes(
        f"autolabel/{session_id}/summary.json",
        json.dumps(summary, indent=2).encode(),
        "application/json",
    )
    log.info(
        "autolabel.done",
        **{k: summary[k] for k in ("objects", "by_state", "vlm_calls", "vlm_call_rate", "peak_vram_mb")},
    )
    return summary


@click.command()
@click.option("--session", "session_id", required=True, type=str)
@click.option("--limit", type=int, default=None, help="cap frames processed (smoke runs)")
def main(session_id: str, limit: int | None) -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    summary = asyncio.run(autolabel_session(UUID(session_id), limit))
    click.echo(summary)


if __name__ == "__main__":
    main()
